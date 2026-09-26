"""Helpers for saving workflow result summaries into TSV tables."""

from __future__ import annotations

import csv
import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


from .paths import ensure_dir, resolve_project_path
from .run_helpers import should_include_summary_metric


@contextmanager
def exclusive_table_lock(table_path: Path):
    """Hold the advisory flock that serialises writers of ``table_path``.

    The ``.tsv.lock`` file is created once and deliberately never unlinked.
    Unlinking after release races: a second writer can block on the old
    (now anonymous) inode while a third writer creates a fresh lock file and
    acquires it immediately — two processes inside the critical section at
    once. A persistent zero-byte lock file has no such window.
    """
    ensure_dir(table_path.parent)
    lock_path = table_path.with_suffix(".tsv.lock")
    with open(lock_path, "a") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)

_METADATA_COLUMNS = [
    "started_at",
    "ended_at",
    "duration_sec",
]
_SUMMARY_COLUMNS = [
    "seeds",
    "best_epochs",
]
_PERF_SUFFIXES = ("_mean", "_std")
_OMITTED_COLUMNS = {
    "device",
    "seeds",
    "save_results.save_skipped",
    "pretrain.run_tasks_tsv",
    "pretrain.tasks_tsv",
    "train.run_tasks_tsv",
    "train.tasks_tsv",
    "finetune.run_tasks_tsv",
    "finetune.tasks_tsv",
}


def extract_explicit_cfg_keys(
    argv: Sequence[str],
    *,
    flag_arity: Mapping[str, int] | None = None,
) -> list[str]:
    """Extract CLI config keys from a yacs-style alternating KEY VALUE argv list."""
    cleaned: list[str] = []
    idx = 0
    flag_arity = dict(flag_arity or {})
    while idx < len(argv):
        token = str(argv[idx])
        # Handle both "--flag value" and "--flag=value" forms.
        arity = flag_arity.get(token)
        if arity is not None:
            idx += 1 + int(arity)
            continue
        if "=" in token and token.startswith("-"):
            prefix = token.split("=", 1)[0]
            if prefix in flag_arity:
                idx += 1
                continue
        cleaned.append(token)
        idx += 1

    keys: list[str] = []
    seen: set[str] = set()
    for idx in range(0, len(cleaned) - 1, 2):
        key = str(cleaned[idx]).strip()
        if not key or key.startswith("-") or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def set_explicit_cfg_keys(cfg, keys: Iterable[str]) -> None:
    """Persist the explicit CLI key list onto cfg.save_results for later use."""
    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        text = str(key).strip()
        if not text or text in seen or _omit_column(text):
            continue
        seen.add(text)
        ordered.append(text)
    cfg.save_results.explicit_keys = ordered


def get_explicit_cfg_keys(cfg) -> list[str]:
    """Read the explicit CLI key list stored on cfg.save_results."""
    raw_keys = getattr(getattr(cfg, "save_results", None), "explicit_keys", []) or []
    return [str(key) for key in raw_keys if str(key).strip() and not _omit_column(str(key))]


def result_table_path(cfg, workflow: str) -> Path:
    """Resolve the per-workflow results TSV path."""
    output_root = getattr(getattr(cfg, "save_results", None), "output_dir", "outputs")
    root = resolve_project_path(output_root, default="outputs")
    return root / f"{workflow}.tsv"


def append_workflow_result(
    *,
    cfg,
    workflow: str,
    started_at: datetime,
    ended_at: datetime,
    checkpoint_save_paths: Sequence[str],
    seeds: Sequence[int] | None,
    best_epochs: Sequence[int] | None,
    metric_summary: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    """Append one workflow result row, rewriting the TSV if the schema expands."""
    if not bool(getattr(getattr(cfg, "save_results", None), "enabled", False)):
        return

    table_path = result_table_path(cfg, workflow)

    # Build the new row outside the lock so we hold it as briefly as possible.
    row_data = _build_result_row(
        cfg=cfg,
        started_at=started_at,
        ended_at=ended_at,
        checkpoint_save_paths=checkpoint_save_paths,
        seeds=seeds,
        best_epochs=best_epochs,
        metric_summary=metric_summary,
    )

    # Serialize concurrent writes with an advisory file lock so parallel
    # SLURM tasks on different nodes see the same lock.  The lock file
    # lives next to the results table on the shared filesystem and stays
    # in place between writes (see exclusive_table_lock).
    with exclusive_table_lock(table_path):
        _append_row_locked(table_path, cfg, row_data, metric_summary)


def append_workflow_result_rows(
    *,
    cfg,
    workflow: str,
    rows: Sequence[Mapping[str, Any]],
    started_at: datetime,
    ended_at: datetime,
) -> int:
    """Append several per-record result rows for foundation-model-style methods.

    :func:`append_workflow_result` maps one ``cfg`` to a single row keyed by a
    metric summary -- the right shape for the single-model siblings (gmoe,
    mowst, graphmore).  AnyGraph instead evaluates many held-out datasets per
    run, so it produces N already-flattened rows (one per dataset/task).  This
    helper writes them to ``outputs/results/<workflow>.tsv`` using the same
    advisory file lock and schema-expanding header as the single-row path, so
    the table stays consistent with the sibling ``moe_*.tsv`` tables.  All rows
    in a batch share the run's ``started_at``/``ended_at`` metadata.  Returns
    the number of rows written.
    """
    if not bool(getattr(getattr(cfg, "save_results", None), "enabled", False)):
        return 0
    materialized = [dict(row) for row in rows if row]
    if not materialized:
        return 0

    table_path = result_table_path(cfg, workflow)
    meta = {
        "started_at": _format_timestamp(started_at),
        "ended_at": _format_timestamp(ended_at),
        "duration_sec": f"{max(0.0, (ended_at - started_at).total_seconds()):.2f}",
    }
    stamped = [{**meta, **row} for row in materialized]

    with exclusive_table_lock(table_path):
        _append_rows_locked(table_path, stamped)
    return len(stamped)


def _append_rows_locked(table_path: Path, new_rows: Sequence[Mapping[str, str]]) -> None:
    """Read-modify-write the TSV for a batch of pre-flattened rows under lock."""
    existing_header, existing_rows = _load_existing_table(table_path)
    # Metadata columns first, then existing columns in order, then any new
    # columns introduced by the incoming rows.  Existing rows are preserved
    # verbatim; missing cells are filled blank by ``_write_table``.
    header: list[str] = list(_METADATA_COLUMNS)
    for column in existing_header:
        if column not in header:
            header.append(column)
    for row in new_rows:
        for column in row.keys():
            if column not in header:
                header.append(column)
    combined: list[Mapping[str, str]] = list(existing_rows)
    for row in new_rows:
        combined.append({column: _stringify_value(row.get(column, "")) for column in header})
    _write_table(table_path, header, combined)


def _build_result_row(
    *,
    cfg,
    started_at: datetime,
    ended_at: datetime,
    checkpoint_save_paths: Sequence[str],
    seeds: Sequence[int] | None,
    best_epochs: Sequence[int] | None,
    metric_summary: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, str]:
    """Prepare the field values for one result row (no I/O)."""
    row: dict[str, str] = {}
    row["started_at"] = _format_timestamp(started_at)
    row["ended_at"] = _format_timestamp(ended_at)
    row["duration_sec"] = f"{max(0.0, (ended_at - started_at).total_seconds()):.2f}"
    if seeds is not None:
        row["seeds"] = _stringify_list([int(seed) for seed in seeds])
    if best_epochs is not None:
        row["best_epochs"] = _stringify_list([int(epoch) for epoch in best_epochs])
    for metric_name, stats in sorted((metric_summary or {}).items()):
        if not should_include_summary_metric(metric_name):
            continue
        row[f"{metric_name}_mean"] = _stringify_scalar(stats.get("mean"))
        row[f"{metric_name}_std"] = _stringify_scalar(stats.get("std"))
    return row


def _append_row_locked(
    table_path: Path,
    cfg,
    row_data: dict[str, str],
    metric_summary: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    """Read-modify-write the TSV while the caller holds the file lock."""
    existing_header, existing_rows = _load_existing_table(table_path)

    existing_config_cols = _config_columns(existing_header)
    existing_perf_cols = _performance_columns(existing_header)
    explicit_keys = get_explicit_cfg_keys(cfg)
    new_config_cols = [key for key in explicit_keys if key not in existing_config_cols]
    new_perf_cols = _performance_columns_for_metrics(metric_summary)
    appended_perf_cols = [key for key in new_perf_cols if key not in existing_perf_cols]

    header = (
        list(_METADATA_COLUMNS)
        + existing_config_cols
        + new_config_cols
        + list(_SUMMARY_COLUMNS)
        + existing_perf_cols
        + appended_perf_cols
    )

    if not existing_header:
        existing_rows = []
    elif header != existing_header:
        existing_rows = _rewrite_existing_rows(
            rows=existing_rows,
            old_header=existing_header,
            new_header=header,
            new_config_cols=new_config_cols,
        )

    perf_columns = set(_performance_columns(header))
    row = {
        column: (_performance_default_value() if column in perf_columns else "")
        for column in header
    }
    # Merge pre-built values.
    row.update(row_data)
    # Fill config columns from the cfg object.
    for column in _config_columns(header):
        row[column] = _stringify_cfg_value(cfg, column)

    existing_rows.append(row)
    _write_table(table_path, header, existing_rows)


def _load_existing_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = list(reader.fieldnames or [])
        rows = [{key: value or "" for key, value in row.items()} for row in reader]
    return header, rows


def _rewrite_existing_rows(
    *,
    rows: Sequence[Mapping[str, str]],
    old_header: Sequence[str],
    new_header: Sequence[str],
    new_config_cols: Sequence[str],
) -> list[dict[str, str]]:
    rewritten: list[dict[str, str]] = []
    perf_columns = set(_performance_columns(new_header))
    for old_row in rows:
        new_row = {
            column: (_performance_default_value() if column in perf_columns else "")
            for column in new_header
        }
        for column in old_header:
            if column in new_row:
                value = str(old_row.get(column, "") or "")
                if column in ("started_at", "ended_at"):
                    value = _normalize_timestamp_text(value)
                elif column == "duration_sec" and value:
                    try:
                        value = f"{max(0.0, float(value)):.2f}"
                    except (TypeError, ValueError):
                        pass
                new_row[column] = value
        # Leave new config columns blank for historical rows rather than
        # backfilling with current defaults, which could misrepresent the
        # configuration that was actually used for the older experiment.
        for column in new_config_cols:
            if column in new_row and not new_row[column]:
                new_row[column] = ""
        for column in perf_columns:
            if not str(new_row.get(column, "")).strip():
                new_row[column] = _performance_default_value()
        rewritten.append(new_row)
    return rewritten


def _write_table(path: Path, header: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    # Atomic replace: a job killed mid-write (timelimit/OOM) must not destroy
    # the accumulated table, and unlocked readers must never observe a
    # half-written file.
    ensure_dir(path.parent)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(header), delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in header})
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _config_columns(header: Sequence[str]) -> list[str]:
    return [
        column
        for column in header
        if column not in _METADATA_COLUMNS
        and column not in _SUMMARY_COLUMNS
        and not _omit_column(column)
        and column not in _performance_columns(header)
    ]


def _performance_columns(header: Sequence[str]) -> list[str]:
    return [column for column in header if _is_performance_column(column)]


def _performance_columns_for_metrics(metric_summary: Mapping[str, Mapping[str, Any]] | None) -> list[str]:
    columns: list[str] = []
    for metric_name in sorted((metric_summary or {}).keys()):
        if not should_include_summary_metric(metric_name):
            continue
        columns.extend([f"{metric_name}_mean", f"{metric_name}_std"])
    return columns


def _is_performance_column(column: str) -> bool:
    if column in _METADATA_COLUMNS or column in _SUMMARY_COLUMNS:
        return False
    return any(column.endswith(suffix) for suffix in _PERF_SUFFIXES)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_timestamp_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return _format_timestamp(datetime.fromisoformat(text))
    except ValueError:
        return text


def _stringify_cfg_value(cfg, dotted_key: str) -> str:
    return _stringify_value(_resolve_cfg_value(cfg, dotted_key))


def _resolve_cfg_value(obj: Any, dotted_key: str) -> Any:
    current = obj
    for part in str(dotted_key).split("."):
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current.get(part)
            continue
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _stringify_list(values: Sequence[Any]) -> str:
    if not values:
        return ""
    return json.dumps(list(values))


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _performance_default_value() -> str:
    return "-1"


def _omit_column(column: str) -> bool:
    return str(column).strip() in _OMITTED_COLUMNS or str(column).strip().startswith("save_results.")


__all__ = [
    "append_workflow_result",
    "append_workflow_result_rows",
    "extract_explicit_cfg_keys",
    "get_explicit_cfg_keys",
    "result_table_path",
    "set_explicit_cfg_keys",
]
