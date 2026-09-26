"""Runtime orchestration for the Mowst MoE method.

Dispatched from ``src.moe.run`` when ``moe.method == 'mowst'``. Supports two
execution modes:

* single-config multi-seed runs from ``cfg.moe.mowst.dataset`` settings
  (default), and
* batch execution over a header-based TSV
  (``cfg.moe.mowst.run_tasks_tsv == True``), one config per row.

Each config trains :class:`MowstRunner` over ``cfg.moe.mowst.num_runs`` seeds,
prints a multi-run summary, and appends an aggregated row to the per-method
MoE results table ``outputs/results/moe_mowst.tsv``. Mirrors ``src.moe.graphmore.run`` /
``src.moe.gmoe.run``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.utils.random import set_seed
from src.utils.run_helpers import (
    aggregate_run_metrics,
    collect_checkpoint_paths,
    collect_run_metrics,
    resolve_seeds,
    should_save_result,
    summarize_runs,
)
from src.utils.save_results import (
    append_workflow_result,
    get_explicit_cfg_keys,
    set_explicit_cfg_keys,
)
from src.utils.tsv_parsing import (
    dedup_tasks,
    parse_row_by_header,
    read_tsv_rows,
    set_cfg_field_and_track,
)

from .trainer import MowstRunner

# Config keys recorded as result-table columns so Mowst rows are
# self-describing and comparable across runs. ``moe.method`` is tracked
# first so ``outputs/results/moe_mowst.tsv`` identifies the producing method.
_IDENTITY_KEYS = [
    "moe.method",
    "moe.mowst.dataset.name",
    "moe.mowst.dataset.task_level",
    "moe.mowst.dataset.induced",
    "moe.mowst.variant",
    "moe.mowst.subloss",
    "moe.mowst.submethod",
    "moe.mowst.weak.model",
    "moe.mowst.strong.model",
    "moe.mowst.hidden_dim",
    "moe.mowst.epochs",
    "moe.mowst.lr",
    "moe.mowst.lr_gate",
    "moe.mowst.batch_size",
]


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------

_HEADER_COLUMNS = {
    "dataset", "task_level", "task_type", "induced",
    "variant", "subloss", "submethod", "weak_model", "strong_model",
    "hidden_dim", "fixed_split", "epochs", "batch", "skip_if_exists",
}
_REQUIRED_COLUMNS = ("dataset", "task_level", "induced")
_MOWST_INT_COLUMNS = {"hidden_dim"}
_DEFAULTS = {
    "task_type": None,
    "variant": None,
    "subloss": None,
    "submethod": None,
    "weak_model": None,
    "strong_model": None,
    "hidden_dim": None,
    "fixed_split": None,
    "epochs": None,
    "batch": None,
    "skip_if_exists": None,
}


def _mowst_custom_parser(col: str, val: str, line_no: int):
    """Parse Mowst-specific numeric columns; fall through for the rest."""
    if col in _MOWST_INT_COLUMNS:
        try:
            return int(val), True
        except ValueError:
            print(f"[Mowst] Skipping malformed task row {line_no}: invalid {col} '{val}'")
            return None, False
    return None


def _task_identity(task: dict[str, Any]) -> tuple:
    return (
        task["dataset"],
        task["task_level"],
        task.get("task_type"),
        bool(task["induced"]),
        task.get("variant"),
        task.get("subloss"),
        task.get("submethod"),
        task.get("weak_model"),
        task.get("strong_model"),
        task.get("hidden_dim"),
        task.get("fixed_split"),
        task.get("epochs"),
        task.get("batch"),
        task.get("skip_if_exists"),
    )


def parse_mowst_tasks(tsv_path: str) -> list[dict[str, Any]]:
    """Read Mowst tasks from a header-based TSV."""
    header_columns, data_rows = read_tsv_rows(
        tsv_path, _HEADER_COLUMNS, min_header_columns=3, log_prefix="[Mowst]",
    )
    tasks: list[dict[str, Any]] = []
    if header_columns is None:
        if data_rows:
            print("[Mowst] Missing header row; expected a '#'-prefixed header with known column names.")
        return tasks

    for line_no, parts in data_rows:
        task = parse_row_by_header(
            parts, header_columns, line_no, "[Mowst]",
            required_columns=_REQUIRED_COLUMNS,
            defaults=_DEFAULTS,
            custom_parser=_mowst_custom_parser,
        )
        if task is not None:
            tasks.append(task)
    return dedup_tasks(tasks, _task_identity)


def _build_task_cfg(base_cfg, task: dict[str, Any]):
    run_cfg = base_cfg.clone()
    m_cfg = run_cfg.moe.mowst
    explicit_keys = list(_IDENTITY_KEYS)

    m_cfg.dataset.name = task["dataset"]
    m_cfg.dataset.task_level = task["task_level"]
    m_cfg.dataset.induced = bool(task["induced"])
    m_cfg.dataset.num_classes = None
    m_cfg.dataset.label_dim = None
    m_cfg.in_dim = 0
    m_cfg.run_tasks_tsv = False

    set_cfg_field_and_track(m_cfg.dataset, "task_type", task.get("task_type"), "moe.mowst.dataset.task_type", explicit_keys)
    set_cfg_field_and_track(m_cfg.dataset, "fixed_split", task.get("fixed_split"), "moe.mowst.dataset.fixed_split", explicit_keys)
    set_cfg_field_and_track(m_cfg, "variant", task.get("variant"), "moe.mowst.variant", explicit_keys)
    set_cfg_field_and_track(m_cfg, "subloss", task.get("subloss"), "moe.mowst.subloss", explicit_keys)
    set_cfg_field_and_track(m_cfg, "submethod", task.get("submethod"), "moe.mowst.submethod", explicit_keys)
    set_cfg_field_and_track(m_cfg.weak, "model", task.get("weak_model"), "moe.mowst.weak.model", explicit_keys)
    set_cfg_field_and_track(m_cfg.strong, "model", task.get("strong_model"), "moe.mowst.strong.model", explicit_keys)
    set_cfg_field_and_track(m_cfg, "hidden_dim", task.get("hidden_dim"), "moe.mowst.hidden_dim", explicit_keys)
    set_cfg_field_and_track(m_cfg, "epochs", task.get("epochs"), "moe.mowst.epochs", explicit_keys)
    set_cfg_field_and_track(m_cfg, "batch_size", task.get("batch"), "moe.mowst.batch_size", explicit_keys)
    set_cfg_field_and_track(m_cfg, "skip_if_exists", task.get("skip_if_exists"), "moe.mowst.skip_if_exists", explicit_keys)

    set_explicit_cfg_keys(run_cfg, explicit_keys)
    return run_cfg


# ---------------------------------------------------------------------------
# Multi-run execution + result saving
# ---------------------------------------------------------------------------

def _run_seeds(base_cfg) -> bool:
    """Train one config over all resolved seeds and append a result row."""
    started_at = datetime.now().astimezone()
    requested_runs = int(getattr(base_cfg.moe.mowst, "num_runs", 0) or 0)
    seeds: list[int] = []
    run_metrics: list[dict[str, float]] = []
    runners: list[MowstRunner] = []
    try:
        seeds = resolve_seeds(base_cfg, requested_count=requested_runs)
    except ValueError as exc:
        print(f"[Mowst][Multi-run] {exc}")
        return False

    total_runs = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        run_cfg = base_cfg.clone()
        run_cfg.seed = int(seed)
        if total_runs > 1:
            print(f"[Mowst][Multi-run] Running {index}/{total_runs} with seed={seed}")
        set_seed(run_cfg.seed)
        runner = MowstRunner(cfg=run_cfg)
        runner.fit()
        runners.append(runner)
        run_metrics.append(collect_run_metrics(runner, log_prefix="[Mowst][Summary]"))

    summarize_runs(run_metrics, seeds, log_prefix="[Mowst][Summary]")
    ended_at = datetime.now().astimezone()
    _append_mowst_result(base_cfg, runners, run_metrics, started_at, ended_at, seeds)
    return True


def _append_mowst_result(cfg, runners, run_metrics, started_at, ended_at, seeds) -> None:
    if not should_save_result(runners, cfg):
        return
    summary = aggregate_run_metrics(run_metrics)
    append_workflow_result(
        cfg=cfg,
        # Mowst rows go into the per-method MoE results table
        # (outputs/results/moe_mowst.tsv).
        workflow="moe_mowst",
        started_at=started_at,
        ended_at=ended_at,
        checkpoint_save_paths=collect_checkpoint_paths(runners),
        seeds=seeds,
        best_epochs=summary["epoch_values"].get("best_epoch"),
        metric_summary=summary["metric_stats"],
    )


def _run_tasks_tsv(cfg) -> int:
    tasks = parse_mowst_tasks(getattr(cfg.moe.mowst, "tasks_tsv", ""))
    if not tasks:
        print("[Mowst] No tasks found to run.")
        return 1

    results: list[bool] = []
    for task in tasks:
        base_cfg = _build_task_cfg(cfg, task)
        try:
            results.append(_run_seeds(base_cfg))
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[Mowst] Failed {task['dataset']} (task_level={task['task_level']}, "
                f"induced={task['induced']}): {exc}"
            )
            results.append(False)
    return 0 if all(results) else 1


def run_mowst(cfg) -> int:
    """Execute one or more Mowst training runs for the provided config."""
    if bool(getattr(cfg.moe.mowst, "run_tasks_tsv", False)):
        return _run_tasks_tsv(cfg)

    # Single-config path: seed identity keys so the result row is self-describing.
    explicit_keys = get_explicit_cfg_keys(cfg)
    for key in _IDENTITY_KEYS:
        if key not in explicit_keys:
            explicit_keys.append(key)
    set_explicit_cfg_keys(cfg, explicit_keys)

    return 0 if _run_seeds(cfg) else 1


__all__ = ["run_mowst", "parse_mowst_tasks"]
