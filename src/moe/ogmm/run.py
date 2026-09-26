"""Runtime orchestration for the OGMM MoE baseline.

Dispatched from ``src.moe.run`` when ``moe.method == 'ogmm'``. Runs one config
over ``cfg.moe.ogmm.num_runs`` seeds, or every row of a header-based TSV
(``cfg.moe.ogmm.run_tasks_tsv``), and appends one aggregated row per config to
``outputs/results/moe_ogmm.tsv``. The ``split_root`` column selects a shift
condition (``data/splits_shift/<condition>``) for Table 15.
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
from src.utils.save_results import append_workflow_result, get_explicit_cfg_keys, set_explicit_cfg_keys
from src.utils.tsv_parsing import dedup_tasks, parse_row_by_header, read_tsv_rows, set_cfg_field_and_track

from .trainer import OGMMRunner

# Config keys recorded as result-table columns. The split root identifies the
# shift condition, and the budget is part of fixed_split.
_IDENTITY_KEYS = [
    "moe.method",
    "moe.ogmm.dataset.name",
    "moe.ogmm.dataset.task_level",
    "moe.ogmm.dataset.induced",
    "moe.ogmm.dataset.fixed_split",
    "data_preparation.dataset.split_root",
    "moe.ogmm.num_domains",
    "moe.ogmm.expert_archs",
    "moe.ogmm.expert_hidden_dim",
    "moe.ogmm.expert_epochs",
    "moe.ogmm.gen_epochs",
    "moe.ogmm.gen_num_graphs",
    "moe.ogmm.merge_epochs",
    "moe.ogmm.top_k",
    "moe.ogmm.lambda_gate",
    "moe.ogmm.lambda_mask",
    "moe.ogmm.batch_size",
]

# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------

_HEADER_COLUMNS = {
    "dataset", "task_level", "task_type", "induced", "fixed_split", "split_root",
    "num_domains", "top_k", "expert_epochs", "gen_epochs", "merge_epochs", "batch", "skip_if_exists",
}
_REQUIRED_COLUMNS = ("dataset", "task_level", "induced")
_INT_COLUMNS = {"num_domains", "top_k", "expert_epochs", "gen_epochs", "merge_epochs"}
_DEFAULTS = {key: None for key in _HEADER_COLUMNS if key not in _REQUIRED_COLUMNS}


def _ogmm_custom_parser(col: str, val: str, line_no: int):
    """Parse OGMM integer columns and the split root (``-`` keeps the default); fall through otherwise."""
    if col in _INT_COLUMNS:
        try:
            return int(val), True
        except ValueError:
            print(f"[OGMM] Skipping malformed task row {line_no}: invalid {col} '{val}'")
            return None, False
    if col == "split_root":
        return (None if val.strip() in ("", "-") else val.strip()), True
    return None


def _task_identity(task: dict[str, Any]) -> tuple:
    return tuple(task.get(key) for key in sorted(_HEADER_COLUMNS))


def parse_ogmm_tasks(tsv_path: str) -> list[dict[str, Any]]:
    """Read OGMM tasks from a header-based TSV."""
    header_columns, data_rows = read_tsv_rows(tsv_path, _HEADER_COLUMNS, min_header_columns=3, log_prefix="[OGMM]")
    tasks: list[dict[str, Any]] = []
    if header_columns is None:
        if data_rows:
            print("[OGMM] Missing header row; expected a '#'-prefixed header with known column names.")
        return tasks
    for line_no, parts in data_rows:
        task = parse_row_by_header(
            parts, header_columns, line_no, "[OGMM]",
            required_columns=_REQUIRED_COLUMNS,
            defaults=_DEFAULTS,
            custom_parser=_ogmm_custom_parser,
        )
        if task is not None:
            tasks.append(task)
    return dedup_tasks(tasks, _task_identity)


def _build_task_cfg(base_cfg, task: dict[str, Any]):
    run_cfg = base_cfg.clone()
    o = run_cfg.moe.ogmm
    explicit_keys = list(_IDENTITY_KEYS)

    o.dataset.name = task["dataset"]
    o.dataset.task_level = task["task_level"]
    o.dataset.induced = bool(task["induced"])
    o.dataset.num_classes = None
    o.dataset.label_dim = None
    o.in_dim = 0
    o.run_tasks_tsv = False

    track = set_cfg_field_and_track
    track(o.dataset, "task_type", task.get("task_type"), "moe.ogmm.dataset.task_type", explicit_keys)
    track(o.dataset, "fixed_split", task.get("fixed_split"), "moe.ogmm.dataset.fixed_split", explicit_keys)
    track(run_cfg.data_preparation.dataset, "split_root", task.get("split_root"), "data_preparation.dataset.split_root", explicit_keys)
    for column in ("num_domains", "top_k", "expert_epochs", "gen_epochs", "merge_epochs", "skip_if_exists"):
        track(o, column, task.get(column), f"moe.ogmm.{column}", explicit_keys)
    track(o, "batch_size", task.get("batch"), "moe.ogmm.batch_size", explicit_keys)

    set_explicit_cfg_keys(run_cfg, explicit_keys)
    return run_cfg


# ---------------------------------------------------------------------------
# Multi-run execution + result saving
# ---------------------------------------------------------------------------

def _run_seeds(base_cfg) -> bool:
    """Run one config over all resolved seeds and append a result row."""
    started_at = datetime.now().astimezone()
    try:
        seeds = resolve_seeds(base_cfg, requested_count=int(getattr(base_cfg.moe.ogmm, "num_runs", 0) or 0))
    except ValueError as exc:
        print(f"[OGMM][Multi-run] {exc}")
        return False

    runners: list[OGMMRunner] = []
    run_metrics: list[dict[str, float]] = []
    for index, seed in enumerate(seeds, start=1):
        run_cfg = base_cfg.clone()
        run_cfg.seed = int(seed)
        if len(seeds) > 1:
            print(f"[OGMM][Multi-run] Running {index}/{len(seeds)} with seed={seed}")
        set_seed(run_cfg.seed)
        runner = OGMMRunner(cfg=run_cfg)
        runner.fit()
        runners.append(runner)
        run_metrics.append(collect_run_metrics(runner, log_prefix="[OGMM][Summary]"))

    summarize_runs(run_metrics, seeds, log_prefix="[OGMM][Summary]")
    if should_save_result(runners, base_cfg):
        summary = aggregate_run_metrics(run_metrics)
        append_workflow_result(
            cfg=base_cfg,
            workflow="moe_ogmm",
            started_at=started_at,
            ended_at=datetime.now().astimezone(),
            checkpoint_save_paths=collect_checkpoint_paths(runners),
            seeds=seeds,
            best_epochs=summary["epoch_values"].get("best_epoch"),
            metric_summary=summary["metric_stats"],
        )
    return True


def _run_tasks_tsv(cfg) -> int:
    tasks = parse_ogmm_tasks(getattr(cfg.moe.ogmm, "tasks_tsv", ""))
    if not tasks:
        print("[OGMM] No tasks found to run.")
        return 1
    results: list[bool] = []
    for task in tasks:
        try:
            results.append(_run_seeds(_build_task_cfg(cfg, task)))
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[OGMM] Failed {task['dataset']} (task_level={task['task_level']}, "
                f"split_root={task.get('split_root')}): {exc}"
            )
            results.append(False)
    return 0 if all(results) else 1


def run_ogmm(cfg) -> int:
    """Execute one or more OGMM runs for the provided config."""
    if bool(getattr(cfg.moe.ogmm, "run_tasks_tsv", False)):
        return _run_tasks_tsv(cfg)

    explicit_keys = get_explicit_cfg_keys(cfg)
    for key in _IDENTITY_KEYS:
        if key not in explicit_keys:
            explicit_keys.append(key)
    set_explicit_cfg_keys(cfg, explicit_keys)
    return 0 if _run_seeds(cfg) else 1


__all__ = ["parse_ogmm_tasks", "run_ogmm"]
