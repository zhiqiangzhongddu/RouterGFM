"""Runtime orchestration for the GraphMoRE MoE method.

Dispatched from ``src.moe.run`` when ``moe.method == 'graphmore'``. Supports
two execution modes:

* single-config multi-seed runs from ``cfg.moe.graphmore.dataset`` settings
  (default), and
* batch execution over a header-based TSV
  (``cfg.moe.graphmore.run_tasks_tsv == True``), one config per row.

Each config trains :class:`GraphMoRERunner` over ``cfg.moe.graphmore.num_runs``
seeds, prints a multi-run summary, and appends an aggregated row to
``outputs/results/moe_graphmore.tsv``. Mirrors ``src.moe.gmoe.run`` /
``src.train.runtime``.
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

from .trainer import GraphMoRERunner

# Config keys recorded as result-table columns so GraphMoRE rows are
# self-describing and comparable across runs. ``moe.method`` is tracked
# first so ``outputs/results/moe_graphmore.tsv`` identifies the producing method.
_IDENTITY_KEYS = [
    "moe.method",
    "moe.graphmore.dataset.name",
    "moe.graphmore.dataset.task_level",
    "moe.graphmore.dataset.induced",
    "moe.graphmore.hidden_dim",
    "moe.graphmore.embed_dim",
    "moe.graphmore.coef_dis",
    "moe.graphmore.epochs",
    "moe.graphmore.lr",
    "moe.graphmore.lr_riemann",
    "moe.graphmore.batch_size",
]


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------

_HEADER_COLUMNS = {
    "dataset", "task_level", "task_type", "induced",
    "hidden_dim", "embed_dim", "coef_dis",
    "fixed_split", "epochs", "batch", "skip_if_exists",
}
_REQUIRED_COLUMNS = ("dataset", "task_level", "induced")
_GRAPHMORE_INT_COLUMNS = {"hidden_dim", "embed_dim"}
_GRAPHMORE_FLOAT_COLUMNS = {"coef_dis"}
_DEFAULTS = {
    "task_type": None,
    "hidden_dim": None,
    "embed_dim": None,
    "coef_dis": None,
    "fixed_split": None,
    "epochs": None,
    "batch": None,
    "skip_if_exists": None,
}


def _graphmore_custom_parser(col: str, val: str, line_no: int):
    """Parse GraphMoRE-specific numeric columns; fall through for the rest."""
    if col in _GRAPHMORE_INT_COLUMNS:
        try:
            return int(val), True
        except ValueError:
            print(f"[GraphMoRE] Skipping malformed task row {line_no}: invalid {col} '{val}'")
            return None, False
    if col in _GRAPHMORE_FLOAT_COLUMNS:
        try:
            return float(val), True
        except ValueError:
            print(f"[GraphMoRE] Skipping malformed task row {line_no}: invalid {col} '{val}'")
            return None, False
    return None


def _task_identity(task: dict[str, Any]) -> tuple:
    return (
        task["dataset"],
        task["task_level"],
        task.get("task_type"),
        bool(task["induced"]),
        task.get("hidden_dim"),
        task.get("embed_dim"),
        task.get("coef_dis"),
        task.get("fixed_split"),
        task.get("epochs"),
        task.get("batch"),
        task.get("skip_if_exists"),
    )


def parse_graphmore_tasks(tsv_path: str) -> list[dict[str, Any]]:
    """Read GraphMoRE tasks from a header-based TSV."""
    header_columns, data_rows = read_tsv_rows(
        tsv_path, _HEADER_COLUMNS, min_header_columns=3, log_prefix="[GraphMoRE]",
    )
    tasks: list[dict[str, Any]] = []
    if header_columns is None:
        if data_rows:
            print("[GraphMoRE] Missing header row; expected a '#'-prefixed header with known column names.")
        return tasks

    for line_no, parts in data_rows:
        task = parse_row_by_header(
            parts, header_columns, line_no, "[GraphMoRE]",
            required_columns=_REQUIRED_COLUMNS,
            defaults=_DEFAULTS,
            custom_parser=_graphmore_custom_parser,
        )
        if task is not None:
            tasks.append(task)
    return dedup_tasks(tasks, _task_identity)


def _build_task_cfg(base_cfg, task: dict[str, Any]):
    run_cfg = base_cfg.clone()
    gm_cfg = run_cfg.moe.graphmore
    explicit_keys = list(_IDENTITY_KEYS)

    gm_cfg.dataset.name = task["dataset"]
    gm_cfg.dataset.task_level = task["task_level"]
    gm_cfg.dataset.induced = bool(task["induced"])
    gm_cfg.dataset.num_classes = None
    gm_cfg.dataset.label_dim = None
    gm_cfg.in_dim = 0
    gm_cfg.run_tasks_tsv = False

    set_cfg_field_and_track(gm_cfg.dataset, "task_type", task.get("task_type"), "moe.graphmore.dataset.task_type", explicit_keys)
    set_cfg_field_and_track(gm_cfg.dataset, "fixed_split", task.get("fixed_split"), "moe.graphmore.dataset.fixed_split", explicit_keys)
    set_cfg_field_and_track(gm_cfg, "hidden_dim", task.get("hidden_dim"), "moe.graphmore.hidden_dim", explicit_keys)
    set_cfg_field_and_track(gm_cfg, "embed_dim", task.get("embed_dim"), "moe.graphmore.embed_dim", explicit_keys)
    set_cfg_field_and_track(gm_cfg, "coef_dis", task.get("coef_dis"), "moe.graphmore.coef_dis", explicit_keys)
    set_cfg_field_and_track(gm_cfg, "epochs", task.get("epochs"), "moe.graphmore.epochs", explicit_keys)
    set_cfg_field_and_track(gm_cfg, "batch_size", task.get("batch"), "moe.graphmore.batch_size", explicit_keys)
    set_cfg_field_and_track(gm_cfg, "skip_if_exists", task.get("skip_if_exists"), "moe.graphmore.skip_if_exists", explicit_keys)

    set_explicit_cfg_keys(run_cfg, explicit_keys)
    return run_cfg


# ---------------------------------------------------------------------------
# Multi-run execution + result saving
# ---------------------------------------------------------------------------

def _run_seeds(base_cfg) -> bool:
    """Train one config over all resolved seeds and append a result row."""
    started_at = datetime.now().astimezone()
    requested_runs = int(getattr(base_cfg.moe.graphmore, "num_runs", 0) or 0)
    seeds: list[int] = []
    run_metrics: list[dict[str, float]] = []
    runners: list[GraphMoRERunner] = []
    try:
        seeds = resolve_seeds(base_cfg, requested_count=requested_runs)
    except ValueError as exc:
        print(f"[GraphMoRE][Multi-run] {exc}")
        return False

    total_runs = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        run_cfg = base_cfg.clone()
        run_cfg.seed = int(seed)
        if total_runs > 1:
            print(f"[GraphMoRE][Multi-run] Running {index}/{total_runs} with seed={seed}")
        set_seed(run_cfg.seed)
        runner = GraphMoRERunner(cfg=run_cfg)
        runner.fit()
        runners.append(runner)
        run_metrics.append(collect_run_metrics(runner, log_prefix="[GraphMoRE][Summary]"))

    summarize_runs(run_metrics, seeds, log_prefix="[GraphMoRE][Summary]")
    ended_at = datetime.now().astimezone()
    _append_graphmore_result(base_cfg, runners, run_metrics, started_at, ended_at, seeds)
    return True


def _append_graphmore_result(cfg, runners, run_metrics, started_at, ended_at, seeds) -> None:
    if not should_save_result(runners, cfg):
        return
    summary = aggregate_run_metrics(run_metrics)
    append_workflow_result(
        cfg=cfg,
        # GraphMoRE rows go into the per-method MoE results table
        # (outputs/results/moe_graphmore.tsv).
        workflow="moe_graphmore",
        started_at=started_at,
        ended_at=ended_at,
        checkpoint_save_paths=collect_checkpoint_paths(runners),
        seeds=seeds,
        best_epochs=summary["epoch_values"].get("best_epoch"),
        metric_summary=summary["metric_stats"],
    )


def _run_tasks_tsv(cfg) -> int:
    tasks = parse_graphmore_tasks(getattr(cfg.moe.graphmore, "tasks_tsv", ""))
    if not tasks:
        print("[GraphMoRE] No tasks found to run.")
        return 1

    results: list[bool] = []
    for task in tasks:
        base_cfg = _build_task_cfg(cfg, task)
        try:
            results.append(_run_seeds(base_cfg))
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[GraphMoRE] Failed {task['dataset']} (task_level={task['task_level']}, "
                f"induced={task['induced']}): {exc}"
            )
            results.append(False)
    return 0 if all(results) else 1


def run_graphmore(cfg) -> int:
    """Execute one or more GraphMoRE training runs for the provided config."""
    if bool(getattr(cfg.moe.graphmore, "run_tasks_tsv", False)):
        return _run_tasks_tsv(cfg)

    # Single-config path: seed identity keys so the result row is self-describing.
    explicit_keys = get_explicit_cfg_keys(cfg)
    for key in _IDENTITY_KEYS:
        if key not in explicit_keys:
            explicit_keys.append(key)
    set_explicit_cfg_keys(cfg, explicit_keys)

    return 0 if _run_seeds(cfg) else 1


__all__ = ["run_graphmore", "parse_graphmore_tasks"]
