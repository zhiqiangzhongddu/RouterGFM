"""Runtime orchestration for the GMoE MoE method.

Dispatched from ``src.moe.run`` when ``moe.method == 'gmoe'``. Supports
two execution modes:

* single-config multi-seed runs from ``cfg.moe.gmoe.dataset`` settings
  (default), and
* batch execution over a header-based TSV
  (``cfg.moe.gmoe.run_tasks_tsv == True``), one config per row.

Each config trains :class:`GMoERunner` over ``cfg.moe.gmoe.num_runs``
seeds, prints a multi-run summary, and appends an aggregated row to the
per-method MoE results table ``outputs/results/moe_gmoe.tsv``. Mirrors the
structure of ``src.train.runtime`` / ``src.train.utils``.
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

from .trainer import GMoERunner

# Config keys recorded as result-table columns so GMoE rows are
# self-describing and comparable across runs. ``moe.method`` is tracked
# first so ``outputs/results/moe_gmoe.tsv`` carries a column identifying
# which method produced each row; once the column exists,
# append_workflow_result auto-fills it from ``cfg.moe.method``.
_IDENTITY_KEYS = [
    "moe.method",
    "moe.gmoe.dataset.name",
    "moe.gmoe.dataset.task_level",
    "moe.gmoe.dataset.induced",
    "moe.gmoe.gnn_type",
    "moe.gmoe.num_experts",
    "moe.gmoe.num_experts_1hop",
    "moe.gmoe.k",
    "moe.gmoe.expert_hop",
    "moe.gmoe.coef",
    "moe.gmoe.hidden_dim",
    "moe.gmoe.num_layers",
    "moe.gmoe.epochs",
    "moe.gmoe.lr",
    "moe.gmoe.batch_size",
]


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------

_HEADER_COLUMNS = {
    "dataset", "task_level", "task_type", "induced", "gnn_type",
    "num_experts", "num_experts_1hop", "k", "expert_hop",
    "fixed_split", "epochs", "batch", "skip_if_exists",
}
_REQUIRED_COLUMNS = ("dataset", "task_level", "induced")
_GMOE_INT_COLUMNS = {"num_experts", "num_experts_1hop", "k", "expert_hop"}
_DEFAULTS = {
    "task_type": None,
    "gnn_type": None,
    "num_experts": None,
    "num_experts_1hop": None,
    "k": None,
    "expert_hop": None,
    "fixed_split": None,
    "epochs": None,
    "batch": None,
    "skip_if_exists": None,
}


def _gmoe_custom_parser(col: str, val: str, line_no: int):
    """Parse GMoE-specific integer columns; fall through for the rest."""
    if col in _GMOE_INT_COLUMNS:
        try:
            return int(val), True
        except ValueError:
            print(f"[GMoE] Skipping malformed task row {line_no}: invalid {col} '{val}'")
            return None, False
    return None


def _task_identity(task: dict[str, Any]) -> tuple:
    return (
        task["dataset"],
        task["task_level"],
        task.get("task_type"),
        bool(task["induced"]),
        task.get("gnn_type"),
        task.get("num_experts"),
        task.get("num_experts_1hop"),
        task.get("k"),
        task.get("expert_hop"),
        task.get("fixed_split"),
        task.get("epochs"),
        task.get("batch"),
        task.get("skip_if_exists"),
    )


def parse_gmoe_tasks(tsv_path: str) -> list[dict[str, Any]]:
    """Read GMoE tasks from a header-based TSV."""
    header_columns, data_rows = read_tsv_rows(
        tsv_path, _HEADER_COLUMNS, min_header_columns=3, log_prefix="[GMoE]",
    )
    tasks: list[dict[str, Any]] = []
    if header_columns is None:
        if data_rows:
            print("[GMoE] Missing header row; expected a '#'-prefixed header with known column names.")
        return tasks

    for line_no, parts in data_rows:
        task = parse_row_by_header(
            parts, header_columns, line_no, "[GMoE]",
            required_columns=_REQUIRED_COLUMNS,
            defaults=_DEFAULTS,
            custom_parser=_gmoe_custom_parser,
        )
        if task is not None:
            tasks.append(task)
    return dedup_tasks(tasks, _task_identity)


def _build_task_cfg(base_cfg, task: dict[str, Any]):
    run_cfg = base_cfg.clone()
    gmoe_cfg = run_cfg.moe.gmoe
    explicit_keys = list(_IDENTITY_KEYS)

    gmoe_cfg.dataset.name = task["dataset"]
    gmoe_cfg.dataset.task_level = task["task_level"]
    gmoe_cfg.dataset.induced = bool(task["induced"])
    gmoe_cfg.dataset.num_classes = None
    gmoe_cfg.dataset.label_dim = None
    gmoe_cfg.in_dim = 0
    gmoe_cfg.run_tasks_tsv = False

    set_cfg_field_and_track(gmoe_cfg.dataset, "task_type", task.get("task_type"), "moe.gmoe.dataset.task_type", explicit_keys)
    set_cfg_field_and_track(gmoe_cfg.dataset, "fixed_split", task.get("fixed_split"), "moe.gmoe.dataset.fixed_split", explicit_keys)
    set_cfg_field_and_track(gmoe_cfg, "gnn_type", task.get("gnn_type"), "moe.gmoe.gnn_type", explicit_keys)
    set_cfg_field_and_track(gmoe_cfg, "num_experts", task.get("num_experts"), "moe.gmoe.num_experts", explicit_keys)
    set_cfg_field_and_track(gmoe_cfg, "num_experts_1hop", task.get("num_experts_1hop"), "moe.gmoe.num_experts_1hop", explicit_keys)
    set_cfg_field_and_track(gmoe_cfg, "k", task.get("k"), "moe.gmoe.k", explicit_keys)
    set_cfg_field_and_track(gmoe_cfg, "expert_hop", task.get("expert_hop"), "moe.gmoe.expert_hop", explicit_keys)
    set_cfg_field_and_track(gmoe_cfg, "epochs", task.get("epochs"), "moe.gmoe.epochs", explicit_keys)
    set_cfg_field_and_track(gmoe_cfg, "batch_size", task.get("batch"), "moe.gmoe.batch_size", explicit_keys)
    set_cfg_field_and_track(gmoe_cfg, "skip_if_exists", task.get("skip_if_exists"), "moe.gmoe.skip_if_exists", explicit_keys)

    set_explicit_cfg_keys(run_cfg, explicit_keys)
    return run_cfg


# ---------------------------------------------------------------------------
# Multi-run execution + result saving
# ---------------------------------------------------------------------------

def _run_seeds(base_cfg) -> bool:
    """Train one config over all resolved seeds and append a result row."""
    started_at = datetime.now().astimezone()
    requested_runs = int(getattr(base_cfg.moe.gmoe, "num_runs", 0) or 0)
    seeds: list[int] = []
    run_metrics: list[dict[str, float]] = []
    runners: list[GMoERunner] = []
    try:
        seeds = resolve_seeds(base_cfg, requested_count=requested_runs)
    except ValueError as exc:
        print(f"[GMoE][Multi-run] {exc}")
        return False

    total_runs = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        run_cfg = base_cfg.clone()
        run_cfg.seed = int(seed)
        if total_runs > 1:
            print(f"[GMoE][Multi-run] Running {index}/{total_runs} with seed={seed}")
        set_seed(run_cfg.seed)
        runner = GMoERunner(cfg=run_cfg)
        runner.fit()
        runners.append(runner)
        run_metrics.append(collect_run_metrics(runner, log_prefix="[GMoE][Summary]"))

    summarize_runs(run_metrics, seeds, log_prefix="[GMoE][Summary]")
    ended_at = datetime.now().astimezone()
    _append_gmoe_result(base_cfg, runners, run_metrics, started_at, ended_at, seeds)
    return True


def _append_gmoe_result(cfg, runners, run_metrics, started_at, ended_at, seeds) -> None:
    if not should_save_result(runners, cfg):
        return
    summary = aggregate_run_metrics(run_metrics)
    append_workflow_result(
        cfg=cfg,
        # GMoE rows go into the per-method MoE results table
        # (outputs/results/moe_gmoe.tsv).
        workflow="moe_gmoe",
        started_at=started_at,
        ended_at=ended_at,
        checkpoint_save_paths=collect_checkpoint_paths(runners),
        seeds=seeds,
        best_epochs=summary["epoch_values"].get("best_epoch"),
        metric_summary=summary["metric_stats"],
    )


def _run_tasks_tsv(cfg) -> int:
    tasks = parse_gmoe_tasks(getattr(cfg.moe.gmoe, "tasks_tsv", ""))
    if not tasks:
        print("[GMoE] No tasks found to run.")
        return 1

    results: list[bool] = []
    for task in tasks:
        base_cfg = _build_task_cfg(cfg, task)
        try:
            results.append(_run_seeds(base_cfg))
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[GMoE] Failed {task['dataset']} (task_level={task['task_level']}, "
                f"induced={task['induced']}): {exc}"
            )
            results.append(False)
    return 0 if all(results) else 1


def run_gmoe(cfg) -> int:
    """Execute one or more GMoE training runs for the provided config."""
    if bool(getattr(cfg.moe.gmoe, "run_tasks_tsv", False)):
        return _run_tasks_tsv(cfg)

    # Single-config path: seed identity keys so the result row is self-describing.
    explicit_keys = get_explicit_cfg_keys(cfg)
    for key in _IDENTITY_KEYS:
        if key not in explicit_keys:
            explicit_keys.append(key)
    set_explicit_cfg_keys(cfg, explicit_keys)

    return 0 if _run_seeds(cfg) else 1


__all__ = ["run_gmoe", "parse_gmoe_tasks"]
