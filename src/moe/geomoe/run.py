"""Runtime orchestration for the GeoMoE MoE method.

Dispatched from ``src.moe.run`` when ``moe.method == 'geomoe'``. Supports
single-config multi-seed runs from ``cfg.moe.geomoe.dataset`` (default) and
batch execution over a header-based TSV (``cfg.moe.geomoe.run_tasks_tsv``),
one config per row; a ``split_root`` column selects the shift condition
(``data_preparation.dataset.split_root``). Each config trains
:class:`GeoMoERunner` over ``num_runs`` seeds and appends an aggregated row
(including ``test_brier``) to ``outputs/results/moe_geomoe.tsv``. Mirrors
``src.moe.graphmore.run``.
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

from .trainer import GeoMoERunner

# Result-table columns; the split root identifies the shift condition.
_IDENTITY_KEYS = [
    "moe.method",
    "moe.geomoe.dataset.name",
    "moe.geomoe.dataset.task_level",
    "moe.geomoe.dataset.induced",
    "moe.geomoe.dataset.fixed_split",
    "data_preparation.dataset.split_root",
    "moe.geomoe.hidden_dim",
    "moe.geomoe.num_layers",
    "moe.geomoe.num_negatives",
    "moe.geomoe.theta",
    "moe.geomoe.eta",
    "moe.geomoe.epochs",
    "moe.geomoe.lr",
    "moe.geomoe.batch_size",
]


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------

_HEADER_COLUMNS = {
    "dataset", "task_level", "task_type", "induced",
    "fixed_split", "split_root", "epochs", "batch", "skip_if_exists",
}
_REQUIRED_COLUMNS = ("dataset", "task_level", "induced")
_DEFAULTS = {
    "task_type": None,
    "fixed_split": None,
    "split_root": None,
    "epochs": None,
    "batch": None,
    "skip_if_exists": None,
}


def _geomoe_custom_parser(col: str, val: str, line_no: int):
    """``split_root``: ``-`` keeps the configured root; fall through for the rest."""
    if col == "split_root":
        value = val.strip()
        return (None if value in ("", "-") else value), True
    return None


def _task_identity(task: dict[str, Any]) -> tuple:
    return (
        task["dataset"],
        task["task_level"],
        task.get("task_type"),
        bool(task["induced"]),
        task.get("fixed_split"),
        task.get("split_root"),
        task.get("epochs"),
        task.get("batch"),
        task.get("skip_if_exists"),
    )


def parse_geomoe_tasks(tsv_path: str) -> list[dict[str, Any]]:
    """Read GeoMoE tasks from a header-based TSV."""
    header_columns, data_rows = read_tsv_rows(
        tsv_path, _HEADER_COLUMNS, min_header_columns=3, log_prefix="[GeoMoE]",
    )
    tasks: list[dict[str, Any]] = []
    if header_columns is None:
        if data_rows:
            print("[GeoMoE] Missing header row; expected a '#'-prefixed header with known column names.")
        return tasks

    for line_no, parts in data_rows:
        task = parse_row_by_header(
            parts, header_columns, line_no, "[GeoMoE]",
            required_columns=_REQUIRED_COLUMNS,
            defaults=_DEFAULTS,
            custom_parser=_geomoe_custom_parser,
        )
        if task is not None:
            tasks.append(task)
    return dedup_tasks(tasks, _task_identity)


def _build_task_cfg(base_cfg, task: dict[str, Any]):
    run_cfg = base_cfg.clone()
    g = run_cfg.moe.geomoe
    explicit_keys = list(_IDENTITY_KEYS)

    g.dataset.name = task["dataset"]
    g.dataset.task_level = task["task_level"]
    g.dataset.induced = bool(task["induced"])
    g.dataset.num_classes = None
    g.dataset.label_dim = None
    g.in_dim = 0
    g.run_tasks_tsv = False

    set_cfg_field_and_track(g.dataset, "task_type", task.get("task_type"), "moe.geomoe.dataset.task_type", explicit_keys)
    set_cfg_field_and_track(g.dataset, "fixed_split", task.get("fixed_split"), "moe.geomoe.dataset.fixed_split", explicit_keys)
    set_cfg_field_and_track(
        run_cfg.data_preparation.dataset, "split_root", task.get("split_root"),
        "data_preparation.dataset.split_root", explicit_keys,
    )
    set_cfg_field_and_track(g, "epochs", task.get("epochs"), "moe.geomoe.epochs", explicit_keys)
    set_cfg_field_and_track(g, "batch_size", task.get("batch"), "moe.geomoe.batch_size", explicit_keys)
    set_cfg_field_and_track(g, "skip_if_exists", task.get("skip_if_exists"), "moe.geomoe.skip_if_exists", explicit_keys)

    set_explicit_cfg_keys(run_cfg, explicit_keys)
    return run_cfg


# ---------------------------------------------------------------------------
# Multi-run execution + result saving
# ---------------------------------------------------------------------------

def _run_seeds(base_cfg) -> bool:
    """Train one config over all resolved seeds and append a result row."""
    started_at = datetime.now().astimezone()
    requested_runs = int(getattr(base_cfg.moe.geomoe, "num_runs", 0) or 0)
    run_metrics: list[dict[str, float]] = []
    runners: list[GeoMoERunner] = []
    try:
        seeds = resolve_seeds(base_cfg, requested_count=requested_runs)
    except ValueError as exc:
        print(f"[GeoMoE][Multi-run] {exc}")
        return False

    total_runs = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        run_cfg = base_cfg.clone()
        run_cfg.seed = int(seed)
        if total_runs > 1:
            print(f"[GeoMoE][Multi-run] Running {index}/{total_runs} with seed={seed}")
        set_seed(run_cfg.seed)
        runner = GeoMoERunner(cfg=run_cfg)
        runner.fit()
        runners.append(runner)
        run_metrics.append(collect_run_metrics(runner, log_prefix="[GeoMoE][Summary]"))

    summarize_runs(run_metrics, seeds, log_prefix="[GeoMoE][Summary]")
    ended_at = datetime.now().astimezone()
    if should_save_result(runners, base_cfg):
        summary = aggregate_run_metrics(run_metrics)
        append_workflow_result(
            cfg=base_cfg,
            workflow="moe_geomoe",
            started_at=started_at,
            ended_at=ended_at,
            checkpoint_save_paths=collect_checkpoint_paths(runners),
            seeds=seeds,
            best_epochs=summary["epoch_values"].get("best_epoch"),
            metric_summary=summary["metric_stats"],
        )
    return True


def _run_tasks_tsv(cfg) -> int:
    tasks = parse_geomoe_tasks(getattr(cfg.moe.geomoe, "tasks_tsv", ""))
    if not tasks:
        print("[GeoMoE] No tasks found to run.")
        return 1

    results: list[bool] = []
    for task in tasks:
        try:
            results.append(_run_seeds(_build_task_cfg(cfg, task)))
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[GeoMoE] Failed {task['dataset']} (task_level={task['task_level']}, "
                f"induced={task['induced']}, split_root={task.get('split_root')}): {exc}"
            )
            results.append(False)
    return 0 if all(results) else 1


def run_geomoe(cfg) -> int:
    """Execute one or more GeoMoE training runs for the provided config."""
    if bool(getattr(cfg.moe.geomoe, "run_tasks_tsv", False)):
        return _run_tasks_tsv(cfg)

    # Single-config path: seed identity keys so the result row is self-describing.
    explicit_keys = get_explicit_cfg_keys(cfg)
    for key in _IDENTITY_KEYS:
        if key not in explicit_keys:
            explicit_keys.append(key)
    set_explicit_cfg_keys(cfg, explicit_keys)
    return 0 if _run_seeds(cfg) else 1


__all__ = ["run_geomoe", "parse_geomoe_tasks"]
