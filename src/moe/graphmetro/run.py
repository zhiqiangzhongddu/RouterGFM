"""Runtime orchestration for the GraphMETRO MoE method.

Dispatched from ``src.moe.run`` when ``moe.method == 'graphmetro'``. Each config
(from ``cfg.moe.graphmetro.dataset`` or, with ``run_tasks_tsv``, one per TSV
row) trains :class:`GraphMETRORunner` over ``num_runs`` seeds and appends an
aggregated row (task metric + ``test_brier``) to ``outputs/results/moe_graphmetro.tsv``.
The TSV ``split_root`` column selects the standard or a shift split root
(``data_preparation.dataset.split_root``).
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

from .trainer import GraphMETRORunner

# Result-table columns; the split root identifies the shift condition.
_IDENTITY_KEYS = [
    "moe.method",
    "moe.graphmetro.dataset.name",
    "moe.graphmetro.dataset.task_level",
    "moe.graphmetro.dataset.induced",
    "moe.graphmetro.dataset.fixed_split",
    "data_preparation.dataset.split_root",
    "moe.graphmetro.backbone",
    "moe.graphmetro.hidden_dim",
    "moe.graphmetro.num_layers",
    "moe.graphmetro.align_lambda",
    "moe.graphmetro.num_shift_samples",
    "moe.graphmetro.moe_lr",
    "moe.graphmetro.classifier_lr",
    "moe.graphmetro.epochs",
    "moe.graphmetro.batch_size",
]


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------

_HEADER_COLUMNS = {
    "dataset", "task_level", "task_type", "induced", "fixed_split", "split_root",
    "backbone", "moe_lr", "epochs", "batch", "skip_if_exists",
}
_REQUIRED_COLUMNS = ("dataset", "task_level", "induced")
_DEFAULTS = {
    "task_type": None,
    "fixed_split": None,
    "split_root": None,
    "backbone": None,
    "moe_lr": None,
    "epochs": None,
    "batch": None,
    "skip_if_exists": None,
}


def _graphmetro_custom_parser(col: str, val: str, line_no: int):
    """``-`` means "config default" for the method columns; ``moe_lr`` is a float."""
    if col in {"split_root", "backbone", "moe_lr"} and val == "-":
        return None, True
    if col == "moe_lr":
        try:
            return float(val), True
        except ValueError:
            print(f"[GraphMETRO] Skipping malformed task row {line_no}: invalid moe_lr '{val}'")
            return None, False
    return None


def _task_identity(task: dict[str, Any]) -> tuple:
    return tuple(task.get(key) for key in (
        "dataset", "task_level", "task_type", "induced", "fixed_split", "split_root",
        "backbone", "moe_lr", "epochs", "batch", "skip_if_exists",
    ))


def parse_graphmetro_tasks(tsv_path: str) -> list[dict[str, Any]]:
    """Read GraphMETRO tasks from a header-based TSV."""
    header_columns, data_rows = read_tsv_rows(
        tsv_path, _HEADER_COLUMNS, min_header_columns=3, log_prefix="[GraphMETRO]",
    )
    tasks: list[dict[str, Any]] = []
    if header_columns is None:
        if data_rows:
            print("[GraphMETRO] Missing header row; expected a '#'-prefixed header with known column names.")
        return tasks

    for line_no, parts in data_rows:
        task = parse_row_by_header(
            parts, header_columns, line_no, "[GraphMETRO]",
            required_columns=_REQUIRED_COLUMNS,
            defaults=_DEFAULTS,
            custom_parser=_graphmetro_custom_parser,
        )
        if task is not None:
            tasks.append(task)
    return dedup_tasks(tasks, _task_identity)


def _build_task_cfg(base_cfg, task: dict[str, Any]):
    run_cfg = base_cfg.clone()
    gm_cfg = run_cfg.moe.graphmetro
    explicit_keys = list(_IDENTITY_KEYS)

    gm_cfg.dataset.name = task["dataset"]
    gm_cfg.dataset.task_level = task["task_level"]
    gm_cfg.dataset.induced = bool(task["induced"])
    gm_cfg.dataset.num_classes = None
    gm_cfg.dataset.label_dim = None
    gm_cfg.in_dim = 0
    gm_cfg.run_tasks_tsv = False

    set_cfg_field_and_track(gm_cfg.dataset, "task_type", task.get("task_type"), "moe.graphmetro.dataset.task_type", explicit_keys)
    set_cfg_field_and_track(gm_cfg.dataset, "fixed_split", task.get("fixed_split"), "moe.graphmetro.dataset.fixed_split", explicit_keys)
    set_cfg_field_and_track(run_cfg.data_preparation.dataset, "split_root", task.get("split_root"), "data_preparation.dataset.split_root", explicit_keys)
    set_cfg_field_and_track(gm_cfg, "backbone", task.get("backbone"), "moe.graphmetro.backbone", explicit_keys)
    set_cfg_field_and_track(gm_cfg, "moe_lr", task.get("moe_lr"), "moe.graphmetro.moe_lr", explicit_keys)
    set_cfg_field_and_track(gm_cfg, "epochs", task.get("epochs"), "moe.graphmetro.epochs", explicit_keys)
    set_cfg_field_and_track(gm_cfg, "batch_size", task.get("batch"), "moe.graphmetro.batch_size", explicit_keys)
    set_cfg_field_and_track(gm_cfg, "skip_if_exists", task.get("skip_if_exists"), "moe.graphmetro.skip_if_exists", explicit_keys)

    set_explicit_cfg_keys(run_cfg, explicit_keys)
    return run_cfg


# ---------------------------------------------------------------------------
# Multi-run execution + result saving
# ---------------------------------------------------------------------------

def _run_seeds(base_cfg) -> bool:
    """Train one config over all resolved seeds and append a result row."""
    started_at = datetime.now().astimezone()
    requested_runs = int(getattr(base_cfg.moe.graphmetro, "num_runs", 0) or 0)
    run_metrics: list[dict[str, float]] = []
    runners: list[GraphMETRORunner] = []
    try:
        seeds = resolve_seeds(base_cfg, requested_count=requested_runs)
    except ValueError as exc:
        print(f"[GraphMETRO][Multi-run] {exc}")
        return False

    total_runs = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        run_cfg = base_cfg.clone()
        run_cfg.seed = int(seed)
        if total_runs > 1:
            print(f"[GraphMETRO][Multi-run] Running {index}/{total_runs} with seed={seed}")
        set_seed(run_cfg.seed)
        runner = GraphMETRORunner(cfg=run_cfg)
        runner.fit()
        runners.append(runner)
        run_metrics.append(collect_run_metrics(runner, log_prefix="[GraphMETRO][Summary]"))

    summarize_runs(run_metrics, seeds, log_prefix="[GraphMETRO][Summary]")
    ended_at = datetime.now().astimezone()
    if should_save_result(runners, base_cfg):
        summary = aggregate_run_metrics(run_metrics)
        append_workflow_result(
            cfg=base_cfg,
            workflow="moe_graphmetro",
            started_at=started_at,
            ended_at=ended_at,
            checkpoint_save_paths=collect_checkpoint_paths(runners),
            seeds=seeds,
            best_epochs=summary["epoch_values"].get("best_epoch"),
            metric_summary=summary["metric_stats"],
        )
    return True


def _run_tasks_tsv(cfg) -> int:
    tasks = parse_graphmetro_tasks(getattr(cfg.moe.graphmetro, "tasks_tsv", ""))
    if not tasks:
        print("[GraphMETRO] No tasks found to run.")
        return 1

    results: list[bool] = []
    for task in tasks:
        try:
            results.append(_run_seeds(_build_task_cfg(cfg, task)))
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[GraphMETRO] Failed {task['dataset']} (task_level={task['task_level']}, "
                f"split={task.get('fixed_split')}, split_root={task.get('split_root')}): {exc}"
            )
            results.append(False)
    return 0 if all(results) else 1


def run_graphmetro(cfg) -> int:
    """Execute one or more GraphMETRO training runs for the provided config."""
    if bool(getattr(cfg.moe.graphmetro, "run_tasks_tsv", False)):
        return _run_tasks_tsv(cfg)

    # Single-config path: seed identity keys so the result row is self-describing.
    explicit_keys = get_explicit_cfg_keys(cfg)
    for key in _IDENTITY_KEYS:
        if key not in explicit_keys:
            explicit_keys.append(key)
    set_explicit_cfg_keys(cfg, explicit_keys)
    return 0 if _run_seeds(cfg) else 1


__all__ = ["run_graphmetro", "parse_graphmetro_tasks"]
