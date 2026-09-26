"""Runtime orchestration for the Link-MoE MoE method.

Dispatched from ``src.moe.run`` when ``moe.method == 'linkmoe'``. Supports
single-config multi-seed runs from ``cfg.moe.linkmoe.dataset`` settings
(default) and batch execution over a header-based TSV
(``cfg.moe.linkmoe.run_tasks_tsv == True``), one config per row. Each config
runs :class:`LinkMoERunner` over ``cfg.moe.linkmoe.num_runs`` seeds and
appends one aggregated row to ``outputs/results/moe_linkmoe.tsv``. Only link
prediction (``task_level == 'edge'``) is accepted.
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

from .trainer import LinkMoERunner, validate_link_task

_IDENTITY_KEYS = [
    "moe.method",
    "moe.linkmoe.dataset.name",
    "moe.linkmoe.dataset.task_level",
    "moe.linkmoe.dataset.fixed_split",
    "moe.linkmoe.experts",
    "moe.linkmoe.expert_max_epochs",
    "moe.linkmoe.expert_patience",
    "moe.linkmoe.gate.hidden_dim",
    "moe.linkmoe.gate.num_layers",
    "moe.linkmoe.gate.num_layers_predictor",
    "moe.linkmoe.gate.dropout",
    "moe.linkmoe.gate.lr",
    "moe.linkmoe.gate.epochs",
    "moe.linkmoe.gate.val_train_ratio",
    "moe.linkmoe.gate.neg_loss_weight",
]


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------

_HEADER_COLUMNS = {
    "dataset", "task_level", "task_type", "experts", "fixed_split", "gate_epochs", "skip_if_exists",
}
_REQUIRED_COLUMNS = ("dataset", "task_level")
_DEFAULTS = {
    "task_type": None,
    "experts": None,
    "fixed_split": None,
    "gate_epochs": None,
    "skip_if_exists": None,
}


def _linkmoe_custom_parser(col: str, val: str, line_no: int):
    """Parse ``experts`` (comma-separated) and ``gate_epochs`` (``-`` = default); fall through for the rest."""
    if col in ("experts", "gate_epochs") and val == "-":
        return None, True
    if col == "experts":
        names = tuple(part.strip().lower() for part in val.split(",") if part.strip())
        if not names:
            print(f"[LinkMoE] Skipping malformed task row {line_no}: empty experts '{val}'")
            return None, False
        return names, True
    if col == "gate_epochs":
        try:
            return int(val), True
        except ValueError:
            print(f"[LinkMoE] Skipping malformed task row {line_no}: invalid gate_epochs '{val}'")
            return None, False
    return None


def _task_identity(task: dict[str, Any]) -> tuple:
    return (
        task["dataset"],
        task["task_level"],
        task.get("task_type"),
        task.get("experts"),
        task.get("fixed_split"),
        task.get("gate_epochs"),
        task.get("skip_if_exists"),
    )


def parse_linkmoe_tasks(tsv_path: str) -> list[dict[str, Any]]:
    """Read Link-MoE tasks from a header-based TSV."""
    header_columns, data_rows = read_tsv_rows(
        tsv_path, _HEADER_COLUMNS, min_header_columns=2, log_prefix="[LinkMoE]",
    )
    tasks: list[dict[str, Any]] = []
    if header_columns is None:
        if data_rows:
            print("[LinkMoE] Missing header row; expected a '#'-prefixed header with known column names.")
        return tasks

    for line_no, parts in data_rows:
        task = parse_row_by_header(
            parts, header_columns, line_no, "[LinkMoE]",
            required_columns=_REQUIRED_COLUMNS,
            defaults=_DEFAULTS,
            custom_parser=_linkmoe_custom_parser,
        )
        if task is not None:
            tasks.append(task)
    return dedup_tasks(tasks, _task_identity)


def _build_task_cfg(base_cfg, task: dict[str, Any]):
    run_cfg = base_cfg.clone()
    lcfg = run_cfg.moe.linkmoe
    explicit_keys = list(_IDENTITY_KEYS)

    lcfg.dataset.name = task["dataset"]
    lcfg.dataset.task_level = task["task_level"]
    lcfg.run_tasks_tsv = False

    set_cfg_field_and_track(lcfg.dataset, "task_type", task.get("task_type"), "moe.linkmoe.dataset.task_type", explicit_keys)
    set_cfg_field_and_track(lcfg.dataset, "fixed_split", task.get("fixed_split"), "moe.linkmoe.dataset.fixed_split", explicit_keys)
    set_cfg_field_and_track(lcfg, "experts", task.get("experts"), "moe.linkmoe.experts", explicit_keys)
    set_cfg_field_and_track(lcfg.gate, "epochs", task.get("gate_epochs"), "moe.linkmoe.gate.epochs", explicit_keys)
    set_cfg_field_and_track(lcfg, "skip_if_exists", task.get("skip_if_exists"), "moe.linkmoe.skip_if_exists", explicit_keys)

    set_explicit_cfg_keys(run_cfg, explicit_keys)
    return run_cfg


# ---------------------------------------------------------------------------
# Multi-run execution + result saving
# ---------------------------------------------------------------------------

def _run_seeds(base_cfg) -> bool:
    """Run one config over all resolved seeds and append a result row."""
    validate_link_task(base_cfg.moe.linkmoe.dataset)  # reject non-link tasks before any work
    started_at = datetime.now().astimezone()
    try:
        seeds = resolve_seeds(base_cfg, requested_count=int(getattr(base_cfg.moe.linkmoe, "num_runs", 0) or 0))
    except ValueError as exc:
        print(f"[LinkMoE][Multi-run] {exc}")
        return False

    runners: list[LinkMoERunner] = []
    run_metrics: list[dict[str, float]] = []
    for index, seed in enumerate(seeds, start=1):
        run_cfg = base_cfg.clone()
        run_cfg.seed = int(seed)
        if len(seeds) > 1:
            print(f"[LinkMoE][Multi-run] Running {index}/{len(seeds)} with seed={seed}")
        set_seed(run_cfg.seed)
        runner = LinkMoERunner(cfg=run_cfg)
        runner.fit()
        runners.append(runner)
        run_metrics.append(collect_run_metrics(runner, log_prefix="[LinkMoE][Summary]"))

    summarize_runs(run_metrics, seeds, log_prefix="[LinkMoE][Summary]")
    ended_at = datetime.now().astimezone()
    if should_save_result(runners, base_cfg):
        summary = aggregate_run_metrics(run_metrics)
        append_workflow_result(
            cfg=base_cfg,
            workflow="moe_linkmoe",
            started_at=started_at,
            ended_at=ended_at,
            checkpoint_save_paths=collect_checkpoint_paths(runners),
            seeds=seeds,
            best_epochs=summary["epoch_values"].get("best_epoch"),
            metric_summary=summary["metric_stats"],
        )
    return True


def _run_tasks_tsv(cfg) -> int:
    tasks = parse_linkmoe_tasks(getattr(cfg.moe.linkmoe, "tasks_tsv", ""))
    if not tasks:
        print("[LinkMoE] No tasks found to run.")
        return 1

    results: list[bool] = []
    for task in tasks:
        try:
            results.append(_run_seeds(_build_task_cfg(cfg, task)))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[LinkMoE] Failed {task['dataset']} (task_level={task['task_level']}): {exc}")
            results.append(False)
    return 0 if all(results) else 1


def run_linkmoe(cfg) -> int:
    """Execute one or more Link-MoE runs for the provided config."""
    if bool(getattr(cfg.moe.linkmoe, "run_tasks_tsv", False)):
        return _run_tasks_tsv(cfg)

    explicit_keys = get_explicit_cfg_keys(cfg)
    for key in _IDENTITY_KEYS:
        if key not in explicit_keys:
            explicit_keys.append(key)
    set_explicit_cfg_keys(cfg, explicit_keys)
    return 0 if _run_seeds(cfg) else 1


__all__ = ["run_linkmoe", "parse_linkmoe_tasks"]
