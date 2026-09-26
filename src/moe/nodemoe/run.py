"""Runtime orchestration for the Node-MoE MoE method.

Dispatched from ``src.moe.run`` when ``moe.method == 'nodemoe'``. Mirrors
``src.moe.gmoe.run``: single-config multi-seed runs from
``cfg.moe.nodemoe.dataset`` or batch execution over a header-based TSV
(``cfg.moe.nodemoe.run_tasks_tsv``). Each config trains
:class:`NodeMoERunner` over ``cfg.moe.nodemoe.num_runs`` seeds and appends
one aggregated row to ``outputs/results/moe_nodemoe.tsv``.
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

from .chebnet2 import FILTER_INITS
from .task import READOUTS
from .trainer import NodeMoERunner

# Config keys recorded as result-table columns (``moe.method`` first so the
# table carries the producing method).
_IDENTITY_KEYS = [
    "moe.method",
    "moe.nodemoe.dataset.name",
    "moe.nodemoe.dataset.task_level",
    "moe.nodemoe.dataset.induced",
    # The support budget distinguishes the 5-/100-shot rows even for
    # single-config CLI runs (run_moe.py records no explicit keys).
    "moe.nodemoe.dataset.fixed_split",
    "moe.nodemoe.expert_inits",
    "moe.nodemoe.expert_alphas",
    "moe.nodemoe.K",
    "moe.nodemoe.expert_hidden_dim",
    "moe.nodemoe.gate_hidden_dim",
    "moe.nodemoe.smoothing_gamma",
    "moe.nodemoe.readout",
    "moe.nodemoe.epochs",
    "moe.nodemoe.expert_lr",
    "moe.nodemoe.filter_lr",
    "moe.nodemoe.gate_lr",
    "moe.nodemoe.batch_size",
]


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------

# Header tokens are lower-cased by the reader, so the ``K`` column is ``k``.
_HEADER_COLUMNS = {
    "dataset", "task_level", "task_type", "induced", "expert_inits", "k",
    "readout", "fixed_split", "epochs", "batch", "skip_if_exists",
}
_REQUIRED_COLUMNS = ("dataset", "task_level", "induced")
_DEFAULTS = {
    "task_type": None,
    "expert_inits": None,
    "k": None,
    "readout": None,
    "fixed_split": None,
    "epochs": None,
    "batch": None,
    "skip_if_exists": None,
}


def _nodemoe_custom_parser(col: str, val: str, line_no: int):
    """Parse Node-MoE-specific columns; fall through for the rest."""
    if col == "expert_inits":
        inits = tuple(token.strip().lower() for token in val.split(",") if token.strip())
        if len(inits) < 2 or any(kind not in FILTER_INITS for kind in inits):
            print(
                f"[NodeMoE] Skipping malformed task row {line_no}: invalid expert_inits '{val}' "
                f"(comma-separated, >= 2 of {FILTER_INITS})"
            )
            return None, False
        return inits, True
    if col == "k":
        try:
            return int(val), True
        except ValueError:
            print(f"[NodeMoE] Skipping malformed task row {line_no}: invalid K '{val}'")
            return None, False
    if col == "readout":
        readout = val.strip().lower()
        if readout not in READOUTS:
            print(f"[NodeMoE] Skipping malformed task row {line_no}: invalid readout '{val}'")
            return None, False
        return readout, True
    return None


def _task_identity(task: dict[str, Any]) -> tuple:
    return (
        task["dataset"],
        task["task_level"],
        task.get("task_type"),
        bool(task["induced"]),
        task.get("expert_inits"),
        task.get("k"),
        task.get("readout"),
        task.get("fixed_split"),
        task.get("epochs"),
        task.get("batch"),
        task.get("skip_if_exists"),
    )


def parse_nodemoe_tasks(tsv_path: str) -> list[dict[str, Any]]:
    """Read Node-MoE tasks from a header-based TSV."""
    header_columns, data_rows = read_tsv_rows(
        tsv_path, _HEADER_COLUMNS, min_header_columns=3, log_prefix="[NodeMoE]",
    )
    tasks: list[dict[str, Any]] = []
    if header_columns is None:
        if data_rows:
            print("[NodeMoE] Missing header row; expected a '#'-prefixed header with known column names.")
        return tasks

    for line_no, parts in data_rows:
        task = parse_row_by_header(
            parts, header_columns, line_no, "[NodeMoE]",
            required_columns=_REQUIRED_COLUMNS,
            defaults=_DEFAULTS,
            custom_parser=_nodemoe_custom_parser,
        )
        if task is not None:
            tasks.append(task)
    return dedup_tasks(tasks, _task_identity)


def _build_task_cfg(base_cfg, task: dict[str, Any]):
    run_cfg = base_cfg.clone()
    nodemoe_cfg = run_cfg.moe.nodemoe
    explicit_keys = list(_IDENTITY_KEYS)

    nodemoe_cfg.dataset.name = task["dataset"]
    nodemoe_cfg.dataset.task_level = task["task_level"]
    nodemoe_cfg.dataset.induced = bool(task["induced"])
    nodemoe_cfg.dataset.num_classes = None
    nodemoe_cfg.dataset.label_dim = None
    nodemoe_cfg.in_dim = 0
    nodemoe_cfg.run_tasks_tsv = False

    set_cfg_field_and_track(nodemoe_cfg.dataset, "task_type", task.get("task_type"), "moe.nodemoe.dataset.task_type", explicit_keys)
    set_cfg_field_and_track(nodemoe_cfg.dataset, "fixed_split", task.get("fixed_split"), "moe.nodemoe.dataset.fixed_split", explicit_keys)
    set_cfg_field_and_track(nodemoe_cfg, "expert_inits", task.get("expert_inits"), "moe.nodemoe.expert_inits", explicit_keys)
    set_cfg_field_and_track(nodemoe_cfg, "K", task.get("k"), "moe.nodemoe.K", explicit_keys)
    set_cfg_field_and_track(nodemoe_cfg, "readout", task.get("readout"), "moe.nodemoe.readout", explicit_keys)
    set_cfg_field_and_track(nodemoe_cfg, "epochs", task.get("epochs"), "moe.nodemoe.epochs", explicit_keys)
    set_cfg_field_and_track(nodemoe_cfg, "batch_size", task.get("batch"), "moe.nodemoe.batch_size", explicit_keys)
    set_cfg_field_and_track(nodemoe_cfg, "skip_if_exists", task.get("skip_if_exists"), "moe.nodemoe.skip_if_exists", explicit_keys)

    set_explicit_cfg_keys(run_cfg, explicit_keys)
    return run_cfg


# ---------------------------------------------------------------------------
# Multi-run execution + result saving
# ---------------------------------------------------------------------------

def _run_seeds(base_cfg) -> bool:
    """Train one config over all resolved seeds and append a result row."""
    started_at = datetime.now().astimezone()
    requested_runs = int(getattr(base_cfg.moe.nodemoe, "num_runs", 0) or 0)
    run_metrics: list[dict[str, float]] = []
    runners: list[NodeMoERunner] = []
    try:
        seeds = resolve_seeds(base_cfg, requested_count=requested_runs)
    except ValueError as exc:
        print(f"[NodeMoE][Multi-run] {exc}")
        return False

    total_runs = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        run_cfg = base_cfg.clone()
        run_cfg.seed = int(seed)
        if total_runs > 1:
            print(f"[NodeMoE][Multi-run] Running {index}/{total_runs} with seed={seed}")
        set_seed(run_cfg.seed)
        runner = NodeMoERunner(cfg=run_cfg)
        runner.fit()
        runners.append(runner)
        run_metrics.append(collect_run_metrics(runner, log_prefix="[NodeMoE][Summary]"))

    summarize_runs(run_metrics, seeds, log_prefix="[NodeMoE][Summary]")
    ended_at = datetime.now().astimezone()
    _append_nodemoe_result(base_cfg, runners, run_metrics, started_at, ended_at, seeds)
    return True


def _append_nodemoe_result(cfg, runners, run_metrics, started_at, ended_at, seeds) -> None:
    if not should_save_result(runners, cfg):
        return
    summary = aggregate_run_metrics(run_metrics)
    append_workflow_result(
        cfg=cfg,
        workflow="moe_nodemoe",
        started_at=started_at,
        ended_at=ended_at,
        checkpoint_save_paths=collect_checkpoint_paths(runners),
        seeds=seeds,
        best_epochs=summary["epoch_values"].get("best_epoch"),
        metric_summary=summary["metric_stats"],
    )


def _run_tasks_tsv(cfg) -> int:
    tasks = parse_nodemoe_tasks(getattr(cfg.moe.nodemoe, "tasks_tsv", ""))
    if not tasks:
        print("[NodeMoE] No tasks found to run.")
        return 1

    results: list[bool] = []
    for task in tasks:
        base_cfg = _build_task_cfg(cfg, task)
        try:
            results.append(_run_seeds(base_cfg))
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[NodeMoE] Failed {task['dataset']} (task_level={task['task_level']}, "
                f"induced={task['induced']}): {exc}"
            )
            results.append(False)
    return 0 if all(results) else 1


def run_nodemoe(cfg) -> int:
    """Execute one or more Node-MoE training runs for the provided config."""
    if bool(getattr(cfg.moe.nodemoe, "run_tasks_tsv", False)):
        return _run_tasks_tsv(cfg)

    # Single-config path: seed identity keys so the result row is self-describing.
    explicit_keys = get_explicit_cfg_keys(cfg)
    for key in _IDENTITY_KEYS:
        if key not in explicit_keys:
            explicit_keys.append(key)
    set_explicit_cfg_keys(cfg, explicit_keys)

    return 0 if _run_seeds(cfg) else 1


__all__ = ["run_nodemoe", "parse_nodemoe_tasks"]
