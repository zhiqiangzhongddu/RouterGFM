"""Runtime orchestration for the GMoPE MoE method.

Dispatched from ``src.moe.run`` when ``moe.method == 'gmope'``:

* ``moe.gmope.stage pretrain`` pretrains the route checkpoints listed in
  ``moe.gmope.pretrain.routes`` (target independent);
* otherwise (``all`` / ``finetune``) each config — from
  ``cfg.moe.gmope.dataset`` or, with ``run_tasks_tsv``, one per TSV row —
  prompt-tunes :class:`GMoPERunner` over ``num_runs`` seeds and appends an
  aggregated row to ``outputs/results/moe_gmope.tsv``.
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

from .pretrain import ROUTES, GMoPEPretrainer, resolve_num_experts, resolve_route, resolve_top_k
from .trainer import GMoPERunner

# Result-table columns; M and the top-K values are written back resolved.
_IDENTITY_KEYS = [
    "moe.method",
    "moe.gmope.dataset.name",
    "moe.gmope.dataset.task_level",
    "moe.gmope.dataset.induced",
    "moe.gmope.dataset.fixed_split",
    "moe.gmope.pretrain.objective",
    "moe.gmope.num_experts",
    "moe.gmope.prompt_dim",
    "moe.gmope.ortho_weight",
    "moe.gmope.tau",
    "moe.gmope.finetune.top_k",
    "moe.gmope.finetune.epochs",
    "moe.gmope.finetune.lr",
    "moe.gmope.finetune.batch_size",
]


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------

_HEADER_COLUMNS = {
    "dataset", "task_level", "task_type", "induced",
    "fixed_split", "epochs", "batch", "skip_if_exists",
}
_REQUIRED_COLUMNS = ("dataset", "task_level", "induced")
_DEFAULTS = {
    "task_type": None,
    "fixed_split": None,
    "epochs": None,
    "batch": None,
    "skip_if_exists": None,
}


def _task_identity(task: dict[str, Any]) -> tuple:
    return (
        task["dataset"],
        task["task_level"],
        task.get("task_type"),
        bool(task["induced"]),
        task.get("fixed_split"),
        task.get("epochs"),
        task.get("batch"),
        task.get("skip_if_exists"),
    )


def parse_gmope_tasks(tsv_path: str) -> list[dict[str, Any]]:
    """Read GMoPE tasks from a header-based TSV."""
    header_columns, data_rows = read_tsv_rows(
        tsv_path, _HEADER_COLUMNS, min_header_columns=3, log_prefix="[GMoPE]",
    )
    tasks: list[dict[str, Any]] = []
    if header_columns is None:
        if data_rows:
            print("[GMoPE] Missing header row; expected a '#'-prefixed header with known column names.")
        return tasks

    for line_no, parts in data_rows:
        task = parse_row_by_header(
            parts, header_columns, line_no, "[GMoPE]",
            required_columns=_REQUIRED_COLUMNS,
            defaults=_DEFAULTS,
        )
        if task is not None:
            tasks.append(task)
    return dedup_tasks(tasks, _task_identity)


def _resolve_sizes(cfg) -> None:
    """Write the resolved M and top-K values back so result rows are self-describing."""
    gmope_cfg = cfg.moe.gmope
    route = resolve_route(gmope_cfg.dataset.task_level)
    pretrain_k = resolve_top_k(gmope_cfg, route, "pretrain")
    finetune_k = resolve_top_k(gmope_cfg, route, "finetune")
    gmope_cfg.num_experts = resolve_num_experts(gmope_cfg, route)
    gmope_cfg.pretrain.top_k = pretrain_k
    gmope_cfg.finetune.top_k = finetune_k


def _build_task_cfg(base_cfg, task: dict[str, Any]):
    run_cfg = base_cfg.clone()
    gmope_cfg = run_cfg.moe.gmope
    explicit_keys = list(_IDENTITY_KEYS)

    gmope_cfg.dataset.name = task["dataset"]
    gmope_cfg.dataset.task_level = task["task_level"]
    gmope_cfg.dataset.induced = bool(task["induced"])
    gmope_cfg.dataset.num_classes = None
    gmope_cfg.dataset.label_dim = None
    gmope_cfg.run_tasks_tsv = False

    set_cfg_field_and_track(gmope_cfg.dataset, "task_type", task.get("task_type"), "moe.gmope.dataset.task_type", explicit_keys)
    set_cfg_field_and_track(gmope_cfg.dataset, "fixed_split", task.get("fixed_split"), "moe.gmope.dataset.fixed_split", explicit_keys)
    set_cfg_field_and_track(gmope_cfg.finetune, "epochs", task.get("epochs"), "moe.gmope.finetune.epochs", explicit_keys)
    set_cfg_field_and_track(gmope_cfg.finetune, "batch_size", task.get("batch"), "moe.gmope.finetune.batch_size", explicit_keys)
    set_cfg_field_and_track(gmope_cfg, "skip_if_exists", task.get("skip_if_exists"), "moe.gmope.skip_if_exists", explicit_keys)

    _resolve_sizes(run_cfg)
    set_explicit_cfg_keys(run_cfg, explicit_keys)
    return run_cfg


# ---------------------------------------------------------------------------
# Multi-run execution + result saving
# ---------------------------------------------------------------------------

def _run_seeds(base_cfg) -> bool:
    """Train one config over all resolved seeds and append a result row."""
    started_at = datetime.now().astimezone()
    requested_runs = int(getattr(base_cfg.moe.gmope, "num_runs", 0) or 0)
    run_metrics: list[dict[str, float]] = []
    runners: list[GMoPERunner] = []
    try:
        seeds = resolve_seeds(base_cfg, requested_count=requested_runs)
    except ValueError as exc:
        print(f"[GMoPE][Multi-run] {exc}")
        return False

    total_runs = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        run_cfg = base_cfg.clone()
        run_cfg.seed = int(seed)
        if total_runs > 1:
            print(f"[GMoPE][Multi-run] Running {index}/{total_runs} with seed={seed}")
        set_seed(run_cfg.seed)
        runner = GMoPERunner(cfg=run_cfg)
        runner.fit()
        runners.append(runner)
        run_metrics.append(collect_run_metrics(runner, log_prefix="[GMoPE][Summary]"))

    summarize_runs(run_metrics, seeds, log_prefix="[GMoPE][Summary]")
    ended_at = datetime.now().astimezone()
    if should_save_result(runners, base_cfg):
        summary = aggregate_run_metrics(run_metrics)
        append_workflow_result(
            cfg=base_cfg,
            workflow="moe_gmope",
            started_at=started_at,
            ended_at=ended_at,
            checkpoint_save_paths=collect_checkpoint_paths(runners),
            seeds=seeds,
            best_epochs=summary["epoch_values"].get("best_epoch"),
            metric_summary=summary["metric_stats"],
        )
    return True


def _run_tasks_tsv(cfg) -> int:
    tasks = parse_gmope_tasks(getattr(cfg.moe.gmope, "tasks_tsv", ""))
    if not tasks:
        print("[GMoPE] No tasks found to run.")
        return 1

    results: list[bool] = []
    for task in tasks:
        try:
            results.append(_run_seeds(_build_task_cfg(cfg, task)))
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[GMoPE] Failed {task['dataset']} (task_level={task['task_level']}, "
                f"induced={task['induced']}): {exc}"
            )
            results.append(False)
    return 0 if all(results) else 1


def _run_pretrain(cfg) -> int:
    routes = [str(route).lower() for route in cfg.moe.gmope.pretrain.routes]
    unknown = [route for route in routes if route not in ROUTES]
    if unknown or not routes:
        print(f"[GMoPE] moe.gmope.pretrain.routes must be a non-empty subset of {ROUTES} (got {routes}).")
        return 1
    for route in routes:
        GMoPEPretrainer(cfg, route).fit()
    return 0


def run_gmope(cfg) -> int:
    """Execute GMoPE pretraining or one or more prompt-tuning runs for ``cfg``."""
    stage = str(cfg.moe.gmope.stage).lower()
    if stage == "pretrain":
        return _run_pretrain(cfg)
    if stage not in {"all", "finetune"}:
        print(f"[GMoPE] Unknown moe.gmope.stage '{stage}' (all | pretrain | finetune).")
        return 1
    if bool(getattr(cfg.moe.gmope, "run_tasks_tsv", False)):
        return _run_tasks_tsv(cfg)

    # Single-config path: seed identity keys so the result row is self-describing.
    _resolve_sizes(cfg)
    explicit_keys = get_explicit_cfg_keys(cfg)
    for key in _IDENTITY_KEYS:
        if key not in explicit_keys:
            explicit_keys.append(key)
    set_explicit_cfg_keys(cfg, explicit_keys)
    return 0 if _run_seeds(cfg) else 1


__all__ = ["run_gmope", "parse_gmope_tasks"]
