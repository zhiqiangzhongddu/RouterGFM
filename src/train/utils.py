"""Train-specific helper utilities and runtime orchestration."""

from __future__ import annotations

import traceback
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
from src.utils.tsv_parsing import (
    dedup_tasks,
    parse_row_by_header,
    read_tsv_rows,
    set_cfg_field_and_track,
)

from .trainer import TrainRunner


# ---------------------------------------------------------------------------
# TSV parsing helpers
# ---------------------------------------------------------------------------

_HEADER_COLUMNS = {
    "dataset", "model", "task_level", "task_type", "induced",
    "fixed_split", "epochs", "batch", "skip_if_exists",
}

_REQUIRED_COLUMNS = ("dataset", "model", "task_level", "induced")

_DEFAULTS = {
    "task_type": None,
    "fixed_split": None,
    "skip_if_exists": None,
    "epochs": None,
    "batch": None,
}


def _task_identity(task: dict[str, Any]) -> tuple:
    return (
        task["dataset"],
        task.get("model"),
        task["task_level"],
        task.get("task_type"),
        bool(task["induced"]),
        task.get("fixed_split"),
        task.get("epochs"),
        task.get("batch"),
        task.get("skip_if_exists"),
    )


def parse_train_tasks(tsv_path: str) -> list[dict[str, Any]]:
    """Read train tasks from a header-based TSV.

    The file must begin with a header row whose tokens are all known
    column names.  Rows are parsed by column name.
    """
    header_columns, data_rows = read_tsv_rows(
        tsv_path, _HEADER_COLUMNS, min_header_columns=5, log_prefix="[Train]",
    )

    tasks: list[dict[str, Any]] = []
    if header_columns is None:
        if data_rows:
            print("[Train] Missing header row; expected a '#'-prefixed header with known column names.")
        return tasks

    for line_no, parts in data_rows:
        task = parse_row_by_header(
            parts, header_columns, line_no, "[Train]",
            required_columns=_REQUIRED_COLUMNS,
            defaults=_DEFAULTS,
        )
        if task is not None:
            tasks.append(task)

    return dedup_tasks(tasks, _task_identity)


def _build_task_cfg(base_cfg, task: dict[str, Any]):
    run_cfg = base_cfg.clone()
    explicit_keys = get_explicit_cfg_keys(run_cfg)

    run_cfg.model.name = task["model"]
    run_cfg.train.dataset.name = task["dataset"]
    run_cfg.train.dataset.task_level = task["task_level"]
    run_cfg.train.dataset.induced = bool(task["induced"])
    run_cfg.train.dataset.num_classes = None
    run_cfg.train.dataset.label_dim = None
    run_cfg.model.in_dim = 0
    run_cfg.train.run_tasks_tsv = False
    explicit_keys.extend([
        "train.dataset.name",
        "train.dataset.task_level",
        "train.dataset.induced",
        "model.name",
    ])

    set_cfg_field_and_track(run_cfg.train.dataset, "task_type", task.get("task_type"), "train.dataset.task_type", explicit_keys)
    set_cfg_field_and_track(run_cfg.train.dataset, "fixed_split", task.get("fixed_split"), "train.dataset.fixed_split", explicit_keys)
    set_cfg_field_and_track(run_cfg.train, "epochs", task.get("epochs"), "train.epochs", explicit_keys)
    set_cfg_field_and_track(run_cfg.train, "batch_size", task.get("batch"), "train.batch_size", explicit_keys)
    set_cfg_field_and_track(run_cfg.train, "skip_if_exists", task.get("skip_if_exists"), "train.skip_if_exists", explicit_keys)

    set_explicit_cfg_keys(run_cfg, explicit_keys)
    return run_cfg


def run_train_tasks(cfg) -> int:
    """Run training tasks defined in cfg.train.tasks_tsv."""
    tasks = parse_train_tasks(getattr(cfg.train, "tasks_tsv", ""))
    if not tasks:
        print("[Train] No tasks found to run.")
        return 1

    requested_runs = int(getattr(getattr(cfg, "train", None), "num_runs", 0) or 0)

    results: list[bool] = []
    for task in tasks:
        started_at = datetime.now().astimezone()
        base_cfg = _build_task_cfg(cfg, task)
        seeds: list[int] = []
        run_metrics: list[dict[str, float]] = []
        runners: list[TrainRunner] = []
        try:
            seeds = resolve_seeds(base_cfg, requested_count=requested_runs)
            total_runs = len(seeds)
            for index, seed in enumerate(seeds, start=1):
                run_cfg = base_cfg.clone()
                run_cfg.seed = int(seed)
                if total_runs > 1:
                    print(f"[Train][Multi-run] Running {index}/{total_runs} with seed={seed}")
                set_seed(run_cfg.seed)
                runner = TrainRunner(run_cfg)
                runner.fit()
                runners.append(runner)
                run_metrics.append(collect_run_metrics(runner, log_prefix="[Train][Summary]"))

            summarize_runs(run_metrics, seeds, log_prefix="[Train][Summary]")
            ended_at = datetime.now().astimezone()
            _append_train_result(base_cfg, runners, run_metrics, started_at, ended_at, seeds)
            results.append(True)
        except Exception as exc:
            traceback.print_exc()
            print(
                f"[Train] Failed {task['dataset']} (model={task['model']}, "
                f"task_level={task['task_level']}, induced={task['induced']}) "
                f"seeds={seeds}: {exc}"
            )
            results.append(False)

    return 0 if all(results) else 1


def _append_train_result(cfg, runners: list["TrainRunner"], run_metrics, started_at, ended_at, seeds) -> None:
    if not should_save_result(runners, cfg):
        return

    summary = aggregate_run_metrics(run_metrics)
    append_workflow_result(
        cfg=cfg,
        workflow="train",
        started_at=started_at,
        ended_at=ended_at,
        checkpoint_save_paths=collect_checkpoint_paths(runners),
        seeds=seeds,
        best_epochs=summary["epoch_values"].get("best_epoch"),
        metric_summary=summary["metric_stats"],
    )
