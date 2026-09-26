from __future__ import annotations

from typing import Any
import traceback
from datetime import datetime

from src.pretrain.trainer import PretrainRunner
from src.utils.run_helpers import (
    aggregate_run_metrics,
    checkpoint_path_for_runner,
    collect_run_metrics,
    resolve_seeds,
    should_save_result,
)
from src.utils.save_results import append_workflow_result, get_explicit_cfg_keys, set_explicit_cfg_keys
from src.utils.tsv_parsing import (
    dedup_tasks,
    parse_row_by_header,
    read_tsv_rows,
    set_cfg_field_and_track,
)


_HEADER_COLUMNS = {
    "model", "dataset", "task_level", "task_type", "induced",
    "method", "fixed_split", "epochs", "batch", "seed",
    "skip_if_exists",
}

_REQUIRED_COLUMNS = ("dataset", "task_level", "induced", "method")

_DEFAULTS = {
    "task_type": None,
    "fixed_split": None,
    "skip_if_exists": None,
    "epochs": None,
    "batch": None,
    "seed": None,
}


def _task_identity(task: dict[str, Any]) -> tuple:
    return (
        task.get("model"),
        task["dataset"],
        task["task_level"],
        task.get("task_type"),
        bool(task["induced"]),
        task.get("fixed_split"),
        task.get("method"),
        task.get("epochs"),
        task.get("batch"),
        task.get("seed"),
        task.get("skip_if_exists"),
    )


def parse_pretrain_tasks(tsv_path: str) -> list[dict[str, Any]]:
    """Read pretrain tasks from a header-based TSV.

    The file must begin with a header row whose tokens are all known
    column names.  Rows are parsed by column name.
    """
    header_columns, data_rows = read_tsv_rows(
        tsv_path, _HEADER_COLUMNS, min_header_columns=4, log_prefix="[Pretrain]",
    )

    tasks: list[dict[str, Any]] = []
    if header_columns is None:
        if data_rows:
            print("[Pretrain] Missing header row; expected a '#'-prefixed header with known column names.")
        return tasks

    for line_no, parts in data_rows:
        task = parse_row_by_header(
            parts, header_columns, line_no, "[Pretrain]",
            required_columns=_REQUIRED_COLUMNS,
            defaults=_DEFAULTS,
        )
        if task is not None:
            tasks.append(task)

    return dedup_tasks(tasks, _task_identity)


def _build_task_cfg(base_cfg, task: dict[str, Any]):
    run_cfg = base_cfg.clone()
    explicit_keys = get_explicit_cfg_keys(run_cfg)

    set_cfg_field_and_track(run_cfg.model, "name", task.get("model"), "model.name", explicit_keys)
    run_cfg.pretrain.dataset.name = task["dataset"]
    run_cfg.pretrain.dataset.task_level = task["task_level"]
    run_cfg.pretrain.dataset.induced = task["induced"]
    run_cfg.pretrain.dataset.num_classes = None
    run_cfg.pretrain.dataset.label_dim = None
    run_cfg.model.in_dim = 0
    run_cfg.pretrain.method = task["method"]
    explicit_keys.extend([
        "pretrain.dataset.name",
        "pretrain.dataset.task_level",
        "pretrain.dataset.induced",
        "pretrain.method",
    ])

    set_cfg_field_and_track(run_cfg.pretrain.dataset, "task_type", task.get("task_type"), "pretrain.dataset.task_type", explicit_keys)
    set_cfg_field_and_track(run_cfg.pretrain.dataset, "fixed_split", task.get("fixed_split"), "pretrain.dataset.fixed_split", explicit_keys)
    set_cfg_field_and_track(run_cfg.pretrain, "epochs", task.get("epochs"), "pretrain.epochs", explicit_keys)
    set_cfg_field_and_track(run_cfg.pretrain, "batch_size", task.get("batch"), "pretrain.batch_size", explicit_keys)
    if task.get("seed") is not None:
        run_cfg.seeds = [int(task["seed"])]
    set_cfg_field_and_track(run_cfg.pretrain, "skip_if_exists", task.get("skip_if_exists"), "pretrain.skip_if_exists", explicit_keys)
    run_cfg.pretrain.run_tasks_tsv = False

    set_explicit_cfg_keys(run_cfg, explicit_keys)
    return run_cfg


def run_pretrain_tasks(cfg) -> int:
    """Run all pretrain tasks defined in cfg.pretrain.tasks_tsv."""
    tasks = parse_pretrain_tasks(getattr(cfg.pretrain, "tasks_tsv", ""))
    if not tasks:
        print("[Pretrain] No tasks found to run.")
        return 1

    results: list[bool] = []
    for task in tasks:
        started_at = datetime.now().astimezone()
        run_cfg = _build_task_cfg(cfg, task)
        seed_value = task.get("seed")
        try:
            run_cfg.seed = int(resolve_seeds(run_cfg, requested_count=1)[0])
            seed_value = run_cfg.seed
            runner = PretrainRunner(run_cfg)
            runner.fit()
            ended_at = datetime.now().astimezone()
            if should_save_result(runner, run_cfg):
                summary = aggregate_run_metrics([collect_run_metrics(runner, log_prefix="[Pretrain][Summary]")])
                metric_summary = summary["metric_stats"]
                best_epochs = summary["epoch_values"].get("best_epoch")
                seeds = [int(run_cfg.seed)]

                append_workflow_result(
                    cfg=run_cfg,
                    workflow="pretrain",
                    started_at=started_at,
                    ended_at=ended_at,
                    checkpoint_save_paths=[checkpoint_path_for_runner(runner)],
                    seeds=seeds,
                    best_epochs=best_epochs,
                    metric_summary=metric_summary,
                )
            results.append(True)
        except Exception as exc:
            traceback.print_exc()
            print(
                f"[Pretrain] Failed {task['dataset']} (task level={task['task_level']}, induced={task['induced']}) "
                f"{task['method']} seed={seed_value}: {exc}"
            )
            results.append(False)

    return 0 if all(results) else 1
