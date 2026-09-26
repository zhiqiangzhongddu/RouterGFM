"""Runtime orchestration for the ``run_pretrain.py`` entrypoint.

Pretrain intentionally runs a single seed per invocation: the first configured
seed in ``cfg.seeds``. Finetune may run several seeds against that one
pretrained checkpoint, but pretraining itself stays single-seed because the
datasets are large and expensive to repeat.
"""

from __future__ import annotations

from datetime import datetime

from src.utils.run_helpers import (
    aggregate_run_metrics,
    checkpoint_path_for_runner,
    collect_run_metrics,
    resolve_seeds,
    should_save_result,
)
from src.utils.save_results import append_workflow_result

from .trainer import PretrainRunner
from .utils import run_pretrain_tasks


def run_pretrain(cfg) -> int:
    """Execute one pretraining workflow for the provided config."""
    if cfg.pretrain.run_tasks_tsv:
        return run_pretrain_tasks(cfg)

    try:
        seed = resolve_seeds(cfg, requested_count=1)[0]
    except ValueError as exc:
        print(f"[Pretrain] {exc}")
        return 1

    started_at = datetime.now().astimezone()
    run_cfg = cfg.clone()
    run_cfg.seed = int(seed)
    runner = PretrainRunner(run_cfg)
    runner.fit()
    ended_at = datetime.now().astimezone()
    if not should_save_result(runner, run_cfg):
        return 0

    summary = aggregate_run_metrics([collect_run_metrics(runner, log_prefix="[Pretrain][Summary]")])
    metric_summary = summary["metric_stats"]
    best_epochs = summary["epoch_values"].get("best_epoch")

    append_workflow_result(
        cfg=run_cfg,
        workflow="pretrain",
        started_at=started_at,
        ended_at=ended_at,
        checkpoint_save_paths=[checkpoint_path_for_runner(runner)],
        seeds=[int(run_cfg.seed)],
        best_epochs=best_epochs,
        metric_summary=metric_summary,
    )
    return 0
