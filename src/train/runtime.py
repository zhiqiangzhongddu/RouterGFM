"""Runtime orchestration for the `run_train.py` entrypoint."""

from __future__ import annotations

from datetime import datetime

from src.utils.random import set_seed
from src.utils.run_helpers import (
    collect_run_metrics,
    resolve_seeds,
    summarize_runs,
)

from .trainer import TrainRunner
from .utils import _append_train_result, run_train_tasks


def run_train(cfg) -> int:
    """Execute one or more training runs for the provided config."""
    if cfg.train.run_tasks_tsv:
        return run_train_tasks(cfg)

    requested_runs = int(getattr(getattr(cfg, "train", None), "num_runs", 0) or 0)
    try:
        seeds = resolve_seeds(cfg, requested_count=requested_runs)
    except ValueError as exc:
        print(f"[Train][Multi-run] {exc}")
        return 1

    started_at = datetime.now().astimezone()
    run_metrics: list[dict] = []
    runners: list[TrainRunner] = []
    total_runs = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        run_cfg = cfg.clone()
        run_cfg.seed = int(seed)
        if total_runs > 1:
            print(f"[Train][Multi-run] Running {index}/{total_runs} with seed={seed}")

        set_seed(run_cfg.seed)
        runner = TrainRunner(cfg=run_cfg)
        runner.fit()
        runners.append(runner)
        run_metrics.append(collect_run_metrics(runner, log_prefix="[Train][Summary]"))

    summarize_runs(run_metrics, seeds, log_prefix="[Train][Summary]")
    ended_at = datetime.now().astimezone()
    _append_train_result(cfg, runners, run_metrics, started_at, ended_at, seeds)
    return 0
