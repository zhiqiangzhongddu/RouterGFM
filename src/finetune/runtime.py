"""Runtime orchestration for the `run_finetune.py` entrypoint."""

from __future__ import annotations

import os
from datetime import datetime

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

from .finetuner import FinetuneRunner
from .utils import resolve_pretrained_checkpoint, run_finetune_tasks

def _resolve_checkpoint_once(cfg):
    explicit_ckpt = str(getattr(getattr(cfg, "finetune", None), "pretrained_checkpoint", "") or "").strip()
    if explicit_ckpt:
        if not os.path.isfile(explicit_ckpt):
            print(f"[Finetune] Explicit checkpoint does not exist: {explicit_ckpt}")
            return None, None, 1
        run_name = os.path.splitext(os.path.basename(explicit_ckpt))[0]
        return explicit_ckpt, run_name, 0

    ckpt_path, run_name = resolve_pretrained_checkpoint(cfg)
    if not ckpt_path:
        print("[Finetune] Unable to resolve pretrained checkpoint from pretrain/model config.")
        return None, None, 1
    return ckpt_path, run_name, 0


def run_finetune(cfg) -> int:
    """Execute one or more finetuning runs for the provided config."""
    if cfg.finetune.run_tasks_tsv:
        return run_finetune_tasks(cfg)

    requested_runs = int(getattr(getattr(cfg, "finetune", None), "num_runs", 0) or 0)
    try:
        seeds = resolve_seeds(cfg, requested_count=requested_runs)
    except ValueError as exc:
        print(f"[Finetune][Multi-run] {exc}")
        return 1

    pretrain_seed = int(resolve_seeds(cfg, requested_count=1)[0])
    checkpoint_cfg = cfg.clone()
    checkpoint_cfg.seed = pretrain_seed
    ckpt_path, run_name, resolve_status = _resolve_checkpoint_once(checkpoint_cfg)
    if resolve_status != 0 or not ckpt_path or not run_name:
        summarize_runs([], seeds, log_prefix="[Finetune][Summary]")
        return resolve_status or 1

    # Store provenance on the outer cfg because append_workflow_result() saves
    # config columns from this object after the multi-run loop.
    cfg.finetune.pretrained_checkpoint = ckpt_path
    cfg.finetune.pretrained_run_name = run_name
    explicit_keys = get_explicit_cfg_keys(cfg)
    explicit_keys.extend([
        "finetune.pretrained_run_name",
        "finetune.pretrained_checkpoint",
    ])
    set_explicit_cfg_keys(cfg, explicit_keys)

    started_at = datetime.now().astimezone()
    run_metrics: list[dict[str, float]] = []
    runners: list[FinetuneRunner] = []
    total_runs = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        run_cfg = cfg.clone()
        run_cfg.seed = int(seed)
        if total_runs > 1:
            print(f"[Finetune][Multi-run] Running {index}/{total_runs} with seed={seed}")

        set_seed(run_cfg.seed)
        print(f"[Finetune] Loaded checkpoint for run '{run_name}': {ckpt_path}")
        runner = FinetuneRunner(cfg=run_cfg, pretrained_checkpoint=ckpt_path, pretrained_run_name=run_name)
        runner.fit()
        runners.append(runner)
        run_metrics.append(collect_run_metrics(runner, log_prefix="[Finetune][Summary]"))

    summarize_runs(run_metrics, seeds, log_prefix="[Finetune][Summary]")
    ended_at = datetime.now().astimezone()

    if not should_save_result(runners, cfg):
        return 0

    summary = aggregate_run_metrics(run_metrics)
    append_workflow_result(
        cfg=cfg,
        workflow="finetune",
        started_at=started_at,
        ended_at=ended_at,
        checkpoint_save_paths=collect_checkpoint_paths(runners),
        seeds=seeds,
        best_epochs=summary["epoch_values"].get("best_epoch"),
        metric_summary=summary["metric_stats"],
    )
    return 0
