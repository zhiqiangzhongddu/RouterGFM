"""Finetune entrypoint helpers and CLI wiring."""

from __future__ import annotations

import sys
import warnings
from typing import Iterable

from src.utils.cli_helpers import build_workflow_cfg

from .runtime import run_finetune
from .utils import extract_few_shot


def _preprocess_finetune_argv(raw_argv: list[str]):
    """Lift ``--fewshot`` into ``finetune.dataset.fixed_split``."""
    forwarded_argv, few_shot_split = extract_few_shot(raw_argv)
    extra_explicit: list[tuple[str, object]] = []
    if few_shot_split is not None:
        extra_explicit.append(("finetune.dataset.fixed_split", few_shot_split))
    return forwarded_argv, extra_explicit


def build_finetune_cfg(argv: Iterable[str]):
    """
    Parse CLI overrides for finetuning and validate required inputs.

    `--fewshot` overrides `finetune.dataset.fixed_split`.
    """
    return build_workflow_cfg(
        workflow="Finetune",
        argv=argv,
        dataset_name_key="finetune.dataset.name",
        task_level_key="finetune.dataset.task_level",
        run_tasks_tsv_key="finetune.run_tasks_tsv",
        empty_argv_hint="Please specify at least finetune.dataset.name/task_level.",
        preprocess_argv=_preprocess_finetune_argv,
    )


def run_finetune_from_cli(argv: Iterable[str]) -> int:
    """Parse CLI overrides and execute the finetuning runtime."""
    warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch_sparse")
    try:
        cfg = build_finetune_cfg(argv)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return run_finetune(cfg)
