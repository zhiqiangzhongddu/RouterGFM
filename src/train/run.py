"""Train entrypoint helpers and CLI wiring."""

from __future__ import annotations

import sys
from typing import Iterable

from src.utils.cli_helpers import build_workflow_cfg

from .runtime import run_train


def build_train_cfg(argv: Iterable[str]):
    """Parse train CLI overrides and validate required dataset inputs."""
    return build_workflow_cfg(
        workflow="Train",
        argv=argv,
        dataset_name_key="train.dataset.name",
        task_level_key="train.dataset.task_level",
        run_tasks_tsv_key="train.run_tasks_tsv",
        empty_argv_hint="Please specify at least train.dataset.name/task_level.",
    )


def run_train_from_cli(argv: Iterable[str]) -> int:
    """Parse CLI overrides and execute the training runtime."""
    try:
        cfg = build_train_cfg(argv)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return run_train(cfg)
