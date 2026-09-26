"""Pretrain entrypoint helpers and CLI wiring."""

from __future__ import annotations

import sys
from typing import Iterable

from src.utils.cli_helpers import build_workflow_cfg

from .runtime import run_pretrain


def build_pretrain_cfg(argv: Iterable[str]):
    """Parse pretrain CLI overrides and validate required dataset inputs."""
    return build_workflow_cfg(
        workflow="Pretrain",
        argv=argv,
        dataset_name_key="pretrain.dataset.name",
        task_level_key="pretrain.dataset.task_level",
        run_tasks_tsv_key="pretrain.run_tasks_tsv",
        empty_argv_hint=(
            "Refusing to run with default dataset/model. "
            "Please specify at least pretrain.dataset.name/task_level."
        ),
    )


def run_pretrain_from_cli(argv: Iterable[str]) -> int:
    """Parse CLI overrides and execute the pretraining runtime."""
    try:
        cfg = build_pretrain_cfg(argv)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return run_pretrain(cfg)
