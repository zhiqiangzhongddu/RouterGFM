"""Data-preparation entrypoint helpers and CLI wiring."""

from __future__ import annotations

import sys
from typing import Iterable, List, Optional, Tuple

from src.config import cfg as base_cfg, update_cfg

from .runtime import parse_target_value, run_data_preparation_runtime


def _extract_target_override(argv: List[str]) -> Tuple[List[str], Optional[List[str]]]:
    """Extract `data_preparation.target_datasets` from argv for robust singleton/list parsing."""
    cleaned: List[str] = []
    override = None
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == "data_preparation.target_datasets":
            if idx + 1 >= len(argv):
                raise ValueError("data_preparation.target_datasets requires a value")
            override = parse_target_value(argv[idx + 1])
            idx += 2
            continue
        cleaned.append(token)
        idx += 1
    return cleaned, override


def build_data_preparation_cfg(argv: Iterable[str]):
    """Parse data-preparation CLI overrides into a config object."""
    argv_list, target_override = _extract_target_override(list(argv))
    cfg = update_cfg(base_cfg, argv_list)
    if target_override is not None:
        cfg.data_preparation.target_datasets = target_override
    return cfg


def run_data_preparation_from_cli(argv: Iterable[str]) -> int:
    """Parse CLI overrides and execute the data-preparation runtime."""
    try:
        cfg = build_data_preparation_cfg(argv)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return run_data_preparation_runtime(cfg)
