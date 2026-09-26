"""Run-name builder for the AnyGraph wrapper."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Sequence


_TAG_SAFE_RE = re.compile(r"[^A-Za-z0-9+._-]+")
_MAX_DATASET_TOKEN_LEN = 64


def _sanitize_token(value: str) -> str:
    cleaned = _TAG_SAFE_RE.sub("-", str(value).strip()).strip("-")
    return cleaned or "none"


def _dataset_token(dataset_setting: str, datasets: Sequence[str]) -> str:
    raw = str(dataset_setting or "").strip()
    if not raw:
        raw = "+".join(str(d).strip() for d in datasets if str(d).strip())
    if not raw:
        raw = "none"
    token = _sanitize_token(raw)
    if len(token) <= _MAX_DATASET_TOKEN_LEN:
        return token
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    head = token[: _MAX_DATASET_TOKEN_LEN - 10].rstrip("-")
    return f"{head}-h{digest}"


def _seeds_token(seeds: Iterable[int]) -> str:
    ordered = sorted({int(seed) for seed in seeds})
    return "-".join(str(seed) for seed in ordered) if ordered else "none"


def build_anygraph_route_run_name_from_cfg(
    cfg,
    *,
    route: str,
    dataset_setting: str,
    datasets: Sequence[str],
    seeds: Iterable[int],
) -> str:
    """Canonical AnyGraph run-name builder.

    Encodes training semantics (route, dataset selection, seeds, epoch)
    so the checkpoint tag differs whenever a new training job would
    produce different weights. Evaluation-only knobs (eval_protocol,
    assignment, threshold_mode, ...) are deliberately excluded -- they
    surface as distinct CSV rows, not distinct checkpoints.
    """
    if route not in {"link", "node", "graph"}:
        raise ValueError(f"build_anygraph_route_run_name_from_cfg: unknown route='{route}'")
    train_cfg = cfg.moe.anygraph.train
    dataset_token = _dataset_token(dataset_setting, datasets)
    seeds_token = _seeds_token(seeds)
    try:
        epoch = int(getattr(train_cfg, "epoch", 0))
    except (TypeError, ValueError):
        epoch = 0
    return "_".join(
        [
            "anygraph",
            route,
            dataset_token,
            f"seed{seeds_token}",
            f"e{epoch}",
        ]
    )


__all__ = ["build_anygraph_route_run_name_from_cfg"]
