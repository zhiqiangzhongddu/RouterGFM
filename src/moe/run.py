"""Runtime orchestration for the `run_moe.py` entrypoint.

Dispatches to a mixture-of-experts method selected by ``moe.method``:

- ``anygraph`` — the AnyGraph pipeline (stages via ``moe.anygraph.execution.step``).
- ``routergfm`` — the router-based GFM pipeline (config under ``moe.routergfm``).
- ``mowst`` — the Mowst weak/strong per-node expert mixture (variant via ``moe.mowst.variant``).
- ``gmoe`` — the Graph Mixture of Experts encoder, trained end-to-end (config under ``moe.gmoe``).
- ``graphmore`` — the Mixture of Riemannian Experts, trained end-to-end (config under ``moe.graphmore``).
- ``gmope`` — the Graph Mixture of Prompt-Experts (config under ``moe.gmope``).
- ``nodemoe`` — Node-MoE node-wise filtering experts, node tasks only (config under ``moe.nodemoe``).
- ``linkmoe`` — Link-MoE mixture of link predictors, link tasks only (config under ``moe.linkmoe``).
- ``graphmetro`` — GraphMETRO mixture of aligned experts (config under ``moe.graphmetro``).
- ``ogmm`` — out-of-distribution graph models merging (config under ``moe.ogmm``).
- ``geomoe`` — geometric MoE with curvature-guided routing (config under ``moe.geomoe``).
"""

from __future__ import annotations

import importlib
import sys
import traceback
from typing import Callable, Dict, Iterable, List, Tuple

from src.config import cfg as base_cfg, update_cfg
from src.utils import set_seed
from src.utils.run_helpers import resolve_seeds

RunnerFn = Callable[[object], int]
# method -> (package, runner function), imported on dispatch so a missing or
# broken method package does not break the CLI for the other methods.
_MOE_RUNNERS: Dict[str, Tuple[str, str]] = {
    "anygraph": ("src.moe.anygraph", "run_anygraph"),
    "routergfm": ("src.moe.routergfm", "run_routergfm"),
    "mowst": ("src.moe.mowst", "run_mowst"),
    "gmoe": ("src.moe.gmoe", "run_gmoe"),
    "graphmore": ("src.moe.graphmore", "run_graphmore"),
    "gmope": ("src.moe.gmope", "run_gmope"),
    "nodemoe": ("src.moe.nodemoe", "run_nodemoe"),
    "linkmoe": ("src.moe.linkmoe", "run_linkmoe"),
    "graphmetro": ("src.moe.graphmetro", "run_graphmetro"),
    "ogmm": ("src.moe.ogmm", "run_ogmm"),
    "geomoe": ("src.moe.geomoe", "run_geomoe"),
}
# Methods inferred from their own ``moe.<method>.`` prefix when ``moe.method``
# is omitted (checked before the stage aliases below).
_PREFIX_METHODS = ("gmope", "nodemoe", "linkmoe", "graphmetro", "ogmm", "geomoe")
_ANYGRAPH_STAGE_PREFIXES = (
    "moe.anygraph.",
    "execution.",
    "paths.",
    "conversion.",
    "train.",
    "eval.",
    "prediction.",
    "output.",
)
_ROUTERGFM_PREFIXES = (
    "moe.routergfm.",
)
_MOWST_PREFIXES = (
    "moe.mowst.",
)
_GMOE_PREFIXES = (
    "moe.gmoe.",
)
_GRAPHMORE_PREFIXES = (
    "moe.graphmore.",
)


def _normalize_moe_aliases(argv: List[str]) -> List[str]:
    """Support stage-scoped CLI aliases in run_moe.py."""
    normalized: List[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == "--config":
            normalized.append(token)
            if idx + 1 < len(argv):
                normalized.append(argv[idx + 1])
            idx += 2
            continue
        if token.startswith("--"):
            normalized.append(token)
            idx += 1
            continue
        if idx + 1 >= len(argv):
            normalized.append(token)
            break

        value = argv[idx + 1]
        key = token
        if token == "method":
            key = "moe.method"
        elif token.startswith("execution."):
            key = f"moe.anygraph.{token}"
        elif token.startswith("paths."):
            key = f"moe.anygraph.{token}"
        elif token.startswith("conversion."):
            key = f"moe.anygraph.{token}"
        elif token.startswith("train."):
            key = f"moe.anygraph.{token}"
        elif token.startswith("eval."):
            key = f"moe.anygraph.{token}"
        elif token.startswith("prediction."):
            key = f"moe.anygraph.{token}"
        elif token.startswith("output."):
            key = f"moe.anygraph.{token}"
        normalized.extend([key, value])
        idx += 2
    return normalized


def _build_moe_cfg(argv: Iterable[str]):
    raw_argv = list(argv)
    if not raw_argv:
        print(
            "[MoE] No CLI overrides provided. "
            "Refusing to run with default MoE config. "
            "Please specify at least moe.method and stage-specific options."
        )
        return None

    normalized_argv = _normalize_moe_aliases(raw_argv)
    has_method_override = any(
        token == "moe.method" and idx + 1 < len(normalized_argv)
        for idx, token in enumerate(normalized_argv)
    )
    if not has_method_override:
        prefixed = [
            method for method in _PREFIX_METHODS
            if any(token.startswith(f"moe.{method}.") for token in normalized_argv)
        ]
        if prefixed:
            normalized_argv = ["moe.method", prefixed[0], *normalized_argv]
        elif any(token.startswith(_GRAPHMORE_PREFIXES) for token in normalized_argv):
            normalized_argv = ["moe.method", "graphmore", *normalized_argv]
        elif any(token.startswith(_GMOE_PREFIXES) for token in normalized_argv):
            normalized_argv = ["moe.method", "gmoe", *normalized_argv]
        elif any(token.startswith(_MOWST_PREFIXES) for token in normalized_argv):
            normalized_argv = ["moe.method", "mowst", *normalized_argv]
        elif any(token.startswith(_ANYGRAPH_STAGE_PREFIXES) for token in normalized_argv):
            normalized_argv = ["moe.method", "anygraph", *normalized_argv]
        elif any(token.startswith(_ROUTERGFM_PREFIXES) for token in normalized_argv):
            normalized_argv = ["moe.method", "routergfm", *normalized_argv]
    return update_cfg(base_cfg, normalized_argv)


def _resolve_method(cfg) -> str:
    moe_cfg = getattr(cfg, "moe", None)
    if moe_cfg is None:
        return ""

    method = str(getattr(moe_cfg, "method", "") or "").strip()
    if method:
        return method.lower()
    return ""


def _load_runner(method: str) -> RunnerFn:
    module_name, attr = _MOE_RUNNERS[method]
    return getattr(importlib.import_module(module_name), attr)


def run_moe(cfg) -> int:
    method = _resolve_method(cfg)
    if not method:
        print("[MoE] Missing MoE method. Set `moe.method`.")
        return 1

    if method not in _MOE_RUNNERS:
        available = ", ".join(sorted(_MOE_RUNNERS.keys()))
        print(f"[MoE] Unsupported method '{method}'. Available methods: {available}")
        return 1

    try:
        runner = _load_runner(method)
        cfg.seed = resolve_seeds(cfg, requested_count=1)[0]
        set_seed(cfg.seed)
        return int(runner(cfg))
    except Exception as exc:  # pylint: disable=broad-except
        # Keep the non-zero exit for batch drivers, but never swallow the
        # traceback: a bare message hides the actual failure site.
        traceback.print_exc()
        print(f"[MoE][{method}] Failed: {exc}")
        return 1


def run_moe_from_cli(argv: Iterable[str]) -> int:
    try:
        cfg = _build_moe_cfg(argv)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if cfg is None:
        return 1
    return run_moe(cfg)
