"""Matched-pool baselines (App. C) sharing the RouterGFM inventory (DESIGN 11).

``BASELINE_RUNNERS`` maps a matched-pool method key to ``(module, runner
class)``; modules are imported on first use, so a missing or broken baseline
does not affect the others. A runner is ``Runner(cfg, app, infra)`` with
``fit()``, ``predict_queries()``, ``evaluate()``, ``best_metrics`` and
``best_epoch``; :mod:`.run` drives it. Mixtures over a candidate subset of E_a
(SAGMM-PE, META-DES) share :mod:`.candidates`. Selection baselines live in
:mod:`.selection`.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_PKG = "src.moe.routergfm.baselines"

BASELINE_RUNNERS: Dict[str, Tuple[str, str]] = {
    "metagl_u": (f"{_PKG}.metagl_u", "MetaGLURunner"),
    "sagmm_pe": (f"{_PKG}.sagmm_pe", "SAGMMPERunner"),
    "meta_des": (f"{_PKG}.meta_des", "METADESRunner"),
    # One runner, two paper rows: the variant follows cfg.moe.routergfm.baselines.method.
    "kdem": (f"{_PKG}.kdem_ppem", "KDEMPPEMRunner"),
    "ppem": (f"{_PKG}.kdem_ppem", "KDEMPPEMRunner"),
}


def load_runner_class(method: str):
    """Import and return the runner class registered for *method*."""
    if method not in BASELINE_RUNNERS:
        available = ", ".join(sorted(BASELINE_RUNNERS))
        raise ValueError(f"Unknown matched-pool baseline {method!r}. Available: {available}")
    module_name, attr = BASELINE_RUNNERS[method]
    return getattr(importlib.import_module(module_name), attr)


def config_block_name(method: str) -> str:
    """Name of the method's config subtree ``cfg.moe.routergfm.baselines.<name>``: its module name
    (``kdem``/``ppem`` -> ``kdem_ppem``), or *method* itself when unregistered."""
    return BASELINE_RUNNERS[method][0].rsplit(".", 1)[-1] if method in BASELINE_RUNNERS else method


__all__ = ["BASELINE_RUNNERS", "config_block_name", "load_runner_class"]
