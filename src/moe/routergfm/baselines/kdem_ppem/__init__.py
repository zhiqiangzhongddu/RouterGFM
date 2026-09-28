"""KDEM / PPEM enhanced expert merging (Liu et al., NeurIPS 2025) on the matched pool.

Registered as ``kdem`` and ``ppem`` in ``baselines.BASELINE_RUNNERS``; the
variant follows ``cfg.moe.routergfm.baselines.method``.
"""

from .trainer import VARIANTS, KDEMPPEMRunner

__all__ = ["KDEMPPEMRunner", "VARIANTS"]
