"""SAGMM-PE: self-adaptive graph mixture over the matched frozen-expert pool (Meena et al., AAAI 2026).

RouterGFM App. C, Table 1. Registered as ``sagmm_pe`` in ``baselines.BASELINE_RUNNERS``.
"""

from .trainer import SAGMMPERunner

__all__ = ["SAGMMPERunner"]
