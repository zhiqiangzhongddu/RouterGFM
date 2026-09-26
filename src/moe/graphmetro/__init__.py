"""GraphMETRO (mixture of aligned experts) MoE method package.

A gating GNN detects which stochastic shift components are present in an
instance and mixes ``K+1`` independent GNN experts whose representations are
aligned to a reference expert; trained end-to-end on the target support.
Reference: Wu et al., "GraphMETRO: Mitigating Complex Graph Distribution
Shifts via Mixture of Aligned Experts", NeurIPS 2024. Shift-oriented baseline
of RouterGFM Table 15 (scored with ``src.moe.shift_eval``).
"""

from .run import run_graphmetro

__all__ = ["run_graphmetro"]
