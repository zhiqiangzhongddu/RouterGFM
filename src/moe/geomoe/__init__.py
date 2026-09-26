"""GeoMoE (geometric mixture of experts with curvature-guided routing) MoE method package.

GeoMoE (Cao et al., "Geometric Mixture-of-Experts with Curvature-Guided
Adaptive Routing for Graph Representation Learning", arXiv:2603.22317, 2026)
fuses Euclidean, hyperbolic and spherical GNN experts with a node-wise
graph-aware gate, regularised by Ollivier-Ricci-curvature alignment and
contrastive losses. A shift-oriented baseline of RouterGFM Table 15, trained
end-to-end on the target support of (induced) subgraph instances.
"""

from .run import run_geomoe

__all__ = ["run_geomoe"]
