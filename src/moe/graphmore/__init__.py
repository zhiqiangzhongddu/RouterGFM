"""GraphMoRE (Mixture of Riemannian Experts) MoE method package.

GraphMoRE (Guo et al., "GraphMoRE: Mitigating Topological Heterogeneity via
Mixture of Riemannian Experts", AAAI 2025; ``ref_repos/GraphMoRE/``) routes
each input through a mixture of curvature-specific Riemannian (kappa-GCN)
experts via a topology-aware gate. Following the repo design, node- / edge- /
graph-level tasks are all consumed as (induced) subgraph graph-level batches.
"""

from .run import run_graphmore

__all__ = ["run_graphmore"]
