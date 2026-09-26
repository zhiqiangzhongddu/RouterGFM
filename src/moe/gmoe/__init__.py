"""GMoE (Graph Mixture of Experts) MoE method package.

GMoE replaces each GNN message-passing layer with a sparsely-gated
mixture of multi-hop GNN-conv experts and trains end-to-end on a single
dataset. Reference: Wang et al., "Graph Mixture of Experts: Learning on
Large-Scale Graphs with Explicit Diversity Modeling", NeurIPS 2023
(``ref_repos/Graph-Mixture-of-Experts/``).
"""

from .run import run_gmoe

__all__ = ["run_gmoe"]
