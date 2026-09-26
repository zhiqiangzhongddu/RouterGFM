"""GMoPE (Graph Mixture of Prompt-Experts) MoE method package.

GMoPE pretrains M GNN experts, each conditioned on its own learnable prompt
concatenated to the node features, with a structure-aware top-K router and a
soft orthogonality loss on the prompts; downstream it freezes the experts,
tunes only the prompts and a shared task head, and mixes the experts'
embeddings with entropy-based confidence weights. Reference: Wang et al.,
"GMoPE: A Prompt-Expert Mixture Framework for Graph Foundation Models",
arXiv:2511.03251 (no official code).
"""

from .run import run_gmope

__all__ = ["run_gmope"]
