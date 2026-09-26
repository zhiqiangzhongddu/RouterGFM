"""OGMM (Out-of-Distribution Graph Models Merging) MoE baseline package.

Builds its own density-domain experts from the target support, inverts each
into label-conditional graphs, and merges the experts source-free with a
noisy top-k gate and classifier masks. Reference: Wang et al.,
"Out-of-Distribution Graph Models Merging", ICLR 2026 (arXiv:2506.03674).
"""

from .run import run_ogmm

__all__ = ["run_ogmm"]
