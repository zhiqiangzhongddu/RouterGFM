"""Node-MoE (node-wise filtering mixture of experts) MoE method package.

A node-wise GIN gate mixes ChebNetII experts with diverse spectral-filter
initialisations, trained end-to-end on the target support set; node tasks
only. Reference: Han et al., "Node-wise Filtering in Graph Neural Networks:
A Mixture of Experts Approach", arXiv:2406.03464 (2024).
"""

from .run import run_nodemoe

__all__ = ["run_nodemoe"]
