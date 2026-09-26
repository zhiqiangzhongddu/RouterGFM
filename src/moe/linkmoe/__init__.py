"""Link-MoE (mixture of link predictors) MoE method package.

Heterogeneous link predictors are trained independently on the target's
train pairs; a gate over pair features and structural heuristics, trained
on the validation pairs, mixes their probabilities per node pair. Link
prediction only. Reference: Ma et al., "Mixture of Link Predictors on
Graphs", NeurIPS 2024 (https://github.com/ml-ml/Link-MoE).
"""

from .run import run_linkmoe

__all__ = ["run_linkmoe"]
