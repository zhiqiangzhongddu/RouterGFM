"""Task tokens: class partitions of a support set and their weighted centres (Model Spider Eq. 5).

Every label type becomes a nonnegative weight matrix ``W [n, C]`` over the
support instances; tokens are the weighted feature centres of its nonzero
columns. Single-label and link: one-hot classes. Regression: equal-frequency
bins of the first target column. Multi-label: per-instance counts of observed
positive and negative assays (NaN = missing).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from ....common import MULTILABEL, REGRESSION
from ....losses import is_simplex_family


def partition_weights(labels: torch.Tensor, family: str, *, regression_bins: int = 5) -> torch.Tensor:
    """``[n, C]`` float32 weights >= 0; rows without a usable label are all zero."""
    n = int(labels.size(0))
    if is_simplex_family(family):
        y = labels.reshape(n).long()
        valid = y >= 0
        num = int(y[valid].max()) + 1 if bool(valid.any()) else 1
        w = torch.zeros(n, num)
        w[valid.nonzero(as_tuple=True)[0], y[valid]] = 1.0
        return w
    rows = labels.reshape(n, -1).float()
    if family == MULTILABEL:
        observed = torch.isfinite(rows)
        pos = (observed & (rows > 0.5)).sum(1)
        neg = (observed & (rows <= 0.5)).sum(1)
        return torch.stack([pos, neg], dim=1).float()
    if family == REGRESSION:
        y = rows[:, 0]
        valid = torch.isfinite(y).nonzero(as_tuple=True)[0]
        bins = max(1, min(int(regression_bins), valid.numel()))
        w = torch.zeros(n, bins)
        if valid.numel():
            order = valid[torch.argsort(y[valid], stable=True)]  # ties: support order
            rank = torch.arange(valid.numel())
            w[order, rank * bins // valid.numel()] = 1.0
        return w
    raise ValueError(f"Unsupported task family {family!r}")


def weighted_centers(feats: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``[C', d]`` centres ``diag(1/colsum W) W^T feats`` of the columns with positive mass."""
    weights = weights.to(feats.dtype)
    mass = weights.sum(0)
    keep = mass > 0
    return (weights[:, keep].t() @ feats) / mass[keep, None]


def pad_token_sets(sets: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad ``[C_i, d]`` token sets to ``[B, C_max, d]`` with a validity mask ``[B, C_max]``."""
    size = max(int(s.size(0)) for s in sets)
    ref = sets[0]
    out = ref.new_zeros(len(sets), size, ref.size(1))
    mask = torch.zeros(len(sets), size, dtype=torch.bool, device=ref.device)
    for i, s in enumerate(sets):
        out[i, : s.size(0)] = s
        mask[i, : s.size(0)] = True
    return out, mask


__all__: List[str] = ["pad_token_sets", "partition_weights", "weighted_centers"]
