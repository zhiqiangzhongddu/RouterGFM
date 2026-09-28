"""Plackett-Luce ranking loss (Model Spider Eq. 3-4; official ``HierarchicalCE``), masked."""

from __future__ import annotations

from typing import Optional

import torch

_MASKED = -1e9  # finite stand-in for excluded scores: exp underflows to exactly 0


def plackett_luce_loss(
    scores: torch.Tensor,
    target_losses: torch.Tensor,
    valid: torch.Tensor,
    tie_break: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean over tasks of ``sum_m [logsumexp(s_{dsc(m..M)}) - s_{dsc(m)}]``.

    ``scores``/``target_losses``/``valid``: ``[B, M]``. The ground-truth order
    sorts the valid items by ascending target loss (lower loss = better), ties
    by ascending ``tie_break`` (default: column index). Invalid items (or
    non-finite targets) are removed from the permutation; tasks without a valid
    item are skipped (at least one task must have one).
    """
    valid = valid & torch.isfinite(target_losses)
    if tie_break is None:
        tie_break = torch.arange(scores.size(1), device=scores.device).expand_as(scores)
    order = torch.argsort(tie_break, dim=1, stable=True)
    key = target_losses.masked_fill(~valid, float("inf")).gather(1, order)
    order = order.gather(1, torch.argsort(key, dim=1, stable=True))
    v = valid.gather(1, order)
    s = scores.gather(1, order).masked_fill(~v, _MASKED)
    rest = torch.logcumsumexp(s.flip(1), dim=1).flip(1)  # logsumexp over positions m..M
    per_task = torch.where(v, rest - s, torch.zeros_like(s)).sum(1)
    return per_task[v.any(1)].mean()


__all__ = ["plackett_luce_loss"]
