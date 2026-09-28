"""Adaptive expert pruning (SAGMM Alg. 1, Eqs. 14-16; official ``update_ema_contributions`` + ``main.py``)."""

from __future__ import annotations

from typing import List

import torch


def threshold_factor(base: float, current: float, best: float, mode: str) -> float:
    """Official factor rule: worse than best -> ``max(0.75 f0, 0.01)``, better -> ``min(1.5 f0, 2.0)``,
    equal -> ``min(1.2 f0, 0.8)``; "better" follows the monitor mode (``min`` or ``max``)."""
    sign = 1.0 if mode == "max" else -1.0
    gap = sign * (float(current) - float(best))
    if gap < 0:
        return max(base * 0.75, 0.01)
    if gap > 0:
        return min(base * 1.5, 2.0)
    return min(base * 1.2, 0.8)


class ExpertPruner:
    """EMA importance ``I_j <- (1 - decay) I_j + decay * ||sum_u S[u, j] H_j(u)||`` and threshold pruning."""

    def __init__(self, num_experts: int, *, ema_decay: float, threshold_factor: float, min_experts: int):
        self.importance = torch.zeros(int(num_experts))
        self.ema_decay = float(ema_decay)
        self.base_factor = float(threshold_factor)
        self.min_experts = int(min_experts)

    def update(self, contributions: torch.Tensor, expert_mask: torch.Tensor) -> None:
        valid = expert_mask.cpu() > 0
        new = contributions.detach().float().cpu()
        self.importance[valid] = self.importance[valid] * (1.0 - self.ema_decay) + self.ema_decay * new[valid]

    def prune(self, expert_mask: torch.Tensor, *, n_train: int, current: float, best: float, mode: str) -> List[int]:
        """Mask experts whose normalised importance ``I_j / n_train`` is below ``f * mean`` over unpruned
        experts; at most ``#unpruned - min_experts`` of them, lowest index first. Updates ``expert_mask`` in place."""
        valid = expert_mask.cpu() > 0
        importance = self.importance / max(int(n_train), 1)
        tau = threshold_factor(self.base_factor, current, best, mode) * float(importance[valid].mean())
        below = torch.nonzero(valid & (importance < tau), as_tuple=False).view(-1)
        removed = below[: max(int(valid.sum()) - self.min_experts, 0)]
        if removed.numel():
            expert_mask[removed.to(expert_mask.device)] = 0.0
            self.importance[removed] = float("-inf")
        return removed.tolist()


__all__ = ["ExpertPruner", "threshold_factor"]
