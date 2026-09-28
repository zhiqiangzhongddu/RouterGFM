"""Parameter merging (Eq. 8) and the PPEM proximity pull (Eq. 13)."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn as nn


def module_tensors(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Parameters and buffers by name (shared tensors once), keeping autograd links."""
    return {**dict(module.named_parameters()), **dict(module.named_buffers())}


def merged_state(states: Sequence[Mapping[str, torch.Tensor]], alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
    """``theta_bar = sum_i alpha_i theta_i`` for floating tensors (differentiable); other tensors from ``states[0]``."""
    if len(states) != int(alpha.numel()):
        raise ValueError(f"{len(states)} expert states for {alpha.numel()} merge weights.")
    out = {}
    for name, ref in states[0].items():
        if ref.is_floating_point():
            weights = alpha.to(device=ref.device, dtype=ref.dtype)
            out[name] = sum(weights[i] * state[name] for i, state in enumerate(states))
        else:
            out[name] = ref
    return out


@torch.no_grad()
def ema_pull_(experts: List[nn.Module], alpha: torch.Tensor, beta: float) -> None:
    """PPEM Eq. 13 in place: ``theta_i <- beta theta_i + (1 - beta) theta_bar`` for the team's parameters."""
    params = [dict(expert.named_parameters()) for expert in experts]
    merged = merged_state(params, alpha)
    for state in params:
        for name, param in state.items():
            param.mul_(float(beta)).add_(merged[name], alpha=1.0 - float(beta))


def resolve_ema_beta(*, target_retention: float, fallback_beta: float, total_steps: int, period: int) -> float:
    """``beta = r ** (1 / max(1, total_steps // period))`` so ``beta ** N_pulls = r``; ``fallback_beta`` when ``r <= 0``."""
    if float(target_retention) <= 0:
        return float(fallback_beta)
    pulls = max(1, int(total_steps) // max(1, int(period)))
    return math.exp(math.log(float(target_retention)) / pulls)


__all__ = ["ema_pull_", "merged_state", "module_tensors", "resolve_ema_beta"]
