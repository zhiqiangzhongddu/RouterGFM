"""SAGMM-PE model: TAAG gate over frozen expert readouts and a trainable task head.

``y(x) = sum_e G_e(x) H_e(x) / ||H_e(x)||`` and ``logits = Linear(y)``; only the
gate and the head are trained (SAGMM Sec. 3.4, App. F.1). The head is one
``Linear`` for every task level; the official graph head (two stacked Linear
layers) has the same function class, and link prediction scores the frozen pair
readout with one logit because endpoint rows are not exposed separately.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .gate_features import GateInputs
from .gating import GateOutput, TAAGGate

_EXPERT_CHUNK = 64  # experts per einsum pass


def mix(weights: torch.Tensor, experts: torch.Tensor) -> torch.Tensor:
    """``[B, d] = sum_j weights[:, j] * experts[:, j]`` for ``weights [B, N]``, ``experts [B, N, d]``."""
    out = weights.new_zeros(experts.size(0), experts.size(2))
    for start in range(0, experts.size(1), _EXPERT_CHUNK):
        stop = start + _EXPERT_CHUNK
        out = out + torch.einsum("bn,bnd->bd", weights[:, start:stop], experts[:, start:stop].to(weights.dtype))
    return out


@torch.no_grad()
def contributions(weights: torch.Tensor, experts: torch.Tensor) -> torch.Tensor:
    """``[N]`` norms ``||sum_u weights[u, j] * experts[u, j]||`` (pruning importance).

    Pass the forward's mixing weights (``S * inv_norm``) with the raw readouts, so the
    importance scores the normalised readouts the model mixes, not their raw magnitude.
    """
    out = []
    for start in range(0, experts.size(1), _EXPERT_CHUNK):
        stop = start + _EXPERT_CHUNK
        summed = torch.einsum("bn,bnd->nd", weights[:, start:stop], experts[:, start:stop].to(weights.dtype))
        out.append(summed.norm(dim=-1))
    return torch.cat(out)


class SAGMMPE(nn.Module):
    def __init__(self, gate_in_dim: int, num_experts: int, expert_dim: int, out_dim: int, *, score_act: str, threshold_init: str):
        super().__init__()
        self.gate = TAAGGate(gate_in_dim, num_experts, score_act=score_act, threshold_init=threshold_init)
        self.head = nn.Linear(expert_dim, out_dim)

    def forward(self, inputs: GateInputs, experts: torch.Tensor, inv_norm: torch.Tensor) -> Tuple[torch.Tensor, GateOutput]:
        """``experts [B, N, d]`` raw readouts, ``inv_norm [B, N]`` their per-row normalisers."""
        gate = self.gate(inputs)
        return self.head(mix(gate.gates * inv_norm, experts)), gate


__all__ = ["SAGMMPE", "contributions", "mix"]
