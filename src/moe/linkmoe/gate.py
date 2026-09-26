"""Step 2 of Link-MoE: the pair-wise gate (Eq. 2) and its training objective.

``G(x_ij, s_ij) = softmax(f_score(f_feat(x_i * x_j) || f_struct(s_ij)))`` and
the mixture probability ``q = sum_o G_o p_o`` (official code: expert
probabilities are mixed, then ``sigmoid(q)`` enters the loss).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BranchMLP(nn.Module):
    """Official ``mlp_model``: every layer (including the last) is Lin -> ReLU -> Dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.lins:
            x = self.dropout(torch.relu(lin(x)))
        return x


class LinkMoEGate(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        struct_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_layers_predictor: int,
        num_experts: int,
        dropout: float,
    ):
        super().__init__()
        self.f_feat = BranchMLP(feat_dim, hidden_dim, num_layers, dropout)
        self.f_struct = BranchMLP(struct_dim, hidden_dim, num_layers, dropout)
        dims = [2 * hidden_dim] + [hidden_dim] * (num_layers_predictor - 1) + [num_experts]
        self.f_score = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat: torch.Tensor, struct: torch.Tensor) -> torch.Tensor:
        """``[P, m]`` expert weights on the simplex."""
        h = torch.cat([self.f_feat(feat), self.f_struct(struct)], dim=-1)
        for lin in self.f_score[:-1]:
            h = self.dropout(torch.relu(lin(h)))
        return torch.softmax(self.f_score[-1](h), dim=-1)


def mixture_probability(weights: torch.Tensor, expert_probs: torch.Tensor) -> torch.Tensor:
    """``q = sum_o w_o p_o`` with ``weights [P, m]`` and ``expert_probs [m, P]`` -> ``[P]``."""
    return (weights * expert_probs.t()).sum(dim=-1)


def gate_loss(q: torch.Tensor, labels: torch.Tensor, neg_loss_weight: float) -> torch.Tensor:
    """``-mean_pos log sig(q) - w * mean_neg log(1 - sig(q))`` (official ``main.py::train``)."""
    pos = labels > 0.5
    loss = q.new_zeros(())
    if pos.any():
        loss = loss - F.logsigmoid(q[pos]).mean()
    if (~pos).any():
        loss = loss - float(neg_loss_weight) * F.logsigmoid(-q[~pos]).mean()
    return loss


__all__ = ["BranchMLP", "LinkMoEGate", "gate_loss", "mixture_probability"]
