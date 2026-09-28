"""Topology-aware attention gating (TAAG, SAGMM Eqs. 6-13; official ``sagmm_gating.py``).

The paper's ``beta(...) + (1 - beta) X`` residual of Eq. 9 is dimensionally
inconsistent and absent from the official code; it is omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gate_features import GateInputs


class StraightThroughSign(torch.autograd.Function):
    """``sign`` forward, identity backward (official ``SAGMMGateBackward``)."""

    @staticmethod
    def forward(ctx, scores):
        return torch.sign(scores)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def sga_scores(
    query: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    pop: Optional[torch.Tensor] = None,
    pop_size: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Simplified global attention ``Z = (V + Q^ (K^T V) / n) / (1 + Q^ (K^T 1) / n)`` at the query rows.

    ``Q^ = Q / ||Q||_F`` and ``K^ = K / ||K||_F`` are normalised over the
    population. ``w_* [D, N]``. Without ``pop`` the query rows ``[B, D]`` are the
    population. With ``pop [B, n_max, D]`` (zero-padded; the maps have no bias,
    so padding rows add nothing) each query row attends over its own population
    of ``pop_size`` rows and is scored against that population's norms.
    """
    q, v = query @ w_q, query @ w_v
    if pop is None:
        k = query @ w_k
        n = q.size(0)
        q_hat = q / q.norm().clamp_min(1e-12)
        k_hat = k / k.norm().clamp_min(1e-12)
        num = v + q_hat @ (k_hat.t() @ v) / n
        den = 1.0 + q_hat @ k_hat.sum(0) / n
        return num / den[:, None]
    n = pop_size.to(q.dtype)
    qp, kp, vp = pop @ w_q, pop @ w_k, pop @ w_v
    q_hat = q / qp.pow(2).sum((1, 2)).sqrt().clamp_min(1e-12)[:, None]
    k_hat = kp / kp.pow(2).sum((1, 2)).sqrt().clamp_min(1e-12)[:, None, None]
    attn = torch.einsum("bn,bln->bl", q_hat, k_hat)
    num = v + torch.einsum("bl,bln->bn", attn, vp) / n[:, None]
    den = 1.0 + attn.sum(1) / n
    return num / den[:, None]


def cv_squared(x: torch.Tensor) -> torch.Tensor:
    """Squared coefficient of variation (unbiased variance); 0 for a single entry."""
    if x.numel() == 1:
        return x.new_zeros(())
    x = x.float()
    return x.var() / (x.mean() ** 2 + 1e-10)


def diversity_loss(weight: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
    """``||(W^T W^ - I) * (m m^T)||_F + mean_j ||W[:, j]||`` for ``weight [D, N]`` (columns = experts).

    As in the official code the norm term averages over all columns, pruned ones included.
    """
    unit = F.normalize(weight, dim=0)
    sims = unit.t() @ unit
    pair_mask = expert_mask[:, None] * expert_mask[None, :]
    eye = torch.eye(sims.size(0), device=sims.device, dtype=sims.dtype)
    return torch.norm(sims * pair_mask - eye * pair_mask) + torch.norm(weight, dim=0).mean()


@dataclass
class GateOutput:
    gates: torch.Tensor  # G [B, N] = Z' * M (not renormalised)
    active: torch.Tensor  # M [B, N] binary (straight-through)
    scores: torch.Tensor  # Z' [B, N] (pruned experts 0)


class TAAGGate(nn.Module):
    """Single-head SGA scores, activation, expert mask, learnable threshold, and adaptive top-k."""

    def __init__(self, in_dim: int, num_experts: int, score_act: str = "sigmoid", threshold_init: str = "zeros"):
        super().__init__()
        if score_act not in ("sigmoid", "softplus"):
            raise ValueError(f"Unknown score_act {score_act!r} (expected sigmoid|softplus).")
        if threshold_init not in ("zeros", "randn"):
            raise ValueError(f"Unknown threshold_init {threshold_init!r} (expected zeros|randn).")
        self.score_act = score_act
        self.w_q = nn.Linear(in_dim, num_experts, bias=False)
        self.w_k = nn.Linear(in_dim, num_experts, bias=False)
        self.w_v = nn.Linear(in_dim, num_experts, bias=False)
        init = torch.zeros(num_experts) if threshold_init == "zeros" else 0.1 * torch.randn(num_experts)
        self.threshold = nn.Parameter(init)
        self.register_buffer("expert_mask", torch.ones(num_experts))

    def forward(self, inputs: GateInputs) -> GateOutput:
        z = sga_scores(
            inputs.query, self.w_q.weight.t(), self.w_k.weight.t(), self.w_v.weight.t(), inputs.pop, inputs.pop_size
        )
        scores = (torch.sigmoid(z) if self.score_act == "sigmoid" else F.softplus(z)) * self.expert_mask
        active = StraightThroughSign.apply(F.relu(scores - torch.sigmoid(self.threshold)))
        empty = active.sum(-1) == 0
        if bool(empty.any()):  # rows without an expert above threshold keep their best active one
            best = scores.detach().masked_fill(self.expert_mask == 0, float("-inf")).argmax(-1)
            fallback = empty[:, None] & (torch.arange(scores.size(1), device=scores.device)[None, :] == best[:, None])
            active = torch.where(fallback, torch.ones_like(active), active)
        return GateOutput(gates=scores * active, active=active, scores=scores)

    def aux_loss(self, gates: torch.Tensor, imp_weight: float, div_weight: float) -> torch.Tensor:
        """Importance ``cv^2(sum_u G[u])`` plus diversity of ``W_Q, W_K, W_V`` (training only)."""
        loss = gates.new_zeros(())
        if imp_weight:
            loss = loss + float(imp_weight) * cv_squared(gates.sum(0))
        if div_weight:
            div = sum(diversity_loss(lin.weight.t(), self.expert_mask) for lin in (self.w_q, self.w_k, self.w_v))
            loss = loss + float(div_weight) * div
        return loss


__all__ = ["GateOutput", "StraightThroughSign", "TAAGGate", "cv_squared", "diversity_loss", "sga_scores"]
