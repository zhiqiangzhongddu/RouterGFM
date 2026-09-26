"""OGMM stage 2: fine-tuned MoE merging of masked experts (Sec. 3.3, Eqs. 12-20).

Only the classifier heads are masked (``MaskCL``, the paper's recommendation);
the encoders stay frozen. A noisy top-k gate mixes the masked experts' logits,
and the whole model is trained on generated graphs only (source-free).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.utils.supervised_loss import supervised_loss_from_logits

from .experts import DenseExpert, DenseInstances


class MaskedLinear(nn.Module):
    """Eq. 13 on a frozen head: ``W_hat = W * omega_w``, ``b_hat = b * omega_b`` (masks start at 1)."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear.requires_grad_(False)
        self.omega_w = nn.Parameter(torch.ones_like(linear.weight))
        if linear.bias is not None:
            self.omega_b = nn.Parameter(torch.ones_like(linear.bias))
        else:
            self.register_parameter("omega_b", None)

    def forward(self, h: Tensor) -> Tensor:
        bias = None if self.linear.bias is None else self.linear.bias * self.omega_b
        return F.linear(h, self.linear.weight * self.omega_w, bias)

    def mask_tensors(self) -> list[Tensor]:
        return [self.omega_w] + ([self.omega_b] if self.omega_b is not None else [])


class NoisyTopKGate(nn.Module):
    """Eqs. 15-16: ``softmax(TopK(g W_g + eps * softplus(g W_n), k))``; noise only in training.

    Same math as ``src.moe.gmoe.moe_layer.SparseMoEConv.noisy_top_k_gating``
    (zero-initialised weights, 1e-2 noise floor).
    """

    def __init__(self, in_dim: int, num_experts: int, k: int, noise_epsilon: float = 1e-2):
        super().__init__()
        if not 1 <= int(k) <= int(num_experts):
            raise ValueError(f"top_k ({k}) must be in [1, num_experts={num_experts}].")
        self.k = int(k)
        self.noise_epsilon = float(noise_epsilon)
        self.w_gate = nn.Parameter(torch.zeros(int(in_dim), int(num_experts)))
        self.w_noise = nn.Parameter(torch.zeros(int(in_dim), int(num_experts)))

    def forward(self, g: Tensor) -> Tensor:
        logits = g @ self.w_gate
        if self.training:
            noise_std = F.softplus(g @ self.w_noise) + self.noise_epsilon
            logits = logits + torch.randn_like(logits) * noise_std
        top_logits, top_index = logits.topk(self.k, dim=1)
        return torch.zeros_like(logits).scatter(1, top_index, F.softmax(top_logits, dim=1))


def cv_squared(x: Tensor) -> Tensor:
    """Squared coefficient of variation (GMoE); 0 for a single expert."""
    if x.numel() <= 1:
        return x.sum() * 0.0
    x = x.float()
    return x.var() / (x.mean() ** 2 + 1e-10)


def mask_regularizer(masks: list[list[Tensor]], gamma_p: float, gamma_v: float) -> Tensor:
    """Second part of Eq. 19, literally: ``sum_j (mean(w_j) - g_p) + (frac(|w_j - 1| < g_v) - g_p)``.

    The count term is piecewise constant (no gradient); the mean term pulls each
    mask entry down with gradient ``1 / |w_j|``.
    """
    total = torch.zeros((), device=masks[0][0].device)
    for expert_masks in masks:
        flat = torch.cat([m.reshape(-1) for m in expert_masks])
        with torch.no_grad():
            near_one = ((flat - 1.0).abs() < float(gamma_v)).float().mean()
        total = total + (flat.mean() - float(gamma_p)) + (near_one - float(gamma_p))
    return total


class OGMMMergedModel(nn.Module):
    """Merging function ``Gamma(G) = sum_j Gate(G)_j f(Theta_j, omega_j, G)`` on expert logits."""

    def __init__(self, experts: list[DenseExpert], gate: NoisyTopKGate):
        super().__init__()
        self.experts = nn.ModuleList(experts).requires_grad_(False)
        self.heads = nn.ModuleList(MaskedLinear(expert.head) for expert in experts)
        self.gate = gate
        self.experts.eval()

    def train(self, mode: bool = True) -> "OGMMMergedModel":
        super().train(mode)
        self.experts.eval()  # frozen encoders: running BN statistics, no dropout
        return self

    @staticmethod
    def gate_input(inst: DenseInstances) -> Tensor:
        """Masked mean of the raw node features (PROPOSED gate input)."""
        weight = inst.mask.unsqueeze(-1).to(inst.x.dtype)
        return (inst.x * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)

    def expert_logits(self, inst: DenseInstances) -> Tensor:
        """``[B, M, out]`` logits of the masked experts."""
        outputs = []
        for expert, head in zip(self.experts, self.heads):
            with torch.no_grad():
                h = expert.encode(inst)
            outputs.append(head(h))
        return torch.stack(outputs, dim=1)

    @staticmethod
    def combine(expert_logits: Tensor, gate: Tensor) -> Tensor:
        return (gate.unsqueeze(-1) * expert_logits).sum(dim=1)

    def forward(self, inst: DenseInstances) -> tuple[Tensor, Tensor]:
        gate = self.gate(self.gate_input(inst))
        return self.combine(self.expert_logits(inst), gate), gate


def merge_loss(
    model: OGMMMergedModel,
    inst: DenseInstances,
    *,
    task_type: str,
    lambda_gate: float,
    lambda_mask: float,
    gamma_p: float,
    gamma_v: float,
) -> tuple[Tensor, dict[str, float]]:
    """Eq. 20: ``C(y, Gamma(G)) + l_gate * cv^2(sum Gate) + l_mask * R_mask`` on one batch."""
    logits = model.expert_logits(inst)
    gate = model.gate(model.gate_input(inst))
    task_loss, _ = supervised_loss_from_logits(logits=model.combine(logits, gate), labels=inst.y, task_type=task_type)
    r_gate = cv_squared(gate.sum(dim=0))
    expert_loss = sum(
        supervised_loss_from_logits(logits=logits[:, j], labels=inst.y, task_type=task_type)[0]
        for j in range(logits.size(1))
    )
    r_mask = expert_loss + mask_regularizer([head.mask_tensors() for head in model.heads], gamma_p, gamma_v)
    loss = task_loss + float(lambda_gate) * r_gate + float(lambda_mask) * r_mask
    return loss, {"task": float(task_loss), "gate": float(r_gate), "mask": float(r_mask)}


__all__ = [
    "MaskedLinear",
    "NoisyTopKGate",
    "OGMMMergedModel",
    "cv_squared",
    "mask_regularizer",
    "merge_loss",
]
