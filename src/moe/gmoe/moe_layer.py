"""Sparsely-gated mixture-of-experts message-passing layer for GMoE.

Ported from the official GMoE implementation
(``ref_repos/Graph-Mixture-of-Experts/graphproppred/{moe,conv}.py``),
which itself adapts David Rau's port of the Sparsely-Gated MoE
("Outrageously Large Neural Networks", https://arxiv.org/abs/1701.06538).

Differences from the reference, all driven by repo scope:

* Experts are this repo's GNN convs (``build_conv``) operating on
  continuous SVD features — no ``AtomEncoder`` / ``BondEncoder`` and no
  ``edge_attr`` (edge features are out of scope).
* The multi-hop expert receptive field is realised by passing a
  ``hop``-hop ``edge_index`` (see :mod:`.two_hop`) instead of a
  separately-built two-hop molecular graph.
* The noisy top-k gate, load-balancing loss (``cv_squared`` of
  importance and load) and the dense gate-weighted expert combination
  are preserved verbatim.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from src.model.encoder import build_conv


class SparseMoEConv(nn.Module):
    """One GMoE message-passing layer.

    Holds ``num_experts`` GNN-conv experts (+ optional per-expert
    BatchNorm). The first ``num_experts_1hop`` experts consume the 1-hop
    ``edge_index``; the rest consume the multi-hop ``edge_index``. A
    noisy top-k gate produces per-node expert weights, and the layer
    returns ``(output, load_balance_loss)`` where the loss is already
    scaled by ``coef`` (matching the reference, which bakes ``coef`` into
    the MoE layer and adds the raw value to the task loss).
    """

    def __init__(
        self,
        emb_dim: int,
        num_experts: int,
        num_experts_1hop: int,
        k: int,
        gnn_type: str,
        act: nn.Module,
        coef: float = 1e-2,
        noisy_gating: bool = True,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if k > num_experts:
            raise ValueError(f"k ({k}) must be <= num_experts ({num_experts}).")
        if not 0 <= num_experts_1hop <= num_experts:
            raise ValueError(
                f"num_experts_1hop ({num_experts_1hop}) must be in [0, num_experts={num_experts}]."
            )

        self.emb_dim = emb_dim
        self.num_experts = num_experts
        self.num_experts_1hop = num_experts_1hop
        self.k = k
        self.loss_coef = float(coef)
        self.noisy_gating = bool(noisy_gating)
        self.use_batchnorm = bool(use_batchnorm)

        self.experts_conv = nn.ModuleList(
            build_conv(gnn_type, emb_dim, emb_dim, act) for _ in range(num_experts)
        )
        if self.use_batchnorm:
            self.experts_bn = nn.ModuleList(
                nn.BatchNorm1d(emb_dim) for _ in range(num_experts)
            )
        else:
            self.experts_bn = None

        self.w_gate = nn.Parameter(torch.zeros(emb_dim, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(emb_dim, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    # ------------------------------------------------------------------ #
    # Load-balancing helpers (verbatim from the reference)
    # ------------------------------------------------------------------ #
    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        """Squared coefficient of variation; 0 for a singleton tensor."""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0.0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates: torch.Tensor) -> torch.Tensor:
        """Number of examples routed to each expert (gate > 0)."""
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Differentiable probability that each value falls in the top-k."""
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def noisy_top_k_gating(self, x: torch.Tensor, train: bool, noise_epsilon: float = 1e-2):
        """Noisy top-k gating (https://arxiv.org/abs/1701.06538)."""
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(
                clean_logits, noisy_logits, noise_stddev, top_logits
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_index_multihop: torch.Tensor | None = None,
    ):
        """Run the MoE layer.

        Args:
            x: ``[num_nodes, emb_dim]`` node features.
            edge_index: 1-hop connectivity for the 1-hop experts.
            edge_index_multihop: multi-hop connectivity for the remaining
                experts. Required when ``num_experts_1hop < num_experts``.

        Returns:
            ``(output [num_nodes, emb_dim], load_balance_loss scalar)``.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss = loss * self.loss_coef

        if self.num_experts_1hop < self.num_experts and edge_index_multihop is None:
            raise ValueError(
                "edge_index_multihop is required when there are multi-hop experts "
                f"(num_experts_1hop={self.num_experts_1hop} < num_experts={self.num_experts})."
            )

        expert_outputs = []
        for i in range(self.num_experts):
            ei = edge_index if i < self.num_experts_1hop else edge_index_multihop
            out_i = self.experts_conv[i](x, ei)
            if self.experts_bn is not None:
                out_i = self.experts_bn[i](out_i)
            expert_outputs.append(out_i)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [num_nodes, num_experts, emb_dim]

        # Dense gate-weighted combination (reference uses mean over experts).
        y = gates.unsqueeze(dim=-1) * expert_outputs
        y = y.mean(dim=1)
        return y, loss


__all__ = ["SparseMoEConv"]
