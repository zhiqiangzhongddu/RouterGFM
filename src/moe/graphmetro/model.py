"""GraphMETRO model: a gating GNN over ``K+1`` independent GNN experts and a shared head.

``h(G) = softmax(phi(G)) . [xi_0(G), ..., xi_K(G)]`` (Eq. 2, soft mixture, the
official default) and ``f = mu o h``. Expert 0 is the reference model. Every
encoder is a repo :class:`GNNEncoder`; one vector per instance comes from the
shared shift-baseline readout (target node / endpoint Hadamard / pooled graph).
"""

from __future__ import annotations

import torch
from torch import nn

from src.model.encoder import GNNEncoder
from src.moe.shift_eval import instance_readout


class GraphMETROModel(nn.Module):
    """Gate GNN + ``num_experts`` expert GNNs (index 0 = reference) + shared classifier head."""

    def __init__(
        self,
        *,
        in_dim: int,
        backbone: str,
        num_layers: int,
        hidden_dim: int,
        dropout: float,
        use_batchnorm: bool,
        num_experts: int,
        task_level_raw: str,
        graph_pooling: str,
        head: nn.Module,
    ):
        super().__init__()
        self.task_level_raw = str(task_level_raw).lower()
        self.pool_mode = str(graph_pooling)

        def encoder() -> GNNEncoder:
            return GNNEncoder(
                in_dim=int(in_dim),
                hidden_dim=int(hidden_dim),
                out_dim=int(hidden_dim),
                num_layers=int(num_layers),
                model_type=str(backbone),
                act="relu",
                dropout=float(dropout),
                graph_pooling=self.pool_mode,
                use_batchnorm=bool(use_batchnorm),
            )

        self.gate = encoder()
        self.gate_head = nn.Linear(int(hidden_dim), int(num_experts))
        self.experts = nn.ModuleList(encoder() for _ in range(int(num_experts)))
        self.head = head

    def instance_repr(self, encoder: nn.Module, data) -> torch.Tensor:
        """``[B, hidden]``: the encoder's node representations read out per instance."""
        node_repr, _ = encoder(data)
        return instance_readout(node_repr, data, self.task_level_raw, self.pool_mode)

    def gate_logits(self, data) -> torch.Tensor:
        """``[B, K+1]`` shift-component logits."""
        return self.gate_head(self.instance_repr(self.gate, data))

    def expert_reprs(self, data) -> torch.Tensor:
        """``[B, K+1, hidden]`` instance representations of every expert."""
        return torch.stack([self.instance_repr(expert, data) for expert in self.experts], dim=1)

    @staticmethod
    def mix(reprs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """``[B, hidden]`` weighted sum of expert representations (weights ``[B, K+1]``)."""
        return torch.bmm(weights.unsqueeze(1), reprs).squeeze(1)

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        """Head logits and softmax gate weights on the (untransformed) instances."""
        weights = torch.softmax(self.gate_logits(data), dim=-1)
        return self.head(self.mix(self.expert_reprs(data), weights)), weights

    def param_groups(self, *, moe_lr: float, classifier_lr: float) -> list[dict]:
        """Official Adam groups: gate and experts at ``moe_lr``, the classifier at ``classifier_lr``."""
        return [
            {"params": list(self.gate.parameters()) + list(self.gate_head.parameters()), "lr": float(moe_lr)},
            {"params": list(self.experts.parameters()), "lr": float(moe_lr)},
            {"params": list(self.head.parameters()), "lr": float(classifier_lr)},
        ]


__all__ = ["GraphMETROModel"]
