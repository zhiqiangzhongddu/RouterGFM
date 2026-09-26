"""GMoPE prompt-conditioned expert pool (Eq. 7)."""

from __future__ import annotations

import copy

import torch
from torch import nn

from src.model.encoder import GNNEncoder
from src.utils.pool import resolve_graph_repr

from .routing import soft_orthogonality_loss


class _BoundExpert(nn.Module):
    """``forward(data) = owner.expert_forward(index, data)``.

    Lets any repo ``PretrainTask.step(model, data, device)`` drive one
    prompted expert unchanged (the prompt is appended after the objective's
    own corruption/augmentation of ``data.x``).
    """

    returns_layer_cache = True

    def __init__(self, owner: "GMoPEModel", index: int):
        super().__init__()
        self.expert = owner.experts[index]
        self.index = int(index)
        self._owner = (owner,)  # tuple: do not register the parent as a submodule

    def forward(self, data):
        return self._owner[0].expert_forward(self.index, data)

    def get_layer_node_reprs(self) -> list[torch.Tensor]:
        return self.expert.get_layer_node_reprs()


class GMoPEModel(nn.Module):
    """M GNN experts, each fed ``[x || 1 p_m^T]`` with its own prompt ``p_m``."""

    def __init__(
        self,
        *,
        num_experts: int,
        in_dim: int,
        prompt_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        gnn_type: str,
        dropout: float,
        act: str,
        graph_pooling: str,
        use_batchnorm: bool,
    ):
        super().__init__()
        if int(num_experts) < 1:
            raise ValueError(f"[GMoPE] num_experts must be >= 1 (got {num_experts}).")
        self.num_experts = int(num_experts)
        self.in_dim = int(in_dim)
        self.prompt_dim = int(prompt_dim)
        self.out_dim = int(out_dim)
        self.pool_mode = str(graph_pooling)
        self.experts = nn.ModuleList(
            GNNEncoder(
                in_dim=self.in_dim + self.prompt_dim,
                hidden_dim=int(hidden_dim),
                out_dim=self.out_dim,
                num_layers=int(num_layers),
                model_type=str(gnn_type),
                act=str(act),
                dropout=float(dropout),
                graph_pooling=self.pool_mode,
                use_batchnorm=bool(use_batchnorm),
            )
            for _ in range(self.num_experts)
        )
        self.prompts = nn.Parameter(torch.empty(self.num_experts, self.prompt_dim))
        nn.init.xavier_uniform_(self.prompts)
        self._experts_frozen = False

    def expert_forward(self, m: int, data) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run expert ``m`` on a shallow copy of ``data`` whose ``x`` carries prompt ``m``."""
        x = data.x
        if int(x.size(-1)) != self.in_dim:
            raise ValueError(
                f"[GMoPE] Feature dim {int(x.size(-1))} != moe.gmope.in_dim {self.in_dim}."
            )
        prompted = copy.copy(data)
        prompted.x = torch.cat([x, self.prompts[m].unsqueeze(0).expand(x.size(0), -1)], dim=-1)
        return self.experts[m](prompted)

    def bound(self, m: int) -> nn.Module:
        return _BoundExpert(self, m)

    def pooled_all(self, data) -> torch.Tensor:
        """``[M, B, out_dim]`` pooled (sub)graph embeddings of every expert."""
        pooled = []
        for m in range(self.num_experts):
            node_repr, graph_repr = self.expert_forward(m, data)
            pooled.append(resolve_graph_repr(node_repr, graph_repr, data, self.pool_mode))
        return torch.stack(pooled, dim=0)

    def ortho_loss(self) -> torch.Tensor:
        return soft_orthogonality_loss(self.prompts)

    def freeze_experts(self) -> None:
        """Freeze expert weights and keep the experts in eval mode from now on."""
        self.experts.requires_grad_(False)
        self.experts.eval()
        self._experts_frozen = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self._experts_frozen:
            self.experts.eval()
        return self


__all__ = ["GMoPEModel"]
