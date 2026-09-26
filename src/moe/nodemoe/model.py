"""Node-MoE model: GIN gate over neighbour-difference features + ChebNetII experts (Eq. 1)."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected

from src.model.encoder import GNNEncoder

from .chebnet2 import FILTER_INITS, ChebNetIIExpert

GATE_FEATURE_NORMS = ("mean", "sym")


def resolve_expert_specs(expert_inits: Sequence[str], expert_alphas: Sequence[float]) -> list[tuple[str, float]]:
    """Validate and pair per-expert (filter init, alpha); m = len(expert_inits) >= 2."""
    inits = [str(kind).lower() for kind in expert_inits]
    alphas = [float(alpha) for alpha in expert_alphas]
    if len(inits) < 2:
        raise ValueError(f"Node-MoE needs at least 2 experts (got expert_inits={tuple(inits)}).")
    if len(inits) != len(alphas):
        raise ValueError(
            f"Node-MoE expert_inits {tuple(inits)} and expert_alphas {tuple(alphas)} differ in length; "
            "set moe.nodemoe.expert_alphas to one alpha per expert."
        )
    unknown = [kind for kind in inits if kind not in FILTER_INITS]
    if unknown:
        raise ValueError(f"Unknown Node-MoE filter init(s) {unknown}; expected {FILTER_INITS}.")
    return list(zip(inits, alphas))


def gate_input_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    norm: str = "mean",
) -> torch.Tensor:
    """``[x, |Ax - x|, |A^2 x - x|]`` with A = D^-1 A ("mean") or D^-1/2 A D^-1/2 ("sym").

    ``edge_index`` is expected symmetric; self loops are dropped and isolated
    nodes get ``Ax = 0``.
    """
    if norm not in GATE_FEATURE_NORMS:
        raise ValueError(f"Unknown Node-MoE gate_feature_norm '{norm}'; expected {GATE_FEATURE_NORMS}.")
    edge_index, _ = remove_self_loops(edge_index)
    src, dst = edge_index
    deg = torch.zeros(num_nodes, dtype=x.dtype, device=x.device)
    deg.index_add_(0, dst, torch.ones(dst.numel(), dtype=x.dtype, device=x.device))
    has_edges = deg > 0
    if norm == "mean":
        weight = (has_edges / deg.clamp(min=1.0))[dst]
    else:
        inv_sqrt = has_edges / deg.clamp(min=1.0).sqrt()
        weight = inv_sqrt[src] * inv_sqrt[dst]

    def _apply(h: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(h).index_add_(0, dst, weight.unsqueeze(-1) * h[src])

    ax = _apply(x)
    a2x = _apply(ax)
    return torch.cat([x, (ax - x).abs(), (a2x - x).abs()], dim=-1)


class NodeMoEModel(nn.Module):
    """``mix_i = sum_o softmax(GIN(Z, A))_{i,o} * E_o(A, X)_i``; returns node-level mixed logits."""

    returns_layer_cache = False

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        expert_inits: Sequence[str],
        expert_alphas: Sequence[float],
        K: int,
        expert_hidden_dim: int,
        expert_dropout: float,
        dprate: float,
        gate_hidden_dim: int,
        gate_num_layers: int,
        gate_dropout: float,
        gate_feature_norm: str,
        act: str = "relu",
    ):
        super().__init__()
        specs = resolve_expert_specs(expert_inits, expert_alphas)
        if gate_feature_norm not in GATE_FEATURE_NORMS:
            raise ValueError(
                f"Unknown Node-MoE gate_feature_norm '{gate_feature_norm}'; expected {GATE_FEATURE_NORMS}."
            )
        self.num_experts = len(specs)
        self.gate_feature_norm = str(gate_feature_norm)
        self.gate = GNNEncoder(
            in_dim=3 * int(in_dim),
            hidden_dim=int(gate_hidden_dim),
            out_dim=self.num_experts,
            num_layers=int(gate_num_layers),
            model_type="gin",
            act=act,
            dropout=float(gate_dropout),
            use_batchnorm=False,
        )
        self.experts = nn.ModuleList(
            ChebNetIIExpert(
                in_dim=int(in_dim),
                hidden_dim=int(expert_hidden_dim),
                out_dim=int(out_dim),
                K=int(K),
                init=kind,
                alpha=alpha,
                dropout=float(expert_dropout),
                dprate=float(dprate),
            )
            for kind, alpha in specs
        )
        self.last_gate_weights: torch.Tensor | None = None

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(mixed_logits [N, out_dim], gate_weights [N, m])``; symmetrises edges."""
        x = data.x.float()
        num_nodes = x.size(0)
        edge_index = to_undirected(data.edge_index, num_nodes=num_nodes)
        z = gate_input_features(x, edge_index, num_nodes, norm=self.gate_feature_norm)
        gate_logits, _ = self.gate(Data(x=z, edge_index=edge_index))
        gate = torch.softmax(gate_logits, dim=-1)
        expert_logits = torch.stack([expert(x, edge_index) for expert in self.experts], dim=1)
        mixed = (gate.unsqueeze(-1) * expert_logits).sum(dim=1)
        self.last_gate_weights = gate.detach()
        return mixed, gate

    def smoothing_loss(self) -> torch.Tensor:
        return torch.stack([expert.prop.smoothing_loss() for expert in self.experts]).sum()

    def param_groups(
        self,
        *,
        gate_lr: float,
        gate_wd: float,
        expert_lr: float,
        expert_wd: float,
        filter_lr: float,
        filter_wd: float,
    ) -> list[dict]:
        """Three Adam groups (App. C.2): gate, expert dense layers, expert filters."""
        return [
            {"params": list(self.gate.parameters()), "lr": float(gate_lr), "weight_decay": float(gate_wd)},
            {
                "params": [p for expert in self.experts for p in expert.dense_parameters()],
                "lr": float(expert_lr),
                "weight_decay": float(expert_wd),
            },
            {
                "params": [p for expert in self.experts for p in expert.filter_parameters()],
                "lr": float(filter_lr),
                "weight_decay": float(filter_wd),
            },
        ]


__all__ = ["GATE_FEATURE_NORMS", "NodeMoEModel", "gate_input_features", "resolve_expert_specs"]
