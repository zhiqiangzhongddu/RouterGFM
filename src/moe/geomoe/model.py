"""GeoMoE model: Euclidean / hyperbolic / spherical GNN experts fused by a node-wise graph-aware gate.

Cao et al. (2026), Eqs. 3-6: ``h_fused(v) = sum_m w_m(v) t_m(v)`` over the
tangent-space expert embeddings ``t = (h^E, log_0 h^H, log_0 h^S)``, with
``w(v) = softmax(MLP(GCN(X, A))(v) / tau_g)``. The hyperbolic and spherical
experts reuse GraphMoRE's kappa-stereographic GCN with fixed curvatures (the
paper's hyperbolic expert is HGAT; no official code exists), so every
parameter is an ordinary Euclidean tensor and one Adam optimizer suffices.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from src.model.encoder import GNNEncoder
from src.moe.graphmore.manifold import RiemannianEncoder


class GraphAwareGate(nn.Module):
    """Eqs. 5-6: one GCN layer, a 2-layer MLP, and a tempered softmax over the experts."""

    def __init__(self, in_dim: int, hidden_dim: int, num_experts: int = 3, temperature: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.conv = GCNConv(in_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.mlp(self.conv(x, edge_index)) / self.temperature, dim=-1)


class GeoMoEModel(nn.Module):
    """Three geometric experts and a node-wise gate; ``forward(data) -> (h_fused, None)``.

    ``graph_repr`` is ``None``: the task owns the instance readout. The last
    gate ``[N, 3]`` and expert tangents ``[N, 3, d]`` are cached for the losses.
    """

    returns_layer_cache: bool = False

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        curvatures: Sequence[float] = (-1.0, 1.0),
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        k_hyp, k_sph = (float(k) for k in curvatures)
        if not (k_hyp < 0.0 < k_sph):
            raise ValueError(f"GeoMoE curvatures must be (hyperbolic < 0, spherical > 0); got {list(curvatures)}.")
        self.out_dim = int(hidden_dim)
        self.euclidean = GNNEncoder(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, num_layers=num_layers,
            model_type="gcn", dropout=dropout,
        )
        self.hyperbolic = RiemannianEncoder(k_hyp, in_dim, hidden_dim, hidden_dim, learnable=False)
        self.spherical = RiemannianEncoder(k_sph, in_dim, hidden_dim, hidden_dim, learnable=False)
        self.gate = GraphAwareGate(in_dim, hidden_dim, 3, gate_temperature, dropout)
        self.last_gate: Optional[torch.Tensor] = None
        self.last_expert: Optional[torch.Tensor] = None

    def expert_tangent(self, data) -> torch.Tensor:
        """Tangent-space expert embeddings ``[N, 3, d]`` in the order (E, H, S)."""
        h_euc, _ = self.euclidean(data)
        return torch.stack(
            [
                h_euc,
                self.hyperbolic.encode(data.x, data.edge_index),
                self.spherical.encode(data.x, data.edge_index),
            ],
            dim=1,
        )

    def forward(self, data):
        experts = self.expert_tangent(data)
        gate = self.gate(data.x, data.edge_index)
        self.last_gate, self.last_expert = gate, experts
        return (gate.unsqueeze(-1) * experts).sum(dim=1), None


__all__ = ["GraphAwareGate", "GeoMoEModel"]
