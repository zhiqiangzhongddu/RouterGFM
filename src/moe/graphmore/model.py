"""GraphMoRE model: a mixture of Riemannian experts over induced subgraphs.

Adapts the official GraphMoRE (AAAI 2025) design to this repo's unified
subgraph-task representation. The reference is transductive single-graph;
here the prediction unit is an **induced subgraph** (one per node/edge/
graph instance), so:

* ``RiemannianExperts`` runs ``K`` :class:`RiemannianEncoder` experts (one
  per curvature in ``init_curvs``) over the batched subgraph and returns
  per-node tangent-space embeddings concatenated across experts.
* ``SubgraphGating`` replaces the reference's per-node ego-subgraph
  ``Sampler`` + gating GNN: it encodes the batched subgraph with a plain
  GCN, mean-pools per subgraph, and produces one softmax weight vector per
  subgraph (each induced subgraph already *is* the local structural sample
  the reference's ego-sampler produced).
* ``GraphMoREModel`` broadcasts each subgraph's gate weights to its nodes,
  scales the per-expert embedding blocks, and exposes the GMoE-style
  ``forward(data) -> (node_repr, graph_repr)`` contract so the shared
  supervised head / pooling / loss / metric helpers can be reused.

The optional distortion regulariser (the paper's signature term) is
computed within each subgraph and is **disabled by default** (``coef_dis``
defaults to ``0.0``): on small induced subgraphs nearly all intra-subgraph
edges are distance-1 and the gate is per-subgraph, so it degenerates into a
weak stabiliser.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from .manifold import RiemannianEncoder


class RiemannianExperts(nn.Module):
    """K curvature-specific kappa-GCN experts producing tangent embeddings."""

    def __init__(
        self,
        init_curvs: Sequence[float],
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        learnable: bool = True,
    ):
        super().__init__()
        self.init_curvs = [float(c) for c in init_curvs]
        self.out_dim = int(out_dim)
        self.experts = nn.ModuleList()
        for curv in self.init_curvs:
            # Euclidean (k==0) experts keep a fixed curvature; signed
            # curvatures are learnable so the manifold adapts during training.
            expert_learnable = learnable and curv != 0
            self.experts.append(
                RiemannianEncoder(curv, in_dim, hidden_dim, out_dim, learnable=expert_learnable)
            )

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return concatenated per-node tangent embeddings ``[N, K*out_dim]``."""
        embeds = [expert.encode(x, edge_index) for expert in self.experts]
        return torch.cat(embeds, dim=-1)


class SubgraphGating(nn.Module):
    """Per-subgraph expert router: a GCN + mean-pool + softmax over K experts."""

    def __init__(self, in_dim: int, hidden_dim: int, num_experts: int, temperature: float = 1.0):
        super().__init__()
        self.encoder1 = GCNConv(in_dim, hidden_dim)
        self.encoder2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_experts, bias=True)
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.encoder1(x, edge_index)
        h = self.encoder2(h, edge_index)
        pooled = global_mean_pool(h, batch)  # [B, hidden_dim]
        logits = self.classifier(pooled)  # [B, K]
        return F.softmax(logits / self.temperature, dim=-1)


class GraphMoREModel(nn.Module):
    """Mixture of Riemannian experts with a per-subgraph topology gate.

    Exposes the ``forward(data) -> (node_repr, graph_repr)`` contract of
    :class:`src.model.encoder.GNNEncoder`: ``node_repr`` is the gate-weighted
    per-node mixture embedding (``[N, K*embed_dim]``) and ``graph_repr`` is
    ``None`` so the task head owns pooling (single pooling source of truth).
    """

    #: GraphMoRE does not cache per-layer node representations.
    returns_layer_cache: bool = False

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        embed_dim: int,
        init_curvs: Sequence[float],
        gating_hidden_dim: int,
        learnable: bool = True,
        coef_dis: float = 0.0,
        gating_temperature: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.coef_dis = float(coef_dis)
        self.experts = RiemannianExperts(
            init_curvs=init_curvs,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            learnable=learnable,
        )
        self.num_experts = self.experts.num_experts
        self.out_dim = self.num_experts * self.embed_dim
        self.gating = SubgraphGating(
            in_dim=in_dim,
            hidden_dim=gating_hidden_dim,
            num_experts=self.num_experts,
            temperature=gating_temperature,
        )
        # Populated each forward pass; consumed by the task / inspection.
        self.last_gating: Optional[torch.Tensor] = None
        self.distortion_loss: torch.Tensor | float = 0.0

    def forward(self, data):
        x = data.x
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("GraphMoRE requires edge_index for message passing.")
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        expert_emb = self.experts(x, edge_index)  # [N, K*embed_dim]
        gate = self.gating(x, edge_index, batch)  # [B, K]
        self.last_gating = gate

        # Broadcast each subgraph's gate to its nodes, then to per-expert blocks.
        gate_nodes = gate[batch]  # [N, K]
        gate_rep = gate_nodes.repeat_interleave(self.embed_dim, dim=1)  # [N, K*embed_dim]
        node_repr = expert_emb * gate_rep

        if self.coef_dis > 0.0:
            self.distortion_loss = self._compute_distortion(expert_emb, gate_nodes, edge_index)
        else:
            self.distortion_loss = x.new_zeros(())

        return node_repr, None

    def _compute_distortion(
        self,
        expert_emb: torch.Tensor,
        gate_nodes: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Within-subgraph distortion regulariser (off by default).

        Faithful to the reference ``compute_distortion`` but over the
        batched subgraph edges: each edge's topological distance is 1 (a
        direct edge), and the per-edge expert weights come from the owning
        subgraph's gate (identical for both endpoints, since induced
        subgraphs are disconnected in the batch). Mostly a stabiliser.
        """
        if edge_index.numel() == 0:
            return expert_emb.new_zeros(())
        src, dst = edge_index[0], edge_index[1]
        diff = (expert_emb[src] - expert_emb[dst]) ** 2  # [E, K*embed_dim]
        diff = diff.reshape(diff.size(0), self.num_experts, self.embed_dim).sum(dim=2)  # [E, K]
        weights = F.softmax(gate_nodes[src] * gate_nodes[dst], dim=1)  # [E, K]
        dis = torch.sum(diff * weights, dim=-1)  # [E]; target graph distance == 1
        return torch.abs(dis - 1.0).mean()


__all__ = ["RiemannianExperts", "SubgraphGating", "GraphMoREModel"]
