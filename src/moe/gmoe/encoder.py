"""GMoE encoder: a GNN whose layers are sparse mixtures of multi-hop experts.

Mirrors ``GNN_SpMoE_node`` from the official GMoE implementation, adapted
to this repo's continuous SVD features (no ``AtomEncoder``) and edge-free
message passing. The encoder exposes the same
``forward(data) -> (node_repr, graph_repr)`` contract as
:class:`src.model.encoder.GNNEncoder` so the supervised task heads and
pooling helpers can be reused unchanged. After each forward pass the
aggregated (already ``coef``-scaled) load-balancing loss is available on
``self.load_balance_loss`` for the task to add to the utility loss.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.model.activations import get_activation

from .moe_layer import SparseMoEConv
from .two_hop import compute_multi_hop_edge_index


class GMoEEncoder(nn.Module):
    """Sparsely-gated mixture-of-experts GNN encoder."""

    #: GMoE does not cache per-layer node representations.
    returns_layer_cache: bool = False

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        gnn_type: str = "gcn",
        num_experts: int = 8,
        num_experts_1hop: int = 4,
        k: int = 4,
        coef: float = 1.0,
        expert_hop: int = 2,
        noisy_gating: bool = True,
        dropout: float = 0.5,
        act: str = "relu",
        residual: bool = False,
        jk: str = "last",
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("GMoE requires num_layers >= 2.")
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.num_experts_1hop = num_experts_1hop
        self.expert_hop = int(expert_hop)
        self.residual = bool(residual)
        self.jk = str(jk).lower()
        if self.jk not in {"last", "sum"}:
            raise ValueError(f"Unsupported jk='{jk}'. Use 'last' or 'sum'.")
        self.act = get_activation(act)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim

        # Continuous-feature replacement for the reference AtomEncoder.
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList(
            SparseMoEConv(
                emb_dim=hidden_dim,
                num_experts=num_experts,
                num_experts_1hop=num_experts_1hop,
                k=k,
                gnn_type=gnn_type,
                act=self.act,
                coef=coef,
                noisy_gating=noisy_gating,
                use_batchnorm=use_batchnorm,
            )
            for _ in range(num_layers)
        )

        # Populated each forward pass; consumed by the GMoE task.
        self.load_balance_loss: torch.Tensor | float = 0.0

    @property
    def _needs_multihop(self) -> bool:
        # Whenever some experts are not 1-hop experts they need a second
        # edge_index. compute_multi_hop_edge_index degrades gracefully to
        # the 1-hop graph when expert_hop <= 1, so this stays valid (just
        # no receptive-field diversity) instead of raising at runtime.
        return self.num_experts_1hop < self.num_experts

    def forward(self, data):
        x = data.x
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("GMoE requires edge_index for message passing.")
        batch = getattr(data, "batch", None)

        edge_index_multihop: Optional[torch.Tensor] = None
        if self._needs_multihop:
            edge_index_multihop = compute_multi_hop_edge_index(
                edge_index, num_nodes=x.size(0), hop=self.expert_hop
            )

        h_list = [self.input_proj(x)]
        load_balance_loss = x.new_zeros(())
        for layer in range(self.num_layers):
            h, lb = self.layers[layer](h_list[layer], edge_index, edge_index_multihop)
            load_balance_loss = load_balance_loss + lb

            if layer == self.num_layers - 1:
                # No activation on the final layer (matches the reference).
                h = self.dropout(h)
            else:
                h = self.dropout(self.act(h))

            if self.residual:
                h = h + h_list[layer]
            h_list.append(h)

        self.load_balance_loss = load_balance_loss / self.num_layers

        if self.jk == "last":
            node_repr = h_list[-1]
        else:  # sum
            node_repr = torch.stack(h_list, dim=0).sum(dim=0)

        # graph_repr is intentionally None: pooling is applied by the task
        # via the configured pooling mode (matches GNNEncoder's contract
        # when batch is absent), keeping a single pooling source of truth.
        graph_repr = None
        return node_repr, graph_repr


__all__ = ["GMoEEncoder"]
