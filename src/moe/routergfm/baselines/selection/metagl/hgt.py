"""Heterogeneous Graph Transformer with the semantics of MetaGL's DGL layer, in plain torch.

Per relation r = (s -> t): ``k = K_s h_s``, ``q = Q_t h_t``, ``v = V_s h_s`` per
head; ``k' = k W_att[r]``, ``v' = v W_msg[r]``; score ``<q, k'> mu_r / sqrt(d_k)``;
softmax over the in-edges of each destination within r; message sum. Across
relations the mean over every relation type targeting the node type (a
relation without in-edges contributes 0; DGL ``cross_reducer='mean'``).
Update ``LayerNorm(sigmoid(s) Dropout(A t) + (1 - sigmoid(s)) h)`` without an
activation on t. (PyG's ``HGTConv`` differs: joint softmax, sum, GELU.)
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

from .network import Relation


class HGTLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        node_types: Sequence[str],
        relations: Sequence[Relation],
        n_heads: int = 4,
        dropout: float = 0.5,
        use_norm: bool = True,
    ):
        super().__init__()
        self.node_types, self.relations = list(node_types), list(relations)
        self.dim, self.n_heads, self.d_k = int(dim), int(n_heads), int(dim) // int(n_heads)
        self.k_lin = nn.ModuleDict({t: nn.Linear(dim, dim) for t in self.node_types})
        self.q_lin = nn.ModuleDict({t: nn.Linear(dim, dim) for t in self.node_types})
        self.v_lin = nn.ModuleDict({t: nn.Linear(dim, dim) for t in self.node_types})
        self.a_lin = nn.ModuleDict({t: nn.Linear(dim, dim) for t in self.node_types})
        self.norms = nn.ModuleDict({t: nn.LayerNorm(dim) for t in self.node_types}) if use_norm else None
        n_rel = len(self.relations)
        self.relation_pri = nn.Parameter(torch.ones(n_rel, self.n_heads))
        self.relation_att = nn.Parameter(torch.empty(n_rel, self.n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.empty(n_rel, self.n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(len(self.node_types)))
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def relation_attention(self, r: int, h: Dict[str, torch.Tensor], edge_index: torch.Tensor):
        """``(attention [E, H], transformed messages v' [N_src, H, d_k])`` of relation index *r*."""
        src_t, _, dst_t = self.relations[r]
        H, dk = self.n_heads, self.d_k
        k = torch.einsum("bij,ijk->bik", self.k_lin[src_t](h[src_t]).view(-1, H, dk), self.relation_att[r])
        v = torch.einsum("bij,ijk->bik", self.v_lin[src_t](h[src_t]).view(-1, H, dk), self.relation_msg[r])
        q = self.q_lin[dst_t](h[dst_t]).view(-1, H, dk)
        src, dst = edge_index
        score = (q[dst] * k[src]).sum(-1) * self.relation_pri[r] / math.sqrt(dk)
        return softmax(score, dst, num_nodes=h[dst_t].size(0)), v

    def forward(self, h: Dict[str, torch.Tensor], edges: Dict[Relation, torch.Tensor]) -> Dict[str, torch.Tensor]:
        incoming = {t: [] for t in self.node_types}
        for r, rel in enumerate(self.relations):
            dst_t = rel[2]
            n_dst = h[dst_t].size(0)
            out = h[dst_t].new_zeros(n_dst, self.n_heads, self.d_k)
            edge_index = edges.get(rel)
            if edge_index is not None and edge_index.numel() > 0:
                att, v = self.relation_attention(r, h, edge_index)
                out = out.index_add(0, edge_index[1], v[edge_index[0]] * att.unsqueeze(-1))
            incoming[dst_t].append(out.view(n_dst, self.dim))
        new_h = {}
        for i, t in enumerate(self.node_types):
            agg = torch.stack(incoming[t]).mean(0)
            alpha = torch.sigmoid(self.skip[i])
            out = self.drop(self.a_lin[t](agg)) * alpha + h[t] * (1 - alpha)
            new_h[t] = self.norms[t](out) if self.norms is not None else out
        return new_h


class HGT(nn.Module):
    """``h0 = GELU(Adapt_t x)``; ``n_layers`` :class:`HGTLayer`; per-type output Linear."""

    def __init__(
        self,
        n_inp: int,
        n_hid: int,
        n_out: int,
        node_types: Sequence[str],
        relations: Sequence[Relation],
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.5,
        use_norm: bool = True,
    ):
        super().__init__()
        self.adapt = nn.ModuleDict({t: nn.Linear(n_inp, n_hid) for t in node_types})
        self.layers = nn.ModuleList(
            [HGTLayer(n_hid, node_types, relations, n_heads, dropout, use_norm) for _ in range(int(n_layers))]
        )
        self.out = nn.ModuleDict({t: nn.Linear(n_hid, n_out) for t in node_types})

    def forward(self, x: Dict[str, torch.Tensor], edges: Dict[Relation, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = {t: F.gelu(self.adapt[t](x[t])) for t in self.adapt}
        for layer in self.layers:
            h = layer(h, edges)
        return {t: self.out[t](h[t]) for t in h}


__all__ = ["HGT", "HGTLayer"]
