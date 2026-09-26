"""Context-graph encoder (Eq. 3), application-expert scorer (Eq. 4), retrieval key k_phi.

Node types and relations come from the context graph (``context_graph.py``);
every relation (including reverses) has its own message function, so messages
flow in both directions. Parameters are shared within a node type and within a
relation, and aggregation is a mean, so the encoder is equivariant to
relabeling nodes within a type (Prop. 4) and its per-node scale does not grow
with degree when applications or experts are inserted.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Relation = Tuple[str, str, str]


def _relation_key(relation: Relation) -> str:
    return "__".join(relation)


class RelationConv(nn.Module):
    """``psi_r(h_u, h_v, b_uv) = MLP([h_u; h_v; b_uv])`` mean-aggregated over incoming neighbours."""

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        msg = self.mlp(torch.cat([h_src[src], h_dst[dst], edge_attr.to(h_src.dtype)], dim=-1))
        out = msg.new_zeros(h_dst.shape[0], msg.shape[-1]).index_add_(0, dst, msg)
        deg = torch.bincount(dst, minlength=h_dst.shape[0]).clamp(min=1).to(out.dtype)
        return out / deg.unsqueeze(-1)


class RouterGFMModel(nn.Module):
    """Type projections ``P_kappa``, relation-wise message passing, pair scorer ``q_theta``, key ``k_phi``."""

    def __init__(
        self,
        in_dims: Mapping[str, int],
        relations: Iterable[Relation],
        cfg_router,
        desc_dim: int,
        *,
        edge_dim: int = 2,
    ) -> None:
        super().__init__()
        hidden = int(cfg_router.hidden_dim)
        key_hidden = int(cfg_router.key_hidden_dim)
        self.hidden_dim = hidden
        self.relations = [tuple(r) for r in relations]
        self.proj = nn.ModuleDict({t: nn.Linear(int(d), hidden) for t, d in in_dims.items()})
        self.layers = nn.ModuleList(
            nn.ModuleDict({_relation_key(r): RelationConv(hidden, int(edge_dim)) for r in self.relations})
            for _ in range(int(cfg_router.num_layers))
        )
        self.dropout = nn.Dropout(float(cfg_router.dropout))
        self.scorer = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.key_net = nn.Sequential(
            nn.Linear(int(desc_dim) + hidden, key_hidden),
            nn.ReLU(),
            nn.Linear(key_hidden, int(cfg_router.key_dim)),
        )

    def project(self, x: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """``h^(0)_v = P_kappa(v)(x_v)``."""
        return {t: F.relu(self.proj[t](feat)) for t, feat in x.items()}

    def encode(
        self,
        x: Mapping[str, torch.Tensor],
        edge_index: Mapping[Relation, torch.Tensor],
        edge_attr: Mapping[Relation, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Eq. 3: ``h^(l+1) = h^(l) + sigma(sum_r mean_{u in N_r(v)} psi_r(h_u, h_v, b_uv))``."""
        h = self.project(x)
        for convs in self.layers:
            agg = {t: torch.zeros_like(v) for t, v in h.items()}
            for relation, index in edge_index.items():
                src_t, _, dst_t = relation
                agg[dst_t] = agg[dst_t] + convs[_relation_key(relation)](
                    h[src_t], h[dst_t], index, edge_attr[relation]
                )
            h = {t: h[t] + self.dropout(F.relu(agg[t])) for t in h}
        return h

    def score(self, h_app: torch.Tensor, h_exp: torch.Tensor) -> torch.Tensor:
        """Eq. 4: ``mu_hat = softplus(q_theta([h_a; h_e]))`` for aligned pairs ``[B, d] -> [B]``."""
        return F.softplus(self.scorer(torch.cat([h_app, h_exp], dim=-1)).squeeze(-1))

    def keys(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """``k_phi(z, v_e)`` with ``v_e = h^(0)_e`` for aligned rows ``[n, D], [n, d] -> [n, d_k]``."""
        return self.key_net(torch.cat([z, v], dim=-1))


__all__ = ["RelationConv", "RouterGFMModel"]
