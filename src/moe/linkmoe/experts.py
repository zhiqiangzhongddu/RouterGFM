"""Link-MoE experts: HeaRT-style MLP / GCN, NCN, and SEAL-GCN link predictors.

Full-graph experts take ``(x, edge_index, pairs)`` and return one logit per
pair; ``edge_index`` is the message graph during training and the eval
context graph at evaluation. SEAL scores enclosing subgraphs whose target
link is absent. All experts return logits; probabilities are ``sigmoid``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import shortest_path
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops, to_undirected

from src.model.encoder import GNNEncoder


class HadamardPredictor(nn.Module):
    """HeaRT ``mlp_score`` on ``h_i * h_j`` without the final sigmoid (returns logits)."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [1]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        x = h_i * h_j
        for lin in self.lins[:-1]:
            x = self.dropout(torch.relu(lin(x)))
        return self.lins[-1](x).view(-1)


class MLPLinkExpert(nn.Module):
    """Feature-proximity expert: ``h = Dropout(ReLU(Lin(x)))`` per layer, Hadamard predictor."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, predictor_layers: int, dropout: float):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = nn.Dropout(dropout)
        self.predictor = HadamardPredictor(hidden_dim, hidden_dim, predictor_layers, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        del edge_index  # feature-only expert
        h = x
        for lin in self.lins:
            h = self.dropout(torch.relu(lin(h)))
        return self.predictor(h[pairs[0]], h[pairs[1]])


class GCNLinkExpert(nn.Module):
    """HeaRT GCN: ``GNNEncoder(gcn)`` node embeddings + Hadamard predictor."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, predictor_layers: int, dropout: float):
        super().__init__()
        self.encoder = GNNEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            model_type="gcn",
            act="relu",
            dropout=dropout,
        )
        self.predictor = HadamardPredictor(hidden_dim, hidden_dim, predictor_layers, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(Data(x=x, edge_index=edge_index))
        return self.predictor(h[pairs[0]], h[pairs[1]])


def common_neighbor_index(edge_index: torch.Tensor, num_nodes: int, pairs: torch.Tensor):
    """``(pair_pos, node)`` for every common neighbour of each pair on the undirected graph.

    Expands the lower-degree endpoint's neighbours and keeps those adjacent to
    the other endpoint (sorted edge-key lookup), so memory is ``O(sum min deg)``.
    """
    ei, _ = remove_self_loops(edge_index)
    ei = to_undirected(ei, num_nodes=num_nodes)  # coalesced: sorted by (row, col)
    row, col = ei
    keys = row * num_nodes + col
    deg = torch.bincount(row, minlength=num_nodes)
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=ei.device)
    rowptr[1:] = torch.cumsum(deg, 0)

    i, j = pairs[0].to(ei.device), pairs[1].to(ei.device)
    swap = deg[i] > deg[j]
    a, b = torch.where(swap, j, i), torch.where(swap, i, j)
    counts = deg[a]
    total = int(counts.sum())
    if total == 0 or keys.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=ei.device)
        return empty, empty
    pair_pos = torch.repeat_interleave(torch.arange(a.numel(), device=ei.device), counts)
    group_start = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    offsets = torch.arange(total, device=ei.device) - group_start
    nbr = col[rowptr[a][pair_pos] + offsets]
    query = b[pair_pos] * num_nodes + nbr
    loc = torch.searchsorted(keys, query).clamp(max=keys.numel() - 1)
    hit = keys[loc] == query
    return pair_pos[hit], nbr[hit]


def _ln(dim: int, enabled: bool) -> nn.Module:
    return nn.LayerNorm(dim) if enabled else nn.Identity()


class NCNLinkExpert(nn.Module):
    """Neural Common Neighbour (``CNLinkPredictor``) over a GCNConv encoder.

    ``s = lin(beta * xcnlin(sum_{u in N(i) & N(j)} h_u) + xijlin(h_i * h_j))``.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        layer_norm: bool = True,
        beta: float = 1.0,
    ):
        super().__init__()
        H = hidden_dim
        self.convs = nn.ModuleList(GCNConv(in_dim if k == 0 else H, H) for k in range(num_layers))
        self.acts = nn.ModuleList(
            nn.Sequential(_ln(H, layer_norm), nn.Dropout(dropout), nn.ReLU()) for _ in range(num_layers - 1)
        )
        self.beta = nn.Parameter(beta * torch.ones(1))
        self.xcnlin = nn.Sequential(
            nn.Linear(H, H), nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(H, H), _ln(H, layer_norm), nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(H, H),
        )
        self.xijlin = nn.Sequential(nn.Linear(H, H), _ln(H, layer_norm), nn.Dropout(dropout), nn.ReLU(), nn.Linear(H, H))
        self.lin = nn.Sequential(nn.Linear(H, H), _ln(H, layer_norm), nn.Dropout(dropout), nn.ReLU(), nn.Linear(H, 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        h = x
        for k, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if k < len(self.acts):
                h = self.acts[k](h)
        pair_pos, nbr = common_neighbor_index(edge_index, x.size(0), pairs)
        cn = torch.zeros(pairs.size(1), h.size(1), dtype=h.dtype, device=h.device).index_add_(0, pair_pos, h[nbr])
        xij = self.xijlin(h[pairs[0]] * h[pairs[1]])
        return self.lin(self.xcnlin(cn) * self.beta + xij).view(-1)


class SEALGCN(nn.Module):
    """SEAL_OGB ``GCN`` with center pooling on ``[Emb(z_DRNL) || x]`` node inputs."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, max_z: int, dropout: float):
        super().__init__()
        self.z_embedding = nn.Embedding(max_z, hidden_dim)
        self.convs = nn.ModuleList(
            GCNConv(hidden_dim + in_dim if k == 0 else hidden_dim, hidden_dim) for k in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, batch) -> torch.Tensor:
        h = torch.cat([self.z_embedding(batch.z), batch.x.float()], dim=-1)
        for conv in self.convs[:-1]:
            h = self.dropout(torch.relu(conv(h, batch.edge_index)))
        h = self.convs[-1](h, batch.edge_index)
        u, v = batch.edge_label_index
        h = self.dropout(torch.relu(self.lin1(h[u] * h[v])))
        return self.lin2(h).view(-1)


def drnl_labels(edge_index: torch.Tensor, num_nodes: int, u: int, v: int, max_z: int) -> torch.Tensor:
    """Double-radius node labels (SEAL_OGB ``drnl_node_labeling``), clamped to ``max_z - 1``.

    ``d_u`` is measured with ``v`` removed and ``d_v`` with ``u`` removed;
    endpoints get 1 and nodes unreachable from either endpoint get 0.
    """
    src, dst = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
    ei = edge_index.detach().cpu().numpy().astype(np.int64)
    A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(num_nodes, num_nodes))
    keep_wo_dst = np.array([k for k in range(num_nodes) if k != dst], dtype=np.int64)
    keep_wo_src = np.array([k for k in range(num_nodes) if k != src], dtype=np.int64)
    d_src = shortest_path(A[keep_wo_dst][:, keep_wo_dst], directed=False, unweighted=True, indices=src)
    d_dst = shortest_path(A[keep_wo_src][:, keep_wo_src], directed=False, unweighted=True, indices=dst - 1)
    d_src = np.insert(d_src, dst, 0.0)
    d_dst = np.insert(d_dst, src, 0.0)
    finite = np.isfinite(d_src) & np.isfinite(d_dst)
    d = np.where(finite, d_src + d_dst, 0.0)
    half, odd = d // 2, d % 2
    z = np.where(finite, 1.0 + np.minimum(d_src, d_dst) + half * (half + odd - 1.0), 0.0)
    z[src] = 1.0
    z[dst] = 1.0
    return torch.from_numpy(np.clip(z, 0, max_z - 1).astype(np.int64))


def _build_mlp(lcfg, in_dim: int) -> nn.Module:
    c = lcfg.mlp
    return MLPLinkExpert(in_dim, int(c.hidden_dim), int(c.num_layers), int(c.predictor_layers), float(c.dropout))


def _build_gcn(lcfg, in_dim: int) -> nn.Module:
    c = lcfg.gcn
    return GCNLinkExpert(in_dim, int(c.hidden_dim), int(c.num_layers), int(c.predictor_layers), float(c.dropout))


def _build_ncn(lcfg, in_dim: int) -> nn.Module:
    c = lcfg.ncn
    return NCNLinkExpert(in_dim, int(c.hidden_dim), int(c.num_layers), float(c.dropout), bool(c.layer_norm), float(c.beta))


def _build_seal(lcfg, in_dim: int) -> nn.Module:
    c = lcfg.seal
    return SEALGCN(in_dim, int(c.hidden_dim), int(c.num_layers), int(c.max_z), float(c.dropout))


# name -> builder(cfg.moe.linkmoe, in_dim); "seal" consumes the induced view.
EXPERT_BUILDERS: dict[str, Callable[[object, int], nn.Module]] = {
    "mlp": _build_mlp,
    "gcn": _build_gcn,
    "ncn": _build_ncn,
    "seal": _build_seal,
}
SUBGRAPH_EXPERTS = ("seal",)


__all__ = [
    "EXPERT_BUILDERS",
    "GCNLinkExpert",
    "HadamardPredictor",
    "MLPLinkExpert",
    "NCNLinkExpert",
    "SEALGCN",
    "SUBGRAPH_EXPERTS",
    "common_neighbor_index",
    "drnl_labels",
]
