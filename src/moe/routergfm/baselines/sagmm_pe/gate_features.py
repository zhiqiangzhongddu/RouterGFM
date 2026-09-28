"""Label-free gate inputs X' of SAGMM (paper Eqs. 3-5; official ``get_graph_info``, ``multihop_lap``).

RouterGFM applications are induced: an instance is a node ego-subgraph, a link
enclosing subgraph (queried edge absent), or a whole graph. The node-context
module is therefore computed on each instance graph, whose nodes also form the
SGA population of that instance (the official node/LP code uses the full graph):

* node: population = subgraph nodes, query row = the target node's row;
* edge: population = subgraph nodes, query row = mean of the two endpoint rows
  (one gate per pair, because the frozen readout is a pair representation);
* graph: query row = mean node feature (continuous analogue of the official
  ``global_mean_pool(AtomEncoder(x))``); the population is the batch of graphs.

The official X_g is a full-graph positional code; eigenvectors (and their
random signs) of different instance graphs are not comparable across
instances, so X' defaults to X_loc (``sagmm_pe.gate_input_p = 0``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


def multihop_lap_features(x: torch.Tensor, edge_index: torch.Tensor, p: int, *, sign_seed: int) -> torch.Tensor:
    """``[n, d + p]`` = ``[(X + X1 + X2) / 3 || X_g]`` of one (undirected) graph.

    ``X1 = D~^-1 A~ X``, ``X2 = D~^-1 A~ X1`` with ``A~ = A + I``. ``X_g`` holds the
    eigenvectors 2..p+1 (ascending eigenvalues) of ``I - D^-1/2 A D^-1/2`` (no self
    loops; isolated nodes get a unit diagonal), each with a seeded random sign, and
    is rescaled to the mean absolute value of the local block. A graph with
    ``n - 1 < p`` nontrivial eigenvectors zero-pads the remaining columns (the
    rescaling uses the computed columns). Dense ``eigh`` is exact for the small
    instance graphs; the official code switches to ``eigsh`` only above 100 nodes.
    """
    x = x.float()
    n = x.size(0)
    adj = torch.zeros(n, n)
    if edge_index.numel():
        adj[edge_index[0], edge_index[1]] = 1.0
    adj = torch.maximum(adj, adj.t())
    adj.fill_diagonal_(0.0)
    hop = adj + torch.eye(n)
    hop = hop / hop.sum(1, keepdim=True)
    x1 = hop @ x
    x_loc = (x + x1 + hop @ x1) / 3.0

    x_g = torch.zeros(n, int(p))
    k = min(int(p), n - 1)
    if k > 0:
        deg = adj.sum(1)
        dinv = torch.where(deg > 0, deg.clamp_min(1e-12).rsqrt(), torch.zeros_like(deg))
        lap = torch.eye(n, dtype=torch.float64) - (dinv[:, None] * adj * dinv[None, :]).double()
        _, vecs = torch.linalg.eigh(lap)
        block = vecs[:, 1:k + 1].float()
        generator = torch.Generator().manual_seed(int(sign_seed))
        signs = torch.randint(0, 2, (k,), generator=generator).float() * 2.0 - 1.0
        block = block * signs
        block = block * (x_loc.abs().mean() / (block.abs().mean() + 1e-8))
        x_g[:, :k] = block
    return torch.cat([x_loc, x_g], dim=1)


@dataclass
class GateInputs:
    """Gate inputs of a set of instances.

    ``query [B, D]``: one gate-input row per instance. ``pop [B, n_max, D]``
    (zero-padded) and ``pop_size [B]``: each instance's own SGA population; when
    ``None``, the query rows of a forward pass are the population (graph level).
    """

    query: torch.Tensor
    pop: Optional[torch.Tensor] = None
    pop_size: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return int(self.query.size(0))

    def subset(self, idx: torch.Tensor) -> "GateInputs":
        if self.pop is None:
            return GateInputs(self.query[idx])
        size = self.pop_size[idx]
        width = int(size.max()) if size.numel() else 0
        return GateInputs(self.query[idx], self.pop[idx][:, :width], size)

    def to(self, device) -> "GateInputs":
        move = (lambda t: None if t is None else t.to(device))
        return GateInputs(move(self.query), move(self.pop), move(self.pop_size))


def build_gate_inputs(graphs: Sequence, positions: torch.Tensor, level: str, p: int) -> GateInputs:
    """Gate inputs of instance *graphs* (``positions`` seed the eigenvector signs)."""
    if level == "graph":
        return GateInputs(torch.stack([g.x.float().mean(0) for g in graphs]))
    rows, pops = [], []
    for graph, pos in zip(graphs, positions.tolist()):
        feats = multihop_lap_features(graph.x, graph.edge_index, p, sign_seed=int(pos))
        if level == "node":
            rows.append(feats[int(graph.target_node_index.view(-1)[0])])
        else:
            src, dst = graph.edge_label_index.view(2, -1)[:, 0].tolist()
            rows.append((feats[src] + feats[dst]) / 2.0)
        pops.append(feats)
    size = torch.tensor([f.size(0) for f in pops], dtype=torch.long)
    pop = torch.zeros(len(pops), int(size.max()), pops[0].size(1))
    for i, feats in enumerate(pops):
        pop[i, : feats.size(0)] = feats
    return GateInputs(torch.stack(rows), pop, size)


__all__ = ["GateInputs", "build_gate_inputs", "multihop_lap_features"]
