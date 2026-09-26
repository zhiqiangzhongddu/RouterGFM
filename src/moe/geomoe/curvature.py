"""Ollivier-Ricci curvature (ORC) for GeoMoE's training losses (Cao et al., 2026, Def. 3.1-3.2, A.1-A.2).

ORC is computed per instance graph on its undirected simple version (direction
ignored, duplicates and self-loops dropped). Lazy random-walk measure
``mu_v(v) = p``, ``mu_v(u) = (1 - p) / deg(v)`` on neighbours; edge curvature
``kappa(u, v) = 1 - W1(mu_u, mu_v)`` (``d(u, v) = 1``) with hop-distance ground
cost; node curvature is the mean over incident edges and 0 (Euclidean) for
isolated nodes. W1 is exact (the paper's LP route, App. D.1): the transport
problems of all edges of a graph are solved as one block-diagonal LP (the
blocks are independent, so the joint optimum is optimal per block), each block
reduced to the excess/deficit of ``mu_u - mu_v`` (exact for a metric cost).
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional, Sequence

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from src.utils.checkpoint import save_torch_atomic

# Expert / region order everywhere in GeoMoE: Euclidean, hyperbolic, spherical.
EUCLIDEAN, HYPERBOLIC, SPHERICAL = 0, 1, 2


def _undirected_pairs(edge_index, num_nodes: int) -> np.ndarray:
    """Unique undirected pairs ``(u < v)`` as ``[2, E_u]``, lexicographically sorted."""
    ei = np.asarray(torch.as_tensor(edge_index, dtype=torch.long).cpu().numpy()).reshape(2, -1)
    lo, hi = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
    keep = lo != hi
    keys = np.unique(lo[keep] * int(num_nodes) + hi[keep])
    return np.stack([keys // int(num_nodes), keys % int(num_nodes)]).astype(np.int64)


def _pair_orc(pairs: np.ndarray, num_nodes: int, idleness: float) -> np.ndarray:
    """Exact ORC of each undirected pair (all pairs are edges of the graph)."""
    num_pairs = pairs.shape[1]
    if num_pairs == 0:
        return np.zeros(0)
    n = int(num_nodes)
    sym = np.concatenate([pairs, pairs[::-1]], axis=1)
    adj = coo_matrix((np.ones(sym.shape[1]), (sym[0], sym[1])), shape=(n, n)).tocsr()
    deg = np.diff(adj.indptr)
    dist = shortest_path(adj, unweighted=True, directed=False)
    p = float(idleness)

    def _measure(node, support):
        mass = np.zeros(support.size)
        mass[np.searchsorted(support, adj.indices[adj.indptr[node]:adj.indptr[node + 1]])] = (1.0 - p) / deg[node]
        mass[np.searchsorted(support, node)] = p
        return mass

    costs, rows, cols, b_eq, block = [], [], [], [], []
    num_vars = num_cons = 0
    for e, (u, v) in enumerate(pairs.T):
        support = np.union1d(adj.indices[adj.indptr[u]:adj.indptr[u + 1]], adj.indices[adj.indptr[v]:adj.indptr[v + 1]])
        support = np.union1d(support, [u, v])
        # W1 under a metric cost only depends on mu_u - mu_v: shared mass stays put.
        diff = _measure(u, support) - _measure(v, support)
        src, dst = support[diff > 0], support[diff < 0]
        a, b = src.size, dst.size
        if a == 0 or b == 0:
            continue
        var = num_vars + np.arange(a * b)  # x[i, j] at var[i * b + j]
        costs.append(dist[np.ix_(src, dst)].ravel())
        rows.append(num_cons + np.repeat(np.arange(a), b))  # sum_j x[i, j] = excess at src[i]
        rows.append(num_cons + a + np.tile(np.arange(b), a))  # sum_i x[i, j] = deficit at dst[j]
        cols.extend([var, var])
        b_eq.extend([diff[diff > 0], -diff[diff < 0]])
        block.append(np.full(a * b, e))
        num_vars += a * b
        num_cons += a + b
    if num_vars == 0:
        return np.ones(num_pairs)

    cost = np.concatenate(costs)
    row, col = np.concatenate(rows), np.concatenate(cols)
    a_eq = coo_matrix((np.ones(row.size), (row, col)), shape=(num_cons, num_vars)).tocsr()
    res = linprog(
        cost, A_eq=a_eq, b_eq=np.concatenate(b_eq), bounds=(0, None), method="highs",
        options={"presolve": False},  # the block transport LPs gain nothing from presolve; ~1.4x faster
    )
    if res.status != 0:
        raise RuntimeError(f"ORC transport LP failed: {res.message}")
    w1 = np.bincount(np.concatenate(block), weights=cost * res.x, minlength=num_pairs)
    return 1.0 - w1


def edge_orc(edge_index, num_nodes: int, idleness: float = 0.5) -> torch.Tensor:
    """ORC per column of ``edge_index`` (both directions of an edge agree); self-loop columns are NaN."""
    n = int(num_nodes)
    pairs = _undirected_pairs(edge_index, n)
    kappa = _pair_orc(pairs, n, idleness)
    ei = torch.as_tensor(edge_index, dtype=torch.long).cpu().numpy().reshape(2, -1)
    lo, hi = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
    out = np.full(ei.shape[1], np.nan)
    edge = lo != hi
    if edge.any():
        out[edge] = kappa[np.searchsorted(pairs[0] * n + pairs[1], lo[edge] * n + hi[edge])]
    return torch.as_tensor(out, dtype=torch.float32)


def node_orc(edge_index, num_nodes: int, idleness: float = 0.5) -> torch.Tensor:
    """Mean ORC of each node's incident edges; isolated nodes get 0.0 (Euclidean). Shape ``[num_nodes]``."""
    n = int(num_nodes)
    pairs = _undirected_pairs(edge_index, n)
    kappa = _pair_orc(pairs, n, idleness)
    ends = pairs.reshape(-1)
    total = np.bincount(ends, weights=np.concatenate([kappa, kappa]), minlength=n)
    count = np.bincount(ends, minlength=n)
    return torch.as_tensor(np.divide(total, np.maximum(count, 1)), dtype=torch.float32)


def orc_region(kappa: torch.Tensor, theta: float) -> torch.Tensor:
    """Geometric region per node: 0 = E (``|k| <= theta``), 1 = H (``k < -theta``), 2 = S (``k > theta``)."""
    region = torch.full_like(kappa, EUCLIDEAN, dtype=torch.long)
    region[kappa < -theta] = HYPERBOLIC
    region[kappa > theta] = SPHERICAL
    return region


def orc_target_weights(kappa: torch.Tensor, theta: float, eta: float) -> torch.Tensor:
    """Eq. 7 targets ``(w*_E, w*_H, w*_S)`` normalised to sum to one; shape ``[N, 3]``."""
    kappa = kappa.float()
    raw = torch.stack(
        [
            torch.sigmoid((theta - kappa.abs()) / eta),
            torch.sigmoid((-kappa - theta) / eta),
            torch.sigmoid((kappa - theta) / eta),
        ],
        dim=-1,
    )
    return raw / raw.sum(dim=-1, keepdim=True)


def _structure_digest(graphs: Sequence, idleness: float) -> str:
    """Content hash of the graph structures (ORC depends on nothing else)."""
    digest = hashlib.sha256(f"orc-v1|p={float(idleness)!r}".encode())
    for graph in graphs:
        edge_index = getattr(graph, "edge_index", None)
        ei = np.zeros((2, 0), dtype=np.int64) if edge_index is None else edge_index.cpu().numpy().astype(np.int64)
        digest.update(np.asarray([int(graph.num_nodes), ei.shape[1]], dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(ei).tobytes())
    return digest.hexdigest()[:32]


def attach_node_orc(graphs: Sequence, idleness: float, cache_dir: Optional[str] = None) -> None:
    """Set ``graph.node_orc`` (``[num_nodes]`` float32) on every graph, in place.

    With ``cache_dir`` the values are cached in a file named by the content hash
    of the structures, so a cache hit can never be stale.
    """
    path = os.path.join(cache_dir, f"{_structure_digest(graphs, idleness)}.pt") if cache_dir else None
    values = None
    if path is not None and os.path.isfile(path):
        cached = torch.load(path, map_location="cpu")
        if len(cached) == len(graphs):
            values = cached
    if values is None:
        values = []
        for graph in graphs:
            edge_index = getattr(graph, "edge_index", None)
            if edge_index is None:
                edge_index = torch.zeros(2, 0, dtype=torch.long)
            values.append(node_orc(edge_index, int(graph.num_nodes), idleness))
        if path is not None:
            save_torch_atomic(path, values)
    for graph, value in zip(graphs, values):
        graph.node_orc = value


__all__ = [
    "EUCLIDEAN",
    "HYPERBOLIC",
    "SPHERICAL",
    "attach_node_orc",
    "edge_orc",
    "node_orc",
    "orc_region",
    "orc_target_weights",
]
