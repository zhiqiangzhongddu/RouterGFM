"""Structural pair heuristics consumed by the Link-MoE gate (HeaRT ``get_heuristic.py``).

All heuristics run on one binary, symmetric adjacency without self-loops (the
eval context graph). Every heuristic is symmetric in the pair, so the pair
orientation (which differs between positives and negatives in the repo's
splits) cannot leak the label. :func:`pair_heuristics` refuses pairs that are
edges of the graph: a scored pair's own link, and hence any held-out
positive among the scored pairs, can never contribute to its features.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import shortest_path

HEURISTIC_NAMES = ("aa", "cn", "katz", "ppr", "ra", "inv_sp", "deg_min", "deg_max")
_PAIR_CHUNK = 100_000


def symmetric_csr(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    """Binary symmetric CSR adjacency without self-loops."""
    ei = edge_index.detach().cpu().numpy().astype(np.int64)
    rows = np.concatenate([ei[0], ei[1]])
    cols = np.concatenate([ei[1], ei[0]])
    keep = rows != cols
    A = sp.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.float64), (rows[keep], cols[keep])),
        shape=(num_nodes, num_nodes),
    )
    A.sum_duplicates()
    A.data[:] = 1.0
    return A


def _as_numpy_pairs(pairs) -> tuple[np.ndarray, np.ndarray]:
    arr = torch.as_tensor(pairs).detach().cpu().numpy().astype(np.int64)
    return arr[0], arr[1]


def _degrees(A: sp.csr_matrix) -> np.ndarray:
    return np.asarray(A.sum(axis=1)).ravel()


def _row_dot(A: sp.csr_matrix, B: sp.csr_matrix, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """``sum_k A[src, k] * B[dst, k]`` per pair, chunked."""
    out = np.zeros(src.shape[0], dtype=np.float64)
    for start in range(0, src.shape[0], _PAIR_CHUNK):
        s, d = src[start:start + _PAIR_CHUNK], dst[start:start + _PAIR_CHUNK]
        out[start:start + s.shape[0]] = np.asarray(A[s].multiply(B[d]).sum(axis=1)).ravel()
    return out


def _scaled_columns(A: sp.csr_matrix, multiplier: np.ndarray) -> sp.csr_matrix:
    multiplier = multiplier.copy()
    multiplier[~np.isfinite(multiplier)] = 0.0
    return sp.csr_matrix(A.multiply(multiplier[None, :]))


def edge_mask(A: sp.csr_matrix, pairs) -> np.ndarray:
    """Boolean mask of pairs that are edges of ``A``."""
    src, dst = _as_numpy_pairs(pairs)
    if src.size == 0:
        return np.zeros(0, dtype=bool)
    return np.asarray(A[src, dst]).ravel() > 0


def common_neighbors(A: sp.csr_matrix, pairs) -> np.ndarray:
    src, dst = _as_numpy_pairs(pairs)
    return _row_dot(A, A, src, dst)


def adamic_adar(A: sp.csr_matrix, pairs) -> np.ndarray:
    """``sum_{k in N(i) & N(j)} 1 / log d_k`` (terms with ``d_k <= 1`` set to 0)."""
    src, dst = _as_numpy_pairs(pairs)
    deg = _degrees(A)
    with np.errstate(divide="ignore"):
        mult = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 1.0)), 0.0)
    return _row_dot(A, _scaled_columns(A, mult), src, dst)


def resource_allocation(A: sp.csr_matrix, pairs) -> np.ndarray:
    src, dst = _as_numpy_pairs(pairs)
    deg = _degrees(A)
    with np.errstate(divide="ignore"):
        mult = 1.0 / deg
    return _row_dot(A, _scaled_columns(A, mult), src, dst)


def katz3(A: sp.csr_matrix, pairs, beta: float = 0.005) -> np.ndarray:
    """``sum_{l=1..3} beta^l * #simple paths of length l`` (HeaRT ``katz_apro``, exact).

    Length-3 walks ``i-k-l-j`` that are not simple revisit an endpoint, which
    requires the edge ``(i, j)``; there are ``d_i + d_j - 1`` of them.
    """
    src, dst = _as_numpy_pairs(pairs)
    deg = _degrees(A)
    A2 = sp.csr_matrix(A @ A)
    a1 = np.asarray(A[src, dst]).ravel() if src.size else np.zeros(0)
    a2 = _row_dot(A, A, src, dst)
    a3 = _row_dot(A, A2, src, dst) - a1 * (deg[src] + deg[dst] - 1.0)
    return beta * a1 + beta ** 2 * a2 + beta ** 3 * a3


def inverse_shortest_path(A: sp.csr_matrix, pairs, batch_size: int = 1024) -> np.ndarray:
    """``1 / dist(i, j)`` on ``A``, 0 when unreachable (BFS from the smaller endpoint id)."""
    src, dst = _as_numpy_pairs(pairs)
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    out = np.zeros(src.shape[0], dtype=np.float64)
    sources = np.unique(lo)
    for start in range(0, sources.shape[0], batch_size):
        batch = sources[start:start + batch_size]
        dist = shortest_path(A, directed=False, unweighted=True, indices=batch)
        dist = np.atleast_2d(dist)
        sel = np.nonzero(np.isin(lo, batch))[0]
        d = dist[np.searchsorted(batch, lo[sel]), hi[sel]]
        with np.errstate(divide="ignore"):
            out[sel] = np.where(np.isfinite(d) & (d > 0), 1.0 / d, 0.0)
    return out


def ppr_symmetric(
    A: sp.csr_matrix,
    pairs,
    damping: float = 0.85,
    tol: float = 1e-7,
    max_iter: int = 100,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """``(pi_i(j) + pi_j(i)) / 2`` with fast_pagerank's ``pagerank_power`` per source.

    Per source ``s``: ``x0 = n e_s``; ``x <- p A^T D^-1 x + e_s (sum_k c_k x_k)``
    with ``c_k = 1-p`` (non-dangling) or ``1`` (dangling), until the column's
    L2 change is ``<= tol`` or ``max_iter`` updates; then ``x / sum(x)``.
    Sources are batched as dense ``[N, B]`` blocks (float64 so the absolute
    tolerance keeps its meaning).
    """
    src, dst = _as_numpy_pairs(pairs)
    n = A.shape[0]
    out = np.zeros(src.shape[0], dtype=np.float64)
    if src.size == 0:
        return out
    device = torch.device(device)
    deg = _degrees(A)
    inv_deg = np.where(deg > 0, 1.0 / np.maximum(deg, 1.0), 0.0)
    W = sp.coo_matrix(A.T.multiply(inv_deg[None, :]) * damping)  # p A^T D^-1
    W_t = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack([W.row, W.col]).astype(np.int64)),
        torch.from_numpy(W.data.astype(np.float64)),
        size=(n, n),
    ).coalesce().to(device)
    c = torch.from_numpy(np.where(deg > 0, 1.0 - damping, 1.0)).to(device)

    nodes = np.unique(np.concatenate([src, dst]))
    pos = {int(v): i for i, v in enumerate(nodes)}
    col_of_src = np.array([pos[int(v)] for v in src])
    col_of_dst = np.array([pos[int(v)] for v in dst])
    for start in range(0, nodes.shape[0], batch_size):
        batch = torch.from_numpy(nodes[start:start + batch_size]).to(device)
        b = int(batch.numel())
        E = torch.zeros(n, b, dtype=torch.float64, device=device)
        E[batch, torch.arange(b, device=device)] = 1.0
        x = E * n
        active = torch.ones(b, dtype=torch.bool, device=device)
        for _ in range(int(max_iter)):
            x_new = torch.sparse.mm(W_t, x) + E * (c @ x)[None, :]
            diff = torch.linalg.vector_norm(x_new - x, dim=0)
            x = torch.where(active[None, :], x_new, x)
            active &= diff > tol
            if not bool(active.any()):
                break
        x = (x / x.sum(dim=0, keepdim=True)).cpu().numpy()
        from_i = (col_of_src >= start) & (col_of_src < start + b)  # pi_i(j): source i in batch
        from_j = (col_of_dst >= start) & (col_of_dst < start + b)  # pi_j(i): source j in batch
        out[from_i] += 0.5 * x[dst[from_i], col_of_src[from_i] - start]
        out[from_j] += 0.5 * x[src[from_j], col_of_dst[from_j] - start]
    return out


def pair_heuristics(
    A: sp.csr_matrix,
    pairs: torch.Tensor,
    *,
    katz_beta: float = 0.005,
    ppr_damping: float = 0.85,
    ppr_tol: float = 1e-7,
    ppr_max_iter: int = 100,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """``[P, 8]`` float32 raw heuristics in :data:`HEURISTIC_NAMES` order.

    Raises if any pair is an edge of ``A``: features of a candidate pair must
    be computed on a graph without that link (and, passing all held-out pairs
    at once, without any held-out positive).
    """
    leaked = edge_mask(A, pairs)
    if leaked.any():
        raise ValueError(
            f"[LinkMoE] {int(leaked.sum())} scored pair(s) are edges of the heuristic graph; "
            "candidate or held-out positive links must be absent."
        )
    src, dst = _as_numpy_pairs(pairs)
    deg = _degrees(A)
    cols = [
        adamic_adar(A, pairs),
        common_neighbors(A, pairs),
        katz3(A, pairs, beta=katz_beta),
        ppr_symmetric(A, pairs, damping=ppr_damping, tol=ppr_tol, max_iter=ppr_max_iter, device=device),
        resource_allocation(A, pairs),
        inverse_shortest_path(A, pairs),
        np.minimum(deg[src], deg[dst]),
        np.maximum(deg[src], deg[dst]),
    ]
    return torch.from_numpy(np.stack(cols, axis=1)).float()


__all__ = [
    "HEURISTIC_NAMES",
    "adamic_adar",
    "common_neighbors",
    "edge_mask",
    "inverse_shortest_path",
    "katz3",
    "pair_heuristics",
    "ppr_symmetric",
    "resource_allocation",
    "symmetric_csr",
]
