"""MetaGL meta-graph features m in R^318 (structure only).

Port of GLEMOS ``regular_meta_graph_features`` / ``statdist`` (the extractor
behind MetaGL's released M): three graph-level scalars ``[density(A),
density(A A^T), degree assortativity]`` followed by 63 distribution statistics
of each of degree, core number, PageRank (alpha 0.7), wedges, and triangles.
``density(A A^T)`` uses the matrix product (MetaGL paper; GLEMOS computes an
element-wise product). Every non-finite entry is set to 0.

An application's structure graph (labels are never read) is its input graph,
as in MetaGL: the full dataset graph for node applications (cached once per
dataset) and the message-passing graph of the split for link applications
(held-out positives excluded; cached per data key). Graph-level applications
have no single input graph: they use the disjoint union of at most
``graph_sample_max`` instance graphs sampled from the support and query pools
(cached per data key).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Sequence, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from scipy import stats

from src.utils.checkpoint import save_torch_atomic

from ....applications import instance_set_key
from ....common import AppSpec, RouterPaths

N_STAT_FEATURES = 63
N_META_GRAPH_FEATURES = 3 + 5 * N_STAT_FEATURES  # 318


def gini(x: np.ndarray) -> float:
    """Gini coefficient (GLEMOS: shift to non-negative, add 1e-7, sorted formula)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.min() < 0:
        x = x - x.min()
    x = np.sort(x + 1e-7)
    n = x.size
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * x) / (n * np.sum(x)))


def _outlier_block(x: np.ndarray, lb: float, ub: float) -> Tuple[int, int, int]:
    """(total, below, above) counts; ``> ub`` is checked before ``< lb`` as in GLEMOS."""
    above = x > ub
    below = ~above & (x < lb)
    return int(np.sum((x < lb) | (x > ub))), int(below.sum()), int(above.sum())


def structural_stat_vector(x: np.ndarray) -> np.ndarray:
    """The 63 statistics of one per-node distribution (GLEMOS ``meta_features_per_structural_property``)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        mean, median = np.nanmean(x), np.nanmedian(x)
        std = np.nanstd(x)
        ent = stats.entropy(x)
        xs = np.sort(x)
        med_idx, q_idx = n // 2, n // 4
        q1, q3 = xs[med_idx - q_idx], xs[med_idx + q_idx]
        iqr = q3 - q1
        lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        lb3, ub3 = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        out, out_lb, out_ub = _outlier_block(x, lb, ub)
        out3, out3_lb, out3_ub = _outlier_block(x, lb3, ub3)
        std_blocks = []
        for fac in (1.0, 2.0, 3.0):
            c, c_lb, c_ub = _outlier_block(x, mean - fac * std, mean + fac * std)
            std_blocks += [c, c_lb, c_ub, c / n, c_lb / n, c_ub / n]
        values, counts = np.unique(x, return_counts=True)
        mode, mode_count = values[np.argmax(counts)], counts.max()  # smallest most frequent value
        feats = [
            mean, median, np.nanmin(x), np.nanmax(x), np.nanvar(x), std,
            ent, ent / math.log2(n) if n > 1 else np.nan, gini(x), stats.variation(x),
            np.nanmedian(np.abs(x - median)), np.nanmean(np.abs(x - mean)),
            (q3 - q1) / (q3 + q1), mean * mean / (std * std), std * std / (mean * mean), std * std / mean,
            stats.skew(x),
            stats.kurtosis(x, fisher=True), stats.kurtosis(x, fisher=False),
            stats.kurtosis(x, fisher=True, bias=False), stats.kurtosis(x, fisher=False, bias=False),
            stats.gmean(x), stats.hmean(x + 1e-10),
            out / n, out_lb / n, out_ub / n, out, out_lb, out_ub,
            out3 / n, out3_lb / n, out3_ub / n, out3, out3_lb, out3_ub,
            lb3, ub3,
            iqr, q1, q3, lb, ub,
            *std_blocks,
            mode, mode_count, mode_count / n,
        ]
        vec = np.array(feats, dtype=np.float64)
    vec[~np.isfinite(vec)] = 0.0
    return vec


def aat_nnz(A: sp.csr_matrix, chunk: int = 4096) -> int:
    """``nnz(A @ A.T)`` computed over row chunks (never materializes A^2)."""
    A = sp.csr_matrix(A)
    At = A.T.tocsc()
    return int(sum((A[i : i + chunk] @ At).count_nonzero() for i in range(0, A.shape[0], chunk)))


def meta_graph_features(edge_index, num_nodes: int) -> np.ndarray:
    """318-d meta-graph feature vector of the undirected simple graph (self-loops kept once in A)."""
    ei = np.asarray(torch.as_tensor(edge_index).cpu().numpy(), dtype=np.int64).reshape(2, -1)
    n = int(num_nodes)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr")
    A = sp.csr_matrix((np.ones(A.nnz), A.indices, A.indptr), shape=A.shape)  # binary
    pairs = n * (n - 1.0)
    with np.errstate(all="ignore"):
        density_a = A.nnz / pairs
        density_aat = aat_nnz(A) / pairs
    deg = np.diff((A + A.T).tocsr().indptr).astype(np.float64)  # distinct neighbours; a self-loop counts once

    G.remove_edges_from(list(nx.selfloop_edges(G)))
    r = 0.0
    if deg.size and deg.max() < 1000:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            r = nx.degree_assortativity_coefficient(G)
    core = np.array(list(nx.core_number(G).values()), dtype=np.float64)
    try:
        pagerank = nx.pagerank(G, alpha=0.7)
    except nx.PowerIterationFailedConvergence:
        try:
            pagerank = nx.pagerank(G, alpha=0.7, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            print(f"[MetaGL] PageRank did not converge on a {n}-node graph; using uniform scores.")
            pagerank = {v: 1.0 / n for v in G}
    pagerank = np.array(list(pagerank.values()), dtype=np.float64)
    wedges = deg * (deg - 1.0) / 2.0
    triangles = np.array(list(nx.triangles(G).values()), dtype=np.float64)

    vec = np.concatenate(
        [np.array([density_a, density_aat, r], dtype=np.float64)]
        + [structural_stat_vector(v) for v in (deg, core, pagerank, wedges, triangles)]
    )
    vec[~np.isfinite(vec)] = 0.0
    return vec


def disjoint_union(graphs: Sequence) -> Tuple[torch.Tensor, int]:
    """``(edge_index, num_nodes)`` of the disjoint union of PyG graphs (structure only)."""
    parts, offset = [], 0
    for g in graphs:
        parts.append(g.edge_index.cpu().long() + offset)
        offset += int(g.num_nodes)
    edge_index = torch.cat(parts, dim=1) if parts else torch.zeros((2, 0), dtype=torch.long)
    return edge_index, offset


def application_meta_features(infra, app: AppSpec, cfg) -> np.ndarray:
    """Cached 318-d features of *app*'s structure graph (label-free; see the module docstring).

    Node / link: ``infra.base_graph``. Graph level: sampled union of ``infra.instance_graphs``.
    """
    paths = RouterPaths.from_cfg(cfg)
    if app.task_level in ("node", "edge"):
        name = "full" if app.task_level == "node" else "split"
        path = Path(paths.data_dir(instance_set_key(app))) / f"metagl_features__{name}.pt"
    else:
        m = cfg.moe.routergfm.baselines.metagl
        n_max, seed = int(m.graph_sample_max), int(m.graph_sample_seed)
        path = Path(paths.data_dir(app.data_key)) / f"metagl_features__sample{n_max}-rs{seed}.pt"
    if path.is_file():
        return torch.load(path, map_location="cpu").numpy()
    if app.task_level in ("node", "edge"):
        base = infra.base_graph(app)
        feats = meta_graph_features(base.edge_index, int(base.num_nodes))
    else:
        support, query = infra.instance_graphs(app, "support"), infra.instance_graphs(app, "query")
        total = len(support) + len(query)
        chosen = np.sort(np.random.default_rng(seed).choice(total, size=min(n_max, total), replace=False))
        graphs = [support[i] if i < len(support) else query[i - len(support)] for i in chosen.tolist()]
        feats = meta_graph_features(*disjoint_union(graphs))
    save_torch_atomic(str(path), torch.from_numpy(feats))
    return feats


__all__ = [
    "N_META_GRAPH_FEATURES",
    "N_STAT_FEATURES",
    "aat_nnz",
    "application_meta_features",
    "disjoint_union",
    "gini",
    "meta_graph_features",
    "structural_stat_vector",
]
