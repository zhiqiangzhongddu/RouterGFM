"""The multi-relational G-M kNN network of MetaGL (official ``create_knn_graph`` /
``create_train_test_graphs``) and its model-side extension for MetaGL+metadata.

``knn_pairs(X, Y, k)`` returns, for each row of X, its top-min(k, |Y|) rows of Y
by cosine similarity (itself included when X is Y). Edges point from the
querying node to its neighbour, as in the DGL code: a node aggregates from the
nodes that chose it. New graphs (targets) and new models receive edges from
known nodes and send ``P_g2m`` / ``P_m2g`` edges; new nodes are never linked
to each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

Relation = Tuple[str, str, str]
GRAPH, MODEL = "graph", "model"
NODE_TYPES = (GRAPH, MODEL)
M_G2G: Relation = (GRAPH, "M_g2g", GRAPH)
P_G2G: Relation = (GRAPH, "P_g2g", GRAPH)
P_M2M: Relation = (MODEL, "P_m2m", MODEL)
P_G2M: Relation = (GRAPH, "P_g2m", MODEL)
P_M2G: Relation = (MODEL, "P_m2g", GRAPH)
M_M2M: Relation = (MODEL, "M_m2m", MODEL)  # MetaGL+metadata: kNN on expert metadata
RELATIONS = (M_G2G, P_G2G, P_M2M, P_G2M, P_M2G)
METADATA_RELATIONS = RELATIONS + (M_M2M,)


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a / a.norm(dim=1, keepdim=True).clamp_min(eps)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(eps)
    return a @ b.t()


def knn_pairs(X: torch.Tensor, Y: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(query index, neighbour index)`` for each row of X and its top-min(k, |Y|) rows of Y."""
    k = min(int(k), Y.size(0))
    idx = torch.topk(cosine_sim(X, Y), k, dim=1).indices
    return torch.arange(X.size(0), device=X.device).repeat_interleave(k), idx.reshape(-1)


@dataclass
class GMNetwork:
    num_nodes: Dict[str, int]
    edges: Dict[Relation, torch.Tensor] = field(default_factory=dict)  # relation -> [2, E] (src, dst)

    def add_edges(self, rel: Relation, src: torch.Tensor, dst: torch.Tensor) -> None:
        new = torch.stack([src, dst])
        self.edges[rel] = torch.cat([self.edges[rel], new], dim=1) if rel in self.edges else new


def build_network(
    Mp: torch.Tensor, U: torch.Tensor, V: torch.Tensor, k: int, v_meta: Optional[torch.Tensor] = None
) -> GMNetwork:
    """Network over known graphs (meta-features ``Mp``, graph factors ``U``) and models (factors ``V``)."""
    net = GMNetwork({GRAPH: Mp.size(0), MODEL: V.size(0)})
    net.add_edges(M_G2G, *knn_pairs(Mp, Mp, k))
    net.add_edges(P_G2G, *knn_pairs(U, U, k))
    net.add_edges(P_M2M, *knn_pairs(V, V, k))
    net.add_edges(P_G2M, *knn_pairs(U, V, k))
    net.add_edges(P_M2G, *knn_pairs(V, U, k))
    if v_meta is not None:
        net.add_edges(M_M2M, *knn_pairs(v_meta, v_meta, k))
    return net


def add_graphs(
    net: GMNetwork,
    Mp_known: torch.Tensor,
    U_known: torch.Tensor,
    V: torch.Tensor,
    Mp_new: torch.Tensor,
    U_new: torch.Tensor,
    k: int,
) -> GMNetwork:
    """Copy of *net* with new graph nodes appended after the known ones (official test extension)."""
    out = GMNetwork(dict(net.num_nodes), dict(net.edges))
    off = out.num_nodes[GRAPH]
    out.num_nodes[GRAPH] += Mp_new.size(0)
    t, j = knn_pairs(Mp_new, Mp_known, k)
    out.add_edges(M_G2G, j, t + off)
    t, j = knn_pairs(U_new, U_known, k)
    out.add_edges(P_G2G, j, t + off)
    t, e = knn_pairs(U_new, V, k)
    out.add_edges(P_G2M, t + off, e)
    e, t = knn_pairs(V, U_new, k)
    out.add_edges(P_M2G, e, t + off)
    return out


def add_models(
    net: GMNetwork,
    V_known: torch.Tensor,
    V_new: torch.Tensor,
    U_all: torch.Tensor,
    k: int,
    v_known: torch.Tensor,
    v_new: torch.Tensor,
) -> GMNetwork:
    """Copy of *net* with new model nodes appended (MetaGL+metadata insertion; roles of the graph extension swapped).

    ``U_all`` holds the factors of every graph node currently in *net*.
    """
    out = GMNetwork(dict(net.num_nodes), dict(net.edges))
    off = out.num_nodes[MODEL]
    out.num_nodes[MODEL] += V_new.size(0)
    n, f = knn_pairs(v_new, v_known, k)
    out.add_edges(M_M2M, f, n + off)
    n, f = knn_pairs(V_new, V_known, k)
    out.add_edges(P_M2M, f, n + off)
    n, i = knn_pairs(V_new, U_all, k)
    out.add_edges(P_M2G, n + off, i)
    i, n = knn_pairs(U_all, V_new, k)
    out.add_edges(P_G2M, i, n + off)
    return out


__all__ = [
    "GMNetwork",
    "METADATA_RELATIONS",
    "NODE_TYPES",
    "RELATIONS",
    "add_graphs",
    "add_models",
    "build_network",
    "cosine_sim",
    "knn_pairs",
]
