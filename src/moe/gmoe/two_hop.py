"""Multi-hop edge_index construction for GMoE experts.

GMoE assigns different experts different receptive fields: the first
``num_experts_1hop`` experts message-pass over the original 1-hop
``edge_index``; the rest use a ``hop``-hop ``edge_index`` so they
aggregate from a wider neighbourhood. The official implementation
pre-builds a separate 2-hop molecular graph (with two-hop bond
features); since edge features are out of scope in this repo, the
multi-hop expert is realised purely by
expanding connectivity — the union of neighbourhoods reachable within
``hop`` steps of the batched graph.

The batched graph is block-diagonal, so powers of the adjacency stay
within each subgraph and never leak across graphs in a batch.
"""

from __future__ import annotations

import torch
from torch_geometric.utils import coalesce, remove_self_loops


def _sparse_matmul(a: torch.Tensor, b: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return the connectivity of ``A @ B`` as an ``edge_index``.

    Prefers ``torch_sparse`` (device-resident); falls back to a scipy
    CSR product on CPU when ``torch_sparse`` is unavailable.
    """
    try:
        from torch_sparse import SparseTensor  # optional PyG dependency

        adj_a = SparseTensor(row=a[0], col=a[1], sparse_sizes=(num_nodes, num_nodes))
        adj_b = SparseTensor(row=b[0], col=b[1], sparse_sizes=(num_nodes, num_nodes))
        row, col, _ = (adj_a @ adj_b).coo()
        return torch.stack([row, col], dim=0)
    except Exception:
        from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix

        mat_a = to_scipy_sparse_matrix(a.cpu(), num_nodes=num_nodes).tocsr()
        mat_b = to_scipy_sparse_matrix(b.cpu(), num_nodes=num_nodes).tocsr()
        ei, _ = from_scipy_sparse_matrix(mat_a @ mat_b)
        return ei.to(a.device)


def compute_multi_hop_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
    hop: int = 2,
) -> torch.Tensor:
    """Return the ``hop``-hop ``edge_index`` (self-loops removed, coalesced).

    For ``hop <= 1`` the input is returned unchanged (after coalescing).
    For larger hops the 1-hop adjacency is multiplied in repeatedly and
    the reachable edges accumulated, so an edge exists between any two
    nodes reachable within ``hop`` steps.
    """
    one_hop = coalesce(edge_index, num_nodes=num_nodes)
    if hop <= 1:
        return one_hop

    accum = one_hop
    frontier = one_hop
    for _ in range(hop - 1):
        frontier = _sparse_matmul(frontier, one_hop, num_nodes)
        accum = coalesce(torch.cat([accum, frontier], dim=1), num_nodes=num_nodes)

    accum, _ = remove_self_loops(accum)
    return coalesce(accum, num_nodes=num_nodes)


__all__ = ["compute_multi_hop_edge_index"]
