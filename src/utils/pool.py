"""Shared graph-pooling helpers.

Single source of truth for the pool map and for per-graph batch
resolution. Used by the model encoder and by pretrain / train /
finetune methods. Keeping this module dependency-light (stdlib +
torch + torch_geometric only) lets every workflow import it without
circular dependencies.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


#: Canonical pool-function map. Callers that accept user-facing strings
#: should first pass the value through ``normalize_pool_mode`` so that
#: "sum" and "add" resolve to the same entry.
POOLERS = {
    "mean": global_mean_pool,
    "add": global_add_pool,
    "max": global_max_pool,
}


def normalize_pool_mode(mode: str) -> str:
    """Map user-facing pool names to canonical keys.

    ``"sum"`` is accepted as a synonym for ``"add"`` because several
    reference papers use the former spelling in their config files.
    """
    key = str(mode).lower()
    return "add" if key == "sum" else key


def get_pool_fn(mode: str):
    """Return the pooling function for ``mode``. Falls back to mean pooling."""
    return POOLERS.get(normalize_pool_mode(mode), global_mean_pool)


def pool_nodes(
    x: torch.Tensor,
    batch: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """Pool node embeddings into graph embeddings.

    Args:
        x: Node embeddings ``[num_nodes, dim]``.
        batch: Batch assignment ``[num_nodes]``.
        mode: ``"mean"``, ``"add"``/``"sum"``, or ``"max"``.
    """
    return get_pool_fn(mode)(x, batch)


def pool_target_nodes(x: torch.Tensor, data) -> torch.Tensor:
    """Graph representation = the embedding of each graph's target node.

    Requires ``data.target_node_index`` (written by the induced-subgraph
    builder; its "index" suffix makes PyG batching offset it per graph, so
    it always indexes into the batched node dimension directly).
    """
    target = getattr(data, "target_node_index", None)
    if target is None:
        raise ValueError(
            "graph_pooling='target' requires data.target_node_index. Induced "
            "subgraph caches generated before this attribute existed must be "
            "regenerated (delete the dataset's cache under data/induced_subgraphs)."
        )
    return x[torch.as_tensor(target).view(-1)]


def get_batch_vector(data) -> torch.Tensor:
    """Return the batch assignment tensor for ``data``.

    Falls back to an all-zero vector for single-graph inputs so callers
    can always use graph-level reductions without branching. The device
    is inferred from ``data.x`` when present, otherwise from
    ``data.edge_index`` (featureless graphs), and finally from CPU as a
    last resort.
    """
    batch = getattr(data, "batch", None)
    if batch is not None:
        return batch
    x = getattr(data, "x", None)
    if x is not None:
        device = x.device
    else:
        edge_index = getattr(data, "edge_index", None)
        device = edge_index.device if edge_index is not None else torch.device("cpu")
    return torch.zeros(int(data.num_nodes), dtype=torch.long, device=device)


def resolve_graph_repr(
    node_repr: torch.Tensor,
    graph_repr: Optional[torch.Tensor],
    data,
    mode: str,
) -> torch.Tensor:
    """Return a graph-level representation, reusing ``graph_repr`` when non-None.

    Policy (by design, not a detectable invariant): callers must ensure
    the ``graph_repr`` they pass in -- if any -- was produced with the
    **same** pooling mode as ``mode``. This helper cannot verify that;
    it always returns ``graph_repr`` as-is when provided, and pools
    ``node_repr`` with ``mode`` only when ``graph_repr`` is ``None``.

    If the caller's pooling mode differs from the encoder's (e.g.
    InfoGraph uses its own ``graph_pooling`` that may not match
    ``cfg.model.graph_pooling``), bypass this helper and call
    ``pool_nodes`` directly.
    """
    if graph_repr is not None:
        return graph_repr
    batch = get_batch_vector(data)
    return pool_nodes(node_repr, batch, mode=mode)


__all__ = [
    "POOLERS",
    "normalize_pool_mode",
    "get_pool_fn",
    "pool_nodes",
    "pool_target_nodes",
    "get_batch_vector",
    "resolve_graph_repr",
]
