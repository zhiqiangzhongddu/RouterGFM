"""Task-aware query readouts of frozen encoders (App. B.1)."""

from __future__ import annotations

from typing import Optional

import torch

from src.utils.pool import get_batch_vector, pool_nodes


def readout_dim(out_dim: int, task_level: str) -> int:
    """Width of :func:`graph_query_representation` for an encoder of width ``out_dim``."""
    return int(out_dim) * (4 if str(task_level).lower() == "edge" else 1)


def graph_query_representation(
    node_repr: torch.Tensor,
    graph_repr: Optional[torch.Tensor],
    data,
    *,
    task_level: str,
    pool_mode: str,
) -> torch.Tensor:
    """Return one graph-batch representation per supervised query.

    Edge queries use their target endpoints plus pooled enclosing-subgraph
    context. Whole-graph pooling alone erases which pair is being scored.
    """
    level = str(task_level).lower()
    if level == "node":
        target_node_index = getattr(data, "target_node_index", None)
        if target_node_index is None or target_node_index.numel() == 0:
            raise ValueError(
                "RouterGFM induced node prediction requires "
                "target_node_index on each query graph."
            )
        target_node_index = target_node_index.view(-1)
        target_repr = node_repr[target_node_index]
        num_graphs = int(get_batch_vector(data).max().item()) + 1
        if target_repr.size(0) != num_graphs:
            raise ValueError(
                "RouterGFM expects one target node per induced query graph; "
                f"found {target_repr.size(0)} targets for {num_graphs} graphs."
            )
        return target_repr

    pooled = graph_repr
    if pooled is None:
        pooled = pool_nodes(
            node_repr,
            get_batch_vector(data),
            mode=pool_mode,
        )
    if level != "edge":
        return pooled

    edge_label_index = getattr(data, "edge_label_index", None)
    if edge_label_index is None or edge_label_index.numel() == 0:
        raise ValueError(
            "RouterGFM edge prediction requires edge_label_index on each "
            "induced query graph."
        )
    src, dst = edge_label_index
    src_repr = node_repr[src]
    dst_repr = node_repr[dst]
    if src_repr.size(0) != pooled.size(0):
        raise ValueError(
            "RouterGFM expects one target edge per induced query graph; "
            f"found {src_repr.size(0)} edges for {pooled.size(0)} graphs."
        )
    return torch.cat(
        [
            src_repr + dst_repr,
            torch.abs(src_repr - dst_repr),
            src_repr * dst_repr,
            pooled,
        ],
        dim=-1,
    )


__all__ = ["graph_query_representation", "readout_dim"]
