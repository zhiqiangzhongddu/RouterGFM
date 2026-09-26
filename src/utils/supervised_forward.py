"""Shared supervised forward-pass helper.

Pretrain (``src/pretrain/methods/supervised.py``) and Train
(``src/train/trainer.py``) both need to route encoder outputs through
a per-task-level representation selector before a linear classifier:

* ``node``  -> mask the node representations with ``{train,val,test}_mask``
* ``edge``  -> gather endpoint reps via ``data.edge_label_index`` and take
  the elementwise product (pretrain-gnns dot-product family)
* ``graph`` -> pool node reps via the configured pooling mode (or reuse
  the encoder-supplied ``graph_repr`` if already available)

``select_supervised_logits_and_labels`` performs that routing once. The
finetune workflow has its own more elaborate objective
(``src/finetune/task_heads.TaskAwareObjective``) with prompt-method
alignment logic and is left untouched.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .dataset_helpers import normalize_node_mask
from .pool import resolve_graph_repr


def select_supervised_logits_and_labels(
    *,
    node_repr: torch.Tensor,
    graph_repr: torch.Tensor | None,
    data,
    classifier: nn.Module,
    task_level: str,
    pool_mode: str,
    mask_attr: str = "train_mask",
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Route encoder outputs through ``classifier`` and return ``(logits, labels)``.

    Args:
        node_repr: ``[num_nodes, dim]`` node representations from the encoder.
        graph_repr: Optional ``[num_graphs, dim]`` pooled graph representation.
            When ``None`` and ``task_level == "graph"`` the helper pools
            ``node_repr`` with ``pool_mode``.
        data: The ``torch_geometric`` batch holding labels and masks.
        classifier: The trainable head to apply to the selected representations.
        task_level: ``"node"``, ``"edge"``, or ``"graph"``.
        pool_mode: Pooling mode for the graph-level branch (e.g. ``"mean"``).
        mask_attr: Attribute name on ``data`` holding the node mask (node-level
            path only). Defaults to ``"train_mask"``.
        device: Device to fall back on when ``data`` has no mask tensor. When
            ``None`` the device is inferred from ``node_repr``.
    """
    level = str(task_level).lower()
    if device is None:
        device = node_repr.device

    if level == "node":
        logits = classifier(node_repr)
        mask = normalize_node_mask(data, mask_attr, device)
        logits_used = logits[mask]
        labels = data.y[mask]
        return logits_used, labels

    if level == "edge":
        edge_label_index = getattr(data, "edge_label_index", None)
        if edge_label_index is None:
            raise ValueError(
                "Edge-level supervised forward requires data.edge_label_index."
            )
        edge_label = getattr(data, "edge_label", None)
        if edge_label is None:
            raise ValueError(
                "Edge-level supervised forward requires data.edge_label; "
                "fabricating edge labels from data.y is not supported."
            )
        src, dst = edge_label_index
        edge_repr = node_repr[src] * node_repr[dst]
        logits_used = classifier(edge_repr)
        return logits_used, edge_label

    # graph (or any other promoted level)
    pooled = resolve_graph_repr(
        node_repr=node_repr,
        graph_repr=graph_repr,
        data=data,
        mode=pool_mode,
    )
    logits_used = classifier(pooled)
    return logits_used, data.y


__all__ = ["select_supervised_logits_and_labels"]
