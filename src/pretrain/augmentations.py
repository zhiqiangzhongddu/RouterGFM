"""Graph augmentations reusable by pretraining methods.

These are the classical GraphCL augmentations (``dropN`` / ``permE`` /
``maskN`` / ``subgraph``).  They were historically private to
``src/pretrain/methods/graphcl.py``; lifting them out keeps graphcl.py
focused on the contrastive objective and lets future contrastive /
augmentation-based pretrain methods reuse the same primitives instead
of re-implementing them.

The per-graph driver :func:`apply_per_graph_augment` preserves positive-
pair alignment by unbatching with ``data.to_data_list()`` before applying
the augmentation (GraphCL semantics).
"""

from __future__ import annotations

import random
from typing import Callable

import torch
from torch_geometric.data import Batch


#: Canonical augmentation names accepted by graphcl.aug1 / graphcl.aug2.
VALID_AUG_NAMES: tuple[str, ...] = ("dropN", "permE", "maskN", "subgraph", "random")


def _apply_edge_mask(data, edge_mask: torch.Tensor) -> None:
    """Apply an edge mask to edge attributes when available."""
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is None:
        return
    if edge_attr.size(0) == edge_mask.numel():
        data.edge_attr = edge_attr[edge_mask]


def edge_perturbation(data, p: float, add_random_edges: bool = False):
    """
    Perturb edges with ratio p.

    For plain edge-index graphs, `permE` can be either delete-only (TU-style)
    or delete-and-add (chem GraphCL style). When edge attributes exist,
    keep a delete-only variant to avoid fabricating invalid edge attributes.
    """
    aug = data.clone()
    edge_index = getattr(aug, "edge_index", None)
    if edge_index is None:
        return aug
    if p <= 0:
        return aug

    edge_num = int(edge_index.size(1))
    if edge_num == 0:
        return aug
    permute_num = int(edge_num * float(p))
    if permute_num <= 0:
        return aug

    edge_attr = getattr(aug, "edge_attr", None)
    # When edge_attr is present, use delete-only permE so edge_index and
    # edge_attr stay aligned. The official chem loader adds random edges
    # without updating edge_attr, which silently misaligns the two tensors;
    # we deliberately diverge from that behavior.
    if edge_attr is not None:
        keep_num = max(1, edge_num - permute_num)
        keep_idx = torch.randperm(edge_num, device=edge_index.device)[:keep_num]
        aug.edge_index = edge_index[:, keep_idx]
        if edge_attr.size(0) == edge_num:
            aug.edge_attr = edge_attr[keep_idx]
        return aug

    # permE for plain graphs.
    keep_num = max(1, edge_num - permute_num)
    keep_idx = torch.randperm(edge_num, device=edge_index.device)[:keep_num]
    kept_edge_index = edge_index[:, keep_idx]
    if not add_random_edges:
        aug.edge_index = kept_edge_index
        return aug

    num_nodes = int(getattr(aug, "num_nodes", 0) or 0)
    if num_nodes <= 0 and getattr(aug, "x", None) is not None:
        num_nodes = int(aug.x.size(0))
    if num_nodes <= 0 and edge_index.numel() > 0:
        num_nodes = int(edge_index.max().item()) + 1
    if num_nodes <= 0:
        aug.edge_index = kept_edge_index
        return aug

    added = torch.randint(
        low=0,
        high=num_nodes,
        size=(2, permute_num),
        device=edge_index.device,
        dtype=edge_index.dtype,
    )
    aug.edge_index = torch.cat([kept_edge_index, added], dim=1)
    return aug


def _subset_nodes(data, keep_mask: torch.Tensor):
    """Filter graph nodes and relabel edges accordingly."""
    aug = data.clone()
    num_nodes = keep_mask.size(0)
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    if keep_idx.numel() == 0:
        keep_idx = torch.zeros(1, dtype=torch.long, device=keep_mask.device)
        keep_mask = torch.zeros_like(keep_mask)
        keep_mask[0] = True

    if getattr(aug, "x", None) is not None:
        aug.x = aug.x[keep_idx]

    if getattr(aug, "batch", None) is not None:
        aug.batch = aug.batch[keep_idx]

    if getattr(aug, "pos", None) is not None and torch.is_tensor(aug.pos) and aug.pos.size(0) == num_nodes:
        aug.pos = aug.pos[keep_idx]
    # Keep num_nodes explicit for downstream components that rely on it.
    aug.num_nodes = int(keep_idx.numel())

    edge_index = getattr(aug, "edge_index", None)
    if edge_index is not None:
        edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
        filtered_edge_index = edge_index[:, edge_mask]
        node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
        node_map[keep_idx] = torch.arange(keep_idx.numel(), dtype=torch.long, device=edge_index.device)
        aug.edge_index = node_map[filtered_edge_index]
        _apply_edge_mask(aug, edge_mask)

    return aug


def subgraph_sampling(data, p: float):
    """GraphCL-style subgraph augmentation: keep ~p fraction of nodes via neighborhood expansion."""
    aug = data.clone()
    if getattr(aug, "x", None) is None or getattr(aug, "edge_index", None) is None:
        return aug

    node_num = int(aug.x.size(0))
    if node_num <= 1:
        return aug

    keep_target = max(1, int(node_num * float(p)))
    edge_index = aug.edge_index
    src = edge_index[0]
    dst = edge_index[1]

    seed = int(torch.randint(node_num, (1,), device=src.device).item())
    selected = [seed]
    selected_set = {seed}

    def adjacent_nodes(node: int) -> set[int]:
        # Treat graph connectivity as undirected for frontier expansion. A
        # directed-only incoming edge still connects the two nodes and must
        # not make the sampler stop prematurely.
        outgoing = dst[src == node]
        incoming = src[dst == node]
        return set(torch.cat([outgoing, incoming]).tolist())

    frontier = adjacent_nodes(seed) - selected_set
    while len(selected) < keep_target:
        if frontier:
            nxt = random.choice(tuple(frontier))
            frontier.discard(nxt)
        else:
            # Disconnected graph: restart from an unselected component so the
            # requested keep count remains exact whenever enough nodes exist.
            remaining = list(set(range(node_num)) - selected_set)
            if not remaining:
                break
            nxt = random.choice(remaining)
        if nxt in selected_set:
            continue
        selected.append(nxt)
        selected_set.add(nxt)
        frontier.update(adjacent_nodes(nxt) - selected_set)

    keep_mask = torch.zeros(node_num, dtype=torch.bool, device=aug.x.device)
    keep_mask[torch.as_tensor(selected, dtype=torch.long, device=aug.x.device)] = True
    return _subset_nodes(aug, keep_mask)


def node_dropping(data, p: float):
    """Drop an exact p-ratio of nodes (GraphCL-style dropN)."""
    if getattr(data, "x", None) is None:
        return data.clone()
    num_nodes = data.x.size(0)
    drop_num = int(num_nodes * float(p))
    if drop_num <= 0:
        return data.clone()

    idx_perm = torch.randperm(num_nodes, device=data.x.device)
    keep_idx = idx_perm[drop_num:]
    if keep_idx.numel() == 0:
        keep_idx = idx_perm[:1]
    keep_idx, _ = torch.sort(keep_idx)
    keep_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    keep_mask[keep_idx] = True
    return _subset_nodes(data, keep_mask)


def feature_masking(data, p: float):
    """Mask node features with probability p (node-wise token masking)."""
    aug = data.clone()
    if getattr(aug, "x", None) is not None:
        node_num = aug.x.size(0)
        mask_num = int(node_num * p)
        if mask_num > 0:
            idx_mask = torch.randperm(node_num, device=aug.x.device)[:mask_num]
            token = aug.x.float().mean(dim=0, keepdim=True)
            if torch.is_floating_point(aug.x):
                token = token.to(dtype=aug.x.dtype)
            else:
                token = token.round().to(dtype=aug.x.dtype)
            aug.x[idx_mask] = token.expand_as(aug.x[idx_mask])
    return aug


def ensure_num_nodes_attr(data) -> None:
    """Normalize ``num_nodes`` so all items can be collated by PyG Batch."""
    x = getattr(data, "x", None)
    edge_index = getattr(data, "edge_index", None)
    num_nodes = getattr(data, "num_nodes", None)
    if x is not None:
        num_nodes = int(x.size(0))
    elif edge_index is not None and edge_index.numel() > 0:
        num_nodes = int(edge_index.max().item()) + 1
    elif num_nodes is None:
        num_nodes = 0
    data.num_nodes = int(num_nodes)


def apply_per_graph_augment(data, augment: Callable):
    """Apply *augment* per graph in a batch, preserving positive-pair alignment."""
    if getattr(data, "batch", None) is None:
        aug = augment(data)
        ensure_num_nodes_attr(aug)
        return aug
    graphs = data.to_data_list()
    aug_graphs = [augment(g) for g in graphs]
    for graph in aug_graphs:
        ensure_num_nodes_attr(graph)
    return Batch.from_data_list(aug_graphs)


__all__ = [
    "VALID_AUG_NAMES",
    "apply_per_graph_augment",
    "edge_perturbation",
    "ensure_num_nodes_attr",
    "feature_masking",
    "node_dropping",
    "subgraph_sampling",
]
