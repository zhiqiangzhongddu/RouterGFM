"""Pretrain-method-local helpers.

**Ownership rule:** this module is strictly method-local. Helpers that
need to be shared with ``train/``, ``finetune/``, or any other workflow
must live under ``src/utils/`` instead. Pool primitives live in
``src/utils/pool.py``; pretrain methods import them from there directly.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch


def resolve_ptr(data) -> Optional[torch.Tensor]:
    """Return a ``ptr`` tensor for ``data``, reconstructing from ``batch`` if needed.

    Matches the ``torch_geometric.data.Batch.ptr`` layout: a 1-D
    ``LongTensor`` of length ``num_graphs + 1`` with the cumulative node
    offsets. Returns ``None`` for single-graph inputs that carry neither
    ``ptr`` nor ``batch``.
    """
    ptr = getattr(data, "ptr", None)
    if ptr is not None:
        return ptr
    batch = getattr(data, "batch", None)
    if batch is None:
        return None
    counts = torch.bincount(batch)
    out = torch.zeros(counts.numel() + 1, dtype=torch.long, device=batch.device)
    torch.cumsum(counts, dim=0, out=out[1:])
    return out


def iter_graph_slices(data) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` node-index spans for each graph in ``data``.

    For single-graph inputs, yields a single span covering ``num_nodes``.
    """
    ptr = resolve_ptr(data)
    if ptr is None or ptr.numel() <= 1:
        num_nodes = int(getattr(data, "num_nodes", None) or data.x.size(0))
        yield 0, num_nodes
        return
    for g in range(int(ptr.numel()) - 1):
        yield int(ptr[g].item()), int(ptr[g + 1].item())


def make_zero_loss(
    task: torch.nn.Module,
    device: torch.device,
    *extras: torch.Tensor,
) -> torch.Tensor:
    """Return a zero-valued loss that is still attached to the graph.

    Sums ``.sum() * 0.0`` across the task's parameters (and any extra
    tensors passed in) so that the caller can ``loss.backward()`` safely
    on a degenerate batch without producing NaNs or detached tensors.
    """
    anchors: list[torch.Tensor] = []
    for p in task.parameters():
        anchors.append(p)
        break  # one anchor is enough to preserve the autograd graph
    for t in extras:
        if t is not None and torch.is_tensor(t):
            anchors.append(t)
    if not anchors:
        return torch.zeros((), device=device, requires_grad=True)
    return sum(a.sum() for a in anchors) * 0.0


def sample_masked_node_indices(
    num_nodes: int,
    ptr: Optional[torch.Tensor],
    batch: Optional[torch.Tensor],
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample node indices to mask, guaranteeing at least one per graph.

    When ``ptr`` (or ``batch``) is provided, samples
    ``int(n_g * mask_ratio + 1)`` indices independently within each graph,
    mirroring the per-graph behavior of Hu et al. 2020's ``MaskAtom``
    transform. Without batching info, falls back to a single global draw.

    Args:
        num_nodes: Total number of nodes in the (possibly batched) graph.
        ptr: Optional cumulative node counts, shape ``[num_graphs + 1]``.
        batch: Optional batch assignment, shape ``[num_nodes]``. Used to
            reconstruct ``ptr`` when ``ptr`` itself is not given.
        mask_ratio: Fraction of nodes to mask per graph.
        device: Device to place the returned index tensor on.

    Returns:
        A 1-D ``LongTensor`` of global node indices, sorted per graph.
    """
    if num_nodes <= 0:
        return torch.empty(0, dtype=torch.long, device=device)

    if ptr is None and batch is not None:
        counts = torch.bincount(batch)
        ptr = torch.zeros(counts.numel() + 1, dtype=torch.long, device=batch.device)
        torch.cumsum(counts, dim=0, out=ptr[1:])

    if ptr is None:
        num_mask = int(num_nodes * mask_ratio + 1)
        num_mask = min(num_mask, num_nodes)
        return torch.randperm(num_nodes, device=device)[:num_mask]

    ptr = ptr.to(device=device)
    parts: list[torch.Tensor] = []
    num_graphs = int(ptr.numel() - 1)
    for g in range(num_graphs):
        start = int(ptr[g].item())
        end = int(ptr[g + 1].item())
        n_g = end - start
        if n_g <= 0:
            continue
        k = int(n_g * mask_ratio + 1)
        k = min(k, n_g)
        local = torch.randperm(n_g, device=device)[:k]
        parts.append(local + start)

    if not parts:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(parts, dim=0)


__all__ = [
    "iter_graph_slices",
    "make_zero_loss",
    "resolve_ptr",
    "sample_masked_node_indices",
]
