"""GraphMETRO shift components: stochastic graph transforms applied per instance.

The five transforms of the official library (``graphmetro/transform``) used in all
paper experiments. Adaptations for induced instances (target node / LP pair
subgraphs): target nodes and LP endpoints are never dropped, node / LP random
subgraphs are centred on them (graph instances keep the official random centre),
and ``add_edge`` never inserts the LP target pair (the SEAL builder removed it;
re-adding it would leak the label). All randomness comes from an explicit
``torch.Generator`` (CPU) so transforms are reproducible on any device.
"""

from __future__ import annotations

import copy
from typing import Sequence

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import is_undirected, k_hop_subgraph, subgraph, to_undirected

TRANSFORM_NAMES: tuple[str, ...] = ("noisy_node_feat", "add_edge", "drop_edge", "drop_node", "random_subgraph")
IDENTITY = "id"  # the reference (in-distribution) component, expert 0

# Instance-level attributes: never sliced as node / edge attributes.
_INSTANCE_KEYS = frozenset({
    "y", "target_node_index", "edge_label_index", "edge_label", "global_target_pair",
    "base_node_id", "index", "num_nodes", "batch", "ptr", "edge_index",
})
# Local node positions that follow a node relabelling.
_POSITION_KEYS = ("target_node_index", "edge_label_index")


def parse_shift_list(spec: str) -> list[tuple[str, ...]]:
    """``'a/b/a-b'`` -> ``[('a',), ('b',), ('a', 'b')]``; ``-`` composes left to right."""
    shifts = []
    for entry in str(spec).split("/"):
        if not entry.strip():
            continue
        parts = tuple(part.strip() for part in entry.split("-"))
        unknown = [part for part in parts if part not in TRANSFORM_NAMES and part != IDENTITY]
        if unknown or (IDENTITY in parts and len(parts) > 1):
            raise ValueError(
                f"Invalid GraphMETRO shift {entry!r}: components must be in {TRANSFORM_NAMES} "
                f"(or the lone identity {IDENTITY!r})."
            )
        shifts.append(parts)
    if not shifts:
        raise ValueError("GraphMETRO needs at least one training shift.")
    return shifts


def expert_names(shifts: Sequence[tuple[str, ...]]) -> list[str]:
    """``['id']`` + the single components used by ``shifts`` (in ``TRANSFORM_NAMES`` order)."""
    used = {name for shift in shifts for name in shift}
    return [IDENTITY] + [name for name in TRANSFORM_NAMES if name in used]


def shift_target(shift: tuple[str, ...], names: Sequence[str]) -> torch.Tensor:
    """Multi-hot gate target ``[K+1]``: 1 at every component of ``shift`` (index 0 only for ``('id',)``)."""
    target = torch.zeros(len(names))
    for name in shift:
        target[list(names).index(name)] = 1.0
    return target


# --------------------------------------------------------------------------- #
# Per-instance transforms
# --------------------------------------------------------------------------- #
def _protected(data: Data, level: str) -> torch.Tensor:
    """Local positions that must survive: the target node (node) or both endpoints (edge)."""
    if level == "node":
        return torch.as_tensor(data.target_node_index).view(-1)
    if level == "edge":
        return torch.as_tensor(data.edge_label_index).view(-1)
    return torch.empty(0, dtype=torch.long)


def _keep_nodes(data: Data, keep: torch.Tensor, node_keys: Sequence[str]) -> Data:
    """Induced subgraph on the ``keep`` mask, remapping target / endpoint positions."""
    n = keep.numel()
    edge_index, _ = subgraph(keep, data.edge_index, relabel_nodes=True, num_nodes=n)
    remap = torch.full((n,), -1, dtype=torch.long, device=keep.device)
    remap[keep] = torch.arange(int(keep.sum()), device=keep.device)
    out = copy.copy(data)
    out.edge_index = edge_index
    for key in node_keys:
        out[key] = data[key][keep]
    for key in _POSITION_KEYS:
        if key in data:
            out[key] = remap[data[key]]
    if "num_nodes" in data:
        out.num_nodes = int(keep.sum())
    return out


def _add_edge(data: Data, p: float, level: str, g: torch.Generator) -> Data:
    """Add ``round(p * |E|)`` random new edges (undirected if the graph is), like ``add_random_edge``."""
    edge_index, n = data.edge_index, int(data.num_nodes)
    num_new = round(edge_index.size(1) * p)
    if num_new == 0:
        return data
    undirected = is_undirected(edge_index, num_nodes=n)
    blocked = torch.eye(n, dtype=torch.bool, device=edge_index.device)
    blocked[edge_index[0], edge_index[1]] = True
    if level == "edge":
        u, v = _protected(data, level).tolist()
        blocked[u, v] = blocked[v, u] = True
    if undirected:
        blocked = blocked | blocked.t()
        candidates = (~blocked).triu(1).nonzero()
        num_new //= 2
    else:
        candidates = (~blocked).nonzero()
    order = torch.randperm(candidates.size(0), generator=g).to(candidates.device)
    new = candidates[order[:num_new]].t()
    if undirected:
        new = torch.cat([new, new.flip(0)], dim=1)
    out = copy.copy(data)
    out.edge_index = torch.cat([edge_index, new], dim=1)
    return out


def _drop_edge(data: Data, p: float, g: torch.Generator) -> Data:
    """Drop every (directed) edge independently with probability ``p`` (``dropout_edge``)."""
    keep = torch.rand(data.edge_index.size(1), generator=g) >= p
    out = copy.copy(data)
    out.edge_index = data.edge_index[:, keep.to(data.edge_index.device)]
    return out


def _drop_node(data: Data, p: float, level: str, g: torch.Generator, node_keys: Sequence[str]) -> Data:
    """Drop nodes with probability ``p``; protected nodes stay, and at least one node always survives."""
    n = int(data.num_nodes)
    protected = _protected(data, level)
    while True:
        keep = torch.rand(n, generator=g) > p
        keep[protected.cpu()] = True
        if bool(keep.any()):
            break
    return _keep_nodes(data, keep.to(data.edge_index.device), node_keys)


def _random_subgraph(data: Data, k: int, level: str, g: torch.Generator, node_keys: Sequence[str]) -> Data:
    """Bidirectional ``k``-hop subgraph around the target(s), or a random centre for graph instances."""
    n = int(data.num_nodes)
    device = data.edge_index.device
    centres = _protected(data, level)
    if centres.numel() == 0:
        centres = torch.randint(n, (1,), generator=g)
    subset, _, _, _ = k_hop_subgraph(
        centres.to(device), int(k), to_undirected(data.edge_index, num_nodes=n), num_nodes=n
    )
    keep = torch.zeros(n, dtype=torch.bool, device=device)
    keep[subset] = True
    return _keep_nodes(data, keep, node_keys)


def _noisy_node_feat(data_list: list[Data], p: float, g: torch.Generator) -> None:
    """``x + p * std(x) * N(0, 1)`` with the per-dimension std over all nodes of the batch (official)."""
    std = torch.cat([data.x for data in data_list]).float().std(dim=0).nan_to_num(0.0)
    for data in data_list:
        noise = torch.randn(data.x.shape, generator=g).to(data.x.device)
        data.x = data.x + p * std.to(data.x.dtype) * noise.to(data.x.dtype)


def apply_shift(
    batch: Batch,
    shift: tuple[str, ...],
    *,
    p: float,
    k: int,
    task_level_raw: str,
    generator: torch.Generator,
) -> Batch:
    """Apply the components of ``shift`` left to right to every instance; returns a new Batch.

    The input batch is not modified. Edge-level attributes other than
    ``edge_index`` are dropped from the result (edge features are out of scope
    for the repo encoders, and added edges would have none).
    """
    level = str(task_level_raw).lower()
    node_keys = [key for key in batch.keys() if key not in _INSTANCE_KEYS and batch.is_node_attr(key)]
    edge_keys = [key for key in batch.keys() if key not in _INSTANCE_KEYS and batch.is_edge_attr(key)]
    data_list = batch.to_data_list()
    for data in data_list:
        for key in edge_keys:
            del data[key]
    for name in shift:
        if name == IDENTITY:
            continue
        if name == "noisy_node_feat":
            _noisy_node_feat(data_list, p, generator)
        elif name == "add_edge":
            data_list = [_add_edge(data, p, level, generator) for data in data_list]
        elif name == "drop_edge":
            data_list = [_drop_edge(data, p, generator) for data in data_list]
        elif name == "drop_node":
            data_list = [_drop_node(data, p, level, generator, node_keys) for data in data_list]
        elif name == "random_subgraph":
            data_list = [_random_subgraph(data, k, level, generator, node_keys) for data in data_list]
        else:
            raise ValueError(f"Unknown GraphMETRO transform {name!r}.")
    return Batch.from_data_list(data_list)


__all__ = [
    "IDENTITY",
    "TRANSFORM_NAMES",
    "apply_shift",
    "expert_names",
    "parse_shift_list",
    "shift_target",
]
