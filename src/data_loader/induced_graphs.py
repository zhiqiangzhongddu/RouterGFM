"""Utilities for generating and caching node- and edge-induced subgraph datasets."""

import fcntl
import os
import secrets
from pathlib import Path
from typing import BinaryIO, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph

from src.utils import ensure_dir

from .dataset_paths import _dataset_scoped_dir, _sanitize_name
from .utils import safe_torch_load


EDGE_INDUCED_GRAPH_SCHEMA = "edge-induced-graph-v3-global-target-pair"


def _induced_cache_path(
    dataset_name: str,
    task_level: str,
    cache_root_path: Path,
    suffix: str,
):
    sanitized = _sanitize_name(dataset_name)
    dataset_dir = _dataset_scoped_dir(cache_root_path, sanitized)
    return dataset_dir / f"{sanitized}_induced_{task_level}_{suffix}.pt"


def _load_induced_cache(path: Path, expected_meta: Dict) -> Dict | None:
    if not path.is_file():
        return None
    try:
        payload = safe_torch_load(path)
        meta = payload.get("meta", {})
        for key, value in expected_meta.items():
            if meta.get(key) != value:
                return None
        return payload
    except Exception:
        return None


def _acquire_induced_cache_build_lock(path: Path) -> BinaryIO:
    """Hold an exclusive cross-process lock while a missing cache is built."""
    ensure_dir(path.parent)
    lock_path = path.with_name(f".{path.name}.lock")
    handle = lock_path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except BaseException:
        handle.close()
        raise
    return handle


def _release_induced_cache_build_lock(handle: BinaryIO | None) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _save_induced_cache(path: Path, payload: Dict):
    # Atomic write: save to a per-PID temp file, then rename.  Prevents
    # killed/OOM jobs from leaving a corrupt .pt at the final path, and
    # prevents concurrent jobs on the shared filesystem from colliding
    # on a deterministic temp name.
    ensure_dir(path.parent)
    # PIDs are only node-local, so two SLURM workers on different nodes can
    # have the same PID while sharing this directory.  Add random entropy to
    # keep their atomic-save siblings distinct.
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


class SingleGraphDataLoader:
    """Lightweight wrapper so node-level datasets match the DataLoader interface."""

    def __init__(self, data: Data):
        self.data = data

    def __iter__(self):
        yield self.data

    def __len__(self):
        return 1


def _build_reverse_csr(edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a compact inbound adjacency index for bounded node-induced expansion."""
    if edge_index is None or edge_index.numel() == 0 or num_nodes <= 0:
        return (
            torch.zeros(max(num_nodes, 0) + 1, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )

    reverse_edge_index = edge_index[[1, 0]].detach().cpu()
    values = torch.ones(reverse_edge_index.size(1), dtype=torch.uint8)
    reverse_adj = (
        torch.sparse_coo_tensor(
            reverse_edge_index,
            values,
            size=(num_nodes, num_nodes),
            device="cpu",
        )
        .coalesce()
        .to_sparse_csr()
    )
    return reverse_adj.crow_indices(), reverse_adj.col_indices()


def _sample_bounded_k_hop_subset(
    center_idx: int,
    reverse_crow: torch.Tensor,
    reverse_cols: torch.Tensor,
    smallest_size: int,
    largest_size: int | None,
    max_hops: int,
    start_hops: int,
) -> torch.Tensor:
    """Sample a bounded inbound k-hop neighborhood without materializing large frontiers."""
    min_size = max(1, int(smallest_size))
    max_size = int(largest_size) if largest_size is not None and int(largest_size) > 0 else None
    if max_size is not None:
        min_size = min(min_size, max_size)

    min_hops = max(0, int(start_hops))
    hop_limit = max(min_hops, int(max_hops))
    selected = [int(center_idx)]
    frontier = [int(center_idx)]
    visited = {int(center_idx)}
    hop = 0

    while frontier and hop < hop_limit:
        hop += 1
        if max_size is not None and len(selected) >= max_size:
            break

        remaining_slots = (max_size - len(selected)) if max_size is not None else max(min_size - len(selected), 1)
        if remaining_slots <= 0:
            break

        per_frontier_budget = max(1, (remaining_slots + len(frontier) - 1) // max(len(frontier), 1))
        sample_budget = max(per_frontier_budget * 2, min(remaining_slots, 32))
        next_frontier: List[int] = []

        for node_id in frontier:
            start = int(reverse_crow[node_id].item())
            end = int(reverse_crow[node_id + 1].item())
            degree = end - start
            if degree <= 0:
                continue

            neighbors = reverse_cols[start:end]
            if degree > sample_budget:
                choice = torch.randperm(degree)[:sample_budget]
                neighbors = neighbors[choice]

            for neighbor in neighbors.tolist():
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_frontier.append(int(neighbor))
                if len(selected) + len(next_frontier) >= (max_size or (len(selected) + remaining_slots)):
                    break
            if len(selected) + len(next_frontier) >= (max_size or (len(selected) + remaining_slots)):
                break

        if not next_frontier:
            break

        if max_size is not None and len(selected) + len(next_frontier) > max_size:
            next_frontier = next_frontier[: max_size - len(selected)]
        selected.extend(next_frontier)
        frontier = next_frontier

        if hop >= min_hops and len(selected) >= min_size:
            break

    return torch.tensor(selected, dtype=torch.long)


def build_induced_graphs(
    data: Data,
    smallest_size: int = 10,
    largest_size: int = 30,
    max_hops: int = 5,
    start_hops: int = 2,
):
    """Generate induced subgraphs centered at each node."""
    induced_graphs = []
    if data is None:
        return induced_graphs

    edge_index = getattr(data, "edge_index", None)
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is None and getattr(data, "x", None) is not None:
        num_nodes = data.x.size(0)
    if num_nodes is None or num_nodes <= 0:
        return induced_graphs

    labels = getattr(data, "y", None)
    if labels is not None:
        labels = labels.view(-1)
    device = data.x.device if getattr(data, "x", None) is not None else torch.device("cpu")
    has_edges = edge_index is not None and edge_index.numel() > 0
    reverse_crow = reverse_cols = None
    use_bounded_expansion = has_edges and largest_size is not None and largest_size > 0
    if use_bounded_expansion:
        reverse_crow, reverse_cols = _build_reverse_csr(edge_index, int(num_nodes))

    # When labels exist, skip unlabeled nodes (NaN) and negative labels.
    # When labels are absent (unsupervised pretraining), iterate all nodes.
    if labels is not None:
        if labels.is_floating_point():
            valid_mask = torch.isfinite(labels) & (labels >= 0)
        else:
            valid_mask = labels >= 0
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).view(-1)
    else:
        valid_indices = torch.arange(num_nodes)

    for idx in valid_indices.tolist():
        label = int(labels[idx].item()) if labels is not None else None
        hops = start_hops
        subset = torch.tensor([idx], device=device)

        if has_edges:
            if use_bounded_expansion and reverse_crow is not None and reverse_cols is not None:
                subset = _sample_bounded_k_hop_subset(
                    center_idx=idx,
                    reverse_crow=reverse_crow,
                    reverse_cols=reverse_cols,
                    smallest_size=smallest_size,
                    largest_size=largest_size,
                    max_hops=max_hops,
                    start_hops=start_hops,
                )
            else:
                subset, _, _, _ = k_hop_subgraph(
                    node_idx=idx,
                    num_hops=hops,
                    edge_index=edge_index,
                    relabel_nodes=True,
                    num_nodes=num_nodes,
                )
                while subset.numel() < smallest_size and hops < max_hops:
                    hops += 1
                    subset, _, _, _ = k_hop_subgraph(
                        node_idx=idx,
                        num_hops=hops,
                        edge_index=edge_index,
                        relabel_nodes=True,
                        num_nodes=num_nodes,
                    )

        subset_cpu = subset.cpu()
        if subset_cpu.numel() > largest_size:
            keep = subset_cpu[torch.randperm(subset_cpu.numel())[: largest_size - 1]]
            subset_cpu = torch.unique(torch.cat([torch.tensor([idx], dtype=torch.long), keep]))

        subset = subset_cpu.to(device)
        if has_edges:
            sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        else:
            sub_edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        x = data.x[subset] if getattr(data, "x", None) is not None else None

        graph = Data(x=x, edge_index=sub_edge_index)
        if label is not None:
            graph.y = torch.tensor(label)
        graph.base_node_id = idx
        graph.index = idx
        # Local position of the centre node inside this subgraph. The
        # "index" suffix makes PyG batching offset it per graph, so after
        # batching it indexes directly into the batched node dimension
        # (used by graph_pooling="target").
        graph.target_node_index = (subset == idx).nonzero(as_tuple=False).view(-1)[:1].cpu()
        induced_graphs.append(graph)
    return induced_graphs


def _edge_induced_subgraph(
    data: Data,
    endpoints: torch.Tensor,
    max_hops: int,
    max_size: int | None = None,
    label: int | None = None,
):
    """Build a single edge-centered induced subgraph given endpoints."""
    if endpoints.numel() != 2:
        raise ValueError("endpoints must contain exactly two node indices.")

    device = data.edge_index.device
    x = getattr(data, "x", None)
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is None and x is not None:
        num_nodes = x.size(0)
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=endpoints.to(device=device),
        edge_index=data.edge_index,
        num_hops=max_hops,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )
    if max_size is not None and max_size > 0 and subset.numel() > max_size:
        target = max(max_size, 2)
        keep = torch.randperm(subset.numel(), device=subset.device)[: target - 2]
        keep = torch.unique(torch.cat([keep, torch.tensor([mapping[0], mapping[1]], device=subset.device)]))
        subset = subset[keep]
        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
        map_u = torch.where(subset == endpoints[0])[0][0]
        map_v = torch.where(subset == endpoints[1])[0][0]
        mapping = torch.stack([map_u, map_v])
    sub_x = x[subset] if x is not None else None
    mapped_u, mapped_v = int(mapping[0]), int(mapping[1])
    # SEAL-style target removal: drop the queried edge (both directions) from
    # the subgraph. Train positives would otherwise contain the edge they are
    # asked to predict while val/test positives never do, so "is the edge
    # present" becomes a train-only shortcut.
    keep = ~(
        ((sub_edge_index[0] == mapped_u) & (sub_edge_index[1] == mapped_v))
        | ((sub_edge_index[0] == mapped_v) & (sub_edge_index[1] == mapped_u))
    )
    sub_edge_index = sub_edge_index[:, keep]
    target_edge = torch.tensor([[mapped_u], [mapped_v]], dtype=torch.long, device=device)
    graph = Data(
        x=sub_x,
        edge_index=sub_edge_index,
        edge_label_index=target_edge,
        # Keep the source-graph endpoints under a name that does not contain
        # ``index``.  PyG increments attributes with that substring while
        # batching; these ids must remain in the global node-id space so
        # downstream link decoders can establish an exact pair bijection.
        global_target_pair=endpoints.detach().cpu().to(torch.long).reshape(2),
    )
    if label is not None:
        graph.y = torch.tensor(label, dtype=torch.long)
    return graph


def build_edge_induced_graphs(
    data: Data,
    edge_indices: torch.Tensor,
    max_hops: int = 2,
    max_size: int | None = None,
):
    """Generate edge-centered induced subgraphs (one per target edge)."""
    if data is None or edge_indices.numel() == 0:
        return []

    edge_pairs = data.edge_index[:, edge_indices.to(dtype=torch.long)]
    graphs = []

    for idx in range(edge_pairs.size(1)):
        endpoints = edge_pairs[:, idx]
        graphs.append(_edge_induced_subgraph(data, endpoints=endpoints, max_hops=max_hops, max_size=max_size))
    return graphs


# Shared by forked edge-cache workers (set in the parent right before the
# pool is created; children inherit it copy-on-write).
_EDGE_PARALLEL_DATA: Data | None = None


def _resolve_edge_induced_workers(total_pairs: int, data: Data) -> int:
    """Worker count for parallel edge-induced generation (0 = serial).

    Controlled by ICG_EDGE_INDUCED_WORKERS. Parallelism uses fork, so it is
    disabled for CUDA-resident graphs and for small workloads where pool
    overhead dominates.
    """
    try:
        workers = int(os.environ.get("ICG_EDGE_INDUCED_WORKERS", "0") or 0)
    except ValueError:
        return 0
    if workers <= 1 or total_pairs < 256:
        return 0
    if data.edge_index.is_cuda or (getattr(data, "x", None) is not None and data.x.is_cuda):
        return 0
    return workers


def _edge_pair_chunk_worker(args):
    # Inputs/outputs cross the pool as plain pickled bytes: torch's
    # multiprocessing reducers (fd or shm-file passing per tensor) fall over
    # at this volume of tiny tensors ("received 0 items of ancdata" /
    # "unable to mmap ... Cannot allocate memory").
    import pickle

    start, pairs_list, max_hops, max_size, label = args
    torch.set_num_threads(1)
    pairs = torch.tensor(pairs_list, dtype=torch.long)
    graphs = [
        _edge_induced_subgraph(
            _EDGE_PARALLEL_DATA, endpoints=pairs[:, i], max_hops=max_hops, max_size=max_size, label=label
        )
        for i in range(pairs.size(1))
    ]
    return start, len(graphs), pickle.dumps(graphs, protocol=pickle.HIGHEST_PROTOCOL)


def _build_edge_graphs_for_pairs(data: Data, pairs: torch.Tensor, label: int, max_hops: int, max_size: int | None):
    total = pairs.size(1)
    workers = _resolve_edge_induced_workers(total, data)
    if workers <= 0:
        return [
            _edge_induced_subgraph(data, endpoints=pairs[:, i], max_hops=max_hops, max_size=max_size, label=label)
            for i in range(total)
        ]

    import multiprocessing as mp
    import pickle

    global _EDGE_PARALLEL_DATA
    chunk = max(32, (total + workers * 8 - 1) // (workers * 8))
    tasks = [
        (start, pairs[:, start : start + chunk].tolist(), max_hops, max_size, label)
        for start in range(0, total, chunk)
    ]
    _EDGE_PARALLEL_DATA = data
    try:
        with mp.get_context("fork").Pool(processes=workers) as pool:
            results = []
            done_pairs = 0
            next_report = 0.1
            for start, n_graphs, blob in pool.imap_unordered(_edge_pair_chunk_worker, tasks):
                results.append((start, blob))
                done_pairs += n_graphs
                if total and done_pairs / total >= next_report:
                    print(f"[Induced] edge subgraphs (label={label}): {done_pairs}/{total}")
                    next_report += 0.1
    finally:
        _EDGE_PARALLEL_DATA = None
    results.sort(key=lambda item: item[0])
    return [graph for _, blob in results for graph in pickle.loads(blob)]


def build_edge_induced_graphs_supervised(
    data: Data,
    pos_edge_pairs: torch.Tensor,
    neg_edge_pairs: torch.Tensor,
    max_hops: int = 2,
    max_size: int | None = None,
):
    """Generate edge-centered induced subgraphs with binary labels."""
    if data is None:
        return []

    graphs = []
    if pos_edge_pairs is not None and pos_edge_pairs.numel() > 0:
        graphs.extend(_build_edge_graphs_for_pairs(data, pos_edge_pairs, label=1, max_hops=max_hops, max_size=max_size))

    if neg_edge_pairs is not None and neg_edge_pairs.numel() > 0:
        graphs.extend(_build_edge_graphs_for_pairs(data, neg_edge_pairs, label=0, max_hops=max_hops, max_size=max_size))
    return graphs


class InducedGraphDataset(Dataset):
    """Dataset of induced subgraphs (one per original node)."""

    def __init__(
        self,
        graphs,
        base_num_nodes: int = None,
        base_num_edges: int = None,
        base_info: Dict = None,
        split_tags=None,
    ):
        self.graphs = graphs
        self.num_features = graphs[0].num_node_features if graphs else 0
        labels_list = []
        for graph in graphs:
            y = getattr(graph, "y", None)
            if y is not None:
                labels_list.append(y.view(-1)[0])
        labels = torch.stack(labels_list) if labels_list else torch.tensor([])
        self.num_classes = int(labels.max().item() + 1) if labels.numel() > 0 else None
        self.base_num_nodes = base_num_nodes
        self.base_num_edges = base_num_edges
        self.base_dataset_info = base_info or {}
        self.split_tags = split_tags or ["train"] * len(graphs)
        if self.base_dataset_info.get("name"):
            self.name = f"Induced({self.base_dataset_info['name']})"

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        if not hasattr(data, "base_node_id"):
            data.base_node_id = idx
        if self.split_tags:
            data.split = self.split_tags[idx]
        return data
