"""Feature processing utilities for SVD-based dimensionality reduction and caching."""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

from .dataset_paths import _dataset_scoped_dir
from .dataset_storage import _get_dataset_data_storage, _set_dataset_data_storage
from .dataset_splits import _split_suffix


class EnsureFeatureTransform:
    """If a dataset has no node features, attach degree features."""

    def __call__(self, data: Data) -> Data:
        x = getattr(data, "x", None)
        if x is not None:
            if torch.is_tensor(x) and x.dim() == 2 and x.size(-1) > 0:
                return data
            if hasattr(x, "numel") and x.numel() > 0:
                return data

        num_nodes = data.num_nodes
        if num_nodes is None:
            raise ValueError("Cannot infer num_nodes to build degree features.")
        edge_index = getattr(data, "edge_index", None)
        if edge_index is not None and edge_index.numel() > 0:
            deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
        else:
            deg = torch.zeros(num_nodes, dtype=torch.float)
        data.x = deg.view(num_nodes, 1)
        return data


class SafeSVDFeatureReduction:
    """Robust feature reduction with padding/truncation fallbacks."""

    def __init__(self, out_channels: int):
        self.out_channels = out_channels

    def __call__(self, data: Data) -> Data:
        x = getattr(data, "x", None)
        if x is None or x.numel() == 0:
            return data

        target = self.out_channels
        in_dim = x.size(-1)
        if in_dim < target:
            x_f = x.to(torch.float32) if not torch.is_floating_point(x) else x
            pad = x_f.new_zeros(x_f.size(0), target - in_dim)
            data.x = torch.cat([x_f, pad], dim=1).to(torch.float32)
            return data

        try:
            u, s, _ = torch.linalg.svd(x.float(), full_matrices=False)
            u = u[:, :target]
            s = s[:target]
            reduced = u * s
            data.x = reduced.to(dtype=torch.float32, device=x.device)
        except Exception:
            data.x = x[:, :target].to(torch.float32)
        return data


def _feature_task_suffix(task_level: str | None) -> str:
    if task_level == "edge":
        return "_edge"
    if task_level == "graph":
        return "_graph"
    if task_level == "node":
        return "_node"
    if task_level is None:
        return ""
    raise ValueError(f"Unsupported task level for feature SVD: {task_level}")


def _feature_svd_path(dataset_root: str, name: str, dim: int, task_level: str | None = None) -> Path:
    suffix = _feature_task_suffix(task_level)
    dataset_dir = _dataset_scoped_dir(dataset_root, name)
    return dataset_dir / f"feature_svd_{name}{suffix}_d{dim}.pt"


def _apply_feature_svd(
    dataset,
    name: str,
    dim: int,
    reducer: SafeSVDFeatureReduction,
    task_level: str | None = None,
    output_root: str | None = None,
):
    """Apply SVD once and persist reduced features to avoid recompute."""
    root = output_root or dataset.root
    save_path = _feature_svd_path(root, name, dim, task_level=task_level)

    dataset_data = _get_dataset_data_storage(dataset)
    target_x = getattr(dataset_data, "x", None) if dataset_data is not None else None
    target_device = target_x.device if target_x is not None else "cpu"
    if save_path.exists():
        try:
            payload = torch.load(save_path, map_location=target_device)
            dataset_data = _get_dataset_data_storage(dataset)
            if dataset_data is None:
                raise ValueError("Dataset has no in-memory data storage.")
            dataset_data.x = payload["x"].to(target_device)
            return
        except Exception:
            save_path.unlink(missing_ok=True)

    print(f"[FeatureSVD] Processing features for {name} (task level={task_level or 'node'})...")

    data = _get_dataset_data_storage(dataset)
    if data is None:
        raise ValueError("Dataset has no in-memory data storage for feature SVD.")
    if getattr(data, "x", None) is None and getattr(dataset, "transform", None) is not None:
        try:
            data = dataset.transform(data.clone())
        except Exception:
            pass
    reduced = reducer(data)
    _set_dataset_data_storage(dataset, reduced)
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"x": reduced.x.cpu()}, save_path)
        print(f"[FeatureSVD] Saved features to {save_path}")
    except Exception:
        pass

    try:
        dataset._svd_dim = dim  # type: ignore[attr-defined]
        dataset._svd_task_level = task_level  # type: ignore[attr-defined]
        dataset._feature_svd_root = root  # type: ignore[attr-defined]
    except Exception:
        pass


def _subgraph_svd_path(
    output_dir: str,
    dataset_name: str,
    task_level: str,
    feat_dim: int,
    struct_dim: int,
    matrix_type: str,
    edge_split: Optional[Tuple[float, float, float]] = None,
    edge_seed: Optional[int] = None,
) -> Path:
    sanitized = str(dataset_name).replace("/", "_").replace(" ", "_")
    matrix_type = str(matrix_type or "adjacency").lower()
    dataset_dir = _dataset_scoped_dir(output_dir, sanitized)
    parts = [f"subgraph_svd_{sanitized}_{task_level}"]
    if task_level == "edge" and edge_split is not None:
        parts.append(f"split{_split_suffix(edge_split)}")
    if task_level == "edge" and edge_seed is not None:
        parts.append(f"seed{int(edge_seed)}")
    if feat_dim and feat_dim > 0:
        parts.append(f"feat{feat_dim}")
    parts.append(f"struct{struct_dim}")
    parts.append(matrix_type)
    return dataset_dir / f"{'_'.join(parts)}.pt"


def _svd_singular_values(matrix: torch.Tensor, target_dim: int) -> torch.Tensor:
    if target_dim <= 0:
        return torch.empty((0,), dtype=torch.float32)
    if matrix is None or matrix.numel() == 0:
        return torch.zeros(target_dim, dtype=torch.float32)
    try:
        _, s, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
        vec = s[:target_dim]
    except Exception:
        flat = matrix.flatten()
        vec = flat[:target_dim].float() if flat.numel() else torch.zeros(target_dim, dtype=torch.float32)
    if vec.numel() < target_dim:
        pad = torch.zeros(target_dim - vec.numel(), dtype=torch.float32)
        vec = torch.cat([vec, pad], dim=0)
    return vec.to(torch.float32)


def _subgraph_structure_matrix(data: Data, matrix_type: str) -> torch.Tensor:
    num_nodes = int(getattr(data, "num_nodes", 0) or 0)
    if num_nodes <= 0:
        return torch.empty((0, 0), dtype=torch.float32)
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = (adj + adj.t()).clamp(max=1.0)
    matrix_type = str(matrix_type or "adjacency").lower()
    if matrix_type == "laplacian":
        deg = torch.diag(adj.sum(dim=1))
        return deg - adj
    return adj


def compute_subgraph_svd_features(
    dataset,
    dataset_name: str,
    task_level: str,
    feat_dim: int,
    struct_dim: int,
    matrix_type: str,
    output_dir: str,
    overwrite: bool = False,
) -> Path | None:
    """Compute and cache per-subgraph SVD feature/structure vectors."""
    if feat_dim <= 0 and struct_dim <= 0:
        return None
    edge_split = None
    edge_seed = None
    edge_context = None
    if task_level == "edge":
        raw_split = getattr(dataset, "edge_split", None)
        if isinstance(raw_split, (list, tuple)) and len(raw_split) >= 3:
            edge_split = (float(raw_split[0]), float(raw_split[1]), float(raw_split[2]))
        raw_seed = getattr(dataset, "edge_seed", None)
        if raw_seed is not None:
            edge_seed = int(raw_seed)
        edge_context = getattr(dataset, "edge_context", None)

    output_dir = str(output_dir)
    path = _subgraph_svd_path(
        output_dir,
        dataset_name,
        task_level,
        feat_dim,
        struct_dim,
        matrix_type,
        edge_split=edge_split,
        edge_seed=edge_seed,
    )
    if path.exists() and not overwrite:
        print(f"[SubgraphSVD] Found cached features at {path}, skipping.")
        return path
    print(f"[SubgraphSVD] Processing features for {dataset_name} (task level={task_level})...")

    graphs = getattr(dataset, "graphs", None)
    if graphs is None:
        if hasattr(dataset, "__len__"):
            graphs = [dataset[idx] for idx in range(len(dataset))]
        else:
            return None
    if not graphs:
        return None

    feat_out = torch.zeros((len(graphs), feat_dim), dtype=torch.float32) if feat_dim > 0 else torch.empty((len(graphs), 0))
    struct_out = torch.zeros((len(graphs), struct_dim), dtype=torch.float32) if struct_dim > 0 else torch.empty((len(graphs), 0))
    for idx, data in enumerate(graphs):
        if feat_dim > 0:
            x = getattr(data, "x", None)
            feat_out[idx] = _svd_singular_values(x, feat_dim)
        if struct_dim > 0:
            mat = _subgraph_structure_matrix(data, matrix_type)
            struct_out[idx] = _svd_singular_values(mat, struct_dim)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feat_svd": feat_out,
        "struct_svd": struct_out,
        "meta": {
            "dataset": dataset_name,
            "task_level": task_level,
            "feat_dim": feat_dim,
            "struct_dim": struct_dim,
            "matrix_type": matrix_type,
            "edge_split": edge_split,
            "edge_seed": edge_seed,
            "edge_context": edge_context,
        },
    }
    torch.save(payload, path)
    print(f"[SubgraphSVD] Saved features to {path}")
    return path
