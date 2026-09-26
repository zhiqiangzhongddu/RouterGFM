"""Graph-level dataset filter to remove empty graph instances."""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Subset

from src.utils import ensure_dir

from .dataset_metadata import _safe_dataset_num_classes
from .dataset_paths import _base_dataset_for_split_name, _dataset_scoped_dir
from .utils import safe_torch_load


GRAPH_FILTER_REASON = "num_nodes<=0"


def _graph_filter_manifest_path(filter_root: Path | str, dataset_name: str) -> Path:
    filter_root_path = Path(filter_root)
    base_name = _base_dataset_for_split_name(dataset_name)
    dataset_filter_dir = _dataset_scoped_dir(filter_root_path, base_name)
    return dataset_filter_dir / f"{base_name}_graph_filter.pt"


def _graph_num_nodes(data) -> int:
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is None and getattr(data, "x", None) is not None:
        num_nodes = data.x.size(0)
    if num_nodes is None:
        edge_index = getattr(data, "edge_index", None)
        if edge_index is not None and edge_index.numel() > 0:
            num_nodes = int(edge_index.max().item()) + 1
    return int(num_nodes) if num_nodes is not None else 0


def _graph_num_edges(data) -> int | None:
    num_edges = getattr(data, "num_edges", None)
    if num_edges is not None:
        return int(num_edges)
    edge_index = getattr(data, "edge_index", None)
    if edge_index is not None:
        return int(edge_index.size(1))
    return None


def _validate_graph_filter_indices(
    keep_indices,
    drop_indices,
    raw_total: int,
) -> Tuple[List[int], List[int]] | None:
    if not isinstance(keep_indices, (list, tuple)) or not isinstance(drop_indices, (list, tuple)):
        return None
    try:
        keep = [int(idx) for idx in keep_indices]
        drop = [int(idx) for idx in drop_indices]
    except Exception:
        return None

    if len(keep) + len(drop) != int(raw_total):
        return None

    seen = [False] * int(raw_total)
    for idx in keep + drop:
        if idx < 0 or idx >= int(raw_total):
            return None
        if seen[idx]:
            return None
        seen[idx] = True

    if not all(seen):
        return None
    return keep, drop


def _graph_filter_cache_meta(payload_meta: Dict) -> Dict:
    return {
        "dataset_name": str(payload_meta.get("dataset_name") or ""),
        "task_level": "graph",
        "reason": str(payload_meta.get("reason") or GRAPH_FILTER_REASON),
        "raw_total": int(payload_meta.get("raw_total", 0) or 0),
        "filtered_total": int(payload_meta.get("filtered_total", 0) or 0),
        "drop_count": int(payload_meta.get("drop_count", 0) or 0),
    }


def _graph_filter_cache_meta_for_dataset(dataset) -> Dict | None:
    meta = getattr(dataset, "graph_filter_meta", None)
    if not isinstance(meta, dict):
        return None
    try:
        return _graph_filter_cache_meta(meta)
    except Exception:
        return None


def _set_graph_filter_attrs(dataset, meta: Dict, manifest_path: Path, keep_indices: List[int], drop_indices: List[int]) -> None:
    try:
        dataset.graph_filter_meta = dict(meta)  # type: ignore[attr-defined]
        dataset.graph_filter_manifest = str(manifest_path)  # type: ignore[attr-defined]
        dataset.graph_filter_keep_indices = list(keep_indices)  # type: ignore[attr-defined]
        dataset.graph_filter_drop_indices = list(drop_indices)  # type: ignore[attr-defined]
    except Exception:
        pass


def _set_filtered_graph_dataset_metadata(
    filtered_dataset,
    base_dataset,
    *,
    dataset_name: str,
    filtered_total_nodes: int | None,
    filtered_total_edges: int | None,
    filtered_total_graphs: int,
) -> None:
    attrs = {
        "name": getattr(base_dataset, "name", dataset_name),
        "root": getattr(base_dataset, "root", None),
        "num_classes": _safe_dataset_num_classes(base_dataset),
        "has_labels": getattr(base_dataset, "has_labels", None),
        "num_features": getattr(base_dataset, "num_features", None),
        "num_node_features": getattr(base_dataset, "num_node_features", None),
        "total_nodes": filtered_total_nodes,
        "total_edges": filtered_total_edges,
        "avg_nodes_per_graph": (
            float(filtered_total_nodes) / float(filtered_total_graphs)
            if filtered_total_nodes is not None and filtered_total_graphs > 0
            else None
        ),
        "avg_edges_per_graph": (
            float(filtered_total_edges) / float(filtered_total_graphs)
            if filtered_total_edges is not None and filtered_total_graphs > 0
            else None
        ),
    }
    for key, value in attrs.items():
        if value is None:
            continue
        try:
            setattr(filtered_dataset, key, value)
        except Exception:
            continue


def _load_graph_filter_manifest(path: Path, dataset_name: str, raw_total: int) -> Dict | None:
    if not path.is_file():
        return None
    try:
        payload = safe_torch_load(path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        return None
    if str(meta.get("dataset_name") or "") != str(dataset_name):
        return None
    if str(meta.get("task_level") or "") != "graph":
        return None
    if str(meta.get("reason") or "") != GRAPH_FILTER_REASON:
        return None
    if int(meta.get("raw_total", -1) or -1) != int(raw_total):
        return None

    validated = _validate_graph_filter_indices(
        keep_indices=payload.get("keep_indices"),
        drop_indices=payload.get("drop_indices"),
        raw_total=int(raw_total),
    )
    if validated is None:
        return None
    keep_indices, drop_indices = validated

    filtered_total = int(meta.get("filtered_total", -1) or -1)
    drop_count = int(meta.get("drop_count", -1) or -1)
    if filtered_total != len(keep_indices) or drop_count != len(drop_indices):
        return None

    payload["keep_indices"] = keep_indices
    payload["drop_indices"] = drop_indices
    payload["meta"] = {
        "dataset_name": str(dataset_name),
        "task_level": "graph",
        "reason": GRAPH_FILTER_REASON,
        "raw_total": int(raw_total),
        "filtered_total": len(keep_indices),
        "drop_count": len(drop_indices),
    }
    return payload


def _create_graph_filter_manifest(dataset, dataset_name: str) -> Dict:
    raw_total = int(len(dataset))
    keep_indices: List[int] = []
    drop_indices: List[int] = []
    filtered_total_nodes = 0
    filtered_total_edges = 0
    has_total_edges = True

    for idx in range(raw_total):
        graph = dataset[idx]
        num_nodes = _graph_num_nodes(graph)
        if num_nodes <= 0:
            drop_indices.append(idx)
            continue
        keep_indices.append(idx)
        filtered_total_nodes += int(num_nodes)
        num_edges = _graph_num_edges(graph)
        if num_edges is None:
            has_total_edges = False
        elif has_total_edges:
            filtered_total_edges += int(num_edges)

    return {
        "keep_indices": keep_indices,
        "drop_indices": drop_indices,
        "meta": {
            "dataset_name": str(dataset_name),
            "task_level": "graph",
            "reason": GRAPH_FILTER_REASON,
            "raw_total": raw_total,
            "filtered_total": len(keep_indices),
            "drop_count": len(drop_indices),
        },
        "stats": {
            "filtered_total_nodes": int(filtered_total_nodes),
            "filtered_total_edges": int(filtered_total_edges) if has_total_edges else None,
        },
    }


def _sanitize_graph_dataset(
    dataset,
    dataset_name: str,
    graph_filter_dir: str = "",
):
    manifest_root = Path(graph_filter_dir) if graph_filter_dir else Path("data/filters")
    manifest_path = _graph_filter_manifest_path(manifest_root, dataset_name)
    raw_total = int(len(dataset))

    payload = _load_graph_filter_manifest(manifest_path, dataset_name=dataset_name, raw_total=raw_total)
    manifest_was_created = payload is None
    if payload is None:
        payload = _create_graph_filter_manifest(dataset, dataset_name=dataset_name)
        ensure_dir(str(manifest_path.parent))
        torch.save(payload, manifest_path)
        print(f"[GraphFilter] Saved manifest to {manifest_path}")
    else:
        print(f"[GraphFilter] Loaded manifest from {manifest_path}")

    keep_indices = list(payload["keep_indices"])
    drop_indices = list(payload["drop_indices"])
    meta = dict(payload["meta"])
    stats = payload.get("stats", {}) if isinstance(payload.get("stats"), dict) else {}
    filtered_total = int(meta.get("filtered_total", len(keep_indices)) or len(keep_indices))
    if filtered_total <= 0:
        raise RuntimeError(f"[GraphFilter] All graph instances were filtered for dataset={dataset_name}.")

    filtered_total_nodes = stats.get("filtered_total_nodes")
    filtered_total_edges = stats.get("filtered_total_edges")
    if filtered_total_nodes is not None:
        filtered_total_nodes = int(filtered_total_nodes)
    if filtered_total_edges is not None:
        filtered_total_edges = int(filtered_total_edges)

    if len(drop_indices) > 0:
        filtered_dataset = Subset(dataset, keep_indices)
        _set_filtered_graph_dataset_metadata(
            filtered_dataset,
            dataset,
            dataset_name=dataset_name,
            filtered_total_nodes=filtered_total_nodes,
            filtered_total_edges=filtered_total_edges,
            filtered_total_graphs=filtered_total,
        )
    else:
        filtered_dataset = dataset
        if filtered_total_nodes is not None:
            try:
                filtered_dataset.total_nodes = int(filtered_total_nodes)  # type: ignore[attr-defined]
                filtered_dataset.avg_nodes_per_graph = float(filtered_total_nodes) / float(filtered_total)  # type: ignore[attr-defined]
            except Exception:
                pass
        if filtered_total_edges is not None:
            try:
                filtered_dataset.total_edges = int(filtered_total_edges)  # type: ignore[attr-defined]
                filtered_dataset.avg_edges_per_graph = float(filtered_total_edges) / float(filtered_total)  # type: ignore[attr-defined]
            except Exception:
                pass

    _set_graph_filter_attrs(filtered_dataset, meta, manifest_path, keep_indices, drop_indices)
    if manifest_was_created:
        # Full verification only when the manifest was just built: on cached
        # loads this O(N) materialisation of every graph (~437k for
        # ogbg-molpcba) would defeat the point of the manifest.
        for local_idx in range(len(filtered_dataset)):
            if _graph_num_nodes(filtered_dataset[local_idx]) <= 0:
                raise RuntimeError(
                    f"[GraphFilter] Empty graph remained after filtering for dataset={dataset_name} "
                    f"at filtered_index={local_idx}."
                )
    print(
        f"[GraphFilter] dataset={dataset_name} raw_total={int(meta.get('raw_total', raw_total))} "
        f"filtered_total={filtered_total} drop_count={int(meta.get('drop_count', len(drop_indices)))}"
    )
    return filtered_dataset
