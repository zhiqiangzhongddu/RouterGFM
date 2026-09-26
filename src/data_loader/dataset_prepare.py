"""
Prepare datasets listed in TSV/list inputs.
For each dataset task, this runs dataset creation, split generation, and optional SVD artifacts.
"""
import os
import io
import contextlib
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
os.environ.setdefault("OGB_ASSUME_YES", "1")

from src.data_loader.datasets import (  # noqa: E402
    create_dataset,
    infer_task_level,
)
from src.data_loader.dataset_loader import make_loaders  # noqa: E402
from src.data_loader.dataset_metadata import (  # noqa: E402
    dataset_info,
    is_regression_dataset,
    resolve_count_split_strategy,
)
from src.data_loader.dataset_paths import _dataset_scoped_dir  # noqa: E402
from src.data_loader.dataset_splits import (  # noqa: E402
    _canonical_split_dataset_name,
    _is_few_shot_split_def,
)
from src.data_loader.svd_features import (  # noqa: E402
    _feature_svd_path,
    compute_subgraph_svd_features,
)

DATASET_SEPARATOR = "=" * 72
TASK_SEPARATOR = "-" * 72


def _print_feature_svd_info(
    dataset_name: str,
    task_level: str,
    feat_reduction: bool,
    feat_reduction_dim: int,
    feature_svd_dir: str,
) -> None:
    if not feat_reduction or int(feat_reduction_dim) <= 0:
        return
    primary = _feature_svd_path(feature_svd_dir, dataset_name, int(feat_reduction_dim), task_level=task_level)
    if primary.exists():
        print(f"[FeatureSVD] Ready features at {primary}")


def _fmt_count(value: Optional[int]) -> str:
    return str(value) if value is not None else "?"


def _fmt_avg(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "?"


def _avg_per_graph(total: Optional[int], graphs: Optional[int]) -> Optional[float]:
    if graphs is not None and int(graphs) == 0:
        return 0.0
    if total is None or graphs is None or int(graphs) < 0:
        return None
    return float(total) / float(graphs)


def _num_nodes(data) -> Optional[int]:
    if data is None:
        return None
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is None and getattr(data, "x", None) is not None:
        num_nodes = data.x.size(0)
    return int(num_nodes) if num_nodes is not None else None


def _num_edges(data) -> Optional[int]:
    if data is None:
        return None
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        return None
    return int(edge_index.size(1))


def _dataset_num_nodes(dataset_obj) -> Optional[int]:
    if dataset_obj is None:
        return None
    try:
        if len(dataset_obj) <= 0:
            return None
        data = dataset_obj[0]
    except (TypeError, IndexError):
        return None
    return _num_nodes(data)


def _should_prepare_edge_level_for_node_dataset(
    dataset_name: str,
    root: str,
    *,
    node_limit: int | None = None,
) -> bool:
    """Return False when a node dataset is too large for derived edge-level preparation."""
    if node_limit is None or int(node_limit) <= 0:
        return True

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            dataset_obj = create_dataset(
                name=dataset_name,
                root=root,
                task_level="node",
                induced=False,
                feat_reduction=False,
                persist_feature_svd=False,
            )
    except Exception as exc:  # pylint: disable=broad-except
        print(
            f"[DataPrep] Failed to inspect node count for {dataset_name}; "
            f"keeping edge-level preparation enabled: {exc}"
        )
        return True

    num_nodes = _dataset_num_nodes(dataset_obj)
    if num_nodes is None:
        print(
            f"[DataPrep] Could not determine node count for {dataset_name}; "
            "keeping edge-level preparation enabled."
        )
        return True

    if int(num_nodes) > int(node_limit):
        print(
            f"[DataPrep] Skipping edge-level preparation for {dataset_name}: "
            f"num_nodes={int(num_nodes)} exceeds limit={int(node_limit)}."
        )
        return False

    return True


def _split_mask_count(data, mask_key: str) -> Optional[int]:
    if data is None:
        return None
    mask = getattr(data, mask_key, None)
    if mask is None:
        return None
    try:
        return int(torch.as_tensor(mask).sum().item())
    except (TypeError, ValueError, RuntimeError):
        return None


def _dataset_len(loader) -> Optional[int]:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    try:
        return int(len(dataset))
    except TypeError:
        return None


def _resolve_subset_base_and_indices(dataset):
    """Unwrap nested Subset wrappers and map indices back to the base dataset."""
    if dataset is None:
        return None, None

    base_dataset = dataset
    resolved_indices = None

    while hasattr(base_dataset, "dataset") and hasattr(base_dataset, "indices"):
        try:
            current_indices = list(base_dataset.indices)
        except (TypeError, AttributeError):
            current_indices = None
        parent = getattr(base_dataset, "dataset", None)
        if parent is None or parent is base_dataset:
            break

        if current_indices is None:
            resolved_indices = None
        elif resolved_indices is None:
            resolved_indices = current_indices
        else:
            try:
                resolved_indices = [current_indices[idx] for idx in resolved_indices]
            except Exception:
                resolved_indices = None

        base_dataset = parent

    return base_dataset, resolved_indices


def _subset_total_from_slices(dataset, key: str) -> Optional[int]:
    """Best-effort fast aggregate from InMemoryDataset slices without loading every graph."""
    if dataset is None:
        return None

    base_dataset, indices = _resolve_subset_base_and_indices(dataset)
    if base_dataset is None:
        return None

    if indices is None:
        try:
            indices = list(range(len(dataset)))
        except Exception:
            return None

    slices = getattr(base_dataset, "slices", None)
    if slices is None or key not in slices:
        return None
    boundaries = slices[key]

    try:
        if torch.is_tensor(boundaries):
            idx = torch.as_tensor(indices, dtype=torch.long)
            if idx.numel() == 0:
                return 0
            span = boundaries[idx + 1] - boundaries[idx]
            return int(span.sum().item())

        total = 0
        for idx in indices:
            total += int(boundaries[idx + 1] - boundaries[idx])
        return total
    except Exception:
        return None


def _dataset_data_storage(dataset):
    """Return dataset storage without triggering PyG's `dataset.data` warning when possible."""
    if "_data" in getattr(dataset, "__dict__", {}):
        return dataset.__dict__.get("_data", None)
    return getattr(dataset, "data", None)


def _subset_total_from_values(dataset, key: str) -> Optional[int]:
    """Aggregate per-graph scalar metadata such as `_num_nodes`."""
    if dataset is None:
        return None

    base_dataset, indices = _resolve_subset_base_and_indices(dataset)
    if base_dataset is None:
        return None

    storage = _dataset_data_storage(base_dataset)
    if storage is None:
        return None

    if isinstance(storage, dict):
        values = storage.get(key)
    else:
        values = getattr(storage, key, None)
    if values is None:
        return None

    try:
        if torch.is_tensor(values):
            if values.dim() == 0:
                return int(values.item()) if indices is None else None
            if indices is None:
                return int(values.sum().item())
            idx = torch.as_tensor(indices, dtype=torch.long)
            if idx.numel() == 0:
                return 0
            return int(values[idx].sum().item())

        if isinstance(values, (list, tuple)):
            if indices is None:
                return int(sum(int(value) for value in values))
            return int(sum(int(values[idx]) for idx in indices))

        scalar = int(values)
        return scalar if indices is None else None
    except Exception:
        return None


def _graph_nodes_edges(loader) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None, None, None

    graphs = _dataset_len(loader)
    nodes = _subset_total_from_slices(dataset, "x")
    if nodes is None:
        nodes = _subset_total_from_values(dataset, "_num_nodes")
    if nodes is None:
        nodes = _subset_total_from_values(dataset, "num_nodes")
    edges = _subset_total_from_slices(dataset, "edge_index")
    if edges is None and not hasattr(dataset, "dataset"):
        total_edges = getattr(dataset, "total_edges", None)
        if total_edges is not None:
            edges = int(total_edges)
    if nodes is not None and edges is not None:
        return nodes, edges, graphs

    # Fallback: materialize only for small datasets.
    if graphs is not None and graphs <= 128:
        total_nodes = 0
        total_edges = 0
        try:
            for item in dataset:
                item_nodes = getattr(item, "num_nodes", None)
                if item_nodes is None and getattr(item, "x", None) is not None:
                    item_nodes = item.x.size(0)
                item_edges = getattr(item, "num_edges", None)
                if item_edges is None and getattr(item, "edge_index", None) is not None:
                    item_edges = item.edge_index.size(1)
                total_nodes += int(item_nodes) if item_nodes is not None else 0
                total_edges += int(item_edges) if item_edges is not None else 0
            return total_nodes, total_edges, graphs
        except Exception:
            pass

    return None, None, graphs


def _split_stats_line(task_level: str, train_loader, val_loader, test_loader) -> str:
    split_meta = (
        ("train", "train", train_loader),
        ("valid", "val", val_loader),
        ("test", "test", test_loader),
    )

    if task_level == "node":
        parts = []
        for display_name, mask_name, loader in split_meta:
            data = getattr(loader, "data", None)
            nodes = _split_mask_count(data, f"{mask_name}_mask")
            if nodes is None:
                nodes = _num_nodes(data)
            edges = _num_edges(data)
            graphs = int(getattr(data, "num_graphs", 1)) if data is not None else _dataset_len(loader)
            parts.append(
                f"{display_name}(nodes={_fmt_count(nodes)},edges={_fmt_count(edges)},graphs={_fmt_count(graphs)})"
            )
        return " ".join(parts)

    if task_level == "edge":
        parts = []
        message_edges = None
        for display_name, _, loader in split_meta:
            data = getattr(loader, "data", None)
            nodes = _num_nodes(data)
            graphs = int(getattr(data, "num_graphs", 1)) if data is not None else _dataset_len(loader)
            if message_edges is None:
                message_edges = _num_edges(data)

            edge_label_index = getattr(data, "edge_label_index", None) if data is not None else None
            labeled_edges = int(edge_label_index.size(1)) if edge_label_index is not None else None

            pos_edges = None
            neg_edges = None
            edge_label = getattr(data, "edge_label", None) if data is not None else None
            if edge_label is not None:
                try:
                    label_tensor = torch.as_tensor(edge_label).view(-1)
                    pos_edges = int((label_tensor > 0.5).sum().item())
                    neg_edges = int((label_tensor <= 0.5).sum().item())
                except Exception:
                    pos_edges = None
                    neg_edges = None

            parts.append(
                f"{display_name}(nodes={_fmt_count(nodes)},edges={_fmt_count(labeled_edges)},"
                f"graphs={_fmt_count(graphs)},pos={_fmt_count(pos_edges)},neg={_fmt_count(neg_edges)})"
            )
        return f"{' '.join(parts)} message_edges={_fmt_count(message_edges)}"

    if task_level == "graph":
        parts = []
        for display_name, _, loader in split_meta:
            nodes, edges, graphs = _graph_nodes_edges(loader)
            avg_nodes = _avg_per_graph(nodes, graphs)
            avg_edges = _avg_per_graph(edges, graphs)
            parts.append(
                f"{display_name}(avg_nodes={_fmt_avg(avg_nodes)},"
                f"avg_edges={_fmt_avg(avg_edges)},graphs={_fmt_count(graphs)})"
            )
        return " ".join(parts)

    return ""



def _filter_unsupported_split_defs(
    dataset_obj,
    dataset_name: str,
    task_level: str,
    split_defs: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:
    """Drop unsupported split definitions for the given dataset/task pair."""
    if not split_defs:
        return split_defs
    split_strategy = resolve_count_split_strategy(dataset_obj, task_level)
    if split_strategy != "unsupported":
        return split_defs

    kept: List[Tuple[float, float, float]] = []
    skipped: List[Tuple[float, float, float]] = []
    for split_def in split_defs:
        if _is_few_shot_split_def(split_def):
            skipped.append(split_def)
        else:
            kept.append(split_def)

    if getattr(dataset_obj, "has_labels", None) is False:
        reason = "unlabeled"
    elif is_regression_dataset(dataset_obj, task_level):
        reason = "single-target regression"
    else:
        reason = "unsupported"

    for split_def in skipped:
        print(
            f"[DataPrep][Split] Skip unsupported integer-first split for {reason} "
            f"dataset={dataset_name} task level={task_level} split={split_def}"
        )
    return kept


def _should_generate_splits_for_dataset(dataset_obj) -> bool:
    """Return False for fully unlabeled datasets that are only prepared for pretraining artifacts."""
    return getattr(dataset_obj, "has_labels", None) is not False


def _generate_task_splits(
    dataset_obj,
    dataset_name: str,
    task_level: str,
    split_defs: List[Tuple[float, float, float]],
    split_seeds: List[int],
    split_root: str,
    batch_size: int,
    num_workers: int,
) -> int:
    if not split_root or not split_defs or not split_seeds:
        return 0

    dataset_split_root = Path(split_root)
    failures = 0
    total_jobs = len(split_defs) * len(split_seeds)
    status_counts: Dict[str, int] = {
        "loaded": 0,
        "saved": 0,
        "builtin": 0,
    }
    for split_def in split_defs:
        for seed in split_seeds:
            split_name = _canonical_split_dataset_name(dataset_name, task_level, int(seed))
            try:
                # Suppress per-file split logs; report one concise summary per task.
                with contextlib.redirect_stdout(io.StringIO()):
                    train_loader, val_loader, test_loader, split_meta = make_loaders(
                        dataset=dataset_obj,
                        dataset_name=split_name,
                        task_level=task_level,
                        batch_size=int(batch_size),
                        num_workers=int(num_workers),
                        split=split_def,
                        seed=int(seed),
                        induced=False,
                        split_root=str(dataset_split_root),
                        return_split_meta=True,
                    )
                status = str(split_meta.get("status") or "builtin") if isinstance(split_meta, dict) else "builtin"
                if status not in status_counts:
                    status = "builtin"
                status_counts[status] += 1
                split_stats = _split_stats_line(
                    task_level=task_level,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                )
                if split_stats:
                    print(
                        f"[DataPrep][Split][Stats] dataset={dataset_name} task level={task_level} "
                        f"split={split_def} seed={seed} status={status} {split_stats}"
                    )
            except Exception as exc:  # pylint: disable=broad-except
                failures += 1
                print(
                    f"[DataPrep][Split] Failed dataset={dataset_name} task level={task_level} "
                    f"split={split_def} seed={seed} root={dataset_split_root}: {exc}"
                )

    completed = total_jobs - failures
    print(
        f"[DataPrep][Split] dataset={dataset_name} task level={task_level} "
        f"splits={len(split_defs)} seeds={len(split_seeds)} completed={completed} "
        f"loaded={status_counts['loaded']} saved={status_counts['saved']} "
        f"builtin={status_counts['builtin']} failed={failures} root={dataset_split_root}"
    )
    return failures


def read_datasets(tsv_path: str) -> List[str]:
    names: List[str] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if not parts:
                continue
            names.append(parts[0])
    return names


def cleanup_artifacts(dataset: str, root: Path) -> None:
    """Remove stale zip/raw/processed files that can cause zip parsing errors."""
    names = {dataset}
    # Common OGB naming: ogbn-arxiv -> arxiv, ogbg-ppa -> ppa
    if dataset.startswith("ogbn-") or dataset.startswith("ogbg-"):
        names.add(dataset.split("-", 1)[1])
    # LRGB naming: coco-sp -> cocosp, pascalvoc-sp -> pascalvocsp
    names.add(dataset.replace("-", ""))

    candidates = []
    for name in names:
        candidates.extend(
            [
                root / name,
                root / name / "raw",
                root / name / "processed",
                root / f"{name}.zip",
            ]
        )
    for path in candidates:
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass
        elif path.is_dir():
            try:
                shutil.rmtree(path)
            except OSError:
                pass


def already_processed(dataset: str, root: Path) -> str | None:
    """Return a processed path if dataset appears already prepared, else None."""
    names = {dataset}
    if dataset.startswith("ogbn-") or dataset.startswith("ogbg-"):
        names.add(dataset.split("-", 1)[1])
    names.add(dataset.replace("-", ""))

    candidates = []
    for name in names:
        candidates.extend(
            [
                root / name / "processed",
            ]
        )
    for path in candidates:
        if path.exists() and any(path.iterdir()):
            return str(path)
    return None


def _is_zip_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "zip" in msg or "file is not a zip file" in msg


def _run_with_zip_retry(dataset: str, root_path: Path, process_fn: Callable[[], bool]) -> bool:
    """Retry once after cleaning stale artifacts when dataset archives are corrupted."""
    for attempt in (1, 2):
        try:
            return process_fn()
        except Exception as exc:
            if attempt == 1 and _is_zip_error(exc):
                print(f"[CLEAN] removing stale files for {dataset} due to zip error")
                cleanup_artifacts(dataset, root_path)
                continue
            raise
    return False


def _compute_subgraph_svd_with_fallback(
    dataset_obj,
    dataset_name: str,
    task_level: str,
    feat_dim: int,
    struct_dim: int,
    matrix_type: str,
    output_dir: str,
) -> None:
    """Compute subgraph SVD features and fallback to local default path on permission errors."""
    try:
        compute_subgraph_svd_features(
            dataset_obj,
            dataset_name=dataset_name,
            task_level=task_level,
            feat_dim=feat_dim,
            struct_dim=struct_dim,
            matrix_type=matrix_type,
            output_dir=output_dir,
            overwrite=False,
        )
    except PermissionError:
        fallback_dir = str(_dataset_scoped_dir("data/subgraph_svd", dataset_name))
        print(
            f"[SubgraphSVD] Permission denied for {output_dir}; retrying with {fallback_dir}."
        )
        compute_subgraph_svd_features(
            dataset_obj,
            dataset_name=dataset_name,
            task_level=task_level,
            feat_dim=feat_dim,
            struct_dim=struct_dim,
            matrix_type=matrix_type,
            output_dir=fallback_dir,
            overwrite=False,
        )


def _log_node_dataset_overview(dataset: str, dataset_obj) -> None:
    """Print node dataset overview."""
    dataset_info(
        dataset=dataset_obj,
        task_level="node",
        name=dataset,
        induced=hasattr(dataset_obj, "graphs"),
    )


def _log_graph_dataset_overview(dataset: str, dataset_obj) -> None:
    """Print graph dataset overview."""
    graph_filter_meta = getattr(dataset_obj, "graph_filter_meta", None)
    if isinstance(graph_filter_meta, dict):
        print(
            f"[DataPrep][GraphFilter] dataset={dataset} "
            f"raw_total={int(graph_filter_meta.get('raw_total', len(dataset_obj)))} "
            f"filtered_total={int(graph_filter_meta.get('filtered_total', len(dataset_obj)))} "
            f"drop_count={int(graph_filter_meta.get('drop_count', 0))}"
        )
    dataset_info(
        dataset=dataset_obj,
        task_level="graph",
        name=dataset,
        induced=False,
    )


def _log_edge_dataset_overview(dataset: str, dataset_obj) -> None:
    """Print edge dataset overview."""
    dataset_info(
        dataset=dataset_obj,
        task_level="edge",
        name=dataset,
        induced=False,
    )


def _process_edge_task(
    dataset: str,
    root: str,
    split_defs: List[Tuple[float, float, float]],
    split_seeds: List[int],
    *,
    dataset_split_root: str,
    dataset_feature_svd_dir: str,
    dataset_induced_root: str,
    dataset_subgraph_svd_dir: str,
    split_batch_size: int,
    split_num_workers: int,
    induced: bool,
    induced_min_size: int,
    induced_max_size: int,
    induced_max_hops: int,
    edge_max_size: int | None,
    feat_reduction: bool,
    feat_reduction_dim: int,
    subgraph_svd: bool,
    subgraph_svd_feat_dim: int,
    subgraph_svd_struct_dim: int,
    subgraph_svd_matrix: str,
) -> bool:
    """Prepare edge-level artifacts: fixed splits, induced edge subgraphs, and optional subgraph SVD."""
    if not split_defs:
        raise ValueError("No edge split definitions found in cfg.data_preparation.edge_task_splits.")

    split_failures = 0
    induced_failures = 0
    primary_seed = int(split_seeds[0]) if split_seeds else 42

    # Step 1: materialize all edge splits on non-induced data.
    split_dataset_obj = create_dataset(
        name=dataset,
        root=root,
        task_level="edge",
        induced=False,
        induced_min_size=induced_min_size,
        induced_max_size=induced_max_size,
        induced_max_hops=induced_max_hops,
        edge_max_size=edge_max_size,
        induced_root=dataset_induced_root,
        split_root=dataset_split_root,
        split=None,
        seed=primary_seed,
        feat_reduction=feat_reduction,
        feat_reduction_dim=feat_reduction_dim,
        feature_svd_dir=dataset_feature_svd_dir,
    )
    _log_edge_dataset_overview(dataset=dataset, dataset_obj=split_dataset_obj)
    split_failures = _generate_task_splits(
        dataset_obj=split_dataset_obj,
        dataset_name=dataset,
        task_level="edge",
        split_defs=split_defs,
        split_seeds=split_seeds,
        split_root=dataset_split_root,
        batch_size=split_batch_size,
        num_workers=split_num_workers,
    )

    # Step 2: for each split+seed, generate induced edge subgraphs and split-specific SVD.
    if induced:
        total_jobs = len(split_defs) * len(split_seeds)
        for split_def in split_defs:
            for split_seed in split_seeds:
                try:
                    edge_dataset = create_dataset(
                        name=dataset,
                        root=root,
                        task_level="edge",
                        induced=True,
                        induced_min_size=induced_min_size,
                        induced_max_size=induced_max_size,
                        induced_max_hops=induced_max_hops,
                        edge_max_size=edge_max_size,
                        induced_root=dataset_induced_root,
                        split_root=dataset_split_root,
                        split=split_def,
                        seed=int(split_seed),
                        feat_reduction=feat_reduction,
                        feat_reduction_dim=feat_reduction_dim,
                        feature_svd_dir=dataset_feature_svd_dir,
                    )
                    if subgraph_svd:
                        _compute_subgraph_svd_with_fallback(
                            edge_dataset,
                            dataset_name=dataset,
                            task_level="edge",
                            feat_dim=subgraph_svd_feat_dim,
                            struct_dim=subgraph_svd_struct_dim,
                            matrix_type=subgraph_svd_matrix,
                            output_dir=dataset_subgraph_svd_dir,
                        )
                except Exception as exc:  # pylint: disable=broad-except
                    induced_failures += 1
                    print(
                        f"[DataPrep][EdgeInduced] Failed dataset={dataset} "
                        f"split={split_def} seed={split_seed}: {exc}"
                    )
        generated = total_jobs - induced_failures
        print(
            f"[DataPrep][EdgeInduced] dataset={dataset} "
            f"splits={len(split_defs)} seeds={len(split_seeds)} "
            f"generated={generated} failed={induced_failures} root={dataset_induced_root}"
        )

    _print_feature_svd_info(
        dataset_name=dataset,
        task_level="edge",
        feat_reduction=feat_reduction,
        feat_reduction_dim=feat_reduction_dim,
        feature_svd_dir=dataset_feature_svd_dir,
    )
    return (split_failures == 0) and (induced_failures == 0)


def _process_non_edge_task(
    dataset: str,
    task_level: str,
    root: str,
    split_defs: List[Tuple[float, float, float]],
    split_seeds: List[int],
    *,
    dataset_split_root: str,
    dataset_feature_svd_dir: str,
    dataset_induced_root: str,
    dataset_subgraph_svd_dir: str,
    graph_filter_dir: str,
    split_batch_size: int,
    split_num_workers: int,
    induced: bool,
    induced_min_size: int,
    induced_max_size: int,
    induced_max_hops: int,
    feat_reduction: bool,
    feat_reduction_dim: int,
    subgraph_svd: bool,
    subgraph_svd_feat_dim: int,
    subgraph_svd_struct_dim: int,
    subgraph_svd_matrix: str,
) -> bool:
    """
    Prepare node/graph artifacts and optionally generate fixed splits.

    Node tasks may use induced subgraphs during preparation; split generation is done
    on the non-induced view to keep split semantics consistent.
    """
    split_failures = 0
    primary_seed = int(split_seeds[0]) if split_seeds else 42
    effective_split_defs = list(split_defs)
    split_for_dataset = effective_split_defs[0] if effective_split_defs else None
    requested_task_uses_induced = induced and task_level == "node"
    effective_task_uses_induced = requested_task_uses_induced
    base_dataset_obj = None
    dataset_obj = create_dataset(
        name=dataset,
        root=root,
        task_level=task_level,
        induced=effective_task_uses_induced,
        induced_min_size=induced_min_size,
        induced_max_size=induced_max_size,
        induced_max_hops=induced_max_hops,
        induced_root=dataset_induced_root,
        split_root=dataset_split_root,
        split=split_for_dataset,
        seed=primary_seed,
        feat_reduction=feat_reduction,
        feat_reduction_dim=feat_reduction_dim,
        feature_svd_dir=dataset_feature_svd_dir,
        graph_filter_dir=graph_filter_dir,
    )
    if task_level == "node":
        _log_node_dataset_overview(dataset=dataset, dataset_obj=dataset_obj)
    elif task_level == "graph":
        _log_graph_dataset_overview(dataset=dataset, dataset_obj=dataset_obj)

    if _should_generate_splits_for_dataset(dataset_obj):
        # Regression tasks do not support few-shot split definitions.
        effective_split_defs = _filter_unsupported_split_defs(
            dataset_obj=dataset_obj,
            dataset_name=dataset,
            task_level=task_level,
            split_defs=effective_split_defs,
        )
    else:
        effective_split_defs = []

    if effective_split_defs:
        split_dataset_obj = dataset_obj
        if effective_task_uses_induced:
            # Split indices should be produced from the base (non-induced) graph.
            if base_dataset_obj is not None:
                split_dataset_obj = base_dataset_obj
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    split_dataset_obj = create_dataset(
                        name=dataset,
                        root=root,
                        task_level=task_level,
                        induced=False,
                        induced_min_size=induced_min_size,
                        induced_max_size=induced_max_size,
                        induced_max_hops=induced_max_hops,
                        induced_root=dataset_induced_root,
                        split_root=dataset_split_root,
                        split=None,
                        seed=primary_seed,
                        feat_reduction=feat_reduction,
                        feat_reduction_dim=feat_reduction_dim,
                        feature_svd_dir=dataset_feature_svd_dir,
                        graph_filter_dir=graph_filter_dir,
                    )
        split_failures = _generate_task_splits(
            dataset_obj=split_dataset_obj,
            dataset_name=dataset,
            task_level=task_level,
            split_defs=effective_split_defs,
            split_seeds=split_seeds,
            split_root=dataset_split_root,
            batch_size=split_batch_size,
            num_workers=split_num_workers,
        )

    _print_feature_svd_info(
        dataset_name=dataset,
        task_level=task_level,
        feat_reduction=feat_reduction,
        feat_reduction_dim=feat_reduction_dim,
        feature_svd_dir=dataset_feature_svd_dir,
    )
    should_run_subgraph_svd = subgraph_svd and (effective_task_uses_induced or task_level == "graph")
    if should_run_subgraph_svd:
        _compute_subgraph_svd_with_fallback(
            dataset_obj,
            dataset_name=dataset,
            task_level=task_level,
            feat_dim=subgraph_svd_feat_dim,
            struct_dim=subgraph_svd_struct_dim,
            matrix_type=subgraph_svd_matrix,
            output_dir=dataset_subgraph_svd_dir,
        )

    return split_failures == 0


def try_load(
    dataset: str,
    root: str,
    task_levels: Iterable[str] = ("node", "graph"),
    feat_reduction: bool = True,
    feat_reduction_dim: int = 100,
    feature_svd_dir: str = "data/feature_svd",
    induced: bool = False,
    induced_min_size: int = 10,
    induced_max_size: int = 30,
    induced_max_hops: int = 5,
    edge_max_size: int | None = 60,
    edge_level_max_num_nodes: int | None = None,
    induced_root: str = "",
    graph_filter_dir: str = "data/filters",
    subgraph_svd: bool = False,
    subgraph_svd_feat_dim: int = 100,
    subgraph_svd_struct_dim: int = 100,
    subgraph_svd_matrix: str = "adjacency",
    subgraph_svd_dir: str = "data/subgraph_svd",
    split_root: str = "",
    split_defs_by_task: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
    split_seeds: Optional[List[int]] = None,
    split_batch_size: int = 64,
    split_num_workers: int = 0,
) -> bool:
    """Prepare one dataset for all requested task levels."""
    root_path = Path(root)
    normalized_task_levels = tuple(str(level) for level in task_levels)
    dataset_split_root = str(_dataset_scoped_dir(split_root, dataset))
    dataset_feature_svd_dir = str(_dataset_scoped_dir(feature_svd_dir, dataset))
    dataset_induced_root = str(_dataset_scoped_dir(induced_root, dataset)) if induced_root else ""
    dataset_subgraph_svd_dir = str(_dataset_scoped_dir(subgraph_svd_dir, dataset))
    processed_path = already_processed(dataset, root_path)
    if processed_path:
        print(f"[DataPrep] Reusing cached processed data at {processed_path}")

    split_defs_map = split_defs_by_task or {}
    normalized_split_seeds = [int(seed) for seed in (split_seeds or [])]
    if not normalized_split_seeds:
        normalized_split_seeds = [42]
    skip_edge_task = (
        "edge" in normalized_task_levels
        and infer_task_level(dataset) == "node"
        and not _should_prepare_edge_level_for_node_dataset(
            dataset,
            root,
            node_limit=edge_level_max_num_nodes,
        )
    )

    all_ok = True
    for task_level in normalized_task_levels:
        if task_level == "edge" and skip_edge_task:
            continue
        print(TASK_SEPARATOR)
        print(f"[DataPrep] Processing {dataset} (task level={task_level})...")
        print(TASK_SEPARATOR)
        task_split_defs = list(split_defs_map.get(task_level, []))
        try:
            if task_level == "edge":
                task_ok = _run_with_zip_retry(
                    dataset,
                    root_path,
                    lambda: _process_edge_task(
                        dataset=dataset,
                        root=root,
                        split_defs=task_split_defs,
                        split_seeds=normalized_split_seeds,
                        dataset_split_root=dataset_split_root,
                        dataset_feature_svd_dir=dataset_feature_svd_dir,
                        dataset_induced_root=dataset_induced_root,
                        dataset_subgraph_svd_dir=dataset_subgraph_svd_dir,
                        split_batch_size=split_batch_size,
                        split_num_workers=split_num_workers,
                        induced=induced,
                        induced_min_size=induced_min_size,
                        induced_max_size=induced_max_size,
                        induced_max_hops=induced_max_hops,
                        edge_max_size=edge_max_size,
                        feat_reduction=feat_reduction,
                        feat_reduction_dim=feat_reduction_dim,
                        subgraph_svd=subgraph_svd,
                        subgraph_svd_feat_dim=subgraph_svd_feat_dim,
                        subgraph_svd_struct_dim=subgraph_svd_struct_dim,
                        subgraph_svd_matrix=subgraph_svd_matrix,
                    ),
                )
            else:
                task_ok = _run_with_zip_retry(
                    dataset,
                    root_path,
                    lambda: _process_non_edge_task(
                        dataset=dataset,
                        task_level=task_level,
                        root=root,
                        split_defs=task_split_defs,
                        split_seeds=normalized_split_seeds,
                        dataset_split_root=dataset_split_root,
                        dataset_feature_svd_dir=dataset_feature_svd_dir,
                        dataset_induced_root=dataset_induced_root,
                        dataset_subgraph_svd_dir=dataset_subgraph_svd_dir,
                        graph_filter_dir=graph_filter_dir,
                        split_batch_size=split_batch_size,
                        split_num_workers=split_num_workers,
                        induced=induced,
                        induced_min_size=induced_min_size,
                        induced_max_size=induced_max_size,
                        induced_max_hops=induced_max_hops,
                        feat_reduction=feat_reduction,
                        feat_reduction_dim=feat_reduction_dim,
                        subgraph_svd=subgraph_svd,
                        subgraph_svd_feat_dim=subgraph_svd_feat_dim,
                        subgraph_svd_struct_dim=subgraph_svd_struct_dim,
                        subgraph_svd_matrix=subgraph_svd_matrix,
                    ),
                )
        except Exception as exc:  # pylint: disable=broad-except
            all_ok = False
            print(f"[FAIL] {dataset} (task level={task_level}): {exc}")
            continue

        if not task_ok:
            all_ok = False
            print(f"[FAIL] {dataset} (task level={task_level}): preparation completed with failures.")
            continue
        print(f"[OK] {dataset} (task level={task_level})")
    return all_ok


def prepare_datasets(
    tsv_path: str | Iterable[str],
    root: Path | str,
    task_levels: Iterable[str] = ("node", "graph"),
    feat_reduction: bool = True,
    feat_reduction_dim: int = 100,
    feature_svd_dir: str = "data/feature_svd",
    induced: bool = False,
    induced_min_size: int = 10,
    induced_max_size: int = 30,
    induced_max_hops: int = 5,
    edge_max_size: int | None = 60,
    edge_level_max_num_nodes: int | None = None,
    induced_root: str = "",
    graph_filter_dir: str = "data/filters",
    subgraph_svd: bool = False,
    subgraph_svd_feat_dim: int = 100,
    subgraph_svd_struct_dim: int = 100,
    subgraph_svd_matrix: str = "adjacency",
    subgraph_svd_dir: str = "data/subgraph_svd",
    use_infer: bool = True,
    split_root: str = "",
    generate_edge_level: bool = True,
    split_defs_by_task: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
    split_seeds: Optional[List[int]] = None,
    split_batch_size: int = 64,
    split_num_workers: int = 0,
) -> int:
    """Prepare all requested datasets and return a process exit code."""
    if isinstance(tsv_path, (list, tuple, set)):
        datasets = [str(name) for name in tsv_path]
    else:
        datasets = read_datasets(str(tsv_path))
    if not datasets:
        print(f"[INFO] No datasets found in {tsv_path}")
        return 0

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    success = 0
    for idx, name in enumerate(datasets, 1):
        print(DATASET_SEPARATOR)
        print(f"[{idx}/{len(datasets)}] Preparing {name}")
        print(DATASET_SEPARATOR)
        inferred = infer_task_level(name) if use_infer else None
        if inferred == "node" and generate_edge_level:
            levels = ("node", "edge")
        else:
            levels = (inferred,) if inferred else task_levels
        if try_load(
            name,
            root=str(root),
            task_levels=levels,
            feat_reduction=feat_reduction,
            feat_reduction_dim=feat_reduction_dim,
            feature_svd_dir=feature_svd_dir,
            induced=induced,
            induced_min_size=induced_min_size,
            induced_max_size=induced_max_size,
            induced_max_hops=induced_max_hops,
            edge_max_size=edge_max_size,
            edge_level_max_num_nodes=edge_level_max_num_nodes,
            induced_root=induced_root,
            graph_filter_dir=graph_filter_dir,
            subgraph_svd=subgraph_svd,
            subgraph_svd_feat_dim=subgraph_svd_feat_dim,
            subgraph_svd_struct_dim=subgraph_svd_struct_dim,
            subgraph_svd_matrix=subgraph_svd_matrix,
            subgraph_svd_dir=subgraph_svd_dir,
            split_root=split_root,
            split_defs_by_task=split_defs_by_task,
            split_seeds=split_seeds,
            split_batch_size=split_batch_size,
            split_num_workers=split_num_workers,
        ):
            success += 1
    print(f"[SUMMARY] success={success}, fail={len(datasets)-success}, total={len(datasets)}")
    return 0 if success == len(datasets) else 1
