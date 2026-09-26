"""Metadata helpers for PyG datasets used throughout the data-loading stack.

This module infers task properties such as regression vs. classification,
resolves split-strategy hints from label structure, derives lightweight
summaries for node, edge, and graph tasks, and reports split instance counts
for logging.
"""

from typing import Dict, Optional, Tuple
import warnings

import torch

from src.utils import format_split_for_name

from .dataset_domains import CLASS_TO_DOMAIN, KEYWORD_DOMAINS, NAME_TO_DOMAIN
from .dataset_storage import _get_dataset_data_storage


def _labels_indicate_regression(labels: torch.Tensor) -> bool:
    """Best-effort regression detection from labels."""
    label_tensor = torch.as_tensor(labels).view(-1)
    if label_tensor.numel() == 0:
        return False

    if label_tensor.dtype.is_floating_point:
        finite = label_tensor[torch.isfinite(label_tensor)]
        if finite.numel() == 0:
            return False
        rounded = finite.round()
        if not torch.allclose(finite, rounded, atol=1e-6):
            return True

        if int(torch.unique(finite).numel()) > 32:
            return True

    return False


def _safe_dataset_num_classes(dataset) -> int | None:
    """Read dataset.num_classes without surfacing PyG's float-label warning."""
    raw_num_classes = getattr(dataset, "__dict__", {}).get("num_classes", None)
    try:
        if raw_num_classes is not None:
            return int(raw_num_classes)
    except Exception:
        pass

    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings(
            "always",
            message="Found floating-point labels while calling `dataset.num_classes`.*",
            category=UserWarning,
        )
        try:
            num_classes = getattr(dataset, "num_classes", None)
        except Exception:
            return None

    for caught_warning in caught:
        if "Found floating-point labels while calling `dataset.num_classes`" in str(caught_warning.message):
            return None

    try:
        return int(num_classes) if num_classes is not None else None
    except Exception:
        return None


def _extract_task_labels(dataset, task_level: str, max_graphs: int = 4096) -> torch.Tensor | None:
    """Best-effort label extraction for node/graph datasets."""
    if getattr(dataset, "has_labels", None) is False:
        return None

    level = str(task_level).lower()
    if level == "node":
        try:
            data = dataset[0]
        except Exception:
            return None
        labels = getattr(data, "y", None)
        return torch.as_tensor(labels) if labels is not None else None

    if level != "graph":
        return None

    data_obj = _get_dataset_data_storage(dataset)
    labels = getattr(data_obj, "y", None) if data_obj is not None else None
    if labels is not None:
        labels_tensor = torch.as_tensor(labels)
        try:
            if labels_tensor.dim() == 0 or labels_tensor.size(0) == len(dataset):
                return labels_tensor
        except Exception:
            if labels_tensor.dim() == 0:
                return labels_tensor

    collected = []
    target_dim = None
    try:
        sample_count = min(int(len(dataset)), int(max_graphs))
        for idx in range(sample_count):
            item = dataset[idx]
            y = getattr(item, "y", None)
            if y is None:
                continue
            y_tensor = torch.as_tensor(y).detach().cpu().view(-1)
            if y_tensor.numel() == 0:
                continue
            if target_dim is None:
                target_dim = int(y_tensor.numel())
            if int(y_tensor.numel()) != target_dim:
                return torch.cat(collected + [y_tensor], dim=0) if collected else y_tensor
            collected.append(y_tensor)
    except Exception:
        return None

    if not collected:
        return None
    return torch.stack(collected, dim=0)


def _label_target_dim(labels: torch.Tensor | None) -> int:
    """Return the number of target values per node/graph item."""
    if labels is None:
        return 0
    label_tensor = torch.as_tensor(labels)
    if label_tensor.numel() == 0:
        return 0
    if label_tensor.dim() <= 1:
        return 1

    dim = 1
    for size in label_tensor.shape[1:]:
        dim *= int(size)
    return dim


def is_regression_dataset(dataset, task_level: str) -> bool:
    """Infer whether the current task uses regression labels."""
    level = str(task_level).lower()
    if level == "edge":
        return False
    if getattr(dataset, "has_labels", None) is False:
        return False

    labels = _extract_task_labels(dataset, level)
    if labels is not None:
        label_tensor = torch.as_tensor(labels)
        if label_tensor.dtype.is_floating_point:
            finite = label_tensor[torch.isfinite(label_tensor)]
            if finite.numel() > 0 and not torch.allclose(finite, finite.round(), atol=1e-6):
                return True

    num_classes = _safe_dataset_num_classes(dataset)
    if num_classes is not None and num_classes > 0:
        return False

    if labels is not None:
        return _labels_indicate_regression(labels)

    return False


def resolve_count_split_strategy(dataset, task_level: str) -> str:
    """
    Decide how integer-first split definitions should be interpreted.

    Returns:
      - "balanced": single-label classification few-shot
      - "random": multi-target labels or regression -> random train-count split
      - "unsupported": unlabeled tasks
    """
    level = str(task_level).lower()
    if level == "edge":
        return "unsupported"
    if getattr(dataset, "has_labels", None) is False:
        return "unsupported"

    labels = _extract_task_labels(dataset, level)
    if labels is None or torch.as_tensor(labels).numel() == 0:
        return "unsupported"
    if _label_target_dim(labels) > 1:
        return "random"
    if is_regression_dataset(dataset, level):
        return "random"
    return "balanced"


def get_basic_dataset_info(dataset) -> dict:
    """
    Extract minimal context about a PyTorch Geometric dataset.
    Returns a dictionary with:
        - name: resolved dataset name
        - domain: estimated domain
        - source: dataset class path
    """
    cls = dataset.__class__
    cls_name = cls.__name__
    base_info = getattr(dataset, "base_dataset_info", None)
    dataset_name = getattr(dataset, "name", None) or cls_name
    if base_info and base_info.get("name"):
        dataset_name = base_info["name"]

    name_key = str(dataset_name).lower()
    dataset_domain = base_info.get("domain") if base_info else None
    if dataset_domain is None:
        dataset_domain = CLASS_TO_DOMAIN.get(cls_name)
    if dataset_domain is None:
        dataset_domain = NAME_TO_DOMAIN.get(name_key)
    if dataset_domain is None:
        combined = f"{cls_name.lower()} {name_key}"
        for domain, keywords in KEYWORD_DOMAINS:
            if any(token in combined for token in keywords):
                dataset_domain = domain
                break
    dataset_domain = dataset_domain or "unknown"

    dataset_source = f"{cls.__module__}.{cls_name}"

    return {
        # Without the name, generic source classes (Planetoid, TUDataset)
        # leave LLM prompts saying "Dataset name: None".
        "name": str(dataset_name),
        "domain": dataset_domain,
        "source": dataset_source,
    }


def dataset_info(
    dataset,
    task_level: str,
    name: str = "",
    induced: bool = False,
) -> Dict:
    """Get meta information about the dataset."""

    def _task_type_label() -> str:
        if getattr(dataset, "has_labels", None) is False:
            return "none"
        return "regression" if is_regression_dataset(dataset, task_level) else "classification"

    def _log_info(info: Dict) -> None:
        ordered = [f"task level={task_level}", f"task type={_task_type_label()}"]
        if induced:
            ordered.append("induced=True")
        for key in (
            "num_node_features",
            "num_classes",
            "num_nodes",
            "num_edges",
            "num_graphs",
            "avg_nodes_per_graph",
            "avg_edges_per_graph",
        ):
            if key in info:
                ordered.append(f"{key}={info[key]}")
        summary = ", ".join(ordered)
        prefix = f"dataset={name}, " if name else ""
        print(f"[Dataset info] {prefix}{summary}")

    if task_level == "node":
        data = dataset[0]
        if induced and not hasattr(dataset, "graphs"):
            raise RuntimeError("Induced node dataset missing graphs; generation failed.")
        num_classes = None
        label_dim = None
        if hasattr(dataset, "num_classes") and dataset.num_classes is not None:
            num_classes = dataset.num_classes
        elif getattr(data, "y", None) is not None and data.y.numel() > 0:
            num_classes = int(data.y.max().item() + 1)
        if getattr(data, "y", None) is not None:
            y = torch.as_tensor(data.y)
            if y.dim() <= 1:
                label_dim = 1
            else:
                label_dim = int(y.size(-1))

        info = {
            "num_node_features": data.num_node_features,
            "num_classes": num_classes,
            "label_dim": label_dim,
            "task_type": _task_type_label(),
            "num_nodes": dataset.base_num_nodes if induced and getattr(dataset, "base_num_nodes", None) is not None else (len(dataset) if induced else data.num_nodes),
            "num_edges": dataset.base_num_edges if induced and getattr(dataset, "base_num_edges", None) is not None else data.num_edges,
        }
        if induced:
            info["num_graphs"] = len(dataset)
            info["avg_nodes_per_graph"] = round(float(sum(g.num_nodes for g in dataset.graphs) / len(dataset)), 2)
            info["avg_edges_per_graph"] = round(float(sum(g.num_edges for g in dataset.graphs) / len(dataset)), 2)
        _log_info(info)
        return info

    if task_level == "edge":
        data = dataset[0]
        if induced and not hasattr(dataset, "graphs"):
            raise RuntimeError("Induced edge dataset missing graphs; generation failed.")
        num_classes = None
        label_dim = None
        if hasattr(dataset, "num_classes") and dataset.num_classes is not None:
            num_classes = dataset.num_classes
        elif getattr(data, "y", None) is not None and data.y.numel() > 0:
            num_classes = int(data.y.max().item() + 1)
        if getattr(data, "y", None) is not None:
            y = torch.as_tensor(data.y)
            if y.dim() <= 1:
                label_dim = 1
            else:
                label_dim = int(y.size(-1))
        info = {
            "num_node_features": data.num_node_features,
            "num_classes": num_classes,
            "label_dim": label_dim,
            "task_type": _task_type_label(),
            "num_nodes": dataset.base_num_nodes if induced and getattr(dataset, "base_num_nodes", None) is not None else (len(dataset) if induced else data.num_nodes),
            "num_edges": dataset.base_num_edges if induced and getattr(dataset, "base_num_edges", None) is not None else data.num_edges,
        }
        if induced and hasattr(dataset, "graphs"):
            info["num_graphs"] = len(dataset)
            info["avg_nodes_per_graph"] = round(float(sum(g.num_nodes for g in dataset.graphs) / len(dataset)), 2)
            info["avg_edges_per_graph"] = round(float(sum(g.num_edges for g in dataset.graphs) / len(dataset)), 2)
        _log_info(info)
        return info

    if task_level == "graph":
        sample = dataset[0]
        sample_y = getattr(sample, "y", None)
        if sample_y is None:
            label_dim = None
        else:
            y = torch.as_tensor(sample_y)
            if y.dim() <= 1:
                label_dim = 1
            else:
                label_dim = int(y.size(-1))
        num_graphs = int(len(dataset))
        avg_nodes = getattr(dataset, "avg_nodes_per_graph", None)
        avg_edges = getattr(dataset, "avg_edges_per_graph", None)
        if avg_nodes is None or avg_edges is None:
            total_nodes = getattr(dataset, "total_nodes", None)
            total_edges = getattr(dataset, "total_edges", None)
            if total_nodes is not None and total_edges is not None and num_graphs > 0:
                avg_nodes = float(total_nodes) / float(num_graphs)
                avg_edges = float(total_edges) / float(num_graphs)
        if avg_nodes is None or avg_edges is None:
            avg_nodes = float(sum(g.num_nodes for g in dataset) / len(dataset))
            avg_edges = float(sum(g.num_edges for g in dataset) / len(dataset))
        info = {
            "num_node_features": sample.num_features,
            "num_classes": _safe_dataset_num_classes(dataset),
            "label_dim": label_dim,
            "task_type": _task_type_label(),
            "num_graphs": num_graphs,
            "avg_nodes_per_graph": round(float(avg_nodes), 2),
            "avg_edges_per_graph": round(float(avg_edges), 2),
        }
        _log_info(info)
        return info

    raise ValueError(f"Unsupported task_level: {task_level}")


def split_instance_counts(train_loader, val_loader, test_loader, task_level: str) -> Dict[str, Optional[int]]:
    """Best-effort counts of items per split based on available loader metadata."""

    def _mask_count(data, split_name: str):
        mask = getattr(data, f"{split_name}_mask", None)
        if mask is None:
            return None
        try:
            return int(torch.as_tensor(mask).sum().item())
        except Exception:
            return None

    def _count(loader, split_name: str):
        if loader is None:
            return None
        data = getattr(loader, "data", None)
        if data is not None:
            if task_level == "node":
                mask_total = _mask_count(data, split_name)
                if mask_total is not None:
                    return mask_total
                num_nodes = getattr(data, "num_nodes", None)
                if num_nodes is None and getattr(data, "x", None) is not None:
                    num_nodes = data.x.size(0)
                return int(num_nodes) if num_nodes is not None else None
            if task_level == "edge":
                mask_total = _mask_count(data, split_name)
                if mask_total is not None:
                    return mask_total
                edge_label_index = getattr(data, "edge_label_index", None)
                if edge_label_index is not None:
                    return int(edge_label_index.size(1))
                edge_index = getattr(data, "edge_index", None)
                if edge_index is not None:
                    return int(edge_index.size(1))
        dataset = getattr(loader, "dataset", None)
        if dataset is not None:
            try:
                return len(dataset)
            except Exception:
                return None
        return None

    return {
        split_name: _count(loader, split_name)
        for split_name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader))
    }


def log_split_instance_counts(
    train_loader,
    val_loader,
    test_loader,
    task_level: str,
    split: Optional[Tuple[float, float, float]] = None,
    induced: bool = False,
    prefix: str = "[Dataset split]",
) -> Dict[str, Optional[int]]:
    """Print a concise summary of split instance counts."""
    counts = split_instance_counts(train_loader, val_loader, test_loader, task_level=task_level)
    parts = []
    for key in ("train", "val", "test"):
        value = counts.get(key)
        parts.append(f"{key}={value if value is not None else '?'}")
    split_label = ""
    if split is not None:
        formatted = format_split_for_name(split)
        split_label = f", split={formatted or split}"
    induced_label = ", induced" if induced else ""
    print(f"{prefix} (task level={task_level}{induced_label}{split_label}): " + ", ".join(parts))
    return counts
