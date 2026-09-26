"""Dataset summary generation for configured or explicitly selected datasets."""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from src.config import cfg as base_cfg, update_cfg
from src.utils.save_results import exclusive_table_lock
from .dataset_prepare import read_datasets
from .datasets import create_dataset, is_regression_dataset


@dataclass(frozen=True)
class DatasetSummaryRow:
    """One summary row written to `data_summary.tsv`."""

    name: str
    dataset_type: str
    num_graphs: int
    num_nodes: int
    avg_nodes: float
    num_edges: int
    avg_edges: float
    num_features: int
    task_type: str
    num_labels: int


def _resolve_list_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    typo_fix = Path(str(path).replace("avaulable", "available"))
    if typo_fix.exists():
        return typo_fix

    raise FileNotFoundError(f"Dataset list file not found: {path_str}")


def _sum_from_slices(dataset, key: str) -> Optional[int]:
    """Fast aggregate using InMemoryDataset slices when available."""
    slices = getattr(dataset, "slices", None)
    if slices is None or key not in slices:
        return None

    boundaries = slices[key]
    try:
        if torch.is_tensor(boundaries):
            boundaries = boundaries.view(-1)
            if boundaries.numel() < 2:
                return 0
            span = boundaries[1:] - boundaries[:-1]
            return int(span.sum().item())

        if len(boundaries) < 2:
            return 0
        total = 0
        for idx in range(len(boundaries) - 1):
            total += int(boundaries[idx + 1] - boundaries[idx])
        return total
    except Exception:
        return None


def _compute_graph_totals(dataset) -> tuple[int, int, int]:
    num_graphs = int(len(dataset))
    cached_nodes = getattr(dataset, "total_nodes", None)
    cached_edges = getattr(dataset, "total_edges", None)
    if cached_nodes is not None and cached_edges is not None:
        return num_graphs, int(cached_nodes), int(cached_edges)

    total_nodes = _sum_from_slices(dataset, "x")
    total_edges = _sum_from_slices(dataset, "edge_index")

    if total_nodes is not None and total_edges is not None:
        return num_graphs, int(total_nodes), int(total_edges)

    # Fallback for datasets without slices.
    nodes = 0
    edges = 0
    for graph in dataset:
        graph_nodes = getattr(graph, "num_nodes", None)
        if graph_nodes is None and getattr(graph, "x", None) is not None:
            graph_nodes = graph.x.size(0)
        graph_edges = getattr(graph, "num_edges", None)
        if graph_edges is None and getattr(graph, "edge_index", None) is not None:
            graph_edges = graph.edge_index.size(1)
        nodes += int(graph_nodes or 0)
        edges += int(graph_edges or 0)
    return num_graphs, nodes, edges


def _extract_labels(dataset, task_level: str) -> Optional[torch.Tensor]:
    try:
        if getattr(dataset, "has_labels", None) is False:
            return None
        if task_level == "node":
            data = dataset[0]
            labels = getattr(data, "y", None)
            if labels is None:
                return None
            return torch.as_tensor(labels)

        data_obj = getattr(dataset, "data", None)
        labels = getattr(data_obj, "y", None) if data_obj is not None else None
        if labels is not None:
            return torch.as_tensor(labels)

        collected = []
        for graph in dataset:
            y = getattr(graph, "y", None)
            if y is None:
                continue
            collected.append(torch.as_tensor(y).view(-1))
        if not collected:
            return None
        return torch.cat(collected, dim=0)
    except Exception:
        return None


def _num_classes_for_classification(dataset, task_level: str) -> int:
    num_classes = getattr(dataset, "num_classes", None)
    try:
        if num_classes is not None and int(num_classes) > 0:
            return int(num_classes)
    except Exception:
        pass

    labels = _extract_labels(dataset, task_level)
    if labels is None or labels.numel() == 0:
        return -1

    label_tensor = labels.detach().view(-1)
    if label_tensor.dtype.is_floating_point:
        finite = label_tensor[torch.isfinite(label_tensor)]
        if finite.numel() == 0:
            return -1
        label_tensor = finite.round().long()
    else:
        label_tensor = label_tensor.long()

    # Ignore common missing-label markers.
    label_tensor = label_tensor[label_tensor >= 0]
    if label_tensor.numel() == 0:
        return -1

    return int(torch.unique(label_tensor).numel())


def _num_features(dataset, task_level: str) -> int:
    for attr in ("num_node_features", "num_features"):
        value = getattr(dataset, attr, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass

    try:
        if task_level == "node":
            data = dataset[0]
            x = getattr(data, "x", None)
            return int(x.size(1)) if x is not None and x.dim() == 2 else -1

        if len(dataset) == 0:
            return -1
        graph = dataset[0]
        x = getattr(graph, "x", None)
        return int(x.size(1)) if x is not None and x.dim() == 2 else -1
    except Exception:
        return -1


def _summarize_node_dataset(name: str, dataset_root: str) -> DatasetSummaryRow:
    dataset = create_dataset(
        name=name,
        root=dataset_root,
        task_level="node",
        feat_reduction=False,
        induced=False,
    )
    data = dataset[0]

    num_nodes = int(getattr(data, "num_nodes", 0) or (data.x.size(0) if getattr(data, "x", None) is not None else 0))
    edge_index = getattr(data, "edge_index", None)
    num_edges = int(edge_index.size(1)) if edge_index is not None else 0

    regression = is_regression_dataset(dataset, "node")
    task_type = "regression" if regression else "classification"
    num_labels = -1 if regression else _num_classes_for_classification(dataset, "node")
    num_features = _num_features(dataset, "node")

    return DatasetSummaryRow(
        name=name,
        dataset_type="node",
        num_graphs=1,
        num_nodes=num_nodes,
        avg_nodes=float(num_nodes),
        num_edges=num_edges,
        avg_edges=float(num_edges),
        num_features=num_features,
        task_type=task_type,
        num_labels=num_labels,
    )


def _summarize_graph_dataset(
    name: str,
    dataset_root: str,
    graph_filter_dir: str,
) -> DatasetSummaryRow:
    dataset = create_dataset(
        name=name,
        root=dataset_root,
        task_level="graph",
        feat_reduction=False,
        induced=False,
        graph_filter_dir=graph_filter_dir,
    )

    num_graphs, total_nodes, total_edges = _compute_graph_totals(dataset)
    avg_nodes = float(total_nodes) / float(num_graphs) if num_graphs > 0 else 0.0
    avg_edges = float(total_edges) / float(num_graphs) if num_graphs > 0 else 0.0

    if getattr(dataset, "has_labels", None) is False:
        task_type = "none"
        num_labels = -1
    else:
        regression = is_regression_dataset(dataset, "graph")
        task_type = "regression" if regression else "classification"
        num_labels = -1 if regression else _num_classes_for_classification(dataset, "graph")
    num_features = _num_features(dataset, "graph")

    return DatasetSummaryRow(
        name=name,
        dataset_type="graph",
        num_graphs=num_graphs,
        num_nodes=total_nodes,
        avg_nodes=avg_nodes,
        num_edges=total_edges,
        avg_edges=avg_edges,
        num_features=num_features,
        task_type=task_type,
        num_labels=num_labels,
    )


def _summary_row_key(row: DatasetSummaryRow) -> Tuple[str, str]:
    return row.name.casefold(), row.dataset_type.casefold()


def _row_from_record(record: dict) -> Optional[DatasetSummaryRow]:
    try:
        name = str(record.get("name", "")).strip()
        dataset_type = str(record.get("type", "")).strip().lower()
        if not name or dataset_type not in {"node", "graph"}:
            return None
        return DatasetSummaryRow(
            name=name,
            dataset_type=dataset_type,
            num_graphs=int(record.get("#graphs", 0)),
            num_nodes=int(record.get("#nodes", 0)),
            avg_nodes=float(record.get("#avg. nodes", 0.0)),
            num_edges=int(record.get("#edges", 0)),
            avg_edges=float(record.get("#avg. edges", 0.0)),
            num_features=int(record.get("#features", -1)),
            task_type=str(record.get("task type") or record.get("task_type") or record.get("task") or "").strip(),
            num_labels=int(record.get("#labels", -1)),
        )
    except Exception:
        return None


def _load_existing_summary_rows(output_path: Path) -> List[DatasetSummaryRow]:
    if not output_path.is_file():
        return []

    rows: List[DatasetSummaryRow] = []
    try:
        with output_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for record in reader:
                row = _row_from_record(record)
                if row is not None:
                    rows.append(row)
    except Exception:
        return []
    return rows


def _merge_summary_rows(
    existing_rows: Sequence[DatasetSummaryRow],
    new_rows: Sequence[DatasetSummaryRow],
) -> List[DatasetSummaryRow]:
    merged_by_key = {}

    for row in existing_rows:
        key = _summary_row_key(row)
        merged_by_key[key] = row

    for row in new_rows:
        key = _summary_row_key(row)
        merged_by_key[key] = row

    return sorted(merged_by_key.values(), key=_summary_row_key)


def _rows_to_tsv(rows: Sequence[DatasetSummaryRow], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Summary generation is commonly launched as a SLURM array.  The read,
    # merge, and replacement must therefore be one critical section; locking
    # only the write would still let two workers discard each other's rows.
    with exclusive_table_lock(output_path):
        merged_rows = _merge_summary_rows(
            existing_rows=_load_existing_summary_rows(output_path),
            new_rows=rows,
        )
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                tmp_path = Path(f.name)
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(
                    [
                        "name",
                        "type",
                        "#graphs",
                        "#nodes",
                        "#avg. nodes",
                        "#edges",
                        "#avg. edges",
                        "#features",
                        "task type",
                        "#labels",
                    ]
                )
                for row in merged_rows:
                    writer.writerow(
                        [
                            row.name,
                            row.dataset_type,
                            row.num_graphs,
                            row.num_nodes,
                            f"{row.avg_nodes:.2f}",
                            row.num_edges,
                            f"{row.avg_edges:.2f}",
                            row.num_features,
                            row.task_type,
                            row.num_labels,
                        ]
                    )
                f.flush()
                os.fsync(f.fileno())
            output_mode = (output_path.stat().st_mode & 0o777) if output_path.exists() else 0o640
            os.chmod(tmp_path, output_mode)
            os.replace(tmp_path, output_path)
        except BaseException:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise
    return len(merged_rows)


def _print_summary_status(name: str, dataset_type: str) -> None:
    print(f"[DataSummary] Processing {name} (type={dataset_type})...")


def _sorted_dataset_names(names: Sequence[str]) -> List[str]:
    """Return deterministic A->Z ordering for dataset names."""
    return sorted(names, key=lambda item: item.casefold())


def _normalize_dataset_names(names: Sequence[str]) -> List[str]:
    unique_by_key = {}
    for raw_name in names:
        name = str(raw_name).strip()
        if not name:
            continue
        key = name.casefold()
        if key not in unique_by_key:
            unique_by_key[key] = name
    return _sorted_dataset_names(list(unique_by_key.values()))


def _read_names_from_cfg_list(path_value: str) -> List[str]:
    list_path = _resolve_list_path(path_value)
    return read_datasets(str(list_path))


def run_data_summary(
    cfg,
    node_names: Optional[Sequence[str]] = None,
    graph_names: Optional[Sequence[str]] = None,
) -> int:
    prep_cfg = cfg.data_preparation
    ds_cfg = prep_cfg.dataset
    dataset_root = str(getattr(ds_cfg, "root", "data/datasets"))
    graph_filter_dir = str(getattr(ds_cfg, "graph_filter_dir", "data/filters"))

    if node_names is None:
        node_names = _read_names_from_cfg_list(str(getattr(ds_cfg, "available_node_datasets", "")))
    if graph_names is None:
        graph_names = _read_names_from_cfg_list(str(getattr(ds_cfg, "available_graph_datasets", "")))

    node_names = _normalize_dataset_names(node_names)
    graph_names = _normalize_dataset_names(graph_names)

    rows: List[DatasetSummaryRow] = []
    failures: List[str] = []

    for name in node_names:
        _print_summary_status(name, "node")
        try:
            rows.append(_summarize_node_dataset(name=name, dataset_root=dataset_root))
        except Exception as exc:  # pylint: disable=broad-except
            failures.append(f"{name} (node): {exc}")
            print(f"[DataSummary][FAIL] {name} (node): {exc}")

    for name in graph_names:
        _print_summary_status(name, "graph")
        try:
            rows.append(
                _summarize_graph_dataset(
                    name=name,
                    dataset_root=dataset_root,
                    graph_filter_dir=graph_filter_dir,
                )
            )
        except Exception as exc:  # pylint: disable=broad-except
            failures.append(f"{name} (graph): {exc}")
            print(f"[DataSummary][FAIL] {name} (graph): {exc}")

    output_path = Path(str(prep_cfg.summary_file))
    total_rows = _rows_to_tsv(rows, output_path)
    print(f"[DataSummary] Updated {len(rows)} dataset summaries into {output_path} (total={total_rows})")

    if failures:
        print(f"[DataSummary] Failed datasets: {len(failures)}")
        return 1
    return 0


def run_data_summary_from_cli(argv: Iterable[str]) -> int:
    cfg = update_cfg(base_cfg, list(argv))
    return run_data_summary(cfg)
