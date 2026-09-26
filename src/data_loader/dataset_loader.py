"""DataLoader creation and split orchestration for train/val/test partitioning."""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .dataset_metadata import resolve_count_split_strategy
from .dataset_splits import (
    _canonical_split_dataset_name,
    _get_or_create_count_split,
    _get_or_create_edge_split_payload,
    _get_or_create_few_shot_split,
    _get_or_create_split_indices,
    _get_or_create_split_indices_subset,
    _is_few_shot_split_def,
    _validate_edge_split_def,
    _validate_split_def,
    split_graph_dataset,
)
from .dataset_storage import _get_dataset_data_storage
from .induced_graphs import SingleGraphDataLoader


def make_loaders(
    dataset,
    dataset_name: str,
    task_level: str,
    batch_size: int,
    num_workers: int,
    split: Tuple[float, float, float],
    seed: int,
    induced: bool = False,
    split_root: str = "",
    return_split_meta: bool = False,
):
    """Create data loaders for training, validation, and testing."""
    raw_dataset_name = str(dataset_name)
    split_dataset_name = _canonical_split_dataset_name(raw_dataset_name, task_level, int(seed))
    split_meta = None

    def _builtin_split_meta():
        return {"status": "builtin", "path": None}

    def _finalize_loaders(train_loader, val_loader, test_loader):
        if return_split_meta:
            meta = dict(split_meta) if isinstance(split_meta, dict) else _builtin_split_meta()
            return train_loader, val_loader, test_loader, meta
        return train_loader, val_loader, test_loader

    if split is not None:
        if task_level == "edge":
            _validate_edge_split_def(split)
        else:
            _validate_split_def(split)

    def _few_shot_indices_from_graphs():
        if len(split) < 3:
            raise ValueError("Few-shot split must provide [shots_per_class, val_ratio, test_ratio].")
        shots_per_class = int(split[0])
        val_ratio = float(split[1])
        test_ratio = float(split[2])
        split_root_path = Path(split_root) if split_root else None

        # Induced node datasets share the split FILE with the non-induced
        # workflow (same canonical name), but their natural index space is
        # "induced-graph position", which only coincides with node ids on
        # fully-labeled datasets. Always split in node-id space and map back
        # to positions, so the artifact has one meaning for both workflows.
        node_position_by_id = None
        labels = None
        if task_level == "node" and induced:
            graphs = getattr(dataset, "graphs", None)
            if graphs is None:
                graphs = list(dataset)
            base_total = int(getattr(dataset, "base_num_nodes", 0) or 0)
            if base_total > 0 and all(hasattr(g, "base_node_id") for g in graphs):
                node_position_by_id = {}
                node_labels = torch.full((base_total,), -1, dtype=torch.long)
                for pos, item in enumerate(graphs):
                    node_id = int(item.base_node_id)
                    node_position_by_id[node_id] = pos
                    target = item.y.view(-1) if getattr(item, "y", None) is not None else None
                    if target is not None and target.numel() == 1:
                        node_labels[node_id] = int(target[0].item())
                labels = node_labels

        if labels is None:
            dataset_data = _get_dataset_data_storage(dataset)
            if dataset_data is not None and getattr(dataset_data, "y", None) is not None:
                labels = dataset_data.y
                if labels.numel() == len(dataset):
                    labels = labels.view(-1)
            if labels is None or labels.numel() != len(dataset):
                collected = []
                for item in dataset:
                    if not hasattr(item, "y") or item.y is None:
                        raise ValueError("Few-shot split requires labels for each graph instance.")
                    target = item.y.view(-1)
                    if target.numel() != 1:
                        raise ValueError("Few-shot split currently supports single-label targets.")
                    collected.append(int(target[0].item()))
                labels = torch.tensor(collected, dtype=torch.long)

        result = _get_or_create_few_shot_split(
            dataset_name=split_dataset_name,
            labels=labels,
            shots_per_class=shots_per_class,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            split_root_path=split_root_path,
            return_split_meta=return_split_meta,
        )
        if node_position_by_id is None:
            return result

        def _to_positions(node_ids):
            positions = [node_position_by_id[i] for i in node_ids if i in node_position_by_id]
            if len(positions) != len(node_ids):
                print(
                    "[Dataset split] WARNING: "
                    f"{len(node_ids) - len(positions)} few-shot node ids have no "
                    "induced subgraph and were dropped."
                )
            return positions

        if return_split_meta:
            train_ids, val_ids, test_ids, meta = result
            return _to_positions(train_ids), _to_positions(val_ids), _to_positions(test_ids), meta
        train_ids, val_ids, test_ids = result
        return _to_positions(train_ids), _to_positions(val_ids), _to_positions(test_ids)

    def _count_indices_from_graphs():
        if len(split) < 3:
            raise ValueError("Integer-first split must provide [train_count, val_ratio, test_ratio].")
        split_root_path = Path(split_root) if split_root else None
        return _get_or_create_count_split(
            dataset_name=split_dataset_name,
            train_count=int(split[0]),
            val_ratio=float(split[1]),
            test_ratio=float(split[2]),
            seed=seed,
            split_root_path=split_root_path,
            total=len(dataset),
            return_split_meta=return_split_meta,
        )

    use_few_shot = _is_few_shot_split_def(split)
    split_strategy = resolve_count_split_strategy(dataset, task_level) if use_few_shot else "ratios"
    if use_few_shot and split_strategy == "unsupported":
        raise ValueError("Integer-first splits are not supported for unlabeled or single-target regression datasets.")

    if task_level == "node" and not induced:
        data = dataset[0]
        if not split_root:
            raise ValueError("split_root is required to save or load fixed splits for node datasets.")
        split_root_path = Path(split_root)
        labels = getattr(data, "y", None)
        labeled_idx = None
        if labels is not None:
            label_tensor = torch.as_tensor(labels)
            if label_tensor.dim() <= 1:
                label_matrix = label_tensor.view(-1, 1)
            else:
                label_matrix = label_tensor.view(label_tensor.size(0), -1)
            valid_mask = torch.ones_like(label_matrix, dtype=torch.bool)
            if label_matrix.dtype.is_floating_point:
                valid_mask &= torch.isfinite(label_matrix)
            if label_matrix.dtype != torch.bool:
                valid_mask &= label_matrix >= 0
            labeled_idx = torch.nonzero(valid_mask.any(dim=1), as_tuple=False).view(-1).tolist()
        if use_few_shot and split_strategy == "balanced":
            if len(split) < 3:
                raise ValueError("Few-shot split must provide [shots_per_class, val_ratio, test_ratio].")
            result = _get_or_create_few_shot_split(
                dataset_name=split_dataset_name,
                labels=data.y,
                shots_per_class=int(split[0]),
                val_ratio=float(split[1]),
                test_ratio=float(split[2]),
                seed=seed,
                split_root_path=split_root_path,
                return_split_meta=return_split_meta,
            )
        elif use_few_shot and split_strategy == "random":
            result = _get_or_create_count_split(
                dataset_name=split_dataset_name,
                train_count=int(split[0]),
                val_ratio=float(split[1]),
                test_ratio=float(split[2]),
                seed=seed,
                split_root_path=split_root_path,
                total=data.num_nodes,
                subset_indices=labeled_idx if labeled_idx is not None and len(labeled_idx) < data.num_nodes else None,
                return_split_meta=return_split_meta,
            )
        else:
            if labeled_idx is not None and len(labeled_idx) < data.num_nodes:
                result = _get_or_create_split_indices_subset(
                    dataset_name=split_dataset_name,
                    split=split,
                    seed=seed,
                    split_root_path=split_root_path,
                    subset_indices=labeled_idx,
                    return_split_meta=return_split_meta,
                )
            else:
                result = _get_or_create_split_indices(
                    dataset_name=split_dataset_name,
                    split=split,
                    seed=seed,
                    split_root_path=split_root_path,
                    total=data.num_nodes,
                    return_split_meta=return_split_meta,
                )
        if return_split_meta:
            train_idx, val_idx, test_idx, split_meta = result
        else:
            train_idx, val_idx, test_idx = result
        for mask_name, indices in (
            ("train_mask", train_idx),
            ("val_mask", val_idx),
            ("test_mask", test_idx),
        ):
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[indices] = True
            setattr(data, mask_name, mask)
        return _finalize_loaders(
            SingleGraphDataLoader(data),
            SingleGraphDataLoader(data),
            SingleGraphDataLoader(data),
        )

    if task_level == "edge" and not induced:
        data = dataset[0]
        if not split_root:
            raise ValueError("split_root is required to save or load fixed splits for edge datasets.")
        split_root_path = Path(split_root)
        if use_few_shot:
            raise ValueError("Few-shot split is not supported for edge-level tasks.")
        result = _get_or_create_edge_split_payload(
            dataset_name=split_dataset_name,
            split=split,
            seed=seed,
            split_root_path=split_root_path,
            data=data,
            return_split_meta=return_split_meta,
        )
        if return_split_meta:
            split_payload, split_meta = result
        else:
            split_payload = result

        edge_device = data.edge_index.device
        def _message_edges(index_key: str):
            indices = torch.as_tensor(split_payload[index_key], dtype=torch.long, device=edge_device)
            if indices.numel() == 0:
                return torch.empty((2, 0), dtype=torch.long, device=edge_device)
            return data.edge_index[:, indices]

        # Training targets must not appear in their own message graph.  Once
        # training is complete, however, those observed positives are valid
        # context for both validation and test queries.  Held-out val/test
        # targets remain excluded by construction of context_pos_idx.
        train_message_edge_index = _message_edges("message_pos_idx")
        eval_message_edge_index = _message_edges("context_pos_idx")

        def _edge_subset(pos_key: str, neg_key: str, message_edge_index: torch.Tensor):
            pos_idx = torch.as_tensor(split_payload[pos_key], dtype=torch.long, device=edge_device)
            if pos_idx.numel() == 0:
                pos_pairs = torch.empty((2, 0), dtype=torch.long, device=edge_device)
            else:
                pos_pairs = data.edge_index[:, pos_idx]

            neg_source = split_payload.get(neg_key)
            if neg_source is None:
                neg_pairs = torch.empty((2, 0), dtype=torch.long, device=edge_device)
            else:
                neg_pairs = torch.as_tensor(neg_source, dtype=torch.long, device=edge_device)
                if neg_pairs.numel() == 0:
                    neg_pairs = torch.empty((2, 0), dtype=torch.long, device=edge_device)
                elif neg_pairs.dim() != 2 or neg_pairs.size(0) != 2:
                    raise ValueError(f"Invalid negative edge tensor for key={neg_key}.")

            edge_label_index = torch.cat([pos_pairs, neg_pairs], dim=1)
            edge_label = torch.cat(
                [
                    torch.ones(pos_pairs.size(1), dtype=torch.float, device=edge_label_index.device),
                    torch.zeros(neg_pairs.size(1), dtype=torch.float, device=edge_label_index.device),
                ],
                dim=0,
            )
            subset = Data(
                x=getattr(data, "x", None),
                edge_index=message_edge_index,
                edge_label_index=edge_label_index,
                num_nodes=data.num_nodes,
            )
            subset.edge_label = edge_label
            return subset

        train_data = _edge_subset("train_pos_idx", "train_neg_edge_index", train_message_edge_index)
        val_data = _edge_subset("val_pos_idx", "val_neg_edge_index", eval_message_edge_index)
        test_data = _edge_subset("test_pos_idx", "test_neg_edge_index", eval_message_edge_index)

        return _finalize_loaders(
            SingleGraphDataLoader(train_data),
            SingleGraphDataLoader(val_data),
            SingleGraphDataLoader(test_data),
        )

    if task_level == "edge" and induced and hasattr(dataset, "split_tags"):
        split_meta = _builtin_split_meta()
        train_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "train"]
        val_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "val"]
        test_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "test"]
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)
    elif task_level == "graph" and hasattr(dataset, "split_tags") and dataset.split_tags:
        split_meta = _builtin_split_meta()
        train_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "train"]
        val_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "val"]
        test_idx = [i for i, tag in enumerate(dataset.split_tags) if tag == "test"]
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)
    elif use_few_shot and split_strategy == "balanced":
        result = _few_shot_indices_from_graphs()
        if return_split_meta:
            train_idx, val_idx, test_idx, split_meta = result
        else:
            train_idx, val_idx, test_idx = result
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)
    elif use_few_shot and split_strategy == "random":
        result = _count_indices_from_graphs()
        if return_split_meta:
            train_idx, val_idx, test_idx, split_meta = result
        else:
            train_idx, val_idx, test_idx = result
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)
    else:
        result = split_graph_dataset(
            dataset=dataset,
            dataset_name=split_dataset_name,
            split=split,
            seed=seed,
            split_root=split_root,
            return_split_meta=return_split_meta,
        )
        if return_split_meta:
            train_set, val_set, test_set, split_meta = result
        else:
            train_set, val_set, test_set = result

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    train_loader = DataLoader(dataset=train_set, **loader_kwargs)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return _finalize_loaders(train_loader, val_loader, test_loader)
