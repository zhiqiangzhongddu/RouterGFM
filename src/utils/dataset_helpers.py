"""Shared dataset-related helpers used by pretrain, train, and finetune runners.

Four distinct task-level concepts live in this codebase. Keep them separate:

``raw_task_level``
    User intent from TSV/CLI (``node``, ``edge``, or ``graph``). Used for
    run names and checkpoint identity.
``effective_task_level``
    What methods consume for forward/loss/head logic. Induced node/edge tasks
    become graph-level subgraph batches.
``split_task_level``
    Which on-disk split file keys apply. Induced node runs keep node splits;
    induced edge runs keep edge splits.
``loader_task_level``
    Which ``make_loaders`` branch to take. Induced edge stays ``edge`` so the
    edge split validator and split-tags branch remain active.
"""

from __future__ import annotations

import re
from numbers import Integral

import torch

from src.utils.parsing import to_bool


def shared_split_root(cfg) -> str:
    """Resolve the split root directory from the data_preparation config block."""
    ds_cfg = getattr(getattr(cfg, "data_preparation", None), "dataset", None)
    return getattr(ds_cfg, "split_root", "data/splits")


def shared_induced_root(cfg, fallback: str = "") -> str:
    """Resolve the induced-subgraph root directory from the data_preparation config block."""
    ds_cfg = getattr(getattr(cfg, "data_preparation", None), "dataset", None)
    root = getattr(ds_cfg, "induced_root", "")
    return root or fallback


def split_dataset_name(base_name: str, task_level: str, seed: int) -> str:
    """Build a split-aware dataset name, skipping if the name already encodes a seed."""
    name = str(base_name)
    if any(tag in name for tag in ("_node_seed", "_graph_seed", "_edge_seed")):
        return name
    return f"{name}_{task_level}_seed{int(seed)}"


_SPLIT_NAME_SUFFIX = re.compile(r"_(?:node|graph|edge)_seed\d+$")


def canonical_source_dataset_name(value) -> str:
    """Canonical source-dataset identity for same-source (diagonal) comparisons.

    Inverts :func:`split_dataset_name` by stripping a trailing
    ``_<level>_seed<seed>`` tag, then normalizes case and the ``_``/``-``
    spelling so ``Cora``, ``cora_node_seed42``, and ``cora`` all map to the
    same key. Single source of truth for detecting "expert pretrained on the
    same canonical source dataset as the downstream application" — the
    interaction table, graph construction, and the router trainers must all
    route through it so the empty-diagonal rule cannot drift.
    """
    name = str(value or "").strip().lower()
    name = _SPLIT_NAME_SUFFIX.sub("", name)
    return name.replace("_", "-")


def is_few_shot_split(split) -> bool:
    """Return True when *split* uses few-shot form ``(shots_per_class, val_ratio, test_ratio)``.

    Shared across train, finetune, and prompt methods so split semantics
    are detected identically everywhere.
    """
    if not isinstance(split, (tuple, list)) or not split:
        return False
    first = split[0]
    if isinstance(first, bool):
        return False
    if isinstance(first, Integral):
        return True
    if isinstance(first, float) and first.is_integer():
        return True
    return False


def populate_dataset_cfg_from_meta(model_cfg, ds_cfg, dataset_meta: dict) -> None:
    """Auto-fill model and dataset config fields from loaded dataset metadata.

    This replaces the ~8-line block that was copy-pasted across all three
    runner ``_setup()`` methods.
    """
    model_cfg.in_dim = model_cfg.in_dim or dataset_meta.get("num_node_features")
    if getattr(ds_cfg, "num_classes", None) is None:
        ds_cfg.num_classes = dataset_meta.get("num_classes")
    if getattr(ds_cfg, "label_dim", None) is None:
        ds_cfg.label_dim = dataset_meta.get("label_dim")
    if getattr(ds_cfg, "task_type", "none") in (None, "", "none"):
        meta_task_type = dataset_meta.get("task_type")
        if meta_task_type:
            ds_cfg.task_type = meta_task_type


def normalize_node_mask(data, mask_attr: str, device, *, num_nodes: int | None = None) -> "torch.Tensor":
    """Return a boolean mask tensor for the given mask attribute.

    Handles missing masks (returns all-True), wrong-dtype masks, and
    ensures the result is a 1-D boolean tensor on *device*.  Shared
    across TaskAwareObjective, GPPT, GraphPrompt, and supervised
    forward paths.

    When *num_nodes* is provided and the stored mask is longer than
    *num_nodes* (batched data), it is trimmed to match.
    """
    expected = num_nodes if num_nodes is not None else data.num_nodes
    mask = getattr(data, mask_attr, None)
    if mask is None:
        if mask_attr in ("train_mask", "val_mask", "test_mask"):
            # An all-True fallback here turns a plumbing mistake into silent
            # train to/eval on every node (i.e. leakage); make it loud while
            # keeping the permissive behaviour for full-graph semantics.
            print(
                f"[Dataset] WARNING: {mask_attr} missing on data; "
                "falling back to an all-True mask (all nodes selected)."
            )
        return torch.ones(expected, dtype=torch.bool, device=device)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=device).view(-1)
    if num_nodes is not None and mask.size(0) > num_nodes:
        mask = mask[:num_nodes]
    return mask


def checkpoint_dataset_dir_name(dataset_name: str) -> str:
    """Sanitize a dataset name for use as a checkpoint subdirectory."""
    name = str(dataset_name or "").strip()
    if not name:
        return "unknown_dataset"
    name = name.replace("\\", "_").replace("/", "_")
    if name in (".", ".."):
        # "." / ".." would resolve the checkpoint dir into (the parent of)
        # the checkpoint root itself.
        return f"dataset{name.replace('.', '_dot')}"
    return name


def resolve_effective_task_level(task_level_raw: str, induced: bool, default: str = "graph") -> str:
    """Resolve the method-level task level consumed by forward/loss/head logic."""
    task_level = str(task_level_raw or default).lower()
    if to_bool(induced) and task_level in {"node", "edge"}:
        return "graph"
    return task_level


def read_effective_task_level(ds_cfg, default: str = "graph") -> str:
    """Read ``task_level_effective`` from a dataset cfg, resolving raw fallback."""
    effective = getattr(ds_cfg, "task_level_effective", None)
    if effective:
        return str(effective).lower()
    return resolve_effective_task_level(
        getattr(ds_cfg, "task_level", default),
        getattr(ds_cfg, "induced", False),
        default=default,
    )


def resolve_split_task_level(task_level_raw: str, effective_task_level: str, induced: bool) -> str:
    """Resolve the split-file task level used to find on-disk split payloads."""
    raw = str(task_level_raw or "").lower()
    effective = str(effective_task_level or "").lower()
    return raw if to_bool(induced) else effective


def resolve_loader_task_level(task_level_raw: str, effective_task_level: str, induced: bool) -> str:
    """Resolve the task level to pass into ``make_loaders``.

    This is a loader-construction concern, not method forward behavior.
    Induced edge runs keep the edge loader branch so edge split validation
    and split-tag filtering stay in effect.
    """
    if to_bool(induced) and str(task_level_raw).lower() == "edge":
        return "edge"
    return effective_task_level


def make_workflow_loaders(
    *,
    dataset,
    dataset_name: str,
    task_level_raw: str,
    effective_task_level: str,
    batch_size: int,
    num_workers: int,
    split: tuple,
    seed: int,
    induced: bool,
    split_root: str,
):
    """Build train/val/test data loaders for any workflow.

    Shared by pretrain (supervised path), train, and finetune runners.
    """
    from src.data_loader import make_loaders

    split_task_level = resolve_split_task_level(task_level_raw, effective_task_level, induced)
    loader_task_level = resolve_loader_task_level(task_level_raw, effective_task_level, induced)
    split_ds_name = split_dataset_name(dataset_name, split_task_level, seed)
    return make_loaders(
        dataset=dataset,
        dataset_name=split_ds_name,
        task_level=loader_task_level,
        batch_size=batch_size,
        num_workers=num_workers,
        split=split,
        seed=seed,
        induced=induced,
        split_root=split_root,
    )
