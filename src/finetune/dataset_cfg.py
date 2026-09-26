"""Helpers for resolving the finetune target dataset configuration."""

from __future__ import annotations

from typing import Any

from src.utils.dataset_helpers import shared_split_root


def resolve_target_dataset_cfg(
    cfg,
    task_cls,
) -> dict[str, Any]:
    """Resolve target dataset settings, allowing finetune overrides.

    Extracted from ``FinetuneRunner._get_target_dataset_cfg`` to reduce
    the runner's size and make the resolution logic independently testable.
    """
    target_cfg = getattr(getattr(cfg, "finetune", None), "dataset", None)
    pretrain_ds = getattr(getattr(cfg, "pretrain", None), "dataset", None)
    name = getattr(target_cfg, "name", None) or getattr(pretrain_ds, "name", None)
    task_level = getattr(target_cfg, "task_level", None) or getattr(pretrain_ds, "task_level", None)
    induced = getattr(target_cfg, "induced", None)
    if induced is None and pretrain_ds is not None:
        induced = getattr(pretrain_ds, "induced", False)
    root = getattr(target_cfg, "root", None) or getattr(pretrain_ds, "root", None)
    split_root = shared_split_root(cfg)
    feat_reduction = getattr(target_cfg, "feat_reduction", True)
    feat_reduction_dim = getattr(
        target_cfg,
        "feat_reduction_svd_dim",
        getattr(target_cfg, "feat_reduction_dim", None),
    )
    if feat_reduction_dim is None:
        feat_reduction_dim = getattr(
            pretrain_ds,
            "feat_reduction_svd_dim",
            getattr(pretrain_ds, "feat_reduction_dim", 100),
        )
    feature_svd_dir = getattr(target_cfg, "feature_svd_dir", None) or getattr(
        pretrain_ds, "feature_svd_dir", "data/feature_svd"
    )
    if feature_svd_dir in (None, ""):
        feature_svd_dir = "data/feature_svd"
    subgraph_svd = getattr(target_cfg, "subgraph_svd", False)
    induced_root = getattr(target_cfg, "induced_root", None) or getattr(pretrain_ds, "induced_root", None) or ""
    induced_min_size = getattr(target_cfg, "induced_min_size", 10)
    induced_max_size = getattr(target_cfg, "induced_max_size", 30)
    induced_max_hops = getattr(target_cfg, "induced_max_hops", 5)

    # Let the method class adjust dataset parameters (e.g. GPF overrides
    # induced mode, EdgePrompt adjusts subgraph sizing).
    dataset_params = {
        "name": name,
        "task_level": task_level,
        "induced": induced,
        "root": root,
        "split_root": split_root,
        "feat_reduction": feat_reduction,
        "feat_reduction_dim": feat_reduction_dim,
        "feature_svd_dir": feature_svd_dir,
        "subgraph_svd": subgraph_svd,
        "induced_root": induced_root,
        "induced_min_size": induced_min_size,
        "induced_max_size": induced_max_size,
        "induced_max_hops": induced_max_hops,
    }
    return task_cls.adjust_dataset_cfg(cfg, dataset_params)
