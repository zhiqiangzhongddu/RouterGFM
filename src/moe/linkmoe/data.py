"""Graph views and pair bookkeeping for Link-MoE.

Both views read the repo's shared (pair-level, v2) edge split, so every
method scores the same positive/negative pairs:

* full-graph view (``make_loaders`` edge branch, ``induced=False``): node
  features, message graph ``M`` (remaining positives; train positives
  excluded) and context graph ``C = M + train positives``; val/test
  positives are absent from both;
* induced view (SEAL expert only): enclosing subgraphs built on ``C`` with
  the target link removed, aligned to the full-graph pair order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from src.data_loader import create_dataset, dataset_info, make_loaders
from src.utils.dataset_helpers import shared_induced_root, shared_split_root

from .experts import drnl_labels

SPLITS = ("train", "val", "test")


@dataclass
class LinkViews:
    x: torch.Tensor  # [N, d] node features (SVD-reduced)
    num_nodes: int
    message_edge_index: torch.Tensor  # M, undirected (train-time message graph)
    context_edge_index: torch.Tensor  # C, undirected (eval-time graph)
    pairs: dict[str, torch.Tensor]  # split -> [2, P] (positives first, then negatives)
    labels: dict[str, torch.Tensor]  # split -> [P] float {0, 1}
    meta: dict = field(default_factory=dict)  # dataset_info of the base graph


def link_views_from_loaders(train_loader, val_loader, test_loader, meta: dict | None = None) -> LinkViews:
    """Assemble :class:`LinkViews` from the non-induced edge loaders of ``make_loaders``."""
    items = {name: next(iter(loader)) for name, loader in zip(SPLITS, (train_loader, val_loader, test_loader))}
    if not torch.equal(items["val"].edge_index, items["test"].edge_index):
        raise ValueError("[LinkMoE] val and test must share the eval context graph.")
    num_nodes = int(items["train"].num_nodes)
    return LinkViews(
        x=items["train"].x.float().cpu(),
        num_nodes=num_nodes,
        message_edge_index=to_undirected(items["train"].edge_index.cpu(), num_nodes=num_nodes),
        context_edge_index=to_undirected(items["val"].edge_index.cpu(), num_nodes=num_nodes),
        pairs={k: v.edge_label_index.long().cpu() for k, v in items.items()},
        labels={k: v.edge_label.float().cpu() for k, v in items.items()},
        meta=dict(meta or {}),
    )


def _create_link_dataset(cfg, seed: int, induced: bool):
    ds_cfg = cfg.moe.linkmoe.dataset
    return create_dataset(
        name=ds_cfg.name,
        root=ds_cfg.root,
        task_level="edge",
        feat_reduction=ds_cfg.feat_reduction,
        feat_reduction_dim=int(ds_cfg.feat_reduction_svd_dim),
        persist_feature_svd=ds_cfg.feat_reduction,
        feature_svd_dir=ds_cfg.feature_svd_dir,
        induced=induced,
        induced_max_hops=int(ds_cfg.induced_max_hops),
        edge_max_size=ds_cfg.edge_max_size,
        split_root=shared_split_root(cfg),
        induced_root=shared_induced_root(cfg, ds_cfg.induced_root),
        split=tuple(ds_cfg.fixed_split),
        seed=int(seed),
    )


def build_link_views(cfg, seed: int) -> LinkViews:
    """Full-graph view of ``cfg.moe.linkmoe.dataset`` for one split seed."""
    ds_cfg = cfg.moe.linkmoe.dataset
    dataset = _create_link_dataset(cfg, seed, induced=False)
    meta = dataset_info(dataset=dataset, task_level="edge", name=ds_cfg.name, induced=False)
    loaders = make_loaders(
        dataset=dataset,
        dataset_name=ds_cfg.name,
        task_level="edge",
        batch_size=1,
        num_workers=0,
        split=tuple(ds_cfg.fixed_split),
        seed=int(seed),
        induced=False,
        split_root=shared_split_root(cfg),
    )
    return link_views_from_loaders(*loaders, meta=meta)


def _pair_keys(pairs: torch.Tensor, labels: torch.Tensor) -> list[tuple[int, int, int]]:
    lo = torch.minimum(pairs[0], pairs[1]).tolist()
    hi = torch.maximum(pairs[0], pairs[1]).tolist()
    return list(zip(lo, hi, [int(round(float(y))) for y in labels.view(-1).tolist()]))


def align_induced_to_pairs(graphs: list[Data], pairs: torch.Tensor, labels: torch.Tensor) -> list[Data]:
    """Reorder induced graphs to ``pairs`` order by ``(unordered global pair, label)``; bijection or raise."""
    index: dict[tuple[int, int, int], Data] = {}
    for g in graphs:
        gp = g.global_target_pair.view(-1)
        key = (int(min(gp[0], gp[1])), int(max(gp[0], gp[1])), int(g.y.view(-1)[0]))
        if key in index:
            raise ValueError(f"[LinkMoE] duplicated induced graph for pair {key}.")
        index[key] = g
    keys = _pair_keys(pairs, labels)
    missing = [k for k in keys if k not in index]
    if missing or len(index) != len(keys):
        raise ValueError(
            f"[LinkMoE] induced graphs do not match the split pairs "
            f"({len(missing)} missing, {len(index)} graphs vs {len(keys)} pairs)."
        )
    return [index[k] for k in keys]


def prepare_seal_graphs(graphs: list[Data], pairs: torch.Tensor, labels: torch.Tensor, max_z: int) -> list[Data]:
    """Aligned SEAL inputs (undirected subgraph, DRNL labels ``z``, local target, label)."""
    out = []
    for g in align_induced_to_pairs(graphs, pairs, labels):
        n = int(g.num_nodes)
        edge_index = to_undirected(g.edge_index, num_nodes=n)
        u, v = (int(t) for t in g.edge_label_index.view(-1))
        out.append(Data(
            x=g.x.float(),
            edge_index=edge_index,
            edge_label_index=g.edge_label_index.view(2, 1).long(),
            z=drnl_labels(edge_index, n, u, v, max_z),
            y=g.y.view(-1)[:1].long(),
            num_nodes=n,
        ))
    return out


def build_seal_view(cfg, seed: int, views: LinkViews) -> dict[str, list[Data]]:
    """Induced enclosing subgraphs per split, aligned to ``views.pairs`` order."""
    dataset = _create_link_dataset(cfg, seed, induced=True)
    by_split: dict[str, list[Data]] = {name: [] for name in SPLITS}
    for g, tag in zip(dataset.graphs, dataset.split_tags):
        by_split[tag].append(g)
    max_z = int(cfg.moe.linkmoe.seal.max_z)
    return {
        name: prepare_seal_graphs(by_split[name], views.pairs[name], views.labels[name], max_z)
        for name in SPLITS
    }


def stratified_split(labels: torch.Tensor, ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-class shuffled ``floor(ratio * n_c)`` / rest partition (each side non-empty when ``n_c >= 2``)."""
    generator = torch.Generator().manual_seed(int(seed))
    y = labels.view(-1).round().long()
    first, second = [], []
    for cls in torch.unique(y).tolist():
        idx = torch.nonzero(y == cls, as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.numel(), generator=generator)]
        n_first = int(float(ratio) * idx.numel())
        if idx.numel() >= 2:
            n_first = min(max(n_first, 1), idx.numel() - 1)
        first.append(idx[:n_first])
        second.append(idx[n_first:])
    return torch.sort(torch.cat(first)).values, torch.sort(torch.cat(second)).values


__all__ = [
    "LinkViews",
    "SPLITS",
    "align_induced_to_pairs",
    "build_link_views",
    "build_seal_view",
    "link_views_from_loaders",
    "prepare_seal_graphs",
    "stratified_split",
]
