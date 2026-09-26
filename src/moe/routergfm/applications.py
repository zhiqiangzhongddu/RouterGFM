"""Application data: instances, support/query/diagnostic positions, labels, statistics.

An application's support set ``S_a`` is the train split and its query set
``Q_a`` the test split of ``AppSpec.split``; the diagnostic set ``D_a`` is a
fixed, label-stratified subsample of ``Q_a`` (disjoint from the support used to
fit heads, App. B.3). Few-shot splits have an empty validation part; the routed
LP convention ignores its validation positives.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Protocol

import torch

from src.data_loader.dataset_domains import CLASS_TO_DOMAIN, KEYWORD_DOMAINS, NAME_TO_DOMAIN
from src.utils.checkpoint import save_torch_atomic
from src.utils.supervised_loss import binary_targets_and_valid, prepare_class_labels

from .common import (
    LINK,
    MULTILABEL,
    REGRESSION,
    TASK_FAMILIES,
    AppSpec,
    RouterPaths,
    infer_task_family,
    stable_hash,
)
from .losses import is_simplex_family

TASK_LEVELS = ("node", "edge", "graph")
DOMAINS = tuple(
    sorted(set(CLASS_TO_DOMAIN.values()) | set(NAME_TO_DOMAIN.values()) | {d for d, _ in KEYWORD_DOMAINS})
) + ("unknown",)
# Count-valued statistics; metadata features use log1p of these.
STAT_COUNT_KEYS = ("num_instances", "num_nodes", "num_edges", "avg_degree", "feature_dim", "num_classes", "budget")
_META_VERSION = 1


@dataclass
class AppData:
    app: AppSpec
    dataset: Any  # indexable PyG dataset of (sub)graphs; built lazily by RealDataProvider
    level: str  # node | edge | graph (raw task level = readout level)
    task_family: str
    num_classes: int  # C for node/graph cls; 2 for link; L for multilabel; T for regression
    label_dim: int
    support_pos: torch.Tensor  # S_a positions into dataset
    query_pos: torch.Tensor  # Q_a (full test split)
    diag_pos: torch.Tensor  # D_a subset of Q_a, at most apps.max_diagnostic
    labels: Dict[str, torch.Tensor]  # 'support' | 'query' | 'diag', aligned with the positions
    stats: Dict[str, float]  # label-free statistics (+ class/target count) for metadata features
    in_dim: int


class DataProvider(Protocol):
    def load(self, app: AppSpec) -> AppData: ...


# --------------------------------------------------------------------------- #
# Pure helpers (shared with the synthetic test provider)
# --------------------------------------------------------------------------- #
def derive_seed(*parts: Any) -> int:
    """Deterministic 31-bit seed from arbitrary JSON-serializable parts."""
    return int(stable_hash(list(parts), length=8), 16) % (2**31 - 1)


def dataset_domain(name: str) -> str:
    """Coarse domain of a dataset name (``src.data_loader.dataset_domains`` rules)."""
    key = str(name).lower()
    if key in NAME_TO_DOMAIN:
        return NAME_TO_DOMAIN[key]
    for domain, keywords in KEYWORD_DOMAINS:
        if any(token in key for token in keywords):
            return domain
    return "unknown"


def build_stats(
    *,
    level: str,
    family: str,
    num_instances: int,
    num_nodes: float,
    num_edges: float,
    feature_dim: int,
    num_classes: int,
    domain: str,
    budget: int,
) -> Dict[str, float]:
    """Application statistics with a fixed key set (counts + one-hots).

    ``num_nodes``/``num_edges`` are base-graph sizes for node/edge tasks and
    mean graph sizes for graph datasets; ``avg_degree = num_edges / num_nodes``
    (PyG directed edge count).
    """
    stats = {
        "num_instances": float(num_instances),
        "num_nodes": float(num_nodes),
        "num_edges": float(num_edges),
        "avg_degree": float(num_edges) / max(float(num_nodes), 1.0),
        "feature_dim": float(feature_dim),
        "num_classes": float(num_classes),
        "budget": float(budget),
    }
    stats.update({f"family_{f}": float(f == family) for f in TASK_FAMILIES})
    stats.update({f"level_{lvl}": float(lvl == level) for lvl in TASK_LEVELS})
    domain = domain if domain in DOMAINS else "unknown"
    stats.update({f"domain_{d}": float(d == domain) for d in DOMAINS})
    return stats


def stat_feature_names() -> List[str]:
    return (
        [f"log_{k}" for k in STAT_COUNT_KEYS]
        + [f"family_{f}" for f in TASK_FAMILIES]
        + [f"level_{lvl}" for lvl in TASK_LEVELS]
        + [f"domain_{d}" for d in DOMAINS]
    )


def stats_feature_vector(stats: Dict[str, float]) -> torch.Tensor:
    """Numeric metadata vector (order of :func:`stat_feature_names`): log1p counts + one-hots."""
    values = [math.log1p(max(float(stats[k]), 0.0)) for k in STAT_COUNT_KEYS]
    values += [float(stats[name]) for name in stat_feature_names()[len(STAT_COUNT_KEYS):]]
    return torch.tensor(values, dtype=torch.float32)


def convert_labels(raw: torch.Tensor, family: str) -> torch.Tensor:
    """Stacked raw targets ``[n, d]`` -> canonical labels of the family.

    Single-label families: long ``[n]``; multi-label: float ``[n, L]`` in {0,1}
    with NaN for missing (signed {-1,0,1} conventions converted); regression:
    float ``[n, T]`` in raw units. Convert all positions of an application in
    one call: the signed-convention check looks at the whole tensor.
    """
    if is_simplex_family(family):
        return prepare_class_labels(raw)
    if family == MULTILABEL:
        target, valid = binary_targets_and_valid(raw.float())
        return torch.where(valid, target, torch.full_like(target, float("nan")))
    return raw.float()


def valid_label_mask(labels: torch.Tensor, family: str) -> torch.Tensor:
    """Rows that carry a usable label (class >= 0; any observed assay; all targets finite)."""
    if is_simplex_family(family):
        return labels >= 0
    finite = torch.isfinite(labels.reshape(labels.size(0), -1))
    return finite.any(dim=1) if family == MULTILABEL else finite.all(dim=1)


def gather_labels(dataset, positions: torch.Tensor, family: str) -> torch.Tensor:
    """Canonical labels of ``dataset[pos].y`` for each position (see :func:`convert_labels`)."""
    rows = [torch.as_tensor(dataset[int(p)].y).reshape(-1) for p in positions.tolist()]
    if not rows:
        empty = torch.empty(0, 0)
        return torch.empty(0, dtype=torch.long) if is_simplex_family(family) else empty
    return convert_labels(torch.stack(rows), family)


def diagnostic_subsample(labels: torch.Tensor, family: str, cap: int, seed: int) -> torch.Tensor:
    """Sorted indices of a fixed subsample of at most ``cap`` query rows.

    Single-label families are stratified by label (largest-remainder
    proportional allocation, at least one per class when ``cap`` allows);
    other families are sampled uniformly.
    """
    n = int(labels.size(0))
    cap = int(cap)
    if cap <= 0 or n <= cap:
        return torch.arange(n)
    generator = torch.Generator().manual_seed(int(seed))
    if not is_simplex_family(family):
        return torch.sort(torch.randperm(n, generator=generator)[:cap]).values
    y = labels.reshape(-1)
    classes, counts = torch.unique(y, return_counts=True)
    quota = counts.double() * cap / n
    take = quota.floor().long()
    if cap >= classes.numel():
        take = take.clamp_min(1)
    remaining = cap - int(take.sum())
    if remaining > 0:
        order = torch.argsort(-(quota - quota.floor()), stable=True)
        room = counts - take
        for idx in order.tolist():
            if remaining == 0:
                break
            if room[idx] > 0:
                take[idx] += 1
                remaining -= 1
    elif remaining < 0:  # the per-class minimum overshot: trim the largest classes
        for idx in torch.argsort(-take, stable=True).tolist():
            if remaining == 0:
                break
            spare = min(int(take[idx]) - 1, -remaining)
            take[idx] -= spare
            remaining += spare
    chosen = []
    for cls, k in zip(classes.tolist(), take.tolist()):
        members = torch.nonzero(y == cls, as_tuple=False).view(-1)
        chosen.append(members[torch.randperm(members.numel(), generator=generator)[:k]])
    return torch.sort(torch.cat(chosen)).values


def instance_set_key(app: AppSpec) -> str:
    """Identity of an application's instance set: split-independent except for LP.

    Induced node subgraphs and graph datasets do not depend on the split, so
    positions of every budget/seed index the same instances; induced LP
    subgraphs are built per edge split (data key).
    """
    return app.data_key if app.task_level == "edge" else f"{app.dataset}__{app.task_level}"


def assemble_app_data(app: AppSpec, meta: Dict[str, Any], dataset: Any, max_diagnostic: int) -> AppData:
    """AppData from split-level metadata (see ``RealDataProvider``); adds D_a and budget stats."""
    family = meta["task_family"]
    query_pos, query_labels = meta["query_pos"], meta["query_labels"]
    diag_idx = diagnostic_subsample(query_labels, family, max_diagnostic, derive_seed(app.seed, "diag", app.data_key))
    stats = dict(meta["stats"])
    stats["budget"] = float(app.budget)
    return AppData(
        app=app,
        dataset=dataset,
        level=meta["level"],
        task_family=family,
        num_classes=int(meta["num_classes"]),
        label_dim=int(meta["label_dim"]),
        support_pos=meta["support_pos"],
        query_pos=query_pos,
        diag_pos=query_pos[diag_idx],
        labels={
            "support": meta["support_labels"],
            "query": query_labels,
            "diag": query_labels[diag_idx],
        },
        stats=stats,
        in_dim=int(meta["in_dim"]),
    )


# --------------------------------------------------------------------------- #
# Real datasets
# --------------------------------------------------------------------------- #
class _LazyDataset:
    """Defers building a dataset until an item, length, or attribute is requested."""

    def __init__(self, build: Callable[[], Any]):
        self._build = build
        self._obj = None

    def _get(self):
        if self._obj is None:
            self._obj = self._build()
        return self._obj

    def __len__(self) -> int:
        return len(self._get())

    def __getitem__(self, idx):
        return self._get()[idx]

    def __getattr__(self, name: str):
        if name.startswith("__") or name in ("_build", "_obj"):
            raise AttributeError(name)
        return getattr(self._get(), name)


def _loader_positions(loader) -> torch.Tensor:
    indices = getattr(loader.dataset, "indices", None)
    if indices is None:
        raise ValueError("RouterGFM applications expect Subset-backed loaders (induced or graph datasets).")
    return torch.as_tensor(list(indices), dtype=torch.long)


class RealDataProvider:
    """Builds applications with ``src.data_loader`` exactly like the expert-evaluation protocol.

    Mirrors the reference recorder's ``_build_dataset``: 100-d padded features,
    induced subgraphs for node/edge tasks, dataset parameters from
    ``cfg.moe.routergfm.apps.data``. Split-level metadata is cached on disk per
    data key; dataset objects are rebuilt on demand and shared in-process
    (non-LP instance sets do not depend on the split).
    """

    max_cached_datasets = 4

    def __init__(self, cfg):
        self.cfg = cfg
        self.paths = RouterPaths.from_cfg(cfg)
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._datasets: "OrderedDict[str, Any]" = OrderedDict()

    # -- public -------------------------------------------------------------
    def load(self, app: AppSpec) -> AppData:
        meta = self._load_meta(app)
        dataset = _LazyDataset(lambda: self._dataset(app))
        return assemble_app_data(app, meta, dataset, int(self.cfg.moe.routergfm.apps.max_diagnostic))

    # -- datasets -----------------------------------------------------------
    def _remember(self, key: str, dataset: Any) -> None:
        self._datasets[key] = dataset
        self._datasets.move_to_end(key)
        while len(self._datasets) > self.max_cached_datasets:
            self._datasets.popitem(last=False)

    def _dataset(self, app: AppSpec):
        key = instance_set_key(app)
        if key in self._datasets:
            self._datasets.move_to_end(key)
            return self._datasets[key]
        dataset = self._create(app)
        self._remember(key, dataset)
        return dataset

    def _split_root(self) -> str:
        from src.utils.dataset_helpers import shared_split_root

        return str(self.cfg.moe.routergfm.apps.data.split_root or shared_split_root(self.cfg))

    def _create(self, app: AppSpec):
        from src.data_loader import create_dataset
        from src.utils.dataset_helpers import shared_induced_root

        ds = self.cfg.moe.routergfm.apps.data
        return create_dataset(
            name=app.dataset,
            root=ds.root,
            task_level=app.task_level,
            feat_reduction=ds.feat_reduction,
            feat_reduction_dim=ds.feat_reduction_svd_dim,
            # Experts are trained at in_dim=100; featureless query datasets
            # (qm7b) must be padded into that feature space.
            pad_featureless_features=True,
            induced=app.task_level != "graph",
            induced_min_size=ds.induced_min_size,
            induced_max_size=ds.induced_max_size,
            induced_max_hops=ds.induced_max_hops,
            edge_max_size=ds.edge_max_size,
            require_induced_cache_hit=bool(ds.require_induced_cache_hit),
            split=app.split,
            split_root=self._split_root(),
            feature_svd_dir=ds.feature_svd_dir,
            induced_root=ds.induced_root or shared_induced_root(self.cfg),
            graph_filter_dir=ds.graph_filter_dir,
            seed=app.seed,
        )

    # -- split-level metadata ------------------------------------------------
    def _load_meta(self, app: AppSpec) -> Dict[str, Any]:
        key = app.data_key
        if key in self._meta:
            return self._meta[key]
        path = self.paths.data_meta_file(key)
        meta = None
        if path.is_file():
            meta = torch.load(path, map_location="cpu")
            if meta.get("version") != _META_VERSION:
                meta = None
        if meta is None:
            meta = self._build_meta(app)
            save_torch_atomic(str(path), meta)
        self._meta[key] = meta
        return meta

    def _build_meta(self, app: AppSpec) -> Dict[str, Any]:
        from src.data_loader import dataset_info
        from src.data_loader.dataset_metadata import get_basic_dataset_info
        from src.utils.dataset_helpers import make_workflow_loaders, resolve_effective_task_level

        level = app.task_level
        induced = level != "graph"
        # A fresh build: the node split is attached by create_dataset itself.
        dataset = self._create(app)
        train_loader, val_loader, test_loader = make_workflow_loaders(
            dataset=dataset,
            dataset_name=app.dataset,
            task_level_raw=level,
            effective_task_level=resolve_effective_task_level(level, induced),
            batch_size=int(self.cfg.moe.routergfm.device_batch_size),
            num_workers=0,
            split=app.split,
            seed=app.seed,
            induced=induced,
            split_root=self._split_root(),
        )
        support = torch.sort(_loader_positions(train_loader)).values
        query = torch.sort(_loader_positions(test_loader)).values
        if level != "edge" and len(val_loader.dataset) > 0:
            raise ValueError(f"{app.key}: few-shot split {app.split} unexpectedly has validation items.")

        info = dataset_info(dataset=dataset, task_level=level, name=app.dataset, induced=induced)
        task_type = str(info.get("task_type") or "classification").lower()
        label_dim = int(info.get("label_dim") or 1)
        family = infer_task_family(level, task_type, label_dim)

        labels = gather_labels(dataset, torch.cat([support, query]), family)
        support_labels, query_labels = labels[: support.numel()], labels[support.numel():]
        keep_s = valid_label_mask(support_labels, family)
        keep_q = valid_label_mask(query_labels, family)
        support, support_labels = support[keep_s], support_labels[keep_s]
        query, query_labels = query[keep_q], query_labels[keep_q]

        if family == LINK:
            num_classes = 2
        elif family in (MULTILABEL, REGRESSION):
            num_classes = label_dim
        else:
            observed = int(labels[labels >= 0].max().item()) + 1 if bool((labels >= 0).any()) else 0
            num_classes = max(int(info.get("num_classes") or 0), observed)

        if level == "graph":
            num_nodes, num_edges = info["avg_nodes_per_graph"], info["avg_edges_per_graph"]
        else:
            num_nodes, num_edges = info["num_nodes"], info["num_edges"]
        domain = get_basic_dataset_info(dataset).get("domain")
        if not domain or domain == "unknown":
            domain = dataset_domain(app.dataset)
        stats = build_stats(
            level=level,
            family=family,
            num_instances=len(dataset),
            num_nodes=float(num_nodes),
            num_edges=float(num_edges),
            feature_dim=int(info["num_node_features"]),
            num_classes=num_classes,
            domain=domain,
            budget=int(app.budget),
        )
        if instance_set_key(app) not in self._datasets:
            self._remember(instance_set_key(app), dataset)
        return {
            "version": _META_VERSION,
            "app": app.to_dict(),
            "level": level,
            "task_family": family,
            "task_type": task_type,
            "num_classes": int(num_classes),
            "label_dim": label_dim,
            "in_dim": int(info["num_node_features"]),
            "support_pos": support,
            "query_pos": query,
            "support_labels": support_labels,
            "query_labels": query_labels,
            "stats": stats,
        }


__all__ = [
    "AppData",
    "DOMAINS",
    "DataProvider",
    "RealDataProvider",
    "STAT_COUNT_KEYS",
    "assemble_app_data",
    "build_stats",
    "convert_labels",
    "dataset_domain",
    "derive_seed",
    "diagnostic_subsample",
    "gather_labels",
    "instance_set_key",
    "stat_feature_names",
    "stats_feature_vector",
    "valid_label_mask",
]
