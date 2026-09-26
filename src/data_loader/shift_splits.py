"""Covariate-shift split roots for the RouterGFM Table 15 comparison (our protocol).

The paper does not describe how its shift conditions were built, so this is a
documented construction, not a reproduction. Queries are real instances
re-split from the standard split universe by label-free statistics, in the
style of the GOOD covariate splits:

* ``feature``: feature norm (node: target node, graph: mean over nodes);
* ``structural``: degree of the target node in the full base graph / graph size;
* ``mixed``: both axes jointly (queries extreme on both, never seen in support).

Statistics are rank-normalized within the universe ``U`` (the standard split's
train + val + test, seed-tie-broken). The support is sampled from the low
region with the standard sampler and budget, all high-region items become the
queries (``test``), and everything else goes to ``val`` (unused; the few-shot
runners never evaluate it, and it keeps the loader's size check satisfied).
Files mirror the standard names under ``<shift.root>/<condition>/`` so every
method consumes them via ``data_preparation.dataset.split_root``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.utils import degree, to_undirected

from src.utils.checkpoint import save_torch_atomic
from src.utils.run_helpers import resolve_seeds
from src.utils.supervised_loss import binary_targets_and_valid

from .dataset_loader import make_loaders
from .dataset_metadata import is_regression_dataset, resolve_count_split_strategy
from .dataset_paths import _split_dataset_dir
from .dataset_prepare import read_datasets
from .dataset_splits import (
    _canonical_split_dataset_name,
    _is_few_shot_split_def,
    _load_existing_indices,
    _split_suffix,
)
from .dataset_storage import _get_dataset_data_storage, _unwrap_subset_dataset
from .datasets import _load_node_dataset, create_dataset, infer_task_level
from .utils import safe_torch_load

SHIFT_CONDITIONS: Tuple[str, ...] = ("feature", "structural", "mixed")
SHIFT_SPLIT_TYPE = "shift_covariate"
_AXES = ("feature", "structural")
_AXIS_SEED_OFFSET = {"feature": 0, "structural": 1}
# Mixed condition: lower the target quantile in these steps, never below the floor.
_MIXED_TARGET_STEP = 0.05
_MIXED_TARGET_FLOOR = 0.65
_STATISTIC_NAMES = {
    "node": {
        "feature": "l2 norm of the target node's features",
        "structural": "degree of the target node in the full base graph (undirected)",
    },
    "graph": {"feature": "mean node-feature l2 norm", "structural": "number of nodes"},
}
_LP_UNSUPPORTED = (
    "Link-prediction shift splits (phase 2 of the protocol) are not implemented: they need "
    "region-matched negatives and enclosing subgraphs rebuilt under a separate induced root."
)


# --------------------------------------------------------------------------- #
# Instance statistics (label-free) and regions
# --------------------------------------------------------------------------- #
def _check_level(task_level: str) -> str:
    level = str(task_level).lower()
    if level == "edge":
        raise NotImplementedError(_LP_UNSUPPORTED)
    if level not in ("node", "graph"):
        raise ValueError(f"Unsupported task level for shift splits: {task_level!r}")
    return level


def _check_condition(condition: str) -> str:
    if condition not in SHIFT_CONDITIONS:
        raise ValueError(f"Unknown shift condition {condition!r}; expected one of {SHIFT_CONDITIONS}.")
    return condition


def _graphs(dataset) -> Sequence[Any]:
    graphs = getattr(dataset, "graphs", None)
    return graphs if graphs is not None else [dataset[i] for i in range(len(dataset))]


def instance_statistics(
    dataset,
    task_level_raw: str,
    induced: bool,
    universe: Sequence[int],
    base_graph=None,
) -> Dict[str, np.ndarray]:
    """``{'feature': s_F, 'structural': s_S}`` aligned to ``universe``. Never reads labels.

    ``universe`` holds split-file ids: node ids for node tasks (``base_node_id``
    of induced instances), dataset positions for graph tasks. Node tasks need the
    full ``base_graph`` because induced ego-subgraphs truncate degree.
    """
    level = _check_level(task_level_raw)
    ids = [int(i) for i in universe]
    if level == "graph":
        feature, size = [], []
        for i in ids:
            item = dataset[i]
            x = getattr(item, "x", None)
            has_x = x is not None and x.numel() > 0
            feature.append(float(x.float().norm(dim=-1).mean()) if has_x else 0.0)
            size.append(float(item.num_nodes))
        return {"feature": np.asarray(feature, dtype=np.float64), "structural": np.asarray(size, dtype=np.float64)}

    if base_graph is None:
        raise ValueError("Node-level shift statistics need the full base graph.")
    num_nodes = int(base_graph.num_nodes)
    deg = degree(to_undirected(base_graph.edge_index, num_nodes=num_nodes)[0], num_nodes=num_nodes)
    structural = deg[torch.as_tensor(ids, dtype=torch.long)].double().numpy()
    if induced:
        by_id = {int(g.base_node_id): g for g in _graphs(dataset)}
        feature = []
        for i in ids:
            graph = by_id[i]
            target = int(torch.as_tensor(graph.target_node_index).view(-1)[0])
            feature.append(float(graph.x[target].float().norm()))
        feature = np.asarray(feature, dtype=np.float64)
    else:
        x = dataset[0].x.float()
        feature = x[torch.as_tensor(ids, dtype=torch.long)].norm(dim=-1).double().numpy()
    return {"feature": feature, "structural": structural}


def rank_normalise(stat: np.ndarray, seed: int) -> np.ndarray:
    """Ranks divided by ``n`` in ``(0, 1]``; ties broken by a seed-derived permutation."""
    values = np.asarray(stat, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Shift statistics must be finite.")
    n = values.size
    tie_break = np.random.default_rng(int(seed)).permutation(n)
    order = np.lexsort((tie_break, values))
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    return ranks / max(n, 1)


def shift_regions(
    stats: Dict[str, np.ndarray],
    condition: str,
    *,
    source_q: float,
    target_q: float,
    min_queries: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return ``(source_mask, target_mask, info)`` over the universe; disjoint by construction.

    Single-axis: source ``u <= source_q``, target ``u >= target_q``. Mixed: both
    axes in the source region / both in the target region; the target quantile is
    lowered in steps of 0.05 (floor 0.65, always above ``source_q``) until at least
    ``min_queries`` items qualify.
    """
    _check_condition(condition)
    source_q, target_q = float(source_q), float(target_q)
    if not 0.0 < source_q < target_q <= 1.0:
        raise ValueError(f"Need 0 < source_quantile < target_quantile <= 1, got {source_q}, {target_q}.")
    axes = _AXES if condition == "mixed" else (condition,)
    ranks = {axis: rank_normalise(stats[axis], int(seed) + _AXIS_SEED_OFFSET[axis]) for axis in axes}

    def region(test) -> np.ndarray:
        mask = np.ones(len(ranks[axes[0]]), dtype=bool)
        for axis in axes:
            mask &= test(ranks[axis])
        return mask

    source = region(lambda u: u <= source_q)
    q = target_q
    target = region(lambda u: u >= q)
    if condition == "mixed":
        while int(target.sum()) < int(min_queries):
            lowered = round(q - _MIXED_TARGET_STEP, 6)
            if lowered < _MIXED_TARGET_FLOOR or lowered <= source_q:
                break
            q = lowered
            target = region(lambda u: u >= q)
    if (source & target).any():
        raise RuntimeError("Shift source and target regions overlap.")
    info = {
        "source_quantile": source_q,
        "target_quantile": float(q),
        "source_region_size": int(source.sum()),
        "target_region_size": int(target.sum()),
        "min_queries": int(min_queries),
        "min_queries_met": bool(int(target.sum()) >= int(min_queries)),
    }
    return source, target, info


def sample_support(
    source_ids: Sequence[int],
    labels: Optional[torch.Tensor],
    shots: int,
    strategy: str,
    seed: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """Support from the source region with the standard samplers.

    ``balanced`` mirrors the class-balanced few-shot sampler (``shots`` per class,
    all of a class when it has fewer; ``labels`` aligned to ``source_ids``);
    ``random`` mirrors the count sampler (``shots`` items in total).
    """
    ids = [int(i) for i in source_ids]
    generator = torch.Generator().manual_seed(int(seed))
    if strategy == "random":
        perm = torch.randperm(len(ids), generator=generator).tolist() if ids else []
        support = [ids[i] for i in perm[: int(shots)]]
        return support, {"support_size": len(support)}
    if strategy != "balanced":
        raise ValueError(f"Unknown support strategy {strategy!r} (expected balanced|random).")
    y = torch.as_tensor(labels).reshape(-1).long()
    if y.numel() != len(ids):
        raise ValueError(f"{y.numel()} labels for {len(ids)} source items.")
    support: List[int] = []
    counts: Dict[int, int] = {}
    for label in torch.unique(y[y >= 0]).tolist():
        members = torch.nonzero(y == label, as_tuple=False).view(-1)
        chosen = members[torch.randperm(members.numel(), generator=generator)][: int(shots)].tolist()
        support.extend(ids[i] for i in chosen)
        counts[int(label)] = len(chosen)
    short = sorted(label for label, count in counts.items() if count < int(shots))
    return support, {"support_class_counts": counts, "short_classes": short}


# --------------------------------------------------------------------------- #
# Instance sets
# --------------------------------------------------------------------------- #
@dataclass
class ShiftSource:
    """One (dataset, task level): split-independent ids, statistics and labels."""

    name: str
    task_level: str
    induced: bool
    dataset: Any
    ids: np.ndarray  # sorted split-file ids
    stats: Dict[str, np.ndarray]  # aligned with ids
    labels: torch.Tensor  # aligned with ids: long [n] (single-label) or float [n, d]
    strategy: str  # balanced | random
    label_kind: str  # single | multilabel | regression
    featureless: bool  # no native node features: the feature statistic would be structural

    def rows(self, universe: Sequence[int]) -> np.ndarray:
        wanted = np.asarray(list(universe), dtype=np.int64)
        rows = np.searchsorted(self.ids, wanted)
        found = rows < len(self.ids)
        found[found] = self.ids[rows[found]] == wanted[found]
        if not found.all():
            raise KeyError(f"{int((~found).sum())} split ids of {self.name} have no instance.")
        return rows


def _instance_ids(dataset, level: str, induced: bool) -> np.ndarray:
    if level == "graph":
        return np.arange(len(dataset), dtype=np.int64)
    if induced:
        return np.sort(np.asarray([int(g.base_node_id) for g in _graphs(dataset)], dtype=np.int64))
    return np.arange(int(dataset[0].num_nodes), dtype=np.int64)


def _instance_labels(dataset, level: str, induced: bool, ids: np.ndarray, strategy: str) -> torch.Tensor:
    if level == "graph":
        raw = torch.stack([torch.as_tensor(dataset[int(i)].y).reshape(-1) for i in ids])
    elif induced:
        by_id = {int(g.base_node_id): g for g in _graphs(dataset)}
        raw = torch.stack([torch.as_tensor(by_id[int(i)].y).reshape(-1) for i in ids])
    else:
        y = torch.as_tensor(dataset[0].y)
        raw = y.reshape(y.size(0), -1)[torch.as_tensor(ids, dtype=torch.long)]
    raw = raw.float()
    if strategy == "balanced":
        return torch.nan_to_num(raw[:, 0], nan=-1.0).long()
    return raw


def _is_featureless(dataset, level: str, base_graph) -> bool:
    if level == "node":
        return getattr(base_graph, "x", None) is None
    storage = _get_dataset_data_storage(_unwrap_subset_dataset(dataset))
    if storage is not None:
        return getattr(storage, "x", None) is None
    return getattr(dataset[0], "x", None) is None


def make_shift_source(dataset, name: str, task_level: str, *, induced: bool, base_graph=None) -> ShiftSource:
    """Build a :class:`ShiftSource` from an already constructed dataset (+ raw base graph for nodes)."""
    level = _check_level(task_level)
    induced = bool(induced) and level == "node"
    strategy = resolve_count_split_strategy(dataset, level)
    if strategy not in ("balanced", "random"):
        raise ValueError(f"{name}: shift splits need labeled data (count-split strategy {strategy!r}).")
    if strategy == "balanced":
        label_kind = "single"
    else:
        label_kind = "regression" if is_regression_dataset(dataset, level) else "multilabel"
    ids = _instance_ids(dataset, level, induced)
    return ShiftSource(
        name=str(name),
        task_level=level,
        induced=induced,
        dataset=dataset,
        ids=ids,
        stats=instance_statistics(dataset, level, induced, ids, base_graph),
        labels=_instance_labels(dataset, level, induced, ids, strategy),
        strategy=strategy,
        label_kind=label_kind,
        featureless=_is_featureless(dataset, level, base_graph),
    )


def load_shift_source(cfg, dataset_name: str, task_level: str) -> ShiftSource:
    """Build the dataset as the consumers do (100-d features, induced node instances)."""
    level = _check_level(task_level)
    ds = cfg.data_preparation.dataset
    induced = level == "node" and bool(ds.induced)
    dataset = create_dataset(
        name=dataset_name,
        root=ds.root,
        task_level=level,
        feat_reduction=bool(ds.feat_reduction),
        feat_reduction_dim=int(ds.feat_reduction_svd_dim),
        feature_svd_dir=ds.feature_svd_dir,
        induced=induced,
        induced_min_size=int(ds.induced_min_size),
        induced_max_size=int(ds.induced_max_size),
        induced_max_hops=int(ds.induced_max_hops),
        edge_max_size=ds.edge_max_size,
        require_induced_cache_hit=bool(ds.require_induced_cache_hit),
        induced_root=ds.induced_root,
        graph_filter_dir=ds.graph_filter_dir,
        pad_featureless_features=True,
    )
    base_graph = _load_node_dataset(name=dataset_name, root=ds.root, transform=None)[0] if level == "node" else None
    return make_shift_source(dataset, dataset_name, level, induced=induced, base_graph=base_graph)


# --------------------------------------------------------------------------- #
# Split files
# --------------------------------------------------------------------------- #
def split_file_path(split_root, dataset_name: str, task_level: str, seed: int, split) -> Path:
    """Path of a (standard or shift) few-shot split file, named exactly like the standard one."""
    name = _canonical_split_dataset_name(dataset_name, str(task_level).lower(), int(seed))
    return _split_dataset_dir(Path(split_root), name) / f"{name}_splits-{_split_suffix(tuple(split))}.pt"


def _standard_split_path(source: ShiftSource, seed: int, split, split_root: str) -> Path:
    """Load (or create, as any consumer would) the standard split and return its file."""
    *_, meta = make_loaders(
        dataset=source.dataset,
        dataset_name=_canonical_split_dataset_name(source.name, source.task_level, int(seed)),
        task_level=source.task_level,
        batch_size=1,
        num_workers=0,
        split=split,
        seed=int(seed),
        induced=source.induced,
        split_root=split_root,
        return_split_meta=True,
    )
    if not meta.get("path"):
        raise ValueError(f"{source.name}: split {split} has no split file under {split_root}.")
    return Path(meta["path"])


def _class_counts(labels: torch.Tensor) -> Dict[int, int]:
    classes, counts = torch.unique(labels, return_counts=True)
    return {int(c): int(n) for c, n in zip(classes.tolist(), counts.tolist())}


def _label_diagnostics(labels: torch.Tensor, support: np.ndarray, query: np.ndarray, label_kind: str) -> Dict[str, Any]:
    """Support vs query label shift (diagnostics only; regions never read labels)."""
    s, q = torch.as_tensor(support, dtype=torch.long), torch.as_tensor(query, dtype=torch.long)
    if label_kind == "single":
        ys, yq = labels[s], labels[q]
        cs, cq = _class_counts(ys), _class_counts(yq)
        tv = float("nan")
        if ys.numel() and yq.numel():
            tv = 0.5 * sum(abs(cs.get(c, 0) / ys.numel() - cq.get(c, 0) / yq.numel()) for c in set(cs) | set(cq))
        return {"query_class_counts": cq, "label_tv_support_vs_query": float(tv)}
    if label_kind == "regression":
        center = torch.nanmean(labels, dim=0)
        spread = torch.nanmean((labels - center).pow(2), dim=0).sqrt()
        diff = (torch.nanmean(labels[q], dim=0) - torch.nanmean(labels[s], dim=0)).abs() / spread
        diff = diff[torch.isfinite(diff)]
        return {"label_mean_shift": float(diff.mean()) if diff.numel() else float("nan")}
    target, valid = binary_targets_and_valid(labels)  # whole universe: one sign convention

    def pos_rate(rows):
        observed = valid[rows].float().sum(dim=0)
        return (target[rows] * valid[rows].float()).sum(dim=0) / observed, observed > 0

    rate_s, seen_s = pos_rate(s)
    rate_q, seen_q = pos_rate(q)
    both = seen_s & seen_q
    shift = (rate_q[both] - rate_s[both]).abs()
    return {"assay_pos_rate_shift": float(shift.mean()) if shift.numel() else float("nan")}


def _stat_ranges(stats: Dict[str, np.ndarray], source: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    def span(values: np.ndarray, mask: np.ndarray):
        return [float(values[mask].min()), float(values[mask].max())] if mask.any() else None

    return {
        region: {axis: span(stats[axis], mask) for axis in _AXES}
        for region, mask in (("source", source), ("target", target))
    }


def build_shift_split(
    cfg,
    dataset_name: str,
    task_level: str,
    seed: int,
    split,
    condition: str,
    *,
    source: Optional[ShiftSource] = None,
) -> Optional[Path]:
    """Write ``<shift.root>/<condition>/<dataset>/<standard name>`` and return its path.

    Returns ``None`` (nothing written) for the feature condition on datasets
    without native node features. The written file is made read-only so a
    consumer that fails to validate it cannot silently regenerate a standard
    split in its place.
    """
    level = _check_level(task_level)
    _check_condition(condition)
    if not _is_few_shot_split_def(split):
        raise ValueError(f"Shift splits need a count-first split (shots, 0.0, 1.0); got {split}.")
    split = (int(split[0]), float(split[1]), float(split[2]))
    sh = cfg.data_preparation.shift
    source = source or load_shift_source(cfg, dataset_name, level)
    if condition == "feature" and source.featureless:
        print(f"[Shift] Skip {dataset_name} (feature): no native node features, the norm would be structural.")
        return None

    std_path = _standard_split_path(source, seed, split, cfg.data_preparation.dataset.split_root)
    std = safe_torch_load(std_path)
    universe = np.asarray(list(std["train"]) + list(std["val"]) + list(std["test"]), dtype=np.int64)
    rows = source.rows(universe)
    stats = {axis: source.stats[axis][rows] for axis in _AXES}
    labels = source.labels[torch.as_tensor(rows, dtype=torch.long)]

    in_source, in_target, info = shift_regions(
        stats,
        condition,
        source_q=float(sh.source_quantile),
        target_q=float(sh.target_quantile),
        min_queries=int(sh.min_queries),
        seed=int(seed),
    )
    source_rows = np.nonzero(in_source)[0]
    support_rows, support_info = sample_support(
        source_rows,
        labels[torch.as_tensor(source_rows, dtype=torch.long)] if source.strategy == "balanced" else None,
        split[0],
        source.strategy,
        int(seed),
    )
    support_rows = np.asarray(support_rows, dtype=np.int64)
    query_rows = np.nonzero(in_target)[0]
    unused = np.ones(len(universe), dtype=bool)
    unused[support_rows] = False
    unused[query_rows] = False

    diagnostics = dict(support_info)
    if source.label_kind == "single":
        present = set(torch.unique(labels).tolist())
        diagnostics["missing_classes"] = sorted(present - set(support_info["support_class_counts"]))
    diagnostics.update(_label_diagnostics(labels, support_rows, query_rows, source.label_kind))

    std_meta = std.get("meta") if isinstance(std.get("meta"), dict) else {}
    split_name = _canonical_split_dataset_name(dataset_name, level, int(seed))
    payload = {
        "train": [int(i) for i in universe[support_rows]],
        "val": [int(i) for i in np.sort(universe[unused])],
        "test": [int(i) for i in np.sort(universe[query_rows])],
        "meta": {
            **{key: std_meta[key] for key in ("graph_filter", "total_nodes") if key in std_meta},
            "dataset_name": split_name,
            "total": int(len(universe)),
            "split": split,
            "seed": int(seed),
            "type": SHIFT_SPLIT_TYPE,
            "condition": condition,
            "statistic": {
                axis: _STATISTIC_NAMES[level][axis] for axis in (_AXES if condition == "mixed" else (condition,))
            },
            "feature_statistic_is_structural": bool(source.featureless),
            "direction": "low_to_high",
            "universe_from": str(std_path),
            "sampling": source.strategy,
            "val_semantics": "unused",
            **info,
            "sizes": {"support": int(len(support_rows)), "query": int(len(query_rows)), "unused": int(unused.sum())},
            "stat_ranges": _stat_ranges(stats, in_source, in_target),
            **diagnostics,
        },
    }

    path = split_file_path(Path(sh.root) / condition, dataset_name, level, seed, split)
    previous = safe_torch_load(path) if path.is_file() else None
    unchanged = (
        isinstance(previous, dict)
        and (previous.get("meta") or {}).get("type") == SHIFT_SPLIT_TYPE
        and all(previous.get(key) == payload[key] for key in ("train", "val", "test"))
    )
    if not unchanged:
        save_torch_atomic(str(path), payload)
        os.chmod(path, 0o444)
    loaded = _load_existing_indices(path, len(universe))
    if loaded is None or list(loaded) != [payload["train"], payload["val"], payload["test"]]:
        raise RuntimeError(f"Shift split {path} does not pass the loader's split validation.")
    print(
        f"[Shift] {'Kept' if unchanged else 'Saved'} {condition} split {path} "
        f"(support={len(support_rows)}, query={len(query_rows)}, q={info['target_quantile']})"
    )
    return path


def verify_shift_root(split_root, entries: Sequence[Tuple[str, str, int, Sequence]]) -> None:
    """Assert every ``(dataset, task_level, seed, split)`` file under ``split_root`` is an intact shift split.

    Catches files a consumer regenerated as standard splits, files missing from
    the root (a consumer would create a standard split there), and files whose
    standard universe changed since they were built.
    """
    root_condition = Path(split_root).name
    problems: List[str] = []
    for dataset_name, task_level, seed, split in entries:
        path = split_file_path(split_root, dataset_name, task_level, seed, split)
        if not path.is_file():
            problems.append(f"missing: {path}")
            continue
        payload = safe_torch_load(path)
        meta = payload.get("meta") if isinstance(payload, dict) else None
        if not isinstance(meta, dict) or meta.get("type") != SHIFT_SPLIT_TYPE:
            problems.append(f"not a shift split (meta.type={None if meta is None else meta.get('type')!r}): {path}")
            continue
        if root_condition in SHIFT_CONDITIONS and meta.get("condition") != root_condition:
            problems.append(f"condition {meta.get('condition')!r} under the {root_condition!r} root: {path}")
        total = int(meta.get("total", -1))
        if _load_existing_indices(path, total) is None:
            problems.append(f"fails the loader's size check (total={total}): {path}")
            continue
        parts = [set(payload["train"]), set(payload["val"]), set(payload["test"])]
        if sum(len(p) for p in parts) != len(set().union(*parts)):
            problems.append(f"train/val/test overlap: {path}")
        if not parts[2]:
            problems.append(f"no queries: {path}")
        standard = Path(str(meta.get("universe_from", "")))
        if standard.is_file():
            std_meta = safe_torch_load(standard).get("meta") or {}
            if "total" in std_meta and int(std_meta["total"]) != total:
                problems.append(f"standard universe changed ({std_meta['total']} != {total}): {path}")
    if problems:
        raise ValueError("Shift split verification failed:\n  " + "\n  ".join(problems))
    print(f"[Shift] Verified {len(entries)} split files under {split_root}")


# --------------------------------------------------------------------------- #
# Data-preparation stage
# --------------------------------------------------------------------------- #
def _few_shot_splits(cfg, level: str) -> List[Tuple[int, float, float]]:
    dp = cfg.data_preparation
    defs = dp.node_task_splits if level == "node" else dp.graph_task_splits
    return [(int(s[0]), float(s[1]), float(s[2])) for s in defs if _is_few_shot_split_def(tuple(s))]


def run_shift_preparation(cfg) -> int:
    """Build and/or verify the shift split roots for ``data_preparation.target_datasets``.

    Uses the few-shot entries of ``node_task_splits`` / ``graph_task_splits`` and
    the first ``dataset.num_splits`` seeds, like the standard split generation.
    Link prediction is not covered (see ``_LP_UNSUPPORTED``).
    """
    from .runtime import normalize_targets

    dp = cfg.data_preparation
    sh = dp.shift
    targets = normalize_targets(dp.target_datasets)
    if not targets:
        print("[Shift][Error] `data_preparation.target_datasets` is required.")
        return 1
    names = targets if isinstance(targets, list) else read_datasets(str(targets))
    conditions = [_check_condition(str(c)) for c in sh.conditions]
    seeds = resolve_seeds(cfg, requested_count=int(dp.dataset.num_splits or 5))
    override = str(dp.task_level_override or "").strip().lower()

    plan = []
    for name in names:
        level = override or infer_task_level(name)
        if level not in ("node", "graph"):
            print(f"[Shift] Skip {name}: task level {level!r} ({'LP is phase 2' if level == 'edge' else 'unknown'}).")
            continue
        plan.append((name, level, _few_shot_splits(cfg, level)))

    status = 0
    if bool(sh.build):
        for name, level, splits in plan:
            try:
                source = load_shift_source(cfg, name, level)
                for seed in seeds:
                    for split in splits:
                        for condition in conditions:
                            build_shift_split(cfg, name, level, seed, split, condition, source=source)
            except Exception as exc:  # keep going with the other datasets, like the standard prep
                print(f"[Shift][Error] {name} ({level}): {exc}")
                status = 1
    if bool(sh.verify):
        for condition in conditions:
            entries = [(name, level, seed, split) for name, level, splits in plan for seed in seeds for split in splits]
            try:
                verify_shift_root(Path(sh.root) / condition, entries)
            except ValueError as exc:
                print(f"[Shift][Error] {exc}")
                status = 1
    return status


__all__ = [
    "SHIFT_CONDITIONS",
    "SHIFT_SPLIT_TYPE",
    "ShiftSource",
    "build_shift_split",
    "instance_statistics",
    "load_shift_source",
    "make_shift_source",
    "rank_normalise",
    "run_shift_preparation",
    "sample_support",
    "shift_regions",
    "split_file_path",
    "verify_shift_root",
]


if __name__ == "__main__":
    from .run import build_data_preparation_cfg

    raise SystemExit(run_shift_preparation(build_data_preparation_cfg(sys.argv[1:])))
