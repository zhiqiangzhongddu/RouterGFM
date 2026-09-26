#!/usr/bin/env python3
"""
Convert IcG datasets into AnyGraph-native matrix format.

Outputs per dataset:
- link task:
  - trn_mat.pkl / val_mat.pkl / tst_mat.pkl   (N x N sparse matrices)
  - feats.pkl                                  (optional, N x F)
- node task:
  - trn_mat.pkl                                (N+C x N+C sparse, graph + train label edges)
  - val_mat.pkl / tst_mat.pkl                  (N+C x C sparse label matrices)
  - feats.pkl                                  (optional, (N+C) x F, with class-node features)

Only datasets that resolve to single-graph node/edge tasks are converted.
"""

import argparse
import ast
import json
import pickle
import sys
import traceback
from datetime import datetime
from numbers import Integral
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[3]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from src.data_loader.datasets import create_dataset, infer_task_level  # noqa: E402
from src.utils import (
    format_split_for_name,
    parse_csv_list,
    project_path,
    read_name_list_file,
    resolve_project_path,
)  # noqa: E402


def _parse_split(raw: str) -> Tuple[float, float, float]:
    vals = tuple(float(x.strip()) for x in raw.split(","))
    if len(vals) != 3:
        raise ValueError(f"Invalid split '{raw}'. Expected 'a,b,c'.")
    return vals  # type: ignore[return-value]


def _parse_split_value(value) -> Tuple[float, float, float]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Split definition cannot be empty.")
        try:
            value = ast.literal_eval(text)
        except Exception:
            return _parse_split(text)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Invalid split definition '{value}'. Expected 3 values.")
    return tuple(float(x) for x in value)  # type: ignore[return-value]


def _parse_split_list(raw: str) -> List[Tuple[float, float, float]]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        chunks = [chunk.strip() for chunk in text.split(";") if chunk.strip()]
        if len(chunks) > 1:
            return [_parse_split(chunk) for chunk in chunks]
        return [_parse_split(text)]
    if isinstance(parsed, (list, tuple)):
        if len(parsed) == 0:
            return []
        if len(parsed) == 3 and not isinstance(parsed[0], (list, tuple)):
            return [_parse_split_value(parsed)]
        return [_parse_split_value(item) for item in parsed]
    return [_parse_split_value(parsed)]


def _parse_int_list(raw: str) -> List[int]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = parse_csv_list(text)
    if isinstance(parsed, (list, tuple, set)):
        return [int(tok) for tok in parsed]
    return [int(parsed)]


def _add_bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str = "") -> None:
    """
    Python 3.6-compatible replacement for argparse.BooleanOptionalAction.
    Creates paired flags: --<name> / --no-<name>.
    """
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--" + name, dest=dest, action="store_true", help=help_text)
    group.add_argument("--no-" + name, dest=dest, action="store_false")
    parser.set_defaults(**{dest: bool(default)})


def _split_suffix(portions: Tuple[float, float, float]) -> str:
    return "-".join(str(int(round(float(p) * 100))) for p in portions)


def _is_few_shot_split(split_def: Tuple[float, float, float]) -> bool:
    if not isinstance(split_def, (list, tuple)) or len(split_def) < 3:
        return False
    first = split_def[0]
    if isinstance(first, bool):
        return False
    try:
        first_val = float(first)
        shots_like = first_val.is_integer() and first_val >= 1.0
    except Exception:
        shots_like = isinstance(first, Integral) and not isinstance(first, bool)
    if not shots_like:
        return False
    try:
        val_ratio = float(split_def[1])
        test_ratio = float(split_def[2])
    except Exception:
        return False
    return val_ratio >= 0.0 and test_ratio >= 0.0 and (val_ratio + test_ratio) > 0.0


def _normalized_split_name(split: Tuple[float, float, float]) -> str:
    parts = tuple(float(item) for item in split)
    if _is_few_shot_split(parts):
        parts = (int(round(parts[0])), float(parts[1]), float(parts[2]))
    return str(format_split_for_name(parts))


def _split_file_tag(split: Tuple[float, float, float]) -> str:
    """Canonical suffix of ``*_splits-<suffix>.pt`` files on disk.

    Mirrors src/data_loader/dataset_splits._split_suffix: few-shot uses
    ``<shots>-<val_pct>-<test_pct>``, ratio uses
    ``<train_pct>-<val_pct>-<test_pct>``. The ``fewshot`` / ``split``
    display prefixes only appear in alias strings, never in filenames.
    """
    parts = tuple(float(item) for item in split)
    if _is_few_shot_split(parts):
        return f"{int(round(parts[0]))}-{int(round(parts[1] * 100))}-{int(round(parts[2] * 100))}"
    return "-".join(str(int(round(part * 100))) for part in parts)


def _split_alias_tag(split: Tuple[float, float, float]) -> str:
    split_name = _normalized_split_name(split)
    if split_name.startswith("split"):
        return f"split-{split_name[len('split') :]}"
    return split_name


def _converted_dataset_name(dataset_name: str, split: Tuple[float, float, float], seed: int) -> str:
    return f"{dataset_name}_seed{int(seed)}_{_split_alias_tag(split)}"


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _row_l1_normalize(feats: np.ndarray) -> np.ndarray:
    if feats.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape={feats.shape}")
    feats = feats.astype(np.float32, copy=False)
    row_sum = feats.sum(axis=1, keepdims=True)
    nonzero = row_sum.squeeze(-1) > 0
    if np.any(nonzero):
        feats[nonzero] = feats[nonzero] / row_sum[nonzero]
    return feats


def _reduce_features_svd(feats: np.ndarray, out_dim: int) -> np.ndarray:
    if out_dim <= 0:
        return feats.astype(np.float32, copy=False)
    feats_t = torch.as_tensor(feats)
    if feats_t.dim() != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape={tuple(feats_t.shape)}")
    in_dim = int(feats_t.size(1))
    if in_dim == out_dim:
        return feats_t.detach().cpu().numpy().astype(np.float32, copy=False)
    if in_dim < out_dim:
        x_f = feats_t.to(torch.float32) if not torch.is_floating_point(feats_t) else feats_t
        pad = x_f.new_zeros(x_f.size(0), out_dim - in_dim)
        out = torch.cat([x_f, pad], dim=1)
        return out.detach().cpu().numpy().astype(np.float32, copy=False)
    try:
        u, s, _ = torch.linalg.svd(feats_t.float(), full_matrices=False)
        reduced = u[:, :out_dim] * s[:out_dim]
        return reduced.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception as exc:
        raise RuntimeError(
            f"SVD feature reduction failed for shape={tuple(feats_t.shape)} out_dim={out_dim}"
        ) from exc


def _to_1d_label(y: torch.Tensor) -> np.ndarray:
    if y is None:
        raise ValueError("Missing labels (y is None).")
    y_t = torch.as_tensor(y)
    if y_t.dim() > 1:
        if y_t.size(-1) != 1:
            raise ValueError(f"Expected single-label y, got shape={tuple(y_t.shape)}")
        y_t = y_t.view(-1)
    return y_t.detach().cpu().to(torch.long).numpy()


def _mask_to_bool(mask: Optional[torch.Tensor], n: int, mask_col: int) -> np.ndarray:
    if mask is None:
        return np.zeros((n,), dtype=bool)
    m = torch.as_tensor(mask).detach().cpu()
    if m.dtype != torch.bool:
        m = m.to(torch.bool)
    if m.dim() == 1:
        if m.numel() != n:
            raise ValueError(f"Mask length mismatch: got {m.numel()} expected {n}")
        return m.numpy()
    if m.dim() == 2:
        if m.size(0) != n:
            raise ValueError(f"2D mask first dim mismatch: got {m.size(0)} expected {n}")
        col = max(0, min(mask_col, m.size(1) - 1))
        return m[:, col].numpy()
    raise ValueError(f"Unsupported mask shape: {tuple(m.shape)}")


def _coo_from_edges(rows: np.ndarray, cols: np.ndarray, shape: Tuple[int, int]) -> sp.coo_matrix:
    if rows.size == 0:
        return sp.coo_matrix(shape, dtype=np.float32)
    data = np.ones(rows.shape[0], dtype=np.float32)
    mat = sp.coo_matrix((data, (rows, cols)), shape=shape, dtype=np.float32)
    mat.sum_duplicates()
    mat.data[:] = 1.0
    return mat


def _edge_index_from_data(data) -> np.ndarray:
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        raise ValueError("Dataset has no edge_index.")
    ei = torch.as_tensor(edge_index).detach().cpu().to(torch.long)
    if ei.dim() != 2 or ei.size(0) != 2:
        raise ValueError(f"Invalid edge_index shape: {tuple(ei.shape)}")
    return ei.numpy()


def _save_sparse(path: Path, mat: sp.coo_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(mat.tocoo(), f, protocol=pickle.HIGHEST_PROTOCOL)


def _save_feats(path: Path, feats: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(feats.astype(np.float32, copy=False), f, protocol=pickle.HIGHEST_PROTOCOL)


def _validate_single_graph_dataset(ds, name: str, task_level: str) -> None:
    if len(ds) != 1:
        raise ValueError(
            f"Only single-graph {task_level} datasets are supported for AnyGraph conversion; "
            f"dataset={name} len={len(ds)}"
        )


def _unique_sorted_edges(edge_index: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    if edge_index.shape[1] == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    rows = edge_index[0].astype(np.int64, copy=False)
    cols = edge_index[1].astype(np.int64, copy=False)
    keep = (rows >= 0) & (cols >= 0) & (rows < num_nodes) & (cols < num_nodes)
    rows = rows[keep]
    cols = cols[keep]
    if rows.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    hashed = rows * int(num_nodes) + cols
    uniq = np.unique(hashed.astype(np.int64, copy=False))
    out_rows = (uniq // int(num_nodes)).astype(np.int64, copy=False)
    out_cols = (uniq % int(num_nodes)).astype(np.int64, copy=False)
    return out_rows, out_cols


def _as_int_list(values) -> Optional[List[int]]:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return [int(v) for v in values.view(-1).tolist()]
    if isinstance(values, np.ndarray):
        return [int(v) for v in values.reshape(-1).tolist()]
    if isinstance(values, (list, tuple)):
        return [int(v) for v in values]
    return None


def _as_edge_index(values) -> Optional[np.ndarray]:
    if values is None:
        return np.empty((2, 0), dtype=np.int64)
    edge_index = torch.as_tensor(values, dtype=torch.long)
    if edge_index.numel() == 0:
        return np.empty((2, 0), dtype=np.int64)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        return None
    return edge_index.detach().cpu().numpy().astype(np.int64, copy=False)


def _lookup_payload(payload: Mapping[str, object], *keys: str):
    for key in keys:
        if key in payload:
            return payload.get(key)
    return None


def _dedup_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _split_roots(split_root: Path, dataset_name: str) -> List[Path]:
    roots = [split_root, split_root / dataset_name]
    existing = [p for p in roots if p.is_dir()]
    missing = [p for p in roots if not p.is_dir()]
    return _dedup_paths(existing + missing)


def _find_split_file(
    roots: Sequence[Path],
    names: Sequence[str],
    globs: Sequence[str],
    *,
    split_kind: str,
) -> Optional[Path]:
    for name in names:
        matches_for_name: List[Path] = []
        for root in roots:
            path = root / name
            if path.is_file():
                matches_for_name.append(path)
        matches_for_name = _dedup_paths(matches_for_name)
        if not matches_for_name:
            continue
        if len(matches_for_name) == 1:
            return matches_for_name[0]
        raise RuntimeError(
            f"Ambiguous exact {split_kind} split files found for candidate '{name}': "
            f"{[str(p) for p in matches_for_name]}. Please keep only one exact match."
        )

    fallback_matches: List[Path] = []
    for root in roots:
        for pattern in globs:
            fallback_matches.extend(sorted(root.glob(pattern)))
    fallback_matches = _dedup_paths(fallback_matches)
    if fallback_matches:
        raise FileNotFoundError(
            f"No exact {split_kind} split file found; non-exact candidates exist: "
            f"{[str(p) for p in fallback_matches]}. "
            "Please rename to one of the expected exact filenames."
        )
    return None


def _resolve_node_split_file(
    *,
    split_root: Path,
    dataset_name: str,
    split: Tuple[float, float, float],
    seed: int,
) -> Optional[Path]:
    suffix = _split_file_tag(split)
    roots = _split_roots(split_root, dataset_name)
    names = [
        f"{dataset_name}_node_seed{int(seed)}_splits-{suffix}.pt",
        f"{dataset_name}_seed{int(seed)}_splits-{suffix}.pt",
        f"{dataset_name}_splits-{suffix}.pt",
    ]
    globs = [f"{dataset_name}*_splits-{suffix}.pt"]
    return _find_split_file(roots, names, globs, split_kind="node")


def _resolve_edge_split_file(
    *,
    split_root: Path,
    dataset_name: str,
    split: Tuple[float, float, float],
    seed: int,
) -> Optional[Path]:
    suffix = _split_suffix(split)
    neg_pct = int(round(float(split[0]) * 100))
    edge_name = dataset_name if dataset_name.endswith("_edge") else f"{dataset_name}_edge"
    seeded_edge_name = f"{edge_name}_seed{int(seed)}"
    roots = _split_roots(split_root, dataset_name)
    names = [
        f"{seeded_edge_name}_splits-{suffix}.pt",
        f"{seeded_edge_name}_splits-pos{suffix}-neg{neg_pct}.pt",
        f"{edge_name}_splits-{suffix}.pt",
        f"{edge_name}_splits-pos{suffix}-neg{neg_pct}.pt",
    ]
    return _find_split_file(roots, names, (), split_kind="edge")


def _load_node_split_indices(path: Path, num_nodes: int) -> Tuple[List[int], List[int], List[int]]:
    payload = _safe_torch_load(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Node split payload is not a dict: {path}")

    train_idx = _as_int_list(_lookup_payload(payload, "train_indices", "train"))
    val_idx = _as_int_list(_lookup_payload(payload, "val_indices", "val"))
    test_idx = _as_int_list(_lookup_payload(payload, "test_indices", "test"))
    if train_idx is None or val_idx is None or test_idx is None:
        raise ValueError(f"Node split payload missing train/val/test indices: {path}")

    def _validate(indices: List[int], tag: str) -> List[int]:
        out = [int(idx) for idx in indices]
        bad = [idx for idx in out if idx < 0 or idx >= int(num_nodes)]
        if bad:
            raise ValueError(f"Node split file contains out-of-range {tag} indices in {path}: sample={bad[:5]}")
        return out

    train_idx = _validate(train_idx, "train")
    val_idx = _validate(val_idx, "val")
    test_idx = _validate(test_idx, "test")
    return train_idx, val_idx, test_idx


def _resolve_graph_split_file(
    *,
    split_root: Path,
    dataset_name: str,
    split: Tuple[float, float, float],
    seed: int,
) -> Optional[Path]:
    suffix = _split_file_tag(split)
    roots = _split_roots(split_root, dataset_name)
    names = [
        f"{dataset_name}_graph_seed{int(seed)}_splits-{suffix}.pt",
        f"{dataset_name}_seed{int(seed)}_splits-{suffix}.pt",
        f"{dataset_name}_splits-{suffix}.pt",
    ]
    globs = [f"{dataset_name}*_splits-{suffix}.pt"]
    return _find_split_file(roots, names, globs, split_kind="graph")


def _load_graph_split_indices(path: Path, num_graphs: int) -> Tuple[List[int], List[int], List[int]]:
    """Load train/val/test graph indices (over the empty-graph-filtered ordering)."""
    payload = _safe_torch_load(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Graph split payload is not a dict: {path}")

    train_idx = _as_int_list(_lookup_payload(payload, "train_indices", "train"))
    val_idx = _as_int_list(_lookup_payload(payload, "val_indices", "val"))
    test_idx = _as_int_list(_lookup_payload(payload, "test_indices", "test"))
    if train_idx is None or val_idx is None or test_idx is None:
        raise ValueError(f"Graph split payload missing train/val/test indices: {path}")

    def _validate(indices: List[int], tag: str) -> List[int]:
        out = [int(idx) for idx in indices]
        bad = [idx for idx in out if idx < 0 or idx >= int(num_graphs)]
        if bad:
            raise ValueError(
                f"Graph split file contains out-of-range {tag} indices in {path} "
                f"(num_graphs={num_graphs}): sample={bad[:5]}. The empty-graph filter "
                "ordering used at conversion must match the one used when the split was created."
            )
        return out

    return _validate(train_idx, "train"), _validate(val_idx, "val"), _validate(test_idx, "test")


def _graph_label_array(g) -> Optional[np.ndarray]:
    """Return a graph's label as a 1-D float array (K,), or None if absent."""
    y = getattr(g, "y", None)
    if y is None:
        return None
    t = torch.as_tensor(y).detach().cpu()
    if t.numel() == 0:
        return None
    return t.reshape(-1).to(torch.float64).numpy()


def _infer_graph_task_family(Y: np.ndarray) -> Tuple[str, int]:
    """Classify a (G, K) graph-label matrix into a task family.

    Returns ``(family, label_dim)`` where family is one of
    ``single_label`` (K==1 integer labels, no missing), ``multilabel``
    (K>=1 binary 0/1 labels, possibly missing), or ``regression``
    (continuous targets). ``label_dim`` is K for multilabel/regression and
    1 for single_label.
    """
    if Y.ndim != 2:
        raise ValueError(f"Expected 2-D label matrix, got shape={Y.shape}")
    num_targets = int(Y.shape[1])
    finite = Y[np.isfinite(Y)]
    if finite.size == 0:
        raise ValueError("No finite graph labels found.")
    integerish = bool(np.allclose(finite, np.round(finite)))
    rounded = np.round(finite)
    only01 = bool(integerish and np.all((rounded == 0) | (rounded == 1)))
    has_nan = bool(np.isnan(Y).any())

    if num_targets > 1:
        if only01:
            return "multilabel", num_targets
        return "regression", num_targets
    # single target column
    if integerish and not has_nan:
        return "single_label", 1
    return "regression", 1


def _subsample_graph_splits(
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    node_counts: Sequence[int],
    *,
    max_graphs: int,
    max_total_nodes: int,
    seed: int,
) -> Tuple[List[int], List[int], List[int], Dict[str, int]]:
    """Seeded, split-preserving subsampling for the scale guard.

    Returns filtered (train, val, test) original-index lists plus a report of
    how many graphs were dropped. Never drops silently: callers log the report.
    """
    splits = {"train": list(train_idx), "val": list(val_idx), "test": list(test_idx)}
    kept = {k: list(v) for k, v in splits.items()}
    rng = np.random.default_rng(int(seed))

    def _total_kept_graphs() -> int:
        return sum(len(v) for v in kept.values())

    def _total_kept_nodes() -> int:
        return sum(int(node_counts[i]) for v in kept.values() for i in v)

    # Cap by graph count: subsample each split proportionally.
    if max_graphs > 0 and _total_kept_graphs() > max_graphs:
        total = _total_kept_graphs()
        for k, v in kept.items():
            if not v:
                continue
            quota = max(1, int(round(len(v) * max_graphs / total)))
            quota = min(quota, len(v))
            sel = rng.choice(len(v), size=quota, replace=False)
            kept[k] = [v[i] for i in sorted(sel.tolist())]

    # Cap by total node budget: drop whole graphs (largest-first within a
    # seeded shuffle) until under budget, preserving at least one per split.
    if max_total_nodes > 0 and _total_kept_nodes() > max_total_nodes:
        order = [(k, i) for k in kept for i in kept[k]]
        perm = rng.permutation(len(order)).tolist()
        budget = max_total_nodes
        chosen = {"train": [], "val": [], "test": []}
        running = 0
        for j in perm:
            k, i = order[j]
            n = int(node_counts[i])
            if running + n <= budget:
                chosen[k].append(i)
                running += n
        # guarantee non-empty splits where the original was non-empty
        for k, v in kept.items():
            if v and not chosen[k]:
                pick = v[int(rng.integers(len(v)))]
                chosen[k].append(pick)
        kept = {k: sorted(v) for k, v in chosen.items()}

    report = {
        "train_dropped": len(splits["train"]) - len(kept["train"]),
        "val_dropped": len(splits["val"]) - len(kept["val"]),
        "test_dropped": len(splits["test"]) - len(kept["test"]),
    }
    return kept["train"], kept["val"], kept["test"], report


def _load_edge_split_payload(path: Path, total_edges: int) -> Dict[str, np.ndarray]:
    payload = _safe_torch_load(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Edge split payload is not a dict: {path}")

    meta = payload.get("meta") or {}
    if int(meta.get("format_version", 1)) != 2:
        raise ValueError(
            "Edge split payload uses the legacy directed-slot format (pre-v2), "
            "which leaks reverse directions of held-out edges into the training "
            f"context. Delete and regenerate the split: {path}"
        )

    out: Dict[str, np.ndarray] = {}
    for key in ("train_pos_idx", "val_pos_idx", "test_pos_idx", "message_pos_idx", "context_pos_idx"):
        idx = _as_int_list(payload.get(key))
        if idx is None:
            raise ValueError(f"Edge split payload missing key={key}: {path}")
        out[key] = np.asarray(idx, dtype=np.int64)
        if out[key].size > 0:
            if int(out[key].min()) < 0 or int(out[key].max()) >= int(total_edges):
                raise ValueError(f"Edge split payload has out-of-range indices for key={key}: {path}")

    # v2 splits assign unordered pairs to partitions: supervision lists hold
    # one canonical slot per pair, the reverse slots of val/test pairs are
    # excluded entirely, and context = train + message slots. The supervision
    # lists must still be pairwise disjoint.
    merged = np.concatenate(
        [
            out["train_pos_idx"],
            out["val_pos_idx"],
            out["test_pos_idx"],
            out["message_pos_idx"],
        ],
        axis=0,
    ).astype(np.int64, copy=False)
    if int(np.unique(merged).size) != int(merged.size):
        raise ValueError(
            f"Edge split payload has overlapping train/val/test/message indices in {path}"
        )

    for key, meta_key in (
        ("train_pos_idx", "train_positives"),
        ("val_pos_idx", "val_positives"),
        ("test_pos_idx", "test_positives"),
    ):
        if int(meta.get(meta_key, -1)) != int(out[key].size):
            raise ValueError(
                f"Edge split payload count mismatch for key={key} "
                f"(meta says {meta.get(meta_key)}, payload has {out[key].size}): {path}"
            )

    for key in ("train_neg_edge_index", "val_neg_edge_index", "test_neg_edge_index"):
        edge_index = _as_edge_index(payload.get(key))
        if edge_index is None:
            raise ValueError(f"Invalid edge index tensor for key={key}: {path}")
        out[key] = edge_index

    for key, meta_key, pos_key in (
        ("train_neg_edge_index", "train_negatives", "train_pos_idx"),
        ("val_neg_edge_index", "val_negatives", "val_pos_idx"),
        ("test_neg_edge_index", "test_negatives", "test_pos_idx"),
    ):
        if int(meta.get(meta_key, -1)) != int(out[key].shape[1]):
            raise ValueError(
                f"Edge split payload negative-count mismatch for key={key}: {path}"
            )
        if int(out[key].shape[1]) > int(out[pos_key].size):
            raise ValueError(
                f"Edge split payload has more negatives than positives for key={key}: {path}"
            )

    return out


class ConvertResult:
    def __init__(
        self,
        dataset: str,
        task: str,
        status: str,
        source_dataset: str = "",
        split: Optional[Tuple[float, float, float]] = None,
        seed: int = 0,
        out_dir: str = "",
        message: str = "",
        num_nodes: int = 0,
        num_edges_train: int = 0,
        num_edges_val: int = 0,
        num_edges_test: int = 0,
        num_classes: int = 0,
        num_graphs: int = 0,
        task_family: str = "",
    ):
        self.dataset = dataset
        self.task = task
        self.status = status
        self.source_dataset = source_dataset or dataset
        self.split = tuple(float(item) for item in split) if split is not None else None
        self.seed = int(seed)
        self.out_dir = out_dir
        self.message = message
        self.num_nodes = int(num_nodes)
        self.num_edges_train = int(num_edges_train)
        self.num_edges_val = int(num_edges_val)
        self.num_edges_test = int(num_edges_test)
        self.num_classes = int(num_classes)
        self.num_graphs = int(num_graphs)
        self.task_family = str(task_family)

    def to_dict(self) -> Dict[str, object]:
        split_tag = _split_alias_tag(self.split) if self.split is not None else ""
        return {
            "dataset": self.dataset,
            "source_dataset": self.source_dataset,
            "split": list(self.split) if self.split is not None else [],
            "split_tag": split_tag,
            "seed": self.seed,
            "task": self.task,
            "status": self.status,
            "out_dir": self.out_dir,
            "message": self.message,
            "num_nodes": self.num_nodes,
            "num_edges_train": self.num_edges_train,
            "num_edges_val": self.num_edges_val,
            "num_edges_test": self.num_edges_test,
            "num_classes": self.num_classes,
            "num_graphs": self.num_graphs,
            "task_family": self.task_family,
        }


class AnyGraphConverter:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dataset_root = resolve_project_path(args.dataset_root)
        self.split_root = resolve_project_path(args.split_root)
        self.out_root = resolve_project_path(args.out_root)
        self.link_root = self.out_root / "link"
        self.node_root = self.out_root / "node"
        self.graph_root = self.out_root / "graph"
        if not self.split_root.exists():
            raise FileNotFoundError(f"Split root does not exist: {self.split_root}")
        if not self.split_root.is_dir():
            raise NotADirectoryError(f"Split root is not a directory: {self.split_root}")
        self.link_root.mkdir(parents=True, exist_ok=True)
        self.node_root.mkdir(parents=True, exist_ok=True)
        self.graph_root.mkdir(parents=True, exist_ok=True)

    def _reset_output_dir(self, out_dir: Path) -> None:
        if out_dir.is_dir():
            for item in out_dir.iterdir():
                if item.is_file():
                    item.unlink()
        else:
            out_dir.mkdir(parents=True, exist_ok=True)

    def convert_link(self, dataset_name: str, split: Tuple[float, float, float], seed: int, dataset_alias: str) -> ConvertResult:
        try:
            ds = create_dataset(
                name=dataset_name,
                root=str(self.dataset_root),
                task_level="edge",
                induced=False,
                feat_reduction=bool(self.args.feat_reduction),
                feat_reduction_dim=int(self.args.feat_dim),
            )
            _validate_single_graph_dataset(ds, dataset_name, "edge")
            data = ds[0]
            num_nodes = int(getattr(data, "num_nodes", 0) or 0)
            if num_nodes <= 0:
                raise ValueError("Invalid num_nodes from dataset.")
            edge_index = _edge_index_from_data(data)
            total_edges = int(edge_index.shape[1])
            split_path = _resolve_edge_split_file(
                split_root=self.split_root,
                dataset_name=dataset_name,
                split=split,
                seed=int(seed),
            )
            if split_path is None:
                raise FileNotFoundError(
                    f"Edge split payload not found for dataset={dataset_name}, split={split}, seed={seed}, "
                    f"split_root={self.split_root}"
                )
            split_payload = _load_edge_split_payload(split_path, total_edges=total_edges)

            all_rows = edge_index[0].astype(np.int64, copy=False)
            all_cols = edge_index[1].astype(np.int64, copy=False)

            def _rows_cols_from_indices(indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                if indices.size == 0:
                    return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
                if int(indices.max()) >= total_edges or int(indices.min()) < 0:
                    raise ValueError(f"Found edge index out of range in split payload: {split_path}")
                return all_rows[indices], all_cols[indices]

            # v2 payloads provide the full training context (all directions of
            # train + message pairs; never any direction of val/test pairs).
            train_all_idx = split_payload["context_pos_idx"]

            trn_rows, trn_cols = _rows_cols_from_indices(train_all_idx)
            val_rows, val_cols = _rows_cols_from_indices(split_payload["val_pos_idx"])
            tst_rows, tst_cols = _rows_cols_from_indices(split_payload["test_pos_idx"])

            trn_mat = _coo_from_edges(trn_rows, trn_cols, (num_nodes, num_nodes))
            val_mat = _coo_from_edges(val_rows, val_cols, (num_nodes, num_nodes))
            tst_mat = _coo_from_edges(tst_rows, tst_cols, (num_nodes, num_nodes))

            out_dir = self.link_root / dataset_alias
            self._reset_output_dir(out_dir)
            _save_sparse(out_dir / "trn_mat.pkl", trn_mat)
            _save_sparse(out_dir / "val_mat.pkl", val_mat)
            _save_sparse(out_dir / "tst_mat.pkl", tst_mat)
            torch.save(
                {
                    "val_pos_edge_index": np.stack([val_rows, val_cols], axis=0).astype(np.int64, copy=False),
                    "val_neg_edge_index": split_payload["val_neg_edge_index"].astype(np.int64, copy=False),
                    "test_pos_edge_index": np.stack([tst_rows, tst_cols], axis=0).astype(np.int64, copy=False),
                    "test_neg_edge_index": split_payload["test_neg_edge_index"].astype(np.int64, copy=False),
                    "meta": {
                        "dataset": dataset_alias,
                        "source_dataset": dataset_name,
                        "split_file": str(split_path),
                        "edge_split": tuple(float(v) for v in split),
                        "split_seed": int(seed),
                    },
                },
                out_dir / str(self.args.edge_eval_payload_name),
            )

            if getattr(data, "x", None) is not None:
                feats = torch.as_tensor(data.x).detach().cpu().numpy().astype(np.float32, copy=False)
                if feats.shape[0] == num_nodes:
                    if bool(self.args.l1_normalize_features):
                        feats = _row_l1_normalize(feats)
                    _save_feats(out_dir / "feats.pkl", feats)

            meta = {
                "dataset": dataset_alias,
                "source_dataset": dataset_name,
                "task": "link",
                "source": "agae_conversion",
                "num_nodes": num_nodes,
                "num_edges_total": total_edges,
                "num_edges_train": int(trn_mat.nnz),
                "num_edges_val": int(val_mat.nnz),
                "num_edges_test": int(tst_mat.nnz),
                "edge_split": list(split),
                "split_seed": int(seed),
                "split_strategy": "edge_payload_train_plus_message",
                "split_file": str(split_path),
                "edge_eval_payload": str(out_dir / str(self.args.edge_eval_payload_name)),
            }
            (out_dir / "conversion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return ConvertResult(
                dataset=dataset_alias,
                source_dataset=dataset_name,
                split=split,
                seed=seed,
                task="link",
                status="ok",
                out_dir=str(out_dir),
                num_nodes=num_nodes,
                num_edges_train=int(trn_mat.nnz),
                num_edges_val=int(val_mat.nnz),
                num_edges_test=int(tst_mat.nnz),
            )
        except Exception as exc:  # pylint: disable=broad-except
            return ConvertResult(
                dataset=dataset_alias,
                source_dataset=dataset_name,
                split=split,
                seed=seed,
                task="link",
                status="fail",
                message=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )

    def convert_node(self, dataset_name: str, split: Tuple[float, float, float], seed: int, dataset_alias: str) -> ConvertResult:
        try:
            ds = create_dataset(
                name=dataset_name,
                root=str(self.dataset_root),
                task_level="node",
                induced=False,
                feat_reduction=bool(self.args.feat_reduction),
                feat_reduction_dim=int(self.args.feat_dim),
            )
            _validate_single_graph_dataset(ds, dataset_name, "node")

            data = ds[0]
            num_nodes = int(getattr(data, "num_nodes", 0) or 0)
            if num_nodes <= 0:
                raise ValueError("Invalid num_nodes from dataset.")

            y = _to_1d_label(data.y)
            valid = y >= 0
            if not np.any(valid):
                raise ValueError("No non-negative labels found.")
            num_classes = int(y[valid].max() + 1)
            total_nodes = num_nodes + num_classes

            split_path = _resolve_node_split_file(
                split_root=self.split_root,
                dataset_name=dataset_name,
                split=split,
                seed=int(seed),
            )
            if split_path is None:
                raise FileNotFoundError(
                    f"Node split file not found for dataset={dataset_name}, split={split}, seed={seed}, "
                    f"split_root={self.split_root}"
                )
            if not bool(self.args.emit_node_val):
                raise ValueError("Strict node conversion requires --emit_node_val to keep val/test separated.")

            split_train_idx, split_val_idx, split_test_idx = _load_node_split_indices(split_path, num_nodes)
            for tag, indices in (
                ("train", split_train_idx),
                ("val", split_val_idx),
                ("test", split_test_idx),
            ):
                if len(indices) == 0:
                    continue
                idx_np = np.asarray(indices, dtype=np.int64)
                if not bool(np.all(valid[idx_np])):
                    raise ValueError(
                        f"Node split contains unlabeled or invalid {tag} indices for dataset={dataset_name}: {split_path}"
                    )

            train_mask = np.zeros((num_nodes,), dtype=bool)
            val_mask = np.zeros((num_nodes,), dtype=bool)
            test_mask = np.zeros((num_nodes,), dtype=bool)
            train_mask[np.asarray(split_train_idx, dtype=np.int64)] = True
            val_mask[np.asarray(split_val_idx, dtype=np.int64)] = True
            test_mask[np.asarray(split_test_idx, dtype=np.int64)] = True
            mask_source = "split_ckpt"
            coverage = float(np.sum(train_mask | val_mask | test_mask)) / float(max(np.sum(valid), 1))

            edge_index = _edge_index_from_data(data)
            keep = (
                (edge_index[0] >= 0)
                & (edge_index[1] >= 0)
                & (edge_index[0] < num_nodes)
                & (edge_index[1] < num_nodes)
            )
            graph_edges = edge_index[:, keep]

            trn_rows: List[np.ndarray] = [graph_edges[0].astype(np.int64, copy=False)]
            trn_cols: List[np.ndarray] = [graph_edges[1].astype(np.int64, copy=False)]

            train_nodes = np.where(train_mask)[0].astype(np.int64, copy=False)
            if train_nodes.size > 0:
                train_labels = y[train_nodes].astype(np.int64, copy=False)
                trn_rows.append(train_nodes)
                trn_cols.append((num_nodes + train_labels).astype(np.int64, copy=False))
                trn_rows.append((num_nodes + train_labels).astype(np.int64, copy=False))
                trn_cols.append(train_nodes)

            trn_row = np.concatenate(trn_rows) if trn_rows else np.empty((0,), dtype=np.int64)
            trn_col = np.concatenate(trn_cols) if trn_cols else np.empty((0,), dtype=np.int64)
            trn_mat = _coo_from_edges(trn_row, trn_col, (total_nodes, total_nodes))

            val_nodes = np.where(val_mask)[0].astype(np.int64, copy=False)
            val_labels = y[val_nodes].astype(np.int64, copy=False) if val_nodes.size > 0 else np.empty((0,), dtype=np.int64)
            val_mat = _coo_from_edges(val_nodes, val_labels, (total_nodes, num_classes))

            test_nodes = np.where(test_mask)[0].astype(np.int64, copy=False)
            test_labels = y[test_nodes].astype(np.int64, copy=False) if test_nodes.size > 0 else np.empty((0,), dtype=np.int64)
            tst_mat = _coo_from_edges(test_nodes, test_labels, (total_nodes, num_classes))

            out_dir = self.node_root / dataset_alias
            self._reset_output_dir(out_dir)
            _save_sparse(out_dir / "trn_mat.pkl", trn_mat)
            if bool(self.args.emit_node_val):
                _save_sparse(out_dir / "val_mat.pkl", val_mat)
            _save_sparse(out_dir / "tst_mat.pkl", tst_mat)

            x = getattr(data, "x", None)
            if x is not None:
                x_np = torch.as_tensor(x).detach().cpu().numpy().astype(np.float32, copy=False)
                if x_np.shape[0] == num_nodes:
                    node_feat_dim = int(self.args.node_output_feat_dim)
                    if node_feat_dim > 0:
                        x_np = _reduce_features_svd(x_np, node_feat_dim)
                    feat_dim = int(x_np.shape[1])
                    class_feats = np.zeros((num_classes, feat_dim), dtype=np.float32)
                    feats = np.concatenate([x_np, class_feats], axis=0)
                    _save_feats(out_dir / "feats.pkl", feats)

            meta = {
                "dataset": dataset_alias,
                "source_dataset": dataset_name,
                "task": "node",
                "source": "agae_conversion",
                "mask_source": mask_source,
                "split_file": str(split_path) if split_path is not None else "",
                "existing_mask_coverage": coverage,
                "num_nodes_real": num_nodes,
                "num_nodes_total": total_nodes,
                "num_classes": num_classes,
                "num_edges_graph": int(graph_edges.shape[1]),
                "num_train_labels": int(train_nodes.size),
                "num_val_labels": int(val_nodes.size),
                "num_test_labels": int(test_nodes.size),
                "node_split": list(split),
                "split_seed": int(seed),
                "node_output_feat_dim": int(self.args.node_output_feat_dim),
            }
            (out_dir / "conversion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return ConvertResult(
                dataset=dataset_alias,
                source_dataset=dataset_name,
                split=split,
                seed=seed,
                task="node",
                status="ok",
                out_dir=str(out_dir),
                num_nodes=total_nodes,
                num_edges_train=int(trn_mat.nnz),
                num_edges_val=int(val_mat.nnz),
                num_edges_test=int(tst_mat.nnz),
                num_classes=num_classes,
            )
        except Exception as exc:  # pylint: disable=broad-except
            return ConvertResult(
                dataset=dataset_alias,
                source_dataset=dataset_name,
                split=split,
                seed=seed,
                task="node",
                status="fail",
                message=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )

    def convert_graph(self, dataset_name: str, split: Tuple[float, float, float], seed: int, dataset_alias: str) -> ConvertResult:
        """Convert a graph-level dataset to the AnyGraph super-node mega-graph.

        Builds one block-diagonal graph of all (kept) graphs plus one
        super-node per graph wired to every node of that graph (a readout via
        topo propagation). Single-label classification additionally appends
        class-nodes and encodes train labels as super-node↔class-node edges,
        identical in shape to the node task. Multi-label and regression instead
        emit ``labels.pkl``/``label_mask.pkl`` consumed by a head over the
        super-node embeddings.
        """
        try:
            ds = create_dataset(
                name=dataset_name,
                root=str(self.dataset_root),
                task_level="graph",
                induced=False,
                feat_reduction=bool(self.args.feat_reduction),
                feat_reduction_dim=int(self.args.feat_dim),
                graph_filter_dir=str(getattr(self.args, "graph_filter_dir", "") or ""),
            )
            num_graphs_total = len(ds)
            if num_graphs_total <= 0:
                raise ValueError("Empty graph dataset.")

            split_path = _resolve_graph_split_file(
                split_root=self.split_root,
                dataset_name=dataset_name,
                split=split,
                seed=int(seed),
            )
            if split_path is None:
                raise FileNotFoundError(
                    f"Graph split file not found for dataset={dataset_name}, split={split}, seed={seed}, "
                    f"split_root={self.split_root}"
                )
            train_idx, val_idx, test_idx = _load_graph_split_indices(split_path, num_graphs_total)

            # --- labels + task-family inference (over all graphs) ---
            labels: List[np.ndarray] = []
            for i in range(num_graphs_total):
                arr = _graph_label_array(ds[i])
                if arr is None:
                    raise ValueError(f"Dataset has graphs without labels (graph index {i}); cannot build a supervised graph task.")
                labels.append(arr)
            label_dim_in = int(labels[0].shape[0])
            if any(int(a.shape[0]) != label_dim_in for a in labels):
                raise ValueError("Inconsistent label dimension across graphs in this dataset.")
            Y = np.stack(labels).astype(np.float64)  # (G_total, K)
            family, label_dim = _infer_graph_task_family(Y)

            # --- scale guard (seeded, split-preserving) ---
            node_counts = [int(getattr(ds[i], "num_nodes", 0) or 0) for i in range(num_graphs_total)]
            max_graphs = int(getattr(self.args, "max_graphs", 0) or 0)
            max_total_nodes = int(getattr(self.args, "max_total_nodes", 0) or 0)
            train_idx, val_idx, test_idx, drop_report = _subsample_graph_splits(
                train_idx, val_idx, test_idx, node_counts,
                max_graphs=max_graphs, max_total_nodes=max_total_nodes, seed=int(seed),
            )
            if any(drop_report.values()):
                print(
                    f"[MoE][AnyGraph][Graph] scale guard dropped graphs for {dataset_name}: "
                    f"train-{drop_report['train_dropped']} val-{drop_report['val_dropped']} "
                    f"test-{drop_report['test_dropped']} (max_graphs={max_graphs}, max_total_nodes={max_total_nodes})"
                )

            kept = sorted(set(train_idx) | set(val_idx) | set(test_idx))
            if not kept:
                raise ValueError("No graphs left after split resolution / scale guard.")
            local_pos = {orig: p for p, orig in enumerate(kept)}
            num_graphs = len(kept)

            # node-count offsets per kept graph (block-diagonal layout)
            offsets: List[int] = []
            running = 0
            for orig in kept:
                offsets.append(running)
                running += node_counts[orig]
            real_total = running
            super_offset = real_total                     # super-node ids: [real_total, real_total + G)
            single_label = family == "single_label"
            num_classes = int(np.round(Y[np.isfinite(Y[:, 0]), 0].max()) + 1) if single_label else 0
            num_classes = max(num_classes, 2) if single_label else 0
            class_offset = real_total + num_graphs        # class-node ids (single-label only)
            total_nodes = real_total + num_graphs + (num_classes if single_label else 0)

            # --- structural edges + super-node star edges ---
            rows: List[np.ndarray] = []
            cols: List[np.ndarray] = []
            feats_blocks: List[np.ndarray] = []
            have_feats = True
            for p, orig in enumerate(kept):
                g = ds[orig]
                ni = node_counts[orig]
                off = offsets[p]
                ei = _edge_index_from_data(g)
                r = ei[0].astype(np.int64, copy=False)
                c = ei[1].astype(np.int64, copy=False)
                keep_e = (r >= 0) & (c >= 0) & (r < ni) & (c < ni)
                rows.append(r[keep_e] + off)
                cols.append(c[keep_e] + off)
                # star edges: super-node sp <-> every node of graph p (bidirectional)
                sp = super_offset + p
                node_ids = np.arange(off, off + ni, dtype=np.int64)
                if ni > 0:
                    rows.append(np.full((ni,), sp, dtype=np.int64))
                    cols.append(node_ids)
                    rows.append(node_ids)
                    cols.append(np.full((ni,), sp, dtype=np.int64))
                # collect node features
                if have_feats:
                    x = getattr(g, "x", None)
                    if x is None:
                        have_feats = False
                    else:
                        feats_blocks.append(torch.as_tensor(x).detach().cpu().numpy().astype(np.float32, copy=False))

            # train-only label edges (single-label classification)
            if single_label:
                for orig in train_idx:
                    cls = int(np.round(Y[orig, 0]))
                    sp = super_offset + local_pos[orig]
                    cn = class_offset + cls
                    rows.append(np.array([sp, cn], dtype=np.int64))
                    cols.append(np.array([cn, sp], dtype=np.int64))

            trn_row = np.concatenate(rows) if rows else np.empty((0,), dtype=np.int64)
            trn_col = np.concatenate(cols) if cols else np.empty((0,), dtype=np.int64)
            trn_mat = _coo_from_edges(trn_row, trn_col, (total_nodes, total_nodes))

            out_dir = self.graph_root / dataset_alias
            self._reset_output_dir(out_dir)
            _save_sparse(out_dir / "trn_mat.pkl", trn_mat)

            # --- features (rows: real nodes, then zero super-nodes, then zero class-nodes) ---
            feat_dim = 0
            if have_feats and feats_blocks:
                x_all = np.concatenate(feats_blocks, axis=0)
                if x_all.shape[0] == real_total:
                    out_feat_dim = int(getattr(self.args, "graph_output_feat_dim", self.args.node_output_feat_dim))
                    if out_feat_dim > 0:
                        x_all = _reduce_features_svd(x_all, out_feat_dim)
                    feat_dim = int(x_all.shape[1])
                    pad_rows = num_graphs + (num_classes if single_label else 0)
                    feats = np.concatenate([x_all, np.zeros((pad_rows, feat_dim), dtype=np.float32)], axis=0)
                    _save_feats(out_dir / "feats.pkl", feats)

            # --- per-family label payloads ---
            num_val = len(val_idx)
            num_test = len(test_idx)
            reg_mean: List[float] = []
            reg_std: List[float] = []
            if single_label:
                val_super = np.array([super_offset + local_pos[o] for o in val_idx], dtype=np.int64)
                val_lab = np.array([int(np.round(Y[o, 0])) for o in val_idx], dtype=np.int64)
                tst_super = np.array([super_offset + local_pos[o] for o in test_idx], dtype=np.int64)
                tst_lab = np.array([int(np.round(Y[o, 0])) for o in test_idx], dtype=np.int64)
                val_mat = _coo_from_edges(val_super, val_lab, (total_nodes, num_classes))
                tst_mat = _coo_from_edges(tst_super, tst_lab, (total_nodes, num_classes))
                _save_sparse(out_dir / "val_mat.pkl", val_mat)
                _save_sparse(out_dir / "tst_mat.pkl", tst_mat)
            else:
                # labels indexed by local super-node order p = 0..G-1
                lab = np.zeros((num_graphs, label_dim), dtype=np.float32)
                mask = np.zeros((num_graphs, label_dim), dtype=bool)
                for p, orig in enumerate(kept):
                    row = Y[orig].reshape(-1)
                    finite = np.isfinite(row)
                    mask[p] = finite
                    lab[p][finite] = row[finite].astype(np.float32)
                if family == "regression":
                    # standardize per target using TRAIN graphs only (no leakage)
                    train_local = [local_pos[o] for o in train_idx]
                    for d in range(label_dim):
                        col_mask = mask[train_local, d]
                        vals = lab[train_local, d][col_mask] if np.any(col_mask) else np.array([0.0], dtype=np.float32)
                        m = float(vals.mean()) if vals.size else 0.0
                        s = float(vals.std()) if vals.size else 1.0
                        s = s if s > 1e-8 else 1.0
                        reg_mean.append(m)
                        reg_std.append(s)
                        sel = mask[:, d]
                        lab[sel, d] = (lab[sel, d] - m) / s
                with (out_dir / "labels.pkl").open("wb") as f:
                    pickle.dump(lab, f, protocol=pickle.HIGHEST_PROTOCOL)
                with (out_dir / "label_mask.pkl").open("wb") as f:
                    pickle.dump(mask, f, protocol=pickle.HIGHEST_PROTOCOL)
                graph_index = {
                    "train": np.array([super_offset + local_pos[o] for o in train_idx], dtype=np.int64),
                    "val": np.array([super_offset + local_pos[o] for o in val_idx], dtype=np.int64),
                    "test": np.array([super_offset + local_pos[o] for o in test_idx], dtype=np.int64),
                    "super_offset": int(super_offset),
                }
                with (out_dir / "graph_index.pkl").open("wb") as f:
                    pickle.dump(graph_index, f, protocol=pickle.HIGHEST_PROTOCOL)

            meta = {
                "dataset": dataset_alias,
                "source_dataset": dataset_name,
                "task": "graph",
                "task_family": family,
                "source": "agae_conversion",
                "split_file": str(split_path),
                "num_graphs_total": int(num_graphs_total),
                "num_graphs": int(num_graphs),
                "num_nodes_real": int(real_total),
                "num_nodes_total": int(total_nodes),
                "super_offset": int(super_offset),
                "class_offset": int(class_offset) if single_label else -1,
                "num_classes": int(num_classes),
                "label_dim": int(label_dim),
                "num_train_graphs": int(len(train_idx)),
                "num_val_graphs": int(num_val),
                "num_test_graphs": int(num_test),
                "feat_dim": int(feat_dim),
                "graph_split": list(split),
                "split_seed": int(seed),
                "scale_guard": {
                    "max_graphs": max_graphs,
                    "max_total_nodes": max_total_nodes,
                    **drop_report,
                },
                "graph_output_feat_dim": int(getattr(self.args, "graph_output_feat_dim", self.args.node_output_feat_dim)),
            }
            if family == "regression":
                meta["regression_target_mean"] = reg_mean
                meta["regression_target_std"] = reg_std
            (out_dir / "conversion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return ConvertResult(
                dataset=dataset_alias,
                source_dataset=dataset_name,
                split=split,
                seed=seed,
                task="graph",
                status="ok",
                out_dir=str(out_dir),
                num_nodes=total_nodes,
                num_edges_train=int(trn_mat.nnz),
                num_classes=num_classes,
                num_graphs=num_graphs,
                task_family=family,
            )
        except Exception as exc:  # pylint: disable=broad-except
            return ConvertResult(
                dataset=dataset_alias,
                source_dataset=dataset_name,
                split=split,
                seed=seed,
                task="graph",
                status="fail",
                message=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )


def _parse_dataset_spec(spec) -> List[Tuple[str, str]]:
    """Parse a ``name:level`` selection spec into ``(name, level)`` pairs.

    ``level`` is one of ``node`` / ``edge`` / ``graph`` (from the expert-pool
    TSV's task_level column), or ``auto`` when no level is given (the level is
    then inferred or taken from ``--task``). Same dataset may appear at multiple
    levels (e.g. ``cora:node,cora:edge``) to convert it for several tasks.
    """
    pairs: List[Tuple[str, str]] = []
    for token in str(spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            name, _, level = token.partition(":")
            name = name.strip()
            level = level.strip().lower() or "auto"
        else:
            name, level = token, "auto"
        if name:
            pairs.append((name, level))
    return pairs


def _resolve_tasks_for(name: str, level: str, global_task: str) -> List[str]:
    """Map a (dataset, level) selection to AnyGraph conversion task(s).

    Explicit levels win (``edge`` -> ``link``); ``auto`` defers to ``--task``
    (``all`` -> node+link) or, when ``--task`` is also auto, to
    ``infer_task_level``. Returns ``[]`` when nothing can be resolved.
    """
    level = (level or "auto").strip().lower()
    if level in ("node", "graph"):
        return [level]
    if level in ("edge", "link"):
        return ["link"]
    # auto level: honour the global --task, else infer from the name
    if global_task == "all":
        return ["node", "link"]
    if global_task in ("node", "link", "graph"):
        return [global_task]
    inferred = infer_task_level(name)
    if inferred == "node":
        return ["node"]
    if inferred == "edge":
        return ["link"]
    if inferred == "graph":
        return ["graph"]
    return []


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert IcG datasets into AnyGraph matrix format.")
    parser.add_argument("--dataset", default="", help="Comma-separated dataset names. Empty means use --dataset_file.")
    parser.add_argument(
        "--dataset_spec",
        default="",
        help="Comma-separated name:level pairs (level in node/edge/graph/auto). Takes precedence over "
        "--dataset/--task: each pair is converted at its level, so a dataset can be converted for "
        "multiple tasks (e.g. 'cora:node,cora:edge'). Built from expert-pool TSVs by run.py.",
    )
    parser.add_argument(
        "--dataset_file",
        default=str(project_path("data", "available_node_datasets.tsv")),
        help="TSV file containing dataset names when --dataset is empty.",
    )
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "all", "node", "link", "graph"],
        help="Conversion mode: auto picks node/edge/graph by infer_task_level.",
    )
    parser.add_argument("--dataset_root", default=str(project_path("data", "datasets")))
    parser.add_argument("--split_root", default=str(project_path("data", "splits")))
    parser.add_argument("--out_root", default=str(project_path("data", "anygraph_data")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_col", type=int, default=0)

    _add_bool_flag(parser, "feat_reduction", default=False)
    _add_bool_flag(
        parser,
        "l1_normalize_features",
        default=True,
        help_text="Apply row-wise L1 normalization for link-task feats.pkl.",
    )
    parser.add_argument("--feat_dim", type=int, default=100)

    parser.add_argument("--edge_split", default="0.8,0.1,0.1")
    parser.add_argument("--edge_splits", default="", help="Optional list of edge splits; empty falls back to --edge_split.")
    parser.add_argument(
        "--edge_eval_payload_name",
        default="agae_edge_eval_payload.pt",
        help="Filename (under each link dataset dir) for agae-protocol edge eval payload.",
    )

    parser.add_argument("--node_split", default="0.8,0.1,0.1")
    parser.add_argument("--node_splits", default="", help="Optional list of node splits; empty falls back to --node_split.")
    parser.add_argument("--seeds", default="", help="Comma-separated split seeds; empty falls back to --seed.")
    parser.add_argument("--node_output_feat_dim", type=int, default=128, help="Output feature dimension for node task; <=0 keeps original.")
    _add_bool_flag(parser, "emit_node_val", default=True, help_text="Emit val_mat.pkl for node task.")

    parser.add_argument("--graph_split", default="0.8,0.1,0.1")
    parser.add_argument("--graph_splits", default="", help="Optional list of graph splits; empty falls back to --graph_split.")
    parser.add_argument("--graph_output_feat_dim", type=int, default=128, help="Output feature dim for graph task; <=0 keeps original.")
    parser.add_argument("--graph_filter_dir", default=str(project_path("data", "filters")), help="Empty-graph filter dir; must match split generation for index alignment.")
    parser.add_argument("--max_graphs", type=int, default=0, help="Scale guard: cap graphs per graph dataset (0 = all). Subsampling is seeded and logged.")
    parser.add_argument("--max_total_nodes", type=int, default=0, help="Scale guard: cap ΣNᵢ per graph dataset (0 = unbounded). Seeded and logged.")

    parser.add_argument("--index_out", default="", help="Optional explicit conversion index output path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    args.edge_split = _parse_split(args.edge_split)
    args.node_split = _parse_split(args.node_split)
    args.graph_split = _parse_split(args.graph_split)
    args.edge_splits = _parse_split_list(args.edge_splits) or [args.edge_split]
    args.node_splits = _parse_split_list(args.node_splits) or [args.node_split]
    args.graph_splits = _parse_split_list(args.graph_splits) or [args.graph_split]
    args.seeds = _parse_int_list(args.seeds) or [int(args.seed)]

    # Build (name, level) selection pairs. --dataset_spec (name:level, from the
    # expert-pool TSVs) wins; otherwise fall back to --dataset names, then the
    # --dataset_file list. Names without a level are "auto" (level via --task /
    # infer_task_level), preserving the original behavior.
    if args.dataset_spec:
        spec_pairs = _parse_dataset_spec(args.dataset_spec)
    elif args.dataset:
        spec_pairs = [(name, "auto") for name in parse_csv_list(args.dataset)]
    else:
        dataset_file = resolve_project_path(args.dataset_file)
        if not dataset_file.is_file():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        spec_pairs = [(name, "auto") for name in read_name_list_file(dataset_file)]

    converter = AnyGraphConverter(args)

    results: List[ConvertResult] = []
    # Resolve to ordered-unique (name, task) work items so a dataset listed at
    # several levels (e.g. cora node + cora edge) is converted for each task once.
    work: List[Tuple[str, str]] = []
    seen: set = set()
    for name, level in spec_pairs:
        tasks = _resolve_tasks_for(name, level, args.task)
        if not tasks:
            key = (name, "skip")
            if key not in seen:
                seen.add(key)
                results.append(
                    ConvertResult(
                        dataset=name,
                        task="skip",
                        status="fail",
                        message=f"unsupported level='{level}' for dataset={name} (inferred={infer_task_level(name)})",
                    )
                )
            continue
        for task in tasks:
            key = (name, task)
            if key not in seen:
                seen.add(key)
                work.append((name, task))

    for idx, (name, task) in enumerate(work, 1):
        print(f"[{idx}/{len(work)}] dataset={name} task={task}")
        if task == "node":
            split_defs = args.node_splits
        elif task == "graph":
            split_defs = args.graph_splits
        else:
            split_defs = args.edge_splits
        if not split_defs:
            results.append(
                ConvertResult(
                    dataset=name,
                    source_dataset=name,
                    split=None,
                    seed=0,
                    task=task,
                    status="fail",
                    message=f"missing split definitions for task={task}",
                )
            )
            continue
        for split_def in split_defs:
            for split_seed in args.seeds:
                dataset_alias = _converted_dataset_name(name, split_def, int(split_seed))
                if task == "node":
                    res = converter.convert_node(
                        name,
                        split=split_def,
                        seed=int(split_seed),
                        dataset_alias=dataset_alias,
                    )
                elif task == "graph":
                    res = converter.convert_graph(
                        name,
                        split=split_def,
                        seed=int(split_seed),
                        dataset_alias=dataset_alias,
                    )
                else:
                    res = converter.convert_link(
                        name,
                        split=split_def,
                        seed=int(split_seed),
                        dataset_alias=dataset_alias,
                    )
                print(
                    f"  - task={task} split={list(split_def)} seed={int(split_seed)} "
                    f"dataset={dataset_alias} status={res.status} out={res.out_dir or '-'} msg={res.message or '-'}"
                )
                results.append(res)

    ok = [r for r in results if r.status == "ok"]
    fail = [r for r in results if r.status == "fail"]
    skip = [r for r in results if r.status == "skip"]

    conversion_index = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "dataset_root": str(resolve_project_path(args.dataset_root)),
        "split_root": str(resolve_project_path(args.split_root)),
        "out_root": str(resolve_project_path(args.out_root)),
        "config": {
            "task": args.task,
            "dataset_spec": str(args.dataset_spec or ""),
            "seed": int(args.seed),
            "seeds": list(args.seeds),
            "feat_reduction": bool(args.feat_reduction),
            "l1_normalize_features": bool(args.l1_normalize_features),
            "feat_dim": int(args.feat_dim),
            "edge_splits": [list(split) for split in args.edge_splits],
            "edge_eval_payload_name": str(args.edge_eval_payload_name),
            "node_splits": [list(split) for split in args.node_splits],
            "node_output_feat_dim": int(args.node_output_feat_dim),
            "emit_node_val": bool(args.emit_node_val),
            "graph_splits": [list(split) for split in args.graph_splits],
            "graph_output_feat_dim": int(args.graph_output_feat_dim),
            "max_graphs": int(args.max_graphs),
            "max_total_nodes": int(args.max_total_nodes),
        },
        "results": [r.to_dict() for r in results],
        "ok_count": len(ok),
        "fail_count": len(fail),
        "skip_count": len(skip),
        "node_datasets": sorted({r.dataset for r in ok if r.task == "node"}),
        "link_datasets": sorted({r.dataset for r in ok if r.task == "link"}),
        "graph_datasets": sorted({r.dataset for r in ok if r.task == "graph"}),
        "node_source_datasets": sorted({r.source_dataset for r in ok if r.task == "node"}),
        "link_source_datasets": sorted({r.source_dataset for r in ok if r.task == "link"}),
        "graph_source_datasets": sorted({r.source_dataset for r in ok if r.task == "graph"}),
    }

    if args.index_out:
        index_path = resolve_project_path(args.index_out)
    else:
        index_path = resolve_project_path(args.out_root) / "conversion_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(conversion_index, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"Conversion done: ok={len(ok)} fail={len(fail)} skip={len(skip)}")
    print(f"Conversion index: {index_path}")
    print("=" * 72)

    return 0 if len(fail) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
