"""Historical evaluations (paper Sec. 3.1, Eq. 2; Alg. 1 l.1-4).

For every (application, expert) pair, a task head is fitted on the frozen
expert's support embeddings, and its predictions and per-instance routing
losses on the diagnostic set D_a are recorded. Heads depend only on the split
(support) and the expert, so one record per (data key, expert) serves both LP
budget blocks. :func:`predict_queries` refits the identical head (same support
embeddings, seed, and head config) to predict the full query set Q_a at
deployment.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from src.data_loader.induced_graphs import _acquire_induced_cache_build_lock, _release_induced_cache_build_lock
from src.utils.checkpoint import save_torch_atomic

from .applications import AppData, derive_seed, instance_set_key
from .common import REGRESSION, AppSpec, RouterPaths, enumerate_applications, is_same_source, stable_hash
from .embeddings import embed_instances
from .heads import FittedHead, fit_head, fit_predict_oof, predict_head
from .losses import RegressionNormalizer, routing_loss

SPLITS = ("support", "diag", "query")
_MATRIX_STEM = "_matrix"  # consolidated per-data-key cache next to the records
# NodeFormer seeds its random-feature projection from the whole batch, so an
# instance's embedding depends on its batch mates. Such encoders embed every
# canonical position list (support, D_a, Q_a minus D_a) on its own, so the
# history, predict_queries, and cached embeddings all see identical batches.
BATCH_DEPENDENT_ARCHITECTURES = ("nodeformer",)
_PRED_KEYS = ("pred", "support_pred", "support_oof_pred")
_REFRESH_SECONDS = 1.0  # a record directory's mtime is checked at most this often per store
# Directory mtimes can be coarse (1 s on Lustre): a listing taken within this
# long of the directory's last change is re-read at the next check.
_MTIME_SLACK_NS = 2_000_000_000


def _device(cfg) -> torch.device:
    return torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")


def _load_matrix(path) -> Optional[Dict[str, Any]]:
    """Memory-mapped ``_matrix.pt`` (page-cache backed, not a heap copy), or None when absent."""
    return torch.load(str(path), map_location="cpu", mmap=True) if path.is_file() else None


# --------------------------------------------------------------------------- #
# Shared fitting procedure (history, predict_queries, RouterInfra)
# --------------------------------------------------------------------------- #
def head_cfg_hash(cfg) -> str:
    """Identity of the head-fitting procedure (head block, scale floor, embedding batch size, OOF normalization)."""
    rg = cfg.moe.routergfm
    return stable_hash(
        {
            "heads": dict(rg.heads),
            "scale_floor": float(rg.loss.scale_floor),
            "batch_size": int(rg.device_batch_size),
            "oof_normalizer": "per_fold",  # regression OOF heads normalize with their own training labels
        }
    )


def head_seed(app: AppSpec, expert_id: str) -> int:
    return derive_seed(app.seed, "head", app.data_key, expert_id)


def app_normalizer(cfg, data: AppData) -> Optional[RegressionNormalizer]:
    """Support median/MAD normalizer of a regression application (else ``None``)."""
    if data.task_family != REGRESSION:
        return None
    return RegressionNormalizer(float(cfg.moe.routergfm.loss.scale_floor)).fit(data.labels["support"])


def fit_expert_head(
    cfg, data: AppData, expert_id: str, emb_support: torch.Tensor, *, normalizer=None, device=None
) -> FittedHead:
    """F_{a,e}: head on S_a with the seed of (application split, expert)."""
    return fit_head(
        emb_support,
        data.labels["support"],
        data.task_family,
        int(data.num_classes),
        cfg,
        seed=head_seed(data.app, expert_id),
        normalizer=normalizer,
        device=device,
    )


def stored_predictions(pred: torch.Tensor, family: str) -> torch.Tensor:
    """Predictions as stored: float16 for bounded probabilities, float32 for unbounded regression outputs."""
    return pred.float() if family == REGRESSION else pred.half()


def instance_losses(cfg, pred: torch.Tensor, target: torch.Tensor, family: str, normalizer=None) -> torch.Tensor:
    """Per-instance routing losses of family-space predictions; ``target`` in raw units (NaN = invalid)."""
    if family == REGRESSION:
        target = normalizer.transform(target)
    return routing_loss(pred, target, family, reg_kind=str(cfg.moe.routergfm.loss.regression))


def embed_parts(
    encoder, model_cfg, data: AppData, parts: Mapping[Any, torch.Tensor], device, batch_size: int, *, per_part: bool
) -> Dict[Any, torch.Tensor]:
    """Embeddings of several position lists of one instance set, rows in each list's order.

    Batch-invariant encoders embed the sorted union once; ``per_part`` embeds
    each list on its own (batch-dependent encoders).
    """
    if per_part:
        return {k: embed_instances(encoder, model_cfg, data, pos, device, batch_size) for k, pos in parts.items()}
    lists = [torch.as_tensor(p, dtype=torch.long).reshape(-1) for p in parts.values()]
    union = torch.unique(torch.cat(lists)) if lists else torch.empty(0, dtype=torch.long)
    emb = embed_instances(encoder, model_cfg, data, union, device, batch_size)
    return {k: emb[torch.searchsorted(union, pos)] for k, pos in zip(parts, lists)}


def embed_splits(
    encoder, model_cfg, spec, data: AppData, splits: Sequence[str], device, batch_size: int
) -> Dict[str, torch.Tensor]:
    """Readouts of ``data.<split>_pos`` for each split, rows in position order.

    Q_a is embedded as D_a plus Q_a minus D_a, so a batch-dependent encoder
    gives D_a the same embeddings as the history records.
    """
    in_diag = torch.isin(data.query_pos, data.diag_pos)
    parts: Dict[str, torch.Tensor] = {}
    for split in splits:
        if split == "query":
            parts["diag"] = data.diag_pos
            parts["query_rest"] = data.query_pos[~in_diag]
        elif split in SPLITS:
            parts[split] = getattr(data, f"{split}_pos")
        else:
            raise ValueError(f"Unknown split {split!r} (expected one of {SPLITS}).")
    per_part = spec.architecture in BATCH_DEPENDENT_ARCHITECTURES
    emb = embed_parts(encoder, model_cfg, data, parts, device, batch_size, per_part=per_part)
    out = {}
    for split in splits:
        if split != "query":
            out[split] = emb[split]
            continue
        diag_emb = emb["diag"]
        query = diag_emb.new_empty(data.query_pos.numel(), diag_emb.size(1))
        order = torch.argsort(data.diag_pos)
        rows = order[torch.searchsorted(data.diag_pos[order], data.query_pos[in_diag])]
        query[in_diag] = diag_emb[rows]
        query[~in_diag] = emb["query_rest"]
        out["query"] = query
    return out


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
@dataclass
class _Listing:
    """Cached expert ids of one ``history/<data_key>`` directory."""

    ids: List[str]
    members: set
    mtime: Optional[int]  # directory st_mtime_ns when listed (None: no directory yet)
    settled: bool  # listed >= _MTIME_SLACK_NS after that mtime: later writes must change it
    checked: float  # time.monotonic() of the last freshness check


class HistoryStore:
    """Records at ``RouterPaths.history_file(data_key, expert_id)`` plus a consolidated matrix per data key.

    Records are write-once. ``<root>/history/<data_key>/_matrix.pt`` stacks the
    diagnostic losses (float32, NaN = invalid) and predictions (``stored_predictions``) of every
    recorded expert; it is rebuilt (reusing unchanged columns) when the expert
    set on disk differs from the one it was built from. Directory listings and
    matrices are cached per store instance and refreshed when the directory
    changes (mtime checked at most every ``_REFRESH_SECONDS``), so a
    long-running process sees records that other shards write meanwhile.
    """

    def __init__(self, paths: RouterPaths):
        self.paths = paths
        self._ids: Dict[str, _Listing] = {}
        self._matrices: Dict[str, Tuple[List[str], Dict[str, Any]]] = {}  # (listing ids it was built for, matrix)
        self._columns: Dict[str, Dict[str, int]] = {}

    def _matrix_file(self, data_key: str):
        return self.paths.history_file(data_key, _MATRIX_STEM)

    def _listing(self, data_key: str) -> _Listing:
        entry = self._ids.get(data_key)
        now = time.monotonic()
        if entry is not None and now - entry.checked < _REFRESH_SECONDS:
            return entry
        directory = self._matrix_file(data_key).parent
        listed_at = time.time_ns()
        try:
            mtime = os.stat(directory).st_mtime_ns
        except FileNotFoundError:
            mtime = None
        if entry is not None and entry.settled and entry.mtime == mtime:
            entry.checked = now
            return entry
        names = os.listdir(directory) if mtime is not None else []
        ids = sorted(n[:-3] for n in names if n.endswith(".pt") and not n.startswith((".", "_")))
        if entry is not None and ids == entry.ids:
            ids = entry.ids  # unchanged expert set: keep the cached matrix valid
        settled = mtime is None or listed_at - mtime >= _MTIME_SLACK_NS
        self._ids[data_key] = _Listing(ids, set(ids), mtime, settled, now)
        return self._ids[data_key]

    def expert_ids(self, data_key: str) -> List[str]:
        """Experts with a record on *data_key* (sorted)."""
        return list(self._listing(data_key).ids)

    def has(self, data_key: str, expert_id: str) -> bool:
        return expert_id in self._listing(data_key).members

    def save(self, data_key: str, expert_id: str, record: Dict[str, Any]) -> None:
        save_torch_atomic(str(self.paths.history_file(data_key, expert_id)), record)
        entry = self._listing(data_key)
        entry.ids = sorted(entry.members | {expert_id})  # the changed mtime triggers a re-list later
        entry.members = set(entry.ids)

    def load(self, data_key: str, expert_id: str) -> Dict[str, Any]:
        """One record; tensors are memory-mapped, so reading a few fields stays cheap."""
        return torch.load(str(self.paths.history_file(data_key, expert_id)), map_location="cpu", mmap=True)

    # -- consolidated matrix -------------------------------------------------
    def matrix(self, data_key: str) -> Dict[str, Any]:
        """``{'expert_ids', 'diag_pos', 'family', 'num_classes', 'loss' [n, E], 'pred' [n, E, C], 'mu' [E], 'count' [E]}``."""
        ids = self._listing(data_key).ids
        built = self._matrices.get(data_key)
        if built is not None and built[0] is ids:
            return built[1]
        path = self._matrix_file(data_key)
        cached = _load_matrix(path)
        if cached is None or list(cached["expert_ids"]) != ids:
            if not ids:
                cached = self._build_matrix(data_key, ids, cached)
            else:
                # One process rebuilds a stale matrix; concurrent readers wait and then
                # memory-map the saved file instead of keeping a heap copy.
                lock = _acquire_induced_cache_build_lock(path)
                try:
                    cached = _load_matrix(path)
                    if cached is None or list(cached["expert_ids"]) != ids:
                        save_torch_atomic(str(path), self._build_matrix(data_key, ids, cached))
                        cached = _load_matrix(path)
                finally:
                    _release_induced_cache_build_lock(lock)
        self._matrices[data_key] = (ids, cached)
        self._columns[data_key] = {e: i for i, e in enumerate(cached["expert_ids"])}
        return cached

    def _build_matrix(self, data_key: str, ids: List[str], old: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not ids:
            return {
                "expert_ids": [], "diag_pos": torch.empty(0, dtype=torch.long), "family": None, "num_classes": 0,
                "loss": torch.zeros(0, 0), "pred": torch.zeros(0, 0, 0, dtype=torch.float16),
                "mu": torch.zeros(0), "count": torch.zeros(0, dtype=torch.long),
            }
        old_col = {e: i for i, e in enumerate(old["expert_ids"])} if old is not None else {}
        ref = old if any(e in old_col for e in ids) else None
        losses, preds = [], []
        for eid in ids:
            if eid in old_col:
                losses.append(old["loss"][:, old_col[eid]])
                preds.append(stored_predictions(old["pred"][:, old_col[eid]], str(old["family"])))
                continue
            record = self.load(data_key, eid)
            if ref is None:
                ref = record
            elif not torch.equal(record["diag_pos"], ref["diag_pos"]):
                raise ValueError(f"{data_key}: record {eid} was built on different diagnostic positions (stale history).")
            losses.append(record["loss"].float())
            preds.append(stored_predictions(record["pred"], str(record["family"])))
        loss = torch.stack(losses, dim=1)
        valid = torch.isfinite(loss)
        count = valid.sum(dim=0)
        mu = torch.where(valid, loss, torch.zeros_like(loss)).sum(dim=0) / count.clamp_min(1)
        return {
            "expert_ids": list(ids),
            "diag_pos": ref["diag_pos"].clone(),
            "family": str(ref["family"]),
            "num_classes": int(ref["num_classes"]),
            "loss": loss,
            "pred": torch.stack(preds, dim=1),
            "mu": torch.where(count > 0, mu, torch.full_like(mu, float("nan"))),
            "count": count,
        }

    def loss_matrix(self, data_key: str) -> Tuple[List[str], torch.Tensor]:
        """``(expert_ids, loss [n_diag, n_exp])`` float32, NaN for invalid; rows follow the records' ``diag_pos``."""
        m = self.matrix(data_key)
        return list(m["expert_ids"]), m["loss"].float().clone()

    def _select(self, data_key: str, name: str, expert_ids: Sequence[str]) -> torch.Tensor:
        m = self.matrix(data_key)
        src = m[name]
        col = self._columns[data_key]
        out = torch.full((src.size(0), len(expert_ids)) + tuple(src.shape[2:]), float("nan"))
        pairs = [(k, col[e]) for k, e in enumerate(expert_ids) if e in col]
        if pairs:
            dst_idx, src_idx = (torch.tensor(v, dtype=torch.long) for v in zip(*pairs))
            out[:, dst_idx] = src[:, src_idx].float()
        return out

    def pred_matrix(self, data_key: str, expert_ids: Sequence[str]) -> torch.Tensor:
        """Diagnostic predictions ``[n_diag, K, C]`` float32 (NaN for experts without a record)."""
        return self._select(data_key, "pred", expert_ids)

    def losses(self, app: AppSpec, expert_ids: Sequence[str]) -> torch.Tensor:
        """Per-instance diagnostic losses ``[|D_a|, K]`` (NaN where missing or invalid)."""
        return self._select(app.data_key, "loss", expert_ids)

    def preds(self, app: AppSpec, expert_ids: Sequence[str]) -> torch.Tensor:
        return self.pred_matrix(app.data_key, expert_ids)

    def app_average(self, app: AppSpec, expert_id: str) -> Tuple[float, int]:
        """Eq. 2: mean routing loss over the valid diagnostic observations and their count (NaN, 0 if unrecorded)."""
        if not self.has(app.data_key, expert_id):
            return float("nan"), 0
        m = self.matrix(app.data_key)
        j = self._columns[app.data_key][expert_id]
        return float(m["mu"][j]), int(m["count"][j])


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def _history_record(cfg, data: AppData, expert_id: str, emb_support, emb_diag, device) -> Dict[str, Any]:
    normalizer = app_normalizer(cfg, data)
    head = fit_expert_head(cfg, data, expert_id, emb_support, normalizer=normalizer, device=device)
    pred = predict_head(head, emb_diag)
    loss = instance_losses(cfg, pred, data.labels["diag"], data.task_family, normalizer).float()
    valid = torch.isfinite(loss)
    count = int(valid.sum())
    return {
        "expert_id": expert_id,
        "data_key": data.app.data_key,
        "diag_pos": data.diag_pos.clone(),
        "pred": stored_predictions(pred, data.task_family),
        "loss": loss,
        "mu": float(loss[valid].mean()) if count else float("nan"),
        "count": count,
        "family": data.task_family,
        "num_classes": int(data.num_classes),
        "normalizer": normalizer.state_dict() if normalizer is not None else None,
        "support_pos": data.support_pos.clone(),
        "support_size": int(data.support_pos.numel()),
        "support_emb": emb_support.float(),
        "head_cfg_hash": head_cfg_hash(cfg),
    }


def generate_history(cfg, provider=None, *, apps: Optional[Sequence[AppSpec]] = None, expert_ids: Optional[Sequence[str]] = None) -> None:
    """Record F_{a,e} on D_a for every application and every expert of this shard.

    Experts are ``expert_ids`` (default: the catalog) restricted to
    ``experts[shard_index::num_shards]``. Existing records and same-source pairs
    are skipped. Each built dataset (instance set) is loaded once; per expert,
    the union of the support and diagnostic positions of all its pending data
    keys is embedded once, then a head is fitted per data key. Shard 0 also
    ensures the (expert-independent) descriptors of every data key, cached
    once per instance set.
    """
    from .applications import RealDataProvider
    from .descriptors import ensure_descriptors
    from .experts import build_expert_catalog, load_frozen_encoder

    rg = cfg.moe.routergfm
    provider = provider if provider is not None else RealDataProvider(cfg)
    apps = list(apps) if apps is not None else enumerate_applications(rg)
    catalog = {spec.expert_id: spec for spec in build_expert_catalog(cfg)}
    ids = list(catalog) if expert_ids is None else [str(e) for e in expert_ids]
    shard, num_shards = int(rg.experts.shard_index), int(rg.experts.num_shards)
    specs = [catalog[e] for e in ids][shard::num_shards]
    exclude_same = bool(rg.experts.exclude_same_source)
    store = HistoryStore(RouterPaths.from_cfg(cfg))
    device, batch_size = _device(cfg), int(rg.device_batch_size)

    groups: "OrderedDict[str, Dict[str, AppSpec]]" = OrderedDict()
    for app in apps:
        groups.setdefault(instance_set_key(app), {}).setdefault(app.data_key, app)

    encoders: Dict[str, Tuple[Any, Any]] = {}
    for set_key, members in groups.items():
        datas: Dict[str, AppData] = {}

        def data_of(key: str) -> AppData:
            if key not in datas:
                datas[key] = provider.load(members[key])
            return datas[key]

        if shard == 0:
            for key in members:
                ensure_descriptors(cfg, data_of(key))
        written = 0
        for spec in specs:
            todo = [
                key for key, app in members.items()
                if not (exclude_same and is_same_source(app, spec)) and not store.has(key, spec.expert_id)
            ]
            if not todo:
                continue
            if spec.expert_id not in encoders:
                encoders[spec.expert_id] = load_frozen_encoder(cfg, spec, device)
            encoder, model_cfg = encoders[spec.expert_id]
            parts = {}
            for key in todo:
                parts[(key, "support")] = data_of(key).support_pos
                parts[(key, "diag")] = data_of(key).diag_pos
            emb = embed_parts(
                encoder, model_cfg, data_of(todo[0]), parts, device, batch_size,
                per_part=spec.architecture in BATCH_DEPENDENT_ARCHITECTURES,
            )
            for key in todo:
                record = _history_record(cfg, data_of(key), spec.expert_id, emb[(key, "support")], emb[(key, "diag")], device)
                store.save(key, spec.expert_id, record)
                written += 1
        print(f"[RouterGFM history] {set_key}: {len(members)} data key(s), {written} new record(s)", flush=True)


# --------------------------------------------------------------------------- #
# Deployment-time predictions
# --------------------------------------------------------------------------- #
def _as_float(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v.float() if k in _PRED_KEYS else v) for k, v in payload.items()}


def predict_queries(cfg, app: AppSpec, expert_ids: Sequence[str], provider=None) -> Dict[str, Dict[str, Any]]:
    """F_{a,e} on the full query set Q_a and on S_a for each expert (cached per data key).

    Entries: ``query_pos``, ``pred`` [|Q_a|, C], ``support_pos``,
    ``support_pred`` (in-sample), ``support_oof_pred`` (each support item
    predicted by a fold head not trained on it), in the family's prediction
    space as float32 (stored as ``stored_predictions``; a cache holding
    non-finite values is recomputed). Heads are refitted exactly as in
    :func:`generate_history`. Only support labels are read.
    """
    from .applications import RealDataProvider
    from .experts import build_expert_catalog, load_frozen_encoder

    paths = RouterPaths.from_cfg(cfg)
    digest = head_cfg_hash(cfg)
    wanted = list(dict.fromkeys(str(e) for e in expert_ids))
    out: Dict[str, Dict[str, Any]] = {}
    for eid in wanted:
        path = paths.prediction_file(app.data_key, eid)
        if path.is_file():
            cached = torch.load(path, map_location="cpu")
            if cached.get("head_cfg_hash") == digest and all(torch.isfinite(cached[k]).all() for k in _PRED_KEYS):
                out[eid] = cached
    missing = [e for e in wanted if e not in out]
    if missing:
        data = (provider if provider is not None else RealDataProvider(cfg)).load(app)
        catalog = {spec.expert_id: spec for spec in build_expert_catalog(cfg)}
        device, batch_size = _device(cfg), int(cfg.moe.routergfm.device_batch_size)
        normalizer = app_normalizer(cfg, data)
        for eid in missing:
            spec = catalog[eid]
            encoder, model_cfg = load_frozen_encoder(cfg, spec, device)
            emb = embed_splits(encoder, model_cfg, spec, data, ("support", "query"), device, batch_size)
            head = fit_expert_head(cfg, data, eid, emb["support"], normalizer=normalizer, device=device)
            oof = fit_predict_oof(
                emb["support"], data.labels["support"], data.task_family, int(data.num_classes), cfg,
                seed=derive_seed(app.seed, "oof", app.data_key, eid), normalizer=normalizer, device=device,
            )
            payload = {
                "query_pos": data.query_pos.clone(),
                "pred": stored_predictions(predict_head(head, emb["query"]), data.task_family),
                "support_pos": data.support_pos.clone(),
                "support_pred": stored_predictions(predict_head(head, emb["support"]), data.task_family),
                "support_oof_pred": stored_predictions(oof, data.task_family),
                "family": data.task_family,
                "normalizer": normalizer.state_dict() if normalizer is not None else None,
                "head_cfg_hash": digest,
            }
            save_torch_atomic(str(paths.prediction_file(app.data_key, eid)), payload)
            out[eid] = payload
    return {eid: _as_float(out[eid]) for eid in wanted}


__all__ = [
    "BATCH_DEPENDENT_ARCHITECTURES",
    "HistoryStore",
    "SPLITS",
    "app_normalizer",
    "embed_parts",
    "embed_splits",
    "fit_expert_head",
    "generate_history",
    "head_cfg_hash",
    "head_seed",
    "instance_losses",
    "predict_queries",
    "stored_predictions",
]
