"""MetaGL and MetaGL+metadata selection baselines (Park et al., ICLR 2023; App. C, Tables 9-10).

MetaGL ranks E_a without evaluating any expert on the target: a meta-learner is
fitted on the historical performance matrix ``P[b, e] = -mu_bar_{b,e}`` (Eq. 2)
of the leave-one-dataset-out applications A_tr(a) and the 318-d meta-graph
features of every application; the target enters as a new graph node of the
G-M network from its label-free structure only.

Adaptation (one selector per target group, slice, and seed):

* Slice: historical applications with the target's task family and budget
  (any budget for LP; LP applications of both budgets share one data key and
  appear once). Below ``min_slice_rows`` rows, all same-budget applications are
  pooled and a task-family one-hot is appended to the meta-features.
* Rows with fewer than two observed entries are dropped; P rows are min-max
  scaled to [1, 11] (lowest loss -> 11). Meta-features are min-max scaled on
  the slice rows (target clipped to [0, 1]); ``M' = [M ; log(1 + M)]``.
* Inner validation split by base dataset (``GroupShuffleSplit``); columns are
  the experts observed in an inner-train row; ``k_in = min(2 hid, d, m)``
  (PCA path: also <= #inner-train rows).
* Factors from masked NMF / PCA of inner-train P; ``phi`` (random forest):
  M' -> U, applied to every graph. Training as the official code; the best
  validation state is always restored. Inference: training part over all
  slice rows, the target as the single new graph node.
* Plain MetaGL scores experts without an inner-train observation ``-inf``.
  MetaGL+metadata represents every expert by ``[v_e ; psi(v_e)]`` with expert
  metadata v_e (``infra.expert_metadata``: min-max numeric ⊕ text, per-block
  L2) and a second forest ``psi``: v -> V, adds the ``M_m2m`` kNN relation, and
  inserts unobserved (or ``hidden_experts``) experts as new model nodes.
* Known application (``known_mu``, Table 10 insertion conditions): the caller
  passes the target's observed averages of the visible experts; the target
  joins the inner-train rows with that P row (the historical rows keep their
  split) and is scored as its own graph node. The selector never reads the
  target's history itself.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ....common import LINK, TASK_FAMILIES, AppSpec
from ....context_graph import EXPERT_NUMERIC_NAMES
from ..common import ColumnScaler, SelectionOutcome, application_averages, block_concat, metadata_blocks
from .factorization import factorize
from .features import application_meta_features
from .model import MetaGLNet, TrainResult, train_metagl
from .network import METADATA_RELATIONS, RELATIONS, GMNetwork, add_graphs, add_models, build_network

RF_RANDOM_STATE = 1
PERF_SCALE = 10.0  # rows of P scaled to [1, 1 + PERF_SCALE]


def row_minmax(P: np.ndarray, scale: float = PERF_SCALE) -> np.ndarray:
    """Per-row min-max to ``[1, 1 + scale]`` over observed entries; constant rows map to 1; NaN kept."""
    lo = np.nanmin(P, axis=1, keepdims=True)
    span = np.nanmax(P, axis=1, keepdims=True) - lo
    span[span < 10 * np.finfo(np.float64).eps] = 1.0
    return (P - lo) / span * scale + 1.0


def inner_split(groups: Sequence[str], val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(train, val)`` row indices: grouped by base dataset, or by row with a single group."""
    n = len(groups)
    if len(set(groups)) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=float(val_ratio), random_state=int(seed))
        tr, va = next(splitter.split(np.zeros(n), groups=list(groups)))
    elif n >= 2:
        tr, va = train_test_split(np.arange(n), test_size=float(val_ratio), shuffle=True, random_state=int(seed))
    else:
        tr, va = np.arange(n), np.arange(0)
    return np.sort(tr), np.sort(va)


def _forest(n_estimators: int) -> RandomForestRegressor:
    # max_features=1.0 is the official "auto" for regressors (removed in sklearn >= 1.3).
    return RandomForestRegressor(
        n_estimators=int(n_estimators), criterion="squared_error", max_features=1.0, max_depth=None,
        random_state=RF_RANDOM_STATE,
    )


def _forest_predict(forest: RandomForestRegressor, X: torch.Tensor, k: int) -> torch.Tensor:
    return torch.as_tensor(forest.predict(X.cpu().numpy()), dtype=torch.float32).reshape(-1, int(k)).to(X.device)


def _mprime(M: torch.Tensor) -> torch.Tensor:
    return torch.cat([M, torch.log(M + 1.0)], dim=1)


def _unique_data(apps: Sequence[AppSpec]) -> List[AppSpec]:
    """First application per data key (LP budget blocks share one split and history)."""
    seen, out = set(), []
    for a in apps:
        if a.data_key not in seen:
            seen.add(a.data_key)
            out.append(a)
    return out


@dataclass
class MetaGLFit:
    slice_name: str
    pooled: bool  # fallback: all same-budget applications + task-family one-hot
    rows: List[AppSpec]  # training graphs (all slice rows kept after filtering)
    expert_ids: List[str]  # model nodes (P columns)
    m_scaler: ColumnScaler
    v_scaler: Optional[ColumnScaler]
    phi: RandomForestRegressor
    psi: Optional[RandomForestRegressor]
    Mp: torch.Tensor  # [n, 2d]
    U: torch.Tensor  # [n, k_in] = phi(M')
    V: torch.Tensor  # [m, k_in] model inputs: V (plain) or psi(v) (metadata)
    v_meta: Optional[torch.Tensor]  # [m, d_v] processed expert metadata
    net: GMNetwork  # network over all rows and model nodes
    model: MetaGLNet
    k_in: int
    factorization: str
    train: TrainResult
    n_train_rows: int
    n_val_rows: int
    target_row: Optional[int] = None  # the target's graph node when it is a known application


class MetaGLSelector:
    """``rank(app) -> SelectionOutcome`` (scores ``p_hat``, higher is better; no target labels or runs)."""

    def __init__(self, cfg, infra, *, use_metadata: bool = False):
        self.cfg = cfg
        self.infra = infra
        self.use_metadata = bool(use_metadata)
        self.name = "metagl_metadata" if self.use_metadata else "metagl"
        self.mcfg = cfg.moe.routergfm.baselines.metagl
        self.topk = int(cfg.moe.routergfm.baselines.topk)
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        self._fits: Dict[Tuple, MetaGLFit] = {}
        self._app_x: Dict[str, np.ndarray] = {}
        self._expert_x: Dict[str, torch.Tensor] = {}

    # -- inputs -------------------------------------------------------------
    def slice_rows(self, app: AppSpec) -> Tuple[List[AppSpec], str, bool]:
        """``(rows, slice name, pooled)`` from A_tr(a) (the target's group is always excluded)."""
        family = self.infra.task_family(app)
        history = [b for b in self.infra.historical_applications(app) if b.group != app.group]
        name = family if family == LINK else f"{family}/b{int(app.budget)}"
        rows = _unique_data(
            [b for b in history if self.infra.task_family(b) == family and (family == LINK or b.budget == app.budget)]
        )
        if len(rows) >= int(self.mcfg.min_slice_rows):
            return rows, name, False
        return _unique_data([b for b in history if b.budget == app.budget]), f"pooled/b{int(app.budget)}", True

    def _meta_features(self, apps: Sequence[AppSpec], pooled: bool) -> torch.Tensor:
        """Raw meta-features ``[n, 318]`` (+ task-family one-hot when pooled)."""
        rows = []
        for a in apps:
            if a.data_key not in self._app_x:
                self._app_x[a.data_key] = application_meta_features(self.infra, a, self.cfg)
            x = self._app_x[a.data_key]
            if pooled:
                family = self.infra.task_family(a)
                x = np.concatenate([x, [float(f == family) for f in TASK_FAMILIES]])
            rows.append(x)
        return torch.tensor(np.stack(rows), dtype=torch.float32)

    def _expert_inputs(self, expert_ids: Sequence[str], scaler: Optional[ColumnScaler] = None):
        """``(processed metadata, fitted scaler)``: min-max numeric (fitted here unless given) ⊕ text, per-block L2."""
        for e in expert_ids:
            if e not in self._expert_x:
                self._expert_x[e] = self.infra.expert_metadata(e).float()
        raw = torch.stack([self._expert_x[e] for e in expert_ids])
        numeric, text = metadata_blocks(raw, len(EXPERT_NUMERIC_NAMES), self.cfg)
        if numeric is not None and scaler is None:
            scaler = ColumnScaler("minmax").fit(numeric)
        return block_concat(scaler.transform(numeric) if numeric is not None else None, text), scaler

    # -- fitting --------------------------------------------------------------
    def fit_for(
        self, app: AppSpec, hidden_experts: Sequence[str] = (), known_mu: Optional[Mapping[str, float]] = None
    ) -> MetaGLFit:
        """Fit (cached per target group, slice, seed, hidden experts, known row) on the historical rows of *app*.

        ``hidden_experts``: evaluations removed from P (the experts are scored as new ones).
        ``known_mu``: the target's observed averages (Eq. 2) of visible experts; with two or
        more of them the target is a known application and its P row is an inner-train row.
        """
        m = self.mcfg
        rows, slice_name, pooled = self.slice_rows(app)
        hidden = tuple(sorted(set(hidden_experts)))
        expert_ids = [s.expert_id for s in self.infra.catalog if s.expert_id not in hidden]
        known_mu = dict(known_mu or {})
        known = {e: float(known_mu[e]) for e in expert_ids if e in known_mu and math.isfinite(float(known_mu[e]))}
        known = known if len(known) >= 2 else {}  # a P row needs two observed entries
        key = (app.group, slice_name, int(app.seed), hidden, tuple(sorted(known.items())))
        if key in self._fits:
            return self._fits[key]
        if pooled:
            print(f"[MetaGL] {app.key}: the task-family slice has < {int(m.min_slice_rows)} rows; pooling "
                  f"{len(rows)} budget-{int(app.budget)} applications with a task-family one-hot.")
        mu, _ = application_averages(self.infra, rows, expert_ids)
        P = -mu.double().numpy()
        keep = np.isfinite(P).sum(axis=1) >= 2
        rows, P = [r for r, k in zip(rows, keep.tolist()) if k], P[keep]
        if not rows:
            raise ValueError(f"{app.key}: no historical application with two or more observed experts ({slice_name}).")
        n_hist, target_row = len(rows), None
        if known:  # a known application: the last row, always an inner-train row
            rows, target_row = rows + [app], len(rows)
            P = np.concatenate([P, [[-known.get(e, math.nan) for e in expert_ids]]])
        P = row_minmax(P)
        M_raw = self._meta_features(rows, pooled)
        m_scaler = ColumnScaler("minmax").fit(M_raw)
        Mp = _mprime(m_scaler.transform(M_raw))

        seed = int(app.seed)
        tr, va = inner_split([r.group for r in rows[:n_hist]], float(m.val_ratio), seed)
        if target_row is not None:
            tr = np.append(tr, target_row)
        cols = np.isfinite(P[tr]).any(axis=0)
        P = P[:, cols]
        col_ids = [e for e, c in zip(expert_ids, cols.tolist()) if c]
        P_tr = P[tr]
        k_in = min(2 * int(m.hid_dim), M_raw.size(1), len(col_ids))
        if not np.isnan(P_tr).any():
            k_in = min(k_in, len(tr))
        U, V, kind = factorize(P_tr, k_in, np.random.default_rng(seed))
        phi = _forest(m.rf_n_estimators).fit(Mp.numpy()[tr], U)
        U_hat = _forest_predict(phi, Mp, k_in)
        V = torch.as_tensor(V, dtype=torch.float32)
        psi, v_scaler, v_meta = None, None, None
        if self.use_metadata:
            v_meta, v_scaler = self._expert_inputs(col_ids)
            psi = _forest(m.rf_n_estimators).fit(v_meta.numpy(), V.numpy())
            V = _forest_predict(psi, v_meta, k_in)

        dev, k = self.device, int(m.knn_k)
        Mp, U_hat, V = Mp.to(dev), U_hat.to(dev), V.to(dev)
        v_meta = v_meta.to(dev) if v_meta is not None else None
        tr_t, va_t = torch.as_tensor(tr, device=dev), torch.as_tensor(va, device=dev)
        train_net = build_network(Mp[tr_t], U_hat[tr_t], V, k, v_meta)
        val_net, val_inputs, P_val = None, None, None
        if any(np.isfinite(P[i]).sum() >= 2 for i in va.tolist()):
            val_net = add_graphs(train_net, Mp[tr_t], U_hat[tr_t], V, Mp[va_t], U_hat[va_t], k)
            val_inputs = (torch.cat([Mp[tr_t], Mp[va_t]]), torch.cat([U_hat[tr_t], U_hat[va_t]]), V, v_meta)
            P_val = P[va]
        with torch.random.fork_rng(devices=[dev.index or 0] if dev.type == "cuda" else []):
            torch.manual_seed(seed)
            model = MetaGLNet(
                Mp.size(1), k_in, int(m.hid_dim), METADATA_RELATIONS if self.use_metadata else RELATIONS,
                n_layers=int(m.hgt_layers), n_heads=int(m.hgt_heads), dropout=float(m.hgt_dropout),
                expert_meta_dim=v_meta.size(1) if v_meta is not None else None,
            ).to(dev)
            result = train_metagl(
                model, train_net, (Mp[tr_t], U_hat[tr_t], V, v_meta),
                torch.as_tensor(P_tr, dtype=torch.float32, device=dev),
                val_net, val_inputs, P_val,
                epochs=int(m.epochs), patience=int(m.patience), batch_size=int(m.batch_size),
                lr=float(m.lr), weight_decay=float(m.weight_decay),
            )
        fit = MetaGLFit(
            slice_name=slice_name, pooled=pooled, rows=rows, expert_ids=col_ids,
            m_scaler=m_scaler, v_scaler=v_scaler, phi=phi, psi=psi,
            Mp=Mp, U=U_hat, V=V, v_meta=v_meta,
            net=build_network(Mp, U_hat, V, k, v_meta),
            model=model, k_in=k_in, factorization=kind, train=result,
            n_train_rows=len(tr), n_val_rows=len(va), target_row=target_row,
        )
        self._fits[key] = fit
        return fit

    # -- inference ------------------------------------------------------------
    def _scores(self, fit: MetaGLFit, app: AppSpec) -> Dict[str, float]:
        """``p_hat`` of every expert of E_a for the target (a new graph node unless known; ``-inf``: unscorable)."""
        pool = self.infra.compatible_pool(app)
        k = int(self.mcfg.knn_k)
        if fit.target_row is None:
            Mp_t = _mprime(fit.m_scaler.transform(self._meta_features([app], fit.pooled))).to(self.device)
            U_t = _forest_predict(fit.phi, Mp_t, fit.k_in)
            net = add_graphs(fit.net, fit.Mp, fit.U, fit.V, Mp_t, U_t, k)
            Mp_all, U_all, row = torch.cat([fit.Mp, Mp_t]), torch.cat([fit.U, U_t]), -1
        else:
            net, Mp_all, U_all, row = fit.net, fit.Mp, fit.U, fit.target_row
        V_all, v_all = fit.V, fit.v_meta
        column = {e: j for j, e in enumerate(fit.expert_ids)}
        new = [e for e in pool if e not in column] if self.use_metadata else []
        if new:
            v_new, _ = self._expert_inputs(new, fit.v_scaler)
            v_new = v_new.to(self.device)
            V_new = _forest_predict(fit.psi, v_new, fit.k_in)
            net = add_models(net, fit.V, V_new, U_all, k, fit.v_meta, v_new)
            V_all, v_all = torch.cat([fit.V, V_new]), torch.cat([fit.v_meta, v_new])
            column.update({e: len(fit.expert_ids) + i for i, e in enumerate(new)})
        with torch.no_grad():
            p_hat = fit.model(net, Mp_all, U_all, V_all, v_all)[row].cpu()
        return {e: float(p_hat[column[e]]) if e in column else -math.inf for e in pool}

    def rank(
        self, app: AppSpec, *, hidden_experts: Sequence[str] = (), known_mu: Optional[Mapping[str, float]] = None
    ) -> SelectionOutcome:
        started = time.perf_counter()
        fit = self.fit_for(app, hidden_experts, known_mu)
        scores = self._scores(fit, app)
        pool = self.infra.compatible_pool(app)
        order = sorted(range(len(pool)), key=lambda j: (-scores[pool[j]], j))  # ties: catalog order
        known = set(fit.expert_ids)
        return SelectionOutcome.from_ranking(
            app,
            [(pool[j], scores[pool[j]]) for j in order],
            self.topk,
            num_target_executions=0,
            wall_time_sec=time.perf_counter() - started,
            extras={
                "slice": fit.slice_name,
                "pooled_fallback": fit.pooled,
                "factorization": fit.factorization,
                "k_in": fit.k_in,
                "best_epoch": fit.train.best_epoch,
                "val_score": fit.train.best_score,
                "known_target": fit.target_row is not None,
                "n_rows": len(fit.rows),
                "n_train_rows": fit.n_train_rows,
                "n_val_rows": fit.n_val_rows,
                "n_columns": len(fit.expert_ids),
                "n_new_experts": sum(e not in known for e in pool),
            },
        )


__all__ = ["MetaGLFit", "MetaGLSelector", "inner_split", "row_minmax"]
