"""LogME transferability selection baseline (You et al., ICML 2021).

LogME scores a frozen feature extractor by the log marginal evidence of a
Bayesian linear model fitted on (features, labels), maximized over the prior
and noise precisions with the Gull/MacKay fixed point (Eq. 2, Algorithm 1).
The selector scores every eligible expert from its frozen support readout and
the target's support labels only; query data are never read.

Ported from the AnyGraphAnyExpert reranker with the spec's fixes: optional
standardization (``standardize=False`` is the paper/official variant), float
target columns with fewer than two finite or two distinct values are skipped,
and expert ids are kept as given (strings).
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Hashable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from ...applications import convert_labels
from ...common import MULTILABEL, REGRESSION, AppSpec
from ...losses import is_simplex_family
from .common import SelectionOutcome

_INT_DTYPES = (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)


def _logme_single_target(
    s: np.ndarray,
    u_t_y: np.ndarray,
    y_norm_sq: float,
    n: int,
    d: int,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """Evidence for one scalar target given the SVD of the feature matrix.

    ``s``: singular values of F [k]; ``u_t_y``: U^T y over those components;
    ``y_norm_sq``: ||y||^2. Fixed-point iteration on (alpha, beta) from the
    LogME reference implementation, returning the per-sample log evidence.
    """
    sigma = s**2
    alpha, beta = 1.0, 1.0
    for _ in range(max_iter):
        gamma = float(np.sum(sigma * beta / (alpha + beta * sigma)))
        m_sq = float(
            np.sum((beta**2 * sigma * u_t_y**2) / (alpha + beta * sigma) ** 2)
        )
        res_sq = float(
            np.sum((alpha**2 * u_t_y**2) / (alpha + beta * sigma) ** 2)
        ) + max(y_norm_sq - float(np.sum(u_t_y**2)), 0.0)
        alpha_new = gamma / m_sq if m_sq > 0 else alpha
        beta_new = (n - gamma) / res_sq if res_sq > 0 else beta
        if abs(alpha_new - alpha) / max(alpha, 1e-12) < tol and abs(
            beta_new - beta
        ) / max(beta, 1e-12) < tol:
            alpha, beta = alpha_new, beta_new
            break
        alpha, beta = alpha_new, beta_new
    m_sq = float(np.sum((beta**2 * sigma * u_t_y**2) / (alpha + beta * sigma) ** 2))
    res_sq = float(
        np.sum((alpha**2 * u_t_y**2) / (alpha + beta * sigma) ** 2)
    ) + max(y_norm_sq - float(np.sum(u_t_y**2)), 0.0)
    # Dimensions beyond the k retained SVD components have sigma = 0, so their
    # log-determinant contribution is log(alpha) each.
    log_det = float(np.sum(np.log(alpha + beta * sigma))) + (d - len(sigma)) * np.log(
        alpha
    )
    evidence = (
        d / 2.0 * np.log(alpha)
        + n / 2.0 * np.log(beta)
        - 0.5 * log_det
        - beta / 2.0 * res_sq
        - alpha / 2.0 * m_sq
        - n / 2.0 * np.log(2 * np.pi)
    )
    return float(evidence) / n


def _target_matrix(targets: torch.Tensor) -> Tuple[np.ndarray, bool]:
    """``(y [n, K] float64, is_float)``: integer labels one-hot over the classes present."""
    t = targets.detach().cpu()
    if t.dtype in _INT_DTYPES:
        t = t.reshape(-1)
        cols = [(t == c).to(torch.float64).numpy() for c in torch.unique(t)]
        return np.stack(cols, axis=1), False
    y_mat = t.to(torch.float64).numpy()
    return (y_mat[:, None] if y_mat.ndim == 1 else y_mat.reshape(y_mat.shape[0], -1)), True


def _usable_column(y: np.ndarray) -> bool:
    """At least two finite entries with at least two distinct values (constant columns carry no evidence)."""
    finite = y[np.isfinite(y)]
    return finite.size >= 2 and np.unique(finite).size >= 2


def num_usable_columns(targets: torch.Tensor) -> int:
    """Target columns :func:`logme_score` averages over (expert independent)."""
    y_mat, is_float = _target_matrix(targets)
    if not is_float:
        return int(y_mat.shape[1])
    return int(sum(_usable_column(y_mat[:, j]) for j in range(y_mat.shape[1])))


def logme_score(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    standardize: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """LogME evidence of ``targets`` under frozen ``features``.

    ``features``: [n, d]. ``targets``: [n] integer class labels (one-hot
    expanded), [n] floats (single regression target), or [n, T] floats
    (vector regression / multilabel with NaN = missing; evidence averaged over
    usable columns). Higher is better. ``standardize=True`` z-scores features
    and float targets on the given rows (legacy reranker); ``False`` is the
    paper/official algorithm (no bias term, so centring changes the score).
    """
    f = features.detach().to(torch.float64).cpu().numpy()
    n, d = f.shape
    if n < 2:
        return float("-inf")
    if standardize:
        mu = f.mean(axis=0, keepdims=True)
        sd = f.std(axis=0, keepdims=True)
        f = (f - mu) / np.maximum(sd, 1e-8)

    y_mat, is_float = _target_matrix(targets)
    if is_float and standardize:
        y_mu = np.nanmean(y_mat, axis=0, keepdims=True)
        y_sd = np.nanstd(y_mat, axis=0, keepdims=True)
        y_mat = (y_mat - y_mu) / np.maximum(y_sd, 1e-8)

    u, s, _ = np.linalg.svd(f, full_matrices=False)
    evidences = []
    for j in range(y_mat.shape[1]):
        y = y_mat[:, j]
        if is_float and not _usable_column(y):
            continue
        finite = np.isfinite(y)
        if finite.sum() < 2:
            continue
        if finite.all():
            u_j, s_j, y_j = u, s, y
        else:
            u_j, s_j, _ = np.linalg.svd(f[finite], full_matrices=False)
            y_j = y[finite]
        u_t_y = u_j.T @ y_j
        evidences.append(
            _logme_single_target(s_j, u_t_y, float(y_j @ y_j), int(len(y_j)), d, max_iter, tol)
        )
    if not evidences:
        return float("-inf")
    return float(np.mean(evidences))


def _sort_by_evidence(scores: Mapping[Hashable, float], order_hint: Sequence[Hashable]) -> List[Tuple[Any, float]]:
    """Descending evidence (non-finite -> -inf); exact ties follow ``order_hint``."""
    hint = {e: i for i, e in enumerate(order_hint)}
    scored = [(e, s if math.isfinite(s) else float("-inf")) for e, s in scores.items()]
    scored.sort(key=lambda kv: (-kv[1], hint.get(kv[0], len(hint))))
    return scored


def rank_experts_by_evidence(
    expert_features: Dict[Hashable, torch.Tensor],
    targets: torch.Tensor,
    order_hint: Sequence[Hashable],
    *,
    standardize: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> List[Tuple[Any, float]]:
    """Rank expert ids by LogME evidence, descending; exact ties follow ``order_hint``."""
    scores = {
        e: logme_score(feats, targets, standardize=standardize, max_iter=max_iter, tol=tol)
        for e, feats in expert_features.items()
    }
    return _sort_by_evidence(scores, order_hint)


def to_logme_targets(labels: torch.Tensor, family: str) -> torch.Tensor:
    """Support labels -> LogME targets.

    Single-label and link: int64 ``[n]`` (one-hot columns inside LogME);
    multilabel: float ``[n, L]`` with NaN for missing (signed conventions
    converted); regression: float ``[n, T]`` in raw units (official code).
    """
    if is_simplex_family(family):
        return labels.reshape(-1).long()
    rows = labels.reshape(labels.size(0), -1)
    if family == MULTILABEL:
        return convert_labels(rows, MULTILABEL)
    if family == REGRESSION:
        return rows.float()
    raise ValueError(f"Unsupported task family {family!r}")


class LogMESelector:
    """Rank the whole eligible pool E_a by LogME on support readouts (ties: ascending expert id)."""

    name = "logme"

    def __init__(self, cfg, infra):
        self.cfg = cfg
        self.infra = infra
        lcfg = cfg.moe.routergfm.baselines.logme
        self.standardize = bool(lcfg.standardize)
        self.max_iter = int(lcfg.max_iter)
        self.tol = float(lcfg.tol)
        self.topk = int(cfg.moe.routergfm.baselines.topk)

    def rank(self, app: AppSpec) -> SelectionOutcome:
        started = time.perf_counter()
        pool = self.infra.compatible_pool(app)
        targets = to_logme_targets(self.infra.support_labels(app), self.infra.task_family(app))
        scores = {
            e: logme_score(
                self.infra.embeddings(app, e, "support"),
                targets,
                standardize=self.standardize,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            for e in pool
        }
        ranking = _sort_by_evidence(scores, sorted(pool))
        non_finite = sum(not math.isfinite(s) for s in scores.values())
        if non_finite:
            print(f"[LogME] {app.key}: {non_finite}/{len(pool)} experts have a non-finite score (-inf).")
        return SelectionOutcome.from_ranking(
            app,
            ranking,
            self.topk,
            num_target_executions=len(pool),
            wall_time_sec=time.perf_counter() - started,
            extras={"num_usable_columns": num_usable_columns(targets), "num_non_finite": non_finite},
        )


__all__ = [
    "LogMESelector",
    "logme_score",
    "num_usable_columns",
    "rank_experts_by_evidence",
    "to_logme_targets",
]
