"""Nearest-application selection baseline (App. C, Table 9; ArgoSmart-style 1-NN).

Rank experts by their historical losses on the most similar historical
application in metadata space:

1. Candidates C(a): historical applications (outside the target's base
   dataset) with the target's task family and budget (budget ignored for LP);
   all of A_tr(a) if none qualifies.
2. Metadata ``z = block_concat(minmax(numeric), text)`` from
   ``infra.app_metadata`` (min-max fitted on A_tr(a), target clipped to [0, 1]);
   cosine similarity; the nearest set N(a) keeps every candidate within
   ``tie_tol`` of the maximum (seeds of one dataset tie exactly).
3. ``s_e`` = mean mu_bar_{b,e} over b in N(a) that observed e; experts nobody in
   N(a) observed get the mean of N(a)'s observed entries (row-mean imputation).
4. Sort by ``s_e``, then the global mean over C(a) (ArgoSmart tie-break;
   +inf when unobserved), then catalog order.
"""

from __future__ import annotations

import math
import time
from typing import List, Sequence, Tuple

import torch

from ...common import LINK, AppSpec
from ...context_graph import APP_NUMERIC_NAMES
from .common import ColumnScaler, SelectionOutcome, application_averages, block_concat, metadata_blocks


def compatible_candidates(infra, app: AppSpec, history: Sequence[AppSpec], restrict: bool) -> Tuple[List[AppSpec], bool]:
    """``(C(a), used_fallback)``: same task family and budget (any budget for LP), else all of *history*."""
    if not restrict:
        return list(history), False
    family = infra.task_family(app)
    cands = [b for b in history if infra.task_family(b) == family and (family == LINK or b.budget == app.budget)]
    if cands:
        return cands, False
    print(f"[NearestApplication] {app.key}: no compatible historical application; using all {len(history)}.")
    return list(history), True


def nearest_set(z_target: torch.Tensor, z_cands: torch.Tensor, tol: float = 1e-9) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(indices of the tie group within tol of the best cosine similarity, similarities)``."""
    sims = torch.nn.functional.cosine_similarity(z_cands.double(), z_target.double()[None], dim=1, eps=1e-12)
    return torch.nonzero(sims >= sims.max() - tol, as_tuple=False).view(-1), sims


def score_experts(mu_nearest: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(s_e, imputed)`` from mu_bar rows of the nearest set ``[|N|, E]`` (NaN = unobserved)."""
    observed = torch.isfinite(mu_nearest)
    total = torch.where(observed, mu_nearest, torch.zeros_like(mu_nearest))
    count = observed.sum(0)
    fill = float(total.sum() / observed.sum()) if bool(observed.any()) else math.inf
    imputed = count == 0
    scores = torch.where(imputed, torch.full_like(total[0], fill), total.sum(0) / count.clamp_min(1))
    return scores, imputed


def global_losses(mu: torch.Tensor) -> torch.Tensor:
    """Mean observed mu_bar per expert over the candidate rows (+inf if never observed)."""
    observed = torch.isfinite(mu)
    count = observed.sum(0)
    mean = torch.where(observed, mu, torch.zeros_like(mu)).sum(0) / count.clamp_min(1)
    return torch.where(count > 0, mean, torch.full_like(mean, math.inf))


class NearestApplicationSelector:
    """Rank E_a by historical losses on the nearest compatible historical application(s)."""

    name = "nearest_application"

    def __init__(self, cfg, infra):
        self.cfg = cfg
        self.infra = infra
        ncfg = cfg.moe.routergfm.baselines.nearest_application
        self.restrict = bool(ncfg.restrict_compatible)
        self.tie_tol = float(ncfg.tie_tol)
        self.topk = int(cfg.moe.routergfm.baselines.topk)
        self._meta = {}

    def _metadata(self, apps: Sequence[AppSpec]) -> torch.Tensor:
        for a in apps:
            if a.key not in self._meta:
                self._meta[a.key] = self.infra.app_metadata(a).float()
        return torch.stack([self._meta[a.key] for a in apps])

    def _embed(self, x: torch.Tensor, scaler) -> torch.Tensor:
        numeric, text = metadata_blocks(x, len(APP_NUMERIC_NAMES), self.cfg)
        return block_concat(scaler.transform(numeric) if numeric is not None else None, text)

    def rank(self, app: AppSpec) -> SelectionOutcome:
        started = time.perf_counter()
        history = [b for b in self.infra.historical_applications(app) if b.group != app.group]
        if not history:
            raise ValueError(f"{app.key}: no historical applications outside group {app.group!r}.")
        cands, fallback = compatible_candidates(self.infra, app, history, self.restrict)
        x_hist = self._metadata(history)
        numeric, _ = metadata_blocks(x_hist, len(APP_NUMERIC_NAMES), self.cfg)
        scaler = ColumnScaler("minmax").fit(numeric) if numeric is not None else None
        z_cands = self._embed(self._metadata(cands), scaler)
        z_target = self._embed(self.infra.app_metadata(app).float()[None], scaler)[0]
        idx, sims = nearest_set(z_target, z_cands, self.tie_tol)

        pool = self.infra.compatible_pool(app)
        mu, _ = application_averages(self.infra, cands, pool)
        scores, imputed = score_experts(mu[idx])
        tie_break = global_losses(mu)
        order = sorted(range(len(pool)), key=lambda j: (float(scores[j]), float(tie_break[j]), j))
        return SelectionOutcome.from_ranking(
            app,
            [(pool[j], -float(scores[j])) for j in order],
            self.topk,
            num_target_executions=0,
            wall_time_sec=time.perf_counter() - started,
            extras={
                "nearest_apps": [cands[i].key for i in idx.tolist()],
                "similarity": float(sims.max()),
                "n_candidates": len(cands),
                "n_observed_in_nearest": int((~imputed).sum()),
                "n_imputed": int(imputed.sum()),
                "fallback": fallback,
            },
        )


__all__ = [
    "NearestApplicationSelector",
    "compatible_candidates",
    "global_losses",
    "nearest_set",
    "score_experts",
]
