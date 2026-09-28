"""Candidate expert set of a matched-pool mixture (DESIGN 11), shared by SAGMM-PE and META-DES.

``baselines.candidate_rule``:

* ``eligible``: all of E_a.
* ``historical_mean``: the ``candidate_pool`` experts of E_a with the lowest mean
  historical rank. Each same-family, leave-one-dataset-out historical
  application ranks the experts it observed by Eq. 2 average loss; ranks are
  normalised to [0, 1] by that application's number of observed experts and
  averaged. Experts never observed rank last; ties keep catalog order.
* ``random``: ``candidate_pool`` experts of E_a, seeded by the application.

``candidate_pool <= 0`` keeps all of E_a under every rule. The target
application's own labels and history are never read. The result is returned in
E_a (catalog) order.
"""

from __future__ import annotations

from typing import List

import torch

from ..applications import derive_seed
from ..common import AppSpec
from .selection.common import application_averages

CANDIDATE_RULES = ("historical_mean", "eligible", "random")


def mean_historical_rank(mu: torch.Tensor) -> torch.Tensor:
    """``[E]`` mean normalised rank over the rows of ``mu [B, E]`` (NaN = unobserved; inf if never observed)."""
    ranks = torch.full_like(mu, float("nan"))
    for b in range(mu.size(0)):
        observed = torch.nonzero(torch.isfinite(mu[b]), as_tuple=False).view(-1)
        if observed.numel() == 0:
            continue
        order = observed[torch.argsort(mu[b, observed], stable=True)]
        ranks[b, order] = torch.arange(order.numel(), dtype=mu.dtype) / max(order.numel() - 1, 1)
    seen = torch.isfinite(ranks)
    total = torch.where(seen, ranks, torch.zeros_like(ranks)).sum(0)
    count = seen.sum(0)
    return torch.where(count > 0, total / count.clamp_min(1), torch.full_like(total, float("inf")))


def candidate_experts(cfg, app: AppSpec, infra) -> List[str]:
    """Candidate expert ids for *app* under ``baselines.candidate_rule`` (E_a order)."""
    b = cfg.moe.routergfm.baselines
    rule = str(b.candidate_rule).lower()
    pool = infra.compatible_pool(app)
    size = len(pool) if int(b.candidate_pool) <= 0 else min(int(b.candidate_pool), len(pool))
    if rule == "eligible":
        return list(pool)
    if rule == "random":
        generator = torch.Generator().manual_seed(derive_seed(app.seed, "candidates", app.key))
        chosen = torch.randperm(len(pool), generator=generator)[:size]
    elif rule == "historical_mean":
        family = infra.task_family(app)
        history = [h for h in infra.historical_applications(app) if infra.task_family(h) == family]
        if not history:
            print(f"[RouterGFM][candidates] {app.key}: no same-family historical application; E_a order.")
        mu, _ = application_averages(infra, history, pool)
        chosen = torch.argsort(mean_historical_rank(mu), stable=True)[:size]
    else:
        raise ValueError(f"Unknown baselines.candidate_rule {rule!r} (expected one of {CANDIDATE_RULES}).")
    return [pool[i] for i in sorted(chosen.tolist())]


__all__ = ["CANDIDATE_RULES", "candidate_experts", "mean_historical_rank"]
