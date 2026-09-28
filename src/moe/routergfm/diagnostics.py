"""Evaluation diagnostics (paper Sec. 4.1, App. A.2, B.4). Evaluation only: callers pass query labels
or the target's recorded history after every prediction and weight is fixed.

* hit@K / regret@K: inclusion of a best eligible expert, and the risk gap
  between the best shortlisted and the best eligible expert.
* mixture risk: mean routing loss of the MIXED predictions (never the weighted
  individual losses).
* evaluation cells: one k-means partition of the standardized query contexts,
  shared by every rule; worst-cell risk is the largest cell-average loss.
* local-winner coverage: mass of cells whose best eligible expert is in the team;
  winner agreement: share of instances whose highest-weight expert is the
  cellwise best team expert.
* specialization index: Prop. 2 ``Delta_loc = min_e sum_j m_j r_je - sum_j m_j min_e r_je``
  over cells.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from src.utils.metrics import compute_supervised_metrics

from .archive import kmeans
from .common import REGRESSION
from .losses import RegressionNormalizer, routing_loss, to_metric_inputs


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def _finite(mu_true: Mapping[str, float]) -> Dict[str, float]:
    return {e: float(v) for e, v in mu_true.items() if v is not None and math.isfinite(float(v))}


def hit_at_k(team: Sequence[str], mu_true: Mapping[str, float], atol: float = 1e-12) -> float:
    """1 if the team contains a best eligible expert (ties within ``atol`` count), else 0; NaN without risks.

    ``mu_true`` maps every eligible expert to its target risk; non-finite entries are ignored.
    """
    risk = _finite(mu_true)
    if not risk:
        return float("nan")
    best = min(risk.values())
    return float(any(e in risk and risk[e] <= best + atol for e in team))


def regret_at_k(team: Sequence[str], mu_true: Mapping[str, float]) -> float:
    """Best shortlisted risk minus best eligible risk (NaN when the team has no finite risk)."""
    risk = _finite(mu_true)
    team_risk = [risk[e] for e in team if e in risk]
    if not team_risk:
        return float("nan")
    return max(0.0, min(team_risk) - min(risk.values()))


# --------------------------------------------------------------------------- #
# Risks
# --------------------------------------------------------------------------- #
def routing_risks(
    pred: torch.Tensor,
    target: torch.Tensor,
    family: str,
    *,
    normalizer: Optional[RegressionNormalizer] = None,
    reg_kind: str = "abs",
) -> torch.Tensor:
    """Per-instance routing losses ``[N]`` (or ``[N, K]`` for ``pred [N, K, C]``); ``target`` raw, NaN = invalid."""
    if family == REGRESSION:
        target = normalizer.transform(target)
    return routing_loss(pred, target, family, reg_kind=reg_kind)


def mixture_risk(
    pred_mix: torch.Tensor,
    target: torch.Tensor,
    family: str,
    *,
    normalizer: Optional[RegressionNormalizer] = None,
    reg_kind: str = "abs",
) -> float:
    """Mean routing loss of the mixed predictions over valid instances (the reported "Brier risk")."""
    loss = routing_risks(pred_mix, target, family, normalizer=normalizer, reg_kind=reg_kind)
    valid = torch.isfinite(loss)
    return float(loss[valid].mean()) if bool(valid.any()) else float("nan")


def task_metrics(
    pred_mix: torch.Tensor, target: torch.Tensor, family: str, normalizer: Optional[RegressionNormalizer] = None
) -> Dict[str, float]:
    """``compute_supervised_metrics`` of family-space predictions (acc / auc / f1, or raw-unit mae / mse)."""
    logits, labels, task_type = to_metric_inputs(pred_mix, family, normalizer, target=target)
    return {k: float(v) for k, v in compute_supervised_metrics(logits, labels, task_type).items()}


# --------------------------------------------------------------------------- #
# Context cells
# --------------------------------------------------------------------------- #
def eval_cells(z_std_query: torch.Tensor, cfg, seed: int) -> torch.Tensor:
    """Cell id per query: k-means with ``integration.eval_cells`` clusters (capped at N)."""
    rg = cfg.moe.routergfm
    if z_std_query.size(0) == 0:
        return torch.empty(0, dtype=torch.long)
    assign, _ = kmeans(z_std_query.float().cpu(), int(rg.integration.eval_cells), seed, num_iters=int(rg.archive.kmeans_iters))
    return assign


def cell_means(values: torch.Tensor, cells: torch.Tensor, num_cells: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cell averages of finite values and their counts: ``[B]`` for ``[N]`` input, ``[B, K]`` for ``[N, K]``.

    Cells without a finite value average to NaN.
    """
    v = values.float().cpu()
    squeeze = v.dim() == 1
    if squeeze:
        v = v.unsqueeze(1)
    cells = cells.long().cpu()
    b = int(num_cells) if num_cells is not None else (int(cells.max()) + 1 if cells.numel() else 0)
    valid = torch.isfinite(v)
    sums = torch.zeros(b, v.size(1)).index_add_(0, cells, torch.where(valid, v, torch.zeros_like(v)))
    counts = torch.zeros(b, v.size(1)).index_add_(0, cells, valid.float())
    means = torch.where(counts > 0, sums / counts.clamp_min(1), torch.full_like(sums, float("nan")))
    return (means[:, 0], counts[:, 0]) if squeeze else (means, counts)


def worst_cell_risk(losses: torch.Tensor, cells: torch.Tensor) -> float:
    """Largest cell-average loss over cells with valid instances."""
    means, _ = cell_means(losses, cells)
    means = means[torch.isfinite(means)]
    return float(means.max()) if means.numel() else float("nan")


def cell_mass(losses: torch.Tensor, cells: torch.Tensor, num_cells: int) -> torch.Tensor:
    """Instances per cell that have at least one finite loss."""
    row_ok = torch.isfinite(losses.float().cpu()).any(dim=1)
    return torch.zeros(num_cells).index_add_(0, cells.long().cpu(), row_ok.float())


def winner_coverage(
    cells: torch.Tensor, losses_all: torch.Tensor, team_cols: Sequence[int], atol: float = 1e-12
) -> float:
    """Mass of cells whose best eligible expert (lowest cell-average loss) is in the team.

    ``losses_all [N, |E_a|]``: per-instance losses of every eligible expert (NaN
    = unrecorded/invalid); ``team_cols``: the team's columns. Ties within ``atol`` count.
    """
    if cells.numel() == 0 or len(team_cols) == 0:
        return float("nan")
    b = int(cells.max()) + 1
    means, _ = cell_means(losses_all, cells, b)
    mass = cell_mass(losses_all, cells, b)
    filled = torch.nan_to_num(means, nan=float("inf"))
    best = filled.min(dim=1).values
    team_best = filled[:, torch.as_tensor(list(team_cols), dtype=torch.long)].min(dim=1).values
    defined = torch.isfinite(best) & (mass > 0)
    if not bool(defined.any()):
        return float("nan")
    covered = defined & (team_best <= best + atol)
    return float(mass[covered].sum() / mass[defined].sum())


def winner_agreement(alpha: torch.Tensor, cells: torch.Tensor, losses_team: torch.Tensor) -> float:
    """Share of valid instances whose highest-weight expert is the cellwise best team expert.

    Weight ties (e.g. uniform weights) resolve to the first team member (lowest ``mu_hat``).
    """
    if cells.numel() == 0:
        return float("nan")
    b = int(cells.max()) + 1
    means, _ = cell_means(losses_team, cells, b)
    filled = torch.nan_to_num(means, nan=float("inf"))
    winner = filled.argmin(dim=1)
    has_winner = torch.isfinite(filled.min(dim=1).values)
    cells = cells.long().cpu()
    ok = has_winner[cells] & torch.isfinite(losses_team.float().cpu()).any(dim=1)
    if not bool(ok.any()):
        return float("nan")
    agree = alpha.float().cpu().argmax(dim=1) == winner[cells]
    return float(agree[ok].float().mean())


def specialization_index(cell_risk: torch.Tensor, mass: torch.Tensor) -> float:
    """Prop. 2 ``Delta_loc = R_fixed - R_loc`` on a cell partition (``cell_risk [B, E]``, cell ``mass [B]``).

    Experts without a finite risk in every populated cell are left out.
    """
    mass = mass.float().cpu()
    keep = mass > 0
    risk = cell_risk.float().cpu()[keep]
    risk = risk[:, torch.isfinite(risk).all(dim=0)]
    if risk.numel() == 0:
        return float("nan")
    m = mass[keep] / mass[keep].sum()
    fixed = (m.unsqueeze(1) * risk).sum(dim=0).min()
    local = (m * risk.min(dim=1).values).sum()
    return float(fixed - local)


__all__ = [
    "cell_mass",
    "cell_means",
    "eval_cells",
    "hit_at_k",
    "mixture_risk",
    "regret_at_k",
    "routing_risks",
    "specialization_index",
    "task_metrics",
    "winner_agreement",
    "winner_coverage",
    "worst_cell_risk",
]
