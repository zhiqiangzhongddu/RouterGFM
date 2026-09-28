"""Prediction integration (paper Eq. 1, 6-8) and the fixed-team rules of App. C / D.2.

Every rule maps one application's fixed team, fitted heads, and query
predictions to mixture weights ``alpha [N, K]``; :func:`mix` applies Eq. 1.

* ``uniform``: 1/K.
* ``global``: ``softmax(-mu_hat / tau)`` (RouterGFM-G, application-level).
* ``routergfm``: Eq. 6-8, centered residual transfer from the archive.
* ``no_centering``: raw local losses instead of centered residuals (control).
* ``shuffled``: ``routergfm`` on residuals permuted within the compat group
  (``archive.perturb_archive(..., 'shuffled')``), same retrieved records.
* ``simplex_stacking``: one weight vector ``softmax(theta)`` fitted on the
  support out-of-fold predictions (mean mixture routing loss).
* ``local_mlp``: ``softmax(MLP(z))`` fitted the same way on support contexts.

Only support labels are used (by the two fitted rules); query labels never are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from .applications import derive_seed
from .archive import Archive, perturb_archive
from .common import PRED_SPACE, REGRESSION
from .losses import RegressionNormalizer, routing_loss
from .router.retrieval import kernel_weights, local_estimate, mixture_weights, search

RULES = ("uniform", "global", "routergfm", "no_centering", "shuffled", "simplex_stacking", "local_mlp")
LOCAL_RULES = ("routergfm", "no_centering", "shuffled")  # rules that read archive evidence
_QUERY_CHUNK = 8192  # queries per key / search / kernel-weight pass


def mix(preds: torch.Tensor, alpha: torch.Tensor, family: str) -> torch.Tensor:
    """Eq. 1: ``F_a(x) = sum_e alpha_e(x) F_e(x)`` for ``preds [N, K, C]`` and ``alpha [N, K]`` or ``[K]``.

    Every family mixes linearly in its prediction space (class simplex,
    assay probabilities, or normalized regression units).
    """
    if family not in PRED_SPACE:
        raise ValueError(f"Unknown task family {family!r}.")
    preds = torch.as_tensor(preds).float()
    alpha = torch.as_tensor(alpha).to(preds)
    if alpha.dim() == 1:
        alpha = alpha.expand(preds.size(0), -1)
    return (alpha.unsqueeze(-1) * preds).sum(dim=1)


@dataclass
class LocalEvidence:
    """Archive records a deployment may retrieve (other groups, same CompatKey) and their keys (Eq. 6)."""

    model: Any  # RouterGFMModel; only ``keys(z, v)`` is used
    team_v: torch.Tensor  # [K, d] v_e = h^(0)_e of the team experts
    archive: Archive  # allowed records only
    record_keys: torch.Tensor  # [R, d_k] k_phi(c_i, v_{e_i})
    retrieval_j: int
    per_app_cap: int
    bandwidth: float


@torch.no_grad()
def retrieve(evidence: LocalEvidence, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Retrieved record indices and kernel weights ``[N, K, J]`` (CPU) for every (query, team expert).

    An empty archive gives all-zero weights (zero local correction).
    """
    n, k, j = z.size(0), evidence.team_v.size(0), int(evidence.retrieval_j)
    idx = torch.zeros(n, k, j, dtype=torch.long)
    w = torch.zeros(n, k, j)
    if len(evidence.archive) == 0 or n == 0 or j <= 0:
        return idx, w
    keys = evidence.record_keys
    device = keys.device
    allowed = torch.ones(keys.size(0), dtype=torch.bool, device=device)
    rec_app = evidence.archive.app.to(device)
    for e in range(k):
        v = evidence.team_v[e].to(device)
        for s in range(0, n, _QUERY_CHUNK):
            zc = z[s:s + _QUERY_CHUNK].to(device=device, dtype=torch.float32)
            q = evidence.model.keys(zc, v.expand(zc.size(0), -1))
            i, valid = search(q, keys, rec_app, allowed, j, int(evidence.per_app_cap))
            w[s:s + zc.size(0), e] = kernel_weights(q, keys[i], valid, float(evidence.bandwidth)).cpu()
            idx[s:s + zc.size(0), e] = i.cpu()
    return idx, w


@dataclass
class IntegrationContext:
    """What the rules of one deployment share: team scores, contexts, evidence, support OOF predictions."""

    family: str
    mu_hat: torch.Tensor  # [K] team scores (Eq. 4)
    tau: float
    rho: float
    z_query: torch.Tensor  # [N, D] standardized query descriptors
    cfg: Any
    seed: int
    evidence: Optional[LocalEvidence] = None
    support_oof: Optional[torch.Tensor] = None  # [N_s, K, C] out-of-fold support predictions
    support_target: Optional[torch.Tensor] = None  # raw support labels
    support_z: Optional[torch.Tensor] = None  # [N_s, D] standardized support descriptors
    normalizer: Optional[RegressionNormalizer] = None
    _retrieved: Optional[Tuple[torch.Tensor, torch.Tensor]] = field(default=None, repr=False)

    @property
    def num_queries(self) -> int:
        return int(self.z_query.size(0))

    @property
    def team_size(self) -> int:
        return int(self.mu_hat.numel())

    def retrieved(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(idx, w)`` of :func:`retrieve`, computed once and shared by the local rules."""
        if self._retrieved is None:
            self._retrieved = retrieve(self.evidence, self.z_query)
        return self._retrieved


def local_estimates(ctx: IntegrationContext, rule: str = "routergfm") -> torch.Tensor:
    """Local loss estimates ``r_hat [N, K]`` of a local rule (Eq. 7, or its no-centering control)."""
    if rule not in LOCAL_RULES:
        raise ValueError(f"{rule!r} is not a local rule ({LOCAL_RULES}).")
    mu = ctx.mu_hat.detach().float().cpu().expand(ctx.num_queries, -1)
    if ctx.evidence is None or len(ctx.evidence.archive) == 0:
        return mu.clone()
    idx, w = ctx.retrieved()
    archive = ctx.evidence.archive
    if rule == "no_centering":
        return local_estimate(mu, w, archive.r_local.float().cpu()[idx], ctx.rho, mode="raw")
    if rule == "shuffled":
        archive = perturb_archive(archive, "shuffled", derive_seed(ctx.seed, "shuffled"))
    return local_estimate(mu, w, archive.residual.float().cpu()[idx], ctx.rho, mode="centered")


# --------------------------------------------------------------------------- #
# Support-fitted rules
# --------------------------------------------------------------------------- #
def _support_inputs(ctx: IntegrationContext) -> Tuple[torch.Tensor, torch.Tensor]:
    if ctx.support_oof is None or ctx.support_target is None:
        raise ValueError("Support-fitted rules need out-of-fold support predictions and support labels.")
    target = ctx.support_target
    if ctx.family == REGRESSION:
        target = ctx.normalizer.transform(target)
    return ctx.support_oof.detach().float().cpu(), target


def _mean_risk(pred: torch.Tensor, target: torch.Tensor, family: str, reg_kind: str) -> torch.Tensor:
    loss = routing_loss(pred, target, family, reg_kind=reg_kind)
    valid = torch.isfinite(loss)
    return loss[valid].mean() if bool(valid.any()) else pred.sum() * 0.0


def _fit_stacking(ctx: IntegrationContext) -> torch.Tensor:
    """Simplex stacking: ``softmax(theta)`` minimizing the mean support mixture routing loss."""
    ic, reg_kind = ctx.cfg.moe.routergfm.integration, str(ctx.cfg.moe.routergfm.loss.regression)
    preds, target = _support_inputs(ctx)
    theta = torch.zeros(ctx.team_size, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=float(ic.stacking_lr))
    for _ in range(int(ic.stacking_epochs)):
        optimizer.zero_grad()
        _mean_risk(mix(preds, torch.softmax(theta, 0), ctx.family), target, ctx.family, reg_kind).backward()
        optimizer.step()
    return torch.softmax(theta.detach(), 0)


def _fit_local_mlp(ctx: IntegrationContext) -> nn.Module:
    """Local-MLP gate ``softmax(MLP(z))`` minimizing the mean support mixture routing loss (weight decay)."""
    ic, reg_kind = ctx.cfg.moe.routergfm.integration, str(ctx.cfg.moe.routergfm.loss.regression)
    preds, target = _support_inputs(ctx)
    if ctx.support_z is None:
        raise ValueError("local_mlp needs the standardized support descriptors.")
    z = ctx.support_z.float().cpu()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(ctx.seed, "local_mlp"))
        net = nn.Sequential(
            nn.Linear(z.size(1), int(ic.local_mlp_hidden)), nn.ReLU(), nn.Linear(int(ic.local_mlp_hidden), ctx.team_size)
        )
    optimizer = torch.optim.Adam(net.parameters(), lr=float(ic.local_mlp_lr), weight_decay=float(ic.local_mlp_weight_decay))
    for _ in range(int(ic.local_mlp_epochs)):
        optimizer.zero_grad()
        alpha = torch.softmax(net(z), dim=-1)
        _mean_risk(mix(preds, alpha, ctx.family), target, ctx.family, reg_kind).backward()
        optimizer.step()
    return net.eval()


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def integration_weights(rule: str, ctx: IntegrationContext) -> torch.Tensor:
    """Mixture weights ``alpha [N, K]`` (float32, CPU) of one integration rule."""
    n, k = ctx.num_queries, ctx.team_size
    if rule == "uniform":
        return torch.full((n, k), 1.0 / k)
    if rule == "global":
        return mixture_weights(ctx.mu_hat.detach().float().cpu(), ctx.tau).expand(n, -1).clone()
    if rule in LOCAL_RULES:
        return mixture_weights(local_estimates(ctx, rule), ctx.tau)
    if rule == "simplex_stacking":
        return _fit_stacking(ctx).expand(n, -1).clone()
    if rule == "local_mlp":
        net = _fit_local_mlp(ctx)
        z = ctx.z_query.float().cpu()
        with torch.no_grad():
            chunks = [torch.softmax(net(z[s:s + _QUERY_CHUNK]), -1) for s in range(0, n, _QUERY_CHUNK)]
        return torch.cat(chunks) if chunks else torch.zeros(0, k)
    raise ValueError(f"Unknown integration rule {rule!r}; expected one of {RULES}.")


__all__ = [
    "IntegrationContext",
    "LOCAL_RULES",
    "LocalEvidence",
    "RULES",
    "integration_weights",
    "local_estimates",
    "mix",
    "retrieve",
]
