"""Router training objectives (Eq. 9-10): Huber, ListMLE, local squared loss."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def huber(mu_hat: torch.Tensor, mu_bar: torch.Tensor, delta: float) -> torch.Tensor:
    """Mean Huber loss between estimated and recorded application-average losses (Eq. 9)."""
    return F.huber_loss(mu_hat, mu_bar.to(mu_hat.dtype), reduction="mean", delta=float(delta))


def listmle(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ListMLE (Xia et al., 2008): ``-log P(pi | scores)`` under the Plackett-Luce model.

    ``pi`` orders the list by ascending ``targets`` (lowest recorded loss first),
    so callers pass ``scores = -mu_hat``. The negative log-likelihood is
    averaged over list positions so its scale does not grow with the number
    of evaluated experts. Lists shorter than two give zero.
    """
    if scores.numel() < 2:
        return scores.sum() * 0.0
    order = torch.argsort(targets, stable=True)
    s = scores[order]
    tail_lse = torch.logcumsumexp(s.flip(0), dim=0).flip(0)
    return (tail_lse - s).mean()


def local_sq(
    r_hat: torch.Tensor,
    loss: torch.Tensor,
    expert: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Squared error between local estimates and recorded per-instance losses (Eq. 10).

    Pairs with a non-finite recorded loss are ignored. With ``expert`` given,
    errors are averaged within each expert and then across experts, exactly as
    Eq. 10; otherwise a plain mean over pairs.
    """
    ok = torch.isfinite(loss)
    if not bool(ok.any()):
        return r_hat.sum() * 0.0
    err = (r_hat[ok] - loss[ok].to(r_hat.dtype)).pow(2)
    if expert is None:
        return err.mean()
    _, inverse = torch.unique(expert[ok], return_inverse=True)
    num = inverse.max() + 1
    sums = err.new_zeros(int(num)).index_add_(0, inverse, err)
    counts = torch.bincount(inverse, minlength=int(num)).to(err.dtype)
    return (sums / counts).mean()


__all__ = ["huber", "listmle", "local_sq"]
