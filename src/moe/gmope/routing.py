"""GMoPE routing, orthogonality and confidence-aggregation primitives.

Pure tensor functions (Eqs. 10-15 of Wang et al., 2025) plus the RNG context
used to give every expert identical sampling within one routing decision.
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F


def soft_orthogonality_loss(prompts: torch.Tensor) -> torch.Tensor:
    """Eq. 15: mean over ordered pairs m != n of exp(cos(p_m, p_n)); 0 for M == 1."""
    num = int(prompts.size(0))
    if num < 2:
        return prompts.sum() * 0.0
    normed = F.normalize(prompts, dim=-1)
    cos = normed @ normed.t()
    off_diag = ~torch.eye(num, dtype=torch.bool, device=prompts.device)
    return torch.exp(cos[off_diag]).sum() / float(num * (num - 1))


def scores_from_losses(losses: torch.Tensor) -> torch.Tensor:
    """Rawscore of Eq. 9 with the sign flipped: lower loss -> higher (detached) score."""
    return -losses.detach()


def _topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` highest scores; ties go to the lower expert index."""
    k = max(1, min(int(k), int(scores.numel())))
    return torch.sort(-scores, stable=True).indices[:k]


def soft_topk_gate(scores: torch.Tensor, k: int, tau: float) -> torch.Tensor:
    """Eq. 10: softmax(score / tau) over the top-K experts, zero elsewhere."""
    idx = _topk_indices(scores, k)
    gate = torch.zeros_like(scores, dtype=torch.float32)
    gate[idx] = torch.softmax(scores[idx].float() / float(tau), dim=0)
    return gate


def hard_topk_gate(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Eq. 11: 1/K on the top-K experts, zero elsewhere."""
    idx = _topk_indices(scores, k)
    gate = torch.zeros_like(scores, dtype=torch.float32)
    gate[idx] = 1.0 / float(idx.numel())
    return gate


def confidence_weights(
    logits: torch.Tensor,
    *,
    task_type: str,
    multilabel: bool,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Eqs. 12-13: ``logits [M, B, out] -> omega [M, B]`` (columns sum to 1).

    ``alpha_m = 1 - H(y_m) / log C`` with softmax entropy for multiclass heads,
    binary entropy (C = 2) for single-logit heads, and the mean binary entropy
    over all assays for multilabel heads. Regression has no entropy and uses
    uniform weights, as does any instance whose confidences are all zero.
    """
    num_experts = int(logits.size(0))
    uniform = torch.full(logits.shape[:2], 1.0 / num_experts, device=logits.device)
    if str(task_type).lower() == "regression":
        return uniform
    z = logits.float()
    if multilabel or z.size(-1) == 1:
        p = torch.sigmoid(z)
        entropy = ((torch.special.entr(p) + torch.special.entr(1.0 - p)) / math.log(2.0)).mean(dim=-1)
    else:
        p = torch.softmax(z, dim=-1)
        entropy = torch.special.entr(p).sum(dim=-1) / math.log(float(z.size(-1)))
    alpha = (1.0 - entropy).clamp_min(0.0)
    total = alpha.sum(dim=0, keepdim=True)
    return torch.where(total > eps, alpha / total.clamp_min(eps), uniform)


def aggregate_embeddings(pooled: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Eq. 14: ``h_final = sum_m omega_m h_m`` for ``pooled [M, B, d]``, ``omega [M, B]``."""
    return (omega.unsqueeze(-1).to(pooled.dtype) * pooled).sum(dim=0)


@contextmanager
def shared_rng(seed: int):
    """Run the body with python/numpy/torch RNGs seeded to ``seed``, then restore them.

    Used so every expert scored on a batch sees identical sampling (negative
    edges, dropout masks, augmentations) and the gradient pass reproduces it.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


__all__ = [
    "aggregate_embeddings",
    "confidence_weights",
    "hard_topk_gate",
    "scores_from_losses",
    "shared_rng",
    "soft_orthogonality_loss",
    "soft_topk_gate",
]
