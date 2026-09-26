"""Utility helpers for the GraphPrompt finetune method.

Extracted from ``graphprompt.py`` to reduce file size and improve
testability of pure functions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def similarity_logits(
    embeddings: torch.Tensor,
    centers: torch.Tensor,
    score_mode: str,
    tau: float,
    is_train: bool,
) -> torch.Tensor:
    """Compute similarity logits between embeddings and prototype centers.

    Args:
        embeddings: (N, D) sample embeddings.
        centers: (K, D) prototype centers.
        score_mode: ``"cosine"``, ``"distance"``, or ``"official"``
            (reciprocal-normalized distance for training, negative for eval).
        tau: Temperature scaling for cosine similarity.
        is_train: Whether to use training-mode scoring for ``"official"`` mode.
    """
    if score_mode == "cosine":
        logits = F.cosine_similarity(embeddings.unsqueeze(1), centers.unsqueeze(0), dim=-1)
        return logits / max(tau, 1e-12)
    n = embeddings.size(0)
    k = centers.size(0)
    emb_power = torch.sum(embeddings * embeddings, dim=1, keepdim=True).expand(n, k)
    center_power = torch.sum(centers * centers, dim=1).expand(n, k)
    distance = emb_power + center_power - 2 * torch.mm(embeddings, centers.transpose(0, 1))
    normed_distance = F.normalize(distance, dim=1)
    if score_mode == "distance":
        return normed_distance
    if is_train:
        return torch.reciprocal(normed_distance.clamp_min(1e-12))
    return -1.0 * normed_distance


def fill_missing_centers(
    centers: torch.Tensor,
    counts: torch.Tensor,
    fallback: torch.Tensor | None,
) -> torch.Tensor:
    """Replace zero-count centers with fallback values."""
    if fallback is None:
        return centers
    present = counts.view(-1) > 0
    if bool(present.all().item()):
        return centers
    mixed = centers.clone()
    mixed[~present] = fallback[~present]
    return mixed
