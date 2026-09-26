"""Shared train-only class balancing for multilabel finetuning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.finetune.target_stats import (
    iter_training_items_deterministically,
    preserve_loader_rng,
)
from src.utils.dataset_helpers import normalize_node_mask


MASKED_BCE = "masked_bce"
MACRO_BALANCED_BCE = "macro_balanced_bce"
_VALID_MULTILABEL_LOSSES = frozenset({MASKED_BCE, MACRO_BALANCED_BCE})


def resolve_multilabel_loss(cfg) -> str:
    """Return and validate the shared finetuning multilabel objective."""
    finetune_cfg = getattr(cfg, "finetune", None)
    raw = getattr(finetune_cfg, "multilabel_loss", MASKED_BCE)
    mode = str(raw or MASKED_BCE).strip().lower().replace("-", "_")
    if mode not in _VALID_MULTILABEL_LOSSES:
        raise ValueError(
            f"Unknown finetune.multilabel_loss='{raw}'; expected one of "
            f"{sorted(_VALID_MULTILABEL_LOSSES)}."
        )
    return mode


class MacroBalancedBCELoss(nn.Module):
    """Macro task/class-balanced BCE fitted from training labels only.

    Each eligible target contributes equal positive and negative mass. A
    target is eligible exactly when its training labels contain at least one
    observed positive and one observed negative; targets with one or neither
    class cannot define an empirical ROC-AUC and contribute no loss.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        target_dim: int,
        task_level: str,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.target_dim = max(1, int(target_dim))
        self.task_level = str(task_level or "graph").lower()
        self.register_buffer(
            "positive_count", torch.zeros(self.target_dim, dtype=torch.long)
        )
        self.register_buffer(
            "negative_count", torch.zeros(self.target_dim, dtype=torch.long)
        )
        self.register_buffer(
            "eligible", torch.zeros(self.target_dim, dtype=torch.bool)
        )
        self.register_buffer("train_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("uses_signed_labels", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("ready", torch.zeros(1, dtype=torch.bool))

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.ready.item())

    def _as_target_matrix(self, values: torch.Tensor, *, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(values).float()
        if tensor.numel() % self.target_dim != 0:
            raise ValueError(
                f"Cannot reshape {name} with {tensor.numel()} values into "
                f"{self.target_dim} multilabel target(s)."
            )
        return tensor.reshape(-1, self.target_dim)

    def _train_labels_from_item(self, item) -> torch.Tensor | None:
        labels = getattr(item, "y", None)
        if labels is None and torch.is_tensor(item):
            labels = item
        if labels is None:
            return None
        labels = torch.as_tensor(labels)
        if self.task_level == "node":
            num_nodes = int(labels.size(0)) if labels.dim() > 0 else 1
            mask = normalize_node_mask(
                item,
                "train_mask",
                labels.device,
                num_nodes=num_nodes,
            )
            labels = labels[mask]
        return self._as_target_matrix(labels, name="training labels")

    @torch.no_grad()
    def fit(self, train_loader) -> bool:
        """Fit class counts without changing loader order or any RNG stream."""
        if not self.enabled:
            return False

        positives = torch.zeros(self.target_dim, dtype=torch.long)
        zeros = torch.zeros(self.target_dim, dtype=torch.long)
        negatives = torch.zeros(self.target_dim, dtype=torch.long)
        train_count = 0

        with preserve_loader_rng(train_loader):
            for item in iter_training_items_deterministically(train_loader):
                labels = self._train_labels_from_item(item)
                if labels is None or labels.numel() == 0:
                    continue
                labels = labels.detach().cpu()
                finite = torch.isfinite(labels)
                positives += (finite & (labels > 0)).sum(dim=0)
                zeros += (finite & (labels == 0)).sum(dim=0)
                negatives += (finite & (labels < 0)).sum(dim=0)
                train_count += int(labels.size(0))

        if train_count <= 0:
            raise RuntimeError(
                "Macro-balanced multilabel BCE requires at least one training example."
            )

        uses_signed = bool((negatives > 0).any().item())
        negative_count = negatives if uses_signed else zeros
        eligible = (positives > 0) & (negative_count > 0)

        self.positive_count.copy_(positives.to(self.positive_count.device))
        self.negative_count.copy_(negative_count.to(self.negative_count.device))
        self.eligible.copy_(eligible.to(self.eligible.device))
        self.train_count.fill_(train_count)
        self.uses_signed_labels.fill_(uses_signed)
        self.ready.fill_(True)

        summary = self.summary()
        print(
            "[Finetune][Multilabel] Train-only macro-balanced BCE: "
            f"{summary['eligible_targets']}/{self.target_dim} eligible target(s), "
            f"positive-only={summary['positive_only_targets']}, "
            f"negative-only={summary['negative_only_targets']}, "
            f"all-missing={summary['all_missing_targets']}, "
            f"training examples={train_count}."
        )
        return True

    def targets_and_valid(
        self, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the fitted label convention to targets and validity masks."""
        raw = self._as_target_matrix(labels, name="multilabel labels")
        valid = torch.isfinite(raw)
        if bool(self.uses_signed_labels.item()):
            valid = valid & (raw != 0)
            raw = (raw + 1.0) / 2.0
        return raw.clamp(min=0.0, max=1.0), valid

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            raise RuntimeError("MacroBalancedBCELoss is disabled for this objective.")
        if not self.active:
            raise RuntimeError(
                "Macro-balanced multilabel BCE must be fitted from the training "
                "loader before loss computation."
            )

        logits_mat = self._as_target_matrix(logits, name="multilabel logits").float()
        targets, valid = self.targets_and_valid(labels)
        targets = targets.to(device=logits_mat.device, dtype=logits_mat.dtype)
        valid = valid.to(device=logits_mat.device)
        if targets.shape != logits_mat.shape:
            raise ValueError(
                "Multilabel logits and labels must have identical matrix shapes; "
                f"got {tuple(logits_mat.shape)} and {tuple(targets.shape)}."
            )

        eligible = self.eligible.to(device=logits_mat.device)
        eligible_count = int(eligible.sum().item())
        if eligible_count == 0:
            return logits_mat.sum() * 0.0

        positive_count = self.positive_count.to(
            device=logits_mat.device, dtype=logits_mat.dtype
        ).clamp_min(1.0)
        negative_count = self.negative_count.to(
            device=logits_mat.device, dtype=logits_mat.dtype
        ).clamp_min(1.0)
        weights = torch.where(
            targets > 0.5,
            0.5 / positive_count,
            0.5 / negative_count,
        )
        mask = valid & eligible.unsqueeze(0)
        safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
        loss_matrix = F.binary_cross_entropy_with_logits(
            logits_mat,
            safe_targets,
            reduction="none",
        )
        batch_count = max(1, int(logits_mat.size(0)))
        sample_scale = float(self.train_count.item()) / float(batch_count)
        return (
            (loss_matrix * weights * mask.to(logits_mat.dtype)).sum()
            * sample_scale
            / float(eligible_count)
        )

    def summary(self) -> dict[str, int | bool | str]:
        """Return compact JSON-safe train-label provenance."""
        positive = self.positive_count.detach().cpu()
        negative = self.negative_count.detach().cpu()
        return {
            "mode": MACRO_BALANCED_BCE,
            "ready": bool(self.ready.item()),
            "target_dim": self.target_dim,
            "train_examples": int(self.train_count.item()),
            "eligible_targets": int(((positive > 0) & (negative > 0)).sum().item()),
            "positive_only_targets": int(((positive > 0) & (negative == 0)).sum().item()),
            "negative_only_targets": int(((positive == 0) & (negative > 0)).sum().item()),
            "all_missing_targets": int(((positive == 0) & (negative == 0)).sum().item()),
            "uses_signed_labels": bool(self.uses_signed_labels.item()),
        }


__all__ = [
    "MACRO_BALANCED_BCE",
    "MASKED_BCE",
    "MacroBalancedBCELoss",
    "resolve_multilabel_loss",
]
