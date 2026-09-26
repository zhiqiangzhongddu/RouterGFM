"""Shared train-only regression normalization and objectives."""

from __future__ import annotations

import torch
from torch import nn

from src.finetune.target_stats import (
    iter_training_items_deterministically,
    preserve_loader_rng,
)
from src.utils.dataset_helpers import normalize_node_mask
from src.utils.parsing import to_bool


NORMALIZED_MSE = "normalized_mse"
METRIC_MAE = "metric_mae"
_VALID_REGRESSION_LOSSES = frozenset({NORMALIZED_MSE, METRIC_MAE})


def resolve_regression_loss(cfg) -> str:
    """Return and validate the shared finetuning regression objective."""
    finetune_cfg = getattr(cfg, "finetune", None)
    raw = getattr(finetune_cfg, "regression_loss", NORMALIZED_MSE)
    mode = str(raw or NORMALIZED_MSE).strip().lower().replace("-", "_")
    if mode not in _VALID_REGRESSION_LOSSES:
        raise ValueError(
            f"Unknown finetune.regression_loss='{raw}'; expected one of "
            f"{sorted(_VALID_REGRESSION_LOSSES)}."
        )
    return mode


def resolve_regression_target_normalization(cfg) -> bool:
    """Return the effective finetune-wide regression-normalization flag."""
    finetune_cfg = getattr(cfg, "finetune", None)
    shared = getattr(finetune_cfg, "normalize_regression_targets", True)
    return to_bool(shared)


class RegressionTargetNormalizer(nn.Module):
    """Per-target z-scoring fitted exclusively from a training loader."""

    def __init__(
        self,
        *,
        enabled: bool,
        target_dim: int,
        task_level: str,
        loss_mode: str = NORMALIZED_MSE,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.target_dim = max(1, int(target_dim))
        self.task_level = str(task_level or "graph").lower()
        self.loss_mode = str(loss_mode)
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(self.target_dim))
        self.register_buffer("std", torch.ones(self.target_dim))
        self.register_buffer("count", torch.zeros(self.target_dim, dtype=torch.long))
        self.register_buffer("ready", torch.zeros(1, dtype=torch.bool))

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.ready.item())

    def _as_target_matrix(self, values: torch.Tensor, *, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(values).float()
        if tensor.numel() % self.target_dim != 0:
            raise ValueError(
                f"Cannot reshape {name} with {tensor.numel()} values into "
                f"{self.target_dim} regression target(s)."
            )
        return tensor.reshape(-1, self.target_dim)

    def _train_labels_from_batch(self, batch) -> torch.Tensor | None:
        labels = getattr(batch, "y", None)
        if labels is None:
            return None
        labels = torch.as_tensor(labels)
        if self.task_level == "node":
            num_nodes = int(labels.size(0)) if labels.dim() > 0 else 1
            mask = normalize_node_mask(
                batch,
                "train_mask",
                labels.device,
                num_nodes=num_nodes,
            )
            labels = labels[mask]
        return self._as_target_matrix(labels, name="training labels")

    @torch.no_grad()
    def fit(self, train_loader) -> bool:
        """Fit finite-value statistics from *train_loader* and no other split."""
        if not self.enabled:
            return False

        sums = torch.zeros(self.target_dim, dtype=torch.float64)
        squared_sums = torch.zeros(self.target_dim, dtype=torch.float64)
        counts = torch.zeros(self.target_dim, dtype=torch.long)
        with preserve_loader_rng(train_loader):
            for batch in iter_training_items_deterministically(train_loader):
                labels = self._train_labels_from_batch(batch)
                if labels is None or labels.numel() == 0:
                    continue
                labels = labels.detach().cpu().to(torch.float64)
                valid = torch.isfinite(labels)
                safe = torch.where(valid, labels, torch.zeros_like(labels))
                sums += safe.sum(dim=0)
                squared_sums += (safe * safe).sum(dim=0)
                counts += valid.sum(dim=0)

        if not bool((counts > 0).any().item()):
            raise RuntimeError(
                "Regression target normalization requires at least one finite "
                "training label."
            )

        safe_counts = counts.clamp_min(1).to(torch.float64)
        means = sums / safe_counts
        variances = (squared_sums / safe_counts - means.square()).clamp_min(0.0)
        stds = variances.sqrt().clamp_min(self.eps)
        # An entirely missing target is ignored by the supervised loss.  Keep
        # an identity transform for that column rather than inventing stats.
        missing = counts == 0
        means[missing] = 0.0
        stds[missing] = 1.0

        self.mean.copy_(means.to(device=self.mean.device, dtype=self.mean.dtype))
        self.std.copy_(stds.to(device=self.std.device, dtype=self.std.dtype))
        self.count.copy_(counts.to(device=self.count.device))
        self.ready.fill_(True)
        print(
            "[Finetune][Regression] Train-only target normalization: "
            f"{self.target_dim} target(s), mean~{float(self.mean.mean()):.4f}, "
            f"std~{float(self.std.mean()):.4f}."
        )
        return True

    def normalize_targets(self, labels: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return torch.as_tensor(labels)
        original_shape = torch.as_tensor(labels).shape
        matrix = self._as_target_matrix(labels, name="regression labels")
        mean = self.mean.to(device=matrix.device, dtype=matrix.dtype)
        std = self.std.to(device=matrix.device, dtype=matrix.dtype)
        return ((matrix - mean) / std).reshape(original_shape)

    def denormalize_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return torch.as_tensor(predictions)
        original_shape = torch.as_tensor(predictions).shape
        matrix = self._as_target_matrix(predictions, name="regression predictions")
        mean = self.mean.to(device=matrix.device, dtype=matrix.dtype)
        std = self.std.to(device=matrix.device, dtype=matrix.dtype)
        return (matrix * std + mean).reshape(original_shape)

    def metric_mae_weights(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return train-std weights with mean one over observed targets."""
        if not self.active:
            raise RuntimeError(
                "Metric-aligned regression MAE requires target statistics fitted "
                "from the training loader."
            )
        observed = self.count.to(device=device) > 0
        if not bool(observed.any().item()):
            raise RuntimeError(
                "Metric-aligned regression MAE requires at least one observed "
                "training target."
            )
        std = self.std.to(device=device, dtype=dtype)
        scale = std[observed].mean().clamp_min(self.eps)
        return torch.where(observed, std / scale, torch.zeros_like(std))

    def metric_aligned_mae_loss(
        self,
        predictions: torch.Tensor,
        normalized_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Train-std-weighted L1 proportional to original-unit flat MAE.

        For each observed target ``j``, denormalization multiplies residuals
        by its train-only standard deviation ``std[j]``. Weighting normalized
        residuals by ``std[j] / mean(std)`` therefore differs from reported
        original-unit flattened MAE only by a positive train-only constant.
        """
        pred = self._as_target_matrix(
            predictions, name="normalized regression predictions"
        )
        target = self._as_target_matrix(
            normalized_targets, name="normalized regression targets"
        ).to(device=pred.device, dtype=pred.dtype)
        if pred.shape != target.shape:
            raise ValueError(
                "Regression predictions and targets must have identical matrix "
                f"shapes; got {tuple(pred.shape)} and {tuple(target.shape)}."
            )

        observed = (self.count.to(device=pred.device) > 0).unsqueeze(0)
        valid = torch.isfinite(target) & observed
        if not bool(valid.any().item()):
            return pred.sum() * 0.0

        safe_pred = torch.where(valid, pred, torch.zeros_like(pred))
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        weights = self.metric_mae_weights(
            device=pred.device,
            dtype=pred.dtype,
        ).unsqueeze(0)
        weighted_abs_error = (safe_pred - safe_target).abs() * weights
        return weighted_abs_error.sum() / valid.to(pred.dtype).sum()

    def summary(self) -> dict[str, int | float | bool | str]:
        """Return compact JSON-safe train-target provenance."""
        observed = self.count.detach().cpu() > 0
        observed_std = self.std.detach().cpu()[observed]
        if observed_std.numel() > 0:
            std_mean = float(observed_std.mean().item())
            std_min = float(observed_std.min().item())
            std_max = float(observed_std.max().item())
        else:
            std_mean = std_min = std_max = 0.0
        return {
            "mode": self.loss_mode,
            "ready": bool(self.ready.item()),
            "target_dim": self.target_dim,
            "observed_targets": int(observed.sum().item()),
            "finite_train_labels": int(self.count.sum().item()),
            "train_std_mean": std_mean,
            "train_std_min": std_min,
            "train_std_max": std_max,
        }


__all__ = [
    "METRIC_MAE",
    "NORMALIZED_MSE",
    "RegressionTargetNormalizer",
    "resolve_regression_loss",
    "resolve_regression_target_normalization",
]
