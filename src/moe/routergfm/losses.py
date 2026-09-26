"""Routing losses and regression normalization (App. B.2).

Prediction tensors live in the family's mixture space (``common.PRED_SPACE``):
class probabilities for node/graph classification and link prediction,
independent assay probabilities for multi-label tasks, and support
median/MAD-normalized outputs for regression.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .common import MULTILABEL, PRED_SPACE, REGRESSION

_PROB_EPS = 1e-6


def is_simplex_family(family: str) -> bool:
    """Single-label families whose predictions are class-probability vectors."""
    return PRED_SPACE[family] == "simplex"


class RegressionNormalizer:
    """Per-target support median and median absolute deviation with a scale floor."""

    def __init__(self, scale_floor: float = 1e-6):
        self.scale_floor = float(scale_floor)
        self.median: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None

    def fit(self, y_support: torch.Tensor) -> "RegressionNormalizer":
        y = torch.as_tensor(y_support).float()
        y = y.reshape(y.size(0), -1)
        median = torch.nanquantile(y, 0.5, dim=0)
        mad = torch.nanquantile((y - median).abs(), 0.5, dim=0)
        # A target with no finite support value keeps raw units.
        self.median = torch.nan_to_num(median, nan=0.0)
        self.scale = torch.nan_to_num(mad, nan=1.0).clamp_min(self.scale_floor)
        return self

    def _check(self) -> None:
        if self.median is None or self.scale is None:
            raise RuntimeError("RegressionNormalizer used before fit().")

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        self._check()
        y = torch.as_tensor(y).float()
        shaped = y.reshape(y.size(0), -1)
        return ((shaped - self.median) / self.scale).reshape(y.shape)

    def inverse(self, pred: torch.Tensor) -> torch.Tensor:
        self._check()
        pred = torch.as_tensor(pred).float()
        shaped = pred.reshape(pred.size(0), -1)
        return (shaped * self.scale + self.median).reshape(pred.shape)

    def state_dict(self) -> Dict[str, object]:
        self._check()
        return {"median": self.median.clone(), "scale": self.scale.clone(), "scale_floor": self.scale_floor}

    def load_state_dict(self, state: Dict[str, object]) -> "RegressionNormalizer":
        self.scale_floor = float(state.get("scale_floor", self.scale_floor))
        self.median = torch.as_tensor(state["median"]).float().clone()
        self.scale = torch.as_tensor(state["scale"]).float().clone()
        return self

    @classmethod
    def from_state_dict(cls, state: Optional[Dict[str, object]]) -> Optional["RegressionNormalizer"]:
        return None if state is None else cls().load_state_dict(state)


def routing_loss(pred: torch.Tensor, target: torch.Tensor, family: str, *, reg_kind: str = "abs") -> torch.Tensor:
    """Per-instance routing loss ``[N]`` (App. B.2).

    * single-label classification / link: ``0.5 * ||p - onehot(y)||^2`` (for the
      two-class link simplex this equals ``(p_1 - y)^2``);
    * multi-label: mean squared probability error over observed assays;
    * regression: mean over targets of ``|pred - y|`` (``abs``) or
      ``0.5 * (pred - y)^2`` (``sq``), both in normalized units (``target``
      must already be normalized).

    Rows without a valid label (negative class, or no observed assay/target)
    get NaN; callers drop NaN rows (they are not valid observations).
    """
    pred = torch.as_tensor(pred).float()
    if is_simplex_family(family):
        num_classes = pred.size(-1)
        y = torch.as_tensor(target, device=pred.device).reshape(-1).long()
        valid = (y >= 0) & (y < num_classes)
        onehot = F.one_hot(y.clamp(0, num_classes - 1), num_classes).float()
        loss = 0.5 * (pred - onehot).pow(2).sum(dim=-1)
        return torch.where(valid, loss, torch.full_like(loss, float("nan")))

    y = torch.as_tensor(target, device=pred.device).float().reshape(pred.shape)
    valid = torch.isfinite(y)
    diff = pred - torch.nan_to_num(y, nan=0.0)
    if family == MULTILABEL:
        elem = diff.pow(2)
    elif family == REGRESSION:
        if reg_kind == "abs":
            elem = diff.abs()
        elif reg_kind == "sq":
            elem = 0.5 * diff.pow(2)
        else:
            raise ValueError(f"Unknown regression routing loss {reg_kind!r} (expected abs|sq).")
    else:
        raise ValueError(f"Unknown task family {family!r}.")
    elem = elem.reshape(elem.size(0), -1)
    valid = valid.reshape(valid.size(0), -1)
    count = valid.sum(dim=-1)
    loss = (elem * valid).sum(dim=-1) / count.clamp_min(1)
    return torch.where(count > 0, loss, torch.full_like(loss, float("nan")))


def mixture_loss(pred_mix: torch.Tensor, target: torch.Tensor, family: str, *, reg_kind: str = "abs") -> torch.Tensor:
    """Routing loss of the mixed prediction (never the weighted individual losses)."""
    return routing_loss(pred_mix, target, family, reg_kind=reg_kind)


def to_metric_inputs(
    pred: torch.Tensor,
    family: str,
    normalizer: Optional[RegressionNormalizer] = None,
    target: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
    """Map family-space predictions to ``compute_supervised_metrics`` inputs.

    Returns ``(logits, labels, task_type)``: log-probabilities for single-label
    classification and link prediction (``log([1-p, p])``), assay logits for
    multi-label tasks, and raw-unit predictions (``normalizer.inverse``) for
    regression. ``labels`` is ``target`` in the metric's convention (raw units
    for regression, NaN for missing assays) or ``None`` when not given.
    """
    pred = torch.as_tensor(pred).float()
    if family == REGRESSION:
        if normalizer is None:
            raise ValueError("Regression metrics need the application's RegressionNormalizer.")
        logits, task_type = normalizer.inverse(pred), "regression"
    elif family == MULTILABEL:
        prob = pred.clamp(_PROB_EPS, 1.0 - _PROB_EPS)
        logits, task_type = torch.log(prob) - torch.log1p(-prob), "classification"
    elif is_simplex_family(family):
        logits, task_type = pred.clamp_min(_PROB_EPS).log(), "classification"
    else:
        raise ValueError(f"Unknown task family {family!r}.")
    labels = None
    if target is not None:
        target = torch.as_tensor(target)
        labels = target.reshape(-1).long() if is_simplex_family(family) else target.float()
    return logits, labels, task_type


__all__ = [
    "RegressionNormalizer",
    "is_simplex_family",
    "mixture_loss",
    "routing_loss",
    "to_metric_inputs",
]
