"""Shared monitoring primitives used by train, pretrain, and finetune."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional


EXPLICIT_MONITOR_MAP: Dict[str, tuple[str, str]] = {
    "balanced_loss": ("balanced_loss", "min"),
    "train_loss": ("train_loss", "min"),
    "val_loss": ("val_loss", "min"),
    "train_acc": ("train_acc", "max"),
    "val_acc": ("val_acc", "max"),
    "train_auc": ("train_auc", "max"),
    "val_auc": ("val_auc", "max"),
    "train_micro_f1": ("train_micro_f1", "max"),
    "val_micro_f1": ("val_micro_f1", "max"),
    "train_macro_f1": ("train_macro_f1", "max"),
    "val_macro_f1": ("val_macro_f1", "max"),
    "train_mae": ("train_mae", "min"),
    "val_mae": ("val_mae", "min"),
    "train_mse": ("train_mse", "min"),
    "val_mse": ("val_mse", "min"),
}


@dataclass(frozen=True)
class MonitorSpec:
    """Resolved monitoring configuration used by training loops."""

    name: Optional[str]
    mode: Optional[str]
    best_metric: float


def normalize_monitor_metric(raw_monitor_metric: object) -> str:
    """Normalize monitor config tokens to the canonical lowercase representation."""
    token = str(raw_monitor_metric or "auto").strip().lower()
    return token or "auto"


def _best_metric_for_mode(mode: Optional[str]) -> float:
    if mode == "max":
        return float("-inf")
    return float("inf")


def make_monitor_spec(name: Optional[str], mode: Optional[str]) -> MonitorSpec:
    """Create a monitor spec with the correct initial best metric."""
    if name is None or mode is None:
        return MonitorSpec(name=None, mode=None, best_metric=float("inf"))
    return MonitorSpec(name=name, mode=mode, best_metric=_best_metric_for_mode(mode))


def supported_monitor_metric_values() -> list[str]:
    """Return all accepted monitor_metric tokens in stable order."""
    return sorted(["auto", "disabled", "none", *EXPLICIT_MONITOR_MAP.keys()])


def resolve_explicit_monitor_spec(
    *,
    raw_monitor_metric: object,
    setting_name: str,
) -> Optional[MonitorSpec]:
    """Resolve disabled and explicit monitor settings; return None for `auto`."""
    monitor_metric = normalize_monitor_metric(raw_monitor_metric)

    if monitor_metric in {"none", "disabled"}:
        return make_monitor_spec(None, None)

    if monitor_metric.startswith("test_"):
        raise ValueError(
            f"Unsupported {setting_name}='{monitor_metric}'. Test-split metrics "
            "are held out for one final evaluation and cannot drive checkpoint "
            "selection or early stopping. Use a val_* or train_* metric instead."
        )

    if monitor_metric in EXPLICIT_MONITOR_MAP:
        name, mode = EXPLICIT_MONITOR_MAP[monitor_metric]
        return make_monitor_spec(name, mode)

    if monitor_metric != "auto":
        supported = ", ".join(supported_monitor_metric_values())
        raise ValueError(
            f"Unsupported {setting_name}='{monitor_metric}'. Supported values: {supported}"
        )

    return None


def resolve_auto_monitor_spec(
    *,
    task_type: str,
    task_level: str,
    label_dim: int,
    no_validation: bool = False,
) -> MonitorSpec:
    """Shared auto-monitor policy used by all three workflows.

    Each workflow's ``resolve_*_monitor_spec`` calls this after handling
    workflow-specific overrides (e.g. finetune prompt methods that force
    train_loss).  The common rules are:

    1. ``no_validation`` (few-shot w/o val, unsupervised pretrain) → train_loss
    2. edge task → val_auc
    3. multi-label classification → val_micro_f1
    4. regression → val_mae
    5. classification → val_acc
    """
    if no_validation:
        return make_monitor_spec("train_loss", "min")
    if task_level == "edge":
        return make_monitor_spec("val_auc", "max")
    if task_type == "classification" and label_dim > 1:
        return make_monitor_spec("val_micro_f1", "max")
    if task_type == "regression":
        return make_monitor_spec("val_mae", "min")
    if task_type == "classification":
        return make_monitor_spec("val_acc", "max")
    raise ValueError(
        f"Unsupported auto-monitoring context: task_type={task_type}, task_level={task_level}"
    )


def monitor_uses_train_split(monitor_name: Optional[str]) -> bool:
    """Return True when the selected monitor depends only on training logs."""
    if not monitor_name:
        return False
    token = str(monitor_name)
    return token == "balanced_loss" or token == "train_loss" or token.startswith("train_")


def resolve_monitor_value(
    monitor_name: Optional[str],
    *,
    train_loss: float,
    train_logs: Mapping[str, float],
    val_metrics: Mapping[str, float],
    test_metrics: Mapping[str, float],
) -> float:
    """Select the scalar used for checkpointing and early stopping."""
    if monitor_name is not None and str(monitor_name).startswith("test_"):
        raise ValueError(
            "Test-split metrics cannot be used for checkpoint selection or early stopping."
        )
    if monitor_name is None or monitor_name == "train_loss":
        return float(train_loss)

    if monitor_name.startswith("train_"):
        monitor_value = train_logs.get(monitor_name)
    elif monitor_name.startswith("val_"):
        monitor_value = val_metrics.get(monitor_name)
    else:
        monitor_value = None
        for metrics in (train_logs, val_metrics):
            if monitor_name in metrics:
                monitor_value = metrics[monitor_name]
                break

    if monitor_value is None:
        print(
            f"[Monitor] WARNING: '{monitor_name}' not found in epoch metrics; "
            f"skipping checkpoint update this epoch."
        )
        return float("nan")

    monitor_value = float(monitor_value)
    if not math.isfinite(monitor_value):
        print(
            f"[Monitor] WARNING: '{monitor_name}' is non-finite ({monitor_value}); "
            f"skipping checkpoint update this epoch."
        )
        return float("nan")
    return monitor_value


def merge_epoch_metrics(
    train_logs: Mapping[str, float],
    val_metrics: Mapping[str, float],
    test_metrics: Mapping[str, float],
) -> dict[str, float]:
    """Merge train/val/test logs into a single flat dict for history storage."""
    merged = {k: float(v) for k, v in train_logs.items()}
    merged.update({k: float(v) for k, v in val_metrics.items()})
    merged.update({k: float(v) for k, v in test_metrics.items()})
    return merged


def is_metric_improved(current: float, best: float, mode: str) -> bool:
    """Return True when *current* improves over *best* according to *mode*.

    Used by both ``PretrainRunner`` and ``FinetuneRunner`` for checkpoint
    selection and early stopping.
    """
    if current is None or not math.isfinite(float(current)):
        return False
    if mode == "max":
        return current > best
    return current < best


def should_print_metric(key: str) -> bool:
    """Return True if *key* should appear in per-epoch stdout.

    Filters out bookkeeping counters (``_count`` suffix) and metrics
    that are redundant or noisy in the log line (``batch_size``,
    ``test_loss``).
    """
    if str(key).endswith("_count"):
        return False
    return key not in {"batch_size", "test_loss"}


__all__ = [
    "EXPLICIT_MONITOR_MAP",
    "MonitorSpec",
    "is_metric_improved",
    "merge_epoch_metrics",
    "make_monitor_spec",
    "monitor_uses_train_split",
    "normalize_monitor_metric",
    "resolve_auto_monitor_spec",
    "resolve_explicit_monitor_spec",
    "resolve_monitor_value",
    "should_print_metric",
    "supported_monitor_metric_values",
]
