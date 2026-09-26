"""Train-specific monitoring policy helpers."""

from __future__ import annotations

from src.utils.monitoring import MonitorSpec, resolve_auto_monitor_spec, resolve_explicit_monitor_spec
from src.utils.parsing import resolve_task_type


def resolve_train_monitor_spec(
    cfg,
    *,
    task_level: str,
    label_dim: int,
    few_shot_without_validation: bool = False,
    default_monitor: str | None = None,
) -> MonitorSpec:
    """Resolve training monitor selection, including train-specific auto policy.

    Precedence: explicit ``train.monitor_metric`` > the no-validation data
    constraint > the task class's ``default_monitor`` declaration > auto.
    """
    spec = resolve_explicit_monitor_spec(
        raw_monitor_metric=getattr(cfg.train, "monitor_metric", "auto"),
        setting_name="train.monitor_metric",
    )
    if spec is not None:
        return spec

    if default_monitor and not few_shot_without_validation:
        spec = resolve_explicit_monitor_spec(
            raw_monitor_metric=default_monitor,
            setting_name="TrainTask.default_monitor",
        )
        if spec is not None:
            return spec

    return resolve_auto_monitor_spec(
        task_type=resolve_task_type(getattr(cfg.train.dataset, "task_type", None)),
        task_level=str(task_level or getattr(cfg.train.dataset, "task_level", "")).lower(),
        label_dim=int(label_dim or 1),
        no_validation=few_shot_without_validation,
    )


__all__ = ["resolve_train_monitor_spec"]
