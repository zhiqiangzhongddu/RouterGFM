"""Pretrain-specific monitoring policy helpers."""

from __future__ import annotations

from src.utils.monitoring import MonitorSpec, make_monitor_spec, resolve_auto_monitor_spec, resolve_explicit_monitor_spec
from src.utils.parsing import resolve_task_type


def resolve_pretrain_monitor_spec(
    cfg,
    *,
    task_level: str,
    label_dim: int,
    uses_dataset_splits: bool,
    default_monitor: str | None = None,
) -> MonitorSpec:
    """Resolve pretraining monitor selection.

    Precedence: explicit ``pretrain.monitor_metric`` > the task class's
    ``default_monitor`` declaration > the splits-based auto policy. The
    ``uses_dataset_splits`` gate comes from the task's capability flag
    (see ``PretrainTask.uses_dataset_splits``) so any future supervised-
    style method automatically gets val-metric monitoring without needing
    to extend a string check here.
    """
    spec = resolve_explicit_monitor_spec(
        raw_monitor_metric=getattr(cfg.pretrain, "monitor_metric", "auto"),
        setting_name="pretrain.monitor_metric",
    )
    if spec is not None:
        return spec

    if default_monitor:
        spec = resolve_explicit_monitor_spec(
            raw_monitor_metric=default_monitor,
            setting_name="PretrainTask.default_monitor",
        )
        if spec is not None:
            return spec

    if not uses_dataset_splits:
        return make_monitor_spec("train_loss", "min")

    return resolve_auto_monitor_spec(
        task_type=resolve_task_type(getattr(cfg.pretrain.dataset, "task_type", None)),
        task_level=str(task_level or getattr(cfg.pretrain.dataset, "task_level", "")).lower(),
        label_dim=int(label_dim or 1),
    )


__all__ = ["resolve_pretrain_monitor_spec"]
