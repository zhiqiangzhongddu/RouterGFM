"""Finetune-specific monitoring policy helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.monitoring import MonitorSpec, resolve_auto_monitor_spec, resolve_explicit_monitor_spec
from src.utils.parsing import resolve_task_type

if TYPE_CHECKING:
    from src.finetune.task_base import _FinetuneBase


def resolve_finetune_monitor_spec(
    cfg,
    *,
    task_level: str,
    label_dim: int,
    few_shot_without_validation: bool = False,
    task_cls: type[_FinetuneBase] | None = None,
    method_name: str = "",
) -> MonitorSpec:
    """Resolve finetuning monitor selection, including prompt-specific auto rules.

    Monitoring policy is determined in this priority order:

    1. Explicit ``finetune.monitor_metric`` config override.
    2. Task class ``resolve_default_monitor(cfg)`` hook (covers both the
       static ``default_monitor`` attribute and cfg-dependent toggles such
       as GPF's ``monitor_train_loss``).
    3. Auto-resolution from task_type / task_level / label_dim.

    *method_name* is accepted for backward compatibility but no longer
    drives any branching — per-method monitor policy belongs on the task
    class.
    """
    del method_name  # kept for backward-compatible kwargs; not used.
    spec = resolve_explicit_monitor_spec(
        raw_monitor_metric=getattr(cfg.finetune, "monitor_metric", "auto"),
        setting_name="finetune.monitor_metric",
    )
    if spec is not None:
        return spec

    # Priority 2: task-class-resolved monitor (may depend on cfg).
    if task_cls is not None:
        task_spec = task_cls.resolve_default_monitor(cfg)
        if task_spec is not None:
            return task_spec

    return resolve_auto_monitor_spec(
        task_type=resolve_task_type(getattr(cfg.finetune.dataset, "task_type", None)),
        task_level=str(task_level or getattr(cfg.finetune.dataset, "task_level", "")).lower(),
        label_dim=int(label_dim or 1),
        no_validation=few_shot_without_validation,
    )


__all__ = ["resolve_finetune_monitor_spec"]
