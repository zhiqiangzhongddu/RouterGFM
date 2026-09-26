"""Base class for training task modules."""

from __future__ import annotations

from typing import Iterable

from torch import nn


class TrainTask(nn.Module):
    """Base class for supervised training tasks.

    Provides the same step-based interface as ``PretrainTask`` but lives
    in the train package so the train workflow does not depend on
    pretrain internals.  New train methods should inherit this class
    and register via ``@register`` in ``src.train.registry``.
    """

    name = "base"

    #: Default monitor metric for this task.  When set (e.g.
    #: ``"train_loss"``), the monitoring resolver uses this value
    #: instead of auto-resolving from task_type/task_level.  ``None``
    #: means "use the auto policy".
    default_monitor: str | None = None

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        """Validate method-specific config before training starts.

        Called by ``build_train_task`` before instantiation and by the
        runner before the skip-if-exists check.  Must raise
        ``ValueError`` on bad input.  Default is a no-op.
        """

    @classmethod
    def run_tag(cls, cfg) -> str:
        """Return the primary method identity tag for the run name.

        This is the coarser tag (e.g. ``"plus1"``), whereas
        :meth:`variant_tag` captures secondary option deviations.
        Default returns ``""``; methods override when needed.
        """
        return ""

    @classmethod
    def variant_tag(cls, cfg) -> str:
        """Return a short filename-safe tag for method-variant options.

        A run using all-default options must return ``""``.
        """
        return ""

    def parameters_to_optimize(self) -> Iterable[nn.Parameter]:
        """Return parameters that should be optimized during training."""
        return self.parameters()

    def step(self, model: nn.Module, data, device):
        """Run one training step and return ``(loss, log_dict)``."""
        raise NotImplementedError

    def evaluate(self, model: nn.Module, data, device, mask_attr="val_mask", return_outputs=False):
        """Supervised evaluation forward pass."""
        raise NotImplementedError
