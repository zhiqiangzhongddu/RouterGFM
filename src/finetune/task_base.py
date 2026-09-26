from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from src.utils.dataset_helpers import resolve_effective_task_level

#: Allowed values for ``_FinetuneBase.encoder_builder``.  Keep in sync with the
#: dispatch in :meth:`FinetuneRunner._setup` (``_setup`` chooses
#: ``build_encoder_from_cfg`` vs ``build_prompt_encoder`` from this key).
_VALID_ENCODER_BUILDERS = frozenset({"default", "prompt"})

#: Allowed values for ``_FinetuneBase.frozen_encoder_mode``.  Keep in sync with
#: the branches in :meth:`FinetuneRunner._apply_frozen_encoder_mode`.
_VALID_FROZEN_ENCODER_MODES = frozenset({"eval", "train_bn_eval", "train"})


class _FinetuneBase(nn.Module):
    """Private mixin carrying class-level declarations shared by both
    step-based and epoch-based finetune task interfaces.

    Not intended for direct subclassing by methods — use
    :class:`FinetuneTask` (epoch-based) or :class:`StepFinetuneTask`
    (step-based) instead.
    """

    name = "base"

    #: Default monitor metric for this task.  When set (e.g.
    #: ``"train_loss"``), the monitoring resolver uses this value
    #: instead of auto-resolving from task_type/task_level.  ``None``
    #: means "use the auto policy".  This replaces method-name string
    #: checks in the monitoring module.
    default_monitor: str | None = None

    #: Whether this method requires the pretrained encoder to be frozen.
    requires_frozen_encoder: bool = False

    #: Whether this method supports early stopping.  When False, the runner
    #: disables patience regardless of the config value (fixed-epoch training).
    supports_early_stopping: bool = True

    #: Encoder builder key.  ``"default"`` uses the standard
    #: :func:`build_encoder_from_cfg`; ``"prompt"`` uses the prompt-aware
    #: :func:`build_prompt_encoder`.  Methods that need a
    #: non-standard encoder override this at class level.
    encoder_builder: str = "default"

    #: Encoder train/eval mode policy when the encoder is frozen.
    #: ``"eval"`` — full eval mode (no dropout, frozen BN stats).
    #: ``"train_bn_eval"`` — dropout active, but BN layers pinned to eval.
    #: ``"train"`` — full train mode (dropout + BN stats update).
    #: The runner applies this in :meth:`_apply_frozen_encoder_mode` via
    #: :meth:`resolve_frozen_encoder_mode`.
    #: Methods that need paper-specific behaviour override at class level,
    #: or (if the choice depends on cfg) override
    #: :meth:`resolve_frozen_encoder_mode`.
    frozen_encoder_mode: str = "eval"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        eb = getattr(cls, "encoder_builder", "default")
        if eb not in _VALID_ENCODER_BUILDERS:
            raise ValueError(
                f"{cls.__name__}.encoder_builder='{eb}' is invalid; "
                f"expected one of {sorted(_VALID_ENCODER_BUILDERS)}."
            )
        fem = getattr(cls, "frozen_encoder_mode", "eval")
        if fem not in _VALID_FROZEN_ENCODER_MODES:
            raise ValueError(
                f"{cls.__name__}.frozen_encoder_mode='{fem}' is invalid; "
                f"expected one of {sorted(_VALID_FROZEN_ENCODER_MODES)}."
            )

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Hooks called by the runner *before* skip-if-exists / setup.
    # Subclasses override to participate; defaults are no-ops.
    # ------------------------------------------------------------------

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        """Validate method-specific config before training starts.

        Called by the runner before the skip-if-exists check so that
        invalid configs are never silently skipped.  Must raise
        ``ValueError`` with a clear message on bad input.
        """

    @classmethod
    def require_node_or_graph_batches(cls, cfg, *, method_label: str) -> None:
        """Require batches whose effective task level is node or graph.

        Prompt methods such as GPF, GPPT, and GraphPrompt support raw node
        datasets directly, and induced node/edge datasets after promotion to
        graph-level subgraph batches. Any task level that does not resolve to
        node or graph is rejected.
        """
        ds_cfg = cfg.finetune.dataset
        raw = str(getattr(ds_cfg, "task_level", "") or "").lower()
        induced = getattr(ds_cfg, "induced", False)
        effective = resolve_effective_task_level(raw, induced)
        if effective not in {"node", "graph"}:
            raise ValueError(
                f"{method_label} requires node/graph batches "
                f"(set finetune.dataset.induced=True for {raw} datasets)."
            )

    @classmethod
    def require_graph_level_batches(cls, cfg, *, method_label: str) -> None:
        """Require graph-level batches after induced promotion."""
        ds_cfg = cfg.finetune.dataset
        raw = str(getattr(ds_cfg, "task_level", "") or "").lower()
        induced = getattr(ds_cfg, "induced", False)
        effective = resolve_effective_task_level(raw, induced)
        if effective != "graph":
            raise ValueError(
                f"{method_label} requires graph-level/subgraph batches "
                f"(set finetune.dataset.induced=True for {raw} datasets)."
            )

    @classmethod
    def variant_tag(cls, cfg) -> str:
        """Return a short filename-safe tag for method-variant options.

        Emitted only for options that differ from the config default.
        A run using all-default options must return ``""``.

        Methods may deviate from this convention with documented
        justification (e.g. EdgePrompt always emits anchor/loop tags
        for backward compatibility with established checkpoint names).
        """
        return ""

    @classmethod
    def run_tag(cls, cfg) -> str:
        """Return the primary method identity tag for the run name.

        This is the coarser tag (e.g. ``"plus1"``), whereas
        :meth:`variant_tag` captures secondary option deviations.
        """
        return ""

    @classmethod
    def resolve_frozen_encoder_mode(cls, cfg) -> str:
        """Return the effective frozen-encoder mode for this run.

        Default implementation returns the class attribute
        :attr:`frozen_encoder_mode`.  Methods whose mode depends on a cfg
        flag (e.g. GPF's ``freeze_encoder_bn_when_frozen``) override this
        classmethod instead of reassigning ``self.frozen_encoder_mode`` at
        instance level, so the return value is always validated against
        :data:`_VALID_FROZEN_ENCODER_MODES` and the class-attribute
        contract declared in :meth:`__init_subclass__` is preserved.
        """
        mode = cls.frozen_encoder_mode
        if mode not in _VALID_FROZEN_ENCODER_MODES:
            raise ValueError(
                f"{cls.__name__}.resolve_frozen_encoder_mode returned invalid "
                f"mode '{mode}'; expected one of {sorted(_VALID_FROZEN_ENCODER_MODES)}."
            )
        return mode

    @classmethod
    def resolve_encoder_builder(cls, cfg) -> str:
        """Return the effective encoder builder for this run.

        Default implementation returns the class attribute
        :attr:`encoder_builder`.  Methods whose builder depends on a cfg
        flag override this classmethod;
        the return value is re-validated here because
        :meth:`__init_subclass__` cannot see override return values.
        """
        del cfg
        kind = cls.encoder_builder
        if kind not in _VALID_ENCODER_BUILDERS:
            raise ValueError(
                f"{cls.__name__}.resolve_encoder_builder returned invalid "
                f"builder '{kind}'; expected one of {sorted(_VALID_ENCODER_BUILDERS)}."
            )
        return kind

    @classmethod
    def resolve_default_monitor(cls, cfg):
        """Return a :class:`MonitorSpec` (or ``None``) for cfg-dependent monitor policy.

        Default implementation uses the static :attr:`default_monitor` class
        attribute.  Methods whose monitor policy depends on method-specific
        config (e.g. GPF's ``monitor_train_loss`` toggle) override this
        classmethod so the monitoring module does not need method-name
        branches.
        """
        from src.utils.monitoring import make_monitor_spec

        if cls.default_monitor is not None:
            return make_monitor_spec(cls.default_monitor, "min")
        return None

    @classmethod
    def adjust_dataset_cfg(cls, cfg, dataset_params: dict) -> dict:
        """Optionally mutate *dataset_params* before dataset creation.

        Called by the runner during ``_setup`` to let the method alter
        induced mode, subgraph sizing, or other dataset-level knobs.
        The default implementation returns *dataset_params* unchanged.
        """
        return dataset_params

    def parameters_to_optimize(self) -> Iterable[nn.Parameter]:
        return self.parameters()

    def fit_regression_target_stats(self, train_loader) -> list[dict]:
        """Fit every shared target normalizer from the training loader only.

        Normalizers are registered inside ``TaskAwareObjective`` modules.
        Traversing registered submodules also covers GPF's nested supervised
        head without giving that method a separate normalization path. The
        returned compact summaries are persisted by the runner; complete
        per-target statistics remain registered in the task checkpoint state.
        """
        from src.finetune.regression import (
            RegressionTargetNormalizer,
            resolve_regression_target_normalization,
        )
        from src.utils.parsing import resolve_task_type

        ds_cfg = getattr(getattr(self.cfg, "finetune", None), "dataset", None)
        is_enabled_regression = (
            resolve_task_type(getattr(ds_cfg, "task_type", None)) == "regression"
            and resolve_regression_target_normalization(self.cfg)
        )
        normalizers = []
        seen = set()
        for module in self.modules():
            if isinstance(module, RegressionTargetNormalizer) and id(module) not in seen:
                normalizers.append(module)
                seen.add(id(module))
        if is_enabled_regression and not normalizers:
            raise RuntimeError(
                f"{type(self).__name__} does not expose the shared regression "
                "target normalizer."
            )
        summaries = []
        for normalizer in normalizers:
            if normalizer.fit(train_loader):
                summaries.append(normalizer.summary())
        return summaries

    def fit_multilabel_target_stats(self, train_loader) -> list[dict]:
        """Fit every shared macro-balanced BCE from training labels only.

        ``TaskAwareObjective`` registers the loss module, so this traversal
        covers direct users and GPF's nested supervised head identically.
        The returned compact summaries are persisted by the runner; complete
        per-target counts remain registered in the task checkpoint state.
        """
        from src.finetune.multilabel import (
            MACRO_BALANCED_BCE,
            MacroBalancedBCELoss,
            resolve_multilabel_loss,
        )
        from src.utils.parsing import resolve_task_type

        ds_cfg = getattr(getattr(self.cfg, "finetune", None), "dataset", None)
        is_enabled_multilabel = (
            resolve_task_type(getattr(ds_cfg, "task_type", None)) == "classification"
            and int(getattr(ds_cfg, "label_dim", 1) or 1) > 1
            and resolve_multilabel_loss(self.cfg) == MACRO_BALANCED_BCE
        )
        balancers = []
        seen = set()
        for module in self.modules():
            if isinstance(module, MacroBalancedBCELoss) and id(module) not in seen:
                balancers.append(module)
                seen.add(id(module))
        if is_enabled_multilabel and not balancers:
            raise RuntimeError(
                f"{type(self).__name__} does not expose the shared macro-balanced "
                "multilabel objective."
            )

        summaries = []
        for balancer in balancers:
            if balancer.fit(train_loader):
                summaries.append(balancer.summary())
        return summaries

    def validate_encoder(self, model: nn.Module) -> None:
        """Check that *model* is compatible with this task. Called after model construction."""


class FinetuneTask(_FinetuneBase):
    """Epoch-based finetune task interface.

    Prompt methods that own the training loop (GraphPrompt, GPF,
    EdgePrompt, GPPT, All-in-One) inherit this class.
    """

    def build_optimizers(self, model: nn.Module) -> dict[str, torch.optim.Optimizer] | None:
        return None

    def train_epoch(self, model: nn.Module, loader, device, optimizers=None):
        raise NotImplementedError

    def evaluate_split(self, model: nn.Module, loader, device, prefix: str, mask_attr: str):
        raise NotImplementedError

    def on_epoch_end(self, model: nn.Module, loader, device):
        return None


class StepFinetuneTask(_FinetuneBase):
    """Step-based finetune task interface.

    Methods that delegate the training loop to the runner and only
    implement a per-batch ``step()`` inherit this class (e.g. supervised
    finetuning).  This replaces the earlier pattern of inheriting
    ``PretrainTask`` from the pretrain package.
    """

    def step(self, model: nn.Module, data, device):
        """Run one training step and return ``(loss, log_dict)``."""
        raise NotImplementedError

    def evaluate(self, model: nn.Module, data, device, mask_attr="val_mask", return_outputs=False):
        """Supervised evaluation forward pass."""
        raise NotImplementedError
