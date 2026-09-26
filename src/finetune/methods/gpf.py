"""GPF prompt finetuning method."""

from __future__ import annotations

from torch import nn

from src.finetune.prompts.gpf import GPFPlusPrompt, GPFPrompt
from src.finetune.registry import register
from src.finetune.methods.supervised import FinetuneSupervised
from src.finetune.task_base import FinetuneTask
from src.utils.config_helpers import build_prompt_head_optimizer, cfg_default, optimizer_variant_tags, tag_if_nondefault
from src.utils.parsing import resolve_task_type, to_bool
from src.utils.save_results import get_explicit_cfg_keys
from src.utils.supervised_eval import evaluate_epoch_split
from src.utils.training import run_epoch_loop


@register("gpf")
class FinetuneGPF(FinetuneTask):
    """GPF / GPF-plus finetuning with a prompted input and supervised task head.

    The official GPF paper targets graph-level molecular property prediction.
    IcG extends the method to node-level masked classification by applying
    the learnable prompt to ``data.x`` before the frozen encoder and training
    a linear/MLP head on the node or pooled graph representations. See
    ``ref_repos/GPF`` for the original graph-level pipeline.
    """

    requires_frozen_encoder = True
    supports_early_stopping = False  # official GPF scripts run fixed epochs
    frozen_encoder_mode = "train_bn_eval"  # dropout active, BN stats frozen

    @classmethod
    def resolve_default_monitor(cls, cfg):
        """GPF optionally monitors ``train_loss`` via a method-specific toggle.

        Keeps the per-method policy local to GPF so the shared monitoring
        helper does not need to special-case method names.
        """
        from src.utils.monitoring import make_monitor_spec

        gpf_cfg = getattr(getattr(cfg, "finetune", None), "gpf", None)
        if gpf_cfg is not None and bool(getattr(gpf_cfg, "monitor_train_loss", False)):
            return make_monitor_spec("train_loss", "min")
        return super().resolve_default_monitor(cfg)

    @classmethod
    def resolve_frozen_encoder_mode(cls, cfg) -> str:
        gpf_cfg = getattr(getattr(cfg, "finetune", None), "gpf", None)
        freeze_bn = (
            to_bool(getattr(gpf_cfg, "freeze_encoder_bn_when_frozen", True))
            if gpf_cfg is not None
            else True
        )
        return "train_bn_eval" if freeze_bn else "train"

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        cls.require_node_or_graph_batches(cfg, method_label="GPF")

        gpf_cfg = getattr(getattr(cfg, "finetune", None), "gpf", None)
        if gpf_cfg is None:
            return

        p_num = getattr(gpf_cfg, "p_num", None)
        if p_num is not None and int(p_num) < 1:
            raise ValueError(f"[GPF] p_num must be >= 1, got {p_num}")
        head_layers = getattr(gpf_cfg, "head_layers", None)
        if head_layers is not None and int(head_layers) < 1:
            raise ValueError(f"[GPF] head_layers must be >= 1, got {head_layers}")
        head_lr_scale = getattr(gpf_cfg, "head_lr_scale", None)
        if head_lr_scale is not None and float(head_lr_scale) <= 0:
            raise ValueError(f"[GPF] head_lr_scale must be > 0, got {head_lr_scale}")
        head_dropout = getattr(gpf_cfg, "head_dropout", None)
        if head_dropout is not None and not (0.0 <= float(head_dropout) < 1.0):
            raise ValueError(f"[GPF] head_dropout must be in [0, 1), got {head_dropout}")

    @classmethod
    def run_tag(cls, cfg) -> str:
        gpf_cfg = getattr(getattr(cfg, "finetune", None), "gpf", None)
        plus = int(to_bool(getattr(gpf_cfg, "plus", False))) if gpf_cfg is not None else 0
        if plus:
            p_num = int(getattr(gpf_cfg, "p_num", 0) or 0)
            return f"plus{plus}_p{p_num}"
        return f"plus{plus}"

    @classmethod
    def variant_tag(cls, cfg) -> str:
        gpf_cfg = getattr(getattr(cfg, "finetune", None), "gpf", None)
        if gpf_cfg is None:
            return ""

        tags = []
        head_layers = int(getattr(gpf_cfg, "head_layers", 1) or 1)
        if head_layers > 1:
            tags.append(f"head{head_layers}")
            head_hidden_dim = int(getattr(gpf_cfg, "head_hidden_dim", 0) or 0)
            if head_hidden_dim > 0:
                tags.append(f"hhd{head_hidden_dim}")
        head_lr_scale = float(getattr(gpf_cfg, "head_lr_scale", 1.0) or 1.0)
        t = tag_if_nondefault("hlrs", head_lr_scale, float(cfg_default("finetune.gpf.head_lr_scale")))
        if t:
            tags.append(t)
        monitor_train_loss = to_bool(getattr(gpf_cfg, "monitor_train_loss", False))
        if monitor_train_loss:
            tags.append("mtrain")
        # head_lr_scale is already tagged as "hlrs" above.
        tags.extend(optimizer_variant_tags(gpf_cfg, "gpf", exclude={"head_lr_scale"}))
        return "-".join(tags)

    @classmethod
    def adjust_dataset_cfg(cls, cfg, dataset_params: dict) -> dict:
        gpf_cfg = getattr(getattr(cfg, "finetune", None), "gpf", None)
        prefer_non_induced = bool(getattr(gpf_cfg, "prefer_non_induced_node", True)) if gpf_cfg is not None else True
        task_level = str(dataset_params.get("task_level", "") or "").lower()
        induced = bool(dataset_params.get("induced", False))
        explicit_keys = get_explicit_cfg_keys(cfg)
        induced_explicitly_set = "finetune.dataset.induced" in explicit_keys
        if prefer_non_induced and induced and task_level == "node" and not induced_explicitly_set:
            dataset_params["induced"] = False
            print(
                "[Finetune][gpf] Overriding finetune.dataset.induced=True to False for node-level tuning. "
                "Set finetune.gpf.prefer_non_induced_node=False to keep induced subgraphs."
            )
        return dataset_params

    def __init__(self, cfg):
        super().__init__(cfg)
        self.supervised_head = FinetuneSupervised(cfg)
        self.task_type = resolve_task_type(getattr(self.supervised_head, "task_type", None))
        self.task_level = str(getattr(self.supervised_head, "task_level", "graph") or "graph").lower()

        method_cfg = getattr(cfg.finetune, "gpf", None)
        plus_flag = to_bool(getattr(method_cfg, "plus", False)) if method_cfg else False
        p_num = int(getattr(method_cfg, "p_num", 5)) if method_cfg else 5

        self.prompt_in_dim = int(getattr(cfg.model, "in_dim", 0) or 0)
        if self.prompt_in_dim <= 0:
            raise ValueError("GPF requires model.in_dim > 0.")

        if plus_flag:
            self.prompt = GPFPlusPrompt(in_channels=self.prompt_in_dim, p_num=p_num)
            self.prompt_variant = "gpf_plus"
        else:
            self.prompt = GPFPrompt(in_channels=self.prompt_in_dim)
            self.prompt_variant = "gpf"

        # Official GPF scripts tune the prediction head depth (num_layers).
        self.head_layers = (
            int(getattr(method_cfg, "head_layers", 1))
            if method_cfg is not None
            else 1
        )
        self.head_layers = max(1, self.head_layers)
        self.head_hidden_dim = (
            int(getattr(method_cfg, "head_hidden_dim", 0))
            if method_cfg is not None
            else 0
        )
        self.head_dropout = (
            float(getattr(method_cfg, "head_dropout", 0.0))
            if method_cfg is not None
            else 0.0
        )
        self._maybe_upgrade_prediction_head()

    def _maybe_upgrade_prediction_head(self) -> None:
        if self.head_layers <= 1:
            return
        classifier = self.supervised_head.classifier
        if not isinstance(classifier, nn.Linear):
            return

        in_dim = int(classifier.in_features)
        out_dim = int(classifier.out_features)
        hidden_dim = int(self.head_hidden_dim) if int(self.head_hidden_dim) > 0 else in_dim

        layers = []
        for idx in range(self.head_layers - 1):
            input_dim = in_dim if idx == 0 else hidden_dim
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if self.head_dropout > 0:
                layers.append(nn.Dropout(self.head_dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.supervised_head.classifier = nn.Sequential(*layers)

    def parameters_to_optimize(self):
        return list(self.prompt.parameters()) + list(self.supervised_head.classifier.parameters())

    def build_optimizers(self, model):
        method_cfg = getattr(self.cfg.finetune, "gpf", None)
        return build_prompt_head_optimizer(
            method_cfg=method_cfg,
            prompt_params=self.prompt.parameters(),
            head_params=self.supervised_head.classifier.parameters(),
            base_lr=float(self.cfg.finetune.lr),
            base_wd=float(self.cfg.finetune.weight_decay),
        )

    def _apply_prompt(self, data, device):
        data = data.to(device)
        x = getattr(data, "x", None)
        if x is None:
            raise ValueError("GPF requires node features in `data.x`.")
        if int(x.size(-1)) != self.prompt_in_dim:
            raise ValueError(
                f"GPF prompt dim mismatch: expected {self.prompt_in_dim}, got {int(x.size(-1))}. "
                "Ensure finetune dataset features match pretrained model input dimension."
            )
        prompted = data.clone()
        prompted.x = self.prompt.add(prompted.x)
        return prompted

    def train_epoch(self, model, loader, device, optimizers=None):
        optimizer = optimizers.get("primary") if isinstance(optimizers, dict) else optimizers
        if optimizer is None:
            raise ValueError("GPF requires an optimizer.")

        # Encoder mode is handled by the runner via _apply_frozen_encoder_mode.
        self.train()
        metric_key = "train_mae" if self.task_type == "regression" else "train_acc"

        def forward_fn(data, device):
            prompted = self._apply_prompt(data, device)
            loss, primary = self.supervised_head.evaluate(
                model=model, data=prompted, device=device,
                mask_attr="train_mask",
            )
            return loss, {metric_key: float(primary)}

        return run_epoch_loop(
            forward_fn=forward_fn,
            loader=loader,
            optimizer=optimizer,
            device=device,
            grad_clip=float(getattr(getattr(self.cfg, "finetune", None), "grad_clip", 0.0) or 0.0),
        )

    def evaluate_split(self, model, loader, device, prefix: str, mask_attr: str) -> dict[str, float]:
        model.eval()
        self.eval()

        def _forward(data, device):
            prompted = self._apply_prompt(data, device)
            loss, _primary, logits, labels = self.supervised_head.evaluate(
                model=model, data=prompted, device=device,
                mask_attr=mask_attr, return_outputs=True,
            )
            return loss, logits, labels

        return evaluate_epoch_split(
            forward_fn=_forward,
            loader=loader,
            device=device,
            prefix=prefix,
            task_type=self.task_type,
        )
