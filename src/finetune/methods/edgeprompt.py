"""EdgePrompt finetuning method."""

from __future__ import annotations

import torch
from torch import nn

from src.finetune.encoders.edgeprompt import (
    PromptAwareEncoder,
    resolve_edgeprompt_prompt_spec,
)
from src.finetune.prompts.edgeprompt import EdgePrompt, EdgePromptPlus
from src.finetune.registry import register
from src.finetune.task_heads import TaskAwareObjective, build_task_aware_classifier
from src.finetune.task_base import FinetuneTask
from src.utils.config_helpers import build_prompt_head_optimizer, cfg_default, optimizer_variant_tags


def _normalize_num_anchors(value):
    """Coerce edgeprompt.num_anchors: None or "auto" mean auto; else int.

    validate_cfg accepts the literal string "auto", so every consumer must
    too — a bare int() raised on exactly the value validation allowed.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    return int(value)
from src.utils.dataset_helpers import read_effective_task_level
from src.utils.parsing import to_bool
from src.utils.supervised_eval import evaluate_epoch_split
from src.utils.training import run_epoch_loop

@register("edgeprompt")
class FinetuneEdgePrompt(FinetuneTask):
    """EdgePrompt / EdgePromptplus finetuning."""

    requires_frozen_encoder = True
    supports_early_stopping = False  # official scripts run a fixed epoch budget
    encoder_builder = "prompt"
    frozen_encoder_mode = "train"  # dropout active, BN stats update (default pin_bn_eval=False)

    @classmethod
    def resolve_frozen_encoder_mode(cls, cfg) -> str:
        ep_cfg = getattr(getattr(cfg, "finetune", None), "edgeprompt", None)
        pin_bn = (
            bool(getattr(ep_cfg, "pin_bn_eval_when_frozen", False))
            if ep_cfg is not None
            else False
        )
        return "train_bn_eval" if pin_bn else "train"

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        cls.require_node_or_graph_batches(cfg, method_label="EdgePrompt")
        ep_cfg = getattr(getattr(cfg, "finetune", None), "edgeprompt", None)
        if ep_cfg is not None:
            raw_plus = getattr(ep_cfg, "plus", None)
            if raw_plus is not None and not isinstance(raw_plus, bool):
                try:
                    to_bool(raw_plus)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"finetune.edgeprompt.plus must be a bool, got {raw_plus!r}"
                    )
            num_anchors = getattr(ep_cfg, "num_anchors", None)
            if num_anchors is not None and str(num_anchors).lower() != "auto":
                try:
                    if int(num_anchors) < 1:
                        raise ValueError
                except (ValueError, TypeError):
                    raise ValueError(
                        f"[EdgePrompt] num_anchors must be 'auto' or a positive int, got {num_anchors!r}"
                    )
            node_hops = getattr(ep_cfg, "node_subgraph_hops", None)
            if node_hops is not None and int(node_hops) < 1:
                raise ValueError(f"[EdgePrompt] node_subgraph_hops must be >= 1, got {node_hops}")

    @classmethod
    def run_tag(cls, cfg) -> str:
        ep_cfg = getattr(getattr(cfg, "finetune", None), "edgeprompt", None)
        plus = int(to_bool(getattr(ep_cfg, "plus", False))) if ep_cfg is not None else 0
        return f"plus{plus}"

    @classmethod
    def variant_tag(cls, cfg) -> str:
        """Return variant tag for EdgePrompt checkpoint naming.

        **Compatibility exception:** Unlike the base-class contract which
        says all-default options must return ``""``, EdgePrompt always
        emits ``anchorsauto`` / ``loopsauto`` tags even for the default
        (None = auto).  These tags are part of the established checkpoint
        naming convention — suppressing them would break
        ``skip_if_exists`` for existing runs.  Future methods should NOT
        copy this pattern; it exists solely for backward compatibility.
        """
        edge_cfg = getattr(getattr(cfg, "finetune", None), "edgeprompt", None)
        if edge_cfg is None:
            return ""
        num_anchors = _normalize_num_anchors(getattr(edge_cfg, "num_anchors", None))
        add_self_loops = getattr(edge_cfg, "add_self_loops", None)
        tags = []
        if num_anchors is None:
            tags.append("anchorsauto")
        else:
            tags.append(f"anchors{num_anchors}")
        if add_self_loops is None:
            tags.append("loopsauto")
        else:
            tags.append(f"loops{int(to_bool(add_self_loops))}")
        # Tag non-default node subgraph settings so different sizing
        # configs produce distinct checkpoint names.
        use_official = to_bool(getattr(edge_cfg, "use_official_node_subgraphs", True))
        if use_official:
            hops = int(getattr(edge_cfg, "node_subgraph_hops", 2))
            min_sz = int(getattr(edge_cfg, "node_subgraph_min_size", 1))
            max_sz = int(getattr(edge_cfg, "node_subgraph_max_size", 100000))
            if hops != cfg_default("finetune.edgeprompt.node_subgraph_hops"):
                tags.append(f"h{hops}")
            if min_sz != cfg_default("finetune.edgeprompt.node_subgraph_min_size"):
                tags.append(f"mn{min_sz}")
            if max_sz != cfg_default("finetune.edgeprompt.node_subgraph_max_size"):
                tags.append(f"mx{max_sz}")
        tags.extend(optimizer_variant_tags(edge_cfg, "edgeprompt"))
        return "-".join(tags)

    @classmethod
    def adjust_dataset_cfg(cls, cfg, dataset_params: dict) -> dict:
        finetune_cfg = getattr(cfg, "finetune", None)
        edge_cfg = getattr(finetune_cfg, "edgeprompt", None)
        use_official = to_bool(getattr(edge_cfg, "use_official_node_subgraphs", True)) if edge_cfg is not None else True
        task_level = str(dataset_params.get("task_level", "") or "").lower()
        induced = bool(dataset_params.get("induced", False))
        if use_official and induced and task_level == "node":
            dataset_params["induced_min_size"] = int(getattr(edge_cfg, "node_subgraph_min_size", 1)) if edge_cfg is not None else 1
            dataset_params["induced_max_size"] = int(getattr(edge_cfg, "node_subgraph_max_size", 100000)) if edge_cfg is not None else 100000
            dataset_params["induced_max_hops"] = int(getattr(edge_cfg, "node_subgraph_hops", 2)) if edge_cfg is not None else 2
            print(
                "[Finetune][EdgePrompt] Using official-style node subgraph settings: "
                f"hops={dataset_params['induced_max_hops']}, "
                f"min_size={dataset_params['induced_min_size']}, "
                f"max_size={dataset_params['induced_max_size']}."
            )
        return dataset_params

    def __init__(self, cfg):
        super().__init__(cfg)
        ds_cfg = cfg.finetune.dataset
        self.task_level = read_effective_task_level(ds_cfg)
        self.task_level_raw = str(ds_cfg.task_level or self.task_level).lower()
        hidden_dim = int(getattr(cfg.model, "hidden_dim", 1) or 1)
        self.repr_dim = int(getattr(cfg.model, "out_dim", hidden_dim) or hidden_dim)
        self.objective = TaskAwareObjective(cfg, task_level=self.task_level, repr_dim=self.repr_dim)
        self.num_classes = int(self.objective.num_classes or 2)
        self.task_type = self.objective.task_type
        self.label_dim = self.objective.label_dim

        method_cfg = getattr(cfg.finetune, "edgeprompt", None)
        self.method_cfg = method_cfg

        use_plus = bool(getattr(method_cfg, "plus", True)) if method_cfg is not None else True

        raw_anchors = getattr(method_cfg, "num_anchors", None) if method_cfg is not None else None
        num_anchors = _normalize_num_anchors(raw_anchors)
        if num_anchors is None:
            # Follow official defaults: node-style downstream uses more anchors.
            num_anchors = 10 if self.task_level_raw == "node" else 5

        # Prompt dim list and self-loop policy come from the single pure
        # resolver shared with the encoder factory, so the two constructions
        # cannot disagree (cf. src/finetune/encoders/edgeprompt/spec.py).
        spec = resolve_edgeprompt_prompt_spec(cfg)
        dim_list = list(spec.dim_list)
        add_self_loops = bool(spec.add_self_loops)

        self.force_mean_pooling = (
            bool(getattr(method_cfg, "force_mean_pooling", True)) if method_cfg is not None else True
        )

        if use_plus:
            self.prompt = EdgePromptPlus(
                dim_list=dim_list,
                num_anchors=num_anchors,
                add_self_loops=add_self_loops,
                replace_self_loops=bool(spec.replace_self_loops),
            )
        else:
            self.prompt = EdgePrompt(dim_list=dim_list)
        self.prompt_type = "EdgePromptplus" if use_plus else "EdgePrompt"
        self.classifier = build_task_aware_classifier(
            input_dim=self.repr_dim,
            task_type=self.task_type,
            label_dim=self.label_dim,
            num_classes=self.objective.num_classes,
        )

    def validate_encoder(self, model: nn.Module) -> None:
        if not isinstance(model, PromptAwareEncoder):
            raise TypeError(
                f"EdgePrompt requires a PromptAwareEncoder but got {type(model).__name__}. "
                "Ensure finetune.method=edgeprompt uses a prompt-aware encoder from "
                "src/finetune/encoders/edgeprompt/ (this is automatic when using the "
                "standard FinetuneRunner)."
            )

    def parameters_to_optimize(self):
        return list(self.prompt.parameters()) + list(self.classifier.parameters())

    def build_optimizers(self, model: nn.Module):
        del model  # interface compatibility
        # EdgePrompt's reference config only exposes ``lr`` / ``weight_decay``
        # (no prompt_lr/head_lr split), so the shared helper produces two
        # groups at the same lr/wd — arithmetically equivalent to the prior
        # single-group Adam call but consistent with the other prompt methods.
        return build_prompt_head_optimizer(
            method_cfg=self.method_cfg,
            prompt_params=self.prompt.parameters(),
            head_params=self.classifier.parameters(),
            base_lr=float(self.cfg.finetune.lr),
            base_wd=float(self.cfg.finetune.weight_decay),
        )

    def _forward(
        self,
        model,
        data,
        device,
        mask_attr: str = "train_mask",
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, float]:
        data = data.to(device)
        node_repr, graph_repr = model(data, prompt=self.prompt, prompt_type=self.prompt_type)
        pooling_mode = "mean" if self.force_mean_pooling else None
        if self.force_mean_pooling:
            graph_repr = None
        return self.objective.forward_with_model_outputs(
            classifier=self.classifier,
            node_repr=node_repr,
            graph_repr=graph_repr,
            data=data,
            device=device,
            mask_attr=mask_attr,
            graph_pooling_mode=pooling_mode,
            return_outputs=return_logits,
        )

    def train_epoch(self, model, loader, device, optimizers=None):
        # Encoder mode is handled by the runner via _apply_frozen_encoder_mode.
        self.prompt.train()
        self.classifier.train()
        optimizer = optimizers.get("primary") if isinstance(optimizers, dict) else optimizers
        if optimizer is None:
            raise ValueError("Optimizer is required for EdgePrompt finetune.")

        metric_key = "train_mae" if self.task_type == "regression" else "train_acc"

        def forward_fn(data, device):
            loss, acc = self._forward(model, data, device, mask_attr="train_mask")
            return loss, {metric_key: float(acc)}

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

        def _eval_forward(data, device):
            loss, _primary, logits, labels = self._forward(
                model, data, device,
                mask_attr=mask_attr, return_logits=True,
            )
            return loss, logits, labels

        return evaluate_epoch_split(
            forward_fn=_eval_forward,
            loader=loader,
            device=device,
            prefix=prefix,
            task_type=self.task_type,
        )
