"""GraphPrompt finetuning method."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.finetune.encoders.graphprompt_plus import (
    GraphPromptPlusAdapter,
    build_graphprompt_plus_adapter,
    resolve_graphprompt_plus_spec,
    supported_graphprompt_plus_backbones,
)
from src.finetune.prompts import (
    GraphPrompt,
    GraphPromptPlusStageWise,
    compute_class_centers,
)
from src.finetune.registry import register
from src.finetune.task_heads import TaskAwareObjective, align_last_dim, build_task_aware_classifier, prepare_single_label_labels
from src.finetune.task_base import FinetuneTask
from src.model.encoder import supports_layer_cache
from src.utils.config_helpers import (
    build_prompt_head_optimizer,
    cfg_default,
    optimizer_variant_tags,
    tag_if_nondefault,
    validate_choice,
)
from src.utils.dataset_helpers import normalize_node_mask, read_effective_task_level
from src.utils.pool import get_batch_vector, pool_nodes, pool_target_nodes
from src.utils.supervised_eval import concat_and_compute_metrics, evaluate_epoch_split
from src.utils.parsing import resolve_task_type, to_bool

from .graphprompt_utils import fill_missing_centers, similarity_logits


@register("graphprompt")
class FinetuneGraphPrompt(FinetuneTask):
    """Prototype-based GraphPrompt finetuning with a feature-weight prompt."""

    requires_frozen_encoder = True
    supports_early_stopping = False  # official GraphPrompt scripts run fixed epochs

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        cls.require_node_or_graph_batches(cfg, method_label="GraphPrompt")

        gp_cfg = getattr(getattr(cfg, "finetune", None), "graphprompt", None)
        if gp_cfg is None:
            return
        score_mode = str(getattr(gp_cfg, "score_mode", "auto") or "auto").lower()
        if score_mode == "auto":
            score_mode = "neg_distance"
        validate_choice(
            "graphprompt.score_mode", score_mode,
            {"neg_distance", "distance", "cosine"},
        )
        train_center = str(getattr(gp_cfg, "train_center_mode", "auto") or "auto").lower()
        if train_center == "auto":
            train_center = "ema"
        validate_choice(
            "graphprompt.train_center_mode", train_center,
            {"batch", "train", "ema"},
        )
        eval_center = str(getattr(gp_cfg, "eval_center_mode", "auto") or "auto").lower()
        if eval_center == "auto":
            eval_center = "train"
        if eval_center == "batch":
            raise ValueError(
                "graphprompt.eval_center_mode='batch' is not supported because it "
                "builds prototypes from validation/test labels before scoring the "
                "same examples. Use 'train' (or 'auto', which maps to 'train')."
            )
        validate_choice("graphprompt.eval_center_mode", eval_center, {"train"})
        repr_source = str(getattr(gp_cfg, "repr_source", "last") or "last").lower()
        validate_choice("graphprompt.repr_source", repr_source, {"last", "layer_concat"})
        loss_red = str(getattr(gp_cfg, "loss_reduction", "auto") or "auto").lower()
        if loss_red == "auto":
            loss_red = "mean"
        validate_choice(
            "graphprompt.loss_reduction", loss_red,
            {"mean", "sum"},
        )
        graph_pool = str(getattr(gp_cfg, "graph_pooling", "auto") or "auto").lower()
        if graph_pool == "auto":
            graph_pool = "sum"
        validate_choice(
            "graphprompt.graph_pooling", graph_pool,
            {"encoder", "sum", "add", "mean", "max", "target"},
        )
        postprocess = str(getattr(gp_cfg, "embedding_postprocess", "auto") or "auto").lower()
        if postprocess == "auto":
            postprocess = "none"
        validate_choice(
            "graphprompt.embedding_postprocess", postprocess,
            {"none", "official_node"},
        )
        prompt_dropout = float(getattr(gp_cfg, "prompt_dropout", 0.0) or 0.0)
        if prompt_dropout < 0.0 or prompt_dropout >= 1.0:
            raise ValueError(
                f"graphprompt.prompt_dropout must be in [0.0, 1.0); got {prompt_dropout}."
            )
        # Resolve GraphPrompt+ backbone support here so unsupported
        # combinations error before the runner's skip-if-exists check.
        # If we deferred this to ``__init__`` (after skip), a stale
        # checkpoint for an invalid (plus=True, h2gcn) combination would
        # silently short-circuit the run.
        use_plus = bool(to_bool(getattr(gp_cfg, "plus", False)))
        if use_plus:
            plus_spec = resolve_graphprompt_plus_spec(cfg)
            if repr_source == "layer_concat" and not plus_spec.supports_layer_concat:
                raise ValueError(
                    "graphprompt.repr_source='layer_concat' is not supported by "
                    f"GraphPrompt+ for model '{plus_spec.model_name}'."
                )
        elif repr_source == "layer_concat":
            model_name = str(getattr(getattr(cfg, "model", None), "name", "") or "").lower()
            if not supports_layer_cache(model_name):
                raise ValueError(
                    "graphprompt.repr_source='layer_concat' requires an encoder "
                    f"with a per-layer representation cache; model '{model_name}' "
                    "does not provide one. Use repr_source='last'."
                )

    @classmethod
    def run_tag(cls, cfg) -> str:
        gp_cfg = getattr(getattr(cfg, "finetune", None), "graphprompt", None)
        plus = int(to_bool(getattr(gp_cfg, "plus", False))) if gp_cfg is not None else 0
        score_mode = str(getattr(gp_cfg, "score_mode", "auto") or "auto").lower()
        return f"plus{plus}_{score_mode}"

    @classmethod
    def variant_tag(cls, cfg) -> str:
        gp_cfg = getattr(getattr(cfg, "finetune", None), "graphprompt", None)
        if gp_cfg is None:
            return ""
        tags = []
        default_repr_source = str(cfg_default("finetune.graphprompt.repr_source")).lower()
        repr_source = str(getattr(gp_cfg, "repr_source", default_repr_source) or default_repr_source).lower()
        if repr_source != default_repr_source:
            tags.append(f"repr_{repr_source}")
        default_train_center = str(cfg_default("finetune.graphprompt.train_center_mode")).lower()
        train_center = str(getattr(gp_cfg, "train_center_mode", default_train_center) or default_train_center).lower()
        if train_center != default_train_center:
            tags.append(f"tc_{train_center}")
        default_postprocess = str(cfg_default("finetune.graphprompt.embedding_postprocess")).lower()
        postprocess = str(getattr(gp_cfg, "embedding_postprocess", default_postprocess) or default_postprocess).lower()
        if postprocess != default_postprocess:
            tags.append(f"post_{postprocess}")
        prompt_lr = float(getattr(gp_cfg, "prompt_lr", 1e-3) or 1e-3)
        t = tag_if_nondefault("plr", prompt_lr, float(cfg_default("finetune.graphprompt.prompt_lr")))
        if t:
            tags.append(t)
        default_loss_red = str(cfg_default("finetune.graphprompt.loss_reduction")).lower()
        loss_red = str(getattr(gp_cfg, "loss_reduction", default_loss_red) or default_loss_red).lower()
        if loss_red != default_loss_red:
            tags.append(f"red_{loss_red}")
        t = tag_if_nondefault(
            "tau",
            float(getattr(gp_cfg, "tau", cfg_default("finetune.graphprompt.tau")) or cfg_default("finetune.graphprompt.tau")),
            float(cfg_default("finetune.graphprompt.tau")),
        )
        if t:
            tags.append(t)
        default_pooling = str(cfg_default("finetune.graphprompt.graph_pooling")).lower()
        pooling = str(getattr(gp_cfg, "graph_pooling", default_pooling) or default_pooling).lower()
        if pooling != default_pooling:
            tags.append(f"pool_{pooling}")
        default_eval_center = str(cfg_default("finetune.graphprompt.eval_center_mode")).lower()
        eval_center = str(getattr(gp_cfg, "eval_center_mode", default_eval_center) or default_eval_center).lower()
        if eval_center != default_eval_center:
            tags.append(f"ec_{eval_center}")
        # prompt_lr is already tagged as "plr" above.
        tags.extend(optimizer_variant_tags(gp_cfg, "graphprompt", exclude={"prompt_lr"}))
        return "-".join(tags)

    def __init__(self, cfg):
        super().__init__(cfg)
        ds_cfg = cfg.finetune.dataset
        self.task_level = read_effective_task_level(ds_cfg)
        self.task_level_raw = str(ds_cfg.task_level or self.task_level).lower()
        self.is_induced = bool(getattr(ds_cfg, "induced", False))
        self.task_type = resolve_task_type(getattr(ds_cfg, "task_type", None))
        self.label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        self.num_classes = max(2, int(getattr(ds_cfg, "num_classes", 2) or 2))
        if self.task_level_raw == "node" and self.is_induced:
            print(
                "[Finetune][GraphPrompt] Running node GraphPrompt with induced subgraphs. "
                "Official GraphPrompt uses non-induced node training; consider finetune.dataset.induced=False."
            )

        hidden_dim = int(getattr(cfg.model, "hidden_dim", 1) or 1)
        out_dim = int(getattr(cfg.model, "out_dim", hidden_dim) or hidden_dim)
        num_layers = int(getattr(cfg.model, "num_layers", 1) or 1)
        method_cfg = cfg.finetune.graphprompt
        raw_repr_source = str(method_cfg.repr_source).lower()
        if raw_repr_source not in {"last", "layer_concat"}:
            raise ValueError("graphprompt.repr_source must be 'last' or 'layer_concat'.")
        self.repr_source = raw_repr_source
        if self.repr_source == "layer_concat":
            # Official GraphPrompt backbones concatenate all layer outputs.
            self.repr_dim = hidden_dim * max(0, num_layers - 1) + out_dim
        else:
            self.repr_dim = out_dim
        self.objective = TaskAwareObjective(cfg, task_level=self.task_level, repr_dim=self.repr_dim)
        self.single_label_classification = self.objective.is_single_label_classification

        self.use_plus = bool(method_cfg.plus)
        self.p_num = int(method_cfg.p_num)
        self.prompt_init = str(method_cfg.init)
        self.prompt_init_std = float(method_cfg.init_std)
        self.tau = float(method_cfg.tau)
        raw_score_mode = str(method_cfg.score_mode).lower()
        self.score_mode = "neg_distance" if raw_score_mode == "auto" else raw_score_mode
        raw_reduction = str(method_cfg.loss_reduction).lower()
        self.loss_reduction = "mean" if raw_reduction == "auto" else raw_reduction
        raw_train_center_mode = str(method_cfg.train_center_mode).lower()
        self.train_center_mode = "ema" if raw_train_center_mode == "auto" else raw_train_center_mode
        raw_eval_center_mode = str(method_cfg.eval_center_mode).lower()
        self.eval_center_mode = "train" if raw_eval_center_mode == "auto" else raw_eval_center_mode
        raw_graph_pooling_mode = str(method_cfg.graph_pooling).lower()
        self.graph_pooling_mode = "sum" if raw_graph_pooling_mode == "auto" else raw_graph_pooling_mode
        self.prompt_dropout = float(method_cfg.prompt_dropout)
        self.center_momentum = float(method_cfg.center_momentum)
        if self.score_mode not in {"neg_distance", "distance", "cosine"}:
            raise ValueError("graphprompt.score_mode must be one of: neg_distance, distance, cosine (auto maps to neg_distance).")
        if self.loss_reduction not in {"mean", "sum"}:
            raise ValueError("graphprompt.loss_reduction must be one of: mean, sum (auto maps to mean).")
        if self.train_center_mode not in {"batch", "train", "ema"}:
            raise ValueError("graphprompt.train_center_mode must be one of: batch, train, ema (auto maps to ema).")
        if self.eval_center_mode != "train":
            raise ValueError(
                "graphprompt.eval_center_mode must be 'train' (auto maps to train); "
                "batch centers leak evaluation labels."
            )
        if self.graph_pooling_mode not in {"encoder", "sum", "add", "mean", "max", "target"}:
            raise ValueError(
                "graphprompt.graph_pooling must be one of: encoder, sum, add, mean, max, target (auto maps to sum)."
            )
        if self.prompt_dropout < 0.0 or self.prompt_dropout >= 1.0:
            raise ValueError("graphprompt.prompt_dropout must be in [0.0, 1.0).")
        # Node-level embedding postprocessing (official GraphPrompt node downstream).
        raw_postprocess = str(method_cfg.embedding_postprocess).lower()
        self.embedding_postprocess = "none" if raw_postprocess == "auto" else raw_postprocess
        if self.embedding_postprocess not in {"none", "official_node"}:
            raise ValueError("graphprompt.embedding_postprocess must be 'none' or 'official_node'.")
        self.nhop_neighbour = int(method_cfg.nhop_neighbour)
        self.self_loop_weight = float(method_cfg.self_loop_weight)

        if self.use_plus:
            # Resolve the per-backbone adapter early so the configuration
            # error for unsupported backbones surfaces during task construction
            # (alongside the rest of the cfg validation), not lazily at the
            # first training batch.
            self._plus_spec = resolve_graphprompt_plus_spec(cfg)
            self._plus_adapter: type[GraphPromptPlusAdapter] = build_graphprompt_plus_adapter(cfg)
            if self.repr_source == "layer_concat" and not self._plus_adapter.supports_layer_concat():
                raise ValueError(
                    f"[GraphPrompt+] repr_source='layer_concat' is not supported "
                    f"by the adapter for model '{self._plus_spec.model_name}'."
                )
            # The adapter is authoritative for which stages exist and at what
            # dim — H2GCN's fixed 2-hop layout, for example, exposes only
            # stages {0, 1, 3} and would silently break a hardcoded prompt
            # module.
            adapter_specs = self._plus_adapter.iter_stage_specs(
                num_layers=num_layers,
                in_dim=int(cfg.model.in_dim or self.repr_dim),
                hidden_dim=hidden_dim,
                out_dim=int(getattr(cfg.model, "out_dim", hidden_dim) or hidden_dim),
                repr_dim=self.repr_dim,
            )
            self.prompt = GraphPromptPlusStageWise(
                in_channels=int(cfg.model.in_dim or self.repr_dim),
                hidden_channels=hidden_dim,
                out_channels=self.repr_dim,
                num_layers=num_layers,
                p_num=self.p_num,
                init=self.prompt_init,
                init_std=self.prompt_init_std,
                stage_specs=adapter_specs,
            )
            active = self.prompt.active_stage_ids
            if self.p_num > len(active):
                print(
                    f"[Finetune][GraphPrompt+] p_num={self.p_num} but only "
                    f"{len(active)} stages available with num_layers={num_layers}. "
                    f"Active stages: {active}. Consider num_layers>=3 for 4-stage GraphPrompt+."
                )
        else:
            self._plus_spec = None
            self._plus_adapter = None
            self.prompt = GraphPrompt(
                in_channels=self.repr_dim,
                init=self.prompt_init,
                init_std=self.prompt_init_std,
            )
        if self.single_label_classification:
            self.classifier = None
        else:
            self.classifier = build_task_aware_classifier(
                input_dim=self.repr_dim,
                task_type=self.task_type,
                label_dim=self.label_dim,
                num_classes=self.objective.num_classes,
            )
        # Prediction for single-label GraphPrompt depends on the learned
        # train-split prototypes. Register fixed-shape buffers so task
        # state_dict checkpoints reproduce predictions after a round-trip.
        # Separate validity buffers preserve the pre-training "not populated"
        # state without using ``None`` (None buffers are omitted by PyTorch).
        center_shape = (self.num_classes, self.repr_dim)
        self.register_buffer("latest_centers", torch.zeros(center_shape))
        self.register_buffer("prototype_bank", torch.zeros(center_shape))
        self.register_buffer("_latest_centers_valid", torch.tensor(False))
        self.register_buffer("_prototype_bank_valid", torch.tensor(False))
        self._generalized_notice_printed = False

    def _has_latest_centers(self) -> bool:
        return bool(self._latest_centers_valid.item())

    def _has_prototype_bank(self) -> bool:
        return bool(self._prototype_bank_valid.item())

    def _latest_centers_or_none(self) -> torch.Tensor | None:
        return self.latest_centers if self._has_latest_centers() else None

    def _prototype_bank_or_none(self) -> torch.Tensor | None:
        return self.prototype_bank if self._has_prototype_bank() else None

    def _store_latest_centers(self, centers: torch.Tensor | None) -> None:
        if centers is None:
            self._latest_centers_valid.fill_(False)
            return
        self.latest_centers.copy_(centers.detach().to(self.latest_centers))
        self._latest_centers_valid.fill_(True)

    def _store_prototype_bank(self, centers: torch.Tensor | None) -> None:
        if centers is None:
            self._prototype_bank_valid.fill_(False)
            return
        self.prototype_bank.copy_(centers.detach().to(self.prototype_bank))
        self._prototype_bank_valid.fill_(True)

    def parameters_to_optimize(self):
        params = list(self.prompt.parameters())
        if self.classifier is not None:
            params.extend(self.classifier.parameters())
        return params

    def validate_encoder(self, model) -> None:
        """Validate encoder compatibility for GraphPrompt+.

        Delegates the runtime check to the per-backbone adapter
        (``GraphPromptPlusAdapter.supports_model``).  The adapter knows
        which encoder attributes its driver needs.
        """
        if not self.use_plus:
            return
        if not self._plus_adapter.supports_model(model):
            supported = supported_graphprompt_plus_backbones()
            raise TypeError(
                f"[GraphPrompt+] The encoder ({type(model).__name__}) is "
                f"incompatible with adapter "
                f"{self._plus_adapter.__name__}. "
                f"Backbones with GraphPrompt+ support: {list(supported)}. "
                f"Use graphprompt.plus=False for unsupported encoders."
            )

    def build_optimizers(self, model: nn.Module):
        method_cfg = self.cfg.finetune.graphprompt
        # Official GraphPrompt trains prompt + classifier at the same LR/WD
        # (no separate ``head_lr``), so head_lr_scale=1.0 with the shared
        # helper reproduces that policy.  AdamW mirrors the reference code.
        head_params = list(self.classifier.parameters()) if self.classifier is not None else None
        return build_prompt_head_optimizer(
            method_cfg=method_cfg,
            prompt_params=self.prompt.parameters(),
            head_params=head_params,
            base_lr=float(method_cfg.prompt_lr),
            base_wd=float(method_cfg.prompt_weight_decay),
            head_lr_scale=1.0,
            optimizer_cls=torch.optim.AdamW,
        )

    def _apply_node_postprocess(self, node_repr: torch.Tensor, data) -> torch.Tensor:
        """Official GraphPrompt node postprocessing: sigmoid + adjacency propagation.

        Mirrors ref_repos/GraphPrompt/nodedownstream/run.py:pre_train():
        ``pred = sigmoid(pred); pred = (adj + self_weight*I) @ pred`` repeated
        nhop_neighbour times.  Uses PyG sparse ops.
        """
        if self.embedding_postprocess != "official_node":
            return node_repr
        if self.task_level_raw != "node" or self.is_induced:
            return node_repr
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            return node_repr
        num_nodes = node_repr.size(0)
        device = node_repr.device
        # sigmoid activation
        x = torch.sigmoid(node_repr)
        # build sparse (adj + self_loop_weight * I)
        row, col = edge_index[0], edge_index[1]
        self_idx = torch.arange(num_nodes, device=device)
        all_row = torch.cat([row, self_idx])
        all_col = torch.cat([col, self_idx])
        edge_vals = torch.ones(row.size(0), device=device, dtype=x.dtype)
        self_vals = torch.full((num_nodes,), self.self_loop_weight, device=device, dtype=x.dtype)
        values = torch.cat([edge_vals, self_vals])
        adj = torch.sparse_coo_tensor(
            torch.stack([all_row, all_col]),
            values,
            size=(num_nodes, num_nodes),
        )
        # nhop propagation
        for _ in range(self.nhop_neighbour):
            x = torch.sparse.mm(adj, x)
        return x

    def _supports_stagewise_plus_backbone(self, model: nn.Module) -> bool:
        if not (self.use_plus and isinstance(self.prompt, GraphPromptPlusStageWise)):
            return False
        if self._plus_adapter is None:
            return False
        return self._plus_adapter.supports_model(model)

    def _forward_with_stage_prompt(self, model: nn.Module, data, stage_id: int):
        """Run one stage-prompted forward pass via the per-backbone adapter.

        See ``GraphPromptPlusAdapter.forward_with_stage_prompt`` for the
        return contract.  Adapters return raw stage representations; the
        task class is responsible for masking, pooling fallback, label
        preparation, and cross-stage mixing.
        """
        return self._plus_adapter.forward_with_stage_prompt(
            model=model,
            data=data,
            stage_id=stage_id,
            prompt=self.prompt,
            repr_source=self.repr_source,
        )

    def _extract_stagewise_plus_embeddings(
        self,
        model: nn.Module,
        data,
        device: torch.device,
        mask_attr: str,
        apply_prompt_dropout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = data.to(device)
        batch = get_batch_vector(data)

        if self.task_level == "node":
            mask = normalize_node_mask(data, mask_attr, device)
            # Keep the same task-aware label handling as the non-plus path.
            labels = self._prepare_task_labels(data.y[mask]).to(device)
        else:
            labels = self._prepare_task_labels(data.y).to(device)

        mixed_embeddings = None
        for stage_id, coeff in self.prompt.iter_stage_coefficients():
            node_repr, stage_graph_repr = self._forward_with_stage_prompt(model=model, data=data, stage_id=stage_id)
            node_repr = align_last_dim(node_repr, self.repr_dim)

            if self.task_level == "node":
                stage_embeddings = node_repr[mask]
            else:
                if self.graph_pooling_mode == "encoder":
                    graph_repr = stage_graph_repr
                    if graph_repr is None:
                        graph_repr = pool_nodes(
                            x=node_repr,
                            batch=batch,
                            mode=self.cfg.model.graph_pooling,
                        )
                elif self.graph_pooling_mode == "target":
                    graph_repr = pool_target_nodes(node_repr, data)
                else:
                    graph_repr = pool_nodes(
                        x=node_repr,
                        batch=batch,
                        mode=self.graph_pooling_mode,
                    )
                stage_embeddings = align_last_dim(graph_repr, self.repr_dim)

            weighted = coeff * stage_embeddings
            mixed_embeddings = weighted if mixed_embeddings is None else (mixed_embeddings + weighted)

        if mixed_embeddings is None:
            raise RuntimeError("Stage-wise GraphPrompt+ produced no embeddings.")

        embeddings = mixed_embeddings
        if apply_prompt_dropout and self.prompt_dropout > 0.0:
            embeddings = F.dropout(embeddings, p=self.prompt_dropout, training=True)
        return embeddings, labels


    def _prepare_task_labels(self, labels: torch.Tensor) -> torch.Tensor:
        labels = torch.as_tensor(labels)
        if self.single_label_classification:
            return prepare_single_label_labels(labels)
        if labels.dim() == 0:
            return labels.view(1)
        return labels

    def _extract_embeddings_and_labels(
        self,
        model: nn.Module,
        data,
        device: torch.device,
        mask_attr: str,
        apply_prompt_dropout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_plus and isinstance(self.prompt, GraphPromptPlusStageWise):
            if not self._supports_stagewise_plus_backbone(model):
                supported = supported_graphprompt_plus_backbones()
                raise ValueError(
                    f"[Finetune][GraphPrompt+] Stage-wise prompting needs an adapter-compatible "
                    f"backbone; got {model.__class__.__name__}. "
                    f"Supported backbones: {list(supported)}."
                )
            return self._extract_stagewise_plus_embeddings(
                model=model,
                data=data,
                device=device,
                mask_attr=mask_attr,
                apply_prompt_dropout=apply_prompt_dropout,
            )

        data = data.to(device)
        node_repr, _graph_repr = model(data)
        if self.repr_source == "layer_concat" and getattr(model, "returns_layer_cache", False):
            layer_nodes = model.get_layer_node_reprs()
            if layer_nodes:
                node_repr = torch.cat(layer_nodes, dim=-1)
        node_repr = align_last_dim(node_repr, self.repr_dim)
        node_repr = self._apply_node_postprocess(node_repr, data)
        prompted_node_repr = self.prompt(node_repr)
        if apply_prompt_dropout and self.prompt_dropout > 0.0:
            prompted_node_repr = F.dropout(prompted_node_repr, p=self.prompt_dropout, training=True)

        if self.task_level == "node":
            mask = normalize_node_mask(data, mask_attr, device)
            labels = self._prepare_task_labels(data.y[mask]).to(device)
            embeddings = prompted_node_repr[mask]
        else:
            if self.graph_pooling_mode == "encoder":
                if self.repr_source == "layer_concat" and hasattr(model, "cached_layer_graph_reprs") and model.cached_layer_graph_reprs:
                    graph_repr = torch.cat(model.cached_layer_graph_reprs, dim=-1)
                else:
                    graph_repr = _graph_repr
                if graph_repr is None:
                    batch = get_batch_vector(data)
                    graph_repr = pool_nodes(
                        x=node_repr,
                        batch=batch,
                        mode=self.cfg.model.graph_pooling,
                    )
                graph_repr = align_last_dim(graph_repr, self.repr_dim)
                graph_repr = self.prompt(graph_repr)
            elif self.graph_pooling_mode == "target":
                graph_repr = pool_target_nodes(prompted_node_repr, data)
            else:
                batch = get_batch_vector(data)
                graph_repr = pool_nodes(
                    x=prompted_node_repr,
                    batch=batch,
                    mode=self.graph_pooling_mode,
                )
            if apply_prompt_dropout and self.prompt_dropout > 0.0:
                graph_repr = F.dropout(graph_repr, p=self.prompt_dropout, training=True)
            labels = self._prepare_task_labels(data.y).to(device)
            embeddings = graph_repr

        return embeddings, labels

    def _maybe_print_generalized_notice(self) -> None:
        if self.single_label_classification or self._generalized_notice_printed:
            return
        print("[Finetune][GraphPrompt] Using task-aware prediction head for multi-label/regression finetuning.")
        self._generalized_notice_printed = True

    def _similarity_logits(self, embeddings: torch.Tensor, centers: torch.Tensor, is_train: bool) -> torch.Tensor:
        return similarity_logits(embeddings, centers, self.score_mode, self.tau, is_train)

    def _fill_missing_centers(
        self,
        centers: torch.Tensor,
        counts: torch.Tensor,
        fallback: torch.Tensor | None,
    ) -> torch.Tensor:
        return fill_missing_centers(centers, counts, fallback)

    def _compute_reference_centers(
        self,
        model: nn.Module,
        loader,
        device: torch.device,
        mask_attr: str,
    ) -> torch.Tensor | None:
        accum_centers = torch.zeros(self.num_classes, self.repr_dim, device=device)
        accum_counts = torch.zeros(self.num_classes, 1, device=device)
        observed = 0

        # Prototype extraction must run in eval mode to avoid dropout/BN noise.
        model_was_training = model.training
        prompt_was_training = self.prompt.training
        model.eval()
        self.prompt.eval()
        try:
            with torch.no_grad():
                for data in loader:
                    embeddings, labels = self._extract_embeddings_and_labels(
                        model=model,
                        data=data,
                        device=device,
                        mask_attr=mask_attr,
                        apply_prompt_dropout=False,
                    )
                    if embeddings.numel() == 0:
                        continue
                    centers, counts = compute_class_centers(embeddings, labels, self.num_classes)
                    accum_centers += centers * counts
                    accum_counts += counts
                    observed += int(labels.numel())
        finally:
            if model_was_training:
                model.train()
            if prompt_was_training:
                self.prompt.train()

        if observed == 0:
            return None
        return accum_centers / accum_counts.clamp_min(1.0)

    def train_epoch(self, model, loader, device, optimizers=None):
        optimizer = optimizers.get("primary") if isinstance(optimizers, dict) else optimizers
        if optimizer is None:
            raise ValueError("GraphPrompt requires an optimizer.")

        if not self.single_label_classification:
            self._maybe_print_generalized_notice()
            # Encoder mode is handled by the runner via _apply_frozen_encoder_mode.
            self.prompt.train()
            if self.classifier is not None:
                self.classifier.train()

            total_loss = 0.0
            total_primary = 0.0
            num_batches = 0
            for data in loader:
                optimizer.zero_grad()
                embeddings, labels = self._extract_embeddings_and_labels(
                    model=model,
                    data=data,
                    device=device,
                    mask_attr="train_mask",
                    apply_prompt_dropout=True,
                )
                if embeddings.numel() == 0:
                    continue
                loss, primary, _logits, _labels = self.objective.forward_with_classifier(
                    classifier=self.classifier,
                    representations=embeddings,
                    labels=labels,
                    input_dim=self.repr_dim,
                    return_outputs=True,
                )
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                total_primary += float(primary)
                num_batches += 1

            if num_batches == 0:
                # All batches had empty embeddings (empty masks).
                # Return gracefully rather than raising — this is a
                # legitimate edge case for plus-variant training.
                return 0.0, {}
            metric_name = "train_mae" if self.task_type == "regression" else "train_acc"
            return total_loss / num_batches, {metric_name: total_primary / num_batches}

        # Encoder mode is handled by the runner via _apply_frozen_encoder_mode.
        self.prompt.train()

        total_loss = 0.0
        total_loss_denom = 0
        total_acc = 0.0
        num_batches = 0

        accum_centers = torch.zeros(self.num_classes, self.repr_dim, device=device)
        accum_counts = torch.zeros(self.num_classes, 1, device=device)
        epoch_train_centers = None
        if self.train_center_mode == "train":
            epoch_train_centers = self._compute_reference_centers(
                model=model,
                loader=loader,
                device=device,
                mask_attr="train_mask",
            )
        warm_start_centers = None
        if self.train_center_mode in {"batch", "ema"}:
            warm_start_centers = self._prototype_bank_or_none()
            if warm_start_centers is None:
                warm_start_centers = self._latest_centers_or_none()
            if warm_start_centers is None:
                warm_start_centers = self._compute_reference_centers(
                    model=model,
                    loader=loader,
                    device=device,
                    mask_attr="train_mask",
                )
        runtime_bank = warm_start_centers.detach().clone() if warm_start_centers is not None else None

        for data in loader:
            optimizer.zero_grad()
            embeddings, labels = self._extract_embeddings_and_labels(
                model=model,
                data=data,
                device=device,
                mask_attr="train_mask",
                apply_prompt_dropout=True,
            )
            if embeddings.numel() == 0:
                continue

            batch_centers, batch_counts = compute_class_centers(embeddings, labels, self.num_classes)
            centers_for_loss = batch_centers
            if self.train_center_mode == "train" and epoch_train_centers is not None:
                centers_for_loss = epoch_train_centers
            elif runtime_bank is not None:
                centers_for_loss = self._fill_missing_centers(
                    centers=batch_centers,
                    counts=batch_counts,
                    fallback=runtime_bank,
                )

            logits = self._similarity_logits(embeddings, centers_for_loss, is_train=True)
            loss = F.cross_entropy(logits, labels, reduction=self.loss_reduction)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                total_acc += float((pred == labels).float().mean().item())
                accum_centers += batch_centers.detach() * batch_counts.detach()
                accum_counts += batch_counts.detach()
                if runtime_bank is None:
                    runtime_bank = batch_centers.detach().clone()
                else:
                    present = batch_counts.view(-1) > 0
                    runtime_bank[present] = batch_centers.detach()[present]
                if self.train_center_mode == "ema":
                    if not self._has_prototype_bank():
                        self._store_prototype_bank(batch_centers)
                    else:
                        present = batch_counts.view(-1) > 0
                        self.prototype_bank[present] = (
                            self.center_momentum * self.prototype_bank[present]
                            + (1.0 - self.center_momentum) * batch_centers.detach()[present]
                        )

            total_loss += float(loss.item())
            total_loss_denom += int(labels.numel()) if self.loss_reduction == "sum" else 1
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("Train loader is empty; unable to run a training epoch.")

        mean_loss = total_loss / max(1, total_loss_denom)
        self._store_latest_centers(accum_centers / accum_counts.clamp_min(1.0))
        if not self._has_prototype_bank():
            self._store_prototype_bank(self.latest_centers)
        return mean_loss, {"train_acc": total_acc / num_batches}

    def on_epoch_end(self, model: nn.Module, loader, device):
        if not self.single_label_classification:
            return None
        latest_centers = self._compute_reference_centers(
            model=model,
            loader=loader,
            device=device,
            mask_attr="train_mask",
        )
        self._store_latest_centers(latest_centers)
        if latest_centers is not None:
            self._store_prototype_bank(latest_centers)
        return None

    def evaluate_split(self, model, loader, device, prefix: str, mask_attr: str) -> dict[str, float]:
        if not self.single_label_classification:
            model.eval()
            self.prompt.eval()
            if self.classifier is not None:
                self.classifier.eval()

            def _generalized_forward(data, device):
                embeddings, labels = self._extract_embeddings_and_labels(
                    model=model, data=data, device=device,
                    mask_attr=mask_attr, apply_prompt_dropout=False,
                )
                if embeddings.numel() == 0:
                    return None, None, None
                loss, _primary, logits, labels = self.objective.forward_with_classifier(
                    classifier=self.classifier,
                    representations=embeddings, labels=labels,
                    input_dim=self.repr_dim, return_outputs=True,
                )
                return loss, logits, labels

            return evaluate_epoch_split(
                forward_fn=_generalized_forward,
                loader=loader,
                device=device,
                prefix=prefix,
                task_type=self.task_type,
            )

        model.eval()
        self.prompt.eval()

        total_loss = 0.0
        total_loss_denom = 0
        num_batches = 0
        all_logits = []
        all_labels = []

        reference_centers = self._latest_centers_or_none()
        if reference_centers is None:
            # Never fall back to computing centers from the evaluation
            # loader: that builds prototypes from val/test labels and then
            # scores those same samples against them (label leakage).
            raise RuntimeError(
                "[Finetune][GraphPrompt] no train-split class centers are "
                "available; evaluate() was called before the first training "
                "epoch populated them."
            )

        with torch.no_grad():
            for data in loader:
                embeddings, labels = self._extract_embeddings_and_labels(
                    model=model,
                    data=data,
                    device=device,
                    mask_attr=mask_attr,
                    apply_prompt_dropout=False,
                )
                if embeddings.numel() == 0:
                    continue
                logits = self._similarity_logits(embeddings, reference_centers, is_train=False)
                loss = F.cross_entropy(logits, labels, reduction=self.loss_reduction)
                total_loss += float(loss.item())
                total_loss_denom += int(labels.numel()) if self.loss_reduction == "sum" else 1
                num_batches += 1
                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())

        if num_batches == 0:
            return {}

        metrics = {
            f"{prefix}_loss": total_loss / max(1, total_loss_denom),
        }
        metrics.update(concat_and_compute_metrics(all_logits, all_labels, self.task_type, prefix))
        return metrics
