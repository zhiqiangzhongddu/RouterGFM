"""ProNoG finetuning method (Non-Homophilic Graph Pre-Training and Prompt Learning, KDD'25)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.finetune.prompts import ProNoGConditionNet, compute_class_centers
from src.finetune.registry import register
from src.finetune.task_base import FinetuneTask
from src.finetune.task_heads import (
    TaskAwareObjective,
    align_last_dim,
    build_task_aware_classifier,
    prepare_single_label_labels,
)
from src.utils.config_helpers import (
    build_prompt_head_optimizer,
    cfg_default,
    optimizer_variant_tags,
    tag_if_nondefault,
    validate_choice,
)
from src.utils.dataset_helpers import normalize_node_mask, read_effective_task_level
from src.utils.parsing import resolve_task_type
from src.utils.pool import get_batch_vector, pool_nodes, pool_target_nodes
from src.utils.supervised_eval import concat_and_compute_metrics, evaluate_epoch_split

from .graphprompt_utils import fill_missing_centers, similarity_logits
from .pronog_utils import build_hop_neighbor_pairs, conditioned_subgraph_readout

#: Logit assigned to classes with no training examples at evaluation time.
#: Large enough to never win argmax over cosine/tau logits, small enough to
#: keep cross-entropy finite when an evaluation label happens to be unseen.
_UNSEEN_CLASS_LOGIT = -1.0e4


@register("pronog")
class FinetuneProNoG(FinetuneTask):
    """Condition-net prompting with node-specific prompts and prototype logits.

    A frozen encoder produces node embeddings; each node's prompt is generated
    by a bottleneck-MLP condition-net from a similarity-weighted readout of
    its capped multi-hop ego-network (paper Eq. 7-8).  Single-label
    classification scores prompted embeddings against class-prototype centers
    with cosine similarity (paper Eq. 10); multi-label/regression falls back
    to a task-aware classifier head on the prompted embeddings.
    """

    requires_frozen_encoder = True
    # Fixed-epoch budget, mirroring FinetuneGraphPrompt: the shared patience
    # (finetune.early_stopping=20) neither matches the official ProNoG
    # patience-50 protocol nor keeps pronog rows budget-comparable to the
    # sibling prototype method in the same results table.
    supports_early_stopping = False

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        cls.require_node_or_graph_batches(cfg, method_label="ProNoG")

        method_cfg = getattr(getattr(cfg, "finetune", None), "pronog", None)
        if method_cfg is None:
            return
        # Raw reads on purpose: `int(x or default)` would swallow an explicit
        # 0 and make the >=1 guards unreachable (set_cfg guarantees the keys).
        hops = int(method_cfg.hops)
        if hops < 1:
            raise ValueError(f"pronog.hops must be >= 1; got {hops}.")
        neighbor_cap = int(method_cfg.neighbor_cap)
        if neighbor_cap < 0:
            raise ValueError(f"pronog.neighbor_cap must be >= 0; got {neighbor_cap}.")
        bottleneck_dim = int(method_cfg.bottleneck_dim)
        if bottleneck_dim < 1:
            raise ValueError(f"pronog.bottleneck_dim must be >= 1; got {bottleneck_dim}.")
        condition_dropout = float(method_cfg.condition_dropout)
        if condition_dropout < 0.0 or condition_dropout >= 1.0:
            raise ValueError(
                f"pronog.condition_dropout must be in [0.0, 1.0); got {condition_dropout}."
            )
        condition_scaling = float(method_cfg.condition_scaling)
        if condition_scaling < 0.0:
            raise ValueError(
                f"pronog.condition_scaling must be >= 0.0; got {condition_scaling}."
            )
        tau = float(method_cfg.tau)
        if tau <= 0.0:
            raise ValueError(f"pronog.tau must be > 0; got {tau}.")
        combine = validate_choice(
            "pronog.prompt_combine", str(method_cfg.prompt_combine), {"add", "mul"},
        )
        if combine == "mul" and condition_scaling == 0.0:
            raise ValueError(
                "pronog.condition_scaling=0 with prompt_combine='mul' collapses "
                "every prompted embedding to zero; use prompt_combine='add' for "
                "the no-prompt ablation."
            )
        validate_choice(
            "pronog.graph_pooling",
            str(method_cfg.graph_pooling),
            {"sum", "add", "mean", "max", "target"},
        )
        validate_choice(
            "pronog.train_center_mode",
            str(method_cfg.train_center_mode),
            {"batch", "train"},
        )
        eval_center = str(method_cfg.eval_center_mode).lower()
        if eval_center == "batch":
            raise ValueError(
                "pronog.eval_center_mode='batch' is not supported because it "
                "builds prototypes from validation/test labels before scoring "
                "the same examples. Use 'train'."
            )
        validate_choice("pronog.eval_center_mode", eval_center, {"train"})

    @classmethod
    def run_tag(cls, cfg) -> str:
        method_cfg = getattr(getattr(cfg, "finetune", None), "pronog", None)
        if method_cfg is None:
            return "h2_add"
        # Raw reads: an `or`-fallback here would let an invalid hops=0 run
        # masquerade under the default h2 run name.
        hops = int(method_cfg.hops)
        combine = str(method_cfg.prompt_combine).lower()
        return f"h{hops}_{combine}"

    @classmethod
    def variant_tag(cls, cfg) -> str:
        method_cfg = getattr(getattr(cfg, "finetune", None), "pronog", None)
        if method_cfg is None:
            return ""
        tags = []
        t = tag_if_nondefault(
            "cap",
            int(method_cfg.neighbor_cap),
            int(cfg_default("finetune.pronog.neighbor_cap")),
        )
        if t:
            tags.append(t)
        t = tag_if_nondefault(
            "bn",
            int(method_cfg.bottleneck_dim),
            int(cfg_default("finetune.pronog.bottleneck_dim")),
        )
        if t:
            tags.append(t)
        t = tag_if_nondefault(
            "cdrop",
            float(method_cfg.condition_dropout),
            float(cfg_default("finetune.pronog.condition_dropout")),
        )
        if t:
            tags.append(t)
        t = tag_if_nondefault(
            "cscale",
            float(method_cfg.condition_scaling),
            float(cfg_default("finetune.pronog.condition_scaling")),
        )
        if t:
            tags.append(t)
        t = tag_if_nondefault(
            "tau",
            float(method_cfg.tau),
            float(cfg_default("finetune.pronog.tau")),
        )
        if t:
            tags.append(t)
        default_pooling = str(cfg_default("finetune.pronog.graph_pooling")).lower()
        pooling = str(method_cfg.graph_pooling).lower()
        if pooling != default_pooling:
            tags.append(f"pool_{pooling}")
        default_train_center = str(cfg_default("finetune.pronog.train_center_mode")).lower()
        train_center = str(method_cfg.train_center_mode).lower()
        if train_center != default_train_center:
            tags.append(f"tc_{train_center}")
        t = tag_if_nondefault(
            "plr",
            float(method_cfg.prompt_lr),
            float(cfg_default("finetune.pronog.prompt_lr")),
        )
        if t:
            tags.append(t)
        tags.extend(optimizer_variant_tags(method_cfg, "pronog", exclude={"prompt_lr"}))
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
                "[Finetune][ProNoG] Running node ProNoG with induced subgraphs. "
                "Official ProNoG uses non-induced node training; consider finetune.dataset.induced=False."
            )

        hidden_dim = int(getattr(cfg.model, "hidden_dim", 1) or 1)
        self.repr_dim = int(getattr(cfg.model, "out_dim", hidden_dim) or hidden_dim)
        self.objective = TaskAwareObjective(cfg, task_level=self.task_level, repr_dim=self.repr_dim)
        self.single_label_classification = self.objective.is_single_label_classification

        # Choice validation lives in validate_cfg (always run by
        # build_finetune_task / the runner before construction); plain reads
        # here avoid a second drift-prone allow-list.
        method_cfg = cfg.finetune.pronog
        self.hops = int(method_cfg.hops)
        self.neighbor_cap = int(method_cfg.neighbor_cap)
        self.tau = float(method_cfg.tau)
        self.prompt_combine = str(method_cfg.prompt_combine).lower()
        self.graph_pooling_mode = str(method_cfg.graph_pooling).lower()
        self.train_center_mode = str(method_cfg.train_center_mode).lower()

        self.prompt = ProNoGConditionNet(
            in_channels=self.repr_dim,
            bottleneck_channels=int(method_cfg.bottleneck_dim),
            dropout=float(method_cfg.condition_dropout),
            scaling=float(method_cfg.condition_scaling),
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
        # Prediction for single-label ProNoG depends on the learned
        # train-split prototypes. Register fixed-shape buffers so task
        # state_dict checkpoints reproduce predictions after a round-trip
        # (same layout as GraphPrompt, plus per-class counts so evaluation
        # can exclude classes that have no training examples).
        center_shape = (self.num_classes, self.repr_dim)
        self.register_buffer("latest_centers", torch.zeros(center_shape))
        self.register_buffer("latest_center_counts", torch.zeros(self.num_classes))
        self.register_buffer("prototype_bank", torch.zeros(center_shape))
        self.register_buffer("_latest_centers_valid", torch.tensor(False))
        self.register_buffer("_prototype_bank_valid", torch.tensor(False))
        self._generalized_notice_printed = False
        # Single-entry cache of the hop-pair list: (edge_index, num_nodes,
        # pairs).  The pair list is a pure function of the graph structure,
        # so full-graph node training reuses one build for the whole run.
        self._pair_cache = None

    def _has_latest_centers(self) -> bool:
        return bool(self._latest_centers_valid.item())

    def _has_prototype_bank(self) -> bool:
        return bool(self._prototype_bank_valid.item())

    def _latest_centers_or_none(self) -> torch.Tensor | None:
        return self.latest_centers if self._has_latest_centers() else None

    def _prototype_bank_or_none(self) -> torch.Tensor | None:
        return self.prototype_bank if self._has_prototype_bank() else None

    def _store_latest_centers(
        self,
        centers: torch.Tensor | None,
        counts: torch.Tensor | None,
    ) -> None:
        """Store train-split centers and their per-class support counts.

        ``counts`` is required (not defaulted) so the counts can never go
        stale behind valid centers — the unseen-class mask reads them.
        """
        if centers is None or counts is None:
            self._latest_centers_valid.fill_(False)
            return
        self.latest_centers.copy_(centers.detach().to(self.latest_centers))
        self.latest_center_counts.copy_(
            counts.detach().view(-1).to(self.latest_center_counts)
        )
        self._latest_centers_valid.fill_(True)

    def _unseen_class_mask(self, counts: torch.Tensor | None) -> torch.Tensor | None:
        """Boolean mask of classes with no training support, or ``None``.

        Returns ``None`` when every class is represented, so callers skip the
        masked_fill (and its device sync) in the common case.
        """
        if counts is None:
            return None
        unseen = counts.view(-1) <= 0
        return unseen if bool(unseen.any()) else None

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

    def build_optimizers(self, model: nn.Module):
        method_cfg = self.cfg.finetune.pronog
        # Official ProNoG optimizes the condition-net with plain Adam at
        # down_lr and no weight decay; the fallback classifier head (when
        # present) shares the same LR/WD policy.
        head_params = list(self.classifier.parameters()) if self.classifier is not None else None
        return build_prompt_head_optimizer(
            method_cfg=method_cfg,
            prompt_params=self.prompt.parameters(),
            head_params=head_params,
            base_lr=float(method_cfg.prompt_lr),
            base_wd=float(method_cfg.prompt_weight_decay),
            head_lr_scale=1.0,
        )

    def _cached_hop_pairs(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        cached = self._pair_cache
        if cached is not None:
            cached_edges, cached_num_nodes, cached_pairs = cached
            if (
                cached_num_nodes == num_nodes
                and cached_edges.shape == edge_index.shape
                and cached_edges.device == edge_index.device
                and torch.equal(cached_edges, edge_index)
            ):
                return cached_pairs
        pairs = build_hop_neighbor_pairs(
            edge_index=edge_index,
            num_nodes=num_nodes,
            hops=self.hops,
            cap=self.neighbor_cap,
        )
        self._pair_cache = (edge_index, num_nodes, pairs)
        return pairs

    def _apply_conditional_prompt(self, node_repr: torch.Tensor, data) -> torch.Tensor:
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None or edge_index.numel() == 0:
            # No structure: the ego-network degenerates to the node itself,
            # so the condition equals the node embedding (cos(h_v, h_v) = 1).
            condition = node_repr
        else:
            pairs = self._cached_hop_pairs(edge_index, node_repr.size(0))
            condition = conditioned_subgraph_readout(node_repr, pairs)
        prompts = self.prompt(condition)
        if self.prompt_combine == "mul":
            return prompts * node_repr
        return node_repr + prompts

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = data.to(device)
        node_repr, _graph_repr = model(data)
        node_repr = align_last_dim(node_repr, self.repr_dim)
        prompted_node_repr = self._apply_conditional_prompt(node_repr, data)

        if self.task_level == "node":
            mask = normalize_node_mask(data, mask_attr, device)
            labels = self._prepare_task_labels(data.y[mask]).to(device)
            embeddings = prompted_node_repr[mask]
        else:
            if self.graph_pooling_mode == "target":
                graph_repr = pool_target_nodes(prompted_node_repr, data)
            else:
                batch = get_batch_vector(data)
                graph_repr = pool_nodes(
                    x=prompted_node_repr,
                    batch=batch,
                    mode=self.graph_pooling_mode,
                )
            labels = self._prepare_task_labels(data.y).to(device)
            embeddings = graph_repr

        return embeddings, labels

    def _maybe_print_generalized_notice(self) -> None:
        if self.single_label_classification or self._generalized_notice_printed:
            return
        print("[Finetune][ProNoG] Using task-aware prediction head for multi-label/regression finetuning.")
        self._generalized_notice_printed = True

    def _similarity_logits(
        self,
        embeddings: torch.Tensor,
        centers: torch.Tensor,
        unseen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cosine prototype logits, with train-absent classes masked out.

        A class with no training examples keeps an all-zero centre, and the
        cosine against a zero vector is exactly 0 — a mid-range logit that
        would outrank every trained class scoring negative.  Masking is
        applied identically in training and evaluation so both see the same
        label space.
        """
        logits = similarity_logits(embeddings, centers, "cosine", self.tau, is_train=False)
        if unseen is not None:
            logits = logits.masked_fill(unseen.unsqueeze(0), _UNSEEN_CLASS_LOGIT)
        return logits

    def _compute_reference_centers(
        self,
        model: nn.Module,
        loader,
        device: torch.device,
        mask_attr: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        accum_centers = torch.zeros(self.num_classes, self.repr_dim, device=device)
        accum_counts = torch.zeros(self.num_classes, 1, device=device)
        observed = 0

        # Prototype extraction must run in eval mode to avoid dropout noise.
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
            return None, None
        return accum_centers / accum_counts.clamp_min(1.0), accum_counts

    def train_epoch(self, model, loader, device, optimizers=None):
        optimizer = optimizers.get("primary") if isinstance(optimizers, dict) else optimizers
        if optimizer is None:
            raise ValueError("ProNoG requires an optimizer.")

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
                return 0.0, {}
            metric_name = "train_mae" if self.task_type == "regression" else "train_acc"
            return total_loss / num_batches, {metric_name: total_primary / num_batches}

        # Encoder mode is handled by the runner via _apply_frozen_encoder_mode.
        self.prompt.train()

        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        accum_centers = torch.zeros(self.num_classes, self.repr_dim, device=device)
        accum_counts = torch.zeros(self.num_classes, 1, device=device)
        epoch_train_centers = None
        train_counts = None
        runtime_bank = None
        if self.train_center_mode == "train":
            epoch_train_centers, train_counts = self._compute_reference_centers(
                model=model,
                loader=loader,
                device=device,
                mask_attr="train_mask",
            )
        else:
            warm_start_centers = self._prototype_bank_or_none()
            if warm_start_centers is None:
                warm_start_centers = self._latest_centers_or_none()
            if warm_start_centers is None:
                warm_start_centers, train_counts = self._compute_reference_centers(
                    model=model,
                    loader=loader,
                    device=device,
                    mask_attr="train_mask",
                )
            # Only batch mode consults the running bank; in train mode every
            # batch scores against the epoch-level centers instead.
            runtime_bank = (
                warm_start_centers.detach().clone() if warm_start_centers is not None else None
            )
        if train_counts is None and self._has_latest_centers():
            # Warm start came from the stored bank, whose support counts are
            # the ones recorded alongside the last train-split recompute.
            train_counts = self.latest_center_counts
        unseen = self._unseen_class_mask(train_counts)

        for data in loader:
            optimizer.zero_grad()
            embeddings, labels = self._extract_embeddings_and_labels(
                model=model,
                data=data,
                device=device,
                mask_attr="train_mask",
            )
            if embeddings.numel() == 0:
                continue

            batch_centers, batch_counts = compute_class_centers(embeddings, labels, self.num_classes)
            centers_for_loss = batch_centers
            if self.train_center_mode == "train" and epoch_train_centers is not None:
                centers_for_loss = epoch_train_centers
            elif runtime_bank is not None:
                centers_for_loss = fill_missing_centers(
                    centers=batch_centers,
                    counts=batch_counts,
                    fallback=runtime_bank,
                )

            logits = self._similarity_logits(embeddings, centers_for_loss, unseen)
            loss = F.cross_entropy(logits, labels)

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

            total_loss += float(loss.item())
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("Train loader is empty; unable to run a training epoch.")

        self._store_latest_centers(
            accum_centers / accum_counts.clamp_min(1.0), accum_counts,
        )
        if not self._has_prototype_bank():
            self._store_prototype_bank(self.latest_centers)
        return total_loss / num_batches, {"train_acc": total_acc / num_batches}

    def on_epoch_end(self, model: nn.Module, loader, device):
        if not self.single_label_classification:
            return None
        latest_centers, latest_counts = self._compute_reference_centers(
            model=model,
            loader=loader,
            device=device,
            mask_attr="train_mask",
        )
        self._store_latest_centers(latest_centers, latest_counts)
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
                    model=model, data=data, device=device, mask_attr=mask_attr,
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
        loss_batches = 0
        all_logits = []
        all_labels = []

        reference_centers = self._latest_centers_or_none()
        if reference_centers is None:
            # Never fall back to computing centers from the evaluation
            # loader: that builds prototypes from val/test labels and then
            # scores those same samples against them (label leakage).
            raise RuntimeError(
                "[Finetune][ProNoG] no train-split class centers are "
                "available; evaluate() was called before the first training "
                "epoch populated them."
            )
        unseen = self._unseen_class_mask(self.latest_center_counts)

        with torch.no_grad():
            for data in loader:
                embeddings, labels = self._extract_embeddings_and_labels(
                    model=model,
                    data=data,
                    device=device,
                    mask_attr=mask_attr,
                )
                if embeddings.numel() == 0:
                    continue
                logits = self._similarity_logits(embeddings, reference_centers, unseen)
                scored = labels
                scored_logits = logits
                if unseen is not None:
                    # A sample whose gold class has no prototype is already
                    # counted as an error by the metrics below; keeping it in
                    # the loss would add an arbitrary sentinel-sized term and
                    # make *_loss (and the val monitor) unreadable.
                    keep = ~unseen[labels]
                    scored_logits, scored = logits[keep], labels[keep]
                if scored.numel() > 0:
                    loss = F.cross_entropy(scored_logits, scored)
                    total_loss += float(loss.item())
                    loss_batches += 1
                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())

        if not all_logits:
            return {}

        metrics = {}
        if loss_batches > 0:
            # Omitted only when every evaluated sample belongs to a class with
            # no training support; metrics below still score all of them.
            metrics[f"{prefix}_loss"] = total_loss / loss_batches
        metrics.update(concat_and_compute_metrics(all_logits, all_labels, self.task_type, prefix))
        return metrics
