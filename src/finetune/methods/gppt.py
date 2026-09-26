"""GPPT prompt finetuning method."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.finetune.prompts.gppt import GPPTPrompt
from src.finetune.registry import register
from src.finetune.task_heads import TaskAwareObjective, align_last_dim, prepare_single_label_labels
from src.finetune.task_base import FinetuneTask
from src.utils.dataset_helpers import normalize_node_mask, read_effective_task_level
from src.utils.pool import get_batch_vector, pool_nodes
from src.utils.supervised_eval import evaluate_epoch_split
from src.utils.config_helpers import (
    build_prompt_head_optimizer,
    cfg_default,
    optimizer_variant_tags,
    tag_if_nondefault,
    validate_choice,
)
from src.utils.parsing import to_bool


@register("gppt")
class FinetuneGPPT(FinetuneTask):
    """Official GPPT-style prompt tuning with fixed structure/task tokens.

    The encoder is frozen during prompt tuning, matching the paper's intent.
    (The original GPPT repo's code optimizes all parameters including the GNN
    — that appears to be a code-vs-paper
    discrepancy rather than an intentional design choice.)
    """

    requires_frozen_encoder = True
    default_monitor = "train_loss"

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        cls.require_node_or_graph_batches(cfg, method_label="GPPT")

        gppt_cfg = getattr(getattr(cfg, "finetune", None), "gppt", None)
        if gppt_cfg is None:
            return
        valid_modes = {"node", "neighbor", "concat"}
        structure_mode = str(getattr(gppt_cfg, "structure_mode", "concat") or "concat").lower()
        validate_choice("gppt.structure_mode", structure_mode, valid_modes)
        task_mode = str(getattr(gppt_cfg, "task_mode", "concat") or "concat").lower()
        validate_choice("gppt.task_mode", task_mode, valid_modes)

    @classmethod
    def variant_tag(cls, cfg) -> str:
        gppt_cfg = getattr(getattr(cfg, "finetune", None), "gppt", None)
        if gppt_cfg is None:
            return ""
        tags = []
        center_num = int(getattr(gppt_cfg, "center_num", 0) or 0)
        if center_num > 0:
            tags.append(f"c{center_num}")
        default_structure_mode = str(cfg_default("finetune.gppt.structure_mode")).lower()
        default_task_mode = str(cfg_default("finetune.gppt.task_mode")).lower()
        structure_mode = str(getattr(gppt_cfg, "structure_mode", default_structure_mode) or default_structure_mode).lower()
        task_mode = str(getattr(gppt_cfg, "task_mode", default_task_mode) or default_task_mode).lower()
        if structure_mode != default_structure_mode:
            tags.append(f"s{structure_mode}")
        if task_mode != default_task_mode:
            tags.append(f"t{task_mode}")
        cw = float(getattr(gppt_cfg, "constraint_weight", 1e-2) or 1e-2)
        t = tag_if_nondefault("cw", cw, float(cfg_default("finetune.gppt.constraint_weight")))
        if t:
            tags.append(t)
        if to_bool(getattr(gppt_cfg, "add_self_loops", False)):
            tags.append("loops1")
        tags.extend(optimizer_variant_tags(gppt_cfg, "gppt"))
        return "-".join(tags)

    def __init__(self, cfg):
        super().__init__(cfg)
        ds_cfg = cfg.finetune.dataset
        self.task_level = read_effective_task_level(ds_cfg)
        hidden_dim = int(getattr(cfg.model, "hidden_dim", 1) or 1)
        self.repr_dim = int(getattr(cfg.model, "out_dim", hidden_dim) or hidden_dim)
        self.objective = TaskAwareObjective(cfg, task_level=self.task_level, repr_dim=self.repr_dim)
        self.task_type = self.objective.task_type
        if self.task_type in {"none", ""}:
            raise ValueError(
                "GPPT requires finetune.dataset.task_type to resolve to 'classification' or 'regression'. "
                f"Got '{self.task_type}'. Set finetune.dataset.task_type explicitly in your config or CLI."
            )
        self.label_dim = self.objective.label_dim
        self.num_classes = max(2, int(getattr(ds_cfg, "num_classes", 2) or 2))
        self.single_label_classification = self.objective.is_single_label_classification
        # GPPT's forward uses F.cross_entropy and class-based voting, so the
        # head must always produce num_classes logits for single-label tasks
        # (even binary graph classification where the supervised convention is
        # a single BCE logit).
        self.task_output_dim = (
            self.num_classes if self.single_label_classification else self.objective.output_dim
        )

        method_cfg = getattr(cfg.finetune, "gppt", None)
        raw_center_num = int(getattr(method_cfg, "center_num", 0)) if method_cfg is not None else 0
        if raw_center_num > 0:
            center_num = raw_center_num
        else:
            center_num = self.num_classes if self.single_label_classification else max(1, self.label_dim)
        self.constraint_weight = (
            float(getattr(method_cfg, "constraint_weight", 1e-2)) if method_cfg is not None else 1e-2
        )
        self.update_structure_every_step = (
            to_bool(getattr(method_cfg, "update_structure_every_step", True))
            if method_cfg is not None
            else True
        )
        self.update_structure_from_mask = (
            to_bool(getattr(method_cfg, "update_structure_from_mask", False))
            if method_cfg is not None
            else False
        )
        structure_mode = str(getattr(method_cfg, "structure_mode", "concat")) if method_cfg is not None else "concat"
        task_mode = str(getattr(method_cfg, "task_mode", "concat")) if method_cfg is not None else "concat"

        self.prompt = GPPTPrompt(
            in_channels=self.repr_dim,
            center_num=center_num,
            num_classes=self.num_classes,
            output_dim=self.task_output_dim,
            structure_mode=structure_mode,
            task_mode=task_mode,
            add_self_loops_in_conv=(
                to_bool(getattr(method_cfg, "add_self_loops", False)) if method_cfg is not None else False
            ),
            kmeans_max_iter=int(getattr(method_cfg, "kmeans_max_iter", 100)) if method_cfg is not None else 100,
            kmeans_tol=float(getattr(method_cfg, "kmeans_tol", 1e-4)) if method_cfg is not None else 1e-4,
            kmeans_restarts=int(getattr(method_cfg, "kmeans_restarts", 1)) if method_cfg is not None else 1,
            use_sklearn_kmeans=(
                to_bool(getattr(method_cfg, "use_sklearn_kmeans", True)) if method_cfg is not None else True
            ),
            kmeans_random_state=int(getattr(method_cfg, "kmeans_random_state", 0)) if method_cfg is not None else 0,
            kmeans_n_init=int(getattr(method_cfg, "kmeans_n_init", 10)) if method_cfg is not None else 10,
            use_prototype_init=self.single_label_classification,
        )
        self._prompt_initialized = False
        self._generalized_notice_printed = False


    def _prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        labels = torch.as_tensor(labels)
        if self.single_label_classification:
            return prepare_single_label_labels(labels)
        if labels.dim() == 0:
            return labels.view(1)
        return labels

    @staticmethod
    def _match_label_size(labels: torch.Tensor, target_size: int) -> torch.Tensor:
        labels = labels.view(-1)
        if target_size <= 0:
            return labels[:0]
        if labels.numel() == target_size:
            return labels
        if labels.numel() == 0:
            return torch.zeros(target_size, dtype=torch.long, device=labels.device)
        if labels.numel() == 1:
            return labels.repeat(target_size)
        if labels.numel() > target_size:
            return labels[:target_size]
        repeat = (target_size + labels.numel() - 1) // labels.numel()
        return labels.repeat(repeat)[:target_size]

    def build_optimizers(self, model: nn.Module):
        method_cfg = getattr(self.cfg.finetune, "gppt", None)
        # GPPT embeds classification tokens inside the prompt module itself
        # (no separate head), so only ``prompt_params`` are registered.
        return build_prompt_head_optimizer(
            method_cfg=method_cfg,
            prompt_params=self.prompt.parameters(),
            head_params=None,
            base_lr=2e-3,
            base_wd=5e-4,
        )

    def _orthogonality_constraint(self) -> torch.Tensor:
        weights = self.prompt.get_TaskToken()
        if not weights:
            return next(self.prompt.parameters()).new_tensor(0.0)
        total = weights[0].new_tensor(0.0)
        for weight in weights:
            gram = weight @ weight.t()
            ident = torch.eye(weight.size(0), device=weight.device, dtype=weight.dtype)
            total = total + torch.norm(gram - ident, p="fro")
        return total / len(weights)

    def _graph_vote_logits(
        self,
        node_logits: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        if num_graphs is None:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        if num_graphs <= 0:
            return node_logits.new_zeros((0, self.num_classes))

        pred = node_logits.argmax(dim=-1)
        one_hot = F.one_hot(pred, num_classes=self.num_classes).to(node_logits.dtype)
        votes = node_logits.new_zeros((num_graphs, self.num_classes))
        votes.scatter_add_(0, batch.view(-1, 1).expand_as(one_hot), one_hot)
        return votes

    def _maybe_print_generalized_notice(self) -> None:
        if self.single_label_classification or self._generalized_notice_printed:
            return
        print("[Finetune][gppt] Using generalized token outputs for multi-label/regression finetuning.")
        self._generalized_notice_printed = True

    def _node_forward(
        self,
        model: nn.Module,
        data,
        device: torch.device,
        mask_attr: str,
    ) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
        data = data.to(device)
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("GPPT requires edge_index for node-level tuning.")

        node_repr, _ = model(data)
        node_repr = align_last_dim(node_repr, self.repr_dim)
        logits = self.prompt(node_repr, edge_index)

        mask = normalize_node_mask(data, mask_attr, device, num_nodes=data.num_nodes)
        if not bool(mask.any().item()):
            return None

        labels = self._prepare_labels(data.y).to(device)
        labels = self._match_label_size(labels, data.num_nodes)
        logits_used = logits[mask]
        labels_used = labels[mask]
        loss = F.cross_entropy(logits_used, labels_used)
        pred = logits_used.argmax(dim=-1)
        acc = float((pred == labels_used).float().mean().item())
        structure_for_update = self.prompt.get_mid_h()
        if structure_for_update is not None:
            if self.update_structure_from_mask:
                structure_for_update = structure_for_update[mask]
            structure_for_update = structure_for_update.detach()
        return loss, acc, logits_used, labels_used, structure_for_update

    def _graph_forward(
        self,
        model: nn.Module,
        data,
        device: torch.device,
    ) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        data = data.to(device)
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("GPPT requires edge_index for graph-level tuning.")

        node_repr, _ = model(data)
        node_repr = align_last_dim(node_repr, self.repr_dim)
        node_logits = self.prompt(node_repr, edge_index)
        batch = get_batch_vector(data).long()
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

        graph_labels = self._prepare_labels(data.y).to(device)
        if num_graphs > 0:
            graph_labels = self._match_label_size(graph_labels, num_graphs)

        if num_graphs > 0:
            graph_losses = []
            for graph_id in range(num_graphs):
                node_mask = batch == graph_id
                if not bool(node_mask.any().item()):
                    continue
                node_label = graph_labels[graph_id].view(1).repeat(int(node_mask.sum().item()))
                graph_losses.append(F.cross_entropy(node_logits[node_mask], node_label))
            if graph_losses:
                loss = torch.stack(graph_losses, dim=0).mean()
            else:
                loss = node_logits.sum() * 0.0
        else:
            loss = node_logits.sum() * 0.0

        graph_votes = self._graph_vote_logits(node_logits=node_logits, batch=batch, num_graphs=num_graphs)
        graph_logits = (
            torch.softmax(graph_votes.float(), dim=-1).to(node_logits.dtype)
            if graph_votes.numel() > 0
            else graph_votes
        )
        pred = graph_logits.argmax(dim=-1) if graph_logits.numel() > 0 else graph_labels.new_zeros((0,), dtype=torch.long)
        acc = float((pred == graph_labels).float().mean().item()) if graph_labels.numel() > 0 else 0.0
        structure_for_update = self.prompt.get_mid_h()
        if structure_for_update is not None:
            structure_for_update = structure_for_update.detach()
        return loss, acc, graph_logits, graph_labels, structure_for_update

    def _node_forward_generalized(
        self,
        model: nn.Module,
        data,
        device: torch.device,
        mask_attr: str,
    ) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
        data = data.to(device)
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("GPPT requires edge_index for node-level tuning.")

        node_repr, _ = model(data)
        node_repr = align_last_dim(node_repr, self.repr_dim)
        node_logits = self.prompt(node_repr, edge_index)

        mask = normalize_node_mask(data, mask_attr, device, num_nodes=data.num_nodes)
        if not bool(mask.any().item()):
            return None

        labels = torch.as_tensor(data.y)
        if labels.dim() > 1 and labels.size(0) == data.num_nodes:
            labels_used = labels[mask]
        else:
            labels_used = labels.view(-1)[mask]
        loss, primary, logits_used, labels_used = self.objective.loss_from_logits(
            logits=node_logits[mask],
            labels=labels_used.to(device),
            return_outputs=True,
        )
        structure_for_update = self.prompt.get_mid_h()
        if structure_for_update is not None:
            if self.update_structure_from_mask:
                structure_for_update = structure_for_update[mask]
            structure_for_update = structure_for_update.detach()
        return loss, primary, logits_used, labels_used, structure_for_update

    def _graph_forward_generalized(
        self,
        model: nn.Module,
        data,
        device: torch.device,
    ) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        data = data.to(device)
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("GPPT requires edge_index for graph-level tuning.")

        node_repr, _ = model(data)
        node_repr = align_last_dim(node_repr, self.repr_dim)
        node_logits = self.prompt(node_repr, edge_index)
        batch = get_batch_vector(data).long()
        graph_logits = pool_nodes(
            x=node_logits,
            batch=batch,
            mode="mean",
        )
        labels = torch.as_tensor(data.y).to(device)
        loss, primary, graph_logits, labels = self.objective.loss_from_logits(
            logits=graph_logits,
            labels=labels,
            return_outputs=True,
        )
        structure_for_update = self.prompt.get_mid_h()
        if structure_for_update is not None:
            structure_for_update = structure_for_update.detach()
        return loss, primary, graph_logits, labels, structure_for_update

    def _maybe_initialize_prompt(self, model: nn.Module, loader, device: torch.device) -> None:
        if self._prompt_initialized:
            return

        node_reprs = []
        edge_indices = []
        node_labels = []
        train_indices = []
        node_offset = 0

        model_was_training = model.training
        model.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                edge_index = getattr(data, "edge_index", None)
                if edge_index is None:
                    continue

                node_repr, _ = model(data)
                node_repr = align_last_dim(node_repr, self.repr_dim)
                num_nodes = int(node_repr.size(0))
                if num_nodes <= 0:
                    continue

                node_reprs.append(node_repr.detach())
                edge_indices.append(edge_index + node_offset)
                if self.task_level == "node":
                    labels = self._prepare_labels(data.y).to(device)
                    if self.single_label_classification:
                        labels = self._match_label_size(labels, num_nodes)
                    elif labels.dim() > 1 and labels.size(0) == num_nodes:
                        labels = labels.view(num_nodes, -1)
                    elif labels.numel() % max(1, num_nodes) == 0:
                        labels = labels.view(num_nodes, -1)
                    else:
                        labels = labels.view(-1)[:num_nodes]
                    node_labels.append(labels)
                    train_mask = getattr(data, "train_mask", None)
                    if train_mask is None:
                        local_idx = torch.arange(num_nodes, device=device)
                    else:
                        local_mask = torch.as_tensor(train_mask, dtype=torch.bool, device=device).view(-1)
                        if local_mask.numel() > num_nodes:
                            local_mask = local_mask[:num_nodes]
                        elif local_mask.numel() < num_nodes:
                            pad = torch.zeros(num_nodes - local_mask.numel(), dtype=torch.bool, device=device)
                            local_mask = torch.cat([local_mask, pad], dim=0)
                        local_idx = torch.nonzero(local_mask, as_tuple=False).view(-1)
                    train_indices.append(local_idx + node_offset)
                else:
                    batch = get_batch_vector(data).long()
                    graph_labels = self._prepare_labels(data.y).to(device)
                    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
                    if num_graphs > 0:
                        if self.single_label_classification:
                            graph_labels = self._match_label_size(graph_labels, num_graphs)
                        elif graph_labels.dim() == 1:
                            if graph_labels.numel() == num_graphs:
                                graph_labels = graph_labels.view(num_graphs, 1)
                            elif graph_labels.numel() % num_graphs == 0:
                                graph_labels = graph_labels.view(num_graphs, -1)
                            else:
                                graph_labels = graph_labels.view(-1, 1)[:num_graphs]
                        elif graph_labels.size(0) != num_graphs and graph_labels.numel() % num_graphs == 0:
                            graph_labels = graph_labels.view(num_graphs, -1)
                        labels = graph_labels[batch]
                    else:
                        labels = graph_labels.new_zeros((0,), dtype=graph_labels.dtype)
                    node_labels.append(labels)
                    train_indices.append(torch.arange(num_nodes, device=device, dtype=torch.long) + node_offset)
                node_offset += num_nodes

        model.train(model_was_training)

        if not node_reprs or not edge_indices or not node_labels:
            return

        all_repr = torch.cat(node_reprs, dim=0)
        all_edge_index = torch.cat(edge_indices, dim=1)
        all_labels = torch.cat(node_labels, dim=0)
        all_indices = (
            torch.cat(train_indices, dim=0) if train_indices else torch.arange(all_repr.size(0), device=device)
        )
        self.prompt.weigth_init(
            h=all_repr,
            edge_index=all_edge_index,
            label=all_labels,
            index=all_indices,
        )
        self._prompt_initialized = True

    def train_epoch(self, model, loader, device, optimizers=None):
        optimizer = optimizers.get("primary") if isinstance(optimizers, dict) else optimizers
        if optimizer is None:
            raise ValueError("GPPT requires an optimizer.")

        self._maybe_print_generalized_notice()
        self._maybe_initialize_prompt(model=model, loader=loader, device=device)
        # Encoder mode is handled by the runner via _apply_frozen_encoder_mode.
        self.prompt.train()

        total_loss = 0.0
        total_primary = 0.0
        num_batches = 0
        for data in loader:
            optimizer.zero_grad()
            structure_for_update = None
            if self.task_level == "node":
                if self.single_label_classification:
                    outputs = self._node_forward(model=model, data=data, device=device, mask_attr="train_mask")
                else:
                    outputs = self._node_forward_generalized(model=model, data=data, device=device, mask_attr="train_mask")
                if outputs is None:
                    continue
                loss, primary, _logits, _labels, structure_for_update = outputs
            else:
                if self.single_label_classification:
                    loss, primary, _logits, _labels, structure_for_update = self._graph_forward(
                        model=model,
                        data=data,
                        device=device,
                    )
                else:
                    loss, primary, _logits, _labels, structure_for_update = self._graph_forward_generalized(
                        model=model,
                        data=data,
                        device=device,
                    )

            if self.constraint_weight > 0:
                loss = loss + self.constraint_weight * self._orthogonality_constraint()

            loss.backward()
            optimizer.step()

            if self.update_structure_every_step:
                self.prompt.update_StructureToken_weight(structure_for_update)

            total_loss += float(loss.item())
            total_primary += float(primary)
            num_batches += 1

        if num_batches == 0:
            # All batches were skipped (e.g. every graph had empty train
            # masks).  This is legitimate for GPPT node-level tasks where
            # some batches may lack labeled nodes — return gracefully.
            return 0.0, {}
        metric_name = "train_mae" if self.task_type == "regression" else "train_acc"
        return total_loss / num_batches, {metric_name: total_primary / num_batches}

    def evaluate_split(self, model, loader, device, prefix: str, mask_attr: str) -> dict[str, float]:
        model.eval()
        self.prompt.eval()

        def _forward(data, device):
            if self.task_level == "node":
                if self.single_label_classification:
                    outputs = self._node_forward(model=model, data=data, device=device, mask_attr=mask_attr)
                else:
                    outputs = self._node_forward_generalized(model=model, data=data, device=device, mask_attr=mask_attr)
                if outputs is None:
                    # Empty mask — return the skip sentinel so the helper
                    # skips this batch entirely (no loss dilution).
                    return None, None, None
                loss, _primary, logits, labels, _structure = outputs
            else:
                if self.single_label_classification:
                    loss, _primary, logits, labels, _structure = self._graph_forward(
                        model=model, data=data, device=device,
                    )
                else:
                    loss, _primary, logits, labels, _structure = self._graph_forward_generalized(
                        model=model, data=data, device=device,
                    )
            return loss, logits, labels

        return evaluate_epoch_split(
            forward_fn=_forward,
            loader=loader,
            device=device,
            prefix=prefix,
            task_type=self.task_type,
        )
