"""Shared task-aware prediction helpers for finetuning methods."""

from __future__ import annotations

import torch
from torch import nn

from src.finetune.multilabel import (
    MACRO_BALANCED_BCE,
    MacroBalancedBCELoss,
    resolve_multilabel_loss,
)
from src.finetune.regression import (
    METRIC_MAE,
    RegressionTargetNormalizer,
    resolve_regression_loss,
    resolve_regression_target_normalization,
)
from src.utils.dataset_helpers import normalize_node_mask, read_effective_task_level
from src.utils.pool import get_batch_vector, pool_nodes
from src.utils.parsing import resolve_task_type
from src.utils.supervised_loss import (
    prepare_class_labels,
    resolve_supervised_output_dim,
    supervised_loss_from_logits,
)


# ---------------------------------------------------------------------------
# Shared tensor helpers (used by multiple finetune methods)
# ---------------------------------------------------------------------------

def align_last_dim(x: torch.Tensor, target_dim: int | None) -> torch.Tensor:
    """Pad or truncate the last dimension of *x* to *target_dim*."""
    if target_dim is None:
        return x
    current_dim = int(x.size(-1))
    if current_dim == target_dim:
        return x
    if current_dim > target_dim:
        return x[:, :target_dim]
    pad = x.new_zeros((x.size(0), target_dim - current_dim))
    return torch.cat([x, pad], dim=-1)


def prepare_single_label_labels(labels: torch.Tensor) -> torch.Tensor:
    """Flatten multi-dim labels to a 1-D long tensor for single-label classification.

    Delegates to the canonical ``prepare_class_labels`` from
    ``src.utils.supervised_loss`` to avoid duplication.
    """
    return prepare_class_labels(torch.as_tensor(labels))


def build_task_aware_classifier(
    *,
    input_dim: int,
    task_type: str,
    label_dim: int,
    num_classes: int | None,
    task_level: str | None = None,
) -> nn.Linear:
    """Construct a linear head sized for the finetune task.

    Delegates to :func:`resolve_supervised_output_dim` from
    ``src.utils.supervised_loss`` for output-dim resolution.  When
    *task_level* is omitted the backward-compat ``max(2, num_classes)``
    path is used for prompt-based methods that don't declare a level.
    """
    return nn.Linear(
        in_features=int(input_dim),
        out_features=resolve_supervised_output_dim(
            task_type=task_type,
            task_level=task_level,
            label_dim=label_dim,
            num_classes=num_classes,
        ),
    )


class TaskAwareObjective(nn.Module):
    """Task-aware label shaping, loss computation, and model-output selection."""

    def __init__(
        self,
        cfg,
        *,
        task_level: str | None = None,
        task_type: str | None = None,
        label_dim: int | None = None,
        num_classes: int | None = None,
        repr_dim: int | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        ds_cfg = getattr(getattr(cfg, "finetune", None), "dataset", None) or getattr(cfg, "dataset", None)
        # Prefer the effective task level (set by the runner after induced
        # promotion) over the raw requested level.
        self.task_level = str(task_level or read_effective_task_level(ds_cfg)).lower()
        self.task_type = resolve_task_type(task_type or getattr(ds_cfg, "task_type", None))
        self.label_dim = max(1, int(label_dim if label_dim is not None else (getattr(ds_cfg, "label_dim", 1) or 1)))
        raw_num_classes = num_classes if num_classes is not None else getattr(ds_cfg, "num_classes", None)
        self.num_classes = None if raw_num_classes in (None, "") else int(raw_num_classes)
        self.repr_dim = int(repr_dim) if repr_dim is not None else None
        # Endpoint readout on induced edge tasks (finetune.edge_readout).
        # Gated on the RAW task level because induced promotion rewrites
        # edge -> graph before this objective is built; on any other task
        # shape the setting is inert. Default "pool" keeps every existing
        # run byte-identical.
        edge_readout = str(
            getattr(getattr(cfg, "finetune", None), "edge_readout", "pool") or "pool"
        ).lower()
        if edge_readout not in {"pool", "endpoints"}:
            raise ValueError(
                f"finetune.edge_readout must be pool or endpoints, got {edge_readout!r}"
            )
        raw_level = str(getattr(ds_cfg, "task_level", "") or "").lower()
        self.edge_endpoint_readout = edge_readout == "endpoints" and raw_level == "edge"
        self.output_dim = resolve_supervised_output_dim(
            task_type=self.task_type,
            task_level=self.task_level,
            label_dim=self.label_dim,
            num_classes=self.num_classes,
        )
        self.regression_loss_mode = resolve_regression_loss(cfg)
        regression_normalization_enabled = (
            self.task_type == "regression"
            and resolve_regression_target_normalization(cfg)
        )
        if (
            self.task_type == "regression"
            and self.regression_loss_mode == METRIC_MAE
            and not regression_normalization_enabled
        ):
            raise ValueError(
                "finetune.regression_loss=metric_mae requires "
                "finetune.normalize_regression_targets=True."
            )
        self.target_normalizer = RegressionTargetNormalizer(
            enabled=regression_normalization_enabled,
            target_dim=self.label_dim,
            task_level=self.task_level,
            loss_mode=self.regression_loss_mode,
        )
        self.multilabel_loss_mode = resolve_multilabel_loss(cfg)
        self.multilabel_balancer = MacroBalancedBCELoss(
            enabled=(
                self.is_multilabel_classification
                and self.multilabel_loss_mode == MACRO_BALANCED_BCE
            ),
            target_dim=self.label_dim,
            task_level=self.task_level,
        )

    @property
    def is_single_label_classification(self) -> bool:
        return self.task_type == "classification" and self.label_dim <= 1

    @property
    def is_multilabel_classification(self) -> bool:
        return self.task_type == "classification" and self.label_dim > 1

    @property
    def primary_metric_name(self) -> str:
        return "mae" if self.task_type == "regression" else "acc"

    @staticmethod
    def _reshape_targets_with_batch_size(labels: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(labels)
        if tensor.dim() == 0:
            tensor = tensor.view(1, 1)
        elif tensor.dim() == 1:
            if batch_size > 0 and tensor.numel() % batch_size == 0:
                tensor = tensor.view(batch_size, -1)
            else:
                tensor = tensor.view(-1, 1)
        else:
            if tensor.size(0) != batch_size and batch_size > 0 and tensor.numel() % batch_size == 0:
                tensor = tensor.view(batch_size, -1)
            elif tensor.size(0) == batch_size:
                tensor = tensor.view(batch_size, -1)
        if tensor.size(0) != batch_size:
            raise ValueError(
                f"Unable to align {name} with batch size {batch_size}: got shape {tuple(tensor.shape)}."
            )
        return tensor


    def _prepare_single_label_targets(self, labels: torch.Tensor, batch_size: int) -> torch.Tensor:
        tensor = torch.as_tensor(labels)
        if tensor.dim() > 1:
            tensor = self._reshape_targets_with_batch_size(tensor, batch_size, "labels")[:, 0]
        else:
            tensor = tensor.view(-1)
        if tensor.numel() != batch_size:
            raise ValueError(
                f"Expected {batch_size} single-label targets, got {tensor.numel()} elements."
            )
        return prepare_class_labels(tensor)

    def select_representations_and_labels(
        self,
        *,
        node_repr: torch.Tensor,
        graph_repr: torch.Tensor | None,
        data,
        device: torch.device,
        mask_attr: str = "train_mask",
        graph_pooling_mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_repr = align_last_dim(node_repr, self.repr_dim)
        if graph_repr is not None:
            graph_repr = align_last_dim(graph_repr, self.repr_dim)

        if self.task_level == "node":
            mask = normalize_node_mask(data, mask_attr, device)
            return node_repr[mask], torch.as_tensor(data.y)[mask]

        if self.edge_endpoint_readout:
            edge_label_index = getattr(data, "edge_label_index", None)
            if edge_label_index is None:
                raise ValueError(
                    "finetune.edge_readout=endpoints requires data.edge_label_index "
                    "(induced edge subgraphs); this batch has none."
                )
            eli = torch.as_tensor(edge_label_index, device=node_repr.device).view(2, -1)
            edge_repr = node_repr[eli[0]] * node_repr[eli[1]]
            return edge_repr, torch.as_tensor(data.y)

        batch = get_batch_vector(data)
        if graph_repr is None:
            graph_repr = pool_nodes(
                x=node_repr,
                batch=batch,
                mode=graph_pooling_mode or self.cfg.model.graph_pooling,
            )
        graph_repr = align_last_dim(graph_repr, self.repr_dim)
        return graph_repr, torch.as_tensor(data.y)

    def loss_from_logits(
        self,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        return_outputs: bool = False,
    ):
        # Shape normalization is applied before delegating to the canonical
        # ``supervised_loss_from_logits`` from ``src.utils.supervised_loss``
        # so the shared function receives well-formed tensors.
        logits_mat = self._reshape_targets_with_batch_size(logits, int(logits.size(0)), "logits")

        if self.task_type == "regression":
            logits_flat = logits_mat.view(-1)
            labels_mat = self._reshape_targets_with_batch_size(
                torch.as_tensor(labels, device=logits_mat.device).float(),
                int(logits.size(0)),
                "labels",
            )
            labels_for_loss = self.target_normalizer.normalize_targets(labels_mat).view(-1)
            loss, normalized_mae = supervised_loss_from_logits(
                logits=logits_flat,
                labels=labels_for_loss,
                task_type="regression",
                return_outputs=False,
            )
            if self.regression_loss_mode == METRIC_MAE:
                loss = self.target_normalizer.metric_aligned_mae_loss(
                    logits_mat,
                    labels_for_loss,
                )
            if not self.target_normalizer.active:
                if return_outputs:
                    return loss, normalized_mae, logits_flat, labels_mat.view(-1)
                return loss, normalized_mae

            # The head predicts normalized targets for optimization.  Expose
            # predictions and the primary MAE only on the original target
            # scale so train/validation/test metrics remain comparable with
            # unnormalized experiments and published baselines.
            original_logits = self.target_normalizer.denormalize_predictions(
                logits_mat
            ).view(-1)
            original_labels = labels_mat.view(-1)
            valid = torch.isfinite(original_labels)
            if valid.any():
                original_mae = float(
                    (original_logits[valid] - original_labels[valid]).abs().mean().item()
                )
            else:
                original_mae = 0.0
            if return_outputs:
                return loss, original_mae, original_logits, original_labels
            return loss, original_mae

        # Classification paths: multilabel/binary/multiclass dispatch.
        labels_tensor = torch.as_tensor(labels, device=logits_mat.device)
        if self.is_multilabel_classification:
            labels_tensor = self._reshape_targets_with_batch_size(labels_tensor, int(logits_mat.size(0)), "labels")
        elif logits_mat.size(1) == 1:
            labels_tensor = self._reshape_targets_with_batch_size(labels_tensor, int(logits_mat.size(0)), "labels").view(-1)
        else:
            labels_tensor = self._prepare_single_label_targets(labels_tensor, int(logits_mat.size(0)))
        legacy_result = supervised_loss_from_logits(
            logits=logits_mat,
            labels=labels_tensor,
            task_type="classification",
            return_outputs=return_outputs,
        )
        if not (
            self.is_multilabel_classification
            and self.multilabel_loss_mode == MACRO_BALANCED_BCE
        ):
            # Preserve the historical masked-BCE path byte-for-byte for the
            # default objective and for every non-multilabel task.
            return legacy_result

        balanced_loss = self.multilabel_balancer(logits_mat, labels_tensor)
        balanced_targets, balanced_valid = self.multilabel_balancer.targets_and_valid(
            labels_tensor
        )
        balanced_targets = balanced_targets.to(logits_mat.device)
        balanced_valid = balanced_valid.to(logits_mat.device)
        balanced_predictions = (torch.sigmoid(logits_mat.float()) >= 0.5).float()
        valid_count = balanced_valid.sum()
        if bool(valid_count.item()):
            balanced_acc = float(
                (
                    (balanced_predictions == balanced_targets).float()
                    * balanced_valid.float()
                ).sum().item()
                / float(valid_count.item())
            )
        else:
            balanced_acc = 0.0
        if return_outputs:
            return balanced_loss, balanced_acc, *legacy_result[2:]
        return balanced_loss, balanced_acc

    def forward_with_classifier(
        self,
        *,
        classifier: nn.Module,
        representations: torch.Tensor,
        labels: torch.Tensor,
        input_dim: int | None = None,
        return_outputs: bool = False,
    ):
        reps = align_last_dim(representations, input_dim or self.repr_dim)
        logits = classifier(reps)
        return self.loss_from_logits(logits=logits, labels=labels, return_outputs=return_outputs)

    def forward_with_model_outputs(
        self,
        *,
        classifier: nn.Module,
        node_repr: torch.Tensor,
        graph_repr: torch.Tensor | None,
        data,
        device: torch.device,
        mask_attr: str = "train_mask",
        graph_pooling_mode: str | None = None,
        return_outputs: bool = False,
    ):
        representations, labels = self.select_representations_and_labels(
            node_repr=node_repr,
            graph_repr=graph_repr,
            data=data,
            device=device,
            mask_attr=mask_attr,
            graph_pooling_mode=graph_pooling_mode,
        )
        return self.forward_with_classifier(
            classifier=classifier,
            representations=representations,
            labels=labels,
            input_dim=getattr(classifier, "in_features", self.repr_dim),
            return_outputs=return_outputs,
        )


__all__ = [
    "TaskAwareObjective",
    "align_last_dim",
    "build_task_aware_classifier",
    "prepare_single_label_labels",
]
