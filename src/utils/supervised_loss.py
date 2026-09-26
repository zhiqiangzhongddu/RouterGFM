"""Shared supervised loss and head-sizing helpers.

Used by pretrain, train, and finetune supervised workflows to ensure
consistent loss computation and output-dimension resolution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ #
# Output-dimension resolution
# ------------------------------------------------------------------ #

def resolve_supervised_output_dim(
    *,
    task_type: str,
    task_level: str | None = None,
    label_dim: int,
    num_classes: int | None,
) -> int:
    """Decide how many logits the supervised head should produce.

    This is the **single canonical resolver** used by pretrain, train, and
    finetune workflows. It replaces the former ``resolve_task_output_dim``
    in ``src.finetune.task_heads`` (which is now a thin delegation).

    When *task_level* is provided and is ``"graph"`` or ``"edge"`` with
    ``num_classes <= 2``, a single-logit BCE head is used (matching the
    official pretrain-gnns implementation).

    When *task_level* is omitted (``None``), the function preserves
    backward compatibility for prompt-based finetune methods that do not
    pass a task level: it returns ``max(2, num_classes)`` instead of raw
    ``num_classes``, avoiding single-logit heads in contexts where the
    caller has not explicitly opted in.
    """
    task_type = str(task_type or "classification").lower()
    label_dim = max(1, int(label_dim or 1))
    num_classes = max(1, int(num_classes or 1))

    if task_type == "regression":
        return label_dim
    if label_dim > 1:
        # Multi-task binary classification (e.g., MoleculeNet).
        return label_dim
    if task_level is not None:
        resolved_level = str(task_level).lower()
        if resolved_level in {"graph", "edge"} and num_classes <= 2:
            # Binary graph/edge: single-logit BCE (official behaviour).
            return 1
        return num_classes
    # task_level=None: backward-compat path for prompt-based finetune
    # methods that don't pass a task level.
    return max(2, num_classes)


def build_supervised_head(
    *,
    in_dim: int,
    task_type: str,
    task_level: str,
    label_dim: int,
    num_classes: int,
) -> torch.nn.Linear:
    """Build a linear classifier head for supervised pretrain/train tasks.

    Shared by ``src.pretrain.methods.supervised`` and ``src.train.methods.supervised``
    to avoid duplicating the output-dimension resolution + Linear
    construction.
    """
    out_dim = resolve_supervised_output_dim(
        task_type=task_type,
        task_level=task_level,
        label_dim=label_dim,
        num_classes=num_classes,
    )
    return torch.nn.Linear(in_features=int(in_dim), out_features=out_dim)


# ------------------------------------------------------------------ #
# Label helpers
# ------------------------------------------------------------------ #

def binary_targets_and_valid(
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert labels to float targets in [0, 1] and a validity mask.

    Handles three label conventions:
    * {0, 1}          — standard binary
    * {-1, 1}         — signed encoding (0 means missing)
    * {-1, 0, 1}      — signed with explicit missing

    This is the canonical implementation; ``src.finetune.task_heads`` and
    ``src.utils.metrics`` import from here.
    """
    target = labels.float()
    valid = torch.isfinite(target)
    uses_signed = bool((target < 0).any().item()) if target.numel() else False
    if uses_signed:
        valid = valid & (target != 0)
        target = (target + 1.0) / 2.0
    target = target.clamp(min=0.0, max=1.0)
    return target, valid


def prepare_class_labels(labels: torch.Tensor) -> torch.Tensor:
    """Flatten and convert labels to a 1-D long tensor for CE loss."""
    cls = labels
    if cls.dim() > 1:
        cls = cls.view(cls.size(0), -1)[:, 0]
    else:
        cls = cls.view(-1)
    if cls.dtype.is_floating_point:
        rounded = cls.round()
        if torch.allclose(cls, rounded, atol=1e-6):
            cls = rounded
        else:
            cls = (cls > 0.5).to(cls.dtype)
    return cls.long()


# ------------------------------------------------------------------ #
# Loss computation
# ------------------------------------------------------------------ #

def supervised_loss_from_logits(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: str,
    return_outputs: bool = False,
):
    """Compute supervised loss and primary metric from raw logits.

    Dispatches to the correct loss based on *task_type* and logit shape:

    * **Regression** → MSE loss, MAE metric.
    * **Multi-task binary** (logits 2-D with cols > 1 and labels match)
      → per-element BCE with validity masking, accuracy metric.
    * **Single-logit binary** (logits shape ``[N, 1]`` or ``[N]``)
      → BCE with logits, accuracy metric.
    * **Multi-class** (logits shape ``[N, C]`` with C ≥ 2)
      → cross-entropy, accuracy metric.

    Returns
    -------
    (loss, metric)  when *return_outputs* is False
    (loss, metric, logits_out, labels_out)  when *return_outputs* is True
    """
    task_type = str(task_type or "classification").lower()

    # -- regression ------------------------------------------------ #
    if task_type == "regression":
        pred = logits.view(-1).float()
        target = torch.as_tensor(labels).view(-1).float()
        valid = torch.isfinite(target)
        if valid.all():
            loss = F.mse_loss(pred, target)
            mae = float((pred - target).abs().mean().item())
        elif valid.any():
            safe_target = torch.where(valid, target, torch.zeros_like(target))
            diff = pred - safe_target
            valid_f = valid.float()
            denom = valid_f.sum().clamp(min=1.0)
            loss = ((diff * diff) * valid_f).sum() / denom
            mae = float(((diff.abs() * valid_f).sum() / denom).item())
        else:
            loss = pred.sum() * 0.0
            mae = 0.0
        if return_outputs:
            return loss, mae, pred, target
        return loss, mae

    labels_tensor = torch.as_tensor(labels)

    # -- multi-task binary (label_dim > 1) ------------------------- #
    if labels_tensor.dim() > 1 and labels_tensor.size(-1) > 1:
        targets = labels_tensor.float()
        targets, valid = binary_targets_and_valid(targets)
        safe_targets = torch.where(valid, targets, torch.zeros_like(targets))

        loss_mat = F.binary_cross_entropy_with_logits(
            logits.float(),
            safe_targets,
            reduction="none",
        )
        valid_f = valid.float()
        denom = valid_f.sum().clamp(min=1.0)
        loss = (loss_mat * valid_f).sum() / denom

        probs = torch.sigmoid(logits.float())
        pred = (probs >= 0.5).float()
        acc = float(((pred == targets).float() * valid_f).sum().item() / float(denom.item()))
        if return_outputs:
            return loss, acc, logits, labels_tensor
        return loss, acc

    # -- single-logit binary --------------------------------------- #
    if logits.dim() == 1 or (logits.dim() == 2 and logits.size(-1) == 1):
        logits_vec = logits.view(-1).float()
        targets = labels_tensor.view(-1).float()
        targets, valid = binary_targets_and_valid(targets)
        if valid.any():
            loss = F.binary_cross_entropy_with_logits(logits_vec[valid], targets[valid])
            pred = (torch.sigmoid(logits_vec[valid]) >= 0.5).long()
            truth = targets[valid].long()
            acc = float((pred == truth).float().mean().item())
        else:
            loss = logits_vec.sum() * 0.0
            acc = 0.0
        if return_outputs:
            return loss, acc, logits_vec.unsqueeze(-1), labels_tensor.view(-1)
        return loss, acc

    # -- multi-class ----------------------------------------------- #
    class_labels = prepare_class_labels(labels_tensor)
    loss = F.cross_entropy(logits, class_labels)
    pred = logits.argmax(dim=-1)
    acc = float((pred == class_labels).float().mean().item())
    if return_outputs:
        return loss, acc, logits, class_labels
    return loss, acc


__all__ = [
    "binary_targets_and_valid",
    "build_supervised_head",
    "prepare_class_labels",
    "resolve_supervised_output_dim",
    "supervised_loss_from_logits",
]
