"""Shared supervised metric helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from src.utils.supervised_loss import binary_targets_and_valid


def _classification_labels(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.detach().view(-1).cpu()
    if labels.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    if labels.dtype.is_floating_point:
        rounded = labels.round()
        if torch.allclose(labels, rounded, atol=1e-6):
            return rounded.long()
        return (labels > 0.5).long()
    return labels.long()


def _binary_targets_and_valid(labels: torch.Tensor):
    """Thin wrapper: detach/cpu then delegate to the canonical implementation."""
    return binary_targets_and_valid(
        torch.as_tensor(labels).detach().float().cpu()
    )


def _safe_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    try:
        from sklearn.metrics import f1_score  # pylint: disable=import-outside-toplevel
    except Exception:
        return {"micro_f1": float("nan"), "macro_f1": float("nan")}
    try:
        micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        return {"micro_f1": micro, "macro_f1": macro}
    except Exception:
        return {"micro_f1": float("nan"), "macro_f1": float("nan")}


def _safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score  # pylint: disable=import-outside-toplevel
    except Exception:
        return float("nan")

    try:
        unique = np.unique(y_true)
        if unique.size < 2:
            return float("nan")
        if probs.ndim == 1:
            return float(roc_auc_score(y_true, probs))
        if probs.ndim == 2 and probs.shape[1] == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        if probs.ndim == 2 and probs.shape[1] > 2:
            return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        return float("nan")
    except Exception:
        return float("nan")


def _safe_macro_multilabel_auc(
    target: np.ndarray,
    probs: np.ndarray,
    valid: np.ndarray,
) -> float:
    """Return unweighted mean ROC-AUC across evaluable multilabel tasks."""
    try:
        target_np = np.asarray(target)
        probs_np = np.asarray(probs)
        valid_np = np.asarray(valid, dtype=bool)
    except Exception:
        return float("nan")

    if target_np.ndim != 2 or probs_np.ndim != 2 or valid_np.ndim != 2:
        return float("nan")
    if target_np.shape != probs_np.shape or target_np.shape != valid_np.shape:
        return float("nan")

    auc_vals = []
    for tid in range(target_np.shape[1]):
        mask = valid_np[:, tid]
        if not np.any(mask):
            continue
        auc_task = _safe_auc(target_np[mask, tid], probs_np[mask, tid])
        if not np.isnan(auc_task):
            auc_vals.append(float(auc_task))
    return float(np.mean(auc_vals)) if auc_vals else float("nan")


def compute_supervised_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: str = "classification",
) -> Dict[str, float]:
    task = str(task_type or "classification").lower()
    logits = torch.as_tensor(logits).detach().cpu()
    labels = torch.as_tensor(labels).detach().cpu()

    if logits.numel() == 0 or labels.numel() == 0:
        return {}

    if task == "regression":
        preds = logits.view(-1).float()
        target = labels.view(-1).float()
        n = min(int(preds.numel()), int(target.numel()))
        if n <= 0:
            return {}
        preds = preds[:n]
        target = target[:n]
        valid = torch.isfinite(target)
        if not valid.any():
            return {"mse": float("nan"), "mae": float("nan")}
        if valid.all():
            diff = preds - target
            mse = float((diff * diff).mean().item())
            mae = float(diff.abs().mean().item())
        else:
            safe_target = torch.where(valid, target, torch.zeros_like(target))
            diff = preds - safe_target
            valid_f = valid.float()
            denom = valid_f.sum().clamp(min=1.0)
            mse = float(((diff * diff) * valid_f).sum().item() / float(denom.item()))
            mae = float(((diff.abs()) * valid_f).sum().item() / float(denom.item()))
        return {"mse": mse, "mae": mae}

    if (
        logits.dim() == 2
        and labels.dim() == 2
        and logits.size(0) == labels.size(0)
        and logits.size(1) == labels.size(1)
        and logits.size(1) > 1
    ):
        target, valid = _binary_targets_and_valid(labels)
        probs = torch.sigmoid(logits.float())
        preds = (probs >= 0.5).float()

        valid_f = valid.float()
        denom = valid_f.sum().clamp(min=1.0)
        acc = float((((preds == target).float() * valid_f).sum() / denom).item())

        valid_flat = valid.view(-1)
        y_true_flat = target.view(-1)[valid_flat].long().numpy() if valid_flat.any() else np.array([], dtype=np.int64)
        y_pred_flat = preds.view(-1)[valid_flat].long().numpy() if valid_flat.any() else np.array([], dtype=np.int64)
        if y_true_flat.size == 0:
            micro_f1 = float("nan")
        else:
            micro_f1 = _safe_f1(y_true_flat, y_pred_flat)["micro_f1"]

        macro_f1_vals = []
        probs_np = probs.numpy()
        target_np = target.numpy()
        valid_np = valid.numpy()
        for tid in range(target_np.shape[1]):
            mask = valid_np[:, tid]
            if not np.any(mask):
                continue
            y_t = target_np[mask, tid]
            y_p = (probs_np[mask, tid] >= 0.5).astype(np.int64)
            f1_task = _safe_f1(y_t.astype(np.int64), y_p)["macro_f1"]
            if not np.isnan(f1_task):
                macro_f1_vals.append(float(f1_task))

        macro_f1 = float(np.mean(macro_f1_vals)) if macro_f1_vals else float("nan")
        auc = _safe_macro_multilabel_auc(target=target_np, probs=probs_np, valid=valid_np)
        return {
            "acc": acc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "auc": auc,
        }

    if logits.dim() == 1:
        logits = logits.unsqueeze(-1)
    if logits.dim() != 2:
        logits = logits.view(logits.size(0), -1)

    y_true = _classification_labels(labels)
    if logits.size(1) == 2 and y_true.numel() > 0 and int(y_true.min().item()) < 0:
        uniq = torch.unique(y_true)
        if torch.all((uniq == -1) | (uniq == 1)):
            y_true = (y_true > 0).long()
    n = min(int(logits.size(0)), int(y_true.numel()))
    if n <= 0:
        return {}
    logits = logits[:n]
    y_true = y_true[:n]

    if logits.size(1) == 1:
        target, valid = _binary_targets_and_valid(labels.view(-1))
        probs_pos = torch.sigmoid(logits.view(-1).float())
        if valid.any():
            probs_pos_v = probs_pos[valid]
            y_true_v = target[valid].long()
            y_pred = (probs_pos_v >= 0.5).long()
            probs = probs_pos_v.numpy()
            y_true_np = y_true_v.numpy()
            y_pred_np = y_pred.numpy()
            acc = float((y_pred == y_true_v).float().mean().item())
            f1 = _safe_f1(y_true_np, y_pred_np)
            auc = _safe_auc(y_true_np, probs)
            return {
                "acc": acc,
                "micro_f1": float(f1["micro_f1"]),
                "macro_f1": float(f1["macro_f1"]),
                "auc": float(auc),
            }
        return {
            "acc": float("nan"),
            "micro_f1": float("nan"),
            "macro_f1": float("nan"),
            "auc": float("nan"),
        }

    probs_tensor = torch.softmax(logits, dim=-1)
    y_pred = probs_tensor.argmax(dim=-1)
    probs = probs_tensor.numpy()

    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    acc = float((y_pred == y_true).float().mean().item())
    f1 = _safe_f1(y_true_np, y_pred_np)
    auc = _safe_auc(y_true_np, probs)
    return {
        "acc": acc,
        "micro_f1": float(f1["micro_f1"]),
        "macro_f1": float(f1["macro_f1"]),
        "auc": float(auc),
    }


__all__ = ["compute_supervised_metrics"]
