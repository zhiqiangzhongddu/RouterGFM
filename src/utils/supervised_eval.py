"""Shared supervised split-evaluation loop.

Pretrain (``src/pretrain/trainer.py``), Train (``src/train/trainer.py``),
and Finetune (``src/finetune/finetuner.py``) all implement the same
evaluation pattern: run the task's forward step in ``torch.no_grad()``,
buffer per-batch logits / labels on CPU, concatenate, and compute
``compute_supervised_metrics`` with the workflow's resolved
``task_type``. This module factors that loop out so the three workflows
share a single implementation.

The per-workflow variation is the task method that produces
``(loss, primary, logits, labels)``: pretrain uses ``task.evaluate``,
train and finetune use ``task._forward``. The helper takes a plain
callable to keep the call-site contract explicit and avoid forcing a
method rename on the three task base classes.
"""

from __future__ import annotations

from typing import Callable

import torch

from .metrics import compute_supervised_metrics


StepFn = Callable[..., tuple]


def concat_and_compute_metrics(
    logits_buffer: list[torch.Tensor],
    labels_buffer: list[torch.Tensor],
    task_type: str,
    prefix: str,
) -> dict[str, float]:
    """Concatenate buffered logits/labels and compute prefixed metrics."""
    if not logits_buffer or not labels_buffer:
        return {}
    logits_cat = torch.cat(
        [x.view(-1, 1) if x.dim() == 1 else x.view(x.size(0), -1) for x in logits_buffer],
        dim=0,
    )
    if any(y.dim() > 1 for y in labels_buffer):
        labels_cat = torch.cat(
            [y.view(-1, 1) if y.dim() == 1 else y.view(y.size(0), -1) for y in labels_buffer],
            dim=0,
        )
        if labels_cat.dim() == 2 and labels_cat.size(1) == 1:
            labels_cat = labels_cat.view(-1)
    else:
        labels_cat = torch.cat([y.view(-1) for y in labels_buffer], dim=0)
    return {
        f"{prefix}_{key}": float(value)
        for key, value in compute_supervised_metrics(
            logits=logits_cat, labels=labels_cat, task_type=task_type,
        ).items()
    }


def evaluate_supervised_split(
    *,
    model: torch.nn.Module,
    task: torch.nn.Module,
    step_fn: StepFn,
    loader,
    device: torch.device,
    prefix: str,
    mask_attr: str,
    task_type: str,
) -> dict[str, float]:
    """Run a supervised split evaluation and return prefixed metrics.

    ``step_fn`` must accept ``(model, data, device, mask_attr, return_outputs=True)``
    and return ``(loss, primary, logits, labels)``.

    Returns an empty dict when the loader has no batches, so callers can
    treat "no split" and "empty split" uniformly.
    """
    model.eval()
    task.eval()
    total_loss = 0.0
    num_batches = 0
    logits_buffer: list[torch.Tensor] = []
    labels_buffer: list[torch.Tensor] = []

    with torch.no_grad():
        for data in loader:
            loss, _primary, logits, labels = step_fn(
                model=model,
                data=data,
                device=device,
                mask_attr=mask_attr,
                return_outputs=True,
            )
            total_loss += loss.item()
            num_batches += 1
            if logits is not None and labels is not None:
                logits_buffer.append(torch.as_tensor(logits).detach().cpu())
                labels_buffer.append(torch.as_tensor(labels).detach().cpu())

    if num_batches == 0:
        return {}

    metrics: dict[str, float] = {
        f"{prefix}_loss": total_loss / num_batches,
    }
    metrics.update(concat_and_compute_metrics(logits_buffer, labels_buffer, task_type, prefix))
    return metrics


def evaluate_epoch_split(
    *,
    forward_fn: Callable[..., tuple],
    loader,
    device: torch.device,
    prefix: str,
    task_type: str,
) -> dict[str, float]:
    """Evaluate a split using a simple forward callable.

    This is the epoch-based counterpart of :func:`evaluate_supervised_split`.
    Prompt-based finetune methods that own their forward logic use this
    helper to avoid duplicating the logits/labels buffering and metric
    computation loop.

    *forward_fn* signature: ``(data, device) -> (loss, logits, labels)``
    where *logits* and *labels* are CPU tensors (or ``None`` to skip a
    batch).
    """
    total_loss = 0.0
    num_batches = 0
    logits_buffer: list[torch.Tensor] = []
    labels_buffer: list[torch.Tensor] = []

    with torch.no_grad():
        for data in loader:
            loss, logits, labels = forward_fn(data, device)
            # Skip the entire batch when the forward signals "nothing to
            # evaluate" (e.g. GPPT with an empty node mask).  Returning
            # ``(None, None, None)`` is the canonical skip sentinel.
            if loss is None:
                continue
            total_loss += float(loss)
            num_batches += 1
            if logits is not None and labels is not None:
                logits_buffer.append(torch.as_tensor(logits).detach().cpu())
                labels_buffer.append(torch.as_tensor(labels).detach().cpu())

    if num_batches == 0:
        return {}

    metrics: dict[str, float] = {f"{prefix}_loss": total_loss / num_batches}
    metrics.update(concat_and_compute_metrics(logits_buffer, labels_buffer, task_type, prefix))
    return metrics


def runner_evaluate_split(
    *,
    model: torch.nn.Module,
    task: torch.nn.Module,
    loader,
    device: torch.device,
    prefix: str,
    mask_attr: str,
    task_type: str,
) -> dict[str, float]:
    """Shared split-evaluation helper for pretrain/train/finetune runners.

    Delegates to :func:`evaluate_supervised_split` using ``task.evaluate``
    as the step function.  This eliminates the identical one-liner
    ``_evaluate_split`` wrappers across all three runners.
    """
    return evaluate_supervised_split(
        model=model,
        task=task,
        step_fn=task.evaluate,
        loader=loader,
        device=device,
        prefix=prefix,
        mask_attr=mask_attr,
        task_type=task_type,
    )


__all__ = [
    "concat_and_compute_metrics",
    "evaluate_supervised_split",
    "evaluate_epoch_split",
    "runner_evaluate_split",
]
