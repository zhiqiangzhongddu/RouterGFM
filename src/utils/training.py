"""Shared training utilities used by pretrain, train, and finetune runners."""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def build_lr_scheduler(
    optimizer,
    scheduler_name: str = "none",
    epochs: int = 500,
    step_size: int = 50,
    gamma: float = 0.5,
):
    """Build a learning-rate scheduler from config-style parameters.

    Returns ``None`` when *scheduler_name* is ``"none"`` or unrecognised.
    """
    name = str(scheduler_name).lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return StepLR(optimizer, step_size=int(step_size), gamma=float(gamma))
    return None


def run_step_epoch(
    model,
    task,
    loader,
    optimizer,
    device,
    grad_clip: float = 0.0,
    skip_model_train: bool = False,
) -> tuple[float, dict[str, float]]:
    """Run one step-based training epoch.

    Iterates *loader*, calling ``task.step(model, data, device)`` for each
    batch.  Returns ``(avg_loss, avg_logs)``.

    When *skip_model_train* is ``True`` the function does **not** call
    ``model.train()`` before the epoch.  This allows the caller (e.g.
    ``FinetuneRunner``) to apply a frozen-encoder mode policy first
    without it being immediately overridden.
    """
    if not skip_model_train:
        model.train()
    task.train()
    total_loss = 0.0
    logs: dict[str, float] = {}

    task_params = (
        list(task.parameters_to_optimize())
        if hasattr(task, "parameters_to_optimize")
        else list(task.parameters())
    )
    all_params = list(model.parameters()) + task_params

    for data in loader:
        optimizer.zero_grad()
        loss, log = task.step(model=model, data=data, device=device)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        for k, v in log.items():
            logs[k] = logs.get(k, 0.0) + float(v)

    num_batches = len(loader)
    if num_batches == 0:
        raise RuntimeError("Train loader is empty; unable to run a training epoch.")
    avg_loss = total_loss / num_batches
    logs = {k: v / num_batches for k, v in logs.items()}
    return avg_loss, logs


def run_epoch_loop(
    forward_fn,
    loader,
    optimizer,
    device,
    grad_clip: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """Run one epoch-based training loop.

    This is the epoch-based counterpart of :func:`run_step_epoch` for
    finetune prompt methods that own the forward logic but want shared
    loop mechanics (zero_grad, backward, step, averaging, grad clip).

    *forward_fn* signature: ``(data, device) -> (loss, log_dict)``
    where *loss* is a scalar tensor and *log_dict* maps metric names
    to scalar values for the batch.
    """
    total_loss = 0.0
    logs: dict[str, float] = {}

    for data in loader:
        optimizer.zero_grad()
        loss, log = forward_fn(data, device)
        loss.backward()
        if grad_clip > 0.0:
            params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        for k, v in log.items():
            logs[k] = logs.get(k, 0.0) + float(v)

    num_batches = len(loader)
    if num_batches == 0:
        raise RuntimeError("Train loader is empty; unable to run a training epoch.")
    avg_loss = total_loss / num_batches
    logs = {k: v / num_batches for k, v in logs.items()}
    return avg_loss, logs
