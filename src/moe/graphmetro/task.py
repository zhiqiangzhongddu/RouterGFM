"""GraphMETRO objective (Eq. 3) over sampled training shifts.

Per mini-batch, ``num_shift_samples`` shifts are drawn uniformly with
replacement from the training list (official ``train_moe.py``). For each shift
``tau`` with multi-hot component target ``Y(tau)``:

* ``L1 = BCEWithLogits(phi(tau(G)), Y(tau), pos_weight)``;
* ``h = mix(xi(tau(G)), softmax(phi(tau(G))).detach())`` -- ``L2`` never reaches the gate;
* ``L2 = task_loss(mu(h), y) + align_lambda * ||h - xi_0(G)||_F / B`` with
  ``xi_0(G)`` the reference expert on the clean batch (not detached, official).

The step loss is the mean of ``L1 + L2`` over the sampled shifts. Evaluation
runs the model on the untransformed queries. The task loss is the repo's shared
supervised loss (CE / single-logit BCE / masked multi-task BCE / MSE); regression
trains on support median/MAD-normalized targets (``normalizer``, set by the
runner) and evaluates raw-unit outputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.moe.shift_eval import normalized_targets, raw_outputs
from src.utils.parsing import resolve_task_type
from src.utils.supervised_loss import supervised_loss_from_logits

from .transforms import apply_shift, expert_names, parse_shift_list, shift_target


class GraphMETROTask(nn.Module):
    """Shift sampling, the GraphMETRO training loss, and plain supervised evaluation."""

    def __init__(self, cfg):
        super().__init__()
        gm_cfg = cfg.moe.graphmetro
        ds_cfg = gm_cfg.dataset
        self.task_level_raw = str(ds_cfg.task_level).lower()
        self.task_type = resolve_task_type(getattr(ds_cfg, "task_type", None))
        self.shifts = parse_shift_list(gm_cfg.shift_train_types)
        self.expert_names = expert_names(self.shifts)
        self.shift_targets = torch.stack([shift_target(shift, self.expert_names) for shift in self.shifts])
        self.p = float(gm_cfg.shift_p)
        self.k = int(gm_cfg.shift_k)
        if not 0.0 <= self.p < 1.0 or self.k < 0:
            raise ValueError(f"[GraphMETRO] Need 0 <= shift_p < 1 and shift_k >= 0 (got {self.p}, {self.k}).")
        self.num_shift_samples = int(gm_cfg.num_shift_samples)
        if self.num_shift_samples < 1:
            raise ValueError("[GraphMETRO] num_shift_samples must be >= 1.")
        self.gate_pos_weight = float(gm_cfg.gate_pos_weight)
        self.align_lambda = float(gm_cfg.align_lambda)
        # Shift sampling and transforms draw from their own seeded stream.
        self.generator = torch.Generator().manual_seed(int(cfg.seed))
        self.normalizer = None  # support RegressionNormalizer for regression (set by the runner)

    def parameters_to_optimize(self):
        """No task-owned parameters: the classifier head lives in the model (own LR group)."""
        return []

    def sample_shifts(self) -> list[int]:
        return torch.randint(len(self.shifts), (self.num_shift_samples,), generator=self.generator).tolist()

    def shift_terms(self, model, shifted, shift_idx: int, z0: torch.Tensor, labels) -> dict:
        """Loss terms of one transformed batch (tensors; ``primary`` / ``gate_acc`` are floats)."""
        gate_logits = model.gate_logits(shifted)
        target = self.shift_targets[shift_idx].to(gate_logits.device).expand_as(gate_logits)
        pos_weight = torch.full((gate_logits.size(-1),), self.gate_pos_weight, device=gate_logits.device)
        gate_loss = F.binary_cross_entropy_with_logits(gate_logits, target, pos_weight=pos_weight)
        weights = torch.softmax(gate_logits.detach(), dim=-1)
        h = model.mix(model.expert_reprs(shifted), weights)
        task_loss, primary = supervised_loss_from_logits(logits=model.head(h), labels=labels, task_type=self.task_type)
        align = torch.linalg.norm(h - z0, "fro") / h.size(0)
        gate_acc = float(((gate_logits.detach() > 0).float() == target).float().mean())
        return {"gate": gate_loss, "task": task_loss, "align": align, "primary": primary, "gate_acc": gate_acc}

    def step(self, model, data, device):
        # Transform the CPU batch before moving it: ``Batch.to`` works in place.
        chosen = self.sample_shifts()
        shifted = [
            apply_shift(data, self.shifts[i], p=self.p, k=self.k, task_level_raw=self.task_level_raw, generator=self.generator)
            for i in chosen
        ]
        clean = data.to(device)
        z0 = model.instance_repr(model.experts[0], clean)

        loss = 0.0
        sums = {"gate": 0.0, "task": 0.0, "align": 0.0, "primary": 0.0, "gate_acc": 0.0}
        for idx, batch in zip(chosen, shifted):
            terms = self.shift_terms(model, batch.to(device), idx, z0, normalized_targets(self.normalizer, clean.y))
            loss = loss + terms["gate"] + terms["task"] + self.align_lambda * terms["align"]
            for key in sums:
                sums[key] += float(terms[key])
        count = len(chosen)
        loss = loss / count
        log = {
            "train_gate_loss": sums["gate"] / count,
            "train_task_loss": sums["task"] / count,
            "train_align_loss": sums["align"] / count,
            "train_gate_acc": sums["gate_acc"] / count,
            ("train_mae" if self.task_type == "regression" else "train_acc"): sums["primary"] / count,
        }
        return loss, log

    def evaluate(self, model, data, device, mask_attr="val_mask", return_outputs=False):
        data = data.to(device)
        logits, _ = model(data)
        return supervised_loss_from_logits(
            logits=raw_outputs(self.normalizer, logits), labels=data.y, task_type=self.task_type,
            return_outputs=return_outputs,
        )


__all__ = ["GraphMETROTask"]
