"""GMoPE downstream task: prompt tuning (Stage B) and confidence aggregation (Stage C).

The experts are frozen; the prompts (owned by the model) and one task head
shared by all experts are trained with the gate-weighted per-expert task
losses plus the prompt orthogonality loss (Eq. 17). The hard top-K router
(Eq. 11) scores experts on each support batch with the label-free
pretraining objective (``finetune.route_loss='pretrain'``) or the support
task loss (``'task'``). Evaluation mixes the experts' pooled embeddings with
normalised-entropy confidence weights and applies the head (Eqs. 12-14).
Regression trains on support z-scored targets and reports raw units.
"""

from __future__ import annotations

import random

import torch
from torch import nn

from src.finetune.regression import RegressionTargetNormalizer
from src.utils.dataset_helpers import resolve_effective_task_level
from src.utils.parsing import resolve_task_type
from src.utils.supervised_loss import build_supervised_head, supervised_loss_from_logits

from .pretrain import resolve_route, resolve_top_k
from .routing import (
    aggregate_embeddings,
    confidence_weights,
    hard_topk_gate,
    scores_from_losses,
    shared_rng,
)

_ROUTE_LOSSES = {"pretrain", "task"}
_AGGREGATIONS = {"all", "routed"}


class GMoPETask(nn.Module):
    """Shared supervised head plus GMoPE routing/aggregation over a :class:`GMoPEModel`."""

    def __init__(self, cfg, *, num_experts: int, route_objectives=None):
        super().__init__()
        self.cfg = cfg
        gmope_cfg = cfg.moe.gmope
        ds_cfg = gmope_cfg.dataset

        raw_task_level = str(ds_cfg.task_level).lower()
        self.task_level = resolve_effective_task_level(raw_task_level, bool(getattr(ds_cfg, "induced", False)))
        if self.task_level == "edge":
            raise ValueError(
                "[GMoPE] Non-induced edge-level training is not supported; set "
                "moe.gmope.dataset.induced=True to use induced subgraphs."
            )
        self.task_type = resolve_task_type(getattr(ds_cfg, "task_type", None))
        self.label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        num_classes = int(getattr(ds_cfg, "num_classes", 1) or 1)
        self.multilabel = self.task_type == "classification" and self.label_dim > 1

        self.num_experts = int(num_experts)
        self.top_k = min(resolve_top_k(gmope_cfg, resolve_route(raw_task_level), "finetune"), self.num_experts)
        self.ortho_weight = float(gmope_cfg.ortho_weight)
        self.route_loss = str(gmope_cfg.finetune.route_loss).lower()
        self.aggregation = str(gmope_cfg.aggregation.experts).lower()
        if self.route_loss not in _ROUTE_LOSSES:
            raise ValueError(f"[GMoPE] finetune.route_loss must be one of {sorted(_ROUTE_LOSSES)}.")
        if self.aggregation not in _AGGREGATIONS:
            raise ValueError(f"[GMoPE] aggregation.experts must be one of {sorted(_AGGREGATIONS)}.")

        # Frozen per-expert pretraining objectives (already on the model's
        # device), kept out of the module tree so they are neither trained,
        # saved, nor switched to train mode.
        self._route_objectives = list(route_objectives or [])
        routed = self.top_k < self.num_experts
        if routed and (self.route_loss == "pretrain" or self.aggregation == "routed"):
            if len(self._route_objectives) != self.num_experts:
                raise ValueError(
                    "[GMoPE] Label-free routing needs one pretraining objective per expert "
                    f"(got {len(self._route_objectives)} for M={self.num_experts})."
                )
        for objective in self._route_objectives:
            objective.requires_grad_(False)
            objective.eval()
        self._route_rng = random.Random(int(cfg.seed))

        self.classifier = build_supervised_head(
            in_dim=int(gmope_cfg.expert.out_dim),
            task_type=self.task_type,
            task_level=self.task_level,
            label_dim=self.label_dim,
            num_classes=num_classes,
        )
        self.normalizer = RegressionTargetNormalizer(
            enabled=self.task_type == "regression",
            target_dim=self.label_dim,
            task_level=self.task_level,
        )

    def parameters_to_optimize(self):
        """Head parameters; the prompts are added by the runner."""
        return self.classifier.parameters()

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def route_scores(self, model, data, device) -> torch.Tensor:
        """Label-free Rawscore (negated pretraining loss) of every expert on this batch."""
        seed = self._route_rng.randrange(2**31)
        losses = []
        for m, objective in enumerate(self._route_objectives):
            with shared_rng(seed):
                loss, _ = objective.step(model.bound(m), data, device)
            losses.append(loss.detach().float())
        return scores_from_losses(torch.stack(losses))

    def _train_gate(self, model, data, device, losses: torch.Tensor) -> torch.Tensor:
        if self.top_k >= self.num_experts:
            return torch.full((self.num_experts,), 1.0 / self.num_experts, device=losses.device)
        if self.route_loss == "task":
            scores = scores_from_losses(losses)
        else:
            scores = self.route_scores(model, data, device)
        return hard_topk_gate(scores, self.top_k).to(losses.device)

    # ------------------------------------------------------------------ #
    # Stage B
    # ------------------------------------------------------------------ #
    def step(self, model, data, device):
        data = data.to(device)
        expert_logits = self.classifier(model.pooled_all(data))  # [M, B, out]
        targets = self.normalizer.normalize_targets(data.y)
        losses, primaries = [], []
        for m in range(self.num_experts):
            loss_m, primary_m = supervised_loss_from_logits(
                logits=expert_logits[m], labels=targets, task_type=self.task_type,
            )
            losses.append(loss_m)
            primaries.append(float(primary_m))
        losses = torch.stack(losses)
        gate = self._train_gate(model, data, device, losses)
        selected = torch.nonzero(gate > 0).view(-1)
        task_loss = (gate[selected] * losses[selected]).sum() / self.num_experts
        ortho = model.ortho_loss()
        loss = self.ortho_weight * ortho + task_loss

        primary = sum(primaries[int(m)] for m in selected) / max(1, int(selected.numel()))
        log = {
            "train_task_loss": float(task_loss.detach().item()),
            "train_ortho_loss": float(ortho.detach().item()),
        }
        log["train_mae" if self.task_type == "regression" else "train_acc"] = primary
        return loss, log

    # ------------------------------------------------------------------ #
    # Stage C
    # ------------------------------------------------------------------ #
    def predict(self, model, data, device) -> torch.Tensor:
        """Aggregated head output ``f(sum_m omega_m h_m)`` (normalised units for regression)."""
        pooled = model.pooled_all(data)
        expert_logits = self.classifier(pooled)
        omega = confidence_weights(expert_logits, task_type=self.task_type, multilabel=self.multilabel)
        if self.aggregation == "routed" and self.top_k < self.num_experts:
            mask = (hard_topk_gate(self.route_scores(model, data, device), self.top_k) > 0).to(omega.device)
            omega = omega * mask.unsqueeze(-1).to(omega.dtype)
            total = omega.sum(dim=0, keepdim=True)
            fallback = mask.unsqueeze(-1).to(omega.dtype).expand_as(omega) / float(self.top_k)
            omega = torch.where(total > 0, omega / total.clamp_min(1e-12), fallback)
        return self.classifier(aggregate_embeddings(pooled, omega))

    def evaluate(self, model, data, device, mask_attr="val_mask", return_outputs=False):
        data = data.to(device)
        logits = self.predict(model, data, device)
        if self.task_type == "regression":
            logits = self.normalizer.denormalize_predictions(logits)
        return supervised_loss_from_logits(
            logits=logits, labels=data.y, task_type=self.task_type, return_outputs=return_outputs,
        )


__all__ = ["GMoPETask"]
