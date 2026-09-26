"""Supervised Mowst task: dual expert heads + routing gate.

Unlike GMoE/GraphMoRE (which mix expert *embeddings* and feed a single shared
head), Mowst mixes expert *predictions*: each expert has its own supervised head
and the gate routes per sample by the dispersion of the weak expert's
prediction. The task therefore owns two heads (``weak_head`` / ``strong_head``)
and the gate, and computes the mixture loss itself.

It still reuses the repo's shared supervised plumbing for everything else:
``select_supervised_logits_and_labels`` selects per-task-level representations
(node mask / edge endpoints / graph pooling), ``build_supervised_head`` sizes
the heads, and ``supervised_loss_from_logits`` computes the loss + primary
metric — so Mowst supports node / edge / graph levels and classification /
regression / multi-task-binary task types identically to the standard path, and
its evaluation flows through ``runner_evaluate_split`` unchanged.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F

from src.utils.dataset_helpers import resolve_effective_task_level
from src.utils.parsing import resolve_task_type
from src.utils.supervised_forward import select_supervised_logits_and_labels
from src.utils.supervised_loss import (
    binary_targets_and_valid,
    build_supervised_head,
    prepare_class_labels,
    supervised_loss_from_logits,
)

from .gating import GateMLP, compute_gating, gate_input_dim


class MowstTask(nn.Module):
    """Weak/strong dual-head supervised task with a dispersion-routed gate."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        m_cfg = cfg.moe.mowst
        ds_cfg = m_cfg.dataset

        raw_task_level = str(ds_cfg.task_level).lower()
        induced = bool(getattr(ds_cfg, "induced", False))
        # Induced node/edge datasets are materialised as graph batches; the
        # heads then use graph-level logic. Mirrors GMoETask / TrainSupervised.
        self.task_level = resolve_effective_task_level(raw_task_level, induced)
        if self.task_level == "edge":
            raise ValueError(
                "[Mowst] Non-induced edge-level training is not supported: "
                "select_supervised_logits_and_labels requires "
                "data.edge_label_index/edge_label which full-graph datasets "
                "don't provide. Set moe.mowst.dataset.induced=True to use "
                "induced subgraphs."
            )

        self.task_type = resolve_task_type(getattr(ds_cfg, "task_type", None))
        self.label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        num_classes = int(getattr(ds_cfg, "num_classes", 1) or 1)
        self.pool_mode = str(m_cfg.graph_pooling)
        self.variant = str(m_cfg.variant).strip().lower()
        self.subloss = "separate" if self.variant == "mowst" else str(m_cfg.subloss).strip().lower()
        self.original_data = bool(m_cfg.original_data)

        hidden_dim = int(m_cfg.hidden_dim)
        self.weak_head = build_supervised_head(
            in_dim=hidden_dim,
            task_type=self.task_type,
            task_level=self.task_level,
            label_dim=self.label_dim,
            num_classes=num_classes,
        )
        self.strong_head = build_supervised_head(
            in_dim=hidden_dim,
            task_type=self.task_type,
            task_level=self.task_level,
            label_dim=self.label_dim,
            num_classes=num_classes,
        )
        self.gate = GateMLP(
            in_dim=gate_input_dim(feature_dim=hidden_dim, original_data=self.original_data),
            hidden_dim=int(m_cfg.gate.hidden_dim),
            num_layers=int(m_cfg.gate.num_layers),
            dropout=float(m_cfg.gate.dropout),
            act=str(m_cfg.activation),
            use_batchnorm=bool(m_cfg.use_batchnorm),
        )
        self._identity = nn.Identity()

    # ------------------------------------------------------------------ #
    # Parameter groups
    # ------------------------------------------------------------------ #
    def parameters_to_optimize(self):
        """All task params (both heads + gate); encoder params added by runner."""
        return self.parameters()

    def weak_branch_parameters(self):
        """Weak head + gate params (the weak turn / joint-weak optimizer side)."""
        return list(self.weak_head.parameters()) + list(self.gate.parameters())

    def strong_branch_parameters(self):
        """Strong head params (the strong turn optimizer side)."""
        return list(self.strong_head.parameters())

    # ------------------------------------------------------------------ #
    # Representation selection (shared per-task-level plumbing)
    # ------------------------------------------------------------------ #
    def _select_rep(self, node_repr, graph_repr, data, mask_attr, device):
        return select_supervised_logits_and_labels(
            node_repr=node_repr,
            graph_repr=graph_repr,
            data=data,
            classifier=self._identity,
            task_level=self.task_level,
            pool_mode=self.pool_mode,
            mask_attr=mask_attr,
            device=device,
        )

    def _gated_forward(
        self,
        model,
        data,
        device,
        mask_attr,
        *,
        weak_no_grad: bool = False,
        strong_no_grad: bool = False,
        gate_no_grad: bool = False,
    ):
        """Return ``(weak_logits, strong_logits, gating, labels)``.

        The ``*_no_grad`` flags freeze a branch (used by the alternating ``mowst``
        turns); evaluation leaves them all False and relies on the outer
        ``torch.no_grad()`` in ``runner_evaluate_split``.
        """
        data = data.to(device)

        with torch.no_grad() if weak_no_grad else nullcontext():
            w_node, w_graph = model.forward_weak(data)
            w_rep, labels = self._select_rep(w_node, w_graph, data, mask_attr, device)
            weak_logits = self.weak_head(w_rep)

        with torch.no_grad() if strong_no_grad else nullcontext():
            s_node, s_graph = model.forward_strong(data)
            s_rep, _ = self._select_rep(s_node, s_graph, data, mask_attr, device)
            strong_logits = self.strong_head(s_rep)

        with torch.no_grad() if gate_no_grad else nullcontext():
            gating = compute_gating(
                self.gate,
                weak_logits,
                w_rep,
                task_type=self.task_type,
                label_dim=self.label_dim,
                original_data=self.original_data,
            )
        return weak_logits, strong_logits, gating, labels

    # ------------------------------------------------------------------ #
    # Losses
    # ------------------------------------------------------------------ #
    def _per_sample_loss(self, logits, labels, task_type):
        """Return ``(loss_vec[M], valid_sample[M])`` for the gate-weighted loss.

        ``valid_sample`` flags samples with at least one usable label; the
        separate loss averages only over those, matching the valid-count
        normalization of the shared ``supervised_loss_from_logits`` path.
        """
        labels_t = torch.as_tensor(labels)
        if task_type == "regression":
            pred = logits.view(logits.size(0), -1).float()
            target = labels_t.float().view(pred.size(0), -1)
            valid = torch.isfinite(target)
            safe = torch.where(valid, target, torch.zeros_like(target))
            diff = (pred - safe) ** 2
            denom = valid.float().sum(dim=1).clamp(min=1.0)
            return (diff * valid.float()).sum(dim=1) / denom, valid.any(dim=1)

        if labels_t.dim() > 1 and labels_t.size(-1) > 1:  # multi-task binary
            targets, valid = binary_targets_and_valid(labels_t.float())
            safe = torch.where(valid, targets, torch.zeros_like(targets))
            loss_mat = F.binary_cross_entropy_with_logits(logits.float(), safe, reduction="none")
            denom = valid.float().sum(dim=1).clamp(min=1.0)
            return (loss_mat * valid.float()).sum(dim=1) / denom, valid.any(dim=1)

        if logits.dim() == 1 or (logits.dim() == 2 and logits.size(-1) == 1):  # single-logit binary
            logits_vec = logits.view(-1).float()
            targets, valid = binary_targets_and_valid(labels_t.view(-1).float())
            loss = F.binary_cross_entropy_with_logits(logits_vec, targets, reduction="none")
            return loss * valid.float(), valid

        class_labels = prepare_class_labels(labels_t).to(logits.device)
        loss = F.cross_entropy(logits, class_labels, reduction="none")
        valid = torch.ones(loss.size(0), dtype=torch.bool, device=loss.device)
        return loss, valid

    def _mixture_loss(self, weak_logits, strong_logits, gating, labels):
        """Return ``(loss, primary)`` for the configured subloss."""
        if self.subloss == "joint":
            mixed = weak_logits * gating + strong_logits * (1.0 - gating)
            return supervised_loss_from_logits(
                logits=mixed, labels=labels, task_type=self.task_type,
            )
        # separate: per-sample gate-weighted mixture of the two expert losses,
        # averaged over samples with a usable label (valid-count normalization,
        # consistent with the joint/eval path).
        loss_w, valid_mask = self._per_sample_loss(weak_logits, labels, self.task_type)
        loss_s, _ = self._per_sample_loss(strong_logits, labels, self.task_type)
        g = gating.view(-1)
        valid_f = valid_mask.float()
        per_sample = (g * loss_w + (1.0 - g) * loss_s) * valid_f
        denom = valid_f.sum().clamp(min=1.0)
        loss = per_sample.sum() / denom
        # Primary metric reported from the deterministic mixture (stable).
        mixed = weak_logits * gating + strong_logits * (1.0 - gating)
        _, primary = supervised_loss_from_logits(
            logits=mixed, labels=labels, task_type=self.task_type,
        )
        return loss, primary

    def _log(self, primary: float) -> dict[str, float]:
        key = "train_mae" if self.task_type == "regression" else "train_acc"
        return {key: float(primary)}

    # ------------------------------------------------------------------ #
    # Training steps
    # ------------------------------------------------------------------ #
    def step(self, model, data, device):
        """mowst_star step: full-grad gated forward + mixture loss."""
        weak_logits, strong_logits, gating, labels = self._gated_forward(
            model, data, device, "train_mask"
        )
        loss, primary = self._mixture_loss(weak_logits, strong_logits, gating, labels)
        return loss, self._log(primary)

    def step_turn(self, model, data, device, turn: str):
        """mowst alternating step: freeze one branch, separate gate-weighted loss."""
        if turn == "weak":
            weak_logits, strong_logits, gating, labels = self._gated_forward(
                model, data, device, "train_mask", strong_no_grad=True
            )
        elif turn == "strong":
            weak_logits, strong_logits, gating, labels = self._gated_forward(
                model, data, device, "train_mask", weak_no_grad=True, gate_no_grad=True
            )
        else:
            raise ValueError(f"Unknown turn '{turn}'. Use 'weak' or 'strong'.")
        loss, primary = self._mixture_loss(weak_logits, strong_logits, gating, labels)
        return loss, self._log(primary)

    def pretrain_step(self, model, data, device, which: str):
        """Supervised warm-up of a single expert + its head (submethod)."""
        data = data.to(device)
        if which == "weak":
            node_repr, graph_repr = model.forward_weak(data)
            head = self.weak_head
        elif which == "strong":
            node_repr, graph_repr = model.forward_strong(data)
            head = self.strong_head
        else:
            raise ValueError(f"Unknown expert '{which}'. Use 'weak' or 'strong'.")
        rep, labels = self._select_rep(node_repr, graph_repr, data, "train_mask", device)
        logits = head(rep)
        return supervised_loss_from_logits(logits=logits, labels=labels, task_type=self.task_type)

    # ------------------------------------------------------------------ #
    # Evaluation (shared supervised-eval contract)
    # ------------------------------------------------------------------ #
    def evaluate(self, model, data, device, mask_attr="val_mask", return_outputs=False):
        weak_logits, strong_logits, gating, labels = self._gated_forward(
            model, data, device, mask_attr
        )
        mixed = weak_logits * gating + strong_logits * (1.0 - gating)
        return supervised_loss_from_logits(
            logits=mixed, labels=labels, task_type=self.task_type, return_outputs=return_outputs,
        )


__all__ = ["MowstTask"]
