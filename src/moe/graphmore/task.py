"""Supervised task head for GraphMoRE with distortion-loss integration.

Mirrors :class:`src.moe.gmoe.task.GMoETask`. The only method-specific
behaviour is that the model exposes an optional distortion regulariser
(``model.distortion_loss``); the training ``step`` adds it to the utility
loss when ``coef_dis > 0`` and evaluation ignores it. The forward / loss /
head plumbing is shared with the rest of the repo
(``select_supervised_logits_and_labels`` + ``supervised_loss_from_logits``)
so GraphMoRE produces identical metrics to the standard supervised path.

The head input dim is ``K * embed_dim`` (the concatenated mixture of
``K = len(init_curvs)`` experts), not ``embed_dim``.
"""

from __future__ import annotations

import torch
from torch import nn

from src.utils.dataset_helpers import resolve_effective_task_level
from src.utils.parsing import resolve_task_type
from src.utils.supervised_forward import select_supervised_logits_and_labels
from src.utils.supervised_loss import build_supervised_head, supervised_loss_from_logits


class GraphMoRETask(nn.Module):
    """Supervised classification/regression head over GraphMoRE embeddings."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        gm_cfg = cfg.moe.graphmore
        ds_cfg = gm_cfg.dataset

        raw_task_level = str(ds_cfg.task_level).lower()
        induced = bool(getattr(ds_cfg, "induced", False))
        # Induced node/edge datasets are materialised as graph batches; the
        # head must then use graph-level logic. Mirrors GMoETask / TrainSupervised.
        self.task_level = resolve_effective_task_level(raw_task_level, induced)
        if self.task_level == "edge":
            raise ValueError(
                "[GraphMoRE] Non-induced edge-level training is not supported: "
                "select_supervised_logits_and_labels requires "
                "data.edge_label_index/edge_label which full-graph datasets "
                "don't provide. Set moe.graphmore.dataset.induced=True to use "
                "induced subgraphs."
            )

        self.task_type = resolve_task_type(getattr(ds_cfg, "task_type", None))
        self.label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        num_classes = int(getattr(ds_cfg, "num_classes", 1) or 1)
        self.pool_mode = str(gm_cfg.graph_pooling)
        self.coef_dis = float(gm_cfg.coef_dis)
        self.coef_dis_active = self.coef_dis > 0.0

        num_experts = len(list(gm_cfg.init_curvs))
        mixture_dim = num_experts * int(gm_cfg.embed_dim)
        self.classifier = build_supervised_head(
            in_dim=mixture_dim,
            task_type=self.task_type,
            task_level=self.task_level,
            label_dim=self.label_dim,
            num_classes=num_classes,
        )

    def parameters_to_optimize(self):
        """Head parameters; the encoder's parameters are added by the runner."""
        return self.parameters()

    def _forward(self, model, data, device, mask_attr: str = "train_mask", return_outputs: bool = False):
        data = data.to(device)
        node_repr, graph_repr = model(data)

        logits_used, labels = select_supervised_logits_and_labels(
            node_repr=node_repr,
            graph_repr=graph_repr,
            data=data,
            classifier=self.classifier,
            task_level=self.task_level,
            pool_mode=self.pool_mode,
            mask_attr=mask_attr,
            device=device,
        )
        return supervised_loss_from_logits(
            logits=logits_used,
            labels=labels,
            task_type=self.task_type,
            return_outputs=return_outputs,
        )

    def step(self, model, data, device):
        loss, primary = self._forward(model=model, data=data, device=device, mask_attr="train_mask")

        log: dict[str, float] = {}
        distortion_loss = getattr(model, "distortion_loss", 0.0)
        if self.coef_dis_active and torch.is_tensor(distortion_loss):
            loss = loss + self.coef_dis * distortion_loss
            log["train_dist_loss"] = float(distortion_loss.detach().item())

        if self.task_type == "regression":
            log["train_mae"] = primary
        else:
            log["train_acc"] = primary
        return loss, log

    def evaluate(self, model, data, device, mask_attr="val_mask", return_outputs=False):
        # The distortion loss is a training-only regulariser; eval uses the
        # plain supervised forward.
        return self._forward(
            model=model, data=data, device=device,
            mask_attr=mask_attr, return_outputs=return_outputs,
        )


__all__ = ["GraphMoRETask"]
