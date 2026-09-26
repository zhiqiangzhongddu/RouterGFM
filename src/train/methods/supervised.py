from __future__ import annotations

import torch

from src.train.task_base import TrainTask
from src.train.registry import register
from src.utils.parsing import resolve_task_type
from src.utils.supervised_forward import select_supervised_logits_and_labels
from src.utils.supervised_loss import build_supervised_head, supervised_loss_from_logits


@register("supervised")
class TrainSupervised(TrainTask):
    """Supervised training head that uses the train dataset config."""

    def __init__(self, cfg):
        super().__init__(cfg)
        ds_cfg = getattr(getattr(cfg, "train", None), "dataset", None) or getattr(cfg, "dataset", None)
        raw_task_level = str(ds_cfg.task_level).lower()
        induced = bool(getattr(ds_cfg, "induced", False))
        # Induced node/edge datasets become graph batches at runtime;
        # the task must use graph-level logic for the forward pass.
        # Mirrors Supervised.__init__ in src/pretrain/methods/supervised.py.
        self.task_level = "graph" if induced and raw_task_level in {"node", "edge"} else raw_task_level
        # Non-induced edge-level training is not supported: the shared
        # supervised-forward helper routes edge tasks through
        # data.edge_label_index/edge_label, which full-graph datasets
        # don't provide. Fail fast at construction instead of leaking a
        # cryptic error from deep inside the forward pass.
        if self.task_level == "edge":
            raise ValueError(
                "[TrainSupervised] Non-induced edge-level training is not "
                "supported: select_supervised_logits_and_labels requires "
                "data.edge_label_index/edge_label which full-graph datasets "
                "don't provide. Set train.dataset.induced=True to use "
                "induced subgraphs."
            )
        self.task_type = resolve_task_type(getattr(ds_cfg, "task_type", None))
        self.label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        num_classes = int(getattr(ds_cfg, "num_classes", 1) or 1)
        self.classifier = build_supervised_head(
            in_dim=cfg.model.out_dim,
            task_type=self.task_type,
            task_level=self.task_level,
            label_dim=self.label_dim,
            num_classes=num_classes,
        )

    def _forward(self, model, data, device, mask_attr: str = "train_mask", return_outputs: bool = False):
        data = data.to(device)
        node_repr, graph_repr = model(data)

        logits_used, labels = select_supervised_logits_and_labels(
            node_repr=node_repr,
            graph_repr=graph_repr,
            data=data,
            classifier=self.classifier,
            task_level=self.task_level,
            pool_mode=self.cfg.model.graph_pooling,
            mask_attr=mask_attr,
            device=device,
        )

        if logits_used.numel() == 0:
            # Empty selection (e.g. an all-False val mask on a (1, 0, 1) split)
            # would feed 0-row logits into the loss and print NaN every epoch;
            # mirror the pretrain Supervised guard.
            zero = node_repr.sum() * 0.0
            if return_outputs:
                empty = logits_used.view(0, 1) if logits_used.dim() < 2 else logits_used
                return zero, 0.0, empty, torch.as_tensor(labels).view(-1)[:0]
            return zero, 0.0

        return supervised_loss_from_logits(
            logits=logits_used,
            labels=labels,
            task_type=self.task_type,
            return_outputs=return_outputs,
        )

    def step(self, model, data, device):
        loss, primary = self._forward(model=model, data=data, device=device, mask_attr="train_mask")
        if self.task_type == "regression":
            return loss, {"train_mae": primary}
        return loss, {"train_acc": primary}

    def evaluate(self, model, data, device, mask_attr="val_mask", return_outputs=False):
        """Supervised evaluation forward pass.

        Supports the same ``(model, data, device, mask_attr, return_outputs)``
        signature expected by ``evaluate_supervised_split``.
        """
        return self._forward(
            model=model, data=data, device=device,
            mask_attr=mask_attr, return_outputs=return_outputs,
        )
