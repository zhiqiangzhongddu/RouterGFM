from __future__ import annotations

import torch

from ..task_base import PretrainTask
from ..registry import register
from .utils import make_zero_loss
from src.utils.parsing import resolve_task_type
from src.utils.supervised_forward import select_supervised_logits_and_labels
from src.utils.supervised_loss import build_supervised_head, supervised_loss_from_logits


@register("supervised")
class Supervised(PretrainTask):
    """Supervised pretraining task.

    Reference: Hu et al. "Strategies for Pre-training Graph Neural Networks" ICLR 2020.

    Representation flow follows the chem-style variant: pooled graph repr
    feeds the classifier head directly. Induced node/edge datasets are
    promoted to graph-level batches, so one induced ego-graph is classified
    by its original node label via pooled-only representation. If a center-
    node-conditioned supervision is desired, run supervised pretraining
    without ``induced`` instead — there is no need to track a center index.

    The encoder weights are the transferable artifact; the classifier head
    lives on the task module so ``PretrainRunner`` saves only the encoder
    ``state_dict`` — consistent with the official scripts that persist
    ``model.gnn.state_dict()`` only.
    """

    uses_dataset_splits = True

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        """No cfg-time validation is needed for supervised pretraining.

        ``num_classes`` / ``label_dim`` / ``task_type`` are populated by
        ``PretrainRunner._setup`` from dataset metadata *after* this hook
        runs, so their presence is asserted in :meth:`__init__` as a
        runtime contract check rather than here.
        """
        return None

    def __init__(self, cfg):
        super().__init__(cfg)
        # ``PretrainRunner._setup`` populates num_classes / label_dim /
        # task_type on cfg.pretrain.dataset after the dataset is loaded,
        # so by the time the task is constructed the fields are always
        # present. Read them directly; a missing value indicates the
        # runner contract was violated and should raise loudly rather
        # than silently default to 1.
        ds_cfg = cfg.pretrain.dataset
        raw_task_level = str(ds_cfg.task_level).lower()
        induced = bool(ds_cfg.induced)
        # Induced node/edge datasets become graph batches at runtime;
        # the task must use graph-level logic for the forward pass.
        self.task_level = "graph" if induced and raw_task_level in {"node", "edge"} else raw_task_level
        self.task_type = resolve_task_type(ds_cfg.task_type)
        if ds_cfg.label_dim is None:
            raise ValueError(
                "[Supervised] cfg.pretrain.dataset.label_dim is None; "
                "PretrainRunner must populate it from dataset metadata before task construction."
            )
        self.label_dim = int(ds_cfg.label_dim)
        if self.task_type != "regression" and ds_cfg.num_classes is None:
            raise ValueError(
                "[Supervised] classification tasks require cfg.pretrain.dataset.num_classes; "
                "PretrainRunner must populate it from dataset metadata before task construction."
            )
        num_classes = int(ds_cfg.num_classes) if ds_cfg.num_classes is not None else 1
        self.classifier = build_supervised_head(
            in_dim=cfg.model.out_dim,
            task_type=self.task_type,
            task_level=self.task_level,
            label_dim=self.label_dim,
            num_classes=num_classes,
        )

    def evaluate(
        self,
        model,
        data,
        device,
        mask_attr: str = "train_mask",
        return_outputs: bool = False,
    ):
        """Run the supervised forward pass used by both train and eval."""
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
            zero = make_zero_loss(self, device, node_repr)
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
        loss, primary = self.evaluate(model=model, data=data, device=device, mask_attr="train_mask")
        if self.task_type == "regression":
            return loss, {"train_mae": primary}
        return loss, {"train_acc": primary}
