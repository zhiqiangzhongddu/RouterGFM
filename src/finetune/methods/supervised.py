from __future__ import annotations

import torch

from src.finetune.task_base import StepFinetuneTask
from src.finetune.registry import register
from src.finetune.task_heads import TaskAwareObjective
from src.utils.config_helpers import resolve_workflow_dataset_cfg
from src.utils.dataset_helpers import read_effective_task_level
from src.utils.parsing import resolve_task_type
from src.utils.supervised_forward import select_supervised_logits_and_labels
from src.utils.supervised_loss import build_supervised_head


@register("supervised")
class FinetuneSupervised(StepFinetuneTask):
    """Supervised finetuning task (step-based).

    Uses the same ``select_supervised_logits_and_labels`` +
    ``supervised_loss_from_logits`` pipeline as pretrain and train
    supervised methods, ensuring consistent forward/loss behaviour
    across all three workflows.
    """

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        ds_cfg = resolve_workflow_dataset_cfg(cfg, "finetune")
        if read_effective_task_level(ds_cfg) == "edge":
            raise ValueError(
                "[FinetuneSupervised] Non-induced edge-level supervised "
                "finetuning is not supported: select_supervised_logits_and_labels "
                "requires data.edge_label_index/edge_label which full-graph "
                "datasets don't provide. Set finetune.dataset.induced=True "
                "to use induced subgraphs."
            )

    def __init__(self, cfg):
        super().__init__(cfg)
        ds_cfg = resolve_workflow_dataset_cfg(cfg, "finetune")
        self.task_level = read_effective_task_level(ds_cfg)
        self.task_type = resolve_task_type(getattr(ds_cfg, "task_type", None))
        self.label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        num_classes = int(getattr(ds_cfg, "num_classes", 1) or 1)
        self.objective = TaskAwareObjective(
            cfg,
            task_level=self.task_level,
            task_type=self.task_type,
            label_dim=self.label_dim,
            num_classes=num_classes,
            repr_dim=cfg.model.out_dim,
        )
        self.classifier = build_supervised_head(
            in_dim=cfg.model.out_dim,
            task_type=self.task_type,
            task_level=self.task_level,
            label_dim=self.label_dim,
            num_classes=num_classes,
        )
        # finetune.edge_readout=endpoints on induced edge cells: resolved once
        # by TaskAwareObjective (raw-level gated, inert elsewhere) and applied
        # here because this method routes its forward through the promoted
        # level, which pools the whole subgraph and cannot see the endpoints.
        self.edge_endpoint_readout = self.objective.edge_endpoint_readout

    def _forward(self, model, data, device, mask_attr: str = "train_mask", return_outputs: bool = False):
        data = data.to(device)
        node_repr, graph_repr = model(data)

        if self.edge_endpoint_readout:
            edge_label_index = getattr(data, "edge_label_index", None)
            if edge_label_index is None:
                raise ValueError(
                    "finetune.edge_readout=endpoints requires data.edge_label_index "
                    "(induced edge subgraphs); this batch has none."
                )
            eli = torch.as_tensor(edge_label_index, device=node_repr.device).view(2, -1)
            logits_used = self.classifier(node_repr[eli[0]] * node_repr[eli[1]])
            labels = data.y
        else:
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

        return self.objective.loss_from_logits(
            logits=logits_used,
            labels=labels,
            return_outputs=return_outputs,
        )

    def evaluate(self, model, data, device, mask_attr="val_mask", return_outputs=False):
        """Supervised evaluation forward pass.

        Supports the same ``(model, data, device, mask_attr, return_outputs)``
        signature expected by ``evaluate_supervised_split``.
        """
        return self._forward(
            model=model, data=data, device=device,
            mask_attr=mask_attr, return_outputs=return_outputs,
        )

    # Backward-compatible alias.
    forward_for_eval = evaluate

    def step(self, model, data, device):
        loss, primary = self._forward(model=model, data=data, device=device, mask_attr="train_mask")
        if self.task_type == "regression":
            return loss, {"train_mae": primary}
        return loss, {"train_acc": primary}
