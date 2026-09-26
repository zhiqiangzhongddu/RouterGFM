"""Node-MoE task: target-node readout, CE loss and the filter-smoothing regulariser (Eq. 2).

Node-MoE is node-task-scoped (the paper evaluates node classification only),
so any other task level is rejected. With induced ego-subgraphs the
prediction for node i is node i's own mixed logits (``target`` readout);
``mean`` pooling is kept only for a like-for-like comparison with GMoE.
"""

from __future__ import annotations

import torch
from torch import nn

from src.utils.dataset_helpers import normalize_node_mask
from src.utils.parsing import resolve_task_type
from src.utils.pool import get_batch_vector, pool_nodes, pool_target_nodes
from src.utils.supervised_loss import supervised_loss_from_logits

READOUTS = ("target", "mean")


def require_node_task_level(task_level) -> str:
    """Return ``"node"`` or raise: Node-MoE is node-task-scoped."""
    level = str(task_level or "").lower()
    if level != "node":
        raise ValueError(
            f"Node-MoE is node-task-scoped: got task_level='{level}', but the method is only "
            "defined (and evaluated) for node classification. Use another MoE method for "
            "edge/graph tasks."
        )
    return level


class NodeMoETask(nn.Module):
    """Loss/readout wrapper; all trainable parameters live in :class:`NodeMoEModel`."""

    def __init__(self, cfg):
        super().__init__()
        nodemoe_cfg = cfg.moe.nodemoe
        ds_cfg = nodemoe_cfg.dataset
        self.task_level = require_node_task_level(ds_cfg.task_level)
        self.induced = bool(getattr(ds_cfg, "induced", False))
        self.readout = str(nodemoe_cfg.readout).lower()
        if self.readout not in READOUTS:
            raise ValueError(f"Unknown moe.nodemoe.readout '{self.readout}'; expected {READOUTS}.")
        self.task_type = resolve_task_type(getattr(ds_cfg, "task_type", None))
        self.smoothing_gamma = float(nodemoe_cfg.smoothing_gamma)
        self.track_gate_weights(False)

    def parameters_to_optimize(self):
        return iter(())

    def track_gate_weights(self, enabled: bool) -> None:
        """Start/stop accumulating readout-row gate weights during ``evaluate`` (analysis only)."""
        self._track_gate = bool(enabled)
        self._gate_sum: torch.Tensor | None = None
        self._gate_count = 0

    def mean_gate_weights(self) -> list[float] | None:
        if self._gate_sum is None or self._gate_count == 0:
            return None
        return (self._gate_sum / self._gate_count).tolist()

    def _select(self, rows: torch.Tensor, data, mask_attr: str, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick the prediction rows (and labels) of ``rows`` [N, *] for this batch."""
        if not self.induced:
            mask = normalize_node_mask(data, mask_attr, device)
            return rows[mask], data.y[mask]
        if self.readout == "target":
            return pool_target_nodes(rows, data), data.y
        return pool_nodes(rows, get_batch_vector(data), "mean"), data.y

    def _forward(self, model, data, device, mask_attr: str = "train_mask", return_outputs: bool = False):
        data = data.to(device)
        mixed, gate = model(data)
        logits, labels = self._select(mixed, data, mask_attr, device)
        if self._track_gate:
            gate_rows, _ = self._select(gate.detach(), data, mask_attr, device)
            batch_sum = gate_rows.double().sum(dim=0).cpu()
            self._gate_sum = batch_sum if self._gate_sum is None else self._gate_sum + batch_sum
            self._gate_count += int(gate_rows.size(0))
        return supervised_loss_from_logits(
            logits=logits,
            labels=labels,
            task_type=self.task_type,
            return_outputs=return_outputs,
        )

    def step(self, model, data, device):
        loss, primary = self._forward(model=model, data=data, device=device, mask_attr="train_mask")
        log: dict[str, float] = {}
        if self.smoothing_gamma > 0.0:
            smooth = model.smoothing_loss()
            loss = loss + self.smoothing_gamma * smooth
            log["train_smooth_loss"] = float(smooth.detach().item())
        if self.task_type == "regression":
            log["train_mae"] = primary
        else:
            log["train_acc"] = primary
        return loss, log

    def evaluate(self, model, data, device, mask_attr="val_mask", return_outputs=False):
        # The smoothing term is a training-only regulariser (matches GMoETask.evaluate).
        return self._forward(
            model=model, data=data, device=device,
            mask_attr=mask_attr, return_outputs=return_outputs,
        )


__all__ = ["READOUTS", "NodeMoETask", "require_node_task_level"]
