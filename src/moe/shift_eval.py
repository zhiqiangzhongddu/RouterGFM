"""Shared evaluation for the Table 15 shift baselines (GraphMETRO, OGMM, GeoMoE).

One readout rule, one way to collect per-query outputs, and a Brier risk that
delegates to RouterGFM's per-instance routing loss (App. B.2), so every method
in the shift comparison is scored by the same function as RouterGFM itself.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional

import torch
import torch.nn.functional as F

from src.moe.routergfm.applications import convert_labels
from src.moe.routergfm.common import REGRESSION
from src.moe.routergfm.losses import RegressionNormalizer, is_simplex_family, routing_loss
from src.utils.checkpoint import save_torch_atomic
from src.utils.pool import get_batch_vector, pool_nodes, pool_target_nodes


def instance_readout(node_repr: torch.Tensor, data, task_level_raw: str, pool_mode: str = "mean") -> torch.Tensor:
    """One vector per instance: target node (node), endpoint Hadamard product (edge), pooled graph (graph)."""
    level = str(task_level_raw).lower()
    if level == "node":
        return pool_target_nodes(node_repr, data)
    if level == "edge":
        edge_label_index = getattr(data, "edge_label_index", None)
        if edge_label_index is None:
            raise ValueError("Edge instances need data.edge_label_index (one target pair per subgraph).")
        src, dst = edge_label_index
        return node_repr[src] * node_repr[dst]
    return pool_nodes(node_repr, get_batch_vector(data), mode=pool_mode)


def _probabilities(logits: torch.Tensor, task_type: str, label_dim: int) -> torch.Tensor:
    """Head outputs -> prediction space: raw (regression), sigmoid (multi-label / single logit), softmax."""
    logits = logits.float().reshape(logits.size(0), -1)
    if str(task_type).lower() == "regression":
        return logits
    if int(label_dim or 1) > 1 or logits.size(-1) == 1:
        return torch.sigmoid(logits)
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def collect_query_outputs(
    model_fn: Callable[[object], torch.Tensor],
    loader,
    device,
    task_type: str,
    label_dim: int,
) -> Dict[str, torch.Tensor]:
    """Run ``model_fn`` (batch -> head logits ``[B, out]``) over an unshuffled loader.

    Returns CPU tensors ``index`` (dataset positions, from the loader's ``Subset``
    when present), ``y`` (raw labels ``[N, d]``) and ``pred`` (probabilities, or
    raw regression outputs). The caller puts the model in eval mode.
    """
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = _probabilities(model_fn(batch), task_type, label_dim)
        preds.append(pred.cpu())
        labels.append(torch.as_tensor(batch.y).reshape(pred.size(0), -1).cpu())
    pred = torch.cat(preds) if preds else torch.empty(0, 0)
    y = torch.cat(labels) if labels else torch.empty(0, 0)
    indices = getattr(loader.dataset, "indices", None)
    index = torch.as_tensor(list(indices), dtype=torch.long) if indices is not None else torch.arange(pred.size(0))
    if index.numel() != pred.size(0):
        raise ValueError(f"{index.numel()} loader indices for {pred.size(0)} predictions.")
    return {"index": index, "y": y, "pred": pred}


def brier_risk(
    outputs: Mapping[str, torch.Tensor],
    *,
    task_family: str,
    support_targets: Optional[torch.Tensor] = None,
    reg_kind: str = "abs",
) -> float:
    """Mean RouterGFM routing loss of the query predictions (the "Brier risk" of Table 15).

    ``task_family`` is a ``src.moe.routergfm.common`` family. Single-logit binary
    outputs become two-class probabilities ``[1-p, p]`` (squared probability
    error). Regression predictions and targets are normalized with the support
    median/MAD (``support_targets`` in raw units, required). Rows without a valid
    label are skipped.
    """
    pred = torch.as_tensor(outputs["pred"]).float()
    pred = pred.reshape(pred.size(0), -1)
    target = convert_labels(torch.as_tensor(outputs["y"]).reshape(pred.size(0), -1), task_family)
    if is_simplex_family(task_family) and pred.size(-1) == 1:
        pred = torch.cat([1.0 - pred, pred], dim=-1)
    if task_family == REGRESSION:
        if support_targets is None:
            raise ValueError("Regression Brier risk needs the support targets (median/MAD normalization).")
        normalizer = RegressionNormalizer().fit(torch.as_tensor(support_targets).float())
        pred, target = normalizer.transform(pred), normalizer.transform(target)
    losses = routing_loss(pred, target, task_family, reg_kind=reg_kind)
    losses = losses[torch.isfinite(losses)]
    return float(losses.mean()) if losses.numel() else float("nan")


def save_query_predictions(path: str, outputs: Mapping[str, torch.Tensor], meta: Mapping) -> None:
    """Atomically save ``{index, y, pred, meta}`` (pred kept in float32: raw regression units need it)."""
    save_torch_atomic(
        str(path),
        {
            "index": torch.as_tensor(outputs["index"]).long().cpu(),
            "y": torch.as_tensor(outputs["y"]).cpu(),
            "pred": torch.as_tensor(outputs["pred"]).float().cpu(),
            "meta": dict(meta),
        },
    )


__all__ = ["brier_risk", "collect_query_outputs", "instance_readout", "save_query_predictions"]
