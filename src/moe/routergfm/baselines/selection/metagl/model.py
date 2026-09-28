"""MetaGL scorer ``p_hat_ij = <h_{G_i}, h_{M_j}>`` over the G-M network, its top-one loss,
validation criterion, and training loop (official ``do_train``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hgt import HGT
from .network import GRAPH, MODEL, NODE_TYPES, GMNetwork, Relation


def top_one_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """ListNet top-one cross-entropy per row, averaged; NaN entries of *y_true* are missing."""
    mask = torch.isnan(y_true)
    y_pred = y_pred.masked_fill(mask, float("-inf"))
    y_true = y_true.masked_fill(mask, float("-inf"))
    log_pred = torch.log(F.softmax(y_pred, dim=1) + eps)
    return torch.mean(-torch.sum(F.softmax(y_true, dim=1) * log_pred, dim=1))


def best_model_auc_mrr(true: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    """ROC-AUC and average precision (= 1 / rank) of the true best entry against the others."""
    best = int(np.argmax(true))
    others = np.delete(pred, best)
    auc = float(np.mean((others < pred[best]) + 0.5 * (others == pred[best])))
    return auc, 1.0 / float(np.sum(pred >= pred[best]))


def validation_score(P_true: np.ndarray, P_hat: np.ndarray) -> float:
    """Official MetaGL criterion ``mean(2 AUC, MRR)`` over rows with >= 2 observed entries."""
    aucs, mrrs = [], []
    for true, pred in zip(P_true, P_hat):
        obs = np.isfinite(true)
        if obs.sum() >= 2:
            auc, mrr = best_model_auc_mrr(true[obs], pred[obs])
            aucs.append(auc)
            mrrs.append(mrr)
    return float(np.mean([2 * np.mean(aucs), np.mean(mrrs)])) if aucs else float("nan")


class MetaGLNet(nn.Module):
    """Graph node input ``Linear([M' ; U_hat])``; model node input ``V`` (fixed) or, with expert
    metadata, ``Linear([v ; V_hat])``; HGT embeddings; dot-product scores."""

    def __init__(
        self,
        meta_dim: int,
        k_in: int,
        hid_dim: int,
        relations: Sequence[Relation],
        *,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.5,
        expert_meta_dim: Optional[int] = None,
    ):
        super().__init__()
        self.graph_emb_net = nn.Linear(int(meta_dim) + int(k_in), int(k_in))
        self.model_emb_net = nn.Linear(int(expert_meta_dim) + int(k_in), int(k_in)) if expert_meta_dim else None
        self.hgt = HGT(k_in, hid_dim, hid_dim, NODE_TYPES, relations, n_layers=n_layers, n_heads=n_heads, dropout=dropout)

    def forward(
        self,
        net: GMNetwork,
        Mp: torch.Tensor,
        U: torch.Tensor,
        V: torch.Tensor,
        v_meta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``P_hat [num graphs, num models]``."""
        x_model = V if self.model_emb_net is None else self.model_emb_net(torch.cat([v_meta, V], dim=1))
        x = {GRAPH: self.graph_emb_net(torch.cat([Mp, U], dim=1)), MODEL: x_model}
        h = self.hgt(x, net.edges)
        return h[GRAPH] @ h[MODEL].t()


@dataclass
class TrainResult:
    best_epoch: int  # 1-based; the last epoch when no validation ran
    best_score: float
    val_scores: List[float]


def train_metagl(
    model: MetaGLNet,
    train_net: GMNetwork,
    train_inputs: Tuple,
    P_train: torch.Tensor,
    val_net: Optional[GMNetwork],
    val_inputs: Optional[Tuple],
    P_val: Optional[np.ndarray],
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> TrainResult:
    """Adam over all parameters; each epoch visits the training rows in random batches, each batch a
    full network forward. From the second epoch on, validation rows (the last ``len(P_val)`` graph
    nodes of *val_net*) are scored with :func:`validation_score`; early stopping on its maximum.
    The best-validation state is always restored (the official code restores it only on early stop).
    """
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    n = P_train.size(0)
    best_score, best_state, best_epoch, bad, scores = -float("inf"), None, int(epochs), 0, []
    for epoch in range(int(epochs)):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, int(batch_size)):
            opt.zero_grad()
            batch = perm[start : start + int(batch_size)].to(P_train.device)
            loss = top_one_loss(model(train_net, *train_inputs)[batch], P_train[batch])
            loss.backward()
            opt.step()
        if val_net is None or epoch == 0:
            continue
        model.eval()
        with torch.no_grad():
            P_hat = model(val_net, *val_inputs)[-len(P_val):].cpu().numpy()
        score = validation_score(P_val, P_hat)
        scores.append(score)
        if score > best_score:
            best_score, best_state, best_epoch, bad = score, copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            bad += 1
            if bad >= int(patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return TrainResult(best_epoch=best_epoch, best_score=best_score if best_state is not None else float("nan"), val_scores=scores)


__all__ = ["MetaGLNet", "TrainResult", "best_model_auc_mrr", "top_one_loss", "train_metagl", "validation_score"]
