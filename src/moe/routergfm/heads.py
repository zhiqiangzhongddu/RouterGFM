"""Task heads fitted on frozen-encoder support embeddings (Sec. 3.1, Alg. 1 l.2/l.15).

Heads are trained full-batch with Adam on CPU from a fixed seed, so the same
(support embeddings, labels, seed, head config) always yields the same head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MULTILABEL, REGRESSION
from .losses import RegressionNormalizer, is_simplex_family

_PREDICT_CHUNK = 65536


@dataclass
class FittedHead:
    module: nn.Module  # on CPU, eval mode
    mean: torch.Tensor  # input standardization (support statistics)
    std: torch.Tensor
    family: str
    out_dim: int


def _heads_cfg(cfg):
    """Accept the full cfg or its ``moe.routergfm.heads`` block."""
    return cfg.moe.routergfm.heads if hasattr(cfg, "moe") else cfg


def _build_module(in_dim: int, out_dim: int, hcfg) -> nn.Module:
    kind = str(hcfg.type).lower()
    if kind == "linear":
        return nn.Linear(in_dim, out_dim)
    if kind == "mlp":
        hidden = int(hcfg.hidden_dim)
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    raise ValueError(f"Unknown head type {hcfg.type!r} (expected linear|mlp).")


def _training_target(y: torch.Tensor, family: str, normalizer: Optional[RegressionNormalizer]) -> torch.Tensor:
    if is_simplex_family(family):
        return torch.as_tensor(y).reshape(-1).long()
    y = torch.as_tensor(y).float()
    y = y.reshape(y.size(0), -1)
    if family == REGRESSION:
        if normalizer is None:
            raise ValueError("Regression heads need the application's RegressionNormalizer.")
        return normalizer.transform(y)
    return y


def _head_loss(out: torch.Tensor, target: torch.Tensor, family: str) -> torch.Tensor:
    if is_simplex_family(family):
        return F.cross_entropy(out, target)
    valid = torch.isfinite(target)
    if not bool(valid.any()):
        return out.sum() * 0.0
    safe = torch.nan_to_num(target, nan=0.0)
    if family == MULTILABEL:
        elem = F.binary_cross_entropy_with_logits(out, safe, reduction="none")
    else:
        elem = (out - safe).pow(2)
    return (elem * valid).sum() / valid.sum()


def _activate(out: torch.Tensor, family: str) -> torch.Tensor:
    if is_simplex_family(family):
        return torch.softmax(out, dim=-1)
    if family == MULTILABEL:
        return torch.sigmoid(out)
    return out


def fit_head(
    emb_s: torch.Tensor,
    y_s: torch.Tensor,
    family: str,
    out_dim: int,
    cfg,
    *,
    seed: int,
    normalizer: Optional[RegressionNormalizer] = None,
    device=None,
) -> FittedHead:
    """Fit a linear (or one-hidden-layer) head on support embeddings.

    Training losses: cross-entropy (node/graph classification, two-class link
    prediction), masked BCE (multi-label, NaN = missing), MSE on
    median/MAD-normalized targets (regression; ``y_s`` in raw units).
    ``device`` only selects where the full-batch optimization runs (default
    CPU); the returned head always lives on CPU.
    """
    hcfg = _heads_cfg(cfg)
    x = torch.as_tensor(emb_s).detach().float().cpu()
    target = _training_target(torch.as_tensor(y_s).cpu(), family, normalizer)
    if x.size(0) != target.size(0):
        raise ValueError(f"Support embeddings ({x.size(0)}) and labels ({target.size(0)}) are misaligned.")
    if bool(hcfg.standardize_inputs) and x.size(0) > 0:
        mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    else:
        mean = torch.zeros(x.size(1))
        std = torch.ones(x.size(1))
    x = (x - mean) / std

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        module = _build_module(x.size(1), int(out_dim), hcfg)
    device = torch.device(device) if device is not None else torch.device("cpu")
    module.to(device)
    x, target = x.to(device), target.to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=float(hcfg.lr), weight_decay=float(hcfg.weight_decay))
    module.train()
    if x.size(0) > 0:
        for _ in range(int(hcfg.epochs)):
            optimizer.zero_grad()
            loss = _head_loss(module(x), target, family)
            loss.backward()
            optimizer.step()
    module.cpu()
    module.eval()
    module.requires_grad_(False)
    return FittedHead(module=module, mean=mean, std=std, family=family, out_dim=int(out_dim))


@torch.no_grad()
def predict_head(head: FittedHead, emb: torch.Tensor) -> torch.Tensor:
    """Predictions in the family's mixture space (probabilities or normalized outputs)."""
    x = torch.as_tensor(emb).float().cpu()
    outs = []
    for start in range(0, x.size(0), _PREDICT_CHUNK):
        chunk = (x[start:start + _PREDICT_CHUNK] - head.mean) / head.std
        outs.append(_activate(head.module(chunk), head.family))
    if not outs:
        return torch.zeros(0, head.out_dim)
    return torch.cat(outs, dim=0)


def oof_fold_ids(y_s: torch.Tensor, family: str, folds: int, seed: int) -> torch.Tensor:
    """Deterministic fold id per support item, stratified by class when single-label.

    Classes are dealt round-robin over folds after a seeded shuffle, continuing
    the rotation across classes, so classes with fewer than ``folds`` items land
    in distinct folds (leave-one-out style) and fold sizes stay balanced.
    """
    n = int(torch.as_tensor(y_s).size(0))
    generator = torch.Generator().manual_seed(int(seed))
    fold_id = torch.empty(n, dtype=torch.long)
    if is_simplex_family(family):
        labels = torch.as_tensor(y_s).reshape(-1).long()
        offset = 0
        for cls in torch.unique(labels).tolist():
            members = torch.nonzero(labels == cls, as_tuple=False).view(-1)
            members = members[torch.randperm(members.numel(), generator=generator)]
            fold_id[members] = (offset + torch.arange(members.numel())) % folds
            offset += members.numel()
    else:
        order = torch.randperm(n, generator=generator)
        fold_id[order] = torch.arange(n) % folds
    return fold_id


def fit_predict_oof(
    emb_s: torch.Tensor,
    y_s: torch.Tensor,
    family: str,
    out_dim: int,
    cfg,
    *,
    seed: int,
    folds: Optional[int] = None,
    normalizer: Optional[RegressionNormalizer] = None,
    device=None,
) -> torch.Tensor:
    """Out-of-fold support predictions: every item is predicted by a head not trained on it.

    ``folds`` defaults to ``heads.oof_folds`` and is capped at the support size.
    Regression folds reuse the application's support normalizer so OOF and
    query predictions share one normalized scale.
    """
    hcfg = _heads_cfg(cfg)
    emb_s = torch.as_tensor(emb_s).float().cpu()
    y_s = torch.as_tensor(y_s).cpu()
    n = emb_s.size(0)
    k = min(int(folds if folds is not None else hcfg.oof_folds), n)
    if k < 2:
        raise ValueError(f"Out-of-fold prediction needs at least two support items (got {n}).")
    fold_id = oof_fold_ids(y_s, family, k, seed)
    oof = torch.full((n, int(out_dim)), float("nan"))
    for fold in range(k):
        held = fold_id == fold
        head = fit_head(
            emb_s[~held], y_s[~held], family, out_dim, cfg, seed=int(seed) + fold + 1, normalizer=normalizer, device=device
        )
        oof[held] = predict_head(head, emb_s[held])
    return oof


__all__ = ["FittedHead", "fit_head", "fit_predict_oof", "oof_fold_ids", "predict_head"]
