"""Step 1 of Link-MoE: train each expert independently, score val/test pairs.

Full-graph experts follow HeaRT's protocol: batches of train positives with
as many fresh uniform non-edges, ``-log sig(s+) - log sig(-s-)``, message
passing on ``M`` during training and on ``C`` for evaluation, val-AUC model
selection with patience. SEAL trains on the fixed split's labelled
enclosing subgraphs with BCE.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling

from src.utils.metrics import compute_supervised_metrics
from src.utils.random import set_seed

from .data import LinkViews


@dataclass
class ExpertScores:
    name: str
    val_prob: torch.Tensor  # [P_val], views.pairs["val"] order
    test_prob: torch.Tensor  # [P_test]
    val_auc: float
    test_auc: float  # diagnostic only (never used for selection)
    best_epoch: int
    state: dict | None = None  # best-epoch state_dict (CPU)


def probability_metrics(prob: torch.Tensor, labels: torch.Tensor, prefix: str = "") -> dict[str, float]:
    """Binary metrics of link probabilities via ``compute_supervised_metrics`` on ``logit(p)``."""
    logits = torch.logit(prob.detach().double().clamp(1e-6, 1 - 1e-6)).float()
    metrics = compute_supervised_metrics(logits, labels.detach().float(), task_type="classification")
    return {f"{prefix}{k}": float(v) for k, v in metrics.items()}


def _snapshot(module: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


@torch.no_grad()
def _score_pairs(expert: nn.Module, x, edge_index, pairs, batch_size: int) -> torch.Tensor:
    expert.eval()
    out = [torch.sigmoid(expert(x, edge_index, pairs[:, s:s + batch_size])) for s in range(0, pairs.size(1), batch_size)]
    return torch.cat(out).float().cpu() if out else torch.empty(0)


def train_full_graph_expert(name: str, expert: nn.Module, views: LinkViews, cfg, device, seed: int) -> ExpertScores:
    lcfg = cfg.moe.linkmoe
    ecfg = getattr(lcfg, name)
    set_seed(int(seed))
    expert = expert.to(device)
    x = views.x.to(device)
    message = views.message_edge_index.to(device)
    context = views.context_edge_index.to(device)
    train_pos = views.pairs["train"][:, views.labels["train"] > 0.5].to(device)
    val_pairs, val_y = views.pairs["val"].to(device), views.labels["val"]
    batch_size = int(lcfg.expert_batch_size)
    optimizer = torch.optim.Adam(expert.parameters(), lr=float(ecfg.lr), weight_decay=float(ecfg.weight_decay))
    generator = torch.Generator().manual_seed(int(seed))

    best_auc, best_epoch, best_state, stale = float("-inf"), 0, None, 0
    for epoch in range(1, int(lcfg.expert_max_epochs) + 1):
        expert.train()
        perm = torch.randperm(train_pos.size(1), generator=generator).to(device)
        for idx in perm.split(batch_size):
            pos = train_pos[:, idx]
            neg = negative_sampling(context, num_nodes=views.num_nodes, num_neg_samples=pos.size(1))
            logits = expert(x, message, torch.cat([pos, neg], dim=1))
            loss = -F.logsigmoid(logits[: pos.size(1)]).mean() - F.logsigmoid(-logits[pos.size(1):]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_auc = probability_metrics(_score_pairs(expert, x, context, val_pairs, batch_size * 64), val_y)["auc"]
        if val_auc > best_auc:
            best_auc, best_epoch, best_state, stale = val_auc, epoch, _snapshot(expert), 0
        else:
            stale += 1
            if stale >= int(lcfg.expert_patience):
                break
    if best_state is None:  # val AUC undefined every epoch: keep the last state
        best_epoch, best_state = epoch, _snapshot(expert)
    expert.load_state_dict(best_state)

    val_prob = _score_pairs(expert, x, context, val_pairs, batch_size * 64)
    test_prob = _score_pairs(expert, x, context, views.pairs["test"].to(device), batch_size * 64)
    scores = ExpertScores(
        name=name,
        val_prob=val_prob,
        test_prob=test_prob,
        val_auc=probability_metrics(val_prob, val_y)["auc"],
        test_auc=probability_metrics(test_prob, views.labels["test"])["auc"],
        best_epoch=int(best_epoch),
        state=best_state,
    )
    print(f"[LinkMoE][{name}] best_epoch={best_epoch} val_auc={scores.val_auc:.4f} test_auc={scores.test_auc:.4f}")
    return scores


@torch.no_grad()
def _score_graphs(expert: nn.Module, graphs, batch_size: int, num_workers: int, device) -> torch.Tensor:
    expert.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = [torch.sigmoid(expert(batch.to(device))) for batch in loader]
    return torch.cat(out).float().cpu() if out else torch.empty(0)


def train_seal_expert(expert: nn.Module, seal_view: dict, views: LinkViews, cfg, device, seed: int) -> ExpertScores:
    """Train SEAL on ``seal_view['train']``; graphs of each split are aligned to ``views.pairs``."""
    lcfg = cfg.moe.linkmoe
    scfg = lcfg.seal
    set_seed(int(seed))
    expert = expert.to(device)
    batch_size, num_workers = int(scfg.batch_size), int(lcfg.num_workers)
    optimizer = torch.optim.Adam(expert.parameters(), lr=float(scfg.lr), weight_decay=float(scfg.weight_decay))
    loader = DataLoader(
        seal_view["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    val_y = views.labels["val"]

    best_auc, best_epoch, best_state, stale = float("-inf"), 0, None, 0
    for epoch in range(1, int(scfg.max_epochs) + 1):
        expert.train()
        for batch in loader:
            batch = batch.to(device)
            loss = F.binary_cross_entropy_with_logits(expert(batch), batch.y.view(-1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_auc = probability_metrics(_score_graphs(expert, seal_view["val"], batch_size, num_workers, device), val_y)["auc"]
        if val_auc > best_auc:
            best_auc, best_epoch, best_state, stale = val_auc, epoch, _snapshot(expert), 0
        else:
            stale += 1
            if stale >= int(scfg.patience):
                break
    if best_state is None:
        best_epoch, best_state = epoch, _snapshot(expert)
    expert.load_state_dict(best_state)

    val_prob = _score_graphs(expert, seal_view["val"], batch_size, num_workers, device)
    test_prob = _score_graphs(expert, seal_view["test"], batch_size, num_workers, device)
    scores = ExpertScores(
        name="seal",
        val_prob=val_prob,
        test_prob=test_prob,
        val_auc=probability_metrics(val_prob, val_y)["auc"],
        test_auc=probability_metrics(test_prob, views.labels["test"])["auc"],
        best_epoch=int(best_epoch),
        state=best_state,
    )
    print(f"[LinkMoE][seal] best_epoch={best_epoch} val_auc={scores.val_auc:.4f} test_auc={scores.test_auc:.4f}")
    return scores


__all__ = ["ExpertScores", "probability_metrics", "train_full_graph_expert", "train_seal_expert"]
