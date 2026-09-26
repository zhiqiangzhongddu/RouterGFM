"""OGMM stage 1: label-conditional graph generation by inverting a frozen expert (Sec. 3.2, Eqs. 7-11).

Per expert, node features ``X`` and an edge encoder are learned so that the
expert assigns the sampled labels ``y_hat`` to the generated graphs:
``L_gen = C(y_hat, f(G)) + R_bn + R_conf`` (unit weights).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data

from src.utils.supervised_loss import binary_targets_and_valid, supervised_loss_from_logits

from .experts import DenseExpert, DenseInstances


class EdgeEncoder(nn.Module):
    """Eq. 7: ``a_jk = sigmoid(MLP([x_j ; x_k]))`` with a three-layer MLP, read on pairs ``j < k``."""

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.in_dim = int(in_dim)
        self.lin1 = nn.Linear(2 * self.in_dim, int(hidden_dim))
        self.lin2 = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.lin3 = nn.Linear(int(hidden_dim), 1)

    def pair_logits(self, x: Tensor) -> Tensor:
        """``[G, n, n]`` logits; entry ``(j, k)`` scores ``[x_j ; x_k]``."""
        # lin1 on the concatenation, split into its x_j and x_k blocks so the
        # [G, n, n, 2d] pair tensor is never materialised.
        weight = self.lin1.weight
        h = (x @ weight[:, : self.in_dim].T).unsqueeze(2) + (x @ weight[:, self.in_dim:].T).unsqueeze(1) + self.lin1.bias
        h = F.relu(self.lin2(F.relu(h)))
        return self.lin3(h).squeeze(-1)

    @staticmethod
    def _symmetric(values: Tensor, forbid_pair: Optional[Tuple[int, int]]) -> Tensor:
        n = values.size(-1)
        upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=values.device), diagonal=1)
        if forbid_pair is not None:
            upper[min(forbid_pair), max(forbid_pair)] = False
        values = values * upper
        return values + values.transpose(-1, -2)

    def edge_probs(self, x: Tensor, forbid_pair: Optional[Tuple[int, int]] = None) -> Tensor:
        """Symmetric edge probabilities with a zero diagonal (and a zero forbidden pair)."""
        return self._symmetric(torch.sigmoid(self.pair_logits(x)), forbid_pair)

    def sample(
        self,
        x: Tensor,
        tau: float,
        forbid_pair: Optional[Tuple[int, int]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Eq. 8, binary Gumbel-softmax: ``sigmoid((log a - log(1 - a) + g1 - g2) / tau)``."""
        logits = self.pair_logits(x)
        uniform = torch.rand((2,) + tuple(logits.shape), generator=generator).to(logits.device)
        gumbel = -torch.log(-torch.log(uniform.clamp(1e-10, 1.0 - 1e-10)))
        relaxed = torch.sigmoid((logits + gumbel[0] - gumbel[1]) / float(tau))
        return self._symmetric(relaxed, forbid_pair)


def bn_stat_loss(captured: list[Tensor], bns: list[nn.BatchNorm1d]) -> Tensor:
    """Eq. 9: ``sum_L ||mu_L - E[mu_L]||^2 + ||sigma^2_L - E[sigma^2_L]||^2`` against running stats."""
    if len(captured) != len(bns):
        raise ValueError(f"{len(captured)} captured BatchNorm inputs for {len(bns)} layers.")
    loss = torch.zeros((), device=bns[0].running_mean.device) if bns else torch.zeros(())
    for inputs, bn in zip(captured, bns):
        mean = inputs.mean(dim=0)
        var = inputs.var(dim=0, unbiased=False)
        loss = loss + (mean - bn.running_mean).pow(2).sum() + (var - bn.running_var).pow(2).sum()
    return loss


def confidence_loss(logits: Tensor, task_type: str, label_dim: int) -> Tensor:
    """Eq. 10: prediction entropy (binary entropy for single-logit / multi-label heads; 0 for regression)."""
    if str(task_type).lower() == "regression":
        return logits.sum() * 0.0
    logits = logits.float()
    if logits.size(-1) == 1 or int(label_dim or 1) > 1:
        p = torch.sigmoid(logits)
        return -(p * F.logsigmoid(logits) + (1.0 - p) * F.logsigmoid(-logits)).mean()
    log_p = F.log_softmax(logits, dim=-1)
    return -(log_p.exp() * log_p).sum(dim=-1).mean()


def sample_generated_labels(
    task_type: str,
    num: int,
    *,
    num_classes: int,
    label_dim: int,
    domain_targets: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Conditional labels ``y_hat``.

    Single-label and link prediction: uniform classes (paper). Regression:
    ``U[min, max]`` of the domain's support targets. Multi-label:
    ``Bernoulli(p_t)``, the observed positive rate of each assay in the domain
    (0.5 when unobserved). The last two are PROPOSED.
    """
    if str(task_type).lower() == "regression":
        y = torch.as_tensor(domain_targets).float().reshape(len(domain_targets), -1)
        finite = torch.isfinite(y)
        low = torch.where(finite, y, torch.full_like(y, float("inf"))).min(dim=0).values
        high = torch.where(finite, y, torch.full_like(y, float("-inf"))).max(dim=0).values
        low, high = torch.nan_to_num(low, posinf=0.0), torch.nan_to_num(high, neginf=0.0)
        return low + (high - low) * torch.rand(int(num), y.size(1), generator=generator)
    if int(label_dim or 1) > 1:
        target, valid = binary_targets_and_valid(torch.as_tensor(domain_targets).reshape(len(domain_targets), -1))
        observed = valid.sum(dim=0)
        positives = torch.where(valid, target, torch.zeros_like(target)).sum(dim=0)
        rate = torch.where(observed > 0, positives / observed.clamp(min=1), torch.full_like(positives, 0.5))
        return torch.bernoulli(rate.expand(int(num), -1).contiguous(), generator=generator)
    return torch.randint(0, max(int(num_classes or 2), 2), (int(num),), generator=generator)


def _generated_anchor(level: str, num: int, device) -> Tensor:
    if level == "node":
        return torch.zeros(num, 2, dtype=torch.long, device=device)
    if level == "edge":
        return torch.tensor([[0, 1]], dtype=torch.long, device=device).expand(num, 2).contiguous()
    return torch.full((num, 2), -1, dtype=torch.long, device=device)


def generate_for_expert(
    expert: DenseExpert,
    *,
    num_graphs: int,
    num_nodes: int,
    in_dim: int,
    task_level_raw: str,
    task_type: str,
    label_dim: int,
    num_classes: int,
    domain_targets: Tensor,
    epochs: int,
    lr: float,
    tau: float,
    edge_threshold: float,
    generator: torch.Generator,
    edge_hidden_dim: int = 64,
) -> list[Data]:
    """Invert a frozen expert into ``num_graphs`` labelled graphs of ``num_nodes`` nodes.

    Node tasks: node 0 is the target. Link prediction: nodes 0 and 1 are the
    endpoints and their pair never gets an edge (SEAL target removal). The
    stored graphs keep the learned features and ``1[a_jk > edge_threshold]``.
    """
    level = str(task_level_raw).lower()
    device = next(expert.parameters()).device
    num_graphs, num_nodes = int(num_graphs), int(num_nodes)
    forbid_pair = (0, 1) if level == "edge" else None

    labels = sample_generated_labels(
        task_type, num_graphs, num_classes=num_classes, label_dim=label_dim,
        domain_targets=domain_targets, generator=generator,
    ).to(device)
    x = torch.randn(num_graphs, num_nodes, int(in_dim), generator=generator).to(device).requires_grad_(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))
        encoder = EdgeEncoder(int(in_dim), int(edge_hidden_dim)).to(device)
    mask = torch.ones(num_graphs, num_nodes, dtype=torch.bool, device=device)
    anchor = _generated_anchor(level, num_graphs, device)
    optimizer = torch.optim.AdamW([x, *encoder.parameters()], lr=float(lr))

    was_training = expert.training
    grad_flags = [p.requires_grad for p in expert.parameters()]
    expert.eval()
    expert.requires_grad_(False)
    bns = expert.bn_layers()
    captured: list[Tensor] = []
    hooks = [bn.register_forward_hook(lambda _m, inputs, _out: captured.append(inputs[0])) for bn in bns]
    parts = {}
    try:
        for _ in range(int(epochs)):
            captured.clear()
            optimizer.zero_grad()
            adj = encoder.sample(x, tau, forbid_pair=forbid_pair, generator=generator)
            logits = expert(DenseInstances(x=x, adj=adj, mask=mask, anchor=anchor, y=labels))
            task_loss, _ = supervised_loss_from_logits(logits=logits, labels=labels, task_type=task_type)
            r_bn = bn_stat_loss(captured, bns)
            r_conf = confidence_loss(logits, task_type, label_dim)
            (task_loss + r_bn + r_conf).backward()
            optimizer.step()
            parts = {"task": float(task_loss), "bn": float(r_bn), "conf": float(r_conf)}
    finally:
        for hook in hooks:
            hook.remove()
        for param, flag in zip(expert.parameters(), grad_flags):
            param.requires_grad_(flag)
        expert.train(was_training)

    with torch.no_grad():
        hard = encoder.edge_probs(x, forbid_pair) > float(edge_threshold)
    features, labels = x.detach().cpu(), labels.cpu()
    graphs = []
    for g in range(num_graphs):
        y = labels[g].view(1) if labels.dim() == 1 else labels[g].view(1, -1)
        data = Data(x=features[g], edge_index=hard[g].nonzero().t().contiguous().cpu(), y=y)
        if level == "node":
            data.target_node_index = torch.tensor([0])
        elif level == "edge":
            data.edge_label_index = torch.tensor([[0], [1]])
        graphs.append(data)
    if parts:
        print(
            f"[OGMM][Gen] {expert.arch}: {num_graphs} graphs x {num_nodes} nodes, "
            f"final task={parts['task']:.4f} bn={parts['bn']:.4f} conf={parts['conf']:.4f}"
        )
    return graphs


__all__ = [
    "EdgeEncoder",
    "bn_stat_loss",
    "confidence_loss",
    "generate_for_expert",
    "sample_generated_labels",
]
