"""Gating: per-sample dispersion of the weak expert and the routing weight.

The gate decides, per sample (a subgraph in the framework's induced-batch
setting), how much to trust the weak expert. Its signal is the *dispersion* of
the weak expert's prediction — how spread-out / unconfident it is — measured two
ways and concatenated into an ``[M, 2]`` vector, optionally followed by the weak
expert's pooled embedding when ``original_data`` is set.

For single-label multi-class classification the dispersion is exactly the
reference's softmax variance ⊕ entropy. Because the framework also feeds binary,
multi-task-binary, and regression datasets through the same path, the dispersion
is computed task-type-aware (see :func:`compute_dispersion`).

The dispersion is always computed from ``weak_logits.detach()`` so that during
training gradients reach the weak expert only through the direct loss term —
never through the gate's routing coefficient. This matches the reference, whose
``compute_confidence`` is wrapped in ``@torch.no_grad()``.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from src.model.activations import get_activation


_LN2 = math.log(2.0)


def _bernoulli_confidence(probs: torch.Tensor) -> torch.Tensor:
    """Per-output Bernoulli [variance, entropy] confidence, averaged over outputs.

    *probs* are sigmoid probabilities of shape ``[M, L]``. Returns ``[M, 2]``:
    a normalised peakedness ``(2p-1)^2`` (1 = certain) and ``1 - H_b(p)/ln2``
    (1 = certain), averaged across the ``L`` outputs.
    """
    p = probs.clamp(min=1e-12, max=1.0 - 1e-12)
    var_conf = (2.0 * p - 1.0).pow(2)  # in [0, 1]; matches 2-class softmax-variance form
    entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))  # natural-log Bernoulli entropy
    ent_conf = 1.0 - entropy / _LN2
    return torch.stack((var_conf.mean(dim=1), ent_conf.mean(dim=1)), dim=1)


def compute_dispersion(logits: torch.Tensor, task_type: str, label_dim: int) -> torch.Tensor:
    """Per-sample ``[M, 2]`` dispersion of the weak expert's prediction.

    * ``regression`` — no probabilistic dispersion; returns zeros so the gate
      routes from its bias (and the pooled embedding when ``original_data``).
    * multi-task binary (``label_dim > 1``) and single-logit binary — averaged
      Bernoulli confidence (:func:`_bernoulli_confidence`).
    * single-label multi-class (``logits`` width ``C >= 2``) — softmax variance
      normalised by the one-hot maximum, and ``1 - entropy / log(C)`` (the
      reference's exact dispersion).
    """
    task = str(task_type or "classification").lower()
    if task == "regression":
        return torch.zeros(logits.size(0), 2, device=logits.device, dtype=logits.dtype)

    if int(label_dim) > 1 or logits.size(-1) == 1:
        probs = torch.sigmoid(logits.float()).view(logits.size(0), -1)
        return _bernoulli_confidence(probs).to(logits.dtype)

    n_classes = logits.size(-1)
    probs = torch.softmax(logits, dim=1)
    variance = torch.var(probs, dim=1, unbiased=False)
    one_hot = torch.zeros(n_classes, device=logits.device, dtype=probs.dtype)
    one_hot[0] = 1.0
    max_variance = torch.var(one_hot, unbiased=False)
    var_conf = variance / max_variance
    log_probs = torch.log(probs.clamp_min(1e-12))
    entropy = -(probs * log_probs).sum(dim=1)
    ent_conf = 1.0 - entropy / math.log(n_classes)
    return torch.stack((var_conf, ent_conf), dim=1)


def compute_gating(
    gate: nn.Module,
    weak_logits: torch.Tensor,
    weak_feat: torch.Tensor | None,
    *,
    task_type: str,
    label_dim: int,
    original_data: bool,
) -> torch.Tensor:
    """Per-sample routing weight on the weak expert, an ``[M, 1]`` tensor in (0, 1).

    The gate consumes the weak expert's (detached) dispersion, optionally
    concatenated with its pooled embedding ``weak_feat`` when ``original_data``.
    """
    disp = compute_dispersion(weak_logits.detach(), task_type, label_dim)
    if original_data:
        if weak_feat is None:
            raise ValueError("original_data=True requires the weak expert's pooled embedding.")
        gate_input = torch.cat((disp, weak_feat.detach()), dim=1)
    else:
        gate_input = disp
    return torch.sigmoid(gate(gate_input))


def gate_input_dim(*, feature_dim: int, original_data: bool) -> int:
    """Gate input width: 2 dispersion channels, plus the embedding when enabled."""
    return 2 + (int(feature_dim) if original_data else 0)


class GateMLP(nn.Module):
    """Per-sample gate: maps a feature vector to one pre-sigmoid routing logit.

    Mirrors the reference ``GateMLP`` (Linear → BatchNorm → ReLU → Dropout
    stack, final Linear to a single unit). The sigmoid is applied by
    :func:`compute_gating`. ``num_layers == 1`` yields a 2-layer net
    (input → hidden → 1), as in the reference, not a bare Linear.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        act: str,
        use_batchnorm: bool,
    ):
        super().__init__()
        assert num_layers >= 1, "gate num_layers must be >= 1"
        self.act = get_activation(act)
        self.dropout = float(dropout)
        self.use_batchnorm = bool(use_batchnorm)

        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lins.append(nn.Linear(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.lins.append(nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.use_batchnorm:
                x = self.bns[idx](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lins[-1](x)

    def reset_parameters(self) -> None:
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


__all__ = ["GateMLP", "compute_dispersion", "compute_gating", "gate_input_dim"]
