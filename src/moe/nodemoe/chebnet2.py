"""ChebNetII expert for Node-MoE (pure modules, no cfg).

Port of the official ChebNetII code (``ivam-he/ChebNetII``:
``ChebnetII_prop`` + ``ChebNetII``). The filter values ``temp`` (gamma_j,
passed through ReLU) sit at the Chebyshev nodes ``x_j`` ordered from low
(lambda = x + 1 ~ 0) to high (lambda ~ 2) frequency; the propagation is the
Chebyshev interpolant of those values applied to ``L_hat = L_sym - I``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian

FILTER_INITS = ("low", "high", "uniform")


def chebyshev_nodes(K: int) -> torch.Tensor:
    """x_j = cos((K - j + 0.5) * pi / (K + 1)), j = 0..K (ascending; lambda = x + 1)."""
    j = torch.arange(K + 1, dtype=torch.float64)
    return torch.cos((K - j + 0.5) * math.pi / (K + 1)).float()


def chebyshev_basis(x: torch.Tensor, K: int) -> torch.Tensor:
    """``[len(x), K+1]`` matrix of T_k(x), k = 0..K (three-term recurrence)."""
    x = x.reshape(-1)
    terms = [torch.ones_like(x)]
    if K >= 1:
        terms.append(x)
    for _ in range(2, K + 1):
        terms.append(2.0 * x * terms[-1] - terms[-2])
    return torch.stack(terms, dim=-1)


def init_filter_values(kind: str, K: int, alpha: float) -> torch.Tensor:
    """Filter init (App. C.2): low ``alpha**j``; high ``alpha**(K-j)``; uniform ones."""
    key = str(kind).lower()
    j = torch.arange(K + 1, dtype=torch.float64)
    if key == "low":
        values = float(alpha) ** j
    elif key == "high":
        values = float(alpha) ** (K - j)
    elif key == "uniform":
        values = torch.ones(K + 1, dtype=torch.float64)
    else:
        raise ValueError(f"Unknown Node-MoE filter init '{kind}'; expected one of {FILTER_INITS}.")
    return values.float()


class ChebIIProp(MessagePassing):
    """ChebNetII propagation with a learnable K-order filter."""

    def __init__(self, K: int = 10, init: str = "uniform", alpha: float = 0.9):
        super().__init__(aggr="add")
        if int(K) < 1:
            raise ValueError(f"ChebNetII order K must be >= 1 (got {K}).")
        self.K = int(K)
        self.temp = nn.Parameter(init_filter_values(init, self.K, alpha))
        # basis[j, k] = T_k(x_j); fixed, so kept out of the state dict.
        self.register_buffer("basis", chebyshev_basis(chebyshev_nodes(self.K), self.K), persistent=False)

    def coefficients(self) -> torch.Tensor:
        """c_k = 2/(K+1) * sum_j relu(temp_j) T_k(x_j), k = 0..K."""
        return (2.0 / (self.K + 1)) * (F.relu(self.temp) @ self.basis)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        coe = self.coefficients()
        num_nodes = x.size(0)
        # L = I - D^-1/2 A D^-1/2, then L_hat = L - I (duplicate self loops sum to 0).
        lap_index, lap_weight = get_laplacian(edge_index, normalization="sym", dtype=x.dtype, num_nodes=num_nodes)
        lap_index, lap_weight = add_self_loops(lap_index, lap_weight, fill_value=-1.0, num_nodes=num_nodes)

        tx_0 = x
        tx_1 = self.propagate(lap_index, x=x, norm=lap_weight)
        out = coe[0] / 2 * tx_0 + coe[1] * tx_1
        for k in range(2, self.K + 1):
            tx_2 = 2 * self.propagate(lap_index, x=tx_1, norm=lap_weight) - tx_0
            out = out + coe[k] * tx_2
            tx_0, tx_1 = tx_1, tx_2
        return out

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j

    def response(self, lam: torch.Tensor) -> torch.Tensor:
        """Filter response f(lam) = c_0/2 + sum_{k>=1} c_k T_k(lam - 1), lam in [0, 2]."""
        coe = self.coefficients()
        half_first = torch.ones_like(coe)
        half_first[0] = 0.5
        lam = torch.as_tensor(lam, dtype=coe.dtype, device=coe.device)
        return chebyshev_basis(lam - 1.0, self.K) @ (coe * half_first)

    def smoothing_loss(self) -> torch.Tensor:
        """Eq. 2: sum_j (f(x_j) - f(x_{j-1}))^2, with f(x_j) = relu(temp_j) at the nodes."""
        values = F.relu(self.temp)
        return (values[1:] - values[:-1]).pow(2).sum()


class ChebNetIIExpert(nn.Module):
    """ChebNetII: ``prop(dropout_dprate(lin2(dropout(relu(lin1(dropout(x)))))))``; returns logits."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        K: int,
        init: str,
        alpha: float,
        dropout: float,
        dprate: float,
    ):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.prop = ChebIIProp(K=K, init=init, alpha=alpha)
        self.dropout = float(dropout)
        self.dprate = float(dprate)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dprate, training=self.training)
        # No log_softmax: Node-MoE mixes expert logits before the softmax head.
        return self.prop(x, edge_index)

    def filter_parameters(self) -> list[nn.Parameter]:
        return [self.prop.temp]

    def dense_parameters(self) -> list[nn.Parameter]:
        return [*self.lin1.parameters(), *self.lin2.parameters()]


__all__ = [
    "FILTER_INITS",
    "ChebIIProp",
    "ChebNetIIExpert",
    "chebyshev_basis",
    "chebyshev_nodes",
    "init_filter_values",
]
