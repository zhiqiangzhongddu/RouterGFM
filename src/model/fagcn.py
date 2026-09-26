from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import FAConv

from .activations import get_activation
from src.utils.pool import get_pool_fn


class FAGCNEncoder(nn.Module):
    """Lightweight FAGCN encoder with configurable depth."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        act: str = "relu",
        dropout: float = 0.1,
        eps: float = 0.1,
        graph_pooling: str = "mean",
        use_batchnorm: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.act = get_activation(act)
        self.dropout = nn.Dropout(dropout)
        self.pool = get_pool_fn(graph_pooling)
        self.use_batchnorm = bool(use_batchnorm)

        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            # FAConv uses a single channels arg (in == out), so we keep hidden_dim inside the stack.
            self.convs.append(FAConv(channels=hidden_dim, eps=eps, dropout=dropout))
            if self.use_batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.out_lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        x = self.lin_in(x)
        x0 = x  # FAConv expects both current features and the initial features
        for idx, conv in enumerate(self.convs):
            x = conv(x, x0, edge_index)
            if self.use_batchnorm:
                x = self.bns[idx](x)
            if idx != len(self.convs) - 1:
                x = self.act(x)
                x = self.dropout(x)
        x = self.out_lin(x)

        node_repr = x
        graph_repr: Optional[torch.Tensor] = None
        if batch is not None:
            graph_repr = self.pool(node_repr, batch)
        return node_repr, graph_repr
