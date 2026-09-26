from typing import Optional

import torch
from torch import nn
from torch_geometric.utils import add_self_loops

from .activations import get_activation
from src.utils.pool import get_pool_fn


class H2GCNEncoder(nn.Module):
    """
    Minimal H2GCN-style encoder: combines 1-hop and 2-hop neighborhood features.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        act: str = "relu",
        dropout: float = 0.1,
        use_batchnorm: bool = False,
        graph_pooling: str = "mean",
    ):
        super().__init__()
        self.act = get_activation(act)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batchnorm else None
        self.bn2 = nn.BatchNorm1d(hidden_dim) if use_batchnorm else None
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        # second hop consumes hidden_dim features, not raw in_dim
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_lin = nn.Linear(2 * hidden_dim, out_dim)
        self.pool = get_pool_fn(graph_pooling)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 1-hop aggregation
        edge_with_self, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_with_self
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1).to(x.device)
        x1 = torch.zeros_like(x)
        x1.index_add_(0, row, x[col])
        x1 = x1 / deg.view(-1, 1)
        x1 = self.lin1(x1)
        if self.bn1:
            x1 = self.bn1(x1)
        x1 = self.act(x1)
        x1 = self.dropout(x1)

        # 2-hop aggregation (apply 1-hop again)
        x2 = torch.zeros_like(x1)
        x2.index_add_(0, row, x1[col])
        x2 = x2 / deg.view(-1, 1)
        x2 = self.lin2(x2)
        if self.bn2:
            x2 = self.bn2(x2)
        x2 = self.act(x2)
        x2 = self.dropout(x2)

        h = torch.cat([x1, x2], dim=-1)
        node_repr = self.out_lin(h)

        batch = getattr(data, "batch", None)
        graph_repr: Optional[torch.Tensor] = None
        if batch is not None:
            graph_repr = self.pool(node_repr, batch)
        return node_repr, graph_repr
