from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import TransformerConv

from .activations import get_activation
from src.utils.pool import get_pool_fn


class TransformerEncoder(nn.Module):
    """TransformerConv-based encoder that returns node and optional graph embeddings."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        act: str = "relu",
        graph_pooling: str = "mean",
    ):
        super().__init__()
        self.act = get_activation(act)
        self.dropout = nn.Dropout(dropout)
        self.pool = get_pool_fn(graph_pooling)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            out_c = out_dim if i == num_layers - 1 else hidden_dim
            self.convs.append(
                TransformerConv(
                    in_channels=in_c,
                    out_channels=out_c,
                    heads=heads,
                    dropout=dropout,
                    concat=False,
                )
            )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if idx != len(self.convs) - 1:
                x = self.act(x)
                x = self.dropout(x)

        node_repr = x
        graph_repr: Optional[torch.Tensor] = None
        if batch is not None:
            graph_repr = self.pool(node_repr, batch)
        return node_repr, graph_repr
