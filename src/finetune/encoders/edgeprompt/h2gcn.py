"""Prompt-aware H2GCN (extension).

Formula (preserves the repo's H2GCN stage structure):

    u_i^(1) = (1/|N_hat(i)|) sum_{j in N_hat(i)} (x_j + p_ji^(0))
    h_i^(1) = dropout( act( lin1(u_i^(1)) ) )

    u_i^(2) = (1/|N_hat(i)|) sum_{j in N_hat(i)} (h_j^(1) + p_ji^(1))
    h_i^(2) = dropout( act( lin2(u_i^(2)) ) )

    node_repr = out_lin( [h_i^(1) || h_i^(2)] )

Prompt enters only the aggregation inputs; the rest of the pipeline is
bit-identical to ``src/model/h2gcn.py`` so vanilla H2GCN pretrain
checkpoints load through the whitelist without key renaming.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import add_self_loops

from src.model.activations import get_activation
from src.utils.pool import get_pool_fn

from .base import PromptAwareEncoder


class PromptH2GCNEncoder(PromptAwareEncoder):
    """Repo-style 2-hop H2GCN with per-hop prompt injection."""

    edgeprompt_support = "extension"
    edgeprompt_formula = "prompted per-hop aggregation; lin/act/dropout/concat preserved"
    returns_layer_cache = True
    layer_cache_count = 2

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        act: str = "relu",
        dropout: float = 0.1,
        use_batchnorm: bool = False,
        graph_pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.act = get_activation(act)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batchnorm else None
        self.bn2 = nn.BatchNorm1d(hidden_dim) if use_batchnorm else None
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_lin = nn.Linear(2 * hidden_dim, out_dim)
        self.pool = get_pool_fn(graph_pooling)
        self.out_dim = out_dim
        self.cached_layer_node_reprs: list[torch.Tensor] = []

    @staticmethod
    def _aggregate_with_prompt(
        x: torch.Tensor,
        edge_index_with_self: torch.Tensor,
        deg: torch.Tensor,
        edge_prompt: torch.Tensor | None,
    ) -> torch.Tensor:
        row, col = edge_index_with_self
        src_feats = x[col]
        if edge_prompt is not None:
            src_feats = src_feats + edge_prompt
        out = torch.zeros_like(x)
        out.index_add_(0, row, src_feats)
        return out / deg.view(-1, 1)

    def forward(
        self,
        data,
        prompt=None,
        prompt_type: str | None = None,
    ):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        # Match vanilla H2GCN exactly: append one loop per node without
        # removing pre-existing loops. The H2 prompt spec uses the same policy.
        edge_with_self, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, _ = edge_with_self
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1).to(x.device)

        use_prompt = prompt is not None and prompt_type in ("EdgePrompt", "EdgePromptplus")

        prompt_0 = None
        if use_prompt:
            # Pass raw edge_index: EdgePromptPlus adds self-loops internally
            # when configured, producing the same [E+N] ordering as
            # ``add_self_loops`` above.  Passing edge_with_self here would
            # re-add self-loops and yield E+2N rows.
            prompt_0 = prompt.get_prompt(x, edge_index, layer=0)

        x1 = self._aggregate_with_prompt(x, edge_with_self, deg, prompt_0)
        x1 = self.lin1(x1)
        if self.bn1 is not None:
            x1 = self.bn1(x1)
        x1 = self.act(x1)
        x1 = self.dropout(x1)

        prompt_1 = None
        if use_prompt:
            prompt_1 = prompt.get_prompt(x1, edge_index, layer=1)

        x2 = self._aggregate_with_prompt(x1, edge_with_self, deg, prompt_1)
        x2 = self.lin2(x2)
        if self.bn2 is not None:
            x2 = self.bn2(x2)
        x2 = self.act(x2)
        x2 = self.dropout(x2)

        # Ordinary attributes only: the cache follows the existing encoder
        # capability contract without adding buffers or checkpoint keys.
        self.cached_layer_node_reprs = [x1, x2]
        h = torch.cat([x1, x2], dim=-1)
        node_repr = self.out_lin(h)

        graph_repr = None
        if batch is not None:
            graph_repr = self.pool(node_repr, batch)
        return node_repr, graph_repr

    def get_layer_node_reprs(self) -> list[torch.Tensor]:
        """Return the one-hop and two-hop states from the latest forward."""
        return self.cached_layer_node_reprs


def build_prompt_h2gcn_encoder_from_cfg(cfg, in_dim: int) -> PromptH2GCNEncoder:
    return PromptH2GCNEncoder(
        in_dim=in_dim,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.out_dim,
        act=cfg.model.activation,
        dropout=cfg.model.dropout,
        use_batchnorm=(
            bool(getattr(cfg.model, "use_batchnorm", False))
            or bool(getattr(getattr(cfg.model, "h2gcn", None), "use_batchnorm", False))
        ),
        graph_pooling=cfg.model.graph_pooling,
    )
