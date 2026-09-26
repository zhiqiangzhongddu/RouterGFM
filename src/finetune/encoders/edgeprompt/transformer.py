"""Prompt-aware Transformer (extension).

Formula:
    v_ji^(l)   = W_V (h_j + p_ji)
    h_i^(l+1)  = sum_j alpha_ij * v_ji,  alpha_ij from unprompted (h_i, h_j)
    h_i^(l+1) += W_O h_i   (skip projection)

Reimplements the surface of PyG's ``TransformerConv(concat=False)`` --
same state_dict key names (``lin_query``, ``lin_key``, ``lin_value``,
``lin_skip``) so vanilla transformer pretrain checkpoints load without
key renaming.  ``edge_attr`` / ``beta`` / ``root_weight=False`` paths
of the original are intentionally not carried over: they would drag
extra state_dict keys that the vanilla encoder never uses.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import softmax as scatter_softmax

from src.model.activations import get_activation
from src.utils.pool import pool_nodes

from .base import PromptAwareEncoder


class PromptTransformerConv(nn.Module):
    """Single TransformerConv-style layer with prompted value path.

    Key layout mirrors PyG's ``TransformerConv(heads, concat=False,
    root_weight=True, beta=False, edge_dim=None, bias=True)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = float(dropout)

        self.lin_query = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_key = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_value = nn.Linear(in_channels, heads * out_channels, bias=bias)
        skip_out = heads * out_channels if concat else out_channels
        self.lin_skip = nn.Linear(in_channels, skip_out, bias=bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_query.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_skip.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_prompt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        H, C = self.heads, self.out_channels
        N = x.size(0)

        query = self.lin_query(x).view(N, H, C)
        key = self.lin_key(x).view(N, H, C)
        value = self.lin_value(x).view(N, H, C)

        row, col = edge_index  # source -> target
        q_t = query[col]  # [E, H, C]
        k_s = key[row]    # [E, H, C]
        val = value[row]  # [E, H, C]
        if edge_prompt is not None:
            # v_ji = W_V(h_j + p_ji) = (W_V h_j + b_V) + W_V p_ji.
            # ``val`` already contains W_V h_j + b_V (from lin_value above),
            # so we must apply W_V WITHOUT the bias here -- otherwise a
            # zero-valued prompt would still add b_V a second time.
            val = val + F.linear(edge_prompt, self.lin_value.weight).view(-1, H, C)

        alpha = (q_t * k_s).sum(dim=-1) / math.sqrt(C)  # [E, H]
        alpha = scatter_softmax(alpha, col, num_nodes=N)
        if self.dropout > 0 and self.training:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        weighted = val * alpha.unsqueeze(-1)  # [E, H, C]
        out = torch.zeros(N, H, C, device=x.device, dtype=weighted.dtype)
        out.scatter_add_(0, col.view(-1, 1, 1).expand_as(weighted), weighted)

        if self.concat:
            out = out.reshape(N, H * C)
        else:
            out = out.mean(dim=1)

        out = out + self.lin_skip(x)  # skip projection
        return out


class PromptTransformerEncoder(PromptAwareEncoder):
    """Stacked ``PromptTransformerConv`` layers mirroring the vanilla
    ``TransformerEncoder`` for state-dict compatibility."""

    edgeprompt_support = "extension"
    edgeprompt_formula = "value = W_V(h_j + p_ji); query/key unchanged"

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
    ) -> None:
        super().__init__()
        self.act = get_activation(act)
        self.dropout = nn.Dropout(dropout)
        self.graph_pooling = graph_pooling

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            out_c = out_dim if i == num_layers - 1 else hidden_dim
            self.convs.append(
                PromptTransformerConv(
                    in_channels=in_c,
                    out_channels=out_c,
                    heads=heads,
                    dropout=dropout,
                    concat=False,
                )
            )
        self.out_dim = out_dim

    def forward(
        self,
        data,
        prompt=None,
        prompt_type: str | None = None,
    ):
        x, edge_index = data.x, getattr(data, "edge_index", None)
        batch = getattr(data, "batch", None)
        if edge_index is None:
            raise ValueError("edge_index is required for EdgePrompt transformer encoder")

        use_prompt = prompt is not None and prompt_type in ("EdgePrompt", "EdgePromptplus")
        for idx, conv in enumerate(self.convs):
            edge_prompt = None
            if use_prompt:
                edge_prompt = prompt.get_prompt(x, edge_index, layer=idx)
            x = conv(x, edge_index, edge_prompt=edge_prompt)
            if idx != len(self.convs) - 1:
                x = self.act(x)
                x = self.dropout(x)

        node_repr = x
        graph_repr = None
        if batch is not None:
            graph_repr = pool_nodes(node_repr, batch, mode=self.graph_pooling)
        return node_repr, graph_repr


def build_prompt_transformer_encoder_from_cfg(cfg, in_dim: int) -> PromptTransformerEncoder:
    heads = int(getattr(cfg.model.gat, "heads", 4))
    return PromptTransformerEncoder(
        in_dim=in_dim,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.out_dim,
        num_layers=cfg.model.num_layers,
        heads=heads,
        dropout=cfg.model.dropout,
        act=cfg.model.activation,
        graph_pooling=cfg.model.graph_pooling,
    )
