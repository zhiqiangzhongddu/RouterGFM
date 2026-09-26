"""Prompt-aware GAT (extension).

Formula:
    h_i^{l+1} = sigma( sum_j alpha_ij * W (h_j + p_ji) )

Attention coefficients alpha_ij are computed from the UNPROMPTED features
h_i, h_j.  Only the value/message path sees the edge prompt.

State-dict key names mirror PyG's ``GATConv(heads, concat=False)``:
    - ``lin.weight``
    - ``att_src``  shape [1, heads, out]
    - ``att_dst``  shape [1, heads, out]
    - ``bias`` (when ``bias=True``)

This lets a vanilla ``GATConv`` pretrain checkpoint load into
``PromptGATConv`` through the frozen-load whitelist without key
renaming.  The zero-prompt parity test enforces it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import (
    add_self_loops as add_self_loops_fn,
    remove_self_loops,
    softmax as scatter_softmax,
)

from src.model.activations import get_activation
from src.utils.pool import pool_nodes

from .base import PromptAwareEncoder


class PromptGATConv(nn.Module):
    """Single-conv GAT variant with prompt-conditioned value path.

    Matches the PyG ``GATConv`` state-dict surface so vanilla-GAT
    pretrained weights load in without key renaming.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = float(negative_slope)
        self.dropout = float(dropout)
        # Default add_self_loops=True matches PyG's ``GATConv(add_self_loops=True)``
        # default.  Vanilla-GAT pretrain checkpoints were trained with
        # self-loops in the edge set; omitting them here would silently
        # change the forward pass of a loaded pretrain.  The
        # vanilla-parity test locks this.
        self.add_self_loops = bool(add_self_loops)

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        if bias:
            bias_dim = heads * out_channels if concat else out_channels
            self.bias = nn.Parameter(torch.empty(bias_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot(self.lin.weight)
        glorot(self.att_src)
        glorot(self.att_dst)
        if self.bias is not None:
            zeros(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_prompt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        H, C = self.heads, self.out_channels
        N = x.size(0)

        # Unprompted projection used for attention scoring.
        x_lin = self.lin(x).view(N, H, C)
        alpha_src = (x_lin * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x_lin * self.att_dst).sum(dim=-1)  # [N, H]

        if self.add_self_loops:
            # Mirror PyG GATConv's default: remove then re-add self-loops.
            # The caller's prompt module must also use add_self_loops=True so
            # that its per-edge output aligns with this expanded edge_index.
            # ``spec._AUTO_ADD_SELF_LOOPS["gat"]`` pins the matching default.
            ei_no_loops, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops_fn(ei_no_loops, num_nodes=N)

        row, col = edge_index  # source -> target

        # Per-edge value: W(h_j + p_ji).  W is linear, so this equals
        # W(h_j) + W(p_ji).  W(h_j) is already cached in x_lin.
        val = x_lin[row]  # [E, H, C]
        if edge_prompt is not None:
            prompt_proj = self.lin(edge_prompt).view(-1, H, C)
            val = val + prompt_proj

        alpha = alpha_src[row] + alpha_dst[col]  # [E, H]
        alpha = F.leaky_relu(alpha, self.negative_slope)
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

        if self.bias is not None:
            out = out + self.bias
        return out


class PromptGATEncoder(PromptAwareEncoder):
    """Stacked ``PromptGATConv`` layers with the standard act/bn/dropout
    pipeline.  Mirrors the vanilla ``GNNEncoder`` layer structure for
    GAT so pretrained backbones load without key renaming."""

    edgeprompt_support = "extension"
    edgeprompt_formula = "attention from unprompted h; value = W(h_j + p_ji)"

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        gat_heads: int = 2,
        act: str = "relu",
        dropout: float = 0.1,
        graph_pooling: str = "mean",
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.act = get_activation(act)
        self.dropout = nn.Dropout(dropout)
        self.graph_pooling = graph_pooling
        self.use_batchnorm = bool(use_batchnorm)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for idx in range(num_layers):
            in_c = in_dim if idx == 0 else hidden_dim
            out_c = out_dim if idx == num_layers - 1 else hidden_dim
            self.convs.append(
                PromptGATConv(
                    in_channels=in_c,
                    out_channels=out_c,
                    heads=gat_heads,
                    concat=False,
                )
            )
            self.bns.append(nn.BatchNorm1d(out_c))

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
            raise ValueError("edge_index is required for EdgePrompt GAT encoder")

        for idx, conv in enumerate(self.convs):
            edge_prompt = None
            if prompt is not None and prompt_type in ("EdgePrompt", "EdgePromptplus"):
                edge_prompt = prompt.get_prompt(x, edge_index, layer=idx)
            x = conv(x, edge_index, edge_prompt=edge_prompt)
            if idx != len(self.convs) - 1:
                x = self.act(x)
                if self.use_batchnorm:
                    x = self.bns[idx](x)
                x = self.dropout(x)
            else:
                if self.use_batchnorm:
                    x = self.bns[idx](x)

        node_repr = x
        graph_repr = None
        if batch is not None:
            graph_repr = pool_nodes(node_repr, batch, mode=self.graph_pooling)
        return node_repr, graph_repr


def build_prompt_gat_encoder_from_cfg(cfg, in_dim: int) -> PromptGATEncoder:
    return PromptGATEncoder(
        in_dim=in_dim,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.out_dim,
        num_layers=cfg.model.num_layers,
        gat_heads=int(getattr(cfg.model.gat, "heads", 2)),
        act=cfg.model.activation,
        dropout=cfg.model.dropout,
        graph_pooling=cfg.model.graph_pooling,
        use_batchnorm=bool(getattr(cfg.model, "use_batchnorm", False)),
    )
