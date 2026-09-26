"""Prompt-aware FAGCN (extension).

Formula (per layer):
    g_ij = tanh( a_l^T h_i + a_r^T h_j )      # gate from unprompted h
    h_i^(l+1) = eps * h^(0) + sum_j norm_ij * g_ij * (h_j + p_ji)

``norm_ij`` is the GCN normalization ``d_hat(i)^{-1/2} d_hat(j)^{-1/2}``
with self-loops added (mirrors PyG's ``FAConv`` default).  The gate is
computed from the UNPROMPTED node features so the prompt only enters
the message path.

State-dict keys mirror PyG's ``FAConv``: ``att_l.weight`` and
``att_r.weight`` (both ``Linear(channels, 1, bias=False)``).  The
encoder shell adds ``lin_in``, ``out_lin`` mirroring
``src/model/fagcn.py`` so vanilla FAGCN pretrain checkpoints load
without renaming.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import remove_self_loops

from src.model.activations import get_activation
from src.utils.pool import get_pool_fn

from .base import PromptAwareEncoder


class PromptFAConv(nn.Module):
    """FAConv with prompt-conditioned message path.

    Matches PyG ``FAConv`` state-dict: ``att_l.weight``, ``att_r.weight``.
    """

    def __init__(
        self,
        channels: int,
        eps: float = 0.1,
        dropout: float = 0.0,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.eps = float(eps)
        self.dropout = float(dropout)
        self.add_self_loops = bool(add_self_loops)
        self.att_l = nn.Linear(channels, 1, bias=False)
        self.att_r = nn.Linear(channels, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.att_l.reset_parameters()
        self.att_r.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        x_0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_prompt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N = x.size(0)
        if self.add_self_loops:
            # remove pre-existing self-loops so gcn_norm's add_self_loops
            # stage does not produce duplicates (would desynchronize from
            # the prompt module's remove+add).
            edge_index, _ = remove_self_loops(edge_index)
        ei, edge_weight = gcn_norm(
            edge_index, None, N, False, self.add_self_loops, "source_to_target", dtype=x.dtype
        )
        row, col = ei

        # FAConv conventions (PyG, flow="source_to_target"):
        #   alpha_j  -> sourced via ``alpha_l`` = att_l(x)[source] = att_l(x)[row]
        #   alpha_i  -> sourced via ``alpha_r`` = att_r(x)[target] = att_r(x)[col]
        #   gate = tanh(alpha_j + alpha_i) = tanh(att_l(x)[row] + att_r(x)[col])
        # Swapping the indices silently corrupts any pretrained FAConv
        # weights -- att_l would be applied to the target side and vice
        # versa.  The vanilla-parity test locks this ordering.
        alpha_l = self.att_l(x).squeeze(-1)  # [N]
        alpha_r = self.att_r(x).squeeze(-1)  # [N]
        gate = torch.tanh(alpha_l[row] + alpha_r[col])  # [E]
        if self.dropout > 0 and self.training:
            gate = nn.functional.dropout(gate, p=self.dropout, training=True)

        msg_src = x[row]  # [E, C]
        if edge_prompt is not None:
            msg_src = msg_src + edge_prompt

        weighted = msg_src * (gate * edge_weight).view(-1, 1)  # [E, C]
        out = torch.zeros_like(x)
        out.scatter_add_(0, col.view(-1, 1).expand_as(weighted), weighted)

        if self.eps != 0.0:
            out = out + self.eps * x_0
        return out


class PromptFAGCNEncoder(PromptAwareEncoder):
    """FAGCN encoder with prompt-conditioned per-layer message path.

    Mirrors ``src/model/fagcn.py`` (``lin_in`` -> num_layers x FAConv
    -> ``out_lin``) so a vanilla FAGCN pretrain checkpoint loads
    without key renaming.
    """

    edgeprompt_support = "extension"
    edgeprompt_formula = "adaptive gate from unprompted features; message on (h_j + p_ji)"

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
    ) -> None:
        super().__init__()
        assert num_layers >= 1
        self.act = get_activation(act)
        self.dropout = nn.Dropout(dropout)
        self.pool = get_pool_fn(graph_pooling)
        self.use_batchnorm = bool(use_batchnorm)

        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            PromptFAConv(channels=hidden_dim, eps=eps, dropout=dropout)
            for _ in range(num_layers)
        )
        self.bns = nn.ModuleList(
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ) if self.use_batchnorm else nn.ModuleList()
        self.out_lin = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(
        self,
        data,
        prompt=None,
        prompt_type: str | None = None,
    ):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        x = self.lin_in(x)
        x0 = x
        use_prompt = prompt is not None and prompt_type in ("EdgePrompt", "EdgePromptplus")

        for idx, conv in enumerate(self.convs):
            edge_prompt = None
            if use_prompt:
                edge_prompt = prompt.get_prompt(x, edge_index, layer=idx)
            x = conv(x, x0, edge_index, edge_prompt=edge_prompt)
            if self.use_batchnorm:
                x = self.bns[idx](x)
            if idx != len(self.convs) - 1:
                x = self.act(x)
                x = self.dropout(x)

        x = self.out_lin(x)
        node_repr = x
        graph_repr = None
        if batch is not None:
            graph_repr = self.pool(node_repr, batch)
        return node_repr, graph_repr


def build_prompt_fagcn_encoder_from_cfg(cfg, in_dim: int) -> PromptFAGCNEncoder:
    fagcn_cfg = getattr(cfg.model, "fagcn", None)
    eps = float(getattr(fagcn_cfg, "eps", 0.1)) if fagcn_cfg is not None else 0.1
    return PromptFAGCNEncoder(
        in_dim=in_dim,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.out_dim,
        num_layers=cfg.model.num_layers,
        act=cfg.model.activation,
        dropout=cfg.model.dropout,
        eps=eps,
        graph_pooling=cfg.model.graph_pooling,
        use_batchnorm=(
            bool(getattr(cfg.model, "use_batchnorm", False))
            or bool(getattr(fagcn_cfg, "use_batchnorm", False))
        ),
    )
