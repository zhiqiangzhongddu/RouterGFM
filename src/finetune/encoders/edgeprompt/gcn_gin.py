"""Prompt-aware GCN/GIN convs and the shared per-layer encoder shell.

Moved verbatim (minus the old GCN-only guard in ``PromptGNNEncoder``)
from ``src/finetune/encoders/prompt_encoder.py`` during the
edgeprompt/ refactor.  ``PromptGNNEncoder`` now subclasses
``PromptAwareEncoder`` and advertises ``edgeprompt_support="official"``
for gcn/gin.

The rest of this module is deliberately bit-identical to the pre-refactor
implementation: the parity test locks this.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, remove_self_loops

from src.model.activations import get_activation
from src.utils.pool import pool_nodes

from .base import PromptAwareEncoder


class PromptGCNConv(nn.Module):
    """GCN-style convolution with optional edge prompts.

    When ``add_self_loops_in_conv=True``, follows the baseline EdgePrompt
    recipe: add self-loops, degree-normalize, ``lin(x_j + edge_prompt)``
    inside messages.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        add_self_loops_in_conv: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops_in_conv = bool(add_self_loops_in_conv)
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_prompt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        if self.add_self_loops_in_conv:
            # remove_self_loops first so a pre-existing self-loop in the
            # input does not produce duplicated loops after add_self_loops.
            # Matches the behavior used in EdgePromptPlus.get_prompt so
            # prompt rows and conv rows stay aligned on any input.
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        row, col = edge_index
        deg = degree(col, num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # source->target flow: edge_index[0]=source, edge_index[1]=target
        x_j = x[row]
        if edge_prompt is not None:
            msg = self.lin(x_j + edge_prompt)
        else:
            msg = self.lin(x_j)
        msg = norm.view(-1, 1) * msg

        out = torch.zeros(
            num_nodes,
            self.out_channels,
            device=x.device,
            dtype=msg.dtype,
        )
        out.scatter_add_(0, col.unsqueeze(1).expand_as(msg), msg)
        if self.bias is not None:
            out = out + self.bias
        return out


class PromptGINConv(MessagePassing):
    """GIN-style message passing with optional edge prompt additions."""

    def __init__(
        self,
        nn_module: nn.Module,
        eps: float = 0.0,
        train_eps: bool = False,
        message_relu: bool = True,
    ):
        super().__init__(aggr="add")
        self.nn = nn_module
        self.initial_eps = float(eps)
        self.message_relu = bool(message_relu)
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.tensor([eps], dtype=torch.float))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_prompt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        post_mp = self.propagate(
            edge_index=edge_index, x=x, edge_attr=edge_prompt, size=None
        )
        post_mp = (1 + self.eps) * x + post_mp
        return self.nn(post_mp)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        if edge_attr is not None:
            x_j = x_j + edge_attr
        if self.message_relu:
            x_j = F.relu(x_j)
        return x_j


class PromptGNNEncoder(PromptAwareEncoder):
    """GNN encoder with edge-prompt aware convolutions for GCN/GIN."""

    edgeprompt_support = "official"
    edgeprompt_formula = "normalized (h_j + p_ji) message (gcn) / summed (h_j + p_ji) then MLP (gin)"

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        model_type: str = "gcn",
        act: str = "relu",
        dropout: float = 0.1,
        graph_pooling: str = "mean",
        gcn_add_self_loops: bool = True,
        use_batchnorm: bool = False,
        gin_message_relu: bool = True,
        gin_train_eps: bool = True,
    ) -> None:
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.model_type = model_type.lower()
        self.act = get_activation(act)
        self.dropout = nn.Dropout(dropout)
        self.graph_pooling = graph_pooling
        self.use_batchnorm = bool(use_batchnorm)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for idx in range(num_layers):
            in_c = in_dim if idx == 0 else hidden_dim
            out_c = out_dim if idx == num_layers - 1 else hidden_dim

            if self.model_type == "gcn":
                conv = PromptGCNConv(
                    in_channels=in_c,
                    out_channels=out_c,
                    add_self_loops_in_conv=gcn_add_self_loops,
                )
            elif self.model_type == "gin":
                hidden = max(out_c, in_c)
                mlp = nn.Sequential(
                    nn.Linear(in_c, hidden),
                    copy.deepcopy(self.act),
                    nn.Linear(hidden, out_c),
                )
                conv = PromptGINConv(
                    mlp,
                    train_eps=gin_train_eps,
                    message_relu=gin_message_relu,
                )
            else:
                raise ValueError(
                    f"PromptGNNEncoder handles gcn/gin only; got '{self.model_type}'. "
                    f"Other backbones should be dispatched to their dedicated "
                    f"PromptAwareEncoder via build_prompt_encoder."
                )
            self.convs.append(conv)
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
            raise ValueError("edge_index is required for EdgePrompt encoder")

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
                # Last layer: match GNNEncoder — apply BN (if enabled) but
                # no activation or dropout.
                if self.use_batchnorm:
                    x = self.bns[idx](x)

        node_repr = x
        graph_repr = None
        if batch is not None:
            graph_repr = pool_nodes(node_repr, batch, mode=self.graph_pooling)
        return node_repr, graph_repr


def build_prompt_gnn_encoder_from_cfg(cfg, in_dim: int) -> PromptGNNEncoder:
    """Instantiate the shared GCN/GIN prompt encoder from cfg."""
    from .spec import resolve_edgeprompt_prompt_spec

    gcn_add_self_loops = resolve_edgeprompt_prompt_spec(cfg).add_self_loops
    edge_cfg = getattr(getattr(cfg, "finetune", None), "edgeprompt", None)
    gin_message_relu = bool(getattr(edge_cfg, "gin_message_relu", True)) if edge_cfg is not None else True
    gin_train_eps = bool(getattr(edge_cfg, "gin_train_eps", True)) if edge_cfg is not None else True
    if gin_message_relu and str(getattr(cfg.model, "name", "gcn")).lower() == "gin":
        # Surface the operator switch: the official EdgePrompt GIN applies
        # relu(x_j) in message() even with a zero prompt, while IcG's vanilla
        # pretrain GIN (PyG GINConv) has no message nonlinearity — the frozen
        # backbone therefore computes a different function than it was
        # pretrained for. Set finetune.edgeprompt.gin_message_relu=False for
        # pretrain/finetune operator consistency with IcG checkpoints.
        print(
            "[Finetune][EdgePrompt] gin_message_relu=True: GIN messages use "
            "relu(x_j) (official EdgePrompt operator), which differs from the "
            "vanilla pretrained GIN."
        )
    return PromptGNNEncoder(
        in_dim=in_dim,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.out_dim,
        num_layers=cfg.model.num_layers,
        model_type=getattr(cfg.model, "name", "gcn"),
        act=cfg.model.activation,
        dropout=cfg.model.dropout,
        graph_pooling=cfg.model.graph_pooling,
        gcn_add_self_loops=gcn_add_self_loops,
        use_batchnorm=bool(getattr(cfg.model, "use_batchnorm", False)),
        gin_message_relu=gin_message_relu,
        gin_train_eps=gin_train_eps,
    )
