"""GraphPrompt+ adapter for ``NodeFormerEncoder``.

NodeFormer's forward pass is more complex than the GNN stack:

    fcs[0] (input proj) -> bns[0] -> activation -> dropout
    -> for each conv: conv(z, adjs, tau) [+ residual] [+ bn] [+ act] -> dropout
    -> [optional jk-cat]
    -> fcs[-1] (output proj)

The injection points for the four stage prompts:

    0 — input features (before fcs[0])
    1 — between conv 0 and conv 1 (after dropout of conv 0)
    2 — between conv 1 and conv 2 (after dropout of conv 1)
    3 — final node representation (after fcs[-1])

Stage 1/2 prompts apply to the post-dropout state, mirroring the
GNN-stack convention: the prompt is whatever the *next* layer would
have read.

This adapter does not support ``repr_source="layer_concat"``: NodeFormer
already has a built-in ``use_jk`` mechanism for layer concatenation that
we do not override, and the base encoder does not expose a per-layer
cache that would let GraphPrompt+ build its own layer_concat
representation.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .base import GraphPromptPlusAdapter


class NodeFormerStackAdapter(GraphPromptPlusAdapter):
    """Stage-prompt driver for ``src.model.nodeformer.NodeFormerEncoder``."""

    @classmethod
    def supports_model(cls, model: nn.Module) -> bool:
        # NodeFormerEncoder wraps the actual NodeFormer in ``self.model``
        # and exposes ``tau`` / ``use_edge_loss`` flags.
        if not hasattr(model, "model") or not hasattr(model, "tau"):
            return False
        inner = getattr(model, "model")
        for attr in ("convs", "fcs", "bns", "use_bn", "use_residual", "use_act", "dropout"):
            if not hasattr(inner, attr):
                return False
        return True

    @classmethod
    def iter_stage_specs(
        cls,
        num_layers: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        repr_dim: int,
    ) -> list[tuple[int, int]]:
        # Stage 0 lives BEFORE fcs[0], so its dim is in_dim.
        # Stages 1/2 live between conv layers in hidden space.
        # Stage 3 lives AFTER fcs[-1], so its dim is out_dim.
        # ``num_layers`` for NodeFormer counts the conv layers (matching
        # cfg.model.num_layers); stages 1/2 require >=2 / >=3 convs.
        stages: list[tuple[int, int]] = [(0, int(in_dim))]
        if num_layers >= 2:
            stages.append((1, int(hidden_dim)))
        if num_layers >= 3:
            stages.append((2, int(hidden_dim)))
        stages.append((3, int(repr_dim)))
        return stages

    @classmethod
    def supports_layer_concat(cls) -> bool:
        return False

    @classmethod
    def forward_with_stage_prompt(
        cls,
        model: nn.Module,
        data,
        stage_id: int,
        prompt,
        repr_source: str,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if repr_source == "layer_concat":
            raise ValueError(
                "NodeFormerStackAdapter does not support repr_source='layer_concat'."
            )
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("edge_index is required for NodeFormer GraphPrompt+.")
        # Mirror NodeFormerEncoder.forward's adjs construction.
        from torch_sparse import SparseTensor

        if isinstance(edge_index, SparseTensor):
            coo = edge_index.coo()
            edge_index_t = (coo[0].to(data.x.device), coo[1].to(data.x.device))
        else:
            edge_index_t = (edge_index[0].to(data.x.device), edge_index[1].to(data.x.device))
        adjs = [edge_index_t]

        inner = model.model
        tau = float(getattr(model, "tau", 1.0))

        x = data.x
        if stage_id == 0 and prompt.has_stage(0):
            x = prompt.apply_stage(0, x)

        # Replay NodeFormer.forward.  The encoder has use_edge_loss=False
        # in finetune (the link-loss path is for pretrain), but we mirror
        # both branches so the parity test exercises the realistic config.
        x = x.unsqueeze(0)
        node_batch = getattr(data, "batch", None)
        layer_ = []
        z = inner.fcs[0](x)
        if inner.use_bn:
            z = inner.bns[0](z)
        z = inner.activation(z)
        z = F.dropout(z, p=inner.dropout, training=inner.training)
        layer_.append(z)

        for i, conv in enumerate(inner.convs):
            if inner.use_edge_loss:
                z, _link = conv(z, adjs, tau, batch=node_batch)
            else:
                z = conv(z, adjs, tau, batch=node_batch)
            if inner.use_residual:
                z = z + layer_[i]
            if inner.use_bn:
                z = inner.bns[i + 1](z)
            if inner.use_act:
                z = inner.activation(z)
            z = F.dropout(z, p=inner.dropout, training=inner.training)
            # Stage-1/2 prompts inject AFTER dropout — the same point a
            # subsequent layer would consume.  The dim contract is hidden
            # dim, set in ``iter_stage_specs``.  We squeeze/unsqueeze the
            # leading "batch=1" axis so the prompt apply is identical to
            # the per-node prompt application elsewhere.
            if not (i == len(inner.convs) - 1):
                if stage_id == 1 and i == 0 and prompt.has_stage(1):
                    z2d = prompt.apply_stage(1, z.squeeze(0))
                    z = z2d.unsqueeze(0)
                if stage_id == 2 and i == 1 and prompt.has_stage(2):
                    z2d = prompt.apply_stage(2, z.squeeze(0))
                    z = z2d.unsqueeze(0)
            layer_.append(z)

        if inner.use_jk:
            z = torch.cat(layer_, dim=-1)
        node_repr = inner.fcs[-1](z).squeeze(0)

        if stage_id == 3 and prompt.has_stage(3):
            node_repr = prompt.apply_stage(3, node_repr)

        graph_repr: Optional[torch.Tensor] = None
        batch = getattr(data, "batch", None)
        if batch is not None and hasattr(model, "pool"):
            graph_repr = model.pool(node_repr, batch)
        return node_repr, graph_repr


def build_graphprompt_plus_nodeformer_adapter(cfg) -> type[GraphPromptPlusAdapter]:
    return NodeFormerStackAdapter
