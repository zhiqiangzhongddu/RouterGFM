"""GraphPrompt+ adapter for ``TransformerEncoder``.

``TransformerEncoder`` is a uniform stack of ``TransformerConv`` layers
with the pipeline ``conv -> act -> dropout`` between every non-final
layer (no batch norm).  The stage-prompt injection points map to the
same four-stage layout as the GNN-stack adapter:

    0 — input features (before layer 0)
    1 — between layer 0 and layer 1 (after act/dropout of layer 0)
    2 — between layer 1 and layer 2 (after act/dropout of layer 1)
    3 — final node representation (after the last layer's conv)

The adapter does NOT support ``repr_source="layer_concat"`` because
the base ``TransformerEncoder`` does not cache per-layer
representations (no ``returns_layer_cache`` flag), so there is no
encoder-side reference for the parity contract under that mode.
``FinetuneGraphPrompt`` raises early when ``layer_concat`` is paired
with this adapter.

This is an extension over the original GraphPrompt+ formulation,
which targeted message-passing GNN backbones.  The injection points
are the natural analogue: between attention layers, on the same
hidden activations the next layer would consume.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .base import GraphPromptPlusAdapter


_REQUIRED_ATTRS = ("convs", "act", "dropout")


class TransformerStackAdapter(GraphPromptPlusAdapter):
    """Stage-prompt driver for ``src.model.transformer.TransformerEncoder``."""

    @classmethod
    def supports_model(cls, model: nn.Module) -> bool:
        if any(not hasattr(model, attr) for attr in _REQUIRED_ATTRS):
            return False
        # Distinguish from GNNEncoder: TransformerEncoder has no
        # ``model_type`` attribute and no ``bns`` ModuleList of
        # BatchNorm1d, so we accept by absence of GNNEncoder's marker.
        return not hasattr(model, "model_type")

    @classmethod
    def iter_stage_specs(
        cls,
        num_layers: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        repr_dim: int,
    ) -> list[tuple[int, int]]:
        # Same dim schedule as the GNN stack: in / hidden / hidden / repr.
        # repr_dim equals out_dim here because layer_concat is not
        # supported (see class docstring).
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
                "TransformerStackAdapter does not support repr_source='layer_concat'."
            )
        x = data.x
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("edge_index is required for Transformer GraphPrompt+.")
        batch = getattr(data, "batch", None)
        convs = getattr(model, "convs")
        act = getattr(model, "act")
        dropout = getattr(model, "dropout")

        if stage_id == 0 and prompt.has_stage(0):
            x = prompt.apply_stage(0, x)

        for idx, conv in enumerate(convs):
            x = conv(x, edge_index)
            is_last = idx == len(convs) - 1
            if is_last:
                continue
            x = act(x)
            x = dropout(x)
            # Stage 1/2 land between the post-activation/post-dropout
            # output of layer ``idx`` and the input of layer ``idx+1``,
            # mirroring the GNN-stack injection point.
            if stage_id == 1 and idx == 0 and prompt.has_stage(1):
                x = prompt.apply_stage(1, x)
            if stage_id == 2 and idx == 1 and prompt.has_stage(2):
                x = prompt.apply_stage(2, x)

        node_repr = x
        if stage_id == 3 and prompt.has_stage(3):
            node_repr = prompt.apply_stage(3, node_repr)

        graph_repr: Optional[torch.Tensor] = None
        if batch is not None and hasattr(model, "pool"):
            graph_repr = model.pool(node_repr, batch)
        return node_repr, graph_repr


def build_graphprompt_plus_transformer_adapter(cfg) -> type[GraphPromptPlusAdapter]:
    return TransformerStackAdapter
