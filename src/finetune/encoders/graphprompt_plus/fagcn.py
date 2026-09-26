"""GraphPrompt+ adapter for ``FAGCNEncoder``.

FAGCN's forward pass is:

    x = lin_in(x_input)
    x0 = x   # initial reference re-used by every FAConv
    for i, conv in enumerate(convs):
        x = conv(x, x0, edge_index)
        if not last:
            x = act(x); x = dropout(x)
    x = out_lin(x)

Stage layout:

    0 — input features (before lin_in)
    1 — between conv 0 and conv 1 (after act/dropout of conv 0)
    2 — between conv 1 and conv 2 (after act/dropout of conv 1)
    3 — final node representation (after out_lin)

**Design decision (extension):** stage-1/2 prompts modify only the
*current* features ``x`` flowing forward — the initial reference
``x0`` stays unprompted.  Reasoning: FAConv mixes ``x_0`` into the
hidden state at *every* layer via the FAGCN gating coefficient.  If
the prompt also modified ``x_0`` it would compound across all
subsequent layers, which collapses the per-stage independence
GraphPrompt+ relies on.  Keeping ``x_0`` unprompted preserves the
"one stage one branch" property that the cross-stage softmax mixing
expects.

State-dict compatibility: this adapter does not subclass FAGCNEncoder,
so the standard pretrained encoder is consumed as-is.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .base import GraphPromptPlusAdapter


_REQUIRED_ATTRS = ("lin_in", "convs", "out_lin", "act", "dropout")


class FAGCNStackAdapter(GraphPromptPlusAdapter):
    """Stage-prompt driver for ``src.model.fagcn.FAGCNEncoder``."""

    @classmethod
    def supports_model(cls, model: nn.Module) -> bool:
        return all(hasattr(model, attr) for attr in _REQUIRED_ATTRS)

    @classmethod
    def iter_stage_specs(
        cls,
        num_layers: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        repr_dim: int,
    ) -> list[tuple[int, int]]:
        # Stage 0 lives BEFORE lin_in, hence in_dim.
        # Stages 1/2 are between FAConv layers — all hidden_dim
        # because FAConv keeps channels constant (single ``channels``
        # arg).  Stage 3 is AFTER out_lin, hence repr_dim.
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
                "FAGCNStackAdapter does not support repr_source='layer_concat'."
            )
        x = data.x
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("edge_index is required for FAGCN GraphPrompt+.")
        batch = getattr(data, "batch", None)
        lin_in = model.lin_in
        convs = model.convs
        out_lin = model.out_lin
        act = model.act
        dropout = model.dropout

        if stage_id == 0 and prompt.has_stage(0):
            x = prompt.apply_stage(0, x)
        x = lin_in(x)
        x0 = x  # FAConv residual reference; intentionally not prompted.

        for idx, conv in enumerate(convs):
            x = conv(x, x0, edge_index)
            is_last = idx == len(convs) - 1
            if is_last:
                continue
            x = act(x)
            x = dropout(x)
            if stage_id == 1 and idx == 0 and prompt.has_stage(1):
                x = prompt.apply_stage(1, x)
            if stage_id == 2 and idx == 1 and prompt.has_stage(2):
                x = prompt.apply_stage(2, x)

        node_repr = out_lin(x)
        if stage_id == 3 and prompt.has_stage(3):
            node_repr = prompt.apply_stage(3, node_repr)

        graph_repr: Optional[torch.Tensor] = None
        if batch is not None and hasattr(model, "pool"):
            graph_repr = model.pool(node_repr, batch)
        return node_repr, graph_repr


def build_graphprompt_plus_fagcn_adapter(cfg) -> type[GraphPromptPlusAdapter]:
    return FAGCNStackAdapter
