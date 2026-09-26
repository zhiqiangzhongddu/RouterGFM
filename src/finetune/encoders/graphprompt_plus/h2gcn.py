"""GraphPrompt+ adapter for ``H2GCNEncoder``.

H2GCN does not have a uniform layer stack.  Its forward is a fixed
two-hop structure:

    x  -> 1-hop agg -> lin1 -> [bn1] -> act -> dropout            =: x1
    x1 -> 2-hop agg -> lin2 -> [bn2] -> act -> dropout            =: x2
    cat(x1, x2) -> out_lin                                         =: node_repr

Because the structure is fixed, only three of the four stage-prompt
slots have a natural injection point:

    0 — input features (before 1-hop aggregation)
    1 — between the 1-hop and 2-hop branches
        (applied to ``x1`` after dropout, BEFORE x1 is re-aggregated
         to produce x2; ``x1`` itself is also concatenated into the
         final readout, so this prompt also flows through the
         skip path)
    3 — final node representation (after out_lin)

Stage 2 has no meaningful injection point — there is no third hop.
``iter_stage_specs`` therefore omits stage 2 entirely; the prompt
module skips its mask.

**Design decision (extension):** stage-1 prompts modify ``x1`` once.
Both the 2-hop aggregation and the final concat see the prompted
``x1``.  This is the natural reading of "prompt between layers"
because both downstream consumers read ``x1``; prompting only one of
them would be an arbitrary asymmetry.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch_geometric.utils import add_self_loops

from .base import GraphPromptPlusAdapter


_REQUIRED_ATTRS = ("lin1", "lin2", "out_lin", "act", "dropout")


class H2GCNAdapter(GraphPromptPlusAdapter):
    """Stage-prompt driver for ``src.model.h2gcn.H2GCNEncoder``."""

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
        # H2GCN's ``num_layers`` is irrelevant — the architecture is
        # fixed at 2 hops.  Active stages are {0, 1, 3}; stage 2 has no
        # injection point.
        return [
            (0, int(in_dim)),
            (1, int(hidden_dim)),
            (3, int(repr_dim)),
        ]

    @classmethod
    def supports_layer_concat(cls) -> bool:
        # The "layer concat" idea has no meaning for H2GCN: its readout
        # already concatenates x1 and x2.  Refusing layer_concat keeps
        # the contract tight — repr_dim equals out_dim.
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
                "H2GCNAdapter does not support repr_source='layer_concat'."
            )
        x = data.x
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("edge_index is required for H2GCN GraphPrompt+.")
        batch = getattr(data, "batch", None)

        if stage_id == 0 and prompt.has_stage(0):
            x = prompt.apply_stage(0, x)

        # Mirror H2GCNEncoder.forward verbatim from src/model/h2gcn.py.
        edge_with_self, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_with_self
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1).to(x.device)

        x1 = torch.zeros_like(x)
        x1.index_add_(0, row, x[col])
        x1 = x1 / deg.view(-1, 1)
        x1 = model.lin1(x1)
        if model.bn1 is not None:
            x1 = model.bn1(x1)
        x1 = model.act(x1)
        x1 = model.dropout(x1)
        # Stage-1 prompt fires AFTER dropout — both the 2-hop aggregation
        # and the final concat consume the prompted ``x1``.
        if stage_id == 1 and prompt.has_stage(1):
            x1 = prompt.apply_stage(1, x1)

        x2 = torch.zeros_like(x1)
        x2.index_add_(0, row, x1[col])
        x2 = x2 / deg.view(-1, 1)
        x2 = model.lin2(x2)
        if model.bn2 is not None:
            x2 = model.bn2(x2)
        x2 = model.act(x2)
        x2 = model.dropout(x2)

        h = torch.cat([x1, x2], dim=-1)
        node_repr = model.out_lin(h)
        if stage_id == 3 and prompt.has_stage(3):
            node_repr = prompt.apply_stage(3, node_repr)

        graph_repr: Optional[torch.Tensor] = None
        if batch is not None and hasattr(model, "pool"):
            graph_repr = model.pool(node_repr, batch)
        return node_repr, graph_repr


def build_graphprompt_plus_h2gcn_adapter(cfg) -> type[GraphPromptPlusAdapter]:
    return H2GCNAdapter
