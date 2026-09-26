"""GraphPrompt+ adapter for the standard ``GNNEncoder`` backbones.

Drives ``src.model.encoder.GNNEncoder`` (gcn/gin/gat/mlp) through a
forward pass with one stage prompt injected at the corresponding
position.  The implementation is a direct port of the pre-refactor
``_forward_with_stage_prompt`` body in
``src/finetune/methods/graphprompt.py``; an identity-mask parity check
in the project's local test harness locks bit-identity with the
previous behavior across gcn/gin/gat/mlp x {last, layer_concat}.

Stage layout (matches ``GraphPromptPlusStageWise``):
    0 — input features (before any layer)
    1 — between layers 0 and 1 (after act/bn/dropout of layer 0)
    2 — between layers 1 and 2 (after act/bn/dropout of layer 1)
    3 — final representation (after the last layer's BN, before pooling)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .base import GraphPromptPlusAdapter


_REQUIRED_ATTRS = ("convs", "model_type", "dropout", "act")
_SUPPORTED_MODEL_TYPES = frozenset({"gcn", "gin", "gat", "mlp"})


class GNNStackAdapter(GraphPromptPlusAdapter):
    """Adapter for the uniform ``conv -> act -> bn -> dropout`` stack used
    by ``GNNEncoder`` for gcn/gin/gat/mlp.

    Per-backbone support tier and formula are owned by
    :func:`resolve_graphprompt_plus_spec` because one adapter covers
    multiple backbones (gcn/gin = official, gat/mlp = extension).
    """

    @classmethod
    def supports_model(cls, model: nn.Module) -> bool:
        if any(not hasattr(model, attr) for attr in _REQUIRED_ATTRS):
            return False
        model_type = str(getattr(model, "model_type", "") or "").lower()
        return model_type in _SUPPORTED_MODEL_TYPES

    @classmethod
    def iter_stage_specs(
        cls,
        num_layers: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        repr_dim: int,
    ) -> list[tuple[int, int]]:
        # Stage layout matches ``GraphPromptPlusStageWise``: stages 1 and
        # 2 require enough hidden layers to land between them.  ``repr_dim``
        # already accounts for ``repr_source`` (last vs layer_concat) so
        # stage 3's mask sizes correctly when concatenation is in effect.
        stages: list[tuple[int, int]] = [(0, int(in_dim))]
        if num_layers >= 2:
            stages.append((1, int(hidden_dim)))
        if num_layers >= 3:
            stages.append((2, int(hidden_dim)))
        stages.append((3, int(repr_dim)))
        return stages

    @classmethod
    def supports_layer_concat(cls) -> bool:
        return True

    @classmethod
    def forward_with_stage_prompt(
        cls,
        model: nn.Module,
        data,
        stage_id: int,
        prompt,
        repr_source: str,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = data.x
        edge_index = getattr(data, "edge_index", None)
        batch = getattr(data, "batch", None)
        model_type = str(getattr(model, "model_type", "") or "").lower()
        convs = getattr(model, "convs")
        bns = getattr(model, "bns", [])
        use_batchnorm = bool(getattr(model, "use_batchnorm", False))
        act = getattr(model, "act")
        dropout = getattr(model, "dropout")
        collect_layers = repr_source == "layer_concat"

        if stage_id == 0 and prompt.has_stage(0):
            x = prompt.apply_stage(0, x)

        layer_outputs: list[torch.Tensor] = []
        for idx, conv in enumerate(convs):
            if model_type == "mlp":
                x = conv(x)
            else:
                if edge_index is None:
                    raise ValueError(
                        "edge_index is required for stage-wise GraphPrompt+ on GNN backbones."
                    )
                x = conv(x, edge_index)

            is_last = idx == len(convs) - 1
            if not is_last:
                x = act(x)
                if use_batchnorm and idx < len(bns):
                    x = bns[idx](x)
                if collect_layers:
                    layer_outputs.append(x)
                x = dropout(x)
                if stage_id == 1 and idx == 0 and prompt.has_stage(1):
                    x = prompt.apply_stage(1, x)
                if stage_id == 2 and idx == 1 and prompt.has_stage(2):
                    x = prompt.apply_stage(2, x)
                continue

            if use_batchnorm and idx < len(bns):
                x = bns[idx](x)
            if collect_layers:
                layer_outputs.append(x)

        if collect_layers and len(layer_outputs) > 1:
            node_repr = torch.cat(layer_outputs, dim=-1)
        else:
            node_repr = x

        if stage_id == 3 and prompt.has_stage(3):
            node_repr = prompt.apply_stage(3, node_repr)

        graph_repr: Optional[torch.Tensor] = None
        if batch is not None and hasattr(model, "pool"):
            graph_repr = model.pool(node_repr, batch)
        return node_repr, graph_repr


def build_graphprompt_plus_gnn_adapter(cfg) -> type[GraphPromptPlusAdapter]:
    """Return the GNN-stack adapter class for gcn/gin/gat/mlp.

    The adapter has no per-cfg state — its methods are classmethods
    operating on whatever ``model`` and ``prompt`` the task class hands
    in.  Returning the class (rather than an instance) keeps the
    factory contract uniform with future per-backbone adapters that
    may carry state.
    """
    return GNNStackAdapter
