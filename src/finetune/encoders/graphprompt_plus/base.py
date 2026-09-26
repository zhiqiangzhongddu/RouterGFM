"""Shared base class for GraphPrompt+ stage adapters.

Every backbone supported by GraphPrompt+ has exactly one
``GraphPromptPlusAdapter`` subclass that knows:

- where the four stage prompts (input / between-layer 0-1 / between-layer
  1-2 / final repr) enter the backbone's forward pass;
- which of those stages are applicable for a given encoder shape;
- whether the backbone supports ``repr_source="layer_concat"``.

The adapter consumes the *standard* pretrained encoder produced by
``src.model.build_encoder_from_cfg`` — there is no encoder substitution
in Phase 1.  Adapters are forward-pass drivers, not encoder wrappers, so
state-dict compatibility is automatic for the migrated backbones
(gcn/gin/gat/mlp).  Later phases may introduce true ``PromptAware*``
subclasses for backbones whose call signature cannot be expressed by
driving the existing encoder.

Adapter return contract: ``forward_with_stage_prompt`` returns
``(node_repr, graph_repr_or_None)`` exactly like the pre-refactor
``_forward_with_stage_prompt`` in ``src/finetune/methods/graphprompt.py``
returned.  Node masking, label preparation, graph-pooling fallback, and
cross-stage mixing remain owned by the task class
(``_extract_stagewise_plus_embeddings``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class GraphPromptPlusAdapter(ABC):
    """Backbone-specific stage-prompt driver for GraphPrompt+.

    A single adapter class can serve multiple backbones (the
    ``GNNStackAdapter`` covers gcn/gin/gat/mlp), so support tier and
    formula are intentionally **not** class attributes here — they are
    per-backbone and live in :func:`resolve_graphprompt_plus_spec`.
    Adapters carry only structural concerns (which stages exist for an
    encoder shape, how to drive the forward).
    """

    @classmethod
    @abstractmethod
    def supports_model(cls, model: nn.Module) -> bool:
        """Sanity check that *model* exposes the API this adapter needs.

        Distinct from registry membership: ``build_graphprompt_plus_adapter``
        already filters by ``cfg.model.name``; this is a runtime guard for
        when the chosen adapter is asked to drive a model whose shape
        violates its assumptions (e.g. missing ``convs`` attribute).
        """

    @classmethod
    @abstractmethod
    def iter_stage_specs(
        cls,
        num_layers: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        repr_dim: int,
    ) -> list[tuple[int, int]]:
        """Return the list of ``(stage_id, dim)`` slots this backbone exposes.

        Stage IDs follow the GraphPrompt+ convention: ``0`` is the input,
        ``1`` and ``2`` are between-layer points, ``3`` is the final
        representation.  ``dim`` is the channel count at that injection
        point so the prompt module can size its mask correctly.
        """

    @classmethod
    @abstractmethod
    def supports_layer_concat(cls) -> bool:
        """Whether ``repr_source="layer_concat"`` is meaningful for this backbone."""

    @classmethod
    @abstractmethod
    def forward_with_stage_prompt(
        cls,
        model: nn.Module,
        data,
        stage_id: int,
        prompt,
        repr_source: str,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Drive *model*'s forward pass with one stage prompt injected.

        Args:
            model: the standard pretrained encoder.
            data: a PyG ``Data``/``Batch`` object with ``x`` and optionally
                ``edge_index`` and ``batch``.
            stage_id: which stage's prompt to apply this pass.
            prompt: a ``GraphPromptPlusStageWise`` module exposing
                ``has_stage(stage_id)`` and ``apply_stage(stage_id, x)``.
            repr_source: ``"last"`` or ``"layer_concat"`` — controls
                whether per-layer outputs are concatenated.

        Returns:
            ``(node_repr, graph_repr_or_None)``.  ``graph_repr`` may be
            ``None`` when the backbone has no native pooling at this
            point; the task class will fall back to its configured
            pooling mode in that case.
        """
