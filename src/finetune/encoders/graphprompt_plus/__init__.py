"""GraphPrompt+ stage-prompt adapters.

Public API:
- ``GraphPromptPlusAdapter`` — abstract base for per-backbone stage drivers.
- ``GraphPromptPlusSpec`` + ``resolve_graphprompt_plus_spec(cfg)`` — single
  source of truth for which backbones GraphPrompt+ supports and at what
  tier (``official``/``extension``).
- ``build_graphprompt_plus_adapter(cfg)`` — factory that dispatches to
  the per-backbone adapter; raises on unsupported models.

Phase 1 ships gcn/gin/gat/mlp via the shared ``GNNStackAdapter``.
Future phases land transformer / nodeformer / h2gcn / fagcn.
"""

from __future__ import annotations

from .base import GraphPromptPlusAdapter
from .factory import (
    GRAPHPROMPT_PLUS_ADAPTERS,
    build_graphprompt_plus_adapter,
)
from .spec import (
    GraphPromptPlusSpec,
    resolve_graphprompt_plus_spec,
    supported_graphprompt_plus_backbones,
)

__all__ = [
    "GRAPHPROMPT_PLUS_ADAPTERS",
    "GraphPromptPlusAdapter",
    "GraphPromptPlusSpec",
    "build_graphprompt_plus_adapter",
    "resolve_graphprompt_plus_spec",
    "supported_graphprompt_plus_backbones",
]
