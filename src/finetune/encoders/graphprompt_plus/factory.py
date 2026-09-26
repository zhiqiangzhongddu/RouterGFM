"""GraphPrompt+ adapter factory.

``build_graphprompt_plus_adapter(cfg)`` dispatches to the right
``GraphPromptPlusAdapter`` subclass based on ``cfg.model.name``.  The
spec registry in ``spec.py`` is authoritative for "which backbones are
supported" so unsupported backbones surface a single, consistent error
message regardless of whether the per-backbone builder has landed.

Phase 1 ships only the GNN-stack adapter (gcn/gin/gat/mlp).  Later
phases register transformer / nodeformer / h2gcn / fagcn.
"""

from __future__ import annotations

from typing import Callable

from .base import GraphPromptPlusAdapter
from .fagcn import build_graphprompt_plus_fagcn_adapter
from .gnn import build_graphprompt_plus_gnn_adapter
from .h2gcn import build_graphprompt_plus_h2gcn_adapter
from .nodeformer import build_graphprompt_plus_nodeformer_adapter
from .spec import (
    GraphPromptPlusSpec,
    resolve_graphprompt_plus_spec,
    supported_graphprompt_plus_backbones,
)
from .transformer import build_graphprompt_plus_transformer_adapter


# Registry: backbone name -> builder callable(cfg) -> adapter class.
# Validation against the spec registry is done first so the error
# message stays consistent across "unsupported by spec" and "supported
# by spec but no builder yet".
GRAPHPROMPT_PLUS_ADAPTERS: dict[str, Callable[..., type[GraphPromptPlusAdapter]]] = {
    "gcn": build_graphprompt_plus_gnn_adapter,
    "gin": build_graphprompt_plus_gnn_adapter,
    "gat": build_graphprompt_plus_gnn_adapter,
    "mlp": build_graphprompt_plus_gnn_adapter,
    "transformer": build_graphprompt_plus_transformer_adapter,
    "nodeformer": build_graphprompt_plus_nodeformer_adapter,
    "fagcn": build_graphprompt_plus_fagcn_adapter,
    "h2gcn": build_graphprompt_plus_h2gcn_adapter,
}


def build_graphprompt_plus_adapter(cfg) -> type[GraphPromptPlusAdapter]:
    """Build the GraphPrompt+ stage adapter class for ``cfg.model.name``.

    Raises ``ValueError`` when the backbone has no GraphPrompt+ spec
    entry, ``NotImplementedError`` when the spec exists but the
    per-backbone builder has not landed yet.
    """
    resolve_graphprompt_plus_spec(cfg)  # raises ValueError on unknown/unsupported

    name = str(cfg.model.name).lower()
    if name not in GRAPHPROMPT_PLUS_ADAPTERS:
        registered = sorted(GRAPHPROMPT_PLUS_ADAPTERS.keys())
        raise NotImplementedError(
            f"GraphPrompt+ for model '{name}' is declared supported but no "
            f"adapter is registered yet. Backbones with an active adapter: "
            f"{registered}. See src/finetune/encoders/graphprompt_plus/factory.py."
        )
    return GRAPHPROMPT_PLUS_ADAPTERS[name](cfg)


__all__ = [
    "GRAPHPROMPT_PLUS_ADAPTERS",
    "build_graphprompt_plus_adapter",
    "GraphPromptPlusSpec",
    "resolve_graphprompt_plus_spec",
    "supported_graphprompt_plus_backbones",
]
