"""EdgePrompt encoder factory.

``build_prompt_encoder(cfg, in_dim)`` dispatches to the right
``PromptAwareEncoder`` subclass based on ``cfg.model.name``.  Each
backbone registers exactly one builder here; ``mlp`` and any unknown
name raise with the supported list.

Later commits add entries for gat, transformer, h2gcn, fagcn,
nodeformer.  The initial refactor keeps only gcn/gin wired so Commit 5
is a pure no-behavior-change move.
"""

from __future__ import annotations

from typing import Callable

from .base import PromptAwareEncoder
from .fagcn import build_prompt_fagcn_encoder_from_cfg
from .gat import build_prompt_gat_encoder_from_cfg
from .gcn_gin import build_prompt_gnn_encoder_from_cfg
from .h2gcn import build_prompt_h2gcn_encoder_from_cfg
from .nodeformer import build_prompt_nodeformer_encoder_from_cfg
from .transformer import build_prompt_transformer_encoder_from_cfg
from .spec import (
    EdgePromptSpec,
    resolve_edgeprompt_prompt_spec,
    supported_edgeprompt_backbones,
)

# Registry: backbone name -> builder callable(cfg, in_dim) -> PromptAwareEncoder.
# Populated lazily as each backbone lands; resolve_edgeprompt_prompt_spec
# is authoritative for "which backbones exist", so the factory validates
# against it, not against this dict, to keep the error message aligned.
PROMPT_ENCODERS: dict[str, Callable[..., PromptAwareEncoder]] = {
    "gcn": build_prompt_gnn_encoder_from_cfg,
    "gin": build_prompt_gnn_encoder_from_cfg,
    "gat": build_prompt_gat_encoder_from_cfg,
    "h2gcn": build_prompt_h2gcn_encoder_from_cfg,
    "transformer": build_prompt_transformer_encoder_from_cfg,
    "fagcn": build_prompt_fagcn_encoder_from_cfg,
    "nodeformer": build_prompt_nodeformer_encoder_from_cfg,
}


def build_prompt_encoder(cfg, in_dim: int) -> PromptAwareEncoder:
    """Build the prompt-aware encoder for ``cfg.model.name``.

    Raises ``ValueError`` when the backbone is unsupported (e.g. ``mlp``)
    or when EdgePrompt has not yet landed for that backbone.
    """
    # Validate backbone against the full spec registry first so the error
    # message is consistent ("gcn/gin/gat/...") regardless of whether the
    # backbone's builder has landed yet.
    resolve_edgeprompt_prompt_spec(cfg)  # raises ValueError on unknown/unsupported

    name = str(cfg.model.name).lower()
    if name not in PROMPT_ENCODERS:
        supported_now = sorted(PROMPT_ENCODERS.keys())
        raise NotImplementedError(
            f"EdgePrompt for model '{name}' is declared supported but no "
            f"builder is registered yet. Backbones with an active builder: "
            f"{supported_now}. See src/finetune/encoders/edgeprompt/factory.py."
        )
    return PROMPT_ENCODERS[name](cfg, in_dim)


__all__ = [
    "PROMPT_ENCODERS",
    "build_prompt_encoder",
    "EdgePromptSpec",
    "resolve_edgeprompt_prompt_spec",
    "supported_edgeprompt_backbones",
]
