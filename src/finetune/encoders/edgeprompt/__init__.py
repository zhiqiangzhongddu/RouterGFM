"""EdgePrompt-aware encoders.

Public API:
- ``PromptAwareEncoder`` — marker base class; every prompt-aware encoder
  subclasses it and advertises support tier + formula.
- ``EdgePromptSpec`` + ``resolve_edgeprompt_prompt_spec(cfg)`` — single
  source of truth for prompt dim list and self-loop policy; used by
  both the encoder factory and ``FinetuneEdgePrompt`` so the two
  constructions cannot drift.
- ``build_prompt_encoder(cfg, in_dim)`` — factory that dispatches to
  per-backbone builders; raises on unsupported models (e.g. ``mlp``).
"""

from __future__ import annotations

from .base import PromptAwareEncoder
from .factory import PROMPT_ENCODERS, build_prompt_encoder
from .spec import (
    EdgePromptSpec,
    resolve_edgeprompt_prompt_spec,
    supported_edgeprompt_backbones,
)

__all__ = [
    "PROMPT_ENCODERS",
    "EdgePromptSpec",
    "PromptAwareEncoder",
    "build_prompt_encoder",
    "resolve_edgeprompt_prompt_spec",
    "supported_edgeprompt_backbones",
]
