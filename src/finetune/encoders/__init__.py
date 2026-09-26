"""Prompt-aware encoder modules for finetuning.

Canonical implementations live in :mod:`src.finetune.encoders.edgeprompt`
and :mod:`src.finetune.encoders.graphprompt_plus`.
"""

from .edgeprompt import (
    PROMPT_ENCODERS,
    EdgePromptSpec,
    PromptAwareEncoder,
    build_prompt_encoder,
    resolve_edgeprompt_prompt_spec,
    supported_edgeprompt_backbones,
)
from .graphprompt_plus import (
    GRAPHPROMPT_PLUS_ADAPTERS,
    GraphPromptPlusAdapter,
    GraphPromptPlusSpec,
    build_graphprompt_plus_adapter,
    resolve_graphprompt_plus_spec,
    supported_graphprompt_plus_backbones,
)

__all__ = [
    "PROMPT_ENCODERS",
    "EdgePromptSpec",
    "PromptAwareEncoder",
    "build_prompt_encoder",
    "resolve_edgeprompt_prompt_spec",
    "supported_edgeprompt_backbones",
    "GRAPHPROMPT_PLUS_ADAPTERS",
    "GraphPromptPlusAdapter",
    "GraphPromptPlusSpec",
    "build_graphprompt_plus_adapter",
    "resolve_graphprompt_plus_spec",
    "supported_graphprompt_plus_backbones",
]
