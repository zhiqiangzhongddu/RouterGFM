"""Prompt modules for finetuning."""

from .gpf import GPFPlusPrompt, GPFPrompt
from .edgeprompt import EdgePrompt, EdgePromptPlus
from .gppt import GPPTPrompt
from .graphprompt import (
    GraphPrompt,
    GraphPromptPlusStageWise,
    compute_class_centers,
)
from .pronog import ProNoGConditionNet

__all__ = [
    "GPFPrompt",
    "GPFPlusPrompt",
    "EdgePrompt",
    "EdgePromptPlus",
    "GPPTPrompt",
    "GraphPrompt",
    "GraphPromptPlusStageWise",
    "ProNoGConditionNet",
    "compute_class_centers",
]

try:
    from .all_in_one import HeavyPrompt, LightPrompt

    __all__.extend(["HeavyPrompt", "LightPrompt"])
except ModuleNotFoundError:
    pass
