"""Shared base class for EdgePrompt-aware encoders.

Every prompt-aware encoder under ``src/finetune/encoders/edgeprompt/``
subclasses ``PromptAwareEncoder`` and declares:

- ``edgeprompt_support``: ``"official"`` (paper-faithful, no design
  decisions) or ``"extension"`` (repo-specific adaptation with a
  documented injection point).
- ``edgeprompt_formula``: short human-readable string; surfaces in
  error messages and the README table.
- ``pretrained_key_whitelist``: state_dict keys that MUST load from a
  pretrained checkpoint when the encoder is frozen.  Empty means "auto
  derive from ``state_dict().keys()`` minus ``prompt_only_keys``".
- ``prompt_only_keys``: state_dict keys that are legitimately new in
  the prompt-aware encoder (e.g. prompt tensors).  Never counted
  against the frozen-load match ratio.

The ``forward`` signature is fixed: ``forward(data, prompt=None,
prompt_type=None) -> (node_repr, graph_repr)``.  ``FinetuneEdgePrompt``
duck-types encoders through this signature (plus ``isinstance``
against this class as a fast-path check).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class PromptAwareEncoder(nn.Module):
    edgeprompt_support: str = "extension"
    edgeprompt_formula: str = ""
    pretrained_key_whitelist: frozenset[str] = frozenset()
    prompt_only_keys: frozenset[str] = frozenset()

    def forward(
        self,
        data,
        prompt: Optional[nn.Module] = None,
        prompt_type: Optional[str] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError
