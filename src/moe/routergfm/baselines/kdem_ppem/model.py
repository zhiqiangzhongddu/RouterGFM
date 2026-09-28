"""Trainable experts forwarded through their merged parameters (Eq. 8, 10)."""

from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.nn.modules.batchnorm import _BatchNorm

from .merge import merged_state, module_tensors
from .selection import assert_state_compatible


class MergedExpertModel(nn.Module):
    """Experts ``theta_i`` (trainable) and fixed weights ``alpha``; ``forward`` runs ``M(.; sum_i alpha_i theta_i)``.

    Gradients reach each ``theta_i`` scaled by ``alpha_i``. The structural
    template (a copy of the top-1 expert) is held outside the registered
    modules, so its own weights never enter the optimizer. A second copy is
    never run: encoders cache autograd-linked layer outputs on themselves,
    which ``copy.deepcopy`` rejects. BatchNorm layers always run in eval mode
    (frozen running statistics).
    """

    def __init__(self, experts: List[nn.Module], alpha: torch.Tensor):
        super().__init__()
        assert_state_compatible([e.state_dict() for e in experts])
        self.experts = nn.ModuleList(experts)
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32).detach().clone().view(-1))
        # Lists keep both copies unregistered: [forwarded template, never-run copy source].
        self._template = [copy.deepcopy(experts[0]).requires_grad_(False) for _ in range(2)]

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        for module in self._template:
            module._apply(fn, *args, **kwargs)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self._template[0].train(mode)
        for module in (*self.modules(), *self._template[0].modules()):
            if isinstance(module, _BatchNorm):
                module.eval()
        return self

    def merged_state(self) -> Dict[str, torch.Tensor]:
        return merged_state([module_tensors(e) for e in self.experts], self.alpha)

    def forward(self, data):
        """Student: ``(node_repr, graph_repr)`` of the merged expert."""
        return functional_call(self._template[0], self.merged_state(), (data,))

    def ensemble_node_repr(self, data) -> torch.Tensor:
        """Teacher: ``sum_i alpha_i node_repr_i`` of the individual experts."""
        return sum(self.alpha[i] * expert(data)[0] for i, expert in enumerate(self.experts))

    @torch.no_grad()
    def merged_module(self) -> nn.Module:
        """A standalone encoder holding ``theta_bar`` in eval mode (merge once for inference)."""
        module = copy.deepcopy(self._template[1])
        merged = self.merged_state()
        for name, tensor in module_tensors(module).items():
            tensor.copy_(merged[name])
        return module.eval()


__all__ = ["MergedExpertModel"]
