"""ProNoG condition-net prompt module."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ProNoGConditionNet(nn.Module):
    """Bottleneck-MLP condition-net generating node-specific prompt vectors.

    Mirrors the official ProNoG ``PromptVector``: dropout on the conditioning
    input, a down-projection to ``bottleneck_channels``, tanh, an
    up-projection back to ``in_channels``, and a fixed output scaling.  The
    conditioning input is the similarity-weighted multi-hop neighborhood
    readout ``s_v`` (paper Eq. 7); the output is the prompt ``p_v`` used to
    modify the node's frozen embedding.
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int = 64,
        dropout: float = 0.1,
        scaling: float = 0.1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.bottleneck_channels = int(bottleneck_channels)
        self.scaling = float(scaling)
        self.down = nn.Linear(self.in_channels, self.bottleneck_channels)
        self.up = nn.Linear(self.bottleneck_channels, self.in_channels)
        self.dropout = nn.Dropout(p=float(dropout))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Official PromptVector re-inits weights only; biases keep the
        # nn.Linear default init.
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, condition: Tensor) -> Tensor:
        # Unconditional multiply: scaling=0.0 must zero the prompts (the
        # official code's truthiness guard would silently emit full-strength
        # prompts instead, inverting a no-prompt ablation).
        prompts = self.up(torch.tanh(self.down(self.dropout(condition))))
        return prompts * self.scaling
