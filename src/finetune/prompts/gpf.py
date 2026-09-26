"""Prompt modules for GPF-style finetuning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.inits import glorot


class GPFPrompt(nn.Module):
    """Original GPF prompt: a single learnable global prompt vector."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.global_emb = nn.Parameter(torch.empty(1, self.in_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot(self.global_emb)

    def add(self, x: Tensor) -> Tensor:
        return x + self.global_emb


class GPFPlusPrompt(nn.Module):
    """GPF-plus prompt with input-conditioned basis interpolation."""

    def __init__(self, in_channels: int, p_num: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.p_num = max(1, int(p_num))
        self.p_list = nn.Parameter(torch.empty(self.p_num, self.in_channels))
        self.attn = nn.Linear(self.in_channels, self.p_num)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot(self.p_list)
        self.attn.reset_parameters()

    def add(self, x: Tensor) -> Tensor:
        score = self.attn(x)
        weight = F.softmax(score, dim=1)
        prompt = weight.mm(self.p_list)
        return x + prompt
