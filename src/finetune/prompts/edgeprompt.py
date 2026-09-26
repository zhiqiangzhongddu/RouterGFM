"""EdgePrompt modules for finetuning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import add_self_loops, remove_self_loops


class EdgePrompt(nn.Module):
    """Global (layer-wise) edge prompts."""

    def __init__(self, dim_list):
        super().__init__()
        self.global_prompt = nn.ParameterList([nn.Parameter(torch.Tensor(1, dim)) for dim in dim_list])
        self.reset_parameters()

    def reset_parameters(self):
        for prompt in self.global_prompt:
            glorot(prompt)

    def get_prompt(self, x, edge_index, layer):
        del x, edge_index
        return self.global_prompt[layer]


class EdgePromptPlus(nn.Module):
    """Anchor-based edge prompts conditioned on endpoint features."""

    def __init__(
        self,
        dim_list,
        num_anchors: int,
        add_self_loops: bool = True,
        replace_self_loops: bool = True,
    ):
        super().__init__()
        self.add_self_loops = add_self_loops
        self.replace_self_loops = bool(replace_self_loops)
        self.anchor_prompt = nn.ParameterList([nn.Parameter(torch.Tensor(num_anchors, dim)) for dim in dim_list])
        self.w = nn.ModuleList([nn.Linear(2 * dim, num_anchors) for dim in dim_list])
        self.reset_parameters()

    def reset_parameters(self):
        for anchor in self.anchor_prompt:
            glorot(anchor)
        for w in self.w:
            w.reset_parameters()

    def get_prompt(self, x, edge_index, layer):
        if self.add_self_loops:
            # remove_self_loops first so that an input with pre-existing
            # self-loops does not produce E+2N rows.  This mirrors the
            # PyG convention used by GCNConv/GATConv/FAConv (via
            # gcn_norm or the conv's own remove+add sequence) and keeps
            # the prompt tensor aligned with the conv's edge_index.
            if self.replace_self_loops:
                edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Ensure float32 for the weight network (guards mixed-precision scenarios;
        # the official code assumes float32 inputs).
        x_f = x.float()
        combined_x = torch.cat([x_f[edge_index[0]], x_f[edge_index[1]]], dim=1)
        b = F.softmax(F.leaky_relu(self.w[layer](combined_x)), dim=1)
        prompt = b.mm(self.anchor_prompt[layer])
        return prompt
