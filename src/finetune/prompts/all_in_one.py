"""Prompt modules for the All-in-One finetuning method."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data


class LightPrompt(torch.nn.Module):
    """Token-only prompt graph with learnable prompt nodes."""

    def __init__(self, token_dim: int, token_num_per_group: int, group_num: int = 1, inner_prune: float | None = None):
        super().__init__()
        self.inner_prune = inner_prune
        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for _ in range(group_num)]
        )
        self.token_init()

    def token_init(self) -> None:
        for token in self.token_list:
            torch.nn.init.kaiming_uniform_(token, nonlinearity="leaky_relu", mode="fan_in", a=0.01)

    def inner_structure_update(self) -> Batch:
        return self.token_view()

    def token_view(self) -> Batch:
        prompt_graphs = []
        for idx, tokens in enumerate(self.token_list):
            token_dot = torch.mm(tokens, tokens.t())
            token_sim = torch.sigmoid(token_dot)

            if self.inner_prune is None:
                inner_adj = token_sim
            else:
                inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            edge_index = inner_adj.nonzero(as_tuple=False).t().contiguous()
            prompt_graphs.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([idx], dtype=torch.long)))

        return Batch.from_data_list(prompt_graphs)


class HeavyPrompt(LightPrompt):
    """Connect prompt tokens to each input graph via learned cross edges.

    Architectural exception: unlike other modules under ``src/finetune/prompts``
    (which expose a stateless ``nn.Module`` forward that returns a prompted
    tensor), ``HeavyPrompt`` returns a modified ``torch_geometric.data.Batch``
    with a ``prompt_node_mask`` attribute and owns graph-merging logic that
    is conceptually part of the forward pass.  This is intentional: All-in-One
    treats prompt tokens as *additional graph nodes* with learned cross-edges,
    a graph-level construction that cannot be expressed as a plain tensor
    transform.  Future methods should not copy this shape — add a new module
    under ``src/finetune/encoders`` if encoder-level changes are needed, or
    keep prompts stateless otherwise.
    """

    def __init__(
        self,
        token_dim: int,
        token_num: int,
        cross_prune: float = 0.1,
        inner_prune: float = 0.01,
        bidirectional_cross_edges: bool = True,
    ):
        super().__init__(token_dim=token_dim, token_num_per_group=token_num, group_num=1, inner_prune=inner_prune)
        self.cross_prune = cross_prune
        self.bidirectional_cross_edges = bool(bidirectional_cross_edges)

    def forward(self, graph_batch: Batch) -> Batch:
        prompt_batch = self.inner_structure_update()
        inner_edge_index = prompt_batch.edge_index
        token_num = prompt_batch.x.shape[0]

        merged_graphs = []
        for graph in Batch.to_data_list(graph_batch):
            graph_edge_index = graph.edge_index + token_num

            cross_dot = torch.mm(prompt_batch.x, graph.x.t())
            cross_sim = torch.sigmoid(cross_dot)
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)

            cross_edge_index = cross_adj.nonzero(as_tuple=False).t().contiguous()
            if cross_edge_index.numel() > 0:
                cross_edge_index[1] = cross_edge_index[1] + token_num
                if self.bidirectional_cross_edges:
                    reverse_edge_index = torch.stack([cross_edge_index[1], cross_edge_index[0]], dim=0)
                    cross_edge_index = torch.cat([cross_edge_index, reverse_edge_index], dim=1)

            merged_x = torch.cat([prompt_batch.x, graph.x], dim=0)
            merged_y = graph.y
            merged_edge_index = torch.cat([inner_edge_index, graph_edge_index, cross_edge_index], dim=1)
            merged = Data(x=merged_x, edge_index=merged_edge_index, y=merged_y)

            # Track prompt vs original nodes so the task can exclude prompt nodes from pooling.
            merged.prompt_node_mask = torch.cat(
                [
                    torch.ones(token_num, dtype=torch.bool, device=merged_x.device),
                    torch.zeros(graph.x.size(0), dtype=torch.bool, device=merged_x.device),
                ],
                dim=0,
            )
            if hasattr(graph, "target_node"):
                merged.target_node = graph.target_node + token_num
            if hasattr(graph, "base_node_id"):
                merged.base_node_id = graph.base_node_id
            if hasattr(graph, "index"):
                merged.index = graph.index
            if hasattr(graph, "split"):
                merged.split = graph.split

            merged_graphs.append(merged)

        return Batch.from_data_list(merged_graphs)
