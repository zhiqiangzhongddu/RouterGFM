"""Frozen-encoder query embeddings with the task readout (App. B.1)."""

from __future__ import annotations

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from .readout import graph_query_representation, readout_dim


@torch.no_grad()
def embed_instances(encoder, model_cfg, data, positions: torch.Tensor, device, batch_size: int) -> torch.Tensor:
    """Readout of ``encoder`` for ``data.dataset[positions]``: CPU float32 ``[n, d]`` in position order.

    ``model_cfg`` is the checkpoint-restored cfg returned by
    ``experts.load_frozen_encoder`` (``model.graph_pooling``, ``model.out_dim``).
    """
    positions = torch.as_tensor(positions, dtype=torch.long).view(-1)
    model_in_dim = int(getattr(model_cfg.model, "in_dim", 0) or 0)
    if model_in_dim and int(data.in_dim) != model_in_dim:
        raise ValueError(
            f"{data.app.key}: feature width {data.in_dim} does not match the expert's in_dim {model_in_dim}."
        )
    width = readout_dim(int(model_cfg.model.out_dim), data.level)
    if positions.numel() == 0:
        return torch.zeros(0, width)
    encoder.eval()
    loader = DataLoader(Subset(data.dataset, positions.tolist()), batch_size=int(batch_size), shuffle=False)
    chunks = []
    for batch in loader:
        batch = batch.to(device)
        node_repr, graph_repr = encoder(batch)
        rep = graph_query_representation(
            node_repr,
            graph_repr,
            batch,
            task_level=data.level,
            pool_mode=model_cfg.model.graph_pooling,
        )
        chunks.append(rep.float().cpu())
    out = torch.cat(chunks, dim=0)
    if out.size(0) != positions.numel():
        raise RuntimeError(f"{data.app.key}: embedded {out.size(0)} of {positions.numel()} instances.")
    return out


__all__ = ["embed_instances"]
