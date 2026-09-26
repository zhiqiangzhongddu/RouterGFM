"""Dense OGMM experts: instance conversion, edge-density domains, 2-layer GNNs with BatchNorm.

OGMM's stage 1 back-propagates into a soft adjacency, which the repo's sparse
encoders cannot take (they ignore edge weights). All instances are small
(ego-subgraphs, enclosing subgraphs, small graphs), so the experts run on dense
``[B, N, N]`` adjacencies. BatchNorm only sees valid (unpadded) nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import DenseGATConv, DenseGCNConv, DenseGINConv
from torch_geometric.utils import to_dense_adj, to_dense_batch

from src.utils.pool import get_batch_vector

EXPERT_ARCHS = ("gcn", "gat", "gin")


@dataclass
class DenseInstances:
    """A padded batch of instance graphs.

    ``anchor`` holds dense node positions: ``(t, t)`` for node tasks, ``(u, v)``
    for link prediction, ``(-1, -1)`` for graph tasks.
    """

    x: Tensor  # [B, N, d]
    adj: Tensor  # [B, N, N] float, symmetric, zero diagonal
    mask: Tensor  # [B, N] bool
    anchor: Tensor  # [B, 2] long
    y: Optional[Tensor] = None


def _anchor_positions(batch, level: str, ptr: Tensor) -> Tensor:
    num_graphs = ptr.numel()
    if level == "node":
        target = getattr(batch, "target_node_index", None)
        if target is None:
            raise ValueError("Node instances need data.target_node_index (induced ego-subgraphs).")
        local = torch.as_tensor(target).view(-1).to(ptr.device) - ptr
        if local.numel() != num_graphs:
            raise ValueError(f"{local.numel()} target nodes for {num_graphs} instances.")
        return torch.stack([local, local], dim=1)
    if level == "edge":
        pair = getattr(batch, "edge_label_index", None)
        if pair is None:
            raise ValueError("Edge instances need data.edge_label_index (one target pair per subgraph).")
        pair = torch.as_tensor(pair).to(ptr.device)
        if pair.size(1) != num_graphs:
            raise ValueError(f"{pair.size(1)} target pairs for {num_graphs} instances.")
        return torch.stack([pair[0] - ptr, pair[1] - ptr], dim=1)
    return torch.full((num_graphs, 2), -1, dtype=torch.long, device=ptr.device)


def to_dense_instances(batch, task_level_raw: str) -> DenseInstances:
    """Convert a PyG batch of instance graphs to dense tensors (edges symmetrised, no self-loops)."""
    level = str(task_level_raw).lower()
    batch_vec = get_batch_vector(batch)
    num_graphs = int(getattr(batch, "num_graphs", 1) or 1)
    x, mask = to_dense_batch(batch.x.float(), batch_vec, batch_size=num_graphs)
    adj = to_dense_adj(batch.edge_index, batch_vec, max_num_nodes=x.size(1), batch_size=num_graphs)
    adj = ((adj + adj.transpose(1, 2)) > 0).float()
    adj.diagonal(dim1=1, dim2=2).zero_()
    counts = torch.bincount(batch_vec, minlength=num_graphs)
    ptr = torch.cumsum(counts, dim=0) - counts
    return DenseInstances(x=x, adj=adj, mask=mask, anchor=_anchor_positions(batch, level, ptr), y=getattr(batch, "y", None))


def edge_density(data) -> float:
    """Undirected edges per node, ``|E| / |V|`` (self-loops and duplicates ignored)."""
    num_nodes = int(data.num_nodes or 0)
    edge_index = data.edge_index
    if num_nodes == 0 or edge_index.numel() == 0:
        return 0.0
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    pairs = torch.unique(torch.sort(edge_index, dim=0).values, dim=1)
    return float(pairs.size(1)) / num_nodes


def partition_by_edge_density(items: Sequence, num_domains: int) -> list[list[int]]:
    """Sort instances by edge density (ties by index) and cut into contiguous near-equal domains."""
    if int(num_domains) < 1:
        raise ValueError(f"num_domains must be >= 1 (got {num_domains}).")
    density = np.asarray([edge_density(item) for item in items], dtype=np.float64)
    order = np.lexsort((np.arange(len(items)), density))
    return [chunk.tolist() for chunk in np.array_split(order, int(num_domains))]


class SoftAdjDenseGAT(DenseGATConv):
    """``DenseGATConv`` whose attention is weighted by ``adj`` instead of masked by ``adj != 0``.

    ``alpha_ij = a_ij exp(e_ij) / sum_k a_ik exp(e_ik)``: identical to
    ``DenseGATConv`` on a binary adjacency, and differentiable in soft
    adjacency entries (the stock layer only reads the zero pattern).
    """

    def forward(self, x: Tensor, adj: Tensor, mask: Optional[Tensor] = None, add_loop: bool = True) -> Tensor:
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        heads, channels = self.heads, self.out_channels
        batch_size, num_nodes, _ = x.size()
        if add_loop:
            eye = torch.eye(num_nodes, dtype=adj.dtype, device=adj.device)
            adj = adj * (1.0 - eye) + eye

        x = self.lin(x).view(batch_size, num_nodes, heads, channels)
        alpha_src = torch.sum(x * self.att_src, dim=-1)
        alpha_dst = torch.sum(x * self.att_dst, dim=-1)
        alpha = F.leaky_relu(alpha_src.unsqueeze(1) + alpha_dst.unsqueeze(2), self.negative_slope)  # [B, N, N, H]

        weight = adj.unsqueeze(-1)
        shift = alpha.masked_fill(weight == 0, float("-inf")).amax(dim=2, keepdim=True).detach()
        scores = weight * torch.exp((alpha - shift).clamp(max=0.0))
        alpha = scores / scores.sum(dim=2, keepdim=True)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.matmul(alpha.movedim(3, 1), x.movedim(2, 1)).movedim(1, 2)
        out = out.reshape(batch_size, num_nodes, heads * channels) if self.concat else out.mean(dim=2)
        if self.bias is not None:
            out = out + self.bias
        if mask is not None:
            out = out * mask.view(-1, num_nodes, 1).to(x.dtype)
        return out


def _dense_conv(arch: str, in_dim: int, out_dim: int) -> nn.Module:
    if arch == "gcn":
        return DenseGCNConv(in_dim, out_dim)
    if arch == "gat":
        return SoftAdjDenseGAT(in_dim, out_dim, heads=1)
    if arch == "gin":
        return DenseGINConv(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))
    raise ValueError(f"Unknown OGMM expert architecture {arch!r} (expected one of {EXPERT_ARCHS}).")


def _masked_batch_norm(bn: nn.BatchNorm1d, h: Tensor, mask: Tensor) -> Tensor:
    """BatchNorm over valid nodes only; padded positions stay zero."""
    valid = h[mask]
    if bn.training and valid.size(0) < 2:
        # One node cannot give batch statistics; normalise with the running ones.
        normed = F.batch_norm(valid, bn.running_mean, bn.running_var, bn.weight, bn.bias, False, 0.0, bn.eps)
    else:
        normed = bn(valid)
    out = torch.zeros_like(h)
    out[mask] = normed
    return out


def dense_readout(h: Tensor, inst: DenseInstances, level: str) -> Tensor:
    """Target node (node), endpoint Hadamard product (edge), or masked mean (graph)."""
    rows = torch.arange(h.size(0), device=h.device)
    if level == "node":
        return h[rows, inst.anchor[:, 0]]
    if level == "edge":
        return h[rows, inst.anchor[:, 0]] * h[rows, inst.anchor[:, 1]]
    weight = inst.mask.unsqueeze(-1).to(h.dtype)
    return (h * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)


class DenseExpert(nn.Module):
    """Two dense conv layers + BatchNorm (one for GCN/GAT, two for GIN) + linear head (OGMM App. B)."""

    def __init__(self, arch: str, in_dim: int, hidden_dim: int, out_dim: int, task_level_raw: str, dropout: float):
        super().__init__()
        self.arch = str(arch).lower()
        self.level = str(task_level_raw).lower()
        self.dropout = float(dropout)
        self.conv1 = _dense_conv(self.arch, int(in_dim), int(hidden_dim))
        self.bn1 = nn.BatchNorm1d(int(hidden_dim))
        self.conv2 = _dense_conv(self.arch, int(hidden_dim), int(hidden_dim))
        self.bn2 = nn.BatchNorm1d(int(hidden_dim)) if self.arch == "gin" else None
        self.head = nn.Linear(int(hidden_dim), int(out_dim))

    def bn_layers(self) -> list[nn.BatchNorm1d]:
        return [bn for bn in (self.bn1, self.bn2) if bn is not None]

    def encode(self, inst: DenseInstances) -> Tensor:
        h = self.conv1(inst.x, inst.adj, inst.mask)
        h = F.relu(_masked_batch_norm(self.bn1, h, inst.mask))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, inst.adj, inst.mask)
        if self.bn2 is not None:
            h = _masked_batch_norm(self.bn2, h, inst.mask)
        return dense_readout(h, inst, self.level)

    def forward(self, inst: DenseInstances) -> Tensor:
        return self.head(self.encode(inst))


__all__ = [
    "EXPERT_ARCHS",
    "DenseExpert",
    "DenseInstances",
    "SoftAdjDenseGAT",
    "edge_density",
    "dense_readout",
    "partition_by_edge_density",
    "to_dense_instances",
]
