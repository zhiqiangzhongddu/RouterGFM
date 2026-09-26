"""Riemannian (kappa-stereographic) GNN building blocks for GraphMoRE.

Ported from the official GraphMoRE implementation
(``ref_repos/GraphMoRE/models.py``) with two adaptations for this repo:

* ``kappaGCNConv.forward`` calls ``add_self_loops(edge_index,
  num_nodes=x.size(0))`` explicitly so that isolated centre nodes in an
  induced subgraph still receive their self-loop and normalise correctly.
* No hard-coded ``.cuda()`` — every op runs on the input tensor's device,
  so the same module works on CPU (smoke tests) and GPU (training).

Each expert is a ``RiemannianEncoder`` parameterised by a curvature
``k``: ``k<0`` hyperbolic, ``k>0`` spherical, ``k==0`` Euclidean (fixed,
non-learnable). The Stereographic manifold from ``geoopt`` covers all
three signs with a single set of Möbius operations.
"""

from __future__ import annotations

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class kappaLinear(nn.Module):
    """Möbius linear layer on a kappa-stereographic manifold."""

    def __init__(self, manifold, in_dim: int, out_dim: int, dropout: float = 0.0, use_bias: bool = True):
        super().__init__()
        self.manifold = manifold
        self.dropout = dropout
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = self.manifold.mobius_matvec(drop_weight, x, project=True)
        if self.use_bias:
            bias = self.manifold.proju(self.manifold.origin(self.bias.shape), self.bias)
            kappa_bias = self.manifold.expmap0(bias, project=True)
            res = self.manifold.mobius_add(res, kappa_bias, project=True)
        return res


class kappaGCNConv(MessagePassing):
    """Single kappa-GCN convolution: Möbius linear + tangent-space aggregation."""

    def __init__(self, k, in_dim: int, out_dim: int, learnable: bool = True):
        super().__init__(aggr="add")
        self.manifold = geoopt.Stereographic(k=k, learnable=learnable)
        self.lin = kappaLinear(manifold=self.manifold, in_dim=in_dim, out_dim=out_dim, use_bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Explicit num_nodes keeps degree-0 centre nodes (isolated in an
        # induced subgraph) in the graph so they still get a self-loop.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)

        x_tan0 = self.manifold.logmap0(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x_tan0, norm=norm)
        out = self.manifold.expmap0(out, project=True)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class RiemannianEncoder(nn.Module):
    """Two-layer kappa-GCN encoder (one GraphMoRE expert).

    ``forward`` returns manifold points; ``encode`` additionally maps the
    output back to the tangent space at the origin (``logmap0``) so the
    rest of the pipeline operates on ordinary Euclidean vectors.
    """

    def __init__(self, k, in_dim: int, hidden_dim: int, out_dim: int, learnable: bool = True):
        super().__init__()
        self.manifold = geoopt.Stereographic(k=k, learnable=learnable)
        self.encoder1 = kappaGCNConv(k, in_dim, hidden_dim, learnable=learnable)
        self.encoder2 = kappaGCNConv(k, hidden_dim, out_dim, learnable=learnable)

    def _embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.manifold.proju(self.manifold.origin(x.shape), x)
        x = self.manifold.expmap0(x, project=True)
        h = self.encoder1(x, edge_index)
        z = self.encoder2(h, edge_index)
        return z

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self._embed(x, edge_index)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self._embed(x, edge_index)
        return self.manifold.logmap0(z)


__all__ = ["kappaLinear", "kappaGCNConv", "RiemannianEncoder"]
