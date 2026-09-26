"""Mowst encoder: a weak expert and a strong expert.

The model holds two ordinary repo encoders (:class:`GNNEncoder`) — a *weak*
expert (typically an MLP over features) and a *strong* expert (a message-passing
GNN) — each producing a ``hidden_dim`` embedding. The per-expert supervised
heads and the routing gate live in :class:`~src.moe.mowst.task.MowstTask`,
matching the repo convention that the encoder library stays task-agnostic.

``forward`` returns both experts' ``(node_repr, graph_repr)`` so the task can
build per-expert logits via the shared supervised forward. ``forward_weak`` /
``forward_strong`` expose each expert alone for per-expert pretraining and for
the alternating ``mowst`` turns (where one expert is frozen under
``torch.no_grad()``).
"""

from __future__ import annotations

from torch import nn

from src.model.encoder import GNNEncoder


def _build_encoder(
    *,
    in_dim: int,
    hidden_dim: int,
    model_type: str,
    num_layers: int,
    dropout: float,
    act: str,
    use_batchnorm: bool,
    graph_pooling: str,
    gat_heads: int = 2,
) -> GNNEncoder:
    return GNNEncoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=hidden_dim,
        num_layers=num_layers,
        model_type=model_type,
        act=act,
        dropout=dropout,
        graph_pooling=graph_pooling,
        gat_heads=gat_heads,
        use_batchnorm=use_batchnorm,
    )


class MowstModel(nn.Module):
    """Two-expert encoder: ``weak`` (MLP) + ``strong`` (GNN)."""

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        weak_model: str,
        weak_num_layers: int,
        weak_dropout: float,
        strong_model: str,
        strong_num_layers: int,
        strong_dropout: float,
        gat_heads: int,
        act: str,
        use_batchnorm: bool,
        graph_pooling: str,
    ):
        super().__init__()
        self.weak = _build_encoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            model_type=weak_model,
            num_layers=weak_num_layers,
            dropout=weak_dropout,
            act=act,
            use_batchnorm=use_batchnorm,
            graph_pooling=graph_pooling,
        )
        self.strong = _build_encoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            model_type=strong_model,
            num_layers=strong_num_layers,
            dropout=strong_dropout,
            act=act,
            use_batchnorm=use_batchnorm,
            graph_pooling=graph_pooling,
            gat_heads=gat_heads,
        )

    def forward_weak(self, data):
        """Return ``(node_repr, graph_repr)`` from the weak expert."""
        return self.weak(data)

    def forward_strong(self, data):
        """Return ``(node_repr, graph_repr)`` from the strong expert."""
        return self.strong(data)

    def forward(self, data):
        """Return ``(weak_node, weak_graph, strong_node, strong_graph)``."""
        w_node, w_graph = self.weak(data)
        s_node, s_graph = self.strong(data)
        return w_node, w_graph, s_node, s_graph


__all__ = ["MowstModel"]
