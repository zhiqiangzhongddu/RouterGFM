import copy
from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
)

from src.utils.pool import get_pool_fn, normalize_pool_mode

from .activations import get_activation
from .h2gcn import H2GCNEncoder
from .fagcn import FAGCNEncoder
from .transformer import TransformerEncoder
from .nodeformer import NodeFormerEncoder


def _build_gin_mlp(
    in_channels: int, 
    out_channels: int, 
    act: nn.Module
) -> nn.Sequential:
    hidden = max(out_channels, in_channels)
    return nn.Sequential(
        nn.Linear(in_channels, hidden),
        act,
        nn.Linear(hidden, out_channels),
    )


def build_conv(
    model_type: str,
    in_channels: int,
    out_channels: int,
    act: nn.Module,
    gat_heads: int = 2,
) -> nn.Module:
    mtype = model_type.lower()
    if mtype == "gcn":
        return GCNConv(
            in_channels=in_channels, 
            out_channels=out_channels
        )
    if mtype == "gin":
        # Fresh activation instance per GIN MLP: embedding the encoder's
        # shared module would tie parametric activations (PReLU) across all
        # layers and register one tensor under several state_dict keys.
        return GINConv(
            _build_gin_mlp(in_channels, out_channels, copy.deepcopy(act))
        )
    if mtype == "gat":
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=gat_heads,
            concat=False,
        )
    if mtype == "mlp":
        return nn.Linear(
            in_features=in_channels, 
            out_features=out_channels
        )
    raise ValueError(f"Unknown model type: {model_type}")


class GNNEncoder(nn.Module):
    """
    Flexible encoder that supports MLP, GCN, GIN, and GAT backbones.
    Returns both node-level and graph-level representations.

    Exposes a per-layer node-representation cache via
    ``get_layer_node_reprs()`` and advertises the capability through
    ``returns_layer_cache = True`` so consumers (InfoGraph, GraphPrompt)
    can check availability without relying on attribute presence.
    """

    #: Advertises per-layer node-representation caching to downstream
    #: tasks. Any alternative encoder that does not cache per-layer
    #: representations should leave this at ``False`` (the default on
    #: ``nn.Module`` subclasses without this attribute — callers should
    #: read it via ``getattr(encoder, "returns_layer_cache", False)``).
    returns_layer_cache: bool = True

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        model_type: str = "gcn",
        act: str = "relu",
        dropout: float = 0.1,
        graph_pooling: str = "mean",
        gat_heads: int = 2,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.model_type = model_type.lower()
        self.act = get_activation(act)
        self.dropout = nn.Dropout(dropout)
        pool_key = normalize_pool_mode(graph_pooling)
        self.pool = get_pool_fn(pool_key)
        self.use_batchnorm = bool(use_batchnorm)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            out_c = out_dim if i == num_layers - 1 else hidden_dim
            conv = build_conv(self.model_type, in_c, out_c, self.act, gat_heads)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(out_c))

        self.out_dim = out_dim
        self.cached_layer_node_reprs = []
        self.cached_layer_graph_reprs = []

    def forward(self, data):
        x, edge_index = data.x, getattr(data, "edge_index", None)
        batch = getattr(data, "batch", None)
        layer_node_reprs = []

        for idx, conv in enumerate(self.convs):
            if self.model_type == "mlp":
                x = conv(x)
            else:
                if edge_index is None:
                    raise ValueError("edge_index is required for GNN models")
                x = conv(x, edge_index)
            if idx != len(self.convs) - 1:
                x = self.act(x)
                if self.use_batchnorm:
                    x = self.bns[idx](x)
                layer_node_reprs.append(x)
                x = self.dropout(x)
                continue
            layer_last = x
            if self.use_batchnorm:
                layer_last = self.bns[idx](layer_last)
            layer_node_reprs.append(layer_last)

        node_repr = layer_last
        graph_repr: Optional[torch.Tensor] = None
        if batch is not None:
            graph_repr = self.pool(node_repr, batch)

        self.cached_layer_node_reprs = layer_node_reprs
        if batch is not None:
            self.cached_layer_graph_reprs = [self.pool(layer_x, batch) for layer_x in layer_node_reprs]
        else:
            self.cached_layer_graph_reprs = []
        return node_repr, graph_repr

    def get_layer_node_reprs(self) -> list[torch.Tensor]:
        """Return per-layer node representations from the last forward pass.

        Documented accessor for the layer-cache capability advertised by
        ``returns_layer_cache``. Callers should prefer this over reading
        ``cached_layer_node_reprs`` directly.
        """
        return self.cached_layer_node_reprs


def build_encoder_from_cfg(cfg, in_dim: int) -> GNNEncoder:
    model_name = getattr(cfg.model, "name", "").lower()
    if model_name == "h2gcn":
        return H2GCNEncoder(
            in_dim=in_dim,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            act=cfg.model.activation,
            dropout=cfg.model.dropout,
            use_batchnorm=(
                bool(getattr(cfg.model, "use_batchnorm", False))
                or bool(getattr(getattr(cfg.model, "h2gcn", None), "use_batchnorm", False))
            ),
            graph_pooling=cfg.model.graph_pooling,
        )
    if model_name == "fagcn":
        fagcn_cfg = getattr(cfg.model, "fagcn", None)
        fagcn_eps = getattr(fagcn_cfg, "eps", getattr(cfg.model, "fagcn_eps", 0.1))
        return FAGCNEncoder(
            in_dim=in_dim,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            num_layers=cfg.model.num_layers,
            act=cfg.model.activation,
            dropout=cfg.model.dropout,
            eps=fagcn_eps,
            graph_pooling=cfg.model.graph_pooling,
            use_batchnorm=(
                bool(getattr(cfg.model, "use_batchnorm", False))
                or bool(getattr(fagcn_cfg, "use_batchnorm", False))
            ),
        )
    if model_name == "transformer":
        heads = getattr(cfg.model.gat, "heads", 4)
        return TransformerEncoder(
            in_dim=in_dim,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            num_layers=cfg.model.num_layers,
            heads=heads,
            dropout=cfg.model.dropout,
            act=cfg.model.activation,
            graph_pooling=cfg.model.graph_pooling,
        )
    if model_name == "nodeformer":
        return NodeFormerEncoder(
            cfg=cfg,
            in_dim=in_dim,
        )
    return GNNEncoder(
        in_dim=in_dim,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.out_dim,
        num_layers=cfg.model.num_layers,
        model_type=getattr(cfg.model, "name", model_name),
        act=cfg.model.activation,
        dropout=cfg.model.dropout,
        graph_pooling=cfg.model.graph_pooling,
        gat_heads=cfg.model.gat.heads,
        use_batchnorm=bool(getattr(cfg.model, "use_batchnorm", False)),
    )


#: Encoder names whose class caches per-layer node representations
#: (i.e. ``returns_layer_cache = True``).  GNNEncoder covers gcn, gin,
#: gat, and mlp; all other encoder backends do not.
LAYER_CACHE_COMPATIBLE_MODELS = frozenset({"gcn", "gin", "gat", "mlp"})


def supports_layer_cache(model_name: str) -> bool:
    """Return True when *model_name* resolves to a layer-cache-capable encoder."""
    return str(model_name).lower() in LAYER_CACHE_COMPATIBLE_MODELS
