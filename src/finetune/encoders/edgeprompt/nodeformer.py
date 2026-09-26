"""Prompt-aware NodeFormer (extension, requires ``rb_order >= 1``).

Formula:
    z_i^(l)      = KernelAttention(h^(l))_i          # unchanged
    v_ji^(l)     = W_V h_j + p_ji                     # prompted value (raw prompt,
                                                      # NOT projected through W_V;
                                                      # per-head channels == prompt dim)
    RB_i^(l,r)   = sum_{j in N^(r)(i)} norm_ji * b^(l,r) * v_ji^(l)
    h_i^(l+1)    = W_O ( z_i^(l) + sum_r RB_i^(l,r) )

The prompt enters NodeFormer's NATIVE relational-bias path (already
present in ``src/model/nodeformer.py``) rather than inventing a new
local branch.  This means EdgePrompt on NodeFormer requires
``cfg.model.nodeformer.rb_order >= 1``; existing ``rb_order=0``
NodeFormer pretrain checkpoints do not expose the ``b`` parameter and
must be re-pretrained with a non-zero order to use NodeFormer
EdgePrompt.

State-dict keys are byte-identical to the vanilla NodeFormer model for
``rb_order=R`` -- we subclass ``NodeFormerConv`` and override only
``forward`` / add a prompted relational-bias helper.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import degree
from torch_sparse import SparseTensor

from src.model.nodeformer import (
    BIG_CONSTANT,
    NodeFormer,
    NodeFormerConv,
    create_projection_matrix,
    kernelized_gumbel_softmax,
    kernelized_softmax,
    transform_relational_bias,
)
from src.utils.pool import get_pool_fn

from .base import PromptAwareEncoder


def _prompted_relational_bias(
    value: torch.Tensor,
    edge_index: tuple[torch.Tensor, torch.Tensor],
    b: torch.Tensor,
    trans: str,
    edge_prompt: Optional[torch.Tensor],
) -> torch.Tensor:
    """Relational-bias aggregation with per-edge prompt injection.

    Mirrors ``add_conv_relational_bias`` from
    ``src/model/nodeformer.py`` exactly when ``edge_prompt`` is ``None``
    (checked by the zero-prompt parity test).  When a prompt is
    supplied, each edge's aggregated value becomes
    ``weight_e * (value[:, src, h, :] + edge_prompt[e])``.

    ``value`` has shape [B, N, H, C], ``edge_prompt`` has shape [E, C]
    (broadcast across heads and batch) or ``None``.
    """
    row, col = edge_index
    B, N, H, C = value.shape
    d_in = degree(col, N).float()
    d_norm_in = (1.0 / d_in[col]).sqrt()
    d_out = degree(row, N).float()
    d_norm_out = (1.0 / d_out[row]).sqrt()

    out = torch.zeros(B, N, H, C, device=value.device, dtype=value.dtype)
    for h in range(H):
        b_h = transform_relational_bias(b[h], trans)
        edge_val = (b_h * d_norm_in * d_norm_out).view(1, -1, 1)  # [1, E, 1]
        x_src = value[:, row, h, :]  # [B, E, C]
        if edge_prompt is not None:
            x_src = x_src + edge_prompt.unsqueeze(0)  # [B, E, C]
        weighted = x_src * edge_val  # [B, E, C]
        index = col.view(1, -1, 1).expand(B, -1, C)
        out[:, :, h, :].scatter_add_(1, index, weighted)
    return out


class PromptNodeFormerConv(NodeFormerConv):
    """NodeFormerConv with prompt-conditioned relational-bias path."""

    def forward(self, z, adjs, tau, edge_prompt_per_rb=None, batch=None):
        B, N = z.size(0), z.size(1)
        query = self.Wq(z).reshape(-1, N, self.num_heads, self.out_channels)
        key = self.Wk(z).reshape(-1, N, self.num_heads, self.out_channels)
        value = self.Wv(z).reshape(-1, N, self.num_heads, self.out_channels)

        if self.projection_matrix_type is None:
            projection_matrix = None
        else:
            dim = query.shape[-1]
            # Mirror the vanilla conv: data-dependent projection seed inside a
            # forked RNG so the global stream is never hijacked, with overflow
            # clamping (see src/model/nodeformer.py).
            seed_val = torch.nan_to_num(
                torch.abs(torch.sum(query.detach())) * BIG_CONSTANT,
                nan=0.0,
                posinf=float(2**31 - 1),
            )
            seed = int(seed_val.item()) % (2**31 - 1)
            with torch.random.fork_rng(devices=[]):
                projection_matrix = create_projection_matrix(
                    self.nb_random_features, dim, seed=seed
                ).to(query.device)

        if self.use_gumbel and self.training:
            attn_out = kernelized_gumbel_softmax(
                query, key, value, self.kernel_transformation,
                projection_matrix, adjs[0], self.nb_gumbel_sample, tau,
                self.use_edge_loss, batch=batch,
            )
        else:
            attn_out = kernelized_softmax(
                query, key, value, self.kernel_transformation,
                projection_matrix, adjs[0], tau, self.use_edge_loss, batch=batch,
            )

        if self.use_edge_loss:
            z_next, weight = attn_out
        else:
            z_next, weight = attn_out, None

        # Relational-bias path is the prompt injection point.
        # ``edge_prompt_per_rb`` is a list with one tensor per rb order
        # (or None in each slot to skip that order's prompt).
        prompts = edge_prompt_per_rb or [None] * self.rb_order
        for i in range(self.rb_order):
            prompt_i = prompts[i] if i < len(prompts) else None
            if prompt_i is None:
                # Preserve vanilla behavior: PyG's add_conv_relational_bias
                # is equivalent to _prompted_relational_bias with a None
                # prompt; using our helper keeps the code path uniform.
                rb = _prompted_relational_bias(
                    value, adjs[i], self.b[i], self.rb_trans, None
                )
            else:
                rb = _prompted_relational_bias(
                    value, adjs[i], self.b[i], self.rb_trans, prompt_i
                )
            z_next = z_next + rb

        z_next = self.Wo(z_next.flatten(-2, -1))

        if self.use_edge_loss:
            row, col = adjs[0]
            d_in = degree(col, query.shape[1]).float()
            d_norm = 1.0 / d_in[col]
            d_norm_ = d_norm.reshape(1, -1, 1).repeat(1, 1, weight.shape[-1])
            link_loss = torch.mean(weight.log() * d_norm_)
            return z_next, link_loss
        return z_next


class PromptNodeFormer(NodeFormer):
    """NodeFormer subclass that swaps in ``PromptNodeFormerConv``.

    All other state (bns, fcs, hyperparameters) is identical to the
    vanilla model, so a vanilla pretrain checkpoint loads cleanly
    *when* it was trained with the same ``rb_order``.  EdgePrompt
    requires ``rb_order >= 1``; that is enforced at the encoder level.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Swap vanilla NodeFormerConv -> PromptNodeFormerConv without
        # rebuilding any parameters.  Since PromptNodeFormerConv is a
        # pure subclass that only overrides ``forward``, rebinding
        # ``__class__`` is sufficient and preserves all weights, buffers,
        # and state_dict keys (same ``_modules`` / ``_parameters`` dict).
        for conv in self.convs:
            conv.__class__ = PromptNodeFormerConv
        self.cached_layer_node_reprs: list[torch.Tensor] = []

    def forward(
        self,
        x,
        adjs,
        tau=1.0,
        prompt=None,
        edge_index_for_prompt=None,
        rb_order: int = 0,
        batch=None,
    ):
        """Forward with per-layer prompt calls on the current hidden
        representation.

        ``prompt`` is the EdgePromptPlus/EdgePrompt module; we call
        ``prompt.get_prompt(z_layer, edge_index_for_prompt, layer=i)``
        inside the loop so the prompt sees the POST-``lin_in`` hidden
        features (dim=hidden_channels) rather than the raw input.  The
        resulting prompt is replicated ``rb_order`` times and fed to
        the conv's relational-bias path.
        """
        x = x.unsqueeze(0)
        layer_ = []
        link_loss_ = []
        z = self.fcs[0](x)
        if self.use_bn:
            z = self.bns[0](z)
        z = self.activation(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        layer_.append(z)

        for i, conv in enumerate(self.convs):
            edge_prompt_per_rb = None
            if prompt is not None:
                # z has shape [1, N, hidden]; the prompt's anchor weights
                # expect 2D [N, hidden] input.
                prompt_tensor = prompt.get_prompt(
                    z.squeeze(0), edge_index_for_prompt, layer=i
                )
                edge_prompt_per_rb = [prompt_tensor] * rb_order
            if self.use_edge_loss:
                z, link_loss = conv(z, adjs, tau, edge_prompt_per_rb=edge_prompt_per_rb, batch=batch)
                link_loss_.append(link_loss)
            else:
                z = conv(z, adjs, tau, edge_prompt_per_rb=edge_prompt_per_rb, batch=batch)
            if self.use_residual:
                z = z + layer_[i]
            if self.use_bn:
                z = self.bns[i + 1](z)
            if self.use_act:
                z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
            layer_.append(z)

        # Exclude the input projection and final output projection: these are
        # exactly the two post-convolution hop states for the fixed two-layer
        # reserve.  This plain list is intentionally absent from state_dict.
        self.cached_layer_node_reprs = [value.squeeze(0) for value in layer_[1:]]
        if self.use_jk:
            z = torch.cat(layer_, dim=-1)
        x_out = self.fcs[-1](z).squeeze(0)
        if self.use_edge_loss:
            return x_out, link_loss_
        return x_out


class PromptNodeFormerEncoder(PromptAwareEncoder):
    """NodeFormer encoder with prompt-conditioned relational-bias path.

    Requires ``cfg.model.nodeformer.rb_order >= 1``.  At construction
    time this is enforced; running with ``rb_order=0`` raises a clear
    error before any forward pass.
    """

    edgeprompt_support = "extension"
    edgeprompt_formula = "prompted relational-bias path (requires rb_order >= 1)"
    returns_layer_cache = True

    def __init__(self, cfg, in_dim: int):
        super().__init__()
        ncfg = getattr(cfg.model, "nodeformer", None)
        rb_order = int(getattr(ncfg, "rb_order", 0)) if ncfg is not None else 0
        if rb_order < 1:
            raise ValueError(
                "EdgePrompt on NodeFormer requires cfg.model.nodeformer.rb_order "
                f">= 1 (got {rb_order}). The prompt enters the relational-bias "
                "path, which only exists when rb_order is at least 1. Existing "
                "rb_order=0 NodeFormer pretrains cannot be used with "
                "EdgePrompt and must be re-pretrained with rb_order>=1."
            )
        if rb_order > 1:
            raise ValueError(
                "EdgePrompt's project NodeFormer encoder supports rb_order=1; "
                f"got {rb_order}. Higher orders require explicit adjacency powers."
            )
        heads = int(getattr(ncfg, "heads", 4))
        nb_random_features = int(getattr(ncfg, "num_random_features", 30))
        tau = float(getattr(ncfg, "tau", 1.0))
        dropout = float(getattr(cfg.model, "dropout", 0.1))
        self.tau = tau
        self.rb_order = rb_order
        self.use_edge_loss = bool(getattr(ncfg, "use_edge_loss", False))

        self.model = PromptNodeFormer(
            in_channels=in_dim,
            hidden_channels=cfg.model.hidden_dim,
            out_channels=cfg.model.out_dim,
            num_layers=cfg.model.num_layers,
            num_heads=heads,
            dropout=dropout,
            nb_random_features=nb_random_features,
            use_bn=bool(getattr(ncfg, "use_layernorm", True)),
            use_gumbel=bool(getattr(ncfg, "use_gumbel", True)),
            use_residual=bool(getattr(ncfg, "use_residual", True)),
            use_act=bool(getattr(ncfg, "use_activation", True)),
            use_jk=bool(getattr(ncfg, "use_jk", False)),
            nb_gumbel_sample=int(getattr(ncfg, "num_gumbel_samples", 10)),
            rb_order=rb_order,
            rb_trans=str(getattr(ncfg, "rb_trans", "sigmoid")),
            use_edge_loss=self.use_edge_loss,
        )
        self.pool = get_pool_fn(getattr(cfg.model, "graph_pooling", "mean"))
        self.out_dim = cfg.model.out_dim
        self.layer_cache_count = int(cfg.model.num_layers)

    def forward(
        self,
        data,
        prompt=None,
        prompt_type: str | None = None,
    ):
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("EdgePrompt NodeFormer requires edge_index.")
        if isinstance(edge_index, SparseTensor):
            coo = edge_index.coo()
            edge_index = (coo[0], coo[1])
        adjs = [(edge_index[0].to(data.x.device), edge_index[1].to(data.x.device))]
        use_prompt = prompt is not None and prompt_type in ("EdgePrompt", "EdgePromptplus")

        batch = getattr(data, "batch", None)
        out = self.model(
            data.x,
            adjs,
            tau=self.tau,
            prompt=prompt if use_prompt else None,
            edge_index_for_prompt=edge_index,
            rb_order=self.rb_order,
            batch=batch,
        )
        if self.use_edge_loss and isinstance(out, tuple):
            node_repr, _ = out
        else:
            node_repr = out

        graph_repr = None
        if batch is not None:
            graph_repr = self.pool(node_repr, batch)
        return node_repr, graph_repr

    def get_layer_node_reprs(self) -> list[torch.Tensor]:
        """Return post-convolution states from the latest forward."""
        return self.model.cached_layer_node_reprs


def build_prompt_nodeformer_encoder_from_cfg(cfg, in_dim: int) -> PromptNodeFormerEncoder:
    return PromptNodeFormerEncoder(cfg=cfg, in_dim=in_dim)
