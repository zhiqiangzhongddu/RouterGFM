from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.encoder import LAYER_CACHE_COMPATIBLE_MODELS, supports_layer_cache
from src.utils.pool import POOLERS, get_batch_vector, normalize_pool_mode, pool_nodes

from src.utils.config_helpers import cfg_default, tag_if_nondefault, validate_choice

from ..task_base import PretrainTask
from ..registry import register
from .utils import make_zero_loss


def get_positive_expectation(
    p_samples: torch.Tensor,
    measure: str = "JSD",
    average: bool = True,
) -> torch.Tensor:
    if measure != "JSD":
        raise ValueError(f"Unsupported InfoGraph measure: {measure}")
    ep = math.log(2.0) - F.softplus(-p_samples)
    return ep.mean() if average else ep


def get_negative_expectation(
    q_samples: torch.Tensor,
    measure: str = "JSD",
    average: bool = True,
) -> torch.Tensor:
    if measure != "JSD":
        raise ValueError(f"Unsupported InfoGraph measure: {measure}")
    eq = F.softplus(-q_samples) + q_samples - math.log(2.0)
    return eq.mean() if average else eq


def local_global_loss(
    l_enc: torch.Tensor,
    g_enc: torch.Tensor,
    batch: torch.Tensor,
    measure: str = "JSD",
) -> torch.Tensor:
    """Local-global JSD MI estimator.

    The caller is responsible for guarding degenerate batches
    (``num_graphs < 2`` or ``num_nodes == 0``) — see ``InfoGraph.step``.
    """
    num_nodes = l_enc.size(0)
    num_graphs = g_enc.size(0)
    pos_mask = F.one_hot(batch, num_classes=num_graphs).to(l_enc.dtype)
    neg_mask = 1.0 - pos_mask

    scores = torch.mm(l_enc, g_enc.t())

    # Masked-out positions evaluate to 0 under JSD expectations, so summing
    # directly matches the official normalization (cortex_DIM gan_losses).
    e_pos = get_positive_expectation(scores * pos_mask, measure=measure, average=False).sum()
    e_pos = e_pos / num_nodes

    e_neg = get_negative_expectation(scores * neg_mask, measure=measure, average=False).sum()
    e_neg = e_neg / (num_nodes * (num_graphs - 1))
    return e_neg - e_pos


class _FF(nn.Module):
    """Residual MLP discriminator used by InfoGraph."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.linear_shortcut(x)


class _PriorDiscriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


def prior_loss(y: torch.Tensor, prior_d: nn.Module, gamma: float) -> torch.Tensor:
    prior = torch.rand_like(y)
    eps = 1e-12
    term_a = torch.log(prior_d(prior).clamp(min=eps, max=1.0 - eps)).mean()
    term_b = torch.log((1.0 - prior_d(y)).clamp(min=eps, max=1.0)).mean()
    return -(term_a + term_b) * float(gamma)


# Backward-compat alias — canonical set now lives in src/model/encoder.py.
_LAYERWISE_COMPATIBLE_MODELS = LAYER_CACHE_COMPATIBLE_MODELS


@register("infograph")
class InfoGraph(PretrainTask):
    """
    Generalized InfoGraph objective from Sun et al. (ICLR 2020):
    - local-global MI with JSD (Fenchel-dual) estimator
    - optional prior matching regularization

    NOTE: This is a generalized implementation wired into the project's shared
    ``GNNEncoder`` (see ``src/model/encoder.py``). The InfoGraph *objective* is
    faithful to the paper, but the surrounding pipeline differs from the
    official reference in several ways that change effective behavior:
      - Project default backbone is 2-layer GCN (not the paper's 5-layer GIN).
      - Project default features go through SVD reduction (``feat_reduction``)
        — the paper uses raw TUDataset node features (or an all-ones fallback).
      - Node/edge datasets are converted to induced subgraphs
        (see ``src/data_loader/datasets.py``) to satisfy InfoGraph's
        graph-level batching requirement.
      - Training defaults (500 epochs, early stop on train loss,
        checkpointing) differ from the paper's 20-epoch no-early-stop loop.
    Set ``model.use_batchnorm True`` to more closely follow the paper's
    GIN-with-BN backbone.
    """

    requires_graph_batches = True
    min_graphs_per_batch = 2

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        task_cfg = cfg.pretrain.infograph
        validate_choice("pretrain.infograph.measure", str(task_cfg.measure), {"jsd"})
        pooling = normalize_pool_mode(task_cfg.graph_pooling)
        validate_choice("pretrain.infograph.graph_pooling", pooling, frozenset(POOLERS))
        gamma = float(task_cfg.gamma)
        if bool(task_cfg.prior) and gamma < 0.0:
            raise ValueError(
                f"[InfoGraph] pretrain.infograph.gamma must be >= 0 when prior=True; got {gamma}."
            )
        # Fail fast when the selected backbone does not cache per-layer
        # node representations. Previously this only surfaced inside
        # step() after dataset and encoder setup, wasting node-hours on
        # slurm. Only the GNNEncoder family (gcn/gin/gat/mlp) advertises
        # ``returns_layer_cache = True``; every other encoder backend
        # reaches an unreachable error path with use_layerwise=True.
        if bool(task_cfg.use_layerwise):
            model_name = str(getattr(getattr(cfg, "model", None), "name", "") or "").lower()
            if model_name and not supports_layer_cache(model_name):
                raise ValueError(
                    f"[InfoGraph] pretrain.infograph.use_layerwise=True requires an "
                    f"encoder that caches per-layer node representations. "
                    f"Model '{model_name}' is not supported. "
                    f"Fix: either set model.name to one of "
                    f"{sorted(_LAYERWISE_COMPATIBLE_MODELS)}, or set "
                    "pretrain.infograph.use_layerwise=False."
                )

    @classmethod
    def variant_tag(cls, cfg) -> str:
        _d = "pretrain.infograph"
        task_cfg = cfg.pretrain.infograph
        parts: list[str] = []
        pooling = normalize_pool_mode(task_cfg.graph_pooling)
        default_pooling = normalize_pool_mode(cfg_default(f"{_d}.graph_pooling"))
        parts.append(tag_if_nondefault("", pooling, default_pooling))
        if bool(task_cfg.use_layerwise) != bool(cfg_default(f"{_d}.use_layerwise")):
            parts.append("nolw")
        if bool(task_cfg.prior) != bool(cfg_default(f"{_d}.prior")):
            gamma = float(task_cfg.gamma)
            parts.append(f"pr{gamma:g}")
        measure = str(task_cfg.measure).upper()
        default_measure = str(cfg_default(f"{_d}.measure")).upper()
        parts.append(tag_if_nondefault("", measure.lower(), default_measure.lower()))
        return "-".join(p for p in parts if p)

    def __init__(self, cfg):
        super().__init__(cfg)
        task_cfg = cfg.pretrain.infograph
        self.measure = str(task_cfg.measure).upper()
        self.graph_pooling = normalize_pool_mode(task_cfg.graph_pooling)
        self.use_layerwise = bool(task_cfg.use_layerwise)
        self.use_prior = bool(task_cfg.prior)
        self.gamma = float(task_cfg.gamma)

        num_layers = max(1, int(cfg.model.num_layers))
        hidden_dim = int(cfg.model.hidden_dim)
        out_dim = int(cfg.model.out_dim)
        if self.use_layerwise:
            self.repr_dim = hidden_dim * max(0, num_layers - 1) + out_dim
        else:
            self.repr_dim = out_dim

        self.local_d = _FF(self.repr_dim)
        self.global_d = _FF(self.repr_dim)
        self.prior_d = _PriorDiscriminator(self.repr_dim) if self.use_prior else None

    def _resolve_representations(self, model, node_repr, graph_repr, batch):
        if self.use_layerwise:
            # The encoder's ``returns_layer_cache`` capability is checked
            # in ``step`` before we get here; at this point the accessor
            # is guaranteed to exist.
            layer_nodes = model.get_layer_node_reprs()
            if not layer_nodes:
                raise RuntimeError(
                    "InfoGraph use_layerwise=True but the encoder produced an "
                    "empty layer cache for this batch. This indicates an "
                    "encoder contract violation."
                )
            local_repr = torch.cat(layer_nodes, dim=-1)
            # Always pool with InfoGraph's own pooling setting, not the
            # encoder's cached graph reps (which may use a different mode).
            global_repr = torch.cat(
                [pool_nodes(x=h, batch=batch, mode=self.graph_pooling) for h in layer_nodes],
                dim=-1,
            )
        else:
            local_repr = node_repr
            encoder_pooling = normalize_pool_mode(self.cfg.model.graph_pooling)
            if graph_repr is None or encoder_pooling != self.graph_pooling:
                global_repr = pool_nodes(x=node_repr, batch=batch, mode=self.graph_pooling)
            else:
                global_repr = graph_repr

        if local_repr.size(-1) != self.repr_dim or global_repr.size(-1) != self.repr_dim:
            raise RuntimeError(
                "InfoGraph representation dim mismatch: "
                f"local={local_repr.size(-1)}, global={global_repr.size(-1)}, "
                f"expected={self.repr_dim}. Check "
                f"model.num_layers={self.cfg.model.num_layers}, "
                f"model.hidden_dim={self.cfg.model.hidden_dim}, "
                f"model.out_dim={self.cfg.model.out_dim}, "
                f"and whether the encoder's get_layer_node_reprs() output matches "
                f"the expected (hidden_dim*(num_layers-1) + out_dim) layout."
            )
        return local_repr, global_repr

    def step(self, model, data, device):
        # validate_cfg already rejects incompatible cfg.model.name + use_layerwise
        # combinations at task-construction time. This runtime check is a
        # safety net for cases where InfoGraph is constructed with a
        # custom encoder that bypasses cfg.model.name (e.g. direct tooling
        # use), so the error message mirrors validate_cfg's message.
        if self.use_layerwise and not getattr(model, "returns_layer_cache", False):
            raise RuntimeError(
                "[InfoGraph] use_layerwise=True but the encoder does not "
                "advertise returns_layer_cache. Fix: either use an encoder "
                f"in the layer-cache-compatible set "
                f"{sorted(_LAYERWISE_COMPATIBLE_MODELS)}, or set "
                "pretrain.infograph.use_layerwise=False."
            )
        data = data.to(device)
        node_repr, graph_repr = model(data)
        batch = get_batch_vector(data)
        local_repr, global_repr = self._resolve_representations(
            model=model,
            node_repr=node_repr,
            graph_repr=graph_repr,
            batch=batch,
        )

        g_enc = self.global_d(global_repr)
        l_enc = self.local_d(local_repr)

        # Guard degenerate batches before calling local_global_loss.
        # ``min_graphs_per_batch = 2`` is already set, but a rare custom
        # loader path could still hand us a 1-graph batch; return an
        # anchored zero-loss so backward() is safe, matching graphcl /
        # context_pred / edge_pred.
        if l_enc.size(0) == 0 or g_enc.size(0) <= 1:
            zero = make_zero_loss(self, device, l_enc, g_enc)
            return zero, {"mi_loss": 0.0}

        mi_loss = local_global_loss(l_enc=l_enc, g_enc=g_enc, batch=batch, measure=self.measure)

        prior_reg = make_zero_loss(self, device, global_repr)
        if self.prior_d is not None:
            prior_reg = prior_loss(y=global_repr, prior_d=self.prior_d, gamma=self.gamma)

        loss = mi_loss + prior_reg
        logs = {
            "mi_loss": float(mi_loss.detach().item()),
        }
        if self.prior_d is not None:
            logs["prior_loss"] = float(prior_reg.detach().item())
        return loss, logs
