"""IcG GraphCL variant.

This is a GraphCL-style graph contrastive pretraining task adapted to the
IcG framework. It is NOT a byte-for-byte reproduction of any
single official implementation; deliberate divergences from the official
GraphCL (TU / chem) references are:

* Node / edge datasets are bridged to graph batches via the project-wide
  induced-subgraph pipeline (``pretrain.dataset.induced=True``) rather than
  the original Cora/Citeseer DGI-style branch.
* Node features are typically projected through the project-wide SVD
  reduction (``cfg.pretrain.dataset.feat_reduction=True``) to unify feature
  dimensions across datasets; official GraphCL uses raw features.
* The contrastive loss uses the GraphCL ratio form
  ``-log(pos / (sum_row - pos))`` in a single direction, matching the
  unsupervised-TU, , and chem references.
* View augmentations can be selected per side via
  ``cfg.pretrain.graphcl.aug1 / aug2`` (``"random"`` samples independently
  with replacement from the enabled augs, so same-family pairs are allowed).

Reference: You et al. "Graph Contrastive Learning with Augmentations"
NeurIPS 2020.
"""

from __future__ import annotations

import random
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pretrain.augmentations import (
    VALID_AUG_NAMES as _VALID_AUG_NAMES_TUPLE,
    apply_per_graph_augment,
    edge_perturbation,
    feature_masking,
    node_dropping,
    subgraph_sampling,
)
from src.utils.pool import get_batch_vector, pool_nodes

from src.utils.config_helpers import cfg_default, tag_if_nondefault, validate_choice, validate_probability

from ..task_base import PretrainTask
from ..registry import register
from .utils import make_zero_loss


class _Projector(nn.Module):
    """Projection head for contrastive learning."""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


_VALID_AUG_NAMES = _VALID_AUG_NAMES_TUPLE


@register("graphcl")
class GraphCL(PretrainTask):
    """GraphCL: Graph Contrastive Learning with configurable augmentation pairs.

    See the module docstring for intentional divergences from the official
    reference implementations.
    """

    requires_graph_batches = True
    min_graphs_per_batch = 2

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        task_cfg = cfg.pretrain.graphcl
        _valid_augs = frozenset(_VALID_AUG_NAMES)
        for side in ("aug1", "aug2"):
            validate_choice(f"pretrain.graphcl.{side}", getattr(task_cfg, side), _valid_augs)
        for field in ("edge_remove_prob", "node_drop_prob", "feature_mask_prob"):
            validate_probability(f"pretrain.graphcl.{field}", getattr(task_cfg, field))
        temperature = float(task_cfg.temperature)
        if temperature <= 0.0:
            raise ValueError(
                f"[GraphCL] pretrain.graphcl.temperature must be > 0; got {temperature}."
            )
        subgraph_keep_ratio = float(task_cfg.subgraph_keep_ratio)
        if subgraph_keep_ratio <= 0.0 or subgraph_keep_ratio > 1.0:
            raise ValueError(
                f"[GraphCL] pretrain.graphcl.subgraph_keep_ratio must be in (0, 1]; "
                f"got {subgraph_keep_ratio}."
            )
        proj_hidden = int(task_cfg.proj_hidden)
        if proj_hidden < 1:
            raise ValueError(
                f"[GraphCL] pretrain.graphcl.proj_hidden must be >= 1; got {proj_hidden}."
            )
        # "subgraph" is only in the enabled aug set when use_subgraph_aug
        # is True AND subgraph_keep_ratio > 0. Requesting it otherwise
        # must fail here so the error shows up at run-name construction
        # time, not deep in __init__ after the skip-if-exists check.
        use_subgraph_aug = bool(task_cfg.use_subgraph_aug)
        subgraph_enabled = use_subgraph_aug and subgraph_keep_ratio > 0.0
        for side in ("aug1", "aug2"):
            name = str(getattr(task_cfg, side))
            if name == "subgraph" and not subgraph_enabled:
                raise ValueError(
                    f"[GraphCL] pretrain.graphcl.{side}='subgraph' requires "
                    "pretrain.graphcl.use_subgraph_aug=True and "
                    "pretrain.graphcl.subgraph_keep_ratio > 0."
                )

    @classmethod
    def variant_tag(cls, cfg) -> str:
        _d = "pretrain.graphcl"
        task_cfg = cfg.pretrain.graphcl
        parts: list[str] = []
        aug1 = str(task_cfg.aug1)
        aug2 = str(task_cfg.aug2)
        default_aug = str(cfg_default(f"{_d}.aug1"))
        if not (aug1 == default_aug and aug2 == default_aug):
            parts.append(f"{aug1}-{aug2}")
        parts.append(tag_if_nondefault("t", float(task_cfg.temperature), cfg_default(f"{_d}.temperature")))
        parts.append(tag_if_nondefault("ep", float(task_cfg.edge_remove_prob), cfg_default(f"{_d}.edge_remove_prob")))
        parts.append(tag_if_nondefault("np", float(task_cfg.node_drop_prob), cfg_default(f"{_d}.node_drop_prob")))
        parts.append(tag_if_nondefault("fp", float(task_cfg.feature_mask_prob), cfg_default(f"{_d}.feature_mask_prob")))
        if not bool(task_cfg.use_subgraph_aug):
            parts.append("nosub")
        elif tag_if_nondefault("sk", float(task_cfg.subgraph_keep_ratio), cfg_default(f"{_d}.subgraph_keep_ratio")):
            parts.append(tag_if_nondefault("sk", float(task_cfg.subgraph_keep_ratio), cfg_default(f"{_d}.subgraph_keep_ratio")))
        if bool(task_cfg.permE_add_edges):
            parts.append("peAdd")
        parts.append(tag_if_nondefault("ph", int(task_cfg.proj_hidden), cfg_default(f"{_d}.proj_hidden")))
        return "-".join(p for p in parts if p)

    def __init__(self, cfg):
        super().__init__(cfg)
        task_cfg = cfg.pretrain.graphcl
        self.tau = float(task_cfg.temperature)
        self.edge_remove_prob = float(task_cfg.edge_remove_prob)
        self.node_drop_prob = float(task_cfg.node_drop_prob)
        self.feature_mask_prob = float(task_cfg.feature_mask_prob)
        self.permE_add_edges = bool(task_cfg.permE_add_edges)
        self.use_subgraph_aug = bool(task_cfg.use_subgraph_aug)
        self.subgraph_keep_ratio = float(task_cfg.subgraph_keep_ratio)

        # permE_add_edges only takes effect on plain (edge_attr-less) graphs;
        # for datasets with edge_attr we always use delete-only permE so the
        # two tensors stay aligned. Warn once at init so the user isn't
        # surprised if the flag silently has no effect on their dataset.
        if self.permE_add_edges:
            print(
                "[GraphCL] permE_add_edges=True: random edges are added only "
                "for graphs without edge_attr. On datasets carrying edge_attr "
                "this flag is ignored and permE stays delete-only."
            )

        # Named augmentation registry. "subgraph" is included only when
        # explicitly enabled so the user-visible aug set matches the aug1/aug2
        # config values.
        self._aug_fns: dict[str, Callable] = {
            "dropN": lambda data: node_dropping(data, self.node_drop_prob),
            "permE": lambda data: edge_perturbation(
                data,
                self.edge_remove_prob,
                add_random_edges=self.permE_add_edges,
            ),
            "maskN": lambda data: feature_masking(data, self.feature_mask_prob),
        }
        if self.use_subgraph_aug and self.subgraph_keep_ratio > 0:
            self._aug_fns["subgraph"] = lambda data: subgraph_sampling(data, self.subgraph_keep_ratio)

        # aug1/aug2 names and their coherence with use_subgraph_aug are
        # fully validated in validate_cfg; no runtime-only invariants to
        # check here.
        self.aug1_name = str(task_cfg.aug1)
        self.aug2_name = str(task_cfg.aug2)

        hidden = int(task_cfg.proj_hidden)
        self.projector = _Projector(cfg.model.out_dim, hidden)

    def _readout(self, node_repr, data):
        """Pool node representations to graph level."""
        batch = get_batch_vector(data)
        return pool_nodes(node_repr, batch, mode=self.cfg.model.graph_pooling)

    def _contrastive_loss(self, z1, z2):
        """GraphCL ratio-form contrastive objective (one direction).

        Matches the official unsupervised-TU, and chem implementations::

            L_i = -log( exp(sim(z1_i, z2_i) / tau)
                        / sum_{j != i} exp(sim(z1_i, z2_j) / tau) )
        """
        sim = torch.exp((z1 @ z2.t()) / self.tau)        # [B, B]
        pos = torch.diag(sim)                             # [B]
        denom = (sim.sum(dim=1) - pos).clamp(min=1e-12)   # row-wise negatives
        loss = -torch.log(pos / denom).mean()

        with torch.no_grad():
            diag_sim = torch.diag(z1 @ z2.t()).mean().item()
        return loss, diag_sim

    def step(self, model, data, device):
        """Perform one pretraining step.

        Args:
            model: GNN encoder
            data: Batch of graphs
            device: torch device

        Returns:
            loss: scalar tensor
            logs: dict with logging info
        """
        data = data.to(device)

        # Resolve augmentation for each view. When aug1/aug2 is "random" we
        # sample independently (with replacement) from the enabled aug set,
        # so same-family pairs (e.g. dropN/dropN) are allowed -- matching
        # official GraphCL behavior.
        # Documented divergences from the official TU pipeline: the aug TYPE
        # is sampled once per step per view and applied to the whole batch
        # (official samples per graph in dataset.get), and maskN substitutes
        # the per-graph mean-feature token instead of N(0.5, 0.5) noise (see
        # augmentations.mask_nodes).
        enabled = list(self._aug_fns.keys())
        name1 = random.choice(enabled) if self.aug1_name == "random" else self.aug1_name
        name2 = random.choice(enabled) if self.aug2_name == "random" else self.aug2_name
        aug1 = apply_per_graph_augment(data, self._aug_fns[name1])
        aug2 = apply_per_graph_augment(data, self._aug_fns[name2])

        # Encode augmented views
        z1_nodes, g1 = model(aug1)
        z2_nodes, g2 = model(aug2)

        # Graph-level representations (pool if model doesn't return graph embedding)
        if g1 is None:
            g1 = self._readout(z1_nodes, aug1)
            g2 = self._readout(z2_nodes, aug2)

        # Pair alignment should stay exact because augmentations are per-graph.
        if g1.size(0) != g2.size(0):
            raise RuntimeError(
                "GraphCL positive-pair alignment failed (view batch sizes differ)."
            )

        # Project and normalize
        p1 = F.normalize(self.projector(g1), dim=-1)
        p2 = F.normalize(self.projector(g2), dim=-1)

        # The runner deliberately keeps partial batches. A trailing one-graph
        # batch cannot form a negative pair, so return an anchored zero-loss;
        # non-singleton partial batches remain fully useful.
        if p1.size(0) < 2:
            zero = make_zero_loss(self, device, p1)
            return zero, {"sim": 0.0, "batch_size_count": float(p1.size(0))}

        # Compute contrastive loss
        loss, sim_stat = self._contrastive_loss(p1, p2)

        return loss, {"sim": sim_stat, "batch_size_count": float(p1.size(0))}
