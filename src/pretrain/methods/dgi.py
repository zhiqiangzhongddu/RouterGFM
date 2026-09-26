from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.pool import POOLERS, get_batch_vector, normalize_pool_mode, pool_nodes

from src.utils.config_helpers import cfg_default, tag_if_nondefault, validate_choice

from ..task_base import PretrainTask
from ..registry import register


class _Discriminator(nn.Module):
    """Bilinear discriminator used in the original DGI objective."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Bilinear(hidden_dim, hidden_dim, 1)
        nn.init.xavier_uniform_(self.scorer.weight)
        if self.scorer.bias is not None:
            nn.init.zeros_(self.scorer.bias)

    def forward(
        self,
        summary: torch.Tensor,
        pos_repr: torch.Tensor,
        neg_repr: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = summary[batch]
        pos_logits = self.scorer(pos_repr, context).view(-1)
        neg_logits = self.scorer(neg_repr, context).view(-1)
        return pos_logits, neg_logits


@register("dgi")
class DGI(PretrainTask):
    """DGI pretraining task.

    Reference: Veličković et al. "Deep Graph Infomax" ICLR 2019.
    """

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        readout = normalize_pool_mode(cfg.pretrain.dgi.readout)
        validate_choice("pretrain.dgi.readout", readout, frozenset(POOLERS))

    @classmethod
    def variant_tag(cls, cfg) -> str:
        readout = normalize_pool_mode(cfg.pretrain.dgi.readout)
        return tag_if_nondefault(
            "", readout, normalize_pool_mode(cfg_default("pretrain.dgi.readout")),
        )

    def __init__(self, cfg):
        super().__init__(cfg)
        task_cfg = cfg.pretrain.dgi
        self.readout_mode = normalize_pool_mode(task_cfg.readout)
        self.discriminator = _Discriminator(cfg.model.out_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def step(self, model, data, device):
        if getattr(data, "x", None) is None:
            raise ValueError("DGI requires node features `x` for corruption.")

        data = data.to(device)
        node_repr, _ = model(data)
        batch = get_batch_vector(data)
        # Original DGI uses average readout before sigmoid.
        summary = torch.sigmoid(pool_nodes(node_repr, batch, mode=self.readout_mode))

        # Corruption: globally shuffle node features across the entire
        # batch.  For single-graph inputs this is identical to the original
        # DGI paper (Veličković et al. 2019).  For batched graph-level
        # inputs the permutation crosses graph boundaries, which is an
        # intentional design choice: it provides stronger negative
        # examples and matches standard DGI implementations that operate
        # on a single mega-graph.
        corrupted = data.clone()
        perm = torch.randperm(data.num_nodes, device=device)
        corrupted.x = data.x[perm]

        neg_repr, _ = model(corrupted)
        pos_logits, neg_logits = self.discriminator(summary, node_repr, neg_repr, batch)

        logits = torch.cat((pos_logits, neg_logits), dim=0)
        labels = torch.cat((torch.ones_like(pos_logits), torch.zeros_like(neg_logits)), dim=0)
        loss = self.loss_fn(logits, labels)

        return loss, {
            "pos_mean": float(torch.sigmoid(pos_logits).detach().mean().item()),
            "neg_mean": float(torch.sigmoid(neg_logits).detach().mean().item()),
        }
