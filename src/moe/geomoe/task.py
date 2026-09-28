"""GeoMoE objective: ``alpha * L_task + beta * L_align + gamma * L_contr`` (Cao et al., 2026, Eq. 12).

* ``L_task``: the repo's shared supervised loss on one vector per instance
  (``src.moe.shift_eval.instance_readout``: target node / endpoint Hadamard /
  pooled graph); regression trains on support median/MAD-normalized targets
  (``normalizer``, set by the runner) and ``logits`` returns raw-unit outputs.
* ``L_align`` (Eqs. 7-8): ``KL(w* || w)`` between the ORC targets and the gate,
  averaged over the nodes of the batch.
* ``L_contr`` (Eqs. 9-10): InfoNCE with ``h_fused(v)`` as anchor, the expert
  of v's ORC region as positive, the two other experts of v as intra-node
  negatives, and the ``K - 2`` most similar other nodes of the mini-batch as
  hard negatives (nodes of a different ORC region first).

Training batches need ``data.node_orc`` (``curvature.attach_node_orc``);
evaluation never uses curvature.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.moe.shift_eval import instance_readout, normalized_targets, raw_outputs
from src.utils.parsing import resolve_task_type
from src.utils.supervised_loss import build_supervised_head, supervised_loss_from_logits

from .curvature import orc_region, orc_target_weights

# Row chunk for the [chunk, N] hard-negative similarity matrix.
_NEGATIVE_CHUNK = 4096


@torch.no_grad()
def hard_negative_index(h_pos: torch.Tensor, fused: torch.Tensor, region: torch.Tensor, k: int) -> torch.Tensor:
    """``[N, k]`` other nodes ranked by ``cos(h_pos(v), h_fused(u))``, different-region candidates first."""
    n = fused.size(0)
    query = F.normalize(h_pos, dim=-1)
    keys = F.normalize(fused, dim=-1)
    out = []
    for start in range(0, n, _NEGATIVE_CHUNK):
        stop = min(start + _NEGATIVE_CHUNK, n)
        score = query[start:stop] @ keys.T
        # Cosines lie in [-1, 1]; the offset ranks every same-region node below every other one.
        score = score - 3.0 * (region[start:stop, None] == region[None, :]).to(score.dtype)
        rows = torch.arange(stop - start, device=score.device)
        score[rows, rows + start] = float("-inf")
        out.append(score.topk(k, dim=1).indices)
    return torch.cat(out, dim=0)


class GeoMoETask(nn.Module):
    """Supervised head plus GeoMoE's curvature-guided regularisers."""

    def __init__(self, cfg):
        super().__init__()
        g = cfg.moe.geomoe
        ds_cfg = g.dataset
        self.task_level_raw = str(ds_cfg.task_level).lower()
        self.task_type = resolve_task_type(getattr(ds_cfg, "task_type", None))
        self.pool_mode = str(g.graph_pooling)
        self.theta = float(g.theta)
        self.eta = float(g.eta)
        self.num_negatives = int(g.num_negatives)
        if self.num_negatives < 2:
            raise ValueError("[GeoMoE] num_negatives (K) counts the 2 intra-node negatives; it must be >= 2.")
        self.contrast_temperature = float(g.contrast_temperature)
        self.alpha, self.beta, self.gamma = float(g.alpha), float(g.beta), float(g.gamma)
        # Every task is one vector per (sub)graph instance -> graph-level head sizing.
        self.classifier = build_supervised_head(
            in_dim=int(g.hidden_dim),
            task_type=self.task_type,
            task_level="graph",
            label_dim=int(getattr(ds_cfg, "label_dim", 1) or 1),
            num_classes=int(getattr(ds_cfg, "num_classes", 1) or 1),
        )
        self.normalizer = None  # support RegressionNormalizer for regression (set by the runner)

    def parameters_to_optimize(self):
        """Head parameters; the model's parameters are added by the runner."""
        return self.parameters()

    def logits(self, model, data) -> torch.Tensor:
        """Head outputs (raw target units for regression)."""
        node_repr, _ = model(data)
        logits = self.classifier(instance_readout(node_repr, data, self.task_level_raw, self.pool_mode))
        return raw_outputs(self.normalizer, logits)

    def align_loss(self, gate: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """``KL(w* || w)`` averaged over nodes; ``w*`` (Eq. 7) is a constant."""
        target = orc_target_weights(kappa, self.theta, self.eta).to(gate.dtype)
        return F.kl_div(gate.clamp_min(1e-8).log(), target, reduction="batchmean")

    def contrastive_logits(self, fused: torch.Tensor, experts: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """``[N, 1 + 2 + min(K - 2, N - 1)]`` cosine logits over ``tau_c``; column 0 is the positive."""
        n = fused.size(0)
        rows = torch.arange(n, device=fused.device)
        region = orc_region(kappa, self.theta)
        h_pos = experts[rows, region]
        intra = experts[rows[:, None], torch.stack([(region + 1) % 3, (region + 2) % 3], dim=1)]
        anchor = F.normalize(fused, dim=-1)
        sims = [
            (anchor * F.normalize(h_pos, dim=-1)).sum(-1, keepdim=True),
            (anchor[:, None] * F.normalize(intra, dim=-1)).sum(-1),
        ]
        k_inter = min(self.num_negatives - 2, n - 1)
        if k_inter > 0:
            index = hard_negative_index(h_pos, fused, region, k_inter)
            sims.append((anchor[:, None] * anchor[index]).sum(-1))
        return torch.cat(sims, dim=1) / self.contrast_temperature

    def contrastive_loss(self, fused: torch.Tensor, experts: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Eq. 10: mean over nodes of ``-log softmax(logits)[positive]``."""
        logits = self.contrastive_logits(fused, experts, kappa)
        return F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long, device=logits.device))

    def step(self, model, data, device):
        data = data.to(device)
        kappa = getattr(data, "node_orc", None)
        if kappa is None:
            raise ValueError("[GeoMoE] Training batches need data.node_orc (see curvature.attach_node_orc).")
        node_repr, _ = model(data)
        logits = self.classifier(instance_readout(node_repr, data, self.task_level_raw, self.pool_mode))
        task_loss, primary = supervised_loss_from_logits(
            logits=logits, labels=normalized_targets(self.normalizer, data.y), task_type=self.task_type
        )
        kappa = kappa.to(node_repr.dtype)
        align = self.align_loss(model.last_gate, kappa)
        contr = self.contrastive_loss(node_repr, model.last_expert, kappa)
        loss = self.alpha * task_loss + self.beta * align + self.gamma * contr
        log = {
            "train_task_loss": float(task_loss.detach().item()),
            "train_align_loss": float(align.detach().item()),
            "train_contr_loss": float(contr.detach().item()),
            ("train_mae" if self.task_type == "regression" else "train_acc"): primary,
        }
        return loss, log

    def evaluate(self, model, data, device, mask_attr="val_mask", return_outputs=False):
        """Task loss only (the regularisers are training-only); ``mask_attr`` is unused for instance batches."""
        data = data.to(device)
        return supervised_loss_from_logits(
            logits=self.logits(model, data), labels=data.y, task_type=self.task_type,
            return_outputs=return_outputs,
        )


__all__ = ["GeoMoETask", "hard_negative_index"]
