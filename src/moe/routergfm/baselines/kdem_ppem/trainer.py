"""KDEM / PPEM matched-pool runner (Liu et al., NeurIPS 2025; RouterGFM App. C).

Per application: score every eligible expert with the label-free competence of
Eq. 5 on the support (sub)graphs, keep the compatible group of the best one,
and merge its top-k experts with ``alpha = softmax(psi)`` (Eq. 7-8). The k
experts and a new task head are fine-tuned on S_a through the merged
parameters. KDEM adds ``gamma * MSE(merged, alpha-ensemble)`` on node
representations every ``kd.period`` global steps (Eq. 10-11); PPEM pulls the
team towards its merge every ``ema.period`` steps (Eq. 13). Q_a is predicted
by one merged encoder. The variant is ``baselines.method``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from src.utils.checkpoint import save_json_atomic
from src.utils.monitoring import is_metric_improved

from ...applications import derive_seed
from ...experts import load_frozen_encoder
from ...heads import activate, build_head_module, head_loss, training_target
from ...readout import graph_query_representation, readout_dim
from .merge import ema_pull_, resolve_ema_beta
from .model import MergedExpertModel
from .selection import (
    _LOG,
    MergeTeam,
    compat_key,
    competence_batches,
    competence_score,
    select_merge_team,
    team_candidates,
)

VARIANTS = ("kdem", "ppem")


def kd_loss(model: MergedExpertModel, batch, student_node_repr: torch.Tensor, *, detach: bool) -> torch.Tensor:
    """KDEM Eq. 10: MSE between the merged expert's and the alpha-weighted ensemble's node representations.

    The teacher is the in-training ensemble: the experts run in the model's current mode (dropout included).
    """
    if detach:
        with torch.no_grad():
            teacher = model.ensemble_node_repr(batch)
    else:
        teacher = model.ensemble_node_repr(batch)
    return F.mse_loss(student_node_repr, teacher)


def _cpu_state(module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


class KDEMPPEMRunner:
    """``Runner(cfg, app, infra)``: ``fit()``, ``predict_queries()``, ``evaluate()``, ``best_metrics``, ``best_epoch``."""

    def __init__(self, cfg, app, infra):
        variant = str(cfg.moe.routergfm.baselines.method).lower()
        if variant not in VARIANTS:
            raise ValueError(f"KDEMPPEMRunner needs baselines.method in {VARIANTS} (got {variant!r}).")
        self.cfg, self.app, self.infra, self.variant = cfg, app, infra, variant
        self.kcfg = cfg.moe.routergfm.baselines.kdem_ppem
        self.device = infra.device
        self.team: Optional[MergeTeam] = None
        self.beta: Optional[float] = None
        self.history: List[Dict[str, float]] = []
        self.best_epoch: Optional[int] = None
        self.best_metrics: Dict[str, float] = {}
        self.model: Optional[MergedExpertModel] = None  # trained experts (best epoch)
        self.encoder = None  # merged encoder used for Q_a
        self.head = None
        self._pool_mode = "mean"
        self._query_pred: Optional[torch.Tensor] = None

    def _seed(self, tag: str) -> int:
        # Shared by both variants: KDEM and PPEM see the same team, head init, and batches.
        return derive_seed(self.app.seed, "kdem_ppem", tag, self.app.data_key)

    def _load(self, expert_id: str, in_dim: int):
        spec = self.infra.catalog[self.infra.expert_index[expert_id]]
        encoder, model_cfg = load_frozen_encoder(self.cfg, spec, self.device)
        expert_in_dim = int(getattr(model_cfg.model, "in_dim", 0) or 0)
        if expert_in_dim and expert_in_dim != int(in_dim):
            raise ValueError(f"{self.app.key}: feature width {in_dim} does not match {expert_id} in_dim {expert_in_dim}.")
        return encoder, model_cfg

    # -- routing (Eq. 5, 7) ---------------------------------------------------------
    def _select_team(self, support, data) -> MergeTeam:
        kc = self.kcfg
        architecture = {spec.expert_id: spec.architecture for spec in self.infra.catalog}
        candidates = team_candidates(
            self.infra.compatible_pool(self.app), architecture, str(kc.group_policy), str(kc.fixed_arch)
        )
        batches = competence_batches(
            support,
            max_triplets=int(kc.competence.max_triplets),
            seed=self._seed("competence"),
            batch_size=int(self.cfg.moe.routergfm.device_batch_size),
            device=self.device,
        )
        if not batches:
            print(f"{_LOG} {self.app.key}: the support graphs have no edges; every competence score is 0.5.")
        scores, keys = {}, {}
        for expert_id in candidates:
            encoder, model_cfg = self._load(expert_id, data.in_dim)
            keys[expert_id] = compat_key(model_cfg)
            scores[expert_id] = competence_score(encoder, batches)
        return select_merge_team(scores, keys, int(kc.k))

    # -- fine-tuning through the merge (Eq. 8, 10-13) ----------------------------------
    def _train(self, support, data) -> None:
        kc, rg = self.kcfg, self.cfg.moe.routergfm
        loaded = [self._load(e, data.in_dim) for e in self.team.expert_ids]
        model_cfg = loaded[0][1].model
        self._pool_mode = str(model_cfg.graph_pooling)
        model = MergedExpertModel([enc.requires_grad_(True) for enc, _ in loaded], self.team.alpha).to(self.device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._seed("head"))
            head = build_head_module(readout_dim(int(model_cfg.out_dim), data.level), int(data.num_classes), rg.heads)
        head = head.to(self.device)
        family = data.task_family
        target = training_target(self.infra.support_labels(self.app), family, self.infra.normalizer(self.app))

        n, batch_size, epochs = len(support), int(kc.batch_size), int(kc.epochs)
        if self.variant == "ppem":
            self.beta = resolve_ema_beta(
                target_retention=float(kc.ema.target_retention),
                fallback_beta=float(kc.ema.beta),
                total_steps=epochs * math.ceil(n / batch_size),
                period=int(kc.ema.period),
            )
        optimizer = torch.optim.Adam(
            [
                {"params": list(model.experts.parameters()), "lr": float(kc.lr_expert)},
                {"params": list(head.parameters()), "lr": float(kc.lr_head)},
            ],
            weight_decay=float(kc.weight_decay),
        )
        kd_on = self.variant == "kdem" and float(kc.kd.weight) > 0
        generator = torch.Generator().manual_seed(self._seed("batches"))
        torch.manual_seed(self._seed("dropout"))
        best, best_state, wait, step = math.inf, None, 0, 0
        for epoch in range(1, epochs + 1):
            model.train()
            head.train()
            total, kd_steps = 0.0, 0
            for idx in torch.randperm(n, generator=generator).split(batch_size):
                step += 1
                batch = Batch.from_data_list([support[i] for i in idx.tolist()]).to(self.device)
                node_repr, graph_repr = model(batch)
                rep = graph_query_representation(node_repr, graph_repr, batch, task_level=data.level, pool_mode=self._pool_mode)
                task_loss = head_loss(head(rep), target[idx].to(self.device), family)
                loss = task_loss
                if kd_on and step % int(kc.kd.period) == 0:
                    loss = loss + float(kc.kd.weight) * kd_loss(model, batch, node_repr, detach=bool(kc.kd.detach_teacher))
                    kd_steps += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if self.variant == "ppem" and step % int(kc.ema.period) == 0:
                    ema_pull_(list(model.experts), model.alpha, self.beta)
                total += float(task_loss.detach()) * idx.numel()
            epoch_loss = total / n
            self.history.append({"epoch": epoch, "train_loss": epoch_loss, "kd_steps": kd_steps})
            if is_metric_improved(epoch_loss, best, "min"):
                best, wait, self.best_epoch = epoch_loss, 0, epoch
                best_state = (_cpu_state(model.experts), _cpu_state(head))
            else:
                wait += 1
                if wait >= int(kc.early_stopping):
                    break
        if best_state is not None:
            model.experts.load_state_dict(best_state[0])
            head.load_state_dict(best_state[1])
        self.model = model.eval()
        self.encoder = model.merged_module()
        self.head = head.eval()

    # -- runner contract -----------------------------------------------------------
    def fit(self) -> Dict[str, float]:
        data = self.infra.data(self.app)
        support = list(self.infra.instance_graphs(self.app, "support"))  # labels removed
        self.team = self._select_team(support, data)
        print(
            f"{_LOG}[{self.variant}] {self.app.key}: team={self.team.expert_ids} "
            f"psi={[round(p, 4) for p in self.team.competence]} alpha={[round(a, 4) for a in self.team.alpha.tolist()]}"
        )
        self._train(support, data)
        self.best_metrics = {f"test_{k}": float(v) for k, v in self.evaluate().items()}
        self._save_log()
        return self.best_metrics

    @torch.no_grad()
    def predict_queries(self) -> torch.Tensor:
        """Family-space predictions on Q_a (``data.query_pos`` order) from the merged encoder."""
        if self._query_pred is None:
            rg = self.cfg.moe.routergfm
            data = self.infra.data(self.app)
            loader = DataLoader(
                self.infra.instance_graphs(self.app, "query"),
                batch_size=int(rg.device_batch_size),
                shuffle=False,
                num_workers=int(self.kcfg.num_workers),
            )
            preds = []
            for batch in loader:
                batch = batch.to(self.device)
                node_repr, graph_repr = self.encoder(batch)
                rep = graph_query_representation(node_repr, graph_repr, batch, task_level=data.level, pool_mode=self._pool_mode)
                preds.append(activate(self.head(rep), data.task_family).float().cpu())
            self._query_pred = torch.cat(preds) if preds else torch.zeros(0, int(data.num_classes))
        return self._query_pred

    def evaluate(self) -> Dict[str, float]:
        return self.infra.evaluate_outputs(self.app, self.predict_queries())

    def _save_log(self) -> None:
        path = Path(str(self.cfg.moe.routergfm.baselines.output_dir)) / self.variant / "logs" / f"{self.app.key}.json"
        team = self.team
        save_json_atomic(
            str(path),
            {
                "method": self.variant,
                "app": self.app.to_dict(),
                "team": {
                    "compat_key": list(team.compat_key),
                    "expert_ids": team.expert_ids,
                    "competence": team.competence,
                    "alpha": team.alpha.tolist(),
                },
                "competence": team.all_scores,
                "beta": self.beta,
                "best_epoch": self.best_epoch,
                "history": self.history,
                "metrics": self.best_metrics,
            },
        )


__all__ = ["KDEMPPEMRunner", "VARIANTS", "kd_loss"]
