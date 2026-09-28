"""SAGMM-PE matched-pool runner: ``SAGMMPERunner(cfg, app, infra)`` (DESIGN 11).

Frozen candidate experts (``baselines.candidate_rule``) contribute their RouterGFM
task readouts on S_a and Q_a; a TAAG gate and a task head are trained jointly on
the support labels (Adam, ``epochs``), with auxiliary losses and adaptive expert
pruning. Few-shot splits have no validation set, so the monitor is the support
task loss evaluated after each epoch (repo policy for splits without
validation); the best state within the last expert configuration predicts Q_a
once. Query labels are read only by ``infra.evaluate_outputs``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from yacs.config import CfgNode as CN

from src.utils.supervised_loss import supervised_loss_from_logits

from ...applications import derive_seed
from ...common import LINK, MULTILABEL, REGRESSION, AppSpec
from ...losses import is_simplex_family
from ..candidates import candidate_experts
from .gate_features import GateInputs, build_gate_inputs
from .model import SAGMMPE, contributions
from .pruning import ExpertPruner

_LOG = "[RouterGFM][sagmm_pe]"
_LEVEL_BLOCKS = ("edge", "graph")
_CHUNK = 4096  # instances per gate/head pass where the population is per instance


def level_params(block: CN, level: str) -> CN:
    """``baselines.sagmm_pe`` with the ``edge``/``graph`` sub-block of *level* applied."""
    params = CN({k: v for k, v in block.items() if k not in _LEVEL_BLOCKS})
    for key, value in block.get(level, CN()).items():
        if key not in params:
            raise KeyError(f"baselines.sagmm_pe.{level}.{key} overrides no base key.")
        params[key] = value
    return params


class SAGMMPERunner:
    def __init__(self, cfg, app: AppSpec, infra):
        self.cfg = cfg
        self.app = app
        self.infra = infra
        self.params = level_params(cfg.moe.routergfm.baselines.sagmm_pe, app.task_level)
        self.device = infra.device
        self.data = infra.data(app)
        self.family = self.data.task_family
        self.level = self.data.level
        self.seed = derive_seed(app.seed, "sagmm_pe", app.key)
        p = int(self.params.gate_input_p)
        self.gate_p = int(self.data.in_dim) if p < 0 else p
        self.expert_ids: List[str] = []
        self.model: Optional[SAGMMPE] = None
        self.best_epoch: Optional[int] = None
        self.best_metrics: Dict[str, float] = {}
        self.mean_active_experts = float("nan")
        self._query_pred: Optional[torch.Tensor] = None

    # -- inputs ------------------------------------------------------------------
    def _inv_norm(self, readout: torch.Tensor) -> torch.Tensor:
        kind = str(self.params.expert_norm).lower()
        if kind == "l2":
            return 1.0 / readout.norm(dim=-1).clamp_min(1e-12)
        if kind == "none":
            return torch.ones(readout.shape[:-1], device=readout.device)
        raise ValueError(f"Unknown sagmm_pe.expert_norm {kind!r} (expected l2|none).")

    def _support_readouts(self):
        """``[n, N, d]`` float32 raw readouts (float16 overflows large activations) and ``[n, N]`` normalisers."""
        experts, inv_norm = [], []
        for eid in self.expert_ids:
            emb = self.infra.embeddings(self.app, eid, "support").float()
            inv_norm.append(self._inv_norm(emb))
            experts.append(emb)
        widths = sorted({int(e.size(1)) for e in experts})
        if len(widths) != 1:
            raise ValueError(f"{self.app.key}: candidate readouts differ in width {widths}; SAGMM mixes one space.")
        return torch.stack(experts, dim=1).to(self.device), torch.stack(inv_norm, dim=1).to(self.device)

    def _gate_inputs(self, split: str, start: int = 0, stop: Optional[int] = None) -> GateInputs:
        positions = getattr(self.data, f"{split}_pos")
        stop = positions.numel() if stop is None else stop
        graphs = self.infra.instance_graphs(self.app, split)[start:stop]
        return build_gate_inputs(graphs, positions[start:stop], self.level, self.gate_p)

    def _support_target(self) -> torch.Tensor:
        target = self.infra.support_labels(self.app)
        if self.family == REGRESSION:
            target = self.infra.normalizer(self.app).transform(target)
        return target

    # -- task head -----------------------------------------------------------------
    def _task_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """CE (single-label), BCE (link logit), masked BCE (multi-label), MSE on median/MAD units (regression)."""
        task_type = "regression" if self.family == REGRESSION else "classification"
        return supervised_loss_from_logits(logits=logits, labels=target, task_type=task_type)[0]

    def _to_pred(self, logits: torch.Tensor) -> torch.Tensor:
        """Logits -> the family's prediction space (``common.PRED_SPACE``)."""
        if self.family == LINK:
            prob = torch.sigmoid(logits[:, 0])
            return torch.stack([1.0 - prob, prob], dim=-1)
        if is_simplex_family(self.family):
            return torch.softmax(logits, dim=-1)
        if self.family == MULTILABEL:
            return torch.sigmoid(logits)
        return logits

    # -- training ------------------------------------------------------------------
    def _train_batches(self, n: int, generator: torch.Generator) -> List[Optional[torch.Tensor]]:
        size = int(self.params.batch_size)
        if size <= 0 or size >= n:
            return [None]  # full support batch
        return list(torch.randperm(n, generator=generator).split(size))

    def _eval_batches(self, n: int) -> List[torch.Tensor]:
        """Graph level: consecutive SGA populations of ``batch_size``; otherwise memory chunks."""
        size = int(self.params.batch_size) if self.level == "graph" else _CHUNK
        return list(torch.arange(n).split(size if size > 0 else max(n, 1)))

    @torch.no_grad()
    def _support_loss(self, inputs: GateInputs, experts, inv_norm, target) -> float:
        self.model.eval()
        logits = []
        for idx in self._eval_batches(len(inputs)):
            idx = idx.to(self.device)
            logits.append(self.model(inputs.subset(idx), experts[idx], inv_norm[idx])[0])
        return float(self._task_loss(torch.cat(logits), target))

    def fit(self) -> None:
        p = self.params
        self.expert_ids = candidate_experts(self.cfg, self.app, self.infra)
        experts, inv_norm = self._support_readouts()
        inputs = self._gate_inputs("support").to(self.device)
        target = self._support_target().to(self.device)
        n = len(inputs)
        out_dim = 1 if self.family == LINK else int(self.data.num_classes)
        print(f"{_LOG} {self.app.key}: {len(self.expert_ids)} candidates ({self.cfg.moe.routergfm.baselines.candidate_rule}), "
              f"|S_a|={n}, gate input {inputs.query.size(1)}, readout {experts.size(2)}")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.model = SAGMMPE(
                inputs.query.size(1), len(self.expert_ids), experts.size(2), out_dim,
                score_act=str(p.score_act), threshold_init=str(p.threshold_init),
            ).to(self.device)
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=float(p.lr), weight_decay=float(p.weight_decay))
        pruner = ExpertPruner(
            len(self.expert_ids), ema_decay=float(p.ema_decay),
            threshold_factor=float(p.importance_threshold_factor), min_experts=int(p.min_experts),
        )
        generator = torch.Generator().manual_seed(self.seed)
        best_value, best_state = math.inf, None
        for epoch in range(1, int(p.epochs) + 1):
            model.train()
            for idx in self._train_batches(n, generator):
                if idx is None:
                    batch, h, w, y = inputs, experts, inv_norm, target
                else:
                    idx = idx.to(self.device)
                    batch, h, w, y = inputs.subset(idx), experts[idx], inv_norm[idx], target[idx]
                logits, gate = model(batch, h, w)
                loss = self._task_loss(logits, y) + model.gate.aux_loss(gate.gates, float(p.imp_weight), float(p.div_weight))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pick = gate.active if str(p.prune_type) == "new_logits" else gate.gates
                pruner.update(contributions(pick.detach() * w, h), model.gate.expert_mask)
            current = self._support_loss(inputs, experts, inv_norm, target)
            if current <= best_value:  # official node code keeps the latest tie
                best_value, self.best_epoch = current, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if bool(p.prune) and epoch % int(p.prune_interval) == 0:
                removed = pruner.prune(model.gate.expert_mask, n_train=n, current=current, best=best_value, mode="min")
                if removed:
                    left = int(model.gate.expert_mask.sum())
                    print(f"{_LOG} {self.app.key} epoch {epoch}: pruned {len(removed)} experts, {left} left")
                    best_value = math.inf  # best tracking restarts for the new expert configuration
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        metrics = self.evaluate()
        self.best_metrics = {f"test_{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        self.best_metrics["test_mean_active_experts"] = self.mean_active_experts
        self.best_metrics["final_num_experts"] = float(model.gate.expert_mask.sum())
        print(f"{_LOG} {self.app.key}: best epoch {self.best_epoch}, {self.best_metrics}")

    # -- inference -----------------------------------------------------------------
    @torch.no_grad()
    def _query_gates(self):
        """``G [|Q_a|, N]`` and active-expert counts ``k_u`` on CPU."""
        n = self.data.query_pos.numel()
        gates = torch.zeros(n, len(self.expert_ids))
        active = torch.zeros(n)
        whole = self._gate_inputs("query") if self.level == "graph" else None
        for idx in self._eval_batches(n):
            start, stop = int(idx[0]), int(idx[-1]) + 1
            inputs = whole.subset(idx) if whole is not None else self._gate_inputs("query", start, stop)
            out = self.model.gate(inputs.to(self.device))
            gates[start:stop] = out.gates.float().cpu()
            active[start:stop] = out.active.sum(-1).float().cpu()
        return gates, active

    @torch.no_grad()
    def predict_queries(self) -> torch.Tensor:
        """Query predictions in the family's prediction space (streams one expert's readouts at a time)."""
        if self._query_pred is None:
            self.model.eval()
            gates, active = self._query_gates()
            mixed = None
            for j, eid in enumerate(self.expert_ids):
                if not bool((gates[:, j] != 0).any()):
                    continue  # pruned or never selected
                emb = self.infra.embeddings(self.app, eid, "query").float().to(self.device)
                term = (gates[:, j].to(self.device) * self._inv_norm(emb))[:, None] * emb
                mixed = term if mixed is None else mixed + term
            logits = torch.cat([self.model.head(chunk) for chunk in mixed.split(_CHUNK)])
            self.mean_active_experts = float(active.mean())
            self._query_pred = self._to_pred(logits).float().cpu()
        return self._query_pred

    def evaluate(self) -> Dict[str, float]:
        return self.infra.evaluate_outputs(self.app, self.predict_queries())


__all__ = ["SAGMMPERunner", "level_params"]
