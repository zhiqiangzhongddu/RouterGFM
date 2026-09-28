"""Model Spider training and two-stage deployment ranking (Zhang et al., 2023, Sec. 4.3-4.4, Alg. 2).

Training follows the official loop: batches of historical tasks, a random
number k ~ U{train_specific_min..max} of experts per task receiving their
specific tokens, the Plackett-Luce loss on the true order of mu_{b,e}, Adam
(no weight decay), cosine schedule stepped per epoch. The epoch and the
re-ranking depth k_r are chosen jointly on validation applications (lowest
mean regret@K, then higher hit@K, then smaller k_r), never on targets.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.utils.checkpoint import save_torch_atomic

from ....descriptors import DescriptorStandardizer
from ..common import selection_metrics
from .data import SpecificCenterCache, SpiderTask, spider_dir
from .loss import plackett_luce_loss
from .model import ModelSpiderRanker
from .tokens import pad_token_sets

Ranking = List[Tuple[str, float]]


class ModelSpiderTrainer:
    """One ranker (per held-out base dataset and seed): fit on historical tasks, rank new ones."""

    def __init__(
        self,
        cfg,
        infra,
        expert_ids: Sequence[str],
        descriptor_dim: int,
        specific_dims: Sequence[int],
        seed: int,
        *,
        standardizer: Optional[DescriptorStandardizer] = None,
        centers: Optional[SpecificCenterCache] = None,
    ):
        m = cfg.moe.routergfm.baselines.model_spider
        self.cfg, self.mcfg = cfg, m
        self.topk = int(cfg.moe.routergfm.baselines.topk)
        self.seed = int(seed)
        self.descriptor_dim = int(descriptor_dim)
        self.standardizer = standardizer
        self.centers = centers or SpecificCenterCache(infra, spider_dir(cfg, "cache_dir"), int(m.regression_bins))
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        with torch.random.fork_rng(devices=self._rng_devices()):
            torch.manual_seed(self.seed)
            self.model = ModelSpiderRanker(
                expert_ids,
                descriptor_dim,
                specific_dims,
                token_dim=int(m.token_dim),
                num_heads=int(m.num_heads),
                dropout=float(m.dropout),
                type_prompts=bool(m.type_prompts),
                specific_token_mode=str(m.specific_token_mode),
            ).to(self.device)
        self.expert_index = {e: i for i, e in enumerate(self.model.expert_ids)}
        self._id_rank = {e: r for r, e in enumerate(sorted(self.model.expert_ids))}
        self.info: Dict[str, float] = {}
        self.trained_ids: List[str] = []  # experts with a historical evaluation in the training tasks

    def _rng_devices(self) -> List[int]:
        return [self.device.index or 0] if self.device.type == "cuda" else []

    # -- batching -----------------------------------------------------------------
    def _batch(self, tasks: Sequence[SpiderTask]):
        general, general_mask = pad_token_sets([t.general_centers.float() for t in tasks])
        width = max(len(t.candidate_ids) for t in tasks)
        idx = torch.zeros(len(tasks), width, dtype=torch.long)
        tie = torch.zeros(len(tasks), width, dtype=torch.long)
        mask = torch.zeros(len(tasks), width, dtype=torch.bool)
        target = torch.full((len(tasks), width), float("nan"))
        for b, t in enumerate(tasks):
            n = len(t.candidate_ids)
            idx[b, :n] = torch.tensor([self.expert_index[e] for e in t.candidate_ids])
            tie[b, :n] = torch.tensor([self._id_rank[e] for e in t.candidate_ids])
            mask[b, :n] = True
            if t.target_losses is not None:
                target[b, :n] = t.target_losses.float()
        dev = self.device
        return general.to(dev), general_mask.to(dev), idx.to(dev), mask.to(dev), target.to(dev), tie.to(dev)

    def _specific(self, task: SpiderTask, j: int) -> Optional[torch.Tensor]:
        """Specific centres of candidate *j*, or None when its readout width has no projection."""
        centers = self.centers.get(task.app, task.candidate_ids[j])
        return centers if int(centers.size(-1)) in self.model.specific_dims else None

    # -- two-stage ranking ---------------------------------------------------------
    @staticmethod
    def _order(task: SpiderTask, scores: torch.Tensor) -> List[int]:
        return sorted(range(len(task.candidate_ids)), key=lambda j: (-float(scores[j]), task.candidate_ids[j]))

    @torch.no_grad()
    def _stages(self, tasks: Sequence[SpiderTask], k_max: int):
        """Coarse scores, coarse orders, and scores with specific tokens for each task's top ``k_max``."""
        self.model.eval()
        general, general_mask, idx, mask, _, _ = self._batch(tasks)
        coarse = self.model.score(general, general_mask, idx, mask).cpu()
        orders = [self._order(t, coarse[b]) for b, t in enumerate(tasks)]
        specific = {}
        for b, t in enumerate(tasks):
            for j in orders[b][:k_max]:
                centers = self._specific(t, j)
                if centers is not None:
                    specific[(b, j)] = centers
        full = self.model.score(general, general_mask, idx, mask, specific).cpu() if specific else coarse
        return coarse, orders, full

    def _final(self, task: SpiderTask, coarse: torch.Tensor, full: torch.Tensor, order: List[int], k: int) -> Ranking:
        """Re-score the top-``k`` coarse experts; all others keep their coarse scores."""
        scores = coarse[: len(task.candidate_ids)].clone()
        top = order[:k]
        scores[top] = full[top]
        return [(task.candidate_ids[j], float(scores[j])) for j in self._order(task, scores)]

    def rank(self, task: SpiderTask, rerank_topk: int) -> Tuple[Ranking, Ranking, int]:
        """``(final ranking, coarse k=0 ranking, target executions)``, best first."""
        k = min(int(rerank_topk), len(task.candidate_ids))
        coarse, orders, full = self._stages([task], k)
        final = self._final(task, coarse[0], full[0], orders[0], k)
        return final, self._final(task, coarse[0], coarse[0], orders[0], 0), k

    def validate(self, tasks: Sequence[SpiderTask], grid: Sequence[int]) -> Dict[int, Tuple[float, float]]:
        """``{k_r: (mean regret@K, mean hit@K)}`` against the tasks' recorded mu over their candidates."""
        per_k: Dict[int, List[Tuple[float, float]]] = {int(k): [] for k in grid}
        bs = int(self.mcfg.batch_size)
        for start in range(0, len(tasks), bs):
            chunk = tasks[start:start + bs]
            coarse, orders, full = self._stages(chunk, max(per_k))
            for b, t in enumerate(chunk):
                risk = dict(zip(t.candidate_ids, t.target_losses.tolist()))
                for k in per_k:
                    m = selection_metrics(self._final(t, coarse[b], full[b], orders[b], k), risk, self.topk)
                    per_k[k].append((m["regret_at_k"], m["hit_at_k"]))
        return {k: (sum(r for r, _ in v) / len(v), sum(h for _, h in v) / len(v)) for k, v in per_k.items()}

    # -- training -------------------------------------------------------------------
    def fit(self, train_tasks: Sequence[SpiderTask], val_tasks: Sequence[SpiderTask]) -> Dict[str, float]:
        m = self.mcfg
        grid = sorted({int(k) for k in m.rerank_topk_grid})
        epochs, bs = int(m.epochs), int(m.batch_size)
        lo, hi = int(m.train_specific_min), int(m.train_specific_max)
        self.trained_ids = sorted({e for t in train_tasks for e in t.candidate_ids})
        best = None  # ((regret, -hit, k_r), epoch, state)
        with torch.random.fork_rng(devices=self._rng_devices()):
            torch.manual_seed(self.seed)
            gen = torch.Generator().manual_seed(self.seed)
            opt = torch.optim.Adam(self.model.parameters(), lr=float(m.lr), weight_decay=float(m.weight_decay))
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=float(m.lr_min))
            for epoch in range(1, epochs + 1):
                self.model.train()
                perm = torch.randperm(len(train_tasks), generator=gen).tolist()
                for start in range(0, len(perm), bs):
                    batch = [train_tasks[i] for i in perm[start:start + bs]]
                    k_spec = int(torch.randint(lo, hi + 1, (1,), generator=gen))
                    specific = {}
                    for b, t in enumerate(batch):
                        for j in torch.randperm(len(t.candidate_ids), generator=gen)[:k_spec].tolist():
                            centers = self._specific(t, j)
                            if centers is not None:
                                specific[(b, j)] = centers
                    general, general_mask, idx, mask, target, tie = self._batch(batch)
                    scores = self.model.score(general, general_mask, idx, mask, specific)
                    loss = plackett_luce_loss(scores, target, mask, tie)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                sched.step()
                crit = min((regret, -hit, k) for k, (regret, hit) in self.validate(val_tasks, grid).items())
                if best is None or crit < best[0]:  # exact ties keep the earlier epoch
                    best = (crit, epoch, copy.deepcopy(self.model.state_dict()))
        self.model.load_state_dict(best[2])
        self.model.eval()
        (regret, neg_hit, k_r), epoch, _ = best
        self.info = {
            "best_epoch": int(epoch),
            "rerank_topk": int(k_r),
            "val_regret": float(regret),
            "val_hit": float(-neg_hit),
            "n_train_tasks": len(train_tasks),
            "n_val_tasks": len(val_tasks),
        }
        return dict(self.info)

    # -- persistence ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        save_torch_atomic(str(path), {
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "expert_ids": list(self.model.expert_ids),
            "descriptor_dim": self.descriptor_dim,
            "specific_dims": list(self.model.specific_dims),
            "seed": self.seed,
            "standardizer": self.standardizer.state_dict() if self.standardizer is not None else None,
            "info": dict(self.info),
            "trained_ids": list(self.trained_ids),
        })

    @classmethod
    def load(cls, path: Path, cfg, infra, *, centers: Optional[SpecificCenterCache] = None) -> "ModelSpiderTrainer":
        payload = torch.load(str(path), map_location="cpu")
        state = payload["standardizer"]
        trainer = cls(
            cfg, infra, payload["expert_ids"], payload["descriptor_dim"], payload["specific_dims"], payload["seed"],
            standardizer=DescriptorStandardizer().load_state_dict(state) if state is not None else None,
            centers=centers,
        )
        trainer.model.load_state_dict(payload["state_dict"])
        trainer.model.eval()
        trainer.info = dict(payload["info"])
        trainer.trained_ids = list(payload["trained_ids"])
        return trainer


__all__ = ["ModelSpiderTrainer"]
