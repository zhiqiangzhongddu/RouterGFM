"""``ModelSpiderSelector``: Model Spider (Zhang et al., NeurIPS 2023) as a RouterGFM selection baseline.

One ranker per (target base dataset, seed) is trained on the historical
applications of every other base dataset (leave-one-dataset-out; router
validation groups select the epoch and k_r) and serves both budgets. At
deployment the target's support labels only partition its label-free support
descriptors into general tokens (coarse ranking of E_a, no expert executed);
the top-k_r experts then run on the support once for their specific tokens.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch

from src.moe.identity import behavior_fingerprint
from src.utils.checkpoint import cfg_to_dict

from ....common import AppSpec, enumerate_applications, stable_hash
from ....descriptors import DescriptorStandardizer
from ....router.trainer import validation_groups
from ..common import SelectionOutcome
from .data import SpecificCenterCache, build_spider_task, spider_dir
from .trainer import ModelSpiderTrainer


class ModelSpiderSelector:
    """Rank E_a by the two-stage Model Spider score (higher is better)."""

    name = "model_spider"

    def __init__(self, cfg, infra):
        self.cfg = cfg
        self.infra = infra
        self.mcfg = cfg.moe.routergfm.baselines.model_spider
        self.topk = int(cfg.moe.routergfm.baselines.topk)
        self.bins = int(self.mcfg.regression_bins)
        self.checkpoint_dir = spider_dir(cfg, "checkpoint_dir")
        self.centers = SpecificCenterCache(infra, spider_dir(cfg, "cache_dir"), self.bins)
        self._trainers: Dict[Tuple[str, int], ModelSpiderTrainer] = {}

    def split(self, app: AppSpec) -> Tuple[List[AppSpec], List[AppSpec]]:
        """Historical (train, val) applications outside the target group, one per data key.

        Validation groups are the router's (``router.val_datasets`` or the first
        ``router.num_val_datasets`` groups in declaration order, target family first).
        """
        declared = enumerate_applications(self.cfg.moe.routergfm)
        order = {a.key: i for i, a in enumerate(declared)}
        seen, history = set(), []
        for b in sorted(self.infra.historical_applications(app), key=lambda b: order.get(b.key, len(order))):
            if b.group != app.group and b.data_key not in seen:  # LP budgets share one support and history
                seen.add(b.data_key)
                history.append(b)
        families = {b.key: self.infra.task_family(b) for b in history}
        target_families = {self.infra.task_family(a) for a in declared + [app] if a.group == app.group}
        val_groups = validation_groups(self.cfg.moe.routergfm.router, app.group, history, families, target_families)
        return [b for b in history if b.group not in val_groups], [b for b in history if b.group in val_groups]

    def trainer_for(self, app: AppSpec) -> ModelSpiderTrainer:
        """Load (``baselines.skip_if_exists``) or train the ranker of (app.group, app.seed)."""
        key = (app.group, int(app.seed))
        if key in self._trainers:
            return self._trainers[key]
        rg = self.cfg.moe.routergfm
        train_apps, val_apps = self.split(app)
        expert_ids = [s.expert_id for s in self.infra.catalog]
        fingerprint = behavior_fingerprint(self.mcfg, external_behavior={
            "output_root": str(rg.output_root),
            "topk": self.topk,
            "descriptors": cfg_to_dict(rg.descriptors),
            "train": [b.data_key for b in train_apps],
            "val": [b.data_key for b in val_apps],
            "experts": stable_hash(expert_ids),
        })
        path = self.checkpoint_dir / f"heldout-{app.group}_seed{int(app.seed)}_{fingerprint}.pt"
        if bool(rg.baselines.skip_if_exists) and path.is_file():
            trainer = ModelSpiderTrainer.load(path, self.cfg, self.infra, centers=self.centers)
        else:
            z_train = torch.cat([self.infra.descriptors(b, "support") for b in train_apps])
            standardizer = DescriptorStandardizer(clip=float(rg.descriptors.clip)).fit(z_train)
            tasks = [
                build_spider_task(self.infra, b, standardizer, with_targets=True, regression_bins=self.bins)
                for b in train_apps + val_apps
            ]
            dims = set()
            for t in tasks:  # specific centres of every candidate: training samples them at random
                self.centers.warm(t.app, t.candidate_ids)
                dims |= {int(self.centers.get(t.app, e).size(-1)) for e in t.candidate_ids}
            trainer = ModelSpiderTrainer(
                self.cfg, self.infra, expert_ids, z_train.size(1), sorted(dims), int(app.seed),
                standardizer=standardizer, centers=self.centers,
            )
            info = trainer.fit(tasks[: len(train_apps)], tasks[len(train_apps):])
            print(f"[ModelSpider] {app.group} seed {app.seed}: {info}", flush=True)
            trainer.save(path)
        self._trainers[key] = trainer
        return trainer

    def rank(self, app: AppSpec) -> SelectionOutcome:
        started = time.perf_counter()
        trainer = self.trainer_for(app)
        task = build_spider_task(self.infra, app, trainer.standardizer, with_targets=False, regression_bins=self.bins)
        k_r = int(trainer.info["rerank_topk"])
        final, coarse, executions = trainer.rank(task, k_r)
        trained = set(trainer.trained_ids)
        return SelectionOutcome.from_ranking(
            app,
            final,
            self.topk,
            num_target_executions=executions,
            wall_time_sec=time.perf_counter() - started,
            variants={"k0": [e for e, _ in coarse[: self.topk]]},
            extras={
                **trainer.info,
                "num_untrained_tokens": sum(e not in trained for e in task.candidate_ids),
                "coarse_ranking": [[e, s] for e, s in coarse],
            },
        )


__all__ = ["ModelSpiderSelector"]
