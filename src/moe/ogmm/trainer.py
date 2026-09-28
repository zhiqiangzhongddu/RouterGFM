"""OGMM runner: stage 0 experts -> stage 1 generation -> stage 2 merging -> query evaluation.

The inventory is OGMM's own domain protocol applied to the target support:
the support is split into edge-density domains, and one dense GCN / GAT / GIN
expert is trained per domain. Merging never sees real instances (source-free).
Only ``evaluate`` reads the test loader. The val loader is never read: under
few-shot and shift splits it holds unused instances.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path

import torch
from torch import optim
from torch_geometric.loader import DataLoader

from src.data_loader import create_dataset, dataset_info, log_split_instance_counts
from src.data_loader.shift_splits import verify_shift_root
from src.moe.identity import behavior_fingerprint
from src.moe.routergfm.common import infer_task_family
from src.moe.routergfm.losses import is_simplex_family
from src.moe.shift_eval import (
    brier_risk,
    collect_query_outputs,
    normalized_targets,
    raw_outputs,
    save_query_predictions,
    support_normalizer,
)
from src.utils.checkpoint import save_checkpoint, save_training_log
from src.utils.dataset_helpers import (
    is_few_shot_split,
    make_workflow_loaders,
    populate_dataset_cfg_from_meta,
    resolve_effective_task_level,
    shared_induced_root,
    shared_split_root,
)
from src.utils.naming import format_split_for_name
from src.utils.parsing import resolve_task_type, resolve_workflow_split
from src.utils.random import set_seed
from src.utils.supervised_eval import concat_and_compute_metrics
from src.utils.supervised_loss import resolve_supervised_output_dim, supervised_loss_from_logits

from .experts import DenseExpert, edge_density, partition_by_edge_density, to_dense_instances
from .generator import generate_for_expert
from .merge import NoisyTopKGate, OGMMMergedModel, merge_loss

# Generated graph size: median node count of the expert's domain, clamped (PROPOSED).
_GEN_MIN_NODES = 2
_GEN_MAX_NODES = 75


def _fmt(value) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


class OGMMRunner:
    """Run OGMM once (one seed) from ``cfg.moe.ogmm`` settings."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.ogmm_cfg = cfg.moe.ogmm
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        set_seed(seed=self.cfg.seed)

        self.best_metrics: dict[str, float] = {}
        self.best_epoch = None
        self.train_history: list[dict] = []

        ds_cfg = self.ogmm_cfg.dataset
        self.task_level_raw = str(ds_cfg.task_level).lower()
        self.induced = bool(getattr(ds_cfg, "induced", False))
        if self.task_level_raw in ("node", "edge") and not self.induced:
            raise ValueError("[OGMM] Node and edge tasks are consumed as induced subgraphs; set moe.ogmm.dataset.induced=True.")
        self.split = self._resolve_split()
        self.run_name = self._build_run_name()
        self.run_group = f"{ds_cfg.name}-{self.task_level_raw}"
        self.run_dir = os.path.join(self.ogmm_cfg.checkpoint_dir, self.run_group)
        self._skip_due_to_existing_checkpoint = False
        self._is_setup = False

        if self.ogmm_cfg.skip_if_exists and os.path.isfile(self._checkpoint_path()):
            self._skip_due_to_existing_checkpoint = True
            print(f"[OGMM] Checkpoint already exists, skipping: {self._checkpoint_path()}")

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _setup(self) -> None:
        if self._is_setup:
            return
        self._is_setup = True
        cfg, ogmm_cfg = self.cfg, self.ogmm_cfg
        ds_cfg = ogmm_cfg.dataset
        split_root = shared_split_root(cfg)
        if Path(split_root).resolve().parent == Path(str(cfg.data_preparation.shift.root)).resolve():
            # A missing/regenerated shift file would otherwise be silently replaced by a standard split.
            verify_shift_root(split_root, [(str(ds_cfg.name), self.task_level_raw, int(cfg.seed), self.split)])

        self.dataset = create_dataset(
            name=ds_cfg.name,
            root=ds_cfg.root,
            task_level=self.task_level_raw,
            feat_reduction=ds_cfg.feat_reduction,
            feat_reduction_dim=getattr(ds_cfg, "feat_reduction_svd_dim", 100),
            persist_feature_svd=ds_cfg.feat_reduction,
            feature_svd_dir=getattr(ds_cfg, "feature_svd_dir", "data/feature_svd"),
            induced=self.induced,
            induced_min_size=getattr(ds_cfg, "induced_min_size", 10),
            induced_max_size=getattr(ds_cfg, "induced_max_size", 30),
            induced_max_hops=getattr(ds_cfg, "induced_max_hops", 5),
            split_root=shared_split_root(cfg),
            induced_root=shared_induced_root(cfg, getattr(ds_cfg, "induced_root", "")),
            split=self.split,
            seed=cfg.seed,
        )
        self.effective_task_level = resolve_effective_task_level(self.task_level_raw, self.induced)
        self.dataset_meta = dataset_info(
            dataset=self.dataset, task_level=self.task_level_raw, name=ds_cfg.name, induced=self.induced,
        )
        # Fills ogmm_cfg.in_dim and ds_cfg.{num_classes,label_dim,task_type}.
        populate_dataset_cfg_from_meta(ogmm_cfg, ds_cfg, self.dataset_meta)
        self.in_dim = int(ogmm_cfg.in_dim)
        self.task_type = resolve_task_type(getattr(ds_cfg, "task_type", None))
        self.label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        self.num_classes = int(getattr(ds_cfg, "num_classes", 1) or 1)
        self.out_dim = resolve_supervised_output_dim(
            task_type=self.task_type, task_level=self.effective_task_level,
            label_dim=self.label_dim, num_classes=self.num_classes,
        )
        self.task_family = infer_task_family(self.task_level_raw, self.task_type, self.label_dim)

        self.train_loader, self.val_loader, self.test_loader = make_workflow_loaders(
            dataset=self.dataset,
            dataset_name=ds_cfg.name,
            task_level_raw=self.task_level_raw,
            effective_task_level=self.effective_task_level,
            batch_size=int(ogmm_cfg.batch_size),
            num_workers=int(ogmm_cfg.num_workers),
            split=self.split,
            seed=cfg.seed,
            induced=self.induced,
            split_root=shared_split_root(cfg),
        )
        log_split_instance_counts(
            self.train_loader, self.val_loader, self.test_loader,
            task_level=self.task_level_raw, split=self.split, induced=self.induced, prefix="[OGMM][Split]",
        )
        support = self.train_loader.dataset
        self.support = [support[i] for i in range(len(support))]
        if not self.support:
            raise ValueError("[OGMM] Empty support set.")
        self.support_targets = torch.cat([torch.as_tensor(item.y).reshape(1, -1).float() for item in self.support])
        self.normalizer = support_normalizer(self.support, self.task_family)
        if self.normalizer is not None:
            # Experts, generated labels (U[min, max] of the domain targets), and merging work in
            # support median/MAD-normalized units, as RouterGFM's heads; evaluation reports raw units.
            self.support = [item.clone() for item in self.support]
            for item in self.support:
                item.y = normalized_targets(self.normalizer, torch.as_tensor(item.y).float())

    # ------------------------------------------------------------------ #
    # Run-name / paths
    # ------------------------------------------------------------------ #
    def _resolve_split(self) -> tuple:
        split = resolve_workflow_split(getattr(self.ogmm_cfg.dataset, "fixed_split", None), default=(5, 0.0, 1.0))
        if is_few_shot_split(split) and (abs(float(split[1])) > 1e-6 or abs(float(split[2]) - 1.0) > 1e-6):
            raise ValueError("Few-shot split must be (shots, 0.0, 1.0).")
        return split

    def _build_run_name(self) -> str:
        o = self.ogmm_cfg
        ds_cfg = o.dataset
        behavior = o.clone()
        behavior.pop("prediction_dir", None)  # an output location, not behaviour
        fingerprint = behavior_fingerprint(
            behavior,
            external_behavior={
                "shared_split_root": shared_split_root(self.cfg),
                "shared_induced_root": shared_induced_root(self.cfg, getattr(ds_cfg, "induced_root", "")),
            },
        )
        parts = [
            "ogmm",
            ds_cfg.name,
            f"induced{int(self.induced)}",
            format_split_for_name(self.split),
            f"task{self.task_level_raw}",
            f"d{o.num_domains}",
            "-".join(str(a) for a in o.expert_archs),
            f"h{o.expert_hidden_dim}",
            f"k{o.top_k}",
            f"lg{_fmt(o.lambda_gate)}",
            f"lm{_fmt(o.lambda_mask)}",
            f"ee{o.expert_epochs}",
            f"ge{o.gen_epochs}",
            f"me{o.merge_epochs}",
            f"bs{o.batch_size}",
            f"cfg{fingerprint}",
            f"seed{self.cfg.seed}",
        ]
        return "_".join(str(p) for p in parts if p not in ("", None))

    def _checkpoint_path(self) -> str:
        return os.path.join(self.run_dir, f"{self.run_name}.pt")

    def get_checkpoint_path_for_metrics(self) -> str:
        return self._checkpoint_path()

    def _log_path(self) -> str:
        log_dir = getattr(self.ogmm_cfg, "log_dir", "")
        if log_dir:
            return os.path.join(log_dir, self.run_group, f"{self.run_name}_log.json")
        return os.path.join(self.run_dir, f"{self.run_name}_log.json")

    def _prediction_path(self) -> str:
        return os.path.join(self.ogmm_cfg.prediction_dir, self.run_group, f"{self.run_name}.pt")

    def _loader(self, items, shuffle: bool, seed_offset: int) -> DataLoader:
        generator = torch.Generator().manual_seed(int(self.cfg.seed) + seed_offset)
        return DataLoader(items, batch_size=int(self.ogmm_cfg.batch_size), shuffle=shuffle, generator=generator)

    def _loss(self, logits, labels):
        return supervised_loss_from_logits(logits=logits, labels=labels, task_type=self.task_type)[0]

    # ------------------------------------------------------------------ #
    # Stage 0: density domains and domain experts
    # ------------------------------------------------------------------ #
    def build_domains(self) -> list[list[int]]:
        """Partition the support by edge density; fall back to one domain if a domain has < 2 instances."""
        requested = int(self.ogmm_cfg.num_domains)
        domains = partition_by_edge_density(self.support, requested)
        if requested > 1 and min(len(d) for d in domains) < 2:
            print(f"[OGMM] Support of {len(self.support)} cannot fill {requested} domains of >= 2; using one domain.")
            domains = partition_by_edge_density(self.support, 1)
        self.domain_info = []
        for index, domain in enumerate(domains):
            density = [edge_density(self.support[i]) for i in domain]
            info = {"size": len(domain), "density_range": [min(density), max(density)]}
            if is_simplex_family(self.task_family):
                labels = Counter(int(torch.as_tensor(self.support[i].y).view(-1)[0]) for i in domain)
                info["class_counts"] = {str(k): v for k, v in sorted(labels.items())}
            self.domain_info.append(info)
            print(f"[OGMM] Domain {index}: {info}")
        return domains

    def _train_expert(self, expert: DenseExpert, items: list, seed_offset: int) -> dict:
        loader = self._loader(items, shuffle=True, seed_offset=seed_offset)
        optimizer = optim.Adam(expert.parameters(), lr=float(self.ogmm_cfg.expert_lr))
        best = {"train_loss": float("inf"), "epoch": 0}
        best_state = None
        for epoch in range(1, int(self.ogmm_cfg.expert_epochs) + 1):
            expert.train()
            total, count = 0.0, 0
            for batch in loader:
                inst = to_dense_instances(batch.to(self.device), self.task_level_raw)
                loss = self._loss(expert(inst), inst.y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss) * inst.x.size(0)
                count += inst.x.size(0)
            epoch_loss = total / max(count, 1)
            # Few-shot: no validation, so the monitor is the train loss.
            if epoch_loss < best["train_loss"]:
                best = {"train_loss": epoch_loss, "epoch": epoch}
                best_state = {k: v.detach().clone() for k, v in expert.state_dict().items()}
        if best_state is not None:
            expert.load_state_dict(best_state)
        expert.eval()
        return best

    def train_experts(self, domains: list[list[int]]) -> list[DenseExpert]:
        o = self.ogmm_cfg
        experts, self.expert_info = [], []
        for d, domain in enumerate(domains):
            items = [self.support[i] for i in domain]
            for arch in o.expert_archs:
                expert = DenseExpert(
                    arch, self.in_dim, int(o.expert_hidden_dim), self.out_dim, self.task_level_raw, float(o.expert_dropout),
                ).to(self.device)
                best = self._train_expert(expert, items, seed_offset=len(experts))
                info = {"arch": str(arch), "domain": d, **best}
                print(f"[OGMM] Expert {len(experts)}: {info}")
                experts.append(expert)
                self.expert_info.append(info)
        return experts

    # ------------------------------------------------------------------ #
    # Stage 1: generation
    # ------------------------------------------------------------------ #
    def generate(self, experts: list[DenseExpert], domains: list[list[int]]) -> list:
        """Invert every expert with its own domain's size and label statistics; concatenate."""
        o = self.ogmm_cfg
        generator = torch.Generator().manual_seed(int(self.cfg.seed))
        generated, self.generation_info = [], []
        for expert, info in zip(experts, self.expert_info):
            items = [self.support[i] for i in domains[info["domain"]]]
            sizes = torch.tensor([float(item.num_nodes) for item in items])
            num_nodes = int(min(max(round(float(sizes.quantile(0.5))), _GEN_MIN_NODES), _GEN_MAX_NODES))
            targets = torch.cat([torch.as_tensor(item.y).reshape(1, -1).float() for item in items])
            graphs = generate_for_expert(
                expert,
                num_graphs=int(o.gen_num_graphs),
                num_nodes=num_nodes,
                in_dim=self.in_dim,
                task_level_raw=self.task_level_raw,
                task_type=self.task_type,
                label_dim=self.label_dim,
                num_classes=self.num_classes,
                domain_targets=targets,
                epochs=int(o.gen_epochs),
                lr=float(o.gen_lr),
                tau=float(o.gen_gumbel_tau),
                edge_threshold=float(o.gen_edge_threshold),
                generator=generator,
                edge_hidden_dim=int(o.gen_edge_hidden_dim),
            )
            density = sum(edge_density(g) for g in graphs) / max(len(graphs), 1)
            self.generation_info.append({"num_nodes": num_nodes, "mean_edge_density": density})
            generated.extend(graphs)
        return generated

    # ------------------------------------------------------------------ #
    # Stage 2: merging
    # ------------------------------------------------------------------ #
    def merge(self, experts: list[DenseExpert], generated: list) -> OGMMMergedModel:
        """Train gate + head masks on the generated graphs only; keep the last epoch (paper: fixed epochs)."""
        o = self.ogmm_cfg
        self.top_k = min(int(o.top_k), len(experts))
        if self.top_k != int(o.top_k):
            print(f"[OGMM] top_k={o.top_k} exceeds {len(experts)} experts; using {self.top_k}.")
        model = OGMMMergedModel(experts, NoisyTopKGate(self.in_dim, len(experts), self.top_k)).to(self.device)
        self.merge_optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(o.merge_lr))
        loader = self._loader(generated, shuffle=True, seed_offset=10_000)
        for epoch in range(1, int(o.merge_epochs) + 1):
            start = time.time()
            model.train()
            total, count, parts = 0.0, 0, Counter()
            for batch in loader:
                inst = to_dense_instances(batch.to(self.device), self.task_level_raw)
                loss, log = merge_loss(
                    model, inst, task_type=self.task_type,
                    lambda_gate=float(o.lambda_gate), lambda_mask=float(o.lambda_mask),
                    gamma_p=float(o.gamma_p), gamma_v=float(o.gamma_v),
                )
                self.merge_optimizer.zero_grad()
                loss.backward()
                self.merge_optimizer.step()
                size = inst.x.size(0)
                total, count = total + float(loss) * size, count + size
                parts.update({k: v * size for k, v in log.items()})
            epoch_log = {"epoch": epoch, "loss": total / max(count, 1), **{k: v / max(count, 1) for k, v in parts.items()}}
            self.train_history.append({**epoch_log, "duration_sec": time.time() - start})
            print(
                f"[OGMM][Merge {epoch}/{o.merge_epochs}] loss={epoch_log['loss']:.4f} "
                f"task={epoch_log.get('task', float('nan')):.4f} gate={epoch_log.get('gate', float('nan')):.4f} "
                f"mask={epoch_log.get('mask', float('nan')):.4f}"
            )
        model.eval()
        return model

    # ------------------------------------------------------------------ #
    # Evaluation (the only reader of query labels)
    # ------------------------------------------------------------------ #
    def evaluate(self, model: OGMMMergedModel) -> dict[str, float]:
        model.eval()
        logits_buffer, gate_sum = [], torch.zeros(len(model.experts))

        def model_fn(batch):
            logits, gate = model(to_dense_instances(batch, self.task_level_raw))
            logits = raw_outputs(self.normalizer, logits)
            logits_buffer.append(logits.detach().cpu())
            gate_sum.add_(gate.detach().sum(dim=0).cpu())
            return logits

        self.query_outputs = collect_query_outputs(model_fn, self.test_loader, self.device, self.task_type, self.label_dim)
        num_queries = int(self.query_outputs["pred"].size(0))
        self.test_mean_gate = (gate_sum / max(num_queries, 1)).tolist()
        metrics = concat_and_compute_metrics(logits_buffer, [self.query_outputs["y"]], self.task_type, "test")
        metrics["test_brier"] = brier_risk(
            self.query_outputs, task_family=self.task_family, support_targets=self.support_targets,
            reg_kind=str(self.cfg.moe.routergfm.loss.regression),
        )
        return metrics

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def fit(self) -> None:
        if self._skip_due_to_existing_checkpoint:
            return
        self._setup()
        domains = self.build_domains()
        experts = self.train_experts(domains)
        generated = self.generate(experts, domains)
        model = self.merge(experts, generated)
        test_metrics = self.evaluate(model)

        self.best_epoch = int(self.ogmm_cfg.merge_epochs)
        final_loss = self.train_history[-1]["loss"] if self.train_history else float("nan")
        self.best_metrics = {"train_loss": float(final_loss), "best_epoch": self.best_epoch, **test_metrics}
        save_query_predictions(
            self._prediction_path(),
            self.query_outputs,
            {
                "method": "ogmm",
                "run_name": self.run_name,
                "dataset": str(self.ogmm_cfg.dataset.name),
                "task_level": self.task_level_raw,
                "task_family": self.task_family,
                "split": list(self.split),
                "split_root": shared_split_root(self.cfg),
                "seed": int(self.cfg.seed),
                "test_brier": float(test_metrics["test_brier"]),
            },
        )
        extra = {
            "ogmm_domains": self.domain_info,
            "ogmm_experts": self.expert_info,
            "ogmm_generation": self.generation_info,
            "ogmm_generated_graphs": [g.to_dict() for g in generated],
            "ogmm_top_k": self.top_k,
            "ogmm_test_mean_gate": self.test_mean_gate,
        }
        save_checkpoint(
            path=self._checkpoint_path(),
            model=model,
            optimizer=self.merge_optimizer,
            epoch=self.best_epoch,
            cfg=self.cfg,
            dataset_meta=self.dataset_meta,
            metrics=self.best_metrics,
            extra=extra,
        )
        save_training_log(
            path=self._log_path(),
            cfg=self.cfg,
            dataset_meta=self.dataset_meta,
            history=self.train_history,
            best_info={"epoch": self.best_epoch, "metric": float(final_loss), "monitor": None},
            extra={key: value for key, value in extra.items() if key != "ogmm_generated_graphs"},
        )
        print("[OGMM] Complete. " + " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))


__all__ = ["OGMMRunner"]
