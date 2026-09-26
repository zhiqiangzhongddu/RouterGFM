"""GMoPE downstream runner.

Mirrors ``src.moe.gmoe.trainer.GMoERunner``: same dataset loading, split
handling, monitoring, best-state selection, single final test evaluation,
checkpointing and logging. GMoPE-specific parts: the route checkpoint
(pretrained here when missing and ``stage='all'``), frozen experts, and an
optimizer over the prompts and the shared head only.
"""

from __future__ import annotations

import os
import time

import torch
from torch import optim

from src.data_loader import create_dataset, dataset_info, log_split_instance_counts
from src.moe.identity import behavior_fingerprint
from src.utils.checkpoint import cfg_to_dict, save_checkpoint, save_training_log
from src.utils.dataset_helpers import (
    is_few_shot_split,
    make_workflow_loaders,
    populate_dataset_cfg_from_meta,
    resolve_effective_task_level,
    shared_induced_root,
    shared_split_root,
)
from src.utils.monitoring import (
    is_metric_improved,
    merge_epoch_metrics,
    monitor_uses_train_split,
    resolve_auto_monitor_spec,
    resolve_explicit_monitor_spec,
    resolve_monitor_value,
    should_print_metric,
)
from src.utils.naming import format_split_for_name
from src.utils.parsing import resolve_task_type, resolve_workflow_split
from src.utils.paths import ensure_dir
from src.utils.random import set_seed
from src.utils.supervised_eval import runner_evaluate_split
from src.utils.training import run_step_epoch

from .pretrain import (
    GMoPEPretrainer,
    load_gmope_checkpoint,
    resolve_num_experts,
    resolve_pretrain_seed,
    resolve_route,
    resolve_top_k,
)
from .task import GMoPETask

_STAGES = {"all", "pretrain", "finetune"}
# Orchestration-only keys excluded from the run identity (on top of
# src.moe.identity._OPERATIONAL_KEYS).
_PRETRAIN_OPERATIONAL = ("routes", "num_workers", "checkpoint_dir", "skip_if_exists", "seed_policy")


class GMoPERunner:
    """Prompt-tune a pretrained GMoPE route checkpoint on one target application."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.gmope_cfg = cfg.moe.gmope
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        set_seed(seed=self.cfg.seed)

        self.best_metrics: dict[str, float] = {}
        self.best_epoch = None
        self.best_metric = float("nan")
        self.monitor_name = "val_acc"
        self.monitor_mode = "max"
        self.train_history: list[dict] = []

        stage = str(self.gmope_cfg.stage).lower()
        if stage not in _STAGES:
            raise ValueError(f"[GMoPE] moe.gmope.stage must be one of {sorted(_STAGES)} (got '{stage}').")
        self.stage = stage

        ds_cfg = self.gmope_cfg.dataset
        self.task_level_raw = str(ds_cfg.task_level)
        self.route = resolve_route(self.task_level_raw)
        self.num_experts = resolve_num_experts(self.gmope_cfg, self.route)
        self.top_k = resolve_top_k(self.gmope_cfg, self.route, "finetune")
        self.split = self._resolve_split()
        self.run_name = self._build_run_name()
        self.run_group = f"{ds_cfg.name}-{self.task_level_raw}"
        self.run_dir = os.path.join(self.gmope_cfg.checkpoint_dir, self.run_group)
        self._skip_due_to_existing_checkpoint = False
        self._is_setup = False

        ckpt_path = self._existing_checkpoint_path()
        if self.gmope_cfg.skip_if_exists and ckpt_path is not None:
            self._skip_due_to_existing_checkpoint = True
            print(f"[GMoPE] Checkpoint already exists, skipping: {ckpt_path}")
            return

    # ------------------------------------------------------------------ #
    # Lazy heavy setup
    # ------------------------------------------------------------------ #
    def _load_pretrained(self):
        pretrainer = GMoPEPretrainer(self.cfg, self.route)
        path = pretrainer.checkpoint_path()
        if not os.path.isfile(path):
            if self.stage != "all":
                raise FileNotFoundError(
                    f"[GMoPE] Missing pretrained '{self.route}' route checkpoint: {path}. "
                    "Run moe.gmope.stage pretrain (or all) first."
                )
            path = pretrainer.fit()
        self.pretrained_path = path
        model, objectives, meta = load_gmope_checkpoint(self.cfg, path, self.device)
        print(f"[GMoPE] Loaded '{self.route}' route checkpoint (M={meta['num_experts']}): {path}")
        return model, objectives

    def _setup(self) -> None:
        if self._is_setup:
            return
        self._is_setup = True

        cfg = self.cfg
        gmope_cfg = self.gmope_cfg
        ds_cfg = gmope_cfg.dataset
        raw_task_level = self.task_level_raw
        induced = bool(getattr(ds_cfg, "induced", False))

        self.model, route_objectives = self._load_pretrained()
        # Pretraining (when it ran in-process) consumed the global RNG.
        set_seed(seed=cfg.seed)

        self.dataset = create_dataset(
            name=ds_cfg.name,
            root=ds_cfg.root,
            task_level=raw_task_level,
            feat_reduction=ds_cfg.feat_reduction,
            feat_reduction_dim=getattr(ds_cfg, "feat_reduction_svd_dim", getattr(ds_cfg, "feat_reduction_dim", 100)),
            persist_feature_svd=ds_cfg.feat_reduction,
            feature_svd_dir=getattr(ds_cfg, "feature_svd_dir", "data/feature_svd"),
            induced=induced,
            induced_min_size=getattr(ds_cfg, "induced_min_size", 10),
            induced_max_size=getattr(ds_cfg, "induced_max_size", 30),
            induced_max_hops=getattr(ds_cfg, "induced_max_hops", 5),
            split_root=shared_split_root(cfg),
            induced_root=shared_induced_root(cfg, getattr(ds_cfg, "induced_root", "")),
            split=self.split,
            seed=cfg.seed,
            pad_featureless_features=True,
        )
        self.effective_task_level = resolve_effective_task_level(raw_task_level, induced)
        self.dataset_meta = dataset_info(
            dataset=self.dataset,
            task_level=raw_task_level,
            name=ds_cfg.name,
            induced=induced,
        )
        # Fills ds_cfg.{num_classes,label_dim,task_type}; in_dim is the fixed d0.
        populate_dataset_cfg_from_meta(gmope_cfg, ds_cfg, self.dataset_meta)
        feat_dim = int(self.dataset_meta.get("num_node_features") or 0)
        if feat_dim != int(gmope_cfg.in_dim):
            raise ValueError(
                f"[GMoPE] Target '{ds_cfg.name}' has feature dim {feat_dim}; moe.gmope.in_dim is "
                f"{int(gmope_cfg.in_dim)} (all sources and targets must share d0)."
            )

        self.model.freeze_experts()
        self.task = GMoPETask(
            cfg, num_experts=self.model.num_experts, route_objectives=list(route_objectives),
        ).to(self.device)
        print(
            f"[GMoPE] route={self.route} M={self.model.num_experts} K_down={self.task.top_k} "
            f"route_loss={self.task.route_loss} aggregation={self.task.aggregation}\n"
            f"[GMoPE] Task head:\n{self.task.classifier}"
        )

        params = [self.model.prompts] + list(self.task.parameters_to_optimize())
        self.optimizer = optim.Adam(
            params=params,
            lr=float(gmope_cfg.finetune.lr),
            weight_decay=float(gmope_cfg.finetune.weight_decay),
        )

        self._init_monitoring()
        ensure_dir(self.run_dir)
        self.train_loader, self.val_loader, self.test_loader = make_workflow_loaders(
            dataset=self.dataset,
            dataset_name=ds_cfg.name,
            task_level_raw=raw_task_level,
            effective_task_level=self.effective_task_level,
            batch_size=int(gmope_cfg.finetune.batch_size),
            num_workers=int(gmope_cfg.finetune.num_workers),
            split=self.split,
            seed=cfg.seed,
            induced=induced,
            split_root=shared_split_root(cfg),
        )
        log_split_instance_counts(
            self.train_loader,
            self.val_loader,
            self.test_loader,
            task_level=raw_task_level,
            split=self.split,
            induced=induced,
            prefix="[GMoPE][Split]",
        )
        if self.task.normalizer.enabled:
            self.task.normalizer.fit(self.train_loader)

    # ------------------------------------------------------------------ #
    # Monitoring
    # ------------------------------------------------------------------ #
    def _init_monitoring(self) -> None:
        ds_cfg = self.gmope_cfg.dataset
        label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        spec = resolve_explicit_monitor_spec(
            raw_monitor_metric=getattr(self.gmope_cfg.finetune, "monitor_metric", "auto"),
            setting_name="moe.gmope.finetune.monitor_metric",
        )
        if spec is None:
            spec = resolve_auto_monitor_spec(
                task_type=resolve_task_type(getattr(ds_cfg, "task_type", None)),
                task_level=str(self.task_level_raw or "").lower(),
                label_dim=label_dim,
                no_validation=is_few_shot_split(self.split),
            )
        self.monitor_name = spec.name
        self.monitor_mode = spec.mode
        self.best_metric = spec.best_metric
        self.best_epoch = None
        self.best_metrics = {}

    # ------------------------------------------------------------------ #
    # Run-name / paths
    # ------------------------------------------------------------------ #
    def _resolve_split(self) -> tuple:
        split = resolve_workflow_split(getattr(self.gmope_cfg.dataset, "fixed_split", None), default=(0.8, 0.1, 0.1))
        if is_few_shot_split(split):
            if abs(float(split[1])) > 1e-6 or abs(float(split[2]) - 1.0) > 1e-6:
                raise ValueError("Few-shot split must be (shots, 0.0, 1.0).")
        return split

    def _behavior_payload(self) -> dict:
        """``cfg.moe.gmope`` without orchestration keys, with M and K resolved."""
        payload = cfg_to_dict(self.gmope_cfg)
        payload.pop("stage", None)
        payload["num_experts"] = self.num_experts
        for key in _PRETRAIN_OPERATIONAL:
            payload["pretrain"].pop(key, None)
        payload["pretrain"]["top_k"] = resolve_top_k(self.gmope_cfg, self.route, "pretrain")
        payload["finetune"].pop("num_workers", None)
        payload["finetune"]["top_k"] = self.top_k
        return payload

    def _build_run_name(self) -> str:
        g = self.gmope_cfg
        ds_cfg = g.dataset
        objective = str(g.pretrain.objective)
        objective_block = getattr(self.cfg.pretrain, objective, None)
        fingerprint = behavior_fingerprint(
            self._behavior_payload(),
            external_behavior={
                "model_activation": str(getattr(self.cfg.model, "activation", "relu")),
                "shared_split_root": shared_split_root(self.cfg),
                "shared_induced_root": shared_induced_root(self.cfg, getattr(ds_cfg, "induced_root", "")),
                "objective_cfg": cfg_to_dict(objective_block) if objective_block is not None else {},
                "pretrain_seed": resolve_pretrain_seed(self.cfg),
            },
        )
        ft = g.finetune
        parts = [
            "gmope",
            ds_cfg.name,
            f"induced{int(getattr(ds_cfg, 'induced', False))}",
            format_split_for_name(self.split),
            f"task{self.task_level_raw}",
            self.route,
            objective,
            str(g.expert.gnn_type),
            f"m{self.num_experts}",
            f"k{self.top_k}",
            f"dp{int(g.prompt_dim)}",
            f"h{int(g.expert.hidden_dim)}",
            f"l{int(g.expert.num_layers)}",
            f"e{int(ft.epochs)}",
            f"lr{float(ft.lr):g}",
            f"bs{int(ft.batch_size)}",
            f"cfg{fingerprint}",
            f"seed{self.cfg.seed}",
        ]
        return "_".join(str(p) for p in parts if p not in ("", None))

    def _checkpoint_path(self) -> str:
        return os.path.join(self.run_dir, f"{self.run_name}.pt")

    def _existing_checkpoint_path(self) -> str | None:
        ckpt_path = self._checkpoint_path()
        return ckpt_path if os.path.isfile(ckpt_path) else None

    def get_checkpoint_path_for_metrics(self) -> str:
        return self._checkpoint_path()

    def _log_path(self) -> str:
        log_dir = getattr(self.gmope_cfg, "log_dir", "")
        if log_dir:
            return os.path.join(log_dir, self.run_group, f"{self.run_name}_log.json")
        return os.path.join(self.run_dir, f"{self.run_name}_log.json")

    def _save_training_log(self) -> None:
        save_training_log(
            path=self._log_path(),
            cfg=self.cfg,
            dataset_meta=self.dataset_meta,
            history=self.train_history,
            best_info={
                "epoch": self.best_epoch,
                "metric": self.best_metric,
                "monitor": self.monitor_name,
            },
            extra={"pretrained_from": self.pretrained_path},
        )

    # ------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------ #
    def _is_improved(self, metric: float) -> bool:
        return is_metric_improved(metric, self.best_metric, self.monitor_mode)

    def _record_best(self, epoch, train_loss, train_logs, val_metrics, monitor_value) -> bool:
        """Track the best epoch in memory (trainable state = prompts + head)."""
        if self.monitor_name is not None and not self._is_improved(monitor_value):
            return False
        self.best_metric = float(monitor_value)
        self.best_epoch = epoch
        self._best_context = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_logs": dict(train_logs),
            "val_metrics": dict(val_metrics),
            "monitor_value": float(monitor_value),
        }
        self._best_prompts = self.model.prompts.detach().cpu().clone()
        self._best_task_state = {k: v.detach().cpu().clone() for k, v in self.task.state_dict().items()}
        if self.monitor_name is None:
            print(f"[GMoPE] Best state updated at epoch={epoch} (monitor disabled).")
        else:
            print(f"[GMoPE] Best epoch updated: epoch={epoch} {self.monitor_name}={monitor_value:.4f}")
        return True

    def _finalize_best_checkpoint(self) -> None:
        context = self._best_context or self._last_context
        if context is None:
            return
        if self._best_prompts is not None:
            with torch.no_grad():
                self.model.prompts.copy_(self._best_prompts.to(self.model.prompts.device))
            self.task.load_state_dict(self._best_task_state)
        epoch = int(context["epoch"])
        if self.best_epoch is None:
            self.best_epoch = epoch
        test_metrics = self._evaluate_split(self.test_loader, prefix="test", mask_attr="test_mask")
        metrics = {
            "train_loss": context["train_loss"],
            "best_epoch": epoch,
            **context["train_logs"],
            **context["val_metrics"],
            **test_metrics,
        }
        if self.monitor_name is not None:
            metrics[self.monitor_name] = float(context["monitor_value"])
        save_checkpoint(
            path=self._checkpoint_path(),
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            cfg=self.cfg,
            dataset_meta=self.dataset_meta,
            metrics=metrics,
            extra={"gmope_task_state": self.task.state_dict(), "pretrained_from": self.pretrained_path},
        )
        self.best_metrics = metrics
        self._save_training_log()
        print(f"[GMoPE] Saved best-epoch checkpoint (epoch={epoch}) after final evaluation.")

    # ------------------------------------------------------------------ #
    # Train / eval loops
    # ------------------------------------------------------------------ #
    def train_epoch(self) -> tuple[float, dict]:
        # skip_model_train: the frozen experts stay in eval mode (dropout off).
        return run_step_epoch(
            model=self.model,
            task=self.task,
            loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            grad_clip=float(getattr(self.gmope_cfg.finetune, "grad_clip", 0.0) or 0.0),
            skip_model_train=True,
        )

    def _evaluate_split(self, loader, prefix: str, mask_attr: str) -> dict[str, float]:
        return runner_evaluate_split(
            model=self.model,
            task=self.task,
            loader=loader,
            device=self.device,
            prefix=prefix,
            mask_attr=mask_attr,
            task_type=resolve_task_type(getattr(self.gmope_cfg.dataset, "task_type", None)),
        )

    def fit(self) -> None:
        if getattr(self, "_skip_due_to_existing_checkpoint", False):
            return

        self._setup()

        if os.path.isfile(self._checkpoint_path()):
            print(f"[GMoPE] Overwriting existing checkpoint: {self._checkpoint_path()}")

        ft = self.gmope_cfg.finetune
        patience = int(getattr(ft, "early_stopping", 0) or 0)
        epochs_since_improvement = 0
        monitor_on_train = monitor_uses_train_split(self.monitor_name)
        self._best_context = None
        self._last_context = None
        self._best_prompts = None
        self._best_task_state = None

        for epoch in range(1, int(ft.epochs) + 1):
            start = time.time()
            train_loss, train_logs = self.train_epoch()
            if monitor_on_train:
                val_metrics = {}
            else:
                val_metrics = self._evaluate_split(self.val_loader, prefix="val", mask_attr="val_mask")

            duration = time.time() - start
            log_parts = [f"[GMoPE][Epoch {epoch}/{ft.epochs}]", f"train_loss={train_loss:.4f}"]
            for metrics in (train_logs, val_metrics):
                for k, v in metrics.items():
                    if should_print_metric(k):
                        log_parts.append(f"{k}={v:.4f}")
            log_parts.append(f"time={duration:.1f}s")
            print(" ".join(log_parts))

            merged_metrics = merge_epoch_metrics(train_logs, val_metrics, {})
            self.train_history.append(
                {
                    "epoch": epoch,
                    "loss": float(train_loss),
                    "duration_sec": float(duration),
                    "metrics": merged_metrics,
                }
            )

            monitor_value = resolve_monitor_value(
                self.monitor_name,
                train_loss=train_loss,
                train_logs=train_logs,
                val_metrics=val_metrics,
                test_metrics={},
            )
            self._last_context = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_logs": dict(train_logs),
                "val_metrics": dict(val_metrics),
                "monitor_value": float(monitor_value),
            }
            improved = self._record_best(
                epoch=epoch,
                train_loss=train_loss,
                train_logs=train_logs,
                val_metrics=val_metrics,
                monitor_value=monitor_value,
            )

            if patience > 0:
                epochs_since_improvement = 0 if improved else epochs_since_improvement + 1
                if epochs_since_improvement >= patience:
                    print(f"[GMoPE] Early stopping at epoch {epoch} (no improvement in {patience} epochs).")
                    break

        self._finalize_best_checkpoint()

        if self.monitor_name is not None:
            print(f"[GMoPE] Complete. Best {self.monitor_name}: {self.best_metric:.4f} at epoch {self.best_epoch}.")
        for metric_name in ("test_acc", "test_micro_f1", "test_macro_f1", "test_auc", "test_mae", "test_mse"):
            metric_value = self.best_metrics.get(metric_name)
            if metric_value is not None:
                print(f"[GMoPE] Best-epoch {metric_name}={metric_value:.4f}")


__all__ = ["GMoPERunner"]
