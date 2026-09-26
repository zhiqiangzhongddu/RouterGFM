"""Node-MoE training runner.

Mirrors ``src.moe.gmoe.trainer.GMoERunner`` (dataset loading, split handling,
monitoring, best-state tracking, single final test evaluation, checkpoint and
log writing) but reads ``cfg.moe.nodemoe``, builds :class:`NodeMoEModel` +
:class:`NodeMoETask`, and optimises three Adam parameter groups (gate, expert
dense layers, expert filters) as in the paper.
"""

from __future__ import annotations

import os
import time

import torch
from torch import optim

from src.data_loader import create_dataset, dataset_info, log_split_instance_counts
from src.moe.identity import behavior_fingerprint
from src.utils.checkpoint import save_checkpoint, save_training_log
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
from src.utils.paths import ensure_dir
from src.utils.parsing import resolve_task_type, resolve_workflow_split
from src.utils.random import set_seed
from src.utils.supervised_eval import runner_evaluate_split
from src.utils.supervised_loss import resolve_supervised_output_dim
from src.utils.training import run_step_epoch

from .model import NodeMoEModel, resolve_expert_specs
from .task import NodeMoETask, require_node_task_level


class NodeMoERunner:
    """Train a Node-MoE model from scratch using ``cfg.moe.nodemoe`` settings."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.nodemoe_cfg = cfg.moe.nodemoe
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        set_seed(seed=self.cfg.seed)

        # Summary fields defined up front so skip paths are safe.
        self.best_metrics: dict[str, float] = {}
        self.best_epoch = None
        self.best_metric = float("nan")
        self.monitor_name = "val_acc"
        self.monitor_mode = "max"
        self.train_history: list[dict] = []

        ds_cfg = self.nodemoe_cfg.dataset
        # Fail before any dataset loading: Node-MoE is node-task-scoped.
        self.task_level_raw = require_node_task_level(ds_cfg.task_level)
        resolve_expert_specs(self.nodemoe_cfg.expert_inits, self.nodemoe_cfg.expert_alphas)
        self.split = self._resolve_split()
        self.run_name = self._build_run_name()
        self.run_group = f"{ds_cfg.name}-{self.task_level_raw}"
        self.run_dir = os.path.join(self.nodemoe_cfg.checkpoint_dir, self.run_group)
        self._skip_due_to_existing_checkpoint = False
        self._is_setup = False

        ckpt_path = self._existing_checkpoint_path()
        if self.nodemoe_cfg.skip_if_exists and ckpt_path is not None:
            self._skip_due_to_existing_checkpoint = True
            print(f"[NodeMoE] Checkpoint already exists, skipping: {ckpt_path}")
            return

    # ------------------------------------------------------------------ #
    # Lazy heavy setup
    # ------------------------------------------------------------------ #
    def _setup(self) -> None:
        if self._is_setup:
            return
        self._is_setup = True

        cfg = self.cfg
        nodemoe_cfg = self.nodemoe_cfg
        ds_cfg = nodemoe_cfg.dataset
        raw_task_level = self.task_level_raw
        induced = bool(getattr(ds_cfg, "induced", False))

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
        )
        self.effective_task_level = resolve_effective_task_level(raw_task_level, induced)

        self.dataset_meta = dataset_info(
            dataset=self.dataset,
            task_level=raw_task_level,
            name=ds_cfg.name,
            induced=induced,
        )
        # Fills nodemoe_cfg.in_dim and ds_cfg.{num_classes,label_dim,task_type}.
        populate_dataset_cfg_from_meta(nodemoe_cfg, ds_cfg, self.dataset_meta)

        out_dim = resolve_supervised_output_dim(
            task_type=resolve_task_type(getattr(ds_cfg, "task_type", None)),
            task_level=raw_task_level,
            label_dim=int(getattr(ds_cfg, "label_dim", 1) or 1),
            num_classes=int(getattr(ds_cfg, "num_classes", 1) or 1),
        )
        self.model = NodeMoEModel(
            in_dim=int(nodemoe_cfg.in_dim),
            out_dim=out_dim,
            expert_inits=tuple(nodemoe_cfg.expert_inits),
            expert_alphas=tuple(nodemoe_cfg.expert_alphas),
            K=int(nodemoe_cfg.K),
            expert_hidden_dim=int(nodemoe_cfg.expert_hidden_dim),
            expert_dropout=float(nodemoe_cfg.expert_dropout),
            dprate=float(nodemoe_cfg.dprate),
            gate_hidden_dim=int(nodemoe_cfg.gate_hidden_dim),
            gate_num_layers=int(nodemoe_cfg.gate_num_layers),
            gate_dropout=float(nodemoe_cfg.gate_dropout),
            gate_feature_norm=str(nodemoe_cfg.gate_feature_norm),
            act=str(getattr(cfg.model, "activation", "relu")),
        ).to(self.device)
        print(f"[NodeMoE] Model architecture:\n{self.model}")

        self.task = NodeMoETask(cfg).to(self.device)

        # ChebNetII uses no LR scheduler; three groups, no global lr.
        self.optimizer = optim.Adam(
            self.model.param_groups(
                gate_lr=float(nodemoe_cfg.gate_lr),
                gate_wd=float(nodemoe_cfg.gate_weight_decay),
                expert_lr=float(nodemoe_cfg.expert_lr),
                expert_wd=float(nodemoe_cfg.expert_weight_decay),
                filter_lr=float(nodemoe_cfg.filter_lr),
                filter_wd=float(nodemoe_cfg.filter_weight_decay),
            )
        )

        self._init_monitoring()
        ensure_dir(self.run_dir)
        self.train_loader, self.val_loader, self.test_loader = make_workflow_loaders(
            dataset=self.dataset,
            dataset_name=ds_cfg.name,
            task_level_raw=raw_task_level,
            effective_task_level=self.effective_task_level,
            batch_size=int(nodemoe_cfg.batch_size),
            num_workers=int(nodemoe_cfg.num_workers),
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
            prefix="[NodeMoE][Split]",
        )

    # ------------------------------------------------------------------ #
    # Monitoring
    # ------------------------------------------------------------------ #
    def _init_monitoring(self) -> None:
        ds_cfg = self.nodemoe_cfg.dataset
        label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        spec = resolve_explicit_monitor_spec(
            raw_monitor_metric=getattr(self.nodemoe_cfg, "monitor_metric", "auto"),
            setting_name="moe.nodemoe.monitor_metric",
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
        split = resolve_workflow_split(getattr(self.nodemoe_cfg.dataset, "fixed_split", None), default=(0.8, 0.1, 0.1))
        if is_few_shot_split(split):
            val_ratio = float(split[1])
            test_ratio = float(split[2])
            if abs(val_ratio) > 1e-6 or abs(test_ratio - 1.0) > 1e-6:
                raise ValueError("Few-shot split must be (shots, 0.0, 1.0).")
        return split

    def _build_run_name(self) -> str:
        nodemoe_cfg = self.nodemoe_cfg
        ds_cfg = nodemoe_cfg.dataset
        fingerprint = behavior_fingerprint(
            nodemoe_cfg,
            external_behavior={
                "model_activation": str(getattr(self.cfg.model, "activation", "relu")),
                "shared_split_root": shared_split_root(self.cfg),
                "shared_induced_root": shared_induced_root(
                    self.cfg, getattr(ds_cfg, "induced_root", "")
                ),
            },
        )
        gamma = nodemoe_cfg.smoothing_gamma
        parts = [
            "nodemoe",
            ds_cfg.name,
            f"induced{int(getattr(ds_cfg, 'induced', False))}",
            format_split_for_name(self.split),
            f"task{self.task_level_raw}",
            f"m{len(nodemoe_cfg.expert_inits)}",
            f"K{nodemoe_cfg.K}",
            f"eh{nodemoe_cfg.expert_hidden_dim}",
            f"g{gamma:g}" if isinstance(gamma, (int, float)) else f"g{gamma}",
            str(nodemoe_cfg.readout),
            f"e{nodemoe_cfg.epochs}",
            f"bs{nodemoe_cfg.batch_size}",
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
        log_dir = getattr(self.nodemoe_cfg, "log_dir", "")
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
        )

    # ------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------ #
    def _is_improved(self, metric: float) -> bool:
        return is_metric_improved(metric, self.best_metric, self.monitor_mode)

    @staticmethod
    def _snapshot_state(module) -> dict:
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    def _record_best(
        self,
        epoch: int,
        train_loss: float,
        train_logs: dict[str, float],
        val_metrics: dict[str, float],
        monitor_value: float,
    ) -> bool:
        """Track the best epoch in memory; the checkpoint is written once by
        ``_finalize_best_checkpoint`` after the single test evaluation."""
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
        self._best_model_state = self._snapshot_state(self.model)
        if self.monitor_name is None:
            print(f"[NodeMoE] Best state updated at epoch={epoch} (monitor disabled).")
        else:
            print(f"[NodeMoE] Best epoch updated: epoch={epoch} {self.monitor_name}={monitor_value:.4f}")
        return True

    def _finalize_best_checkpoint(self) -> None:
        context = self._best_context or self._last_context
        if context is None:
            return
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
        epoch = int(context["epoch"])
        if self.best_epoch is None:
            self.best_epoch = epoch
        self.task.track_gate_weights(True)
        test_metrics = self._evaluate_split(self.test_loader, prefix="test", mask_attr="test_mask")
        test_mean_gate = self.task.mean_gate_weights()
        self.task.track_gate_weights(False)
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
            # Analysis only: learned filter values and mean test gate weights.
            extra={
                "nodemoe_filter_temps": [
                    expert.prop.temp.detach().cpu().clone() for expert in self.model.experts
                ],
                "nodemoe_test_mean_gate": test_mean_gate,
            },
        )
        self.best_metrics = metrics
        self._save_training_log()
        print(f"[NodeMoE] Saved best-epoch checkpoint (epoch={epoch}) after final evaluation.")

    # ------------------------------------------------------------------ #
    # Train / eval loops
    # ------------------------------------------------------------------ #
    def train_epoch(self) -> tuple[float, dict]:
        return run_step_epoch(
            model=self.model,
            task=self.task,
            loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            grad_clip=float(getattr(self.nodemoe_cfg, "grad_clip", 0.0) or 0.0),
        )

    def _evaluate_split(self, loader, prefix: str, mask_attr: str) -> dict[str, float]:
        return runner_evaluate_split(
            model=self.model,
            task=self.task,
            loader=loader,
            device=self.device,
            prefix=prefix,
            mask_attr=mask_attr,
            task_type=resolve_task_type(getattr(self.nodemoe_cfg.dataset, "task_type", None)),
        )

    def fit(self) -> None:
        if getattr(self, "_skip_due_to_existing_checkpoint", False):
            return

        self._setup()

        if os.path.isfile(self._checkpoint_path()):
            print(f"[NodeMoE] Overwriting existing checkpoint: {self._checkpoint_path()}")

        patience = int(getattr(self.nodemoe_cfg, "early_stopping", 0) or 0)
        epochs_since_improvement = 0
        # Few-shot monitors need only train metrics; the test split is
        # evaluated once, on the best epoch's weights.
        monitor_on_train = monitor_uses_train_split(self.monitor_name)
        self._best_context = None
        self._last_context = None
        self._best_model_state = None

        for epoch in range(1, int(self.nodemoe_cfg.epochs) + 1):
            start = time.time()
            train_loss, train_logs = self.train_epoch()
            if monitor_on_train:
                val_metrics = {}
            else:
                val_metrics = self._evaluate_split(self.val_loader, prefix="val", mask_attr="val_mask")

            duration = time.time() - start
            log_parts = [
                f"[NodeMoE][Epoch {epoch}/{self.nodemoe_cfg.epochs}]",
                f"train_loss={train_loss:.4f}",
            ]
            for metrics in (train_logs, val_metrics):
                for k, v in metrics.items():
                    if not should_print_metric(k):
                        continue
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
                if improved:
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(
                        f"[NodeMoE] Early stopping at epoch {epoch} "
                        f"(no improvement in {patience} epochs)."
                    )
                    break

        self._finalize_best_checkpoint()

        if self.monitor_name is not None:
            print(
                f"[NodeMoE] Complete. Best {self.monitor_name}: "
                f"{self.best_metric:.4f} at epoch {self.best_epoch}."
            )
        else:
            final_epoch = self.train_history[-1]["epoch"] if self.train_history else 0
            print(f"[NodeMoE] Complete. Final epoch: {final_epoch} (early stopping disabled).")
        metric_value = self.best_metrics.get("test_acc")
        if metric_value is not None:
            print(f"[NodeMoE] Best-epoch test_acc={metric_value:.4f}")


__all__ = ["NodeMoERunner"]
