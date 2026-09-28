"""GraphMETRO training runner.

Mirrors ``src.moe.gmoe.trainer.GMoERunner`` (dataset loading, split handling,
monitoring, best-state selection, one final test evaluation, checkpoint and
log) with GraphMETRO's model / task and the official three Adam groups. The
final evaluation also scores the queries with the shared Table 15 Brier risk
(``src.moe.shift_eval``) and saves the per-query predictions.

Under a shift split root (a path inside ``data_preparation.shift.root``) the
split file is verified before any data is loaded, so a missing or regenerated
file fails loudly instead of being silently replaced by a standard split.
"""

from __future__ import annotations

import os
import time

import torch
from torch import optim

from src.data_loader import create_dataset, dataset_info, log_split_instance_counts
from src.moe.identity import behavior_fingerprint
from src.moe.routergfm.common import REGRESSION, infer_task_family
from src.moe.shift_eval import (
    brier_risk,
    collect_query_outputs,
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
from src.utils.supervised_loss import build_supervised_head
from src.utils.training import run_step_epoch

from .model import GraphMETROModel
from .task import GraphMETROTask


def _is_within(path: str, root: str) -> bool:
    path, root = os.path.abspath(str(path)), os.path.abspath(str(root))
    return os.path.commonpath([path, root]) == root


class GraphMETRORunner:
    """Train GraphMETRO from scratch on one target application using ``cfg.moe.graphmetro``."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.gm_cfg = cfg.moe.graphmetro
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        set_seed(seed=self.cfg.seed)

        # Summary fields defined up front so skip paths are safe.
        self.best_metrics: dict[str, float] = {}
        self.best_epoch = None
        self.best_metric = float("nan")
        self.monitor_name = "val_acc"
        self.monitor_mode = "max"
        self.train_history: list[dict] = []

        ds_cfg = self.gm_cfg.dataset
        self.task_level_raw = str(ds_cfg.task_level).lower()
        induced = bool(getattr(ds_cfg, "induced", False))
        if resolve_effective_task_level(self.task_level_raw, induced) != "graph":
            raise ValueError(
                "[GraphMETRO] Node / edge tasks must be induced: the shift transforms act on "
                "per-instance subgraphs. Set moe.graphmetro.dataset.induced=True."
            )
        self.split = self._resolve_split()
        self.run_name = self._build_run_name()
        self.run_group = f"{ds_cfg.name}-{self.task_level_raw}"
        self.run_dir = os.path.join(self.gm_cfg.checkpoint_dir, self.run_group)
        self._skip_due_to_existing_checkpoint = False
        self._is_setup = False

        ckpt_path = self._existing_checkpoint_path()
        if self.gm_cfg.skip_if_exists and ckpt_path is not None:
            self._skip_due_to_existing_checkpoint = True
            print(f"[GraphMETRO] Checkpoint already exists, skipping: {ckpt_path}")
            return

    # ------------------------------------------------------------------ #
    # Lazy heavy setup
    # ------------------------------------------------------------------ #
    def _verify_shift_split(self) -> None:
        """Assert the run's split file is an intact shift split when reading a shift root."""
        split_root = shared_split_root(self.cfg)
        if not _is_within(split_root, self.cfg.data_preparation.shift.root):
            return
        from src.data_loader.shift_splits import verify_shift_root

        ds_cfg = self.gm_cfg.dataset
        verify_shift_root(split_root, [(ds_cfg.name, self.task_level_raw, int(self.cfg.seed), self.split)])

    def _setup(self) -> None:
        if self._is_setup:
            return
        self._is_setup = True

        cfg = self.cfg
        gm_cfg = self.gm_cfg
        ds_cfg = gm_cfg.dataset
        raw_task_level = self.task_level_raw
        induced = bool(getattr(ds_cfg, "induced", False))

        self._verify_shift_split()
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
        # Fills gm_cfg.in_dim and ds_cfg.{num_classes,label_dim,task_type}.
        populate_dataset_cfg_from_meta(gm_cfg, ds_cfg, self.dataset_meta)

        self.task = GraphMETROTask(cfg).to(self.device)
        head = build_supervised_head(
            in_dim=int(gm_cfg.hidden_dim),
            task_type=resolve_task_type(getattr(ds_cfg, "task_type", None)),
            task_level=self.effective_task_level,
            label_dim=int(getattr(ds_cfg, "label_dim", 1) or 1),
            num_classes=int(getattr(ds_cfg, "num_classes", 1) or 1),
        )
        self.model = GraphMETROModel(
            in_dim=int(gm_cfg.in_dim),
            backbone=str(gm_cfg.backbone),
            num_layers=int(gm_cfg.num_layers),
            hidden_dim=int(gm_cfg.hidden_dim),
            dropout=float(gm_cfg.dropout),
            use_batchnorm=bool(gm_cfg.use_batchnorm),
            num_experts=len(self.task.expert_names),
            task_level_raw=raw_task_level,
            graph_pooling=str(gm_cfg.graph_pooling),
            head=head,
        ).to(self.device)
        print(f"[GraphMETRO] Experts: {self.task.expert_names}; training shifts: {len(self.task.shifts)}")

        self.optimizer = optim.Adam(
            self.model.param_groups(moe_lr=float(gm_cfg.moe_lr), classifier_lr=float(gm_cfg.classifier_lr)),
            weight_decay=float(gm_cfg.weight_decay),
        )

        self._init_monitoring()
        ensure_dir(self.run_dir)
        self.train_loader, self.val_loader, self.test_loader = make_workflow_loaders(
            dataset=self.dataset,
            dataset_name=ds_cfg.name,
            task_level_raw=raw_task_level,
            effective_task_level=self.effective_task_level,
            batch_size=int(gm_cfg.batch_size),
            num_workers=int(gm_cfg.num_workers),
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
            prefix="[GraphMETRO][Split]",
        )
        task_type = resolve_task_type(getattr(ds_cfg, "task_type", None))
        family = infer_task_family(raw_task_level, task_type, int(getattr(ds_cfg, "label_dim", 1) or 1))
        self.task.normalizer = support_normalizer(self.train_loader.dataset, family)

    # ------------------------------------------------------------------ #
    # Monitoring
    # ------------------------------------------------------------------ #
    def _init_monitoring(self) -> None:
        ds_cfg = self.gm_cfg.dataset
        label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        spec = resolve_explicit_monitor_spec(
            raw_monitor_metric=getattr(self.gm_cfg, "monitor_metric", "auto"),
            setting_name="moe.graphmetro.monitor_metric",
        )
        if spec is None:
            spec = resolve_auto_monitor_spec(
                task_type=resolve_task_type(getattr(ds_cfg, "task_type", None)),
                task_level=self.task_level_raw,
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
        split = resolve_workflow_split(getattr(self.gm_cfg.dataset, "fixed_split", None), default=(5, 0.0, 1.0))
        if is_few_shot_split(split):
            if abs(float(split[1])) > 1e-6 or abs(float(split[2]) - 1.0) > 1e-6:
                raise ValueError("Few-shot split must be (shots, 0.0, 1.0).")
        return split

    def _build_run_name(self) -> str:
        gm_cfg = self.gm_cfg
        ds_cfg = gm_cfg.dataset
        behavior_cfg = gm_cfg.clone()
        behavior_cfg.pop("prediction_dir", None)  # output location only
        fingerprint = behavior_fingerprint(
            behavior_cfg,
            external_behavior={
                "shared_split_root": shared_split_root(self.cfg),
                "shared_induced_root": shared_induced_root(self.cfg, getattr(ds_cfg, "induced_root", "")),
            },
        )
        parts = [
            "graphmetro",
            ds_cfg.name,
            f"induced{int(getattr(ds_cfg, 'induced', False))}",
            format_split_for_name(self.split),
            f"task{self.task_level_raw}",
            str(gm_cfg.backbone),
            f"lam{float(gm_cfg.align_lambda):g}",
            f"h{gm_cfg.hidden_dim}",
            f"l{gm_cfg.num_layers}",
            f"e{gm_cfg.epochs}",
            f"lr{float(gm_cfg.moe_lr):g}",
            f"bs{gm_cfg.batch_size}",
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

    def prediction_path(self) -> str:
        return os.path.join(self.gm_cfg.prediction_dir, self.run_group, f"{self.run_name}.pt")

    def _log_path(self) -> str:
        log_dir = getattr(self.gm_cfg, "log_dir", "")
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
        """Track the best epoch in memory; ``_finalize_best_checkpoint`` writes it once."""
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
            print(f"[GraphMETRO] Best state updated at epoch={epoch} (monitor disabled).")
        else:
            print(f"[GraphMETRO] Best epoch updated: epoch={epoch} {self.monitor_name}={monitor_value:.4f}")
        return True

    def _query_brier(self) -> float:
        """Brier risk of the test queries (shared Table 15 scorer); saves the per-query predictions."""
        ds_cfg = self.gm_cfg.dataset
        task_type = resolve_task_type(getattr(ds_cfg, "task_type", None))
        label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        family = infer_task_family(self.task_level_raw, task_type, label_dim)
        self.model.eval()
        outputs = collect_query_outputs(
            lambda batch: raw_outputs(self.task.normalizer, self.model(batch)[0]),
            self.test_loader, self.device, task_type, label_dim,
        )
        support_targets = None
        if family == REGRESSION:
            support_targets = torch.stack(
                [torch.as_tensor(graph.y).reshape(-1).float() for graph in self.train_loader.dataset]
            )
        brier = brier_risk(
            outputs,
            task_family=family,
            support_targets=support_targets,
            reg_kind=str(self.cfg.moe.routergfm.loss.regression),
        )
        save_query_predictions(
            self.prediction_path(),
            outputs,
            {
                "method": "graphmetro",
                "dataset": str(ds_cfg.name),
                "task_level": self.task_level_raw,
                "task_family": family,
                "split": list(self.split),
                "split_root": shared_split_root(self.cfg),
                "seed": int(self.cfg.seed),
                "run_name": self.run_name,
                "expert_names": list(self.task.expert_names),
                "test_brier": brier,
            },
        )
        return brier

    def _finalize_best_checkpoint(self) -> None:
        context = self._best_context or self._last_context
        if context is None:
            return
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
        epoch = int(context["epoch"])
        if self.best_epoch is None:
            self.best_epoch = epoch
        test_metrics = self._evaluate_split(self.test_loader, prefix="test", mask_attr="test_mask")
        test_metrics["test_brier"] = self._query_brier()
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
            extra={
                "graphmetro_expert_names": list(self.task.expert_names),
                "graphmetro_prediction_path": self.prediction_path(),
            },
        )
        self.best_metrics = metrics
        self._save_training_log()
        print(f"[GraphMETRO] Saved best-epoch checkpoint (epoch={epoch}) after final evaluation.")

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
            grad_clip=float(getattr(self.gm_cfg, "grad_clip", 0.0) or 0.0),
        )

    def _evaluate_split(self, loader, prefix: str, mask_attr: str) -> dict[str, float]:
        return runner_evaluate_split(
            model=self.model,
            task=self.task,
            loader=loader,
            device=self.device,
            prefix=prefix,
            mask_attr=mask_attr,
            task_type=resolve_task_type(getattr(self.gm_cfg.dataset, "task_type", None)),
        )

    def fit(self) -> None:
        if getattr(self, "_skip_due_to_existing_checkpoint", False):
            return

        self._setup()

        if os.path.isfile(self._checkpoint_path()):
            print(f"[GraphMETRO] Overwriting existing checkpoint: {self._checkpoint_path()}")

        patience = int(getattr(self.gm_cfg, "early_stopping", 0) or 0)
        epochs_since_improvement = 0
        # Few-shot splits monitor the train loss, so val (the unused region under
        # shift roots) is never evaluated; test is evaluated once at the end.
        monitor_on_train = monitor_uses_train_split(self.monitor_name)
        self._best_context = None
        self._last_context = None
        self._best_model_state = None

        for epoch in range(1, int(self.gm_cfg.epochs) + 1):
            start = time.time()
            train_loss, train_logs = self.train_epoch()
            if monitor_on_train:
                val_metrics = {}
            else:
                val_metrics = self._evaluate_split(self.val_loader, prefix="val", mask_attr="val_mask")

            duration = time.time() - start
            log_parts = [
                f"[GraphMETRO][Epoch {epoch}/{self.gm_cfg.epochs}]",
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
                        f"[GraphMETRO] Early stopping at epoch {epoch} "
                        f"(no improvement in {patience} epochs)."
                    )
                    break

        self._finalize_best_checkpoint()

        if self.monitor_name is not None:
            print(
                f"[GraphMETRO] Complete. Best {self.monitor_name}: "
                f"{self.best_metric:.4f} at epoch {self.best_epoch}."
            )
        else:
            final_epoch = self.train_history[-1]["epoch"] if self.train_history else 0
            print(f"[GraphMETRO] Complete. Final epoch: {final_epoch} (early stopping disabled).")
        for metric_name in ("test_acc", "test_micro_f1", "test_macro_f1", "test_auc", "test_mae", "test_mse", "test_brier"):
            metric_value = self.best_metrics.get(metric_name)
            if metric_value is not None:
                print(f"[GraphMETRO] Best-epoch {metric_name}={metric_value:.4f}")


__all__ = ["GraphMETRORunner"]
