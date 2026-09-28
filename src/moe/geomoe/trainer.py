"""GeoMoE training runner.

Mirrors :class:`src.moe.graphmore.trainer.GraphMoRERunner` (dataset loading,
split handling, monitoring, best-state tracking, single final test
evaluation, checkpoint and log writing) with GeoMoE specifics:

* one Adam optimizer (fixed curvatures leave no manifold parameters);
* node ORC is attached to copies of the **support** (train) instances only,
  the sole consumers of curvature (``L_align`` / ``L_contr``);
* the final evaluation adds the Table 15 query Brier risk
  (``src.moe.shift_eval``) and saves the query predictions;
* a split root under ``data_preparation.shift.root`` must hold an intact shift
  split, otherwise the loader would silently write a standard split there.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from torch import optim
from torch_geometric.loader import DataLoader

from src.data_loader import create_dataset, dataset_info, log_split_instance_counts
from src.data_loader.shift_splits import verify_shift_root
from src.moe.identity import behavior_fingerprint
from src.moe.routergfm.common import REGRESSION, infer_task_family
from src.moe.shift_eval import brier_risk, collect_query_outputs, save_query_predictions, support_normalizer
from src.utils.checkpoint import cfg_to_dict, save_checkpoint, save_training_log
from src.utils.dataset_helpers import (
    is_few_shot_split,
    make_workflow_loaders,
    populate_dataset_cfg_from_meta,
    resolve_effective_task_level,
    resolve_split_task_level,
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

from .curvature import attach_node_orc
from .model import GeoMoEModel
from .task import GeoMoETask

# Output locations excluded from the run identity (on top of src.moe.identity._OPERATIONAL_KEYS).
_NON_BEHAVIOR_KEYS = ("num_workers", "orc_cache_dir", "prediction_dir")


class GeoMoERunner:
    """Train GeoMoE from scratch on the target support using ``cfg.moe.geomoe`` settings."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.geo_cfg = cfg.moe.geomoe
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        set_seed(seed=self.cfg.seed)

        # Summary fields defined up front so skip paths are safe.
        self.best_metrics: dict[str, float] = {}
        self.best_epoch = None
        self.best_metric = float("nan")
        self.monitor_name = "val_acc"
        self.monitor_mode = "max"
        self.train_history: list[dict] = []

        ds_cfg = self.geo_cfg.dataset
        self.task_level_raw = str(ds_cfg.task_level).lower()
        if self.task_level_raw in {"node", "edge"} and not bool(ds_cfg.induced):
            raise ValueError("[GeoMoE] Node/edge tasks need induced subgraph instances (moe.geomoe.dataset.induced=True).")
        self.split = self._resolve_split()
        self.run_name = self._build_run_name()
        self.run_group = f"{ds_cfg.name}-{self.task_level_raw}"
        self.run_dir = os.path.join(self.geo_cfg.checkpoint_dir, self.run_group)
        self._skip_due_to_existing_checkpoint = False
        self._is_setup = False

        ckpt_path = self._existing_checkpoint_path()
        if self.geo_cfg.skip_if_exists and ckpt_path is not None:
            self._skip_due_to_existing_checkpoint = True
            print(f"[GeoMoE] Checkpoint already exists, skipping: {ckpt_path}")
            return

    # ------------------------------------------------------------------ #
    # Lazy heavy setup
    # ------------------------------------------------------------------ #
    def _setup(self) -> None:
        if self._is_setup:
            return
        self._is_setup = True

        cfg = self.cfg
        g = self.geo_cfg
        ds_cfg = g.dataset
        raw_task_level = self.task_level_raw
        induced = bool(getattr(ds_cfg, "induced", False))
        self.effective_task_level = resolve_effective_task_level(raw_task_level, induced)
        self._require_intact_shift_split(resolve_split_task_level(raw_task_level, self.effective_task_level, induced))

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
        self.dataset_meta = dataset_info(
            dataset=self.dataset,
            task_level=raw_task_level,
            name=ds_cfg.name,
            induced=induced,
        )
        # Fills g.in_dim and ds_cfg.{num_classes,label_dim,task_type}.
        populate_dataset_cfg_from_meta(g, ds_cfg, self.dataset_meta)

        self.model = GeoMoEModel(
            in_dim=int(g.in_dim),
            hidden_dim=int(g.hidden_dim),
            num_layers=int(g.num_layers),
            dropout=float(g.dropout),
            curvatures=[float(k) for k in g.curvatures],
            gate_temperature=float(g.gate_temperature),
        ).to(self.device)
        print(f"[GeoMoE] Model architecture:\n{self.model}")
        self.task = GeoMoETask(cfg).to(self.device)
        print(f"[GeoMoE] Task head:\n{self.task}")

        params = [p for p in list(self.model.parameters()) + list(self.task.parameters_to_optimize()) if p.requires_grad]
        self.optimizer = optim.Adam(params=params, lr=float(g.lr), weight_decay=float(g.weight_decay))

        self._init_monitoring()
        ensure_dir(self.run_dir)
        self.train_loader, self.val_loader, self.test_loader = make_workflow_loaders(
            dataset=self.dataset,
            dataset_name=ds_cfg.name,
            task_level_raw=raw_task_level,
            effective_task_level=self.effective_task_level,
            batch_size=int(g.batch_size),
            num_workers=int(g.num_workers),
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
            prefix="[GeoMoE][Split]",
        )
        self._attach_support_orc()
        task_type = resolve_task_type(getattr(ds_cfg, "task_type", None))
        family = infer_task_family(raw_task_level, task_type, int(getattr(ds_cfg, "label_dim", 1) or 1))
        self.task.normalizer = support_normalizer(self.train_loader.dataset, family)

    def _require_intact_shift_split(self, split_task_level: str) -> None:
        """Verify the shift split file before the loader could replace a missing/invalid one."""
        split_root = Path(shared_split_root(self.cfg))
        if split_root.resolve().parent != Path(str(self.cfg.data_preparation.shift.root)).resolve():
            return
        verify_shift_root(
            split_root, [(str(self.geo_cfg.dataset.name), split_task_level, int(self.cfg.seed), self.split)]
        )

    def _attach_support_orc(self) -> None:
        """Rebuild the train loader over copies of the support instances carrying ``node_orc``."""
        support = [self.train_loader.dataset[i].clone() for i in range(len(self.train_loader.dataset))]
        start = time.time()
        attach_node_orc(
            support,
            idleness=float(self.geo_cfg.orc_idleness),
            cache_dir=os.path.join(str(self.geo_cfg.orc_cache_dir), self.run_group) if self.geo_cfg.orc_cache_dir else None,
        )
        print(f"[GeoMoE] Node ORC for {len(support)} support instances in {time.time() - start:.1f}s.")
        self.train_loader = DataLoader(
            support, batch_size=int(self.geo_cfg.batch_size), shuffle=True, num_workers=int(self.geo_cfg.num_workers),
        )

    # ------------------------------------------------------------------ #
    # Monitoring
    # ------------------------------------------------------------------ #
    def _init_monitoring(self) -> None:
        ds_cfg = self.geo_cfg.dataset
        label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        spec = resolve_explicit_monitor_spec(
            raw_monitor_metric=getattr(self.geo_cfg, "monitor_metric", "auto"),
            setting_name="moe.geomoe.monitor_metric",
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
        split = resolve_workflow_split(getattr(self.geo_cfg.dataset, "fixed_split", None), default=(5, 0.0, 1.0))
        if is_few_shot_split(split):
            if abs(float(split[1])) > 1e-6 or abs(float(split[2]) - 1.0) > 1e-6:
                raise ValueError("Few-shot split must be (shots, 0.0, 1.0).")
        return split

    def _build_run_name(self) -> str:
        g = self.geo_cfg
        ds_cfg = g.dataset
        payload = cfg_to_dict(g)
        for key in _NON_BEHAVIOR_KEYS:
            payload.pop(key, None)
        fingerprint = behavior_fingerprint(
            payload,
            external_behavior={
                "shared_split_root": shared_split_root(self.cfg),
                "shared_induced_root": shared_induced_root(self.cfg, getattr(ds_cfg, "induced_root", "")),
            },
        )
        parts = [
            "geomoe",
            ds_cfg.name,
            f"induced{int(getattr(ds_cfg, 'induced', False))}",
            format_split_for_name(self.split),
            f"task{self.task_level_raw}",
            f"h{g.hidden_dim}",
            f"l{g.num_layers}",
            f"K{g.num_negatives}",
            f"e{g.epochs}",
            f"lr{g.lr:g}" if isinstance(g.lr, (int, float)) else f"lr{g.lr}",
            f"bs{g.batch_size}",
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

    def _prediction_path(self) -> str:
        return os.path.join(str(self.geo_cfg.prediction_dir), self.run_group, f"{self.run_name}.pt")

    def _log_path(self) -> str:
        log_dir = getattr(self.geo_cfg, "log_dir", "")
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
        self._best_task_state = self._snapshot_state(self.task)
        if self.monitor_name is None:
            print(f"[GeoMoE] Best state updated at epoch={epoch} (monitor disabled).")
        else:
            print(f"[GeoMoE] Best epoch updated: epoch={epoch} {self.monitor_name}={monitor_value:.4f}")
        return True

    def _query_brier(self) -> float:
        """Table 15 Brier risk of the query (test) predictions; also saves them."""
        ds_cfg = self.geo_cfg.dataset
        task_type = resolve_task_type(getattr(ds_cfg, "task_type", None))
        label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        family = infer_task_family(self.task_level_raw, task_type, label_dim)
        self.model.eval()
        self.task.eval()
        outputs = collect_query_outputs(
            lambda batch: self.task.logits(self.model, batch), self.test_loader, self.device, task_type, label_dim,
        )
        support_targets = None
        if family == REGRESSION:
            support_targets = torch.stack(
                [torch.as_tensor(item.y).reshape(-1).float() for item in self.train_loader.dataset]
            )
        brier = brier_risk(
            outputs,
            task_family=family,
            support_targets=support_targets,
            reg_kind=str(self.cfg.moe.routergfm.loss.regression),
        )
        save_query_predictions(
            self._prediction_path(),
            outputs,
            meta={
                "method": "geomoe",
                "dataset": str(ds_cfg.name),
                "task_level": self.task_level_raw,
                "task_family": family,
                "split": list(self.split),
                "split_root": shared_split_root(self.cfg),
                "seed": int(self.cfg.seed),
                "run_name": self.run_name,
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
            self.task.load_state_dict(self._best_task_state)
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
            extra={"geomoe_task_state": self.task.state_dict(), "prediction_path": self._prediction_path()},
        )
        self.best_metrics = metrics
        self._save_training_log()
        print(f"[GeoMoE] Saved best-epoch checkpoint (epoch={epoch}) after final evaluation.")

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
            grad_clip=float(getattr(self.geo_cfg, "grad_clip", 0.0) or 0.0),
        )

    def _evaluate_split(self, loader, prefix: str, mask_attr: str) -> dict[str, float]:
        return runner_evaluate_split(
            model=self.model,
            task=self.task,
            loader=loader,
            device=self.device,
            prefix=prefix,
            mask_attr=mask_attr,
            task_type=resolve_task_type(getattr(self.geo_cfg.dataset, "task_type", None)),
        )

    def fit(self) -> None:
        if getattr(self, "_skip_due_to_existing_checkpoint", False):
            return

        self._setup()

        if os.path.isfile(self._checkpoint_path()):
            print(f"[GeoMoE] Overwriting existing checkpoint: {self._checkpoint_path()}")

        patience = int(getattr(self.geo_cfg, "early_stopping", 0) or 0)
        epochs_since_improvement = 0
        # Few-shot splits monitor the train loss, so the val loader (on shift
        # roots: the unused region) is never evaluated; test is evaluated once
        # for the best epoch's weights.
        monitor_on_train = monitor_uses_train_split(self.monitor_name)
        self._best_context = None
        self._last_context = None
        self._best_model_state = None
        self._best_task_state = None

        for epoch in range(1, int(self.geo_cfg.epochs) + 1):
            start = time.time()
            train_loss, train_logs = self.train_epoch()
            if monitor_on_train:
                val_metrics = {}
            else:
                val_metrics = self._evaluate_split(self.val_loader, prefix="val", mask_attr="val_mask")

            duration = time.time() - start
            log_parts = [
                f"[GeoMoE][Epoch {epoch}/{self.geo_cfg.epochs}]",
                f"train_loss={train_loss:.4f}",
            ]
            for metrics in (train_logs, val_metrics):
                for k, v in metrics.items():
                    if should_print_metric(k):
                        log_parts.append(f"{k}={v:.4f}")
            log_parts.append(f"time={duration:.1f}s")
            print(" ".join(log_parts))

            self.train_history.append(
                {
                    "epoch": epoch,
                    "loss": float(train_loss),
                    "duration_sec": float(duration),
                    "metrics": merge_epoch_metrics(train_logs, val_metrics, {}),
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
                    print(f"[GeoMoE] Early stopping at epoch {epoch} (no improvement in {patience} epochs).")
                    break

        self._finalize_best_checkpoint()

        if self.monitor_name is not None:
            print(f"[GeoMoE] Complete. Best {self.monitor_name}: {self.best_metric:.4f} at epoch {self.best_epoch}.")
        for metric_name in ("test_acc", "test_auc", "test_mae", "test_brier"):
            metric_value = self.best_metrics.get(metric_name)
            if metric_value is not None:
                print(f"[GeoMoE] Best-epoch {metric_name}={metric_value:.4f}")


__all__ = ["GeoMoERunner"]
