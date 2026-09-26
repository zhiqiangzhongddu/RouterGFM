from __future__ import annotations

import os
import time

import torch
from torch import optim

from src.data_loader import create_dataset, dataset_info, log_split_instance_counts
from src.model import build_encoder_from_cfg
from src.utils.checkpoint import save_checkpoint, save_training_log
from src.train.monitoring import resolve_train_monitor_spec
from src.train.registry import build_train_task, get_train_task_class
from src.utils.dataset_helpers import (
    is_few_shot_split,
    make_workflow_loaders,
    populate_dataset_cfg_from_meta,
    resolve_effective_task_level,
    shared_induced_root,
    shared_split_root,
)
from src.utils.monitoring import is_metric_improved, merge_epoch_metrics, resolve_monitor_value, should_print_metric
from src.utils.naming import build_train_run_name_from_cfg, format_split_for_name
from src.utils.paths import ensure_dir
from src.utils.parsing import resolve_task_type, resolve_workflow_split
from src.utils.random import set_seed
from src.utils.supervised_eval import runner_evaluate_split
from src.utils.training import build_lr_scheduler, run_step_epoch


class TrainRunner:
    """Train a model from scratch using cfg.train settings."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        set_seed(seed=self.cfg.seed)
        # Define summary fields up front so skip paths are safe for run-level aggregation.
        self.best_metrics: dict[str, float] = {}
        self.best_epoch = None
        self.best_metric = float("nan")
        self.monitor_name = "val_acc"
        self.monitor_mode = "max"
        self.train_history = []
        self._checkpoint_written_this_run = False

        # Resolve task class once (class-level capability flags are
        # consulted before task instantiation in _setup).
        method = getattr(cfg.train, "method", "supervised") or "supervised"
        self.task_cls = get_train_task_class(method)
        if self.task_cls is None:
            raise ValueError(f"Unknown train method: {method}")

        # Validate method-specific cfg up front, before the skip-if-exists
        # check below.  build_train_task calls this again inside _setup
        # as a second line of defense, but running it here guarantees an
        # invalid cfg cannot silently be "skipped" just because an old
        # checkpoint with the same filename happens to exist.
        self.task_cls.validate_cfg(cfg)

        ds_cfg = cfg.train.dataset
        raw_task_level = ds_cfg.task_level
        self.task_level_raw = raw_task_level
        self.split = self._resolve_split()
        self.run_name = self._build_run_name()
        self.run_group = self._build_run_group()
        self.run_dir = os.path.join(self.cfg.train.checkpoint_dir, self.run_group)
        self._skip_due_to_existing_checkpoint = False
        self._is_setup = False

        ckpt_path = self._existing_checkpoint_path()
        if self.cfg.train.skip_if_exists and ckpt_path is not None:
            self._skip_due_to_existing_checkpoint = True
            print(f"[Train] Checkpoint already exists, skipping: {ckpt_path}")
            return

    # ------------------------------------------------------------------ #
    # Lazy heavy setup
    # ------------------------------------------------------------------ #
    def _setup(self) -> None:
        """Load dataset, build model/optimizer/loaders. Called once by fit()."""
        if self._is_setup:
            return
        self._is_setup = True

        cfg = self.cfg
        ds_cfg = cfg.train.dataset
        raw_task_level = self.task_level_raw
        induced = getattr(ds_cfg, "induced", False)

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
        effective_task_level = resolve_effective_task_level(raw_task_level, induced)
        self.effective_task_level = effective_task_level
        # Preserve the raw requested level in cfg for provenance (run name,
        # checkpoint identity). The effective level is stored on self and
        # passed explicitly — matching pretrain's pattern.

        self.dataset_meta = dataset_info(
            dataset=self.dataset,
            task_level=raw_task_level,
            name=ds_cfg.name,
            induced=induced,
        )
        populate_dataset_cfg_from_meta(cfg.model, ds_cfg, self.dataset_meta)

        self.model = build_encoder_from_cfg(
            cfg=cfg,
            in_dim=cfg.model.in_dim,
        ).to(self.device)
        model_name = getattr(cfg.model, "name", "model")
        print(f"[Train] Encoder architecture ({model_name}):\n{self.model}")
        method = getattr(cfg.train, "method", "supervised") or "supervised"
        self.task = build_train_task(method, cfg).to(self.device)
        print(f"[Train] Task head:\n{self.task}")
        params = list(self.model.parameters()) + list(self.task.parameters_to_optimize())
        self.optimizer = optim.Adam(
            params=params,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

        # Create LR scheduler
        self.scheduler = self._build_scheduler()

        self._init_monitoring()
        ensure_dir(self.run_dir)
        self.train_loader, self.val_loader, self.test_loader = self._make_loaders(induced=induced)
        log_split_instance_counts(
            self.train_loader,
            self.val_loader,
            self.test_loader,
            task_level=self.task_level_raw,
            split=self.split,
            induced=induced,
            prefix="[Train][Split]",
        )

    # ------------------------------------------------------------------ #
    # LR scheduler
    # ------------------------------------------------------------------ #
    def _build_scheduler(self):
        """Build an LR scheduler based on config. Returns None when disabled."""
        return build_lr_scheduler(
            optimizer=self.optimizer,
            scheduler_name=getattr(self.cfg.train, "scheduler", "none"),
            epochs=self.cfg.train.epochs,
            step_size=int(getattr(self.cfg.train, "scheduler_step_size", 50)),
            gamma=float(getattr(self.cfg.train, "scheduler_gamma", 0.5)),
        )

    def _init_monitoring(self) -> None:
        label_dim = int(getattr(self.cfg.train.dataset, "label_dim", 1) or 1)
        split = getattr(self, "split", None)
        spec = resolve_train_monitor_spec(
            self.cfg,
            task_level=str(getattr(self, "task_level_raw", self.cfg.train.dataset.task_level) or "").lower(),
            label_dim=label_dim,
            few_shot_without_validation=is_few_shot_split(split),
            default_monitor=getattr(self.task_cls, "default_monitor", None),
        )
        self.monitor_name = spec.name
        self.monitor_mode = spec.mode
        self.best_metric = spec.best_metric
        self.best_epoch = None
        self.best_metrics: dict[str, float] = {}

    def _make_loaders(self, induced: bool):
        return make_workflow_loaders(
            dataset=self.dataset,
            dataset_name=self.cfg.train.dataset.name,
            task_level_raw=self.task_level_raw,
            effective_task_level=self.effective_task_level,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            split=self.split,
            seed=self.cfg.seed,
            induced=induced,
            split_root=shared_split_root(self.cfg),
        )

    def _resolve_split(self) -> tuple:
        ds_cfg = self.cfg.train.dataset
        raw_split = getattr(ds_cfg, "fixed_split", None)
        if raw_split is None:
            raw_split = getattr(self.cfg.train, "fixed_split", None)
        split = resolve_workflow_split(raw_split, default=(0.8, 0.1, 0.1))
        # Train-specific: few-shot splits must be (shots, 0.0, 1.0).
        if is_few_shot_split(split):
            val_ratio = float(split[1])
            test_ratio = float(split[2])
            if abs(val_ratio) > 1e-6 or abs(test_ratio - 1.0) > 1e-6:
                raise ValueError("Few-shot split must be (shots, 0.0, 1.0).")
        return split

    def _build_run_name(self) -> str:
        dataset_cfg = self.cfg.train.dataset
        raw_task_level = getattr(self, "task_level_raw", getattr(dataset_cfg, "task_level", ""))
        return build_train_run_name_from_cfg(
            self.cfg,
            split=self.split,
            task_level_raw=raw_task_level,
            task_cls=getattr(self, "task_cls", None),
        )

    def _build_run_group(self) -> str:
        dataset_cfg = self.cfg.train.dataset
        raw_task_level = getattr(self, "task_level_raw", getattr(dataset_cfg, "task_level", ""))
        return f"{dataset_cfg.name}-{raw_task_level}"

    def _checkpoint_path(self) -> str:
        return os.path.join(self.run_dir, f"{self.run_name}.pt")

    def _existing_checkpoint_path(self) -> str | None:
        ckpt_path = self._checkpoint_path()
        if os.path.isfile(ckpt_path):
            return ckpt_path
        return None

    def get_checkpoint_path_for_metrics(self) -> str:
        return self._checkpoint_path()

    def _log_path(self) -> str:
        log_dir = getattr(self.cfg.train, "log_dir", "")
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
                "metrics": self.best_metrics,
            },
        )

    def _is_improved(self, metric: float) -> bool:
        return is_metric_improved(metric, self.best_metric, self.monitor_mode)

    def _save_best_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        train_logs: dict[str, float],
        val_metrics: dict[str, float],
        test_metrics: dict[str, float],
        monitor_value: float,
    ) -> bool:
        """Persist the best-performing checkpoint. Returns True if improved."""
        if self.monitor_name is None:
            self.best_metric = float(monitor_value)
            self.best_epoch = epoch
        else:
            if not self._is_improved(monitor_value):
                return False
            self.best_metric = float(monitor_value)
            self.best_epoch = epoch
        metrics = {
            "train_loss": train_loss,
            "best_epoch": epoch,
            **train_logs,
            **val_metrics,
            **test_metrics,
        }
        if self.monitor_name is not None:
            metrics[self.monitor_name] = float(monitor_value)
        save_checkpoint(
            path=self._checkpoint_path(),
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            cfg=self.cfg,
            dataset_meta=self.dataset_meta,
            metrics=metrics,
            extra={"train_task_state": self.task.state_dict()},
        )
        self._checkpoint_written_this_run = True
        self.best_metrics = metrics
        self._save_training_log()
        if self.monitor_name is None:
            print(f"[Train] Checkpoint updated at epoch={epoch} (monitor disabled).")
        else:
            print(f"[Train] Best epoch updated: epoch={epoch} {self.monitor_name}={monitor_value:.4f}")
        return True

    def train_epoch(self) -> tuple[float, dict]:
        return run_step_epoch(
            model=self.model,
            task=self.task,
            loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            grad_clip=float(getattr(self.cfg.train, "grad_clip", 0.0) or 0.0),
        )

    def _evaluate_split(self, loader, prefix: str, mask_attr: str) -> dict[str, float]:
        return runner_evaluate_split(
            model=self.model,
            task=self.task,
            loader=loader,
            device=self.device,
            prefix=prefix,
            mask_attr=mask_attr,
            task_type=resolve_task_type(getattr(self.cfg.train.dataset, "task_type", None)),
        )

    def _evaluate_selected_test_checkpoint(self) -> dict[str, float]:
        """Restore the selected checkpoint and evaluate the held-out test once."""
        if self.test_loader is None or not self._checkpoint_written_this_run:
            return {}

        payload = torch.load(self._checkpoint_path(), map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
        task_state = (payload.get("extra") or {}).get("train_task_state")
        if task_state is not None:
            self.task.load_state_dict(task_state)
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)

        test_metrics = self._evaluate_split(
            self.test_loader, prefix="test", mask_attr="test_mask"
        )
        metrics = dict(payload.get("metrics") or {})
        metrics.update(test_metrics)
        save_checkpoint(
            path=self._checkpoint_path(),
            model=self.model,
            optimizer=self.optimizer,
            epoch=int(payload.get("epoch", self.best_epoch or 0)),
            cfg=payload.get("cfg", self.cfg),
            dataset_meta=payload.get("dataset", self.dataset_meta),
            metrics=metrics,
            extra=payload.get("extra"),
        )
        self.best_metrics = metrics
        return test_metrics

    def fit(self) -> None:
        if getattr(self, "_skip_due_to_existing_checkpoint", False):
            return

        self._setup()

        if os.path.isfile(self._checkpoint_path()):
            print(f"[Train] Overwriting existing checkpoint: {self._checkpoint_path()}")

        patience = int(getattr(self.cfg.train, "early_stopping", 0) or 0)
        epochs_since_improvement = 0

        for epoch in range(1, self.cfg.train.epochs + 1):
            start = time.time()
            train_loss, train_logs = self.train_epoch()
            val_metrics = self._evaluate_split(self.val_loader, prefix="val", mask_attr="val_mask")

            duration = time.time() - start
            log_parts = [
                f"[Train][Epoch {epoch}/{self.cfg.train.epochs}]",
                f"train_loss={train_loss:.4f}",
            ]
            for metrics in (train_logs, val_metrics):
                for k, v in metrics.items():
                    if not should_print_metric(k):
                        continue
                    log_parts.append(f"{k}={v:.4f}")

            # Step LR scheduler after each epoch.
            if self.scheduler is not None:
                log_parts.append(f"lr={self.optimizer.param_groups[0]['lr']:.2e}")
                self.scheduler.step()

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
            improved = self._save_best_checkpoint(
                epoch=epoch,
                train_loss=train_loss,
                train_logs=train_logs,
                val_metrics=val_metrics,
                test_metrics={},
                monitor_value=monitor_value,
            )

            if patience > 0:
                if improved:
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(
                        f"[Train] Early stopping at epoch {epoch} "
                        f"(no improvement in {patience} epochs)."
                    )
                    break

        if self.train_history and not self._checkpoint_written_this_run:
            # The monitor never produced a usable value (e.g. empty val split
            # → NaN every epoch, no improvement ever recorded). Persist the
            # final state so the run leaves a checkpoint and a log instead of
            # exiting "successfully" with nothing (PretrainRunner has the same
            # fallback).
            last = self.train_history[-1]
            self.best_epoch = int(last.get("epoch", 0))
            fallback_metrics = {
                "train_loss": float(last.get("loss", float("nan"))),
                "best_epoch": self.best_epoch,
                **(last.get("metrics") or {}),
            }
            print(
                "[Train] WARNING: monitor "
                f"{self.monitor_name!r} never resolved to a finite value; "
                "saving final-epoch state as fallback checkpoint."
            )
            save_checkpoint(
                path=self._checkpoint_path(),
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.best_epoch,
                cfg=self.cfg,
                dataset_meta=self.dataset_meta,
                metrics=fallback_metrics,
                extra={"train_task_state": self.task.state_dict(), "fallback_save": True},
            )
            self._checkpoint_written_this_run = True
            self.best_metrics = fallback_metrics
            self.best_metric = float("nan")
            self._save_training_log()

        final_test_metrics = self._evaluate_selected_test_checkpoint()
        if final_test_metrics:
            print(
                "[Train] Selected-checkpoint test: "
                + " ".join(f"{key}={value:.4f}" for key, value in final_test_metrics.items())
            )
        if self.train_history:
            self._save_training_log()

        if self.monitor_name is not None:
            print(
                f"[Train] Complete. Best {self.monitor_name}: "
                f"{self.best_metric:.4f} at epoch {self.best_epoch}."
            )
        else:
            final_epoch = self.train_history[-1]["epoch"] if self.train_history else 0
            print(f"[Train] Complete. Final epoch: {final_epoch} (early stopping disabled).")
        for metric_name in ("test_acc", "test_micro_f1", "test_macro_f1", "test_auc", "test_mae", "test_mse"):
            metric_value = self.best_metrics.get(metric_name)
            if metric_value is not None:
                print(f"[Train] Best-epoch {metric_name}={metric_value:.4f}")
