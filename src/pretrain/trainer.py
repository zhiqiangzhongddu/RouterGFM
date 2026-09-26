from __future__ import annotations

import json
import os
import time
from typing import Any

import torch
from torch import optim
from torch_geometric.loader import DataLoader

from src.data_loader import (
    SingleGraphDataLoader,
    create_dataset,
    dataset_info,
    log_split_instance_counts,
)
from src.model import build_encoder_from_cfg
from src.utils.training import build_lr_scheduler, run_step_epoch
from src.utils.checkpoint import cfg_to_dict, save_checkpoint, save_training_log
from src.pretrain.monitoring import resolve_pretrain_monitor_spec
from src.pretrain.registry import build_pretrain_task, get_pretrain_task_class
from src.utils.monitoring import is_metric_improved, merge_epoch_metrics, resolve_monitor_value, should_print_metric
from src.utils.naming import build_pretrain_run_name_from_cfg
from src.utils.paths import ensure_dir
from src.utils.supervised_eval import runner_evaluate_split
from src.utils.dataset_helpers import (
    checkpoint_dataset_dir_name,
    make_workflow_loaders,
    populate_dataset_cfg_from_meta,
    resolve_effective_task_level,
    resolve_loader_task_level,
    shared_induced_root,
    shared_split_root,
)
from src.utils.parsing import resolve_task_type, resolve_workflow_split
from src.utils.random import set_seed


class PretrainRunner:
    """Class to handle pretraining of graph models."""
    def __init__(self, cfg):
        # Initialize settings
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        set_seed(seed=self.cfg.seed)
        # Define summary fields up front so skip paths are safe for run-level aggregation.
        self.best_metrics: dict[str, float] = {}
        self.best_epoch = None
        self.best_metric = float("nan")
        self.monitor_name = "train_loss"
        self.monitor_mode = "min"
        self.train_history = []
        self._checkpoint_written_this_run = False

        # Resolve task class once (class-level capability flags are
        # consulted before task instantiation in _setup).
        self.task_cls = get_pretrain_task_class(cfg.pretrain.method)
        if self.task_cls is None:
            raise ValueError(f"Unknown pretraining method: {cfg.pretrain.method}")

        # Validate method-specific cfg up front, before the skip-if-exists
        # check below. build_pretrain_task calls this again inside _setup
        # as a second line of defense, but running it here guarantees an
        # invalid cfg cannot silently be "skipped" just because an old
        # checkpoint with the same filename happens to exist.
        self.task_cls.validate_cfg(cfg)

        # Resolve naming / path fields needed even when skipping.
        ds_cfg = cfg.pretrain.dataset
        self.task_level_raw = ds_cfg.task_level
        self.effective_task_level = resolve_effective_task_level(
            ds_cfg.task_level,
            getattr(ds_cfg, "induced", False),
        )
        self.split = self._resolve_split()
        self.use_dataset_splits = bool(self.task_cls.uses_dataset_splits)
        self.run_name = self._build_run_name()
        self.run_group = self._build_run_group()
        self.run_dir = os.path.join(self.cfg.pretrain.checkpoint_dir, self.run_group)
        self._skip_due_to_existing_checkpoint = False
        self._is_setup = False

        # Check whether we can skip entirely.
        ckpt_path = self._existing_checkpoint_path()
        if self.cfg.pretrain.skip_if_exists and ckpt_path is not None:
            self._skip_due_to_existing_checkpoint = True
            print(f"[Pretrain] Checkpoint already exists, skipping: {ckpt_path}")
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
        ds_cfg = cfg.pretrain.dataset
        raw_task_level = self.task_level_raw
        induced = getattr(ds_cfg, "induced", False)

        split_root_for_dataset = shared_split_root(cfg)
        split_for_dataset = self.split
        if not self.use_dataset_splits:
            if not (str(raw_task_level).lower() == "edge" and induced):
                if str(raw_task_level).lower() != "graph":
                    split_root_for_dataset = ""
                split_for_dataset = None
        if (
            self.task_cls.requires_graph_batches
            and str(raw_task_level).lower() in {"node", "edge"}
            and not induced
        ):
            # Methods declaring ``requires_graph_batches`` produce degenerate
            # batches when a node/edge dataset is consumed as a single graph:
            # contrastive objectives collapse (only one positive pair per
            # step), and the ContextPred pair sampler yields <2 valid pairs.
            # Induced subgraphs resolve both failure modes.
            raise ValueError(
                f"{cfg.pretrain.method} produces degenerate batches when each "
                "batch contains fewer than 2 graph instances. For node/edge "
                "datasets, set pretrain.dataset.induced=True so the loader "
                "yields induced subgraphs."
            )
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
            cache_induced=getattr(ds_cfg, "cache_induced", True),
            split_root=split_root_for_dataset,
            induced_root=shared_induced_root(cfg, getattr(ds_cfg, "induced_root", "")),
            split=split_for_dataset,
            seed=cfg.seed,
        )
        # Induced subgraphs are handled as graph-level datasets downstream.
        # Store on the runner instead of mutating cfg so the original
        # user-requested task_level is preserved in config / result provenance.
        self.effective_task_level = resolve_effective_task_level(raw_task_level, induced)
        # Get dataset meta info
        self.dataset_meta = dataset_info(
            dataset=self.dataset,
            task_level=raw_task_level,
            name=ds_cfg.name,
            induced=induced,
        )
        # Ensure model input dim matches actual feature dim when not reducing features.
        if not getattr(ds_cfg, "feat_reduction", False):
            cfg.model.in_dim = self.dataset_meta.get("num_node_features")
        populate_dataset_cfg_from_meta(cfg.model, ds_cfg, self.dataset_meta)
        # Build encoder model
        self.model = build_encoder_from_cfg(
            cfg=cfg,
            in_dim=cfg.model.in_dim
        ).to(self.device)
        model_name = getattr(cfg.model, "name", "model")
        print(f"[Pretrain] Encoder architecture ({model_name}):\n{self.model}")
        # Optional warm-start from a prior pretrain checkpoint, matching the
        # official edge_pred -> supervised chaining pattern via --input_model_file.
        self._load_input_checkpoint()
        # Build pretraining task
        self.task = build_pretrain_task(
            name=cfg.pretrain.method,
            cfg=cfg
        ).to(self.device)
        print(f"[Pretrain] Task head ({cfg.pretrain.method}):\n{self.task}")
        # Create optimizer
        task_params = list(self.task.parameters_to_optimize())
        params = list(self.model.parameters()) + task_params
        self.optimizer = optim.Adam(
            params=params,
            lr=cfg.pretrain.lr,
            weight_decay=cfg.pretrain.weight_decay,
        )

        # Create LR scheduler
        self.scheduler = self._build_scheduler()

        # Prepare training and validation loaders and settings
        self._init_monitoring()
        self.train_history = []
        ensure_dir(self.run_dir)
        self.train_loader, self.val_loader, self.test_loader = self._make_loaders(induced=induced)
        if self.use_dataset_splits:
            log_split_instance_counts(
                self.train_loader,
                self.val_loader,
                self.test_loader,
                task_level=self._loader_task_level(induced=induced),
                split=self.split,
                induced=induced,
                prefix="[Pretrain][Split]",
            )
        else:
            print(
                f"[Pretrain] Using full dataset for {cfg.pretrain.method} "
                "(no explicit split loading)."
            )

    # ------------------------------------------------------------------ #
    # Warm-start
    # ------------------------------------------------------------------ #
    def _load_input_checkpoint(self) -> None:
        """Load encoder weights from a prior pretrain checkpoint if requested.

        Accepts either a full pretrain checkpoint payload (saved by
        ``src.utils.checkpoint.save_checkpoint``, keyed under
        ``model_state``) or a raw ``state_dict``. Missing/unexpected keys
        are reported but do not raise, so shape-compatible partial loads
        (e.g. a GCN -> GCN transfer between different heads) succeed.
        """
        ckpt_path = str(getattr(self.cfg.pretrain, "input_checkpoint", "") or "")
        if not ckpt_path:
            return
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"[Pretrain] pretrain.input_checkpoint not found: {ckpt_path}"
            )
        payload = torch.load(ckpt_path, map_location=self.device)
        state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        result = self.model.load_state_dict(state_dict, strict=False)
        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        print(f"[Pretrain] Loaded warm-start encoder weights from {ckpt_path}")
        if missing:
            print(f"[Pretrain][Warn] Missing keys during warm-start load: {missing}")
        if unexpected:
            print(f"[Pretrain][Warn] Unexpected keys during warm-start load: {unexpected}")

    # ------------------------------------------------------------------ #
    # LR scheduler
    # ------------------------------------------------------------------ #
    def _build_scheduler(self):
        """Build an LR scheduler based on config. Returns None when disabled."""
        return build_lr_scheduler(
            optimizer=self.optimizer,
            scheduler_name=getattr(self.cfg.pretrain, "scheduler", "none"),
            epochs=self.cfg.pretrain.epochs,
            step_size=int(getattr(self.cfg.pretrain, "scheduler_step_size", 50)),
            gamma=float(getattr(self.cfg.pretrain, "scheduler_gamma", 0.5)),
        )

    def _init_monitoring(self) -> None:
        """Resolve checkpoint monitoring for supervised and unsupervised pretraining."""
        label_dim = int(getattr(self.cfg.pretrain.dataset, "label_dim", 1) or 1)

        spec = resolve_pretrain_monitor_spec(
            self.cfg,
            task_level=str(getattr(self, "task_level_raw", self.cfg.pretrain.dataset.task_level) or "").lower(),
            label_dim=label_dim,
            uses_dataset_splits=self.use_dataset_splits,
            default_monitor=getattr(self.task_cls, "default_monitor", None),
        )
        self.monitor_name = spec.name
        self.monitor_mode = spec.mode
        self.best_metric = spec.best_metric
        self.best_epoch = None
        self.best_metrics: dict[str, float] = {}

    def _make_loaders(self, induced: bool):
        """Create train/val/test loaders with the appropriate split."""
        if not self.use_dataset_splits:
            return self._make_full_dataset_loaders(induced=induced)

        return make_workflow_loaders(
            dataset=self.dataset,
            dataset_name=self.cfg.pretrain.dataset.name,
            task_level_raw=self.task_level_raw,
            effective_task_level=self.effective_task_level,
            batch_size=self.cfg.pretrain.batch_size,
            num_workers=self.cfg.pretrain.num_workers,
            split=self.split,
            seed=self.cfg.seed,
            induced=induced,
            split_root=shared_split_root(self.cfg),
        )

    def _make_full_dataset_loaders(self, induced: bool):
        """
        Create train loader without split files for unsupervised pretraining.
        Validation/test loaders are intentionally omitted.
        """
        task_level = self._loader_task_level(induced=induced)
        if task_level in {"node", "edge"} and not induced:
            data = self.dataset[0]
            train_loader = SingleGraphDataLoader(data)
            return train_loader, None, None

        # Contrastive / cross-graph pretraining objectives (GraphCL,
        # InfoGraph, ContextPred) need >= 2 graphs per batch to produce a
        # meaningful loss. Keep valid partial batches: ``drop_last=True``
        # would discard every graph when len(dataset) < batch_size, and also
        # discards non-singleton tails that satisfy the objective. Task
        # implementations safely return an anchored zero loss only for the
        # rare trailing batch that is actually below their minimum.
        min_graphs = int(getattr(self.task_cls, "min_graphs_per_batch", 1))
        if int(self.cfg.pretrain.batch_size) < min_graphs:
            # With batch_size < min_graphs EVERY batch is "complete" yet
            # undersized, so the whole run silently trains on zero loss and
            # checkpoints near-random weights.
            raise ValueError(
                f"[Pretrain] pretrain.batch_size={self.cfg.pretrain.batch_size} is "
                f"below {self.task_cls.__name__}.min_graphs_per_batch={min_graphs}; "
                "the objective would be identically zero."
            )
        train_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.cfg.pretrain.batch_size,
            num_workers=self.cfg.pretrain.num_workers,
            shuffle=True,
            drop_last=False,
        )
        return train_loader, None, None

    def _loader_task_level(self, induced: bool) -> str:
        return resolve_loader_task_level(self.task_level_raw, self.effective_task_level, induced)

    def _resolve_split(self) -> tuple:
        ds_cfg = self.cfg.pretrain.dataset
        return resolve_workflow_split(
            getattr(ds_cfg, "fixed_split", None),
            default=(0.8, 0.1, 0.1),
        )

    def _build_run_name(self) -> str:
        """Generate run name from requested cfg.

        Delegates to ``src.utils.naming.build_pretrain_run_name_from_cfg``
        so the convention has a single source of truth shared with
        finetune checkpoint resolution.
        """
        return build_pretrain_run_name_from_cfg(
            self.cfg,
            include_split=self.use_dataset_splits,
        )

    def _build_run_group(self) -> str:
        dataset_cfg = self.cfg.pretrain.dataset
        return checkpoint_dataset_dir_name(dataset_cfg.name)

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
        log_dir = getattr(self.cfg.pretrain, "log_dir", "")
        if log_dir:
            return os.path.join(log_dir, self.run_group, f"{self.run_name}_log.json")
        return os.path.join(self.run_dir, f"{self.run_name}_log.json")

    def _architecture_path(self) -> str:
        log_dir = getattr(self.cfg.pretrain, "log_dir", "")
        if log_dir:
            return os.path.join(log_dir, self.run_group, f"{self.run_name}_architecture.json")
        return os.path.join(self.run_dir, f"{self.run_name}_architecture.json")

    def _artifact_config_dict(self) -> dict[str, Any]:
        """
        Return config dict used for artifact serialization.
        For methods that do not use splits, omit fixed_split entirely.
        """
        cfg_dict = cfg_to_dict(self.cfg)
        if self.use_dataset_splits:
            return cfg_dict
        pretrain_cfg = cfg_dict.get("pretrain")
        if isinstance(pretrain_cfg, dict):
            dataset_cfg = pretrain_cfg.get("dataset")
            if isinstance(dataset_cfg, dict):
                dataset_cfg.pop("fixed_split", None)
        return cfg_dict

    def _save_training_log(self) -> None:
        """Save the full pretraining log and config alongside the checkpoint."""
        save_training_log(
            path=self._log_path(),
            cfg=self._artifact_config_dict(),
            dataset_meta=self.dataset_meta,
            history=self.train_history,
            best_info={
                "epoch": self.best_epoch,
                "metric": self.best_metric,
                "monitor": self.monitor_name,
                "metrics": self.best_metrics,
            },
        )

    def _save_model_architecture(self) -> None:
        """Persist the model and task definitions in a readable JSON file."""
        arch_path = self._architecture_path()
        ensure_dir(os.path.dirname(arch_path))
        artifact_cfg = self._artifact_config_dict()
        payload = {
            "model": str(self.model),
            "task": str(self.task),
            "config": artifact_cfg,
            "dataset_meta": self.dataset_meta,
        }
        with open(arch_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _is_improved(self, metric: float) -> bool:
        return is_metric_improved(metric, self.best_metric, self.monitor_mode)

    def _save_best_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        logs: dict[str, float],
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
            **logs,
        }
        if self.monitor_name is not None:
            metrics[self.monitor_name] = float(monitor_value)
        ensure_dir(self.run_dir)
        save_checkpoint(
            path=self._checkpoint_path(),
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            cfg=self._artifact_config_dict(),
            dataset_meta=self.dataset_meta,
            metrics=metrics,
            extra={"pretrain_task_state": self.task.state_dict()},
        )
        self._checkpoint_written_this_run = True
        self.best_metrics = metrics
        self._save_training_log()
        self._save_model_architecture()
        if self.monitor_name is None:
            print(f"[Pretrain] Checkpoint updated at epoch={epoch} (monitor disabled).")
        else:
            print(f"[Best epoch updated] epoch={epoch} {self.monitor_name}={monitor_value:.4f}")
        return True

    def train_epoch(self):
        return run_step_epoch(
            model=self.model,
            task=self.task,
            loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            grad_clip=float(getattr(self.cfg.pretrain, "grad_clip", 0.0) or 0.0),
        )

    def _evaluate_split(self, loader, prefix: str, mask_attr: str) -> dict[str, float]:
        """Run evaluation for a supervised-style pretraining task with splits."""
        return runner_evaluate_split(
            model=self.model,
            task=self.task,
            loader=loader,
            device=self.device,
            prefix=prefix,
            mask_attr=mask_attr,
            task_type=resolve_task_type(getattr(self.cfg.pretrain.dataset, "task_type", None)),
        )

    def _evaluate_selected_test_checkpoint(self) -> dict[str, float]:
        """Restore the selected checkpoint and evaluate the held-out test once."""
        if not self.use_dataset_splits or self.test_loader is None:
            return {}
        if not self._checkpoint_written_this_run:
            return {}

        payload = torch.load(self._checkpoint_path(), map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
        task_state = (payload.get("extra") or {}).get("pretrain_task_state")
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
            cfg=payload.get("cfg", self._artifact_config_dict()),
            dataset_meta=payload.get("dataset", self.dataset_meta),
            metrics=metrics,
            extra=payload.get("extra"),
        )
        self.best_metrics = metrics
        return test_metrics

    def fit(self):
        """Run the pretraining process."""
        if getattr(self, "_skip_due_to_existing_checkpoint", False):
            return

        self._setup()

        ensure_dir(path=self.cfg.pretrain.checkpoint_dir)
        ensure_dir(self.run_dir)

        patience = int(getattr(self.cfg.pretrain, "early_stopping", 0) or 0)
        epochs_since_improvement = 0

        for epoch in range(1, self.cfg.pretrain.epochs + 1):
            start = time.time()
            loss, logs = self.train_epoch()
            duration = time.time() - start
            log_parts = [
                f"[Epoch {epoch}/{self.cfg.pretrain.epochs}]",
                f"train_loss={loss:.4f}",
            ]
            if logs:
                for k, v in logs.items():
                    if should_print_metric(k):
                        log_parts.append(f"{k}={v:.4f}")

            val_metrics: dict[str, float] = {}
            if self.task.uses_dataset_splits:
                val_metrics = self._evaluate_split(self.val_loader, prefix="val", mask_attr="val_mask")
                for k, v in val_metrics.items():
                    if should_print_metric(k):
                        log_parts.append(f"{k}={v:.4f}")

            # Step LR scheduler after each epoch.
            if self.scheduler is not None:
                log_parts.append(f"lr={self.optimizer.param_groups[0]['lr']:.2e}")
                self.scheduler.step()

            log_parts.append(f"time={duration:.1f}s")
            log_str = " ".join(log_parts)
            print(log_str)

            merged_metrics = merge_epoch_metrics(logs, val_metrics, {})
            self.train_history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss),
                    "duration_sec": float(duration),
                    "metrics": merged_metrics,
                }
            )

            monitor_value = resolve_monitor_value(
                self.monitor_name,
                train_loss=loss,
                train_logs=logs,
                val_metrics=val_metrics,
                test_metrics={},
            )
            improved = self._save_best_checkpoint(
                epoch=epoch,
                train_loss=loss,
                logs=merged_metrics,
                monitor_value=monitor_value,
            )

            if patience > 0:
                if improved:
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(
                        f"[Pretrain] Early stopping at epoch {epoch} "
                        f"(no improvement in {patience} epochs)."
                    )
                    break
        # If no checkpoint was saved (e.g., monitor_value never improved), save the last state.
        if self.train_history and not self._checkpoint_written_this_run:
            last = self.train_history[-1]
            last_metrics = dict(last.get("metrics") or {})
            self.best_epoch = int(last.get("epoch", 0))
            fallback_metrics = {
                "train_loss": float(last.get("loss", float("nan"))),
                "best_epoch": self.best_epoch,
                **last_metrics,
            }
            save_checkpoint(
                path=self._checkpoint_path(),
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.best_epoch,
                cfg=self._artifact_config_dict(),
                dataset_meta=self.dataset_meta,
                metrics=fallback_metrics,
                extra={
                    "pretrain_task_state": self.task.state_dict(),
                    "fallback_save": True,
                },
            )
            self._checkpoint_written_this_run = True
            self.best_metrics = fallback_metrics
            self.best_metric = float("nan")
            print(f"[Pretrain] No best checkpoint was saved; wrote last state to {self._checkpoint_path()}")

        final_test_metrics = self._evaluate_selected_test_checkpoint()
        if final_test_metrics:
            print(
                "[Pretrain] Selected-checkpoint test: "
                + " ".join(f"{key}={value:.4f}" for key, value in final_test_metrics.items())
            )
        # Checkpoint saves happen only on improvements, so explicitly write
        # the log once more to retain epochs after the selected one.
        if self.train_history:
            self._save_training_log()

        if self.monitor_name is not None:
            print(f"Pretraining complete. Best {self.monitor_name}: {self.best_metric:.4f} at epoch {self.best_epoch}.")
        else:
            final_epoch = self.train_history[-1]["epoch"] if self.train_history else 0
            print(f"[Pretrain] Complete. Final epoch: {final_epoch} (early stopping disabled).")
        if hasattr(self, "best_metrics"):
            for metric_name in ("test_acc", "test_micro_f1", "test_macro_f1", "test_auc", "test_mae", "test_mse"):
                metric_value = self.best_metrics.get(metric_name)
                if metric_value is not None:
                    print(f"[Pretrain] Best-epoch {metric_name}={metric_value:.4f}")
