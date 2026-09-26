"""Mowst training runner.

Mirrors :class:`src.moe.gmoe.trainer.GMoERunner` /
:class:`src.moe.graphmore.trainer.GraphMoRERunner` (and therefore
``src.train.trainer.TrainRunner``) but reads ``cfg.moe.mowst`` and builds a
:class:`MowstModel` + :class:`MowstTask`. All dataset loading, split handling,
metric computation, monitoring, checkpointing, and logging reuse the same shared
helpers, so Mowst results are directly comparable.

Two method-specific behaviours:

* **Per-expert warm-up** (``submethod``) trains the weak and/or strong expert +
  its head alone before the gated stage.
* The **mowst** variant alternates weak/strong *turns*: even epochs train the
  weak expert + gate (strong frozen), odd epochs train the strong expert (weak +
  gate frozen), each with the separate gate-weighted loss. The **mowst_star**
  variant trains everything jointly with one optimizer. (The reference's nested
  big-epoch / inner-turn / rollback schedule, designed for full-batch
  transductive training, is adapted to per-epoch alternation in the batched
  subgraph setting.)
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
from src.utils.training import build_lr_scheduler

from .model import MowstModel
from .task import MowstTask


class MowstRunner:
    """Train a Mowst model from scratch using ``cfg.moe.mowst`` settings."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.m_cfg = cfg.moe.mowst
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        set_seed(seed=self.cfg.seed)

        self.best_metrics: dict[str, float] = {}
        self.best_epoch = None
        self.best_metric = float("nan")
        self.monitor_name = "val_acc"
        self.monitor_mode = "max"
        self.train_history: list[dict] = []

        self.variant = str(self.m_cfg.variant).strip().lower()
        if self.variant not in {"mowst", "mowst_star"}:
            raise ValueError(
                f"Unsupported moe.mowst.variant='{self.m_cfg.variant}'. Use mowst / mowst_star."
            )

        ds_cfg = self.m_cfg.dataset
        self.task_level_raw = str(ds_cfg.task_level)
        self.split = self._resolve_split()
        self.run_name = self._build_run_name()
        self.run_group = f"{ds_cfg.name}-{self.task_level_raw}"
        self.run_dir = os.path.join(self.m_cfg.checkpoint_dir, self.run_group)
        self._skip_due_to_existing_checkpoint = False
        self._is_setup = False
        self._turn_epoch = 0

        ckpt_path = self._existing_checkpoint_path()
        if self.m_cfg.skip_if_exists and ckpt_path is not None:
            self._skip_due_to_existing_checkpoint = True
            print(f"[Mowst] Checkpoint already exists, skipping: {ckpt_path}")
            return

    # ------------------------------------------------------------------ #
    # Lazy heavy setup
    # ------------------------------------------------------------------ #
    def _setup(self) -> None:
        if self._is_setup:
            return
        self._is_setup = True

        cfg = self.cfg
        m_cfg = self.m_cfg
        ds_cfg = m_cfg.dataset
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
        # Fills m_cfg.in_dim and ds_cfg.{num_classes,label_dim,task_type}.
        populate_dataset_cfg_from_meta(m_cfg, ds_cfg, self.dataset_meta)

        self.model = MowstModel(
            in_dim=int(m_cfg.in_dim),
            hidden_dim=int(m_cfg.hidden_dim),
            weak_model=str(m_cfg.weak.model),
            weak_num_layers=int(m_cfg.weak.num_layers),
            weak_dropout=float(m_cfg.weak.dropout),
            strong_model=str(m_cfg.strong.model),
            strong_num_layers=int(m_cfg.strong.num_layers),
            strong_dropout=float(m_cfg.strong.dropout),
            gat_heads=int(getattr(m_cfg.strong, "gat_heads", 2)),
            act=str(m_cfg.activation),
            use_batchnorm=bool(m_cfg.use_batchnorm),
            graph_pooling=str(m_cfg.graph_pooling),
        ).to(self.device)
        print(f"[Mowst] Model architecture:\n{self.model}")

        self.task = MowstTask(cfg).to(self.device)
        print(f"[Mowst] Task heads + gate:\n{self.task}")

        self._build_optimizers()
        self.scheduler = self._build_scheduler(self.optimizer)
        # The alternating mowst variant has a second optimizer for the strong
        # turn; it must decay in lockstep with the weak one, else the two
        # symmetric turns diverge in effective LR under a non-default scheduler.
        self.scheduler_strong = (
            self._build_scheduler(self.optimizer_strong)
            if self.optimizer_strong is not None
            else None
        )

        self._init_monitoring()
        ensure_dir(self.run_dir)
        self.train_loader, self.val_loader, self.test_loader = make_workflow_loaders(
            dataset=self.dataset,
            dataset_name=ds_cfg.name,
            task_level_raw=raw_task_level,
            effective_task_level=self.effective_task_level,
            batch_size=int(m_cfg.batch_size),
            num_workers=int(m_cfg.num_workers),
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
            prefix="[Mowst][Split]",
        )

    def _build_scheduler(self, optimizer):
        m_cfg = self.m_cfg
        return build_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=str(getattr(m_cfg, "scheduler", "none")),
            epochs=int(m_cfg.epochs),
            step_size=int(getattr(m_cfg, "scheduler_step_size", 50)),
            gamma=float(getattr(m_cfg, "scheduler_gamma", 0.5)),
        )

    def _build_optimizers(self) -> None:
        m_cfg = self.m_cfg
        wd = float(m_cfg.weight_decay)
        if self.variant == "mowst_star":
            # One optimizer over both experts + both heads + gate.
            params = list(self.model.parameters()) + list(self.task.parameters_to_optimize())
            self.optimizer = optim.Adam(params, lr=float(m_cfg.lr_gate), weight_decay=wd)
            self.optimizer_strong = None
        else:  # mowst — alternating turns: two optimizers
            weak_params = (
                list(self.model.weak.parameters()) + self.task.weak_branch_parameters()
            )
            strong_params = (
                list(self.model.strong.parameters()) + self.task.strong_branch_parameters()
            )
            self.optimizer = optim.Adam(weak_params, lr=float(m_cfg.lr), weight_decay=wd)
            self.optimizer_strong = optim.Adam(strong_params, lr=float(m_cfg.lr), weight_decay=wd)

    # ------------------------------------------------------------------ #
    # Per-expert warm-up
    # ------------------------------------------------------------------ #
    def _pretrain(self) -> None:
        submethod = str(self.m_cfg.submethod).strip().lower()
        if submethod == "none":
            return
        epochs = int(self.m_cfg.pretrain_epochs)
        if epochs <= 0:
            return
        if submethod in ("pretrain_model1", "pretrain_both"):
            self._pretrain_expert("weak", epochs)
        if submethod in ("pretrain_model2", "pretrain_both"):
            self._pretrain_expert("strong", epochs)

    def _pretrain_expert(self, which: str, epochs: int) -> None:
        if which == "weak":
            params = list(self.model.weak.parameters()) + list(self.task.weak_head.parameters())
        else:
            params = list(self.model.strong.parameters()) + list(self.task.strong_head.parameters())
        opt = optim.Adam(params, lr=float(self.m_cfg.lr), weight_decay=float(self.m_cfg.weight_decay))
        log_every = max(1, epochs // 5)
        print(f"[Mowst] Pretraining {which} expert for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            self.model.train()
            self.task.train()
            total_loss = 0.0
            num_batches = 0
            for data in self.train_loader:
                opt.zero_grad()
                loss, _primary = self.task.pretrain_step(self.model, data, self.device, which)
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
                num_batches += 1
            if epoch == 1 or epoch % log_every == 0:
                avg = total_loss / max(1, num_batches)
                print(f"[Mowst][Pretrain][{which}] epoch={epoch}/{epochs} loss={avg:.4f}")

    # ------------------------------------------------------------------ #
    # Monitoring
    # ------------------------------------------------------------------ #
    def _init_monitoring(self) -> None:
        ds_cfg = self.m_cfg.dataset
        label_dim = int(getattr(ds_cfg, "label_dim", 1) or 1)
        spec = resolve_explicit_monitor_spec(
            raw_monitor_metric=getattr(self.m_cfg, "monitor_metric", "auto"),
            setting_name="moe.mowst.monitor_metric",
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
        raw_split = getattr(self.m_cfg.dataset, "fixed_split", None)
        if raw_split is None:
            raw_split = getattr(self.m_cfg, "fixed_split", None)
        split = resolve_workflow_split(raw_split, default=(0.8, 0.1, 0.1))
        if is_few_shot_split(split):
            val_ratio = float(split[1])
            test_ratio = float(split[2])
            if abs(val_ratio) > 1e-6 or abs(test_ratio - 1.0) > 1e-6:
                raise ValueError("Few-shot split must be (shots, 0.0, 1.0).")
        return split

    def _build_run_name(self) -> str:
        m_cfg = self.m_cfg
        ds_cfg = m_cfg.dataset
        split_tag = format_split_for_name(self.split)
        fingerprint = behavior_fingerprint(
            m_cfg,
            external_behavior={
                "shared_split_root": shared_split_root(self.cfg),
                "shared_induced_root": shared_induced_root(
                    self.cfg, getattr(ds_cfg, "induced_root", "")
                ),
            },
        )
        parts = [
            "mowst",
            self.variant,
            ds_cfg.name,
            f"induced{int(getattr(ds_cfg, 'induced', False))}",
            split_tag,
            f"task{self.task_level_raw}",
            f"w{m_cfg.weak.model}",
            f"s{m_cfg.strong.model}",
            m_cfg.subloss if self.variant == "mowst_star" else "separate",
            f"h{m_cfg.hidden_dim}",
            f"e{m_cfg.epochs}",
            f"bs{m_cfg.batch_size}",
            # Both LRs are embedded so runs differing only in lr (pretrain /
            # mowst turns) or lr_gate (mowst_star joint) get distinct checkpoint
            # paths, matching GMoE/GraphMoRE which embed their used LR.
            f"lr{m_cfg.lr:g}" if isinstance(m_cfg.lr, (int, float)) else f"lr{m_cfg.lr}",
            f"lrg{m_cfg.lr_gate:g}" if isinstance(m_cfg.lr_gate, (int, float)) else f"lrg{m_cfg.lr_gate}",
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
        log_dir = getattr(self.m_cfg, "log_dir", "")
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
        """Track the best epoch in memory; the checkpoint (with its one-time
        val/test evaluation) is written by ``_finalize_best_checkpoint``, so
        interrupted runs never leave a complete-looking checkpoint behind."""
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
            print(f"[Mowst] Best state updated at epoch={epoch} (monitor disabled).")
        else:
            print(f"[Mowst] Best epoch updated: epoch={epoch} {self.monitor_name}={monitor_value:.4f}")
        return True

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
        metrics = {
            "train_loss": context["train_loss"],
            "best_epoch": epoch,
            **context["train_logs"],
            **context["val_metrics"],
            **test_metrics,
        }
        if self.monitor_name is not None:
            metrics[self.monitor_name] = float(context["monitor_value"])
        extra = {"mowst_task_state": self.task.state_dict()}
        if self.optimizer_strong is not None:
            extra["strong_optimizer"] = self.optimizer_strong.state_dict()
        save_checkpoint(
            path=self._checkpoint_path(),
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            cfg=self.cfg,
            dataset_meta=self.dataset_meta,
            metrics=metrics,
            extra=extra,
        )
        self.best_metrics = metrics
        self._save_training_log()
        print(f"[Mowst] Saved best-epoch checkpoint (epoch={epoch}) after final evaluation.")

    # ------------------------------------------------------------------ #
    # Train / eval loops
    # ------------------------------------------------------------------ #
    def train_epoch(self) -> tuple[float, dict]:
        """One step-based epoch.

        ``mowst_star`` runs the joint step over a single optimizer.
        ``mowst`` alternates: even epochs train the weak expert + gate (strong
        frozen), odd epochs train the strong expert (weak + gate frozen).
        """
        model = self.model
        task = self.task
        model.train()
        task.train()
        total_loss = 0.0
        logs: dict[str, float] = {}
        grad_clip = float(getattr(self.m_cfg, "grad_clip", 0.0) or 0.0)

        if self.variant == "mowst_star":
            optimizer = self.optimizer
            opt_params = list(model.parameters()) + list(task.parameters_to_optimize())
            turn = None
        else:
            turn = "weak" if (self._turn_epoch % 2 == 0) else "strong"
            optimizer = self.optimizer if turn == "weak" else self.optimizer_strong
            opt_params = [p for group in optimizer.param_groups for p in group["params"]]

        for data in self.train_loader:
            optimizer.zero_grad()
            if turn is None:
                loss, log = task.step(model=model, data=data, device=self.device)
            else:
                loss, log = task.step_turn(model=model, data=data, device=self.device, turn=turn)
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(opt_params, max_norm=grad_clip)
            optimizer.step()
            total_loss += loss.item()
            for k, v in log.items():
                logs[k] = logs.get(k, 0.0) + float(v)

        num_batches = len(self.train_loader)
        if num_batches == 0:
            raise RuntimeError("Train loader is empty; unable to run a training epoch.")
        if turn is not None:
            print(f"[Mowst][Turn] epoch trained the {turn} branch")
            self._turn_epoch += 1
        avg_loss = total_loss / num_batches
        logs = {k: v / num_batches for k, v in logs.items()}
        return avg_loss, logs

    def _evaluate_split(self, loader, prefix: str, mask_attr: str) -> dict[str, float]:
        return runner_evaluate_split(
            model=self.model,
            task=self.task,
            loader=loader,
            device=self.device,
            prefix=prefix,
            mask_attr=mask_attr,
            task_type=resolve_task_type(getattr(self.m_cfg.dataset, "task_type", None)),
        )

    def fit(self) -> None:
        if getattr(self, "_skip_due_to_existing_checkpoint", False):
            return

        self._setup()
        self._pretrain()

        if os.path.isfile(self._checkpoint_path()):
            print(f"[Mowst] Overwriting existing checkpoint: {self._checkpoint_path()}")

        patience = int(getattr(self.m_cfg, "early_stopping", 0) or 0)
        epochs_since_improvement = 0
        # Val/test evaluation policy: skip the val pass when the monitor only
        # needs train metrics (few-shot splits), and defer the test pass to a
        # single final evaluation of the best epoch's weights — per-epoch test
        # metrics would only feed logs while dominating runtime on the larger
        # datasets.
        monitor_on_train = monitor_uses_train_split(self.monitor_name)
        self._best_context = None
        self._last_context = None
        self._best_model_state = None
        self._best_task_state = None

        for epoch in range(1, int(self.m_cfg.epochs) + 1):
            start = time.time()
            train_loss, train_logs = self.train_epoch()
            if monitor_on_train:
                val_metrics = {}
            else:
                val_metrics = self._evaluate_split(self.val_loader, prefix="val", mask_attr="val_mask")

            duration = time.time() - start
            log_parts = [
                f"[Mowst][Epoch {epoch}/{self.m_cfg.epochs}]",
                f"train_loss={train_loss:.4f}",
            ]
            for metrics in (train_logs, val_metrics):
                for k, v in metrics.items():
                    if not should_print_metric(k):
                        continue
                    log_parts.append(f"{k}={v:.4f}")

            if self.scheduler is not None:
                log_parts.append(f"lr={self.optimizer.param_groups[0]['lr']:.2e}")
                self.scheduler.step()
                if self.scheduler_strong is not None:
                    self.scheduler_strong.step()

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
                        f"[Mowst] Early stopping at epoch {epoch} "
                        f"(no improvement in {patience} epochs)."
                    )
                    break

        self._finalize_best_checkpoint()

        if self.monitor_name is not None:
            print(
                f"[Mowst] Complete. Best {self.monitor_name}: "
                f"{self.best_metric:.4f} at epoch {self.best_epoch}."
            )
        else:
            final_epoch = self.train_history[-1]["epoch"] if self.train_history else 0
            print(f"[Mowst] Complete. Final epoch: {final_epoch} (early stopping disabled).")
        for metric_name in ("test_acc", "test_micro_f1", "test_macro_f1", "test_auc", "test_mae", "test_mse"):
            metric_value = self.best_metrics.get(metric_name)
            if metric_value is not None:
                print(f"[Mowst] Best-epoch {metric_name}={metric_value:.4f}")


__all__ = ["MowstRunner"]
