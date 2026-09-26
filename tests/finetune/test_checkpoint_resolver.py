"""Checkpoint discovery and TSV metadata filter tests (Phase 0 / Commit 1).

Covers:
- ``collect_pretrained_checkpoints`` finds logs under a configured ``log_root``
  (matching the current ``cfg.pretrain.log_dir = "outputs/logs/pretrained_models"``
  default) AND falls back to legacy sidecar logs next to the ``.pt`` file.
- ``_select_checkpoints_for_task`` enforces strict metadata matching: a
  log-less checkpoint does NOT match a TSV row that specifies ``model``.
- ``pretrained_run_name`` still pins a specific checkpoint even when its
  metadata is missing.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.finetune.utils import (
    _build_task_cfg,
    _select_checkpoints_for_task,
    collect_pretrained_checkpoints,
)
from src.finetune.run import build_finetune_cfg
from src.finetune.finetuner import FinetuneRunner
from src.utils.save_results import get_explicit_cfg_keys


def _write_ckpt(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {}}, path)


def _write_log(
    path: Path,
    *,
    model: str,
    method: str,
    dataset: str,
    task_level: str,
    induced: bool,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "model": {"name": model},
            "pretrain": {
                "method": method,
                "dataset": {
                    "name": dataset,
                    "task_level": task_level,
                    "induced": induced,
                },
            },
            "seed": seed,
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


class CollectPretrainedCheckpointsTest(unittest.TestCase):
    def test_reads_log_from_separate_log_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ckpt_root = tmp / "pretrained_models"
            log_root = tmp / "logs" / "pretrained_models"

            run_name = (
                "attr_masking_cora_tasknode_induced1_gcn_h128_o128_l2_"
                "e500_lr0.001_bs128_seed42"
            )
            _write_ckpt(ckpt_root / "cora" / f"{run_name}.pt")
            _write_log(
                log_root / "cora" / f"{run_name}_log.json",
                model="gcn",
                method="attr_masking",
                dataset="cora",
                task_level="node",
                induced=True,
                seed=42,
            )

            ckpts = collect_pretrained_checkpoints(
                str(ckpt_root), log_root=str(log_root)
            )
            self.assertEqual(len(ckpts), 1)
            self.assertEqual(ckpts[0]["model"], "gcn")
            self.assertEqual(ckpts[0]["method"], "attr_masking")
            self.assertEqual(ckpts[0]["dataset"], "cora")
            self.assertEqual(ckpts[0]["seed"], 42)

    def test_falls_back_to_sidecar_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ckpt_root = tmp / "pretrained_models"

            run_name = (
                "attr_masking_cora_tasknode_induced1_gin_h128_o128_l2_"
                "e500_lr0.001_bs128_seed42"
            )
            _write_ckpt(ckpt_root / "cora" / f"{run_name}.pt")
            _write_log(
                ckpt_root / "cora" / f"{run_name}_log.json",
                model="gin",
                method="attr_masking",
                dataset="cora",
                task_level="node",
                induced=True,
                seed=42,
            )

            # No log_root provided → fall back to sidecar next to .pt.
            ckpts = collect_pretrained_checkpoints(str(ckpt_root))
            self.assertEqual(len(ckpts), 1)
            self.assertEqual(ckpts[0]["model"], "gin")

    def test_log_root_preferred_over_sidecar(self):
        # When both locations have a log, log_root wins.  Sidecar is a
        # legacy fallback only.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ckpt_root = tmp / "pretrained_models"
            log_root = tmp / "logs" / "pretrained_models"
            run_name = "attr_masking_cora_tasknode_induced1_gcn_seed42"
            _write_ckpt(ckpt_root / "cora" / f"{run_name}.pt")
            _write_log(
                ckpt_root / "cora" / f"{run_name}_log.json",
                model="WRONG",
                method="attr_masking",
                dataset="cora",
                task_level="node",
                induced=True,
                seed=42,
            )
            _write_log(
                log_root / "cora" / f"{run_name}_log.json",
                model="gcn",
                method="attr_masking",
                dataset="cora",
                task_level="node",
                induced=True,
                seed=42,
            )
            ckpts = collect_pretrained_checkpoints(
                str(ckpt_root), log_root=str(log_root)
            )
            self.assertEqual(len(ckpts), 1)
            self.assertEqual(ckpts[0]["model"], "gcn")

    def test_log_missing_yields_none_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ckpt_root = tmp / "pretrained_models"
            log_root = tmp / "logs" / "pretrained_models"
            run_name = "attr_masking_cora_tasknode_induced1_gcn_seed42"
            _write_ckpt(ckpt_root / "cora" / f"{run_name}.pt")
            # No logs anywhere.
            ckpts = collect_pretrained_checkpoints(
                str(ckpt_root), log_root=str(log_root)
            )
            self.assertEqual(len(ckpts), 1)
            self.assertIsNone(ckpts[0]["model"])
            self.assertIsNone(ckpts[0]["method"])
            # Seed is derivable from the run name, not the log.
            self.assertEqual(ckpts[0]["seed"], 42)


class SelectCheckpointsStrictFilterTest(unittest.TestCase):
    """Strict: a checkpoint with missing metadata cannot match a TSV row
    that specifies the corresponding field."""

    def _make_ckpts(self, with_logs: bool = True) -> list[dict]:
        """Produce one ckpt per model for dataset=cora, seed=42."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ckpt_root = tmp / "pretrained_models"
            log_root = tmp / "logs" / "pretrained_models"
            models = ["gcn", "gin", "gat", "fagcn"]
            for m in models:
                run_name = (
                    f"attr_masking_cora_tasknode_induced1_{m}_h128_o128_l2_"
                    "e500_lr0.001_bs128_seed42"
                )
                _write_ckpt(ckpt_root / "cora" / f"{run_name}.pt")
                if with_logs:
                    _write_log(
                        log_root / "cora" / f"{run_name}_log.json",
                        model=m,
                        method="attr_masking",
                        dataset="cora",
                        task_level="node",
                        induced=True,
                        seed=42,
                    )
            return collect_pretrained_checkpoints(
                str(ckpt_root), log_root=str(log_root)
            )

    def test_model_field_narrows_to_single_checkpoint(self):
        ckpts = self._make_ckpts(with_logs=True)
        task = {
            "dataset": "cora",
            "model": "fagcn",
            "pretrain_dataset": "cora",
            "pretrain_task_level": "node",
            "pretrain_induced": True,
            "pretrain_method": "attr_masking",
        }
        sel = _select_checkpoints_for_task(ckpts, task)
        self.assertEqual(len(sel), 1)
        self.assertEqual(sel[0]["model"], "fagcn")

    def test_strict_rejects_logless_checkpoint_when_model_specified(self):
        ckpts = self._make_ckpts(with_logs=False)
        # All ckpts have model=None → strict filter must reject them when
        # the TSV row specifies model=fagcn.
        task = {"dataset": "cora", "model": "fagcn"}
        sel = _select_checkpoints_for_task(ckpts, task)
        self.assertEqual(sel, [])

    def test_pretrained_run_name_overrides_even_without_logs(self):
        ckpts = self._make_ckpts(with_logs=False)
        target = next(c for c in ckpts if "_fagcn_" in c["run_name"])
        task = {"dataset": "cora", "pretrained_run_name": target["run_name"]}
        sel = _select_checkpoints_for_task(ckpts, task)
        self.assertEqual(len(sel), 1)
        self.assertEqual(sel[0]["run_name"], target["run_name"])

    def test_no_source_columns_returns_all(self):
        ckpts = self._make_ckpts(with_logs=True)
        task = {"dataset": "cora"}
        sel = _select_checkpoints_for_task(ckpts, task)
        self.assertEqual(len(sel), len(ckpts))


class CheckpointVariantReplayTest(unittest.TestCase):
    def test_infograph_nolw_variant_is_replayed_and_recorded(self):
        base_cfg = build_finetune_cfg(
            [
                "finetune.dataset.name",
                "qm7b",
                "finetune.dataset.task_level",
                "graph",
            ]
        )
        task = {
            "dataset": "qm7b",
            "task_level": "graph",
            "induced": False,
            "task_type": "regression",
            "fixed_split": (5, 0.0, 1.0),
            "finetune_method": "supervised",
        }
        checkpoint_meta = {
            "path": "outputs/pretrained_models/qm7b/infograph-nolw.pt",
            "run_name": "infograph-nolw_qm7b_seed42",
            "dataset": "qm7b",
            "task_level": "graph",
            "induced": False,
            "method": "infograph",
            "model": "transformer",
            "_cfg_dict": {
                "pretrain": {
                    "method": "infograph",
                    "infograph": {"use_layerwise": False},
                }
            },
        }

        run_cfg = _build_task_cfg(base_cfg, task, checkpoint_meta)

        self.assertFalse(run_cfg.pretrain.infograph.use_layerwise)
        self.assertIn(
            "pretrain.infograph.use_layerwise",
            get_explicit_cfg_keys(run_cfg),
        )


class PretrainedModelConfigReplayTest(unittest.TestCase):
    def test_explicit_graph_pooling_survives_architecture_replay(self):
        cfg = build_finetune_cfg(
            [
                "finetune.dataset.name",
                "qm7b",
                "finetune.dataset.task_level",
                "graph",
                "model.graph_pooling",
                "sum",
            ]
        )
        cfg.model.hidden_dim = 999
        runner = object.__new__(FinetuneRunner)
        runner.cfg = cfg
        runner.pretrain_cfg = {
            "model": {
                "hidden_dim": 128,
                "graph_pooling": "mean",
            }
        }

        runner._apply_pretrained_model_cfg()

        self.assertEqual(runner.cfg.model.hidden_dim, 128)
        self.assertEqual(runner.cfg.model.graph_pooling, "sum")

    def test_implicit_graph_pooling_is_replayed_from_checkpoint(self):
        cfg = build_finetune_cfg(
            [
                "finetune.dataset.name",
                "qm7b",
                "finetune.dataset.task_level",
                "graph",
            ]
        )
        cfg.model.graph_pooling = "sum"
        runner = object.__new__(FinetuneRunner)
        runner.cfg = cfg
        runner.pretrain_cfg = {"model": {"graph_pooling": "max"}}

        runner._apply_pretrained_model_cfg()

        self.assertEqual(runner.cfg.model.graph_pooling, "max")


if __name__ == "__main__":
    unittest.main()
