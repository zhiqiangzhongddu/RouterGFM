import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch_geometric.data import Data
from yacs.config import CfgNode as CN

from src.config import cfg as base_cfg
from src.data_loader.dataset_loader import make_loaders
from src.data_loader.datasets import _induced_feature_identity, _induced_feature_tag
from src.data_loader.summary import DatasetSummaryRow, _load_existing_summary_rows, _rows_to_tsv
from src.moe.anygraph.run import _checkpoint_exists
from src.moe.anygraph.src.checkpoint_io import load_checkpoint_pair, save_checkpoint_pair
from src.moe.gmoe.trainer import GMoERunner
from src.moe.graphmore.trainer import GraphMoRERunner
from src.moe.identity import behavior_fingerprint
from src.moe.mowst.trainer import MowstRunner


def _summary_row(name):
    return DatasetSummaryRow(name, "node", 1, 3, 3.0, 2, 2.0, 4, "classification", 2)


class SummaryPersistenceTest(unittest.TestCase):
    def test_summary_merges_under_lock_and_replaces_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.tsv"
            self.assertEqual(_rows_to_tsv([_summary_row("cora")], path), 1)
            self.assertEqual(_rows_to_tsv([_summary_row("pubmed")], path), 2)
            self.assertEqual([row.name for row in _load_existing_summary_rows(path)], ["cora", "pubmed"])
            self.assertTrue((Path(tmp) / "summary.tsv.lock").is_file())
            self.assertEqual(list(Path(tmp).glob(".summary.tsv.*.tmp")), [])

    def test_failed_replace_preserves_previous_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.tsv"
            _rows_to_tsv([_summary_row("cora")], path)
            before = path.read_bytes()
            with patch("src.data_loader.summary.os.replace", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    _rows_to_tsv([_summary_row("pubmed")], path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(tmp).glob(".summary.tsv.*.tmp")), [])


class EdgeMessageContextTest(unittest.TestCase):
    def test_eval_context_adds_train_positives_but_not_heldout_targets(self):
        data = Data(
            x=torch.randn(5, 3),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
            num_nodes=5,
        )

        class Dataset:
            def __getitem__(self, index):
                self.assert_index = index
                return data

        payload = {
            "train_pos_idx": [0],
            "val_pos_idx": [1],
            "test_pos_idx": [2],
            "message_pos_idx": [3],
            "context_pos_idx": [0, 3],
            "train_neg_edge_index": torch.empty((2, 0), dtype=torch.long),
            "val_neg_edge_index": torch.empty((2, 0), dtype=torch.long),
            "test_neg_edge_index": torch.empty((2, 0), dtype=torch.long),
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.data_loader.dataset_loader._get_or_create_edge_split_payload",
            return_value=payload,
        ):
            train, val, test = make_loaders(
                Dataset(), "toy", "edge", 4, 0, (0.25, 0.25, 0.25), 1,
                induced=False, split_root=tmp,
            )
        train_data = next(iter(train))
        val_data = next(iter(val))
        test_data = next(iter(test))
        self.assertTrue(torch.equal(train_data.edge_index, data.edge_index[:, [3]]))
        expected_eval = data.edge_index[:, [0, 3]]
        self.assertTrue(torch.equal(val_data.edge_index, expected_eval))
        self.assertTrue(torch.equal(test_data.edge_index, expected_eval))


class InducedCacheIdentityTest(unittest.TestCase):
    def test_feature_source_and_reduction_change_cache_identity(self):
        dataset = SimpleNamespace(root="data/datasets/toy")
        raw = _induced_feature_identity(
            base_dataset=dataset, requested_root="data/datasets", feat_reduction=False,
            feat_reduction_dim=100, persist_feature_svd=True, feature_svd_dir="data/feature_svd",
        )
        svd100 = _induced_feature_identity(
            base_dataset=dataset, requested_root="data/datasets", feat_reduction=True,
            feat_reduction_dim=100, persist_feature_svd=True, feature_svd_dir="data/feature_svd",
        )
        svd64 = dict(svd100, feat_reduction_dim=64)
        other_source = dict(svd100, feature_source="persisted_svd:/other")
        tags = {_induced_feature_tag(value) for value in (raw, svd100, svd64, other_source)}
        self.assertEqual(len(tags), 4)


class AnyGraphCheckpointTest(unittest.TestCase):
    def test_atomic_pair_is_loadable_and_completion_check_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "Models"
            history = Path(tmp) / "History"
            model_path = models / "anygraph_node_test.mod"
            history_path = history / "anygraph_node_test.his"
            save_checkpoint_pair(
                torch_module=torch,
                model_path=str(model_path),
                history_path=str(history_path),
                model_payload={"model": torch.nn.Linear(2, 2)},
                history_payload={"loss": [1.0]},
            )
            self.assertTrue(_checkpoint_exists(models, history, "anygraph_node_test"))
            manifest_path = Path(f"{model_path}.pair.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["generation"])
            for label in ("model", "history"):
                self.assertGreater(manifest["files"][label]["size"], 0)
                self.assertEqual(len(manifest["files"][label]["sha256"]), 64)
            self.assertFalse(Path(f"{model_path}.pending").exists())
            self.assertEqual(list(models.glob("*.tmp.*")), [])
            model_path.write_bytes(b"truncated")
            self.assertFalse(_checkpoint_exists(models, history, "anygraph_node_test"))

    def test_manifest_rejects_new_model_with_old_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "Models"
            history = Path(tmp) / "History"
            model_path = models / "anygraph_node_test.mod"
            history_path = history / "anygraph_node_test.his"
            save_checkpoint_pair(
                torch_module=torch,
                model_path=str(model_path),
                history_path=str(history_path),
                model_payload={"model": torch.nn.Linear(2, 2)},
                history_payload={"loss": [1.0]},
            )

            # Simulate writer death after publishing the next model but before
            # publishing its history/manifest. Both legacy files remain valid
            # individually, but the old manifest must reject the mixed pair.
            replacement = models / "replacement.mod"
            torch.save({"model": torch.nn.Linear(3, 3)}, replacement)
            os.replace(replacement, model_path)

            self.assertFalse(_checkpoint_exists(models, history, "anygraph_node_test"))
            with self.assertRaisesRegex(ValueError, "pair manifest"):
                load_checkpoint_pair(
                    torch_module=torch,
                    model_path=str(model_path),
                    history_path=str(history_path),
                )

    def test_pending_marker_blocks_first_generation_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "Models"
            history = Path(tmp) / "History"
            models.mkdir()
            history.mkdir()
            model_path = models / "anygraph_node_test.mod"
            history_path = history / "anygraph_node_test.his"
            torch.save({"model": torch.nn.Linear(2, 2)}, model_path)
            with history_path.open("wb") as history_fh:
                pickle.dump({"loss": [1.0]}, history_fh)
            Path(f"{model_path}.pending").write_text(
                '{"version":1,"generation":"interrupted"}', encoding="utf-8"
            )

            self.assertFalse(_checkpoint_exists(models, history, "anygraph_node_test"))
            with self.assertRaisesRegex(ValueError, "Incomplete AnyGraph"):
                load_checkpoint_pair(
                    torch_module=torch,
                    model_path=str(model_path),
                    history_path=str(history_path),
                )

    def test_legacy_pair_without_manifest_or_pending_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "Models"
            history = Path(tmp) / "History"
            models.mkdir()
            history.mkdir()
            model_path = models / "anygraph_node_legacy.mod"
            history_path = history / "anygraph_node_legacy.his"
            torch.save({"model": torch.nn.Linear(2, 2)}, model_path)
            with history_path.open("wb") as history_fh:
                pickle.dump({"loss": [1.0]}, history_fh)

            self.assertTrue(_checkpoint_exists(models, history, "anygraph_node_legacy"))
            checkpoint, loaded_history = load_checkpoint_pair(
                torch_module=torch,
                model_path=str(model_path),
                history_path=str(history_path),
            )
            self.assertIn("model", checkpoint)
            self.assertEqual(loaded_history, {"loss": [1.0]})


class MoEIdentityTest(unittest.TestCase):
    RUNNERS = {
        "gmoe": GMoERunner,
        "graphmore": GraphMoRERunner,
        "mowst": MowstRunner,
    }

    @staticmethod
    def _cfg():
        cfg = base_cfg.clone()
        cfg.seed = 42
        for method in MoEIdentityTest.RUNNERS:
            method_cfg = getattr(cfg.moe, method)
            method_cfg.checkpoint_dir = f"/tmp/icg-moe-identity/{method}/checkpoints"
            method_cfg.log_dir = f"/tmp/icg-moe-identity/{method}/logs"
            method_cfg.skip_if_exists = False
        return cfg

    def test_external_behavior_and_orchestration_exclusions(self):
        method_cfg = CN({
            "dropout": 0.1,
            "checkpoint_dir": "checkpoints-a",
            "log_dir": "logs-a",
            "num_runs": 5,
            "run_tasks_tsv": False,
            "skip_if_exists": True,
            "tasks_tsv": "tasks-a.tsv",
        })
        external = {"shared_split_root": "splits-a"}
        baseline = behavior_fingerprint(method_cfg, external_behavior=external)

        behavior_changed = method_cfg.clone()
        behavior_changed.dropout = 0.2
        self.assertNotEqual(
            behavior_fingerprint(behavior_changed, external_behavior=external), baseline
        )
        self.assertNotEqual(
            behavior_fingerprint(
                method_cfg,
                external_behavior={"shared_split_root": "splits-b"},
            ),
            baseline,
        )

        operational_changed = method_cfg.clone()
        operational_changed.checkpoint_dir = "checkpoints-b"
        operational_changed.log_dir = "logs-b"
        operational_changed.num_runs = 9
        operational_changed.run_tasks_tsv = True
        operational_changed.skip_if_exists = False
        operational_changed.tasks_tsv = "tasks-b.tsv"
        self.assertEqual(
            behavior_fingerprint(operational_changed, external_behavior=external),
            baseline,
        )
        with self.assertRaisesRegex(TypeError, "JSON-serializable"):
            behavior_fingerprint(method_cfg, external_behavior={"bad": object()})

    def test_gmoe_activation_changes_run_identity(self):
        cfg = self._cfg()
        baseline = GMoERunner(cfg).run_name

        changed = cfg.clone()
        changed.model.activation = "gelu"
        self.assertNotEqual(GMoERunner(changed).run_name, baseline)

    def test_shared_data_provenance_changes_all_runner_identities(self):
        for method, runner_cls in self.RUNNERS.items():
            with self.subTest(method=method, provenance="split_root"):
                cfg = self._cfg()
                baseline = runner_cls(cfg).run_name
                changed = cfg.clone()
                changed.data_preparation.dataset.split_root = "data/alternate_splits"
                self.assertNotEqual(runner_cls(changed).run_name, baseline)

            with self.subTest(method=method, provenance="induced_root"):
                changed = cfg.clone()
                changed.data_preparation.dataset.induced_root = "data/alternate_induced"
                self.assertNotEqual(runner_cls(changed).run_name, baseline)

    def test_runner_operational_knobs_do_not_change_identity(self):
        for method, runner_cls in self.RUNNERS.items():
            with self.subTest(method=method):
                cfg = self._cfg()
                baseline = runner_cls(cfg).run_name
                changed = cfg.clone()
                method_cfg = getattr(changed.moe, method)
                method_cfg.checkpoint_dir = f"/tmp/alternate/{method}/checkpoints"
                method_cfg.log_dir = f"/tmp/alternate/{method}/logs"
                method_cfg.num_runs = 9
                method_cfg.run_tasks_tsv = True
                method_cfg.skip_if_exists = True
                method_cfg.tasks_tsv = f"alternate-{method}.tsv"
                self.assertEqual(runner_cls(changed).run_name, baseline)


if __name__ == "__main__":
    unittest.main()
