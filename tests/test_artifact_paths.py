from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.finetune.finetuner import FinetuneRunner
from src.utils.checkpoint import _atomic_tmp_path, save_json_atomic
from src.utils.naming import compact_artifact_stem


class ArtifactStemTest(unittest.TestCase):
    def test_short_stem_is_unchanged(self):
        self.assertEqual(compact_artifact_stem("ft_gcn_seed42"), "ft_gcn_seed42")

    def test_long_stem_is_bounded_deterministic_and_collision_safe(self):
        stem_a = "ft_edgeprompt_" + ("context_" * 40) + "seed42"
        stem_b = "ft_edgeprompt_" + ("context_" * 40) + "seed43"
        compact_a = compact_artifact_stem(stem_a)

        self.assertEqual(compact_a, compact_artifact_stem(stem_a))
        self.assertNotEqual(compact_a, compact_artifact_stem(stem_b))
        self.assertLessEqual(len(compact_a.encode("utf-8")), 180)
        self.assertIn("__h", compact_a)
        self.assertTrue(compact_a.endswith("seed42"))

    def test_finetune_checkpoint_and_log_paths_share_compacted_stem(self):
        runner = FinetuneRunner.__new__(FinetuneRunner)
        runner.run_name = "ft_edgeprompt_" + ("variant_" * 40) + "seed42-ctx0123456789ab"
        runner.run_group = "photo"
        runner.run_dir = "/tmp/checkpoints/photo"
        runner.cfg = SimpleNamespace(
            finetune=SimpleNamespace(log_dir="/tmp/logs")
        )

        artifact_stem = runner._artifact_stem()
        self.assertEqual(
            os.path.basename(runner._checkpoint_path()),
            f"{artifact_stem}.pt",
        )
        self.assertEqual(
            os.path.basename(runner._log_path()),
            f"{artifact_stem}_log.json",
        )
        self.assertLess(len(os.path.basename(runner._log_path()).encode("utf-8")), 255)

    def test_existing_legacy_checkpoint_remains_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = FinetuneRunner.__new__(FinetuneRunner)
            runner.run_name = "legacy_" + ("x" * 182)
            runner.run_dir = tmp
            runner._reused_checkpoint_path = None
            runner.cfg = SimpleNamespace(
                finetune=SimpleNamespace(method="supervised")
            )
            legacy_path = runner._legacy_checkpoint_path()
            self.assertIsNotNone(legacy_path)
            Path(legacy_path).write_bytes(b"legacy")

            existing = runner._existing_checkpoint_path()
            runner._reused_checkpoint_path = existing

            self.assertEqual(existing, legacy_path)
            self.assertEqual(runner.get_checkpoint_path_for_metrics(), legacy_path)


class AtomicTempPathTest(unittest.TestCase):
    def test_long_final_component_uses_short_atomic_tmp_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / (("x" * 238) + ".json")
            tmp_path = _atomic_tmp_path(str(path))
            self.assertLess(len(os.path.basename(tmp_path).encode("utf-8")), 255)

            save_json_atomic(str(path), {"ok": True})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"ok": True})
            self.assertEqual(list(path.parent.glob(".atomic-*.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
