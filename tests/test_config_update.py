import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from src.config import cfg, update_cfg
from src.train.run import run_train_from_cli


class UpdateCfgConfigFileTest(unittest.TestCase):
    def test_missing_explicit_config_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "Config file not found"):
            update_cfg(cfg, ["--config", "/definitely/missing/icg-config.yaml"])

    def test_train_cli_reports_missing_config_without_traceback(self):
        stderr = StringIO()
        with redirect_stderr(stderr):
            status = run_train_from_cli(
                ["--config", "/definitely/missing/icg-config.yaml"]
            )

        self.assertEqual(status, 1)
        self.assertEqual(
            stderr.getvalue().strip(),
            "Config file not found: /definitely/missing/icg-config.yaml",
        )

    def test_existing_config_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("device: 7\n", encoding="utf-8")

            resolved = update_cfg(cfg, ["--config", str(path)])

        self.assertEqual(resolved.device, 7)


if __name__ == "__main__":
    unittest.main()
