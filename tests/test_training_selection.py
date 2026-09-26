from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.train.trainer import TrainRunner


class TinyTask(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))


def make_runner(tmp_path: Path, val_values: list[float]) -> tuple[TrainRunner, list[float]]:
    cfg = set_cfg(CN())
    cfg.device = 0
    cfg.train.epochs = len(val_values)
    cfg.train.early_stopping = 0
    cfg.train.skip_if_exists = False
    cfg.train.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.train.log_dir = str(tmp_path / "logs")
    cfg.train.dataset.task_type = "classification"
    runner = TrainRunner.__new__(TrainRunner)
    runner.cfg = cfg
    runner.device = torch.device("cpu")
    runner.run_name = "selection"
    runner.run_group = "dataset-graph"
    runner.run_dir = str(tmp_path / "checkpoints" / runner.run_group)
    runner.dataset_meta = {"name": "dataset"}
    runner.model = torch.nn.Linear(1, 1, bias=False)
    runner.task = TinyTask()
    runner.optimizer = torch.optim.SGD(
        list(runner.model.parameters()) + list(runner.task.parameters()), lr=0.1
    )
    runner.scheduler = None
    runner.monitor_name = "val_acc"
    runner.monitor_mode = "max"
    runner.best_metric = float("-inf")
    runner.best_epoch = None
    runner.best_metrics = {}
    runner.train_history = []
    runner._checkpoint_written_this_run = False
    runner._skip_due_to_existing_checkpoint = False
    runner.test_loader = object()
    runner.val_loader = object()
    runner._setup = lambda: None
    epoch = {"value": 0}
    observed_test_weights: list[float] = []

    def train_epoch():
        epoch["value"] += 1
        with torch.no_grad():
            runner.model.weight.fill_(float(epoch["value"]))
        return float(epoch["value"]), {"train_acc": 0.5}

    def evaluate_split(_loader, prefix: str, mask_attr: str):
        del mask_attr
        if prefix == "val":
            return {"val_acc": val_values[epoch["value"] - 1]}
        observed_test_weights.append(float(runner.model.weight.item()))
        return {"test_acc": 0.75}

    runner.train_epoch = train_epoch
    runner._evaluate_split = evaluate_split
    return runner, observed_test_weights


class TrainingSelectionTest(unittest.TestCase):
    def test_test_once_after_restoring_selected_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner, observed = make_runner(Path(tmp), [0.9, 0.8, 0.7])
            runner.fit()
            self.assertEqual(observed, [1.0])
            payload = torch.load(runner._checkpoint_path(), map_location="cpu")
            self.assertEqual(payload["epoch"], 1)
            self.assertEqual(payload["metrics"]["test_acc"], 0.75)
            self.assertEqual(float(payload["model_state"]["weight"].item()), 1.0)
            with open(runner._log_path(), "r", encoding="utf-8") as handle:
                log = json.load(handle)
            self.assertEqual([e["epoch"] for e in log["history"]], [1, 2, 3])

    def test_forced_rerun_overwrites_stale_checkpoint_on_nonfinite_monitor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner, observed = make_runner(Path(tmp), [float("nan"), float("nan")])
            Path(runner.run_dir).mkdir(parents=True)
            with torch.no_grad():
                runner.model.weight.fill_(99.0)
            torch.save(
                {
                    "epoch": 99,
                    "model_state": runner.model.state_dict(),
                    "optimizer_state": runner.optimizer.state_dict(),
                    "cfg": {},
                    "dataset": {},
                    "metrics": {"test_acc": 1.0},
                    "extra": {"train_task_state": runner.task.state_dict()},
                },
                runner._checkpoint_path(),
            )
            runner.fit()
            self.assertEqual(observed, [2.0])
            payload = torch.load(runner._checkpoint_path(), map_location="cpu")
            self.assertEqual(payload["epoch"], 2)
            self.assertEqual(float(payload["model_state"]["weight"].item()), 2.0)
            self.assertTrue(payload["extra"].get("fallback_save"))


if __name__ == "__main__":
    unittest.main()
