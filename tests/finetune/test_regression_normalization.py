from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Data
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.methods.all_in_one import FinetuneAllInOne
from src.finetune.methods.edgeprompt import FinetuneEdgePrompt
from src.finetune.methods.gpf import FinetuneGPF
from src.finetune.methods.gppt import FinetuneGPPT
from src.finetune.methods.graphprompt import FinetuneGraphPrompt
from src.finetune.methods.supervised import FinetuneSupervised
from src.finetune.regression import (
    METRIC_MAE,
    NORMALIZED_MSE,
    RegressionTargetNormalizer,
    resolve_regression_loss,
    resolve_regression_target_normalization,
)
from src.finetune.task_heads import TaskAwareObjective
from src.utils.naming import build_finetune_run_name_from_cfg
from src.utils.supervised_loss import supervised_loss_from_logits


def _regression_cfg(method: str = "supervised", target_dim: int = 2):
    cfg = set_cfg(CN())
    cfg.seed = 42
    cfg.finetune.method = method
    cfg.finetune.dataset.name = "toy_regression"
    cfg.finetune.dataset.task_level = "graph"
    cfg.finetune.dataset.task_level_raw = "graph"
    cfg.finetune.dataset.task_level_effective = "graph"
    cfg.finetune.dataset.induced = False
    cfg.finetune.dataset.task_type = "regression"
    cfg.finetune.dataset.label_dim = target_dim
    cfg.finetune.dataset.num_classes = 1
    cfg.model.name = "gcn"
    cfg.model.in_dim = 4
    cfg.model.hidden_dim = 4
    cfg.model.out_dim = 4
    cfg.model.num_layers = 2
    return cfg


class _RngConsumingRegressionDataset:
    def __init__(self, labels: torch.Tensor):
        self.labels = labels
        self.seen = []

    def __len__(self):
        return int(self.labels.size(0))

    def __getitem__(self, index):
        self.seen.append(index)
        random.random()
        np.random.rand()
        torch.rand(())
        return Data(y=self.labels[index])


class RegressionTargetNormalizerTest(unittest.TestCase):
    def test_fit_preserves_rng_and_loader_generator_order(self):
        labels = torch.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
        )
        dataset = _RngConsumingRegressionDataset(labels)
        generator = torch.Generator().manual_seed(123)
        loader = SimpleNamespace(dataset=dataset, generator=generator)
        normalizer = RegressionTargetNormalizer(
            enabled=True,
            target_dim=2,
            task_level="graph",
        )

        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        generator_state = generator.get_state().clone()

        normalizer.fit(loader)

        self.assertEqual(dataset.seen, [0, 1, 2, 3])
        self.assertEqual(random.getstate(), python_state)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
        self.assertEqual(after_numpy[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
        self.assertTrue(torch.equal(generator.get_state(), generator_state))

    def test_fits_per_target_population_stats(self):
        normalizer = RegressionTargetNormalizer(
            enabled=True,
            target_dim=2,
            task_level="graph",
        )
        normalizer.fit([Data(y=torch.tensor([[1.0, 10.0], [3.0, 30.0]]))])

        torch.testing.assert_close(normalizer.mean, torch.tensor([2.0, 20.0]))
        torch.testing.assert_close(normalizer.std, torch.tensor([1.0, 10.0]))
        torch.testing.assert_close(normalizer.count, torch.tensor([2, 2]))
        self.assertTrue(normalizer.active)

    def test_node_stats_use_train_mask_only(self):
        data = Data(
            x=torch.zeros(4, 1),
            y=torch.tensor([1.0, 3.0, 101.0, 203.0]),
            train_mask=torch.tensor([True, True, False, False]),
        )
        normalizer = RegressionTargetNormalizer(
            enabled=True,
            target_dim=1,
            task_level="node",
        )
        normalizer.fit([data])

        torch.testing.assert_close(normalizer.mean, torch.tensor([2.0]))
        torch.testing.assert_close(normalizer.std, torch.tensor([1.0]))
        torch.testing.assert_close(normalizer.count, torch.tensor([2]))

    def test_objective_optimizes_normalized_targets_but_reports_original_scale(self):
        cfg = _regression_cfg(target_dim=2)
        objective = TaskAwareObjective(cfg, task_level="graph", repr_dim=4)
        labels = torch.tensor([[10.0, 100.0], [14.0, 200.0]])
        objective.target_normalizer.fit([Data(y=labels)])
        normalized_predictions = torch.tensor(
            [[0.0, 1.0], [-1.0, 0.0]], requires_grad=True
        )

        loss, mae, outputs, returned_labels = objective.loss_from_logits(
            logits=normalized_predictions,
            labels=labels,
            return_outputs=True,
        )

        self.assertAlmostEqual(float(loss), 2.5)
        self.assertAlmostEqual(mae, 39.0)
        torch.testing.assert_close(outputs, torch.tensor([12.0, 200.0, 10.0, 150.0]))
        torch.testing.assert_close(returned_labels, labels.view(-1))
        loss.backward()
        self.assertIsNotNone(normalized_predictions.grad)

    def test_metric_mae_is_proportional_to_original_unit_flattened_mae(self):
        cfg = _regression_cfg(target_dim=2)
        cfg.finetune.regression_loss = METRIC_MAE
        objective = TaskAwareObjective(cfg, task_level="graph", repr_dim=4)
        labels = torch.tensor([[10.0, 100.0], [14.0, 200.0]])
        objective.target_normalizer.fit([Data(y=labels)])
        normalized_predictions = torch.tensor(
            [[0.0, 1.0], [-1.0, 0.0]], requires_grad=True
        )

        loss, original_mae, outputs, returned_labels = objective.loss_from_logits(
            logits=normalized_predictions,
            labels=labels,
            return_outputs=True,
        )

        train_std_mean = float(objective.target_normalizer.std.mean().item())
        self.assertAlmostEqual(float(loss), original_mae / train_std_mean, places=6)
        self.assertAlmostEqual(float(loss), 1.5, places=6)
        self.assertAlmostEqual(original_mae, 39.0, places=6)
        torch.testing.assert_close(
            outputs, torch.tensor([12.0, 200.0, 10.0, 150.0])
        )
        torch.testing.assert_close(returned_labels, labels.view(-1))

    def test_metric_mae_gradient_tracks_tenfold_train_standard_deviation(self):
        cfg = _regression_cfg(target_dim=2)
        cfg.finetune.regression_loss = METRIC_MAE
        objective = TaskAwareObjective(cfg, task_level="graph", repr_dim=4)
        objective.target_normalizer.fit(
            [Data(y=torch.tensor([[-1.0, -10.0], [1.0, 10.0]]))]
        )
        predictions = torch.ones(1, 2, requires_grad=True)

        loss, _mae = objective.loss_from_logits(
            logits=predictions,
            labels=torch.zeros(1, 2),
        )
        loss.backward()

        self.assertAlmostEqual(float(loss), 1.0, places=6)
        ratio = float(predictions.grad[0, 1] / predictions.grad[0, 0])
        self.assertAlmostEqual(ratio, 10.0, places=6)

    def test_metric_mae_ignores_missing_values_in_loss_and_gradients(self):
        cfg = _regression_cfg(target_dim=2)
        cfg.finetune.regression_loss = METRIC_MAE
        objective = TaskAwareObjective(cfg, task_level="graph", repr_dim=4)
        objective.target_normalizer.fit(
            [Data(y=torch.tensor([[-1.0, -10.0], [1.0, 10.0]]))]
        )
        predictions = torch.tensor(
            [[1.0, 1_000_000.0], [1_000_000.0, 1.0]],
            requires_grad=True,
        )
        labels = torch.tensor([[0.0, float("nan")], [float("nan"), 0.0]])

        loss, original_mae = objective.loss_from_logits(
            logits=predictions,
            labels=labels,
        )
        loss.backward()

        self.assertAlmostEqual(float(loss), 1.0, places=6)
        self.assertAlmostEqual(original_mae, 5.5, places=6)
        torch.testing.assert_close(
            predictions.grad,
            torch.tensor([[1.0 / 11.0, 0.0], [0.0, 10.0 / 11.0]]),
        )

    def test_default_normalized_mse_is_exact_shared_legacy_loss(self):
        cfg = _regression_cfg(target_dim=2)
        objective = TaskAwareObjective(cfg, task_level="graph", repr_dim=4)
        labels = torch.tensor([[10.0, 100.0], [14.0, 200.0]])
        objective.target_normalizer.fit([Data(y=labels)])
        logits = torch.tensor([[0.25, -0.5], [1.25, 0.75]])
        normalized_labels = objective.target_normalizer.normalize_targets(labels)
        expected_loss, _expected_mae = supervised_loss_from_logits(
            logits=logits.view(-1),
            labels=normalized_labels.view(-1),
            task_type="regression",
            return_outputs=False,
        )

        actual_loss, _actual_mae = objective.loss_from_logits(
            logits=logits,
            labels=labels,
        )

        self.assertEqual(objective.regression_loss_mode, NORMALIZED_MSE)
        torch.testing.assert_close(
            actual_loss,
            expected_loss,
            rtol=0.0,
            atol=0.0,
        )

    def test_metric_weights_remain_train_only_after_heldout_evaluation(self):
        cfg = _regression_cfg(target_dim=2)
        cfg.finetune.regression_loss = METRIC_MAE
        objective = TaskAwareObjective(cfg, task_level="graph", repr_dim=4)
        train_labels = torch.tensor([[1.0, 10.0], [3.0, 30.0]])
        objective.target_normalizer.fit([Data(y=train_labels)])
        before = {
            key: value.clone()
            for key, value in objective.target_normalizer.state_dict().items()
        }

        objective.loss_from_logits(
            logits=torch.zeros(1, 2),
            labels=torch.tensor([[1_000_000.0, -1_000_000.0]]),
        )
        objective.loss_from_logits(
            logits=torch.zeros(1, 2),
            labels=torch.tensor([[-7.0, 9.0]]),
        )

        for key, expected in before.items():
            torch.testing.assert_close(
                objective.target_normalizer.state_dict()[key], expected
            )


class RegressionNormalizationRoutingTest(unittest.TestCase):
    def test_global_loss_default_validation_and_normalization_contract(self):
        cfg = _regression_cfg()
        self.assertEqual(resolve_regression_loss(cfg), NORMALIZED_MSE)

        cfg.finetune.regression_loss = "not_a_loss"
        with self.assertRaisesRegex(ValueError, "finetune.regression_loss"):
            TaskAwareObjective(cfg, task_level="graph", repr_dim=4)

        cfg.finetune.regression_loss = METRIC_MAE
        cfg.finetune.normalize_regression_targets = False
        with self.assertRaisesRegex(
            ValueError, "normalize_regression_targets=True"
        ):
            TaskAwareObjective(cfg, task_level="graph", repr_dim=4)

    def test_shared_default_normalization_flag(self):
        cfg = _regression_cfg()
        self.assertTrue(cfg.finetune.normalize_regression_targets)
        self.assertTrue(resolve_regression_target_normalization(cfg))

        cfg.finetune.normalize_regression_targets = False
        self.assertFalse(resolve_regression_target_normalization(cfg))

    def test_all_task_aware_methods_register_the_shared_normalizer(self):
        method_classes = (
            ("all_in_one", FinetuneAllInOne),
            ("edgeprompt", FinetuneEdgePrompt),
            ("gppt", FinetuneGPPT),
            ("graphprompt", FinetuneGraphPrompt),
        )
        for method, task_cls in method_classes:
            with self.subTest(method=method):
                task = task_cls(_regression_cfg(method))
                self.assertIsInstance(task.objective, TaskAwareObjective)
                self.assertTrue(task.objective.target_normalizer.enabled)

    def test_supervised_and_gpf_fit_the_same_registered_path(self):
        labels = Data(y=torch.tensor([[1.0, 10.0], [3.0, 30.0]]))
        supervised = FinetuneSupervised(_regression_cfg("supervised"))
        supervised_summaries = supervised.fit_regression_target_stats([labels])
        self.assertTrue(supervised.objective.target_normalizer.active)
        self.assertEqual(supervised_summaries[0]["mode"], NORMALIZED_MSE)

        gpf = FinetuneGPF(_regression_cfg("gpf"))
        gpf_summaries = gpf.fit_regression_target_stats([labels])
        nested = gpf.supervised_head.objective.target_normalizer
        self.assertTrue(nested.active)
        torch.testing.assert_close(nested.mean, torch.tensor([2.0, 20.0]))
        self.assertEqual(gpf_summaries[0]["finite_train_labels"], 4)

    def test_supervised_metric_mae_objective(self):
        labels = Data(y=torch.tensor([[-1.0, -10.0], [1.0, 10.0]]))
        supervised_cfg = _regression_cfg("supervised")
        supervised_cfg.finetune.regression_loss = METRIC_MAE
        supervised = FinetuneSupervised(supervised_cfg)

        summaries = supervised.fit_regression_target_stats([labels])
        self.assertEqual(supervised.objective.regression_loss_mode, METRIC_MAE)
        self.assertEqual(summaries[0]["mode"], METRIC_MAE)
        loss, _mae = supervised.objective.loss_from_logits(
            logits=torch.ones(1, 2),
            labels=torch.zeros(1, 2),
        )
        self.assertAlmostEqual(float(loss), 1.0, places=6)

    def test_regression_run_names_distinguish_both_normalization_states(self):
        cfg = _regression_cfg("supervised")
        kwargs = {
            "split": (5, 0.0, 1.0),
            "task_level_raw": "graph",
            "task_cls": FinetuneSupervised,
            "finetune_method": "supervised",
            "pretrained_run_name": "pretrained",
            "freeze_pretrained_effective": False,
        }
        enabled_name = build_finetune_run_name_from_cfg(cfg, **kwargs)
        cfg.finetune.normalize_regression_targets = False
        disabled_name = build_finetune_run_name_from_cfg(cfg, **kwargs)
        cfg.finetune.normalize_regression_targets = True
        cfg.finetune.regression_loss = METRIC_MAE
        metric_mae_name = build_finetune_run_name_from_cfg(cfg, **kwargs)

        self.assertIn("regnorm1", enabled_name)
        self.assertIn("regnorm0", disabled_name)
        self.assertNotIn("reglossmetricmae", enabled_name)
        self.assertIn("reglossmetricmae", metric_mae_name)
        self.assertNotEqual(enabled_name, disabled_name)
        self.assertNotEqual(enabled_name, metric_mae_name)


if __name__ == "__main__":
    unittest.main()
