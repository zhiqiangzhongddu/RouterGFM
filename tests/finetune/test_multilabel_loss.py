from __future__ import annotations

import math
import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.methods.all_in_one import FinetuneAllInOne
from src.finetune.methods.gpf import FinetuneGPF
from src.finetune.methods.supervised import FinetuneSupervised
from src.finetune.multilabel import MacroBalancedBCELoss
from src.finetune.task_heads import TaskAwareObjective
from src.utils.naming import build_finetune_run_name_from_cfg
from src.utils.supervised_loss import supervised_loss_from_logits


def _multilabel_cfg(method: str = "supervised", target_dim: int = 4):
    cfg = set_cfg(CN())
    cfg.seed = 42
    cfg.finetune.method = method
    cfg.finetune.dataset.name = "toy_multilabel"
    cfg.finetune.dataset.task_level = "graph"
    cfg.finetune.dataset.task_level_raw = "graph"
    cfg.finetune.dataset.task_level_effective = "graph"
    cfg.finetune.dataset.induced = False
    cfg.finetune.dataset.task_type = "classification"
    cfg.finetune.dataset.label_dim = target_dim
    cfg.finetune.dataset.num_classes = 2
    cfg.model.name = "gcn"
    cfg.model.in_dim = 4
    cfg.model.hidden_dim = 4
    cfg.model.out_dim = 4
    cfg.model.num_layers = 2
    return cfg


class _RngConsumingDataset:
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


class MacroBalancedBCELossTest(unittest.TestCase):
    def test_matches_manual_macro_task_and_class_balance(self):
        labels = torch.tensor(
            [
                [1.0, 0.0, float("nan"), 1.0],
                [0.0, 0.0, float("nan"), 1.0],
                [0.0, 1.0, float("nan"), float("nan")],
                [0.0, float("nan"), float("nan"), float("nan")],
            ]
        )
        logits = torch.tensor(
            [
                [0.3, -0.2, 9.0, 1.0],
                [-0.7, 0.4, -9.0, 1.0],
                [0.1, 0.8, 5.0, 1.0],
                [-0.5, 3.0, -5.0, 1.0],
            ],
            requires_grad=True,
        )
        loss_fn = MacroBalancedBCELoss(
            enabled=True,
            target_dim=4,
            task_level="graph",
        )
        loss_fn.fit([Data(y=labels)])

        task0 = (
            F.softplus(-logits[0, 0]) / 2.0
            + F.softplus(logits[1:, 0]).sum() / 6.0
        )
        task1 = (
            F.softplus(-logits[2, 1]) / 2.0
            + F.softplus(logits[:2, 1]).sum() / 4.0
        )
        expected = (task0 + task1) / 2.0

        actual = loss_fn(logits, labels)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(loss_fn.positive_count, torch.tensor([1, 1, 0, 2]))
        torch.testing.assert_close(loss_fn.negative_count, torch.tensor([3, 2, 0, 0]))
        torch.testing.assert_close(
            loss_fn.eligible, torch.tensor([True, True, False, False])
        )
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_signed_zero_is_missing_and_nan_is_ignored(self):
        labels = torch.tensor(
            [
                [1.0, -1.0, 0.0, float("nan")],
                [-1.0, 1.0, 0.0, float("nan")],
            ]
        )
        loss_fn = MacroBalancedBCELoss(
            enabled=True,
            target_dim=4,
            task_level="graph",
        )
        loss_fn.fit([Data(y=labels)])

        self.assertTrue(loss_fn.summary()["uses_signed_labels"])
        torch.testing.assert_close(loss_fn.positive_count, torch.tensor([1, 1, 0, 0]))
        torch.testing.assert_close(loss_fn.negative_count, torch.tensor([1, 1, 0, 0]))
        loss = loss_fn(torch.zeros_like(labels), labels)
        self.assertAlmostEqual(float(loss), math.log(2.0), places=6)

    def test_all_degenerate_targets_return_differentiable_zero(self):
        labels = torch.tensor(
            [
                [1.0, 0.0, float("nan")],
                [1.0, 0.0, float("nan")],
            ]
        )
        logits = torch.randn(2, 3, requires_grad=True)
        loss_fn = MacroBalancedBCELoss(
            enabled=True,
            target_dim=3,
            task_level="graph",
        )
        loss_fn.fit([Data(y=labels)])

        loss = loss_fn(logits, labels)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_equal_minibatch_estimators_average_to_full_loss(self):
        labels = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 1.0],
            ]
        )
        logits = torch.tensor(
            [[0.2, -0.3], [-0.6, 0.7], [0.4, -0.8], [0.9, 0.1]]
        )
        loss_fn = MacroBalancedBCELoss(
            enabled=True,
            target_dim=2,
            task_level="graph",
        )
        loss_fn.fit([Data(y=labels)])

        full = loss_fn(logits, labels)
        first = loss_fn(logits[:2], labels[:2])
        second = loss_fn(logits[2:], labels[2:])
        torch.testing.assert_close((first + second) / 2.0, full)

    def test_fit_preserves_rng_and_loader_generator_order(self):
        labels = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
        )
        dataset = _RngConsumingDataset(labels)
        generator = torch.Generator().manual_seed(123)
        loader = SimpleNamespace(dataset=dataset, generator=generator)
        loss_fn = MacroBalancedBCELoss(
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

        loss_fn.fit(loader)

        self.assertEqual(dataset.seen, [0, 1, 2, 3])
        self.assertEqual(random.getstate(), python_state)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
        self.assertEqual(after_numpy[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
        self.assertTrue(torch.equal(generator.get_state(), generator_state))

    def test_node_counts_use_train_mask_only(self):
        data = Data(
            x=torch.zeros(4, 1),
            y=torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
            ),
            train_mask=torch.tensor([True, True, False, False]),
        )
        loss_fn = MacroBalancedBCELoss(
            enabled=True,
            target_dim=2,
            task_level="node",
        )
        loss_fn.fit([data])

        torch.testing.assert_close(loss_fn.positive_count, torch.tensor([1, 1]))
        torch.testing.assert_close(loss_fn.negative_count, torch.tensor([1, 1]))
        self.assertEqual(int(loss_fn.train_count.item()), 2)

    def test_registered_counts_round_trip_through_state_dict(self):
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        source = MacroBalancedBCELoss(
            enabled=True,
            target_dim=2,
            task_level="graph",
        )
        source.fit([Data(y=labels)])
        restored = MacroBalancedBCELoss(
            enabled=True,
            target_dim=2,
            task_level="graph",
        )
        restored.load_state_dict(source.state_dict())

        self.assertTrue(restored.active)
        self.assertEqual(restored.summary(), source.summary())


class MultilabelObjectiveRoutingTest(unittest.TestCase):
    def test_macro_primary_metric_uses_fitted_signed_mask_per_minibatch(self):
        cfg = _multilabel_cfg(target_dim=2)
        cfg.finetune.multilabel_loss = "macro_balanced_bce"
        objective = TaskAwareObjective(cfg, task_level="graph", repr_dim=4)
        objective.multilabel_balancer.fit(
            [Data(y=torch.tensor([[-1.0, 1.0], [1.0, -1.0]]))]
        )

        labels = torch.tensor([[0.0, 1.0]])
        logits = torch.zeros(1, 2, requires_grad=True)
        loss, acc, returned_logits, returned_labels = objective.loss_from_logits(
            logits=logits,
            labels=labels,
            return_outputs=True,
        )

        self.assertAlmostEqual(acc, 1.0)
        torch.testing.assert_close(returned_logits, logits)
        torch.testing.assert_close(returned_labels, labels)
        loss.backward()
        self.assertEqual(float(logits.grad[0, 0]), 0.0)
        self.assertNotEqual(float(logits.grad[0, 1]), 0.0)

    def test_default_masked_bce_is_exact_legacy_path_without_fit(self):
        cfg = _multilabel_cfg(target_dim=3)
        objective = TaskAwareObjective(cfg, task_level="graph", repr_dim=4)
        logits = torch.tensor([[0.2, -0.3, 0.7], [-0.4, 0.8, -0.1]])
        labels = torch.tensor(
            [[1.0, 0.0, float("nan")], [0.0, 1.0, 1.0]]
        )

        expected = supervised_loss_from_logits(
            logits=logits,
            labels=labels,
            task_type="classification",
            return_outputs=True,
        )
        actual = objective.loss_from_logits(
            logits=logits,
            labels=labels,
            return_outputs=True,
        )

        self.assertFalse(objective.multilabel_balancer.ready.item())
        self.assertEqual(actual[1], expected[1])
        for actual_tensor, expected_tensor in zip(
            (actual[0], actual[2], actual[3]),
            (expected[0], expected[2], expected[3]),
        ):
            torch.testing.assert_close(
                actual_tensor,
                expected_tensor,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )

    def test_supervised_gpf_and_all_in_one_share_fitted_objective(self):
        labels = Data(y=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        tasks = (
            FinetuneSupervised(_macro_cfg("supervised", target_dim=2)),
            FinetuneGPF(_macro_cfg("gpf", target_dim=2)),
            FinetuneAllInOne(_macro_cfg("all_in_one", target_dim=2)),
        )
        for task in tasks:
            with self.subTest(method=task.name):
                summaries = task.fit_multilabel_target_stats([labels])
                balancers = [
                    module
                    for module in task.modules()
                    if isinstance(module, MacroBalancedBCELoss) and module.enabled
                ]
                self.assertEqual(len(balancers), 1)
                self.assertTrue(balancers[0].active)
                self.assertEqual(summaries[0]["eligible_targets"], 2)

    def test_nondefault_run_identity_and_invalid_mode(self):
        cfg = _multilabel_cfg("supervised", target_dim=2)
        kwargs = {
            "split": (5, 0.0, 1.0),
            "task_level_raw": "graph",
            "task_cls": FinetuneSupervised,
            "finetune_method": "supervised",
            "pretrained_run_name": "pretrained",
            "freeze_pretrained_effective": False,
        }
        default_name = build_finetune_run_name_from_cfg(cfg, **kwargs)
        cfg.finetune.multilabel_loss = "macro_balanced_bce"
        macro_name = build_finetune_run_name_from_cfg(cfg, **kwargs)

        self.assertNotIn("mlbce", default_name)
        self.assertIn("mlbce", macro_name)
        self.assertNotEqual(default_name, macro_name)

        cfg.finetune.multilabel_loss = "not_a_loss"
        with self.assertRaisesRegex(ValueError, "finetune.multilabel_loss"):
            TaskAwareObjective(cfg, task_level="graph", repr_dim=4)


def _macro_cfg(method: str, target_dim: int):
    cfg = _multilabel_cfg(method, target_dim)
    cfg.finetune.multilabel_loss = "macro_balanced_bce"
    return cfg


if __name__ == "__main__":
    unittest.main()
