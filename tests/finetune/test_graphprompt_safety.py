from __future__ import annotations

import unittest

import torch
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.methods.graphprompt import FinetuneGraphPrompt


def graphprompt_cfg() -> CN:
    cfg = set_cfg(CN())
    cfg.finetune.method = "graphprompt"
    cfg.finetune.dataset.task_level = "graph"
    cfg.finetune.dataset.task_level_effective = "graph"
    cfg.finetune.dataset.induced = False
    cfg.finetune.dataset.task_type = "classification"
    cfg.finetune.dataset.label_dim = 1
    cfg.finetune.dataset.num_classes = 3
    cfg.model.in_dim = 4
    cfg.model.hidden_dim = 6
    cfg.model.out_dim = 5
    cfg.model.num_layers = 2
    cfg.finetune.graphprompt.plus = False
    cfg.finetune.graphprompt.repr_source = "last"
    cfg.finetune.graphprompt.eval_center_mode = "train"
    return cfg


class GraphPromptSafetyTest(unittest.TestCase):
    def test_batch_eval_centers_are_rejected_before_training(self) -> None:
        cfg = graphprompt_cfg()
        cfg.finetune.graphprompt.eval_center_mode = "batch"
        with self.assertRaisesRegex(ValueError, "validation/test labels"):
            FinetuneGraphPrompt.validate_cfg(cfg)

    def test_layer_concat_rejected_for_encoder_without_cache(self) -> None:
        cfg = graphprompt_cfg()
        cfg.model.name = "nodeformer"
        cfg.finetune.graphprompt.repr_source = "layer_concat"
        with self.assertRaisesRegex(ValueError, "per-layer representation cache"):
            FinetuneGraphPrompt.validate_cfg(cfg)

    def test_centers_survive_task_state_dict_round_trip(self) -> None:
        cfg = graphprompt_cfg()
        source = FinetuneGraphPrompt(cfg)
        latest = torch.randn(source.num_classes, source.repr_dim)
        bank = torch.randn(source.num_classes, source.repr_dim)
        source._store_latest_centers(latest)
        source._store_prototype_bank(bank)
        restored = FinetuneGraphPrompt(cfg)
        restored.load_state_dict(source.state_dict())
        self.assertTrue(restored._has_latest_centers())
        self.assertTrue(restored._has_prototype_bank())
        torch.testing.assert_close(restored.latest_centers, latest)
        torch.testing.assert_close(restored.prototype_bank, bank)


if __name__ == "__main__":
    unittest.main()
