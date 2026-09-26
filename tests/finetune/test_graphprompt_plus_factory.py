"""Factory dispatch tests for GraphPrompt+ adapters.

Mirrors ``tests/finetune/test_edgeprompt_factory.py``.

- Every Phase-1 supported backbone has a ``GraphPromptPlusSpec`` entry.
- ``build_graphprompt_plus_adapter`` returns a ``GraphPromptPlusAdapter``
  subclass (not an instance — adapters are stateless classmethods).
- Unknown model names raise ``ValueError`` from the spec, since the
  spec registry is authoritative for "what GraphPrompt+ supports".
"""

from __future__ import annotations

import unittest

from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.encoders.graphprompt_plus import (
    GraphPromptPlusAdapter,
    build_graphprompt_plus_adapter,
    resolve_graphprompt_plus_spec,
    supported_graphprompt_plus_backbones,
)


def _cfg(model_name: str, num_layers: int = 3):
    cfg = CN()
    set_cfg(cfg)
    cfg.model.name = model_name
    cfg.model.num_layers = num_layers
    cfg.model.in_dim = 16
    cfg.model.hidden_dim = 32
    cfg.model.out_dim = 32
    return cfg


class FactoryTest(unittest.TestCase):
    def test_supported_backbones_complete(self):
        # Phase 1 covers gcn/gin/gat/mlp via the shared GNN-stack
        # adapter; Phases 2-5 add transformer / nodeformer / fagcn /
        # h2gcn.
        self.assertEqual(
            set(supported_graphprompt_plus_backbones()),
            {"gcn", "gin", "gat", "mlp", "transformer", "nodeformer", "fagcn", "h2gcn"},
        )

    def test_spec_resolves_for_every_supported_backbone(self):
        for name in supported_graphprompt_plus_backbones():
            spec = resolve_graphprompt_plus_spec(_cfg(name))
            self.assertEqual(spec.model_name, name)
            self.assertIn(spec.support, {"official", "extension"})
            self.assertTrue(spec.formula)

    def test_gcn_gin_are_official(self):
        # The original GraphPrompt+ paper formulates stage prompts on
        # GCN/GIN; GAT and MLP are repo extensions.
        for name in ("gcn", "gin"):
            self.assertEqual(resolve_graphprompt_plus_spec(_cfg(name)).support, "official")
        for name in ("gat", "mlp"):
            self.assertEqual(resolve_graphprompt_plus_spec(_cfg(name)).support, "extension")

    def test_factory_returns_adapter_subclass(self):
        for name in supported_graphprompt_plus_backbones():
            adapter = build_graphprompt_plus_adapter(_cfg(name))
            self.assertTrue(
                isinstance(adapter, type) and issubclass(adapter, GraphPromptPlusAdapter),
                f"adapter for {name} must be a GraphPromptPlusAdapter subclass",
            )

    def test_unknown_model_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_graphprompt_plus_adapter(_cfg("not_a_real_model"))

    def test_layer_concat_supported_only_for_gnn_stack(self):
        # Phase-1 adapters drive the standard GNNEncoder, which exposes
        # a per-layer cache; layer_concat is meaningful there.
        # Phase 2-5 backbones (transformer/nodeformer/fagcn/h2gcn) do
        # not have an encoder-side per-layer cache and intentionally
        # refuse layer_concat to keep the parity contract tight.
        for name in ("gcn", "gin", "gat", "mlp"):
            spec = resolve_graphprompt_plus_spec(_cfg(name))
            self.assertTrue(spec.supports_layer_concat, f"{name} should support layer_concat")
        for name in ("transformer", "nodeformer", "fagcn", "h2gcn"):
            spec = resolve_graphprompt_plus_spec(_cfg(name))
            self.assertFalse(spec.supports_layer_concat, f"{name} should refuse layer_concat")

    def test_adapter_class_does_not_carry_support_tier(self):
        # Per-class support tier is wrong when one adapter serves
        # multiple backbones (gcn = official vs gat = extension).  The
        # spec is the single source of truth; the adapter base class
        # must not redeclare a class-level ``support`` attribute that
        # could drift.
        from src.finetune.encoders.graphprompt_plus.base import GraphPromptPlusAdapter
        from src.finetune.encoders.graphprompt_plus.gnn import GNNStackAdapter

        self.assertFalse(
            hasattr(GraphPromptPlusAdapter, "support"),
            "GraphPromptPlusAdapter must not declare a class-level "
            "support attribute; per-backbone support lives in spec.",
        )
        self.assertFalse(
            hasattr(GNNStackAdapter, "support"),
            "GNNStackAdapter must not declare a class-level support "
            "attribute; spec is authoritative.",
        )


class ValidateCfgEarlyResolution(unittest.TestCase):
    """``validate_cfg`` must reject unsupported backbones when
    ``graphprompt.plus=True`` so the runner's skip-if-exists path
    cannot silently absorb an invalid run."""

    def _full_cfg(self, model_name: str, plus: bool) -> CN:
        cfg = _cfg(model_name)
        cfg.finetune.dataset.task_level = "node"
        cfg.finetune.dataset.induced = True
        cfg.finetune.graphprompt.plus = plus
        return cfg

    def test_validate_cfg_rejects_unsupported_backbone_with_plus_true(self):
        from src.finetune.methods.graphprompt import FinetuneGraphPrompt

        cfg = self._full_cfg("not_a_real_model", plus=True)
        with self.assertRaises(ValueError) as ctx:
            FinetuneGraphPrompt.validate_cfg(cfg)
        self.assertIn("not_a_real_model", str(ctx.exception))

    def test_validate_cfg_accepts_unsupported_backbone_with_plus_false(self):
        # Non-plus GraphPrompt is backbone-agnostic; validate_cfg must
        # not raise even when the model has no plus adapter.
        from src.finetune.methods.graphprompt import FinetuneGraphPrompt

        cfg = self._full_cfg("not_a_real_model", plus=False)
        FinetuneGraphPrompt.validate_cfg(cfg)  # no raise

    def test_validate_cfg_accepts_supported_backbone_with_plus_true(self):
        from src.finetune.methods.graphprompt import FinetuneGraphPrompt

        cfg = self._full_cfg("gcn", plus=True)
        FinetuneGraphPrompt.validate_cfg(cfg)  # no raise


if __name__ == "__main__":
    unittest.main()
