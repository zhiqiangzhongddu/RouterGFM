"""Factory dispatch tests (Phase 2 / Commit 5).

- Every supported backbone has a ``EdgePromptSpec`` entry.
- ``build_prompt_encoder`` returns a ``PromptAwareEncoder`` subclass
  for backbones whose builder is registered (currently gcn/gin).
- Backbones declared supported but not yet built (gat, transformer,
  h2gcn, fagcn, nodeformer) raise ``NotImplementedError`` with
  a clear message.
- ``mlp`` and unknown names raise ``ValueError`` from the spec.
"""

from __future__ import annotations

import unittest

import torch
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.encoders.edgeprompt import (
    PromptAwareEncoder,
    build_prompt_encoder,
    resolve_edgeprompt_prompt_spec,
    supported_edgeprompt_backbones,
)
from src.model.encoder import GNNEncoder


def _cfg(model_name: str, num_layers: int = 2):
    cfg = CN()
    set_cfg(cfg)
    cfg.model.name = model_name
    cfg.model.num_layers = num_layers
    cfg.model.in_dim = 16
    cfg.model.hidden_dim = 32
    cfg.model.out_dim = 32
    return cfg


class FactoryTest(unittest.TestCase):
    def test_supported_backbones_list_is_complete(self):
        expected = {
            "gcn", "gin", "gat", "transformer",
            "h2gcn", "fagcn", "nodeformer",
        }
        self.assertEqual(set(supported_edgeprompt_backbones()), expected)

    def test_spec_resolves_for_every_supported_backbone(self):
        for name in supported_edgeprompt_backbones():
            cfg = _cfg(name)
            spec = resolve_edgeprompt_prompt_spec(cfg)
            self.assertEqual(spec.model_name, name)
            self.assertIn(spec.support, {"official", "extension"})
            self.assertGreater(len(spec.dim_list), 0)
            self.assertTrue(spec.formula)

    def test_gcn_builder_returns_prompt_aware(self):
        enc = build_prompt_encoder(_cfg("gcn"), in_dim=16)
        self.assertIsInstance(enc, PromptAwareEncoder)
        self.assertEqual(enc.edgeprompt_support, "official")

    def test_gin_builder_returns_prompt_aware(self):
        enc = build_prompt_encoder(_cfg("gin"), in_dim=16)
        self.assertIsInstance(enc, PromptAwareEncoder)
        self.assertEqual(enc.edgeprompt_support, "official")

    def test_gin_frozen_state_parity_with_fixed_eps_and_prelu(self):
        cfg = _cfg("gin")
        cfg.model.activation = "prelu"
        cfg.finetune.edgeprompt.gin_message_relu = False
        cfg.finetune.edgeprompt.gin_train_eps = False
        vanilla = GNNEncoder(
            in_dim=cfg.model.in_dim,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            num_layers=cfg.model.num_layers,
            model_type="gin",
            act="prelu",
            dropout=cfg.model.dropout,
            graph_pooling=cfg.model.graph_pooling,
            use_batchnorm=cfg.model.use_batchnorm,
        )
        prompted = build_prompt_encoder(cfg, in_dim=cfg.model.in_dim)

        self.assertEqual(set(prompted.state_dict()), set(vanilla.state_dict()))
        missing, unexpected = prompted.load_state_dict(
            vanilla.state_dict(), strict=False
        )
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        activation_ids = {
            id(prompted.act.weight),
            id(prompted.convs[0].nn[1].weight),
            id(prompted.convs[1].nn[1].weight),
        }
        self.assertEqual(len(activation_ids), 3)
        self.assertTrue(torch.equal(prompted.convs[0].eps, torch.zeros(1)))

    def test_pending_backbones_raise_not_implemented(self):
        # Only backbones whose builder has not yet landed should raise.
        # As each per-backbone commit registers a builder, it moves out of
        # this set.
        from src.finetune.encoders.edgeprompt.factory import PROMPT_ENCODERS

        pending = {"gat", "transformer", "h2gcn", "fagcn", "nodeformer"} - PROMPT_ENCODERS.keys()
        if not pending:
            self.skipTest("all declared backbones have builders")
        for name in pending:
            cfg = _cfg(name)
            with self.assertRaises(NotImplementedError) as ctx:
                build_prompt_encoder(cfg, in_dim=16)
            # Error message should list at least one backbone that does work.
            self.assertTrue("gcn" in str(ctx.exception) or "gin" in str(ctx.exception))

    def test_mlp_raises_value_error(self):
        cfg = _cfg("mlp")
        with self.assertRaises(ValueError) as ctx:
            build_prompt_encoder(cfg, in_dim=16)
        msg = str(ctx.exception)
        self.assertIn("mlp", msg)
        # Message should mention at least a couple of supported GNN backbones
        # so the user knows what to try instead.
        for name in ("gcn", "gin"):
            self.assertIn(name, msg)

    def test_unknown_model_raises_value_error(self):
        cfg = _cfg("not_a_real_model")
        with self.assertRaises(ValueError):
            build_prompt_encoder(cfg, in_dim=16)

    def test_dim_schedule_tables(self):
        # Spec table correctness: mirrors the README.
        cfg = _cfg("gcn", num_layers=3)
        self.assertEqual(
            resolve_edgeprompt_prompt_spec(cfg).dim_list,
            (16, 32, 32),
        )
        cfg = _cfg("h2gcn", num_layers=2)
        self.assertEqual(
            resolve_edgeprompt_prompt_spec(cfg).dim_list,
            (16, 32),
        )

    def test_add_self_loops_defaults(self):
        # Encoders that add self-loops inside their aggregation must have
        # the prompt add them too, so prompt and encoder agree on the
        # [E+N] edge ordering.
        must_add_loops = {"gcn", "gat", "h2gcn", "fagcn"}
        for name in must_add_loops:
            self.assertTrue(
                resolve_edgeprompt_prompt_spec(_cfg(name)).add_self_loops,
                f"{name} should default to add_self_loops=True",
            )
        for name in ("gin", "transformer", "nodeformer"):
            self.assertFalse(
                resolve_edgeprompt_prompt_spec(_cfg(name)).add_self_loops,
                f"{name} should default to add_self_loops=False",
            )
        self.assertFalse(
            resolve_edgeprompt_prompt_spec(_cfg("h2gcn")).replace_self_loops
        )


if __name__ == "__main__":
    unittest.main()
