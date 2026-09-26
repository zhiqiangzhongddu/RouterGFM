"""Identity-mask parity tests for the Phase 2-5 GraphPrompt+ adapters.

Same gating contract as ``test_graphprompt_plus_parity.py``: with all
stage masks set to 1.0 in ``eval()``, the adapter's
``forward_with_stage_prompt`` for any active stage_id must reproduce
the base encoder's forward output to within ``1e-5`` (slightly looser
than Phase 1 because Transformer/NodeFormer attention has more
floating-point noise; still well below any meaningful drift).

Each adapter has its own test class so failures point at the right
backbone.  Constructing real ``Transformer`` / ``NodeFormer`` etc.
encoders with cfg ensures the adapter sees what the runner would
build.
"""

from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Data
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.encoders.graphprompt_plus.fagcn import FAGCNStackAdapter
from src.finetune.encoders.graphprompt_plus.h2gcn import H2GCNAdapter
from src.finetune.encoders.graphprompt_plus.nodeformer import NodeFormerStackAdapter
from src.finetune.encoders.graphprompt_plus.transformer import TransformerStackAdapter
from src.finetune.prompts import GraphPromptPlusStageWise
from src.model.encoder import build_encoder_from_cfg


ATOL = 1e-5
RTOL = 1e-4


def _make_data(num_nodes: int = 12, num_edges: int = 30, in_dim: int = 8, seed: int = 0) -> Data:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, in_dim, generator=g)
    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=g)
    edge_index = torch.stack([src, dst], dim=0)
    return Data(x=x, edge_index=edge_index)


def _cfg(model_name: str, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
    cfg = CN()
    set_cfg(cfg)
    cfg.model.name = model_name
    cfg.model.num_layers = num_layers
    cfg.model.in_dim = in_dim
    cfg.model.hidden_dim = hidden_dim
    cfg.model.out_dim = out_dim
    cfg.model.dropout = 0.0
    cfg.model.activation = "relu"
    cfg.model.graph_pooling = "mean"
    return cfg


def _identity_prompt(stage_specs, p_num=4) -> GraphPromptPlusStageWise:
    # in/hidden/out fallback args don't matter when stage_specs is given
    # because the prompt module sizes its masks from stage_specs only.
    return GraphPromptPlusStageWise(
        in_channels=1,
        hidden_channels=1,
        out_channels=1,
        num_layers=len(stage_specs),
        p_num=p_num,
        init="ones",
        init_std=0.0,
        stage_specs=stage_specs,
    )


class _ParityHarness:
    """Mixin with the shared parity-check logic."""

    ADAPTER = None  # set by subclass
    MODEL_NAME = None
    IN_DIM = 8
    HIDDEN_DIM = 16
    OUT_DIM = 16
    NUM_LAYERS = 3

    def _build(self):
        cfg = _cfg(
            self.MODEL_NAME,
            in_dim=self.IN_DIM,
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM,
            num_layers=self.NUM_LAYERS,
        )
        torch.manual_seed(123)
        encoder = build_encoder_from_cfg(cfg, in_dim=self.IN_DIM)
        encoder.eval()
        data = _make_data(in_dim=self.IN_DIM)
        return cfg, encoder, data

    def _check_parity(self, encoder, data, prompt):
        with torch.no_grad():
            base_node_repr, _ = encoder(data)
            for stage_id in prompt.active_stage_ids:
                adapter_node, adapter_graph = self.ADAPTER.forward_with_stage_prompt(
                    model=encoder,
                    data=data,
                    stage_id=stage_id,
                    prompt=prompt,
                    repr_source="last",
                )
                self.assertEqual(
                    adapter_node.shape, base_node_repr.shape,
                    f"shape mismatch on {self.MODEL_NAME}/stage{stage_id}",
                )
                if not torch.allclose(adapter_node, base_node_repr, atol=ATOL, rtol=RTOL):
                    diff = (adapter_node - base_node_repr).abs().max().item()
                    self.fail(
                        f"identity-mask parity failed on {self.MODEL_NAME}/stage{stage_id}: "
                        f"max abs diff = {diff:.3e}"
                    )
                self.assertIsNone(adapter_graph)


class TransformerParityTest(_ParityHarness, unittest.TestCase):
    ADAPTER = TransformerStackAdapter
    MODEL_NAME = "transformer"

    def test_identity_parity(self):
        _, encoder, data = self._build()
        specs = self.ADAPTER.iter_stage_specs(
            num_layers=self.NUM_LAYERS,
            in_dim=self.IN_DIM,
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM,
            repr_dim=self.OUT_DIM,
        )
        prompt = _identity_prompt(specs)
        prompt.eval()
        self._check_parity(encoder, data, prompt)

    def test_layer_concat_rejected(self):
        _, encoder, data = self._build()
        specs = self.ADAPTER.iter_stage_specs(
            num_layers=self.NUM_LAYERS,
            in_dim=self.IN_DIM,
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM,
            repr_dim=self.OUT_DIM,
        )
        prompt = _identity_prompt(specs)
        prompt.eval()
        with self.assertRaises(ValueError):
            self.ADAPTER.forward_with_stage_prompt(
                model=encoder, data=data, stage_id=0,
                prompt=prompt, repr_source="layer_concat",
            )


class FAGCNParityTest(_ParityHarness, unittest.TestCase):
    ADAPTER = FAGCNStackAdapter
    MODEL_NAME = "fagcn"

    def test_identity_parity(self):
        _, encoder, data = self._build()
        specs = self.ADAPTER.iter_stage_specs(
            num_layers=self.NUM_LAYERS,
            in_dim=self.IN_DIM,
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM,
            repr_dim=self.OUT_DIM,
        )
        prompt = _identity_prompt(specs)
        prompt.eval()
        self._check_parity(encoder, data, prompt)


class H2GCNParityTest(_ParityHarness, unittest.TestCase):
    ADAPTER = H2GCNAdapter
    MODEL_NAME = "h2gcn"
    NUM_LAYERS = 2  # H2GCNEncoder ignores num_layers; spec is fixed

    def test_identity_parity(self):
        _, encoder, data = self._build()
        specs = self.ADAPTER.iter_stage_specs(
            num_layers=self.NUM_LAYERS,
            in_dim=self.IN_DIM,
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM,
            repr_dim=self.OUT_DIM,
        )
        # H2GCN has only stages {0, 1, 3} — exactly 3 stages.
        self.assertEqual([sid for sid, _ in specs], [0, 1, 3])
        prompt = _identity_prompt(specs)
        prompt.eval()
        self._check_parity(encoder, data, prompt)


class NodeFormerParityTest(_ParityHarness, unittest.TestCase):
    ADAPTER = NodeFormerStackAdapter
    MODEL_NAME = "nodeformer"
    NUM_LAYERS = 2  # NodeFormer's gumbel-softmax kernel is sensitive at deeper depths

    def test_identity_parity(self):
        cfg = _cfg(
            self.MODEL_NAME,
            in_dim=self.IN_DIM,
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM,
            num_layers=self.NUM_LAYERS,
        )
        # NodeFormer's projection_matrix uses a per-call torch.manual_seed
        # tied to the data, so eval()-mode + same data + same encoder
        # produces the same projection on both calls.  Disable the stochastic
        # gumbel path so eval mode is deterministic.
        cfg.model.nodeformer.use_gumbel = False
        cfg.model.nodeformer.use_edge_loss = False
        torch.manual_seed(123)
        encoder = build_encoder_from_cfg(cfg, in_dim=self.IN_DIM)
        encoder.eval()
        data = _make_data(in_dim=self.IN_DIM)
        specs = self.ADAPTER.iter_stage_specs(
            num_layers=self.NUM_LAYERS,
            in_dim=self.IN_DIM,
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM,
            repr_dim=self.OUT_DIM,
        )
        prompt = _identity_prompt(specs)
        prompt.eval()
        self._check_parity(encoder, data, prompt)


class StageSpecsTest(unittest.TestCase):
    def test_h2gcn_omits_stage_two(self):
        specs = H2GCNAdapter.iter_stage_specs(
            num_layers=2, in_dim=8, hidden_dim=16, out_dim=32, repr_dim=32,
        )
        self.assertEqual([sid for sid, _ in specs], [0, 1, 3])

    def test_transformer_full_layout_with_three_layers(self):
        specs = TransformerStackAdapter.iter_stage_specs(
            num_layers=3, in_dim=8, hidden_dim=16, out_dim=32, repr_dim=32,
        )
        self.assertEqual(specs, [(0, 8), (1, 16), (2, 16), (3, 32)])


class FactoryRegistrationTest(unittest.TestCase):
    def test_all_phase2_5_backbones_registered(self):
        from src.finetune.encoders.graphprompt_plus import (
            GRAPHPROMPT_PLUS_ADAPTERS,
            supported_graphprompt_plus_backbones,
        )

        expected = {"gcn", "gin", "gat", "mlp", "transformer", "nodeformer", "fagcn", "h2gcn"}
        self.assertEqual(set(supported_graphprompt_plus_backbones()), expected)
        self.assertTrue(expected.issubset(set(GRAPHPROMPT_PLUS_ADAPTERS.keys())))


if __name__ == "__main__":
    unittest.main()
