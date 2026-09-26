"""Identity-mask parity tests for GraphPrompt+ adapters.

Gating test for the Phase 1 refactor: GraphPrompt+ used to inline a
manual ``GNNEncoder`` forward replay inside
``_forward_with_stage_prompt``.  That body now lives in
``GNNStackAdapter.forward_with_stage_prompt``.  The refactor must not
change numerics — this test enforces it by checking that, with all
stage masks set to 1.0 (identity) in ``eval()`` mode, the adapter
output equals the base encoder output for every supported stack
backbone (gcn/gin/gat/mlp) under both ``repr_source`` modes.

The test is independent of any cfg validation or full task
construction; it builds the adapter, prompt module, and ``GNNEncoder``
directly so a parity failure points unambiguously at the adapter
forward path.
"""

from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Data

from src.finetune.encoders.graphprompt_plus.gnn import GNNStackAdapter
from src.finetune.prompts import GraphPromptPlusStageWise
from src.model.encoder import GNNEncoder


def _make_data(num_nodes: int = 12, num_edges: int = 30, in_dim: int = 8, seed: int = 0) -> Data:
    """Reproducible toy graph used by every parity case."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, in_dim, generator=g)
    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=g)
    edge_index = torch.stack([src, dst], dim=0)
    return Data(x=x, edge_index=edge_index)


def _make_encoder(model_type: str, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> GNNEncoder:
    torch.manual_seed(42)
    return GNNEncoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=num_layers,
        model_type=model_type,
        act="relu",
        dropout=0.0,  # belt-and-braces; eval() makes this irrelevant anyway
        graph_pooling="mean",
        gat_heads=2,
        use_batchnorm=False,
    )


def _make_identity_prompt(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, p_num: int = 4) -> GraphPromptPlusStageWise:
    # init="ones" + init_std=0.0 sets every stage mask exactly to 1.0,
    # so apply_stage(i, x) is the identity for every stage.
    return GraphPromptPlusStageWise(
        in_channels=in_dim,
        hidden_channels=hidden_dim,
        out_channels=out_dim,
        num_layers=num_layers,
        p_num=p_num,
        init="ones",
        init_std=0.0,
    )


class IdentityMaskParityTest(unittest.TestCase):
    """With identity stage masks, every ``stage_id`` must reproduce the
    base ``GNNEncoder`` output.  This locks the manual-replay logic
    inside ``GNNStackAdapter`` to the encoder's own forward."""

    BACKBONES = ("gcn", "gin", "gat", "mlp")
    IN_DIM = 8
    HIDDEN_DIM = 16
    OUT_DIM = 16
    NUM_LAYERS = 3  # enough for stages 0/1/2/3 to all be active

    def _run_one(self, model_type: str, repr_source: str) -> None:
        data = _make_data(in_dim=self.IN_DIM)
        encoder = _make_encoder(
            model_type=model_type,
            in_dim=self.IN_DIM,
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM,
            num_layers=self.NUM_LAYERS,
        )
        encoder.eval()
        prompt = _make_identity_prompt(
            in_dim=self.IN_DIM,
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM if repr_source == "last"
            else self.HIDDEN_DIM * (self.NUM_LAYERS - 1) + self.OUT_DIM,
            num_layers=self.NUM_LAYERS,
        )
        prompt.eval()

        with torch.no_grad():
            base_node_repr, _ = encoder(data)
            if repr_source == "layer_concat":
                base_node_repr = torch.cat(encoder.get_layer_node_reprs(), dim=-1)

            for stage_id in prompt.active_stage_ids:
                adapter_node, adapter_graph = GNNStackAdapter.forward_with_stage_prompt(
                    model=encoder,
                    data=data,
                    stage_id=stage_id,
                    prompt=prompt,
                    repr_source=repr_source,
                )
                self.assertEqual(
                    adapter_node.shape, base_node_repr.shape,
                    f"shape mismatch on {model_type}/{repr_source}/stage{stage_id}",
                )
                self.assertTrue(
                    torch.allclose(adapter_node, base_node_repr, atol=1e-6, rtol=1e-5),
                    f"identity-mask parity failed on {model_type}/{repr_source}/stage{stage_id}",
                )
                # No batch in the test data → graph_repr must be None to
                # match the pre-refactor return contract.
                self.assertIsNone(adapter_graph)

    def test_parity_last(self):
        for backbone in self.BACKBONES:
            with self.subTest(backbone=backbone, repr_source="last"):
                self._run_one(backbone, "last")

    def test_parity_layer_concat(self):
        for backbone in self.BACKBONES:
            with self.subTest(backbone=backbone, repr_source="layer_concat"):
                self._run_one(backbone, "layer_concat")


class GraphPoolReturnedWhenBatchPresent(unittest.TestCase):
    """When ``data.batch`` is present, the adapter must return the
    encoder-pooled graph repr instead of ``None``.  The pre-refactor
    code did this; verify the adapter preserves the contract."""

    def test_graph_repr_returned_with_batch(self):
        in_dim, hidden, out, num_layers = 8, 16, 16, 2
        data = _make_data(num_nodes=10, num_edges=20, in_dim=in_dim)
        # Two graphs of 5 nodes each.
        data.batch = torch.tensor([0] * 5 + [1] * 5, dtype=torch.long)

        encoder = _make_encoder("gcn", in_dim, hidden, out, num_layers)
        encoder.eval()
        prompt = _make_identity_prompt(in_dim, hidden, out, num_layers, p_num=4)
        prompt.eval()

        with torch.no_grad():
            for stage_id in prompt.active_stage_ids:
                _, graph_repr = GNNStackAdapter.forward_with_stage_prompt(
                    model=encoder,
                    data=data,
                    stage_id=stage_id,
                    prompt=prompt,
                    repr_source="last",
                )
                self.assertIsNotNone(graph_repr, f"graph_repr None at stage {stage_id}")
                self.assertEqual(graph_repr.shape, (2, out))


class StageSpecsTest(unittest.TestCase):
    """``iter_stage_specs`` must reflect the available stages for a
    given encoder shape and match the prompt module's hardcoded stage
    layout."""

    def test_stage_specs_match_prompt_module(self):
        # 3-layer encoder: all four stages available.
        specs = GNNStackAdapter.iter_stage_specs(
            num_layers=3, in_dim=8, hidden_dim=16, out_dim=32, repr_dim=32,
        )
        self.assertEqual(specs, [(0, 8), (1, 16), (2, 16), (3, 32)])

    def test_stage_specs_drop_stage_two_for_two_layer(self):
        specs = GNNStackAdapter.iter_stage_specs(
            num_layers=2, in_dim=8, hidden_dim=16, out_dim=32, repr_dim=32,
        )
        self.assertEqual(specs, [(0, 8), (1, 16), (3, 32)])

    def test_stage_specs_single_layer(self):
        specs = GNNStackAdapter.iter_stage_specs(
            num_layers=1, in_dim=8, hidden_dim=16, out_dim=32, repr_dim=32,
        )
        self.assertEqual(specs, [(0, 8), (3, 32)])


class SupportsModelTest(unittest.TestCase):
    def test_accepts_standard_gnn_encoder(self):
        encoder = _make_encoder("gcn", 8, 16, 16, 2)
        self.assertTrue(GNNStackAdapter.supports_model(encoder))

    def test_rejects_object_missing_required_attrs(self):
        class FakeEncoder:
            pass

        self.assertFalse(GNNStackAdapter.supports_model(FakeEncoder()))


if __name__ == "__main__":
    unittest.main()
