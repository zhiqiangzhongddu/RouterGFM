"""PromptGATConv / PromptGATEncoder tests (Commit 6)."""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch_geometric.data import Data
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.encoders.edgeprompt import PromptAwareEncoder, build_prompt_encoder
from src.finetune.encoders.edgeprompt.gat import PromptGATConv, PromptGATEncoder
from src.finetune.prompts.edgeprompt import EdgePrompt, EdgePromptPlus


def _cfg(heads=2, num_layers=2, in_dim=8, hidden=16):
    cfg = CN()
    set_cfg(cfg)
    cfg.model.name = "gat"
    cfg.model.num_layers = num_layers
    cfg.model.in_dim = in_dim
    cfg.model.hidden_dim = hidden
    cfg.model.out_dim = hidden
    cfg.model.use_batchnorm = False
    cfg.model.gat.heads = heads
    return cfg


def _batch(n_nodes=8, in_dim=8):
    torch.manual_seed(0)
    x = torch.randn(n_nodes, in_dim)
    edges = torch.tensor(
        [[i, (i + 1) % n_nodes] for i in range(n_nodes)]
        + [[(i + 1) % n_nodes, i] for i in range(n_nodes)],
        dtype=torch.long,
    ).t().contiguous()
    batch = torch.zeros(n_nodes, dtype=torch.long)
    return Data(x=x, edge_index=edges, batch=batch)


class FactoryDispatchTest(unittest.TestCase):
    def test_factory_builds_gat_encoder(self):
        enc = build_prompt_encoder(_cfg(), in_dim=8)
        self.assertIsInstance(enc, PromptGATEncoder)
        self.assertIsInstance(enc, PromptAwareEncoder)
        self.assertEqual(enc.edgeprompt_support, "extension")


class ForwardShapeTest(unittest.TestCase):
    def test_shape_without_prompt(self):
        cfg = _cfg(heads=4)
        enc = build_prompt_encoder(cfg, in_dim=8).eval()
        data = _batch(n_nodes=6, in_dim=8)
        with torch.no_grad():
            node_repr, graph_repr = enc(data)
        self.assertEqual(node_repr.shape, (6, cfg.model.out_dim))
        self.assertEqual(graph_repr.shape, (1, cfg.model.out_dim))

    def test_shape_with_edge_prompt(self):
        cfg = _cfg(heads=2, num_layers=2)
        enc = build_prompt_encoder(cfg, in_dim=8).eval()
        dim_list = [cfg.model.in_dim] + [cfg.model.hidden_dim] * (cfg.model.num_layers - 1)
        # PromptGATConv adds self-loops internally (matches PyG GATConv
        # default); the prompt must also add self-loops so its output has
        # E+N rows aligned with the expanded edge_index.
        prompt = EdgePromptPlus(dim_list=dim_list, num_anchors=5, add_self_loops=True).eval()
        data = _batch(n_nodes=6, in_dim=8)
        with torch.no_grad():
            node_repr, graph_repr = enc(data, prompt=prompt, prompt_type="EdgePromptplus")
        self.assertEqual(node_repr.shape, (6, cfg.model.out_dim))


class ZeroPromptParityTest(unittest.TestCase):
    """Zero-valued prompt must reproduce the unprompted forward pass."""

    def test_zero_prompt_equals_no_prompt(self):
        from torch_geometric.utils import add_self_loops as add_sl, remove_self_loops

        cfg = _cfg()
        enc = build_prompt_encoder(cfg, in_dim=8).eval()
        data = _batch(n_nodes=6, in_dim=8)

        dim_list = [cfg.model.in_dim] + [cfg.model.hidden_dim] * (cfg.model.num_layers - 1)

        class _ZeroPrompt(nn.Module):
            def get_prompt(self, x, edge_index, layer):
                # PromptGATConv does remove_self_loops + add_self_loops; the
                # zero-prompt stub must produce an E+N row tensor aligned with
                # the same expansion.
                ei_no_loops, _ = remove_self_loops(edge_index)
                edge_with_self, _ = add_sl(ei_no_loops, num_nodes=x.size(0))
                return torch.zeros(edge_with_self.size(1), dim_list[layer], device=x.device)

        with torch.no_grad():
            out_noprompt = enc(data)
            out_zero = enc(data, prompt=_ZeroPrompt(), prompt_type="EdgePromptplus")

        torch.testing.assert_close(out_noprompt[0], out_zero[0], atol=0, rtol=0)
        torch.testing.assert_close(out_noprompt[1], out_zero[1], atol=0, rtol=0)


class VanillaParityTest(unittest.TestCase):
    """PromptGATConv with no prompt must match PyG's GATConv exactly on
    identical weights.  This test caught the self-loop omission bug
    during self-review."""

    def test_matches_pyg_gatconv_no_prompt(self):
        from torch_geometric.nn import GATConv

        torch.manual_seed(0)
        ref = GATConv(4, 8, heads=2, concat=False)
        ours = PromptGATConv(4, 8, heads=2, concat=False)
        ours.load_state_dict(ref.state_dict())

        x = torch.randn(5, 4)
        ei = torch.tensor(
            [[0, 1, 2, 3, 4, 1, 2, 3, 4, 0],
             [1, 2, 3, 4, 0, 0, 1, 2, 3, 4]],
            dtype=torch.long,
        )
        ref.eval()
        ours.eval()
        with torch.no_grad():
            a = ref(x, ei)
            b = ours(x, ei)
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


class GradientFlowTest(unittest.TestCase):
    def test_prompt_grads_flow_encoder_frozen(self):
        cfg = _cfg()
        enc = build_prompt_encoder(cfg, in_dim=8).train()
        # Freeze encoder.
        for p in enc.parameters():
            p.requires_grad_(False)

        dim_list = [cfg.model.in_dim] + [cfg.model.hidden_dim] * (cfg.model.num_layers - 1)
        prompt = EdgePromptPlus(dim_list=dim_list, num_anchors=3, add_self_loops=True)
        classifier = nn.Linear(cfg.model.out_dim, 2)
        data = _batch(n_nodes=6, in_dim=8)

        node_repr, _ = enc(data, prompt=prompt, prompt_type="EdgePromptplus")
        logits = classifier(node_repr)
        loss = logits.sum()
        loss.backward()

        # Prompt grads must be non-trivial.
        any_prompt_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in prompt.parameters()
        )
        self.assertTrue(any_prompt_grad)
        # Classifier grads too.
        any_cls_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in classifier.parameters()
        )
        self.assertTrue(any_cls_grad)
        # Encoder is frozen: all grads should be None or zero.
        for p in enc.parameters():
            self.assertTrue(p.grad is None or p.grad.abs().sum().item() == 0)


class KeyMappingTest(unittest.TestCase):
    """State-dict key names should mirror PyG's ``GATConv(heads, concat=False)``."""

    def test_conv_state_dict_keys_match_pyg_gatconv(self):
        from torch_geometric.nn import GATConv

        ref = GATConv(8, 16, heads=2, concat=False)
        ours = PromptGATConv(8, 16, heads=2, concat=False)
        self.assertEqual(
            sorted(ours.state_dict().keys()),
            sorted(ref.state_dict().keys()),
        )


if __name__ == "__main__":
    unittest.main()
