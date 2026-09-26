"""PromptNodeFormerEncoder tests (Commit 11).

EdgePrompt on NodeFormer requires ``cfg.model.nodeformer.rb_order >= 1``
because the prompt injects into the native relational-bias path.  These
tests cover:

- rb_order=0 raises at construction time (clear error, not a runtime
  shape mismatch).
- Factory build succeeds for rb_order=1.
- Forward output shape.
- Zero-prompt parity: supplying an all-zero prompt must match the
  unprompted forward (unprompted uses the same
  ``_prompted_relational_bias`` helper with a None prompt, so this
  test also exercises the fallback path).
- Gradient flow: prompt params receive grads, encoder (frozen) params
  don't.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch_geometric.data import Data
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.encoders.edgeprompt import PromptAwareEncoder, build_prompt_encoder
from src.finetune.encoders.edgeprompt.nodeformer import PromptNodeFormerEncoder
from src.finetune.prompts.edgeprompt import EdgePromptPlus


def _cfg(rb_order: int = 1, num_layers=2, in_dim=8, hidden=16):
    cfg = CN()
    set_cfg(cfg)
    cfg.model.name = "nodeformer"
    cfg.model.num_layers = num_layers
    cfg.model.in_dim = in_dim
    cfg.model.hidden_dim = hidden
    cfg.model.out_dim = hidden
    cfg.model.dropout = 0.0
    cfg.model.nodeformer.rb_order = rb_order
    cfg.model.nodeformer.heads = 2
    cfg.model.nodeformer.num_random_features = 8
    cfg.model.nodeformer.use_gumbel = False  # deterministic for parity tests
    cfg.model.nodeformer.use_edge_loss = False
    cfg.model.nodeformer.use_layernorm = True
    cfg.model.nodeformer.use_activation = True
    cfg.model.nodeformer.use_residual = True
    cfg.model.nodeformer.use_jk = False
    cfg.model.nodeformer.tau = 1.0
    return cfg


def _batch(n_nodes=6, in_dim=8):
    torch.manual_seed(0)
    x = torch.randn(n_nodes, in_dim)
    edges = torch.tensor(
        [[i, (i + 1) % n_nodes] for i in range(n_nodes)]
        + [[(i + 1) % n_nodes, i] for i in range(n_nodes)],
        dtype=torch.long,
    ).t().contiguous()
    batch = torch.zeros(n_nodes, dtype=torch.long)
    return Data(x=x, edge_index=edges, batch=batch)


class RbOrderGateTest(unittest.TestCase):
    def test_rb_order_zero_raises(self):
        cfg = _cfg(rb_order=0)
        with self.assertRaises(ValueError) as ctx:
            build_prompt_encoder(cfg, in_dim=8)
        self.assertIn("rb_order", str(ctx.exception).lower())

    def test_rb_order_one_builds(self):
        enc = build_prompt_encoder(_cfg(rb_order=1), in_dim=8)
        self.assertIsInstance(enc, PromptNodeFormerEncoder)
        self.assertIsInstance(enc, PromptAwareEncoder)

    def test_rb_order_above_one_requires_adjacency_powers(self):
        with self.assertRaisesRegex(ValueError, "adjacency powers"):
            build_prompt_encoder(_cfg(rb_order=2), in_dim=8)


class ShapeTest(unittest.TestCase):
    def test_shape(self):
        cfg = _cfg()
        enc = build_prompt_encoder(cfg, in_dim=8).eval()
        data = _batch()
        with torch.no_grad():
            node, graph = enc(data)
        self.assertEqual(node.shape, (6, cfg.model.out_dim))
        self.assertEqual(graph.shape, (1, cfg.model.out_dim))


class ZeroPromptParityTest(unittest.TestCase):
    def test_zero_prompt_equals_no_prompt(self):
        cfg = _cfg()
        enc = build_prompt_encoder(cfg, in_dim=8).eval()
        data = _batch()

        dim_list = [cfg.model.hidden_dim] * cfg.model.num_layers

        class _ZeroPrompt(nn.Module):
            def get_prompt(self, x, edge_index, layer):
                # NodeFormer's relational-bias uses the raw edge_index
                # (no self-loops added by PyG), so E rows match.
                if isinstance(edge_index, tuple):
                    n_edges = edge_index[0].size(0)
                else:
                    n_edges = edge_index.size(1)
                return torch.zeros(n_edges, dim_list[layer], device=x.device)

        with torch.no_grad():
            # Deterministic seed: kernel_attention uses torch.randn
            # internally via create_projection_matrix which seeds off
            # query.sum(). Freeze input to keep it reproducible.
            torch.manual_seed(111)
            out_noprompt = enc(data)
            torch.manual_seed(111)
            out_zero = enc(data, prompt=_ZeroPrompt(), prompt_type="EdgePromptplus")
        torch.testing.assert_close(out_noprompt[0], out_zero[0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(out_noprompt[1], out_zero[1], atol=1e-6, rtol=1e-5)


class GradientFlowTest(unittest.TestCase):
    def test_prompt_grads_with_frozen_encoder(self):
        cfg = _cfg()
        enc = build_prompt_encoder(cfg, in_dim=8).train()
        for p in enc.parameters():
            p.requires_grad_(False)

        dim_list = [cfg.model.hidden_dim] * cfg.model.num_layers
        prompt = EdgePromptPlus(dim_list=dim_list, num_anchors=3, add_self_loops=False)
        classifier = nn.Linear(cfg.model.out_dim, 2)
        data = _batch()

        node_repr, _ = enc(data, prompt=prompt, prompt_type="EdgePromptplus")
        classifier(node_repr).sum().backward()

        self.assertTrue(any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in prompt.parameters()
        ))
        for p in enc.parameters():
            self.assertTrue(p.grad is None or p.grad.abs().sum().item() == 0)


if __name__ == "__main__":
    unittest.main()
