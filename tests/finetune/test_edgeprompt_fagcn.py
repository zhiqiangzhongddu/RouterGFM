"""PromptFAConv / PromptFAGCNEncoder tests (Commit 10)."""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops as add_sl
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.encoders.edgeprompt import PromptAwareEncoder, build_prompt_encoder
from src.finetune.encoders.edgeprompt.fagcn import PromptFAConv, PromptFAGCNEncoder
from src.finetune.prompts.edgeprompt import EdgePromptPlus


def _cfg(num_layers=2, in_dim=8, hidden=16):
    cfg = CN()
    set_cfg(cfg)
    cfg.model.name = "fagcn"
    cfg.model.num_layers = num_layers
    cfg.model.in_dim = in_dim
    cfg.model.hidden_dim = hidden
    cfg.model.out_dim = hidden
    cfg.model.dropout = 0.0
    cfg.model.fagcn.eps = 0.1
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


class FactoryDispatchTest(unittest.TestCase):
    def test_factory_builds(self):
        enc = build_prompt_encoder(_cfg(), in_dim=8)
        self.assertIsInstance(enc, PromptFAGCNEncoder)
        self.assertIsInstance(enc, PromptAwareEncoder)
        self.assertEqual(enc.edgeprompt_support, "extension")


class ShapeTest(unittest.TestCase):
    def test_shape(self):
        cfg = _cfg()
        enc = build_prompt_encoder(cfg, in_dim=8).eval()
        data = _batch()
        with torch.no_grad():
            node, graph = enc(data)
        self.assertEqual(node.shape, (6, cfg.model.out_dim))
        self.assertEqual(graph.shape, (1, cfg.model.out_dim))


class KeyMappingTest(unittest.TestCase):
    def test_conv_keys_match_pyg_faconv(self):
        from torch_geometric.nn import FAConv

        ref = FAConv(channels=16, eps=0.1)
        ours = PromptFAConv(channels=16, eps=0.1)
        self.assertEqual(
            sorted(ours.state_dict().keys()),
            sorted(ref.state_dict().keys()),
        )


class VanillaParityTest(unittest.TestCase):
    """PromptFAConv without a prompt must match PyG's FAConv exactly on
    identical weights.  Caught the gate-direction swap during self-review."""

    def test_matches_pyg_faconv_no_prompt(self):
        from torch_geometric.nn import FAConv

        torch.manual_seed(0)
        ref = FAConv(channels=4, eps=0.1)
        ours = PromptFAConv(channels=4, eps=0.1)
        ours.load_state_dict(ref.state_dict())

        x = torch.randn(5, 4)
        x0 = x.clone()
        ei = torch.tensor(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long
        )
        ref.eval()
        ours.eval()
        with torch.no_grad():
            a = ref(x, x0, ei)
            b = ours(x, x0, ei)
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


class ZeroPromptParityTest(unittest.TestCase):
    def test_zero_prompt_equals_no_prompt(self):
        cfg = _cfg()
        enc = build_prompt_encoder(cfg, in_dim=8).eval()
        data = _batch()

        dim_list = [cfg.model.hidden_dim] * cfg.model.num_layers

        class _ZeroPrompt(nn.Module):
            def get_prompt(self, x, edge_index, layer):
                # gcn_norm adds self-loops (E+N rows).  EdgePromptPlus
                # with add_self_loops=True mirrors this ordering.
                edge_with_self, _ = add_sl(edge_index, num_nodes=x.size(0))
                return torch.zeros(edge_with_self.size(1), dim_list[layer], device=x.device)

        with torch.no_grad():
            out_noprompt = enc(data)
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
        prompt = EdgePromptPlus(dim_list=dim_list, num_anchors=3, add_self_loops=True)
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
