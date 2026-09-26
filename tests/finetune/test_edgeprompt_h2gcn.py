"""PromptH2GCNEncoder tests (Commit 7)."""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch_geometric.data import Data
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.encoders.edgeprompt import PromptAwareEncoder, build_prompt_encoder
from src.finetune.encoders.edgeprompt.h2gcn import PromptH2GCNEncoder
from src.finetune.prompts.edgeprompt import EdgePromptPlus
from src.model.h2gcn import H2GCNEncoder


def _cfg(in_dim=8, hidden=16, use_bn=False):
    cfg = CN()
    set_cfg(cfg)
    cfg.model.name = "h2gcn"
    cfg.model.num_layers = 2  # h2gcn dim schedule is 2 regardless
    cfg.model.in_dim = in_dim
    cfg.model.hidden_dim = hidden
    cfg.model.out_dim = hidden
    cfg.model.use_batchnorm = use_bn
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
        self.assertIsInstance(enc, PromptH2GCNEncoder)
        self.assertIsInstance(enc, PromptAwareEncoder)


class ShapeTest(unittest.TestCase):
    def test_shape(self):
        cfg = _cfg()
        enc = build_prompt_encoder(cfg, in_dim=8).eval()
        data = _batch(n_nodes=6, in_dim=8)
        with torch.no_grad():
            node, graph = enc(data)
        self.assertEqual(node.shape, (6, cfg.model.out_dim))
        self.assertEqual(graph.shape, (1, cfg.model.out_dim))


class StateDictMatchVanillaTest(unittest.TestCase):
    """State-dict keys must match the vanilla H2GCNEncoder exactly, so
    pretrain weights from the vanilla backbone can be loaded via
    ``pretrained_key_whitelist`` without renaming."""

    def test_keys_match(self):
        vanilla = H2GCNEncoder(in_dim=8, hidden_dim=16, out_dim=16)
        prompt_enc = PromptH2GCNEncoder(in_dim=8, hidden_dim=16, out_dim=16)
        self.assertEqual(
            sorted(vanilla.state_dict().keys()),
            sorted(prompt_enc.state_dict().keys()),
        )

    def test_vanilla_checkpoint_loads_fully(self):
        vanilla = H2GCNEncoder(in_dim=8, hidden_dim=16, out_dim=16)
        prompt_enc = PromptH2GCNEncoder(in_dim=8, hidden_dim=16, out_dim=16)
        missing, unexpected = prompt_enc.load_state_dict(
            vanilla.state_dict(), strict=False
        )
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])

    def test_unprompted_parity_with_preexisting_self_loop(self):
        vanilla = H2GCNEncoder(
            in_dim=8, hidden_dim=16, out_dim=16, dropout=0.0
        ).eval()
        prompt_enc = PromptH2GCNEncoder(
            in_dim=8, hidden_dim=16, out_dim=16, dropout=0.0
        ).eval()
        prompt_enc.load_state_dict(vanilla.state_dict())
        data = _batch(n_nodes=6, in_dim=8)
        data.edge_index = torch.cat(
            [data.edge_index, torch.tensor([[0], [0]])], dim=1
        )

        with torch.no_grad():
            expected = vanilla(data)
            actual = prompt_enc(data)

        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


class ZeroPromptParityTest(unittest.TestCase):
    """Zero-valued prompt must reproduce the unprompted forward exactly."""

    def test_zero_prompt_equals_no_prompt(self):
        from torch_geometric.utils import add_self_loops as add_sl

        cfg = _cfg()
        enc = build_prompt_encoder(cfg, in_dim=8).eval()
        data = _batch(n_nodes=6, in_dim=8)

        # Prompt dims match the H2GCN dim schedule.
        # layer 0 sees in_dim, layer 1 sees hidden_dim.
        dim_list = (cfg.model.in_dim, cfg.model.hidden_dim)

        class _ZeroPrompt(nn.Module):
            """Reproduces EdgePromptPlus's add-self-loops behavior with
            zero values, so prompt rows match the encoder's
            ``edge_with_self`` ordering."""

            def get_prompt(self, x, edge_index, layer):
                edge_with_self, _ = add_sl(edge_index, num_nodes=x.size(0))
                return torch.zeros(edge_with_self.size(1), dim_list[layer], device=x.device)

        with torch.no_grad():
            out_noprompt = enc(data)
            out_zero = enc(data, prompt=_ZeroPrompt(), prompt_type="EdgePromptplus")
        torch.testing.assert_close(out_noprompt[0], out_zero[0], atol=0, rtol=0)
        torch.testing.assert_close(out_noprompt[1], out_zero[1], atol=0, rtol=0)


class GradientFlowTest(unittest.TestCase):
    def test_prompt_grads_with_frozen_encoder(self):
        cfg = _cfg()
        enc = build_prompt_encoder(cfg, in_dim=8).train()
        for p in enc.parameters():
            p.requires_grad_(False)

        dim_list = [cfg.model.in_dim, cfg.model.hidden_dim]
        prompt = EdgePromptPlus(dim_list=dim_list, num_anchors=3, add_self_loops=True)
        classifier = nn.Linear(cfg.model.out_dim, 2)
        data = _batch(n_nodes=6, in_dim=8)

        node_repr, _ = enc(data, prompt=prompt, prompt_type="EdgePromptplus")
        logits = classifier(node_repr)
        logits.sum().backward()

        self.assertTrue(any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in prompt.parameters()
        ))
        for p in enc.parameters():
            self.assertTrue(p.grad is None or p.grad.abs().sum().item() == 0)


if __name__ == "__main__":
    unittest.main()
