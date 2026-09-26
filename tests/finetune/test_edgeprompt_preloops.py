"""Regression test for the GAT/H2GCN/FAGCN self-loop alignment bug.

Before the P1.1 fix, inputs that already contained self-loops produced
mismatched prompt/conv edge counts: ``EdgePromptPlus.get_prompt`` called
``add_self_loops`` without removing existing ones (-> E + N + existing
loops rows), while the prompt-aware conv called ``remove_self_loops``
first then ``add_self_loops`` (-> E + N - existing loops rows or E + N
depending on the conv).  The mismatch silently raised a shape error on
any graph with duplicate/pre-existing self-loops.

This test constructs an input edge_index that explicitly includes
self-loops and asserts every supported "must add loops" backbone runs
end-to-end with its resolved replace-or-append self-loop policy.
"""

from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Data
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.encoders.edgeprompt import build_prompt_encoder
from src.finetune.encoders.edgeprompt.spec import resolve_edgeprompt_prompt_spec
from src.finetune.prompts.edgeprompt import EdgePromptPlus


def _cfg(model_name: str, num_layers: int = 2):
    cfg = CN()
    set_cfg(cfg)
    cfg.model.name = model_name
    cfg.model.num_layers = num_layers
    cfg.model.in_dim = 8
    cfg.model.hidden_dim = 16
    cfg.model.out_dim = 16
    cfg.model.dropout = 0.0
    cfg.model.use_batchnorm = False
    # NodeFormer is not in this test suite (no self-loops; its own test
    # covers the rb_order gate).  GAT + FAGCN pick up their backbone
    # configs here for clean defaults.
    cfg.model.gat.heads = 2
    cfg.model.fagcn.eps = 0.1
    return cfg


def _batch_with_preexisting_self_loops(n_nodes: int = 6, in_dim: int = 8):
    """Cycle graph in both directions plus an explicit self-loop on node 0.

    Most prompt-aware convs replace existing loops; H2GCN mirrors its vanilla
    append policy. The prompt must use the same per-backbone ordering.
    """
    torch.manual_seed(0)
    x = torch.randn(n_nodes, in_dim)
    edges = []
    for i in range(n_nodes):
        edges.append([i, (i + 1) % n_nodes])
        edges.append([(i + 1) % n_nodes, i])
    edges.append([0, 0])  # pre-existing self-loop on node 0
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    batch = torch.zeros(n_nodes, dtype=torch.long)
    return Data(x=x, edge_index=ei, batch=batch)


class PreExistingSelfLoopsTest(unittest.TestCase):
    """Each "must add self-loops" backbone must run end-to-end with a
    graph that already has self-loops."""

    def _run(self, model_name: str, dim_list):
        cfg = _cfg(model_name)
        enc = build_prompt_encoder(cfg, in_dim=cfg.model.in_dim).eval()
        spec = resolve_edgeprompt_prompt_spec(cfg)
        prompt = EdgePromptPlus(
            dim_list=list(dim_list),
            num_anchors=3,
            add_self_loops=True,
            replace_self_loops=spec.replace_self_loops,
        ).eval()
        data = _batch_with_preexisting_self_loops()
        with torch.no_grad():
            node, graph = enc(data, prompt=prompt, prompt_type="EdgePromptplus")
        self.assertEqual(node.shape, (6, cfg.model.out_dim))
        self.assertEqual(graph.shape, (1, cfg.model.out_dim))

    def test_gcn_with_preexisting_loops(self):
        self._run("gcn", [8, 16])

    def test_gat_with_preexisting_loops(self):
        self._run("gat", [8, 16])

    def test_h2gcn_with_preexisting_loops(self):
        self._run("h2gcn", [8, 16])

    def test_fagcn_with_preexisting_loops(self):
        self._run("fagcn", [16, 16])


if __name__ == "__main__":
    unittest.main()
