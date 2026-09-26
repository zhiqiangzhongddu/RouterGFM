"""EdgePrompt ``variant_tag`` lock test (Phase 1 / Commit 4).

``variant_tag`` feeds into checkpoint filenames via
``build_finetune_run_name_from_cfg``.  Any drift renames every existing
EdgePrompt checkpoint and breaks ``skip_if_exists``, effectively
invalidating months of finetune runs.  This test pins the exact tag
strings for common configs so accidental changes trip CI.

The cases below were chosen to cover:
- all defaults (``anchorsauto-loopsauto``)
- explicit num_anchors
- explicit add_self_loops (True/False)
- non-default node subgraph knobs (h/mn/mx tags appear)
- use_official_node_subgraphs=False (subgraph knobs suppressed)
"""

from __future__ import annotations

import unittest

from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.methods.edgeprompt import FinetuneEdgePrompt


def _fresh_cfg():
    cfg = CN()
    set_cfg(cfg)
    return cfg


def _cfg(
    *,
    num_anchors=None,
    add_self_loops=None,
    use_official_node_subgraphs=True,
    node_subgraph_hops=None,
    node_subgraph_min_size=None,
    node_subgraph_max_size=None,
):
    cfg = _fresh_cfg()
    ep = cfg.finetune.edgeprompt
    ep.num_anchors = num_anchors
    ep.add_self_loops = add_self_loops
    ep.use_official_node_subgraphs = use_official_node_subgraphs
    if node_subgraph_hops is not None:
        ep.node_subgraph_hops = node_subgraph_hops
    if node_subgraph_min_size is not None:
        ep.node_subgraph_min_size = node_subgraph_min_size
    if node_subgraph_max_size is not None:
        ep.node_subgraph_max_size = node_subgraph_max_size
    return cfg


class VariantTagLockTest(unittest.TestCase):
    def test_all_defaults_emits_anchorsauto_loopsauto(self):
        cfg = _cfg()
        self.assertEqual(FinetuneEdgePrompt.variant_tag(cfg), "anchorsauto-loopsauto")

    def test_explicit_num_anchors_renders_integer(self):
        cfg = _cfg(num_anchors=5)
        self.assertEqual(FinetuneEdgePrompt.variant_tag(cfg), "anchors5-loopsauto")

    def test_explicit_add_self_loops_true(self):
        cfg = _cfg(add_self_loops=True)
        self.assertEqual(FinetuneEdgePrompt.variant_tag(cfg), "anchorsauto-loops1")

    def test_explicit_add_self_loops_false(self):
        cfg = _cfg(add_self_loops=False)
        self.assertEqual(FinetuneEdgePrompt.variant_tag(cfg), "anchorsauto-loops0")

    def test_both_explicit(self):
        cfg = _cfg(num_anchors=10, add_self_loops=True)
        self.assertEqual(FinetuneEdgePrompt.variant_tag(cfg), "anchors10-loops1")

    def test_non_default_node_subgraph_hops_appends_h_tag(self):
        cfg = _cfg(node_subgraph_hops=3)
        self.assertEqual(FinetuneEdgePrompt.variant_tag(cfg), "anchorsauto-loopsauto-h3")

    def test_non_default_node_subgraph_min_size_appends_mn_tag(self):
        cfg = _cfg(node_subgraph_min_size=5)
        self.assertEqual(FinetuneEdgePrompt.variant_tag(cfg), "anchorsauto-loopsauto-mn5")

    def test_non_default_node_subgraph_max_size_appends_mx_tag(self):
        cfg = _cfg(node_subgraph_max_size=50)
        self.assertEqual(FinetuneEdgePrompt.variant_tag(cfg), "anchorsauto-loopsauto-mx50")

    def test_use_official_false_suppresses_subgraph_tags(self):
        # Even with non-default subgraph knobs, setting use_official=False
        # should suppress h/mn/mx tags.
        cfg = _cfg(
            use_official_node_subgraphs=False,
            node_subgraph_hops=3,
            node_subgraph_min_size=5,
            node_subgraph_max_size=50,
        )
        self.assertEqual(FinetuneEdgePrompt.variant_tag(cfg), "anchorsauto-loopsauto")

    def test_combined_non_default_subgraph(self):
        cfg = _cfg(
            num_anchors=10,
            add_self_loops=True,
            node_subgraph_hops=3,
            node_subgraph_min_size=5,
            node_subgraph_max_size=50,
        )
        self.assertEqual(
            FinetuneEdgePrompt.variant_tag(cfg),
            "anchors10-loops1-h3-mn5-mx50",
        )


if __name__ == "__main__":
    unittest.main()
