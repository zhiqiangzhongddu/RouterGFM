"""ProNoG condition-net prompting tests.

Covers the three pieces that are specific to this method and not shared
with GraphPrompt:

- ``build_hop_neighbor_pairs`` — the capped multi-hop ego-network builder,
  checked against a brute-force reference over random graphs, plus the
  symmetrization that keeps directed datasets from giving sink nodes an
  empty neighbourhood.
- ``conditioned_subgraph_readout`` — the similarity-weighted readout of
  paper Eq. 7, checked against a per-node reference including the chunked
  path.
- The prototype/label-space contract: classes with no training support are
  masked out of the logits (their zero centre otherwise scores a spurious
  mid-range cosine of 0), and support counts cannot go stale behind valid
  centres.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.methods.pronog import _UNSEEN_CLASS_LOGIT, FinetuneProNoG
from src.finetune.methods.pronog_utils import (
    build_hop_neighbor_pairs,
    conditioned_subgraph_readout,
)
from src.finetune.prompts.pronog import ProNoGConditionNet


def pronog_cfg() -> CN:
    cfg = set_cfg(CN())
    cfg.finetune.method = "pronog"
    cfg.finetune.dataset.name = "cora"
    cfg.finetune.dataset.task_level = "node"
    cfg.finetune.dataset.task_level_effective = "node"
    cfg.finetune.dataset.induced = False
    cfg.finetune.dataset.task_type = "classification"
    cfg.finetune.dataset.label_dim = 1
    cfg.finetune.dataset.num_classes = 3
    cfg.model.in_dim = 4
    cfg.model.hidden_dim = 6
    cfg.model.out_dim = 4
    cfg.model.num_layers = 2
    return cfg


def brute_force_pairs(edge_index, num_nodes: int, hops: int, cap: int) -> set:
    """Reference: symmetrized+coalesced adjacency, capped hop expansion, set dedup."""
    neighbours = [set() for _ in range(num_nodes)]
    for row, col in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if row != col:
            neighbours[row].add(col)
            neighbours[col].add(row)
    adj = [sorted(entry) for entry in neighbours]  # coalesce order: ascending id
    if cap > 0:
        adj = [entry[:cap] for entry in adj]
    pairs = set()
    for node in range(num_nodes):
        pairs.add((node, node))
        frontier = [(node, member) for member in adj[node]]
        pairs.update(frontier)
        for _hop in range(2, hops + 1):
            expanded = [
                (center, far)
                for (center, member) in frontier
                for far in adj[member]
                if far != center
            ]
            if cap > 0:
                expanded = expanded[:cap]
            pairs.update(expanded)
            frontier = expanded
    return pairs


class ProNoGHopPairsTest(unittest.TestCase):
    def test_matches_brute_force_reference(self) -> None:
        torch.manual_seed(0)
        for _trial in range(10):
            num_nodes = int(torch.randint(1, 40, (1,)))
            num_edges = int(torch.randint(0, 120, (1,)))
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            for hops in (1, 2, 3):
                for cap in (0, 1, 3, 20):
                    pairs = build_hop_neighbor_pairs(
                        edge_index, num_nodes, hops=hops, cap=cap,
                    )
                    got = set(zip(pairs[0].tolist(), pairs[1].tolist()))
                    self.assertEqual(
                        got,
                        brute_force_pairs(edge_index, num_nodes, hops, cap),
                        msg=f"hops={hops} cap={cap}",
                    )

    def test_directed_edges_are_symmetrized(self) -> None:
        # 0->1->2->3 stored one-way: without symmetrization the sink node 3
        # would only ever see itself.
        chain = torch.tensor([[0, 1, 2], [1, 2, 3]])
        pairs = build_hop_neighbor_pairs(chain, 4, hops=2, cap=20)
        got = set(zip(pairs[0].tolist(), pairs[1].tolist()))
        self.assertIn((3, 2), got)
        self.assertIn((3, 1), got)

    def test_isolated_node_keeps_only_itself(self) -> None:
        pairs = build_hop_neighbor_pairs(torch.tensor([[1], [2]]), 3, hops=2, cap=20)
        got = set(zip(pairs[0].tolist(), pairs[1].tolist()))
        self.assertEqual({member for center, member in got if center == 0}, {0})


class ProNoGReadoutTest(unittest.TestCase):
    def test_matches_equation_7_reference(self) -> None:
        torch.manual_seed(1)
        num_nodes, dim = 15, 8
        edge_index = torch.randint(0, num_nodes, (2, 40))
        node_repr = torch.randn(num_nodes, dim)
        pairs = build_hop_neighbor_pairs(edge_index, num_nodes, hops=2, cap=20)
        # chunk_size forces the multi-chunk accumulation path.
        got = conditioned_subgraph_readout(node_repr, pairs, chunk_size=7)

        normed = F.normalize(node_repr, dim=-1)
        want = torch.zeros_like(node_repr)
        for node in range(num_nodes):
            members = pairs[1][pairs[0] == node]
            acc = torch.zeros(dim)
            for member in members.tolist():
                acc += node_repr[member] * float((normed[node] * normed[member]).sum())
            want[node] = acc / max(1, members.numel())
        torch.testing.assert_close(got, want)

    def test_structureless_node_conditions_on_itself(self) -> None:
        node_repr = torch.randn(3, 4)
        pairs = build_hop_neighbor_pairs(torch.tensor([[1], [2]]), 3, hops=2, cap=20)
        readout = conditioned_subgraph_readout(node_repr, pairs)
        # cos(h_v, h_v) == 1, so the self-only ego-network returns h_v.
        torch.testing.assert_close(readout[0], node_repr[0])


class ProNoGConditionNetTest(unittest.TestCase):
    def test_zero_scaling_zeroes_the_prompts(self) -> None:
        # A truthiness guard here would emit full-strength prompts instead,
        # inverting the no-prompt ablation.
        net = ProNoGConditionNet(8, 4, dropout=0.0, scaling=0.0)
        self.assertEqual(float(net(torch.randn(5, 8)).abs().max()), 0.0)


class ProNoGValidationTest(unittest.TestCase):
    def test_explicit_zero_and_out_of_range_values_are_rejected(self) -> None:
        for key, bad in [
            ("hops", 0),
            ("bottleneck_dim", 0),
            ("neighbor_cap", -1),
            ("tau", 0.0),
            ("tau", -1.0),
            ("condition_scaling", -0.5),
            ("condition_dropout", 1.0),
        ]:
            cfg = pronog_cfg()
            setattr(cfg.finetune.pronog, key, bad)
            with self.subTest(key=key, value=bad):
                with self.assertRaises(ValueError):
                    FinetuneProNoG.validate_cfg(cfg)

    def test_zero_scaling_with_mul_combine_is_rejected(self) -> None:
        cfg = pronog_cfg()
        cfg.finetune.pronog.prompt_combine = "mul"
        cfg.finetune.pronog.condition_scaling = 0.0
        with self.assertRaisesRegex(ValueError, "collapses"):
            FinetuneProNoG.validate_cfg(cfg)

    def test_batch_eval_centers_are_rejected(self) -> None:
        cfg = pronog_cfg()
        cfg.finetune.pronog.eval_center_mode = "batch"
        with self.assertRaisesRegex(ValueError, "validation/test labels"):
            FinetuneProNoG.validate_cfg(cfg)

    def test_run_tag_reports_the_raw_hop_count(self) -> None:
        cfg = pronog_cfg()
        cfg.finetune.pronog.hops = 3
        self.assertEqual(FinetuneProNoG.run_tag(cfg), "h3_add")


class ProNoGUnseenClassTest(unittest.TestCase):
    def _centers(self) -> torch.Tensor:
        centers = torch.zeros(3, 4)
        centers[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        centers[1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        return centers  # class 2 has no training support -> zero centre

    def test_zero_prototype_wins_argmax_without_the_mask(self) -> None:
        task = FinetuneProNoG(pronog_cfg())
        # Points away from both trained centres, so both cosines are negative
        # while the zero centre scores exactly 0.
        embeddings = torch.tensor([[-1.0, -1.0, 0.0, 0.0]])
        logits = task._similarity_logits(embeddings, self._centers())
        self.assertEqual(int(logits.argmax()), 2)

    def test_mask_excludes_unsupported_classes(self) -> None:
        task = FinetuneProNoG(pronog_cfg())
        unseen = task._unseen_class_mask(torch.tensor([3.0, 5.0, 0.0]))
        self.assertIsNotNone(unseen)
        self.assertEqual(unseen.tolist(), [False, False, True])

        embeddings = torch.tensor([[-1.0, -1.0, 0.0, 0.0]])
        logits = task._similarity_logits(embeddings, self._centers(), unseen)
        self.assertEqual(float(logits[0, 2]), _UNSEEN_CLASS_LOGIT)
        self.assertNotEqual(int(logits.argmax()), 2)

    def test_no_mask_when_every_class_has_support(self) -> None:
        task = FinetuneProNoG(pronog_cfg())
        self.assertIsNone(task._unseen_class_mask(torch.tensor([1.0, 2.0, 3.0])))

    def test_centers_require_support_counts(self) -> None:
        task = FinetuneProNoG(pronog_cfg())
        counts = torch.tensor([3.0, 5.0, 0.0])
        task._store_latest_centers(self._centers(), counts)
        self.assertTrue(task._has_latest_centers())
        self.assertEqual(task.latest_center_counts.tolist(), counts.tolist())
        # Centres without counts must not be marked valid: the unseen mask
        # reads the counts, so a stale pair would silently mis-mask.
        task._store_latest_centers(self._centers(), None)
        self.assertFalse(task._has_latest_centers())

    def test_centers_and_counts_survive_state_dict_round_trip(self) -> None:
        cfg = pronog_cfg()
        source = FinetuneProNoG(cfg)
        counts = torch.tensor([3.0, 5.0, 0.0])
        source._store_latest_centers(self._centers(), counts)
        restored = FinetuneProNoG(cfg)
        restored.load_state_dict(source.state_dict())
        self.assertTrue(restored._has_latest_centers())
        torch.testing.assert_close(restored.latest_centers, source.latest_centers)
        torch.testing.assert_close(restored.latest_center_counts, counts)


if __name__ == "__main__":
    unittest.main()
