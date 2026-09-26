"""Regression tests for the v2 (pair-level) edge splits.

The v1 edge splits permuted *directed* edge slots, so for undirected-storage
datasets the reverse copy of a held-out val/test edge could land in the
training/message context (~83% of cora test positives were reachable that
way). v2 splits unordered pairs, excludes every direction of held-out pairs
from the context, samples negatives as unique unordered non-edges, and
removes the target edge from each induced subgraph (SEAL-style).
"""

import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from src.data_loader.dataset_splits import (
    EDGE_SPLIT_FORMAT_VERSION,
    _get_or_create_edge_split_payload,
    _load_existing_edge_split_payload,
)
from src.data_loader.induced_graphs import build_edge_induced_graphs_supervised
from src.pretrain.methods.edge_pred import _unique_undirected_edges


def _random_undirected_graph(num_nodes: int = 60, num_edges: int = 240, seed: int = 0) -> Data:
    generator = torch.Generator().manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_edges,), generator=generator)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=generator)
    mask = src != dst
    edge_index = to_undirected(torch.stack([src[mask], dst[mask]], dim=0))
    x = torch.randn(num_nodes, 8, generator=generator)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def _pair_keys(edge_index: torch.Tensor, num_nodes: int) -> set:
    lo = torch.minimum(edge_index[0], edge_index[1])
    hi = torch.maximum(edge_index[0], edge_index[1])
    return set((lo * num_nodes + hi).tolist())


class EdgeSplitV2Test(unittest.TestCase):
    def _payload(self, data: Data, split=(0.1, 0.1, 0.1), seed: int = 42, persist: bool = False,
                 split_root: Path | None = None):
        return _get_or_create_edge_split_payload(
            dataset_name="synthetic_edge_seed%d" % seed,
            split=split,
            seed=seed,
            split_root_path=split_root or Path(tempfile.mkdtemp()),
            data=data,
            persist=persist,
            verbose=False,
        )

    def test_heldout_pairs_never_in_context(self):
        data = _random_undirected_graph()
        payload = self._payload(data)
        num_nodes = data.num_nodes
        context = data.edge_index[:, torch.tensor(payload["context_pos_idx"], dtype=torch.long)]
        message = data.edge_index[:, torch.tensor(payload["message_pos_idx"], dtype=torch.long)]
        context_keys = _pair_keys(context, num_nodes) | _pair_keys(message, num_nodes)
        for key in ("val_pos_idx", "test_pos_idx"):
            held = data.edge_index[:, torch.tensor(payload[key], dtype=torch.long)]
            held_keys = _pair_keys(held, num_nodes)
            self.assertTrue(held_keys, f"{key} unexpectedly empty")
            leaked = held_keys & context_keys
            self.assertFalse(leaked, f"{len(leaked)} held-out pairs from {key} leak into context")

    def test_supervision_pairs_disjoint(self):
        data = _random_undirected_graph()
        payload = self._payload(data)
        num_nodes = data.num_nodes
        keys = {}
        for key in ("train_pos_idx", "val_pos_idx", "test_pos_idx"):
            pairs = data.edge_index[:, torch.tensor(payload[key], dtype=torch.long)]
            keys[key] = _pair_keys(pairs, num_nodes)
            # one canonical slot per pair: no internal duplicates
            self.assertEqual(len(keys[key]), len(payload[key]))
        self.assertFalse(keys["train_pos_idx"] & keys["val_pos_idx"])
        self.assertFalse(keys["train_pos_idx"] & keys["test_pos_idx"])
        self.assertFalse(keys["val_pos_idx"] & keys["test_pos_idx"])

    def test_negatives_unique_nonedges_disjoint(self):
        data = _random_undirected_graph()
        payload = self._payload(data)
        num_nodes = data.num_nodes
        edge_keys = _pair_keys(data.edge_index, num_nodes)
        all_neg = set()
        for key in ("train_neg_edge_index", "val_neg_edge_index", "test_neg_edge_index"):
            neg = payload[key]
            self.assertGreater(neg.size(1), 0)
            neg_keys = _pair_keys(neg, num_nodes)
            self.assertEqual(len(neg_keys), neg.size(1), f"duplicate negatives within {key}")
            self.assertFalse(neg_keys & edge_keys, f"{key} contains real (possibly reversed) edges")
            self.assertFalse(neg_keys & all_neg, f"{key} shares negatives with another split")
            all_neg |= neg_keys

    def test_directed_only_edges_survive(self):
        # WebKB-style storage: some edges exist in one direction only.
        edge_index = torch.tensor([[5, 4, 9, 8, 3], [1, 7, 2, 6, 0]], dtype=torch.long)
        data = Data(x=torch.randn(10, 4), edge_index=edge_index, num_nodes=10)
        payload = self._payload(data, split=(0.4, 0.2, 0.2))
        covered = (
            len(payload["train_pos_idx"]) + len(payload["val_pos_idx"]) + len(payload["test_pos_idx"])
            + len({s for s in payload["message_pos_idx"]})
        )
        self.assertEqual(covered, 5)

    def test_deterministic_and_v2_roundtrip(self):
        data = _random_undirected_graph()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._payload(data, persist=True, split_root=root)
            second = self._payload(data, persist=True, split_root=root)
            self.assertEqual(first["train_pos_idx"], second["train_pos_idx"])
            self.assertEqual(first["context_pos_idx"], second["context_pos_idx"])
            self.assertEqual(int(first["meta"]["format_version"]), EDGE_SPLIT_FORMAT_VERSION)

    def test_legacy_payload_rejected(self):
        data = _random_undirected_graph()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            total = int(data.edge_index.size(1))
            legacy = {
                "train_pos_idx": list(range(0, total - 20)),
                "val_pos_idx": list(range(total - 20, total - 10)),
                "test_pos_idx": list(range(total - 10, total)),
                "message_pos_idx": [],
                "train_neg_edge_index": torch.zeros((2, total - 20), dtype=torch.long),
                "val_neg_edge_index": torch.zeros((2, 10), dtype=torch.long),
                "test_neg_edge_index": torch.zeros((2, 10), dtype=torch.long),
                "meta": {"total_edges": total},
            }
            torch.save(legacy, path)
            self.assertIsNone(_load_existing_edge_split_payload(path, total))

    def test_induced_subgraph_removes_target_edge(self):
        data = _random_undirected_graph()
        payload = self._payload(data)
        context_pairs = data.edge_index[:, torch.tensor(payload["context_pos_idx"], dtype=torch.long)]
        context_data = Data(x=data.x, edge_index=context_pairs, num_nodes=data.num_nodes)
        train_pairs = data.edge_index[:, torch.tensor(payload["train_pos_idx"], dtype=torch.long)]
        graphs = build_edge_induced_graphs_supervised(
            data=context_data,
            pos_edge_pairs=train_pairs,
            neg_edge_pairs=payload["train_neg_edge_index"],
            max_hops=1,
        )
        self.assertEqual(len(graphs), train_pairs.size(1) + payload["train_neg_edge_index"].size(1))
        for graph in graphs:
            u, v = int(graph.edge_label_index[0, 0]), int(graph.edge_label_index[1, 0])
            ei = graph.edge_index
            present = (((ei[0] == u) & (ei[1] == v)) | ((ei[0] == v) & (ei[1] == u))).any()
            self.assertFalse(bool(present), "target edge still present in induced subgraph")

    def test_edge_pred_dedup_keeps_directed_only(self):
        edge_index = torch.tensor([[5, 1, 4, 2, 2], [1, 5, 7, 9, 2]], dtype=torch.long)
        unique = _unique_undirected_edges(edge_index)
        keys = _pair_keys(unique, 10)
        # (5,1)+(1,5) collapse to one pair; (4,7) and (2,9) survive although
        # only stored in one direction; the (2,2) self-loop is dropped.
        self.assertEqual(keys, {1 * 10 + 5, 4 * 10 + 7, 2 * 10 + 9})


if __name__ == "__main__":
    unittest.main()
