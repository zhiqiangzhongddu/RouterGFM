"""Smoke test: parallel edge-induced generation matches the serial path."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Data

from src.data_loader.induced_graphs import build_edge_induced_graphs_supervised


def _make_graph(num_nodes=300, num_edges=1200, dim=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), generator=gen)
    x = torch.randn(num_nodes, dim, generator=gen)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def _assert_parallel_matches_serial() -> None:
    data = _make_graph()
    gen = torch.Generator().manual_seed(1)
    pos = data.edge_index[:, torch.randperm(data.edge_index.size(1), generator=gen)[:400]]
    neg = torch.randint(0, data.num_nodes, (2, 400), generator=gen)

    previous_workers = os.environ.get("ICG_EDGE_INDUCED_WORKERS")
    try:
        os.environ["ICG_EDGE_INDUCED_WORKERS"] = "0"
        serial = build_edge_induced_graphs_supervised(data, pos, neg, max_hops=2, max_size=None)

        os.environ["ICG_EDGE_INDUCED_WORKERS"] = "4"
        parallel = build_edge_induced_graphs_supervised(data, pos, neg, max_hops=2, max_size=None)
    finally:
        if previous_workers is None:
            os.environ.pop("ICG_EDGE_INDUCED_WORKERS", None)
        else:
            os.environ["ICG_EDGE_INDUCED_WORKERS"] = previous_workers

    assert len(serial) == len(parallel) == 800, (len(serial), len(parallel))
    for i, (s, p) in enumerate(zip(serial, parallel)):
        assert int(s.y) == int(p.y), f"label mismatch at {i}"
        assert s.num_nodes == p.num_nodes, f"num_nodes mismatch at {i}"
        assert s.edge_index.shape == p.edge_index.shape, f"edge_index shape mismatch at {i}"
        assert torch.equal(s.edge_index, p.edge_index), f"edge_index mismatch at {i}"
        assert torch.equal(s.x, p.x), f"x mismatch at {i}"
        assert torch.equal(s.edge_label_index, p.edge_label_index), f"edge_label_index mismatch at {i}"

def test_parallel_output_matches_serial():
    _assert_parallel_matches_serial()


def main() -> int:
    _assert_parallel_matches_serial()
    print("OK: parallel output matches serial for 800 edge-induced subgraphs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
