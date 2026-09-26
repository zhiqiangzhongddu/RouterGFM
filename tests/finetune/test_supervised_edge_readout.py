"""finetune.edge_readout=endpoints for baselines on induced edge cells.

By default the baselines use whole-subgraph pooling on induced edge (LP)
tasks. `finetune.edge_readout` is the shared, opt-in endpoint readout: default
"pool" must keep every existing run byte-identical; "endpoints" must use the
Hadamard readout of the two endpoint representations, gated on the RAW task
level so it is inert on node/graph cells.
"""

import torch
from torch import nn
from yacs.config import CfgNode as CN

from src.config import set_cfg
from src.finetune.task_heads import TaskAwareObjective

D = 4


def _cfg(edge_readout=None, raw_level="edge"):
    cfg = CN()
    set_cfg(cfg)
    cfg.finetune.method = "supervised"
    cfg.finetune.dataset.task_level = raw_level  # raw level; induced promotion is separate
    cfg.finetune.dataset.induced = True
    cfg.finetune.dataset.task_type = "classification"
    cfg.finetune.dataset.num_classes = 2
    cfg.model.out_dim = D
    if edge_readout is not None:
        cfg.finetune.edge_readout = edge_readout
    return cfg


def _objective(cfg):
    return TaskAwareObjective(
        cfg,
        task_level="graph",  # the promoted level the runner hands over on induced edge
        task_type="classification",
        label_dim=1,
        num_classes=2,
        repr_dim=D,
    )


def _batch(num_graphs=2, nodes_per_graph=3):
    n = num_graphs * nodes_per_graph
    data = type("B", (), {})()
    data.y = torch.tensor([1, 0][:num_graphs])
    data.batch = torch.repeat_interleave(torch.arange(num_graphs), nodes_per_graph)
    # one query edge per graph, batch-global indices (PyG collation convention)
    data.edge_label_index = torch.tensor(
        [[g * nodes_per_graph, g * nodes_per_graph + 1] for g in range(num_graphs)]
    ).t()
    return data, torch.arange(n * D, dtype=torch.float32).view(n, D)


class DefaultIsPoolTest:
    def test_default_off(self):
        obj = _objective(_cfg())
        assert obj.edge_endpoint_readout is False

    def test_default_pools_whole_subgraph(self):
        obj = _objective(_cfg())
        data, node_repr = _batch()
        repr_used, labels = obj.select_representations_and_labels(
            node_repr=node_repr, graph_repr=None, data=data, device=torch.device("cpu")
        )
        # mean-pool over each 3-node subgraph, independent of the query pair
        expected = torch.stack([node_repr[0:3].mean(0), node_repr[3:6].mean(0)])
        assert torch.allclose(repr_used, expected)
        assert torch.equal(labels, data.y)


class EndpointReadoutTest:
    def test_endpoints_hadamard_and_labels(self):
        obj = _objective(_cfg(edge_readout="endpoints"))
        assert obj.edge_endpoint_readout is True
        data, node_repr = _batch()
        repr_used, labels = obj.select_representations_and_labels(
            node_repr=node_repr, graph_repr=None, data=data, device=torch.device("cpu")
        )
        expected = torch.stack(
            [node_repr[0] * node_repr[1], node_repr[3] * node_repr[4]]
        )
        assert torch.allclose(repr_used, expected)
        assert torch.equal(labels, data.y)

    def test_missing_edge_label_index_raises(self):
        obj = _objective(_cfg(edge_readout="endpoints"))
        data, node_repr = _batch()
        del data.edge_label_index
        try:
            obj.select_representations_and_labels(
                node_repr=node_repr, graph_repr=None, data=data, device=torch.device("cpu")
            )
        except ValueError as err:
            assert "edge_label_index" in str(err)
        else:
            raise AssertionError("expected ValueError without edge_label_index")


class RawLevelGateTest:
    def test_inert_on_non_edge_cells(self):
        obj = _objective(_cfg(edge_readout="endpoints", raw_level="node"))
        assert obj.edge_endpoint_readout is False
        data, node_repr = _batch()
        repr_used, _ = obj.select_representations_and_labels(
            node_repr=node_repr, graph_repr=None, data=data, device=torch.device("cpu")
        )
        expected = torch.stack([node_repr[0:3].mean(0), node_repr[3:6].mean(0)])
        assert torch.allclose(repr_used, expected)

    def test_invalid_value_raises(self):
        try:
            _objective(_cfg(edge_readout="hadamard"))
        except ValueError as err:
            assert "edge_readout" in str(err)
        else:
            raise AssertionError("expected ValueError for invalid edge_readout")


def main() -> int:
    failures = 0
    for cls in (DefaultIsPoolTest, EndpointReadoutTest, RawLevelGateTest):
        inst = cls()
        for name in dir(inst):
            if not name.startswith("test_"):
                continue
            try:
                getattr(inst, name)()
                print(f"PASS {cls.__name__}.{name}")
            except Exception as err:  # noqa: BLE001 - test harness
                failures += 1
                print(f"FAIL {cls.__name__}.{name}: {err}")
    print("OK" if failures == 0 else f"{failures} FAILURES")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
