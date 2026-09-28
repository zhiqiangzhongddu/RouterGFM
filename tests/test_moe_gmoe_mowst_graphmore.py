"""Smoke runs of the GMoE, Mowst, and GraphMoRE runners on tiny synthetic instance sets.

Every supported task family (induced node / edge instances, graph
classification, multi-target regression, multi-label with missing assays) is
trained for two epochs through ``Runner(cfg).fit()`` with the data loaders
patched, and one node task goes through the method's ``run_<method>(cfg)``
entry point to append a result row.
"""

import math
import os
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from src.config import cfg as base_cfg

IN_DIM = 6
KINDS = ("node", "edge", "graph", "regression", "multilabel")
_META = {
    "node": {"num_node_features": IN_DIM, "num_classes": 3, "label_dim": 1, "task_type": "classification"},
    "edge": {"num_node_features": IN_DIM, "num_classes": 2, "label_dim": 1, "task_type": "classification"},
    "graph": {"num_node_features": IN_DIM, "num_classes": 3, "label_dim": 1, "task_type": "classification"},
    "regression": {"num_node_features": IN_DIM, "num_classes": 1, "label_dim": 2, "task_type": "regression"},
    "multilabel": {"num_node_features": IN_DIM, "num_classes": 2, "label_dim": 3, "task_type": "classification"},
}
_METRIC = {"edge": "test_auc", "regression": "test_mae", "multilabel": "test_auc"}
# method -> (package, runner class, run function, tiny overrides of cfg.moe.<method>)
METHODS = {
    "gmoe": ("src.moe.gmoe", "GMoERunner", "run_gmoe", {"hidden_dim": 8, "num_layers": 2, "num_experts": 4,
                                                          "num_experts_1hop": 2, "k": 2}),
    "mowst": ("src.moe.mowst", "MowstRunner", "run_mowst", {"hidden_dim": 8, "pretrain_epochs": 1}),
    "graphmore": ("src.moe.graphmore", "GraphMoRERunner", "run_graphmore", {"hidden_dim": 8, "embed_dim": 4,
                                                                           "gating_hidden_dim": 4}),
}


def _graph(kind, i, gen):
    n = 5 + i % 4
    ring = [(j, (j + 1) % n) for j in range(n)] + ([(0, 2), (1, 3)] if i % 3 == 0 else [])
    x = torch.randn(n, IN_DIM, generator=gen)
    x[: n // 2] += float(i % 3)
    g = Data(x=x, edge_index=to_undirected(torch.tensor(ring, dtype=torch.long).T))
    if kind == "node":
        g.y, g.target_node_index = torch.tensor(i % 3), torch.tensor([1])
    elif kind == "edge":
        g.y, g.edge_label_index = torch.tensor(i % 2), torch.tensor([[0], [2]])
    elif kind == "graph":
        g.y = torch.tensor([i % 3])
    elif kind == "regression":
        g.y = torch.randn(1, 2, generator=gen) * 10.0 + 3.0
    else:
        y = (torch.rand(1, 3, generator=gen) > 0.5).float()
        y[0, i % 3] = float("nan")
        g.y = y
    return g


def _patched_data(method, kind, count=16, num_train=8):
    gen = torch.Generator().manual_seed(0)
    graphs = [_graph(kind, i, gen) for i in range(count)]
    val = list(range(num_train, num_train + 2)) if kind == "edge" else []

    def loaders(**_kwargs):
        return (
            DataLoader(Subset(graphs, list(range(num_train))), batch_size=4, shuffle=True),
            DataLoader(Subset(graphs, val), batch_size=4),
            DataLoader(Subset(graphs, list(range(num_train, count))), batch_size=4),
        )

    return patch.multiple(
        f"{METHODS[method][0]}.trainer",
        create_dataset=lambda **_kwargs: graphs,
        dataset_info=lambda **_kwargs: dict(_META[kind]),
        make_workflow_loaders=loaders,
        log_split_instance_counts=lambda *args, **kwargs: None,
    )


def _cfg(method, kind, tmp):
    cfg = base_cfg.clone()
    cfg.seed, cfg.seeds = 42, [42]
    cfg.save_results.output_dir = os.path.join(tmp, "results")
    m = cfg.moe[method]
    m.epochs, m.num_runs, m.batch_size, m.skip_if_exists = 2, 1, 4, False
    m.checkpoint_dir, m.log_dir = os.path.join(tmp, "ckpt"), os.path.join(tmp, "logs")
    for key, value in METHODS[method][3].items():
        m[key] = value
    ds = m.dataset
    ds.name = "toy"
    ds.task_level = {"node": "node", "edge": "edge"}.get(kind, "graph")
    ds.induced = True
    ds.task_type, ds.num_classes, ds.label_dim = "none", None, None
    ds.fixed_split = (0.5, 0.1, 0.4) if kind == "edge" else (4, 0.0, 1.0)
    cfg.moe.method = method
    return cfg


class RunnerSmokeTest(unittest.TestCase):
    def test_every_task_family_trains_and_reports_its_metric(self):
        for method, (package, runner_name, _, _) in METHODS.items():
            runner_cls = getattr(import_module(f"{package}.trainer"), runner_name)
            for kind in KINDS:
                with self.subTest(method=method, kind=kind), tempfile.TemporaryDirectory() as tmp, _patched_data(method, kind):
                    runner = runner_cls(_cfg(method, kind, tmp))
                    runner.fit()
                    metric = _METRIC.get(kind, "test_acc")
                    self.assertTrue(math.isfinite(runner.best_metrics[metric]), runner.best_metrics)
                    self.assertTrue(os.path.isfile(runner.get_checkpoint_path_for_metrics()))

    def test_run_entry_point_appends_a_result_row(self):
        for method, (package, _, run_name, _) in METHODS.items():
            run = getattr(import_module(f"{package}.run"), run_name)
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp, _patched_data(method, "node"):
                self.assertEqual(run(_cfg(method, "node", tmp)), 0)
                table = (Path(tmp) / "results" / f"moe_{method}.tsv").read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(table), 2)
                self.assertIn("test_acc_mean", table[0].split("\t"))


if __name__ == "__main__":
    unittest.main()
