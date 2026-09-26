import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import torch
from geoopt import ManifoldParameter
from scipy.optimize import linprog
from torch.utils.data import Subset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from src.config import cfg as base_cfg
from src.moe import run as moe_run
from src.moe.geomoe import curvature
from src.moe.geomoe.curvature import attach_node_orc, edge_orc, node_orc, orc_region, orc_target_weights
from src.moe.geomoe.model import GeoMoEModel
from src.moe.geomoe.run import _build_task_cfg, parse_geomoe_tasks, run_geomoe
from src.moe.geomoe.task import GeoMoETask, hard_negative_index
from src.moe.geomoe.trainer import GeoMoERunner

ROOT = Path(__file__).resolve().parents[1]
TSV_HEADER = "# dataset\ttask_level\ttask_type\tinduced\tfixed_split\tsplit_root\tskip_if_exists\n"


def _cfg(tmp=None, **geomoe):
    cfg = base_cfg.clone()
    cfg.seed = 42
    cfg.seeds = [42]
    g = cfg.moe.geomoe
    g.dataset.name = "toy"
    g.dataset.task_type = "classification"
    g.dataset.num_classes = 3
    g.dataset.label_dim = 1
    g.skip_if_exists = False
    g.orc_cache_dir = ""
    if tmp is not None:
        g.checkpoint_dir = os.path.join(tmp, "ckpt")
        g.log_dir = os.path.join(tmp, "logs")
        g.prediction_dir = os.path.join(tmp, "pred")
        cfg.save_results.output_dir = os.path.join(tmp, "results")
    for key, value in geomoe.items():
        setattr(g, key, value)
    return cfg


def _undirected(pairs):
    return to_undirected(torch.tensor(pairs, dtype=torch.long).T)


def _reference_edge_orc(graph: nx.Graph, u, v, p):
    """Per-edge exact W1 on the full neighbourhood supports (no cancellation, no blocks)."""
    src = [u] + sorted(graph[u])
    dst = [v] + sorted(graph[v])
    mu = np.array([p] + [(1 - p) / graph.degree[u]] * graph.degree[u])
    nu = np.array([p] + [(1 - p) / graph.degree[v]] * graph.degree[v])
    cost = np.array([[nx.shortest_path_length(graph, a, b) for b in dst] for a in src], dtype=float)
    a, b = len(src), len(dst)
    a_eq = np.zeros((a + b, a * b))
    for i in range(a):
        a_eq[i, i * b:(i + 1) * b] = 1
    for j in range(b):
        a_eq[a + j, j::b] = 1
    res = linprog(cost.ravel(), A_eq=a_eq, b_eq=np.concatenate([mu, nu]), bounds=(0, None), method="highs")
    return 1.0 - res.fun


def _graph(kind, i, gen, in_dim=6):
    n = 5 + i % 4
    ring = [(j, (j + 1) % n) for j in range(n)]
    chords = [(0, 2), (1, 3)] if i % 3 == 0 else []
    x = torch.randn(n, in_dim, generator=gen)
    x[: n // 2] += float(i % 3)
    g = Data(x=x, edge_index=_undirected(ring + chords))
    if kind == "node":
        g.y = torch.tensor(i % 3)
        g.target_node_index = torch.tensor([1])
    elif kind == "edge":
        g.y = torch.tensor(i % 2)
        g.edge_label_index = torch.tensor([[0], [2]])
    elif kind == "graph":
        g.y = torch.tensor([i % 3])
    elif kind == "regression":
        g.y = torch.randn(1, 2, generator=gen) * 10.0 + 3.0
    else:  # multilabel with missing assays
        y = (torch.rand(1, 3, generator=gen) > 0.5).float()
        y[0, i % 3] = float("nan")
        g.y = y
    return g


_META = {
    "node": {"num_node_features": 6, "num_classes": 3, "label_dim": 1, "task_type": "classification"},
    "edge": {"num_node_features": 6, "num_classes": 2, "label_dim": 1, "task_type": "classification"},
    "graph": {"num_node_features": 6, "num_classes": 3, "label_dim": 1, "task_type": "classification"},
    "regression": {"num_node_features": 6, "num_classes": 1, "label_dim": 2, "task_type": "regression"},
    "multilabel": {"num_node_features": 6, "num_classes": 2, "label_dim": 3, "task_type": "classification"},
}


def _patched_data(kind, count=16, num_train=8):
    gen = torch.Generator().manual_seed(0)
    graphs = [_graph(kind, i, gen) for i in range(count)]
    val = list(range(num_train, num_train + 2)) if kind == "edge" else []

    def _loaders(**_kwargs):
        return (
            DataLoader(Subset(graphs, list(range(num_train))), batch_size=4, shuffle=True),
            DataLoader(Subset(graphs, val), batch_size=4),
            DataLoader(Subset(graphs, list(range(num_train, count))), batch_size=4),
        )

    create = MagicMock(return_value=graphs)
    return graphs, create, patch.multiple(
        "src.moe.geomoe.trainer",
        create_dataset=create,
        dataset_info=lambda **_kwargs: dict(_META[kind]),
        make_workflow_loaders=_loaders,
        log_split_instance_counts=lambda *args, **kwargs: None,
    )


def _smoke_cfg(tmp, kind, **extra):
    level = {"node": "node", "edge": "edge"}.get(kind, "graph")
    cfg = _cfg(tmp, epochs=2, hidden_dim=4, num_runs=1, batch_size=4, **extra)
    ds = cfg.moe.geomoe.dataset
    ds.task_level = level
    ds.task_type = "none"
    ds.num_classes = None
    ds.label_dim = None
    ds.fixed_split = (0.5, 0.1, 0.4) if kind == "edge" else (4, 0.0, 1.0)
    return cfg


class CurvatureTest(unittest.TestCase):
    def test_path_interior_edges_are_flat(self):
        kappa = edge_orc(_undirected([(i, i + 1) for i in range(7)]), 8)
        ei = _undirected([(i, i + 1) for i in range(7)])
        interior = (ei.min(0).values > 0) & (ei.max(0).values < 7)
        self.assertTrue(torch.allclose(kappa[interior], torch.zeros(int(interior.sum())), atol=1e-6))

    def test_triangle_positive_and_bridge_negative(self):
        self.assertTrue(torch.allclose(edge_orc(_undirected([(0, 1), (1, 2), (2, 0)]), 3), torch.full((6,), 0.75), atol=1e-6))
        stars = [(0, 1)] + [(0, i) for i in range(2, 6)] + [(1, i) for i in range(6, 10)]
        ei = _undirected(stars)
        kappa = edge_orc(ei, 10)
        hub = ((ei[0] == 0) & (ei[1] == 1)) | ((ei[0] == 1) & (ei[1] == 0))
        self.assertTrue(bool((kappa[hub] < 0).all()))
        self.assertAlmostEqual(float(kappa[hub][0]), -0.6, places=6)

    def test_symmetric_and_matches_per_edge_reference(self):
        graph = nx.gnm_random_graph(14, 30, seed=3)
        ei = _undirected(list(graph.edges()))
        for p in (0.0, 0.5, 0.8):
            kappa = edge_orc(ei, 14, idleness=p)
            for col in range(ei.size(1)):
                u, v = int(ei[0, col]), int(ei[1, col])
                self.assertAlmostEqual(float(kappa[col]), _reference_edge_orc(graph, u, v, p), places=5)
        # Direction, duplicates and self-loops do not matter; self-loop columns are NaN.
        directed = torch.cat([ei[:, ei[0] < ei[1]], ei[:, :3], torch.tensor([[2], [2]])], dim=1)
        k_dir = edge_orc(directed, 14)
        self.assertTrue(torch.isnan(k_dir[-1]))
        k_ref = edge_orc(ei, 14)
        lookup = {(int(a), int(b)): float(k) for a, b, k in zip(ei[0], ei[1], k_ref)}
        for col in range(directed.size(1) - 1):
            self.assertAlmostEqual(float(k_dir[col]), lookup[(int(directed[0, col]), int(directed[1, col]))], places=6)

    def test_node_orc_mean_and_isolated_zero(self):
        ei = _undirected([(0, 1), (1, 2), (2, 0), (2, 3)])
        kappa_e = edge_orc(ei, 5)
        kappa_v = node_orc(ei, 5)
        self.assertEqual(tuple(kappa_v.shape), (5,))
        for v in range(4):
            incident = (ei[0] == v)
            self.assertAlmostEqual(float(kappa_v[v]), float(kappa_e[incident].mean()), places=6)
        self.assertEqual(float(kappa_v[4]), 0.0)
        self.assertTrue(torch.equal(node_orc(torch.zeros(2, 0, dtype=torch.long), 3), torch.zeros(3)))

    def test_regions_and_target_weights(self):
        kappa = torch.tensor([0.0, -0.5, 0.5, 5e-5, -2e-4])
        self.assertEqual(orc_region(kappa, 1e-4).tolist(), [0, 1, 2, 0, 1])
        w = orc_target_weights(kappa, 1e-4, 0.1)
        self.assertTrue(torch.allclose(w.sum(-1), torch.ones(5)))
        self.assertEqual(w[:3].argmax(-1).tolist(), [0, 1, 2])
        grid = torch.linspace(1e-4, 1.0, 50)
        self.assertTrue(bool((orc_target_weights(grid, 1e-4, 0.1)[:, 2].diff() > 0).all()))
        self.assertTrue(bool((orc_target_weights(-grid, 1e-4, 0.1)[:, 1].diff() > 0).all()))

    def test_attach_node_orc_uses_content_addressed_cache(self):
        graphs = [_graph("node", i, torch.Generator().manual_seed(i)) for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(curvature, "node_orc", wraps=curvature.node_orc) as spy:
            attach_node_orc(graphs, 0.5, cache_dir=tmp)
            self.assertEqual(spy.call_count, 4)
            first = [g.node_orc.clone() for g in graphs]
            copies = [g.clone() for g in graphs]
            attach_node_orc(copies, 0.5, cache_dir=tmp)
            self.assertEqual(spy.call_count, 4)
            attach_node_orc(copies, 0.3, cache_dir=tmp)  # other idleness -> other file
            self.assertEqual(spy.call_count, 8)
        for g, k in zip(graphs, first):
            self.assertEqual(tuple(g.node_orc.shape), (g.num_nodes,))
            self.assertTrue(torch.equal(g.node_orc, k))


def _batch(kind="node", count=6):
    gen = torch.Generator().manual_seed(1)
    graphs = [_graph(kind, i, gen) for i in range(count)]
    attach_node_orc(graphs, 0.5)
    return Batch.from_data_list(graphs)


class ModelAndLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = GeoMoEModel(in_dim=6, hidden_dim=8, num_layers=2, dropout=0.0)
        cfg = _cfg(hidden_dim=8)
        self.task = GeoMoETask(cfg)

    def test_gate_and_fusion(self):
        batch = _batch()
        fused, graph_repr = self.model(batch)
        self.assertIsNone(graph_repr)
        gate, experts = self.model.last_gate, self.model.last_expert
        self.assertEqual(tuple(gate.shape), (batch.num_nodes, 3))
        self.assertEqual(tuple(experts.shape), (batch.num_nodes, 3, 8))
        self.assertTrue(torch.allclose(gate.sum(-1), torch.ones(batch.num_nodes), atol=1e-6))
        self.assertTrue(torch.allclose(fused, sum(gate[:, m:m + 1] * experts[:, m] for m in range(3)), atol=1e-6))

    def test_single_adam_is_valid(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertFalse(any(isinstance(p, ManifoldParameter) for p in self.model.parameters()))
        frozen = [n for n, p in self.model.named_parameters() if not p.requires_grad]
        self.assertTrue(frozen and all(n.endswith("manifold.k") for n in frozen))
        with self.assertRaises(ValueError):
            GeoMoEModel(in_dim=6, hidden_dim=8, num_layers=2, dropout=0.0, curvatures=(1.0, -1.0))

    def test_align_loss(self):
        kappa = torch.tensor([0.0, -0.4, 0.3, 0.02])
        target = orc_target_weights(kappa, self.task.theta, self.task.eta)
        self.assertLess(float(self.task.align_loss(target, kappa)), 1e-6)
        gate = torch.softmax(torch.randn(4, 3), dim=-1)
        self.assertGreater(float(self.task.align_loss(gate, kappa)), 0.0)

    def test_contrastive_negatives_and_fallback(self):
        n, d = 7, 5
        experts = torch.randn(n, 3, d)
        fused = torch.randn(n, d)
        kappa = torch.tensor([-0.5, -0.5, 0.0, 0.0, 0.5, 0.5, 0.5])
        logits = self.task.contrastive_logits(fused, experts, kappa)
        self.assertEqual(tuple(logits.shape), (n, 1 + self.task.num_negatives))
        region = orc_region(kappa, self.task.theta)
        h_pos = experts[torch.arange(n), region]
        index = hard_negative_index(h_pos, fused, region, 4)
        for v in range(n):
            chosen = index[v].tolist()
            self.assertNotIn(v, chosen)
            sims = torch.nn.functional.cosine_similarity(h_pos[v][None], fused, dim=-1)
            others = sorted((u for u in range(n) if region[u] != region[v]), key=lambda u: -float(sims[u]))
            if len(others) >= 4:  # the most similar different-region nodes
                self.assertEqual(chosen, others[:4])
            else:  # all different-region nodes first, then same-region fill
                self.assertEqual(chosen[: len(others)], others)
        # All nodes in one region: fall back to same-region nodes ranked by similarity.
        flat = torch.zeros(n)
        idx = hard_negative_index(experts[:, 0], fused, orc_region(flat, 1e-4), 3)
        for v in range(n):
            self.assertNotIn(v, idx[v].tolist())
            sims = torch.nn.functional.cosine_similarity(experts[v, 0][None], fused, dim=-1)
            sims[v] = -2.0
            self.assertEqual(set(idx[v].tolist()), set(sims.topk(3).indices.tolist()))

    def test_contrastive_prefers_positive_direction(self):
        n, d = 6, 5
        experts = torch.randn(n, 3, d)
        kappa = torch.tensor([-0.5, 0.0, 0.5, -0.5, 0.0, 0.5])
        region = orc_region(kappa, self.task.theta)
        aligned = experts[torch.arange(n), region].clone()
        random = torch.randn(n, d)
        self.assertLess(
            float(self.task.contrastive_loss(aligned, experts, kappa)),
            float(self.task.contrastive_loss(random, experts, kappa)),
        )

    def test_one_node_instances_have_no_nan(self):
        graphs = [Data(x=torch.randn(1, 6), edge_index=torch.zeros(2, 0, dtype=torch.long), y=torch.tensor(i % 3),
                       target_node_index=torch.tensor([0])) for i in range(3)]
        attach_node_orc(graphs, 0.5)
        loss, log = self.task.step(self.model, Batch.from_data_list(graphs), torch.device("cpu"))
        self.assertTrue(torch.isfinite(loss))
        single = Batch.from_data_list(graphs[:1])
        loss, _ = self.task.step(self.model, single, torch.device("cpu"))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(self.task.contrastive_logits(torch.randn(1, 8), torch.randn(1, 3, 8), torch.zeros(1)).shape[1], 3)

    def test_step_requires_orc_and_evaluate_does_not(self):
        batch = _batch()
        loss, log = self.task.step(self.model, batch, torch.device("cpu"))
        self.assertTrue(torch.isfinite(loss))
        for key in ("train_task_loss", "train_align_loss", "train_contr_loss", "train_acc"):
            self.assertIn(key, log)
        del batch.node_orc
        with self.assertRaises(ValueError):
            self.task.step(self.model, batch, torch.device("cpu"))
        out = self.task.evaluate(self.model, batch, torch.device("cpu"), return_outputs=True)
        self.assertEqual(tuple(out[2].shape), (6, 3))

    def test_non_induced_node_task_is_rejected(self):
        cfg = _cfg()
        cfg.moe.geomoe.dataset.induced = False
        with self.assertRaises(ValueError):
            GeoMoETask(cfg)
        with self.assertRaises(ValueError):
            GeoMoERunner(cfg)


class RunnerSmokeTest(unittest.TestCase):
    def _fit(self, kind, tmp, **extra):
        graphs, _, patcher = _patched_data(kind)
        with patcher:
            cfg = _smoke_cfg(tmp, kind, **extra)
            runner = GeoMoERunner(cfg)
            runner.fit()
        return runner, graphs

    def test_all_task_families_train_and_report_brier(self):
        for kind in ("node", "edge", "graph", "regression", "multilabel"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                runner, graphs = self._fit(kind, tmp)
                metrics = runner.best_metrics
                self.assertTrue(np.isfinite(metrics["train_loss"]))
                self.assertTrue(np.isfinite(metrics["test_brier"]), metrics)
                self.assertGreaterEqual(metrics["test_brier"], 0.0)
                self.assertIn({"regression": "test_mae", "edge": "test_auc"}.get(kind, "test_acc"), metrics)
                saved = torch.load(runner._prediction_path(), map_location="cpu")
                self.assertEqual(saved["index"].tolist(), list(range(8, 16)))
                self.assertEqual(saved["meta"]["test_brier"], metrics["test_brier"])
                ckpt = torch.load(runner.get_checkpoint_path_for_metrics(), map_location="cpu")
                self.assertIn("test_brier", ckpt["metrics"])
                # The dataset's own instances are never annotated (ORC lives on support copies).
                self.assertFalse(any(hasattr(g, "node_orc") for g in graphs))

    def test_orc_only_for_support_instances(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(curvature, "node_orc", wraps=curvature.node_orc) as spy:
            runner, _ = self._fit("node", tmp, orc_cache_dir=os.path.join(tmp, "orc"))
            self.assertEqual(spy.call_count, 8)
            runner._evaluate_split(runner.test_loader, prefix="test", mask_attr="test_mask")
            runner._query_brier()
            self.assertEqual(spy.call_count, 8)
            # Same support structures: served from the content-addressed cache.
            self._fit("node", tmp, orc_cache_dir=os.path.join(tmp, "orc"))
            self.assertEqual(spy.call_count, 8)

    def test_run_geomoe_appends_result_row_and_skips_existing(self):
        _, _, patcher = _patched_data("node")
        with tempfile.TemporaryDirectory() as tmp, patcher:
            cfg = _smoke_cfg(tmp, "node")
            self.assertEqual(run_geomoe(cfg), 0)
            table = (Path(tmp) / "results" / "moe_geomoe.tsv").read_text(encoding="utf-8").splitlines()
            self.assertIn("data_preparation.dataset.split_root", table[0])
            self.assertIn("test_brier_mean", table[0])
            cfg.moe.geomoe.skip_if_exists = True
            self.assertTrue(GeoMoERunner(cfg)._skip_due_to_existing_checkpoint)


class ShiftRootGuardTest(unittest.TestCase):
    def test_missing_shift_file_fails_before_loading(self):
        _, create, patcher = _patched_data("node")
        with tempfile.TemporaryDirectory() as tmp, patcher:
            cfg = _smoke_cfg(tmp, "node")
            cfg.data_preparation.shift.root = os.path.join(tmp, "shift")
            cfg.data_preparation.dataset.split_root = os.path.join(tmp, "shift", "structural")
            with self.assertRaisesRegex(ValueError, "missing"):
                GeoMoERunner(cfg).fit()
            create.assert_not_called()

    def test_intact_shift_file_passes(self):
        from src.data_loader.shift_splits import SHIFT_SPLIT_TYPE, split_file_path

        _, create, patcher = _patched_data("node")
        with tempfile.TemporaryDirectory() as tmp, patcher:
            cfg = _smoke_cfg(tmp, "node")
            cfg.data_preparation.shift.root = os.path.join(tmp, "shift")
            root = os.path.join(tmp, "shift", "structural")
            cfg.data_preparation.dataset.split_root = root
            path = split_file_path(root, "toy", "node", 42, (4, 0.0, 1.0))
            path.parent.mkdir(parents=True)
            torch.save({"train": [0, 1], "val": [2], "test": [3, 4],
                        "meta": {"type": SHIFT_SPLIT_TYPE, "condition": "structural", "total": 5}}, path)
            GeoMoERunner(cfg).fit()
            create.assert_called_once()
            # A standard (non-shift) file under the shift root is rejected.
            torch.save({"train": [0, 1], "val": [2], "test": [3, 4], "meta": {"total": 5}}, path)
            with self.assertRaisesRegex(ValueError, "not a shift split"):
                GeoMoERunner(cfg).fit()


class PlumbingTest(unittest.TestCase):
    def test_config_defaults(self):
        g = base_cfg.moe.geomoe
        self.assertEqual((g.hidden_dim, g.theta, g.num_negatives), (16, 1e-4, 4))
        self.assertEqual(tuple(g.dataset.fixed_split), (5, 0.0, 1.0))
        self.assertEqual(list(g.curvatures), [-1.0, 1.0])
        self.assertEqual(g.tasks_tsv, "slurm/moe.geomoe.all.tsv")

    def test_run_moe_dispatches_geomoe(self):
        cfg = moe_run._build_moe_cfg(["moe.geomoe.dataset.name", "photo"])
        self.assertEqual(cfg.moe.method, "geomoe")
        with patch("src.moe.geomoe.run_geomoe", return_value=0) as fake:
            self.assertEqual(moe_run.run_moe(cfg), 0)
        fake.assert_called_once()

    def test_parse_and_build_task_cfg(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.tsv"
            path.write_text(
                TSV_HEADER
                + "photo\tnode\tclassification\tTrue\t(5,0.0,1.0)\tdata/splits_shift/mixed\tFalse\n"
                + "photo\tnode\tclassification\tTrue\t(5,0.0,1.0)\t-\tFalse\n"
                + "photo\tnode\tclassification\tTrue\t(5,0.0,1.0)\tdata/splits_shift/mixed\tFalse\n",
                encoding="utf-8",
            )
            tasks = parse_geomoe_tasks(str(path))
        self.assertEqual(len(tasks), 2)
        self.assertEqual([t["split_root"] for t in tasks], ["data/splits_shift/mixed", None])
        shifted = _build_task_cfg(_cfg(), tasks[0])
        self.assertEqual(shifted.data_preparation.dataset.split_root, "data/splits_shift/mixed")
        self.assertEqual(tuple(shifted.moe.geomoe.dataset.fixed_split), (5, 0.0, 1.0))
        self.assertIn("data_preparation.dataset.split_root", shifted.save_results.explicit_keys)
        standard = _build_task_cfg(_cfg(), tasks[1])
        self.assertEqual(standard.data_preparation.dataset.split_root, base_cfg.data_preparation.dataset.split_root)

    def test_repository_tsv_is_the_shift_sweep(self):
        tasks = parse_geomoe_tasks(str(ROOT / "slurm" / "moe.geomoe.all.tsv"))
        self.assertEqual(len(tasks), 40)
        cells = {(t["dataset"], tuple(t["fixed_split"]), t["split_root"]) for t in tasks}
        self.assertEqual(len(cells), 40)
        self.assertNotIn(("qm7b", (5, 0.0, 1.0), "data/splits_shift/feature"), cells)
        self.assertEqual({t["split_root"] for t in tasks}, {f"data/splits_shift/{c}" for c in ("feature", "structural", "mixed")})
        self.assertEqual({t["dataset"] for t in tasks}, {"photo", "ogbn-arxiv", "airports", "chameleon", "mnist", "toxcast", "qm7b"})

    def test_run_name_identity(self):
        cfg = _cfg()
        name = GeoMoERunner(cfg).run_name
        self.assertTrue(name.startswith("geomoe_toy_induced1_fewshot5-0-100_tasknode_h16_l2_K4_e100_lr0.01_bs32_cfg"), name)
        self.assertTrue(name.endswith("_seed42"), name)
        shifted = cfg.clone()
        shifted.data_preparation.dataset.split_root = "data/splits_shift/structural"
        self.assertNotEqual(GeoMoERunner(shifted).run_name, name)
        changed = cfg.clone()
        changed.moe.geomoe.eta = 0.2
        self.assertNotEqual(GeoMoERunner(changed).run_name, name)
        operational = cfg.clone()
        operational.moe.geomoe.prediction_dir = "/tmp/elsewhere"
        operational.moe.geomoe.orc_cache_dir = "/tmp/elsewhere"
        operational.moe.geomoe.num_runs = 2
        self.assertEqual(GeoMoERunner(operational).run_name, name)


if __name__ == "__main__":
    unittest.main()
