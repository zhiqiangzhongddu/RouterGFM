import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from src.config import cfg as base_cfg
from src.moe import run as moe_run
from src.moe.nodemoe.chebnet2 import (
    ChebIIProp,
    chebyshev_basis,
    chebyshev_nodes,
    init_filter_values,
)
from src.moe.nodemoe.model import NodeMoEModel, gate_input_features, resolve_expert_specs
from src.moe.nodemoe.run import _build_task_cfg, parse_nodemoe_tasks, run_nodemoe
from src.moe.nodemoe.task import NodeMoETask
from src.moe.nodemoe.trainer import NodeMoERunner

ROOT = Path(__file__).resolve().parents[1]
TSV_HEADER = "# dataset\ttask_level\ttask_type\tinduced\texpert_inits\tK\treadout\tfixed_split\tepochs\tbatch\tskip_if_exists\n"


def _cfg(tmp=None, **nodemoe):
    cfg = base_cfg.clone()
    cfg.seed = 42
    cfg.seeds = [42]
    n = cfg.moe.nodemoe
    n.dataset.name = "toy"
    n.dataset.task_level = "node"
    n.dataset.task_type = "classification"
    n.skip_if_exists = False
    if tmp is not None:
        n.checkpoint_dir = os.path.join(tmp, "ckpt")
        n.log_dir = os.path.join(tmp, "logs")
        cfg.save_results.output_dir = os.path.join(tmp, "results")
    for key, value in nodemoe.items():
        setattr(n, key, value)
    return cfg


def _model(in_dim=4, out_dim=3, K=4, dropout=0.0, inits=("low", "high", "uniform")):
    torch.manual_seed(0)
    return NodeMoEModel(
        in_dim, out_dim,
        expert_inits=inits, expert_alphas=(0.9,) * len(inits), K=K,
        expert_hidden_dim=8, expert_dropout=dropout, dprate=dropout,
        gate_hidden_dim=8, gate_num_layers=2, gate_dropout=dropout, gate_feature_norm="mean",
    )


def _random_graph(num_nodes, in_dim, gen, target=0, label=0):
    edge_index = torch.randint(0, num_nodes, (2, 2 * num_nodes), generator=gen)
    return Data(
        x=torch.randn(num_nodes, in_dim, generator=gen),
        edge_index=edge_index,
        y=torch.tensor([label]),
        target_node_index=torch.tensor([target]),
    )


def _ego_graphs(count=40, in_dim=4, neighbours=5, seed=0):
    """CSBM-style ego graphs: half homophilic, half heterophilic; label = target's class."""
    gen = torch.Generator().manual_seed(seed)
    direction = torch.ones(in_dim)
    graphs = []
    for i in range(count):
        label = i % 2
        mean = direction if label == 1 else -direction
        neighbour_mean = mean if i < count // 2 else -mean
        x = torch.cat([
            mean + 0.5 * torch.randn(1, in_dim, generator=gen),
            neighbour_mean + 0.5 * torch.randn(neighbours, in_dim, generator=gen),
        ])
        star = [[0] * neighbours, list(range(1, neighbours + 1))]
        ring = [list(range(1, neighbours + 1)), list(range(2, neighbours + 1)) + [1]]
        edge_index = torch.tensor([star[0] + ring[0], star[1] + ring[1]])
        graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([label]), target_node_index=torch.tensor([0])))
    return graphs


class ChebNetIITest(unittest.TestCase):
    def test_response_interpolates_relu_temp_at_chebyshev_nodes(self):
        torch.manual_seed(0)
        for K in (3, 10):
            prop = ChebIIProp(K=K)
            prop.temp.data = torch.randn(K + 1)
            response = prop.response(chebyshev_nodes(K) + 1.0)
            self.assertTrue(torch.allclose(response, torch.relu(prop.temp), atol=1e-5), K)

    def test_filter_init_shapes_and_smoothing(self):
        lam = torch.linspace(0.0, 2.0, 201)
        with torch.no_grad():
            low = ChebIIProp(10, "low", 0.9)
            high = ChebIIProp(10, "high", 0.9)
            uniform = ChebIIProp(10, "uniform", 0.9)
            self.assertTrue(bool((low.response(lam).diff() <= 1e-6).all()))
            self.assertTrue(bool((high.response(lam).diff() >= -1e-6).all()))
            self.assertTrue(torch.allclose(uniform.response(lam), torch.ones_like(lam), atol=1e-5))
            self.assertAlmostEqual(float(low.response(torch.tensor([0.0]))), 1.02, places=2)
            self.assertAlmostEqual(float(low.response(torch.tensor([2.0]))), 0.34, places=2)
        self.assertEqual(float(uniform.smoothing_loss()), 0.0)
        self.assertGreater(float(low.smoothing_loss()), 0.0)
        self.assertGreater(float(high.smoothing_loss()), 0.0)
        self.assertTrue(torch.allclose(init_filter_values("high", 3, 0.5), torch.tensor([0.125, 0.25, 0.5, 1.0])))
        with self.assertRaises(ValueError):
            init_filter_values("bandpass", 3, 0.9)

    def test_propagation_matches_dense_and_spectral_reference(self):
        torch.manual_seed(1)
        num_nodes = 7  # node 6 is isolated
        edges = torch.tensor([[0, 1, 2, 3, 4, 0, 1], [1, 2, 3, 4, 5, 2, 4]])
        edge_index = to_undirected(edges, num_nodes=num_nodes)
        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float64)
        adj[edge_index[0], edge_index[1]] = 1.0
        deg = adj.sum(1)
        inv_sqrt = torch.where(deg > 0, deg.clamp(min=1).rsqrt(), torch.zeros_like(deg))
        norm_adj = inv_sqrt[:, None] * adj * inv_sqrt[None, :]
        l_hat = -norm_adj
        l_sym = torch.eye(num_nodes, dtype=torch.float64) - norm_adj

        K = 6
        prop = ChebIIProp(K=K)
        prop.temp.data = torch.rand(K + 1) * 2.0
        x = torch.randn(num_nodes, 3)
        out = prop(x, edge_index).double()

        coe = prop.coefficients().detach().double()
        xd = x.double()
        t_prev, t_cur = xd, l_hat @ xd
        dense = coe[0] / 2 * t_prev + coe[1] * t_cur
        for k in range(2, K + 1):
            t_prev, t_cur = t_cur, 2 * l_hat @ t_cur - t_prev
            dense = dense + coe[k] * t_cur
        self.assertTrue(torch.allclose(out, dense, atol=1e-4))

        eigvals, eigvecs = torch.linalg.eigh(l_sym)
        response = prop.response(eigvals.float()).detach().double()
        spectral = eigvecs @ torch.diag(response) @ eigvecs.T @ xd
        self.assertTrue(torch.allclose(out, spectral, atol=1e-4))

    def test_chebyshev_basis_recurrence(self):
        x = torch.tensor([-0.5, 0.0, 0.7])
        basis = chebyshev_basis(x, 3)
        self.assertTrue(torch.allclose(basis[:, 3], 4 * x ** 3 - 3 * x))


class NodeMoEModelTest(unittest.TestCase):
    def test_gate_input_features_mean_normalisation(self):
        x = torch.tensor([[1.0], [2.0], [4.0], [3.0]])  # path 0-1-2, node 3 isolated
        edge_index = to_undirected(torch.tensor([[0, 1], [1, 2]]), num_nodes=4)
        z = gate_input_features(x, edge_index, 4, norm="mean")
        expected = torch.tensor([
            [1.0, 1.0, 1.5],
            [2.0, 0.5, 0.0],
            [4.0, 2.0, 1.5],
            [3.0, 3.0, 3.0],
        ])
        self.assertTrue(torch.allclose(z, expected))
        with self.assertRaises(ValueError):
            gate_input_features(x, edge_index, 4, norm="rw")

    def test_block_diagonal_batches_match_single_graphs(self):
        gen = torch.Generator().manual_seed(3)
        graphs = [_random_graph(5, 4, gen), _random_graph(7, 4, gen)]
        model = _model(dropout=0.5).eval()
        with torch.no_grad():
            mixed, gate = model(Batch.from_data_list(graphs))
            singles = [model(graph) for graph in graphs]
        self.assertTrue(torch.allclose(mixed, torch.cat([s[0] for s in singles]), atol=1e-5))
        self.assertTrue(torch.allclose(gate, torch.cat([s[1] for s in singles]), atol=1e-6))
        self.assertTrue(torch.allclose(gate.sum(-1), torch.ones(gate.size(0)), atol=1e-6))
        self.assertEqual(tuple(mixed.shape), (12, 3))

    def test_directed_input_is_symmetrised(self):
        gen = torch.Generator().manual_seed(4)
        graph = Data(x=torch.randn(6, 4, generator=gen), edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]))
        undirected = Data(x=graph.x, edge_index=to_undirected(graph.edge_index, num_nodes=6))
        model = _model().eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(graph)[0], model(undirected)[0], atol=1e-6))

    def test_param_groups_partition_parameters(self):
        model = _model()
        groups = model.param_groups(gate_lr=1e-3, gate_wd=0.0, expert_lr=1e-2, expert_wd=5e-4, filter_lr=1e-1, filter_wd=0.0)
        grouped = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(sorted(grouped), sorted(id(p) for p in model.parameters()))
        self.assertEqual(len(groups[2]["params"]), 3)
        self.assertEqual([group["lr"] for group in groups], [1e-3, 1e-2, 1e-1])

    def test_expert_spec_validation(self):
        self.assertEqual(resolve_expert_specs(("LOW", "high"), (0.9, 0.8)), [("low", 0.9), ("high", 0.8)])
        with self.assertRaisesRegex(ValueError, "expert_alphas"):
            resolve_expert_specs(("low", "high"), (0.9, 0.9, 0.9))
        with self.assertRaises(ValueError):
            resolve_expert_specs(("low",), (0.9,))
        with self.assertRaises(ValueError):
            resolve_expert_specs(("low", "band"), (0.9, 0.9))


class _FixedOutputs(nn.Module):
    def __init__(self, mixed, gate):
        super().__init__()
        self.mixed, self.gate = mixed, gate

    def forward(self, data):
        return self.mixed, self.gate


class NodeMoETaskTest(unittest.TestCase):
    def _batch(self):
        gen = torch.Generator().manual_seed(5)
        sizes, targets = (3, 4, 2), (1, 0, 1)
        graphs = [_random_graph(n, 4, gen, target=t, label=i) for i, (n, t) in enumerate(zip(sizes, targets))]
        return Batch.from_data_list(graphs)  # batched targets: [1, 3, 8]

    def test_target_readout_selects_only_target_rows(self):
        batch = self._batch()
        task = NodeMoETask(_cfg())
        for sentinel_node, expected_row in ((3, 1), (4, None)):
            mixed = torch.zeros(batch.num_nodes, 3)
            mixed[sentinel_node] = 1e6
            fake = _FixedOutputs(mixed, torch.full((batch.num_nodes, 2), 0.5))
            _, _, logits, labels = task.evaluate(fake, batch, torch.device("cpu"), return_outputs=True)
            self.assertEqual(tuple(logits.shape), (3, 3))
            self.assertEqual(labels.tolist(), [0, 1, 2])
            rows = [i for i in range(3) if bool((logits[i] == 1e6).all())]
            self.assertEqual(rows, [] if expected_row is None else [expected_row])

    def test_mean_and_full_graph_readouts(self):
        batch = self._batch()
        mixed = torch.arange(batch.num_nodes * 3, dtype=torch.float32).view(-1, 3)
        fake = _FixedOutputs(mixed, torch.full((batch.num_nodes, 2), 0.5))
        task = NodeMoETask(_cfg(readout="mean"))
        _, _, logits, _ = task.evaluate(fake, batch, torch.device("cpu"), return_outputs=True)
        self.assertTrue(torch.allclose(logits[0], mixed[:3].mean(0)))

        full = Data(x=torch.zeros(4, 1), edge_index=torch.empty(2, 0, dtype=torch.long), y=torch.tensor([0, 1, 1, 0]),
                    train_mask=torch.tensor([True, False, True, False]))
        cfg = _cfg()
        cfg.moe.nodemoe.dataset.induced = False
        fake = _FixedOutputs(torch.randn(4, 2), torch.full((4, 2), 0.5))
        _, _, logits, labels = NodeMoETask(cfg).evaluate(fake, full, torch.device("cpu"), mask_attr="train_mask", return_outputs=True)
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertTrue(torch.equal(logits, fake.mixed[[0, 2]]))

    def test_gate_tracking_reports_mean_readout_gate(self):
        batch = self._batch()
        gate = torch.tensor([[1.0, 0.0]] * batch.num_nodes)
        gate[3] = torch.tensor([0.0, 1.0])  # graph 1's target
        task = NodeMoETask(_cfg())
        task.track_gate_weights(True)
        task.evaluate(_FixedOutputs(torch.zeros(batch.num_nodes, 3), gate), batch, torch.device("cpu"))
        mean_gate = task.mean_gate_weights()
        task.track_gate_weights(False)
        self.assertEqual(len(mean_gate), 2)
        self.assertAlmostEqual(mean_gate[0], 2 / 3)
        self.assertIsNone(task.mean_gate_weights())

    def test_scope_guard_rejects_non_node_tasks(self):
        for level in ("edge", "graph"):
            cfg = _cfg()
            cfg.moe.nodemoe.dataset.task_level = level
            with self.assertRaisesRegex(ValueError, "node-task-scoped"):
                NodeMoETask(cfg)
            with self.assertRaisesRegex(ValueError, "node-task-scoped"):
                NodeMoERunner(cfg)

    def test_learning_signal_reaches_every_parameter_group(self):
        torch.manual_seed(0)
        batch = Batch.from_data_list(_ego_graphs())
        model = _model(out_dim=2, dropout=0.0)
        task = NodeMoETask(_cfg(smoothing_gamma=0.1))
        groups = model.param_groups(gate_lr=0.01, gate_wd=5e-4, expert_lr=0.01, expert_wd=5e-4, filter_lr=0.01, filter_wd=5e-4)
        optimizer = torch.optim.Adam(groups)
        device = torch.device("cpu")
        model.eval()
        with torch.no_grad():
            loss_before = float(task.evaluate(model, batch, device)[0])
        model.train()
        for step in range(30):
            optimizer.zero_grad()
            loss, log = task.step(model, batch, device)
            loss.backward()
            if step == 0:
                self.assertIn("train_smooth_loss", log)
                for group in groups:
                    grad_mass = sum(float(p.grad.abs().sum()) for p in group["params"] if p.grad is not None)
                    self.assertGreater(grad_mass, 0.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss_after = float(task.evaluate(model, batch, device)[0])
        self.assertLess(loss_after, 0.7 * loss_before)


class NodeMoEPlumbingTest(unittest.TestCase):
    def test_config_defaults(self):
        n = base_cfg.moe.nodemoe
        self.assertEqual(tuple(n.expert_inits), ("low", "high", "uniform"))
        self.assertEqual(len(n.expert_alphas), len(n.expert_inits))
        self.assertEqual((n.K, n.readout, n.dataset.task_level), (10, "target", "node"))
        self.assertEqual(n.tasks_tsv, "slurm/moe.nodemoe.all.tsv")

    def test_parse_and_build_task_cfg(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.tsv"
            path.write_text(
                TSV_HEADER
                + "chameleon\tnode\tclassification\tTrue\tlow,high\t5\tmean\t(5,0.0,1.0)\t7\t16\tFalse\n"
                + "chameleon\tnode\tclassification\tTrue\tlow,bogus\t5\tmean\t(5,0.0,1.0)\t7\t16\tFalse\n"
                + "photo\tnode\tclassification\tTrue\n",
                encoding="utf-8",
            )
            tasks = parse_nodemoe_tasks(str(path))
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["expert_inits"], ("low", "high"))
        self.assertEqual((tasks[0]["k"], tasks[0]["readout"]), (5, "mean"))
        self.assertIsNone(tasks[1]["expert_inits"])

        run_cfg = _build_task_cfg(_cfg(), tasks[0])
        n = run_cfg.moe.nodemoe
        self.assertEqual((n.dataset.name, n.K, n.readout, n.epochs, n.batch_size), ("chameleon", 5, "mean", 7, 16))
        self.assertEqual(tuple(n.expert_inits), ("low", "high"))
        self.assertEqual(tuple(n.dataset.fixed_split), (5, 0.0, 1.0))
        self.assertFalse(n.run_tasks_tsv)
        explicit = run_cfg.save_results.explicit_keys
        for key in ("moe.method", "moe.nodemoe.K", "moe.nodemoe.expert_inits", "moe.nodemoe.dataset.fixed_split"):
            self.assertIn(key, explicit)

    def test_repository_tsv_covers_node_datasets_and_budgets(self):
        tasks = parse_nodemoe_tasks(str(ROOT / "slurm" / "moe.nodemoe.all.tsv"))
        self.assertEqual(len(tasks), 8)
        self.assertEqual({t["dataset"] for t in tasks}, {"photo", "ogbn-arxiv", "airports", "chameleon"})
        self.assertEqual({t["task_level"] for t in tasks}, {"node"})
        self.assertEqual({tuple(t["fixed_split"]) for t in tasks}, {(5, 0.0, 1.0), (100, 0.0, 1.0)})
        self.assertEqual({t["expert_inits"] for t in tasks}, {("low", "high", "uniform")})

    def test_run_moe_dispatches_nodemoe(self):
        cfg = moe_run._build_moe_cfg(["moe.nodemoe.dataset.name", "photo"])
        self.assertEqual(cfg.moe.method, "nodemoe")
        with patch("src.moe.nodemoe.run_nodemoe", return_value=0) as fake:
            self.assertEqual(moe_run.run_moe(cfg), 0)
        fake.assert_called_once()

    def test_tsv_mode_rejects_edge_rows_without_loading_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.tsv"
            path.write_text(TSV_HEADER + "cornell\tedge\tclassification\tTrue\n", encoding="utf-8")
            cfg = _cfg(tmp, run_tasks_tsv=True, tasks_tsv=str(path), num_runs=1)
            stdout = io.StringIO()
            with patch("src.moe.nodemoe.trainer.create_dataset") as create, redirect_stdout(stdout):
                self.assertEqual(run_nodemoe(cfg), 1)
            create.assert_not_called()
            self.assertIn("node-task-scoped", stdout.getvalue())

    def test_run_name_identity(self):
        cfg = _cfg()
        name = NodeMoERunner(cfg).run_name
        self.assertTrue(name.startswith("nodemoe_toy_induced1_"), name)
        self.assertIn("_tasknode_m3_K10_eh64_g0.1_target_e100_bs32_cfg", name)
        self.assertTrue(name.endswith("_seed42"), name)
        changed = cfg.clone()
        changed.moe.nodemoe.smoothing_gamma = 0.01
        self.assertNotEqual(NodeMoERunner(changed).run_name, name)
        operational = cfg.clone()
        operational.moe.nodemoe.checkpoint_dir = "/tmp/elsewhere"
        operational.moe.nodemoe.num_runs = 2
        self.assertEqual(NodeMoERunner(operational).run_name, name)


class NodeMoERunnerSmokeTest(unittest.TestCase):
    def test_run_seeds_trains_checkpoints_and_skips_existing(self):
        graphs = _ego_graphs(count=16, in_dim=4)
        meta = {"num_node_features": 4, "num_classes": 2, "label_dim": 1, "task_type": "classification"}

        def _loaders(**_kwargs):
            return (
                DataLoader(graphs[:8], batch_size=4, shuffle=True),
                DataLoader([], batch_size=4),
                DataLoader(graphs[8:], batch_size=4),
            )

        with tempfile.TemporaryDirectory() as tmp, patch.multiple(
            "src.moe.nodemoe.trainer",
            create_dataset=lambda **_kwargs: graphs,
            dataset_info=lambda **_kwargs: dict(meta),
            make_workflow_loaders=_loaders,
            log_split_instance_counts=lambda *args, **kwargs: None,
        ):
            cfg = _cfg(tmp, epochs=3, early_stopping=0, K=3, expert_hidden_dim=8, gate_hidden_dim=8,
                       num_runs=1, batch_size=4)
            cfg.moe.nodemoe.dataset.fixed_split = (4, 0.0, 1.0)
            self.assertEqual(run_nodemoe(cfg), 0)

            runner = NodeMoERunner(cfg)
            ckpt = torch.load(runner.get_checkpoint_path_for_metrics(), map_location="cpu")
            self.assertIn("test_acc", ckpt["metrics"])
            self.assertEqual(len(ckpt["extra"]["nodemoe_filter_temps"]), 3)
            self.assertAlmostEqual(sum(ckpt["extra"]["nodemoe_test_mean_gate"]), 1.0, places=5)
            self.assertTrue(os.path.isfile(runner._log_path()))
            table = (Path(tmp) / "results" / "moe_nodemoe.tsv").read_text(encoding="utf-8")
            self.assertIn("moe.nodemoe.dataset.fixed_split", table.splitlines()[0])

            cfg.moe.nodemoe.skip_if_exists = True
            self.assertTrue(NodeMoERunner(cfg)._skip_due_to_existing_checkpoint)


if __name__ == "__main__":
    unittest.main()
