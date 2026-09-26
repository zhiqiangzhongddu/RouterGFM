import csv
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import is_undirected, to_undirected

from src.config import cfg as base_cfg
from src.data_loader.shift_splits import split_file_path
from src.moe import run as moe_run
from src.moe.graphmetro.model import GraphMETROModel
from src.moe.graphmetro.run import _build_task_cfg, parse_graphmetro_tasks, run_graphmetro
from src.moe.graphmetro.task import GraphMETROTask
from src.moe.graphmetro.trainer import GraphMETRORunner
from src.moe.graphmetro.transforms import (
    TRANSFORM_NAMES,
    apply_shift,
    expert_names,
    parse_shift_list,
    shift_target,
)
from src.utils.supervised_loss import build_supervised_head

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")
IN_DIM = 4
TSV_HEADER = "# dataset\ttask_level\ttask_type\tinduced\tfixed_split\tsplit_root\tbackbone\tmoe_lr\tepochs\tbatch\tskip_if_exists\n"


# --------------------------------------------------------------------------- #
# Synthetic instances shaped like the repo's induced node / SEAL edge / graph data
# --------------------------------------------------------------------------- #
def _random_edges(n, gen, exclude=None):
    edge_index = torch.randint(0, n, (2, 2 * n), generator=gen)
    edge_index = to_undirected(edge_index[:, edge_index[0] != edge_index[1]], num_nodes=n)
    if exclude is not None:
        u, v = exclude
        pair = ((edge_index[0] == u) & (edge_index[1] == v)) | ((edge_index[0] == v) & (edge_index[1] == u))
        edge_index = edge_index[:, ~pair]
    return edge_index


def _node_graphs(count=12, seed=0, num_classes=3):
    gen = torch.Generator().manual_seed(seed)
    graphs = []
    for i in range(count):
        n = 6 + i % 5
        graph = Data(
            x=torch.randn(n, IN_DIM, generator=gen),
            edge_index=_random_edges(n, gen),
            y=torch.tensor(i % num_classes),
            target_node_index=torch.tensor([i % n]),
        )
        graph.base_node_id = 100 + i
        graph.index = 100 + i
        graphs.append(graph)
    return graphs


def _edge_graphs(count=12, seed=1):
    gen = torch.Generator().manual_seed(seed)
    graphs = []
    for i in range(count):
        n = 5 + i % 4
        graphs.append(Data(
            x=torch.randn(n, IN_DIM, generator=gen),
            edge_index=_random_edges(n, gen, exclude=(0, 1)),
            edge_label_index=torch.tensor([[0], [1]]),
            global_target_pair=torch.tensor([10 * i, 10 * i + 1]),
            y=torch.tensor(i % 2),
        ))
    return graphs


def _graph_graphs(count=12, seed=2, kind="cls"):
    gen = torch.Generator().manual_seed(seed)
    graphs = []
    for i in range(count):
        n = 1 if i == 0 else 4 + i % 5  # one single-node graph
        edge_index = _random_edges(n, gen) if n > 1 else torch.empty(2, 0, dtype=torch.long)
        if kind == "cls":
            y = torch.tensor([i % 3])
        elif kind == "multilabel":
            y = torch.tensor([[float((i + j) % 3 - 1) for j in range(5)]])
        else:
            y = torch.randn(1, 2, generator=gen) * 10.0 + 50.0
        graphs.append(Data(
            x=torch.randn(n, IN_DIM, generator=gen),
            pos=torch.arange(n, dtype=torch.float).unsqueeze(-1),
            edge_index=edge_index,
            edge_attr=torch.ones(edge_index.size(1), 3),
            y=y,
        ))
    return graphs


def _instances(level):
    return {"node": _node_graphs, "edge": _edge_graphs, "graph": _graph_graphs}[level]()


def _cfg(tmp=None, level="node", **graphmetro):
    cfg = base_cfg.clone()
    cfg.seed = 42
    cfg.seeds = [42]
    gm = cfg.moe.graphmetro
    gm.dataset.name = "toy"
    gm.dataset.task_level = level
    gm.dataset.task_type = "classification"
    gm.skip_if_exists = False
    gm.hidden_dim = 8
    gm.num_layers = 2
    gm.dropout = 0.0
    if tmp is not None:
        gm.checkpoint_dir = os.path.join(tmp, "ckpt")
        gm.log_dir = os.path.join(tmp, "logs")
        gm.prediction_dir = os.path.join(tmp, "pred")
        cfg.save_results.output_dir = os.path.join(tmp, "results")
    for key, value in graphmetro.items():
        setattr(gm, key, value)
    return cfg


def _model(level="node", out_dim=3, num_experts=6, backbone="gcn"):
    torch.manual_seed(0)
    head = build_supervised_head(in_dim=8, task_type="classification", task_level="graph", label_dim=1, num_classes=out_dim)
    return GraphMETROModel(
        in_dim=IN_DIM, backbone=backbone, num_layers=2, hidden_dim=8, dropout=0.0, use_batchnorm=True,
        num_experts=num_experts, task_level_raw=level, graph_pooling="mean", head=head,
    )


def _edges(edge_index):
    return set(map(tuple, edge_index.t().tolist()))


# --------------------------------------------------------------------------- #
# Shift list / targets
# --------------------------------------------------------------------------- #
class ShiftListTest(unittest.TestCase):
    def test_default_is_official_good_paired_list(self):
        shifts = parse_shift_list(base_cfg.moe.graphmetro.shift_train_types)
        self.assertEqual(len(shifts), 14)
        self.assertEqual(sum(len(s) == 1 for s in shifts), 5)
        self.assertIn(("drop_node", "random_subgraph"), shifts)
        self.assertEqual(expert_names(shifts), ["id", *TRANSFORM_NAMES])

    def test_shift_targets_are_multi_hot(self):
        names = expert_names(parse_shift_list(base_cfg.moe.graphmetro.shift_train_types))
        pair = shift_target(("noisy_node_feat", "drop_edge"), names)
        self.assertEqual(pair.tolist(), [0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(shift_target(("id",), names).tolist(), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(expert_names(parse_shift_list("drop_edge/add_edge")), ["id", "add_edge", "drop_edge"])

    def test_invalid_shift_lists_raise(self):
        for spec in ("flip_edge", "id-add_edge", "", " / "):
            with self.assertRaises(ValueError, msg=spec):
                parse_shift_list(spec)


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
class TransformInvariantTest(unittest.TestCase):
    def _check(self, level, batch, out, shift):
        self.assertEqual(out.num_graphs, batch.num_graphs)
        self.assertTrue(torch.equal(out.y, batch.y))
        src, dst = out.edge_index
        self.assertTrue(bool((out.edge_index < out.num_nodes).all()) if out.edge_index.numel() else True)
        self.assertTrue(torch.equal(out.batch[src], out.batch[dst]), "edges must stay inside their instance")
        self.assertFalse("edge_attr" in out)
        sizes = out.ptr.diff()
        self.assertTrue(bool((sizes >= 1).all()), "no empty instance")
        noisy = "noisy_node_feat" in shift
        if level == "node":
            same = torch.allclose(out.x[out.target_node_index], batch.x[batch.target_node_index])
            self.assertEqual(same, not noisy)
            self.assertTrue(torch.equal(out.base_node_id, batch.base_node_id))
        elif level == "edge":
            for side in (0, 1):
                same = torch.allclose(out.x[out.edge_label_index[side]], batch.x[batch.edge_label_index[side]])
                self.assertEqual(same, not noisy)
            edges = _edges(out.edge_index)
            for u, v in out.edge_label_index.t().tolist():
                self.assertNotIn((u, v), edges)
                self.assertNotIn((v, u), edges)
            self.assertTrue(torch.equal(out.global_target_pair, batch.global_target_pair))
        else:
            self.assertEqual(out.pos.size(0), out.num_nodes)

    def test_every_training_shift_keeps_instances_valid(self):
        shifts = parse_shift_list(base_cfg.moe.graphmetro.shift_train_types)
        gen = torch.Generator().manual_seed(0)
        for level in ("node", "edge", "graph"):
            batch = Batch.from_data_list(_instances(level))
            for shift in shifts:
                for _ in range(5):
                    out = apply_shift(batch, shift, p=0.5, k=2, task_level_raw=level, generator=gen)
                    self._check(level, batch, out, shift)

    def test_deterministic_and_input_untouched(self):
        batch = Batch.from_data_list(_node_graphs())
        x, edge_index = batch.x.clone(), batch.edge_index.clone()
        shift = ("noisy_node_feat", "drop_node")
        runs = [
            apply_shift(batch, shift, p=0.5, k=2, task_level_raw="node", generator=torch.Generator().manual_seed(seed))
            for seed in (3, 3, 4)
        ]
        self.assertTrue(torch.equal(runs[0].x, runs[1].x))
        self.assertTrue(torch.equal(runs[0].edge_index, runs[1].edge_index))
        self.assertFalse(runs[0].x.shape == runs[2].x.shape and torch.equal(runs[0].x, runs[2].x))
        self.assertTrue(torch.equal(batch.x, x))
        self.assertTrue(torch.equal(batch.edge_index, edge_index))


class TransformSemanticsTest(unittest.TestCase):
    @staticmethod
    def _path(n=7, directed=False, target=3):
        edge_index = torch.tensor([list(range(n - 1)), list(range(1, n))])
        if not directed:
            edge_index = to_undirected(edge_index, num_nodes=n)
        graph = Data(x=torch.arange(n, dtype=torch.float).unsqueeze(-1), edge_index=edge_index,
                     y=torch.tensor(0), target_node_index=torch.tensor([target]))
        return Batch.from_data_list([graph])

    def test_add_edge_matches_add_random_edge_semantics(self):
        batch = self._path()  # 12 directed edges
        gen = torch.Generator().manual_seed(0)
        out = apply_shift(batch, ("add_edge",), p=0.5, k=2, task_level_raw="node", generator=gen)
        before, after = _edges(batch.edge_index), _edges(out.edge_index)
        self.assertTrue(before <= after)
        self.assertEqual(len(after) - len(before), 6)  # round(12 * 0.5) = 6, as 3 undirected pairs
        self.assertEqual(out.edge_index.size(1), len(after), "no duplicate edges")
        self.assertTrue(all(u != v for u, v in after))
        self.assertTrue(is_undirected(out.edge_index))

        directed = self._path(directed=True)  # 6 directed edges -> 3 new directed edges
        out = apply_shift(directed, ("add_edge",), p=0.5, k=2, task_level_raw="node", generator=gen)
        self.assertEqual(len(_edges(out.edge_index)) - 6, 3)

    def test_add_edge_never_inserts_the_lp_target_pair(self):
        graph = Data(x=torch.randn(3, 2), edge_index=to_undirected(torch.tensor([[0, 1], [2, 2]])),
                     edge_label_index=torch.tensor([[0], [1]]), y=torch.tensor(1))
        batch = Batch.from_data_list([graph])
        gen = torch.Generator().manual_seed(0)
        for _ in range(30):
            out = apply_shift(batch, ("add_edge",), p=0.9, k=2, task_level_raw="edge", generator=gen)
            self.assertNotIn((0, 1), _edges(out.edge_index))
            self.assertNotIn((1, 0), _edges(out.edge_index))

    def test_drop_edge_keeps_a_subset(self):
        batch = self._path()
        out = apply_shift(batch, ("drop_edge",), p=0.5, k=2, task_level_raw="node",
                          generator=torch.Generator().manual_seed(0))
        self.assertTrue(_edges(out.edge_index) <= _edges(batch.edge_index))
        self.assertLess(out.edge_index.size(1), batch.edge_index.size(1))
        self.assertEqual(out.num_nodes, batch.num_nodes)

    def test_drop_node_keeps_targets_and_relabels(self):
        gen = torch.Generator().manual_seed(0)
        batch = self._path(n=12, target=7)
        for _ in range(20):
            out = apply_shift(batch, ("drop_node",), p=0.9, k=2, task_level_raw="node", generator=gen)
            self.assertEqual(float(out.x[out.target_node_index].item()), 7.0)
            kept = out.x.view(-1).long()
            self.assertEqual(_edges(out.edge_index), {
                (i, j) for i, a in enumerate(kept.tolist()) for j, b in enumerate(kept.tolist()) if abs(a - b) == 1
            })
        single = Batch.from_data_list([Data(x=torch.ones(1, 1), edge_index=torch.empty(2, 0, dtype=torch.long), y=torch.tensor([0]))])
        out = apply_shift(single, ("drop_node",), p=0.9, k=2, task_level_raw="graph", generator=gen)
        self.assertEqual(out.num_nodes, 1)

    def test_random_subgraph_is_the_bidirectional_ball_around_the_target(self):
        for directed in (False, True):
            batch = self._path(directed=directed)
            out = apply_shift(batch, ("random_subgraph",), p=0.5, k=1, task_level_raw="node",
                              generator=torch.Generator().manual_seed(0))
            self.assertEqual(sorted(out.x.view(-1).tolist()), [2.0, 3.0, 4.0])
            self.assertEqual(float(out.x[out.target_node_index].item()), 3.0)

    def test_noisy_node_feat_scales_with_the_batch_std(self):
        graphs = [Data(x=torch.stack([torch.full((5,), 2.0), torch.randn(5)], dim=1),
                       edge_index=torch.empty(2, 0, dtype=torch.long), y=torch.tensor([0])) for _ in range(3)]
        batch = Batch.from_data_list(graphs)
        out = apply_shift(batch, ("noisy_node_feat",), p=0.5, k=2, task_level_raw="graph",
                          generator=torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(out.x[:, 0], batch.x[:, 0]), "a zero-std dimension gets no noise")
        self.assertFalse(torch.allclose(out.x[:, 1], batch.x[:, 1]))


# --------------------------------------------------------------------------- #
# Model / objective
# --------------------------------------------------------------------------- #
class GraphMETROModelTest(unittest.TestCase):
    def test_shapes_mixture_and_param_groups(self):
        model = _model().eval()
        batch = Batch.from_data_list(_node_graphs())
        with torch.no_grad():
            gate = model.gate_logits(batch)
            reprs = model.expert_reprs(batch)
            logits, weights = model(batch)
            onehot = F.one_hot(torch.full((batch.num_graphs,), 2), 6).float()
            self.assertTrue(torch.allclose(model.mix(reprs, onehot), model.instance_repr(model.experts[2], batch)))
            node_repr, _ = model.experts[0](batch)
            self.assertTrue(torch.allclose(reprs[:, 0], node_repr[batch.target_node_index]))
        self.assertEqual(tuple(gate.shape), (12, 6))
        self.assertEqual(tuple(reprs.shape), (12, 6, 8))
        self.assertEqual(tuple(logits.shape), (12, 3))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(12)))

        groups = model.param_groups(moe_lr=1e-2, classifier_lr=1e-4)
        grouped = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(sorted(grouped), sorted(id(p) for p in model.parameters()))
        self.assertEqual([group["lr"] for group in groups], [1e-2, 1e-2, 1e-4])
        self.assertEqual({id(p) for p in groups[2]["params"]}, {id(p) for p in model.head.parameters()})


class GraphMETROTaskTest(unittest.TestCase):
    def _setup(self, **kwargs):
        task = GraphMETROTask(_cfg(**kwargs))
        model = _model(num_experts=len(task.expert_names))
        batch = Batch.from_data_list(_node_graphs())
        return task, model, batch

    def _terms(self, task, model, batch, idx=5):
        shifted = apply_shift(batch, task.shifts[idx], p=task.p, k=task.k, task_level_raw="node",
                              generator=torch.Generator().manual_seed(1))
        z0 = model.instance_repr(model.experts[0], batch)
        return shifted, z0, task.shift_terms(model, shifted, idx, z0, batch.y)

    @staticmethod
    def _grad_mass(module):
        return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)

    def test_gradient_isolation(self):
        task, model, batch = self._setup()
        model.train()
        _, _, terms = self._terms(task, model, batch)
        (terms["task"] + terms["align"]).backward()
        self.assertEqual(self._grad_mass(model.gate) + self._grad_mass(model.gate_head), 0.0)
        self.assertGreater(self._grad_mass(model.experts), 0.0)
        self.assertGreater(self._grad_mass(model.head), 0.0)

        model.zero_grad(set_to_none=True)
        _, _, terms = self._terms(task, model, batch)
        terms["gate"].backward()
        self.assertEqual(self._grad_mass(model.experts) + self._grad_mass(model.head), 0.0)
        self.assertGreater(self._grad_mass(model.gate_head), 0.0)

    def test_gate_pos_weight_and_alignment_values(self):
        task, model, batch = self._setup()
        model.eval()
        with torch.no_grad():
            shifted, z0, terms = self._terms(task, model, batch, idx=5)
            logits = model.gate_logits(shifted)
            target = task.shift_targets[5].expand_as(logits)
            self.assertEqual(target[0].tolist(), [0.0, 1.0, 1.0, 0.0, 0.0, 0.0])  # noisy_node_feat-add_edge
            expected = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.full((6,), 4.0))
            self.assertAlmostEqual(float(terms["gate"]), float(expected), places=5)
            self.assertNotAlmostEqual(float(terms["gate"]), float(F.binary_cross_entropy_with_logits(logits, target)), places=4)
            h = model.mix(model.expert_reprs(shifted), torch.softmax(logits, dim=-1))
            self.assertAlmostEqual(float(terms["align"]), float(torch.linalg.norm(h - z0) / batch.num_graphs), places=5)

    def test_align_lambda_adds_exactly_the_alignment_term(self):
        losses, logs = [], []
        for lam in (0.0, 1.0):
            task, model, batch = self._setup(align_lambda=lam)
            model.eval()
            with torch.no_grad():
                loss, log = task.step(model, batch, CPU)
            losses.append(float(loss))
            logs.append(log)
        self.assertAlmostEqual(losses[1] - losses[0], logs[0]["train_align_loss"], places=4)
        self.assertAlmostEqual(losses[0], logs[0]["train_gate_loss"] + logs[0]["train_task_loss"], places=4)
        self.assertEqual(set(logs[0]), {"train_gate_loss", "train_task_loss", "train_align_loss", "train_gate_acc", "train_acc"})

    def test_evaluate_uses_the_clean_queries(self):
        task, model, batch = self._setup()
        model.eval()
        with torch.no_grad():
            _, _, logits, labels = task.evaluate(model, batch, CPU, return_outputs=True)
            self.assertTrue(torch.allclose(logits, model(batch)[0]))
        self.assertTrue(torch.equal(labels, batch.y))

    def test_invalid_shift_parameters_raise(self):
        for kwargs in ({"shift_p": 1.0}, {"shift_k": -1}, {"num_shift_samples": 0}):
            with self.assertRaises(ValueError, msg=kwargs):
                GraphMETROTask(_cfg(**kwargs))


# --------------------------------------------------------------------------- #
# Runner / orchestration
# --------------------------------------------------------------------------- #
_FAMILIES = {
    # level, graphs, meta, split, task metric
    "node": ("node", _node_graphs, {"num_classes": 3, "label_dim": 1, "task_type": "classification"}, (2, 0.0, 1.0), "test_acc"),
    "link": ("edge", _edge_graphs, {"num_classes": 2, "label_dim": 1, "task_type": "classification"}, (0.1, 0.05, 0.1), "test_auc"),
    "graph": ("graph", lambda: _graph_graphs(kind="cls"), {"num_classes": 3, "label_dim": 1, "task_type": "classification"}, (2, 0.0, 1.0), "test_acc"),
    "multilabel": ("graph", lambda: _graph_graphs(kind="multilabel"), {"num_classes": 2, "label_dim": 5, "task_type": "classification"}, (2, 0.0, 1.0), "test_auc"),
    "regression": ("graph", lambda: _graph_graphs(kind="reg"), {"num_classes": None, "label_dim": 2, "task_type": "regression"}, (2, 0.0, 1.0), "test_mae"),
}


def _patched_data(graphs, meta):
    def _loaders(**_kwargs):
        return (
            DataLoader(Subset(graphs, list(range(6))), batch_size=4, shuffle=True),
            DataLoader(Subset(graphs, [6, 7]), batch_size=4),
            DataLoader(Subset(graphs, list(range(8, 12))), batch_size=4),
        )

    return patch.multiple(
        "src.moe.graphmetro.trainer",
        create_dataset=lambda **_kwargs: graphs,
        dataset_info=lambda **_kwargs: {"num_node_features": IN_DIM, **meta},
        make_workflow_loaders=_loaders,
        log_split_instance_counts=lambda *args, **kwargs: None,
    )


class GraphMETRORunnerSmokeTest(unittest.TestCase):
    def test_every_task_family_trains_scores_brier_and_saves_predictions(self):
        for family, (level, make, meta, split, metric) in _FAMILIES.items():
            with self.subTest(family=family), tempfile.TemporaryDirectory() as tmp:
                graphs = make()
                with _patched_data(graphs, meta):
                    cfg = _cfg(tmp, level=level, epochs=2, batch_size=4, num_runs=1)
                    cfg.moe.graphmetro.dataset.task_type = meta["task_type"]
                    cfg.moe.graphmetro.dataset.fixed_split = split
                    self.assertEqual(run_graphmetro(cfg), 0)
                    runner = GraphMETRORunner(cfg)

                ckpt = torch.load(runner.get_checkpoint_path_for_metrics(), map_location="cpu")
                metrics = ckpt["metrics"]
                self.assertIn(metric, metrics)
                self.assertTrue(math.isfinite(metrics["test_brier"]) and metrics["test_brier"] >= 0.0)
                self.assertTrue(math.isfinite(metrics["train_loss"]))
                self.assertEqual(ckpt["extra"]["graphmetro_expert_names"], ["id", *TRANSFORM_NAMES])

                preds = torch.load(runner.prediction_path(), map_location="cpu")
                self.assertEqual(preds["index"].tolist(), list(range(8, 12)))
                self.assertEqual(preds["pred"].size(0), 4)
                self.assertAlmostEqual(preds["meta"]["test_brier"], metrics["test_brier"], places=6)
                self.assertEqual(preds["meta"]["split_root"], "data/splits")

                with open(Path(tmp) / "results" / "moe_graphmetro.tsv", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh, delimiter="\t"))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["data_preparation.dataset.split_root"], "data/splits")
                self.assertTrue(rows[0]["test_brier_mean"])

                cfg.moe.graphmetro.skip_if_exists = True
                self.assertTrue(GraphMETRORunner(cfg)._skip_due_to_existing_checkpoint)

    def test_run_name_identity(self):
        cfg = _cfg()
        name = GraphMETRORunner(cfg).run_name
        self.assertTrue(name.startswith("graphmetro_toy_induced1_fewshot5-0-100_tasknode_gcn_lam1_h8_l2_e100_lr0.01_bs32_cfg"), name)
        self.assertTrue(name.endswith("_seed42"), name)
        shifted = cfg.clone()
        shifted.data_preparation.dataset.split_root = "data/splits_shift/structural"
        self.assertNotEqual(GraphMETRORunner(shifted).run_name, name)
        changed = cfg.clone()
        changed.moe.graphmetro.gate_pos_weight = 1.0
        self.assertNotEqual(GraphMETRORunner(changed).run_name, name)
        operational = cfg.clone()
        operational.moe.graphmetro.prediction_dir = "/tmp/elsewhere"
        operational.moe.graphmetro.checkpoint_dir = "/tmp/elsewhere"
        operational.moe.graphmetro.num_runs = 2
        self.assertEqual(GraphMETRORunner(operational).run_name, name)

    def test_non_induced_node_tasks_are_rejected(self):
        cfg = _cfg()
        cfg.moe.graphmetro.dataset.induced = False
        with self.assertRaisesRegex(ValueError, "induced"):
            GraphMETRORunner(cfg)

    def test_shift_root_split_files_are_verified_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(tmp)
            cfg.data_preparation.shift.root = os.path.join(tmp, "splits_shift")
            cfg.data_preparation.dataset.split_root = os.path.join(tmp, "splits_shift", "structural")
            with patch("src.moe.graphmetro.trainer.create_dataset") as create:
                with self.assertRaisesRegex(ValueError, "missing"):
                    GraphMETRORunner(cfg).fit()
                create.assert_not_called()

            path = split_file_path(cfg.data_preparation.dataset.split_root, "toy", "node", 42, (5, 0.0, 1.0))
            path.parent.mkdir(parents=True)
            payload = {"train": [0, 1], "val": [2], "test": [3, 4],
                       "meta": {"type": "shift_covariate", "condition": "structural", "total": 5}}
            torch.save(payload, path)
            GraphMETRORunner(cfg)._verify_shift_split()
            payload["meta"]["type"] = "few_shot"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "not a shift split"):
                GraphMETRORunner(cfg)._verify_shift_split()


class GraphMETROPlumbingTest(unittest.TestCase):
    def test_config_defaults(self):
        gm = base_cfg.moe.graphmetro
        self.assertEqual((gm.backbone, gm.num_layers, gm.hidden_dim, gm.dropout), ("gcn", 3, 300, 0.5))
        self.assertEqual((gm.shift_p, gm.shift_k, gm.num_shift_samples), (0.5, 2, 3))
        self.assertEqual((gm.gate_pos_weight, gm.align_lambda, gm.moe_lr, gm.classifier_lr), (4.0, 1.0, 1e-2, 1e-4))
        self.assertEqual(tuple(gm.dataset.fixed_split), (5, 0.0, 1.0))
        self.assertEqual(gm.tasks_tsv, "slurm/moe.graphmetro.all.tsv")

    def test_parse_and_build_task_cfg(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.tsv"
            path.write_text(
                TSV_HEADER
                + "photo\tnode\tclassification\tTrue\t(5,0.0,1.0)\tdata/splits_shift/mixed\tgin\t1e-3\t7\t16\tFalse\n"
                + "photo\tnode\tclassification\tTrue\t(5,0.0,1.0)\tdata/splits_shift/mixed\tgin\tfast\t7\t16\tFalse\n"
                + "mnist\tgraph\tclassification\tTrue\t(100,0.0,1.0)\t-\t-\t-\n",
                encoding="utf-8",
            )
            tasks = parse_graphmetro_tasks(str(path))
        self.assertEqual(len(tasks), 2)
        self.assertEqual((tasks[0]["split_root"], tasks[0]["backbone"], tasks[0]["moe_lr"]), ("data/splits_shift/mixed", "gin", 1e-3))
        self.assertEqual((tasks[1]["split_root"], tasks[1]["backbone"], tasks[1]["moe_lr"]), (None, None, None))

        run_cfg = _build_task_cfg(_cfg(), tasks[0])
        gm = run_cfg.moe.graphmetro
        self.assertEqual((gm.dataset.name, gm.backbone, gm.moe_lr, gm.epochs, gm.batch_size), ("photo", "gin", 1e-3, 7, 16))
        self.assertEqual(run_cfg.data_preparation.dataset.split_root, "data/splits_shift/mixed")
        self.assertEqual(tuple(gm.dataset.fixed_split), (5, 0.0, 1.0))
        self.assertFalse(gm.run_tasks_tsv)
        self.assertIn("data_preparation.dataset.split_root", run_cfg.save_results.explicit_keys)

        default_cfg = _build_task_cfg(_cfg(), tasks[1])
        self.assertEqual(default_cfg.data_preparation.dataset.split_root, "data/splits")
        self.assertEqual(default_cfg.moe.graphmetro.backbone, "gcn")

    def test_repository_tsv_grid(self):
        tasks = parse_graphmetro_tasks(str(ROOT / "slurm" / "moe.graphmetro.all.tsv"))
        self.assertEqual(len(tasks), 56)
        targets = {"photo", "ogbn-arxiv", "airports", "chameleon", "dblp", "cornell", "mnist", "toxcast", "qm7b"}
        self.assertEqual({t["dataset"] for t in tasks}, targets)
        standard = [t for t in tasks if t["split_root"] == "data/splits"]
        self.assertEqual(len(standard), 16)
        self.assertEqual({(t["dataset"], t["fixed_split"]) for t in standard if t["task_level"] != "edge"},
                         {(d, (s, 0.0, 1.0)) for d in targets - {"dblp", "cornell"} for s in (5, 100)})
        for condition in ("feature", "structural", "mixed"):
            rows = [t for t in tasks if t["split_root"] == f"data/splits_shift/{condition}"]
            expected = targets - {"dblp", "cornell"} - ({"qm7b"} if condition == "feature" else set())
            self.assertEqual({(t["dataset"], t["fixed_split"]) for t in rows},
                             {(d, (s, 0.0, 1.0)) for d in expected for s in (5, 100)}, condition)
        for task in tasks:
            graph = task["task_level"] == "graph"
            self.assertEqual((task["backbone"], task["moe_lr"], task["epochs"]),
                             ("gin", 1e-3, 200) if graph else ("gcn", 1e-2, 100), task)
            self.assertTrue(task["induced"])
            if task["task_level"] == "edge":
                self.assertEqual(task["fixed_split"], (0.1, 0.05, 0.1))

    def test_run_moe_dispatches_graphmetro(self):
        cfg = moe_run._build_moe_cfg(["moe.graphmetro.dataset.name", "photo"])
        self.assertEqual(cfg.moe.method, "graphmetro")
        with patch("src.moe.graphmetro.run_graphmetro", return_value=0) as fake:
            self.assertEqual(moe_run.run_moe(cfg), 0)
        fake.assert_called_once()


if __name__ == "__main__":
    unittest.main()
