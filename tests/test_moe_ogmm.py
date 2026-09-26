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
from torch_geometric.nn import DenseGATConv

from src.config import cfg as base_cfg
from src.moe import run as moe_run
from src.moe.ogmm.experts import (
    DenseExpert,
    DenseInstances,
    SoftAdjDenseGAT,
    edge_density,
    partition_by_edge_density,
    to_dense_instances,
)
from src.moe.ogmm.generator import (
    EdgeEncoder,
    bn_stat_loss,
    confidence_loss,
    generate_for_expert,
    sample_generated_labels,
)
from src.moe.ogmm.merge import (
    MaskedLinear,
    NoisyTopKGate,
    OGMMMergedModel,
    cv_squared,
    mask_regularizer,
    merge_loss,
)
from src.moe.ogmm.run import _build_task_cfg, parse_ogmm_tasks, run_ogmm
from src.moe.ogmm.trainer import OGMMRunner
from src.utils.supervised_loss import supervised_loss_from_logits

ROOT = Path(__file__).resolve().parents[1]
ARCHS = ("gcn", "gat", "gin")
_THREADS = torch.get_num_threads()


def setUpModule():
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_THREADS)


# --------------------------------------------------------------------------- #
# Synthetic instances shaped like the repo's induced datasets
# --------------------------------------------------------------------------- #
def _path_edges(n, start=0):
    src = list(range(start, start + n - 1))
    dst = list(range(start + 1, start + n))
    return torch.tensor([src + dst, dst + src], dtype=torch.long)


def _instances(level, count=14, in_dim=4, seed=0, family="cls"):
    """Instances with feature clusters by label; ego targets / LP endpoints like induced_graphs.py."""
    gen = torch.Generator().manual_seed(seed)
    graphs = []
    for i in range(count):
        n = 5 + i % 4
        label = i % 3 if level != "edge" else i % 2
        x = torch.randn(n, in_dim, generator=gen) * 0.3 + (label - 1.0)
        edge_index = _path_edges(n)
        extra = torch.randint(0, n, (2, i % 3), generator=gen)
        edge_index = torch.cat([edge_index, extra, extra.flip(0)], dim=1)
        data = Data(x=x, edge_index=edge_index)
        if level == "node":
            data.y = torch.tensor(label)
            data.target_node_index = torch.tensor([i % n])
        elif level == "edge":
            keep = ~(((edge_index[0] == 0) & (edge_index[1] == 1)) | ((edge_index[0] == 1) & (edge_index[1] == 0)))
            data.edge_index = edge_index[:, keep]
            data.edge_label_index = torch.tensor([[0], [1]])
            data.y = torch.tensor(label)
        elif family == "regression":
            data.y = torch.tensor([[float(n), 10.0 * label + float(i)]])
        elif family == "multilabel":
            y = torch.tensor([[float(label == 0), float(label == 1), float(i % 2)]])
            y[0, 2] = float("nan") if i % 5 == 0 else y[0, 2]
            data.y = y
        else:
            data.y = torch.tensor([label])
        graphs.append(data)
    return graphs


def _soft_adj(batch_size, n, gen):
    a = torch.rand(batch_size, n, n, generator=gen)
    a = (a + a.transpose(1, 2)) / 2
    a.diagonal(dim1=1, dim2=2).zero_()
    return a


def _cfg(tmp=None, **ogmm):
    cfg = base_cfg.clone()
    cfg.seed = 42
    cfg.seeds = [42]
    o = cfg.moe.ogmm
    o.dataset.name = "toy"
    o.dataset.task_level = "node"
    o.skip_if_exists = False
    if tmp is not None:
        o.checkpoint_dir = os.path.join(tmp, "ckpt")
        o.log_dir = os.path.join(tmp, "logs")
        o.prediction_dir = os.path.join(tmp, "pred")
        cfg.save_results.output_dir = os.path.join(tmp, "results")
    for key, value in ogmm.items():
        setattr(o, key, value)
    return cfg


def _tiny(tmp, level, **extra):
    cfg = _cfg(
        tmp, expert_hidden_dim=8, expert_epochs=2, gen_epochs=2, gen_num_graphs=4,
        gen_edge_hidden_dim=8, merge_epochs=1, batch_size=4, num_runs=1, **extra,
    )
    cfg.moe.ogmm.dataset.task_level = level
    cfg.moe.ogmm.dataset.fixed_split = (4, 0.0, 1.0) if level != "edge" else (0.1, 0.05, 0.1)
    return cfg


class _GuardedLoader:
    """A loader that raises unless opened (query-isolation checks)."""

    def __init__(self, loader):
        self.loader, self.dataset, self.open = loader, loader.dataset, False

    def __iter__(self):
        if not self.open:
            raise AssertionError("query/val loader read outside evaluate()")
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def _patched_trainer(graphs, meta, n_support=8, guard=False):
    loaders = {}

    def _loaders(**_kwargs):
        train = DataLoader(Subset(graphs, list(range(n_support))), batch_size=4, shuffle=True)
        val = _GuardedLoader(DataLoader(Subset(graphs, []), batch_size=4))
        test = DataLoader(Subset(graphs, list(range(n_support, len(graphs)))), batch_size=4)
        loaders["val"], loaders["test"] = val, _GuardedLoader(test) if guard else test
        return train, val, loaders["test"]

    return loaders, patch.multiple(
        "src.moe.ogmm.trainer",
        create_dataset=lambda **_kwargs: graphs,
        dataset_info=lambda **_kwargs: dict(meta),
        make_workflow_loaders=_loaders,
        log_split_instance_counts=lambda *args, **kwargs: None,
    )


# --------------------------------------------------------------------------- #
# Dense instances and experts
# --------------------------------------------------------------------------- #
class DenseInstancesTest(unittest.TestCase):
    def test_anchors_point_to_sparse_targets(self):
        for level, attr in (("node", "target_node_index"), ("edge", "edge_label_index")):
            graphs = _instances(level, count=3)
            batch = Batch.from_data_list(graphs)
            inst = to_dense_instances(batch, level)
            rows = torch.arange(3)
            sparse = getattr(batch, attr).view(2 if level == "edge" else 1, -1)
            self.assertTrue(torch.equal(inst.x[rows, inst.anchor[:, 0]], batch.x[sparse[0]]), level)
            self.assertTrue(torch.equal(inst.x[rows, inst.anchor[:, 1]], batch.x[sparse[-1]]), level)
            self.assertEqual(inst.mask.sum(1).tolist(), [g.num_nodes for g in graphs])
            self.assertTrue(torch.equal(inst.adj, inst.adj.transpose(1, 2)))
            self.assertEqual(float(inst.adj.diagonal(dim1=1, dim2=2).abs().sum()), 0.0)
        graph_inst = to_dense_instances(Batch.from_data_list(_instances("graph", count=2)), "graph")
        self.assertTrue(bool((graph_inst.anchor == -1).all()))
        with self.assertRaisesRegex(ValueError, "target_node_index"):
            to_dense_instances(Batch.from_data_list(_instances("graph", count=2)), "node")

    def test_directed_input_is_symmetrised_and_deduplicated(self):
        data = Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 0, 1, 2], [1, 1, 2, 2]]), y=torch.tensor([0]))
        inst = to_dense_instances(Batch.from_data_list([data]), "graph")
        expected = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        self.assertTrue(torch.equal(inst.adj[0], expected))
        self.assertAlmostEqual(edge_density(data), 2 / 3)


class DenseExpertTest(unittest.TestCase):
    def _inst(self, level, gen, batch_size=3, n=6, in_dim=5):
        adj = (_soft_adj(batch_size, n, gen) > 0.5).float()
        anchor = {"node": [[2, 2]], "edge": [[0, 1]], "graph": [[-1, -1]]}[level] * batch_size
        return DenseInstances(
            x=torch.randn(batch_size, n, in_dim, generator=gen), adj=adj,
            mask=torch.ones(batch_size, n, dtype=torch.bool), anchor=torch.tensor(anchor),
        )

    def test_padding_and_permutation_invariance(self):
        gen = torch.Generator().manual_seed(0)
        for arch in ARCHS:
            for level in ("node", "edge", "graph"):
                torch.manual_seed(1)
                expert = DenseExpert(arch, 5, 8, 3, level, dropout=0.5).eval()
                inst = self._inst(level, gen)
                pad = DenseInstances(
                    x=F.pad(inst.x, (0, 0, 0, 4)), adj=F.pad(inst.adj, (0, 4, 0, 4)),
                    mask=F.pad(inst.mask, (0, 4)), anchor=inst.anchor,
                )
                # Permute every node except the anchors (0, 1, 2).
                perm = torch.tensor([0, 1, 2, 5, 3, 4])
                permuted = DenseInstances(
                    x=inst.x[:, perm], adj=inst.adj[:, perm][:, :, perm], mask=inst.mask, anchor=inst.anchor,
                )
                with torch.no_grad():
                    out = expert(inst)
                    self.assertTrue(torch.allclose(out, expert(pad), atol=1e-5), (arch, level))
                    self.assertTrue(torch.allclose(out, expert(permuted), atol=1e-5), (arch, level))
                self.assertEqual(len(expert.bn_layers()), 2 if arch == "gin" else 1)

    def test_batch_norm_ignores_padding(self):
        gen = torch.Generator().manual_seed(2)
        torch.manual_seed(0)
        expert = DenseExpert("gcn", 5, 8, 3, "graph", dropout=0.0).train()
        expert.bn1.momentum = 1.0  # running stats = this batch's statistics
        inst = self._inst("graph", gen)
        inst.mask[0, 4:] = False
        inst.mask[2, 3:] = False
        inst.x = inst.x * inst.mask.unsqueeze(-1)
        inst.adj = inst.adj * inst.mask.unsqueeze(1) * inst.mask.unsqueeze(2)
        expert(inst)
        with torch.no_grad():
            valid = expert.conv1(inst.x, inst.adj, inst.mask)[inst.mask]
        self.assertTrue(torch.allclose(expert.bn1.running_mean, valid.mean(0), atol=1e-5))
        self.assertTrue(torch.allclose(expert.bn1.running_var, valid.var(0, unbiased=True), atol=1e-5))

    def test_soft_adjacency_receives_gradient(self):
        gen = torch.Generator().manual_seed(3)
        x = torch.randn(2, 6, 5, generator=gen)
        mask = torch.ones(2, 6, dtype=torch.bool)
        anchor = torch.full((2, 2), -1)
        for arch in ARCHS:
            adj = _soft_adj(2, 6, gen).requires_grad_(True)
            DenseExpert(arch, 5, 8, 3, "graph", dropout=0.0).eval()(
                DenseInstances(x=x, adj=adj, mask=mask, anchor=anchor)
            ).sum().backward()
            self.assertIsNotNone(adj.grad, arch)
            self.assertGreater(float(adj.grad.abs().sum()), 0.0, arch)
        # The stock layer only reads the zero pattern: no gradient reaches adj.
        adj = _soft_adj(2, 6, gen).requires_grad_(True)
        DenseGATConv(5, 4)(x, adj, mask).sum().backward()
        self.assertTrue(adj.grad is None or float(adj.grad.abs().sum()) == 0.0)

    def test_soft_gat_matches_dense_gat_on_binary_adjacency(self):
        gen = torch.Generator().manual_seed(4)
        torch.manual_seed(0)
        soft, stock = SoftAdjDenseGAT(5, 4), DenseGATConv(5, 4)
        stock.load_state_dict(soft.state_dict())
        x = torch.randn(3, 6, 5, generator=gen)
        adj = (_soft_adj(3, 6, gen) > 0.5).float()
        mask = torch.ones(3, 6, dtype=torch.bool)
        mask[1, 4:] = False
        self.assertTrue(torch.allclose(soft(x, adj, mask), stock(x, adj, mask), atol=1e-5))

    def test_unknown_arch_raises(self):
        with self.assertRaisesRegex(ValueError, "architecture"):
            DenseExpert("sage", 4, 8, 2, "graph", 0.0)


class DensityPartitionTest(unittest.TestCase):
    def test_partition_is_sorted_balanced_and_deterministic(self):
        graphs = _instances("graph", count=11)
        domains = partition_by_edge_density(graphs, 2)
        self.assertEqual(domains, partition_by_edge_density(graphs, 2))
        self.assertEqual(sorted(i for d in domains for i in d), list(range(11)))
        self.assertLessEqual(abs(len(domains[0]) - len(domains[1])), 1)
        densities = [[edge_density(graphs[i]) for i in d] for d in domains]
        self.assertTrue(all(a <= b for d in densities for a, b in zip(d, d[1:])))
        self.assertLessEqual(max(densities[0]), min(densities[1]))
        self.assertEqual(partition_by_edge_density(graphs, 1), [sorted(range(11), key=lambda i: (edge_density(graphs[i]), i))])
        with self.assertRaises(ValueError):
            partition_by_edge_density(graphs, 0)


# --------------------------------------------------------------------------- #
# Stage 1: generation
# --------------------------------------------------------------------------- #
class GenerationTest(unittest.TestCase):
    def test_edge_encoder_sample_properties(self):
        torch.manual_seed(0)
        encoder = EdgeEncoder(5, 16)
        x = torch.randn(4, 7, 5)
        off_diag = ~torch.eye(7, dtype=torch.bool)
        for tau in (0.5, 0.05):
            a = encoder.sample(x, tau, forbid_pair=(0, 1), generator=torch.Generator().manual_seed(1))
            self.assertTrue(torch.equal(a, a.transpose(1, 2)))
            self.assertEqual(float(a.diagonal(dim1=1, dim2=2).abs().sum()), 0.0)
            self.assertTrue(bool((a[:, 0, 1] == 0).all()) and bool((a[:, 1, 0] == 0).all()))
            free = a[:, off_diag].clone()
            free = torch.cat([free[:, 1:6], free[:, 7:]], dim=1)  # drop the forbidden pair
            self.assertTrue(bool(((free >= 0) & (free <= 1)).all()))
            if tau == 0.05:
                self.assertLess(float(((free > 0.05) & (free < 0.95)).float().mean()), 0.15)
        probs = encoder.edge_probs(x)
        self.assertTrue(torch.equal(probs, probs.transpose(1, 2)))
        self.assertTrue(bool((probs[:, off_diag] > 0).all()))

    def test_edge_encoder_pair_logits_match_concatenated_mlp(self):
        torch.manual_seed(0)
        encoder = EdgeEncoder(3, 8)
        x = torch.randn(2, 4, 3)
        pairs = torch.cat([x.unsqueeze(2).expand(2, 4, 4, 3), x.unsqueeze(1).expand(2, 4, 4, 3)], dim=-1)
        reference = encoder.lin3(F.relu(encoder.lin2(F.relu(encoder.lin1(pairs))))).squeeze(-1)
        self.assertTrue(torch.allclose(encoder.pair_logits(x), reference, atol=1e-5))

    def test_bn_stat_loss(self):
        bn = torch.nn.BatchNorm1d(2)
        bn.running_mean.copy_(torch.tensor([1.0, -2.0]))
        bn.running_var.copy_(torch.tensor([4.0, 0.25]))
        matched = torch.stack([bn.running_mean + bn.running_var.sqrt(), bn.running_mean - bn.running_var.sqrt()])
        self.assertAlmostEqual(float(bn_stat_loss([matched], [bn])), 0.0, places=6)
        self.assertAlmostEqual(float(bn_stat_loss([matched + 1.0], [bn])), 2.0, places=5)
        with self.assertRaises(ValueError):
            bn_stat_loss([], [bn])

    def test_confidence_loss(self):
        self.assertAlmostEqual(float(confidence_loss(torch.zeros(4, 5), "classification", 1)), math.log(5), places=5)
        self.assertAlmostEqual(float(confidence_loss(torch.zeros(4, 1), "classification", 1)), math.log(2), places=5)
        self.assertAlmostEqual(float(confidence_loss(torch.zeros(4, 3), "classification", 3)), math.log(2), places=5)
        self.assertEqual(float(confidence_loss(torch.randn(4, 2), "regression", 2)), 0.0)
        self.assertLess(float(confidence_loss(torch.tensor([[20.0, 0.0]]), "classification", 1)), 1e-6)

    def test_sample_generated_labels(self):
        gen = torch.Generator().manual_seed(0)
        cls = sample_generated_labels("classification", 200, num_classes=3, label_dim=1, domain_targets=torch.zeros(2, 1), generator=gen)
        self.assertEqual(set(cls.tolist()), {0, 1, 2})
        reg_targets = torch.tensor([[1.0, 5.0], [3.0, float("nan")], [2.0, 7.0]])
        reg = sample_generated_labels("regression", 100, num_classes=1, label_dim=2, domain_targets=reg_targets, generator=gen)
        self.assertEqual(tuple(reg.shape), (100, 2))
        self.assertTrue(bool((reg[:, 0] >= 1).all() and (reg[:, 0] <= 3).all()))
        self.assertTrue(bool((reg[:, 1] >= 5).all() and (reg[:, 1] <= 7).all()))
        signed = torch.tensor([[1.0, -1.0, 0.0], [1.0, -1.0, 0.0]])  # 0 = missing in the signed convention
        multi = sample_generated_labels("classification", 400, num_classes=2, label_dim=3, domain_targets=signed, generator=gen)
        self.assertTrue(bool((multi[:, 0] == 1).all()) and bool((multi[:, 1] == 0).all()))
        self.assertAlmostEqual(float(multi[:, 2].mean()), 0.5, delta=0.1)

    def _toy_expert(self, level):
        torch.manual_seed(0)
        graphs = _instances(level, count=18, in_dim=4)
        out_dim = 1 if level == "edge" else 3
        expert = DenseExpert("gcn", 4, 8, out_dim, level, dropout=0.0)
        inst = to_dense_instances(Batch.from_data_list(graphs), level)
        optimizer = torch.optim.Adam(expert.parameters(), lr=1e-2)
        for _ in range(60):
            optimizer.zero_grad()
            supervised_loss_from_logits(logits=expert(inst), labels=inst.y, task_type="classification")[0].backward()
            optimizer.step()
        return expert.eval()

    def _generate(self, expert, level, epochs, num_classes=3):
        return generate_for_expert(
            expert, num_graphs=12, num_nodes=6, in_dim=4, task_level_raw=level, task_type="classification",
            label_dim=1, num_classes=num_classes, domain_targets=torch.zeros(2, 1), epochs=epochs, lr=0.05,
            tau=0.5, edge_threshold=0.5, generator=torch.Generator().manual_seed(3), edge_hidden_dim=8,
        )

    def test_generation_lowers_expert_loss_and_leaves_expert_untouched(self):
        expert = self._toy_expert("graph")
        before = {k: v.clone() for k, v in expert.state_dict().items()}

        def ce(graphs):
            inst = to_dense_instances(Batch.from_data_list(graphs), "graph")
            with torch.no_grad():
                return float(supervised_loss_from_logits(logits=expert(inst), labels=inst.y, task_type="classification")[0])

        initial, trained = self._generate(expert, "graph", 0), self._generate(expert, "graph", 60)
        self.assertTrue(all(torch.equal(a.y, b.y) for a, b in zip(initial, trained)))  # same conditional labels
        self.assertLess(ce(trained), ce(initial))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in expert.state_dict().items()))
        self.assertTrue(all(p.requires_grad for p in expert.parameters()))
        self.assertFalse(expert.training)
        self.assertTrue(all(len(bn._forward_hooks) == 0 for bn in expert.bn_layers()))
        self.assertEqual({g.num_nodes for g in trained}, {6})

    def test_generated_instances_carry_task_anchors(self):
        node = self._generate(self._toy_expert("node"), "node", 2)
        self.assertTrue(all(g.target_node_index.tolist() == [0] and g.y.shape == (1,) for g in node))
        edge = self._generate(self._toy_expert("edge"), "edge", 2, num_classes=2)
        for g in edge:
            self.assertEqual(g.edge_label_index.tolist(), [[0], [1]])
            pair = ((g.edge_index[0] == 0) & (g.edge_index[1] == 1)) | ((g.edge_index[0] == 1) & (g.edge_index[1] == 0))
            self.assertFalse(bool(pair.any()))
            self.assertIn(int(g.y), (0, 1))
        inst = to_dense_instances(Batch.from_data_list(edge), "edge")
        self.assertTrue(bool((inst.anchor == torch.tensor([0, 1])).all()))


# --------------------------------------------------------------------------- #
# Stage 2: merging
# --------------------------------------------------------------------------- #
class MergeTest(unittest.TestCase):
    def _merged(self, k=2):
        torch.manual_seed(0)
        experts = [DenseExpert(arch, 4, 8, 3, "graph", dropout=0.5).eval() for arch in ARCHS]
        return OGMMMergedModel(experts, NoisyTopKGate(4, 3, k))

    def test_masked_linear_identity_at_init(self):
        linear = torch.nn.Linear(5, 3)
        masked = MaskedLinear(linear)
        h = torch.randn(4, 5)
        self.assertTrue(torch.allclose(masked(h), linear(h)))
        self.assertEqual([tuple(m.shape) for m in masked.mask_tensors()], [(3, 5), (3,)])
        self.assertFalse(any(p.requires_grad for p in linear.parameters()))

    def test_only_masks_and_gate_train_and_encoders_stay_frozen(self):
        model = self._merged()
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        expected = {f"heads.{j}.omega_{w}" for j in range(3) for w in ("w", "b")} | {"gate.w_gate", "gate.w_noise"}
        self.assertEqual(trainable, expected)
        model.train()
        self.assertTrue(model.gate.training)
        self.assertFalse(any(expert.training for expert in model.experts))

        inst = to_dense_instances(Batch.from_data_list(_instances("graph", count=6)), "graph")
        with torch.no_grad():
            per_expert = model.expert_logits(inst)
            for j, expert in enumerate(model.experts):
                self.assertTrue(torch.allclose(per_expert[:, j], expert(inst), atol=1e-6))
        loss, log = merge_loss(model, inst, task_type="classification", lambda_gate=0.1, lambda_mask=0.01, gamma_p=0.5, gamma_v=0.1)
        loss.backward()
        self.assertEqual(set(log), {"task", "gate", "mask"})
        self.assertTrue(all(p.grad is None for p in model.experts.parameters()))
        self.assertTrue(all(model.heads[j].omega_w.grad is not None for j in range(3)))

    def test_merge_loss_without_regularisers_is_mixture_task_loss(self):
        model = self._merged().eval()
        inst = to_dense_instances(Batch.from_data_list(_instances("graph", count=6)), "graph")
        loss, _ = merge_loss(model, inst, task_type="classification", lambda_gate=0.0, lambda_mask=0.0, gamma_p=0.5, gamma_v=0.1)
        mixed, gate = model(inst)
        expected = supervised_loss_from_logits(logits=mixed, labels=inst.y, task_type="classification")[0]
        self.assertAlmostEqual(float(loss), float(expected), places=5)
        self.assertTrue(torch.allclose(mixed, (gate.unsqueeze(-1) * model.expert_logits(inst)).sum(1)))

    def test_noisy_top_k_gate(self):
        torch.manual_seed(0)
        gate = NoisyTopKGate(4, 5, 2)
        with torch.no_grad():
            gate.w_gate.normal_()
        g = torch.randn(16, 4)
        gate.eval()
        weights = gate(g)
        self.assertTrue(torch.equal(weights, gate(g)))
        self.assertEqual((weights > 0).sum(1).tolist(), [2] * 16)
        self.assertTrue(torch.allclose(weights.sum(1), torch.ones(16)))
        gate.train()
        self.assertFalse(torch.equal(gate(g), gate(g)))
        self.assertEqual((gate(g) > 0).sum(1).tolist(), [2] * 16)
        with self.assertRaises(ValueError):
            NoisyTopKGate(4, 2, 3)

    def test_cv_squared_and_literal_mask_regularizer(self):
        self.assertEqual(float(cv_squared(torch.ones(4))), 0.0)
        self.assertEqual(float(cv_squared(torch.tensor([3.0]))), 0.0)
        self.assertAlmostEqual(float(cv_squared(torch.tensor([1.0, 3.0]))), 0.5, places=6)

        w1 = torch.ones(2, 3, requires_grad=True)
        b1 = torch.ones(2, requires_grad=True)
        w2 = torch.ones(4, requires_grad=True)
        value = mask_regularizer([[w1, b1], [w2]], gamma_p=0.5, gamma_v=0.1)
        self.assertAlmostEqual(float(value), 2.0, places=6)  # per expert: (1 - .5) + (1 - .5)
        value.backward()
        self.assertTrue(torch.allclose(w1.grad, torch.full_like(w1, 1 / 8)))
        self.assertTrue(torch.allclose(b1.grad, torch.full_like(b1, 1 / 8)))
        self.assertTrue(torch.allclose(w2.grad, torch.full_like(w2, 1 / 4)))
        shifted = mask_regularizer([[torch.full((4,), 0.5)]], gamma_p=0.5, gamma_v=0.1)
        self.assertAlmostEqual(float(shifted), -0.5, places=6)  # no entry within 0.1 of 1


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
_META = {
    "node": ({"num_node_features": 4, "num_classes": 3, "label_dim": 1, "task_type": "classification"}, "node_cls", "test_acc"),
    "edge": ({"num_node_features": 4, "num_classes": 2, "label_dim": 1, "task_type": "classification"}, "link", "test_auc"),
    "graph": ({"num_node_features": 4, "num_classes": 3, "label_dim": 1, "task_type": "classification"}, "graph_cls", "test_acc"),
    "regression": ({"num_node_features": 4, "num_classes": 1, "label_dim": 2, "task_type": "regression"}, "regression", "test_mae"),
    "multilabel": ({"num_node_features": 4, "num_classes": 2, "label_dim": 3, "task_type": "classification"}, "multilabel", "test_auc"),
}


class OGMMRunnerTest(unittest.TestCase):
    def test_smoke_every_task_family(self):
        for name, (meta, family, metric) in _META.items():
            level = name if name in ("node", "edge", "graph") else "graph"
            graphs = _instances(level, count=14, family=name)
            _, patcher = _patched_trainer(graphs, meta)
            with self.subTest(family=name), tempfile.TemporaryDirectory() as tmp, patcher:
                runner = OGMMRunner(_tiny(tmp, level))
                runner.fit()
                self.assertEqual(runner.task_family, family)
                self.assertIn(metric, runner.best_metrics)
                self.assertTrue(math.isfinite(runner.best_metrics["test_brier"]))
                self.assertTrue(all(math.isfinite(h["loss"]) for h in runner.train_history))
                self.assertEqual(len(runner.expert_info), 6)
                self.assertTrue(all(math.isfinite(e["train_loss"]) for e in runner.expert_info))

                preds = torch.load(runner._prediction_path(), map_location="cpu")
                self.assertEqual(preds["index"].tolist(), list(range(8, 14)))
                self.assertEqual(preds["pred"].dtype, torch.float32)
                self.assertEqual(preds["meta"]["task_family"], family)
                ckpt = torch.load(runner.get_checkpoint_path_for_metrics(), map_location="cpu")
                self.assertAlmostEqual(ckpt["metrics"]["test_brier"], runner.best_metrics["test_brier"])
                self.assertEqual(len(ckpt["extra"]["ogmm_generated_graphs"]), 6 * 4)
                self.assertAlmostEqual(sum(ckpt["extra"]["ogmm_test_mean_gate"]), 1.0, places=5)
                self.assertTrue(os.path.isfile(runner._log_path()))

    def test_query_labels_and_real_instances_never_reach_training(self):
        graphs = _instances("node", count=14)
        loaders, patcher = _patched_trainer(graphs, _META["node"][0], guard=True)
        record = {}

        class Probe(OGMMRunner):
            def generate(self, experts, domains):
                record["generated"] = super().generate(experts, domains)
                return record["generated"]

            def merge(self, experts, generated):
                record["merged_on"] = generated
                return super().merge(experts, generated)

            def evaluate(self, model):
                loaders["test"].open = True
                return super().evaluate(model)

        with tempfile.TemporaryDirectory() as tmp, patcher:
            Probe(_tiny(tmp, "node")).fit()
        self.assertIs(record["merged_on"], record["generated"])
        real = {id(g) for g in graphs}
        self.assertFalse(any(id(g) in real for g in record["merged_on"]))
        self.assertFalse(loaders["val"].open)

    def test_small_support_falls_back_to_one_domain_and_clamps_top_k(self):
        graphs = _instances("graph", count=7, family="regression")
        _, patcher = _patched_trainer(graphs, _META["regression"][0], n_support=3)
        with tempfile.TemporaryDirectory() as tmp, patcher:
            runner = OGMMRunner(_tiny(tmp, "graph", expert_archs=["gcn"], top_k=3))
            runner.fit()
        self.assertEqual(len(runner.domain_info), 1)
        self.assertEqual(runner.domain_info[0]["size"], 3)
        self.assertEqual(runner.top_k, 1)

    def test_run_ogmm_appends_result_and_skips_existing(self):
        graphs = _instances("graph", count=14)
        _, patcher = _patched_trainer(graphs, _META["graph"][0])
        with tempfile.TemporaryDirectory() as tmp, patcher:
            cfg = _tiny(tmp, "graph")
            cfg.data_preparation.dataset.split_root = "data/splits_shift/structural"
            self.assertEqual(run_ogmm(cfg), 0)
            lines = (Path(tmp) / "results" / "moe_ogmm.tsv").read_text(encoding="utf-8").splitlines()
            self.assertIn("data_preparation.dataset.split_root", lines[0])
            self.assertIn("test_brier", lines[0])
            self.assertIn("data/splits_shift/structural", lines[1])
            cfg.moe.ogmm.skip_if_exists = True
            self.assertTrue(OGMMRunner(cfg)._skip_due_to_existing_checkpoint)

    def test_run_name_identity(self):
        cfg = _cfg()
        name = OGMMRunner(cfg).run_name
        self.assertTrue(name.startswith("ogmm_toy_induced1_fewshot5-0-100_tasknode_d2_gcn-gat-gin_h32_k2_"), name)
        self.assertTrue(name.endswith("_seed42"), name)
        shifted = cfg.clone()
        shifted.data_preparation.dataset.split_root = "data/splits_shift/mixed"
        self.assertNotEqual(OGMMRunner(shifted).run_name, name)
        operational = cfg.clone()
        operational.moe.ogmm.prediction_dir = "/tmp/elsewhere"
        operational.moe.ogmm.checkpoint_dir = "/tmp/elsewhere"
        operational.moe.ogmm.num_runs = 2
        self.assertEqual(OGMMRunner(operational).run_name, name)
        behavioural = cfg.clone()
        behavioural.moe.ogmm.gen_gumbel_tau = 0.25
        self.assertNotEqual(OGMMRunner(behavioural).run_name, name)

    def test_node_and_edge_tasks_require_induced(self):
        for level in ("node", "edge"):
            cfg = _cfg()
            cfg.moe.ogmm.dataset.task_level = level
            cfg.moe.ogmm.dataset.induced = False
            with self.assertRaisesRegex(ValueError, "induced"):
                OGMMRunner(cfg)


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #
class OGMMPlumbingTest(unittest.TestCase):
    def test_config_defaults(self):
        o = base_cfg.moe.ogmm
        self.assertEqual(list(o.expert_archs), ["gcn", "gat", "gin"])
        self.assertEqual((o.num_domains, o.expert_hidden_dim, o.gen_epochs, o.merge_epochs), (2, 32, 200, 20))
        self.assertEqual((o.top_k, o.lambda_gate, o.lambda_mask), (2, 0.1, 0.01))
        self.assertEqual(tuple(o.dataset.fixed_split), (5, 0.0, 1.0))
        self.assertEqual(o.tasks_tsv, "slurm/moe.ogmm.all.tsv")

    def test_parse_and_build_task_cfg(self):
        header = "# dataset\ttask_level\ttask_type\tinduced\tfixed_split\tsplit_root\ttop_k\tmerge_epochs\tbatch\tskip_if_exists\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.tsv"
            path.write_text(
                header
                + "photo\tnode\tclassification\tTrue\t(5,0.0,1.0)\tdata/splits_shift/mixed\t3\t7\t16\tFalse\n"
                + "photo\tnode\tclassification\tTrue\t(5,0.0,1.0)\t-\n"
                + "photo\tnode\tclassification\tTrue\t(5,0.0,1.0)\tdata/splits\tx\n"
                + "mnist\tgraph\n",
                encoding="utf-8",
            )
            tasks = parse_ogmm_tasks(str(path))
        self.assertEqual(len(tasks), 2)
        self.assertEqual((tasks[0]["split_root"], tasks[0]["top_k"], tasks[0]["merge_epochs"]), ("data/splits_shift/mixed", 3, 7))
        self.assertIsNone(tasks[1]["split_root"])

        run_cfg = _build_task_cfg(_cfg(), tasks[0])
        o = run_cfg.moe.ogmm
        self.assertEqual(run_cfg.data_preparation.dataset.split_root, "data/splits_shift/mixed")
        self.assertEqual((o.dataset.name, o.top_k, o.merge_epochs, o.batch_size, o.skip_if_exists), ("photo", 3, 7, 16, False))
        self.assertFalse(o.run_tasks_tsv)
        for key in ("moe.method", "data_preparation.dataset.split_root", "moe.ogmm.dataset.fixed_split", "moe.ogmm.top_k"):
            self.assertIn(key, run_cfg.save_results.explicit_keys)
        default_cfg = _build_task_cfg(_cfg(), tasks[1])
        self.assertEqual(default_cfg.data_preparation.dataset.split_root, base_cfg.data_preparation.dataset.split_root)

    def test_repository_tsv_covers_table15_shift_grid(self):
        tasks = parse_ogmm_tasks(str(ROOT / "slurm" / "moe.ogmm.all.tsv"))
        self.assertEqual(len(tasks), 54)
        datasets = {"photo", "ogbn-arxiv", "airports", "chameleon", "mnist", "toxcast", "qm7b"}
        self.assertEqual({t["dataset"] for t in tasks}, datasets)
        self.assertEqual({tuple(t["fixed_split"]) for t in tasks}, {(5, 0.0, 1.0), (100, 0.0, 1.0)})
        cells = {(t["dataset"], t["split_root"]) for t in tasks}
        self.assertNotIn(("qm7b", "data/splits_shift/feature"), cells)
        for condition in ("feature", "structural", "mixed"):
            expected = datasets - ({"qm7b"} if condition == "feature" else set())
            self.assertEqual({d for d, root in cells if root == f"data/splits_shift/{condition}"}, expected)
        self.assertEqual({t["task_type"] for t in tasks if t["dataset"] == "qm7b"}, {"regression"})

    def test_run_moe_dispatches_ogmm(self):
        cfg = moe_run._build_moe_cfg(["moe.ogmm.dataset.name", "photo"])
        self.assertEqual(cfg.moe.method, "ogmm")
        with patch("src.moe.ogmm.run_ogmm", return_value=0) as fake:
            self.assertEqual(moe_run.run_moe(cfg), 0)
        fake.assert_called_once()


if __name__ == "__main__":
    unittest.main()
