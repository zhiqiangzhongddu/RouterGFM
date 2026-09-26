"""Unit tests for the Link-MoE baseline (tiny synthetic graphs, CPU only)."""

import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from src.config import cfg as base_cfg
from src.data_loader import make_loaders
from src.data_loader.dataset_splits import _get_or_create_edge_split_payload
from src.data_loader.induced_graphs import build_edge_induced_graphs_supervised
from src.moe.linkmoe.data import (
    align_induced_to_pairs,
    link_views_from_loaders,
    prepare_seal_graphs,
    stratified_split,
)
from src.moe.linkmoe.expert_training import probability_metrics, train_seal_expert
from src.moe.linkmoe.experts import SEALGCN, common_neighbor_index, drnl_labels
from src.moe.linkmoe.gate import BranchMLP, LinkMoEGate, gate_loss, mixture_probability
from src.moe.linkmoe.heuristics import (
    HEURISTIC_NAMES,
    adamic_adar,
    common_neighbors,
    edge_mask,
    inverse_shortest_path,
    katz3,
    pair_heuristics,
    ppr_symmetric,
    resource_allocation,
    symmetric_csr,
)
from src.moe.linkmoe.run import parse_linkmoe_tasks, run_linkmoe
from src.moe.linkmoe.trainer import LinkMoERunner

SPLIT = (0.1, 0.1, 0.1)


def _random_graph(num_nodes=60, num_edges=200, seed=0, isolated=0) -> Data:
    g = torch.Generator().manual_seed(seed)
    active = num_nodes - isolated
    src = torch.randint(0, active, (num_edges,), generator=g)
    dst = torch.randint(0, active, (num_edges,), generator=g)
    keep = src != dst
    edge_index = to_undirected(torch.stack([src[keep], dst[keep]]), num_nodes=num_nodes)
    return Data(x=torch.randn(num_nodes, 8, generator=g), edge_index=edge_index, num_nodes=num_nodes)


def _nx_graph(A: sp.csr_matrix) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(A.shape[0]))
    rows, cols = A.nonzero()
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))
    return G


def _random_pairs(n, count, seed):
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < count:
        i, j = rng.integers(0, n, 2)
        if i != j:
            pairs.add((int(i), int(j)))
    return torch.tensor(sorted(pairs)).t()


def _fast_pagerank(A, p=0.85, max_iter=100, tol=1e-7, personalize=None):
    """Direct port of fast_pagerank.pagerank_power (reference for HeaRT's PPR)."""
    n = A.shape[0]
    r = np.asarray(A.sum(axis=1)).reshape(-1)
    k = r.nonzero()[0]
    D_1 = sp.csr_matrix((1 / r[k], (k, k)), shape=(n, n))
    personalize = personalize.reshape(n, 1)
    s = (personalize / personalize.sum()) * n
    z_T = (((1 - p) * (r != 0) + (r == 0)) / n)[np.newaxis, :]
    W = p * A.T @ D_1
    x = s
    oldx = np.zeros((n, 1))
    iteration = 0
    while np.linalg.norm(x - oldx) > tol:
        oldx = x
        x = W @ x + s @ (z_T @ x)
        iteration += 1
        if iteration >= max_iter:
            break
    x = x / sum(x)
    return np.asarray(x).reshape(-1)


def _seal_drnl(adj, src, dst):
    """Direct port of SEAL_OGB utils.drnl_node_labeling."""
    src, dst = (dst, src) if src > dst else (src, dst)
    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]
    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]
    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = torch.from_numpy(np.insert(dist2src, dst, 0, axis=0))
    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = torch.from_numpy(np.insert(dist2dst, src, 0, axis=0))
    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2
    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.0
    z[dst] = 1.0
    z[torch.isnan(z)] = 0.0
    return z.to(torch.long)


def _synthetic_link_task(tmp: str, seed: int = 42, num_nodes: int = 80, num_edges: int = 400):
    """Full-graph views through the real edge split + loaders, plus the matching induced graphs."""
    data = _random_graph(num_nodes, num_edges, seed=seed)
    loaders = make_loaders(
        dataset=[data], dataset_name="synth", task_level="edge", batch_size=1, num_workers=0,
        split=SPLIT, seed=seed, induced=False, split_root=tmp,
    )
    views = link_views_from_loaders(*loaders, meta={"name": "synth"})
    payload = _get_or_create_edge_split_payload(
        dataset_name=f"synth_edge_seed{seed}", split=SPLIT, seed=seed,
        split_root_path=Path(tmp), data=data, verbose=False,
    )
    context = Data(x=data.x, edge_index=data.edge_index[:, torch.tensor(payload["context_pos_idx"])], num_nodes=num_nodes)
    induced = {}
    for name in ("train", "val", "test"):
        pos = data.edge_index[:, torch.tensor(payload[f"{name}_pos_idx"], dtype=torch.long)]
        induced[name] = build_edge_induced_graphs_supervised(
            context, pos, payload[f"{name}_neg_edge_index"], max_hops=2, max_size=20,
        )
    return data, views, induced


class HeuristicsTest(unittest.TestCase):
    def setUp(self):
        data = _random_graph(40, 90, seed=1, isolated=2)
        self.A = symmetric_csr(data.edge_index, data.num_nodes)
        self.G = _nx_graph(self.A)
        self.n = data.num_nodes
        self.pairs = _random_pairs(self.n, 60, seed=3)  # mixes edges, non-edges, isolated nodes

    def test_set_based_heuristics_match_brute_force(self):
        cn = common_neighbors(self.A, self.pairs)
        aa = adamic_adar(self.A, self.pairs)
        ra = resource_allocation(self.A, self.pairs)
        for p, (i, j) in enumerate(self.pairs.t().tolist()):
            common = set(self.G[i]) & set(self.G[j])
            self.assertEqual(cn[p], len(common))
            self.assertAlmostEqual(aa[p], sum(1 / np.log(self.G.degree(k)) for k in common if self.G.degree(k) > 1))
            self.assertAlmostEqual(ra[p], sum(1 / self.G.degree(k) for k in common))

    def test_inverse_shortest_path_matches_networkx(self):
        inv = inverse_shortest_path(self.A, self.pairs, batch_size=7)
        for p, (i, j) in enumerate(self.pairs.t().tolist()):
            expected = 1.0 / nx.shortest_path_length(self.G, i, j) if nx.has_path(self.G, i, j) else 0.0
            self.assertAlmostEqual(inv[p], expected)

    def test_katz_counts_simple_paths_including_edge_pairs(self):
        beta = 0.1
        rows, cols = self.A.nonzero()
        edge_pairs = torch.from_numpy(np.vstack([rows[:10], cols[:10]])).long()
        pairs = torch.cat([self.pairs, edge_pairs], dim=1)
        got = katz3(self.A, pairs, beta=beta)
        for p, (i, j) in enumerate(pairs.t().tolist()):
            counts = np.zeros(3)
            for path in nx.all_simple_paths(self.G, source=i, target=j, cutoff=3):
                counts[len(path) - 2] += 1
            self.assertAlmostEqual(got[p], float(np.sum(beta ** np.arange(1, 4) * counts)), places=12)

    def test_ppr_matches_fast_pagerank_with_dangling_nodes(self):
        got = ppr_symmetric(self.A, self.pairs, batch_size=5)
        cache = {}
        for p, (i, j) in enumerate(self.pairs.t().tolist()):
            for s in (i, j):
                if s not in cache:
                    e = np.zeros(self.n)
                    e[s] = 1.0
                    cache[s] = _fast_pagerank(self.A, personalize=e)
            self.assertAlmostEqual(got[p], 0.5 * (cache[i][j] + cache[j][i]), places=9)

    def test_pair_heuristics_are_symmetric(self):
        pairs = self.pairs[:, torch.from_numpy(~edge_mask(self.A, self.pairs))]
        forward = pair_heuristics(self.A, pairs)
        backward = pair_heuristics(self.A, pairs.flip(0))
        self.assertEqual(tuple(forward.shape), (pairs.size(1), len(HEURISTIC_NAMES)))
        self.assertEqual(forward.dtype, torch.float32)
        torch.testing.assert_close(forward, backward)


class LeakageGuardTest(unittest.TestCase):
    def test_heldout_positives_absent_and_guard_fires(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, views, _ = _synthetic_link_task(tmp)
        heldout = torch.cat([views.pairs["val"], views.pairs["test"]], dim=1)
        A = symmetric_csr(views.context_edge_index, views.num_nodes)
        struct = pair_heuristics(A, heldout)
        self.assertTrue(torch.isfinite(struct).all())
        M = symmetric_csr(views.message_edge_index, views.num_nodes)
        self.assertFalse(edge_mask(M, heldout).any())
        # train positives are in C (eval context) but not in M (message graph)
        train_pos = views.pairs["train"][:, views.labels["train"] > 0.5]
        self.assertTrue(edge_mask(A, train_pos).all())
        self.assertFalse(edge_mask(M, train_pos).any())

        val_pos = views.pairs["val"][:, views.labels["val"] > 0.5][:, :1]
        leaked = symmetric_csr(torch.cat([views.context_edge_index, val_pos], dim=1), views.num_nodes)
        with self.assertRaisesRegex(ValueError, "held-out"):
            pair_heuristics(leaked, heldout)


class ExpertComponentsTest(unittest.TestCase):
    def test_drnl_matches_seal_reference(self):
        graphs = {
            "path": [(0, 1), (1, 2), (2, 3), (3, 4)],
            "cycle": [(k, (k + 1) % 7) for k in range(7)],
            "two_components": [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (2, 6)],
            "star": [(0, 2), (1, 2), (2, 3), (3, 4), (1, 4), (0, 5)],
        }
        for name, edges in graphs.items():
            n = 1 + max(max(e) for e in edges)
            ei = to_undirected(torch.tensor(edges).t(), num_nodes=n)
            adj = sp.csr_matrix((np.ones(ei.size(1)), (ei[0].numpy(), ei[1].numpy())), shape=(n, n))
            for u, v in [(0, n - 1), (n - 1, 1), (1, 3)]:
                with self.subTest(graph=name, pair=(u, v)):
                    z = drnl_labels(ei, n, u, v, max_z=1000)
                    torch.testing.assert_close(z, _seal_drnl(adj, u, v))
                    self.assertEqual(int(z[u]), 1)
                    self.assertEqual(int(z[v]), 1)
        z = drnl_labels(to_undirected(torch.tensor(graphs["two_components"]).t()), 7, 0, 1, max_z=1000)
        self.assertTrue(torch.all(z[[3, 4, 5]] == 0))
        self.assertEqual(int(drnl_labels(to_undirected(torch.tensor(graphs["cycle"]).t()), 7, 0, 3, max_z=3).max()), 2)

    def test_common_neighbor_sum_matches_dense_loop(self):
        data = _random_graph(30, 90, seed=5)
        pairs = _random_pairs(30, 40, seed=6)
        h = torch.randn(30, 4)
        pos, nbr = common_neighbor_index(data.edge_index, 30, pairs)
        got = torch.zeros(pairs.size(1), 4).index_add_(0, pos, h[nbr])
        G = _nx_graph(symmetric_csr(data.edge_index, 30))
        for p, (i, j) in enumerate(pairs.t().tolist()):
            common = sorted(set(G[i]) & set(G[j]))
            expected = h[common].sum(0) if common else torch.zeros(4)
            torch.testing.assert_close(got[p], expected)


class GateTest(unittest.TestCase):
    def test_gate_mechanics(self):
        torch.manual_seed(0)
        gate = LinkMoEGate(feat_dim=6, struct_dim=8, hidden_dim=8, num_layers=2, num_layers_predictor=2,
                           num_experts=3, dropout=0.0)
        w = gate(torch.randn(10, 6), torch.randn(10, 8))
        self.assertEqual(tuple(w.shape), (10, 3))
        torch.testing.assert_close(w.sum(-1), torch.ones(10))
        self.assertTrue(torch.all(w >= 0))
        q = mixture_probability(w, torch.rand(3, 10))
        self.assertTrue(torch.all((q >= 0) & (q <= 1)))
        self.assertTrue(torch.all(BranchMLP(4, 5, 1, 0.0)(torch.randn(20, 4) * 10) >= 0))

        y = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
        q = torch.tensor([0.9, 0.2, 0.4, 0.7, 0.1])
        bce = torch.nn.functional.binary_cross_entropy(torch.sigmoid(q), y, reduction="none")
        expected = bce[y > 0.5].mean() + bce[y < 0.5].mean()
        torch.testing.assert_close(gate_loss(q, y, 1.0), expected)
        torch.testing.assert_close(gate_loss(q, y, 10.0), bce[y > 0.5].mean() + 10 * bce[y < 0.5].mean())

    def test_gate_learns_local_routing(self):
        torch.manual_seed(0)
        n = 400
        y = (torch.rand(n) < 0.5).float()
        cn = torch.where(torch.arange(n) % 2 == 0, torch.tensor(3.0), torch.tensor(0.0))
        correct = torch.where(y > 0.5, torch.tensor(0.9), torch.tensor(0.1))
        noise = torch.rand(2, n)
        probs = torch.stack([
            torch.where(cn > 0, correct, noise[0]),
            torch.where(cn > 0, noise[1], correct),
        ])
        struct = torch.zeros(n, 8)
        struct[:, 1] = cn
        feat = torch.randn(n, 4)
        gate = LinkMoEGate(4, 8, hidden_dim=16, num_layers=2, num_layers_predictor=1, num_experts=2, dropout=0.0)
        opt = torch.optim.Adam(gate.parameters(), lr=1e-2)
        for _ in range(300):
            loss = gate_loss(mixture_probability(gate(feat, struct), probs), y, 10.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            w = gate(feat, struct)
            q = mixture_probability(w, probs)
        self.assertGreater(float(w[cn > 0, 0].mean()), 0.7)
        self.assertLess(float(w[cn == 0, 0].mean()), 0.3)
        mix_auc = probability_metrics(q, y)["auc"]
        self.assertGreaterEqual(mix_auc, 0.95)
        for k in range(2):
            self.assertGreater(mix_auc, probability_metrics(probs[k], y)["auc"])

    def test_stratified_split(self):
        labels = torch.cat([torch.ones(14), torch.zeros(14)])
        tr, va = stratified_split(labels, 0.8, seed=3)
        tr2, va2 = stratified_split(labels, 0.8, seed=3)
        self.assertTrue(torch.equal(tr, tr2) and torch.equal(va, va2))
        self.assertFalse(set(tr.tolist()) & set(va.tolist()))
        self.assertEqual(sorted(tr.tolist() + va.tolist()), list(range(28)))
        self.assertEqual(int(labels[tr].sum()), 11)
        self.assertEqual(int((labels[va] == 0).sum()), 3)
        self.assertFalse(torch.equal(stratified_split(labels, 0.8, seed=4)[0], tr))

    def test_metrics_invariant_to_probability_transform(self):
        q = torch.rand(50) * 0.98 + 0.01
        y = (torch.rand(50) < 0.5).float()
        auc_q = probability_metrics(q, y)["auc"]
        self.assertAlmostEqual(auc_q, probability_metrics(torch.sigmoid(q), y)["auc"])
        from sklearn.metrics import roc_auc_score
        self.assertAlmostEqual(auc_q, roc_auc_score(y.numpy(), q.numpy()))


class SealViewTest(unittest.TestCase):
    def test_alignment_restores_pair_order_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, views, induced = _synthetic_link_task(tmp)
        graphs = list(induced["val"])
        random.Random(0).shuffle(graphs)
        aligned = align_induced_to_pairs(graphs, views.pairs["val"], views.labels["val"])
        for g, (i, j), y in zip(aligned, views.pairs["val"].t().tolist(), views.labels["val"].tolist()):
            self.assertEqual(sorted(g.global_target_pair.tolist()), sorted([i, j]))
            self.assertEqual(int(g.y), int(y))
        with self.assertRaisesRegex(ValueError, "do not match"):
            align_induced_to_pairs(graphs[1:], views.pairs["val"], views.labels["val"])
        with self.assertRaisesRegex(ValueError, "duplicated"):
            align_induced_to_pairs(graphs + graphs[:1], views.pairs["val"], views.labels["val"])

    def test_seal_expert_scores_aligned_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, views, induced = _synthetic_link_task(tmp)
        seal_view = {s: prepare_seal_graphs(induced[s], views.pairs[s], views.labels[s], max_z=10) for s in induced}
        g = seal_view["val"][0]
        self.assertEqual(int(g.z[g.edge_label_index[0, 0]]), 1)
        self.assertLessEqual(int(max(int(x.z.max()) for x in seal_view["train"])), 9)
        cfg = base_cfg.clone()
        cfg.moe.linkmoe.seal.max_epochs = 2
        cfg.moe.linkmoe.seal.batch_size = 16
        expert = SEALGCN(views.x.size(1), hidden_dim=8, num_layers=2, max_z=10, dropout=0.0)
        scores = train_seal_expert(expert, seal_view, views, cfg, torch.device("cpu"), seed=0)
        self.assertEqual(scores.val_prob.numel(), views.pairs["val"].size(1))
        self.assertEqual(scores.test_prob.numel(), views.pairs["test"].size(1))
        self.assertTrue(torch.all((scores.test_prob >= 0) & (scores.test_prob <= 1)))


def _tiny_cfg(tmp: str):
    cfg = base_cfg.clone()
    cfg.seed = 42
    lcfg = cfg.moe.linkmoe
    lcfg.dataset.name = "synth"
    lcfg.dataset.fixed_split = SPLIT
    lcfg.experts = ("mlp", "gcn", "ncn", "seal")
    lcfg.expert_max_epochs = 3
    lcfg.expert_batch_size = 64
    for name in ("mlp", "gcn", "ncn", "seal"):
        getattr(lcfg, name).hidden_dim = 8
    lcfg.seal.max_epochs = 2
    lcfg.seal.max_z = 10
    lcfg.gate.epochs = 20
    lcfg.gate.hidden_dim = 8
    lcfg.checkpoint_dir = os.path.join(tmp, "ckpt")
    lcfg.log_dir = os.path.join(tmp, "logs")
    lcfg.skip_if_exists = False
    return cfg


class RunnerTest(unittest.TestCase):
    def test_tiny_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, views, induced = _synthetic_link_task(os.path.join(tmp, "splits"))
            seal_view = {s: prepare_seal_graphs(induced[s], views.pairs[s], views.labels[s], max_z=10) for s in induced}
            cfg = _tiny_cfg(tmp)
            with mock.patch("src.moe.linkmoe.trainer.build_link_views", return_value=views), \
                    mock.patch("src.moe.linkmoe.trainer.build_seal_view", return_value=seal_view):
                runner = LinkMoERunner(cfg)
                runner.fit()
            metrics = runner.best_metrics
            for key in ("val_auc", "val_acc", "test_auc", "test_acc", "best_epoch", "train_loss"):
                self.assertIn(key, metrics)
            self.assertTrue(0.0 <= metrics["test_auc"] <= 1.0)
            self.assertTrue(1 <= metrics["best_epoch"] <= 20)
            ckpt = torch.load(runner.get_checkpoint_path_for_metrics(), map_location="cpu")
            self.assertEqual(ckpt["extra"]["experts"], ["mlp", "gcn", "ncn", "seal"])
            self.assertEqual(set(ckpt["extra"]["expert_states"]), {"mlp", "gcn", "ncn", "seal"})
            split = ckpt["extra"]["gate_split"]
            self.assertEqual(split["train"].numel() + split["val"].numel(), views.pairs["val"].size(1))
            self.assertTrue(os.path.isfile(runner._log_path()))
            self.assertIn("seed42", runner.run_name)

            cfg.moe.linkmoe.skip_if_exists = True
            self.assertTrue(LinkMoERunner(cfg)._skip_due_to_existing_checkpoint)

    def test_run_identity_ignores_operational_knobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_cfg(tmp)
            baseline = LinkMoERunner(cfg).run_name
            changed = cfg.clone()
            changed.moe.linkmoe.num_runs = 9
            changed.moe.linkmoe.log_dir = "elsewhere"
            self.assertEqual(LinkMoERunner(changed).run_name, baseline)
            changed.moe.linkmoe.gate.lr = 1e-2
            self.assertNotEqual(LinkMoERunner(changed).run_name, baseline)
            changed = cfg.clone()
            changed.data_preparation.dataset.split_root = "data/alternate_splits"
            self.assertNotEqual(LinkMoERunner(changed).run_name, baseline)


class PlumbingTest(unittest.TestCase):
    def test_config_tsv_and_dispatch(self):
        self.assertEqual(base_cfg.moe.linkmoe.dataset.task_level, "edge")
        self.assertEqual(tuple(base_cfg.moe.linkmoe.dataset.fixed_split), (0.1, 0.05, 0.1))
        self.assertEqual(tuple(base_cfg.moe.linkmoe.experts), ("mlp", "gcn", "ncn", "seal"))
        from src.moe.run import _load_runner
        self.assertIs(_load_runner("linkmoe"), run_linkmoe)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tasks.tsv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# dataset\ttask_level\ttask_type\texperts\tfixed_split\tgate_epochs\tskip_if_exists\n")
                fh.write("dblp\tedge\tclassification\tmlp,gcn\t(0.1,0.05,0.1)\t50\tFalse\n")
                fh.write("cornell\tedge\t-\t-\t-\t-\tTrue\n")
            tasks = parse_linkmoe_tasks(path)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["experts"], ("mlp", "gcn"))
        self.assertEqual(tasks[0]["fixed_split"], (0.1, 0.05, 0.1))
        self.assertEqual(tasks[0]["gate_epochs"], 50)
        self.assertIsNone(tasks[1]["experts"])
        self.assertIsNone(tasks[1]["gate_epochs"])

        repo_tasks = parse_linkmoe_tasks(str(Path(__file__).resolve().parents[1] / "slurm" / "moe.linkmoe.all.tsv"))
        self.assertEqual([t["dataset"] for t in repo_tasks], ["dblp", "cornell"])
        self.assertTrue(all(t["task_level"] == "edge" and t["fixed_split"] == (0.1, 0.05, 0.1) for t in repo_tasks))

    def test_non_link_tasks_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for level, split in (("node", (0.1, 0.05, 0.1)), ("graph", (0.1, 0.05, 0.1)), ("edge", (5, 0.0, 1.0))):
                with self.subTest(level=level, split=split):
                    cfg = _tiny_cfg(tmp)
                    cfg.moe.linkmoe.dataset.task_level = level
                    cfg.moe.linkmoe.dataset.fixed_split = split
                    with self.assertRaises(ValueError):
                        LinkMoERunner(cfg)
                    with self.assertRaises(ValueError):
                        run_linkmoe(cfg)
            cfg = _tiny_cfg(tmp)
            cfg.moe.linkmoe.experts = ("mlp", "gin")
            with self.assertRaisesRegex(ValueError, "experts"):
                LinkMoERunner(cfg)


if __name__ == "__main__":
    unittest.main()
