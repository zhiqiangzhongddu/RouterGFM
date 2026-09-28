"""MetaGL / MetaGL+metadata selectors and the MetaGL-U matched runner."""

from __future__ import annotations

import csv
import json
import math
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data

from src.moe.routergfm.baselines import load_runner_class
from src.moe.routergfm.baselines import run as matched_run
from src.moe.routergfm.baselines.metagl_u import MetaGLURunner
from src.moe.routergfm.baselines.selection import run as selection_run
from src.moe.routergfm.baselines.selection.metagl import MetaGLSelector
from src.moe.routergfm.baselines.selection.metagl.factorization import factorize, sparse_nmf
from src.moe.routergfm.baselines.selection.metagl.features import (
    N_META_GRAPH_FEATURES,
    aat_nnz,
    application_meta_features,
    disjoint_union,
    gini,
    meta_graph_features,
    structural_stat_vector,
)
from src.moe.routergfm.baselines.selection.metagl.hgt import HGTLayer
from src.moe.routergfm.baselines.selection.metagl.model import best_model_auc_mrr, top_one_loss, validation_score
from src.moe.routergfm.baselines.selection.metagl.network import (
    M_G2G,
    M_M2M,
    NODE_TYPES,
    P_G2G,
    P_G2M,
    P_M2G,
    P_M2M,
    RELATIONS,
    add_graphs,
    add_models,
    build_network,
    knn_pairs,
)
from src.moe.routergfm.baselines.selection.metagl.selector import inner_split, row_minmax
from src.moe.routergfm.common import AppSpec, enumerate_applications
from src.moe.routergfm.context_graph import EXPERT_NUMERIC_NAMES
from src.moe.routergfm.history import generate_history
from src.moe.routergfm.infra import RouterInfra
from src.utils.metrics import compute_supervised_metrics
from tests.routergfm.fixtures import SyntheticDataProvider, tiny_cfg


def _edges(pairs, directed_both=True) -> torch.Tensor:
    ei = torch.tensor(pairs, dtype=torch.long).t().reshape(2, -1)
    return torch.cat([ei, ei.flip(0)], dim=1) if directed_both else ei


# --------------------------------------------------------------------------- #
# Meta-graph features
# --------------------------------------------------------------------------- #
def test_structural_stat_vector_values():
    rng = np.random.default_rng(0)
    for x in (rng.random(30), np.ones(5), np.array([0.0, 1.0, 2.0]), np.array([3.0])):
        v = structural_stat_vector(x)
        assert v.shape == (63,) and np.isfinite(v).all()
    x = np.array([1.0, 2.0, 2.0, 3.0, 10.0])
    v = structural_stat_vector(x)
    assert v[:6] == pytest.approx([3.6, 2.0, 1.0, 10.0, np.var(x), np.std(x)])
    assert v[6] == pytest.approx(stats.entropy(x)) and v[8] == pytest.approx(gini(x))
    assert v[37:42] == pytest.approx([1.0, 2.0, 3.0, 0.5, 4.5])  # iqr, q1, q3 (index rule), lb, ub
    assert v[23:29] == pytest.approx([0.2, 0.0, 0.2, 1.0, 0.0, 1.0])  # 10 is the single 1.5-IQR outlier
    assert v[-3:] == pytest.approx([2.0, 2.0, 0.4])  # mode, count, fraction


def test_meta_graph_features_blocks_and_invariance():
    path5 = _edges([(0, 1), (1, 2), (2, 3), (3, 4)])
    graphs = [
        (path5, 5),
        (_edges([(0, i) for i in range(1, 6)]), 6),  # star S6
        (_edges([(0, 1), (1, 2), (2, 0), (2, 3)]), 4),  # triangle + pendant
        (torch.tensor([[0, 1], [0, 2]]), 5),  # self-loop, one edge, isolated nodes
    ]
    for ei, n in graphs:
        f = meta_graph_features(ei, n)
        assert f.shape == (N_META_GRAPH_FEATURES,) and np.isfinite(f).all()
    f = meta_graph_features(path5, 5)
    assert f[3:66] == pytest.approx(structural_stat_vector(np.array([1, 2, 2, 2, 1.0])))
    k4 = _edges([(i, j) for i in range(4) for j in range(i + 1, 4)])
    assert meta_graph_features(k4, 4)[3 + 4 * 63] == pytest.approx(3.0)  # triangles per node
    p3 = meta_graph_features(_edges([(0, 1), (1, 2)]), 3)
    assert p3[0] == pytest.approx(4 / 6) and p3[1] == pytest.approx(5 / 6)  # density(A), density(A A^T) (product)

    g = nx.gnm_random_graph(40, 90, seed=1)
    ei = _edges(list(g.edges()))
    perm = torch.randperm(40, generator=torch.Generator().manual_seed(0))
    assert meta_graph_features(perm[ei], 40) == pytest.approx(meta_graph_features(ei, 40), rel=1e-9, abs=1e-12)


def test_aat_nnz_chunked():
    g = nx.gnm_random_graph(50, 120, seed=3)
    A = sp.csr_matrix(nx.to_scipy_sparse_array(g, nodelist=range(50)))
    assert aat_nnz(A, chunk=2) == (A @ A.T).nnz == aat_nnz(A)


class _GraphOnlyInfra:
    def __init__(self, support, query):
        self.pools = {"support": support, "query": query}
        self.calls = 0

    def instance_graphs(self, app, split):
        self.calls += 1
        return self.pools[split]

    def __getattr__(self, name):
        raise AssertionError(f"meta-graph features accessed infra.{name}")


def test_application_meta_features_sample_union_and_cache(tmp_path):
    cfg = tiny_cfg(tmp_path, write_checkpoints=False)
    cfg.moe.routergfm.baselines.metagl.graph_sample_max = 3
    graphs = [Data(edge_index=_edges([(0, 1), (1, i)]), num_nodes=i + 1) for i in range(2, 7)]
    infra = _GraphOnlyInfra(graphs[:2], graphs[2:])
    app = AppSpec("toy", "graph", 5, 0)
    feats = application_meta_features(infra, app, cfg)
    chosen = np.sort(np.random.default_rng(0).choice(5, size=3, replace=False))
    assert feats == pytest.approx(meta_graph_features(*disjoint_union([graphs[i] for i in chosen])))
    calls = infra.calls
    assert application_meta_features(infra, app, cfg) == pytest.approx(feats) and infra.calls == calls  # cached


class _BaseGraphInfra:
    def __init__(self):
        self.calls = []

    def base_graph(self, app):
        self.calls.append(app.key)
        g = nx.gnm_random_graph(30, 60, seed=int(app.seed) if app.task_level == "edge" else 7)
        return Data(edge_index=_edges(list(g.edges())), num_nodes=30)

    def instance_graphs(self, app, split):
        raise AssertionError("node / link meta-graph features read instance subgraphs")

    def __getattr__(self, name):
        raise AssertionError(f"meta-graph features accessed infra.{name}")


def test_application_meta_features_use_the_input_graph_for_node_and_link(tmp_path):
    cfg = tiny_cfg(tmp_path, write_checkpoints=False)
    infra = _BaseGraphInfra()
    node = AppSpec("toy", "node", 5, 42)
    base = infra.base_graph(node)
    infra.calls.clear()
    feats = application_meta_features(infra, node, cfg)
    assert feats == pytest.approx(meta_graph_features(base.edge_index, base.num_nodes))
    for other in (AppSpec("toy", "node", 100, 0), AppSpec("toy", "node", 5, 123)):  # one cache per dataset
        assert application_meta_features(infra, other, cfg) == pytest.approx(feats)
    assert infra.calls == [node.key]

    for seed in (42, 0):  # link: the split's message-passing graph, cached per data key
        link = AppSpec("toy", "edge", 5, seed)
        expected = meta_graph_features(infra.base_graph(link).edge_index, 30)
        assert application_meta_features(infra, link, cfg) == pytest.approx(expected)
    assert infra.calls == [node.key] + [AppSpec("toy", "edge", 5, s).key for s in (42, 42, 0, 0)]


# --------------------------------------------------------------------------- #
# Loss, factorization, network, HGT, validation criterion
# --------------------------------------------------------------------------- #
def test_top_one_loss():
    pred = torch.tensor([[1.0, 2.0, 0.5], [0.0, 0.0, 3.0]], requires_grad=True)
    true = torch.tensor([[11.0, 1.0, 6.0], [1.0, 11.0, 1.0]])
    expected = -(torch.softmax(true, 1) * torch.log(torch.softmax(pred, 1) + 1e-10)).sum(1).mean()
    loss = top_one_loss(pred, true)
    assert float(loss) == pytest.approx(float(expected))
    loss.backward()
    assert pred.grad[0, 0] < 0 and pred.grad[1, 1] < 0  # descent raises the true best logits

    masked = torch.tensor([[11.0, float("nan"), 6.0], [1.0, 11.0, float("nan")]])
    full = top_one_loss(pred.detach(), masked)
    rows = [top_one_loss(pred.detach()[i : i + 1, keep], masked[i : i + 1, keep]) for i, keep in ((0, [0, 2]), (1, [0, 1]))]
    assert float(full) == pytest.approx(float(sum(rows) / 2))


def test_sparse_nmf_and_factorize():
    rng = np.random.default_rng(0)
    X = rng.random((20, 3)) @ rng.random((3, 15))
    missing = rng.random(X.shape) < 0.3
    Xm = X.copy()
    Xm[missing] = np.nan
    A, Y = sparse_nmf(Xm, 3, max_iter=2000, error_limit=1e-12, fit_error_limit=1e-12, rng=np.random.default_rng(1))
    assert A.shape == (20, 3) and Y.shape == (3, 15) and (A > 0).all() and (Y > 0).all()
    obs = ~missing
    assert np.linalg.norm((A @ Y - X)[obs]) / np.linalg.norm(X[obs]) < 0.1
    U, V, kind = factorize(Xm, 3, np.random.default_rng(0))
    assert kind == "sparse_nmf" and U.shape == (20, 3) and V.shape == (15, 3)
    U, V, kind = factorize(X, 3, np.random.default_rng(0))
    assert kind == "pca" and U.shape == (20, 3) and V.shape == (15, 3)


def test_knn_pairs_and_network_extensions():
    X = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    src, dst = knn_pairs(X, X, 2)
    assert src.tolist() == [0, 0, 1, 1, 2, 2] and dst.view(3, 2)[:, 0].tolist() == [0, 1, 2]  # self first
    src, dst = knn_pairs(X[:1], X[1:], 5)  # k clipped to |Y|; src = the querying row
    assert src.tolist() == [0, 0] and dst.tolist() == [0, 1]

    g = torch.Generator().manual_seed(0)
    Mp, U, V, v = (torch.rand(n, d, generator=g) for n, d in ((6, 4), (6, 3), (5, 3), (5, 2)))
    k = 2
    net = build_network(Mp, U, V, k, v)
    assert net.num_nodes == {"graph": 6, "model": 5}
    assert {rel: e.size(1) for rel, e in net.edges.items()} == {
        M_G2G: 12, P_G2G: 12, P_M2M: 10, P_G2M: 12, P_M2G: 10, M_M2M: 10,
    }
    Mp_t, U_t = torch.rand(2, 4, generator=g), torch.rand(2, 3, generator=g)
    ext = add_graphs(net, Mp, U, V, Mp_t, U_t, k)
    assert net.num_nodes["graph"] == 6 and ext.num_nodes["graph"] == 8  # input network untouched
    for rel in (M_G2G, P_G2G):
        new = ext.edges[rel][:, net.edges[rel].size(1):]
        assert (new[0] < 6).all() and torch.bincount(new[1] - 6).tolist() == [k, k]  # train -> test only
    assert ext.edges[P_G2M].size(1) == 12 + 2 * k and ext.edges[P_M2G].size(1) == 10 + 5 * 2  # min(k, #new)=2
    V_new, v_new = torch.rand(3, 3, generator=g), torch.rand(3, 2, generator=g)
    both = add_models(ext, V, V_new, torch.cat([U, U_t]), k, v, v_new)
    assert both.num_nodes["model"] == 8
    for rel in (M_M2M, P_M2M):
        new = both.edges[rel][:, ext.edges[rel].size(1):]
        assert (new[0] < 5).all() and torch.bincount(new[1] - 5).tolist() == [k, k, k]
    assert both.edges[P_M2G].size(1) == ext.edges[P_M2G].size(1) + 3 * k
    assert both.edges[P_G2M].size(1) == ext.edges[P_G2M].size(1) + 8 * k


def _reference_hgt(layer, h, edges):
    """Loop implementation of the DGL HGT layer (per-relation softmax, mean over relation types)."""
    H, dk = layer.n_heads, layer.d_k
    incoming = {t: [] for t in layer.node_types}
    for r, rel in enumerate(layer.relations):
        s, _, t = rel
        K = layer.k_lin[s](h[s]).view(-1, H, dk)
        Vv = layer.v_lin[s](h[s]).view(-1, H, dk)
        Q = layer.q_lin[t](h[t]).view(-1, H, dk)
        msg = torch.zeros(h[t].size(0), layer.dim)
        ei = edges.get(rel, torch.zeros((2, 0), dtype=torch.long))
        for d in range(h[t].size(0)):
            srcs = ei[0][ei[1] == d]
            if srcs.numel() == 0:
                continue
            for hh in range(H):
                kk = K[srcs, hh] @ layer.relation_att[r, hh]
                vv = Vv[srcs, hh] @ layer.relation_msg[r, hh]
                att = torch.softmax((kk @ Q[d, hh]) * layer.relation_pri[r, hh] / math.sqrt(dk), 0)
                msg[d, hh * dk : (hh + 1) * dk] = att @ vv
        incoming[t].append(msg)
    out = {}
    for i, t in enumerate(layer.node_types):
        alpha = torch.sigmoid(layer.skip[i])
        z = layer.a_lin[t](sum(incoming[t]) / len(incoming[t])) * alpha + h[t] * (1 - alpha)
        out[t] = layer.norms[t](z)
    return out


def test_hgt_layer_matches_reference_semantics():
    torch.manual_seed(0)
    layer = HGTLayer(8, NODE_TYPES, RELATIONS, n_heads=2, dropout=0.5).eval()
    with torch.no_grad():
        layer.relation_pri.uniform_(0.5, 2.0)
        layer.skip.uniform_(-1.0, 1.0)
    h = {"graph": torch.randn(3, 8), "model": torch.randn(2, 8)}
    edges = {
        M_G2G: torch.tensor([[0, 1, 1], [1, 0, 1]]),  # graph 2 has no M_g2g in-edge
        P_G2G: torch.tensor([[0, 1, 2], [2, 2, 2]]),
        P_M2M: torch.tensor([[0, 1], [1, 1]]),
        P_G2M: torch.tensor([[0, 2], [0, 0]]),  # model 1 has no P_g2m in-edge
        P_M2G: torch.tensor([[0, 1, 1], [0, 1, 2]]),
    }
    with torch.no_grad():
        out = layer(h, edges)
        ref = _reference_hgt(layer, h, edges)
        att, _ = layer.relation_attention(RELATIONS.index(P_G2G), h, edges[P_G2G])
        assert torch.allclose(att.sum(0), torch.ones(2))  # one destination: attention sums to 1 per head
        for t in NODE_TYPES:
            assert torch.allclose(out[t], ref[t], atol=1e-6)
            assert torch.equal(out[t], layer(h, edges)[t])  # eval: dropout off, deterministic
        layer.train()
        assert not torch.allclose(layer(h, edges)["graph"], out["graph"])  # dropout active in training


def test_validation_criterion_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        true = rng.random(7)
        pred = np.round(rng.random(7), 1)  # ties
        onehot = (np.arange(7) == np.argmax(true)).astype(int)
        auc, mrr = best_model_auc_mrr(true, pred)
        assert auc == pytest.approx(roc_auc_score(onehot, pred)) and mrr == pytest.approx(average_precision_score(onehot, pred))
    P = np.array([[11.0, 1.0, np.nan], [1.0, np.nan, np.nan], [1.0, 11.0, 6.0]])
    P_hat = np.array([[2.0, 1.0, 9.0], [0.0, 0.0, 0.0], [3.0, 1.0, 2.0]])
    # row 2 is skipped (< 2 observed); row 3: best ranked 3rd -> AUC 0, MRR 1/3
    assert validation_score(P, P_hat) == pytest.approx(np.mean([2 * 0.5, (1 + 1 / 3) / 2]))


def test_row_minmax_and_inner_split():
    P = np.array([[-0.4, -0.1, np.nan], [-0.2, -0.2, -0.2]])
    assert np.allclose(row_minmax(P), [[1.0, 11.0, np.nan], [1.0, 1.0, 1.0]], equal_nan=True)
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    tr, va = inner_split(groups, 0.3, 42)
    assert {groups[i] for i in tr}.isdisjoint({groups[i] for i in va}) and len({groups[i] for i in va}) == 2
    tr, va = inner_split(["a"] * 4, 0.3, 0)
    assert len(va) == 2 and sorted(np.concatenate([tr, va]).tolist()) == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# Selectors on a synthetic archive
# --------------------------------------------------------------------------- #
def _structure(cluster: int, seed: int) -> Data:
    g = nx.erdos_renyi_graph(30, 0.1, seed=seed) if cluster == 0 else nx.barabasi_albert_graph(30, 3, seed=seed)
    return Data(edge_index=_edges(list(g.edges())), num_nodes=30)


class _World:
    """Groups g0..: even -> ER structure (experts x00-x04 best), odd -> BA (x05-x09 best).

    Any accessor outside the label-free selection inputs raises.
    """

    def __init__(self, n_groups=12, seeds=(0, 1), n_experts=20, family=None, unobserved=()):
        self.apps, self.mu, self.graphs = [], {}, {}
        self.experts = [f"x{j:02d}" for j in range(n_experts)]
        self.catalog = [SimpleNamespace(expert_id=e) for e in self.experts]
        self.family = dict(family or {})
        g = torch.Generator().manual_seed(0)
        for i in range(n_groups):
            self.graphs[f"g{i}"] = [_structure(i % 2, 100 * i + s) for s in range(4)]
            for seed in seeds:
                app = AppSpec(f"g{i}", "graph", 5, seed)
                self.apps.append(app)
                for j, e in enumerate(self.experts):
                    if e in unobserved:
                        continue
                    good = (j < 5) if i % 2 == 0 else (5 <= j < 10)
                    self.mu[(app.key, e)] = (0.1 if good else 0.4) + 0.05 * float(torch.rand(1, generator=g))
        self.graphs["new"] = [_structure(1, 9000 + s) for s in range(4)]
        g_meta = torch.Generator().manual_seed(1)
        numeric = torch.zeros(len(self.experts), len(EXPERT_NUMERIC_NAMES))
        numeric[:5, 0], numeric[5:10, 1] = 1.0, 1.0  # a metadata factor aligned with expertise
        self.expert_x = {e: torch.cat([torch.randn(6, generator=g_meta), numeric[j]]) for j, e in enumerate(self.experts)}

    def historical_applications(self, target):
        return list(self.apps)  # includes the target's group: the selector must drop it

    def compatible_pool(self, app):
        return list(self.experts)

    def task_family(self, app):
        return self.family.get(app.dataset, "graph_cls")

    def historical_mu(self, app, expert_id):
        value = self.mu.get((app.key, expert_id))
        return (float("nan"), 0) if value is None else (value, 3)

    def instance_graphs(self, app, split):
        graphs = self.graphs[app.dataset]
        return graphs[:1] if split == "support" else graphs[1:]

    def expert_metadata(self, expert_id):
        return self.expert_x[expert_id]

    def __getattr__(self, name):
        raise AssertionError(f"MetaGL accessed infra.{name}")


def _selector_cfg(tmp_path, **overrides):
    cfg = tiny_cfg(tmp_path, write_checkpoints=False)
    m = cfg.moe.routergfm.baselines.metagl
    m.epochs, m.patience, m.knn_k, m.rf_n_estimators = 80, 80, 3, 20
    for key, value in overrides.items():
        setattr(m, key, value)
    cfg.moe.routergfm.baselines.topk = 5
    return cfg


BA_BEST = {f"x{j:02d}" for j in range(5, 10)}


def test_metagl_learns_structure_and_excludes_target_group(tmp_path):
    cfg = _selector_cfg(tmp_path)
    world = _World()
    target = AppSpec("g1", "graph", 5, 42)  # BA group whose rows are in the offered history
    out = MetaGLSelector(cfg, world).rank(target)
    assert out.extras["n_rows"] == len(world.apps) - 2 and not out.extras["pooled_fallback"]
    assert out.extras["slice"] == "graph_cls/b5" and out.num_target_executions == 0
    assert len(set(out.team) & BA_BEST) >= 4 and out.ranking[0][0] in BA_BEST  # lowest mu_bar ranked first
    assert sorted(e for e, _ in out.ranking) == world.experts and out.team == [e for e, _ in out.ranking[:5]]
    scores = [s for _, s in out.ranking]
    assert scores == sorted(scores, reverse=True) and all(math.isfinite(s) for s in scores)
    assert MetaGLSelector(cfg, _World()).rank(target).ranking == out.ranking  # deterministic


def test_metagl_unscorable_experts_and_fallback(tmp_path):
    cfg = _selector_cfg(tmp_path, epochs=5, patience=5)
    world = _World(n_groups=6, unobserved=("x07",), family={"g0": "node_cls", "g2": "node_cls"})
    target = AppSpec("g4", "graph", 5, 42)  # graph_cls slice: g1, g3, g5 x 2 seeds = 6 rows < 10
    selector = MetaGLSelector(cfg, world)
    out = selector.rank(target)
    assert out.extras["pooled_fallback"] and out.extras["n_rows"] == 10 and out.extras["slice"] == "pooled/b5"
    assert selector.fit_for(target).m_scaler.shift.numel() == N_META_GRAPH_FEATURES + 5  # task-family one-hot
    assert out.ranking[-1] == ("x07", -math.inf) and out.extras["n_new_experts"] == 1
    hidden = selector.rank(target, hidden_experts=["x05"])
    assert dict(hidden.ranking)["x05"] == -math.inf and hidden.extras["n_columns"] == 18


def test_metagl_metadata_scores_inserted_experts(tmp_path):
    cfg = _selector_cfg(tmp_path)
    world = _World()
    target = AppSpec("new", "graph", 5, 42)
    hidden = ["x05", "x06"]
    plain = MetaGLSelector(cfg, world).rank(target, hidden_experts=hidden)
    assert {e for e, s in plain.ranking if not math.isfinite(s)} == set(hidden)
    selector = MetaGLSelector(cfg, world, use_metadata=True)
    out = selector.rank(target, hidden_experts=hidden)
    scores = dict(out.ranking)
    assert all(math.isfinite(scores[e]) for e in world.experts) and out.extras["n_new_experts"] == 2
    assert set(hidden) & set(out.team) and len(set(out.team) & BA_BEST) >= 4
    assert selector.fit_for(target, hidden).v_meta.size(1) == 6 + len(EXPERT_NUMERIC_NAMES)


class _TargetBlindWorld(_World):
    def historical_mu(self, app, expert_id):
        assert app.group != "new", "the selector read the target's history"
        return super().historical_mu(app, expert_id)


def test_metagl_known_target_is_an_inner_train_row(tmp_path):
    """Table 10 known-application conditions: the caller's averages form the target's P row."""
    cfg = _selector_cfg(tmp_path)
    world = _TargetBlindWorld()
    target = AppSpec("new", "graph", 5, 42)  # BA structure: x05-x09 best as a new application
    hidden = ["x05", "x06"]
    er_best = {f"x{j:02d}" for j in range(5)}
    # Observed on the target: the ER experts are best (against its structure); hidden ones are ignored.
    known = {e: (0.1 if e in er_best else 0.4) for e in world.experts}
    selector = MetaGLSelector(cfg, world, use_metadata=True)
    new_app = selector.rank(target, hidden_experts=hidden)
    out = selector.rank(target, hidden_experts=hidden, known_mu=known)
    fit, base = selector.fit_for(target, hidden, known), selector.fit_for(target, hidden)
    assert out.extras["known_target"] and not new_app.extras["known_target"]
    assert fit.rows[:-1] == base.rows and fit.rows[-1] == target and fit.target_row == len(base.rows)
    assert fit.n_train_rows == base.n_train_rows + 1 and fit.n_val_rows == base.n_val_rows  # historical split kept
    assert out.extras["n_new_experts"] == 2 and all(math.isfinite(s) for _, s in out.ranking)

    def er_gap(outcome):  # mean score of the ER experts minus that of the others
        scores = dict(outcome.ranking)
        er = [scores[e] for e in er_best]
        rest = [v for e, v in scores.items() if e not in er_best]
        return sum(er) / len(er) - sum(rest) / len(rest)

    assert len(set(new_app.team) & BA_BEST) >= 4 and er_gap(new_app) < 0
    assert er_gap(out) > er_gap(new_app) + 1.0  # the observed row pulls the target's scores toward it
    # Fewer than two usable averages (hidden or non-finite ones dropped): a new application again.
    few = {"x05": 0.1, "x00": 0.1, "x01": float("nan")}
    assert selector.fit_for(target, hidden, few) is base
    assert selector.rank(target, hidden_experts=hidden, known_mu=few).ranking == new_app.ranking


# --------------------------------------------------------------------------- #
# MetaGL-U
# --------------------------------------------------------------------------- #
class _MatchedWorld(_World):
    """Adds fitted-head query predictions and the evaluation helper (3 classes, 6 queries)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(2)
        self.labels = torch.tensor([0, 1, 2, 0, 1, 2])
        self.preds = {e: torch.softmax(torch.randn(6, 3, generator=g), 1) for e in self.experts}
        self.requested = []

    def expert_predictions(self, app, expert_ids):
        self.requested.append(list(expert_ids))
        return {e: {"pred": self.preds[e]} for e in expert_ids}

    def query_expert_risk(self, app, expert_id):
        raise AssertionError("the runner never reads per-expert target risks")

    def evaluate_outputs(self, app, pred):
        metrics = compute_supervised_metrics(torch.log(pred), self.labels, "classification")
        onehot = torch.nn.functional.one_hot(self.labels, 3).float()
        return {**metrics, "risk": float((0.5 * ((pred - onehot) ** 2).sum(1)).mean())}


def test_metagl_u_runner_uniform_mixture(tmp_path):
    cfg = _selector_cfg(tmp_path)
    world = _MatchedWorld()
    app = AppSpec("new", "graph", 5, 42)
    runner = load_runner_class("metagl_u")(cfg, app, world)
    assert isinstance(runner, MetaGLURunner)
    runner.fit()
    expected_team = MetaGLSelector(cfg, _World()).rank(app).team
    assert runner.team == expected_team and world.requested == [expected_team]
    mix = torch.stack([world.preds[e] for e in expected_team]).mean(0)
    assert torch.allclose(runner.predict_queries(), mix)
    assert runner.best_metrics == {f"test_{k}": pytest.approx(v) for k, v in world.evaluate_outputs(app, mix).items()}
    assert "test_acc" in runner.best_metrics and "test_risk" in runner.best_metrics
    assert not any("loss" in k for k in runner.best_metrics) and runner.best_epoch == runner.outcome.extras["best_epoch"]
    cfg.moe.routergfm.baselines.metagl_u.selector = "logme"
    with pytest.raises(ValueError):
        MetaGLURunner(cfg, app, world)


# --------------------------------------------------------------------------- #
# End to end on the synthetic RouterGFM environment
# --------------------------------------------------------------------------- #
TARGET = "nodea:node"


class _LabelLog(dict):
    def __init__(self, labels, key, log):
        super().__init__(labels)
        self.key, self.log = key, log

    def __getitem__(self, name):
        self.log.append((self.key, name))
        return super().__getitem__(name)

    def get(self, name, default=None):
        return self[name] if name in self else default


class _LoggingProvider:
    def __init__(self, inner):
        self.inner, self.log = inner, []

    def load(self, app):
        data = self.inner.load(app)
        data.labels = _LabelLog(data.labels, app.key, self.log)
        return data

    def base_graph(self, app):
        return self.inner.base_graph(app)  # structure only


def _read_tsv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("metagl")
    cfg = tiny_cfg(
        tmp,
        targets=(TARGET,),
        history_extra=("nodeb:node", "nodec:node", "srca:node", "grapha:graph", "linka:edge"),
        budgets=(3,),
        seeds=(42, 0),
    )
    cfg.save_results.output_dir = str(tmp / "results")
    cfg.moe.routergfm.baselines.num_runs = 2
    m = cfg.moe.routergfm.baselines.metagl
    m.epochs, m.patience, m.knn_k, m.rf_n_estimators, m.graph_sample_max = 10, 10, 3, 10, 30
    provider = SyntheticDataProvider()
    generate_history(cfg, provider, apps=enumerate_applications(cfg.moe.routergfm))
    return SimpleNamespace(cfg=cfg, provider=provider, tmp=tmp)


def test_metagl_harness_and_metagl_u_end_to_end(env, tmp_path):
    cfg = env.cfg.clone()
    cfg.moe.routergfm.baselines.output_dir = str(tmp_path / "baselines")
    cfg.save_results.output_dir = str(tmp_path / "results")
    provider = _LoggingProvider(env.provider)
    infra = RouterInfra(cfg, provider)
    for method in ("metagl", "metagl_metadata"):
        cfg.moe.routergfm.baselines.method = method
        assert selection_run.run_selection_baseline(cfg, infra=infra) == 0
    assert not {k for k, _ in provider.log if k.startswith("nodea__")}  # label-free: no target labels read

    rows = _read_tsv(tmp_path / "results" / "moe_routergfm_selection.tsv")
    assert [(r["moe.routergfm.baselines.method"], r["dataset"], r["n_apps"]) for r in rows] == [
        ("metagl", "nodea", "2"), ("metagl_metadata", "nodea", "2"),  # one dataset: no table9_all row
    ]
    assert 0.0 <= float(rows[0]["test_hit_at_2_mean"]) <= 1.0 and float(rows[0]["test_regret_at_2_mean"]) >= 0.0
    (outcome_file,) = [p for p in (tmp_path / "baselines" / "metagl").rglob("*.json") if p.name == "nodea__node__b3__s42.json"]
    outcome = json.loads(outcome_file.read_text())["outcome"]
    assert outcome["extras"]["pooled_fallback"] and outcome["extras"]["n_rows"] == 5  # 3 node rows < 10 -> pooled

    cfg.moe.routergfm.baselines.method = "metagl_u"
    provider.log.clear()
    assert matched_run.run_matched_baseline(cfg, "metagl_u", infra=infra) == 0
    (row,) = _read_tsv(tmp_path / "results" / "moe_metagl_u.tsv")
    assert json.loads(row["moe.routergfm.baselines.datasets"]) == [TARGET] and json.loads(row["seeds"]) == [42, 0]
    assert 0.0 <= float(row["test_acc_mean"]) <= 1.0 and float(row["test_risk_mean"]) >= 0.0
    assert not any("loss" in c for c in row)
    assert {split for key, split in provider.log if key.startswith("nodea__")} <= {"support", "query"}
    payload = json.loads(next((tmp_path / "baselines" / "metagl_u").rglob("nodea__node__b3__s42.json")).read_text())
    runner = MetaGLURunner(cfg, infra.application(TARGET, 3, 42), infra)
    runner.fit()
    assert runner.team == outcome["team"]  # MetaGL-U mixes exactly the MetaGL shortlist
    assert payload["metrics"]["test_acc"] == pytest.approx(runner.best_metrics["test_acc"])
