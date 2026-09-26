"""Context descriptors z_a(x) (Sec. 3.1/3.4, App. B.2) and the local archive M (Eq. 5, App. A.4, D.2/D.5)."""

import math
import time
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from src.config import cfg as base_cfg
from src.moe.routergfm import descriptors as desc_mod
from src.moe.routergfm.archive import build_archive, build_cells, kmeans, perturb_archive
from src.moe.routergfm.common import GRAPH_CLS, LINK, NODE_CLS, REGRESSION, TASK_FAMILIES, AppSpec, CompatKey, RouterPaths
from src.moe.routergfm.descriptors import (
    DEFAULT_NUM_SPECTRAL,
    DESCRIPTOR_NAMES,
    FAMILY_SLICES,
    SHORTEST_PATH_CAP,
    DescriptorStandardizer,
    batch_descriptors,
    compute_descriptors,
    descriptor_names,
    descriptors_at,
    ensure_descriptors,
    family_slices,
    instance_descriptor,
)

K = 4  # num_spectral used by most tests
NAMES = descriptor_names(K)
IDX = {name: i for i, name in enumerate(NAMES)}


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    """Tiny batched linear algebra: avoid thread oversubscription on shared nodes."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _cfg(tmp_path, num_spectral=K):
    cfg = base_cfg.clone()
    cfg.moe.routergfm.output_root = str(tmp_path / "routergfm")
    cfg.moe.routergfm.descriptors.num_spectral = num_spectral
    return cfg


def _undirected(pairs):
    ei = torch.tensor(pairs, dtype=torch.long).t().view(2, -1)
    return torch.cat([ei, ei.flip(0)], dim=1)


def _graph(n, edges, x=None, target=0, pair=(0, 1)):
    g = Data(x=x if x is not None else torch.ones(n, 2), edge_index=_undirected(edges), num_nodes=n)
    g.target_node_index = torch.tensor([target])
    g.edge_label_index = torch.tensor([[pair[0]], [pair[1]]])
    return g


def _random_graph(n, seed, p=0.2, feat_dim=6):
    gen = torch.Generator().manual_seed(seed)
    upper = (torch.rand(n, n, generator=gen) < p).triu(1)
    upper[0, 2] = upper[2, 0] = False  # candidate pair (0, 2) is unlinked
    ei = upper.nonzero().t()
    g = Data(x=torch.randn(n, feat_dim, generator=gen), edge_index=torch.cat([ei, ei.flip(0)], 1), num_nodes=n)
    g.target_node_index = torch.tensor([1])
    g.edge_label_index = torch.tensor([[0], [2]])
    return g


def _permute(g, perm):
    """Relabel nodes: new node i is old node perm[i]."""
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    out = Data(x=g.x[perm], edge_index=inv[g.edge_index], num_nodes=g.num_nodes)
    out.target_node_index = inv[g.target_node_index]
    out.edge_label_index = inv[g.edge_label_index]
    return out


LEVELS = [("node", NODE_CLS), ("edge", LINK), ("graph", GRAPH_CLS)]


# --------------------------------------------------------------------------- #
# Descriptors
# --------------------------------------------------------------------------- #
def test_layout_is_fixed_across_levels():
    assert DESCRIPTOR_NAMES == descriptor_names(DEFAULT_NUM_SPECTRAL)
    assert FAMILY_SLICES == family_slices(DEFAULT_NUM_SPECTRAL)
    assert len(set(NAMES)) == len(NAMES)
    slices = family_slices(K)
    assert list(slices) == ["structure", "feature", "role", "task"]
    assert sorted(i for s in slices.values() for i in range(len(NAMES))[s]) == list(range(len(NAMES)))
    node_role = [i for n, i in IDX.items() if n.startswith("role_node")]
    edge_role = [i for n, i in IDX.items() if n.startswith("role_edge")]

    g = _random_graph(12, seed=0)
    z = {lvl: instance_descriptor(g, lvl, fam, K) for lvl, fam in LEVELS}
    for lvl, fam in LEVELS:
        assert z[lvl].shape == (len(NAMES),) and z[lvl].dtype == torch.float32
        assert torch.isfinite(z[lvl]).all()
        expected_task = torch.zeros(len(TASK_FAMILIES))
        expected_task[TASK_FAMILIES.index(fam)] = 1
        assert torch.equal(z[lvl][slices["task"]], expected_task)
    assert z["graph"][slices["role"]].abs().sum() == 0
    assert z["node"][edge_role].abs().sum() == 0 and z["node"][node_role].abs().sum() > 0
    assert z["edge"][node_role].abs().sum() == 0 and z["edge"][edge_role].abs().sum() > 0
    # structure is level independent (the candidate pair (0, 2) is unlinked here)
    s = slices["structure"]
    assert torch.allclose(z["node"][s], z["graph"][s]) and torch.allclose(z["edge"][s], z["graph"][s])


def test_hand_computed_values():
    # triangle 0-1-2, pendant 2-3, isolated 4
    x = torch.randn(5, 3, generator=torch.Generator().manual_seed(0))
    g = _graph(5, [(0, 1), (1, 2), (0, 2), (2, 3)], x=x, target=2, pair=(0, 3))
    z = instance_descriptor(g, "node", NODE_CLS, K)
    assert z[IDX["struct_log_nodes"]] == pytest.approx(math.log(6))
    assert z[IDX["struct_log_edges"]] == pytest.approx(math.log(5))
    assert z[IDX["struct_density"]] == pytest.approx(0.4)
    assert z[IDX["struct_degree_mean"]] == pytest.approx(1.6)
    assert z[IDX["struct_degree_std"]] == pytest.approx(math.sqrt(1.04), abs=1e-6)
    assert z[IDX["struct_degree_max"]] == pytest.approx(3.0)
    assert z[IDX["struct_clustering"]] == pytest.approx(7 / 15, abs=1e-6)
    assert z[IDX["struct_log_components"]] == pytest.approx(math.log(3))
    assert z[IDX["role_node_degree"]] == 3 and z[IDX["role_node_degree_pct"]] == pytest.approx(1.0)
    assert z[IDX["role_node_clustering"]] == pytest.approx(1 / 3, abs=1e-6)
    assert z[IDX["role_node_feat_norm"]] == pytest.approx(float(x[2].norm()), abs=1e-5)
    cos = torch.nn.functional.cosine_similarity(x[2:3], x[[0, 1, 3, 4]]).mean()
    assert z[IDX["feat_marked_cosine"]] == pytest.approx(float(cos), abs=1e-5)

    e = instance_descriptor(g, "edge", LINK, K)
    assert (e[IDX["role_edge_degree_min"]], e[IDX["role_edge_degree_max"]]) == (1, 2)
    assert e[IDX["role_edge_common_neighbors"]] == 1
    assert e[IDX["role_edge_jaccard"]] == pytest.approx(0.5)
    assert e[IDX["role_edge_adamic_adar"]] == pytest.approx(1 / math.log(3), abs=1e-6)
    assert e[IDX["role_edge_shortest_path"]] == 2
    assert e[IDX["role_edge_feat_cosine"]] == pytest.approx(float(torch.nn.functional.cosine_similarity(x[0], x[3], dim=0)), abs=1e-5)
    g.edge_label_index = torch.tensor([[3], [4]])  # unreachable pair -> capped
    assert instance_descriptor(g, "edge", LINK, K)[IDX["role_edge_shortest_path"]] == SHORTEST_PATH_CAP


def test_spectra_are_leading_and_zero_padded():
    # K3: normalized Laplacian spectrum {0, 1.5, 1.5}; collinear features -> centered rank 1
    x = torch.arange(3, dtype=torch.float32)[:, None] * torch.tensor([[1.0, 2.0]])
    z = instance_descriptor(_graph(3, [(0, 1), (1, 2), (0, 2)], x=x), "graph", GRAPH_CLS, K)
    eig = z[[IDX[f"struct_lap_eig_{i}"] for i in range(K)]]
    sv = z[[IDX[f"feat_sv_{i}"] for i in range(K)]]
    assert torch.allclose(eig, torch.tensor([1.5, 1.5, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(sv, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-5)
    assert z[IDX["struct_log_components"]] == pytest.approx(math.log(2))


@pytest.mark.parametrize("level,family", LEVELS)
def test_node_permutation_invariance(level, family):
    for seed in range(4):
        g = _random_graph(15 + seed, seed=seed)
        perm = torch.randperm(g.num_nodes, generator=torch.Generator().manual_seed(100 + seed))
        z, zp = instance_descriptor(g, level, family, K), instance_descriptor(_permute(g, perm), level, family, K)
        assert torch.allclose(z, zp, atol=1e-5), (z - zp).abs().max()


def test_candidate_link_is_absent():
    g = _random_graph(14, seed=3)
    linked = g.clone()
    linked.edge_index = torch.cat([g.edge_index, torch.tensor([[0, 2], [2, 0]])], dim=1)
    assert torch.equal(instance_descriptor(g, "edge", LINK, K), instance_descriptor(linked, "edge", LINK, K))
    # without the removal the pair would be adjacent and the structure would differ
    assert not torch.equal(instance_descriptor(g, "graph", LINK, K), instance_descriptor(linked, "graph", LINK, K))


def test_batched_equals_single_and_featureless(monkeypatch):
    monkeypatch.setattr(desc_mod, "_BATCH_ELEMENTS", 3 * 20**2)  # force several size buckets
    graphs = [_random_graph(n, seed=n) for n in (5, 20, 3, 11, 20, 8, 3, 4)]
    for level, family in LEVELS:
        batched = batch_descriptors(graphs, level, family, K)
        single = torch.stack([instance_descriptor(g, level, family, K) for g in graphs])
        assert torch.allclose(batched, single, atol=1e-5)
    bare = Data(edge_index=_undirected([(0, 1), (1, 2)]), num_nodes=3)
    bare.target_node_index = torch.tensor([0])
    z = instance_descriptor(bare, "node", REGRESSION, K)
    assert torch.isfinite(z).all() and z[family_slices(K)["feature"]].abs().sum() == 0


def test_descriptors_are_fast():
    graphs = [_random_graph(60, seed=s, p=0.1, feat_dim=100) for s in range(256)]
    batch_descriptors(graphs[:8], "edge", LINK)  # warm-up
    start = time.perf_counter()
    batch_descriptors(graphs, "edge", LINK)
    per_graph = (time.perf_counter() - start) / len(graphs)
    assert per_graph < 2e-3, f"{per_graph * 1e3:.2f} ms per 60-node subgraph"


def _app_data(n=10, seed=42):
    app = AppSpec("toy", "node", 5, seed)
    return SimpleNamespace(
        app=app,
        dataset=[_random_graph(6 + i % 5, seed=i) for i in range(n)],
        level="node",
        task_family=NODE_CLS,
        support_pos=torch.tensor([7, 1]),
        diag_pos=torch.tensor([3, 5]),
        query_pos=torch.tensor([3, 5, 8, 0]),
    )


class _NoProvider:
    def load(self, app):
        raise AssertionError("cached descriptors must not reload the application")


class _Provider:
    def __init__(self, data):
        self.data, self.calls = data, 0

    def load(self, app):
        self.calls += 1
        return self.data


def test_ensure_descriptors_cache(tmp_path):
    cfg = _cfg(tmp_path)
    data = _app_data()
    cache = ensure_descriptors(cfg, data)
    assert torch.equal(cache["positions"], torch.tensor([0, 1, 3, 5, 7, 8]))
    assert cache["z"].shape == (6, len(NAMES))
    assert torch.allclose(cache["z"], compute_descriptors(data, cache["positions"], cfg, device="cpu"), atol=1e-6)
    assert RouterPaths.from_cfg(cfg).descriptor_file(data.app.data_key).exists()

    again = ensure_descriptors(cfg, data.app, _NoProvider())
    assert torch.equal(again["z"], cache["z"])
    assert torch.equal(descriptors_at(again, [8, 1]), cache["z"][[5, 1]])
    with pytest.raises(KeyError):
        descriptors_at(again, [2])

    data.query_pos = torch.tensor([3, 5, 8, 0, 2])  # new position -> cache extended, old rows kept
    extended = ensure_descriptors(cfg, data)
    assert torch.equal(extended["positions"], torch.tensor([0, 1, 2, 3, 5, 7, 8]))
    assert torch.equal(descriptors_at(extended, [0, 1, 3, 5, 7, 8]), cache["z"])

    provider = _Provider(data)
    fresh = ensure_descriptors(_cfg(tmp_path / "other"), data.app, provider)
    assert provider.calls == 1 and torch.equal(fresh["positions"], extended["positions"])
    stale = ensure_descriptors(_cfg(tmp_path, num_spectral=2), data.app, provider)  # layout changed -> rebuilt
    assert provider.calls == 2 and stale["z"].shape[1] == len(descriptor_names(2))


def test_standardizer():
    gen = torch.Generator().manual_seed(0)
    Z = torch.randn(200, 4, generator=gen) * torch.tensor([1.0, 10.0, 0.1, 1.0]) + 3
    Z[:, 3] = 7.0  # constant (e.g. unused role slot)
    Z[0, 0] = 1000.0
    st = DescriptorStandardizer(clip=5.0).fit(Z)
    out = st.transform(Z)
    assert out.abs().max() <= 5.0 and out[0, 0] == 5.0
    assert torch.all(out[:, 3] == 0)
    assert torch.allclose(out[1:, 1].mean(), torch.tensor(0.0), atol=0.2)
    restored = DescriptorStandardizer().load_state_dict(st.state_dict())
    assert torch.equal(restored.transform(Z), out) and restored.clip == 5.0


# --------------------------------------------------------------------------- #
# Cells and archive
# --------------------------------------------------------------------------- #
def _blobs(n_per, centers, seed, scale=0.05):
    gen = torch.Generator().manual_seed(seed)
    c = torch.tensor(centers, dtype=torch.float32)
    return torch.cat([ci + scale * torch.randn(n_per, c.size(1), generator=gen) for ci in c])


def test_kmeans_and_cell_count(tmp_path):
    X = _blobs(10, [[0, 0], [5, 5], [-5, 5]], seed=0, scale=1.0)
    assign, centers = kmeans(X, 3, seed=1)
    assign2, centers2 = kmeans(X, 3, seed=1)
    assert torch.equal(assign, assign2) and torch.equal(centers, centers2)
    assert centers.shape == (3, 2) and torch.equal(assign, torch.cdist(X, centers).argmin(1))
    for c in assign.unique().tolist():  # centers are member means of the returned assignment
        assert torch.allclose(centers[c], X[assign == c].mean(0), atol=1e-5)
    assert kmeans(X[:2], 5, seed=0)[1].shape == (2, 2)
    assert kmeans(X, 1, seed=0)[0].eq(0).all()

    cfg = _cfg(tmp_path)
    cfg.moe.routergfm.archive.num_cells, cfg.moe.routergfm.archive.min_cell_size = 16, 5
    assert build_cells(torch.randn(23, 3), cfg, seed=0)[1].shape[0] == 4
    assert build_cells(torch.randn(3, 3), cfg, seed=0)[1].shape[0] == 1
    assert build_cells(torch.randn(500, 3), cfg, seed=0)[1].shape[0] == 16


class _FakeStore:
    """data_key -> (expert_ids, loss[n_diag, n_exp] with NaN, diag_pos, task family)."""

    def __init__(self, tables):
        self.tables, self.calls = tables, 0

    def loss_matrix(self, data_key):
        self.calls += 1
        ids, loss, _, _ = self.tables[data_key]
        return list(ids), loss.clone()

    def load(self, data_key, expert_id):
        _, _, pos, family = self.tables[data_key]
        return {"diag_pos": pos, "family": family}


def _archive_fixture(tmp_path):
    cfg = _cfg(tmp_path)
    arc = cfg.moe.routergfm.archive
    arc.num_cells, arc.min_cell_size, arc.num_families = 4, 3, 2
    gen = torch.Generator().manual_seed(0)
    apps = [
        AppSpec("cora", "node", 5, 42),
        AppSpec("pubmed", "node", 5, 42),
        AppSpec("cora", "edge", 5, 42),
        AppSpec("cora", "edge", 100, 42),  # LP budgets share one data key
    ]
    catalog = [SimpleNamespace(expert_id=f"e{i}") for i in range(4)]
    tables, caches = {}, {}
    for a, app in enumerate(apps):
        if app.data_key in tables:
            continue
        n = 24
        pos = torch.randperm(100, generator=gen)[:n]
        z = _blobs(6, [[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]], seed=a)[torch.randperm(n, generator=gen)]
        loss = torch.rand(n, 5, generator=gen)
        loss[torch.rand(n, 5, generator=gen) < 0.2] = float("nan")
        loss[:, 1] = torch.rand(n, generator=gen)  # e1 fully observed
        if app.dataset == "pubmed":
            loss[:, 0] = float("nan")  # e2 never valid on pubmed -> no records
        family = LINK if app.task_level == "edge" else NODE_CLS
        tables[app.data_key] = (["e2", "e1", "e0", "e3", "unknown"], loss, pos, family)
        order = torch.argsort(pos)
        caches[app.data_key] = {"positions": pos[order], "z": z[order]}
    standardizer = DescriptorStandardizer(clip=5.0).fit(torch.cat([c["z"] for c in caches.values()]))
    store = _FakeStore(tables)
    archive = build_archive(apps, store, standardizer, cfg, catalog=catalog, descriptors=caches)
    return SimpleNamespace(cfg=cfg, apps=apps, tables=tables, caches=caches, store=store, archive=archive, standardizer=standardizer)


def test_build_archive_records(tmp_path):
    fx = _archive_fixture(tmp_path)
    arc = fx.archive
    assert fx.store.calls == 3  # one loss matrix per data key
    assert arc.group == ["cora", "pubmed", "cora", "cora"]
    assert arc.compat == [
        CompatKey(NODE_CLS, 5).as_tuple(), CompatKey(NODE_CLS, 5).as_tuple(),
        CompatKey(LINK, 5).as_tuple(), CompatKey(LINK, 100).as_tuple(),
    ]
    assert set(arc.expert.tolist()) <= {0, 1, 2, 3}  # 'unknown' is outside the catalog
    assert not bool(((arc.app == 1) & (arc.expert == 2)).any())
    col = {"e2": 0, "e1": 1, "e0": 2, "e3": 3}

    for a, app in enumerate(fx.apps):
        ids, loss, pos, _ = fx.tables[app.data_key]
        rec = arc.app == a
        z_std = fx.standardizer.transform(descriptors_at(fx.caches[app.data_key], pos))
        cells = arc.cell[rec].unique()
        reps = torch.stack([arc.rep[rec & (arc.cell == c)][0] for c in cells])
        member = cells[torch.cdist(z_std, reps).argmin(1)]  # reps are consistent k-means centers
        for c, rep in zip(cells.tolist(), reps):
            assert torch.allclose(rep, z_std[member == c].mean(0), atol=1e-5)
        for eid, j in col.items():
            r = rec & (arc.expert == int(eid[1:]))  # catalog index of e<i> is i
            if not bool(r.any()):
                continue
            values = loss[:, j]
            valid = torch.isfinite(values)
            mu = values[valid].mean()
            assert torch.allclose(arc.mu_app[r], mu.expand(int(r.sum())), atol=1e-6)
            assert int(arc.count[r].sum()) == int(valid.sum())
            for i in torch.nonzero(r).view(-1).tolist():
                in_cell = (member == arc.cell[i]) & valid
                assert int(arc.count[i]) == int(in_cell.sum()) > 0
                assert arc.r_local[i] == pytest.approx(float(values[in_cell].mean()), abs=1e-6)
            # App. A.4 centering identity on the same valid observations
            assert float((arc.count[r] * (arc.mu_app[r] - arc.r_local[r])).sum()) == pytest.approx(0.0, abs=1e-5)

    lp5, lp100 = arc.app == 2, arc.app == 3
    assert torch.equal(arc.r_local[lp5], arc.r_local[lp100]) and torch.equal(arc.rep[lp5], arc.rep[lp100])
    assert arc.family.min() >= 0 and arc.family.max() < 2
    assert torch.equal(arc.residual, arc.r_local - arc.mu_app)


def test_archive_masks_subset_and_device(tmp_path):
    arc = _archive_fixture(tmp_path).archive
    assert torch.equal(arc.records_of_groups({"pubmed"}), arc.app == 1)
    assert torch.equal(arc.records_with_compat(CompatKey(LINK, 100).as_tuple()), arc.app == 3)
    assert torch.equal(arc.compat_index(), torch.tensor([0, 0, 1, 2])[arc.app])
    sub = arc.subset(~arc.records_of_groups({"cora"}))
    assert len(sub) == int((arc.app == 1).sum()) and sub.apps == arc.apps and sub.group == arc.group
    assert torch.equal(sub.r_local, arc.r_local[arc.app == 1])
    moved = arc.to("cpu")
    assert moved.rep.device.type == "cpu" and len(moved) == len(arc)


def test_perturbations(tmp_path):
    arc = _archive_fixture(tmp_path).archive
    assert perturb_archive(arc, "none", 0) is arc
    with pytest.raises(ValueError):
        perturb_archive(arc, "bogus", 0)

    pairs = lambda a: set(zip(a.app.tolist(), a.cell.tolist()))  # noqa: E731
    half = perturb_archive(arc, "half_cells", 3)
    n_pairs = len(pairs(arc))
    assert len(pairs(half)) == n_pairs - n_pairs // 2 and pairs(half) <= pairs(arc)
    for a, c in pairs(half):  # kept cells keep all their experts
        assert int(((half.app == a) & (half.cell == c)).sum()) == int(((arc.app == a) & (arc.cell == c)).sum())
    assert pairs(perturb_archive(arc, "half_cells", 3)) == pairs(half)

    missing = perturb_archive(arc, "missing_family", 0)
    cid, mcid = arc.compat_index(), missing.compat_index()
    for g in cid.unique().tolist():
        fam = arc.family[cid == g]
        top = int(torch.bincount(fam).argmax())
        assert not bool((missing.family[mcid == g] == top).any())
        assert int((mcid == g).sum()) == int((fam != top).sum())

    rev = perturb_archive(arc, "reversed", 0)
    assert torch.allclose(rev.residual, -arc.residual, atol=1e-6) and torch.equal(rev.mu_app, arc.mu_app)
    assert torch.equal(rev.rep, arc.rep) and torch.equal(rev.count, arc.count)

    shuf = perturb_archive(arc, "shuffled", 5)
    assert torch.equal(shuf.mu_app, arc.mu_app) and torch.equal(shuf.rep, arc.rep)
    for g in cid.unique().tolist():
        m = cid == g
        assert torch.allclose(shuf.residual[m].sort().values, arc.residual[m].sort().values, atol=1e-6)
    assert not torch.allclose(shuf.residual, arc.residual)
    assert torch.equal(perturb_archive(arc, "shuffled", 5).r_local, shuf.r_local)
