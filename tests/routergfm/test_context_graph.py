"""Context graph H and node descriptions (Sec. 3.2-3.3, Prop. 4) on synthetic metadata."""

import json
import math

import pytest
import torch

from src.config import cfg as base_cfg
from src.moe.routergfm import context_graph as cg
from src.moe.routergfm.common import LINK, NODE_CLS, REGRESSION, AppSpec, ExpertSpec
from src.moe.routergfm.text import (
    TextEncoder,
    describe_application,
    describe_architecture,
    describe_expert,
)

HASH_DIM = 16


def _cfg(tmp_path, **graph):
    cfg = base_cfg.clone()
    cfg.moe.routergfm.output_root = str(tmp_path / "out")
    g = cfg.moe.routergfm.graph
    g.text_backend = "hash"
    g.hash_dim = HASH_DIM
    g.text_cache_dir = str(tmp_path / "text_cache")
    for key, value in graph.items():
        setattr(g, key, value)
    return cfg


def _spec(arch, obj, source, level, dims="h64_o32_l2"):
    eid = f"{obj}_{source}_task{level}_induced1_{arch}_{dims}_e100_lr0.001_bs32_seed42"
    return ExpertSpec(eid, arch, obj, obj, source, level, f"/nonexistent/{eid}.pt")


CATALOG = [
    _spec(a, o, s, lvl)
    for s, lvl in (("cora", "node"), ("bbbp", "graph"))
    for o in ("dgi", "graphcl")
    for a in ("gcn", "gin")
]
APPS = [
    AppSpec("photo", "node", 5, 42),
    AppSpec("photo", "node", 100, 42),
    AppSpec("cora", "node", 5, 42),
    AppSpec("dblp", "edge", 5, 42),
    AppSpec("qm7b", "graph", 5, 0),
]
FAMILY = dict(zip([a.key for a in APPS], [NODE_CLS, NODE_CLS, NODE_CLS, LINK, REGRESSION]))
_SIZES = {  # num_instances, num_nodes, num_edges, avg_degree, feature_dim, num_classes
    "photo": (7650, 7650, 238162, 31.1, 745, 8),
    "cora": (2708, 2708, 10556, 3.9, 1433, 7),
    "dblp": (10496, 17716, 105734, 5.9, 1639, 2),
    "qm7b": (7211, 15.4, 245.0, 15.9, 1, 14),
}


def _stats(app):
    out = {k: float(v) for k, v in zip(cg.APP_STAT_KEYS, _SIZES[app.dataset])}
    out[f"family_{FAMILY[app.key]}"] = 1.0
    return out


STATS = {app.key: _stats(app) for app in APPS}


class FakeStore:
    """Application averages (Eq. 2) keyed by (data_key, expert_id)."""

    def __init__(self, apps, catalog):
        self.records = {}
        for a, app in enumerate(apps):
            for e, spec in enumerate(catalog):
                if (a + e) % 5 == 0:
                    continue  # missing record
                count = 0 if (a, e) == (1, 1) else 10 * (a + 1) + e
                self.records[(app.data_key, spec.expert_id)] = (0.1 + 0.01 * a + 0.001 * e, count)

    def has(self, data_key, expert_id):
        return (data_key, expert_id) in self.records

    def app_average(self, app, expert_id):
        return self.records[(app.data_key, expert_id)]


def _build(cfg, catalog=CATALOG, apps=APPS, stats=STATS, **kwargs):
    store = FakeStore(APPS, CATALOG)
    return cg.build_context_graph(cfg, catalog, apps, store, TextEncoder(cfg), stats, **kwargs)


def _expected_eval_pairs(apps, catalog):
    store = FakeStore(APPS, CATALOG)
    pairs = {}
    for app in apps:
        for spec in catalog:
            if app.group == spec.source_group or not store.has(app.data_key, spec.expert_id):
                continue
            mu, count = store.app_average(app, spec.expert_id)
            if count > 0:
                pairs[(app.key, spec.expert_id)] = (mu, count)
    return pairs


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #
def test_hash_encoder_is_deterministic_and_normalized(tmp_path):
    enc = TextEncoder(_cfg(tmp_path))
    texts = ["GCN encoder on cora", "graph regression on qm7b", "GCN encoder on cora"]
    a, b = enc.encode(texts), TextEncoder(_cfg(tmp_path)).encode(texts)
    assert a.shape == (3, HASH_DIM) and a.dtype == torch.float32
    assert torch.equal(a, b) and torch.equal(a[0], a[2]) and not torch.equal(a[0], a[1])
    assert torch.allclose(a.norm(dim=1), torch.ones(3))
    assert enc.encode([]).shape == (0, HASH_DIM)


def test_bert_backend_embeds_once_and_reuses_disk_cache(tmp_path, monkeypatch):
    calls = []

    def fake_embed(self, texts):
        calls.append(list(texts))
        return torch.stack([torch.full((4,), float(len(t))) for t in texts])

    monkeypatch.setattr(TextEncoder, "_bert_embed", fake_embed)
    cfg = _cfg(tmp_path, text_backend="bert")
    first = TextEncoder(cfg).encode(["alpha", "beta", "alpha"])
    assert calls == [["alpha", "beta"]]
    second = TextEncoder(cfg).encode(["beta", "alpha"])  # new instance: served from disk
    assert calls == [["alpha", "beta"]]
    assert torch.equal(second, first[[1, 0]])
    cfg.moe.routergfm.graph.text_max_length = 64  # cache identity includes the tokenizer length
    TextEncoder(cfg).encode(["alpha"])
    assert calls[-1] == ["alpha"]


def test_descriptions_are_metadata_driven(tmp_path):
    app = APPS[0]
    text = describe_application(app, STATS[app.key], NODE_CLS)
    for token in ("photo", "social", "node classification", "8 classes", "7650 nodes", "745",
                  "5 labeled support examples per class"):
        assert token in text
    lp = describe_application(APPS[3], STATS[APPS[3].key], LINK)
    assert "link prediction" in lp and "10%-5%-10%" in lp and "budget block 5" in lp
    reg = describe_application(APPS[4], STATS[APPS[4].key], REGRESSION)
    assert "regression with 14 numerical targets" in reg and "7211 graphs" in reg and "in total" in reg
    exp = describe_expert(CATALOG[0])
    for token in ("gcn", "dgi", "cora", "citation", "task head fitted on the labeled support set"):
        assert token in exp.lower()

    desc = tmp_path / "desc"
    (desc / "architecture").mkdir(parents=True)
    (desc / "dataset").mkdir(parents=True)
    (desc / "architecture" / "mixhop.json").write_text(json.dumps({"summary": "Mixes adjacency powers."}))
    (desc / "dataset" / "photo.json").write_text(json.dumps({"description": "Amazon co-purchase graph."}))
    arch = describe_architecture("mixhop", description_dir=str(desc))
    assert "mixhop" in arch and arch.endswith("Mixes adjacency powers.")
    with_json = describe_application(app, STATS[app.key], NODE_CLS, description_dir=str(desc))
    assert with_json.endswith("Amazon co-purchase graph.")


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def test_build_graph_nodes_features_and_relations(tmp_path):
    graph = _build(_cfg(tmp_path))
    assert set(graph.edge_index) == set(cg.RELATIONS) == set(graph.edge_attr)
    assert graph.app_nodes[: len(APPS)] == APPS
    assert graph.app_nodes[len(APPS):] == ["corpus:cora", "corpus:bbbp"]
    assert graph.expert_ids == [s.expert_id for s in CATALOG]
    assert graph.arch_names == ["gcn", "gin"] and graph.objective_names == ["dgi", "graphcl"]
    assert graph.in_dims == {
        "app": HASH_DIM + len(cg.APP_NUMERIC_NAMES) + 1,
        "expert": HASH_DIM + len(cg.EXPERT_NUMERIC_NAMES) + 1,
        "arch": HASH_DIM + 1,
        "objective": HASH_DIM + 1,
    }
    assert graph.x["app"].shape[0] == len(APPS) + 2 and graph.x["expert"].shape[0] == len(CATALOG)
    for x in graph.x.values():
        assert torch.isfinite(x).all() and torch.all(x[:, -1] == 1)

    # Standardized numeric blocks: mean 0 over the nodes present at build time.
    app_num = graph.x["app"][:, HASH_DIM:-1]
    assert torch.allclose(app_num.mean(0), torch.zeros(app_num.shape[1]), atol=1e-5)
    is_corpus = graph.x["app"][:, HASH_DIM + cg.APP_NUMERIC_NAMES.index("is_corpus")]
    assert (is_corpus[len(APPS):] > is_corpus[: len(APPS)].max()).all()

    # Construction edges: one per expert and relation, targets by metadata.
    for rel, names, attr in ((cg.USES_ARCH, graph.arch_names, "architecture"),
                             (cg.USES_OBJECTIVE, graph.objective_names, "objective")):
        src, dst = graph.edge_index[rel]
        assert src.tolist() == list(range(len(CATALOG)))
        assert [names[d] for d in dst.tolist()] == [getattr(s, attr) for s in CATALOG]
    _, dst = graph.edge_index[cg.PRETRAINED_ON]
    assert [graph.app_nodes[d] for d in dst.tolist()] == [f"corpus:{s.source}" for s in CATALOG]
    # Reverses are flipped with identical attributes; construction attributes are zero.
    for rel in (cg.EVALUATES, cg.USES_ARCH, cg.USES_OBJECTIVE, cg.PRETRAINED_ON):
        rev = cg.reverse_relation(rel)
        assert graph.edge_attr[rel].shape == (graph.edge_index[rel].shape[1], cg.EDGE_DIM)
        assert torch.equal(graph.edge_index[rev], graph.edge_index[rel].flip(0))
        assert torch.equal(graph.edge_attr[rev], graph.edge_attr[rel])
        if rel != cg.EVALUATES:
            assert torch.all(graph.edge_attr[rel] == 0)


def test_evaluation_edges_carry_valid_averages_without_diagonal(tmp_path):
    graph = _build(_cfg(tmp_path))
    expected = _expected_eval_pairs(APPS, CATALOG)
    got = {}
    for i, (a, e) in enumerate(graph.edge_index[cg.EVALUATES].t().tolist()):
        assert (a, e) == (graph.eval_app[i].item(), graph.eval_expert[i].item())
        mu, log_count = graph.edge_attr[cg.EVALUATES][i].tolist()
        got[(graph.app_nodes[a].key, graph.expert_ids[e])] = (mu, log_count)
        assert math.isclose(graph.eval_mu[i].item(), mu, rel_tol=1e-6)
        assert math.isclose(math.log1p(graph.eval_count[i].item()), log_count, rel_tol=1e-6)
    assert set(got) == set(expected) and len(got) == graph.eval_app.numel()
    for key, (mu, count) in expected.items():
        assert math.isclose(got[key][0], mu, rel_tol=1e-6)
        assert math.isclose(got[key][1], math.log1p(count), rel_tol=1e-6)
    # Empty diagonal: the cora application never evaluates cora-pretrained experts.
    assert not any(app == APPS[2].key and "_cora_" in eid for app, eid in got)


def test_masked_edges_hide_every_application_of_a_group(tmp_path):
    graph = _build(_cfg(tmp_path))
    edge_index, edge_attr = cg.masked_edges(graph, {"photo"})
    photo_ids = {graph.app_index[APPS[0].key], graph.app_index[APPS[1].key]}
    kept = edge_index[cg.EVALUATES]
    assert not (set(kept[0].tolist()) & photo_ids)
    n_photo = sum(int(a in photo_ids) for a in graph.eval_app.tolist())
    assert n_photo > 0 and kept.shape[1] == graph.eval_app.numel() - n_photo
    rev = cg.reverse_relation(cg.EVALUATES)
    assert torch.equal(edge_index[rev], kept.flip(0))
    assert torch.equal(edge_attr[rev], edge_attr[cg.EVALUATES])
    for rel in cg.RELATIONS:
        if rel not in cg.EVAL_RELATIONS:
            assert edge_index[rel] is graph.edge_index[rel]
    assert graph.edge_index[cg.EVALUATES].shape[1] == graph.eval_app.numel()  # graph itself untouched
    same, _ = cg.masked_edges(graph, set())
    assert torch.equal(same[cg.EVALUATES], graph.edge_index[cg.EVALUATES])


def test_type_shared_features_are_permutation_equivariant(tmp_path):
    """Prop. 4: features depend on metadata, not node ids."""
    cfg = _cfg(tmp_path)
    graph = _build(cfg)
    other = _build(cfg, catalog=CATALOG[::-1], apps=[APPS[i] for i in (3, 0, 4, 2, 1)])
    for eid, idx in graph.expert_index.items():
        assert torch.allclose(graph.x["expert"][idx], other.x["expert"][other.expert_index[eid]], atol=1e-6)
    for key, idx in graph.app_index.items():
        assert torch.allclose(graph.x["app"][idx], other.x["app"][other.app_index[key]], atol=1e-6)
    for i, name in enumerate(graph.arch_names):
        assert torch.equal(graph.x["arch"][i], other.x["arch"][other.arch_names.index(name)])

    def triples(g):
        return {
            (g.app_nodes[a].key, g.expert_ids[e], round(mu, 6))
            for a, e, mu in zip(g.eval_app.tolist(), g.eval_expert.tolist(), g.eval_mu.tolist())
        }

    assert triples(graph) == triples(other)


# --------------------------------------------------------------------------- #
# Insertion
# --------------------------------------------------------------------------- #
def test_insert_application_matches_build_features(tmp_path):
    cfg = _cfg(tmp_path)
    graph = _build(cfg, apps=APPS[:-1])
    n_eval = graph.eval_app.numel()
    target = APPS[-1]
    stats = {k: v for k, v in STATS[target.key].items() if not k.startswith("family_")}
    node = cg.insert_application(graph, target, stats, TextEncoder(cfg), cfg, family=REGRESSION)
    assert node == graph.x["app"].shape[0] - 1 == graph.app_index[target.key]
    assert graph.eval_app.numel() == n_eval == graph.edge_index[cg.EVALUATES].shape[1]  # metadata only
    assert cg.insert_application(graph, target, stats, TextEncoder(cfg), cfg, family=REGRESSION) == node

    full = _build(cfg, numeric_stats=graph.numeric_stats)
    assert torch.allclose(graph.x["app"][node], full.x["app"][full.app_index[target.key]], atol=1e-6)
    with pytest.raises(ValueError):
        cg.insert_application(graph, AppSpec("zinc", "graph", 5, 0), stats, TextEncoder(cfg), cfg)


def test_insert_expert_creates_unseen_factor_nodes(tmp_path):
    cfg = _cfg(tmp_path)
    graph = _build(cfg)
    counts = {rel: graph.edge_index[rel].shape[1] for rel in cg.RELATIONS}
    new = _spec("transformer", "supervised", "qm9", "graph")
    corpus_stats = dict(zip(cg.APP_STAT_KEYS, (130831, 18.0, 37.3, 2.1, 11, 19)))
    node = cg.insert_expert(graph, new, TextEncoder(cfg), cfg, catalog_index=len(CATALOG), corpus_stats=corpus_stats)
    assert node == len(CATALOG) and graph.expert_index[new.expert_id] == node
    assert graph.expert_catalog_index[-1] == len(CATALOG)
    assert graph.arch_names[-1] == "transformer" and graph.objective_names[-1] == "supervised"
    assert graph.app_nodes[-1] == "corpus:qm9" and graph.x["app"].shape[0] == len(APPS) + 3
    for rel, dst in (
        (cg.USES_ARCH, graph.arch_names.index("transformer")),
        (cg.USES_OBJECTIVE, graph.objective_names.index("supervised")),
        (cg.PRETRAINED_ON, graph.app_index["corpus:qm9"]),
    ):
        rev = cg.reverse_relation(rel)
        assert graph.edge_index[rel].shape[1] == counts[rel] + 1
        assert graph.edge_index[rel][:, -1].tolist() == [node, dst]
        assert graph.edge_index[rev][:, -1].tolist() == [dst, node]
        assert torch.all(graph.edge_attr[rel][-1] == 0) and torch.all(graph.edge_attr[rev][-1] == 0)
    assert graph.edge_index[cg.EVALUATES].shape[1] == counts[cg.EVALUATES]
    assert cg.insert_expert(graph, new, TextEncoder(cfg), cfg, catalog_index=99) == node

    # Same features as building with the expert present and the stored standardization.
    stats = dict(STATS, **{"corpus:qm9": corpus_stats})
    full = _build(cfg, catalog=CATALOG + [new], stats=stats, numeric_stats=graph.numeric_stats)
    assert torch.allclose(graph.x["expert"][node], full.x["expert"][full.expert_index[new.expert_id]], atol=1e-6)
    assert torch.allclose(graph.x["app"][-1], full.x["app"][full.app_index["corpus:qm9"]], atol=1e-6)
    assert torch.equal(graph.x["arch"][-1], full.x["arch"][full.arch_names.index("transformer")])

    # A new expert from seen factors reuses the existing arch/objective/corpus nodes.
    sizes = {t: v.shape[0] for t, v in graph.x.items()}
    cg.insert_expert(graph, _spec("gin", "dgi", "cora", "node", dims="h128_o64_l3"), TextEncoder(cfg), cfg, 100)
    assert {t: v.shape[0] for t, v in graph.x.items()} == dict(sizes, expert=sizes["expert"] + 1)


def test_calibration_edges_are_evaluation_edges(tmp_path):
    cfg = _cfg(tmp_path)
    graph = _build(cfg)
    node = cg.insert_expert(graph, _spec("transformer", "dgi", "cora", "node"), TextEncoder(cfg), cfg, 8)
    n_eval = graph.eval_app.numel()
    apps = [graph.app_index[APPS[0].key], graph.app_index[APPS[4].key], graph.app_index[APPS[3].key]]
    cg.add_calibration_edges(graph, node, apps, [0.2, 0.3, float("nan")], [12, 7, 5])
    assert graph.eval_app.numel() == n_eval + 2 == graph.edge_index[cg.EVALUATES].shape[1]
    assert graph.eval_app[-2:].tolist() == apps[:2] and graph.eval_expert[-2:].tolist() == [node, node]
    assert torch.allclose(graph.edge_attr[cg.EVALUATES][-1], torch.tensor([0.3, math.log1p(7)]))
    rev = cg.reverse_relation(cg.EVALUATES)
    assert torch.equal(graph.edge_index[rev], graph.edge_index[cg.EVALUATES].flip(0))
    edge_index, _ = cg.masked_edges(graph, {"qm7b"})
    assert edge_index[cg.EVALUATES].shape[1] == n_eval + 2 - int((graph.eval_app == apps[1]).sum())


# --------------------------------------------------------------------------- #
# Feature builders, ablations, validation
# --------------------------------------------------------------------------- #
def test_raw_feature_vectors_match_graph_blocks(tmp_path):
    cfg = _cfg(tmp_path)
    graph = _build(cfg)
    enc = TextEncoder(cfg)
    app = APPS[0]
    raw = cg.app_feature_vector(app, STATS[app.key], enc, cfg)
    assert raw.shape == (HASH_DIM + len(cg.APP_NUMERIC_NAMES),)
    row = graph.x["app"][graph.app_index[app.key]]
    assert torch.equal(raw[:HASH_DIM], row[:HASH_DIM])
    stats = graph.numeric_stats["app"]
    assert torch.allclose((raw[HASH_DIM:] - stats["mean"]) / stats["std"], row[HASH_DIM:-1], atol=1e-6)
    assert raw[HASH_DIM + cg.APP_NUMERIC_NAMES.index("log_budget")] == pytest.approx(math.log1p(5))
    assert raw[HASH_DIM + cg.APP_NUMERIC_NAMES.index(f"family_{NODE_CLS}")] == 1

    vec = cg.expert_feature_vector(CATALOG[0], enc, cfg, corpus_stats=graph.corpus_stats["cora"])
    num, names = vec[HASH_DIM:], cg.EXPERT_NUMERIC_NAMES
    assert num[names.index("log_hidden_dim")] == pytest.approx(math.log1p(64))
    assert num[names.index("num_layers")] == 2 and num[names.index("source_node")] == 1
    assert num[names.index("log_corpus_size")] == pytest.approx(math.log1p(2708))
    assert math.isnan(num[names.index("log_num_params")])  # checkpoint absent in tests


def test_num_params_read_from_checkpoint(tmp_path):
    path = tmp_path / "ckpt.pt"
    state = {"w": torch.zeros(3, 4), "b": torch.zeros(4), "n": torch.tensor(1)}
    torch.save({"cfg": {"model": {}}, "model_state": state}, path)
    eid = "dgi_cora_tasknode_induced1_gcn_h8_o4_l1_e1_lr0.1_bs2_seed42"
    num = cg.expert_numeric(ExpertSpec(eid, "gcn", "dgi", "dgi", "cora", "node", str(path)))
    assert num[cg.EXPERT_NUMERIC_NAMES.index("log_num_params")] == pytest.approx(math.log1p(16))


def test_ablation_flags_and_explicit_families(tmp_path):
    graph = _build(_cfg(tmp_path, use_text=False))
    assert graph.in_dims == {
        "app": len(cg.APP_NUMERIC_NAMES) + 1, "expert": len(cg.EXPERT_NUMERIC_NAMES) + 1, "arch": 1, "objective": 1,
    }
    graph = _build(_cfg(tmp_path, use_numeric=False), families=FAMILY)
    assert graph.in_dims["app"] == graph.in_dims["expert"] == HASH_DIM + 1


def test_contract_violations_fail_loudly(tmp_path):
    cfg = _cfg(tmp_path)
    no_degree = {k: {s: v for s, v in st.items() if s != "avg_degree"} for k, st in STATS.items()}
    with pytest.raises(ValueError, match="avg_degree"):
        _build(cfg, stats=no_degree)
    no_family = {k: {s: v for s, v in st.items() if not s.startswith("family_")} for k, st in STATS.items()}
    with pytest.raises(ValueError, match="family"):
        _build(cfg, stats=no_family)
    with pytest.raises(KeyError):
        _build(cfg, stats={k: v for k, v in STATS.items() if k != APPS[0].key})
    with pytest.raises(ValueError):
        TextEncoder(_cfg(tmp_path, text_backend="word2vec"))


def test_to_device_copies_mutable_containers(tmp_path):
    cfg = _cfg(tmp_path)
    graph = _build(cfg)
    moved = graph.to("cpu")
    assert all(v.device.type == "cpu" for v in moved.x.values())
    cg.insert_application(moved, AppSpec("zinc", "graph", 5, 0), STATS[APPS[4].key], TextEncoder(cfg), cfg)
    assert moved.x["app"].shape[0] == graph.x["app"].shape[0] + 1
    assert len(moved.app_nodes) == len(graph.app_nodes) + 1 and "zinc__graph__b5__s0" not in graph.app_index
