"""RouterGFM foundations: catalog parsing, readouts, losses, heads, applications, fixtures."""

from __future__ import annotations

import json

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.moe.routergfm import applications as apps_mod
from src.moe.routergfm import heads as heads_mod
from src.moe.routergfm.applications import (
    RealDataProvider,
    build_stats,
    convert_labels,
    diagnostic_subsample,
    stat_feature_names,
    stats_feature_vector,
    valid_label_mask,
)
from src.moe.routergfm.common import (
    GRAPH_CLS,
    LINK,
    MULTILABEL,
    NODE_CLS,
    REGRESSION,
    AppSpec,
    ExpertSpec,
    RouterPaths,
    enumerate_applications,
)
from src.moe.routergfm.embeddings import embed_instances
from src.moe.routergfm.experts import (
    build_expert_catalog,
    compatible_experts,
    load_frozen_encoder,
    parse_checkpoint_stem,
)
from src.moe.routergfm.heads import fit_head, fit_predict_oof, oof_fold_ids, predict_head
from src.moe.routergfm.losses import RegressionNormalizer, mixture_loss, routing_loss, to_metric_inputs
from src.moe.routergfm.readout import graph_query_representation, readout_dim
from src.utils.metrics import compute_supervised_metrics
from tests.routergfm.fixtures import (
    FEATURE_DIM,
    TINY_MAX_DIAG,
    SyntheticDataProvider,
    make_tiny_checkpoints,
    tiny_cfg,
)

ALL_OBJECTIVES = ["attr_masking", "context_pred", "dgi", "edge_pred", "graphcl", "infograph", "supervised"]
ALL_ARCHS = ["gcn", "gat", "gin", "h2gcn", "fagcn", "nodeformer", "transformer"]
_TAIL = "h128_o128_l2_e500_lr0.001_bs128_seed42"
REAL_STEMS = [
    f"attr_masking_cora_tasknode_induced1_gcn_{_TAIL}",
    f"context_pred_cora_tasknode_induced1_gat_{_TAIL}",
    f"dgi_cora_tasknode_induced1_h2gcn_{_TAIL}",
    f"edge_pred_cora_tasknode_induced1_nodeformer_{_TAIL}",
    f"graphcl_cora_tasknode_induced1_transformer_{_TAIL}",
    f"infograph-nolw_cora_tasknode_induced1_fagcn_{_TAIL}",
    f"infograph_cora_tasknode_induced1_gin_{_TAIL}",
    f"supervised_cora_tasknode_induced1_split80-10-10_gcn_{_TAIL}",
    f"attr_masking_qm9_taskgraph_induced0_gin_{_TAIL}",
    f"supervised_qm9_taskgraph_induced0_split80-10-10_fagcn_{_TAIL}",
    f"infograph-nolw_qm9_taskgraph_induced0_transformer_{_TAIL}",
]


def _base_cfg(tmp_path, **experts):
    from src.config import cfg as base_cfg

    cfg = base_cfg.clone()
    cfg.moe.routergfm.output_root = str(tmp_path / "out")
    for key, value in experts.items():
        setattr(cfg.moe.routergfm.experts, key, value)
    return cfg


# --------------------------------------------------------------------------- #
# Expert catalog
# --------------------------------------------------------------------------- #
def test_parse_real_checkpoint_stems():
    parsed = [parse_checkpoint_stem(s, ALL_OBJECTIVES, ALL_ARCHS) for s in REAL_STEMS]
    assert all(p is not None for p in parsed)
    assert [p["objective"] for p in parsed] == [
        "attr_masking", "context_pred", "dgi", "edge_pred", "graphcl", "infograph", "infograph",
        "supervised", "attr_masking", "supervised", "infograph",
    ]
    assert parsed[5]["method"] == "infograph-nolw" and parsed[6]["method"] == "infograph"
    assert parsed[7]["split"] == "split80-10-10" and parsed[7]["architecture"] == "gcn"
    assert parsed[0]["source"] == "cora" and parsed[0]["task_level"] == "node" and parsed[0]["induced"]
    assert parsed[8]["source"] == "qm9" and parsed[8]["task_level"] == "graph" and not parsed[8]["induced"]
    assert parsed[9]["architecture"] == "fagcn" and parsed[9]["seed"] == 42
    # Architectures outside the grid (mlp) and foreign names are rejected.
    assert parse_checkpoint_stem(f"dgi_cora_tasknode_induced1_mlp_{_TAIL}", ALL_OBJECTIVES, ALL_ARCHS) is None
    assert parse_checkpoint_stem("random_file", ALL_OBJECTIVES, ALL_ARCHS) is None
    hyphen = parse_checkpoint_stem(f"dgi_ogbn-arxiv_tasknode_induced1_gcn-bn_{_TAIL}", ALL_OBJECTIVES, ALL_ARCHS)
    assert hyphen["source"] == "ogbn-arxiv" and hyphen["arch_variant"] == "bn"


def _touch(root, source, stem):
    path = root / source / f"{stem}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_catalog_selection_order_and_persistence(tmp_path):
    root = tmp_path / "ckpt"
    for source, level in (("cora", "node"), ("qm9", "graph")):
        induced = int(level == "node")
        for arch in ("gcn", "fagcn"):
            _touch(root, source, f"dgi_{source}_task{level}_induced{induced}_{arch}_{_TAIL}")
            _touch(root, source, f"infograph_{source}_task{level}_induced{induced}_{arch}_{_TAIL}")
        _touch(root, source, f"infograph-nolw_{source}_task{level}_induced{induced}_fagcn_{_TAIL}")
        _touch(root, source, f"infograph-nolw_{source}_task{level}_induced{induced}_gcn_{_TAIL}")
        _touch(root, source, f"dgi_{source}_task{level}_induced{induced}_gcn_h128_o128_l2_e500_lr0.001_bs128_seed0")
        _touch(root, source, f"dgi_{source}_task{level}_induced{induced}_gcn-bn_{_TAIL}")
    _touch(root, "cora", f"dgi_cora_tasknode_induced1_mlp_{_TAIL}")
    cfg = _base_cfg(
        tmp_path, checkpoint_root=str(root), sources=["qm9", "cora"],
        objectives=["infograph", "dgi"], architectures=["fagcn", "gcn"],
    )
    catalog = build_expert_catalog(cfg)
    cells = [(e.source, e.objective, e.architecture) for e in catalog]
    assert cells == [
        (s, o, a) for s in ("qm9", "cora") for o in ("infograph", "dgi") for a in ("fagcn", "gcn")
    ]
    by_cell = {(e.source, e.objective, e.architecture): e for e in catalog}
    assert by_cell[("cora", "infograph", "fagcn")].objective_variant == "infograph-nolw"
    assert by_cell[("cora", "infograph", "gcn")].objective_variant == "infograph"
    assert by_cell[("cora", "dgi", "gcn")].expert_id == f"dgi_cora_tasknode_induced1_gcn_{_TAIL}"
    assert by_cell[("qm9", "dgi", "gcn")].source_task_level == "graph"
    assert len({e.expert_id for e in catalog}) == len(catalog)

    saved = json.loads(RouterPaths.from_cfg(cfg).catalog_file.read_text())
    assert [ExpertSpec.from_dict(d) for d in saved] == catalog
    # Reused unless refreshed.
    (root / "cora" / f"dgi_cora_tasknode_induced1_gcn_{_TAIL}.pt").unlink()
    assert build_expert_catalog(cfg) == catalog
    refreshed = build_expert_catalog(cfg, refresh=True)
    assert {e.source: e for e in refreshed if e.objective == "dgi" and e.architecture == "gcn"}["cora"].expert_id.endswith("seed0")


def test_catalog_strict_missing_cells(tmp_path):
    root = tmp_path / "ckpt"
    _touch(root, "cora", f"dgi_cora_tasknode_induced1_gcn_{_TAIL}")
    cfg = _base_cfg(tmp_path, checkpoint_root=str(root), sources=["cora"], objectives=["dgi", "graphcl"], architectures=["gcn", "gin"])
    with pytest.raises(FileNotFoundError) as err:
        build_expert_catalog(cfg)
    assert "gin/dgi/cora" in str(err.value) and "gcn/graphcl/cora" in str(err.value)
    cfg.moe.routergfm.experts.strict = False
    assert [e.architecture for e in build_expert_catalog(cfg)] == ["gcn"]


def test_compatible_experts_exclude_same_source():
    catalog = [
        ExpertSpec(f"e{i}", "gcn", "dgi", "dgi", src, "node", "") for i, src in enumerate(["cora", "Cora", "qm9"])
    ]
    from src.config import cfg as base_cfg

    cfg = base_cfg.clone()
    app = AppSpec("cora", "edge", 5, 42)
    assert compatible_experts(app, catalog, cfg) == [2]
    cfg.moe.routergfm.experts.exclude_same_source = False
    assert compatible_experts(app, catalog, cfg) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Readout
# --------------------------------------------------------------------------- #
def test_readout_node_and_edge():
    g1 = Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1], [1, 0]]), target_node_index=torch.tensor([2]),
              edge_label_index=torch.tensor([[0], [2]]))
    g2 = Data(x=torch.randn(2, 2), edge_index=torch.tensor([[0, 1], [1, 0]]), target_node_index=torch.tensor([0]),
              edge_label_index=torch.tensor([[1], [0]]))
    batch = Batch.from_data_list([g1, g2])
    h = torch.arange(10, dtype=torch.float32).view(5, 2)
    node = graph_query_representation(h, None, batch, task_level="node", pool_mode="mean")
    assert torch.equal(node, h[[2, 3]])
    edge = graph_query_representation(h, None, batch, task_level="edge", pool_mode="mean")
    s, d = h[[0, 4]], h[[2, 3]]
    pooled = torch.stack([h[:3].mean(0), h[3:].mean(0)])
    assert torch.allclose(edge, torch.cat([s + d, (s - d).abs(), s * d, pooled], dim=-1))
    assert edge.size(1) == readout_dim(2, "edge") and readout_dim(2, "graph") == 2


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def test_routing_loss_families():
    p = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    loss = routing_loss(p, torch.tensor([0, 1]), NODE_CLS)
    assert torch.allclose(loss, torch.tensor([0.5 * (0.09 + 0.04 + 0.01), 0.5 * (0.01 + 0.81 + 0.64)]))
    assert torch.isnan(routing_loss(p, torch.tensor([0, -1]), GRAPH_CLS)[1])

    q = torch.tensor([0.9, 0.3, 0.6])
    link = routing_loss(torch.stack([1 - q, q], 1), torch.tensor([1, 0, 0]), LINK)
    assert torch.allclose(link, (q - torch.tensor([1.0, 0.0, 0.0])) ** 2)

    ml = routing_loss(torch.tensor([[0.8, 0.4], [0.5, 0.5]]), torch.tensor([[1.0, float("nan")], [float("nan")] * 2]), MULTILABEL)
    assert torch.allclose(ml[0], torch.tensor(0.04)) and torch.isnan(ml[1])

    pred, target = torch.tensor([[1.0, 2.0]]), torch.tensor([[0.0, 4.0]])
    assert torch.allclose(routing_loss(pred, target, REGRESSION), torch.tensor([1.5]))
    assert torch.allclose(routing_loss(pred, target, REGRESSION, reg_kind="sq"), torch.tensor([1.25]))
    assert torch.equal(mixture_loss(p, torch.tensor([0, 1]), NODE_CLS), loss)
    with pytest.raises(ValueError):
        routing_loss(pred, target, REGRESSION, reg_kind="huber")


def test_regression_normalizer():
    y = torch.tensor([[1.0, 10.0], [2.0, 10.0], [4.0, 10.0], [8.0, 10.0]])
    norm = RegressionNormalizer(scale_floor=1e-6).fit(y)
    assert torch.allclose(norm.median, torch.tensor([3.0, 10.0]))  # even count: mean of middle values
    assert torch.allclose(norm.scale, torch.tensor([1.5, 1e-6]))  # MAD with floor
    assert torch.allclose(norm.inverse(norm.transform(y)), y)
    clone = RegressionNormalizer.from_state_dict(norm.state_dict())
    assert torch.equal(clone.transform(y), norm.transform(y))
    assert RegressionNormalizer.from_state_dict(None) is None


def test_to_metric_inputs_roundtrip():
    y = torch.tensor([0, 1, 2, 1])
    probs = torch.nn.functional.one_hot(y, 3).float() * 0.9 + 0.1 / 3
    logits, labels, kind = to_metric_inputs(probs, NODE_CLS, target=y)
    assert kind == "classification" and compute_supervised_metrics(logits, labels, kind)["acc"] == 1.0

    q = torch.tensor([0.9, 0.2, 0.7, 0.1])
    logits, labels, kind = to_metric_inputs(torch.stack([1 - q, q], 1), LINK, target=torch.tensor([1, 0, 1, 0]))
    assert compute_supervised_metrics(logits, labels, kind)["auc"] == 1.0

    ml_target = torch.tensor([[1.0, 0.0], [0.0, float("nan")], [1.0, 1.0], [0.0, 0.0]])
    ml_pred = torch.tensor([[0.9, 0.2], [0.1, 0.5], [0.8, 0.7], [0.3, 0.4]])
    logits, labels, kind = to_metric_inputs(ml_pred, MULTILABEL, target=ml_target)
    assert torch.allclose(torch.sigmoid(logits), ml_pred, atol=1e-5)
    assert compute_supervised_metrics(logits, labels, kind)["auc"] == 1.0

    raw = torch.tensor([[1.0], [3.0], [5.0]])
    norm = RegressionNormalizer().fit(raw)
    logits, labels, kind = to_metric_inputs(norm.transform(raw) + 1.0, REGRESSION, norm, target=raw)
    assert kind == "regression"
    assert compute_supervised_metrics(logits, labels, kind)["mae"] == pytest.approx(float(norm.scale))
    with pytest.raises(ValueError):
        to_metric_inputs(raw, REGRESSION)


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #
def _head_cfg(**kw):
    from src.config import cfg as base_cfg

    cfg = base_cfg.clone()
    cfg.moe.routergfm.heads.epochs = 100
    for key, value in kw.items():
        setattr(cfg.moe.routergfm.heads, key, value)
    return cfg


def _blobs(n_per=10, classes=3, dim=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = 3 * torch.randn(classes, dim, generator=g)
    y = torch.arange(classes).repeat_interleave(n_per)
    return centers[y] + 0.5 * torch.randn(len(y), dim, generator=g), y


@pytest.mark.parametrize("head_type", ["linear", "mlp"])
def test_fit_head_classification_learns_and_is_deterministic(head_type):
    cfg = _head_cfg(type=head_type, hidden_dim=8)
    x, y = _blobs()
    head = fit_head(x, y, NODE_CLS, 3, cfg, seed=3)
    pred = predict_head(head, x)
    assert pred.shape == (30, 3) and torch.allclose(pred.sum(1), torch.ones(30))
    assert (pred.argmax(1) == y).float().mean() > 0.95
    again = predict_head(fit_head(x, y, NODE_CLS, 3, cfg, seed=3), x)
    assert torch.equal(pred, again)
    assert not any(p.requires_grad for p in head.module.parameters())


def test_fit_head_link_multilabel_regression():
    cfg = _head_cfg()
    x, y = _blobs(classes=2)
    link = predict_head(fit_head(x, y, LINK, 2, cfg, seed=0), x)
    assert link.shape == (20, 2) and routing_loss(link, y, LINK).mean() < 0.05

    ml_y = torch.stack([(y == 1).float(), (y == 0).float()], 1)
    ml_y[0, 1] = float("nan")
    ml = predict_head(fit_head(x, ml_y, MULTILABEL, 2, cfg, seed=0), x)
    assert ((ml >= 0) & (ml <= 1)).all() and routing_loss(ml, ml_y, MULTILABEL).mean() < 0.05

    reg_y = 100.0 * x[:, :2] + 7.0
    norm = RegressionNormalizer().fit(reg_y)
    with pytest.raises(ValueError):
        fit_head(x, reg_y, REGRESSION, 2, cfg, seed=0)
    reg_cfg = _head_cfg(epochs=400, weight_decay=0.0, lr=0.05)
    reg = predict_head(fit_head(x, reg_y, REGRESSION, 2, reg_cfg, seed=0, normalizer=norm), x)
    assert routing_loss(reg, norm.transform(reg_y), REGRESSION).mean() < 0.1


def test_oof_folds_are_stratified_and_held_out(monkeypatch):
    y = torch.tensor([0] * 7 + [1] * 5 + [2] * 2)
    fold_id = oof_fold_ids(y, NODE_CLS, 3, seed=1)
    assert torch.equal(fold_id, oof_fold_ids(y, NODE_CLS, 3, seed=1))
    for cls in range(3):
        counts = torch.bincount(fold_id[y == cls], minlength=3)
        assert counts.max() - counts.min() <= 1
    assert len(set(fold_id[y == 2].tolist())) == 2  # small class spread over distinct folds

    x = torch.cat([torch.arange(14, dtype=torch.float32).view(-1, 1), torch.randn(14, 3)], 1)
    trained_on = []
    real_fit = heads_mod.fit_head

    def recording_fit(emb, labels, *args, **kwargs):
        trained_on.append(set(emb[:, 0].long().tolist()))
        return real_fit(emb, labels, *args, **kwargs)

    monkeypatch.setattr(heads_mod, "fit_head", recording_fit)
    real_predict = heads_mod.predict_head
    predicted_by = {}

    def recording_predict(head, emb):
        for item in emb[:, 0].long().tolist():
            predicted_by[item] = len(trained_on) - 1
        return real_predict(head, emb)

    monkeypatch.setattr(heads_mod, "predict_head", recording_predict)
    oof = fit_predict_oof(x, y, NODE_CLS, 3, _head_cfg(epochs=5), seed=0, folds=3)
    assert oof.shape == (14, 3) and torch.isfinite(oof).all()
    assert set(predicted_by) == set(range(14))
    assert all(item not in trained_on[head] for item, head in predicted_by.items())
    with pytest.raises(ValueError):
        fit_predict_oof(x[:1], y[:1], NODE_CLS, 3, _head_cfg(epochs=5), seed=0)


# --------------------------------------------------------------------------- #
# Applications: pure helpers
# --------------------------------------------------------------------------- #
def test_diagnostic_subsample_stratified():
    labels = torch.tensor([0] * 60 + [1] * 30 + [2] * 9 + [3] * 1)
    idx = diagnostic_subsample(labels, NODE_CLS, 20, seed=7)
    assert idx.numel() == 20 and torch.equal(idx, torch.sort(idx).values) and idx.unique().numel() == 20
    counts = torch.bincount(labels[idx], minlength=4).tolist()
    assert counts[3] == 1 and counts[0] >= counts[1] >= counts[2] >= 1 and abs(counts[0] - 12) <= 1
    assert torch.equal(idx, diagnostic_subsample(labels, NODE_CLS, 20, seed=7))
    assert not torch.equal(idx, diagnostic_subsample(labels, NODE_CLS, 20, seed=8))
    assert torch.equal(diagnostic_subsample(labels, NODE_CLS, 500, seed=7), torch.arange(100))
    reg = diagnostic_subsample(torch.randn(50, 2), REGRESSION, 10, seed=0)
    assert reg.numel() == 10 and reg.unique().numel() == 10
    tiny = diagnostic_subsample(torch.tensor([0] * 5 + [1] * 5 + [2] * 5), NODE_CLS, 2, seed=0)
    assert tiny.numel() == 2


def test_label_conversion_and_validity():
    signed = torch.tensor([[1.0, -1.0, 0.0], [-1.0, 1.0, 1.0]])
    ml = convert_labels(signed, MULTILABEL)
    assert torch.equal(torch.isnan(ml), torch.tensor([[False, False, True], [False, False, False]]))
    assert torch.equal(torch.nan_to_num(ml, nan=-9), torch.tensor([[1.0, 0.0, -9.0], [0.0, 1.0, 1.0]]))
    assert convert_labels(torch.tensor([[2], [0]]), GRAPH_CLS).tolist() == [2, 0]
    assert valid_label_mask(torch.tensor([1, -1]), NODE_CLS).tolist() == [True, False]
    assert valid_label_mask(torch.tensor([[float("nan"), 1.0], [float("nan")] * 2]), MULTILABEL).tolist() == [True, False]
    assert valid_label_mask(torch.tensor([[float("nan"), 1.0], [1.0, 2.0]]), REGRESSION).tolist() == [False, True]


def test_stats_features():
    stats = build_stats(level="edge", family=LINK, num_instances=10, num_nodes=100, num_edges=400,
                        feature_dim=16, num_classes=2, domain="citation", budget=5)
    assert stats["avg_degree"] == 4.0 and stats["family_link"] == 1.0 and stats["domain_citation"] == 1.0
    vec = stats_feature_vector(stats)
    assert vec.numel() == len(stat_feature_names()) and torch.isfinite(vec).all()
    assert build_stats(level="graph", family=GRAPH_CLS, num_instances=1, num_nodes=1, num_edges=0,
                       feature_dim=1, num_classes=2, domain="nope", budget=1)["domain_unknown"] == 1.0


# --------------------------------------------------------------------------- #
# RealDataProvider wiring (fake create_dataset, real loaders/splits)
# --------------------------------------------------------------------------- #
def _fake_induced(level: str):
    from src.data_loader.induced_graphs import InducedGraphDataset

    graphs, tags = [], []
    for i in range(24):
        g = Data(x=torch.randn(4, FEATURE_DIM), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
        if level == "node":
            g.y, g.base_node_id, g.target_node_index = torch.tensor(i % 3), i, torch.tensor([1])
            tags.append("train" if i < 6 else "test")
        else:
            g.y, g.edge_label_index = torch.tensor(i % 2, dtype=torch.long), torch.tensor([[0], [3]])
            tags.append("train" if i < 8 else ("val" if i < 12 else "test"))
        graphs.append(g)
    return InducedGraphDataset(graphs, base_num_nodes=50, base_num_edges=200,
                               base_info={"name": "cora", "domain": "citation"}, split_tags=tags)


@pytest.mark.parametrize("level", ["node", "edge"])
def test_real_provider_meta_cache_and_lazy_dataset(tmp_path, monkeypatch, level):
    import src.data_loader as data_loader

    calls = []

    def fake_create_dataset(**kwargs):
        calls.append(kwargs)
        return _fake_induced(level)

    monkeypatch.setattr(data_loader, "create_dataset", fake_create_dataset)
    cfg = _base_cfg(tmp_path)
    cfg.moe.routergfm.apps.max_diagnostic = 8
    cfg.moe.routergfm.apps.data.split_root = str(tmp_path / "splits")
    app = AppSpec("cora", level, 2, 42)
    data = RealDataProvider(cfg).load(app)
    assert calls[0]["pad_featureless_features"] and calls[0]["induced"] and calls[0]["seed"] == 42
    assert calls[0]["split"] == app.split and calls[0]["split_root"] == str(tmp_path / "splits")
    n_train = 6 if level == "node" else 8
    assert data.support_pos.tolist() == list(range(n_train))
    assert data.query_pos.tolist() == list(range(n_train if level == "node" else 12, 24))
    assert set(data.diag_pos.tolist()) <= set(data.query_pos.tolist()) and data.diag_pos.numel() == 8
    assert data.task_family == (NODE_CLS if level == "node" else LINK)
    assert data.num_classes == (3 if level == "node" else 2) and data.in_dim == FEATURE_DIM
    expected = [data.dataset[int(p)].y.item() for p in data.diag_pos]
    assert data.labels["diag"].tolist() == expected
    assert data.stats["num_nodes"] == 50 and data.stats["budget"] == 2 and data.stats["domain_citation"] == 1
    assert RouterPaths.from_cfg(cfg).data_meta_file(app.data_key).is_file()

    # A fresh provider reuses the disk metadata and builds the dataset only on access
    # (both LP budgets share one data key).
    calls.clear()
    budget = 7 if level == "edge" else 2
    other = RealDataProvider(cfg).load(AppSpec("cora", level, budget, 42))
    assert calls == []
    assert torch.equal(other.query_pos, data.query_pos) and other.stats["budget"] == budget
    assert len(other.dataset) == 24 and len(calls) == 1


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
def test_synthetic_provider_families(tmp_path):
    cfg = tiny_cfg(tmp_path, write_checkpoints=False)
    provider = SyntheticDataProvider()
    applications = enumerate_applications(cfg.moe.routergfm)
    loaded = {a.key: provider.load(a) for a in applications}
    families = {a.dataset: d.task_family for a, d in zip(applications, loaded.values())}
    assert families["nodea"] == NODE_CLS and families["linka"] == LINK and families["srcb"] == GRAPH_CLS
    assert families["multia"] == MULTILABEL and families["rega"] == REGRESSION
    for data in loaded.values():
        s, q = set(data.support_pos.tolist()), set(data.query_pos.tolist())
        assert not (s & q) and set(data.diag_pos.tolist()) <= q and data.diag_pos.numel() <= TINY_MAX_DIAG
        assert data.labels["support"].size(0) == len(s) and data.labels["diag"].size(0) == data.diag_pos.numel()
        assert set(apps_mod.STAT_COUNT_KEYS) <= set(data.stats) and data.in_dim == FEATURE_DIM
        if data.task_family in (NODE_CLS, GRAPH_CLS):
            assert torch.bincount(data.labels["support"]).tolist() == [data.app.budget] * 3
    node = loaded["nodea__node__b3__s42"]
    g = node.dataset[0]
    assert g.target_node_index.numel() == 1 and hasattr(g, "base_node_id")
    link = next(d for d in loaded.values() if d.task_family == LINK)
    for p in link.query_pos[:10].tolist():
        g = link.dataset[p]
        u, v = g.edge_label_index.view(-1).tolist()
        pairs = set(map(tuple, g.edge_index.t().tolist()))
        assert (u, v) not in pairs and (v, u) not in pairs
    lp_b3, lp_b6 = (provider.load(AppSpec("linka", "edge", b, 42)) for b in (3, 6))
    assert torch.equal(lp_b3.support_pos, lp_b6.support_pos) and torch.equal(lp_b3.diag_pos, lp_b6.diag_pos)
    ml = next(d for d in loaded.values() if d.task_family == MULTILABEL)
    assert torch.isnan(ml.labels["query"]).any() and ml.labels["query"].shape[1] == ml.num_classes
    again = SyntheticDataProvider().load(node.app)
    assert torch.equal(again.dataset[5].x, node.dataset[5].x) and torch.equal(again.diag_pos, node.diag_pos)


def test_tiny_checkpoints_catalog_and_embeddings(tmp_path):
    cfg = tiny_cfg(tmp_path)
    catalog = build_expert_catalog(cfg)
    assert len(catalog) == 8 and {e.source_task_level for e in catalog} == {"node", "graph"}
    provider = SyntheticDataProvider()
    device = torch.device("cpu")
    node = provider.load(AppSpec("nodea", "node", 3, 42))
    link = provider.load(AppSpec("linka", "edge", 3, 42))
    graph = provider.load(AppSpec("grapha", "graph", 3, 42))
    encoder, model_cfg = load_frozen_encoder(cfg, catalog[0], device)
    assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
    assert int(model_cfg.model.out_dim) == 8 and model_cfg.model.name == catalog[0].architecture
    for data, width in ((node, 8), (link, 32), (graph, 8)):
        emb = embed_instances(encoder, model_cfg, data, data.query_pos, device, batch_size=7)
        assert emb.shape == (data.query_pos.numel(), width) and emb.dtype == torch.float32
        rev = embed_instances(encoder, model_cfg, data, data.query_pos.flip(0), device, batch_size=7)
        assert torch.allclose(rev.flip(0), emb, atol=1e-5)
    assert embed_instances(encoder, model_cfg, node, torch.tensor([], dtype=torch.long), device, 4).shape == (0, 8)

    # Heads on frozen random experts learn the synthetic class structure.
    emb_s = embed_instances(encoder, model_cfg, node, node.support_pos, device, 32)
    emb_q = embed_instances(encoder, model_cfg, node, node.query_pos, device, 32)
    head = fit_head(emb_s, node.labels["support"], NODE_CLS, node.num_classes, cfg, seed=0)
    brier = routing_loss(predict_head(head, emb_q), node.labels["query"], NODE_CLS).mean()
    assert brier < 1.0 / 3.0  # uniform prediction scores 1/3


@pytest.mark.parametrize("arch", ["gat", "h2gcn", "fagcn", "nodeformer", "transformer"])
def test_tiny_checkpoints_every_architecture_loads(tmp_path, arch):
    make_tiny_checkpoints(tmp_path / "ckpt", archs=(arch,), objectives=("infograph",), sources=("srca",))
    cfg = _base_cfg(tmp_path, checkpoint_root=str(tmp_path / "ckpt"), sources=["srca"],
                    objectives=["infograph"], architectures=[arch])
    (spec,) = build_expert_catalog(cfg)
    assert spec.objective == "infograph" and spec.objective_variant == (
        "infograph-nolw" if arch in ("fagcn", "h2gcn", "nodeformer", "transformer") else "infograph"
    )
    encoder, model_cfg = load_frozen_encoder(cfg, spec, torch.device("cpu"))
    data = SyntheticDataProvider().load(AppSpec("linka", "edge", 3, 42))
    emb = embed_instances(encoder, model_cfg, data, data.support_pos, torch.device("cpu"), 16)
    assert emb.shape == (data.support_pos.numel(), 32) and torch.isfinite(emb).all()
