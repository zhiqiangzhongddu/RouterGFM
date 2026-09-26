"""History store / generation, query predictions, and the RouterInfra facade on tiny synthetic data."""

from __future__ import annotations

import math
import shutil
from types import SimpleNamespace

import pytest
import torch

from src.moe.routergfm import experts as experts_mod
from src.moe.routergfm.common import REGRESSION, AppSpec, RouterPaths, is_same_source
from src.moe.routergfm.context_graph import APP_NUMERIC_NAMES, EXPERT_NUMERIC_NAMES
from src.moe.routergfm.descriptors import descriptor_names
from src.moe.routergfm.experts import build_expert_catalog, load_frozen_encoder
from src.moe.routergfm.history import (
    HistoryStore,
    embed_parts,
    embed_splits,
    generate_history,
    instance_losses,
    predict_queries,
)
from src.moe.routergfm.infra import RouterInfra
from src.moe.routergfm.losses import RegressionNormalizer
from tests.routergfm.fixtures import SyntheticDataProvider, tiny_cfg

APPS = [
    AppSpec("nodea", "node", 3, 42),  # nodea b3 / b6: one instance set, two data keys
    AppSpec("nodea", "node", 6, 42),
    AppSpec("srca", "node", 3, 42),  # same source as the srca experts
    AppSpec("linka", "edge", 3, 42),  # LP budgets share one data key
    AppSpec("linka", "edge", 6, 42),
    AppSpec("rega", "graph", 3, 42),
    AppSpec("multia", "graph", 3, 42),
]
RECORD_KEYS = {
    "diag_pos", "pred", "loss", "mu", "count", "family", "num_classes", "normalizer",
    "support_size", "head_cfg_hash", "support_emb", "support_pos",
}


def _cfg(tmp):
    return tiny_cfg(
        tmp,
        targets=("nodea:node",),
        history_extra=("srca:node", "linka:edge", "rega:graph", "multia:graph"),
        archs=("gcn", "nodeformer"),
        objectives=("dgi",),
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    cfg = _cfg(tmp_path_factory.mktemp("history"))
    provider = SyntheticDataProvider()
    generate_history(cfg, provider, apps=APPS)
    return SimpleNamespace(
        cfg=cfg, provider=provider, catalog=build_expert_catalog(cfg), paths=RouterPaths.from_cfg(cfg)
    )


def _forbid_encoder_loads(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("an encoder was loaded although everything is cached")

    monkeypatch.setattr(experts_mod, "load_frozen_encoder", _fail)


# --------------------------------------------------------------------------- #
# generate_history
# --------------------------------------------------------------------------- #
def test_records_and_application_averages(env):
    store = HistoryStore(env.paths)
    for app in APPS:
        data = env.provider.load(app)
        for spec in env.catalog:
            expected = not is_same_source(app, spec)
            assert store.has(app.data_key, spec.expert_id) == expected
            assert env.paths.history_file(app.data_key, spec.expert_id).is_file() == expected
            if not expected:
                continue
            rec = store.load(app.data_key, spec.expert_id)
            assert RECORD_KEYS <= set(rec)
            assert torch.equal(rec["diag_pos"], data.diag_pos)
            assert torch.equal(rec["support_pos"], data.support_pos)
            assert rec["pred"].dtype == torch.float16 and rec["pred"].shape == (data.diag_pos.numel(), data.num_classes)
            assert rec["loss"].dtype == torch.float32 and rec["loss"].shape == (data.diag_pos.numel(),)
            assert rec["support_emb"].dtype == torch.float16 and rec["support_emb"].size(0) == data.support_pos.numel()
            assert rec["family"] == data.task_family and rec["support_size"] == data.support_pos.numel()
            assert (rec["normalizer"] is not None) == (data.task_family == REGRESSION)
            # Eq. 2 on exactly the valid observations.
            valid = torch.isfinite(rec["loss"])
            assert rec["count"] == int(valid.sum()) > 0
            assert rec["mu"] == pytest.approx(float(rec["loss"][valid].mean()), rel=1e-6)
            # Losses are the routing losses of the recorded predictions (up to float16 storage).
            norm = RegressionNormalizer.from_state_dict(rec["normalizer"])
            again = instance_losses(env.cfg, rec["pred"].float(), data.labels["diag"], data.task_family, norm)
            assert torch.allclose(again, rec["loss"], atol=5e-3, equal_nan=True)


def test_same_source_skip_and_shared_lp_record(env):
    store = HistoryStore(env.paths)
    srca = APPS[2]
    assert store.expert_ids(srca.data_key) == sorted(s.expert_id for s in env.catalog if s.source != "srca")
    lp3, lp6 = APPS[3], APPS[4]
    assert lp3.data_key == lp6.data_key
    assert store.expert_ids(lp3.data_key) == sorted(s.expert_id for s in env.catalog)
    assert store.app_average(lp3, env.catalog[0].expert_id) == store.app_average(lp6, env.catalog[0].expert_id)


def test_rerun_is_idempotent(env, monkeypatch):
    files = sorted((env.paths.root / "history").rglob("*.pt"))
    stamps = {f: f.stat().st_mtime_ns for f in files}
    _forbid_encoder_loads(monkeypatch)
    generate_history(env.cfg, env.provider, apps=APPS)
    assert sorted((env.paths.root / "history").rglob("*.pt")) == files
    assert all(f.stat().st_mtime_ns == stamps[f] for f in files)


def test_expert_sharding_selection_and_matrix_rebuild(tmp_path):
    cfg = _cfg(tmp_path)
    provider = SyntheticDataProvider()
    paths = RouterPaths.from_cfg(cfg)
    app = AppSpec("nodea", "node", 3, 42)
    ids = [s.expert_id for s in build_expert_catalog(cfg)]

    cfg.moe.routergfm.experts.num_shards = 2
    cfg.moe.routergfm.experts.shard_index = 1
    generate_history(cfg, provider, apps=[app])
    assert HistoryStore(paths).expert_ids(app.data_key) == sorted(ids[1::2])
    assert not paths.descriptor_file(app.data_key).exists()  # descriptors are shard 0's job
    first_ids, first_loss = HistoryStore(paths).loss_matrix(app.data_key)
    assert (paths.history_file(app.data_key, "_matrix")).is_file()

    cfg.moe.routergfm.experts.shard_index = 0
    generate_history(cfg, provider, apps=[app], expert_ids=ids[:1])  # explicit subset, then sharded
    assert HistoryStore(paths).expert_ids(app.data_key) == sorted(ids[1::2] + ids[:1])
    assert paths.descriptor_file(app.data_key).is_file()

    store = HistoryStore(paths)
    all_ids, loss = store.loss_matrix(app.data_key)  # expert set changed on disk -> rebuilt
    assert all_ids == sorted(ids[1::2] + ids[:1])
    for j, eid in enumerate(first_ids):
        assert torch.allclose(loss[:, all_ids.index(eid)], first_loss[:, j], atol=0.0, equal_nan=True)
    persisted = torch.load(str(paths.history_file(app.data_key, "_matrix")))
    assert persisted["expert_ids"] == all_ids


def test_matrix_views(env):
    store = HistoryStore(env.paths)
    app = APPS[0]
    ids, loss = store.loss_matrix(app.data_key)
    assert ids == sorted(s.expert_id for s in env.catalog)
    records = [store.load(app.data_key, e) for e in ids]
    assert loss.dtype == torch.float32
    assert torch.equal(loss, torch.stack([r["loss"] for r in records], dim=1))
    preds = store.pred_matrix(app.data_key, ids[::-1] + ["missing"])
    assert preds.shape == (loss.size(0), len(ids) + 1, 3)
    assert torch.equal(preds[:, 0], records[-1]["pred"].float())
    assert torch.isnan(preds[:, -1]).all()
    assert torch.equal(store.preds(app, ids), store.pred_matrix(app.data_key, ids))
    partial = store.losses(app, [ids[1], "missing"])
    assert torch.equal(partial[:, 0], loss[:, 1]) and torch.isnan(partial[:, 1]).all()
    for eid, rec in zip(ids, records):
        mu, count = store.app_average(app, eid)
        assert mu == pytest.approx(rec["mu"], rel=1e-6) and count == rec["count"]
    mu, count = store.app_average(APPS[2], env.catalog[0].expert_id)  # same-source: never recorded
    assert math.isnan(mu) and count == 0
    assert HistoryStore(env.paths).loss_matrix(app.data_key)[0] == ids  # fresh store reads _matrix.pt


# --------------------------------------------------------------------------- #
# predict_queries
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("app_idx", [1, 3, 5, 6])
def test_predict_queries_matches_history_heads_and_caches(env, monkeypatch, app_idx):
    app = APPS[app_idx]
    store = HistoryStore(env.paths)
    data = env.provider.load(app)
    ids = store.expert_ids(app.data_key)
    out = predict_queries(env.cfg, app, ids, env.provider)
    assert list(out) == ids
    for eid in ids:
        entry = out[eid]
        assert torch.equal(entry["query_pos"], data.query_pos) and torch.equal(entry["support_pos"], data.support_pos)
        assert entry["pred"].shape == (data.query_pos.numel(), data.num_classes)
        assert entry["support_pred"].shape == entry["support_oof_pred"].shape == (data.support_pos.numel(), data.num_classes)
        assert torch.isfinite(entry["support_oof_pred"]).all()
        assert env.paths.prediction_file(app.data_key, eid).is_file()
        # Same head as the history record: identical predictions on D_a (NodeFormer included).
        rows = torch.searchsorted(data.query_pos, data.diag_pos)
        assert torch.allclose(entry["pred"][rows], store.load(app.data_key, eid)["pred"].float(), atol=1e-3)

    _forbid_encoder_loads(monkeypatch)
    again = predict_queries(env.cfg, app, ids, env.provider)
    for eid in ids:
        for key in ("pred", "support_pred", "support_oof_pred"):
            assert again[eid][key].dtype == torch.float32
            assert torch.equal(again[eid][key], out[eid][key])


def test_nodeformer_embeddings_follow_canonical_parts(env):
    spec = next(s for s in env.catalog if s.architecture == "nodeformer")
    encoder, model_cfg = load_frozen_encoder(env.cfg, spec, torch.device("cpu"))
    data = env.provider.load(APPS[0])
    parts = {"support": data.support_pos, "diag": data.diag_pos}
    per_part = embed_parts(encoder, model_cfg, data, parts, "cpu", 32, per_part=True)
    union = embed_parts(encoder, model_cfg, data, parts, "cpu", 32, per_part=False)
    assert not torch.allclose(per_part["support"], union["support"])  # batch-dependent encoder
    emb = embed_splits(encoder, model_cfg, spec, data, ("support", "diag", "query"), "cpu", 32)
    assert torch.equal(emb["support"], per_part["support"]) and torch.equal(emb["diag"], per_part["diag"])
    assert torch.equal(emb["query"][torch.searchsorted(data.query_pos, data.diag_pos)], emb["diag"])


# --------------------------------------------------------------------------- #
# RouterInfra
# --------------------------------------------------------------------------- #
def test_infra_facade(env):
    infra = RouterInfra(env.cfg, env.provider)
    store = HistoryStore(env.paths)
    app = infra.application("nodea:node", 3, 42)
    assert app == APPS[0]
    assert infra.data(app) is infra.data(app)
    assert infra.task_family(app) == "node_cls"
    srca = APPS[2]
    assert infra.compatible_pool(srca) == [s.expert_id for s in env.catalog if s.source != "srca"]
    assert [e.expert_id for e in infra.catalog] == [s.expert_id for s in env.catalog]

    hist = infra.historical_applications(app)
    assert all(a.group != "nodea" for a in hist)
    assert hist[0] == srca and set(hist) == set(APPS[2:])  # same family first; only apps with history

    data = infra.data(app)
    eid = next(s.expert_id for s in env.catalog if s.architecture == "nodeformer")
    support = infra.embeddings(app, eid, "support")
    assert torch.equal(support, store.load(app.data_key, eid)["support_emb"].float())
    query = infra.embeddings(app, eid, "query")
    diag = infra.embeddings(app, eid, "diag")
    assert query.shape == (data.query_pos.numel(), support.size(1))
    assert torch.equal(query[torch.searchsorted(data.query_pos, data.diag_pos)], diag)
    cached = env.paths.root / "embeddings" / app.data_key / f"{eid}__query.pt"
    assert cached.is_file() and torch.load(cached).dtype == torch.float16
    assert torch.equal(infra.embeddings(app, eid, "query"), query)

    assert torch.equal(infra.support_labels(app), data.labels["support"])
    assert infra.normalizer(app) is None and isinstance(infra.normalizer(APPS[5]), RegressionNormalizer)
    preds = infra.expert_predictions(app, [eid])
    assert torch.equal(preds[eid]["pred"], predict_queries(env.cfg, app, [eid], env.provider)[eid]["pred"])
    assert infra.historical_mu(app, eid) == store.app_average(app, eid)

    num_spectral = int(env.cfg.moe.routergfm.descriptors.num_spectral)
    assert infra.descriptors(app, "query").shape == (data.query_pos.numel(), len(descriptor_names(num_spectral)))
    assert infra.descriptors(app, "support").shape[0] == data.support_pos.numel()
    graphs = infra.instance_graphs(app, "query")
    assert len(graphs) == data.query_pos.numel()
    assert all("y" not in g for g in graphs[:5]) and "y" in data.dataset[int(data.query_pos[0])]

    hash_dim = int(env.cfg.moe.routergfm.graph.hash_dim)
    assert infra.app_metadata(app).shape == (hash_dim + len(APP_NUMERIC_NAMES),)
    assert infra.expert_metadata(eid).shape == (hash_dim + len(EXPERT_NUMERIC_NAMES),)

    assert infra.query_expert_risk(app, eid) == store.app_average(app, eid)[0]
    with pytest.raises(KeyError):
        infra.query_expert_risk(srca, env.catalog[0].expert_id)


@pytest.mark.parametrize("app_idx,metric", [(0, "acc"), (3, "auc"), (5, "mae"), (6, "auc")])
def test_evaluate_outputs(env, app_idx, metric):
    infra = RouterInfra(env.cfg, env.provider)
    app = APPS[app_idx]
    data = infra.data(app)
    eid = HistoryStore(env.paths).expert_ids(app.data_key)[0]
    pred = infra.expert_predictions(app, [eid])[eid]["pred"]
    result = infra.evaluate_outputs(app, pred)
    assert math.isfinite(result[metric])
    loss = instance_losses(env.cfg, pred, data.labels["query"], data.task_family, infra.normalizer(app))
    assert result["risk"] == pytest.approx(float(loss[torch.isfinite(loss)].mean()), rel=1e-5)
    with pytest.raises(ValueError):
        infra.evaluate_outputs(app, pred[:-1])


class _GuardedLabels(dict):
    def __init__(self, labels, log):
        super().__init__(labels)
        self.log = log

    def __getitem__(self, key):
        if key != "support":
            self.log.append(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self[key] if key in self else default


class _GuardedProvider:
    def __init__(self, inner):
        self.inner = inner
        self.log = []

    def load(self, app):
        data = self.inner.load(app)
        data.labels = _GuardedLabels(data.labels, self.log)
        return data


def test_no_query_labels_outside_evaluation(env):
    provider = _GuardedProvider(env.provider)
    infra = RouterInfra(env.cfg, provider)
    app = APPS[5]  # regression: exercises the normalizer path too
    shutil.rmtree(env.paths.root / "predictions" / app.data_key, ignore_errors=True)
    shutil.rmtree(env.paths.root / "embeddings" / app.data_key, ignore_errors=True)
    pool = infra.compatible_pool(app)
    infra.historical_applications(app)
    for eid in pool:
        for split in ("support", "diag", "query"):
            infra.embeddings(app, eid, split)
            infra.descriptors(app, split)
        infra.historical_mu(app, eid)
        infra.expert_metadata(eid)
    infra.expert_predictions(app, pool)  # refits every head
    infra.support_labels(app)
    infra.normalizer(app)
    infra.app_metadata(app)
    list(infra.instance_graphs(app, "query"))
    assert provider.log == []

    infra.evaluate_outputs(app, infra.expert_predictions(app, pool[:1])[pool[0]]["pred"])
    assert provider.log == ["query"]
