"""Selection harness, LogME / metadata MLP / nearest-application selectors, and the matched-pool runner."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.moe.routergfm.baselines import BASELINE_RUNNERS, config_block_name
from src.moe.routergfm.baselines import run as matched_run
from src.moe.routergfm.baselines.selection import run as selection_run
from src.moe.routergfm.baselines.selection.common import (
    ColumnScaler,
    QueryAccessError,
    QueryGuard,
    SelectionOutcome,
    block_concat,
    episode_metrics,
    selection_metrics,
)
from src.moe.routergfm.baselines.selection.logme import (
    LogMESelector,
    logme_score,
    num_usable_columns,
    to_logme_targets,
)
from src.moe.routergfm.baselines.selection.metadata_mlp import (
    MetadataMLP,
    MetadataMLPSelector,
    application_balanced_huber,
    validation_regret_at_k,
)
from src.moe.routergfm.baselines.selection.nearest_application import (
    NearestApplicationSelector,
    global_losses,
    nearest_set,
    score_experts,
)
from src.moe.routergfm.common import AppSpec, enumerate_applications, parse_dataset_spec
from src.moe.routergfm.context_graph import APP_NUMERIC_NAMES
from src.moe.routergfm.history import HistoryStore, generate_history
from src.moe.routergfm.infra import RouterInfra
from tests.routergfm.fixtures import SyntheticDataProvider, tiny_cfg

TARGET = "nodea:node"


# --------------------------------------------------------------------------- #
# selection_metrics
# --------------------------------------------------------------------------- #
def test_selection_metrics_hand_made_cases():
    risk = {"a": 0.1, "b": 0.1, "c": 0.3, "d": 0.5}
    tie = selection_metrics(["c", "b", "a", "d"], risk, 2)  # b ties the best
    assert tie == {"hit_at_k": 1.0, "regret_at_k": 0.0, "best_rank": 2.0}
    miss = selection_metrics([("d", 9.0), ("c", 8.0), ("a", 1.0)], risk, 2)
    assert miss["hit_at_k"] == 0.0 and miss["regret_at_k"] == pytest.approx(0.2) and miss["best_rank"] == 3.0
    near = selection_metrics(["b"], {"a": 0.1, "b": 0.1 + 1e-13}, 1)  # within the tie tolerance
    assert near["hit_at_k"] == 1.0 and near["regret_at_k"] == 0.0
    partial = selection_metrics(["c", "x"], {**risk, "x": float("nan")}, 5)  # NaN/unknown ignored; best unlisted
    assert partial["hit_at_k"] == 0.0 and partial["regret_at_k"] == pytest.approx(0.2) and math.isnan(partial["best_rank"])
    assert all(math.isnan(v) for v in selection_metrics(["a"], {}, 1).values())


def test_episode_metrics_names_and_variants():
    app = AppSpec("nodea", "node", 3, 42)
    out = SelectionOutcome.from_ranking(app, [("c", 3.0), ("a", 2.0), ("b", 1.0)], 1, variants={"k0": ["a"]})
    m = episode_metrics(out, {"a": 0.1, "b": 0.2, "c": 0.4}, 1)
    assert m["test_hit_at_1"] == 0.0 and m["test_regret_at_1"] == pytest.approx(0.3) and m["test_best_rank"] == 2.0
    assert m["test_hit_at_1_k0"] == 1.0 and m["test_regret_at_1_k0"] == 0.0
    assert not any("loss" in k for k in m)
    assert SelectionOutcome.from_dict(json.loads(json.dumps(out.to_dict()))) == out


# --------------------------------------------------------------------------- #
# LogME core
# --------------------------------------------------------------------------- #
def _algorithm1_reference(F: np.ndarray, Y: np.ndarray) -> float:
    """Direct transcription of Algorithm 1 / Eq. 2 (eigendecomposition, explicit A, full log|A|)."""
    n, D = F.shape
    sigma = np.linalg.eigvalsh(F.T @ F)
    out = []
    for y in Y.T:
        alpha, beta = 1.0, 1.0
        for _ in range(10000):
            A = alpha * np.eye(D) + beta * F.T @ F
            m = beta * np.linalg.solve(A, F.T @ y)
            gamma = np.sum(beta * sigma / (alpha + beta * sigma))
            new_alpha, new_beta = gamma / (m @ m), (n - gamma) / np.sum((F @ m - y) ** 2)
            done = abs(new_alpha - alpha) / alpha < 1e-13 and abs(new_beta - beta) / beta < 1e-13
            alpha, beta = new_alpha, new_beta
            if done:
                break
        A = alpha * np.eye(D) + beta * F.T @ F
        m = beta * np.linalg.solve(A, F.T @ y)
        evidence = (
            n / 2 * np.log(beta) + D / 2 * np.log(alpha) - n / 2 * np.log(2 * np.pi)
            - beta / 2 * np.sum((F @ m - y) ** 2) - alpha / 2 * m @ m - 0.5 * np.linalg.slogdet(A)[1]
        )
        out.append(evidence / n)
    return float(np.mean(out))


def _classification(n_per_class=100, d=16, classes=3, noise=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = 2 * torch.randn(classes, d, generator=g)
    y = torch.arange(classes).repeat_interleave(n_per_class)
    return centers[y] + noise * torch.randn(len(y), d, generator=g), y


def test_logme_matches_algorithm1_reference():
    f, y = _classification()
    ref = _algorithm1_reference(f.double().numpy(), torch.nn.functional.one_hot(y).double().numpy())
    assert abs(logme_score(f, y, standardize=False) - ref) < 1e-8


def test_logme_scale_invariance_and_standardize_flag():
    f, y = _classification(n_per_class=30)
    tight = dict(standardize=False, max_iter=2000, tol=1e-13)  # invariance holds at the fixed point
    for c in (0.01, 7.3):
        assert logme_score(c * f, y, **tight) == pytest.approx(logme_score(f, y, **tight), abs=1e-10)
    base = logme_score(f, y, standardize=False)
    shifted = f.clone()
    shifted[:, 0] += 5.0  # no bias term: an offset changes the paper/official score ...
    assert abs(logme_score(shifted, y, standardize=False) - base) > 1e-6
    # ... but not the legacy z-scored variant.
    assert logme_score(shifted, y, standardize=True) == pytest.approx(logme_score(f, y, standardize=True), abs=1e-9)


def test_logme_skips_degenerate_float_columns_and_few_shot():
    g = torch.Generator().manual_seed(3)
    f = torch.randn(30, 8, generator=g)
    good = f @ torch.randn(8, generator=g) + 0.1 * torch.randn(30, generator=g)
    const, empty = torch.full((30,), 2.0), torch.full((30,), float("nan"))
    stacked = torch.stack([good, const, empty], dim=1)
    for standardize in (False, True):
        assert logme_score(f, stacked, standardize=standardize) == pytest.approx(
            logme_score(f, good[:, None], standardize=standardize), abs=1e-12
        )
    assert num_usable_columns(stacked) == 1
    feats, labels = _classification(n_per_class=5, d=128, classes=4, noise=0.05)
    informative = logme_score(feats, labels, standardize=False)
    assert math.isfinite(informative) and informative > logme_score(torch.randn_like(feats), labels, standardize=False)


def test_to_logme_targets():
    signed = torch.tensor([[1, -1, 0], [-1, 0, 1]])
    out = to_logme_targets(signed, "multilabel")
    assert out.dtype == torch.float32 and torch.isnan(out[0, 2]) and out[0, 0] == 1 and out[0, 1] == 0
    assert to_logme_targets(torch.tensor([0, 1, 1]), "link").dtype == torch.int64
    reg = torch.tensor([[1.5, -2.0], [0.3, 4.0]])
    assert torch.equal(to_logme_targets(reg, "regression"), reg)
    assert torch.equal(to_logme_targets(torch.tensor([[2], [0]]), "node_cls"), torch.tensor([2, 0]))


class _FakeLogMEInfra:
    def __init__(self):
        feats, self.y = _classification(n_per_class=6, d=8, classes=3, noise=0.1, seed=4)
        g = torch.Generator().manual_seed(5)
        noise = torch.randn(feats.shape, generator=g)
        self.emb = {f"e{i}": torch.randn(feats.shape, generator=g) for i in range(6)}
        self.emb["e3"] = feats
        self.emb["e1"] = self.emb["e5"] = noise

    def compatible_pool(self, app):
        return ["e5", "e1", "e3", "e0", "e2", "e4"]

    def task_family(self, app):
        return "node_cls"

    def support_labels(self, app):
        return self.y

    def embeddings(self, app, expert_id, split):
        assert split == "support"
        return self.emb[expert_id]


def test_logme_selector_ranks_full_pool_and_breaks_ties_by_id():
    cfg = tiny_cfg(Path("/nonexistent"), write_checkpoints=False)
    cfg.moe.routergfm.baselines.topk = 2
    out = LogMESelector(cfg, _FakeLogMEInfra()).rank(AppSpec("x", "node", 5, 42))
    ids = [e for e, _ in out.ranking]
    assert sorted(ids) == ["e0", "e1", "e2", "e3", "e4", "e5"] and out.team == ids[:2] and ids[0] == "e3"
    assert ids.index("e1") < ids.index("e5")
    assert out.num_target_executions == 6 and out.extras["num_usable_columns"] == 3


# --------------------------------------------------------------------------- #
# Metadata helpers and selectors on fakes
# --------------------------------------------------------------------------- #
def test_scalers_and_block_concat():
    X = torch.tensor([[0.0, 1.0, float("nan")], [2.0, 1.0, 3.0]])
    z = ColumnScaler("standard").fit(X).transform(torch.tensor([[1.0, 5.0, float("nan")]]))
    assert torch.allclose(z, torch.tensor([[0.0, 4.0, 0.0]]))  # constant column keeps scale 1; NaN -> 0
    mm = ColumnScaler("minmax").fit(X).transform(torch.tensor([[4.0, 5.0, 3.0], [-1.0, 1.0, 1.0]]))
    assert torch.allclose(mm, torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    num, text = torch.tensor([[3.0, 4.0]]), torch.tensor([[0.0, 2.0, 0.0]])
    out = block_concat(num, 100 * text)
    assert torch.allclose(out, torch.tensor([[0.6, 0.8, 0.0, 1.0, 0.0]]))


def test_balanced_huber_and_validation_regret():
    mu_hat = torch.tensor([0.0, 0.0, 0.0, 0.0])
    mu_bar = torch.tensor([0.4, 0.1, 0.2, 0.3])
    app_index = torch.tensor([0, 1, 1, 1])
    per = 0.5 * mu_bar**2  # Huber with delta 1 and |r| <= 1 is 0.5 * r^2
    expected = 0.5 * (per[0] + per[1:].mean())
    assert application_balanced_huber(mu_hat, mu_bar, app_index) == pytest.approx(float(expected))
    pred = torch.tensor([0.0, 0.3, 0.1, 0.2])  # app 1: top-1 by pred is mu_bar 0.2 (best 0.1)
    assert validation_regret_at_k(pred, mu_bar, app_index, 1) == pytest.approx(0.05)
    model = MetadataMLP(3, 2, hidden_dim=8)
    out = model(torch.randn(4, 3), torch.randn(5, 2), torch.tensor([0, 1, 2, 3, 0]), torch.arange(5))
    assert out.shape == (5,) and bool((out >= 0).all())


def test_nearest_helpers():
    cands = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 1e-3], [0.0, 1.0]])
    idx, sims = nearest_set(torch.tensor([1.0, 0.0]), cands)
    assert idx.tolist() == [0, 1] and sims[3] == pytest.approx(0.0)
    nan = float("nan")
    mu = torch.tensor([[0.2, nan, 0.5], [0.4, nan, nan]])
    scores, imputed = score_experts(mu)
    assert scores.tolist() == pytest.approx([0.3, (0.2 + 0.5 + 0.4) / 3, 0.5]) and imputed.tolist() == [False, True, False]
    assert global_losses(mu).tolist() == pytest.approx([0.3, math.inf, 0.5])


_NUM = len(APP_NUMERIC_NAMES)


class _FakeMetaInfra:
    """History and metadata of hand-made applications; any label or query accessor raises."""

    def __init__(self, apps, app_x, expert_x, mu, family):
        self.apps, self.app_x, self.expert_x, self.mu, self.family = apps, app_x, expert_x, mu, family
        self.catalog = [SimpleNamespace(expert_id=e) for e in expert_x]

    def historical_applications(self, target):
        return list(self.apps)  # deliberately includes the target's group: selectors must drop it

    def compatible_pool(self, app):
        return list(self.expert_x)

    def task_family(self, app):
        return self.family.get(app.key, "node_cls")

    def historical_mu(self, app, expert_id):
        value = self.mu.get((app.key, expert_id))
        return (float("nan"), 0) if value is None else (value, 3)

    def app_metadata(self, app):
        return self.app_x[app.key]

    def expert_metadata(self, expert_id):
        return self.expert_x[expert_id]

    def __getattr__(self, name):
        raise AssertionError(f"label-free selector accessed infra.{name}")


def _meta_vector(cluster: int, text_seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(text_seed)
    numeric = torch.zeros(_NUM)
    numeric[0] = 3.0 + 4.0 * cluster
    numeric[1] = 1.0 + cluster
    return torch.cat([torch.randn(6, generator=g) + 3.0 * cluster, numeric])


def _cluster_world(num_groups=8):
    """Experts 0-4 are best on cluster-0 applications, 5-9 on cluster-1 applications."""
    apps, app_x, mu = [], {}, {}
    experts = [f"x{j}" for j in range(10)]
    g = torch.Generator().manual_seed(0)
    expert_x = {e: torch.randn(4, generator=g) + (2.0 if j >= 5 else -2.0) for j, e in enumerate(experts)}
    for i in range(num_groups):
        cluster = i % 2
        for seed in (0, 1):
            app = AppSpec(f"ds{i}", "node", 3, seed)
            apps.append(app)
            app_x[app.key] = _meta_vector(cluster, text_seed=i)
            for j, e in enumerate(experts):
                good = (j >= 5) == bool(cluster)
                mu[(app.key, e)] = (0.1 if good else 0.4) + 0.01 * float(torch.rand(1, generator=g))
    return apps, app_x, expert_x, mu


def test_metadata_mlp_learns_and_excludes_target_group():
    apps, app_x, expert_x, mu = _cluster_world()
    target = AppSpec("ds1", "node", 3, 7)  # group ds1 (cluster 1) is in the history: must be excluded
    app_x[target.key] = app_x[apps[2].key].clone()
    cfg = tiny_cfg(Path("/nonexistent"), write_checkpoints=False)
    m = cfg.moe.routergfm.baselines.metadata_mlp
    m.hidden_dim, m.epochs, m.patience, m.lr, m.dropout = 32, 150, 150, 3e-3, 0.0
    cfg.moe.routergfm.baselines.topk = 5
    selector = MetadataMLPSelector(cfg, _FakeMetaInfra(apps, app_x, expert_x, mu, {}))
    out = selector.rank(target)
    assert selector.fit_for(target).n_historical_apps == len(apps) - 2
    assert set(out.team) == {f"x{j}" for j in range(5, 10)}
    assert len(out.ranking) == 10 and out.num_target_executions == 0
    scores = [s for _, s in out.ranking]
    assert scores == sorted(scores, reverse=True)
    again = MetadataMLPSelector(cfg, _FakeMetaInfra(apps, app_x, expert_x, mu, {})).rank(target)
    assert again.ranking == out.ranking  # deterministic


def test_nearest_application_toy():
    apps, app_x, expert_x, mu = _cluster_world(num_groups=4)
    target = AppSpec("new", "node", 3, 42)
    app_x[target.key] = app_x[apps[2].key].clone()  # identical metadata to ds1 (both seeds tie)
    for e in ("x3", "x7"):  # ds1's single best experts among the observed
        for seed in (0, 1):
            mu[(f"ds1__node__b3__s{seed}", e)] = 0.01
    mu.pop(("ds1__node__b3__s0", "x9"))
    mu.pop(("ds1__node__b3__s1", "x9"))  # x9 unobserved on N(a): imputed with N(a)'s row mean
    other = AppSpec("graphy", "graph", 3, 0)
    apps.append(other)
    app_x[other.key] = app_x[target.key].clone()  # identical metadata but another family: never a candidate
    for e in expert_x:
        mu[(other.key, e)] = 0.0
    cfg = tiny_cfg(Path("/nonexistent"), write_checkpoints=False)
    cfg.moe.routergfm.baselines.topk = 2
    infra = _FakeMetaInfra(apps, app_x, expert_x, mu, {other.key: "graph_cls"})
    out = NearestApplicationSelector(cfg, infra).rank(target)
    assert set(out.team) == {"x3", "x7"} and out.extras["nearest_apps"] == ["ds1__node__b3__s0", "ds1__node__b3__s1"]
    assert out.extras["n_imputed"] == 1 and not out.extras["fallback"]
    risk = {e: (0.0 if e in ("x3", "x7") else 0.5) for e in expert_x}
    assert selection_metrics(out.team, risk, 2)["hit_at_k"] == 1.0

    lonely = AppSpec("lonely", "node", 99, 42)  # no historical application with budget 99 -> fallback
    app_x[lonely.key] = app_x[target.key].clone()
    assert NearestApplicationSelector(cfg, infra).rank(lonely).extras["fallback"] is True


# --------------------------------------------------------------------------- #
# Synthetic RouterGFM environment
# --------------------------------------------------------------------------- #
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
    """Records every label read ``(app key, split)`` after loading."""

    def __init__(self, inner):
        self.inner, self.log = inner, []

    def load(self, app):
        data = self.inner.load(app)
        data.labels = _LabelLog(data.labels, app.key, self.log)
        return data


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("selection")
    cfg = tiny_cfg(
        tmp,
        targets=(TARGET,),
        history_extra=("nodeb:node", "nodec:node", "srca:node", "grapha:graph", "linka:edge"),
        budgets=(3,),
        seeds=(42, 0),
    )
    cfg.save_results.output_dir = str(tmp / "results")
    cfg.moe.routergfm.baselines.num_runs = 2
    m = cfg.moe.routergfm.baselines.metadata_mlp
    m.hidden_dim, m.epochs, m.patience = 16, 20, 5
    provider = SyntheticDataProvider()
    generate_history(cfg, provider, apps=enumerate_applications(cfg.moe.routergfm))
    return SimpleNamespace(cfg=cfg, provider=provider, tmp=tmp)


def _read_tsv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_query_guard(env):
    infra = RouterInfra(env.cfg, env.provider)
    guard = QueryGuard(infra)
    app = infra.application(TARGET, 3, 42)
    other = infra.historical_applications(app)[0]
    eid = infra.compatible_pool(app)[0]
    with pytest.raises(QueryAccessError):
        guard.query_expert_risk(other, eid)  # evaluation helpers: always
    guard.target = app
    sibling = infra.application(TARGET, 3, 0)  # same group, other seed: also guarded
    for probe in (
        lambda: guard.evaluate_outputs(app, torch.zeros(1)),
        lambda: guard.historical_mu(sibling, eid),
        lambda: guard.expert_predictions(app, [eid]),
        lambda: guard.embeddings(app, eid, "query"),
        lambda: guard.embeddings(app, eid, "diag"),
        lambda: guard.store.app_average(app, eid),
        lambda: guard.store.loss_matrix(sibling.data_key),
        lambda: guard.store.load(app.data_key, eid),
        lambda: guard.provider,
        lambda: guard._data,
    ):
        with pytest.raises(QueryAccessError):
            probe()
    assert set(guard.data(app).labels) == {"support"} and "query" in infra.data(app).labels
    assert torch.equal(guard.embeddings(app, eid, "support"), infra.embeddings(app, eid, "support"))
    assert guard.historical_mu(other, eid) == infra.historical_mu(other, eid)
    assert guard.store.expert_ids(other.data_key) == infra.store.expert_ids(other.data_key)
    assert guard.support_labels(app) is infra.support_labels(app)
    guard.target = None
    assert guard.historical_mu(app, eid) == infra.historical_mu(app, eid)


class _ProbeSelector:
    """Tries target query-side reads during rank (all must raise), then ranks the pool in order."""

    def __init__(self, cfg, infra):
        self.cfg, self.infra = cfg, infra

    def rank(self, app):
        pool = self.infra.compatible_pool(app)
        for probe in (
            lambda: self.infra.query_expert_risk(app, pool[0]),
            lambda: self.infra.historical_mu(app, pool[0]),
            lambda: self.infra.embeddings(app, pool[0], "query"),
            lambda: self.infra.store.losses(app, pool),
        ):
            with pytest.raises(QueryAccessError):
                probe()
        return SelectionOutcome.from_ranking(app, [(e, -i) for i, e in enumerate(pool)], 2)


def test_harness_guards_selectors_and_scores_after_ranking(env, monkeypatch, tmp_path):
    cfg = env.cfg.clone()
    cfg.moe.routergfm.baselines.method = "probe"
    cfg.moe.routergfm.baselines.output_dir = str(tmp_path / "baselines")
    cfg.save_results.output_dir = str(tmp_path / "results")
    monkeypatch.setitem(selection_run._SELECTORS, "probe", (__name__, "_ProbeSelector", {}))
    infra = RouterInfra(cfg, env.provider)
    assert selection_run.run_selection_baseline(cfg, infra=infra) == 0
    files = sorted((tmp_path / "baselines" / "probe").rglob("*.json"))
    assert len(files) == 2
    payload = json.loads(files[0].read_text())
    app = AppSpec.from_dict(payload["outcome"]["app"])
    store = HistoryStore(infra.paths)
    pool = infra.compatible_pool(app)
    assert payload["risk"] == pytest.approx({e: store.app_average(app, e)[0] for e in pool})
    expected = selection_metrics(pool, payload["risk"], 2)
    assert payload["metrics"]["test_hit_at_2"] == expected["hit_at_k"]
    assert payload["metrics"]["test_regret_at_2"] == pytest.approx(expected["regret_at_k"])


@pytest.mark.parametrize("method", ["logme", "nearest_application", "metadata_mlp"])
def test_run_selection_baseline_rows_and_label_access(env, tmp_path, method):
    cfg = env.cfg.clone()
    cfg.moe.routergfm.baselines.method = method
    cfg.moe.routergfm.baselines.output_dir = str(tmp_path / "baselines")
    cfg.save_results.output_dir = str(tmp_path / "results")
    provider = _LoggingProvider(env.provider)
    infra = RouterInfra(cfg, provider)
    assert selection_run.run_selection_baseline(cfg, infra=infra) == 0

    target_reads = {(key, split) for key, split in provider.log}
    if method == "logme":  # the only label reads are the targets' support labels
        assert target_reads == {(f"nodea__node__b3__s{s}", "support") for s in (42, 0)}
    else:  # label-free selectors read no labels at all
        assert target_reads == set()

    rows = _read_tsv(tmp_path / "results" / "moe_routergfm_selection.tsv")
    assert [(r["dataset"], r["budget"], r["n_apps"]) for r in rows] == [("nodea", "3", "2"), ("table9_all", "3", "2")]
    row = rows[0]
    assert row["moe.routergfm.baselines.method"] == method and row["topk"] == "2"
    assert json.loads(row["seeds"]) == [0, 42]
    assert 0.0 <= float(row["test_hit_at_2_mean"]) <= 1.0 and float(row["test_regret_at_2_mean"]) >= 0.0
    assert not any("loss" in c for c in row)
    files = sorted((tmp_path / "baselines" / method).rglob("*.json"))
    assert [f.name for f in files] == ["nodea__node__b3__s0.json", "nodea__node__b3__s42.json"]
    payload = json.loads(files[1].read_text())
    ranked = [e for e, _ in payload["outcome"]["ranking"]]
    assert sorted(ranked) == sorted(infra.compatible_pool(infra.application(TARGET, 3, 42)))
    assert payload["outcome"]["team"] == ranked[:2]

    # Reuse: no recomputation and no duplicate rows unless save_skipped.
    assert selection_run.run_selection_baseline(cfg, infra=infra) == 0
    assert len(_read_tsv(tmp_path / "results" / "moe_routergfm_selection.tsv")) == 2
    cfg.save_results.save_skipped = True
    assert selection_run.run_selection_baseline(cfg, infra=infra) == 0
    again = _read_tsv(tmp_path / "results" / "moe_routergfm_selection.tsv")
    assert len(again) == 4 and again[2]["test_hit_at_2_mean"] == row["test_hit_at_2_mean"]


def test_registries():
    assert set(BASELINE_RUNNERS) == {"metagl_u", "sagmm_pe", "meta_des", "kdem", "ppem"}
    assert BASELINE_RUNNERS["kdem"] == BASELINE_RUNNERS["ppem"] == ("src.moe.routergfm.baselines.kdem_ppem", "KDEMPPEMRunner")
    assert config_block_name("kdem") == "kdem_ppem" and config_block_name("custom") == "custom"
    assert set(selection_run._SELECTORS) == {
        "metadata_mlp", "nearest_application", "logme", "metagl", "metagl_metadata", "model_spider",
    }
    assert selection_run._SELECTORS["metagl_metadata"][2] == {"use_metadata": True}


# --------------------------------------------------------------------------- #
# Matched-pool runner
# --------------------------------------------------------------------------- #
class _FakeRunner:
    calls = []

    def __init__(self, cfg, app, infra):
        self.cfg, self.app = cfg, app
        self.best_metrics, self.best_epoch = {}, None

    def fit(self):
        type(self).calls.append((self.cfg.moe.routergfm.baselines.method, self.app.key, int(self.cfg.seed)))
        self.best_metrics = {"test_acc": 50.0 + self.app.seed, "test_risk": 0.1, "train_loss": 1.0}
        self.best_epoch = 3


class _EvaluateOnlyRunner(_FakeRunner):
    def fit(self):
        type(self).calls.append((self.cfg.moe.routergfm.baselines.method, self.app.key, int(self.cfg.seed)))

    def evaluate(self):
        return {"acc": 70.0, "risk": 0.2}


def _fake_infra():
    return SimpleNamespace(application=lambda spec, budget, seed: AppSpec(*parse_dataset_spec(spec), int(budget), int(seed)))


def _matched_cfg(env, tmp_path):
    cfg = env.cfg.clone()
    b = cfg.moe.routergfm.baselines
    b.datasets = ["photo:node", "dblp:edge"]
    b.budgets = [5, 100]
    b.num_runs = 2
    cfg.moe.routergfm.apps.seeds = [42, 0, 100]
    b.output_dir = str(tmp_path / "baselines")
    cfg.save_results.output_dir = str(tmp_path / "results")
    return cfg


def test_run_matched_baseline_rows_seeds_and_lp_once(env, tmp_path):
    cfg = _matched_cfg(env, tmp_path)
    _FakeRunner.calls = []
    assert matched_run.run_matched_baseline(cfg, "fake", _FakeRunner, infra=_fake_infra()) == 0
    assert {c[1] for c in _FakeRunner.calls} == {
        f"{d}__{l}__b{b}__s{s}" for d, l, b in (("photo", "node", 5), ("photo", "node", 100), ("dblp", "edge", 5)) for s in (42, 0)
    }
    assert all(method == "fake" and key.endswith(f"s{seed}") for method, key, seed in _FakeRunner.calls)
    rows = _read_tsv(tmp_path / "results" / "moe_fake.tsv")
    assert [json.loads(r["moe.routergfm.baselines.datasets"]) for r in rows] == [["photo:node"]] * 2 + [["dblp:edge"]]
    assert [json.loads(r["moe.routergfm.baselines.budgets"]) for r in rows] == [[5], [100], [5]]
    assert float(rows[0]["test_acc_mean"]) == pytest.approx(71.0) and json.loads(rows[0]["seeds"]) == [42, 0]
    assert json.loads(rows[0]["best_epochs"]) == [3, 3] and rows[0]["moe.routergfm.baselines.method"] == "fake"
    assert not any("loss" in c for c in rows[0])

    _FakeRunner.calls = []  # cached per-seed metrics are reused; no new rows unless save_skipped
    assert matched_run.run_matched_baseline(cfg, "fake", _FakeRunner, infra=_fake_infra()) == 0
    assert _FakeRunner.calls == [] and len(_read_tsv(tmp_path / "results" / "moe_fake.tsv")) == 3


def test_run_matched_baseline_tsv_dispatch_and_evaluate_fallback(env, tmp_path, monkeypatch):
    cfg = _matched_cfg(env, tmp_path)
    tsv = tmp_path / "tasks.tsv"
    tsv.write_text(
        "# method dataset task_level budget\n"
        "kdem photo node 5\n"
        "kdem cornell edge 5\n"
        "kdem cornell edge 100\n"
        "evalonly chameleon node 100\n"
        "logme photo node 5\n"
    )
    b = cfg.moe.routergfm.baselines
    b.run_tasks_tsv, b.tasks_tsv, b.skip_if_exists = True, str(tsv), False
    assert [t["method"] for t in matched_run.parse_baseline_tasks(str(tsv), "kdem")] == ["kdem"] * 3
    assert matched_run.baseline_tasks(cfg, "kdem") == [("photo:node", 5), ("cornell:edge", 5)]

    monkeypatch.setitem(BASELINE_RUNNERS, "kdem", (__name__, "_FakeRunner"))
    monkeypatch.setitem(BASELINE_RUNNERS, "evalonly", (__name__, "_EvaluateOnlyRunner"))
    _FakeRunner.calls, _EvaluateOnlyRunner.calls = [], []
    assert matched_run.run_matched_baseline_from_cfg(cfg, infra=_fake_infra()) == 0  # logme rows are ignored
    assert {c[0] for c in _FakeRunner.calls} == {"kdem"} and len(_FakeRunner.calls) == 4
    assert {c[1] for c in _EvaluateOnlyRunner.calls} == {"chameleon__node__b100__s42", "chameleon__node__b100__s0"}
    assert len(_read_tsv(tmp_path / "results" / "moe_kdem.tsv")) == 2
    (row,) = _read_tsv(tmp_path / "results" / "moe_evalonly.tsv")
    assert float(row["test_acc_mean"]) == 70.0 and float(row["test_risk_mean"]) == pytest.approx(0.2)
