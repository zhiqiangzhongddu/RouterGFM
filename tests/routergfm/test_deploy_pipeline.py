"""Deployment (Alg. 1 l.13-21), stage dispatch, and benchmark rows on tiny synthetic history."""

from __future__ import annotations

import csv
import dataclasses
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.config._moe_routergfm import ROUTERGFM_TARGET_DATASETS
from src.moe.routergfm import run_routergfm
from src.moe.routergfm import run as run_mod
from src.moe.routergfm.common import AppSpec, RouterPaths
from src.moe.routergfm.context_graph import EVALUATES
from src.moe.routergfm.deploy import (
    evaluate_integration,
    integrate_application,
    prepare_router,
    router_seed,
    score_pool,
)
from src.moe.routergfm.diagnostics import hit_at_k, regret_at_k, routing_risks
from src.moe.routergfm.infra import RouterInfra
from src.moe.routergfm.integration import RULES
from src.moe.routergfm.router.trainer import BUNDLE_FILE, router_run_key
from tests.routergfm.fixtures import SyntheticDataProvider, tiny_cfg

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("nodea:node", "regc:graph")
HISTORY_EXTRA = ("srca:node", "nodeb:node", "nodec:node", "linka:edge", "srcb:graph", "rega:graph")


def _stage(env, task, **rg):
    cfg = env.cfg.clone()
    cfg.moe.routergfm.task = task
    for dotted, value in rg.items():
        block, key = dotted.split("__")
        setattr(getattr(cfg.moe.routergfm, block), key, value)
    return cfg


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("deploy")
    cfg = tiny_cfg(tmp, targets=TARGETS, history_extra=HISTORY_EXTRA, seeds=(42, 0))
    cfg.save_results.output_dir = str(tmp / "results")
    cfg.moe.routergfm.benchmark.num_runs = 2
    tasks = tmp / "tasks.tsv"
    tasks.write_text("# dataset\ttask_level\tbudget\nnodea\tnode\t3\nregc\tgraph\t3\n", encoding="utf-8")
    cfg.moe.routergfm.benchmark.tasks_tsv = str(tasks)
    env = SimpleNamespace(cfg=cfg, provider=SyntheticDataProvider(), tmp=tmp, paths=RouterPaths.from_cfg(cfg))
    assert run_routergfm(_stage(env, "history"), provider=env.provider) == 0
    return env


@pytest.fixture(scope="module")
def deployed(env):
    infra = RouterInfra(env.cfg, env.provider)
    app = infra.application("nodea:node", 3, 42)
    return SimpleNamespace(infra=infra, app=app, router=prepare_router(env.cfg, app, infra))


def _with_rho(router, rho):
    return dataclasses.replace(router, bundle=dataclasses.replace(router.bundle, rho=rho))


# --------------------------------------------------------------------------- #
# End to end through run_routergfm
# --------------------------------------------------------------------------- #
def test_history_router_deploy_benchmark_stages(env):
    assert run_routergfm(_stage(env, "router", benchmark__run_tasks_tsv=True), provider=env.provider) == 0
    for group in ("nodea", "regc"):
        assert (env.paths.router_dir(router_run_key(group, 3, 42)) / BUNDLE_FILE).is_file()

    cfg = _stage(env, "deploy", deploy__target="regc:graph", deploy__budget=3, deploy__seed=0)
    assert run_routergfm(cfg, provider=env.provider) == 0
    assert (env.paths.deploy_dir(AppSpec("regc", "graph", 3, 0).key) / "deploy.json").is_file()

    assert run_routergfm(_stage(env, "benchmark", benchmark__run_tasks_tsv=True), provider=env.provider) == 0
    with open(Path(env.cfg.save_results.output_dir) / "moe_routergfm.tsv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    rules = list(env.cfg.moe.routergfm.integration.rules)
    methods = {"routergfm", "routergfm_g"} | {f"fixed_team:{r}" for r in rules}
    for dataset, metric in (("nodea", "test_acc"), ("regc", "test_mae")):
        mine = {r["method"]: r for r in rows if r["dataset"] == dataset}
        assert set(mine) == methods
        for row in mine.values():
            assert row["metric"] == metric and row["budget"] == "3" and row["n_runs"] == "2"
            assert row["seeds"] == "[42, 0]"
            for column in (f"{metric}_mean", f"{metric}_std", "test_risk_mean", "test_worst_cell_risk_mean",
                           "test_winner_agreement_mean", "test_hit_at_2_mean", "test_regret_at_2_mean"):
                assert math.isfinite(float(row[column])), (dataset, row["method"], column)
        assert mine["routergfm_g"]["test_risk_mean"] == mine["fixed_team:global"]["test_risk_mean"]
        assert mine["routergfm"]["test_risk_mean"] == mine["fixed_team:routergfm"]["test_risk_mean"]


def test_task_resolution_and_dispatch(env, monkeypatch):
    cfg = env.cfg.clone()
    rg = cfg.moe.routergfm
    assert run_mod.routergfm_tasks(cfg) == [("nodea:node", 3)]
    rg.deploy.target = "all"
    assert run_mod.routergfm_tasks(cfg) == [(t, b) for t in TARGETS for b in (3, 6)]
    rg.benchmark.run_tasks_tsv = True
    assert run_mod.routergfm_tasks(cfg) == [("nodea:node", 3), ("regc:graph", 3)]
    assert run_mod.benchmark_seeds(cfg) == [42, 0]
    rg.benchmark.num_runs = 3
    with pytest.raises(ValueError):
        run_mod.benchmark_seeds(cfg)
    app = AppSpec("nodea", "node", 3, 0)
    assert router_seed(cfg, app) == 42
    rg.router.per_seed = True
    assert router_seed(cfg, app) == 0

    rg.task = "no_such_stage"
    assert run_routergfm(cfg) == 1
    with pytest.raises(RuntimeError, match="not available"):
        run_mod._external("src.moe.routergfm.no_such_module", "run", "analysis")
    calls = []
    monkeypatch.setattr(run_mod, "_external", lambda module, attr, what: lambda c, **kw: calls.append(module) or 0)
    for task in ("analysis", "selection_baseline", "matched_baseline"):
        rg.task = task
        assert run_routergfm(cfg) == 0
    assert calls == [
        "src.moe.routergfm.analysis",
        "src.moe.routergfm.baselines.selection.run",
        "src.moe.routergfm.baselines.run",
    ]


def test_slurm_task_tables_cover_the_paper_targets():
    expected = sorted((spec, b) for spec in ROUTERGFM_TARGET_DATASETS for b in (5, 100))
    for name in ("moe.routergfm.tsv", "moe.routergfm.router.tsv"):
        assert sorted(run_mod.parse_benchmark_tasks(str(ROOT / "slurm" / name))) == expected, name


# --------------------------------------------------------------------------- #
# Deployment mechanics
# --------------------------------------------------------------------------- #
def test_target_is_inserted_from_metadata_only_and_team_is_topk(env, deployed):
    infra, app, router = deployed.infra, deployed.app, deployed.router
    assert app.group not in {a.group for a in router.bundle.graph_apps + router.bundle.archive_apps}
    pool, mu, graph = score_pool(env.cfg, app, infra, router)
    node = graph.app_index[app.key]
    assert isinstance(graph.app_nodes[node], AppSpec)
    assert node not in graph.eval_app.tolist() and node not in graph.edge_index[EVALUATES][0].tolist()
    assert app.key not in router.graph.app_index  # the router's H is left untouched
    assert pool == infra.compatible_pool(app) and mu.shape == (len(pool),)

    integ = integrate_application(env.cfg, app, infra, router, ["global"])
    k = int(router.bundle.router_cfg["topk"])
    assert integ.team == [pool[i] for i in torch.argsort(mu, stable=True)[:k]]
    assert torch.allclose(integ.mu_hat, torch.sort(mu).values[:k])
    data = infra.data(app)
    assert integ.preds.shape == (data.query_pos.numel(), k, data.num_classes)

    fixed = integrate_application(env.cfg, app, infra, router, ["uniform"], team_override=pool[:3])
    assert fixed.team == pool[:3] and fixed.alpha["uniform"].shape == (data.query_pos.numel(), 3)
    with pytest.raises(ValueError):
        integrate_application(env.cfg, app, infra, router, ["uniform"], team_override=["not_an_expert"])
    with pytest.raises(ValueError):
        integrate_application(env.cfg, app, infra, router, ["no_such_rule"])


def test_prepare_router_rejects_a_router_that_saw_the_target_group(env, deployed):
    bundle = deployed.router.bundle
    leaky = dataclasses.replace(bundle, graph_apps=bundle.graph_apps + [AppSpec("nodea", "node", 6, 42)])
    with pytest.raises(ValueError, match="leave-one-dataset-out"):
        prepare_router(env.cfg, deployed.app, deployed.infra, bundle=leaky)


def test_rho_zero_recovers_routergfm_g_and_uniform_differs(env, deployed):
    infra, app = deployed.infra, deployed.app
    integ = integrate_application(env.cfg, app, infra, _with_rho(deployed.router, 0.0), RULES)
    assert integ.num_records > 0  # compatible evidence exists, yet rho = 0 transfers none of it
    assert torch.allclose(integ.alpha["routergfm"], integ.alpha["global"], atol=1e-7)
    assert torch.allclose(integ.mixed["routergfm"], integ.mixed["global"], atol=1e-6)
    expected = torch.softmax(-integ.mu_hat / integ.tau, 0)
    assert torch.allclose(integ.alpha["global"], expected.expand_as(integ.alpha["global"]), atol=1e-7)
    assert not torch.allclose(integ.alpha["uniform"], integ.alpha["global"])
    for rule, alpha in integ.alpha.items():
        assert bool(torch.isfinite(alpha).all()) and torch.allclose(alpha.sum(-1), torch.ones(alpha.size(0))), rule

    local = integrate_application(env.cfg, app, infra, _with_rho(deployed.router, 1.0), ["routergfm", "global"])
    assert not torch.allclose(local.alpha["routergfm"], local.alpha["global"])  # context-dependent weights
    assert local.alpha["routergfm"].std(dim=0).max() > 0


class _GuardedLabels(dict):
    def __getitem__(self, key):
        if key in ("query", "diag"):
            raise AssertionError(f"target {key} labels were read")
        return super().__getitem__(key)


class _GuardedProvider:
    def __init__(self, base, key):
        self.base, self.key = base, key

    def load(self, app):
        data = self.base.load(app)
        if app.key == self.key:
            data.labels = _GuardedLabels(data.labels)
        return data


class _GuardedStore:
    """History store that refuses every read of the target's data key."""

    def __init__(self, store, data_key):
        self._store, self._key = store, data_key

    def __getattr__(self, name):
        attr = getattr(self._store, name)

        def guarded(*args, **kwargs):
            for arg in args:
                if (arg.data_key if isinstance(arg, AppSpec) else arg) == self._key:
                    raise AssertionError(f"target history read via {name}")
            return attr(*args, **kwargs)

        return guarded


def test_weights_never_read_target_query_labels_or_history(env, deployed):
    app = deployed.app
    shutil.rmtree(env.paths.root / "predictions" / app.data_key, ignore_errors=True)  # refit heads under the guard
    infra = RouterInfra(env.cfg, _GuardedProvider(env.provider, app.key))
    infra.store = _GuardedStore(infra.store, app.data_key)
    guarded = integrate_application(env.cfg, app, infra, deployed.router, RULES)
    plain = integrate_application(env.cfg, app, deployed.infra, deployed.router, RULES)
    assert guarded.team == plain.team
    for rule in RULES:
        assert torch.equal(guarded.alpha[rule], plain.alpha[rule]), rule
    with pytest.raises(AssertionError, match="read"):
        evaluate_integration(env.cfg, app, infra, guarded)  # evaluation is the only reader


def test_local_rules_use_the_bundles_selected_bandwidth(env, deployed):
    """Retrieval weights use the validation-selected h stored in the bundle, not ``router_cfg['bandwidth']``."""
    infra, app, router = deployed.infra, deployed.app, deployed.router
    assert router.bundle.bandwidth in env.cfg.moe.routergfm.router.bandwidth_grid
    alphas = []
    for h in (0.05, 5.0):
        stale_cfg = dict(router.bundle.router_cfg, bandwidth=1.0)
        bundle = dataclasses.replace(router.bundle, rho=1.0, bandwidth=h, router_cfg=stale_cfg)
        integ = integrate_application(env.cfg, app, infra, dataclasses.replace(router, bundle=bundle), ["routergfm"])
        assert integ.context.evidence.bandwidth == h and integ.bandwidth == h
        alphas.append(integ.alpha["routergfm"])
    assert not torch.allclose(*alphas)


def test_evaluation_matches_infra_metrics_and_target_history(env, deployed):
    infra, app = deployed.infra, deployed.app
    integ = integrate_application(env.cfg, app, infra, deployed.router, RULES)
    result = evaluate_integration(env.cfg, app, infra, integ)
    assert set(result["rules"]) == set(RULES) and result["metric"] == "acc"
    for rule in RULES:
        ref, got = infra.evaluate_outputs(app, integ.mixed[rule]), result["rules"][rule]
        assert got["acc"] == pytest.approx(ref["acc"]) and got["risk"] == pytest.approx(ref["risk"], rel=1e-5)
        for key in ("acc", "risk", "worst_cell_risk", "winner_agreement"):
            assert math.isfinite(got[key]), (rule, key)
        assert got["worst_cell_risk"] >= got["risk"] - 1e-6 and 0.0 <= got["winner_agreement"] <= 1.0

    sel = result["selection"]
    mu_true = {e: infra.historical_mu(app, e)[0] for e in integ.pool}
    assert sel["k"] == len(integ.team)
    assert sel["hit_at_k"] == hit_at_k(integ.team, mu_true)
    assert sel["regret_at_k"] == pytest.approx(regret_at_k(integ.team, mu_true))
    assert 0.0 <= sel["winner_coverage"] <= 1.0 and sel["specialization_index"] >= 0.0

    # The deployed heads reproduce the recorded D_a losses of the target's history.
    data = infra.data(app)
    rows = torch.searchsorted(data.query_pos, data.diag_pos)
    losses = routing_risks(integ.preds[rows], data.labels["diag"], data.task_family)
    assert torch.allclose(losses, infra.store.losses(app, integ.team), atol=2e-3, equal_nan=True)
    assert result["pool_mu_hat"] == pytest.approx(dict(zip(integ.pool, integ.pool_mu_hat.tolist())))
