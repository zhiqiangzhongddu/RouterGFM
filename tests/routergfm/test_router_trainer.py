"""Router training with application-masked episodes (Sec. 3.5, Alg. 1 l.5-12) on tiny synthetic history."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.moe.routergfm.common import LINK, NODE_CLS, REGRESSION, AppSpec, RouterPaths
from src.moe.routergfm.context_graph import APP, EVALUATES, EXPERT, build_context_graph, masked_edges
from src.moe.routergfm.history import app_normalizer, generate_history, instance_losses
from src.moe.routergfm.infra import RouterInfra
from src.moe.routergfm.router import huber, listmle
from src.moe.routergfm.router import trainer as trainer_mod
from src.moe.routergfm.router.trainer import (
    BUNDLE_FILE,
    LOG_FILE,
    RouterTrainer,
    _validation_groups,
    build_router_trainer,
    router_run_key,
    train_router,
)
from tests.routergfm.fixtures import SyntheticDataProvider, tiny_cfg

TARGET, BUDGET = "nodea", 3
HISTORY_EXTRA = ("srca:node", "nodeb:node", "nodec:node", "linka:edge", "linkb:edge", "srcb:graph", "rega:graph")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    cfg = tiny_cfg(tmp_path_factory.mktemp("router"), history_extra=HISTORY_EXTRA)
    rt = cfg.moe.routergfm.router
    rt.epochs, rt.patience, rt.lr = 6, 100, 1e-2
    provider = SyntheticDataProvider()
    generate_history(cfg, provider)
    return SimpleNamespace(cfg=cfg, provider=provider)


def _cfg(env, **router):
    cfg = env.cfg.clone()
    for key, value in router.items():
        setattr(cfg.moe.routergfm.router, key, value)
    return cfg


def _train_objective(trainer) -> float:
    """Eq. 11 over all training applications with every valid (x, e) pair."""
    trainer.model.eval()
    with torch.no_grad():
        terms = [trainer.episode_losses(app) for app in trainer.apps_train]
    lam = float(trainer.rt.lambda_local)
    return float(torch.stack([g + lam * l for g, l in terms]).mean())


@pytest.fixture(scope="module")
def fitted(env):
    trainer = build_router_trainer(env.cfg, TARGET, BUDGET, env.provider)
    before = {"val": trainer.evaluate()["loss"], "train": _train_objective(trainer)}
    trainer.fit()
    trainer.select_rho_tau()
    return SimpleNamespace(trainer=trainer, before=before)


def _groups(apps):
    return {a.group for a in apps}


# --------------------------------------------------------------------------- #
# Leave-one-dataset-out split
# --------------------------------------------------------------------------- #
def test_target_group_never_enters_training_validation_graph_or_archive(fitted):
    tr = fitted.trainer
    graph_apps = [n for n in tr.graph.app_nodes if isinstance(n, AppSpec)]
    for apps in (tr.apps_train, tr.apps_val, graph_apps, tr.archive.apps):
        assert TARGET not in _groups(apps)
    assert TARGET not in tr.archive.group
    assert TARGET not in {tr.graph.app_nodes[i].group for i in tr.graph.eval_app.unique().tolist()}
    # Validation: the first node-classification group, held out at the target budget only.
    assert tr.meta["val_groups"] == ["srca"] and _groups(tr.apps_val) == {"srca"}
    assert {a.budget for a in tr.apps_val} == {BUDGET}
    assert "srca" not in _groups(tr.apps_train)
    assert _groups(tr.apps_train) | {"srca"} == {"nodeb", "nodec", "linka", "linkb", "srcb", "rega", "srca"}
    assert {a.budget for a in tr.apps_train} == {3, 6}  # every budget of the training groups
    assert set(a.key for a in graph_apps) == set(a.key for a in tr.archive.apps)
    assert tr.meta["run_key"] == router_run_key(TARGET, BUDGET, 42)


def test_validation_group_choice():
    apps = [
        AppSpec("cora", "edge", 5, 0),
        AppSpec("pubmed", "node", 5, 0),
        AppSpec("photo", "node", 5, 0),
        AppSpec("qm9", "graph", 5, 0),
    ]
    fam = dict(zip([a.key for a in apps], [LINK, NODE_CLS, NODE_CLS, REGRESSION]))
    cfg = lambda num=1, names=(): SimpleNamespace(num_val_datasets=num, val_datasets=list(names))  # noqa: E731
    assert _validation_groups(cfg(1), "tgt", apps, fam, {NODE_CLS}) == ["pubmed"]
    assert _validation_groups(cfg(2), "tgt", apps, fam, {NODE_CLS}) == ["pubmed", "photo"]
    assert _validation_groups(cfg(1), "tgt", apps, fam, {REGRESSION}) == ["qm9"]
    assert _validation_groups(cfg(1), "tgt", apps, fam, set()) == ["cora"]
    assert _validation_groups(cfg(1, ["Photo:node", "tgt"]), "tgt", apps, fam, {LINK}) == ["photo"]
    with pytest.raises(ValueError):
        _validation_groups(cfg(1, ["unknown"]), "tgt", apps, fam, {NODE_CLS})
    with pytest.raises(ValueError):
        _validation_groups(cfg(4), "tgt", apps, fam, {NODE_CLS})  # nothing left for training


# --------------------------------------------------------------------------- #
# Episodes and objectives
# --------------------------------------------------------------------------- #
def test_episodes_hide_own_group_edges_and_records(env, monkeypatch):
    tr = build_router_trainer(_cfg(env, epochs=1), TARGET, BUDGET, env.provider)
    full_eval = tr.graph.edge_index[EVALUATES]
    calls = []
    encode, search = tr.model.encode, trainer_mod.search

    def spy_encode(x, edge_index, edge_attr):
        calls.append(("encode", edge_index[EVALUATES].clone()))
        return encode(x, edge_index, edge_attr)

    def spy_search(query, keys, record_app, allowed, *args, **kwargs):
        calls.append(("search", allowed.clone()))
        return search(query, keys, record_app, allowed, *args, **kwargs)

    tr.model.encode = spy_encode
    monkeypatch.setattr(trainer_mod, "search", spy_search)
    tr.fit()

    num_episodes = len(tr.apps_train) + len(tr.apps_val)
    assert sum(kind == "encode" for kind, _ in calls) == num_episodes
    hidden, retrieved = None, 0
    for kind, value in calls:
        if kind == "encode":
            gone = set(full_eval[0].tolist()) - set(value[0].tolist())
            groups = {tr.graph.app_nodes[i].group for i in gone}
            assert len(groups) == 1  # exactly the episode's group, all of its applications
            hidden = groups.pop()
            assert not any(tr.graph.app_nodes[i].group == hidden for i in value[0].tolist())
        else:
            assert not bool((value & tr.archive.records_of_groups({hidden})).any())
            retrieved += int(value.any())
    assert retrieved > 0  # e.g. the only regression group has no compatible evidence, others do


def test_local_loss_is_exact_eq10_and_global_loss_eq9(env):
    tr = build_router_trainer(_cfg(env, rho_train=0.0), TARGET, BUDGET, env.provider)
    tr.model.eval()
    rt = tr.rt
    for app in tr.apps_train[::3] + tr.apps_val:
        with torch.no_grad():
            glob, loc = tr.episode_losses(app)
        nodes, mu = tr.mu_hat(app)
        ids = [tr.graph.expert_ids[e] for e in nodes.tolist()]
        mu_bar = torch.tensor([tr.store.app_average(app, e)[0] for e in ids])
        expected_glob = huber(mu, mu_bar, rt.huber_delta) + rt.lambda_rank * listmle(-mu, mu_bar)
        assert float(glob) == pytest.approx(float(expected_glob), rel=1e-5)
        losses = tr.store.losses(app, ids)  # [|D_b|, K], NaN invalid
        sq = (mu.unsqueeze(0) - losses).pow(2)
        per_expert = torch.nansum(sq, 0) / torch.isfinite(losses).sum(0)
        assert float(loc) == pytest.approx(float(per_expert.mean()), rel=1e-5)  # rho = 0: r_hat = mu_hat


def test_local_loss_trains_keys_through_kernel_weights(env):
    tr = build_router_trainer(_cfg(env, rho_train=1.0), TARGET, BUDGET, env.provider)
    tr.model.train()
    _, loc = tr.episode_losses(tr.apps_train[0])
    loc.backward()
    grads = [p.grad for p in tr.model.key_net.parameters()]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)
    assert tr.model.scorer[0].weight.grad is not None  # r_hat builds on mu_hat


def test_fit_reduces_losses_and_restores_best_state(fitted):
    tr, log = fitted.trainer, fitted.trainer.log
    assert [e["epoch"] for e in log["epochs"]] == list(range(1, 7))
    val = [e["val_loss"] for e in log["epochs"]]
    assert log["best_epoch"] == 1 + min(range(len(val)), key=val.__getitem__)
    assert tr.evaluate()["loss"] == pytest.approx(log["best_val_loss"], rel=1e-6)
    assert log["best_val_loss"] < fitted.before["val"]
    assert _train_objective(tr) < fitted.before["train"]
    assert all(torch.isfinite(torch.tensor([e["train_loss"], e["train_glob"], e["train_loc"]])).all() for e in log["epochs"])


def test_early_stopping(env):
    tr = build_router_trainer(_cfg(env, epochs=50, patience=1, lr=0.2), TARGET, BUDGET, env.provider)
    log = tr.fit()
    assert len(log["epochs"]) < 50
    assert len(log["epochs"]) == log["best_epoch"] + 1


# --------------------------------------------------------------------------- #
# rho / tau
# --------------------------------------------------------------------------- #
def test_rho_tau_grid_selection_uses_stored_predictions_and_mixture_loss(env, fitted):
    tr = fitted.trainer
    grid = tr.log["rho_tau_grid"]
    rt = tr.rt
    assert len(grid) == len(rt.rho_grid) * len(rt.tau_grid)
    best = min(grid, key=lambda row: row["risk"])
    assert (tr.rho, tr.tau) == (best["rho"], best["tau"])
    # rho = 0 is application-level (global) weighting of the top-K team's stored D_v predictions.
    for tau in rt.tau_grid:
        risks = []
        for app in tr.apps_val:
            nodes, mu = tr.mu_hat(app)
            order = torch.argsort(mu, stable=True)[: rt.topk]
            ids = [tr.graph.expert_ids[e] for e in nodes[order].tolist()]
            alpha = torch.softmax(-mu[order] / tau, dim=0)
            mixed = (alpha.view(1, -1, 1) * tr.store.pred_matrix(app.data_key, ids)).sum(1)
            data = env.provider.load(app)
            loss = instance_losses(env.cfg, mixed, data.labels["diag"], data.task_family, app_normalizer(env.cfg, data))
            risks.append(loss[torch.isfinite(loss)].mean())
        row = next(r for r in grid if r["rho"] == 0.0 and r["tau"] == tau)
        assert row["risk"] == pytest.approx(float(torch.stack(risks).mean()), rel=1e-5)


def test_configured_rho_tau_skip_selection(env):
    tr = build_router_trainer(_cfg(env, rho=0.5, tau=0.1), TARGET, BUDGET, env.provider)
    tr.provider = None  # no labels needed when both are configured
    assert tr.select_rho_tau() == (0.5, 0.1)
    tr = build_router_trainer(_cfg(env, rho=0.5), TARGET, BUDGET, env.provider)
    tr.provider = None
    with pytest.raises(ValueError):
        tr.select_rho_tau()  # tau still needs the validation grid
    with pytest.raises(RuntimeError):
        tr.save(env.cfg.moe.routergfm.output_root)


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #
def test_bundle_round_trip_reproduces_mu_hat(env, fitted, tmp_path):
    tr = fitted.trainer
    directory = tr.save(tmp_path / "router")
    assert (directory / BUNDLE_FILE).is_file() and (directory / LOG_FILE).is_file()
    bundle = RouterTrainer.load(directory, env.cfg, device="cpu")
    assert (bundle.rho, bundle.tau) == (tr.rho, tr.tau)
    assert bundle.catalog_ids == [s.expert_id for s in tr.catalog]
    assert bundle.train_apps == tr.apps_train and bundle.val_apps == tr.apps_val
    assert bundle.graph_apps == [n for n in tr.graph.app_nodes if isinstance(n, AppSpec)]
    assert bundle.archive_apps == tr.archive.apps
    assert bundle.meta == tr.meta and bundle.log == tr.log
    assert bundle.router_cfg["topk"] == tr.rt.topk and not bundle.model.training
    z = torch.randn(7, tr.desc_dim) * 3
    assert torch.equal(bundle.standardizer.transform(z), tr.standardizer.transform(z))

    # Deployment rebuilds H from the bundle's applications and numeric statistics.
    infra = RouterInfra(env.cfg, env.provider)
    stats = {a.key: infra.data(a).stats for a in bundle.graph_apps}
    families = {a.key: infra.task_family(a) for a in bundle.graph_apps}
    graph = build_context_graph(
        env.cfg, infra.catalog, bundle.graph_apps, infra.store, infra.text_encoder, stats,
        families=families, numeric_stats=bundle.numeric_stats,
    )
    for t in tr.graph.x:
        assert torch.equal(graph.x[t], tr.graph.x[t].cpu())
    for app in tr.apps_val + tr.apps_train[:2]:
        nodes, mu = tr.mu_hat(app)
        with torch.no_grad():
            h = bundle.model.encode(graph.x, *masked_edges(graph, {app.group}))
            again = bundle.model.score(h[APP][graph.app_index[app.key]].expand(nodes.numel(), -1), h[EXPERT][nodes])
        assert torch.allclose(again, mu.cpu(), atol=1e-6)


def test_train_router_writes_bundle_and_reuses_it(env, monkeypatch):
    cfg = _cfg(env, epochs=2)
    path = train_router(cfg, f"{TARGET}:node", BUDGET, env.provider)
    assert path == RouterPaths.from_cfg(cfg).router_dir(router_run_key(TARGET, BUDGET, 42))
    bundle = RouterTrainer.load(path, cfg)
    assert bundle.meta["target_group"] == TARGET and bundle.meta["budget"] == BUDGET
    for apps in (bundle.train_apps, bundle.val_apps, bundle.graph_apps, bundle.archive_apps):
        assert apps and TARGET not in _groups(apps)
    assert bundle.rho in cfg.moe.routergfm.router.rho_grid and bundle.tau in cfg.moe.routergfm.router.tau_grid
    assert len(bundle.log["epochs"]) == 2

    def _fail(*args, **kwargs):
        raise AssertionError("an existing router was retrained")

    monkeypatch.setattr(trainer_mod, "build_router_trainer", _fail)
    stamp = (path / BUNDLE_FILE).stat().st_mtime_ns
    assert train_router(cfg, TARGET, BUDGET, env.provider) == path
    assert (path / BUNDLE_FILE).stat().st_mtime_ns == stamp
