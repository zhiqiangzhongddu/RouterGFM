"""App. D analyses (``analysis.py``) and shift-tagged applications on tiny synthetic history."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from src.moe.routergfm import analysis as analysis_mod
from src.moe.routergfm import run_routergfm
from src.moe.routergfm.analysis import (
    ANALYSIS_WORKFLOW,
    INSERTION_CONDITIONS,
    PERTURBATION_CONDITIONS,
    analysis_tasks,
    architecture_holdout,
    calibrate,
    calibration_candidates,
    configuration_holdout,
    holdout_router,
    insert_experts,
    parse_analysis_tasks,
    run_analysis,
    with_known_target,
)
from src.moe.routergfm.applications import RealDataProvider, instance_set_key
from src.moe.routergfm.common import AppSpec, RouterPaths
from src.moe.routergfm.deploy import score_pool
from src.moe.routergfm.infra import RouterInfra
from src.moe.routergfm.router import trainer as trainer_mod
from src.moe.routergfm.router.trainer import RouterTrainer
from tests.routergfm.fixtures import FEATURE_DIM, SyntheticDataProvider, tiny_cfg

ROOT = Path(__file__).resolve().parents[2]
TARGET = "nodea:node"
HISTORY_EXTRA = ("srca:node", "nodeb:node", "nodec:node", "linka:edge", "srcb:graph", "grapha:graph")
ROW_KEYS = ("kind", "condition", "dataset", "task_level", "budget", "seed")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("analysis")
    cfg = tiny_cfg(tmp, targets=(TARGET,), history_extra=HISTORY_EXTRA, budgets=(3,), seeds=(42, 0))
    cfg.save_results.output_dir = str(tmp / "results")
    rg = cfg.moe.routergfm
    rg.analysis.team_sizes = [1, 2, 3, 16]
    rg.analysis.calibration_apps = [0, 1, 2]
    rg.analysis.holdout_architecture = "gin"
    m = rg.baselines.metagl
    m.epochs, m.patience, m.knn_k, m.rf_n_estimators, m.graph_sample_max = 3, 3, 3, 5, 10
    provider = SyntheticDataProvider()
    history = cfg.clone()
    history.moe.routergfm.task = "history"
    assert run_routergfm(history, provider=provider) == 0
    return SimpleNamespace(cfg=cfg, provider=provider, tmp=tmp, infra=RouterInfra(cfg, provider))


def _kind_cfg(env, kind, **analysis):
    cfg = env.cfg.clone()
    cfg.moe.routergfm.task = "analysis"
    cfg.moe.routergfm.analysis.kind = kind
    for key, value in analysis.items():
        setattr(cfg.moe.routergfm.analysis, key, value)
    return cfg


def _rows(env, kind):
    with open(Path(env.cfg.save_results.output_dir) / f"{ANALYSIS_WORKFLOW}.tsv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["kind"] == kind]
    assert rows and all(all(r[k] != "" for k in ROW_KEYS) for r in rows)
    assert not any("loss" in column for column in rows[0])
    return rows


def _finite(row, *columns):
    for column in columns:
        assert math.isfinite(float(row[column])), (row["condition"], column, row[column])


# --------------------------------------------------------------------------- #
# Tasks and shift-tagged applications (no history needed)
# --------------------------------------------------------------------------- #
def test_parse_tasks_and_slurm_table(tmp_path):
    tsv = tmp_path / "rows.tsv"
    tsv.write_text(
        "# kind\tdataset\ttask_level\tbudget\n"
        "team_size\tnodea\tnode\t3\nspecialization\tall\t-\t3\nteam_size\tnodea\tnode\t3\nshift\tnodea\tnode\tx\n",
        encoding="utf-8",
    )
    assert parse_analysis_tasks(str(tsv), "shift") == [("team_size", "nodea:node", 3), ("specialization", "all", 3)]
    no_kind = tmp_path / "plain.tsv"
    no_kind.write_text("# dataset\ttask_level\tbudget\nnodea\tnode\t5\n", encoding="utf-8")
    assert parse_analysis_tasks(str(no_kind), "shift") == [("shift", "nodea:node", 5)]

    cfg = tiny_cfg(tmp_path, targets=("nodea:node", "regc:graph"), write_checkpoints=False)
    cfg.moe.routergfm.benchmark.run_tasks_tsv = True
    cfg.moe.routergfm.benchmark.tasks_tsv = str(tsv)
    tasks = analysis_tasks(cfg)
    assert dict(tasks) == {"team_size": [("nodea:node", 3)], "specialization": [("nodea:node", 3), ("regc:graph", 3)]}
    tsv.write_text("# kind\tdataset\ttask_level\tbudget\nbogus\tnodea\tnode\t3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bogus"):
        analysis_tasks(cfg)
    cfg.moe.routergfm.benchmark.run_tasks_tsv = False
    cfg.moe.routergfm.analysis.kind = "bogus"
    assert run_analysis(cfg) == 1

    rows = parse_analysis_tasks(str(ROOT / "slurm" / "moe.routergfm.analysis.tsv"), "team_size")
    kinds = {k for k, _, _ in rows}
    assert kinds == set(analysis_mod.KINDS) and ("specialization", "all", 5) in rows
    assert not any(k == "shift" and spec.endswith(":edge") for k, spec, _ in rows)  # no LP shift splits


def test_app_spec_split_root_tag():
    app = AppSpec("photo", "node", 5, 42)
    assert app.key == "photo__node__b5__s42" and app.data_key == "photo__node__fewshot5-0-100__s42"
    assert "split_root_tag" not in app.to_dict() and AppSpec.from_dict(app.to_dict()) == app
    shifted = AppSpec("photo", "node", 5, 42, split_root_tag="feature")
    assert shifted.key == app.key + "__feature" and shifted.data_key == app.data_key + "__feature"
    assert shifted.group == app.group and shifted.split == app.split
    assert AppSpec.from_dict(json.loads(json.dumps(shifted.to_dict()))) == shifted


def _fake_node_dataset():
    from src.data_loader.induced_graphs import InducedGraphDataset

    graphs, tags = [], []
    for i in range(24):
        g = Data(x=torch.randn(4, FEATURE_DIM), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
        g.y, g.base_node_id, g.target_node_index = torch.tensor(i % 3), i, torch.tensor([1])
        graphs.append(g)
        tags.append("train" if i < 6 else ("val" if i < 16 else "test"))  # shift files park unused items in val
    return InducedGraphDataset(graphs, base_num_nodes=50, base_num_edges=200,
                               base_info={"name": "cora", "domain": "citation"}, split_tags=tags)


def test_real_provider_reads_the_tagged_shift_root(tmp_path, monkeypatch):
    import src.data_loader as data_loader
    import src.data_loader.shift_splits as shift_splits
    from src.config import cfg as base_cfg

    created, verified = [], []
    monkeypatch.setattr(data_loader, "create_dataset", lambda **kw: created.append(kw) or _fake_node_dataset())
    monkeypatch.setattr(shift_splits, "verify_shift_root", lambda root, entries: verified.append((root, entries)))
    cfg = base_cfg.clone()
    rg = cfg.moe.routergfm
    rg.output_root = str(tmp_path / "out")
    rg.apps.max_diagnostic = 4
    rg.apps.data.split_root = str(tmp_path / "splits")
    rg.analysis.shift_root = str(tmp_path / "splits_shift")
    standard = AppSpec("cora", "node", 2, 42)
    shifted = AppSpec("cora", "node", 2, 42, split_root_tag="structural")

    with pytest.raises(ValueError, match="validation items"):
        RealDataProvider(cfg).load(standard)
    assert created[-1]["split_root"] == str(tmp_path / "splits") and not verified

    data = RealDataProvider(cfg).load(shifted)
    root = str(tmp_path / "splits_shift" / "structural")
    assert created[-1]["split_root"] == root
    assert verified == [(root, [("cora", "node", 42, shifted.split)])]
    assert data.support_pos.tolist() == list(range(6)) and data.query_pos.tolist() == list(range(16, 24))
    paths = RouterPaths.from_cfg(cfg)
    assert paths.data_meta_file(shifted.data_key).is_file() and not paths.data_meta_file(standard.data_key).exists()


# --------------------------------------------------------------------------- #
# Analyses on the synthetic setup
# --------------------------------------------------------------------------- #
def test_team_size(env):
    assert run_analysis(_kind_cfg(env, "team_size"), infra=env.infra) == 0
    rows = _rows(env, "team_size")
    assert [r["condition"] for r in rows] == ["K1", "K2", "K3"]  # K=16 exceeds |E_a| = 8
    for r in rows:
        _finite(r, "test_risk", "test_risk_reduction", "test_winner_coverage", "test_expert_contributions")
    assert float(rows[0]["test_risk_reduction"]) == 0.0 and rows[2]["test_expert_contributions"] == "3.0"
    coverage = [float(r["test_winner_coverage"]) for r in rows]
    assert coverage == sorted(coverage)  # nested teams cover at least as many cells
    payload = json.loads((RouterPaths.from_cfg(env.cfg).analysis_dir("team_size") / "nodea__node__b3.json").read_text())
    assert len(payload["rows"]) == 3 and len(payload["details"]["nodea__node__b3__s42"]["ranking"]) == 8


def test_archive_reliability(env):
    assert run_analysis(_kind_cfg(env, "archive_reliability"), infra=env.infra) == 0
    rows = {r["condition"]: r for r in _rows(env, "archive_reliability")}
    assert set(rows) == set(PERTURBATION_CONDITIONS)
    for r in rows.values():
        _finite(r, "test_global_risk", "test_local_rho1_risk", "test_local_validated_risk", "test_validated_rho")
    globals_ = {r["test_global_risk"] for r in rows.values()}
    assert len(globals_) == 1  # global weights never read the archive
    assert float(rows["half_cells"]["test_num_records"]) < float(rows["matched"]["test_num_records"])
    assert rows["reversed"]["test_local_rho1_risk"] != rows["matched"]["test_local_rho1_risk"]


def test_specialization_buckets(env):
    cfg = _kind_cfg(env, "specialization")
    cfg.moe.routergfm.benchmark.num_runs = 2
    assert run_analysis(cfg, infra=env.infra) == 0
    rows = _rows(env, "specialization")
    apps = [r for r in rows if r["dataset"] == "nodea"]
    summary = [r for r in rows if r["dataset"] == "all"]
    assert sorted(r["seed"] for r in apps) == ["0", "42"]
    indices = {r["condition"]: float(r["test_specialization_index"]) for r in apps}
    assert set(indices) == {"low", "medium"} and indices["low"] <= indices["medium"]
    assert [r["condition"] for r in summary] == ["low", "medium"] and all(r["seed"] == "all" for r in summary)
    for r in rows:
        _finite(r, "test_specialization_index", "test_global_risk", "test_local_risk", "test_risk_difference")


def test_insertion_holds_out_experts_from_training(env, monkeypatch):
    captured = []

    class Spy(RouterTrainer):
        def __init__(self, cfg, catalog, apps_train, apps_val, store, graph, archive, *args, **kwargs):
            captured.append(SimpleNamespace(catalog=list(catalog), graph=graph, archive=archive, meta=kwargs["meta"]))
            super().__init__(cfg, catalog, apps_train, apps_val, store, graph, archive, *args, **kwargs)

    monkeypatch.setattr(trainer_mod, "RouterTrainer", Spy)
    cfg = _kind_cfg(env, "insertion")
    cfg.moe.routergfm.router.skip_if_exists = False  # retrain the held-out routers under the spy
    infra = env.infra
    assert run_analysis(cfg, infra=infra) == 0
    captured = [spy for spy in captured if spy.meta.get("hidden_experts")]  # not the standard router

    app = infra.application(TARGET, 3, 42)
    arch_hidden = architecture_holdout(cfg, infra)
    config_hidden = configuration_holdout(cfg, infra, app, 0)
    assert len(arch_hidden) == 4 and len(config_hidden) == 2  # round(0.2 * |E_a| = 8)
    assert {infra.catalog[infra.expert_index[e]].architecture for e in arch_hidden} == {"gin"}
    assert len(captured) == 2
    for spy in captured:
        hidden = set(spy.meta["hidden_experts"])
        ids = [s.expert_id for s in spy.catalog]
        assert hidden and not hidden & set(ids) and not hidden & set(spy.graph.expert_ids)
        trained = {spy.graph.expert_ids[e] for e in spy.graph.eval_expert.tolist()}
        assert trained and not trained & hidden  # no evaluation edge of a held-out expert
        assert not {ids[e] for e in spy.archive.expert.tolist()} & hidden  # no archive record either
        if hidden == set(arch_hidden):
            assert "gin" not in spy.graph.arch_names
        else:  # seen factors: every held-out expert's architecture, objective, and source stay in training
            for e in hidden:
                spec = infra.catalog[infra.expert_index[e]]
                assert spec.architecture in spy.graph.arch_names and spec.objective in spy.graph.objective_names
                assert any(s.source == spec.source for s in spy.catalog)

    rows = {r["condition"]: r for r in _rows(env, "insertion") if r["seed"] == "42"}
    assert set(rows) == set(INSERTION_CONDITIONS)
    for r in rows.values():
        _finite(r, "test_routergfm_risk", "test_residual_error", "test_metagl_uniform_risk", "test_metagl_local_risk")
    assert [rows[c]["test_num_inserted"] for c in INSERTION_CONDITIONS] == ["0.0", "2.0", "4.0", "4.0"]
    assert rows["joint"]["test_target_evaluation_edges"] == "0.0"
    assert float(rows["new_architecture"]["test_target_evaluation_edges"]) == 4  # the seen gcn experts
    # MetaGL+metadata sees the same target evaluations: a known application exactly when RouterGFM's is.
    details = json.loads((RouterPaths.from_cfg(cfg).analysis_dir("insertion") / "nodea__node__b3.json").read_text())
    for condition, info in details["details"][app.key].items():
        assert info["metagl_known_target"] == (condition in ("new_configuration", "new_architecture")), condition

    # Deployment: inserted experts have construction edges only; a known target's own
    # evaluations reach the scores, a new application's never exist.
    base = holdout_router(cfg, infra, app, arch_hidden, "arch")
    assert "gin" not in base.graph.arch_names and not set(arch_hidden) & set(base.graph.expert_ids)
    joint = insert_experts(cfg, infra, base, arch_hidden)
    assert "gin" in joint.graph.arch_names and "gin" not in base.graph.arch_names
    inserted_nodes = {joint.graph.expert_index[e] for e in arch_hidden}
    assert not inserted_nodes & set(joint.graph.eval_expert.tolist())
    seen = [e for e in base.graph.expert_ids]
    known, averages = with_known_target(cfg, infra, joint, app, seen)
    assert len(averages) == 4 and set(averages) <= set(seen) and app.key not in joint.graph.app_index
    assert all(averages[e] == infra.store.app_average(app, e)[0] for e in averages)
    node = known.graph.app_index[app.key]
    assert set(known.graph.eval_expert[known.graph.eval_app == node].tolist()).isdisjoint(inserted_nodes)
    pool, mu_new, _ = score_pool(cfg, app, infra, joint)
    pool_known, mu_known, _ = score_pool(cfg, app, infra, known)
    assert pool == pool_known and set(arch_hidden) <= set(pool) and not torch.allclose(mu_new, mu_known)


def test_calibration(env):
    cfg = _kind_cfg(env, "calibration")
    infra = env.infra
    assert run_analysis(cfg, infra=infra) == 0
    rows = {r["condition"]: r for r in _rows(env, "calibration") if r["seed"] == "42"}
    assert set(rows) == {"m0", "m1", "m2"}
    for r in rows.values():
        _finite(r, "test_risk", "test_residual_error", "test_num_inserted")
    assert [rows[m]["test_num_calibration_apps"] for m in ("m0", "m1", "m2")] == ["0.0", "1.0", "2.0"]

    app = infra.application(TARGET, 3, 42)
    hidden = architecture_holdout(cfg, infra)
    router = insert_experts(cfg, infra, holdout_router(cfg, infra, app, hidden, "arch"), hidden)
    cands = calibration_candidates(infra, app, router, hidden)
    assert cands and all(b.group != app.group for b in cands)
    assert cands[0].budget == 3 and infra.task_family(cands[0]) == infra.task_family(app)  # same CompatKey first
    calibrated = calibrate(cfg, infra, router, hidden, cands[:2])
    assert calibrated.bundle is router.bundle  # router parameters stay fixed
    nodes = torch.tensor([calibrated.graph.expert_index[e] for e in hidden])
    new_edges = torch.isin(calibrated.graph.eval_expert, nodes)
    assert int(new_edges.sum()) > 0 and not bool(torch.isin(router.graph.eval_expert, nodes).any())
    calib_nodes = {calibrated.graph.app_index[b.key] for b in cands[:2]}
    assert set(calibrated.graph.eval_app[new_edges].tolist()) <= calib_nodes
    positions = torch.tensor([len(router.catalog) - len(hidden) + i for i in range(len(hidden))])
    added = torch.isin(calibrated.archive.expert, positions)
    assert int(added.sum()) > 0 and len(calibrated.archive) == len(router.archive) + int(added.sum())
    assert {calibrated.archive.apps[a].key for a in calibrated.archive.app[added].tolist()} <= {b.key for b in cands[:2]}
    assert calibrate(cfg, infra, router, hidden, []) is router


def test_shift_uses_tagged_applications_and_caches(env):
    from src.data_loader.shift_splits import split_file_path

    cfg = _kind_cfg(env, "shift")
    cfg.data_preparation.shift.conditions = ["feature", "mixed"]
    app = env.infra.application(TARGET, 3, 42)
    path = split_file_path(Path(cfg.moe.routergfm.analysis.shift_root) / "feature", app.dataset, app.task_level, app.seed, app.split)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train": [], "val": [], "test": [], "meta": {"label_tv_support_vs_query": 0.25, "target_quantile": 0.8}}, path)
    assert run_analysis(cfg, infra=env.infra) == 0  # no 'mixed' split file: that condition is skipped

    (row,) = _rows(env, "shift")
    assert row["condition"] == "feature" and row["test_label_tv_support_vs_query"] == "0.25"
    _finite(row, "test_routergfm_risk", "test_routergfm_g_risk", "test_routergfm_acc", "test_routergfm_g_acc")
    shifted = AppSpec(app.dataset, app.task_level, app.budget, app.seed, app.lp_split, split_root_tag="feature")
    assert shifted.key in env.provider.calls
    paths = RouterPaths.from_cfg(cfg)
    # Descriptors are split-free (one cache per instance set) and record the tagged split's coverage.
    cache = torch.load(paths.descriptor_file(instance_set_key(shifted)))
    assert {app.data_key, shifted.data_key} <= set(cache["data_keys"])
    details = json.loads((paths.analysis_dir("shift") / "nodea__node__b3.json").read_text())["details"]
    team = details[shifted.key]["team"]
    shifted_data, standard_data = env.infra.data(shifted), env.infra.data(app)
    assert not torch.equal(shifted_data.support_pos, standard_data.support_pos)
    for e in team:  # heads fitted on the shifted support, predictions on the shifted queries
        assert torch.equal(torch.load(paths.prediction_file(shifted.data_key, e))["query_pos"], shifted_data.query_pos)
        standard = paths.prediction_file(app.data_key, e)
        assert not standard.exists() or torch.equal(torch.load(standard)["query_pos"], standard_data.query_pos)
