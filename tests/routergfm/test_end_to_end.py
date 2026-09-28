"""The whole RouterGFM pipeline through ``run_routergfm`` on the synthetic setup.

history (every shard) -> descriptors -> router -> deploy (every integration rule)
-> benchmark rows (RouterGFM rules; matched baselines listed in benchmark.methods)
-> analyses (every kind) -> selection baselines (every selector) -> matched-pool
baselines (every method). Each stage must write result
rows with finite metrics. Targets cover every task family of the paper's nine
(node / graph classification, LP, multi-label, regression); selection and
META-DES run on the classification targets (Table 9 scope), KDEM/PPEM on the
node and LP targets (the scheduled levels), shift on node / graph targets.

Leakage audit: the provider wraps the target applications' labels and the
history store is patched, so every read of the audited target's query /
diagnostic labels or of its recorded evaluations is logged with the calling
frames. Such reads are allowed only inside the history recorder and the
evaluation helpers, which run after the team's predictions and weights are
fixed. The targets' instance graphs carry no ``y``, so the audited labels are
the only way to reach them. Stages run one target at a time and audit that
target only (another target is ordinary history for it).
"""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.moe.routergfm import run_routergfm
from src.moe.routergfm.analysis import ANALYSIS_WORKFLOW, INSERTION_CONDITIONS, PERTURBATION_CONDITIONS
from src.moe.routergfm.applications import instance_set_key
from src.moe.routergfm.common import AppSpec, RouterPaths, base_group, enumerate_applications, is_same_source
from src.moe.routergfm.experts import build_expert_catalog
from src.moe.routergfm.history import HistoryStore
from src.moe.routergfm.integration import RULES
from src.moe.routergfm.router.trainer import BUNDLE_FILE, router_run_key
from tests.routergfm.fixtures import SyntheticDataProvider, tiny_cfg

CLS_TARGETS = ("nodea:node", "grapha:graph")  # node and graph classification: Table 9 scope, META-DES scope
TARGETS = CLS_TARGETS + ("linka:edge", "multia:graph", "rega:graph")
TARGET_GROUPS = {"nodea", "grapha", "linka", "multia", "rega"}
METRIC = {"nodea": "acc", "grapha": "acc", "linka": "auc", "multia": "auc", "rega": "mae"}
HISTORY_EXTRA = (
    "srca:node", "nodeb:node", "nodec:node", "linkb:edge", "srcb:graph", "graphb:graph", "multib:graph", "regb:graph",
)
BUDGET, SEED = 3, 42
SELECTORS = ("metadata_mlp", "nearest_application", "metagl", "metagl_metadata", "logme", "model_spider")
MATCHED_TARGETS = {
    "metagl_u": TARGETS,
    "sagmm_pe": TARGETS,
    "meta_des": CLS_TARGETS,
    "kdem": ("nodea:node", "linka:edge"),
    "ppem": ("nodea:node", "linka:edge"),
}
ANALYSIS_KINDS = ("team_size", "archive_reliability", "specialization", "insertion", "calibration")
SHIFT_TARGETS = ("nodea:node", "rega:graph")  # no LP shift splits

# Frames allowed to read the target's query / diagnostic labels.
LABEL_READERS = {
    "_history_record",  # history: D_a losses of every declared application (hidden from its own router)
    "evaluate_integration",  # deploy: metrics after every rule's weights are fixed
    "evaluate_outputs",  # RouterInfra evaluation helper (baselines, after their predictions)
    "residual_errors",  # analysis: |r_hat - realized loss| of a finished deployment
}
# Frames allowed to read the target's recorded evaluations (history store values).
HISTORY_READERS = {
    "_selection_diagnostics",  # deploy evaluation: hit@K / regret@K / specialization from D_a
    "query_expert_risk",  # selection evaluation after ranking
    "with_known_target",  # insertion: a known application's own evaluations (Table 10 conditions)
    "embeddings",  # support embeddings stored in the target's record (support side only)
}


class _Audit:
    """Reads of the audited groups' labels / history with the function names on the call stack."""

    def __init__(self):
        self.groups = set()
        self.reads = []

    def record(self, kind: str, what: str) -> None:
        frames, frame = set(), sys._getframe(2)
        while frame is not None:
            frames.add(frame.f_code.co_name)
            frame = frame.f_back
        self.reads.append((kind, what, frames))

    def store_reader(self, fn):
        audit = self

        def read(store, data_key, *args, **kwargs):
            if base_group(str(data_key).split("__", 1)[0]) in audit.groups:
                audit.record("history", str(data_key))
            return fn(store, data_key, *args, **kwargs)

        return read

    def violations(self):
        allowed = {"labels": LABEL_READERS, "history": HISTORY_READERS}
        return [(kind, what) for kind, what, frames in self.reads if not frames & allowed[kind]]

    def readers(self, kind: str):
        allowed = LABEL_READERS if kind == "labels" else HISTORY_READERS
        return {name for k, _, frames in self.reads if k == kind for name in frames & allowed}


class _AuditedLabels(dict):
    def __init__(self, labels, audit: _Audit, app: AppSpec):
        super().__init__(labels)
        self.audit, self.app = audit, app

    def __getitem__(self, split):
        if split != "support" and self.app.group in self.audit.groups:
            self.audit.record("labels", f"{self.app.key}:{split}")
        return super().__getitem__(split)

    def get(self, split, default=None):
        return self[split] if split in self else default

    def values(self):
        return [self[k] for k in self]

    def items(self):
        return [(k, self[k]) for k in self]


class _LabelFreeDataset:
    """A dataset whose items have ``y`` removed (PyG then reads ``y`` as None): no labels this way."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        graph = copy.copy(self.dataset[int(idx)])
        del graph.y
        return graph

    def __getattr__(self, name):  # dataset-level metadata (num_features, base_num_nodes, ...)
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


class _AuditedProvider:
    def __init__(self, base, audit: _Audit):
        self.base, self.audit = base, audit

    def load(self, app: AppSpec):
        data = self.base.load(app)  # the base provider gathers labels from ``y`` before they are removed
        data.labels = _AuditedLabels(data.labels, self.audit, app)
        if app.group in TARGET_GROUPS:
            data.dataset = _LabelFreeDataset(data.dataset)
        return data

    def base_graph(self, app: AppSpec):
        return self.base.base_graph(app)  # structure only: no labels to audit


def _run(env, task: str, groups, *, save_skipped: bool = False, results_dir=None, **blocks) -> None:
    """``run_routergfm`` with ``cfg.moe.routergfm.<block>__<key>`` overrides, auditing *groups*."""
    cfg = env.cfg.clone()
    cfg.moe.routergfm.task = task
    cfg.save_results.save_skipped = bool(save_skipped)
    if results_dir is not None:
        cfg.save_results.output_dir = str(results_dir)
    for dotted, value in blocks.items():
        node = cfg.moe.routergfm
        *parents, leaf = dotted.split("__")
        for parent in parents:
            node = node[parent]
        node[leaf] = value
    env.audit.groups, env.audit.reads = set(groups), []
    assert run_routergfm(cfg, provider=env.provider) == 0, task
    assert env.audit.violations() == [], task


def _group(spec: str) -> str:
    return spec.split(":")[0]


def _rows(env, workflow: str, results_dir=None):
    directory = Path(results_dir if results_dir is not None else env.cfg.save_results.output_dir)
    with open(directory / f"{workflow}.tsv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert rows and not any("loss" in column for column in rows[0]), workflow
    return rows


def _finite(row, *columns):
    for column in columns:
        assert math.isfinite(float(row[column])), (column, row[column])


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("end_to_end")
    cfg = tiny_cfg(tmp, targets=TARGETS, history_extra=HISTORY_EXTRA, budgets=(BUDGET,), seeds=(SEED,))
    cfg.save_results.output_dir = str(tmp / "results")
    rg = cfg.moe.routergfm
    rg.experts.num_shards = 2
    rg.integration.rules = list(RULES)
    rg.analysis.team_sizes = [1, 2]
    rg.analysis.calibration_apps = [0, 1]
    rg.analysis.holdout_architecture = "gin"
    cfg.data_preparation.shift.conditions = ["feature"]
    b = rg.baselines
    b.metagl.epochs, b.metagl.patience, b.metagl.knn_k, b.metagl.rf_n_estimators, b.metagl.graph_sample_max = 3, 3, 3, 5, 10
    b.metadata_mlp.hidden_dim, b.metadata_mlp.epochs, b.metadata_mlp.patience = 16, 20, 5
    b.model_spider.token_dim, b.model_spider.epochs, b.model_spider.batch_size = 8, 3, 2
    b.model_spider.rerank_topk_grid = [0, 2]
    s = b.sagmm_pe
    s.epochs, s.prune_interval, s.edge.prune_interval = 12, 4, 4
    s.graph.epochs, s.graph.prune_interval, s.graph.batch_size = 12, 4, 16
    b.kdem_ppem.epochs, b.kdem_ppem.early_stopping, b.kdem_ppem.batch_size = 3, 5, 8
    b.kdem_ppem.kd.period, b.kdem_ppem.ema.period = 2, 2

    audit = _Audit()
    with pytest.MonkeyPatch.context() as mp:
        for name in ("matrix", "load"):  # every value read of the store goes through one of these
            mp.setattr(HistoryStore, name, audit.store_reader(getattr(HistoryStore, name)))
        yield SimpleNamespace(
            cfg=cfg, tmp=tmp, audit=audit, paths=RouterPaths.from_cfg(cfg),
            provider=_AuditedProvider(SyntheticDataProvider(), audit),
        )


@pytest.fixture(scope="module")
def trained(env):
    """History of every shard, the descriptors stage, and the router of each target."""
    readers = {}
    for shard in range(2):
        _run(env, "history", TARGET_GROUPS, experts__shard_index=shard)
        readers[f"history{shard}"] = env.audit.readers("labels")
    for app in enumerate_applications(env.cfg.moe.routergfm):  # shard 0 wrote them; the stage must rebuild them
        env.paths.descriptor_file(instance_set_key(app)).unlink(missing_ok=True)
    _run(env, "descriptors", TARGET_GROUPS)
    readers["descriptors"] = len(env.audit.reads)
    for spec in TARGETS:
        _run(env, "router", {_group(spec)}, deploy__target=spec, deploy__budget=BUDGET)
        readers[f"router:{spec}"] = len(env.audit.reads)
    return SimpleNamespace(readers=readers)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def test_audit_flags_reads_outside_evaluation(env, trained):
    app = AppSpec("nodea", "node", BUDGET, SEED)
    env.audit.groups, env.audit.reads = {"nodea"}, []
    data = env.provider.load(app)
    data.labels["support"], data.labels["query"]  # support reads are never logged
    store = HistoryStore(env.paths)
    store.losses(app, store.expert_ids(app.data_key)[:1])
    env.provider.load(AppSpec("grapha", "graph", BUDGET, SEED)).labels["query"]  # not the audited target
    assert set(env.audit.violations()) == {("labels", f"{app.key}:query"), ("history", app.data_key)}
    # Instance graphs of targets carry no labels; historical applications' keep theirs.
    assert "y" not in data.dataset[int(data.query_pos[0])] and data.dataset.name == "nodea"
    assert "y" in env.provider.load(AppSpec("nodeb", "node", BUDGET, SEED)).dataset[0]


def test_history_descriptors_and_router_stages(env, trained):
    rg = env.cfg.moe.routergfm
    store = HistoryStore(env.paths)
    catalog = build_expert_catalog(env.cfg)
    for app in enumerate_applications(rg):  # both shards together cover every pair but the same-source ones
        expected = sorted(s.expert_id for s in catalog if not (rg.experts.exclude_same_source and is_same_source(app, s)))
        assert store.expert_ids(app.data_key) == expected, app.key
        cache = torch.load(env.paths.descriptor_file(instance_set_key(app)))
        assert app.data_key in cache["data_keys"] and bool(torch.isfinite(cache["z"]).all()), app.key
    # Targets' D_a losses are recorded like any application's; nothing else reads their labels or
    # history, and a router never touches its own target's.
    assert trained.readers == {
        "history0": {"_history_record"}, "history1": {"_history_record"}, "descriptors": 0,
        **{f"router:{spec}": 0 for spec in TARGETS},
    }
    for spec in TARGETS:
        assert (env.paths.router_dir(router_run_key(_group(spec), BUDGET, SEED)) / BUNDLE_FILE).is_file(), spec


@pytest.mark.parametrize("spec", TARGETS)
def test_deploy_every_integration_rule(env, trained, spec):
    _run(env, "deploy", {_group(spec)}, deploy__target=spec, deploy__budget=BUDGET, deploy__seed=SEED)
    assert env.audit.readers("labels") == {"evaluate_integration"}  # query labels only once weights are fixed
    assert env.audit.readers("history") == {"_selection_diagnostics"}
    name, level = spec.split(":")
    metric = METRIC[name]
    result = json.loads((env.paths.deploy_dir(AppSpec(name, level, BUDGET, SEED).key) / "deploy.json").read_text())
    assert set(result["rules"]) == set(RULES) and result["metric"] == metric
    for metrics in result["rules"].values():
        _finite(metrics, metric, "risk", "worst_cell_risk", "winner_agreement")
    assert len(result["team"]) == int(env.cfg.moe.routergfm.router.topk)
    _finite(result["selection"], "hit_at_k", "regret_at_k", "winner_coverage", "specialization_index")


def test_benchmark_rows(env, trained):
    for spec in TARGETS:
        _run(env, "benchmark", {_group(spec)}, deploy__target=spec)
        assert env.audit.readers("labels") == {"evaluate_integration"}
    rows = _rows(env, "moe_routergfm")
    methods = {"routergfm", "routergfm_g"} | {f"fixed_team:{r}" for r in RULES}
    for spec in TARGETS:
        mine = {r["method"]: r for r in rows if r["dataset"] == _group(spec)}
        assert set(mine) == methods, spec
        metric = f"test_{METRIC[_group(spec)]}"
        for row in mine.values():
            assert row["metric"] == metric and row["n_runs"] == "1"
            _finite(row, f"{metric}_mean", "test_risk_mean", "test_worst_cell_risk_mean", "test_hit_at_2_mean")
        assert mine["routergfm"]["test_risk_mean"] == mine["fixed_team:routergfm"]["test_risk_mean"]


def _analysis_tsv(env, name: str, rows) -> str:
    path = env.tmp / f"analysis_{name}.tsv"
    path.write_text("# kind\tdataset\ttask_level\tbudget\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))
    return str(path)


def test_every_analysis_kind(env, trained):
    from src.data_loader.shift_splits import split_file_path

    root = Path(env.cfg.moe.routergfm.analysis.shift_root) / "feature"
    written, runs = 0, {}
    for spec in TARGETS:  # one invocation per target, as the SLURM rows run them
        name, level = spec.split(":")
        kinds = ANALYSIS_KINDS + (("shift",) if spec in SHIFT_TARGETS else ())
        if "shift" in kinds:
            shift = split_file_path(root, name, level, SEED, AppSpec(name, level, BUDGET, SEED).split)
            shift.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"train": [], "val": [], "test": [], "meta": {"label_tv_support_vs_query": 0.1}}, shift)
        tasks = _analysis_tsv(env, name, [(k, name, level, BUDGET) for k in kinds])
        _run(env, "analysis", {name}, benchmark__run_tasks_tsv=True, benchmark__tasks_tsv=tasks)
        assert env.audit.readers("labels") == {"evaluate_integration", "residual_errors"}, spec
        assert "with_known_target" in env.audit.readers("history")  # only the known-application insertion conditions
        rows = _rows(env, ANALYSIS_WORKFLOW)
        runs[spec], written = rows[written:], len(rows)
    # Table 14 as scheduled: one 'all' row pools every target of the budget into terciles. Several
    # targets deploy in one invocation (each is ordinary history for the others), so this run is not
    # audited; each target's own specialization deployment was audited above.
    _run(env, "analysis", set(), benchmark__run_tasks_tsv=True,
         benchmark__tasks_tsv=_analysis_tsv(env, "all", [("specialization", "all", "-", BUDGET)]))
    rows = _rows(env, ANALYSIS_WORKFLOW)
    pooled = rows[written:]

    for spec, mine in runs.items():
        name = _group(spec)
        by_kind = {}
        for row in mine:
            by_kind.setdefault(row["kind"], []).append(row)
        assert list(by_kind) == list(ANALYSIS_KINDS) + (["shift"] if spec in SHIFT_TARGETS else []), spec
        assert [r["condition"] for r in by_kind["team_size"]] == ["K1", "K2"]
        assert [r["condition"] for r in by_kind["archive_reliability"]] == list(PERTURBATION_CONDITIONS)
        assert [(r["dataset"], r["condition"]) for r in by_kind["specialization"]] == [(name, "low"), ("all", "low")]
        assert [r["condition"] for r in by_kind["insertion"]] == list(INSERTION_CONDITIONS)
        assert [r["condition"] for r in by_kind["calibration"]] == ["m0", "m1"]
        assert [r["condition"] for r in by_kind.get("shift", [])] == (["feature"] if spec in SHIFT_TARGETS else [])
        assert all(r["dataset"] in (name, "all") for r in mine)
        if spec in SHIFT_TARGETS:
            _finite(by_kind["shift"][0], f"test_routergfm_{METRIC[name]}", f"test_routergfm_g_{METRIC[name]}")
    apps, summary = pooled[: len(TARGETS)], pooled[len(TARGETS):]
    assert sorted(r["dataset"] for r in apps) == sorted(_group(s) for s in TARGETS)
    assert sorted(r["condition"] for r in apps) == ["high", "low", "low", "medium", "medium"]
    assert [(r["dataset"], r["condition"], r["test_num_applications"]) for r in summary] == [
        ("all", "low", "2.0"), ("all", "medium", "2.0"), ("all", "high", "1.0"),
    ]
    by_kind = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row)
    finite = {
        "team_size": ("test_risk", "test_winner_coverage"),
        "archive_reliability": ("test_global_risk", "test_local_rho1_risk", "test_local_validated_risk"),
        "specialization": ("test_specialization_index", "test_global_risk", "test_local_risk"),
        "insertion": ("test_routergfm_risk", "test_residual_error", "test_metagl_uniform_risk", "test_metagl_local_risk"),
        "calibration": ("test_risk", "test_residual_error"),
        "shift": ("test_routergfm_risk", "test_routergfm_g_risk"),
    }
    for kind, columns in finite.items():
        for row in by_kind[kind]:
            _finite(row, *columns)


@pytest.mark.parametrize("method", SELECTORS)
def test_selection_baseline(env, trained, method):
    for spec in CLS_TARGETS:  # one SLURM row per target, as in slurm/moe.routergfm_selection.tsv
        _run(env, "selection_baseline", {_group(spec)}, baselines__method=method, baselines__datasets=[spec])
        assert env.audit.readers("labels") == set()  # selectors read support labels at most
        assert env.audit.readers("history") <= {"query_expert_risk", "embeddings"}  # risks scored after ranking
    # The documented pooled pass: cached rankings only, one table9_all row over both targets.
    _run(env, "selection_baseline", {_group(s) for s in CLS_TARGETS}, save_skipped=True, baselines__method=method,
         baselines__datasets=list(CLS_TARGETS))
    assert env.audit.reads == []
    rows = [r for r in _rows(env, "moe_routergfm_selection") if r["moe.routergfm.baselines.method"] == method]
    assert [r["dataset"] for r in rows] == ["nodea", "grapha", "nodea", "grapha", "table9_all"]
    assert rows[-1]["n_apps"] == "2"
    for row in rows:
        _finite(row, "test_hit_at_2_mean", "test_regret_at_2_mean")


@pytest.mark.parametrize("method", sorted(MATCHED_TARGETS))
def test_matched_baseline(env, trained, method):
    targets = MATCHED_TARGETS[method]
    for spec in targets:
        _run(env, "matched_baseline", {_group(spec)}, baselines__method=method, baselines__datasets=[spec])
        assert env.audit.readers("labels") <= {"evaluate_outputs"}  # query labels only to score the finished run
        assert env.audit.readers("history") <= {"embeddings"}
    rows = _rows(env, f"moe_{method}")
    assert [json.loads(r["moe.routergfm.baselines.datasets"]) for r in rows] == [[spec] for spec in targets]
    for spec, row in zip(targets, rows):
        _finite(row, f"test_{METRIC[_group(spec)]}_mean")


def test_benchmark_runs_requested_matched_baselines(env, trained):
    """``benchmark.methods`` beyond ``routergfm`` run as matched-pool baselines on the same task and seeds."""
    results, spec = env.tmp / "benchmark_with_baselines", "nodea:node"
    _run(env, "benchmark", {"nodea"}, results_dir=results, deploy__target=spec,
         benchmark__methods=["routergfm", "metagl_u", "meta_des"], baselines__output_dir=str(env.tmp / "bench_baselines"))
    assert env.audit.readers("labels") == {"evaluate_integration", "evaluate_outputs"}
    assert env.audit.readers("history") <= {"_selection_diagnostics", "embeddings"}
    assert {r["dataset"] for r in _rows(env, "moe_routergfm", results)} == {"nodea"}
    for method in ("metagl_u", "meta_des"):
        (row,) = _rows(env, f"moe_{method}", results)
        assert json.loads(row["moe.routergfm.baselines.datasets"]) == [spec] and json.loads(row["seeds"]) == [SEED]
        _finite(row, "test_acc_mean")
