"""Paper App. D analyses of the deployed router (``moe.routergfm.task analysis``).

``analysis.kind`` selects one; every analysis reuses the deployment steps of
:mod:`.deploy` (label-free integration, then evaluation):

* ``team_size`` (Table 12, Fig. 5): nested teams, the ``analysis.team_sizes``
  prefixes of the router's global ranking of E_a, mixed with the ``routergfm``
  rule. Brier risk, risk reduction relative to the smallest team (K=1 by
  default), local-winner coverage, and expert contributions (predictors mixed).
* ``archive_reliability`` (Table 13, Fig. 7b): global-only, full local
  correction (rho=1), and source-validated rho (the bundle's) risks with the
  archive matched or perturbed (``half_cells``, ``missing_family``, ``reversed``).
* ``specialization`` (Table 14, Fig. 7a): the specialization index of each
  target application (Prop. 2 Delta_loc over the evaluation cells, from the
  target's recorded D_a losses; evaluation only); applications of one budget are
  split into low / medium / high terciles; global vs local risk per bucket.
* ``insertion`` (Table 10): ``new_application`` (standard deployment);
  ``new_configuration`` (``analysis.holdout_fraction`` of E_a whose architecture,
  objective, and source all stay seen) and ``new_architecture`` (every
  ``analysis.holdout_architecture`` checkpoint, arch node included) retrain the
  router without the held-out experts (no expert nodes, evaluation edges, or
  archive records) and insert them metadata-only at deployment while the
  target's own evaluations of the seen experts stay visible (a known
  application); ``joint`` inserts the new-architecture experts for the target as
  a new application. Reports the deployed RouterGFM team's Brier risk, the
  residual-estimation error ``mean |r_hat - realized loss|`` over team x queries,
  and the MetaGL+metadata team (same size; same hidden experts and visible
  target evaluations) mixed uniformly and locally on the same deployed router.
* ``calibration`` (Table 11): the joint-novelty deployment after adding the
  inserted experts' evaluation edges and archive records on ``m`` historical
  applications (``analysis.calibration_apps``), router parameters fixed.
* ``shift`` (Table 15): RouterGFM and RouterGFM-G on target splits read from
  ``analysis.shift_root/<condition>`` (``data_preparation.shift.conditions``;
  conditions without a split file are skipped); router, history, and archive
  stay standard.

Tasks: ``benchmark.tasks_tsv`` rows (``# kind dataset task_level budget``; kind
defaults to ``analysis.kind``; dataset ``all`` = every ``apps.targets`` entry)
when ``benchmark.run_tasks_tsv``, else ``analysis.kind`` on the router/benchmark
tasks; seeds ``apps.seeds[:benchmark.num_runs]``. Rows (kind, condition,
dataset, task_level, budget, seed, ``test_*`` metrics) go to the
``moe_routergfm_analysis`` results table; a JSON per task is saved under
``RouterPaths.analysis_dir(kind)``.
"""

from __future__ import annotations

import dataclasses
import math
import os
import traceback
from collections import Counter, OrderedDict
from datetime import datetime
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch

from src.utils.checkpoint import save_json_atomic
from src.utils.save_results import append_workflow_result_rows
from src.utils.tsv_parsing import dedup_tasks, parse_row_by_header, read_tsv_rows

from .applications import derive_seed
from .archive import Archive, build_archive, perturb_archive
from .common import AppSpec, CompatKey, RouterPaths, parse_dataset_spec, stable_hash
from .context_graph import add_calibration_edges, insert_application, insert_expert
from .deploy import (
    DeployRouter,
    Integration,
    evaluate_integration,
    integrate_application,
    local_evidence,
    prepare_router,
    router_seed,
    score_pool,
)
from .diagnostics import routing_risks
from .integration import local_estimates
from .router.trainer import RouterTrainer, build_router_trainer, reusable_bundle, router_run_key
from .run import benchmark_seeds, routergfm_tasks

ANALYSIS_WORKFLOW = "moe_routergfm_analysis"
KINDS = ("team_size", "archive_reliability", "specialization", "insertion", "calibration", "shift")
PERTURBATION_CONDITIONS = ("matched", "half_cells", "missing_family", "reversed")
INSERTION_CONDITIONS = ("new_application", "new_configuration", "new_architecture", "joint")
SPECIALIZATION_BUCKETS = ("low", "medium", "high")
KNOWN_TARGET_PREFIX = "known:"  # app node of a target whose own evaluations are visible
_ALL = "all"
_LOG = "[RouterGFM analysis]"
_TSV_COLUMNS = {"kind", "dataset", "task_level", "budget"}
_NAN = float("nan")

Task = Tuple[str, int]  # ("dataset:level", budget)


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
def _parse_column(col: str, val: str, line_no: int):
    text = val.strip().lower()
    if col == "budget":
        try:
            return int(text), True
        except ValueError:
            print(f"{_LOG} Skipping malformed task row {line_no}: invalid budget '{val}'")
            return None, False
    if col in ("kind", "dataset", "task_level"):
        return text, True
    return None


def parse_analysis_tasks(tsv_path: str, default_kind: str) -> List[Tuple[str, str, int]]:
    """``(kind, "dataset:level" | "all", budget)`` rows of ``# kind dataset task_level budget`` (kind optional)."""
    header, data_rows = read_tsv_rows(tsv_path, _TSV_COLUMNS, min_header_columns=3, log_prefix=_LOG)
    if header is None:
        if data_rows:
            print(f"{_LOG} Missing header row; expected '# kind dataset task_level budget'.")
        return []
    tasks = []
    for line_no, parts in data_rows:
        row = parse_row_by_header(
            parts, header, line_no, _LOG, required_columns=("dataset", "task_level", "budget"),
            defaults={"kind": default_kind}, custom_parser=_parse_column,
        )
        if row is not None:
            spec = _ALL if row["dataset"] == _ALL else f"{row['dataset']}:{row['task_level']}"
            tasks.append({"kind": row["kind"], "spec": spec, "budget": int(row["budget"])})
    tasks = dedup_tasks(tasks, lambda t: (t["kind"], t["spec"], t["budget"]))
    return [(t["kind"], t["spec"], t["budget"]) for t in tasks]


def analysis_tasks(cfg) -> "OrderedDict[str, List[Task]]":
    """Tasks per analysis kind (``all`` expanded to every ``apps.targets`` entry)."""
    rg = cfg.moe.routergfm
    kind = str(rg.analysis.kind).strip().lower()
    if bool(rg.benchmark.run_tasks_tsv):
        rows = parse_analysis_tasks(str(rg.benchmark.tasks_tsv), kind)
    else:
        rows = [(kind, spec, budget) for spec, budget in routergfm_tasks(cfg)]
    unknown = sorted({k for k, _, _ in rows} - set(KINDS))
    if unknown:
        raise ValueError(f"Unknown analysis kind(s) {unknown}; expected one of {KINDS}.")
    out: "OrderedDict[str, List[Task]]" = OrderedDict()
    for kind, spec, budget in rows:
        specs = [str(t) for t in rg.apps.targets] if spec == _ALL else [spec]
        for s in specs:
            name, level = parse_dataset_spec(s)
            task = (f"{name}:{level}", int(budget))
            if task not in out.setdefault(kind, []):
                out[kind].append(task)
    return out


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _row(kind: str, condition: str, app: AppSpec, metrics: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "kind": kind,
        "condition": condition,
        "dataset": app.dataset,
        "task_level": app.task_level,
        "budget": int(app.budget),
        "seed": int(app.seed),
    }
    row.update({f"test_{name}": float(value) for name, value in metrics.items()})
    return row


def _finite_mean(values) -> float:
    v = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    v = v[torch.isfinite(v)]
    return float(v.mean()) if v.numel() else _NAN


def _task_name(spec: str, budget: int) -> str:
    name, level = parse_dataset_spec(spec)
    return f"{name}__{level}__b{int(budget)}"


def _write(cfg, kind: str, name: str, rows: List[Dict[str, Any]], details: Any, started_at: datetime) -> int:
    """Save the task JSON under ``analysis_dir(kind)`` and append its rows to the results table."""
    path = RouterPaths.from_cfg(cfg).analysis_dir(kind) / f"{name}.json"
    save_json_atomic(str(path), {"kind": kind, "rows": rows, "details": details})
    return append_workflow_result_rows(
        cfg=cfg, workflow=ANALYSIS_WORKFLOW, rows=rows, started_at=started_at, ended_at=datetime.now().astimezone()
    )


def _seed_apps(cfg, infra, spec: str, budget: int, seeds: Sequence[int]) -> Iterator[Tuple[AppSpec, DeployRouter]]:
    """(application, prepared router) per seed; H and M are rebuilt once per router seed."""
    router, key = None, None
    for seed in seeds:
        app = infra.application(spec, budget, seed)
        if router_seed(cfg, app) != key:
            router, key = prepare_router(cfg, app, infra), router_seed(cfg, app)
        yield app, router


def _deploy(cfg, app: AppSpec, infra, router: DeployRouter, rules: Sequence[str], team=None) -> Tuple[Integration, Dict[str, Any]]:
    integ = integrate_application(cfg, app, infra, router, rules, team_override=team)
    return integ, evaluate_integration(cfg, app, infra, integ)


def _with_rho(router: DeployRouter, rho: float) -> DeployRouter:
    return dataclasses.replace(router, bundle=dataclasses.replace(router.bundle, rho=float(rho)))


def _model_device(router: DeployRouter) -> torch.device:
    return next(router.bundle.model.parameters()).device


def residual_errors(cfg, app: AppSpec, infra, router: DeployRouter, integ: Integration, inserted: Sequence[str] = ()) -> Tuple[float, float]:
    """Mean ``|r_hat - realized per-instance routing loss|`` over team x queries, and over the inserted members.

    ``r_hat`` is the deployed Eq. 7 estimate (``routergfm`` rule, the bundle's
    rho), from the deployment's own evidence and retrieval when it ran a local
    rule. Evaluation only: reads the target's query labels.
    """
    ctx = integ.context
    if ctx.evidence is None:  # no local rule was deployed: retrieve as the routergfm rule would
        graph = router.graph.to(_model_device(router))  # expert projections only; the target node is not needed
        ctx = dataclasses.replace(ctx, evidence=local_evidence(app, integ.family, router, graph, integ.team))
    r_hat = local_estimates(ctx, "routergfm")
    data = infra.data(app)
    loss = routing_risks(
        integ.preds, data.labels["query"], integ.family,
        normalizer=infra.normalizer(app), reg_kind=str(cfg.moe.routergfm.loss.regression),
    )
    error = (r_hat - loss).abs()
    cols = [i for i, e in enumerate(integ.team) if e in set(inserted)]
    return _finite_mean(error), (_finite_mean(error[:, cols]) if cols else _NAN)


# --------------------------------------------------------------------------- #
# Team size (Table 12)
# --------------------------------------------------------------------------- #
def _team_size(cfg, infra, spec: str, budget: int, seeds: Sequence[int]):
    sizes = sorted({int(k) for k in cfg.moe.routergfm.analysis.team_sizes if int(k) > 0})
    rows, details = [], {}
    for app, router in _seed_apps(cfg, infra, spec, budget, seeds):
        pool, pool_mu, _ = score_pool(cfg, app, infra, router)
        ranking = [pool[i] for i in torch.argsort(pool_mu, stable=True).tolist()]
        risks: Dict[int, float] = {}
        for k in sizes:
            if k > len(ranking):
                print(f"{_LOG}[team_size] {app.key}: K={k} exceeds |E_a|={len(ranking)}; skipped", flush=True)
                continue
            _, result = _deploy(cfg, app, infra, router, ["routergfm"], team=ranking[:k])
            risks[k] = result["rules"]["routergfm"]["risk"]
            rows.append(_row("team_size", f"K{k}", app, {
                "risk": risks[k],
                "risk_reduction": risks[min(risks)] - risks[k],
                "winner_coverage": result["selection"]["winner_coverage"],
                "expert_contributions": k,
            }))
        details[app.key] = {"ranking": ranking[: max(sizes, default=0)], "risk": risks}
    return rows, details


# --------------------------------------------------------------------------- #
# Archive reliability (Table 13)
# --------------------------------------------------------------------------- #
def _archive_reliability(cfg, infra, spec: str, budget: int, seeds: Sequence[int]):
    cfg = cfg.clone()
    cfg.moe.routergfm.archive.perturbation = "none"  # the matched archive; perturbations are applied here
    seed = int(cfg.moe.routergfm.archive.perturbation_seed)
    rows, details = [], {}
    for app, router in _seed_apps(cfg, infra, spec, budget, seeds):
        details[app.key] = {}
        for condition in PERTURBATION_CONDITIONS:
            archive = router.archive if condition == "matched" else perturb_archive(router.archive, condition, seed)
            perturbed = dataclasses.replace(router, archive=archive)
            _, validated = _deploy(cfg, app, infra, perturbed, ["global", "routergfm"])
            _, full = _deploy(cfg, app, infra, _with_rho(perturbed, 1.0), ["routergfm"])
            rows.append(_row("archive_reliability", condition, app, {
                "global_risk": validated["rules"]["global"]["risk"],
                "local_rho1_risk": full["rules"]["routergfm"]["risk"],
                "local_validated_risk": validated["rules"]["routergfm"]["risk"],
                "validated_rho": validated["rho"],
                "num_records": validated["num_records"],
            }))
            details[app.key][condition] = {"team": validated["team"], "num_archive_records": len(archive)}
    return rows, details


# --------------------------------------------------------------------------- #
# Specialization (Table 14)
# --------------------------------------------------------------------------- #
def _specialization(cfg, infra, tasks: Sequence[Task], seeds: Sequence[int]):
    """Per-application rows (condition = tercile bucket within its budget) and one summary row per bucket."""
    records = []
    for spec, budget in tasks:
        for app, router in _seed_apps(cfg, infra, spec, budget, seeds):
            _, result = _deploy(cfg, app, infra, router, ["global", "routergfm"])
            records.append({
                "app": app,
                "specialization_index": float(result["selection"]["specialization_index"]),
                "global_risk": result["rules"]["global"]["risk"],
                "local_risk": result["rules"]["routergfm"]["risk"],
            })
    rows, summary = [], []
    for budget in sorted({r["app"].budget for r in records}):
        mine = [r for r in records if r["app"].budget == budget]
        defined = sorted((r for r in mine if math.isfinite(r["specialization_index"])), key=lambda r: r["specialization_index"])
        bucket_of = {}
        for name, part in zip(SPECIALIZATION_BUCKETS, np.array_split(np.arange(len(defined)), len(SPECIALIZATION_BUCKETS))):
            bucket_of.update({defined[i]["app"].key: name for i in part.tolist()})
        for r in mine:
            rows.append(_row("specialization", bucket_of.get(r["app"].key, "undefined"), r["app"], {
                "specialization_index": r["specialization_index"],
                "global_risk": r["global_risk"],
                "local_risk": r["local_risk"],
                "risk_difference": r["local_risk"] - r["global_risk"],
            }))
        for name in SPECIALIZATION_BUCKETS:
            members = [r for r in mine if bucket_of.get(r["app"].key) == name]
            if not members:
                continue
            glob = _finite_mean([r["global_risk"] for r in members])
            loc = _finite_mean([r["local_risk"] for r in members])
            summary.append({
                "kind": "specialization", "condition": name, "dataset": _ALL, "task_level": _ALL,
                "budget": int(budget), "seed": _ALL,
                "test_specialization_index": _finite_mean([r["specialization_index"] for r in members]),
                "test_global_risk": glob,
                "test_local_risk": loc,
                "test_risk_difference": loc - glob,
                "test_num_applications": float(len(members)),
            })
    return rows + summary, {"applications": [r["app"].key for r in records]}


# --------------------------------------------------------------------------- #
# Insertion and calibration (Tables 10-11)
# --------------------------------------------------------------------------- #
def configuration_holdout(cfg, infra, app: AppSpec, seed: int) -> List[str]:
    """``analysis.holdout_fraction`` of E_a (at least one) whose architecture, objective, and source stay seen.

    Candidates are visited in a seeded random order; one is held out only if
    every factor keeps another checkpoint in the remaining catalog.
    """
    pool = infra.compatible_pool(app)
    target = max(1, int(round(float(cfg.moe.routergfm.analysis.holdout_fraction) * len(pool))))
    factors = lambda s: (("arch", s.architecture), ("objective", s.objective), ("source", s.source))  # noqa: E731
    left = Counter(f for spec in infra.catalog for f in factors(spec))
    hidden: List[str] = []
    for i in torch.randperm(len(pool), generator=torch.Generator().manual_seed(int(seed))).tolist():
        spec = infra.catalog[infra.expert_index[pool[i]]]
        if all(left[f] > 1 for f in factors(spec)):
            left.subtract(factors(spec))
            hidden.append(spec.expert_id)
            if len(hidden) == target:
                break
    if not hidden:
        raise ValueError(f"{app.key}: no expert can be held out with all of its factors still seen.")
    return hidden


def architecture_holdout(cfg, infra) -> List[str]:
    """Every checkpoint of ``analysis.holdout_architecture`` (checkpoints differing only by seed included)."""
    arch = str(cfg.moe.routergfm.analysis.holdout_architecture)
    hidden = [s.expert_id for s in infra.catalog if s.architecture == arch]
    if not hidden or len(hidden) == len(infra.catalog):
        raise ValueError(f"Cannot hold out architecture {arch!r}: {len(hidden)} of {len(infra.catalog)} experts.")
    return hidden


def holdout_router(cfg, infra, app: AppSpec, hidden: Sequence[str], tag: str) -> DeployRouter:
    """The router of (app.group, app.budget, router seed) retrained without the *hidden* experts.

    H has no node, construction edge, or evaluation edge of a hidden expert
    (nor an arch/objective node used only by them) and M no record of one; the
    applications, validation groups, and descriptor standardizer are those of
    the standard router (``build_router_trainer(..., exclude_experts=hidden)``).
    Bundles live under ``analysis_dir('insertion')/routers`` and are reused with
    ``router.skip_if_exists`` (see :func:`reusable_bundle`).
    """
    hidden = sorted(set(hidden))
    seed = router_seed(cfg, app)
    name = f"{router_run_key(app.group, app.budget, seed)}__{tag}_{stable_hash(hidden)}"
    directory = RouterPaths.from_cfg(cfg).analysis_dir("insertion") / "routers" / name
    if not reusable_bundle(cfg, directory):
        trainer = build_router_trainer(
            cfg, app.group, app.budget, infra.provider, seed, infra.device, exclude_experts=hidden
        )
        trainer.meta["run_key"] = name
        trainer.fit()
        trainer.select_integration_params()
        trainer.save(directory)
    return prepare_router(cfg, app, infra, bundle=RouterTrainer.load(directory, cfg, infra.device))


def insert_experts(cfg, infra, router: DeployRouter, expert_ids: Sequence[str]) -> DeployRouter:
    """A copy of *router* with experts inserted from metadata only (construction edges; new arch nodes as needed)."""
    graph = router.graph.to("cpu")  # copied containers: insertions stay local
    catalog = list(router.catalog)
    for e in expert_ids:
        catalog.append(infra.catalog[infra.expert_index[e]])
        insert_expert(graph, catalog[-1], infra.text_encoder, cfg, len(catalog) - 1)
    return dataclasses.replace(router, catalog=catalog, graph=graph)


def with_known_target(
    cfg, infra, router: DeployRouter, app: AppSpec, experts: Sequence[str]
) -> Tuple[DeployRouter, Dict[str, float]]:
    """A copy of *router* whose target node carries its recorded evaluations of *experts* (a known application).

    The node is labelled ``known:<app.key>`` instead of the AppSpec because
    deployment hides the evaluation edges of every AppSpec node of the target
    group; ``app_index[app.key]`` still points to it, so deployment scores the
    target from this node. The averages come from the target's D_a history (a
    subset of Q_a). Returns the router and the averages added as edges.
    """
    graph = router.graph.to("cpu")
    data = infra.data(app)
    node = insert_application(graph, app, data.stats, infra.text_encoder, cfg, family=data.task_family)
    graph.app_nodes[node] = KNOWN_TARGET_PREFIX + app.key
    added: Dict[str, float] = {}
    for e in experts:
        mu, count = infra.store.app_average(app, e)
        if count > 0 and math.isfinite(mu):
            add_calibration_edges(graph, graph.expert_index[e], [node], [mu], [count])
            added[e] = float(mu)
    return dataclasses.replace(router, graph=graph), added


def extend_archive(base: Archive, extra: Archive) -> Archive:
    """``base`` plus the records of ``extra`` (source applications matched by key)."""
    apps, groups, compats = list(base.apps), list(base.group), list(base.compat)
    index = {a.key: i for i, a in enumerate(apps)}
    remap = []
    for a, g, c in zip(extra.apps, extra.group, extra.compat):
        if a.key not in index:
            index[a.key] = len(apps)
            apps.append(a)
            groups.append(g)
            compats.append(c)
        remap.append(index[a.key])
    device = base.app.device
    tensors = {
        f.name: torch.cat([getattr(base, f.name), getattr(extra, f.name).to(device)])
        for f in dataclasses.fields(Archive) if torch.is_tensor(getattr(base, f.name))
    }
    remap_t = torch.tensor(remap, dtype=torch.long, device=device)
    tensors["app"] = torch.cat([base.app, remap_t[extra.app.to(device)]])
    return Archive(group=groups, compat=compats, apps=apps, **tensors)


def calibration_candidates(infra, app: AppSpec, router: DeployRouter, inserted: Sequence[str]) -> List[AppSpec]:
    """Historical applications of the router (H and M) with a record of an inserted expert, in calibration order.

    Seeded shuffle, then applications sharing the target's CompatKey first and
    round-robin over groups, so the first m form nested, diverse sets.
    """
    archive_keys = {a.key for a in router.bundle.archive_apps}
    cands = [
        b for b in router.bundle.graph_apps
        if b.key in archive_keys and any(infra.store.has(b.data_key, e) for e in inserted)
    ]
    order = torch.randperm(len(cands), generator=torch.Generator().manual_seed(derive_seed(app.seed, "calibration", app.key)))
    target = CompatKey(infra.task_family(app), app.budget).as_tuple()
    seen: Counter = Counter()
    keyed = []
    for i in order.tolist():
        b = cands[i]
        mismatch = CompatKey(infra.task_family(b), b.budget).as_tuple() != target
        keyed.append((mismatch, seen[(mismatch, b.group)], len(keyed), b))
        seen[(mismatch, b.group)] += 1
    return [b for *_, b in sorted(keyed, key=lambda t: t[:3])]


def calibrate(cfg, infra, router: DeployRouter, inserted: Sequence[str], apps: Sequence[AppSpec]) -> DeployRouter:
    """Source calibration: evaluation edges and archive records of the inserted experts on *apps*; router fixed."""
    if not apps:
        return router
    graph = router.graph.to("cpu")
    for e in inserted:
        stats = [infra.store.app_average(b, e) for b in apps]  # (NaN, 0) without a record: skipped
        add_calibration_edges(
            graph, graph.expert_index[e], [graph.app_index[b.key] for b in apps], [m for m, _ in stats], [c for _, c in stats]
        )
    extra = build_archive(list(apps), infra.store, router.bundle.standardizer, cfg, catalog=router.catalog, provider=infra.provider)
    position = {spec.expert_id: i for i, spec in enumerate(router.catalog)}
    ids = torch.tensor([position[e] for e in inserted], dtype=torch.long)
    extra = extra.subset(torch.isin(extra.expert.cpu(), ids))
    return dataclasses.replace(router, graph=graph, archive=extend_archive(router.archive, extra))


class _InsertionSetups:
    """Standard, held-out, and inserted routers of one task, cached per router seed."""

    def __init__(self, cfg, infra):
        self.cfg, self.infra = cfg, infra
        self._cache: Dict[Tuple, Any] = {}

    def _get(self, key, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def standard(self, app: AppSpec) -> DeployRouter:
        return self._get(("standard", router_seed(self.cfg, app)), lambda: prepare_router(self.cfg, app, self.infra))

    def hidden(self, app: AppSpec, tag: str) -> List[str]:
        """Held-out experts of the router of *app* (fixed per group, budget, and router seed)."""
        rseed = router_seed(self.cfg, app)
        if tag == "arch":
            return self._get(("hidden", tag), lambda: architecture_holdout(self.cfg, self.infra))
        seed = derive_seed(rseed, "holdout", app.group, int(app.budget))
        return self._get(("hidden", tag, rseed), lambda: configuration_holdout(self.cfg, self.infra, app, seed))

    def inserted(self, app: AppSpec, tag: str) -> Tuple[DeployRouter, List[str], List[str]]:
        """(router retrained without the held-out experts with those of E_a inserted, hidden, inserted)."""
        hidden = self.hidden(app, tag)
        inserted = [e for e in self.infra.compatible_pool(app) if e in set(hidden)]

        def build():
            base = holdout_router(self.cfg, self.infra, app, hidden, tag)
            return insert_experts(self.cfg, self.infra, base, inserted)

        return self._get(("inserted", tag, router_seed(self.cfg, app)), build), hidden, inserted


def _insertion(cfg, infra, spec: str, budget: int, seeds: Sequence[int]):
    from .baselines.selection.common import QueryGuard
    from .baselines.selection.metagl import MetaGLSelector

    setups = _InsertionSetups(cfg, infra)
    guard = QueryGuard(infra)
    selector = MetaGLSelector(cfg, guard, use_metadata=True)
    rows, details = [], {}
    for seed in seeds:
        app = infra.application(spec, budget, seed)
        details[app.key] = {}
        for condition in INSERTION_CONDITIONS:
            known: Dict[str, float] = {}
            if condition == "new_application":
                router = setups.standard(app)
                hidden, inserted = [], []
            else:
                router, hidden, inserted = setups.inserted(app, "config" if condition == "new_configuration" else "arch")
                if condition != "joint":  # a known application: its evaluations of the seen experts are visible
                    seen = [e for e in router.graph.expert_ids if e not in set(inserted)]
                    router, known = with_known_target(cfg, infra, router, app, seen)
            integ, result = _deploy(cfg, app, infra, router, ["routergfm"])
            error, inserted_error = residual_errors(cfg, app, infra, router, integ, inserted)

            # MetaGL+metadata ranks E_a with the hidden experts' evaluations removed and
            # the same target evaluations visible (a new graph node when there are
            # none); its team (same size) is mixed on the same deployed router.
            guard.target = app
            outcome = selector.rank(app, hidden_experts=hidden, known_mu=known)
            metagl_team = [e for e, _ in outcome.ranking[: len(integ.team)]]
            _, metagl = _deploy(cfg, app, infra, router, ["uniform", "routergfm"], team=metagl_team)

            rows.append(_row("insertion", condition, app, {
                "routergfm_risk": result["rules"]["routergfm"]["risk"],
                "residual_error": error,
                "inserted_residual_error": inserted_error,
                "num_inserted": len(inserted),
                "inserted_in_team": sum(e in set(inserted) for e in integ.team),
                "target_evaluation_edges": len(known),
                "metagl_uniform_risk": metagl["rules"]["uniform"]["risk"],
                "metagl_local_risk": metagl["rules"]["routergfm"]["risk"],
                "metagl_inserted_in_team": sum(e in set(inserted) for e in metagl_team),
            }))
            details[app.key][condition] = {
                "hidden": list(hidden), "inserted": list(inserted), "team": list(integ.team),
                "mu_hat": integ.mu_hat.tolist(), "metagl_team": metagl_team, "rho": integ.rho, "tau": integ.tau,
                "metagl_known_target": outcome.extras["known_target"],
            }
    return rows, details


def _calibration(cfg, infra, spec: str, budget: int, seeds: Sequence[int]):
    sizes = sorted({int(m) for m in cfg.moe.routergfm.analysis.calibration_apps if int(m) >= 0})
    setups = _InsertionSetups(cfg, infra)
    rows, details = [], {}
    for seed in seeds:
        app = infra.application(spec, budget, seed)
        router, _, inserted = setups.inserted(app, "arch")  # joint novelty: new architecture + new application
        cands = calibration_candidates(infra, app, router, inserted)
        details[app.key] = {"inserted": inserted, "candidates": [b.key for b in cands], "teams": {}}
        for m in sizes:
            chosen = cands[:m]
            if len(chosen) < m:
                print(f"{_LOG}[calibration] {app.key}: only {len(chosen)} of {m} calibration applications", flush=True)
            calibrated = calibrate(cfg, infra, router, inserted, chosen)
            integ, result = _deploy(cfg, app, infra, calibrated, ["routergfm"])
            error, inserted_error = residual_errors(cfg, app, infra, calibrated, integ, inserted)
            rows.append(_row("calibration", f"m{m}", app, {
                "risk": result["rules"]["routergfm"]["risk"],
                "residual_error": error,
                "inserted_residual_error": inserted_error,
                "num_calibration_apps": len(chosen),
                "num_inserted": len(inserted),
                "inserted_in_team": sum(e in set(inserted) for e in integ.team),
            }))
            details[app.key]["teams"][f"m{m}"] = list(integ.team)
    return rows, details


# --------------------------------------------------------------------------- #
# Distribution shift (Table 15)
# --------------------------------------------------------------------------- #
_SHIFT_META_KEYS = ("label_tv_support_vs_query", "label_mean_shift", "assay_pos_rate_shift", "target_quantile")


def _shift(cfg, infra, spec: str, budget: int, seeds: Sequence[int]):
    from src.data_loader.shift_splits import split_file_path
    from src.data_loader.utils import safe_torch_load

    root = str(cfg.moe.routergfm.analysis.shift_root)
    conditions = [str(c) for c in cfg.data_preparation.shift.conditions]
    rows, details = [], {}
    for app, router in _seed_apps(cfg, infra, spec, budget, seeds):
        for condition in conditions:
            path = split_file_path(os.path.join(root, condition), app.dataset, app.task_level, app.seed, app.split)
            if not path.is_file():
                print(f"{_LOG}[shift] {app.key}: no {condition} split at {path}; skipped", flush=True)
                continue
            shifted = dataclasses.replace(app, split_root_tag=condition)
            integ, result = _deploy(cfg, shifted, infra, router, ["routergfm", "global"])
            metric = result["metric"]
            metrics = {"num_queries": result["num_queries"]}
            for method, rule in (("routergfm", "routergfm"), ("routergfm_g", "global")):
                metrics[f"{method}_risk"] = result["rules"][rule]["risk"]
                metrics[f"{method}_{metric}"] = result["rules"][rule].get(metric, _NAN)
            meta = safe_torch_load(path).get("meta") or {}
            metrics.update({k: meta[k] for k in _SHIFT_META_KEYS if isinstance(meta.get(k), (int, float))})
            rows.append(_row("shift", condition, app, metrics))
            details[shifted.key] = {"team": list(integ.team), "split_file": str(path)}
    return rows, details


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
_PER_TASK = {
    "team_size": _team_size,
    "archive_reliability": _archive_reliability,
    "insertion": _insertion,
    "calibration": _calibration,
    "shift": _shift,
}


def _units(kind: str, tasks: Sequence[Task]) -> List[Tuple[str, List[Task]]]:
    """(JSON name, tasks) run together: one task each, or all tasks of one budget for ``specialization``."""
    if kind != "specialization":
        return [(_task_name(spec, budget), [(spec, budget)]) for spec, budget in tasks]
    by_budget: "OrderedDict[int, List[Task]]" = OrderedDict()
    for task in tasks:
        by_budget.setdefault(task[1], []).append(task)
    return [(f"b{budget}__{stable_hash(sorted(unit))}", unit) for budget, unit in by_budget.items()]


def run_analysis(cfg, *, provider=None, infra=None) -> int:
    """Run ``analysis.kind`` (or the TSV rows' kinds) on every task; 0 when every task succeeded."""
    try:
        tasks = analysis_tasks(cfg)
    except ValueError as exc:
        print(f"{_LOG} {exc}")
        return 1
    if not tasks:
        print(f"{_LOG} No tasks to run.")
        return 1
    if infra is None:
        from .infra import RouterInfra

        infra = RouterInfra(cfg, provider)
    seeds = benchmark_seeds(cfg)
    ok = True
    for kind, kind_tasks in tasks.items():
        for name, unit in _units(kind, kind_tasks):
            started_at = datetime.now().astimezone()
            try:
                if kind == "specialization":
                    rows, details = _specialization(cfg, infra, unit, seeds)
                else:
                    rows, details = _PER_TASK[kind](cfg, infra, unit[0][0], unit[0][1], seeds)
                written = _write(cfg, kind, name, rows, details, started_at)
                print(f"{_LOG}[{kind}] {name}: {len(rows)} row(s), {written} written", flush=True)
                if not rows:
                    ok = False
            except Exception as exc:  # pylint: disable=broad-except
                traceback.print_exc()
                print(f"{_LOG}[{kind}] Failed {name}: {exc}")
                ok = False
    return 0 if ok else 1


__all__ = [
    "ANALYSIS_WORKFLOW",
    "INSERTION_CONDITIONS",
    "KINDS",
    "PERTURBATION_CONDITIONS",
    "analysis_tasks",
    "architecture_holdout",
    "calibrate",
    "calibration_candidates",
    "configuration_holdout",
    "extend_archive",
    "holdout_router",
    "insert_experts",
    "parse_analysis_tasks",
    "residual_errors",
    "run_analysis",
    "with_known_target",
]
