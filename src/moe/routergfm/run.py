"""RouterGFM stages, dispatched on ``cfg.moe.routergfm.task``.

* ``history``: record heads/losses on D_a for this expert shard (``experts.shard_index/num_shards``).
* ``descriptors``: label-free context descriptors of every declared data key.
* ``router``: train the router of every (target, budget) task (and seed with ``router.per_seed``).
* ``deploy``: one application (``deploy.target/budget/seed``); writes ``deploy_dir/<app>/deploy.json``.
* ``benchmark``: per (target, budget) task, deploy on seeds ``apps.seeds[:benchmark.num_runs]``
  and append mean/std rows per method (``routergfm``, ``routergfm_g`` = rule ``global``,
  ``fixed_team:<rule>``) to the ``moe_routergfm`` results table; other
  ``benchmark.methods`` entries run as matched-pool baselines.
* ``analysis`` / ``selection_baseline`` / ``matched_baseline``: dispatched to their packages.

Tasks: ``benchmark.tasks_tsv`` rows (``# dataset task_level budget``) when
``benchmark.run_tasks_tsv``; else every ``apps.targets`` x ``apps.budgets`` when
``deploy.target == 'all'``; else ``(deploy.target, deploy.budget)``.
"""

from __future__ import annotations

import importlib
import traceback
from collections import OrderedDict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.utils.checkpoint import cfg_to_dict
from src.utils.run_helpers import aggregate_run_metrics, summarize_runs
from src.utils.save_results import append_workflow_result_rows
from src.utils.tsv_parsing import dedup_tasks, parse_row_by_header, read_tsv_rows

from .common import enumerate_applications, parse_dataset_spec, reported_metric, stable_hash

BENCHMARK_WORKFLOW = "moe_routergfm"
_LOG = "[RouterGFM]"
_TSV_COLUMNS = {"dataset", "task_level", "budget"}


# --------------------------------------------------------------------------- #
# Tasks and seeds
# --------------------------------------------------------------------------- #
def _parse_column(col: str, val: str, line_no: int):
    if col == "budget":
        try:
            return int(val), True
        except ValueError:
            print(f"{_LOG} Skipping malformed task row {line_no}: invalid budget '{val}'")
            return None, False
    if col in ("dataset", "task_level"):
        return val.strip().lower(), True
    return None


def parse_benchmark_tasks(tsv_path: str) -> List[Tuple[str, int]]:
    """``(dataset:level, budget)`` rows of a header TSV ``# dataset task_level budget``."""
    header, data_rows = read_tsv_rows(tsv_path, _TSV_COLUMNS, min_header_columns=3, log_prefix=_LOG)
    if header is None:
        if data_rows:
            print(f"{_LOG} Missing header row; expected '# dataset task_level budget'.")
        return []
    tasks = []
    for line_no, parts in data_rows:
        row = parse_row_by_header(
            parts, header, line_no, _LOG, required_columns=("dataset", "task_level", "budget"), custom_parser=_parse_column
        )
        if row is not None:
            tasks.append(row)
    tasks = dedup_tasks(tasks, lambda t: (t["dataset"], t["task_level"], t["budget"]))
    return [(f"{t['dataset']}:{t['task_level']}", int(t["budget"])) for t in tasks]


def routergfm_tasks(cfg) -> List[Tuple[str, int]]:
    """(target spec, budget) pairs of the router and benchmark stages."""
    rg = cfg.moe.routergfm
    if bool(rg.benchmark.run_tasks_tsv):
        return parse_benchmark_tasks(str(rg.benchmark.tasks_tsv))
    if str(rg.deploy.target).strip().lower() == "all":
        return [(str(spec), int(b)) for spec in rg.apps.targets for b in rg.apps.budgets]
    return [(str(rg.deploy.target), int(rg.deploy.budget))]


def benchmark_seeds(cfg) -> List[int]:
    """Application (split) seeds ``apps.seeds[:benchmark.num_runs]``."""
    rg = cfg.moe.routergfm
    seeds, n = [int(s) for s in rg.apps.seeds], int(rg.benchmark.num_runs)
    if n <= 0 or n > len(seeds):
        raise ValueError(f"benchmark.num_runs={n} needs 1..{len(seeds)} seeds from apps.seeds {seeds}.")
    return seeds[:n]


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def _run_history(cfg, provider) -> int:
    from .history import generate_history

    generate_history(cfg, provider)
    return 0


def _run_descriptors(cfg, provider) -> int:
    from .applications import RealDataProvider
    from .descriptors import ensure_descriptors

    provider = provider if provider is not None else RealDataProvider(cfg)
    apps = OrderedDict((a.data_key, a) for a in enumerate_applications(cfg.moe.routergfm))
    for key, app in apps.items():
        ensure_descriptors(cfg, provider.load(app))
        print(f"{_LOG}[descriptors] {key}", flush=True)
    return 0


def _run_router(cfg, provider) -> int:
    from .router.trainer import train_router

    rg = cfg.moe.routergfm
    seeds = benchmark_seeds(cfg) if bool(rg.router.per_seed) else [int(rg.router.seed)]
    tasks = routergfm_tasks(cfg)
    if not tasks:
        print(f"{_LOG}[router] No tasks to run.")
        return 1
    for spec, budget in tasks:
        for seed in seeds:
            print(f"{_LOG}[router] {spec} budget {budget} seed {seed}: {train_router(cfg, spec, budget, provider, seed)}")
    return 0


def _run_deploy(cfg, provider) -> int:
    from .deploy import deploy_application
    from .infra import RouterInfra

    d = cfg.moe.routergfm.deploy
    infra = RouterInfra(cfg, provider)
    deploy_application(cfg, infra.application(str(d.target), int(d.budget), int(d.seed)), infra=infra)
    return 0


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def benchmark_methods(cfg) -> List[Tuple[str, str]]:
    """``(method, rule)`` rows reported per task: RouterGFM, RouterGFM-G, and every fixed-team rule."""
    rules = [str(r) for r in cfg.moe.routergfm.integration.rules]
    return [("routergfm", "routergfm"), ("routergfm_g", "global")] + [(f"fixed_team:{r}", r) for r in rules]


def _run_metrics(result: Dict[str, Any], rule: str) -> Dict[str, float]:
    """Per-seed metrics of one rule plus the team's selection diagnostics, ``test_``-prefixed."""
    sel, k = result["selection"], int(result["selection"]["k"])
    metrics = {f"test_{name}": float(v) for name, v in result["rules"][rule].items()}
    metrics.update({
        f"test_hit_at_{k}": float(sel["hit_at_k"]),
        f"test_regret_at_{k}": float(sel["regret_at_k"]),
        "test_winner_coverage": float(sel["winner_coverage"]),
        "test_specialization_index": float(sel["specialization_index"]),
    })
    return metrics


def _config_hash(cfg) -> str:
    rg = cfg.moe.routergfm
    blocks = ("experts", "apps", "heads", "loss", "descriptors", "archive", "graph", "router", "integration")
    return stable_hash({k: cfg_to_dict(rg[k]) for k in blocks})


def _benchmark_task(cfg, infra, spec: str, budget: int, seeds: List[int]) -> int:
    """Deploy on every seed of one (target, budget) task and append one row per method."""
    from .deploy import deploy_application, prepare_router, router_seed

    started_at = datetime.now().astimezone()
    methods = benchmark_methods(cfg)
    rules = list(dict.fromkeys(rule for _, rule in methods))
    per_method: Dict[str, List[Dict[str, float]]] = {m: [] for m, _ in methods}
    router, router_key, family = None, None, ""
    for seed in seeds:
        app = infra.application(spec, budget, seed)
        rseed = router_seed(cfg, app)
        if rseed != router_key:  # one router (with its H and M) per task unless router.per_seed
            router, router_key = prepare_router(cfg, app, infra), rseed
        result = deploy_application(cfg, app, infra=infra, router=router, rules=rules)
        family = result["family"]
        for method, rule in methods:
            per_method[method].append(_run_metrics(result, rule))
    ended_at = datetime.now().astimezone()

    name, level = parse_dataset_spec(spec)
    rows = []
    for method, runs in per_method.items():
        summarize_runs(runs, seeds, log_prefix=f"{_LOG}[benchmark][{spec} b{budget}][{method}]")
        row = {
            "method": method,
            "dataset": name,
            "task_level": level,
            "budget": int(budget),
            "metric": f"test_{reported_metric(family)}",
            "seeds": list(seeds),
            "n_runs": len(runs),
            "output_root": str(cfg.moe.routergfm.output_root),
            "config_hash": _config_hash(cfg),
        }
        for metric, stats in aggregate_run_metrics(runs)["metric_stats"].items():
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        rows.append(row)
    return append_workflow_result_rows(
        cfg=cfg, workflow=BENCHMARK_WORKFLOW, rows=rows, started_at=started_at, ended_at=ended_at
    )


def _run_matched_for_task(cfg, method: str, spec: str, budget: int, infra) -> int:
    """A matched-pool baseline listed in ``benchmark.methods`` on one task with the benchmark seeds."""
    from .baselines.run import run_matched_baseline

    run_cfg = cfg.clone()
    b = run_cfg.moe.routergfm.baselines
    b.method, b.datasets, b.budgets = method, [spec], [int(budget)]
    b.run_tasks_tsv = False
    b.num_runs = int(cfg.moe.routergfm.benchmark.num_runs)
    return int(run_matched_baseline(run_cfg, method, infra=infra))


def run_benchmark(cfg, provider=None, *, infra=None) -> int:
    """Multi-seed deployment of every task; one ``moe_routergfm`` row per (task, method)."""
    rg = cfg.moe.routergfm
    methods = list(dict.fromkeys(str(m).strip().lower() for m in rg.benchmark.methods))
    tasks, seeds = routergfm_tasks(cfg), benchmark_seeds(cfg)
    if not tasks or not methods:
        print(f"{_LOG}[benchmark] No tasks or methods to run.")
        return 1
    if infra is None:
        from .infra import RouterInfra

        infra = RouterInfra(cfg, provider)
    ok = True
    for spec, budget in tasks:
        for method in methods:
            try:
                if method == "routergfm":
                    written = _benchmark_task(cfg, infra, spec, budget, seeds)
                    print(f"{_LOG}[benchmark] {spec} budget {budget}: {written} result row(s) written")
                elif _run_matched_for_task(cfg, method, spec, budget, infra) != 0:
                    ok = False
            except Exception as exc:  # pylint: disable=broad-except
                traceback.print_exc()
                print(f"{_LOG}[benchmark] Failed {method} on {spec} budget {budget}: {exc}")
                ok = False
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Packages written separately
# --------------------------------------------------------------------------- #
def _external(module: str, attr: str, what: str) -> Callable[..., int]:
    try:
        return getattr(importlib.import_module(module), attr)
    except ModuleNotFoundError as exc:
        if exc.name is not None and (module == exc.name or module.startswith(exc.name + ".")):
            raise RuntimeError(f"RouterGFM {what} is not available: module {module} is missing.") from exc
        raise


def _infra_of(cfg, provider):
    """A ``RouterInfra`` over *provider*, or ``None`` so the package builds its own on real data."""
    if provider is None:
        return None
    from .infra import RouterInfra

    return RouterInfra(cfg, provider)


def _run_analysis(cfg, provider) -> int:
    return int(_external("src.moe.routergfm.analysis", "run_analysis", "analysis")(cfg, provider=provider))


def _run_selection_baseline(cfg, provider) -> int:
    fn = _external("src.moe.routergfm.baselines.selection.run", "run_selection_baseline", "selection baselines")
    return int(fn(cfg, infra=_infra_of(cfg, provider)))


def _run_matched_baseline(cfg, provider) -> int:
    fn = _external("src.moe.routergfm.baselines.run", "run_matched_baseline_from_cfg", "matched-pool baselines")
    return int(fn(cfg, infra=_infra_of(cfg, provider)))


_TASKS: Dict[str, Callable[[Any, Optional[Any]], int]] = {
    "history": _run_history,
    "descriptors": _run_descriptors,
    "router": _run_router,
    "deploy": _run_deploy,
    "benchmark": lambda cfg, provider: run_benchmark(cfg, provider),
    "analysis": _run_analysis,
    "selection_baseline": _run_selection_baseline,
    "matched_baseline": _run_matched_baseline,
}


def run_routergfm(cfg, *, provider=None) -> int:
    """Run the stage named by ``cfg.moe.routergfm.task``; ``provider`` overrides the real data provider."""
    task = str(cfg.moe.routergfm.task).strip().lower()
    if task not in _TASKS:
        print(f"{_LOG} Unknown task {task!r}. Available: {', '.join(_TASKS)}")
        return 1
    return int(_TASKS[task](cfg, provider))


__all__ = [
    "BENCHMARK_WORKFLOW",
    "benchmark_methods",
    "benchmark_seeds",
    "parse_benchmark_tasks",
    "routergfm_tasks",
    "run_benchmark",
    "run_routergfm",
]
