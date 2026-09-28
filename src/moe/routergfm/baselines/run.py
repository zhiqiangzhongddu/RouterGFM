"""Shared matched-pool baseline runner (DESIGN 11).

Tasks are ``(dataset spec, budget)`` pairs from ``baselines.datasets x budgets``
or, with ``baselines.run_tasks_tsv``, from the header TSV ``baselines.tasks_tsv``
(``# method dataset task_level budget``; ``method`` is optional). Each task runs
one ``Runner(cfg, app, infra)`` per seed in ``apps.seeds[:baselines.num_runs]``,
prints a summary, and appends one row to ``outputs/results/moe_<method>.tsv``.
Link prediction runs once per dataset: both budget blocks share one edge split
and report shared results (App. B.3). Per-seed metrics are cached under
``baselines.output_dir/<method>/<fingerprint>/<app.key>.json`` and reused when
``baselines.skip_if_exists``. The fingerprint and the result-row identity cover
every config the method reads: its ``baselines.<block>`` and
:data:`CONFIG_DEPENDENCIES`.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from yacs.config import CfgNode as CN

from src.moe.identity import behavior_fingerprint
from src.utils.checkpoint import cfg_to_dict, save_json_atomic
from src.utils.random import set_seed
from src.utils.run_helpers import aggregate_run_metrics, should_include_summary_metric, summarize_runs
from src.utils.save_results import append_workflow_result, get_explicit_cfg_keys, set_explicit_cfg_keys
from src.utils.tsv_parsing import dedup_tasks, parse_row_by_header, read_tsv_rows

from ..common import parse_dataset_spec
from . import BASELINE_RUNNERS, config_block_name, load_runner_class

_LOG = "[RouterGFM][baselines]"
_TSV_COLUMNS = {"method", "dataset", "task_level", "budget"}
_REQUIRED_COLUMNS = ("dataset", "task_level", "budget")
# Result-table identity columns shared by every matched-pool row (plus the method's identity_paths).
_IDENTITY_KEYS = [
    "moe.method",
    "moe.routergfm.baselines.method",
    "moe.routergfm.baselines.datasets",
    "moe.routergfm.baselines.budgets",
    "moe.routergfm.baselines.topk",
    "moe.routergfm.baselines.candidate_rule",
    "moe.routergfm.baselines.candidate_pool",
]
# Per-row temp TSVs of the SLURM launchers: routing only, never a result-table column.
_ROUTING_KEYS = {"moe.routergfm.baselines.run_tasks_tsv", "moe.routergfm.baselines.tasks_tsv"}
# Config a method reads outside its own ``baselines.<block>`` (paths below ``cfg.moe.routergfm``;
# ``output_root``, i.e. history, embeddings and descriptors, is always fingerprinted). Editing any of
# them re-runs the method instead of reusing its cached results.
CONFIG_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    "metagl_u": ("baselines.metagl", "graph", "heads"),  # MetaGL(+metadata) team; its fitted heads
    "meta_des": ("descriptors", "heads"),  # competence region in z(x); OOF and query heads
    "kdem": ("heads.type", "heads.hidden_dim"),  # the new task head
    "ppem": ("heads.type", "heads.hidden_dim"),
    "metadata_mlp": ("graph",),  # text / numeric metadata blocks
    "nearest_application": ("graph",),
    "metagl_metadata": ("graph",),
    "model_spider": ("descriptors", "router.val_datasets", "router.num_val_datasets"),
}
# Keys of a shared block that only the sibling variant reads (or that never change results), kept out of
# the method's identity: PPEM's ``ema`` never invalidates KDEM results, KDEM's ``kd`` never invalidates
# PPEM results, and ``num_workers`` invalidates neither.
_UNUSED_BLOCK_KEYS: Dict[str, Tuple[str, ...]] = {"kdem": ("ema", "num_workers"), "ppem": ("kd", "num_workers")}


# --------------------------------------------------------------------------- #
# Tasks, seeds, result locations (shared with the selection harness)
# --------------------------------------------------------------------------- #
def _parse_column(col: str, val: str, line_no: int):
    if col == "budget":
        try:
            return int(val), True
        except ValueError:
            print(f"{_LOG} Skipping malformed task row {line_no}: invalid budget '{val}'")
            return None, False
    if col in ("method", "dataset", "task_level"):
        return val.strip().lower(), True
    return None


def parse_baseline_tasks(tsv_path: str, method: Optional[str] = None) -> List[Dict[str, Any]]:
    """Rows ``{method (or None), dataset, task_level, budget}``; with *method*, rows of other methods are dropped."""
    header, data_rows = read_tsv_rows(tsv_path, _TSV_COLUMNS, min_header_columns=3, log_prefix=_LOG)
    if header is None:
        if data_rows:
            print(f"{_LOG} Missing header row; expected '# method dataset task_level budget'.")
        return []
    tasks = []
    for line_no, parts in data_rows:
        task = parse_row_by_header(
            parts, header, line_no, _LOG,
            required_columns=_REQUIRED_COLUMNS,
            defaults={"method": None},
            custom_parser=_parse_column,
        )
        if task is not None and (method is None or task["method"] in (None, method)):
            tasks.append(task)
    return dedup_tasks(tasks, lambda t: (t["method"], t["dataset"], t["task_level"], t["budget"]))


def baseline_tasks(cfg, method: str) -> List[Tuple[str, int]]:
    """``(dataset:level, budget)`` pairs of *method*; link prediction keeps only its first budget."""
    b = cfg.moe.routergfm.baselines
    if bool(b.run_tasks_tsv):
        pairs = [(f"{t['dataset']}:{t['task_level']}", t["budget"]) for t in parse_baseline_tasks(str(b.tasks_tsv), method)]
    else:
        pairs = [(str(spec), int(budget)) for spec in b.datasets for budget in b.budgets]
    out, seen = [], set()
    for spec, budget in pairs:
        name, level = parse_dataset_spec(spec)
        key = (name, level) if level == "edge" else (name, level, int(budget))
        if key in seen:
            if level == "edge":
                print(f"{_LOG} {name}:edge budget {budget}: shared LP result (App. B.3), run once.")
            continue
        seen.add(key)
        out.append((f"{name}:{level}", int(budget)))
    return out


def baseline_seeds(cfg) -> List[int]:
    """Application (split) seeds: ``apps.seeds[:baselines.num_runs]``."""
    seeds = [int(s) for s in cfg.moe.routergfm.apps.seeds]
    n = int(cfg.moe.routergfm.baselines.num_runs)
    if n <= 0 or n > len(seeds):
        raise ValueError(f"baselines.num_runs={n} needs 1..{len(seeds)} seeds from apps.seeds {seeds}.")
    return seeds[:n]


def method_config(cfg, block_name: str) -> CN:
    """``cfg.moe.routergfm.baselines.<block_name>`` (empty when the method has no subtree)."""
    block = cfg.moe.routergfm.baselines.get(block_name)
    return block if isinstance(block, CN) else CN()


def config_value(cfg, path: str) -> Any:
    """Plain (JSON-ready) value of ``cfg.moe.routergfm.<path>``; None when absent."""
    node = cfg.moe.routergfm
    for part in path.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return cfg_to_dict(node) if isinstance(node, dict) else node


def identity_paths(cfg, method: str, block_name: str) -> List[str]:
    """Config paths below ``cfg.moe.routergfm`` that define *method*'s results: its block (the keys
    it reads, when it shares the block with a sibling variant) and :data:`CONFIG_DEPENDENCIES`."""
    block, unused = method_config(cfg, block_name), _UNUSED_BLOCK_KEYS.get(method, ())
    if not block:
        own = []
    elif unused:
        own = [f"baselines.{block_name}.{key}" for key in block if key not in unused]
    else:
        own = [f"baselines.{block_name}"]
    return own + list(CONFIG_DEPENDENCIES.get(method, ()))


def result_dir(cfg, method: str, block_name: str, external: Mapping[str, Any]) -> Path:
    """``baselines.output_dir/<method>/<fingerprint>``: the fingerprint covers the method subtree
    (minus its sibling variant's keys), its :data:`CONFIG_DEPENDENCIES`, ``output_root`` (history,
    heads), and the caller's shared behavior keys."""
    rg = cfg.moe.routergfm
    unused = _UNUSED_BLOCK_KEYS.get(method, ())
    block = {k: v for k, v in cfg_to_dict(method_config(cfg, block_name)).items() if k not in unused}
    payload = {"method": method, "output_root": str(rg.output_root), **dict(external)}
    dependencies = {path: config_value(cfg, path) for path in CONFIG_DEPENDENCIES.get(method, ())}
    if dependencies:
        payload["dependencies"] = dependencies
    fingerprint = behavior_fingerprint(block, external_behavior=payload)
    return Path(str(rg.baselines.output_dir)) / method / fingerprint


# --------------------------------------------------------------------------- #
# Matched-pool runs
# --------------------------------------------------------------------------- #
def runner_metrics(runner) -> Dict[str, float]:
    """``best_metrics`` (else ``evaluate()`` keys prefixed ``test_``) plus ``best_epoch`` when set."""
    metrics = {
        k: float(v) for k, v in (getattr(runner, "best_metrics", None) or {}).items() if isinstance(v, (int, float))
    }
    if not metrics:
        metrics = {
            (k if k.startswith("test_") else f"test_{k}"): float(v)
            for k, v in runner.evaluate().items()
            if isinstance(v, (int, float))
        }
    best_epoch = getattr(runner, "best_epoch", None)
    if best_epoch is not None:
        metrics["best_epoch"] = int(best_epoch)
    dropped = [k for k in metrics if k != "best_epoch" and not should_include_summary_metric(k)]
    if dropped:
        print(f"{_LOG} Metrics {dropped} are not saved: result-table names must not contain 'loss' or end in '_count'.")
    return metrics


def _task_cfg(cfg, method: str, spec: str, budget: int):
    run_cfg = cfg.clone()
    b = run_cfg.moe.routergfm.baselines
    b.method = method
    b.datasets = [spec]
    b.budgets = [int(budget)]
    b.run_tasks_tsv = False
    keys = [k for k in get_explicit_cfg_keys(cfg) if k not in _ROUTING_KEYS] + _IDENTITY_KEYS
    keys += [f"moe.routergfm.{path}" for path in identity_paths(cfg, method, config_block_name(method))]
    set_explicit_cfg_keys(run_cfg, keys)
    return run_cfg


def _run_task(cfg, method: str, runner_cls, infra, spec: str, budget: int) -> None:
    started_at = datetime.now().astimezone()
    task_cfg = _task_cfg(cfg, method, spec, budget)
    b = cfg.moe.routergfm.baselines
    out_dir = result_dir(
        cfg, method, config_block_name(method),
        {"topk": int(b.topk), "candidate_rule": str(b.candidate_rule), "candidate_pool": int(b.candidate_pool)},
    )
    seeds = baseline_seeds(cfg)
    run_metrics, reused = [], 0
    for seed in seeds:
        app = infra.application(spec, budget, seed)
        path = out_dir / f"{app.key}.json"
        if bool(b.skip_if_exists) and path.is_file():
            with open(path, "r", encoding="utf-8") as fh:
                metrics = json.load(fh)["metrics"]
            print(f"{_LOG}[{method}] {app.key}: reusing {path}")
            reused += 1
        else:
            run_cfg = task_cfg.clone()
            run_cfg.seed = int(seed)
            set_seed(int(seed))
            runner = runner_cls(run_cfg, app, infra)
            runner.fit()
            metrics = runner_metrics(runner)
            save_json_atomic(str(path), {"method": method, "app": app.to_dict(), "metrics": metrics})
        run_metrics.append(metrics)
    summarize_runs(run_metrics, seeds, log_prefix=f"{_LOG}[{method}][{spec} b{budget}]")
    if reused == len(seeds) and not bool(cfg.save_results.save_skipped):
        print(f"{_LOG}[{method}] {spec} b{budget}: every seed was reused; no new result row (save_results.save_skipped).")
        return
    ended_at = datetime.now().astimezone()
    summary = aggregate_run_metrics(run_metrics)
    append_workflow_result(
        cfg=task_cfg,
        workflow=f"moe_{method}",
        started_at=started_at,
        ended_at=ended_at,
        checkpoint_save_paths=[],
        seeds=seeds,
        best_epochs=summary["epoch_values"].get("best_epoch"),
        metric_summary=summary["metric_stats"],
    )


def run_matched_baseline(cfg, method: str, runner_cls=None, *, infra=None) -> int:
    """Run *method* on every task and seed; one ``moe_<method>`` result row per task."""
    runner_cls = runner_cls if runner_cls is not None else load_runner_class(method)
    tasks = baseline_tasks(cfg, method)
    if not tasks:
        print(f"{_LOG}[{method}] No tasks to run.")
        return 1
    if infra is None:
        from ..infra import RouterInfra

        infra = RouterInfra(cfg)
    ok = True
    for spec, budget in tasks:
        try:
            _run_task(cfg, method, runner_cls, infra, spec, budget)
        except Exception as exc:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"{_LOG}[{method}] Failed {spec} budget {budget}: {exc}")
            ok = False
    return 0 if ok else 1


def run_matched_baseline_from_cfg(cfg, *, infra=None) -> int:
    """Dispatch on ``baselines.method``; if empty in TSV mode, run every registered method the TSV lists."""
    b = cfg.moe.routergfm.baselines
    method = str(b.method or "").strip().lower()
    if method:
        return run_matched_baseline(cfg, method, infra=infra)
    if not bool(b.run_tasks_tsv):
        print(f"{_LOG} Set moe.routergfm.baselines.method ({', '.join(sorted(BASELINE_RUNNERS))}).")
        return 1
    listed = list(dict.fromkeys(t["method"] for t in parse_baseline_tasks(str(b.tasks_tsv))))
    methods = [m for m in listed if m in BASELINE_RUNNERS]
    ignored = [m for m in listed if m not in BASELINE_RUNNERS]
    if ignored:
        print(f"{_LOG} Ignoring TSV rows of non-matched methods: {ignored}")
    if not methods:
        print(f"{_LOG} No matched-pool methods in {b.tasks_tsv}.")
        return 1
    if infra is None:
        from ..infra import RouterInfra

        infra = RouterInfra(cfg)
    results = [run_matched_baseline(cfg, m, infra=infra) for m in methods]
    return 0 if all(r == 0 for r in results) else 1


__all__ = [
    "CONFIG_DEPENDENCIES",
    "baseline_seeds",
    "baseline_tasks",
    "config_value",
    "identity_paths",
    "method_config",
    "parse_baseline_tasks",
    "result_dir",
    "run_matched_baseline",
    "run_matched_baseline_from_cfg",
    "runner_metrics",
]
