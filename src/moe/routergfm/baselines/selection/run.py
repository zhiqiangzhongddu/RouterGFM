"""Selection-baseline harness (Table 9 / Fig. 3a): rank E_a, then score the shortlist.

``run_selection_baseline(cfg)`` builds the selector named by
``baselines.method`` over a :class:`~.common.QueryGuard` view of the infra, and
for every task (see :func:`..run.baseline_tasks`) and seed ranks the target's
eligible pool. Only afterwards does it read the target's per-expert risks
(``infra.query_expert_risk``, evaluation only) to compute hit@K / regret@K.
Per-application JSON goes to ``baselines.output_dir/<method>/<fingerprint>/``;
aggregated rows to workflow ``moe_routergfm_selection``.
"""

from __future__ import annotations

import importlib
from datetime import datetime
from typing import Any, Dict, Tuple

from ..run import baseline_seeds, baseline_tasks, config_value, identity_paths, result_dir
from .common import (
    QueryGuard,
    append_selection_rows,
    episode_metrics,
    load_outcome_json,
    save_outcome_json,
)

_PKG = "src.moe.routergfm.baselines.selection"
_LOG = "[RouterGFM][selection]"

# method -> (module, selector class, constructor kwargs); imported on first use.
_SELECTORS: Dict[str, Tuple[str, str, Dict[str, Any]]] = {
    "metadata_mlp": (f"{_PKG}.metadata_mlp", "MetadataMLPSelector", {}),
    "nearest_application": (f"{_PKG}.nearest_application", "NearestApplicationSelector", {}),
    "logme": (f"{_PKG}.logme", "LogMESelector", {}),
    "metagl": (f"{_PKG}.metagl", "MetaGLSelector", {}),
    "metagl_metadata": (f"{_PKG}.metagl", "MetaGLSelector", {"use_metadata": True}),
    "model_spider": (f"{_PKG}.model_spider", "ModelSpiderSelector", {}),
}


def build_selector(method: str, cfg, infra):
    """Instantiate the registered selector ``Selector(cfg, infra, **kwargs)``."""
    if method not in _SELECTORS:
        raise ValueError(f"Unknown selection baseline {method!r}. Available: {', '.join(sorted(_SELECTORS))}")
    module_name, attr, kwargs = _SELECTORS[method]
    return getattr(importlib.import_module(module_name), attr)(cfg, infra, **kwargs)


def _target_risks(infra, app, pool) -> Dict[str, float]:
    """Target risk R_a(e) for every eligible expert with a history record (evaluation only)."""
    risk = {}
    for eid in pool:
        try:
            risk[eid] = float(infra.query_expert_risk(app, eid))
        except KeyError:
            continue
    if len(risk) < len(pool):
        print(f"{_LOG} {app.key}: {len(pool) - len(risk)}/{len(pool)} eligible experts have no target record; "
              "metrics use the recorded ones.")
    return risk


def run_selection_baseline(cfg, *, infra=None) -> int:
    """Rank, evaluate, and tabulate one selection baseline over every task and seed."""
    b = cfg.moe.routergfm.baselines
    method = str(b.method or "").strip().lower()
    if method not in _SELECTORS:
        print(f"{_LOG} Set moe.routergfm.baselines.method to one of {sorted(_SELECTORS)} (got {method!r}).")
        return 1
    tasks = baseline_tasks(cfg, method)
    if not tasks:
        print(f"{_LOG}[{method}] No tasks to run.")
        return 1
    if infra is None:
        from ...infra import RouterInfra

        infra = RouterInfra(cfg)
    module_name, _, kwargs = _SELECTORS[method]
    block = module_name.rsplit(".", 1)[-1]
    k = int(b.topk)
    out_dir = result_dir(cfg, method, block, {"topk": k, "selector_kwargs": kwargs})
    guard = QueryGuard(infra)
    selector = None
    started_at = datetime.now().astimezone()
    per_app, reused = [], 0
    for spec, budget in tasks:
        for seed in baseline_seeds(cfg):
            app = infra.application(spec, budget, seed)
            path = out_dir / f"{app.key}.json"
            if bool(b.skip_if_exists) and path.is_file():
                per_app.append(load_outcome_json(path))
                reused += 1
                continue
            if selector is None:
                selector = build_selector(method, cfg, guard)
            guard.target = app
            try:
                outcome = selector.rank(app)
            finally:
                guard.target = None
            risk = _target_risks(infra, app, infra.compatible_pool(app))
            metrics = episode_metrics(outcome, risk, k)
            family = infra.task_family(app)
            save_outcome_json(outcome, metrics, path, family=family, risk=risk)
            per_app.append((outcome, metrics, family))
            print(f"{_LOG}[{method}] {app.key}: team={outcome.team} hit@{k}={metrics[f'test_hit_at_{k}']:.0f} "
                  f"regret@{k}={metrics[f'test_regret_at_{k}']:.4f}")
    if reused == len(per_app) and not bool(cfg.save_results.save_skipped):
        print(f"{_LOG}[{method}] Every application was reused from {out_dir}; no new result rows.")
        return 0
    append_selection_rows(
        cfg, method, per_app, started_at, datetime.now().astimezone(),
        config={path: config_value(cfg, path) for path in identity_paths(cfg, method, block)},
        config_hash=out_dir.name,
    )
    return 0


__all__ = ["build_selector", "run_selection_baseline"]
