"""Shared helpers for multi-run orchestration and run summaries."""

from __future__ import annotations

import math
import os
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

MetricDict = Dict[str, float]
# Metric keys with this suffix are bookkeeping counters (e.g. masked_count,
# num_pairs_count, batch_size_count). They are dropped from multi-run
# summaries and result TSVs by should_include_summary_metric; the same
# convention is enforced in PretrainRunner's per-epoch stdout.
_COUNT_METRIC_SUFFIX = "_count"


def _normalize_seed_list(raw_seeds: object) -> List[int]:
    if isinstance(raw_seeds, (int, float)):
        return [int(raw_seeds)]
    if raw_seeds is None:
        return []
    return [int(seed) for seed in raw_seeds]


def resolve_seeds(
    cfg: Any,
    requested_count: Optional[int] = None,
) -> List[int]:
    """Resolve ordered seeds from ``cfg.seeds``.

    ``cfg.seeds`` is the only configured seed source.  ``requested_count`` may
    select a prefix of that list, but it must not request more seeds than were
    configured.
    """
    seeds = _normalize_seed_list(getattr(cfg, "seeds", None))
    if not seeds:
        raise ValueError("cfg.seeds must contain at least one seed.")

    if requested_count is None:
        return seeds

    requested_count = int(requested_count)
    if requested_count <= 0:
        raise ValueError(f"requested seed count must be positive when provided (got {requested_count}).")
    if requested_count > len(seeds):
        raise ValueError(
            f"Requested {requested_count} seed(s), but cfg.seeds only provides "
            f"{len(seeds)} seed(s): {seeds}. Add more seeds or lower the requested count."
        )
    return seeds[:requested_count]


def load_checkpoint_metrics(path: str, *, log_prefix: str) -> MetricDict:
    """Load persisted numeric metrics from a checkpoint file when present."""
    if not os.path.isfile(path):
        return {}
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"{log_prefix} Failed to load checkpoint metrics: {path} ({exc})")
        return {}

    metrics = payload.get("metrics", {})
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def checkpoint_path_for_runner(runner) -> str:
    """Resolve a runner checkpoint path even when helper methods are absent."""
    helper = getattr(runner, "get_checkpoint_path_for_metrics", None)
    if callable(helper):
        candidate = helper()
        if candidate:
            return str(candidate)
    return os.path.join(runner.run_dir, f"{runner.run_name}.pt")


def collect_checkpoint_paths(runners) -> List[str]:
    """Deduplicate checkpoint paths from a list of runners."""
    paths: List[str] = []
    seen: set = set()
    for runner in runners:
        path = checkpoint_path_for_runner(runner)
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def collect_run_metrics(runner, *, log_prefix: str) -> MetricDict:
    """Collect best metrics from memory, or fallback to persisted checkpoint."""
    in_memory = {
        key: float(value)
        for key, value in getattr(runner, "best_metrics", {}).items()
        if isinstance(value, (int, float))
    }
    if in_memory:
        return in_memory
    return load_checkpoint_metrics(checkpoint_path_for_runner(runner), log_prefix=log_prefix)


def should_include_summary_metric(
    metric_name: str,
    *,
    excluded_metrics: Optional[set[str]] = None,
) -> bool:
    """Return True when a metric should appear in summaries and saved tables."""
    name = str(metric_name or "").strip()
    if not name:
        return False
    if excluded_metrics and name in excluded_metrics:
        return False
    if name.endswith(_COUNT_METRIC_SUFFIX):
        return False
    return "loss" not in name.lower()


def aggregate_run_metrics(
    run_metrics: Sequence[Mapping[str, float]],
    *,
    excluded_metrics: Optional[set[str]] = None,
) -> dict[str, dict[str, object]]:
    """Aggregate per-run metrics into epoch lists and mean/std/n summaries."""
    excluded = excluded_metrics or set()
    epoch_keys = {"best_epoch", "epoch"}
    values_by_key: Dict[str, List[float]] = {}
    for metrics in run_metrics:
        for key, value in metrics.items():
            if key not in epoch_keys and not should_include_summary_metric(key, excluded_metrics=excluded):
                continue
            if isinstance(value, (int, float)):
                values_by_key.setdefault(key, []).append(float(value))

    epoch_values: Dict[str, List[int]] = {}
    metric_stats: Dict[str, Dict[str, float | int]] = {}
    for key in sorted(values_by_key.keys()):
        values = values_by_key[key]
        if key in epoch_keys:
            epoch_values[key] = [int(round(value)) for value in values]
            continue

        finite_values = [v for v in values if not math.isnan(v)]
        if not finite_values:
            metric_stats[key] = {"mean": float("nan"), "std": 0.0, "n": 0}
            continue
        metric_stats[key] = {
            "mean": statistics.mean(finite_values),
            "std": statistics.stdev(finite_values) if len(finite_values) > 1 else 0.0,
            "n": len(finite_values),
        }

    return {
        "epoch_values": epoch_values,
        "metric_stats": metric_stats,
    }


def summarize_runs(
    run_metrics: Sequence[Mapping[str, float]],
    seeds: Iterable[int],
    *,
    log_prefix: str,
    excluded_metrics: Optional[set[str]] = None,
) -> None:
    """Print a compact multi-run summary."""
    seed_list = [int(seed) for seed in seeds]
    summary = aggregate_run_metrics(run_metrics, excluded_metrics=excluded_metrics)
    epoch_values = summary["epoch_values"]
    metric_stats = summary["metric_stats"]

    print(f"{log_prefix} Completed {len(run_metrics)} runs.")
    print(f"{log_prefix} Seeds: {seed_list}")

    if not epoch_values and not metric_stats:
        print(f"{log_prefix} No metrics available to summarize.")
        return

    for key in sorted(epoch_values.keys()):
        print(f"{log_prefix} {key}: {epoch_values[key]}")
    for key in sorted(metric_stats.keys()):
        stats = metric_stats[key]
        print(
            f"{log_prefix} {key}: mean={float(stats['mean']):.4f} "
            f"std={float(stats['std']):.4f} n={int(stats['n'])}"
        )


def all_runs_skipped(runners) -> bool:
    """Return True if every runner was skipped due to existing checkpoints."""
    if not runners:
        return False
    if not isinstance(runners, (list, tuple)):
        runners = [runners]
    return all(bool(getattr(r, "_skip_due_to_existing_checkpoint", False)) for r in runners)


def should_save_result(runners, cfg) -> bool:
    """Return True if we should persist a result row for these runners."""
    if all_runs_skipped(runners):
        return bool(getattr(getattr(cfg, "save_results", None), "save_skipped", False))
    return True


__all__ = [
    "aggregate_run_metrics",
    "all_runs_skipped",
    "MetricDict",
    "checkpoint_path_for_runner",
    "collect_run_metrics",
    "load_checkpoint_metrics",
    "resolve_seeds",
    "should_include_summary_metric",
    "should_save_result",
    "summarize_runs",
]
