"""Shared selection-baseline harness pieces (Table 9 / Fig. 3a; DESIGN 11).

* :class:`SelectionOutcome` — what ``Selector(cfg, infra).rank(app)`` returns.
* :func:`selection_metrics` — hit@K, regret@K, and the rank of a best eligible expert.
* :class:`QueryGuard` — the infra view selectors receive: evaluation helpers and
  the target group's query-side data raise :class:`QueryAccessError`.
* JSON persistence and :func:`append_selection_rows` (workflow ``moe_routergfm_selection``).
* Label-free metadata helpers shared by the metadata selectors
  (:func:`application_averages`, :class:`ColumnScaler`, :func:`block_concat`,
  :func:`metadata_blocks`).
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from src.utils.checkpoint import save_json_atomic
from src.utils.run_helpers import aggregate_run_metrics
from src.utils.save_results import append_workflow_result_rows

from ...common import GRAPH_CLS, NODE_CLS, AppSpec, base_group

SELECTION_WORKFLOW = "moe_routergfm_selection"
# Table 9 scope: node and single-label graph classification.
TABLE9_FAMILIES = (NODE_CLS, GRAPH_CLS)
TABLE9_ROW = "table9_all"


# --------------------------------------------------------------------------- #
# Outcome and metrics
# --------------------------------------------------------------------------- #
@dataclass
class SelectionOutcome:
    """Ranking of E_a for one target application (scores: higher is better)."""

    app: AppSpec
    ranking: List[Tuple[str, float]]  # (expert_id, score) over E_a, best first
    team: List[str]  # top-K
    num_target_executions: int = 0  # frozen forward passes over the target support
    wall_time_sec: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)  # method diagnostics (JSON-serializable)
    variants: Dict[str, List[str]] = field(default_factory=dict)  # extra named teams scored like ``team``

    @classmethod
    def from_ranking(cls, app: AppSpec, ranking: Sequence[Tuple[str, float]], k: int, **kwargs) -> "SelectionOutcome":
        ranking = [(str(e), float(s)) for e, s in ranking]
        return cls(app=app, ranking=ranking, team=[e for e, _ in ranking[: int(k)]], **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["app"] = self.app.to_dict()
        payload["ranking"] = [[e, s] for e, s in self.ranking]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SelectionOutcome":
        return cls(
            app=AppSpec.from_dict(payload["app"]),
            ranking=[(str(e), float(s)) for e, s in payload["ranking"]],
            team=[str(e) for e in payload["team"]],
            num_target_executions=int(payload.get("num_target_executions", 0)),
            wall_time_sec=float(payload.get("wall_time_sec", 0.0)),
            extras=dict(payload.get("extras", {})),
            variants={k: [str(e) for e in v] for k, v in payload.get("variants", {}).items()},
        )


def selection_metrics(
    ranking: Sequence[Union[str, Tuple[str, float]]],
    risk: Mapping[str, float],
    k: int,
    atol: float = 1e-12,
) -> Dict[str, float]:
    """Paper Sec. 4.1 selection diagnostics against per-expert risks over E_a.

    ``ranking``: expert ids (or ``(id, score)``) best first; the shortlist is its
    first ``k`` entries. ``risk``: expert -> risk; non-finite entries are ignored.
    A best eligible expert is any within ``atol`` of the minimum risk (ties count).

    * ``hit_at_k``: 1 if the shortlist contains a best eligible expert, else 0.
    * ``regret_at_k``: best shortlisted risk minus best eligible risk (0 within ``atol``).
    * ``best_rank``: 1-based position in ``ranking`` of the first best eligible
      expert (NaN if none is listed).
    """
    ids = [e if isinstance(e, str) else e[0] for e in ranking]
    finite = {e: float(r) for e, r in risk.items() if r is not None and math.isfinite(float(r))}
    nan = float("nan")
    if not finite:
        return {"hit_at_k": nan, "regret_at_k": nan, "best_rank": nan}
    best = min(finite.values())
    team_risk = [finite[e] for e in ids[: int(k)] if e in finite]
    gap = min(team_risk) - best if team_risk else nan
    best_rank = next((i + 1 for i, e in enumerate(ids) if e in finite and finite[e] <= best + atol), nan)
    return {
        "hit_at_k": float(bool(team_risk) and gap <= atol),
        "regret_at_k": (0.0 if gap <= atol else gap) if team_risk else nan,
        "best_rank": float(best_rank),
    }


def episode_metrics(outcome: SelectionOutcome, risk: Mapping[str, float], k: int) -> Dict[str, float]:
    """Flat per-application metrics as reported in the results table (no name contains 'loss')."""
    m = selection_metrics([e for e, _ in outcome.ranking] or outcome.team, risk, k)
    out = {
        f"test_hit_at_{k}": m["hit_at_k"],
        f"test_regret_at_{k}": m["regret_at_k"],
        "test_best_rank": m["best_rank"],
        "num_target_executions": float(outcome.num_target_executions),
        "wall_time_sec": float(outcome.wall_time_sec),
    }
    for name, team in outcome.variants.items():
        mv = selection_metrics(team, risk, k)
        out[f"test_hit_at_{k}_{name}"] = mv["hit_at_k"]
        out[f"test_regret_at_{k}_{name}"] = mv["regret_at_k"]
    return out


# --------------------------------------------------------------------------- #
# Query guard
# --------------------------------------------------------------------------- #
class QueryAccessError(RuntimeError):
    """A selector touched evaluation-only or target query-side data."""


class _GuardedStore:
    """HistoryStore view refusing every record of the guarded target group."""

    _BY_APP = ("losses", "preds", "app_average")

    def __init__(self, store, guard: "QueryGuard"):
        self._store = store
        self._guard = guard

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._store, name)
        if not callable(attr):
            return attr

        def checked(key, *args, **kwargs):
            data_key = key.data_key if name in self._BY_APP else str(key)
            self._guard._check_group(base_group(data_key.split("__", 1)[0]), f"store.{name}")
            return attr(key, *args, **kwargs)

        return checked


class QueryGuard:
    """``RouterInfra`` view handed to selectors (they never read query data).

    Evaluation helpers always raise. While ``target`` is set, applications of the
    target's group expose only label-free and support-side data: ``data`` keeps
    support labels only, ``embeddings`` only the support split, and history
    records, fitted query predictions, and ``historical_mu`` raise.
    """

    _EVAL_ONLY = frozenset({"query_expert_risk", "evaluate_outputs", "provider"})
    _TARGET_BLOCKED = frozenset({"historical_mu", "expert_predictions"})

    def __init__(self, infra):
        self._infra = infra
        self.target: Optional[AppSpec] = None

    def _check_group(self, group: str, what: str) -> None:
        if self.target is not None and group == self.target.group:
            raise QueryAccessError(f"{what} on the target group {group!r} is not available to selectors.")

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        if name.startswith("_") or name in self._EVAL_ONLY:
            raise QueryAccessError(f"RouterInfra.{name} is not available to selectors.")
        if name == "store":
            return _GuardedStore(self._infra.store, self)
        attr = getattr(self._infra, name)
        if name in self._TARGET_BLOCKED:
            def blocked(app, *args, **kwargs):
                self._check_group(app.group, name)
                return attr(app, *args, **kwargs)

            return blocked
        if name == "embeddings":
            def support_only(app, expert_id, split):
                if split != "support":
                    self._check_group(app.group, f"embeddings(split={split!r})")
                return attr(app, expert_id, split)

            return support_only
        if name == "data":
            def support_labels_only(app):
                data = attr(app)
                if self.target is None or app.group != self.target.group:
                    return data
                return dataclasses.replace(data, labels={"support": data.labels["support"]})

            return support_labels_only
        return attr


# --------------------------------------------------------------------------- #
# Persistence and results table
# --------------------------------------------------------------------------- #
def save_outcome_json(
    outcome: SelectionOutcome,
    metrics: Mapping[str, float],
    path: Path,
    *,
    family: str,
    risk: Mapping[str, float],
) -> None:
    payload = {"outcome": outcome.to_dict(), "family": family, "metrics": dict(metrics), "risk": dict(risk)}
    save_json_atomic(str(path), payload)


def load_outcome_json(path: Path) -> Tuple[SelectionOutcome, Dict[str, float], str]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return SelectionOutcome.from_dict(payload["outcome"]), dict(payload["metrics"]), str(payload["family"])


def append_selection_rows(
    cfg,
    method: str,
    per_app: Sequence[Tuple[SelectionOutcome, Mapping[str, float], str]],
    started_at: datetime,
    ended_at: datetime,
    *,
    config: Optional[Mapping[str, Any]] = None,
    config_hash: str = "",
) -> int:
    """One row per (dataset, task level, budget) plus a ``table9_all`` row per budget.

    ``per_app``: ``(outcome, episode metrics, task family)``. Rows carry
    ``<metric>_mean/_std`` over the applications (``aggregate_run_metrics``),
    ``n_apps``, and the seeds. The ``table9_all`` row pools node and
    single-label graph classification applications (Table 9 scope).
    """
    k = int(cfg.moe.routergfm.baselines.topk)
    groups: Dict[Tuple[str, str, int], List[Tuple[SelectionOutcome, Mapping[str, float], str]]] = {}
    for item in per_app:
        app = item[0].app
        groups.setdefault((app.dataset, app.task_level, int(app.budget)), []).append(item)
    for budget in sorted({key[2] for key in groups}):
        pooled = [it for key, items in groups.items() if key[2] == budget for it in items if it[2] in TABLE9_FAMILIES]
        if pooled:
            groups[(TABLE9_ROW, "", budget)] = pooled

    identity = {
        "moe.routergfm.baselines.method": method,
        "topk": k,
        "config": json.dumps(dict(config or {}), sort_keys=True, default=str),
        "config_hash": config_hash,
    }
    rows = []
    for (dataset, level, budget), items in groups.items():
        stats = aggregate_run_metrics([m for _, m, _ in items])["metric_stats"]
        row = {**identity, "dataset": dataset, "task_level": level, "budget": budget, "n_apps": len(items)}
        row["seeds"] = sorted({int(o.app.seed) for o, _, _ in items})
        for name, s in stats.items():
            row[f"{name}_mean"] = s["mean"]
            row[f"{name}_std"] = s["std"]
        rows.append(row)
    return append_workflow_result_rows(
        cfg=cfg, workflow=SELECTION_WORKFLOW, rows=rows, started_at=started_at, ended_at=ended_at
    )


# --------------------------------------------------------------------------- #
# Label-free metadata helpers
# --------------------------------------------------------------------------- #
def application_averages(infra, apps: Sequence[AppSpec], expert_ids: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. 2 averages ``mu [B, E]`` (NaN where unobserved) and valid counts ``[B, E]`` from the history."""
    mu = torch.full((len(apps), len(expert_ids)), float("nan"))
    count = torch.zeros((len(apps), len(expert_ids)), dtype=torch.long)
    for i, app in enumerate(apps):
        for j, eid in enumerate(expert_ids):
            m, c = infra.historical_mu(app, eid)
            if c > 0 and math.isfinite(m):
                mu[i, j], count[i, j] = m, c
    return mu, count


class ColumnScaler:
    """NaN-aware per-column scaling fitted on training rows; missing entries map to 0.

    ``standard``: z-score (constant columns keep scale 1); ``minmax``: [0, 1]
    with clipping (constant columns map to 0).
    """

    def __init__(self, kind: str = "standard"):
        if kind not in ("standard", "minmax"):
            raise ValueError(f"Unknown scaler kind {kind!r}")
        self.kind = kind
        self.shift: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor) -> "ColumnScaler":
        X = X.float()
        finite = torch.isfinite(X)
        if self.kind == "standard":
            n = finite.sum(0).clamp(min=1)
            mean = torch.where(finite, X, torch.zeros_like(X)).sum(0) / n
            var = torch.where(finite, (X - mean) ** 2, torch.zeros_like(X)).sum(0) / n
            std = var.sqrt()
            self.shift, self.scale = mean, torch.where(std > 1e-6, std, torch.ones_like(std))
        else:
            inf = torch.full_like(X, float("inf"))
            low = torch.where(finite, X, inf).amin(0)
            high = torch.where(finite, X, -inf).amax(0)
            span = high - low
            ok = torch.isfinite(span) & (span > 1e-12)
            self.shift = torch.where(torch.isfinite(low), low, torch.zeros_like(low))
            self.scale = torch.where(ok, span, torch.full_like(span, float("inf")))
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        Z = (X.float() - self.shift) / self.scale
        if self.kind == "minmax":
            Z = Z.clamp(0.0, 1.0)
        return torch.where(torch.isfinite(Z), Z, torch.zeros_like(Z))


def block_concat(numeric: Optional[torch.Tensor], text: Optional[torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2-normalize each present block, then concatenate (equal block weight for cosine)."""
    blocks = [b / b.norm(dim=-1, keepdim=True).clamp_min(eps) for b in (numeric, text) if b is not None]
    return torch.cat(blocks, dim=-1)


def metadata_blocks(x: torch.Tensor, numeric_dim: int, cfg) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Split raw ``text ⊕ numeric`` metadata (``infra.app_metadata`` / ``expert_metadata``) into its blocks."""
    graph_cfg = cfg.moe.routergfm.graph
    n_num = int(numeric_dim) if graph_cfg.use_numeric else 0
    numeric = x[..., x.size(-1) - n_num:] if n_num else None
    text = x[..., : x.size(-1) - n_num] if graph_cfg.use_text else None
    return numeric, text


__all__ = [
    "ColumnScaler",
    "QueryAccessError",
    "QueryGuard",
    "SELECTION_WORKFLOW",
    "SelectionOutcome",
    "TABLE9_FAMILIES",
    "TABLE9_ROW",
    "append_selection_rows",
    "application_averages",
    "block_concat",
    "episode_metrics",
    "load_outcome_json",
    "metadata_blocks",
    "save_outcome_json",
    "selection_metrics",
]
