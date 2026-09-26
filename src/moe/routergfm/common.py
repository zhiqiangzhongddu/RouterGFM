"""Shared RouterGFM vocabulary: applications, experts, task families, artifact layout.

Terminology follows the paper (Sec. 3.1-3.2):

* An *application* ``a`` is a graph dataset, a prediction task, and a labeled
  support set ``S_a``. Here it is ``(dataset, task_level, budget, seed)``; the
  seed fixes the split, the budget the support size.
* A *data key* identifies the concrete split files an application reads.
  Link-prediction applications of both budget blocks share one data key
  because the routed LP convention (App. B.3) uses one positive-edge split.
* A *group* is the canonical base dataset. Tasks, budgets, and splits of one
  base dataset are always masked, validated, and excluded together.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.utils.dataset_helpers import canonical_source_dataset_name
from src.utils.naming import format_split_for_name

# --------------------------------------------------------------------------- #
# Task families and routing-loss conventions (App. B.2)
# --------------------------------------------------------------------------- #
NODE_CLS = "node_cls"
LINK = "link"
GRAPH_CLS = "graph_cls"
MULTILABEL = "multilabel"
REGRESSION = "regression"
TASK_FAMILIES = (NODE_CLS, LINK, GRAPH_CLS, MULTILABEL, REGRESSION)

# Prediction spaces mixed by Eq. 1: class-probability simplex (binary LP is a
# two-class simplex), independent assay probabilities, or regression outputs in
# support median/MAD units.
PRED_SPACE = {
    NODE_CLS: "simplex",
    LINK: "simplex",
    GRAPH_CLS: "simplex",
    MULTILABEL: "probability",
    REGRESSION: "normalized",
}
NORM_CONVENTION = {
    NODE_CLS: "prob",
    LINK: "prob",
    GRAPH_CLS: "prob",
    MULTILABEL: "prob",
    REGRESSION: "median_mad",
}


def infer_task_family(task_level: str, task_type: str, label_dim: int) -> str:
    """Map dataset metadata to one of :data:`TASK_FAMILIES`."""
    level = str(task_level).lower()
    kind = str(task_type or "classification").lower()
    if kind == "regression":
        return REGRESSION
    if level == "edge":
        return LINK
    if int(label_dim or 1) > 1:
        return MULTILABEL
    if level == "node":
        return NODE_CLS
    return GRAPH_CLS


def reported_metric(task_family: str) -> str:
    """Paper Sec. 4.1: accuracy, ROC-AUC (LP, ToxCast), raw-unit MAE (regression)."""
    if task_family == REGRESSION:
        return "mae"
    if task_family in (LINK, MULTILABEL):
        return "auc"
    return "acc"


# --------------------------------------------------------------------------- #
# Applications
# --------------------------------------------------------------------------- #
def parse_dataset_spec(spec: str) -> Tuple[str, str]:
    """Parse ``"dataset:task_level"`` (task level defaults to ``node``)."""
    text = str(spec).strip()
    if ":" in text:
        name, level = text.split(":", 1)
    else:
        name, level = text, "node"
    level = level.strip().lower()
    if level not in {"node", "edge", "graph"}:
        raise ValueError(f"Unsupported task level in application spec {spec!r}")
    return name.strip().lower(), level


def base_group(dataset: str) -> str:
    """Canonical base dataset shared by every task/budget/split of one dataset."""
    return canonical_source_dataset_name(dataset)


@dataclass(frozen=True)
class AppSpec:
    """One application ``a = (dataset, task_level, budget, seed)``."""

    dataset: str
    task_level: str
    budget: int
    seed: int
    lp_split: Tuple[float, float, float] = (0.1, 0.05, 0.1)

    @property
    def split(self) -> Tuple[float, float, float]:
        if self.task_level == "edge":
            return tuple(float(v) for v in self.lp_split)  # type: ignore[return-value]
        return (int(self.budget), 0.0, 1.0)

    @property
    def group(self) -> str:
        return base_group(self.dataset)

    @property
    def key(self) -> str:
        return f"{self.dataset}__{self.task_level}__b{int(self.budget)}__s{int(self.seed)}"

    @property
    def data_key(self) -> str:
        return (
            f"{self.dataset}__{self.task_level}__"
            f"{format_split_for_name(self.split)}__s{int(self.seed)}"
        )

    @property
    def spec(self) -> str:
        return f"{self.dataset}:{self.task_level}"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["lp_split"] = list(self.lp_split)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AppSpec":
        return cls(
            dataset=str(payload["dataset"]),
            task_level=str(payload["task_level"]),
            budget=int(payload["budget"]),
            seed=int(payload["seed"]),
            lp_split=tuple(float(v) for v in payload.get("lp_split", (0.1, 0.05, 0.1))),
        )


def enumerate_applications(rg_cfg, *, include_history_extra: bool = True, seeds: Optional[Sequence[int]] = None) -> List[AppSpec]:
    """All applications declared by ``cfg.moe.routergfm.apps``.

    Targets use ``apps.seeds`` (or *seeds*), historical-only datasets use
    ``apps.history_seeds``. Order is deterministic.
    """
    apps_cfg = rg_cfg.apps
    lp_split = tuple(float(v) for v in apps_cfg.lp_split)
    target_seeds = [int(s) for s in (seeds if seeds is not None else apps_cfg.seeds)]
    history_seeds = [int(s) for s in apps_cfg.history_seeds]
    out: List[AppSpec] = []
    seen = set()

    def _add(spec_text: str, seed_list: Iterable[int]) -> None:
        name, level = parse_dataset_spec(spec_text)
        for budget in apps_cfg.budgets:
            for seed in seed_list:
                app = AppSpec(name, level, int(budget), int(seed), lp_split)
                if app.key not in seen:
                    seen.add(app.key)
                    out.append(app)

    for spec_text in apps_cfg.targets:
        _add(spec_text, target_seeds + [s for s in history_seeds if s not in target_seeds])
    if include_history_extra:
        for spec_text in apps_cfg.history_extra:
            _add(spec_text, history_seeds)
    return out


# --------------------------------------------------------------------------- #
# Experts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExpertSpec:
    """One pretrained checkpoint of the pool (App. B.1)."""

    expert_id: str  # checkpoint stem, unique
    architecture: str  # gcn | gat | gin | h2gcn | fagcn | nodeformer | transformer
    objective: str  # canonical pretraining objective (infograph-nolw -> infograph)
    objective_variant: str  # method token as written in the run name
    source: str  # source corpus dataset name
    source_task_level: str  # node | graph | edge
    checkpoint_path: str

    @property
    def source_group(self) -> str:
        return base_group(self.source)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpertSpec":
        return cls(**{k: payload[k] for k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


def is_same_source(app: AppSpec, expert: ExpertSpec) -> bool:
    """Empty diagonal: an expert is never evaluated on its own source dataset."""
    return app.group == expert.source_group


# --------------------------------------------------------------------------- #
# Artifact layout
# --------------------------------------------------------------------------- #
@dataclass
class RouterPaths:
    """Filesystem layout below ``cfg.moe.routergfm.output_root``."""

    root: Path

    @classmethod
    def from_cfg(cls, cfg) -> "RouterPaths":
        return cls(Path(str(cfg.moe.routergfm.output_root)))

    @property
    def catalog_file(self) -> Path:
        return self.root / "experts" / "catalog.json"

    def data_dir(self, data_key: str) -> Path:
        return self.root / "data" / data_key

    def data_meta_file(self, data_key: str) -> Path:
        """Instance positions, labels, task metadata, dataset statistics."""
        return self.data_dir(data_key) / "meta.pt"

    def descriptor_file(self, data_key: str) -> Path:
        return self.data_dir(data_key) / "descriptors.pt"

    def history_file(self, data_key: str, expert_id: str) -> Path:
        """Per-instance diagnostic losses/predictions of one fitted expert.

        Heads depend only on the split (support) and the expert, so both LP
        budget blocks sharing a data key share these records.
        """
        return self.root / "history" / data_key / f"{expert_id}.pt"

    def prediction_file(self, data_key: str, expert_id: str) -> Path:
        """Full query-set predictions (+ out-of-fold support predictions)."""
        return self.root / "predictions" / data_key / f"{expert_id}.pt"

    def router_dir(self, run_key: str) -> Path:
        return self.root / "routers" / run_key

    def deploy_dir(self, app_key: str) -> Path:
        return self.root / "deploy" / app_key

    def analysis_dir(self, name: str) -> Path:
        return self.root / "analysis" / name


def stable_hash(payload: Any, length: int = 12) -> str:
    """Short deterministic hash of a JSON-serializable payload."""
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


@dataclass
class CompatKey:
    """Retrieval compatibility (Sec. 3.4): task/loss family, budget, normalization."""

    task_family: str
    budget: int
    norm: str = field(default="")

    def __post_init__(self) -> None:
        if not self.norm:
            self.norm = NORM_CONVENTION[self.task_family]

    def as_tuple(self) -> Tuple[str, int, str]:
        return (self.task_family, int(self.budget), self.norm)


__all__ = [
    "AppSpec",
    "CompatKey",
    "ExpertSpec",
    "GRAPH_CLS",
    "LINK",
    "MULTILABEL",
    "NODE_CLS",
    "NORM_CONVENTION",
    "PRED_SPACE",
    "REGRESSION",
    "RouterPaths",
    "TASK_FAMILIES",
    "base_group",
    "enumerate_applications",
    "infer_task_family",
    "is_same_source",
    "parse_dataset_spec",
    "reported_metric",
    "stable_hash",
]
