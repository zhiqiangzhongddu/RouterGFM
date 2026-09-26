"""Which test metric each dataset/task reports — the single source of truth.

Every result table and RouterGFM benchmark row resolves its reported metric
through ``eval_metric`` instead of hard-coding one, so no two code paths can
report the same cell in different metrics. The metric values themselves come
from ``src.utils.metrics.compute_supervised_metrics``.

Kept torch-free so the table renderers stay cheap to import.
"""

from __future__ import annotations

# Classification datasets whose labels are a multi-target 0/1 vector
# (label_dim > 1, NaN = missing), verified against the processed `y` shapes.
# Accuracy is degenerate on them (an all-negative predictor scores ~0.92
# masked accuracy on Toxcast), so they report macro ROC-AUC over targets.
MULTILABEL_DATASETS = frozenset(
    {"clintox", "muv", "ogbg-molpcba", "pcba", "sider", "tox21", "toxcast"}
)

# Paper-table task codes -> (task_level, task_type).
TABLE_TASKS = {
    "NC": ("node", "classification"),
    "LP": ("edge", "classification"),
    "GC": ("graph", "classification"),
    "GR": ("graph", "regression"),
}


def eval_metric(dataset: str, task_level: str, task_type: str) -> str:
    """Return the reported test metric for one dataset/task.

    - regression -> ``test_mae`` (lower is better)
    - edge task or multi-label classification -> ``test_auc``
    - single-label classification -> ``test_acc``
    """
    if str(task_type).strip().lower() == "regression":
        return "test_mae"
    if (
        str(task_level).strip().lower() == "edge"
        or str(dataset).strip().lower() in MULTILABEL_DATASETS
    ):
        return "test_auc"
    return "test_acc"


__all__ = ["MULTILABEL_DATASETS", "TABLE_TASKS", "eval_metric"]
