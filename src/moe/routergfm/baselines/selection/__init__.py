"""Application-level selection baselines (App. C, Table 9): rank E_a without target query data.

Contract: ``Selector(cfg, infra).rank(app) -> SelectionOutcome``; the harness in
:mod:`.run` evaluates the shortlist afterwards. Selector modules are imported on
first use (see ``run._SELECTORS``).
"""

from .common import QueryAccessError, QueryGuard, SelectionOutcome, selection_metrics

__all__ = ["QueryAccessError", "QueryGuard", "SelectionOutcome", "selection_metrics"]
