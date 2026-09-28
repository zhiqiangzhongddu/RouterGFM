"""MetaGL-U matched-pool baseline (Table 1): uniform mixture of the MetaGL top-K team.

The team is the first ``baselines.topk`` experts of the MetaGL ranking of E_a
(``baselines.metagl_u.selector``: ``metagl`` or ``metagl_metadata``), selected
through the selection harness's query guard, so the target contributes only
label-free structure. Each team expert's head is fitted on the target support
with its frozen encoder by the shared infra (``infra.expert_predictions``: the
same heads and query set as RouterGFM-G), and the prediction is Eq. 1 with
alpha = 1/K in the family's prediction space (class / assay probabilities,
normalized regression outputs). Query labels are read only by ``infra.evaluate_outputs``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ..common import AppSpec
from .selection.common import QueryGuard, SelectionOutcome

SELECTORS = ("metagl", "metagl_metadata")


class MetaGLURunner:
    """Matched-pool runner: ``fit()`` selects the team, mixes its query predictions, and evaluates them."""

    def __init__(self, cfg, app: AppSpec, infra):
        self.cfg = cfg
        self.app = app
        self.infra = infra
        self.selector_name = str(cfg.moe.routergfm.baselines.metagl_u.selector).strip().lower()
        if self.selector_name not in SELECTORS:
            raise ValueError(f"baselines.metagl_u.selector must be one of {SELECTORS}, got {self.selector_name!r}")
        self.outcome: Optional[SelectionOutcome] = None
        self.team: List[str] = []
        self.pred: Optional[torch.Tensor] = None
        self.best_metrics: Dict[str, float] = {}
        self.best_epoch: Optional[int] = None

    def select_team(self) -> SelectionOutcome:
        from .selection.run import build_selector

        guard = QueryGuard(self.infra)
        guard.target = self.app
        try:
            return build_selector(self.selector_name, self.cfg, guard).rank(self.app)
        finally:
            guard.target = None

    def fit(self) -> None:
        from ..integration import mix

        self.outcome = self.select_team()
        self.team = list(self.outcome.team)
        outputs = self.infra.expert_predictions(self.app, self.team)
        preds = torch.stack([torch.as_tensor(outputs[e]["pred"]).float() for e in self.team], dim=1)  # [N, K, C]
        self.pred = mix(preds, torch.full((len(self.team),), 1.0 / len(self.team)), self.infra.task_family(self.app))
        self.best_metrics = {f"test_{k}": float(v) for k, v in self.evaluate().items()}
        self.best_epoch = int(self.outcome.extras["best_epoch"])
        print(f"[MetaGL-U] {self.app.key}: team={self.team} " + " ".join(f"{k}={v:.4f}" for k, v in self.best_metrics.items()))

    def predict_queries(self) -> torch.Tensor:
        return self.pred

    def evaluate(self) -> Dict[str, float]:
        return self.infra.evaluate_outputs(self.app, self.pred)


__all__ = ["MetaGLURunner"]
