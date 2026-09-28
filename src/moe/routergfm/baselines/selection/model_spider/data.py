"""Model Spider tasks built from RouterGFM applications (Zhang et al., 2023, Sec. 4.2-4.4).

* General task token (Eq. 5, psi = label-free context descriptor z(x)): class
  centres of the application's standardized support descriptors.
* PTM-specific task token (Sec. 4.4): class centres of an expert's frozen
  support readout, z-scored per dimension over the support (the inputs the
  task heads see), cached per data key.
* Ground truth: the archived application averages mu_{b,e} (Eq. 2) of the
  experts evaluated on b.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

from src.utils.checkpoint import save_torch_atomic

from ....common import AppSpec, RouterPaths
from ..common import application_averages
from .tokens import partition_weights, weighted_centers


_DEFAULT_DIRS = {"checkpoint_dir": "checkpoints", "cache_dir": "cache"}


def spider_dir(cfg, key: str) -> Path:
    """``baselines.model_spider.<key>`` (``checkpoint_dir`` | ``cache_dir``); '' -> below ``<output_root>/model_spider``."""
    value = str(cfg.moe.routergfm.baselines.model_spider.get(key) or "")
    return Path(value) if value else RouterPaths.from_cfg(cfg).root / "model_spider" / _DEFAULT_DIRS[key]


@dataclass
class SpiderTask:
    """One application as a ranking task."""

    app: AppSpec
    general_centers: torch.Tensor  # [C, d_z] class centres of standardized support descriptors
    candidate_ids: List[str]  # history: experts of E_b with a recorded mu_{b,e}; deployment: E_a
    target_losses: Optional[torch.Tensor] = None  # [M_b] mu_{b,e}; None at deployment


def build_spider_task(infra, app: AppSpec, standardizer, *, with_targets: bool, regression_bins: int) -> SpiderTask:
    """General tokens from support labels and descriptors; with targets, the recorded experts and their mu."""
    weights = partition_weights(infra.support_labels(app), infra.task_family(app), regression_bins=regression_bins)
    general = weighted_centers(standardizer.transform(infra.descriptors(app, "support")), weights)
    pool = list(infra.compatible_pool(app))
    if not with_targets:
        return SpiderTask(app, general, pool)
    mu, _ = application_averages(infra, [app], pool)
    keep = torch.isfinite(mu[0])
    return SpiderTask(app, general, [e for e, k in zip(pool, keep.tolist()) if k], mu[0][keep])


class SpecificCenterCache:
    """Class centres ``[C, d_e]`` of ``infra.embeddings(app, e, 'support')``; fp16 file per data key."""

    def __init__(self, infra, cache_dir: Path, regression_bins: int):
        self.infra = infra
        self.cache_dir = Path(cache_dir)
        self.regression_bins = int(regression_bins)
        self._entries: Dict[str, Dict[str, torch.Tensor]] = {}
        self._weights: Dict[str, torch.Tensor] = {}

    def _file(self, data_key: str) -> Path:
        return self.cache_dir / f"{data_key}__rb{self.regression_bins}.pt"

    def _load(self, app: AppSpec) -> Dict[str, torch.Tensor]:
        if app.data_key not in self._entries:
            path = self._file(app.data_key)
            self._entries[app.data_key] = dict(torch.load(path, map_location="cpu")) if path.is_file() else {}
        return self._entries[app.data_key]

    def _compute(self, app: AppSpec, expert_id: str) -> torch.Tensor:
        if app.data_key not in self._weights:
            labels, family = self.infra.support_labels(app), self.infra.task_family(app)
            self._weights[app.data_key] = partition_weights(labels, family, regression_bins=self.regression_bins)
        emb = self.infra.embeddings(app, expert_id, "support").float()
        emb = (emb - emb.mean(0)) / emb.std(0, unbiased=False).clamp_min(1e-6)
        return weighted_centers(emb, self._weights[app.data_key]).half()

    def warm(self, app: AppSpec, expert_ids: Iterable[str]) -> None:
        """Compute every missing or non-finite (built from overflowed float16 readouts) centre of *app*,
        then persist its data key once."""
        entries = self._load(app)
        missing = [e for e in expert_ids if e not in entries or not bool(torch.isfinite(entries[e]).all())]
        for e in missing:
            entries[e] = self._compute(app, e)
        if missing:
            save_torch_atomic(str(self._file(app.data_key)), entries)

    def get(self, app: AppSpec, expert_id: str) -> torch.Tensor:
        entries = self._load(app)
        self.warm(app, [expert_id])
        return entries[expert_id].float()


__all__ = ["SpecificCenterCache", "SpiderTask", "build_spider_task", "spider_dir"]
