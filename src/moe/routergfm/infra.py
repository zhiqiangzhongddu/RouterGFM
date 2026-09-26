"""``RouterInfra``: one access path to applications, experts, history, and predictions.

Used by RouterGFM and by every matched-pool / selection baseline (DESIGN 10),
so all methods share the same eligible pools, frozen readouts, fitted heads,
descriptors, and metadata. Target query labels are read only by the two
evaluation helpers at the end of the class.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.utils.checkpoint import save_torch_atomic
from src.utils.metrics import compute_supervised_metrics

from .applications import AppData, DataProvider, RealDataProvider
from .common import AppSpec, RouterPaths, enumerate_applications, parse_dataset_spec
from .descriptors import descriptors_at, ensure_descriptors
from .experts import build_expert_catalog, compatible_experts, load_frozen_encoder
from .history import SPLITS, HistoryStore, app_normalizer, embed_splits, instance_losses, predict_queries
from .losses import RegressionNormalizer, to_metric_inputs


class _UnlabeledInstances(SequenceABC):
    """Lazy view of instance graphs at given positions, with labels (``y``) removed."""

    def __init__(self, dataset, positions: torch.Tensor):
        self.dataset = dataset
        self.positions = positions

    def __len__(self) -> int:
        return int(self.positions.numel())

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        graph = copy.copy(self.dataset[int(self.positions[idx])])
        if "y" in graph:
            del graph.y
        return graph


class RouterInfra:
    def __init__(self, cfg, provider: Optional[DataProvider] = None, device=None):
        self.cfg = cfg
        self.paths = RouterPaths.from_cfg(cfg)
        self.provider = provider if provider is not None else RealDataProvider(cfg)
        self.device = torch.device(device) if device is not None else torch.device(
            f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu"
        )
        self.catalog = build_expert_catalog(cfg)
        self.expert_index = {spec.expert_id: i for i, spec in enumerate(self.catalog)}
        self.store = HistoryStore(self.paths)
        self._declared = enumerate_applications(cfg.moe.routergfm)
        self._data: Dict[str, AppData] = {}
        self._descriptors: Dict[str, Dict[str, Any]] = {}
        self._encoder: Optional[Tuple[str, Tuple[Any, Any]]] = None
        self._text_encoder = None

    # -- applications -------------------------------------------------------
    def application(self, spec: str, budget: int, seed: int) -> AppSpec:
        """``"dataset:level"`` -> AppSpec with the configured LP split."""
        name, level = parse_dataset_spec(spec)
        lp_split = tuple(float(v) for v in self.cfg.moe.routergfm.apps.lp_split)
        return AppSpec(name, level, int(budget), int(seed), lp_split)

    def data(self, app: AppSpec) -> AppData:
        if app.key not in self._data:
            self._data[app.key] = self.provider.load(app)
        return self._data[app.key]

    def task_family(self, app: AppSpec) -> str:
        return self.data(app).task_family

    def compatible_pool(self, app: AppSpec) -> List[str]:
        """E_a as expert ids, in catalog order."""
        return [self.catalog[i].expert_id for i in compatible_experts(app, self.catalog, self.cfg)]

    def historical_applications(self, target: AppSpec) -> List[AppSpec]:
        """Declared applications with history outside the target's group; same task family first."""
        apps = [a for a in self._declared if a.group != target.group and self.store.expert_ids(a.data_key)]
        family = self.task_family(target)
        return sorted(apps, key=lambda a: self.task_family(a) != family)

    # -- frozen readouts and fitted heads -------------------------------------
    def _embedding_file(self, data_key: str, expert_id: str, split: str) -> Path:
        return self.paths.root / "embeddings" / data_key / f"{expert_id}__{split}.pt"

    def _load_encoder(self, expert_id: str):
        if self._encoder is None or self._encoder[0] != expert_id:
            self._encoder = None  # release the previous encoder first
            spec = self.catalog[self.expert_index[expert_id]]
            self._encoder = (expert_id, load_frozen_encoder(self.cfg, spec, self.device))
        return self._encoder[1]

    def embeddings(self, app: AppSpec, expert_id: str, split: str) -> torch.Tensor:
        """Frozen-encoder task readout of ``data.<split>_pos`` (float32; float16 disk cache).

        Support embeddings come from the history record when one exists.
        """
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r} (expected one of {SPLITS}).")
        path = self._embedding_file(app.data_key, expert_id, split)
        if path.is_file():
            return torch.load(path, map_location="cpu").float()
        if split == "support" and self.store.has(app.data_key, expert_id):
            return self.store.load(app.data_key, expert_id)["support_emb"].float()
        encoder, model_cfg = self._load_encoder(expert_id)
        spec = self.catalog[self.expert_index[expert_id]]
        batch_size = int(self.cfg.moe.routergfm.device_batch_size)
        emb = embed_splits(encoder, model_cfg, spec, self.data(app), (split,), self.device, batch_size)[split].half()
        save_torch_atomic(str(path), emb)
        return emb.float()

    def support_labels(self, app: AppSpec) -> torch.Tensor:
        """Support labels (long classes; float with NaN-missing assays; raw-unit regression targets)."""
        return self.data(app).labels["support"]

    def normalizer(self, app: AppSpec) -> Optional[RegressionNormalizer]:
        return app_normalizer(self.cfg, self.data(app))

    def expert_predictions(self, app: AppSpec, expert_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """``history.predict_queries``: query, in-sample support, and OOF support predictions."""
        return predict_queries(self.cfg, app, expert_ids, self.provider)

    def historical_mu(self, app: AppSpec, expert_id: str) -> Tuple[float, int]:
        """Eq. 2 average and valid count from the history store ((NaN, 0) if unrecorded)."""
        return self.store.app_average(app, expert_id)

    # -- label-free context and metadata ---------------------------------------
    def descriptors(self, app: AppSpec, split: str) -> torch.Tensor:
        """Raw (unstandardized) descriptors z_a(x) of ``data.<split>_pos``."""
        data = self.data(app)
        if app.data_key not in self._descriptors:
            self._descriptors[app.data_key] = ensure_descriptors(self.cfg, data)
        return descriptors_at(self._descriptors[app.data_key], getattr(data, f"{split}_pos"))

    def instance_graphs(self, app: AppSpec, split: str) -> Sequence[Any]:
        """The PyG (sub)graphs of ``data.<split>_pos`` without labels."""
        data = self.data(app)
        return _UnlabeledInstances(data.dataset, getattr(data, f"{split}_pos"))

    @property
    def text_encoder(self):
        if self._text_encoder is None:
            from .text import TextEncoder

            self._text_encoder = TextEncoder(self.cfg)
        return self._text_encoder

    def app_metadata(self, app: AppSpec) -> torch.Tensor:
        """Raw application metadata (text ⊕ numeric, NaN = missing) as in the context graph."""
        from .context_graph import app_feature_vector

        data = self.data(app)
        return app_feature_vector(app, data.stats, self.text_encoder, self.cfg, family=data.task_family)

    def expert_metadata(self, expert_id: str) -> torch.Tensor:
        """Raw expert metadata (text ⊕ numeric, NaN = missing); corpus stats from the source's applications."""
        from .context_graph import corpus_stats_for, expert_feature_vector

        spec = self.catalog[self.expert_index[expert_id]]
        apps = [a for a in self._declared if a.group == spec.source_group]
        stats = {a.key: self.data(a).stats for a in apps}
        corpus = corpus_stats_for(spec.source, spec.source_task_level, apps, stats)
        return expert_feature_vector(spec, self.text_encoder, self.cfg, corpus_stats=corpus)

    # -- evaluation only: the sole readers of target query labels --------------
    def query_expert_risk(self, app: AppSpec, expert_id: str) -> float:
        """The target's recorded mean routing loss of ``expert_id`` on D_a."""
        if not self.store.has(app.data_key, expert_id):
            raise KeyError(f"No history record for {expert_id} on {app.data_key}; run the history stage first.")
        return self.store.app_average(app, expert_id)[0]

    def evaluate_outputs(self, app: AppSpec, pred_query: torch.Tensor) -> Dict[str, float]:
        """Metrics of family-space predictions on Q_a: ``compute_supervised_metrics`` keys + ``risk``.

        ``risk`` is the mean routing loss (Brier for classification/LP) over the
        valid query instances; ``common.reported_metric`` names the reported key.
        """
        data = self.data(app)
        pred = torch.as_tensor(pred_query).float().cpu()
        if pred.size(0) != data.query_pos.numel():
            raise ValueError(f"{app.key}: {pred.size(0)} predictions for {data.query_pos.numel()} queries.")
        family, target = data.task_family, data.labels["query"]
        normalizer = self.normalizer(app)
        loss = instance_losses(self.cfg, pred, target, family, normalizer)
        valid = torch.isfinite(loss)
        logits, labels, task_type = to_metric_inputs(pred, family, normalizer, target=target)
        metrics = dict(compute_supervised_metrics(logits, labels, task_type))
        metrics["risk"] = float(loss[valid].mean()) if bool(valid.any()) else float("nan")
        return metrics


__all__ = ["RouterInfra"]
