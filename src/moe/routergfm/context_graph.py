"""Heterogeneous context graph H (Sec. 3.2) and node features x_v (Sec. 3.3).

Node types: ``app`` (applications and, flagged by ``is_corpus``, the source
corpora), ``expert``, ``arch``, ``objective``. Every relation has a ``rev_*``
reverse with the same edge attributes, so message passing is bidirectional.
Evaluation edges carry ``[mu_bar, log1p(count)]`` (Eq. 2); construction edges
carry zeros. Node features are ``text ⊕ standardized numeric ⊕ 1`` and depend
only on metadata, so historical and inserted nodes share one description and
projection function per type (Prop. 4). Label-derived aggregates never enter
node features; they live on evaluation edges, which :func:`masked_edges` hides.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from src.utils.checkpoint import save_json_atomic

from .common import TASK_FAMILIES, AppSpec, ExpertSpec, RouterPaths, base_group, is_same_source
from .text import (
    describe_application,
    describe_architecture,
    describe_corpus,
    describe_expert,
    describe_objective,
)

APP, EXPERT, ARCH, OBJECTIVE = "app", "expert", "arch", "objective"
NODE_TYPES = (APP, EXPERT, ARCH, OBJECTIVE)
EVALUATES = (APP, "evaluates", EXPERT)
USES_ARCH = (EXPERT, "uses_arch", ARCH)
USES_OBJECTIVE = (EXPERT, "uses_objective", OBJECTIVE)
PRETRAINED_ON = (EXPERT, "pretrained_on", APP)
EDGE_DIM = 2
CORPUS_PREFIX = "corpus:"


def reverse_relation(rel: Tuple[str, str, str]) -> Tuple[str, str, str]:
    return (rel[2], f"rev_{rel[1]}", rel[0])


RELATIONS = tuple(
    r
    for rel in (EVALUATES, USES_ARCH, USES_OBJECTIVE, PRETRAINED_ON)
    for r in (rel, reverse_relation(rel))
)
EVAL_RELATIONS = (EVALUATES, reverse_relation(EVALUATES))

# Label-free AppData.stats entries used as application metadata (raw counts;
# signed-log transformed here because sizes span orders of magnitude).
APP_STAT_KEYS = ("num_instances", "num_nodes", "num_edges", "avg_degree", "feature_dim", "num_classes")
APP_NUMERIC_NAMES = (
    tuple(f"log_{k}" for k in APP_STAT_KEYS)
    + tuple(f"family_{f}" for f in TASK_FAMILIES)
    + ("log_budget", "is_corpus")
)
SOURCE_LEVELS = ("node", "edge", "graph")
EXPERT_NUMERIC_NAMES = (
    ("log_hidden_dim", "log_out_dim", "num_layers", "log_num_params")
    + tuple(f"source_{lvl}" for lvl in SOURCE_LEVELS)
    + ("log_corpus_size",)
)
_DIMS_RE = re.compile(r"_h(\d+)_o(\d+)_l(\d+)(?:_|$)")
_NAN = float("nan")


# --------------------------------------------------------------------------- #
# Numeric metadata and feature builders
# --------------------------------------------------------------------------- #
def _slog(value: Any) -> float:
    if value is None:
        return _NAN
    v = float(value)
    return math.copysign(math.log1p(abs(v)), v) if math.isfinite(v) else _NAN


def resolve_family(app: AppSpec, stats: Mapping[str, float], family: Optional[str] = None) -> str:
    """Explicit *family*, else the ``family_<f>`` one-hot in the label-free stats."""
    if family:
        return str(family)
    hits = [f for f in TASK_FAMILIES if float(stats.get(f"family_{f}", 0.0)) > 0]
    if len(hits) != 1:
        raise ValueError(f"Task family of {app.key} is unknown: pass it or add a family_<f> one-hot to stats")
    return hits[0]


def app_numeric(
    stats: Mapping[str, float], family: Optional[str], budget: int, is_corpus: bool
) -> torch.Tensor:
    """Raw application metadata in :data:`APP_NUMERIC_NAMES` order (NaN = missing)."""
    values = [_slog(stats.get(k)) for k in APP_STAT_KEYS]
    values += [1.0 if family == f else 0.0 for f in TASK_FAMILIES]
    values += [math.log1p(max(int(budget), 0)), 1.0 if is_corpus else 0.0]
    return torch.tensor(values, dtype=torch.float32)


@lru_cache(maxsize=None)
def _num_params(checkpoint_path: str) -> float:
    if not os.path.isfile(checkpoint_path):
        return _NAN
    payload = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    state = payload.get("model_state") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        return _NAN
    return float(sum(v.numel() for v in state.values() if torch.is_tensor(v) and v.is_floating_point()))


def expert_num_params(cfg, specs: Sequence[ExpertSpec]) -> Dict[str, float]:
    """#floating-point parameters per expert id (NaN when the checkpoint is absent).

    Cached next to the catalog in ``<output_root>/experts/params.json``, keyed by
    expert id and invalidated when the checkpoint's path, size or mtime changes,
    so building H does not reload every checkpoint.
    """
    path = RouterPaths.from_cfg(cfg).catalog_file.with_name("params.json")
    cache = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    out, dirty = {}, False
    for spec in specs:
        checkpoint = str(spec.checkpoint_path)
        try:
            st = os.stat(checkpoint)
        except OSError:
            out[spec.expert_id] = _NAN
            continue
        stamp = [checkpoint, st.st_size, st.st_mtime_ns]
        entry = cache.get(spec.expert_id)
        if entry is None or entry.get("stamp") != stamp:
            entry = cache[spec.expert_id] = {"stamp": stamp, "n_params": _num_params(checkpoint)}
            dirty = True
        out[spec.expert_id] = float(entry["n_params"])
    if dirty:
        save_json_atomic(str(path), cache)
    return out


def expert_numeric(
    spec: ExpertSpec, corpus_stats: Optional[Mapping[str, float]] = None, *, num_params: Optional[float] = None
) -> torch.Tensor:
    """Raw expert metadata in :data:`EXPERT_NUMERIC_NAMES` order (NaN = missing).

    No architecture/objective identity one-hots: identity enters through text
    and construction edges so unseen architectures remain representable.
    ``num_params`` (from :func:`expert_num_params`) skips reading the checkpoint.
    """
    match = _DIMS_RE.search(spec.expert_id)
    hidden, out, layers = (float(g) for g in match.groups()) if match else (_NAN, _NAN, _NAN)
    if num_params is None:
        num_params = _num_params(str(spec.checkpoint_path))
    values = [_slog(hidden), _slog(out), layers, _slog(num_params)]
    values += [1.0 if spec.source_task_level == lvl else 0.0 for lvl in SOURCE_LEVELS]
    values += [_slog((corpus_stats or {}).get("num_instances"))]
    return torch.tensor(values, dtype=torch.float32)


def corpus_stats_for(
    source: str,
    source_level: str,
    apps: Sequence[AppSpec],
    stats_by_app: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    """Stats of a source corpus: ``stats_by_app['corpus:<source>']`` if given, else
    those of an application on the same base dataset (same task level first)."""
    if CORPUS_PREFIX + source in stats_by_app:
        return dict(stats_by_app[CORPUS_PREFIX + source])
    group = base_group(source)
    matches = [a for a in apps if a.group == group and a.key in stats_by_app]
    if not matches:
        return {}
    best = next((a for a in matches if a.task_level == source_level), matches[0])
    return dict(stats_by_app[best.key])


def _fit_standardizer(values: torch.Tensor) -> Dict[str, torch.Tensor]:
    """NaN-aware per-column mean/std; constant or all-missing columns get std 1."""
    finite = torch.isfinite(values)
    n = finite.sum(0).clamp(min=1)
    mean = torch.where(finite, values, torch.zeros_like(values)).sum(0) / n
    var = torch.where(finite, (values - mean) ** 2, torch.zeros_like(values)).sum(0) / n
    std = var.sqrt()
    return {"mean": mean, "std": torch.where(std > 1e-6, std, torch.ones_like(std))}


def _standardize(values: torch.Tensor, stats: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """z-score with stored statistics; missing entries are imputed with the mean (0)."""
    z = (values - stats["mean"]) / stats["std"]
    return torch.where(torch.isfinite(z), z, torch.zeros_like(z))


def _raw_features(texts: Sequence[str], numeric: Optional[torch.Tensor], text_encoder, cfg) -> torch.Tensor:
    graph_cfg = cfg.moe.routergfm.graph
    blocks = []
    if graph_cfg.use_text:
        blocks.append(text_encoder.encode(texts))
    if graph_cfg.use_numeric and numeric is not None:
        blocks.append(numeric)
    return torch.cat(blocks, dim=1) if blocks else torch.zeros((len(texts), 0))


def _node_features(
    texts: Sequence[str],
    numeric: Optional[torch.Tensor],
    stats: Optional[Mapping[str, torch.Tensor]],
    text_encoder,
    cfg,
) -> torch.Tensor:
    """``text ⊕ standardized numeric ⊕ 1`` (the constant keeps ablated features non-empty)."""
    std_numeric = _standardize(numeric, stats) if numeric is not None else None
    raw = _raw_features(texts, std_numeric, text_encoder, cfg)
    return torch.cat([raw, torch.ones((len(texts), 1))], dim=1)


def app_feature_vector(
    app: AppSpec, stats: Mapping[str, float], text_encoder, cfg, *, family: Optional[str] = None
) -> torch.Tensor:
    """Raw application metadata vector (text ⊕ unstandardized numeric; NaN = missing)."""
    family = resolve_family(app, stats, family)
    desc_dir = str(cfg.moe.routergfm.graph.description_dir)
    text = describe_application(app, stats, family, description_dir=desc_dir)
    return _raw_features([text], app_numeric(stats, family, app.budget, False)[None], text_encoder, cfg)[0]


def expert_feature_vector(
    spec: ExpertSpec, text_encoder, cfg, *, corpus_stats: Optional[Mapping[str, float]] = None
) -> torch.Tensor:
    """Raw expert metadata vector (text ⊕ unstandardized numeric; NaN = missing)."""
    text = describe_expert(spec, description_dir=str(cfg.moe.routergfm.graph.description_dir))
    numeric = expert_numeric(spec, corpus_stats, num_params=expert_num_params(cfg, [spec])[spec.expert_id])
    return _raw_features([text], numeric[None], text_encoder, cfg)[0]


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
@dataclass
class ContextGraph:
    x: Dict[str, torch.Tensor]  # node features per type (text ⊕ standardized numeric ⊕ 1)
    edge_index: Dict[tuple, torch.Tensor]  # all RELATIONS incl. reverses
    edge_attr: Dict[tuple, torch.Tensor]  # [E, EDGE_DIM]
    app_nodes: List[Union[AppSpec, str]]  # AppSpec, or 'corpus:<name>' for source corpora
    app_index: Dict[str, int]  # app.key / 'corpus:<name>' -> app node id
    expert_ids: List[str]
    arch_names: List[str]
    objective_names: List[str]
    eval_app: torch.Tensor  # evaluation edges, aligned with edge_index[EVALUATES]
    eval_expert: torch.Tensor
    eval_mu: torch.Tensor
    eval_count: torch.Tensor
    expert_index: Dict[str, int] = field(default_factory=dict)
    expert_catalog_index: List[int] = field(default_factory=list)  # expert node -> catalog index
    corpus_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    numeric_stats: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)  # per type, CPU

    @property
    def in_dims(self) -> Dict[str, int]:
        return {t: int(v.shape[1]) for t, v in self.x.items()}

    def to(self, device) -> "ContextGraph":
        """Copy on *device*; node/edge containers are copied so insertions stay local."""
        return replace(
            self,
            x={k: v.to(device) for k, v in self.x.items()},
            edge_index={k: v.to(device) for k, v in self.edge_index.items()},
            edge_attr={k: v.to(device) for k, v in self.edge_attr.items()},
            app_nodes=list(self.app_nodes),
            app_index=dict(self.app_index),
            expert_ids=list(self.expert_ids),
            arch_names=list(self.arch_names),
            objective_names=list(self.objective_names),
            eval_app=self.eval_app.to(device),
            eval_expert=self.eval_expert.to(device),
            eval_mu=self.eval_mu.to(device),
            eval_count=self.eval_count.to(device),
            expert_index=dict(self.expert_index),
            expert_catalog_index=list(self.expert_catalog_index),
            corpus_stats=dict(self.corpus_stats),
        )


def _stack(rows: List[torch.Tensor], dim: int) -> torch.Tensor:
    return torch.stack(rows) if rows else torch.zeros((0, dim))


def _set_relation(edge_index: dict, edge_attr: dict, rel, src: List[int], dst: List[int], attr) -> None:
    ei = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    attr = attr.reshape(-1, EDGE_DIM).float()
    edge_index[rel], edge_attr[rel] = ei, attr
    edge_index[reverse_relation(rel)], edge_attr[reverse_relation(rel)] = ei.flip(0), attr.clone()


def build_context_graph(
    cfg,
    catalog: Sequence[ExpertSpec],
    apps: Sequence[AppSpec],
    store,
    text_encoder,
    stats_by_app: Mapping[str, Mapping[str, float]],
    *,
    families: Optional[Mapping[str, str]] = None,
    numeric_stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> ContextGraph:
    """Build H from the catalog, applications, and recorded averages.

    ``store`` provides ``has(data_key, expert_id)`` and ``app_average(app,
    expert_id) -> (mu, count)``. ``stats_by_app`` maps ``app.key`` (and
    optionally ``'corpus:<name>'``) to label-free ``AppData.stats``;
    ``families`` optionally maps ``app.key`` to its task family (otherwise read
    from the stats one-hot). ``numeric_stats`` reuses stored standardization
    (e.g. a router bundle's) instead of fitting it on the nodes built here.
    """
    rg = cfg.moe.routergfm
    desc_dir = str(rg.graph.description_dir)
    catalog, apps, families = list(catalog), list(apps), dict(families or {})

    # Application-type nodes: applications first, then source corpora (catalog order).
    app_nodes: List[Union[AppSpec, str]] = []
    app_texts: List[str] = []
    app_rows: List[torch.Tensor] = []
    for app in apps:
        if app.key not in stats_by_app:
            raise KeyError(f"No label-free stats for application {app.key}")
        stats = stats_by_app[app.key]
        family = resolve_family(app, stats, families.get(app.key))
        app_nodes.append(app)
        app_texts.append(describe_application(app, stats, family, description_dir=desc_dir))
        app_rows.append(app_numeric(stats, family, app.budget, False))
    if app_rows:
        observed = torch.isfinite(torch.stack(app_rows)[:, : len(APP_STAT_KEYS)]).any(0)
        missing = [k for k, ok in zip(APP_STAT_KEYS, observed.tolist()) if not ok]
        if missing:
            raise ValueError(f"Application stats lack {missing} for every application")
    corpus_stats: Dict[str, Dict[str, float]] = {}
    for spec in catalog:
        if spec.source not in corpus_stats:
            corpus_stats[spec.source] = corpus_stats_for(spec.source, spec.source_task_level, apps, stats_by_app)
    for name, stats in corpus_stats.items():
        app_nodes.append(CORPUS_PREFIX + name)
        app_texts.append(describe_corpus(name, description_dir=desc_dir))
        app_rows.append(app_numeric(stats, None, 0, True))
    app_index = {(n.key if isinstance(n, AppSpec) else n): i for i, n in enumerate(app_nodes)}

    arch_names = list(dict.fromkeys(s.architecture for s in catalog))
    objective_names = list(dict.fromkeys(s.objective for s in catalog))
    app_num = _stack(app_rows, len(APP_NUMERIC_NAMES))
    num_params = expert_num_params(cfg, catalog)
    expert_num = _stack(
        [expert_numeric(s, corpus_stats[s.source], num_params=num_params[s.expert_id]) for s in catalog],
        len(EXPERT_NUMERIC_NAMES),
    )
    if numeric_stats is None:
        numeric_stats = {APP: _fit_standardizer(app_num), EXPERT: _fit_standardizer(expert_num)}
    expert_texts = [describe_expert(s, description_dir=desc_dir) for s in catalog]
    arch_texts = [describe_architecture(n, description_dir=desc_dir) for n in arch_names]
    objective_texts = [describe_objective(n, description_dir=desc_dir) for n in objective_names]
    x = {
        APP: _node_features(app_texts, app_num, numeric_stats[APP], text_encoder, cfg),
        EXPERT: _node_features(expert_texts, expert_num, numeric_stats[EXPERT], text_encoder, cfg),
        ARCH: _node_features(arch_texts, None, None, text_encoder, cfg),
        OBJECTIVE: _node_features(objective_texts, None, None, text_encoder, cfg),
    }

    # Evaluation edges (Eq. 2): valid recorded averages, empty diagonal.
    exclude_same = bool(rg.experts.exclude_same_source)
    ev_app, ev_exp, ev_mu, ev_count = [], [], [], []
    for a, app in enumerate(apps):
        for e, spec in enumerate(catalog):
            if exclude_same and is_same_source(app, spec):
                continue
            if not store.has(app.data_key, spec.expert_id):
                continue
            mu, count = store.app_average(app, spec.expert_id)
            if count > 0 and math.isfinite(float(mu)):
                ev_app.append(a)
                ev_exp.append(e)
                ev_mu.append(float(mu))
                ev_count.append(float(count))
    eval_mu = torch.tensor(ev_mu, dtype=torch.float32)
    eval_count = torch.tensor(ev_count, dtype=torch.float32)

    edge_index: Dict[tuple, torch.Tensor] = {}
    edge_attr: Dict[tuple, torch.Tensor] = {}
    eval_attr = torch.stack([eval_mu, torch.log1p(eval_count)], dim=1)
    _set_relation(edge_index, edge_attr, EVALUATES, ev_app, ev_exp, eval_attr)
    experts = list(range(len(catalog)))
    zeros = torch.zeros((len(catalog), EDGE_DIM))
    for rel, dst in (
        (USES_ARCH, [arch_names.index(s.architecture) for s in catalog]),
        (USES_OBJECTIVE, [objective_names.index(s.objective) for s in catalog]),
        (PRETRAINED_ON, [app_index[CORPUS_PREFIX + s.source] for s in catalog]),
    ):
        _set_relation(edge_index, edge_attr, rel, experts, dst, zeros)

    return ContextGraph(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        app_nodes=app_nodes,
        app_index=app_index,
        expert_ids=[s.expert_id for s in catalog],
        arch_names=arch_names,
        objective_names=objective_names,
        eval_app=torch.tensor(ev_app, dtype=torch.long),
        eval_expert=torch.tensor(ev_exp, dtype=torch.long),
        eval_mu=eval_mu,
        eval_count=eval_count,
        expert_index={s.expert_id: i for i, s in enumerate(catalog)},
        expert_catalog_index=experts,
        corpus_stats=corpus_stats,
        numeric_stats=numeric_stats,
    )


def masked_edges(graph: ContextGraph, hidden_groups) -> Tuple[Dict[tuple, torch.Tensor], Dict[tuple, torch.Tensor]]:
    """Edges with the evaluation edges (and reverses) of every hidden-group application removed."""
    edge_index, edge_attr = dict(graph.edge_index), dict(graph.edge_attr)
    hidden = {base_group(g) for g in hidden_groups}
    ids = [i for i, n in enumerate(graph.app_nodes) if isinstance(n, AppSpec) and n.group in hidden]
    if not ids:
        return edge_index, edge_attr
    keep = ~torch.isin(graph.eval_app, torch.tensor(ids, dtype=torch.long, device=graph.eval_app.device))
    for rel in EVAL_RELATIONS:
        edge_index[rel] = edge_index[rel][:, keep]
        edge_attr[rel] = edge_attr[rel][keep]
    return edge_index, edge_attr


# --------------------------------------------------------------------------- #
# Insertion (Alg. 1 l.13): metadata + construction edges, router fixed
# --------------------------------------------------------------------------- #
def _append_edges(graph: ContextGraph, rel, src: List[int], dst: List[int], attr: torch.Tensor) -> None:
    device = graph.edge_index[rel].device
    ei = torch.tensor([src, dst], dtype=torch.long, device=device).reshape(2, -1)
    attr = attr.to(device=device, dtype=torch.float32).reshape(-1, EDGE_DIM)
    rev = reverse_relation(rel)
    graph.edge_index[rel] = torch.cat([graph.edge_index[rel], ei], dim=1)
    graph.edge_attr[rel] = torch.cat([graph.edge_attr[rel], attr], dim=0)
    graph.edge_index[rev] = torch.cat([graph.edge_index[rev], ei.flip(0)], dim=1)
    graph.edge_attr[rev] = torch.cat([graph.edge_attr[rev], attr], dim=0)


def _append_node(graph: ContextGraph, node_type: str, row: torch.Tensor) -> int:
    node = int(graph.x[node_type].shape[0])
    graph.x[node_type] = torch.cat([graph.x[node_type], row.to(graph.x[node_type].device)], dim=0)
    return node


def _append_app_node(graph: ContextGraph, key: str, node: Union[AppSpec, str], row: torch.Tensor) -> int:
    idx = _append_node(graph, APP, row)
    graph.app_nodes.append(node)
    graph.app_index[key] = idx
    return idx


def insert_application(
    graph: ContextGraph,
    app: AppSpec,
    stats: Mapping[str, float],
    text_encoder,
    cfg,
    *,
    family: Optional[str] = None,
) -> int:
    """Insert an application from metadata only (no evaluation edges); idempotent."""
    if app.key in graph.app_index:
        return graph.app_index[app.key]
    family = resolve_family(app, stats, family)
    text = describe_application(app, stats, family, description_dir=str(cfg.moe.routergfm.graph.description_dir))
    numeric = app_numeric(stats, family, app.budget, False)[None]
    row = _node_features([text], numeric, graph.numeric_stats[APP], text_encoder, cfg)
    return _append_app_node(graph, app.key, app, row)


def _named_node(graph: ContextGraph, node_type: str, names: List[str], name: str, text: str, encoder, cfg) -> int:
    if name not in names:
        _append_node(graph, node_type, _node_features([text], None, None, encoder, cfg))
        names.append(name)
    return names.index(name)


def insert_expert(
    graph: ContextGraph,
    spec: ExpertSpec,
    text_encoder,
    cfg,
    catalog_index: int,
    *,
    corpus_stats: Optional[Mapping[str, float]] = None,
) -> int:
    """Insert an expert with its construction edges; unseen architecture, objective,
    or source corpus nodes are created from their descriptions. Idempotent."""
    if spec.expert_id in graph.expert_index:
        return graph.expert_index[spec.expert_id]
    desc_dir = str(cfg.moe.routergfm.graph.description_dir)
    arch_text = describe_architecture(spec.architecture, description_dir=desc_dir)
    objective_text = describe_objective(spec.objective, description_dir=desc_dir)
    arch = _named_node(graph, ARCH, graph.arch_names, spec.architecture, arch_text, text_encoder, cfg)
    objective = _named_node(
        graph, OBJECTIVE, graph.objective_names, spec.objective, objective_text, text_encoder, cfg
    )
    corpus_key = CORPUS_PREFIX + spec.source
    if corpus_key not in graph.app_index:
        graph.corpus_stats[spec.source] = dict(corpus_stats or {})
        numeric = app_numeric(graph.corpus_stats[spec.source], None, 0, True)[None]
        text = describe_corpus(spec.source, description_dir=desc_dir)
        row = _node_features([text], numeric, graph.numeric_stats[APP], text_encoder, cfg)
        _append_app_node(graph, corpus_key, corpus_key, row)
    corpus = graph.app_index[corpus_key]
    num_params = expert_num_params(cfg, [spec])[spec.expert_id]
    numeric = expert_numeric(spec, graph.corpus_stats.get(spec.source), num_params=num_params)[None]
    text = describe_expert(spec, description_dir=desc_dir)
    node = _append_node(graph, EXPERT, _node_features([text], numeric, graph.numeric_stats[EXPERT], text_encoder, cfg))
    graph.expert_ids.append(spec.expert_id)
    graph.expert_index[spec.expert_id] = node
    graph.expert_catalog_index.append(int(catalog_index))
    zeros = torch.zeros((1, EDGE_DIM))
    for rel, dst in ((USES_ARCH, arch), (USES_OBJECTIVE, objective), (PRETRAINED_ON, corpus)):
        _append_edges(graph, rel, [node], [dst], zeros)
    return node


def add_calibration_edges(graph: ContextGraph, expert_node: int, app_nodes: Sequence[int], mus, counts) -> None:
    """Optional source calibration: evaluation edges from historical application
    nodes to an (inserted) expert; invalid averages (count 0, non-finite) are skipped."""
    mus = torch.as_tensor(mus, dtype=torch.float32).reshape(-1)
    counts = torch.as_tensor(counts, dtype=torch.float32).reshape(-1)
    apps = torch.as_tensor(list(app_nodes), dtype=torch.long).reshape(-1)
    keep = (counts > 0) & torch.isfinite(mus)
    apps, mus, counts = apps[keep], mus[keep], counts[keep]
    if apps.numel() == 0:
        return
    experts = torch.full_like(apps, int(expert_node))
    attr = torch.stack([mus, torch.log1p(counts)], dim=1)
    _append_edges(graph, EVALUATES, apps.tolist(), experts.tolist(), attr)
    device = graph.eval_app.device
    graph.eval_app = torch.cat([graph.eval_app, apps.to(device)])
    graph.eval_expert = torch.cat([graph.eval_expert, experts.to(device)])
    graph.eval_mu = torch.cat([graph.eval_mu, mus.to(device)])
    graph.eval_count = torch.cat([graph.eval_count, counts.to(device)])


__all__ = [
    "APP",
    "APP_NUMERIC_NAMES",
    "APP_STAT_KEYS",
    "ARCH",
    "CORPUS_PREFIX",
    "ContextGraph",
    "EDGE_DIM",
    "EVALUATES",
    "EVAL_RELATIONS",
    "EXPERT",
    "EXPERT_NUMERIC_NAMES",
    "NODE_TYPES",
    "OBJECTIVE",
    "PRETRAINED_ON",
    "RELATIONS",
    "USES_ARCH",
    "USES_OBJECTIVE",
    "add_calibration_edges",
    "app_feature_vector",
    "app_numeric",
    "build_context_graph",
    "corpus_stats_for",
    "expert_feature_vector",
    "expert_num_params",
    "expert_numeric",
    "insert_application",
    "insert_expert",
    "masked_edges",
    "resolve_family",
    "reverse_relation",
]
