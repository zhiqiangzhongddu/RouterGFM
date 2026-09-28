"""Shared RouterGFM test infrastructure: synthetic applications, tiny experts, tiny cfg.

* :class:`SyntheticDataProvider` builds tiny PyG instance sets shaped like the
  real induced datasets (node ego-graphs with ``target_node_index`` /
  ``base_node_id``, link enclosing subgraphs with ``edge_label_index`` and the
  queried edge removed, whole graphs for graph classification, multi-label with
  NaN-missing assays, and multi-target regression). Instances depend only on
  ``(dataset, task_level)``; splits on ``(seed, budget)``. Labels carry learnable
  class structure, and a per-instance "regime" varies size, density, and
  signal strength so contexts differ locally.
* :func:`make_tiny_checkpoints` writes randomly initialised encoders in the real
  checkpoint payload format under ``<root>/<source>/<real-style stem>.pt``.
* :func:`tiny_cfg` returns a cfg clone wired to both, with small budgets/epochs.

Family of a synthetic dataset: explicit ``families`` entry (by ``"name:level"``
or ``name``), else name prefix ``multi*`` -> multilabel, ``reg*`` -> regression,
else by level (node -> node_cls, edge -> link, graph -> graph_cls).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Data

from src.moe.routergfm.applications import (
    AppData,
    assemble_app_data,
    build_stats,
    dataset_domain,
    derive_seed,
    gather_labels,
)
from src.moe.routergfm.common import (
    GRAPH_CLS,
    LINK,
    MULTILABEL,
    NODE_CLS,
    REGRESSION,
    AppSpec,
)

# Tiny CPU workloads: intra-op thread pools only add overhead (heads fit ~10x faster).
torch.set_num_threads(1)

FEATURE_DIM = 16
NUM_CLASSES = 3
NUM_LABELS = 4
NUM_TARGETS = 2
TINY_MAX_DIAG = 40
DEFAULT_NUM_INSTANCES = {NODE_CLS: 90, LINK: 96, GRAPH_CLS: 90, MULTILABEL: 80, REGRESSION: 80}
_LEVEL_FAMILY = {"node": NODE_CLS, "edge": LINK, "graph": GRAPH_CLS}
_NOLW_ARCHS = ("fagcn", "h2gcn", "nodeformer", "transformer")

DEFAULT_TARGETS = ("nodea:node",)
DEFAULT_HISTORY_EXTRA = (
    "srca:node", "nodeb:node", "nodec:node",
    "linka:edge", "linkb:edge",
    "srcb:graph", "grapha:graph",
    "multia:graph", "multib:graph",
    "rega:graph", "regb:graph",
)


# --------------------------------------------------------------------------- #
# Synthetic datasets
# --------------------------------------------------------------------------- #
class SyntheticGraphDataset:
    """Minimal stand-in for ``InducedGraphDataset`` / PyG graph datasets."""

    def __init__(self, graphs, *, base_num_nodes: int, base_num_edges: int, name: str):
        self.graphs = list(graphs)
        self.base_num_nodes = int(base_num_nodes)
        self.base_num_edges = int(base_num_edges)
        self.num_features = FEATURE_DIM
        self.name = name

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[int(idx)]


def _random_edges(n: int, p: float, g: torch.Generator) -> torch.Tensor:
    upper = torch.triu(torch.rand(n, n, generator=g) < p, diagonal=1)
    return upper | upper.t()


def _to_edge_index(adj: torch.Tensor) -> torch.Tensor:
    return adj.nonzero(as_tuple=False).t().contiguous()


def _regime(g: torch.Generator) -> Tuple[int, float, float]:
    """(num_nodes, edge probability, feature signal) of a small or a large/dense/noisy context."""
    if float(torch.rand(1, generator=g)) < 0.5:
        return int(torch.randint(5, 8, (1,), generator=g)), 0.3, 1.2
    return int(torch.randint(9, 13, (1,), generator=g)), 0.5, 0.5


def _node_instances(n_items: int, protos: torch.Tensor, g: torch.Generator):
    graphs = []
    num_classes = protos.size(0)
    for i in range(n_items):
        cls = i % num_classes
        n, p, signal = _regime(g)
        adj = _random_edges(n, p, g)
        center = int(torch.randint(0, n, (1,), generator=g))
        other = (center + 1 + int(torch.randint(0, n - 1, (1,), generator=g))) % n
        adj[center, other] = adj[other, center] = True
        node_cls = torch.where(
            torch.rand(n, generator=g) < 0.7,
            torch.full((n,), cls),
            torch.randint(0, num_classes, (n,), generator=g),
        )
        node_cls[center] = cls
        x = protos[node_cls] * signal + 0.8 * torch.randn(n, FEATURE_DIM, generator=g)
        graphs.append(
            Data(
                x=x,
                edge_index=_to_edge_index(adj),
                y=torch.tensor(cls),
                base_node_id=i,
                target_node_index=torch.tensor([center]),
            )
        )
    return graphs


def _link_instances(n_items: int, protos: torch.Tensor, g: torch.Generator):
    graphs = []
    num_classes = protos.size(0)
    for i in range(n_items):
        label = i % 2
        n, p, signal = _regime(g)
        adj = _random_edges(n, p * 0.6, g)
        u, v = torch.randperm(n, generator=g)[:2].tolist()
        rest = [w for w in range(n) if w not in (u, v)]
        if label == 1:
            for w in rest[:2]:
                adj[u, w] = adj[w, u] = adj[v, w] = adj[w, v] = True
        else:
            for w in rest:
                if adj[u, w]:
                    adj[v, w] = adj[w, v] = False
        adj[u, v] = adj[v, u] = False  # the queried pair is never a message edge
        cls_u = int(torch.randint(0, num_classes, (1,), generator=g))
        cls_v = cls_u if label == 1 else (cls_u + 1 + int(torch.randint(0, num_classes - 1, (1,), generator=g))) % num_classes
        node_cls = torch.randint(0, num_classes, (n,), generator=g)
        node_cls[u], node_cls[v] = cls_u, cls_v
        x = protos[node_cls] * signal + 0.8 * torch.randn(n, FEATURE_DIM, generator=g)
        graphs.append(
            Data(
                x=x,
                edge_index=_to_edge_index(adj),
                edge_label_index=torch.tensor([[u], [v]]),
                y=torch.tensor(label),
            )
        )
    return graphs


def _graph_instances(n_items: int, family: str, protos: torch.Tensor, g: torch.Generator):
    graphs = []
    num_classes = protos.size(0)
    w_labels = torch.randn(NUM_LABELS, FEATURE_DIM, generator=g)
    w_targets = torch.randn(NUM_TARGETS, FEATURE_DIM, generator=g)
    for i in range(n_items):
        n, p, signal = _regime(g)
        if family == GRAPH_CLS:
            cls = i % num_classes
            adj = _random_edges(n, 0.15 + 0.15 * cls, g)
            x = protos[cls] * signal + 0.8 * torch.randn(n, FEATURE_DIM, generator=g)
            y = torch.tensor([cls])
        else:
            adj = _random_edges(n, p, g)
            latent = torch.randn(FEATURE_DIM, generator=g)
            x = latent * signal + 0.5 * torch.randn(n, FEATURE_DIM, generator=g)
            if family == MULTILABEL:
                y = (w_labels @ latent > 0).float()
                missing = torch.rand(NUM_LABELS, generator=g) < 0.2
                missing[int(torch.randint(0, NUM_LABELS, (1,), generator=g))] = False
                y[missing] = float("nan")
                y = y.view(1, -1)
            else:
                y = (10.0 * (w_targets @ latent) + 5.0 + 0.5 * n + 0.3 * torch.randn(NUM_TARGETS, generator=g)).view(1, -1)
        graphs.append(Data(x=x, edge_index=_to_edge_index(adj), y=y))
    return graphs


def make_synthetic_dataset(name: str, level: str, family: str, n_items: int) -> SyntheticGraphDataset:
    """Deterministic instance set of one synthetic ``(dataset, task_level)``."""
    g = torch.Generator().manual_seed(derive_seed("synthetic", name, level))
    protos = 1.5 * torch.randn(NUM_CLASSES, FEATURE_DIM, generator=g)
    if family == NODE_CLS:
        graphs = _node_instances(n_items, protos, g)
    elif family == LINK:
        graphs = _link_instances(n_items, protos, g)
    else:
        graphs = _graph_instances(n_items, family, protos, g)
    num_nodes = sum(int(d.num_nodes) for d in graphs)
    num_edges = sum(int(d.edge_index.size(1)) for d in graphs)
    return SyntheticGraphDataset(graphs, base_num_nodes=num_nodes, base_num_edges=num_edges, name=name)


def synthetic_split(app: AppSpec, family: str, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(support, query) positions: few-shot per class, LP ratio split (val dropped), or count split."""
    g = torch.Generator().manual_seed(derive_seed("split", app.data_key))
    n = int(labels.size(0))
    if family == LINK:
        train, val, test = (float(v) for v in app.split)
        order = torch.randperm(n, generator=g)
        n_train = int(round(n * train / (train + val + test)))
        n_val = int(round(n * val / (train + val + test)))
        support, query = order[:n_train], order[n_train + n_val:]
    elif family in (NODE_CLS, GRAPH_CLS):
        support = []
        for cls in torch.unique(labels).tolist():
            members = torch.nonzero(labels == cls, as_tuple=False).view(-1)
            support.append(members[torch.randperm(members.numel(), generator=g)[: int(app.budget)]])
        support = torch.cat(support)
        mask = torch.ones(n, dtype=torch.bool)
        mask[support] = False
        query = torch.nonzero(mask, as_tuple=False).view(-1)
    else:
        order = torch.randperm(n, generator=g)
        support, query = order[: int(app.budget)], order[int(app.budget):]
    return torch.sort(support).values, torch.sort(query).values


class SyntheticDataProvider:
    """``DataProvider`` over synthetic datasets (tests only). ``calls`` records loaded app keys."""

    def __init__(
        self,
        families: Optional[Mapping[str, str]] = None,
        *,
        num_instances: Optional[int] = None,
        max_diagnostic: int = TINY_MAX_DIAG,
    ):
        self.families = dict(families or {})
        self.num_instances = num_instances
        self.max_diagnostic = int(max_diagnostic)
        self.calls = []
        self._datasets: Dict[Tuple[str, str], SyntheticGraphDataset] = {}

    def family_of(self, dataset: str, level: str) -> str:
        for key in (f"{dataset}:{level}", dataset):
            if key in self.families:
                return self.families[key]
        if level == "graph" and dataset.startswith("multi"):
            return MULTILABEL
        if level == "graph" and dataset.startswith("reg"):
            return REGRESSION
        return _LEVEL_FAMILY[level]

    def dataset(self, name: str, level: str) -> SyntheticGraphDataset:
        key = (name, level)
        if key not in self._datasets:
            family = self.family_of(name, level)
            n_items = int(self.num_instances or DEFAULT_NUM_INSTANCES[family])
            self._datasets[key] = make_synthetic_dataset(name, level, family, n_items)
        return self._datasets[key]

    def load(self, app: AppSpec) -> AppData:
        self.calls.append(app.key)
        family = self.family_of(app.dataset, app.task_level)
        dataset = self.dataset(app.dataset, app.task_level)
        all_pos = torch.arange(len(dataset))
        all_labels = gather_labels(dataset, all_pos, family)
        split_labels = all_labels if family in (NODE_CLS, GRAPH_CLS) else torch.zeros(len(dataset))
        support, query = synthetic_split(app, family, split_labels)
        if family in (NODE_CLS, GRAPH_CLS):
            num_classes = int(all_labels.max()) + 1
        else:
            num_classes = {LINK: 2, MULTILABEL: NUM_LABELS, REGRESSION: NUM_TARGETS}[family]
        if app.task_level == "graph":
            num_nodes = dataset.base_num_nodes / len(dataset)
            num_edges = dataset.base_num_edges / len(dataset)
        else:
            num_nodes, num_edges = dataset.base_num_nodes, dataset.base_num_edges
        meta = {
            "level": app.task_level,
            "task_family": family,
            "num_classes": num_classes,
            "label_dim": 1 if family in (NODE_CLS, GRAPH_CLS, LINK) else num_classes,
            "in_dim": FEATURE_DIM,
            "support_pos": support,
            "query_pos": query,
            "support_labels": all_labels[support],
            "query_labels": all_labels[query],
            "stats": build_stats(
                level=app.task_level,
                family=family,
                num_instances=len(dataset),
                num_nodes=num_nodes,
                num_edges=num_edges,
                feature_dim=FEATURE_DIM,
                num_classes=num_classes,
                domain=dataset_domain(app.dataset),
                budget=int(app.budget),
            ),
        }
        return assemble_app_data(app, meta, dataset, self.max_diagnostic)

    def base_graph(self, app: AppSpec) -> Data:
        """Synthetic input graph: the disjoint union of the dataset's instance graphs (no queried pair is an edge)."""
        if app.task_level not in ("node", "edge"):
            raise ValueError(f"{app.key}: only node and link applications have a single base graph.")
        parts, offset = [], 0
        for graph in self.dataset(app.dataset, app.task_level).graphs:
            parts.append(graph.edge_index + offset)
            offset += int(graph.num_nodes)
        return Data(edge_index=torch.cat(parts, dim=1), num_nodes=offset)


# --------------------------------------------------------------------------- #
# Tiny expert checkpoints
# --------------------------------------------------------------------------- #
def checkpoint_stem(arch: str, objective: str, source: str, level: str, *, hidden_dim: int, out_dim: int, num_layers: int, seed: int = 42) -> str:
    """Pretrain run name in the real ``build_pretrain_run_name_from_cfg`` format."""
    method = "infograph-nolw" if objective == "infograph" and arch in _NOLW_ARCHS else objective
    induced = int(level != "graph")
    split = "_split80-10-10" if objective == "supervised" else ""
    return (
        f"{method}_{source}_task{level}_induced{induced}{split}_{arch}"
        f"_h{hidden_dim}_o{out_dim}_l{num_layers}_e1_lr0.001_bs4_seed{seed}"
    )


def default_source_levels(sources: Sequence[str]) -> Dict[str, str]:
    return {s: ("node" if i % 2 == 0 else "graph") for i, s in enumerate(sources)}


def make_tiny_checkpoints(
    root,
    archs: Sequence[str] = ("gcn", "gin"),
    objectives: Sequence[str] = ("dgi", "edge_pred"),
    sources: Sequence[str] = ("srca", "srcb"),
    *,
    source_levels: Optional[Mapping[str, str]] = None,
    hidden_dim: int = 8,
    out_dim: int = 8,
    num_layers: int = 2,
) -> Dict[Tuple[str, str, str], Path]:
    """Write one random encoder per (arch, objective, source) in the real payload format (idempotent)."""
    from src.config import cfg as base_cfg
    from src.model import build_encoder_from_cfg
    from src.utils.checkpoint import cfg_to_dict

    root = Path(root)
    levels = dict(default_source_levels(sources), **dict(source_levels or {}))
    out: Dict[Tuple[str, str, str], Path] = {}
    for source in sources:
        for objective in objectives:
            for arch in archs:
                stem = checkpoint_stem(
                    arch, objective, source, levels[source], hidden_dim=hidden_dim, out_dim=out_dim, num_layers=num_layers
                )
                path = root / source / f"{stem}.pt"
                out[(arch, objective, source)] = path
                if path.is_file():
                    continue
                cfg = base_cfg.clone()
                cfg.model.name = arch
                cfg.model.in_dim = FEATURE_DIM
                cfg.model.hidden_dim = hidden_dim
                cfg.model.out_dim = out_dim
                cfg.model.num_layers = num_layers
                cfg.model.gat.heads = 2
                cfg.model.nodeformer.heads = 2
                cfg.model.nodeformer.num_random_features = 8
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(derive_seed("ckpt", arch, objective, source))
                    encoder = build_encoder_from_cfg(cfg, FEATURE_DIM)
                payload = {
                    "cfg": {"model": cfg_to_dict(cfg.model)},
                    "model_state": encoder.state_dict(),
                    "dataset": {"num_node_features": FEATURE_DIM},
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".tmp{os.getpid()}")
                torch.save(payload, tmp)
                os.replace(tmp, path)
    return out


# --------------------------------------------------------------------------- #
# Tiny configuration
# --------------------------------------------------------------------------- #
def tiny_cfg(
    tmp_path,
    *,
    targets: Sequence[str] = DEFAULT_TARGETS,
    history_extra: Sequence[str] = DEFAULT_HISTORY_EXTRA,
    budgets: Sequence[int] = (3, 6),
    seeds: Sequence[int] = (42,),
    history_seeds: Sequence[int] = (42,),
    archs: Sequence[str] = ("gcn", "gin"),
    objectives: Sequence[str] = ("dgi", "edge_pred"),
    sources: Sequence[str] = ("srca", "srcb"),
    source_levels: Optional[Mapping[str, str]] = None,
    write_checkpoints: bool = True,
):
    """cfg clone whose ``moe.routergfm`` points at tiny checkpoints and synthetic applications."""
    from src.config import cfg as base_cfg

    tmp_path = Path(tmp_path)
    ckpt_root = tmp_path / "checkpoints"
    if write_checkpoints:
        make_tiny_checkpoints(ckpt_root, archs, objectives, sources, source_levels=source_levels)

    cfg = base_cfg.clone()
    rg = cfg.moe.routergfm
    rg.output_root = str(tmp_path / "routergfm")
    rg.device_batch_size = 32

    rg.experts.checkpoint_root = str(ckpt_root)
    rg.experts.architectures = list(archs)
    rg.experts.objectives = list(objectives)
    rg.experts.sources = list(sources)
    rg.experts.strict = True

    rg.apps.targets = list(targets)
    rg.apps.history_extra = list(history_extra)
    rg.apps.budgets = [int(b) for b in budgets]
    rg.apps.seeds = [int(s) for s in seeds]
    rg.apps.history_seeds = [int(s) for s in history_seeds]
    rg.apps.max_diagnostic = TINY_MAX_DIAG
    rg.apps.data.root = str(tmp_path / "data" / "datasets")
    rg.apps.data.split_root = str(tmp_path / "data" / "splits")
    rg.apps.data.induced_root = str(tmp_path / "data" / "induced")
    rg.apps.data.feature_svd_dir = str(tmp_path / "data" / "feature_svd")
    rg.apps.data.graph_filter_dir = str(tmp_path / "data" / "filters")

    rg.heads.epochs = 40
    rg.heads.hidden_dim = 16
    rg.heads.oof_folds = 3

    rg.descriptors.num_spectral = 3
    rg.archive.num_cells = 3
    rg.archive.min_cell_size = 4
    rg.archive.kmeans_iters = 10
    rg.archive.num_families = 2

    rg.graph.text_backend = "hash"
    rg.graph.hash_dim = 32
    rg.graph.text_cache_dir = str(tmp_path / "text_cache")
    rg.graph.local_files_only = True

    rg.router.hidden_dim = 16
    rg.router.key_dim = 8
    rg.router.key_hidden_dim = 16
    rg.router.topk = 2
    rg.router.retrieval_j = 4
    rg.router.per_app_cap = 2
    rg.router.epochs = 3
    rg.router.episodes_per_step = 2
    rg.router.local_pairs_per_episode = 64
    rg.router.patience = 2
    rg.router.num_val_datasets = 1
    rg.router.rho_grid = [0.0, 1.0]
    rg.router.tau_grid = [0.05, 0.2]

    rg.integration.stacking_epochs = 20
    rg.integration.local_mlp_hidden = 8
    rg.integration.local_mlp_epochs = 20
    rg.integration.eval_cells = 3

    rg.deploy.target = str(targets[0])
    rg.deploy.budget = int(budgets[0])
    rg.deploy.seed = int(seeds[0])
    rg.benchmark.num_runs = 1
    rg.benchmark.tasks_tsv = str(tmp_path / "routergfm_tasks.tsv")

    rg.baselines.datasets = list(targets)
    rg.baselines.budgets = [int(b) for b in budgets]
    rg.baselines.num_runs = 1
    rg.baselines.topk = 2
    rg.baselines.candidate_pool = 4
    rg.baselines.output_dir = str(tmp_path / "baselines")
    rg.analysis.shift_root = str(tmp_path / "splits_shift")
    return cfg


__all__ = [
    "DEFAULT_HISTORY_EXTRA",
    "DEFAULT_TARGETS",
    "FEATURE_DIM",
    "NUM_CLASSES",
    "NUM_LABELS",
    "NUM_TARGETS",
    "SyntheticDataProvider",
    "SyntheticGraphDataset",
    "TINY_MAX_DIAG",
    "checkpoint_stem",
    "make_synthetic_dataset",
    "make_tiny_checkpoints",
    "synthetic_split",
    "tiny_cfg",
]
