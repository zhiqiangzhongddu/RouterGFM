"""GMoPE Stage A: multi-source pretraining of prompt-conditioned experts.

One checkpoint per route (``node``: node-level pool sources as induced
ego-subgraphs, serving node and link targets; ``graph``: graph-level pool
sources, serving graph targets). Each batch is homogeneous (one source);
all M experts are scored under an identical RNG seed without gradients,
the soft top-K router (Eq. 10) weights the selected experts, and only
those are re-run with gradients (Eq. 16). The checkpoint is independent of
the downstream target, budget and seed.
"""

from __future__ import annotations

import fcntl
import os
import random
import time
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader

from src.data_loader import create_dataset
from src.moe.identity import behavior_fingerprint
from src.pretrain.registry import build_pretrain_task, get_pretrain_task_class
from src.utils.checkpoint import cfg_to_dict, save_json_atomic, save_torch_atomic
from src.utils.dataset_helpers import shared_induced_root, shared_split_root
from src.utils.paths import ensure_dir
from src.utils.random import set_seed
from src.utils.run_helpers import resolve_seeds

from .model import GMoPEModel
from .routing import scores_from_losses, shared_rng, soft_topk_gate

ROUTES = ("node", "graph")


# ---------------------------------------------------------------------------
# Route / size resolution (shared with the finetune runner)
# ---------------------------------------------------------------------------

def resolve_route(task_level: str) -> str:
    """Node and edge targets use the node route; graph targets the graph route."""
    level = str(task_level or "").lower()
    if level in {"node", "edge"}:
        return "node"
    if level == "graph":
        return "graph"
    raise ValueError(f"[GMoPE] Cannot resolve a route for task_level '{task_level}'.")


def _check_route(route: str) -> str:
    route = str(route).lower()
    if route not in ROUTES:
        raise ValueError(f"[GMoPE] Unknown route '{route}'; expected one of {ROUTES}.")
    return route


def route_sources(gmope_cfg, route: str) -> list[str]:
    route = _check_route(route)
    sources = gmope_cfg.pretrain.node_sources if route == "node" else gmope_cfg.pretrain.graph_sources
    sources = [str(name) for name in sources]
    if not sources:
        raise ValueError(f"[GMoPE] No pretraining sources configured for route '{route}'.")
    return sources


def resolve_num_experts(gmope_cfg, route: str) -> int:
    """M; 0 means one expert per pretraining source of the route (paper: M = N)."""
    num = int(gmope_cfg.num_experts or 0)
    return num if num > 0 else len(route_sources(gmope_cfg, route))


def resolve_top_k(gmope_cfg, route: str, stage: str) -> int:
    """K for ``stage`` in {pretrain, finetune}; 0 means node route: M, graph route: 1."""
    num = resolve_num_experts(gmope_cfg, route)
    k = int(getattr(gmope_cfg, stage).top_k or 0)
    if k <= 0:
        k = num if _check_route(route) == "node" else 1
    return min(k, num)


def resolve_pretrain_seed(cfg) -> int:
    policy = str(cfg.moe.gmope.pretrain.seed_policy).lower()
    if policy == "shared":
        return int(resolve_seeds(cfg)[0])
    if policy == "per_seed":
        return int(cfg.seed)
    raise ValueError(f"[GMoPE] Unknown pretrain.seed_policy '{policy}' (shared | per_seed).")


def objective_cfg(cfg, objective: str, meta: dict):
    """Clone of ``cfg`` whose ``model`` block describes one expert, for PretrainTask builders."""
    obj_cfg = cfg.clone()
    obj_cfg.pretrain.method = str(objective)
    obj_cfg.model.name = str(meta["gnn_type"])
    obj_cfg.model.in_dim = int(meta["in_dim"])  # objectives see x before the prompt is appended
    obj_cfg.model.hidden_dim = int(meta["hidden_dim"])
    obj_cfg.model.out_dim = int(meta["out_dim"])
    obj_cfg.model.num_layers = int(meta["num_layers"])
    obj_cfg.model.dropout = float(meta["dropout"])
    obj_cfg.model.graph_pooling = str(meta["graph_pooling"])
    obj_cfg.model.use_batchnorm = bool(meta["use_batchnorm"])
    obj_cfg.model.activation = str(meta["act"])
    return obj_cfg


def build_model(meta: dict) -> GMoPEModel:
    return GMoPEModel(
        num_experts=int(meta["num_experts"]),
        in_dim=int(meta["in_dim"]),
        prompt_dim=int(meta["prompt_dim"]),
        hidden_dim=int(meta["hidden_dim"]),
        out_dim=int(meta["out_dim"]),
        num_layers=int(meta["num_layers"]),
        gnn_type=str(meta["gnn_type"]),
        dropout=float(meta["dropout"]),
        act=str(meta["act"]),
        graph_pooling=str(meta["graph_pooling"]),
        use_batchnorm=bool(meta["use_batchnorm"]),
    )


def build_objectives(cfg, meta: dict) -> nn.ModuleList:
    """One objective instance per expert, so objective heads stay per-expert."""
    obj_cfg = objective_cfg(cfg, meta["objective"], meta)
    return nn.ModuleList(
        build_pretrain_task(str(meta["objective"]), obj_cfg) for _ in range(int(meta["num_experts"]))
    )


def load_gmope_checkpoint(cfg, path: str, device) -> tuple[GMoPEModel, nn.ModuleList, dict]:
    """Rebuild ``(model, per-expert objectives, meta)`` from a route checkpoint."""
    payload = torch.load(path, map_location="cpu")
    meta = dict(payload["meta"])
    model = build_model(meta)
    model.load_state_dict(payload["model_state"])
    objectives = build_objectives(cfg, meta)
    for objective, state in zip(objectives, payload["objective_states"]):
        objective.load_state_dict(state)
    return model.to(device), objectives.to(device), meta


# ---------------------------------------------------------------------------
# Batching and one pretraining step
# ---------------------------------------------------------------------------

class MultiSourceBatchStream:
    """Randomly interleaved homogeneous batches, at most ``max_batches_per_source`` per source per epoch."""

    def __init__(self, loaders: dict[str, DataLoader], *, max_batches_per_source: int, seed: int):
        self.loaders = dict(loaders)
        self.max_batches_per_source = int(max_batches_per_source)
        self.seed = int(seed)
        self._epoch = 0

    def _counts(self) -> dict[str, int]:
        cap = self.max_batches_per_source
        return {
            name: (min(len(loader), cap) if cap > 0 else len(loader))
            for name, loader in self.loaders.items()
        }

    def __len__(self) -> int:
        return sum(self._counts().values())

    def __iter__(self) -> Iterator[tuple[str, object]]:
        order = [name for name, count in self._counts().items() for _ in range(count)]
        random.Random(self.seed * 1_000_003 + self._epoch).shuffle(order)
        self._epoch += 1
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        for name in order:
            yield name, next(iterators[name])


def pretrain_step(
    model: GMoPEModel,
    objectives: nn.ModuleList,
    batch,
    device,
    *,
    top_k: int,
    tau: float,
    ortho_weight: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(loss, gate [M], per-expert losses [M] (detached))`` for one batch (Eq. 16).

    Every expert runs under ``shared_rng(seed)``. With K < M the scoring pass
    is gradient-free and only the selected experts are re-run with
    gradients; with K == M one pass with gradients provides both.
    """
    num = model.num_experts

    def _loss(m: int) -> torch.Tensor:
        with shared_rng(seed):
            loss, _ = objectives[m].step(model.bound(m), batch, device)
        return loss

    if top_k >= num:
        losses = torch.stack([_loss(m) for m in range(num)])
        gate = soft_topk_gate(scores_from_losses(losses), top_k, tau)
        task_loss = (gate.to(losses.device) * losses).sum() / num
        detached = losses.detach()
    else:
        with torch.no_grad():
            detached = torch.stack([_loss(m).detach() for m in range(num)])
        gate = soft_topk_gate(scores_from_losses(detached), top_k, tau)
        selected = torch.nonzero(gate > 0).view(-1).tolist()
        task_loss = sum(gate[m].to(detached.device) * _loss(m) for m in selected) / num
    loss = float(ortho_weight) * model.ortho_loss() + task_loss
    return loss, gate.detach().cpu(), detached.float().cpu()


@contextmanager
def _file_lock(path: str):
    """Exclusive advisory lock next to ``path`` so concurrent jobs pretrain a route once."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(f"{path}.lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Pretrainer
# ---------------------------------------------------------------------------

class GMoPEPretrainer:
    """Pretrain (or reuse) the GMoPE checkpoint of one route."""

    def __init__(self, cfg, route: str):
        self.cfg = cfg
        self.gmope_cfg = cfg.moe.gmope
        self.route = _check_route(route)
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        self.sources = route_sources(self.gmope_cfg, self.route)
        self.num_experts = resolve_num_experts(self.gmope_cfg, self.route)
        self.top_k = resolve_top_k(self.gmope_cfg, self.route, "pretrain")
        self.seed = resolve_pretrain_seed(cfg)
        self.objective = str(self.gmope_cfg.pretrain.objective)
        if get_pretrain_task_class(self.objective) is None:
            raise ValueError(f"[GMoPE] Unknown pretraining objective '{self.objective}'.")
        self.meta = self._meta()
        self.history: list[dict] = []

    def _meta(self) -> dict:
        g = self.gmope_cfg
        return {
            "route": self.route,
            "sources": list(self.sources),
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "in_dim": int(g.in_dim),
            "prompt_dim": int(g.prompt_dim),
            "gnn_type": str(g.expert.gnn_type),
            "num_layers": int(g.expert.num_layers),
            "hidden_dim": int(g.expert.hidden_dim),
            "out_dim": int(g.expert.out_dim),
            "dropout": float(g.expert.dropout),
            "graph_pooling": str(g.expert.graph_pooling),
            "use_batchnorm": bool(g.expert.use_batchnorm),
            "act": str(getattr(self.cfg.model, "activation", "relu")),
            "objective": self.objective,
            "tau": float(g.tau),
            "ortho_weight": float(g.ortho_weight),
            "seed": self.seed,
        }

    def _fingerprint(self) -> str:
        g = self.gmope_cfg
        ds = g.dataset
        payload = {
            "in_dim": int(g.in_dim),
            "expert": cfg_to_dict(g.expert),
            "num_experts": self.num_experts,
            "prompt_dim": int(g.prompt_dim),
            "tau": float(g.tau),
            "ortho_weight": float(g.ortho_weight),
            "pretrain": {
                "objective": self.objective,
                "top_k": self.top_k,
                "epochs": int(g.pretrain.epochs),
                "lr": float(g.pretrain.lr),
                "weight_decay": float(g.pretrain.weight_decay),
                "batch_size": int(g.pretrain.batch_size),
                "max_batches_per_source": int(g.pretrain.max_batches_per_source),
                "sources": list(self.sources),
            },
        }
        objective_block = getattr(self.cfg.pretrain, self.objective, None)
        external = {
            "model_activation": self.meta["act"],
            "objective_cfg": cfg_to_dict(objective_block) if objective_block is not None else {},
            "source_data": {
                "root": str(ds.root),
                "feat_reduction": bool(ds.feat_reduction),
                "feat_reduction_svd_dim": int(ds.feat_reduction_svd_dim),
                "feature_svd_dir": str(ds.feature_svd_dir),
                "induced_min_size": int(ds.induced_min_size),
                "induced_max_size": int(ds.induced_max_size),
                "induced_max_hops": int(ds.induced_max_hops),
                "shared_split_root": shared_split_root(self.cfg),
                "shared_induced_root": shared_induced_root(self.cfg, str(ds.induced_root)),
            },
        }
        return behavior_fingerprint(payload, external_behavior=external)

    def run_name(self) -> str:
        g = self.gmope_cfg
        parts = [
            "gmope-pre",
            self.route,
            self.objective,
            f"m{self.num_experts}",
            f"k{self.top_k}",
            f"dp{int(g.prompt_dim)}",
            str(g.expert.gnn_type),
            f"l{int(g.expert.num_layers)}",
            f"h{int(g.expert.hidden_dim)}",
            f"o{int(g.expert.out_dim)}",
            f"e{int(g.pretrain.epochs)}",
            f"cfg{self._fingerprint()}",
            f"seed{self.seed}",
        ]
        return "_".join(parts)

    def checkpoint_path(self) -> str:
        return os.path.join(str(self.gmope_cfg.pretrain.checkpoint_dir), self.route, f"{self.run_name()}.pt")

    def _log_path(self) -> str:
        return os.path.join(str(self.gmope_cfg.pretrain.checkpoint_dir), self.route, f"{self.run_name()}_log.json")

    # ------------------------------------------------------------------ #
    def _build_source_dataset(self, name: str):
        """Full source dataset as the repo's unsupervised pretraining builds it."""
        ds = self.gmope_cfg.dataset
        node_level = self.route == "node"
        dataset = create_dataset(
            name=name,
            root=ds.root,
            task_level="node" if node_level else "graph",
            feat_reduction=ds.feat_reduction,
            feat_reduction_dim=int(ds.feat_reduction_svd_dim),
            persist_feature_svd=ds.feat_reduction,
            feature_svd_dir=str(ds.feature_svd_dir),
            induced=node_level,
            induced_min_size=int(ds.induced_min_size),
            induced_max_size=int(ds.induced_max_size),
            induced_max_hops=int(ds.induced_max_hops),
            cache_induced=True,
            split_root="" if node_level else shared_split_root(self.cfg),
            induced_root=shared_induced_root(self.cfg, str(ds.induced_root)),
            split=None,
            seed=self.seed,
            pad_featureless_features=True,
        )
        feat_dim = int(dataset[0].x.size(-1))
        if feat_dim != int(self.gmope_cfg.in_dim):
            raise ValueError(
                f"[GMoPE] Source '{name}' has feature dim {feat_dim}; moe.gmope.in_dim is "
                f"{int(self.gmope_cfg.in_dim)} (all sources and targets must share d0)."
            )
        return dataset

    def _make_loader(self, dataset) -> DataLoader:
        pre = self.gmope_cfg.pretrain
        min_graphs = int(getattr(get_pretrain_task_class(self.objective), "min_graphs_per_batch", 1))
        if int(pre.batch_size) < min_graphs:
            raise ValueError(
                f"[GMoPE] pretrain.batch_size={pre.batch_size} is below the objective's "
                f"min_graphs_per_batch={min_graphs}."
            )
        return DataLoader(
            dataset=dataset,
            batch_size=int(pre.batch_size),
            num_workers=int(pre.num_workers),
            shuffle=True,
            drop_last=False,
        )

    def fit(self) -> str:
        """Pretrain unless a checkpoint exists (and ``skip_if_exists``); return its path."""
        path = self.checkpoint_path()
        skip = bool(self.gmope_cfg.pretrain.skip_if_exists)
        if skip and os.path.isfile(path):
            print(f"[GMoPE][Pretrain:{self.route}] Checkpoint already exists, skipping: {path}")
            return path
        with _file_lock(path):
            if skip and os.path.isfile(path):
                print(f"[GMoPE][Pretrain:{self.route}] Checkpoint written by another job: {path}")
                return path
            self._train_and_save(path)
        return path

    def _train_and_save(self, path: str) -> None:
        pre = self.gmope_cfg.pretrain
        set_seed(self.seed)
        loaders = {name: self._make_loader(self._build_source_dataset(name)) for name in self.sources}
        # Re-seed so initialisation does not depend on dataset-loading RNG use.
        set_seed(self.seed)
        model = build_model(self.meta).to(self.device)
        objectives = build_objectives(self.cfg, self.meta).to(self.device)
        params = list(model.parameters())
        for objective in objectives:
            params.extend(objective.parameters_to_optimize())
        optimizer = optim.Adam(params, lr=float(pre.lr), weight_decay=float(pre.weight_decay))
        stream = MultiSourceBatchStream(
            loaders, max_batches_per_source=int(pre.max_batches_per_source), seed=self.seed,
        )
        batch_rng = random.Random(self.seed)
        print(
            f"[GMoPE][Pretrain:{self.route}] M={self.num_experts} K={self.top_k} "
            f"objective={self.objective} sources={self.sources} batches/epoch={len(stream)}"
        )

        self.history = []
        num = self.num_experts
        epochs = int(pre.epochs)
        for epoch in range(1, epochs + 1):
            start = time.time()
            model.train()
            objectives.train()
            total_loss = 0.0
            num_batches = 0
            selected = {name: torch.zeros(num) for name in self.sources}
            gate_mass = {name: torch.zeros(num) for name in self.sources}
            counts = {name: 0 for name in self.sources}
            for source, batch in stream:
                optimizer.zero_grad()
                loss, gate, _ = pretrain_step(
                    model, objectives, batch, self.device,
                    top_k=self.top_k,
                    tau=float(self.gmope_cfg.tau),
                    ortho_weight=float(self.gmope_cfg.ortho_weight),
                    seed=batch_rng.randrange(2**31),
                )
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                num_batches += 1
                selected[source] += (gate > 0).float()
                gate_mass[source] += gate
                counts[source] += 1
            if num_batches == 0:
                raise RuntimeError(f"[GMoPE][Pretrain:{self.route}] No pretraining batches.")
            routing = {
                name: {
                    "selected": (selected[name] / max(1, counts[name])).tolist(),
                    "gate": (gate_mass[name] / max(1, counts[name])).tolist(),
                }
                for name in self.sources
            }
            with torch.no_grad():
                ortho = float(model.ortho_loss().item())
            duration = time.time() - start
            self.history.append({
                "epoch": epoch,
                "loss": total_loss / num_batches,
                "ortho_loss": ortho,
                "duration_sec": duration,
                "routing": routing,
            })
            top = " ".join(
                f"{name}->e{int(torch.tensor(routing[name]['gate']).argmax())}" for name in self.sources
            )
            print(
                f"[GMoPE][Pretrain:{self.route}][Epoch {epoch}/{epochs}] "
                f"loss={total_loss / num_batches:.4f} ortho={ortho:.4f} routing[{top}] time={duration:.1f}s"
            )

        save_torch_atomic(path, {
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "objective_states": [
                {k: v.detach().cpu() for k, v in objective.state_dict().items()} for objective in objectives
            ],
            "meta": dict(self.meta, run_name=self.run_name()),
            "history": self.history,
            "cfg": cfg_to_dict(self.cfg),
        })
        save_json_atomic(self._log_path(), {"meta": self.meta, "history": self.history})
        print(f"[GMoPE][Pretrain:{self.route}] Saved checkpoint: {path}")


__all__ = [
    "GMoPEPretrainer",
    "MultiSourceBatchStream",
    "ROUTES",
    "build_model",
    "build_objectives",
    "load_gmope_checkpoint",
    "objective_cfg",
    "pretrain_step",
    "resolve_num_experts",
    "resolve_pretrain_seed",
    "resolve_route",
    "resolve_top_k",
    "route_sources",
]
