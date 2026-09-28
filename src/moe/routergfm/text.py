"""Node descriptions and frozen text embeddings for the context graph (Sec. 3.3, App. B.1).

Templates are metadata-driven so historical and newly inserted applications or
experts are described by the same functions. An optional JSON file
``<graph.description_dir>/{dataset,architecture,objective}/<name>.json`` with a
``summary`` or ``description`` entry is appended to the matching description.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch

from src.data_loader.dataset_domains import CLASS_TO_DOMAIN, KEYWORD_DOMAINS, NAME_TO_DOMAIN
from src.utils.checkpoint import save_torch_atomic

from .common import GRAPH_CLS, LINK, MULTILABEL, NODE_CLS, REGRESSION, AppSpec, ExpertSpec

ADAPTATION_RULE = "frozen pretrained encoder with a task head fitted on the labeled support set"

ARCHITECTURE_TEXT = {
    "gcn": "GCN (graph convolutional network): message passing that averages degree-normalized "
    "neighbor features, a low-pass filter suited to homophilous graphs.",
    "gat": "GAT (graph attention network): message passing with learned multi-head attention "
    "weights over neighbors.",
    "gin": "GIN (graph isomorphism network): sum aggregation followed by an MLP, as expressive as "
    "the Weisfeiler-Lehman test; common for molecular graphs.",
    "h2gcn": "H2GCN: separates ego and neighbor embeddings, aggregates higher-order neighborhoods, "
    "and combines intermediate layers; designed for heterophilous graphs.",
    "fagcn": "FAGCN (frequency adaptive graph convolution): signed attention mixing low- and "
    "high-frequency signals, for homophilous and heterophilous graphs.",
    "nodeformer": "NodeFormer: scalable graph transformer with kernelized all-pair message passing "
    "over a learned latent structure.",
    "transformer": "Graph transformer: global self-attention over all nodes with graph structure "
    "as positional information, capturing long-range interactions.",
}

OBJECTIVE_TEXT = {
    "attr_masking": "Attribute masking: self-supervised pretraining that masks node attributes and "
    "reconstructs them from the graph context.",
    "context_pred": "Context prediction: self-supervised pretraining that matches a node's "
    "neighborhood embedding with its surrounding context subgraph.",
    "dgi": "DGI (deep graph infomax): contrastive pretraining that maximizes mutual information "
    "between node embeddings and a graph summary against corrupted graphs.",
    "edge_pred": "Edge prediction: self-supervised pretraining that predicts whether a link exists "
    "between two nodes.",
    "graphcl": "GraphCL: contrastive pretraining between augmented views of a graph (node dropping, "
    "edge perturbation, attribute masking, subgraph sampling).",
    "infograph": "InfoGraph: contrastive pretraining that maximizes mutual information between "
    "graph-level and substructure-level representations.",
    "supervised": "Supervised pretraining on the labels of the source corpus task.",
}

# Paper datasets that the shared name map does not cover.
_EXTRA_DOMAINS = {
    "flickr": "social",
    "actor": "social",
    "proteins": "biology",
    "mnist": "vision superpixel",
    "cifar10": "vision superpixel",
}
# The one domain vocabulary: every value of dataset_domain and of the dataset-class
# map (real-data metadata), plus "unknown"; the metadata one-hot uses it.
DOMAINS = tuple(
    sorted(
        set(CLASS_TO_DOMAIN.values())
        | set(NAME_TO_DOMAIN.values())
        | {d for d, _ in KEYWORD_DOMAINS}
        | set(_EXTRA_DOMAINS.values())
    )
) + ("unknown",)

_LEVEL_UNIT = {"node": "node", "edge": "node pair", "graph": "graph"}
_INSTANCE_TEXT = {
    "node": "Each instance is the subgraph around a marked target node.",
    "edge": "Each instance is the enclosing subgraph of a candidate node pair with both endpoints "
    "marked and the candidate link removed.",
    "graph": "Each instance is a whole graph.",
}


def dataset_domain(name: str) -> str:
    """Coarse domain of a dataset name (shared name map, then keyword rules)."""
    key = str(name).strip().lower()
    if key in _EXTRA_DOMAINS:
        return _EXTRA_DOMAINS[key]
    if key in NAME_TO_DOMAIN:
        return NAME_TO_DOMAIN[key]
    for domain, keywords in KEYWORD_DOMAINS:
        if any(token in key for token in keywords):
            return domain
    return "unknown"


def _json_description(description_dir: str, kind: str, name: str) -> str:
    if not description_dir:
        return ""
    path = Path(description_dir) / kind / f"{name}.json"
    if not path.is_file():
        return ""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return str(payload.get("summary") or payload.get("description") or "").strip()


def _join(parts: Sequence[str]) -> str:
    return " ".join(part for part in parts if part)


def _count(stats: Mapping[str, float], key: str) -> Optional[float]:
    value = stats.get(key)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _task_text(level: str, family: str, num_out: Optional[float]) -> str:
    n = f"{int(round(num_out))} " if num_out is not None else ""
    if family in (NODE_CLS, GRAPH_CLS):
        return f"{level} classification with {n}classes"
    if family == LINK:
        return "link prediction (binary existence of a candidate link)"
    if family == MULTILABEL:
        return f"multi-label {level} classification with {n}binary targets"
    if family == REGRESSION:
        return f"{level}-level regression with {n}numerical targets"
    raise ValueError(f"Unknown task family {family!r}")


def _budget_text(app: AppSpec, family: str) -> str:
    unit = _LEVEL_UNIT[app.task_level]
    if family == LINK:
        train, val, test = (int(round(100 * v)) for v in app.lp_split)
        return (
            f"Supervision: training links from a {train}%-{val}%-{test}% positive-edge split with "
            f"sampled negatives (budget block {int(app.budget)})."
        )
    if family in (NODE_CLS, GRAPH_CLS):
        return f"Supervision: {int(app.budget)} labeled support examples per class."
    return f"Supervision: {int(app.budget)} labeled support {unit}s in total."


def describe_application(
    app: AppSpec, stats: Mapping[str, float], family: str, *, description_dir: str = ""
) -> str:
    """Dataset, domain, sizes, feature dim, task, output count, and support budget."""
    level = app.task_level
    nodes, edges = _count(stats, "num_nodes"), _count(stats, "num_edges")
    instances, degree = _count(stats, "num_instances"), _count(stats, "avg_degree")
    feat_dim = _count(stats, "feature_dim")
    if level == "graph":
        graph_text = _join([
            f"{int(round(instances))} graphs" if instances is not None else "Graphs",
            f"with on average {nodes:.1f} nodes" if nodes is not None else "",
            f"and {edges:.1f} edges" if edges is not None else "",
        ]) + "."
    else:
        graph_text = _join([
            "One graph",
            f"with {int(round(nodes))} nodes" if nodes is not None else "",
            f"and {int(round(edges))} edges" if edges is not None else "",
            f"(average degree {degree:.2f})" if degree is not None else "",
        ]) + "."
    return _join([
        f"Application: {_task_text(level, family, _count(stats, 'num_classes'))} on the "
        f"{app.dataset} dataset ({dataset_domain(app.dataset)} domain).",
        graph_text,
        f"Node features have {int(round(feat_dim))} dimensions." if feat_dim is not None else "",
        _INSTANCE_TEXT[level],
        _budget_text(app, family),
        _json_description(description_dir, "dataset", app.dataset),
    ])


def describe_corpus(name: str, *, description_dir: str = "") -> str:
    return _join([
        f"Source corpus: the {name} dataset ({dataset_domain(name)} domain), used to pretrain "
        "graph neural network experts.",
        _json_description(description_dir, "dataset", name),
    ])


def describe_architecture(name: str, *, description_dir: str = "") -> str:
    base = ARCHITECTURE_TEXT.get(str(name).lower(), f"{name}: a graph neural network architecture.")
    return _join([f"Architecture: {base}", _json_description(description_dir, "architecture", name)])


def describe_objective(name: str, *, description_dir: str = "") -> str:
    base = OBJECTIVE_TEXT.get(str(name).lower(), f"{name}: a graph pretraining objective.")
    return _join([f"Pretraining objective: {base}", _json_description(description_dir, "objective", name)])


def describe_expert(spec: ExpertSpec, *, description_dir: str = "") -> str:
    """Architecture, objective, source corpus (+ domain), and adaptation rule."""
    return _join([
        f"Expert: a {spec.architecture} encoder pretrained with {spec.objective} on the "
        f"{spec.source} dataset ({dataset_domain(spec.source)} domain, "
        f"{spec.source_task_level}-level pretraining).",
        describe_architecture(spec.architecture, description_dir=description_dir),
        describe_objective(spec.objective, description_dir=description_dir),
        f"Adaptation: {ADAPTATION_RULE}.",
    ])


def _hash_embedding(text: str, dim: int) -> torch.Tensor:
    """Signed feature hashing of lowercase tokens, L2-normalized."""
    vec = torch.zeros(dim, dtype=torch.float32)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        h = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")
        vec[h % dim] += 1.0 if (h >> 63) == 0 else -1.0
    norm = vec.norm()
    return vec / norm if norm > 0 else vec


class TextEncoder:
    """Frozen description embeddings: ``bert`` (mean-pooled last hidden states) or ``hash``.

    BERT embeddings are cached on disk per (model, max_length, text) digest.
    """

    def __init__(self, cfg, batch_size: int = 16) -> None:
        graph_cfg = cfg.moe.routergfm.graph
        self.backend = str(graph_cfg.text_backend).lower()
        if self.backend not in {"bert", "hash"}:
            raise ValueError(f"Unknown text backend {self.backend!r} (bert | hash)")
        self.model_name = str(graph_cfg.text_model)
        self.max_length = int(graph_cfg.text_max_length)
        self.cache_dir = Path(str(graph_cfg.text_cache_dir))
        self.local_files_only = bool(graph_cfg.local_files_only)
        self.hash_dim = int(graph_cfg.hash_dim)
        self.batch_size = int(batch_size)
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None
        self._memo: Dict[str, torch.Tensor] = {}

    @property
    def dim(self) -> int:
        if self.backend == "hash":
            return self.hash_dim
        self._ensure_model()
        return int(self._model.config.hidden_size)

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Embed *texts* -> float32 CPU tensor ``[n, dim]``."""
        texts = [str(t) for t in texts]
        if not texts:
            return torch.zeros((0, self.dim), dtype=torch.float32)
        todo = []
        for text in dict.fromkeys(texts):
            if text in self._memo:
                continue
            if self.backend == "hash":
                self._memo[text] = _hash_embedding(text, self.hash_dim)
                continue
            path = self._cache_path(text)
            if path.is_file():
                self._memo[text] = torch.load(path, map_location="cpu")["embedding"]
            else:
                todo.append(text)
        if todo:
            for text, emb in zip(todo, self._bert_embed(todo)):
                emb = emb.clone()  # own storage: saving a row view would write the whole batch
                save_torch_atomic(str(self._cache_path(text)), {"embedding": emb})
                self._memo[text] = emb
        return torch.stack([self._memo[t] for t in texts]).float()

    def _cache_path(self, text: str) -> Path:
        identity = json.dumps(
            {"model": self.model_name, "max_length": self.max_length, "text": text}, sort_keys=True
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        safe_model = re.sub(r"[^a-z0-9_.-]", "_", self.model_name.lower())
        return self.cache_dir / safe_model / f"{digest}.pt"

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer  # heavy; only the bert backend needs it

        kwargs = {"local_files_only": self.local_files_only}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        self._model = AutoModel.from_pretrained(self.model_name, **kwargs).to(self.device).eval()

    @torch.no_grad()
    def _bert_embed(self, texts: Sequence[str]) -> torch.Tensor:
        self._ensure_model()
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            tokens = self._tokenizer(
                list(texts[start:start + self.batch_size]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            hidden = self._model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            outputs.append(pooled.float().cpu())
        return torch.cat(outputs, dim=0)


__all__ = [
    "ADAPTATION_RULE",
    "DOMAINS",
    "TextEncoder",
    "dataset_domain",
    "describe_application",
    "describe_architecture",
    "describe_corpus",
    "describe_expert",
    "describe_objective",
]
