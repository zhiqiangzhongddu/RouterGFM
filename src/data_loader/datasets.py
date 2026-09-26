"""Dataset loading and processing utilities for node-level, edge-level, and graph-level datasets with support for induced subgraph generation and SVD-based feature reduction."""

import contextlib
import hashlib
import json
from pathlib import Path
import warnings
from typing import Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import (
    Actor,
    Airports,
    Amazon,
    CitationFull,
    Coauthor,
    CoraFull,
    EllipticBitcoinDataset,
    EmailEUCore,
    GNNBenchmarkDataset,
    Flickr,
    HeterophilousGraphDataset,
    LINKXDataset,
    MoleculeNet,
    Planetoid,
    QM7b,
    QM9,
    Reddit,
    Reddit2,
    TUDataset,
    WebKB,
    WikiCS,
    WikipediaNetwork,
    ZINC,
)
from torch_geometric.transforms import Compose

# OGB pulls in `outdated`, which still imports `pkg_resources.parse_version`.
# Keep that one third-party deprecation warning out of experiment logs.
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"outdated(\..*)?",
)
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset

from .dataset_metadata import (
    dataset_info,
    get_basic_dataset_info,
    is_regression_dataset,
    log_split_instance_counts,
    split_instance_counts,
)
from .dataset_paths import _dataset_scoped_dir, _scoped_root
from .dataset_splits import (
    EDGE_SPLIT_FORMAT_VERSION,
    _canonical_split_dataset_name,
    _edge_split_file_path,
    _get_or_create_edge_split_payload,
    _get_or_create_few_shot_split,
    _get_or_create_split_indices,
    _get_or_create_split_indices_subset,
    _is_few_shot_split_def,
    _mask_to_node_indices,
    _split_suffix,
    _validate_edge_split_def,
    _validate_split_def,
)
from .dataset_storage import _get_dataset_data_storage, _unwrap_subset_dataset
from .filter_empty_graph import _graph_filter_cache_meta_for_dataset, _sanitize_graph_dataset
from .induced_graphs import (
    EDGE_INDUCED_GRAPH_SCHEMA,
    InducedGraphDataset,
    SingleGraphDataLoader,
    _acquire_induced_cache_build_lock,
    _induced_cache_path,
    _load_induced_cache,
    _release_induced_cache_build_lock,
    _save_induced_cache,
    build_edge_induced_graphs_supervised,
    build_induced_graphs,
)
from .svd_features import (
    EnsureFeatureTransform,
    SafeSVDFeatureReduction,
    _apply_feature_svd,
    compute_subgraph_svd_features,
)


__all__ = [
    "SingleGraphDataLoader",
    "compute_subgraph_svd_features",
    "create_dataset",
    "dataset_info",
    "get_basic_dataset_info",
    "infer_task_level",
    "is_regression_dataset",
    "log_split_instance_counts",
    "split_instance_counts",
]


# Keep logs clean when third-party internals still touch InMemoryDataset.data.
warnings.filterwarnings(
    "ignore",
    message="It is not recommended to directly access the internal storage format `data` of an 'InMemoryDataset'.*",
    category=UserWarning,
    module="torch_geometric.data.in_memory_dataset",
)
# Sparse CSR tensor support is still in beta, and some datasets trigger related warnings when loading.
warnings.filterwarnings(
    "ignore",
    message="Sparse CSR tensor support is in beta state.*",
    category=UserWarning,
)


# Note: we don't touch any dynamic, 3D, relational or heterogeneous datasets in this project.


def _induced_feature_identity(
    *,
    base_dataset,
    requested_root: str,
    feat_reduction: bool,
    feat_reduction_dim: int,
    persist_feature_svd: bool,
    feature_svd_dir: str,
) -> dict:
    """Describe the feature tensor embedded in an induced-graph cache.

    Node/edge induced caches store ``x`` directly.  Reusing the same cache for
    raw features, a different SVD width, or a different feature-cache root is
    therefore incorrect even when all topology settings are unchanged.
    """
    dataset_root = str(Path(getattr(base_dataset, "root", requested_root)).resolve())
    if not feat_reduction:
        source = f"raw:{dataset_root}"
    elif persist_feature_svd:
        source_root = str(Path(feature_svd_dir or dataset_root).resolve())
        source = f"persisted_svd:{source_root}"
    else:
        source = f"runtime_svd:{dataset_root}"
    return {
        "feat_reduction": bool(feat_reduction),
        "feat_reduction_dim": int(feat_reduction_dim) if feat_reduction else None,
        "feature_source": source,
    }


def _induced_feature_tag(identity: dict) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:10]

# -------------------------------------------------------------------------- #
# Node-level datasets
# -------------------------------------------------------------------------- #
Actor_NAMES = {
    "actor": "actor",
}
Airports_NAMES = {
    "airports": "USA",
}
Amazon_NAMES = {
    "computers": "Computers",
    "photo": "Photo",
}
CitationFull_NAMES = {
    "cora-ml": "cora_ml",
    "dblp": "dblp",
}
Coauthor_NAMES = {
    "cs": "CS",
    "physics": "Physics",
}
CoraFull_NAMES = {
    "corafull": "corafull",
}
EllipticBitcoinDataset_NAMES = {
    "elliptic-bitcoin": "elliptic_bitcoin",
}
EmailEUCore_NAMES = {
    "email": "email_eu_core",
}
Flickr_NAMES = {
    "flickr": "flickr",
}
HeterophilousGraphDataset_NAMES = {
    "amazon-ratings": "amazon_ratings",
    "minesweeper": "minesweeper",
    "questions": "questions",
    "roman-empire": "roman_empire",
    "tolokers": "tolokers",
}
LINKXDataset_NAMES = {
    "amherst41": "amherst41",
    "cornell5": "cornell5",
    "genius": "genius",
    "johnshopkins55": "johnshopkins55",
    "penn94": "penn94",
    "reed98": "reed98",
}
OGBG_NAMES = {
    "ogbg-molhiv": "ogbg-molhiv",
    "ogbg-molpcba": "ogbg-molpcba",
}
OGBN_NAMES = {
    "ogbn-arxiv": "ogbn-arxiv",
    "ogbn-mag": "ogbn-mag",
    "ogbn-papers100m": "ogbn-papers100M",
    "ogbn-products": "ogbn-products",
    "ogbn-proteins": "ogbn-proteins",
}
Planetoid_NAMES = {
    "citeseer": "CiteSeer",
    "cora": "Cora",
    "pubmed": "PubMed",
}
Reddit_NAMES = {
    "reddit": "reddit",
}
Reddit2_NAMES = {
    "reddit2": "reddit2",
}
WebKB_NAMES = {
    "cornell": "cornell",
    "texas": "texas",
    "wisconsin": "wisconsin",
}
WikiCS_NAMES = {
    "wikics": "wikics",
}
WikipediaNetwork_NAMES = {
    "chameleon": "chameleon",
    "squirrel": "squirrel",
}

# -------------------------------------------------------------------------- #
# Graph-level datasets
# -------------------------------------------------------------------------- #
MoleculeNet_NAMES = {
    "bace": "bace",
    "bbbp": "bbbp",
    "clintox": "clintox",
    "esol": "esol",
    "freesolv": "freesolv",
    "hiv": "hiv",
    "lipo": "lipo",
    "muv": "muv",
    "pcba": "pcba",
    "sider": "sider",
    "tox21": "tox21",
    "toxcast": "toxcast",
}
QM7b_NAMES = {
    "qm7b": "qm7b",
}
QM9_NAMES = {
    "qm9": "qm9",
}
TUDataset_NAMES = {
    "collab": "COLLAB",
    "enzymes": "ENZYMES",
    "imdb-binary": "IMDB-BINARY",
    "imdb-multi": "IMDB-MULTI",
    "mutag": "MUTAG",
    "proteins": "PROTEINS",
    "nci1": "NCI1",
    "nci109": "NCI109",
    "dd": "DD",
    "reddit-binary": "REDDIT-BINARY",
    "reddit-multi-5k": "REDDIT-MULTI-5K",
}
GNNBenchmarkDataset_NAMES = {
    "mnist": "MNIST",
    "cifar10": "CIFAR10",
}
ZINC_NAMES = {
    "zinc": "zinc",
}


def _is_node_dataset_key(key: str) -> bool:
    return (
        key.startswith("ogbn-")
        or key in Actor_NAMES
        or key in Airports_NAMES
        or key in Amazon_NAMES
        or key in CitationFull_NAMES
        or key in Coauthor_NAMES
        or key in CoraFull_NAMES
        or key in EllipticBitcoinDataset_NAMES
        or key in EmailEUCore_NAMES
        or key in Flickr_NAMES
        or key in HeterophilousGraphDataset_NAMES
        or key in LINKXDataset_NAMES
        or key in Planetoid_NAMES
        or key in Reddit_NAMES
        or key in Reddit2_NAMES
        or key in WebKB_NAMES
        or key in WikiCS_NAMES
        or key in WikipediaNetwork_NAMES
    )


def _is_graph_dataset_key(key: str) -> bool:
    return (
        key.startswith("ogbg-")
        or key in MoleculeNet_NAMES
        or key in GNNBenchmarkDataset_NAMES
        or key in ZINC_NAMES
        or key in QM7b_NAMES
        or key in QM9_NAMES
        or key in TUDataset_NAMES
    )


def _is_edge_dataset_key(key: str) -> bool:
    return key.startswith("ogbl-")


@contextlib.contextmanager
def _force_ogb_prompts_yes():
    """Force OGB download/version prompts to auto-yes."""
    import builtins

    try:
        import ogb.utils.url as ogb_url
    except Exception:
        ogb_url = None

    orig_input = builtins.input
    orig_decide = getattr(ogb_url, "decide_download", None) if ogb_url else None
    builtins.input = lambda *args, **kwargs: "y"
    if ogb_url and orig_decide:
        ogb_url.decide_download = lambda url: True
    try:
        yield
    finally:
        builtins.input = orig_input
        if ogb_url and orig_decide:
            ogb_url.decide_download = orig_decide


def infer_task_level(name: str) -> str | None:
    key = name.lower()
    in_node = _is_node_dataset_key(key)
    in_graph = _is_graph_dataset_key(key)
    in_edge = _is_edge_dataset_key(key)
    if in_edge and not in_node and not in_graph:
        return "edge"
    if in_node and not in_graph:
        return "node"
    if in_graph and not in_node and not in_edge:
        return "graph"
    return None


def _load_node_dataset(
    name: str,
    root: str,
    transform,
):
    """Load node-level dataset by name."""
    key = name.lower()
    if key in Actor_NAMES:
        return Actor(root=_scoped_root(root, key), transform=transform)
    elif key in Airports_NAMES:
        dataset_key = Airports_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return Airports(root=root_key, name=dataset_key, transform=transform)
    elif key in Amazon_NAMES:
        dataset_key = Amazon_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return Amazon(root=root_key, name=dataset_key, transform=transform)
    elif key in CitationFull_NAMES:
        dataset_key = CitationFull_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return CitationFull(root=root_key, name=dataset_key, transform=transform)
    elif key in Coauthor_NAMES:
        dataset_key = Coauthor_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return Coauthor(root=root_key, name=dataset_key, transform=transform)
    elif key in CoraFull_NAMES:
        root_key = _scoped_root(root, key)
        return CoraFull(root=root_key, transform=transform)
    elif key in EllipticBitcoinDataset_NAMES:
        dataset_key = EllipticBitcoinDataset_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return EllipticBitcoinDataset(root=root_key, transform=transform)
    elif key in EmailEUCore_NAMES:
        dataset_key = EmailEUCore_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return EmailEUCore(root=root_key, transform=transform)
    elif key in Flickr_NAMES:
        dataset_key = Flickr_NAMES[key]
        root_key = _scoped_root(root, key)
        return Flickr(root=root_key, transform=transform)
    elif key in HeterophilousGraphDataset_NAMES:
        dataset_key = HeterophilousGraphDataset_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return HeterophilousGraphDataset(root=root_key, name=dataset_key, transform=transform)
    elif key in LINKXDataset_NAMES:
        dataset_key = LINKXDataset_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return LINKXDataset(root=root_key, name=dataset_key, transform=transform)
    elif key in Planetoid_NAMES:
        dataset_key = Planetoid_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return Planetoid(root=root_key, name=dataset_key, transform=transform)
    elif key in Reddit_NAMES:
        root_key = _scoped_root(root, key)
        return Reddit(root=root_key, transform=transform)
    elif key in Reddit2_NAMES:
        root_key = _scoped_root(root, key)
        return Reddit2(root=root_key, transform=transform)
    elif key.startswith("ogbn-"):
        dataset_key = OGBN_NAMES[key]
        root_key = _scoped_root(root, key)
        with _force_ogb_prompts_yes():
            return PygNodePropPredDataset(name=dataset_key, root=root_key)
    elif key in WebKB_NAMES:
        dataset_key = WebKB_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return WebKB(root=root_key, name=dataset_key, transform=transform)
    elif key in WikiCS_NAMES:
        root_key = _scoped_root(root, key)
        return WikiCS(root=root_key, is_undirected=True, transform=transform)
    elif key in WikipediaNetwork_NAMES:
        dataset_key = WikipediaNetwork_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return WikipediaNetwork(root=root_key, name=dataset_key, transform=transform)
    else:
        raise ValueError(f"Unsupported node-level dataset: {name}")


def _load_graph_dataset(
    name: str,
    root: str,
    transform,
):
    """Load graph-level dataset by name."""
    key = name.lower()
    if key in MoleculeNet_NAMES:
        dataset_key = MoleculeNet_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return MoleculeNet(root=root_key, name=dataset_key, transform=transform)
    elif key in GNNBenchmarkDataset_NAMES:
        dataset_key = GNNBenchmarkDataset_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return GNNBenchmarkDataset(root=root_key, name=dataset_key, transform=transform)
    elif key in QM7b_NAMES:
        dataset_key = QM7b_NAMES[key]
        root_key = _scoped_root(root, key)
        return QM7b(root=root_key, transform=transform)
    elif key in QM9_NAMES:
        root_key = _scoped_root(root, key)
        return QM9(root=root_key, transform=transform)
    elif key in ZINC_NAMES:
        root_key = _scoped_root(root, key)
        return ZINC(root=root_key, subset=False, split="train", transform=transform)
    elif key in TUDataset_NAMES:
        dataset_key = TUDataset_NAMES[key]
        root_key = _scoped_root(root, key) if key != dataset_key else root
        return TUDataset(root=root_key, name=dataset_key, transform=transform)
    elif key.startswith("ogbg-"):
        dataset_key = OGBG_NAMES[key]
        root_key = _scoped_root(root, key)
        with _force_ogb_prompts_yes():
            return PygGraphPropPredDataset(name=dataset_key, root=root_key)
    else:
        raise ValueError(f"Unsupported graph-level dataset: {name}")


def create_dataset(
    name: str,
    root: str,
    task_level: str,
    feat_reduction: bool = True,
    feat_reduction_dim: int = 100,
    persist_feature_svd: bool = True,
    feature_svd_dir: str = "",
    induced: bool = False,
    induced_min_size: int = 10,
    induced_max_size: int = 30,
    induced_max_hops: int = 5,
    edge_max_size: int | None = 60,
    cache_induced: bool = True,
    require_induced_cache_hit: bool = False,
    split: Tuple[float, float, float] | None = None,
    seed: int = 42,
    split_root: str = "",
    induced_root: str = "",
    graph_filter_dir: str = "",
    pad_featureless_features: bool = False,
):
    """Create dataset based on name and task level with optional feature reduction.

    ``pad_featureless_features`` controls SVD padding for graph datasets with
    no native node features (e.g. qm7b, which synthesizes 1-dim degree
    features at access time). The default ``False`` preserves the protocol
    every published pretrain/finetune artifact was built on: such datasets
    keep their synthesized features untouched. Cross-dataset expert
    pipelines that require uniform 100-dim query features opt in
    explicitly at their call site.
    """
    transforms = [EnsureFeatureTransform()]
    reducer = None
    if feat_reduction:
        if persist_feature_svd:
            reducer = SafeSVDFeatureReduction(out_channels=feat_reduction_dim)
        else:
            transforms.append(SafeSVDFeatureReduction(out_channels=feat_reduction_dim))
    transform = Compose(transforms) if transforms else None

    if task_level in ("node", "edge"):
        if induced:
            base_dataset = _load_node_dataset(name=name, root=root, transform=transform)
            scoped_feature_svd_dir = (
                str(_dataset_scoped_dir(feature_svd_dir, name))
                if feature_svd_dir
                else ""
            )
            if reducer and hasattr(base_dataset, "data") and persist_feature_svd:
                _apply_feature_svd(
                    base_dataset,
                    name,
                    feat_reduction_dim,
                    reducer,
                    task_level=task_level,
                    output_root=scoped_feature_svd_dir,
                )
            base_data = base_dataset[0]
            base_name = name
            feature_identity = _induced_feature_identity(
                base_dataset=base_dataset,
                requested_root=root,
                feat_reduction=feat_reduction,
                feat_reduction_dim=feat_reduction_dim,
                persist_feature_svd=persist_feature_svd,
                feature_svd_dir=scoped_feature_svd_dir,
            )
            feature_tag = _induced_feature_tag(feature_identity)
            cache_root_path = Path(induced_root) if induced_root else (Path(split_root) if split_root else None)
            if require_induced_cache_hit and (
                not cache_induced or cache_root_path is None
            ):
                raise ValueError(
                    "require_induced_cache_hit needs cache_induced=True and "
                    "a non-empty induced_root or split_root"
                )
            cache_build_lock = None
            try:
                if task_level == "edge":
                    if not split_root:
                        raise ValueError("split_root is required for induced edge datasets.")
                    if split is None:
                        raise ValueError("split must be provided for induced edge datasets.")
                    split_root_path = Path(split_root)
                    split_def = split
                    _validate_edge_split_def(split_def)
                    edge_size_tag = "none" if edge_max_size is None else str(int(edge_max_size))
                    cache_suffix = (
                        f"h{induced_max_hops}_m{edge_size_tag}_s{_split_suffix(split_def)}_"
                        f"seed{seed}_f{feature_tag}"
                    )
                    cache_meta = {
                        "task_level": "edge",
                        "edge_induced_graph_schema": EDGE_INDUCED_GRAPH_SCHEMA,
                        "max_hops": induced_max_hops,
                        "max_size": edge_max_size,
                        "split": tuple(float(v) for v in split_def),
                        "seed": int(seed),
                        # v2: pair-level splits + target-edge removal in each
                        # induced subgraph; v1 caches must not be reused.
                        "edge_split_format": EDGE_SPLIT_FORMAT_VERSION,
                        **feature_identity,
                    }
                    split_name = _canonical_split_dataset_name(base_name, "edge", int(seed))
                    # Load an existing saved edge split when present (the
                    # helper checks disk first); otherwise build it and persist
                    # under data/splits/ so train/finetune reuse the same split
                    # as data prep instead of regenerating it in-memory.
                    split_payload = _get_or_create_edge_split_payload(
                        dataset_name=split_name,
                        split=split_def,
                        seed=int(seed),
                        split_root_path=split_root_path,
                        data=base_data,
                        persist=True,
                        verbose=False,
                    )
                    split_path = _edge_split_file_path(
                        split_name,
                        split_def,
                        split_root_path,
                    ).resolve()
                    split_digest = hashlib.sha256()
                    with split_path.open("rb") as split_handle:
                        for chunk in iter(lambda: split_handle.read(1024 * 1024), b""):
                            split_digest.update(chunk)
                    cache_meta["edge_split_sha256"] = split_digest.hexdigest()
                    cache_path = _induced_cache_path(base_name, "edge", cache_root_path, cache_suffix) if cache_root_path else None
                    payload = _load_induced_cache(cache_path, cache_meta) if cache_induced and cache_path else None
                    if payload is None and cache_induced and cache_path:
                        if require_induced_cache_hit:
                            raise RuntimeError(
                                "Required induced edge cache miss: "
                                f"split={_split_suffix(split_def)} seed={int(seed)} path={cache_path}"
                            )
                        cache_build_lock = _acquire_induced_cache_build_lock(cache_path)
                        payload = _load_induced_cache(cache_path, cache_meta)
                    if payload:
                        if cache_path:
                            print(f"[Induced] Loaded cached induced edge graphs from {cache_path}")
                        graphs = payload["graphs"]
                        split_tags = payload.get("split_tags")
                        result = InducedGraphDataset(
                            graphs,
                            base_info=get_basic_dataset_info(base_dataset),
                            base_num_nodes=getattr(base_data, "num_nodes", None),
                            base_num_edges=getattr(base_data, "num_edges", None),
                            split_tags=split_tags,
                        )
                        result.edge_split = tuple(float(v) for v in split_def)
                        result.edge_seed = int(seed)
                        result.edge_context = "train+message"
                        return result
                    if cache_induced and cache_path:
                        print(
                            "[Induced] Cache miss for edge induced graphs "
                            f"(split={_split_suffix(split_def)}, seed={int(seed)}), generating: {cache_path}"
                        )
                    print(f"[Induced] Processing edge induced subgraphs for {base_name}...")

                    edge_device = base_data.edge_index.device

                    def _edge_pairs_from_idx(indices) -> torch.Tensor:
                        idx = torch.as_tensor(indices, dtype=torch.long, device=edge_device)
                        if idx.numel() == 0:
                            return torch.empty((2, 0), dtype=torch.long, device=edge_device)
                        return base_data.edge_index[:, idx]

                    def _neg_pairs_from_payload(key: str) -> torch.Tensor:
                        neg_pairs = split_payload.get(key)
                        if neg_pairs is None:
                            return torch.empty((2, 0), dtype=torch.long, device=edge_device)
                        neg_pairs = torch.as_tensor(neg_pairs, dtype=torch.long, device=edge_device)
                        if neg_pairs.numel() == 0:
                            return torch.empty((2, 0), dtype=torch.long, device=edge_device)
                        if neg_pairs.dim() != 2 or neg_pairs.size(0) != 2:
                            raise ValueError(f"Invalid negative edge tensor for key={key}.")
                        return neg_pairs

                    context_pairs = _edge_pairs_from_idx(split_payload["context_pos_idx"])
                    context_data = Data(
                        x=getattr(base_data, "x", None),
                        edge_index=context_pairs,
                        num_nodes=base_data.num_nodes,
                    )

                    all_graphs = []
                    split_tags = []
                    for split_name, pos_key, neg_key in (
                        ("train", "train_pos_idx", "train_neg_edge_index"),
                        ("val", "val_pos_idx", "val_neg_edge_index"),
                        ("test", "test_pos_idx", "test_neg_edge_index"),
                    ):
                        pos_pairs = _edge_pairs_from_idx(split_payload[pos_key])
                        neg_pairs = _neg_pairs_from_payload(neg_key)
                        graphs_for_split = build_edge_induced_graphs_supervised(
                            data=context_data,
                            pos_edge_pairs=pos_pairs,
                            neg_edge_pairs=neg_pairs,
                            max_hops=induced_max_hops,
                            max_size=edge_max_size,
                        )
                        all_graphs.extend(graphs_for_split)
                        split_tags.extend([split_name] * len(graphs_for_split))

                    if cache_induced and cache_path:
                        _save_induced_cache(
                            cache_path,
                            {
                                "graphs": all_graphs,
                                "split_tags": split_tags,
                                "base_num_nodes": getattr(base_data, "num_nodes", None),
                                "base_num_edges": getattr(base_data, "num_edges", None),
                                "meta": cache_meta,
                            },
                        )
                        print(f"[Induced] Saved induced edge graphs to {cache_path}")
                    result = InducedGraphDataset(
                        all_graphs,
                        base_info=get_basic_dataset_info(base_dataset),
                        base_num_nodes=getattr(base_data, "num_nodes", None),
                        base_num_edges=getattr(base_data, "num_edges", None),
                        split_tags=split_tags,
                    )
                    result.edge_split = tuple(float(v) for v in split_def)
                    result.edge_seed = int(seed)
                    result.edge_context = "train+message"
                    return result
                else:
                    graphs = None
                    split_tags = None
                    split_lookup = None
                    if split_root:
                        split_root_path = Path(split_root)
                        labels = getattr(base_data, "y", None)
                        labeled_idx = None
                        if labels is not None:
                            labels = labels.view(-1)
                            labeled_idx = torch.nonzero(labels >= 0, as_tuple=False).view(-1).tolist()

                        if split is not None:
                            split_def = split
                            _validate_split_def(split_def)
                            split_name = _canonical_split_dataset_name(base_name, "node", int(seed))
                            use_few_shot = _is_few_shot_split_def(split_def)
                            if use_few_shot:
                                train_idx, val_idx, test_idx = _get_or_create_few_shot_split(
                                    dataset_name=split_name,
                                    labels=labels,
                                    shots_per_class=int(split_def[0]),
                                    val_ratio=float(split_def[1]),
                                    test_ratio=float(split_def[2]),
                                    seed=seed,
                                    split_root_path=split_root_path,
                                )
                            elif labeled_idx is not None and len(labeled_idx) < base_data.num_nodes:
                                train_idx, val_idx, test_idx = _get_or_create_split_indices_subset(
                                    dataset_name=split_name,
                                    split=split_def,
                                    seed=seed,
                                    split_root_path=split_root_path,
                                    subset_indices=labeled_idx,
                                )
                            else:
                                train_idx, val_idx, test_idx = _get_or_create_split_indices(
                                    dataset_name=split_name,
                                    split=split_def,
                                    seed=seed,
                                    split_root_path=split_root_path,
                                    total=base_data.num_nodes,
                                )
                        else:
                            train_mask = getattr(base_data, "train_mask", None)
                            val_mask = getattr(base_data, "val_mask", None)
                            test_mask = getattr(base_data, "test_mask", None)
                            if train_mask is None or val_mask is None or test_mask is None:
                                raise ValueError(
                                    "split must be provided for induced node datasets when masks are unavailable."
                                )
                            train_idx = _mask_to_node_indices(train_mask, "train_mask")
                            val_idx = _mask_to_node_indices(val_mask, "val_mask")
                            test_idx = _mask_to_node_indices(test_mask, "test_mask")
                        split_lookup = {idx: "train" for idx in train_idx}
                        split_lookup.update({idx: "val" for idx in val_idx})
                        split_lookup.update({idx: "test" for idx in test_idx})
                    if cache_induced and cache_root_path:
                        cache_suffix = (
                            f"h{induced_max_hops}_s{induced_min_size}-{induced_max_size}_f{feature_tag}"
                        )
                        cache_meta = {
                            "task_level": "node",
                            "max_hops": induced_max_hops,
                            "min_size": induced_min_size,
                            "max_size": induced_max_size,
                            **feature_identity,
                        }
                        cache_path = _induced_cache_path(base_name, "node", cache_root_path, cache_suffix)
                        payload = _load_induced_cache(cache_path, cache_meta)
                        if payload is None:
                            if require_induced_cache_hit:
                                raise RuntimeError(
                                    f"Required induced node cache miss: path={cache_path}"
                                )
                            cache_build_lock = _acquire_induced_cache_build_lock(cache_path)
                            payload = _load_induced_cache(cache_path, cache_meta)
                        if payload:
                            if cache_path:
                                print(f"[Induced] Loaded cached induced node graphs from {cache_path}")
                            graphs = payload["graphs"]
                            split_tags = payload.get("split_tags")
                            if graphs is not None and split_lookup is not None:
                                split_tags = [
                                    split_lookup.get(getattr(graph, "base_node_id", idx), "train")
                                    for idx, graph in enumerate(graphs)
                                ]
                    if graphs is None:
                        print(f"[Induced] Processing node induced subgraphs for {base_name}...")
                        graphs = build_induced_graphs(
                            data=base_data,
                            smallest_size=induced_min_size,
                            largest_size=induced_max_size,
                            max_hops=induced_max_hops,
                        )
                        if split_lookup is not None:
                            split_tags = [split_lookup.get(graph.base_node_id, "train") for graph in graphs]
                        if cache_induced and cache_root_path:
                            _save_induced_cache(
                                cache_path,
                                {
                                    "graphs": graphs,
                                    "split_tags": split_tags,
                                    "base_num_nodes": getattr(base_data, "num_nodes", None),
                                    "base_num_edges": getattr(base_data, "num_edges", None),
                                    "meta": cache_meta,
                                },
                            )
                            print(f"[Induced] Saved induced node graphs to {cache_path}")
            except Exception as exc:
                raise RuntimeError(f"Induced graph generation failed for {name} ({task_level}): {exc}") from exc
            finally:
                _release_induced_cache_build_lock(cache_build_lock)
            if graphs:
                return InducedGraphDataset(
                    graphs,
                    base_info=get_basic_dataset_info(base_dataset),
                    base_num_nodes=getattr(base_data, "num_nodes", None),
                    base_num_edges=getattr(base_data, "num_edges", None),
                    split_tags=split_tags,
                )
            raise RuntimeError(f"[Induced] No induced graphs generated for {name} ({task_level}).")

        dataset = _load_node_dataset(name=name, root=root, transform=transform)
        if reducer and _get_dataset_data_storage(dataset) is not None and persist_feature_svd:
            _apply_feature_svd(
                dataset,
                name,
                feat_reduction_dim,
                reducer,
                task_level=task_level,
                output_root=feature_svd_dir,
            )
            try:
                dataset._svd_dim = feat_reduction_dim  # type: ignore[attr-defined]
                dataset._svd_task_level = task_level  # type: ignore[attr-defined]
                dataset._feature_svd_root = feature_svd_dir or dataset.root  # type: ignore[attr-defined]
            except Exception:
                pass
        return dataset

    elif task_level == "graph":
        dataset = _load_graph_dataset(name=name, root=root, transform=transform)
        dataset = _sanitize_graph_dataset(dataset=dataset, dataset_name=name, graph_filter_dir=graph_filter_dir)
        if reducer:
            feature_target = _unwrap_subset_dataset(dataset)
            feature_store = _get_dataset_data_storage(feature_target)
            # Collated-storage SVD only reconstructs per graph when the raw
            # dataset has an x slice. Datasets such as QM7b synthesize degree
            # features at __getitem__ time, so route them through the transform
            # path or the persisted 100-D tensor cannot be separated per graph.
            has_native_x = (
                feature_store is not None
                and getattr(feature_store, "x", None) is not None
            )
            if not has_native_x and not pad_featureless_features:
                # Pre-e8e09000 protocol: featureless graph datasets keep the
                # synthesized per-node features from EnsureFeatureTransform
                # (1-dim degree for qm7b). Every published pretrained encoder
                # for these datasets was built on that feature space, so
                # silently padding to feat_reduction_dim here makes all of
                # them unloadable. Padding is opt-in via
                # pad_featureless_features.
                pass
            elif has_native_x and persist_feature_svd:
                _apply_feature_svd(
                    feature_target,
                    name,
                    feat_reduction_dim,
                    reducer,
                    task_level=task_level,
                    output_root=feature_svd_dir,
                )
                try:
                    dataset._svd_dim = feat_reduction_dim  # type: ignore[attr-defined]
                    dataset._svd_task_level = task_level  # type: ignore[attr-defined]
                    dataset._feature_svd_root = feature_svd_dir or getattr(feature_target, "root", root)  # type: ignore[attr-defined]
                except Exception:
                    pass
            else:
                pending_datasets = [_unwrap_subset_dataset(dataset)]
                while pending_datasets:
                    current_dataset = pending_datasets.pop()
                    children = getattr(current_dataset, "datasets", None)
                    if children is not None:
                        pending_datasets.extend(children)
                        continue
                    current_transform = getattr(current_dataset, "transform", None)
                    try:
                        current_dataset.transform = reducer if current_transform is None else Compose([current_transform, reducer])
                    except Exception:
                        pass
                try:
                    dataset.num_features = int(feat_reduction_dim)  # type: ignore[attr-defined]
                    dataset.num_node_features = int(feat_reduction_dim)  # type: ignore[attr-defined]
                except Exception:
                    pass
        filter_meta = _graph_filter_cache_meta_for_dataset(dataset)
        if filter_meta is None:
            raise RuntimeError(f"[GraphFilter] Missing filter metadata for graph dataset={name}.")
        return dataset

    else:
        raise ValueError(f"Unsupported task_level: {task_level}")
