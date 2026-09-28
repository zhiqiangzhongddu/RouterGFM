"""Table 15 shift protocol: label-free regions, shift split files, shared shift evaluation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Subset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree, to_undirected

from src.config import cfg as base_cfg
from src.data_loader import shift_splits as ss
from src.data_loader.dataset_loader import make_loaders
from src.data_loader.dataset_splits import _get_or_create_few_shot_split
from src.data_loader.induced_graphs import InducedGraphDataset, build_induced_graphs
from src.moe import shift_eval
from src.moe.routergfm.common import GRAPH_CLS, LINK, MULTILABEL, NODE_CLS, REGRESSION
from src.moe.routergfm.losses import RegressionNormalizer, routing_loss

NUM_CLASSES = 3


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def _base_graph(n: int = 200, seed: int = 0, with_x: bool = True) -> Data:
    g = torch.Generator().manual_seed(seed)
    prob = 0.01 + 0.06 * torch.rand(n, generator=g)  # heterogeneous degrees
    upper = torch.triu(torch.rand(n, n, generator=g) < (prob[:, None] + prob[None, :]) / 2, diagonal=1)
    edge_index = (upper | upper.t()).nonzero().t().contiguous()
    scale = 0.2 + 3.0 * torch.rand(n, 1, generator=g)
    x = scale * torch.randn(n, 8, generator=g) if with_x else None
    return Data(x=x, edge_index=edge_index, y=torch.randint(0, NUM_CLASSES, (n,), generator=g), num_nodes=n)


def _node_dataset(base: Data) -> InducedGraphDataset:
    torch.manual_seed(0)  # build_induced_graphs subsamples large ego-nets with the global RNG
    graphs = build_induced_graphs(base, smallest_size=4, largest_size=10, max_hops=2)
    return InducedGraphDataset(graphs, base_num_nodes=base.num_nodes, base_num_edges=base.edge_index.size(1))


class GraphList:
    """Plain list-backed graph dataset (no split_tags, no collated storage)."""

    def __init__(self, graphs, num_classes=None):
        self.graphs = list(graphs)
        if num_classes is not None:
            self.num_classes = num_classes

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[int(idx)]


class FeaturelessGraphList(GraphList):
    """Mimics a dataset whose collated storage has no native ``x`` (e.g. qm7b)."""

    def __init__(self, graphs, num_classes=None):
        super().__init__(graphs, num_classes)
        self.data = Data(x=None)


def _graph_dataset(kind: str, n: int = 150, seed: int = 1, cls=GraphList) -> GraphList:
    g = torch.Generator().manual_seed(seed)
    graphs = []
    for _ in range(n):
        size = int(torch.randint(4, 20, (1,), generator=g))
        upper = torch.triu(torch.rand(size, size, generator=g) < 0.3, diagonal=1)
        edge_index = (upper | upper.t()).nonzero().t().contiguous()
        x = (0.2 + 2.0 * torch.rand(1, generator=g)) * torch.randn(size, 8, generator=g)
        if kind == "cls":
            y = torch.randint(0, NUM_CLASSES, (1,), generator=g)
        elif kind == "multilabel":
            y = torch.randint(0, 2, (1, 4), generator=g).float()
            y[torch.rand(1, 4, generator=g) < 0.2] = float("nan")
        else:
            y = torch.randn(1, 2, generator=g) * 3.0 + 1.0
        graphs.append(Data(x=x, edge_index=edge_index, y=y, num_nodes=size))
    return cls(graphs, num_classes=NUM_CLASSES if kind == "cls" else None)


def _cfg(tmp_path: Path, *, min_queries: int = 20):
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.data_preparation.dataset.split_root = str(tmp_path / "splits")
    cfg.data_preparation.shift.root = str(tmp_path / "shift")
    cfg.data_preparation.shift.min_queries = min_queries
    return cfg


@pytest.fixture(scope="module")
def node_setup():
    base = _base_graph()
    dataset = _node_dataset(base)
    source = ss.make_shift_source(dataset, "synth", "node", induced=True, base_graph=base)
    return base, dataset, source


# --------------------------------------------------------------------------- #
# Statistics and regions
# --------------------------------------------------------------------------- #
def test_config_block_defaults():
    sh = base_cfg.data_preparation.shift
    assert (sh.build, sh.verify, sh.root) == (False, False, "data/splits_shift")
    assert list(sh.conditions) == ["feature", "structural", "mixed"]
    assert (sh.source_quantile, sh.target_quantile, sh.min_queries) == (0.6, 0.8, 100)


def test_node_statistics_use_base_degree_and_target_features(node_setup):
    base, dataset, source = node_setup
    deg = degree(to_undirected(base.edge_index, num_nodes=base.num_nodes)[0], num_nodes=base.num_nodes)
    assert np.allclose(source.stats["structural"], deg[torch.as_tensor(source.ids)].numpy())
    assert np.allclose(source.stats["feature"], base.x[torch.as_tensor(source.ids)].norm(dim=-1).numpy(), atol=1e-5)
    # Ego-subgraphs truncate degree: the statistic must not come from them.
    ego_deg = [int((g.edge_index[0] == g.target_node_index[0]).sum()) for g in dataset.graphs]
    assert any(d > e for d, e in zip(deg.tolist(), ego_deg))


def test_rank_normalise_breaks_ties_by_seed():
    stat = np.array([3.0, 1.0, 1.0, 1.0, 2.0, 5.0])
    u = ss.rank_normalise(stat, seed=0)
    assert sorted(u.tolist()) == pytest.approx([(i + 1) / 6 for i in range(6)])
    assert u[1:4].max() < u[4] < u[0] < u[5]  # distinct values keep their order
    orders = {tuple(np.argsort(ss.rank_normalise(stat, seed=s))[:3]) for s in range(10)}
    assert len(orders) > 1  # tie order depends on the seed
    assert np.array_equal(u, ss.rank_normalise(stat, seed=0))
    with pytest.raises(ValueError):
        ss.rank_normalise(np.array([1.0, np.nan]), seed=0)


@pytest.mark.parametrize("condition", ["feature", "structural"])
def test_single_axis_regions(condition):
    n = 500
    rng = np.random.default_rng(0)
    stats = {"feature": rng.normal(size=n), "structural": rng.integers(0, 10, size=n).astype(float)}
    source, target, info = ss.shift_regions(stats, condition, source_q=0.6, target_q=0.8, min_queries=100, seed=3)
    assert not (source & target).any()
    assert source.sum() == int(0.6 * n)
    assert abs(int(target.sum()) - 0.2 * n) <= 1
    values = stats[condition]
    assert values[target].min() >= values[source].max()  # ties may straddle, never invert
    assert info["target_quantile"] == 0.8 and info["target_region_size"] == int(target.sum())


def test_mixed_regions_relax_until_min_queries():
    n = 2000
    rng = np.random.default_rng(1)
    stats = {"feature": rng.normal(size=n), "structural": rng.normal(size=n)}
    seed = 7
    source, target, info = ss.shift_regions(stats, "mixed", source_q=0.6, target_q=0.8, min_queries=100, seed=seed)
    q = info["target_quantile"]
    u_f = ss.rank_normalise(stats["feature"], seed)
    u_s = ss.rank_normalise(stats["structural"], seed + 1)
    assert ((u_f[target] >= q) & (u_s[target] >= q)).all()
    assert ((u_f[source] <= 0.6) & (u_s[source] <= 0.6)).all()
    assert not (source & target).any()
    assert 0.65 <= q < 0.8 and info["min_queries_met"] and target.sum() >= 100
    stricter = round(q + 0.05, 6)
    assert ((u_f >= stricter) & (u_s >= stricter)).sum() < 100  # stopped at the first sufficient q


def test_mixed_relaxation_stops_at_floor():
    n = 400
    feature = np.arange(n, dtype=float)
    stats = {"feature": feature, "structural": -feature}  # never high on both axes
    _, target, info = ss.shift_regions(stats, "mixed", source_q=0.6, target_q=0.8, min_queries=100, seed=0)
    assert info["target_quantile"] == 0.65 and not info["min_queries_met"] and target.sum() == 0


def test_regions_validate_arguments():
    stats = {"feature": np.arange(10.0), "structural": np.arange(10.0)}
    with pytest.raises(ValueError):
        ss.shift_regions(stats, "feature", source_q=0.8, target_q=0.6, min_queries=1, seed=0)
    with pytest.raises(ValueError):
        ss.shift_regions(stats, "label", source_q=0.6, target_q=0.8, min_queries=1, seed=0)


def test_regions_are_label_free(node_setup):
    base, _, source = node_setup
    permuted = base.clone()
    permuted.y = base.y[torch.randperm(base.num_nodes, generator=torch.Generator().manual_seed(5))]
    other = ss.make_shift_source(_node_dataset(permuted), "synth", "node", induced=True, base_graph=permuted)
    assert not torch.equal(other.labels, source.labels)
    for condition in ss.SHIFT_CONDITIONS:
        a = ss.shift_regions(source.stats, condition, source_q=0.6, target_q=0.8, min_queries=10, seed=42)
        b = ss.shift_regions(other.stats, condition, source_q=0.6, target_q=0.8, min_queries=10, seed=42)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# --------------------------------------------------------------------------- #
# Support sampling
# --------------------------------------------------------------------------- #
def test_sample_support_balanced_and_fallback():
    ids = list(range(100, 130))
    labels = torch.tensor([0] * 12 + [1] * 15 + [2] * 3)
    support, info = ss.sample_support(ids, labels, 5, "balanced", seed=42)
    assert set(support) <= set(ids) and len(set(support)) == len(support)
    by_class = {c: sum(labels[ids.index(i)].item() == c for i in support) for c in range(3)}
    assert by_class == {0: 5, 1: 5, 2: 3}
    assert info["support_class_counts"] == by_class and info["short_classes"] == [2]
    assert ss.sample_support(ids, labels, 5, "balanced", seed=42)[0] == support
    assert ss.sample_support(ids, labels, 5, "balanced", seed=0)[0] != support


def test_sample_support_random_count():
    ids = list(range(50))
    support, info = ss.sample_support(ids, None, 7, "random", seed=1)
    assert len(support) == 7 == info["support_size"] and set(support) <= set(ids)
    assert ss.sample_support(ids, None, 7, "random", seed=1)[0] == support
    assert ss.sample_support(ids, None, 7, "random", seed=2)[0] != support
    assert len(ss.sample_support(ids[:3], None, 7, "random", seed=1)[0]) == 3


# --------------------------------------------------------------------------- #
# Shift split files
# --------------------------------------------------------------------------- #
def _check_payload(path: Path, source, condition: str, shots: int, cfg):
    payload = torch.load(path, weights_only=False)
    meta = payload["meta"]
    assert meta["type"] == ss.SHIFT_SPLIT_TYPE and meta["condition"] == condition
    std = torch.load(meta["universe_from"], weights_only=False)
    universe = std["train"] + std["val"] + std["test"]
    train, val, test = payload["train"], payload["val"], payload["test"]
    assert sorted(train + val + test) == sorted(universe) and meta["total"] == len(universe)
    assert meta["sizes"] == {"support": len(train), "query": len(test), "unused": len(val)}
    assert len(test) > 0 and len(train) > 0

    rows = source.rows(universe)
    axes = ("feature", "structural") if condition == "mixed" else (condition,)
    ranks = {a: ss.rank_normalise(source.stats[a][rows], meta["seed"] + (a == "structural")) for a in axes}
    pos = {int(i): k for k, i in enumerate(universe)}
    q = meta["target_quantile"]
    for a in axes:
        assert all(ranks[a][pos[i]] <= cfg.data_preparation.shift.source_quantile for i in train)
        assert all(ranks[a][pos[i]] >= q for i in test)
    if source.strategy == "balanced":
        labels = source.labels[torch.as_tensor(source.rows(train))]
        assert max(torch.bincount(labels).tolist()) <= shots
        assert 0.0 <= meta["label_tv_support_vs_query"] <= 1.0
    else:
        assert len(train) == shots
    return payload


@pytest.mark.parametrize("condition", ss.SHIFT_CONDITIONS)
def test_node_shift_split_round_trip(tmp_path, node_setup, condition):
    base, dataset, source = node_setup
    cfg = _cfg(tmp_path)
    split = (5, 0.0, 1.0)
    path = ss.build_shift_split(cfg, "synth", "node", 42, split, condition, source=source)
    assert path == tmp_path / "shift" / condition / "synth" / "synth_node_seed42_splits-5-0-100.pt"
    assert (tmp_path / "splits" / "synth" / "synth_node_seed42_splits-5-0-100.pt").is_file()
    payload = _check_payload(path, source, condition, 5, cfg)
    assert not os.access(path, os.W_OK) or os.geteuid() == 0  # read-only against silent regeneration
    mtime = path.stat().st_mtime_ns

    shift_root = str(tmp_path / "shift" / condition)
    # Loader path (node ids mapped to induced positions) and create_dataset's split-tag path.
    train_loader, val_loader, test_loader, meta = make_loaders(
        dataset, "synth_node_seed42", "node", 8, 0, split, 42, induced=True, split_root=shift_root,
        return_split_meta=True,
    )
    assert meta["status"] == "loaded"
    assert len(train_loader.dataset) == len(payload["train"]) and len(test_loader.dataset) == len(payload["test"])
    node_ids = {int(dataset.graphs[i].base_node_id) for i in test_loader.dataset.indices}
    assert node_ids == set(payload["test"])
    loaded = _get_or_create_few_shot_split(
        "synth_node_seed42", labels=base.y, shots_per_class=5, val_ratio=0.0, test_ratio=1.0, seed=42,
        split_root_path=Path(shift_root), return_split_meta=True,
    )
    assert loaded[3]["status"] == "loaded" and loaded[2] == payload["test"]
    assert path.stat().st_mtime_ns == mtime

    # Idempotent rebuild keeps the file untouched.
    assert ss.build_shift_split(cfg, "synth", "node", 42, split, condition, source=source) == path
    assert path.stat().st_mtime_ns == mtime


def test_shift_split_is_deterministic_per_seed(tmp_path, node_setup):
    _, _, source = node_setup
    split = (5, 0.0, 1.0)
    a = torch.load(ss.build_shift_split(_cfg(tmp_path / "a"), "synth", "node", 42, split, "mixed", source=source),
                   weights_only=False)
    b = torch.load(ss.build_shift_split(_cfg(tmp_path / "b"), "synth", "node", 42, split, "mixed", source=source),
                   weights_only=False)
    c = torch.load(ss.build_shift_split(_cfg(tmp_path / "a"), "synth", "node", 0, split, "mixed", source=source),
                   weights_only=False)
    assert all(a[k] == b[k] for k in ("train", "val", "test"))
    assert a["train"] != c["train"]


@pytest.mark.parametrize("kind", ["cls", "multilabel", "regression"])
def test_graph_shift_split_round_trip(tmp_path, kind):
    dataset = _graph_dataset(kind)
    source = ss.make_shift_source(dataset, "gsynth", "graph", induced=False)
    assert source.strategy == ("balanced" if kind == "cls" else "random")
    assert source.label_kind == {"cls": "single", "multilabel": "multilabel", "regression": "regression"}[kind]
    assert np.array_equal(source.stats["structural"], [float(g.num_nodes) for g in dataset.graphs])
    cfg = _cfg(tmp_path)
    split = (5, 0.0, 1.0)
    for condition in ss.SHIFT_CONDITIONS:
        path = ss.build_shift_split(cfg, "gsynth", "graph", 0, split, condition, source=source)
        payload = _check_payload(path, source, condition, 5, cfg)
        key = {"cls": "label_tv_support_vs_query", "multilabel": "assay_pos_rate_shift",
               "regression": "label_mean_shift"}[kind]
        assert key in payload["meta"]
        mtime = path.stat().st_mtime_ns
        train_loader, _, test_loader, meta = make_loaders(
            dataset, "gsynth_graph_seed0", "graph", 8, 0, split, 0, split_root=str(tmp_path / "shift" / condition),
            return_split_meta=True,
        )
        assert meta["status"] == "loaded" and path.stat().st_mtime_ns == mtime
        assert list(test_loader.dataset.indices) == payload["test"]
        assert list(train_loader.dataset.indices) == payload["train"]


def test_featureless_dataset_skips_feature_condition(tmp_path):
    dataset = _graph_dataset("cls", cls=FeaturelessGraphList)
    source = ss.make_shift_source(dataset, "fl", "graph", induced=False)
    assert source.featureless
    cfg = _cfg(tmp_path)
    assert ss.build_shift_split(cfg, "fl", "graph", 0, (5, 0.0, 1.0), "feature", source=source) is None
    path = ss.build_shift_split(cfg, "fl", "graph", 0, (5, 0.0, 1.0), "mixed", source=source)
    assert torch.load(path, weights_only=False)["meta"]["feature_statistic_is_structural"] is True


def test_verify_stage_skips_the_feature_condition_of_featureless_datasets(tmp_path, monkeypatch):
    source = ss.make_shift_source(_graph_dataset("cls", cls=FeaturelessGraphList), "fl", "graph", induced=False)
    cfg = _cfg(tmp_path)
    cfg.seeds = [0]
    cfg.data_preparation.dataset.num_splits = 1
    cfg.data_preparation.graph_task_splits = [(5, 0.0, 1.0)]
    cfg.data_preparation.target_datasets = "fl"
    cfg.data_preparation.task_level_override = "graph"
    cfg.data_preparation.shift.build = True
    cfg.data_preparation.shift.verify = True
    monkeypatch.setattr(ss, "load_shift_source", lambda *_: source)
    assert ss.run_shift_preparation(cfg) == 0
    built = {p.relative_to(tmp_path / "shift").parts[0] for p in (tmp_path / "shift").rglob("*.pt")}
    assert built == set(ss.SHIFT_CONDITIONS) - {"feature"}
    cfg.data_preparation.shift.build = False
    (tmp_path / "shift" / "mixed" / "fl" / "fl_graph_seed0_splits-5-0-100.pt").unlink()
    assert ss.run_shift_preparation(cfg) == 1  # other conditions are still verified

def test_unsupported_inputs_raise(tmp_path, node_setup):
    _, _, source = node_setup
    cfg = _cfg(tmp_path)
    with pytest.raises(NotImplementedError):
        ss.build_shift_split(cfg, "synth", "edge", 42, (0.1, 0.05, 0.1), "feature", source=source)
    with pytest.raises(ValueError):
        ss.build_shift_split(cfg, "synth", "node", 42, (0.1, 0.1, 0.8), "feature", source=source)


def test_verify_shift_root(tmp_path, node_setup):
    _, _, source = node_setup
    cfg = _cfg(tmp_path)
    split = (5, 0.0, 1.0)
    path = ss.build_shift_split(cfg, "synth", "node", 42, split, "structural", source=source)
    root = tmp_path / "shift" / "structural"
    ss.verify_shift_root(root, [("synth", "node", 42, split)])
    with pytest.raises(ValueError, match="missing"):
        ss.verify_shift_root(root, [("synth", "node", 0, split)])
    # A consumer that regenerated a standard split in place must be caught.
    os.chmod(path, 0o644)
    standard = torch.load(tmp_path / "splits" / "synth" / path.name, weights_only=False)
    torch.save(standard, path)
    with pytest.raises(ValueError, match="not a shift split"):
        ss.verify_shift_root(root, [("synth", "node", 42, split)])


def test_run_shift_preparation_stage(tmp_path, node_setup, monkeypatch):
    _, _, source = node_setup
    cfg = _cfg(tmp_path)
    cfg.seeds = [42, 0]
    cfg.data_preparation.dataset.num_splits = 2
    cfg.data_preparation.node_task_splits = [(0.8, 0.1, 0.1), (5, 0.0, 1.0)]
    cfg.data_preparation.target_datasets = "synth"
    cfg.data_preparation.task_level_override = "node"
    cfg.data_preparation.shift.build = True
    cfg.data_preparation.shift.verify = True
    monkeypatch.setattr(ss, "load_shift_source", lambda *_: source)
    assert ss.run_shift_preparation(cfg) == 0
    files = sorted(p.relative_to(tmp_path / "shift").as_posix() for p in (tmp_path / "shift").rglob("*.pt"))
    assert files == [f"{c}/synth/synth_node_seed{s}_splits-5-0-100.pt" for c in sorted(ss.SHIFT_CONDITIONS)
                     for s in (0, 42)]
    cfg.data_preparation.shift.build = False
    (tmp_path / "shift" / "mixed" / "synth" / "synth_node_seed0_splits-5-0-100.pt").unlink()
    assert ss.run_shift_preparation(cfg) == 1


# --------------------------------------------------------------------------- #
# Shared shift evaluation
# --------------------------------------------------------------------------- #
def test_instance_readout_levels():
    g1 = Data(x=torch.randn(3, 4), edge_index=torch.tensor([[0, 1], [1, 2]]), target_node_index=torch.tensor([2]),
              edge_label_index=torch.tensor([[0], [1]]))
    g2 = Data(x=torch.randn(2, 4), edge_index=torch.tensor([[0], [1]]), target_node_index=torch.tensor([0]),
              edge_label_index=torch.tensor([[1], [0]]))
    batch = Batch.from_data_list([g1, g2])
    h = batch.x
    assert torch.equal(shift_eval.instance_readout(h, batch, "node"), torch.stack([g1.x[2], g2.x[0]]))
    assert torch.allclose(shift_eval.instance_readout(h, batch, "edge"),
                          torch.stack([g1.x[0] * g1.x[1], g2.x[1] * g2.x[0]]))
    assert torch.allclose(shift_eval.instance_readout(h, batch, "graph", "mean"),
                          torch.stack([g1.x.mean(0), g2.x.mean(0)]))


def _loader(ys, positions):
    graphs = [Data(x=torch.randn(2, 3), edge_index=torch.empty(2, 0, dtype=torch.long), y=y) for y in ys]
    return DataLoader(Subset(graphs, positions), batch_size=2, shuffle=False)


def test_collect_query_outputs_prediction_spaces():
    loader = _loader([torch.tensor([i % 3]) for i in range(6)], [5, 1, 3])
    model = torch.nn.Linear(3, 3)

    def model_fn(batch):
        from torch_geometric.nn import global_mean_pool
        return model(global_mean_pool(batch.x, batch.batch))

    out = shift_eval.collect_query_outputs(model_fn, loader, "cpu", "classification", 1)
    assert out["index"].tolist() == [5, 1, 3]
    assert torch.allclose(out["pred"].sum(-1), torch.ones(3))
    assert out["y"].view(-1).tolist() == [2, 1, 0]

    binary = shift_eval.collect_query_outputs(lambda b: torch.zeros(b.num_graphs, 1), loader, "cpu", "classification", 1)
    assert torch.allclose(binary["pred"], torch.full((3, 1), 0.5))
    reg = shift_eval.collect_query_outputs(lambda b: torch.full((b.num_graphs, 2), -4.0), loader, "cpu", "regression", 2)
    assert torch.equal(reg["pred"], torch.full((3, 2), -4.0))


def test_brier_risk_classification_and_link():
    pred = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]])
    y = torch.tensor([[0], [1], [-1]])  # the invalid row is skipped
    expected = 0.5 * ((0.3 ** 2 + 0.2 ** 2 + 0.1 ** 2) + (0.1 ** 2 + 0.9 ** 2 + 0.8 ** 2)) / 2
    for family in (NODE_CLS, GRAPH_CLS):
        assert shift_eval.brier_risk({"pred": pred, "y": y}, task_family=family) == pytest.approx(expected)
    p = torch.tensor([[0.9], [0.2], [0.6]])
    y_link = torch.tensor([1, 0, 0])
    assert shift_eval.brier_risk({"pred": p, "y": y_link}, task_family=LINK) == pytest.approx(
        ((0.1 ** 2) + (0.2 ** 2) + (0.6 ** 2)) / 3)


def test_brier_risk_multilabel_masks_missing():
    pred = torch.tensor([[0.8, 0.5], [0.1, 0.9]])
    signed = torch.tensor([[1.0, 0.0], [-1.0, 1.0]])  # {-1, 0, 1}: 0 = missing
    expected = ((0.2 ** 2) + (0.1 ** 2 + 0.1 ** 2) / 2) / 2
    assert shift_eval.brier_risk({"pred": pred, "y": signed}, task_family=MULTILABEL) == pytest.approx(expected)
    nan_y = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
    assert shift_eval.brier_risk({"pred": pred, "y": nan_y}, task_family=MULTILABEL) == pytest.approx(expected)


def test_brier_risk_regression_uses_support_median_mad():
    support = torch.tensor([[0.0], [2.0], [10.0], [4.0]])
    pred, y = torch.tensor([[3.0], [8.0]]), torch.tensor([[5.0], [2.0]])
    norm = RegressionNormalizer().fit(support)
    expected = float((norm.transform(pred) - norm.transform(y)).abs().mean())
    out = {"pred": pred, "y": y}
    assert shift_eval.brier_risk(out, task_family=REGRESSION, support_targets=support) == pytest.approx(expected)
    sq = float((0.5 * (norm.transform(pred) - norm.transform(y)) ** 2).mean())
    assert shift_eval.brier_risk(out, task_family=REGRESSION, support_targets=support, reg_kind="sq") == \
        pytest.approx(sq)
    with pytest.raises(ValueError):
        shift_eval.brier_risk(out, task_family=REGRESSION)


def test_brier_risk_delegates_to_routing_loss(monkeypatch):
    calls = []

    def spy(*args, **kwargs):
        calls.append(args[2])
        return routing_loss(*args, **kwargs)

    monkeypatch.setattr(shift_eval, "routing_loss", spy)
    shift_eval.brier_risk({"pred": torch.tensor([[0.5, 0.5]]), "y": torch.tensor([1])}, task_family=NODE_CLS)
    assert calls == [NODE_CLS]


def test_save_query_predictions(tmp_path):
    out = {"index": torch.tensor([3, 1]), "y": torch.tensor([[1], [0]]), "pred": torch.tensor([[0.2, 0.8], [0.6, 0.4]])}
    path = tmp_path / "preds" / "run.pt"
    shift_eval.save_query_predictions(str(path), out, {"method": "graphmetro", "test_brier": 0.1})
    saved = torch.load(path, weights_only=False)
    assert saved["index"].tolist() == [3, 1] and torch.equal(saved["pred"], out["pred"])
    assert saved["meta"]["method"] == "graphmetro"
