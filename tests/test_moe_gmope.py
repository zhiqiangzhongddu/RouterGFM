"""GMoPE unit tests on tiny synthetic graphs (CPU, seconds)."""

import copy
import csv
import math
import os

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

import src.moe.gmope.pretrain as pretrain_mod
import src.moe.gmope.trainer as trainer_mod
from src.config import cfg as base_cfg
from src.moe.gmope import run_gmope
from src.moe.gmope.pretrain import (
    GMoPEPretrainer,
    MultiSourceBatchStream,
    build_model,
    build_objectives,
    load_gmope_checkpoint,
    pretrain_step,
)
from src.moe.gmope.routing import (
    aggregate_embeddings,
    confidence_weights,
    hard_topk_gate,
    scores_from_losses,
    soft_orthogonality_loss,
    soft_topk_gate,
)
from src.moe.gmope.run import parse_gmope_tasks
from src.moe.gmope.task import GMoPETask
from src.moe.gmope.trainer import GMoPERunner

CPU = torch.device("cpu")
FEAT = 6


def _graph(gen, num_nodes=None, **attrs):
    n = int(num_nodes or torch.randint(5, 9, (1,), generator=gen).item())
    ring = torch.arange(n)
    chord_src = torch.randint(0, n, (2,), generator=gen)
    chord_dst = torch.randint(0, n, (2,), generator=gen)
    edges = torch.stack([torch.cat([ring, chord_src]), torch.cat([(ring + 1) % n, chord_dst])])
    edges = edges[:, edges[0] != edges[1]]
    data = Data(x=torch.randn(n, FEAT, generator=gen), edge_index=to_undirected(edges, num_nodes=n))
    for key, value in attrs.items():
        setattr(data, key, value)
    return data


def _sources():
    gen = torch.Generator().manual_seed(0)
    return {name: [_graph(gen) for _ in range(8)] for name in ("src_a", "src_b")}


def _targets(kind, count=12):
    gen = torch.Generator().manual_seed(1)
    graphs = []
    for i in range(count):
        if kind == "node":
            graphs.append(_graph(gen, y=torch.tensor([i % 3]), target_node_index=torch.tensor([0])))
        elif kind == "edge":
            graphs.append(_graph(gen, y=torch.tensor([i % 2]), edge_label_index=torch.tensor([[0], [1]])))
        elif kind == "graph_cls":
            graphs.append(_graph(gen, y=torch.tensor([i % 3])))
        elif kind == "multilabel":
            y = torch.tensor([[float(i % 2), float((i + 1) % 2), float("nan"), float(i % 3 == 0)]])
            graphs.append(_graph(gen, y=y))
        else:
            graphs.append(_graph(gen, y=100.0 + 10.0 * torch.randn(1, 2, generator=gen)))
    return graphs


_KINDS = {
    "node": dict(task_level="node", split=(2, 0.0, 1.0), num_classes=3, label_dim=1, task_type="classification"),
    "edge": dict(task_level="edge", split=(0.5, 0.25, 0.25), num_classes=2, label_dim=1, task_type="classification"),
    "graph_cls": dict(task_level="graph", split=(2, 0.0, 1.0), num_classes=3, label_dim=1, task_type="classification"),
    "multilabel": dict(task_level="graph", split=(2, 0.0, 1.0), num_classes=2, label_dim=4, task_type="classification"),
    "regression": dict(task_level="graph", split=(2, 0.0, 1.0), num_classes=None, label_dim=2, task_type="regression"),
}


def _cfg(tmp_path, kind="node"):
    spec = _KINDS[kind]
    cfg = base_cfg.clone()
    cfg.seed = 42
    cfg.seeds = [42, 7]
    cfg.moe.method = "gmope"
    cfg.save_results.output_dir = str(tmp_path / "results")
    g = cfg.moe.gmope
    g.dataset.name = f"toy_{kind}"
    g.dataset.task_level = spec["task_level"]
    g.dataset.induced = True
    g.dataset.fixed_split = spec["split"]
    g.in_dim = FEAT
    g.prompt_dim = 3
    g.num_experts = 3
    g.expert.num_layers = 2
    g.expert.hidden_dim = 8
    g.expert.out_dim = 8
    g.pretrain.node_sources = ["src_a", "src_b"]
    g.pretrain.graph_sources = ["src_a", "src_b"]
    g.pretrain.epochs = 2
    g.pretrain.batch_size = 4
    g.pretrain.max_batches_per_source = 2
    g.pretrain.checkpoint_dir = str(tmp_path / "pretrained")
    g.finetune.epochs = 2
    g.finetune.batch_size = 4
    g.finetune.early_stopping = 0
    g.checkpoint_dir = str(tmp_path / "checkpoints")
    g.log_dir = str(tmp_path / "logs")
    g.num_runs = 1
    return cfg


def _meta(num_experts=3):
    return {
        "num_experts": num_experts, "in_dim": FEAT, "prompt_dim": 3, "hidden_dim": 8, "out_dim": 8,
        "num_layers": 2, "gnn_type": "gcn", "dropout": 0.5, "act": "relu", "graph_pooling": "mean",
        "use_batchnorm": False, "objective": "edge_pred",
    }


def _set_task_dims(cfg, kind):
    spec = _KINDS[kind]
    ds = cfg.moe.gmope.dataset
    ds.task_type = spec["task_type"]
    ds.num_classes = spec["num_classes"]
    ds.label_dim = spec["label_dim"]


def _batch(count=3, seed=5, **attrs):
    gen = torch.Generator().manual_seed(seed)
    return Batch.from_data_list([_graph(gen, **attrs) for _ in range(count)])


def _patch_data(monkeypatch, kind):
    """Route dataset creation and split loading to synthetic graphs."""
    spec = _KINDS[kind]
    sources = _sources()

    def fake_source_dataset(**kwargs):
        node_route = kwargs["task_level"] == "node"
        assert kwargs["induced"] is node_route and kwargs["split"] is None
        return sources[kwargs["name"]]

    graphs = _targets(kind)
    meta = {
        "num_node_features": FEAT, "num_classes": spec["num_classes"],
        "label_dim": spec["label_dim"], "task_type": spec["task_type"],
    }

    def fake_loaders(*, dataset, batch_size, split, **_):
        n = len(dataset)
        if isinstance(split[0], int):
            train, val, test = dataset[:6], [], dataset[6:]
        else:
            train, val, test = dataset[: n // 2], dataset[n // 2: 3 * n // 4], dataset[3 * n // 4:]
        return (
            DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(val, batch_size=batch_size),
            DataLoader(test, batch_size=batch_size),
        )

    monkeypatch.setattr(pretrain_mod, "create_dataset", fake_source_dataset)
    monkeypatch.setattr(trainer_mod, "create_dataset", lambda **_: graphs)
    monkeypatch.setattr(trainer_mod, "dataset_info", lambda **_: dict(meta))
    monkeypatch.setattr(trainer_mod, "make_workflow_loaders", fake_loaders)
    monkeypatch.setattr(trainer_mod, "log_split_instance_counts", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# Model and routing primitives
# ---------------------------------------------------------------------------

def test_expert_forward_concatenates_prompt_without_mutating_input():
    torch.manual_seed(0)
    model = build_model(_meta())
    data = _batch(count=2)
    x_before = data.x.clone()
    captured = {}
    handle = model.experts[1].register_forward_pre_hook(
        lambda _mod, args: captured.__setitem__("x", args[0].x.detach().clone())
    )
    model.expert_forward(1, data)
    handle.remove()
    expected = torch.cat([x_before, model.prompts[1].detach().expand(x_before.size(0), -1)], dim=-1)
    assert torch.equal(captured["x"], expected)
    assert torch.equal(data.x, x_before)
    assert model.pooled_all(data).shape == (3, 2, 8)
    with pytest.raises(ValueError, match="in_dim"):
        model.expert_forward(0, Data(x=torch.randn(3, FEAT + 1), edge_index=torch.empty(2, 0, dtype=torch.long)))


def test_soft_orthogonality_loss():
    assert soft_orthogonality_loss(torch.eye(3)).item() == pytest.approx(1.0)
    assert soft_orthogonality_loss(torch.ones(3, 4)).item() == pytest.approx(math.e, rel=1e-5)

    torch.manual_seed(0)
    prompts = nn.Parameter(torch.randn(4, 5))

    def mean_cos(p):
        normed = torch.nn.functional.normalize(p.detach(), dim=-1)
        return (normed @ normed.t())[~torch.eye(4, dtype=torch.bool)].mean().item()

    before = mean_cos(prompts)
    soft_orthogonality_loss(prompts).backward()
    torch.optim.SGD([prompts], lr=0.5).step()
    assert mean_cos(prompts) < before

    single = nn.Parameter(torch.randn(1, 5))
    loss = soft_orthogonality_loss(single)
    loss.backward()
    assert loss.item() == 0.0 and torch.isfinite(single.grad).all()


def test_gates():
    scores = scores_from_losses(torch.tensor([0.3, 0.1, 0.5, 0.2]))
    soft = soft_topk_gate(scores, 2, 0.8)
    assert (soft >= 0).all() and soft.sum().item() == pytest.approx(1.0)
    assert torch.nonzero(soft).view(-1).tolist() == [1, 3]
    assert soft[1] > soft[3]  # lower loss -> larger weight
    cold = soft_topk_gate(scores, 4, 1e-4)
    assert torch.allclose(cold, torch.tensor([0.0, 1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.equal(hard_topk_gate(scores, 2), torch.tensor([0.0, 0.5, 0.0, 0.5]))
    assert torch.allclose(hard_topk_gate(scores, 4), torch.full((4,), 0.25))
    assert torch.equal(hard_topk_gate(torch.zeros(3), 1), torch.tensor([1.0, 0.0, 0.0]))  # ties -> lowest index


def test_confidence_weights():
    sharp_vs_flat = torch.tensor([[[30.0, -30.0, -30.0]], [[0.0, 0.0, 0.0]]])
    omega = confidence_weights(sharp_vs_flat, task_type="classification", multilabel=False)
    assert torch.allclose(omega, torch.tensor([[1.0], [0.0]]), atol=1e-6)
    assert torch.allclose(
        confidence_weights(torch.zeros(3, 2, 4), task_type="classification", multilabel=False),
        torch.full((3, 2), 1.0 / 3),
    )

    def alpha_binary(z):
        p = torch.sigmoid(torch.tensor(z))
        return 1.0 - float(-(p * p.log() + (1 - p) * (1 - p).log()) / math.log(2.0))

    binary = torch.tensor([[[1.0]], [[-2.0]]])
    a = [alpha_binary(1.0), alpha_binary(-2.0)]
    omega = confidence_weights(binary, task_type="classification", multilabel=False)
    assert omega[:, 0].tolist() == pytest.approx([a[0] / sum(a), a[1] / sum(a)], rel=1e-5)

    multi = torch.tensor([[[1.0, -1.0, 3.0]], [[0.5, 0.0, -4.0]]])
    a = [1.0 - sum(1.0 - alpha_binary(z) for z in row) / 3 for row in ([1.0, -1.0, 3.0], [0.5, 0.0, -4.0])]
    omega = confidence_weights(multi, task_type="classification", multilabel=True)
    assert omega[:, 0].tolist() == pytest.approx([a[0] / sum(a), a[1] / sum(a)], rel=1e-5)

    assert torch.allclose(
        confidence_weights(torch.randn(2, 3, 1), task_type="regression", multilabel=False),
        torch.full((2, 3), 0.5),
    )

    torch.manual_seed(0)
    head = nn.Linear(8, 3)
    pooled = torch.randn(3, 5, 8)
    omega = torch.softmax(torch.randn(3, 5), dim=0)
    lhs = head(aggregate_embeddings(pooled, omega))
    rhs = (omega.unsqueeze(-1) * head(pooled)).sum(dim=0)
    assert torch.allclose(lhs, rhs, atol=1e-6)


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------

def test_pretrain_step_backprops_only_selected_experts(tmp_path):
    cfg = _cfg(tmp_path)
    torch.manual_seed(0)
    model = build_model(_meta())
    objectives = build_objectives(cfg, _meta())
    batch = _batch(count=3)

    loss, gate, _ = pretrain_step(model, objectives, batch, CPU, top_k=1, tau=0.8, ortho_weight=0.0, seed=123)
    loss.backward()
    selected = int(gate.argmax())
    assert torch.count_nonzero(gate).item() == 1
    for m, expert in enumerate(model.experts):
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in expert.parameters())
        assert has_grad == (m == selected)
    unselected = [m for m in range(3) if m != selected]
    assert model.prompts.grad[unselected].abs().sum().item() == 0.0

    model.zero_grad()
    loss, gate_again, _ = pretrain_step(model, objectives, batch, CPU, top_k=1, tau=0.8, ortho_weight=1.0, seed=123)
    loss.backward()
    assert torch.equal(gate_again, gate)
    ortho_grad = torch.autograd.grad(soft_orthogonality_loss(model.prompts), model.prompts)[0]
    assert torch.allclose(model.prompts.grad[unselected], ortho_grad[unselected], atol=1e-7)
    assert (model.prompts.grad.abs().sum(dim=1) > 0).all()


def test_pretrain_step_shares_negative_samples_across_experts(tmp_path, monkeypatch):
    import src.pretrain.methods.edge_pred as edge_pred

    calls = []
    original = edge_pred.negative_sampling

    def recording(*args, **kwargs):
        out = original(*args, **kwargs)
        calls.append(out.clone())
        return out

    monkeypatch.setattr(edge_pred, "negative_sampling", recording)
    cfg = _cfg(tmp_path)
    torch.manual_seed(0)
    model = build_model(_meta())
    objectives = build_objectives(cfg, _meta())
    pretrain_step(model, objectives, _batch(count=3), CPU, top_k=1, tau=0.8, ortho_weight=1.0, seed=7)

    passes = 3 + 1  # M scoring passes + K gradient passes
    assert len(calls) % passes == 0 and calls
    per_pass = len(calls) // passes
    chunks = [calls[i * per_pass:(i + 1) * per_pass] for i in range(passes)]
    for chunk in chunks[1:]:
        assert all(torch.equal(a, b) for a, b in zip(chunk, chunks[0]))


def test_multi_source_batch_stream_is_homogeneous_and_capped():
    gen = torch.Generator().manual_seed(0)
    loaders = {
        name: DataLoader([_graph(gen, source_id=torch.tensor([k])) for _ in range(9)], batch_size=2, shuffle=True)
        for k, name in enumerate(("a", "b"))
    }
    stream = MultiSourceBatchStream(loaders, max_batches_per_source=3, seed=1)
    assert len(stream) == 6
    seen = [(name, batch) for name, batch in stream]
    assert [name for name, _ in seen].count("a") == 3 and [name for name, _ in seen].count("b") == 3
    for name, batch in seen:
        assert set(batch.source_id.tolist()) == {("a", "b").index(name)}
    order_a = [name for name, _ in MultiSourceBatchStream(loaders, max_batches_per_source=3, seed=1)]
    order_b = [name for name, _ in MultiSourceBatchStream(loaders, max_batches_per_source=3, seed=1)]
    assert order_a == order_b
    assert len(MultiSourceBatchStream(loaders, max_batches_per_source=0, seed=1)) == 10


@pytest.mark.parametrize("route", ["node", "graph"])
def test_pretrainer_smoke_roundtrip_and_skip(tmp_path, monkeypatch, route):
    _patch_data(monkeypatch, "node")
    cfg = _cfg(tmp_path)
    pretrainer = GMoPEPretrainer(cfg, route)
    path = pretrainer.fit()
    assert os.path.isfile(path) and path.startswith(str(tmp_path / "pretrained" / route))
    assert [entry["epoch"] for entry in pretrainer.history] == [1, 2]
    k = 3 if route == "node" else 1
    for entry in pretrainer.history:
        assert math.isfinite(entry["loss"])
        for stats in entry["routing"].values():
            assert len(stats["selected"]) == 3
            assert sum(stats["selected"]) == pytest.approx(k)
            assert sum(stats["gate"]) == pytest.approx(1.0)

    payload = torch.load(path, map_location="cpu")
    model, objectives, meta = load_gmope_checkpoint(cfg, path, CPU)
    assert meta["route"] == route and meta["num_experts"] == 3 and meta["top_k"] == k
    for key, value in model.state_dict().items():
        assert torch.equal(value, payload["model_state"][key])
    assert len(objectives) == 3

    def must_not_train(self, _path):
        raise AssertionError("existing checkpoint should be reused")

    monkeypatch.setattr(GMoPEPretrainer, "_train_and_save", must_not_train)
    assert GMoPEPretrainer(cfg, route).fit() == path


# ---------------------------------------------------------------------------
# Stages B and C
# ---------------------------------------------------------------------------

def test_prompt_tuning_keeps_experts_frozen(tmp_path):
    cfg = _cfg(tmp_path)
    _set_task_dims(cfg, "node")
    torch.manual_seed(0)
    model = build_model(_meta())
    expert_state = copy.deepcopy(model.experts.state_dict())
    prompts_before = model.prompts.detach().clone()
    model.freeze_experts()
    task = GMoPETask(cfg, num_experts=3)
    head_before = copy.deepcopy(task.classifier.state_dict())
    optimizer = torch.optim.Adam([model.prompts] + list(task.parameters_to_optimize()), lr=0.01)
    loader = DataLoader(_targets("node"), batch_size=4)
    for _, batch in zip(range(3), loader):
        optimizer.zero_grad()
        loss, log = task.step(model, batch, CPU)
        loss.backward()
        optimizer.step()
        assert {"train_task_loss", "train_ortho_loss", "train_acc"} <= set(log)
    model.train()
    assert not any(module.training for module in model.experts.modules())
    for key, value in model.experts.state_dict().items():
        assert torch.equal(value, expert_state[key])
    assert not torch.equal(model.prompts.detach(), prompts_before)
    assert any(not torch.equal(v, head_before[k]) for k, v in task.classifier.state_dict().items())
    batch = _batch(count=2)
    assert torch.equal(model.pooled_all(batch), model.pooled_all(batch))  # dropout off


def test_single_expert_reduces_to_prompted_expert(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.moe.gmope.num_experts = 1
    _set_task_dims(cfg, "node")
    torch.manual_seed(0)
    model = build_model(_meta(num_experts=1)).eval()
    task = GMoPETask(cfg, num_experts=1)
    batch = _batch(count=3)
    _, graph_repr = model.expert_forward(0, batch)
    assert torch.allclose(task.predict(model, batch, CPU), task.classifier(graph_repr), atol=1e-6)
    assert model.ortho_loss().item() == 0.0


def test_label_free_routing_ignores_labels(tmp_path):
    cfg = _cfg(tmp_path, "regression")
    _set_task_dims(cfg, "regression")
    torch.manual_seed(0)
    model = build_model(_meta())
    model.freeze_experts()
    objectives = build_objectives(cfg, _meta())
    with pytest.raises(ValueError, match="one pretraining objective per expert"):
        GMoPETask(cfg, num_experts=3)
    task = GMoPETask(cfg, num_experts=3, route_objectives=list(objectives))
    assert task.top_k == 1  # graph route default
    batch = _batch(count=3, y=torch.full((1, 2), float("nan")))
    gate = task._train_gate(model, batch, CPU, torch.zeros(3))
    assert torch.isfinite(gate).all() and torch.count_nonzero(gate).item() == 1
    assert gate.sum().item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Runner / run orchestration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["edge", "graph_cls", "multilabel", "regression"])
def test_runner_two_epochs_per_task_type(tmp_path, monkeypatch, kind):
    _patch_data(monkeypatch, kind)
    cfg = _cfg(tmp_path, kind)
    runner = GMoPERunner(cfg)
    runner.fit()
    metrics = runner.best_metrics
    expected = {"edge": "test_auc", "regression": "test_mae"}.get(kind, "test_acc")
    assert math.isfinite(metrics[expected])
    assert runner.monitor_name == ("val_auc" if kind == "edge" else "train_loss")
    assert os.path.isfile(runner.get_checkpoint_path_for_metrics())
    assert runner.task.top_k == (3 if kind == "edge" else 1)

    if kind == "regression":
        normalizer = runner.task.normalizer
        assert normalizer.active and normalizer.mean.mean().item() > 50.0  # fitted on raw ~100 targets
        preds, labels = [], []
        runner.task.eval()
        with torch.no_grad():
            for batch in runner.test_loader:
                z = runner.task.predict(runner.model, batch, CPU)
                preds.append(z * normalizer.std + normalizer.mean)
                labels.append(batch.y)
        manual_mae = (torch.cat(preds) - torch.cat(labels)).abs().mean().item()
        assert metrics["test_mae"] == pytest.approx(manual_mae, rel=1e-5)


def test_run_gmope_node_end_to_end_appends_result_row(tmp_path, monkeypatch):
    _patch_data(monkeypatch, "node")
    state = {"finalizing": False}
    make_loaders = trainer_mod.make_workflow_loaders

    class _GuardedLoader:
        """Test split must only be read by the single final evaluation."""

        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            assert state["finalizing"], "query split read before the final evaluation"
            return iter(self.loader)

        def __len__(self):
            return len(self.loader)

    def guarded_loaders(**kwargs):
        train, val, test = make_loaders(**kwargs)
        return train, val, _GuardedLoader(test)

    original_finalize = GMoPERunner._finalize_best_checkpoint

    def finalize(self):
        state["finalizing"] = True
        return original_finalize(self)

    monkeypatch.setattr(trainer_mod, "make_workflow_loaders", guarded_loaders)
    monkeypatch.setattr(GMoPERunner, "_finalize_best_checkpoint", finalize)
    cfg = _cfg(tmp_path, "node")
    cfg.moe.gmope.num_experts = 0  # resolved to one expert per source (M = N = 2)
    assert run_gmope(cfg) == 0

    with open(tmp_path / "results" / "moe_gmope.tsv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 1
    row = rows[0]
    assert row["moe.method"] == "gmope"
    assert row["moe.gmope.dataset.name"] == "toy_node"
    assert row["moe.gmope.num_experts"] == "2" and row["moe.gmope.finetune.top_k"] == "2"
    assert math.isfinite(float(row["test_acc_mean"]))

    # Second invocation reuses the finetune checkpoint (skip_if_exists) and adds no row.
    assert run_gmope(cfg) == 0
    with open(tmp_path / "results" / "moe_gmope.tsv", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle, delimiter="\t"))) == 1


def test_stage_finetune_requires_pretrained_route(tmp_path, monkeypatch):
    _patch_data(monkeypatch, "node")
    cfg = _cfg(tmp_path, "node")
    cfg.moe.gmope.stage = "finetune"
    with pytest.raises(FileNotFoundError, match="route checkpoint"):
        GMoPERunner(cfg).fit()
    cfg.moe.gmope.stage = "pretrain"
    cfg.moe.gmope.pretrain.routes = ["node"]
    assert run_gmope(cfg) == 0
    cfg.moe.gmope.stage = "finetune"
    runner = GMoPERunner(cfg)
    runner.fit()
    assert math.isfinite(runner.best_metrics["test_acc"])


def _set(cfg, dotted, value):
    node = cfg
    *parents, leaf = dotted.split(".")
    for part in parents:
        node = getattr(node, part)
    setattr(node, leaf, value)


def test_run_identity_tracks_behaviour_not_orchestration(tmp_path):
    cfg = _cfg(tmp_path)
    baseline = GMoPERunner(cfg).run_name
    pretrain_path = GMoPEPretrainer(cfg, "node").checkpoint_path()

    behaviour = {
        "moe.gmope.prompt_dim": 4,
        "moe.gmope.ortho_weight": 0.5,
        "moe.gmope.tau": 0.5,
        "moe.gmope.expert.dropout": 0.1,
        "moe.gmope.pretrain.objective": "dgi",
        "moe.gmope.pretrain.node_sources": ["src_a", "src_c"],
        "moe.gmope.finetune.route_loss": "task",
        "moe.gmope.finetune.weight_decay": 0.1,
        "moe.gmope.aggregation.experts": "routed",
        "pretrain.edge_pred.neg_ratio": 2.0,
        "model.activation": "gelu",
        "data_preparation.dataset.split_root": "data/alternate_splits",
        "seeds": [1, 42],
    }
    for key, value in behaviour.items():
        changed = cfg.clone()
        _set(changed, key, value)
        assert GMoPERunner(changed).run_name != baseline, key

    operational = {
        "moe.gmope.stage": "finetune",
        "moe.gmope.pretrain.routes": ["node"],
        "moe.gmope.pretrain.num_workers": 2,
        "moe.gmope.finetune.num_workers": 2,
        "moe.gmope.pretrain.checkpoint_dir": str(tmp_path / "elsewhere"),
        "moe.gmope.pretrain.skip_if_exists": False,
        "moe.gmope.checkpoint_dir": str(tmp_path / "other_ckpt"),
        "moe.gmope.log_dir": str(tmp_path / "other_logs"),
        "moe.gmope.num_runs": 3,
        "moe.gmope.run_tasks_tsv": True,
        "moe.gmope.tasks_tsv": "other.tsv",
    }
    for key, value in operational.items():
        changed = cfg.clone()
        _set(changed, key, value)
        assert GMoPERunner(changed).run_name == baseline, key

    # 0 ("auto") and the value it resolves to are the same run.
    auto, explicit = cfg.clone(), cfg.clone()
    auto.moe.gmope.num_experts = 0
    explicit.moe.gmope.num_experts = 2
    explicit.moe.gmope.finetune.top_k = 2
    assert GMoPERunner(auto).run_name == GMoPERunner(explicit).run_name

    # Route checkpoints are target, budget and finetune independent.
    target_changed = cfg.clone()
    target_changed.moe.gmope.dataset.name = "other"
    target_changed.moe.gmope.dataset.fixed_split = (100, 0.0, 1.0)
    target_changed.moe.gmope.finetune.lr = 0.1
    target_changed.seed = 7
    assert GMoPEPretrainer(target_changed, "node").checkpoint_path() == pretrain_path
    epochs_changed = cfg.clone()
    epochs_changed.moe.gmope.pretrain.epochs = 3
    assert GMoPEPretrainer(epochs_changed, "node").checkpoint_path() != pretrain_path


def test_dispatch_and_task_grid():
    from src.moe.run import _build_moe_cfg, _load_runner

    assert _load_runner("gmope") is run_gmope
    assert _build_moe_cfg(["moe.gmope.dataset.name", "cora"]).moe.method == "gmope"
    assert _build_moe_cfg(["moe.gmoe.dataset.name", "cora"]).moe.method == "gmoe"

    tasks = parse_gmope_tasks(os.path.join(os.path.dirname(__file__), "..", "slurm", "moe.gmope.all.tsv"))
    assert len(tasks) == 16
    lp = [t for t in tasks if t["task_level"] == "edge"]
    assert sorted(t["dataset"] for t in lp) == ["cornell", "dblp"]
    assert all(t["fixed_split"] == (0.1, 0.05, 0.1) for t in lp)
    few_shot = [t for t in tasks if t["task_level"] != "edge"]
    assert len({t["dataset"] for t in few_shot}) == 7
    assert sorted({t["fixed_split"] for t in few_shot}) == [(5, 0.0, 1.0), (100, 0.0, 1.0)]
