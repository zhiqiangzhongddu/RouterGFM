"""SAGMM-PE matched-pool baseline: gate, pruning, gate features, candidates, runner (CPU, synthetic)."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from src.moe.routergfm.baselines import load_runner_class
from src.moe.routergfm.baselines import run as matched_run
from src.moe.routergfm.baselines.sagmm_pe import SAGMMPERunner
from src.moe.routergfm.baselines.sagmm_pe import trainer as sagmm_trainer
from src.moe.routergfm.baselines.candidates import candidate_experts, mean_historical_rank
from src.moe.routergfm.baselines.sagmm_pe.gate_features import GateInputs, build_gate_inputs, multihop_lap_features
from src.moe.routergfm.baselines.sagmm_pe.gating import TAAGGate, cv_squared, diversity_loss, sga_scores
from src.moe.routergfm.baselines.sagmm_pe.model import mix
from src.moe.routergfm.baselines.sagmm_pe.pruning import ExpertPruner, threshold_factor
from src.moe.routergfm.baselines.sagmm_pe.trainer import level_params
from src.moe.routergfm.common import MULTILABEL, NODE_CLS, AppSpec
from src.moe.routergfm.infra import RouterInfra
from tests.routergfm.fixtures import SyntheticDataProvider, tiny_cfg


def _naive_sga(x, w_q, w_k, w_v):
    q, k, v = x @ w_q, x @ w_k, x @ w_v
    q, k = q / q.norm(), k / k.norm()
    n = x.size(0)
    attn = q @ k.t()  # explicit n x n
    return (v + attn @ v / n) / (1.0 + attn @ torch.ones(n, 1) / n)


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def test_sga_scores_match_naive_and_own_populations():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(7, 5, generator=g)
    w = [torch.randn(5, 3, generator=g) for _ in range(3)]
    assert torch.allclose(sga_scores(x, *w), _naive_sga(x, *w), atol=1e-6)

    # Own populations (zero-padded) = the full SGA of each population read at its query row.
    y = torch.randn(4, 5, generator=g)
    pop = torch.zeros(2, 7, 5)
    pop[0], pop[1, :4] = x, y
    inputs = GateInputs(torch.stack([x[2], y[3]]), pop, torch.tensor([7, 4]))
    out = sga_scores(inputs.query, *w, inputs.pop, inputs.pop_size)
    assert torch.allclose(out[0], _naive_sga(x, *w)[2], atol=1e-6)
    assert torch.allclose(out[1], _naive_sga(y, *w)[3], atol=1e-6)
    sub = inputs.subset(torch.tensor([1]))
    assert sub.pop.size(1) == 4
    assert torch.allclose(sga_scores(sub.query, *w, sub.pop, sub.pop_size)[0], out[1], atol=1e-6)


def test_gate_straight_through_fallback_and_masking():
    torch.manual_seed(0)
    gate = TAAGGate(5, 4)
    inputs = GateInputs(torch.randn(12, 5))
    with torch.no_grad():
        gate.expert_mask[3] = 0.0
        scores = gate(inputs).scores[:, :3]
        gate.threshold.fill_(float(torch.logit(scores.max(-1).values.median())))  # about half the rows: none above
    out = gate(inputs)
    assert set(out.active.unique().tolist()) <= {0.0, 1.0}
    thr = torch.sigmoid(gate.threshold)
    empty = ~(out.scores > thr).any(-1)
    assert bool(empty.any()) and bool((~empty).any())
    assert torch.all(out.active[empty].sum(-1) == 1)
    assert torch.equal(out.active[empty].argmax(-1), out.scores[empty].argmax(-1))
    assert torch.all(out.gates[:, 3] == 0) and torch.all(out.active[:, 3] == 0)
    assert torch.allclose(out.gates, out.scores * out.active)

    out.gates.sum().backward()
    assert gate.threshold.grad.abs().sum() > 0 and gate.w_v.weight.grad.abs().sum() > 0
    assert torch.all(gate.w_v.weight.grad[3] == 0) and gate.threshold.grad[3] == 0


def test_mixing_ignores_pruned_experts():
    torch.manual_seed(0)
    weights = torch.rand(6, 4)
    weights[:, 1] = 0.0  # pruned
    experts = torch.randn(6, 4, 3)
    y = mix(weights, experts)
    assert torch.allclose(y, torch.einsum("bn,bnd->bd", weights, experts), atol=1e-6)
    noisy = experts.clone()
    noisy[:, 1] = 100 * torch.randn(6, 3)
    assert torch.allclose(mix(weights, noisy), y)


def test_aux_losses():
    assert float(cv_squared(torch.ones(4))) == 0.0
    assert float(cv_squared(torch.tensor([3.0]))) == 0.0
    ortho = torch.eye(5)[:, :3]
    mask = torch.ones(3)
    assert float(diversity_loss(ortho, mask)) == pytest.approx(1.0)  # 0 + mean column norm 1
    dup = torch.stack([ortho[:, 0], ortho[:, 0], ortho[:, 1]], dim=1)
    assert float(diversity_loss(dup, mask)) > 1.0 + 1e-3
    assert float(diversity_loss(dup, torch.tensor([1.0, 0.0, 1.0]))) == pytest.approx(1.0)  # masked duplicate


# --------------------------------------------------------------------------- #
# Pruning
# --------------------------------------------------------------------------- #
def test_threshold_factor_rules():
    for mode, better, worse in (("max", 0.9, 0.5), ("min", 0.1, 0.5)):
        best = 0.7 if mode == "max" else 0.3
        assert threshold_factor(0.6, better, best, mode) == pytest.approx(0.9)
        assert threshold_factor(0.6, worse, best, mode) == pytest.approx(0.45)
        assert threshold_factor(0.6, best, best, mode) == pytest.approx(0.72)
    assert threshold_factor(1.6, 0.1, 0.3, "min") == 2.0
    assert threshold_factor(0.9, 0.3, 0.3, "min") == 0.8
    assert threshold_factor(0.01, 0.5, 0.3, "min") == 0.01


def test_pruner_removes_low_importance_and_respects_min_experts():
    pruner = ExpertPruner(3, ema_decay=0.9, threshold_factor=0.6, min_experts=1)
    mask = torch.ones(3)
    pruner.update(torch.tensor([10.0, 10.0, 0.1]), mask)
    assert torch.allclose(pruner.importance, torch.tensor([9.0, 9.0, 0.09]))
    removed = pruner.prune(mask, n_train=10, current=0.5, best=0.5, mode="min")  # f = 0.72
    assert removed == [2] and mask.tolist() == [1.0, 1.0, 0.0] and pruner.importance[2] == float("-inf")
    pruner.update(torch.tensor([5.0, 5.0, 99.0]), mask)  # pruned experts keep -inf
    assert pruner.importance[2] == float("-inf")

    pruner = ExpertPruner(3, ema_decay=1.0, threshold_factor=2.0, min_experts=2)
    mask = torch.ones(3)
    pruner.update(torch.tensor([1.0, 2.0, 30.0]), mask)
    assert pruner.prune(mask, n_train=1, current=0.1, best=0.3, mode="min") == [0]  # cap: lowest index first
    assert int(mask.sum()) == 2


# --------------------------------------------------------------------------- #
# Gate features
# --------------------------------------------------------------------------- #
def test_multihop_lap_features_on_path_graph():
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    feats = multihop_lap_features(x, edge_index, 2, sign_seed=0)
    x1 = torch.tensor([1.5, 2.0, 3.0, 3.5])
    x2 = torch.tensor([1.75, 13 / 6, 17 / 6, 3.25])
    assert torch.allclose(feats[:, 0], (x[:, 0] + x1 + x2) / 3, atol=1e-6)
    assert feats.shape == (4, 3)
    x_g = feats[:, 1:]
    deg = torch.tensor([1.0, 2.0, 2.0, 1.0])
    assert torch.allclose(deg.sqrt() @ x_g, torch.zeros(2), atol=1e-5)  # trivial eigenvector excluded
    assert float(x_g.abs().mean()) == pytest.approx(float(feats[:, :1].abs().mean()), rel=1e-5)

    padded = multihop_lap_features(x, edge_index, 5, sign_seed=0)
    assert padded.shape == (4, 6) and torch.all(padded[:, 4:] == 0) and torch.any(padded[:, 1:4] != 0)
    assert torch.equal(multihop_lap_features(x, edge_index, 5, sign_seed=0), padded)
    single = multihop_lap_features(torch.ones(1, 1), torch.zeros(2, 0, dtype=torch.long), 3, sign_seed=0)
    assert torch.all(single[:, 1:] == 0)
    assert torch.equal(multihop_lap_features(x, edge_index, 0, sign_seed=0), feats[:, :1])  # X_loc only


def test_build_gate_inputs_rows_per_level():
    g = torch.Generator().manual_seed(1)
    ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    graphs = [
        Data(x=torch.randn(3, 2, generator=g), edge_index=ei, target_node_index=torch.tensor([1]),
             edge_label_index=torch.tensor([[0], [2]])),
        Data(x=torch.randn(5, 2, generator=g), edge_index=ei, target_node_index=torch.tensor([4]),
             edge_label_index=torch.tensor([[3], [4]])),
    ]
    pos = torch.tensor([7, 9])
    node = build_gate_inputs(graphs, pos, "node", 2)
    assert node.query.shape == (2, 4) and node.pop.shape == (2, 5, 4) and node.pop_size.tolist() == [3, 5]
    feats1 = multihop_lap_features(graphs[1].x, ei, 2, sign_seed=9)
    assert torch.allclose(node.query[1], feats1[4]) and torch.all(node.pop[0, 3:] == 0)
    edge = build_gate_inputs(graphs, pos, "edge", 2)
    assert torch.allclose(edge.query[1], (feats1[3] + feats1[4]) / 2)
    graph = build_gate_inputs(graphs, pos, "graph", 2)
    assert graph.pop is None and torch.allclose(graph.query[0], graphs[0].x.mean(0))


# --------------------------------------------------------------------------- #
# Config and candidates
# --------------------------------------------------------------------------- #
def test_level_params_overrides():
    cfg = tiny_cfg(Path("/nonexistent"), write_checkpoints=False)
    block = cfg.moe.routergfm.baselines.sagmm_pe
    node, edge, graph = (level_params(block, lvl) for lvl in ("node", "edge", "graph"))
    assert "edge" not in node and node.epochs == 1000 and node.score_act == "sigmoid"
    assert edge.prune_interval == 28 and edge.importance_threshold_factor == 0.3 and edge.epochs == 1000
    assert graph.score_act == "softplus" and graph.batch_size == 32 and graph.imp_weight == 0.0
    block.graph.set_new_allowed(True)
    block.graph.bogus = 1
    with pytest.raises(KeyError):
        level_params(block, "graph")


def test_mean_historical_rank():
    nan = float("nan")
    mu = torch.tensor([[0.3, 0.1, 0.2, nan], [0.1, nan, 0.5, nan]])
    ranks = mean_historical_rank(mu)
    assert torch.allclose(ranks[:3], torch.tensor([0.5, 0.0, 0.75]))
    assert ranks[3] == float("inf")
    assert torch.all(mean_historical_rank(torch.zeros(0, 3)) == float("inf"))


class _CandidateInfra:
    def __init__(self):
        self.pool = ["e0", "e1", "e2", "e3", "e4"]
        self.hist = [AppSpec("h1", "node", 5, 42), AppSpec("h2", "node", 5, 42), AppSpec("h3", "graph", 5, 42)]
        self.mu = {
            "h1": [0.5, 0.4, 0.1, 0.3, float("nan")],
            "h2": [0.5, 0.4, 0.2, 0.1, float("nan")],
            "h3": [0.0, 0.0, 9.0, 9.0, 0.0],  # other family: ignored
        }

    def compatible_pool(self, app):
        return list(self.pool)

    def task_family(self, app):
        return NODE_CLS if app.task_level == "node" else "graph_cls"

    def historical_applications(self, app):
        return list(self.hist)

    def historical_mu(self, app, eid):
        m = self.mu[app.dataset][self.pool.index(eid)]
        return (m, 0 if math.isnan(m) else 10)


def test_candidate_rules():
    cfg = tiny_cfg(Path("/nonexistent"), write_checkpoints=False)
    b = cfg.moe.routergfm.baselines
    infra, app = _CandidateInfra(), AppSpec("t", "node", 5, 42)
    b.candidate_pool = 2
    b.candidate_rule = "historical_mean"
    assert candidate_experts(cfg, app, infra) == ["e2", "e3"]
    b.candidate_pool = 3
    assert candidate_experts(cfg, app, infra) == ["e1", "e2", "e3"]
    b.candidate_pool = 4
    assert candidate_experts(cfg, app, infra) == ["e0", "e1", "e2", "e3"]  # unobserved e4 ranks last
    b.candidate_rule = "eligible"
    assert candidate_experts(cfg, app, infra) == infra.pool
    b.candidate_rule, b.candidate_pool = "random", 3
    chosen = candidate_experts(cfg, app, infra)
    assert len(chosen) == 3 and chosen == sorted(chosen) and chosen == candidate_experts(cfg, app, infra)
    b.candidate_rule = "bogus"
    with pytest.raises(ValueError):
        candidate_experts(cfg, app, infra)


# --------------------------------------------------------------------------- #
# Runner on a fake infra (node level)
# --------------------------------------------------------------------------- #
class _ToyInfra:
    """3 experts over 60 node instances: expert 0 encodes the class, experts 1-2 are noise.

    Query labels live only in ``evaluate_outputs``; ``data.labels`` holds support labels only.
    """

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        n, self.num_classes = 60, 3
        labels = torch.arange(n) % 3
        self.device = torch.device("cpu")
        support = torch.cat([torch.nonzero(labels == c).view(-1)[:5] for c in range(3)]).sort().values
        mask = torch.ones(n, dtype=torch.bool)
        mask[support] = False
        query = torch.nonzero(mask).view(-1)
        self._query_labels = labels[query]
        self.graphs = []
        for i in range(n):
            size = 4 + i % 3
            ei = torch.stack([torch.arange(size - 1), torch.arange(1, size)])
            ei = torch.cat([ei, ei.flip(0)], dim=1)
            x = 1.0 + 0.1 * torch.randn(size, 4, generator=g)
            self.graphs.append(Data(x=x, edge_index=ei, target_node_index=torch.tensor([i % size])))
        code = torch.zeros(n, 6)
        code[torch.arange(n), labels] = 3.0
        self.readouts = {
            "e0": code + 0.1 * torch.randn(n, 6, generator=g),
            "e1": torch.randn(n, 6, generator=g),
            "e2": torch.randn(n, 6, generator=g),
        }
        self.data_obj = SimpleNamespace(
            task_family=NODE_CLS, level="node", num_classes=3, in_dim=4,
            support_pos=support, query_pos=query, labels={"support": labels[support]},
        )
        self.embedding_calls = []
        self.evaluated = 0

    def data(self, app):
        return self.data_obj

    def compatible_pool(self, app):
        return ["e0", "e1", "e2"]

    def embeddings(self, app, eid, split):
        self.embedding_calls.append((eid, split))
        return self.readouts[eid][getattr(self.data_obj, f"{split}_pos")].clone()

    def instance_graphs(self, app, split):
        return [self.graphs[int(i)] for i in getattr(self.data_obj, f"{split}_pos")]

    def support_labels(self, app):
        return self.data_obj.labels["support"]

    def normalizer(self, app):
        return None

    def evaluate_outputs(self, app, pred):
        self.evaluated += 1
        assert pred.shape == (self._query_labels.numel(), 3)
        return {"acc": float((pred.argmax(-1) == self._query_labels).float().mean() * 100), "risk": 0.1}


def _toy_cfg():
    cfg = tiny_cfg(Path("/nonexistent"), write_checkpoints=False)
    cfg.moe.routergfm.baselines.candidate_rule = "eligible"
    s = cfg.moe.routergfm.baselines.sagmm_pe
    s.epochs, s.lr, s.prune_interval = 150, 0.01, 30
    return cfg


@pytest.fixture(scope="module")
def toy():
    infra = _ToyInfra()
    runner = SAGMMPERunner(_toy_cfg(), AppSpec("toy", "node", 5, 42), infra)
    runner.fit()
    return SimpleNamespace(infra=infra, runner=runner)


def test_toy_runner_learns_informative_expert(toy):
    runner, infra = toy.runner, toy.infra
    assert infra.evaluated == 1 and runner.best_metrics["test_acc"] > 80.0
    assert runner.gate_p == 0 and runner._gate_inputs("support").query.size(1) == runner.data.in_dim  # X' = X_loc
    assert runner.best_metrics["test_risk"] == 0.1 and 1 <= runner.best_metrics["final_num_experts"] <= 3
    assert 1.0 <= runner.best_metrics["test_mean_active_experts"] <= 3.0 and 1 <= runner.best_epoch <= 150
    experts, inv_norm = runner._support_readouts()
    with torch.no_grad():
        logits, _ = runner.model(runner._gate_inputs("support"), experts, inv_norm)
    assert torch.equal(logits.argmax(-1), infra.support_labels(None))
    gates, _ = runner._query_gates()
    mean_gate = gates.mean(0)
    assert mean_gate[0] > mean_gate[1:].max()
    assert load_runner_class("sagmm_pe") is SAGMMPERunner
    assert all(split in ("support", "query") for _, split in infra.embedding_calls)


def test_toy_runner_is_deterministic_and_chunk_invariant(toy, monkeypatch):
    runner = toy.runner
    pred = runner.predict_queries()
    assert torch.allclose(pred.sum(-1), torch.ones(pred.size(0)), atol=1e-5)
    runner._query_pred = None
    monkeypatch.setattr(sagmm_trainer, "_CHUNK", 7)
    assert torch.allclose(runner.predict_queries(), pred, atol=1e-6)
    runner._query_pred = pred

    again = SAGMMPERunner(_toy_cfg(), AppSpec("toy", "node", 5, 42), _ToyInfra())
    again.fit()
    assert torch.equal(again.predict_queries(), pred) and again.best_epoch == runner.best_epoch


def test_pruning_ignores_raw_readout_scale_under_l2_norm(monkeypatch):
    """The EMA importance scores the mixed (L2-normalised) readouts, so rescaling one expert's raw readouts
    (by 2^20, exact in floating point and beyond the float16 range like real activations) changes nothing."""
    log = []

    class _Recording(ExpertPruner):
        def prune(self, expert_mask, **kwargs):
            removed = super().prune(expert_mask, **kwargs)
            log[-1].append((removed, self.importance.clone()))
            return removed

    monkeypatch.setattr(sagmm_trainer, "ExpertPruner", _Recording)
    runs = []
    for scale in (1.0, 2.0 ** 20):
        log.append([])
        infra = _ToyInfra()
        infra.readouts["e0"] = infra.readouts["e0"] * scale  # the informative expert
        runner = SAGMMPERunner(_toy_cfg(), AppSpec("toy", "node", 5, 42), infra)
        runner.fit()
        runs.append((runner.model.gate.expert_mask.clone(), runner.best_epoch, runner.predict_queries()))
    assert any(removed for removed, _ in log[0])  # the toy run prunes, so the decisions are exercised
    assert [r for r, _ in log[0]] == [r for r, _ in log[1]]
    for (_, a), (_, b) in zip(*log):
        assert torch.allclose(a, b, rtol=1e-5)
    assert torch.equal(runs[0][0], runs[1][0]) and runs[0][1] == runs[1][1]
    assert torch.allclose(runs[0][2], runs[1][2], atol=1e-5)


def test_task_loss_masks_missing_multilabel_entries():
    fake = SimpleNamespace(family=MULTILABEL)
    logits = torch.randn(4, 3, requires_grad=True)
    target = torch.tensor([[1.0, float("nan"), 0.0]] * 4)
    loss = SAGMMPERunner._task_loss(fake, logits, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.all(logits.grad[:, 1] == 0) and torch.all(logits.grad[:, 0] != 0)


# --------------------------------------------------------------------------- #
# Real RouterInfra on synthetic applications of every family
# --------------------------------------------------------------------------- #
_TARGETS = ("nodea:node", "linka:edge", "grapha:graph", "multia:graph", "rega:graph")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sagmm_pe")
    cfg = tiny_cfg(tmp, targets=_TARGETS, history_extra=(), budgets=(3,), seeds=(42,))
    cfg.save_results.output_dir = str(tmp / "results")
    b = cfg.moe.routergfm.baselines
    b.candidate_rule = "eligible"
    s = b.sagmm_pe
    s.epochs, s.prune_interval, s.lr = 12, 4, 0.01
    s.edge.prune_interval = 4
    s.graph.epochs, s.graph.prune_interval, s.graph.batch_size = 12, 4, 16
    infra = RouterInfra(cfg, SyntheticDataProvider())
    return SimpleNamespace(cfg=cfg, infra=infra, tmp=tmp)


@pytest.mark.parametrize("spec,metric", [
    ("nodea:node", "acc"), ("linka:edge", "auc"), ("grapha:graph", "acc"), ("multia:graph", "auc"), ("rega:graph", "mae"),
])
def test_runner_on_router_infra(env, spec, metric):
    infra = env.infra
    app = infra.application(spec, 3, 42)
    runner = SAGMMPERunner(env.cfg, app, infra)
    runner.fit()
    data = infra.data(app)
    pred = runner.predict_queries()
    assert pred.size(0) == data.query_pos.numel() and torch.isfinite(pred).all()
    assert len(runner.expert_ids) == len(infra.compatible_pool(app))
    assert math.isfinite(runner.best_metrics[f"test_{metric}"]) and math.isfinite(runner.best_metrics["test_risk"])
    if spec.startswith("linka"):
        assert pred.shape[1] == 2 and torch.allclose(pred.sum(-1), torch.ones(pred.size(0)), atol=1e-5)
    if spec.startswith("rega"):
        norm = infra.normalizer(app)
        assert torch.allclose(runner._support_target(), norm.transform(data.labels["support"]))
    if spec.startswith("grapha"):  # graph-level populations are fixed query batches: repeatable
        runner._query_pred = None
        assert torch.equal(runner.predict_queries(), pred)


def test_run_matched_baseline_appends_row(env):
    cfg = env.cfg.clone()
    b = cfg.moe.routergfm.baselines
    b.datasets, b.budgets = ["nodea:node"], [3]
    b.candidate_rule, b.candidate_pool = "random", 3
    assert matched_run.run_matched_baseline(cfg, "sagmm_pe", infra=env.infra) == 0
    with open(Path(cfg.save_results.output_dir) / "moe_sagmm_pe.tsv", newline="", encoding="utf-8") as fh:
        (row,) = list(csv.DictReader(fh, delimiter="\t"))
    assert math.isfinite(float(row["test_acc_mean"])) and float(row["final_num_experts_mean"]) <= 3
    assert row["moe.routergfm.baselines.candidate_rule"] == "random"
