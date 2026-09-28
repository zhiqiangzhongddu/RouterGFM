"""KDEM / PPEM: parameter merging, PPEM pull, competence routing, team selection, and the matched runner."""

from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from src.config import cfg as base_cfg
from src.model import build_encoder_from_cfg
from src.moe.routergfm.baselines import load_runner_class
from src.moe.routergfm.baselines import run as matched_run
from src.moe.routergfm.baselines.kdem_ppem import KDEMPPEMRunner
from src.moe.routergfm.baselines.kdem_ppem import trainer as kp_trainer
from src.moe.routergfm.baselines.kdem_ppem.merge import ema_pull_, merged_state, module_tensors, resolve_ema_beta
from src.moe.routergfm.baselines.kdem_ppem.model import MergedExpertModel
from src.moe.routergfm.baselines.kdem_ppem.selection import (
    assert_state_compatible,
    compat_key,
    competence_batches,
    competence_score,
    select_merge_team,
    team_candidates,
)
from src.moe.routergfm.common import LINK, NODE_CLS
from src.moe.routergfm.infra import RouterInfra
from tests.routergfm.fixtures import FEATURE_DIM, SyntheticDataProvider, make_synthetic_dataset, tiny_cfg

NODE, EDGE = "nodea:node", "linka:edge"


def _encoder(arch="gcn", *, hidden=8, seed=0, **model):
    cfg = base_cfg.clone()
    cfg.model.name, cfg.model.in_dim, cfg.model.hidden_dim, cfg.model.out_dim = arch, FEATURE_DIM, hidden, 8
    cfg.model.num_layers, cfg.model.gat.heads = 2, 2
    for key, value in model.items():
        setattr(cfg.model, key, value)
    torch.manual_seed(seed)
    return build_encoder_from_cfg(cfg, FEATURE_DIM), cfg


def _node_batch(n=6):
    return Batch.from_data_list(make_synthetic_dataset("nodea", "node", NODE_CLS, n).graphs)


def _flat(module):
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


# --------------------------------------------------------------------------- #
# Merging (Eq. 8)
# --------------------------------------------------------------------------- #
def test_merged_state_weights_float_tensors_and_copies_int_buffers():
    (a, _), (b, _) = _encoder(seed=0), _encoder(seed=1)
    with torch.no_grad():
        b.bns[0].running_mean.normal_()
    b.bns[0].num_batches_tracked.fill_(7)
    alpha = torch.tensor([0.7, 0.3])
    sa, sb = module_tensors(a), module_tensors(b)
    merged = merged_state([sa, sb], alpha)
    assert merged.keys() == sa.keys()
    for name, ref in sa.items():
        if ref.is_floating_point():
            assert torch.allclose(merged[name], 0.7 * ref + 0.3 * sb[name])
        else:
            assert torch.equal(merged[name], ref)
    assert int(merged["bns.0.num_batches_tracked"]) == 0
    with pytest.raises(ValueError):
        merged_state([sa, sb], torch.ones(3) / 3)


def test_state_compatibility_and_compat_key():
    (gcn, gcn_cfg), (gcn2, _) = _encoder("gcn"), _encoder("gcn", seed=3)
    (gat, gat_cfg), (wide, _) = _encoder("gat"), _encoder("gcn", hidden=12)
    assert_state_compatible([gcn.state_dict(), gcn2.state_dict()])
    with pytest.raises(ValueError, match="key sets"):
        assert_state_compatible([gcn.state_dict(), gat.state_dict()])
    with pytest.raises(ValueError, match="shape"):
        assert_state_compatible([gcn.state_dict(), wide.state_dict()])
    assert compat_key(gcn_cfg) != compat_key(gat_cfg)
    other_dropout = gcn_cfg.clone()
    other_dropout.model.dropout = 0.1
    assert compat_key(other_dropout) == compat_key(gcn_cfg)  # training-only knob
    other_heads = gat_cfg.clone()
    other_heads.model.gat.heads = 4
    assert compat_key(other_heads) != compat_key(gat_cfg)


def test_functional_merged_forward_matches_a_merged_module():
    (a, _), (b, _), (fresh, _) = _encoder(seed=0), _encoder(seed=1), _encoder(seed=5)
    alpha = torch.tensor([0.6, 0.4])
    model = MergedExpertModel([a, b], alpha).eval()
    batch = _node_batch()
    with torch.no_grad():
        out = model(batch)[0]
        assert torch.allclose(out, model.merged_module()(batch)[0], atol=1e-6)
        fresh.load_state_dict(merged_state([a.state_dict(), b.state_dict()], alpha))
        assert torch.allclose(out, fresh.eval()(batch)[0], atol=1e-6)


def test_gradients_reach_experts_scaled_by_alpha_and_template_is_not_trained():
    (a, _), (b, _) = _encoder(seed=0), _encoder(seed=1)
    model = MergedExpertModel([a, b], torch.tensor([0.7, 0.3])).eval()
    batch = _node_batch()
    model(batch)[0].pow(2).sum().backward()
    merged = model.merged_module().requires_grad_(True)
    merged(batch)[0].pow(2).sum().backward()
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    for name, param in merged.named_parameters():
        if param.grad is None:  # unused (e.g. BatchNorm without use_batchnorm)
            assert pa[name].grad is None
            continue
        assert torch.allclose(pa[name].grad, 0.7 * param.grad, atol=1e-6)
        assert torch.allclose(pb[name].grad, 0.3 * param.grad, atol=1e-6)
    registered = {id(p) for p in model.parameters()}
    assert not registered & {id(p) for p in model._template[0].parameters()}
    assert all(name.startswith("experts.") for name, _ in model.named_parameters())


def test_merge_approaches_the_ensemble_as_experts_get_close():
    a, _ = _encoder(seed=0)
    gen = torch.Generator().manual_seed(0)
    noise = {name: torch.randn(p.shape, generator=gen) for name, p in a.named_parameters()}
    alpha, batch, gaps = torch.tensor([0.5, 0.5]), _node_batch(), []
    for eps in (1e-1, 1e-2, 1e-3):
        b = copy.deepcopy(a)
        with torch.no_grad():
            for name, param in b.named_parameters():
                param.add_(eps * noise[name])
            model = MergedExpertModel([a, b], alpha).eval()
            gaps.append(float((model(batch)[0] - model.ensemble_node_repr(batch)).norm()))
    assert gaps[0] > gaps[1] > gaps[2]


def test_kd_loss_teacher_detachment():
    (a, _), (b, _) = _encoder(seed=0), _encoder(seed=1)
    model = MergedExpertModel([a, b], torch.tensor([0.5, 0.5])).eval()
    batch = _node_batch()

    def grads(fn):
        model.zero_grad()
        fn().backward()
        return torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])

    detached = grads(lambda: kp_trainer.kd_loss(model, batch, model(batch)[0], detach=True))
    manual = grads(lambda: F.mse_loss(model(batch)[0], model.ensemble_node_repr(batch).detach()))
    attached = grads(lambda: kp_trainer.kd_loss(model, batch, model(batch)[0], detach=False))
    assert torch.allclose(detached, manual, atol=1e-7)
    assert not torch.allclose(detached, attached, atol=1e-7)


# --------------------------------------------------------------------------- #
# PPEM pull (Eq. 13)
# --------------------------------------------------------------------------- #
def test_ema_pull_contracts_team_keeps_merge_and_leaves_others():
    (a, _), (b, _), (c, _) = _encoder(seed=0), _encoder(seed=1), _encoder(seed=2)
    alpha, beta = torch.tensor([0.7, 0.3]), 0.9
    diff, bar, c_before = _flat(a) - _flat(b), 0.7 * _flat(a) + 0.3 * _flat(b), _flat(c)
    buffer_before = a.bns[0].running_var.clone()
    ema_pull_([a, b], alpha, beta)
    assert torch.allclose(_flat(a) - _flat(b), beta * diff, atol=1e-6)
    assert torch.allclose(0.7 * _flat(a) + 0.3 * _flat(b), bar, atol=1e-6)
    assert torch.equal(_flat(c), c_before)
    assert torch.equal(a.bns[0].running_var, buffer_before)  # parameters only


def test_resolve_ema_beta():
    beta = resolve_ema_beta(target_retention=0.0025, fallback_beta=0.999, total_steps=1000, period=10)
    assert beta ** 100 == pytest.approx(0.0025)
    assert resolve_ema_beta(target_retention=0.0025, fallback_beta=0.999, total_steps=3, period=10) == pytest.approx(0.0025)
    assert resolve_ema_beta(target_retention=0.0, fallback_beta=0.999, total_steps=1000, period=10) == 0.999


# --------------------------------------------------------------------------- #
# Competence score (Eq. 5) and team selection (Eq. 7)
# --------------------------------------------------------------------------- #
class _ReprEncoder(nn.Module):
    """Returns a stored per-node representation (``data.<attr>``)."""

    def __init__(self, attr):
        super().__init__()
        self.attr = attr

    def forward(self, data):
        return getattr(data, self.attr), None


def _two_clique_graphs(num=4, seed=0):
    gen = torch.Generator().manual_seed(seed)
    graphs = []
    for i in range(num):
        n = 4
        clique = torch.tensor([[u, v] for u in range(n) for v in range(n) if u != v]).t()
        edge_index = torch.cat([clique, clique + n], dim=1)
        comp = torch.arange(2 * n) // n
        graphs.append(
            Data(
                x=torch.randn(2 * n, FEATURE_DIM, generator=gen),
                edge_index=edge_index,
                y=torch.tensor(i % 2),
                good=3.0 * F.one_hot(comp, 2).float(),
                rand=torch.randn(2 * n, 2, generator=gen),
            )
        )
    return graphs


def test_competence_prefers_edge_aligned_embeddings_and_ignores_labels():
    graphs = _two_clique_graphs()
    batches = competence_batches(graphs, max_triplets=10_000, seed=0, batch_size=3)
    good = competence_score(_ReprEncoder("good"), batches)
    rand = competence_score(_ReprEncoder("rand"), batches)
    assert 0.0 < rand < good < 1.0 and good > 0.7
    relabeled = [g.clone() for g in graphs]
    for g in relabeled:
        g.y = torch.tensor(float("nan"))
    again = competence_batches(relabeled, max_triplets=10_000, seed=0, batch_size=3)
    assert competence_score(_ReprEncoder("good"), again) == good


def test_competence_triplets_on_link_subgraphs_exclude_the_target_edge():
    graphs = make_synthetic_dataset("linka", "edge", LINK, 30).graphs
    batches = competence_batches(graphs, max_triplets=10**6, seed=0, batch_size=7)
    total = 0
    for batch, anchor, positive, negative in batches:
        edges = set(map(tuple, batch.edge_index.t().tolist()))
        targets = {(int(u), int(v)) for u, v in batch.edge_label_index.t()} | {
            (int(v), int(u)) for u, v in batch.edge_label_index.t()
        }
        pairs = set(zip(anchor.tolist(), positive.tolist()))
        assert pairs <= edges and not pairs & targets
        assert torch.equal(batch.batch[negative], batch.batch[anchor])
        assert bool((anchor != positive).all())
        total += anchor.numel()
    expected = sum(int((g.edge_index[0] < g.edge_index[1]).sum()) for g in graphs)  # symmetric, no self-loops
    assert total == expected


def test_competence_triplet_cap_determinism_and_edgeless_graphs():
    graphs = _two_clique_graphs(num=6)
    first = competence_batches(graphs, max_triplets=10, seed=1, batch_size=2)
    second = competence_batches(graphs, max_triplets=10, seed=1, batch_size=2)
    assert sum(b[1].numel() for b in first) == 10
    for x, y in zip(first, second):
        assert all(torch.equal(u, v) for u, v in zip(x[1:], y[1:]))
    edgeless = [Data(x=torch.randn(3, FEATURE_DIM), edge_index=torch.empty(2, 0, dtype=torch.long)) for _ in range(3)]
    assert competence_batches(edgeless, max_triplets=10, seed=0, batch_size=2) == []
    assert competence_score(_ReprEncoder("x"), []) == 0.5


def test_select_merge_team_and_candidates(capsys):
    scores = {"gcn_a": 0.91, "gat_a": 0.90, "gcn_b": 0.60, "gcn_c": 0.80, "gat_b": 0.85, "gcn_d": 0.80}
    keys = {e: ("gcn", 16) if e.startswith("gcn") else ("gat", 16) for e in scores}
    team = select_merge_team(scores, keys, 3)
    assert team.expert_ids == ["gcn_a", "gcn_c", "gcn_d"] and team.compat_key == ("gcn", 16)
    assert torch.allclose(team.alpha, torch.softmax(torch.tensor([0.91, 0.80, 0.80]), 0))
    assert float(team.alpha.sum()) == pytest.approx(1.0) and team.all_scores == scores
    assert select_merge_team(scores, keys, 10).expert_ids == ["gcn_a", "gcn_c", "gcn_d", "gcn_b"]
    assert "exceeds the compatible group" in capsys.readouterr().out

    arch = {e: e.split("_")[0] for e in scores}
    assert team_candidates(list(scores), arch, "top1_arch", "") == list(scores)
    assert team_candidates(list(scores), arch, "fixed", "gat") == ["gat_a", "gat_b"]
    for policy, fixed in (("fixed", ""), ("fixed", "gin"), ("best_group", "")):
        with pytest.raises(ValueError):
            team_candidates(list(scores), arch, policy, fixed)


# --------------------------------------------------------------------------- #
# Matched runner (synthetic applications, tiny checkpoints)
# --------------------------------------------------------------------------- #
class _LabelLog(dict):
    def __init__(self, labels, log):
        super().__init__(labels)
        self.log = log

    def __getitem__(self, key):
        self.log.append(key)
        return super().__getitem__(key)


class _LoggingProvider:
    def __init__(self, inner):
        self.inner, self.log = inner, []

    def load(self, app):
        data = self.inner.load(app)
        data.labels = _LabelLog(data.labels, self.log)
        return data


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("kdem_ppem")
    cfg = tiny_cfg(tmp, targets=(NODE, EDGE), history_extra=(), budgets=(3,), archs=("gcn", "gin", "gat"))
    cfg.save_results.output_dir = str(tmp / "results")
    b = cfg.moe.routergfm.baselines.kdem_ppem
    b.epochs, b.early_stopping, b.batch_size = 3, 5, 8
    b.kd.period, b.ema.period = 2, 2
    return SimpleNamespace(cfg=cfg, tmp=tmp)


def _runner(env, method, spec=NODE, provider=None, **overrides):
    cfg = env.cfg.clone()
    cfg.moe.routergfm.baselines.method = method
    for key, value in overrides.items():
        node = cfg.moe.routergfm.baselines.kdem_ppem
        *parents, leaf = key.split(".")
        for parent in parents:
            node = node[parent]
        node[leaf] = value
    infra = RouterInfra(cfg, provider or SyntheticDataProvider(), device="cpu")
    return KDEMPPEMRunner(cfg, infra.application(spec, 3, 42), infra)


@pytest.mark.parametrize("spec,metric", [(NODE, "test_acc"), (EDGE, "test_auc")])
def test_runner_fits_both_variants_on_one_team(env, spec, metric):
    runs = {}
    for method in ("kdem", "ppem"):
        runner = _runner(env, method, spec)
        metrics = runner.fit()
        assert math.isfinite(metrics[metric]) and math.isfinite(metrics["test_risk"])
        data = runner.infra.data(runner.app)
        pred = runner.predict_queries()
        assert pred.shape == (data.query_pos.numel(), data.num_classes)
        assert torch.allclose(pred.sum(-1), torch.ones(pred.size(0)), atol=1e-5)
        team = runner.team
        archs = {runner.infra.catalog[runner.infra.expert_index[e]].architecture for e in team.expert_ids}
        assert len(team.expert_ids) == 3 and archs == {team.compat_key[0]}
        assert set(team.all_scores) == set(runner.infra.compatible_pool(runner.app))
        log = json.loads((Path(env.cfg.moe.routergfm.baselines.output_dir) / method / "logs" / f"{runner.app.key}.json").read_text())
        assert log["team"]["expert_ids"] == team.expert_ids and log["metrics"].keys() == metrics.keys()
        assert sum(log["team"]["alpha"]) == pytest.approx(1.0)
        runs[method] = runner
    assert runs["kdem"].team.expert_ids == runs["ppem"].team.expert_ids  # routing is variant-independent
    assert runs["kdem"].beta is None and 0.0 < runs["ppem"].beta < 1.0


def test_runner_is_deterministic_and_rejects_unknown_variants(env):
    first, second = _runner(env, "kdem"), _runner(env, "kdem")
    first.fit(), second.fit()
    assert torch.equal(first.predict_queries(), second.predict_queries())
    with pytest.raises(ValueError, match="baselines.method"):
        _runner(env, "sagmm_pe")


def test_fixed_group_policy_restricts_the_team(env):
    runner = _runner(env, "ppem", group_policy="fixed", fixed_arch="gat")
    runner.fit()
    assert runner.team.compat_key[0] == "gat"
    assert {runner.infra.catalog[runner.infra.expert_index[e]].architecture for e in runner.team.all_scores} == {"gat"}


def test_kd_and_ema_schedules_follow_global_steps(env, monkeypatch):
    events = {"forward": 0, "kd": [], "ema": []}
    forward, ensemble = MergedExpertModel.forward, MergedExpertModel.ensemble_node_repr

    def spy_forward(self, data):
        events["forward"] += 1
        return forward(self, data)

    def spy_ensemble(self, data):
        events["kd"].append(events["forward"])
        return ensemble(self, data)

    def spy_ema(experts, alpha, beta):
        events["ema"].append((events["forward"], beta))
        return ema_pull_(experts, alpha, beta)

    monkeypatch.setattr(MergedExpertModel, "forward", spy_forward)
    monkeypatch.setattr(MergedExpertModel, "ensemble_node_repr", spy_ensemble)
    monkeypatch.setattr(kp_trainer, "ema_pull_", spy_ema)
    common = {"epochs": 7, "early_stopping": 100, "batch_size": 64}  # one step per epoch
    _runner(env, "kdem", **common, **{"kd.period": 3}).fit()
    assert events["forward"] == 7 and events["kd"] == [3, 6] and events["ema"] == []

    events.update(forward=0, kd=[])
    runner = _runner(env, "ppem", **common, **{"ema.period": 2})
    runner.fit()
    assert events["kd"] == [] and [step for step, _ in events["ema"]] == [2, 4, 6]
    assert runner.beta == pytest.approx(0.0025 ** (1 / 3)) and all(b == runner.beta for _, b in events["ema"])
    assert [h["epoch"] for h in runner.history] == list(range(1, 8))


def test_query_labels_are_read_only_by_evaluation(env, monkeypatch):
    provider = _LoggingProvider(SyntheticDataProvider())
    runner = _runner(env, "kdem", provider=provider)
    seen = []
    evaluate_outputs = type(runner.infra).evaluate_outputs

    def spy(self, app, pred):
        seen.append(list(provider.log))
        return evaluate_outputs(self, app, pred)

    monkeypatch.setattr(type(runner.infra), "evaluate_outputs", spy)
    runner.fit()
    assert seen and set(seen[0]) == {"support"}


def test_matched_harness_dispatches_to_the_runner(env, tmp_path):
    assert load_runner_class("kdem") is load_runner_class("ppem") is KDEMPPEMRunner
    cfg = env.cfg.clone()
    cfg.moe.routergfm.baselines.output_dir = str(tmp_path / "baselines")
    cfg.save_results.output_dir = str(tmp_path / "results")
    infra = RouterInfra(cfg, SyntheticDataProvider(), device="cpu")
    assert matched_run.run_matched_baseline(cfg, "ppem", infra=infra) == 0
    with open(tmp_path / "results" / "moe_ppem.tsv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert [json.loads(r["moe.routergfm.baselines.datasets"]) for r in rows] == [[NODE], [EDGE]]
    assert all(r["moe.routergfm.baselines.method"] == "ppem" for r in rows)
    assert math.isfinite(float(rows[0]["test_acc_mean"])) and math.isfinite(float(rows[1]["test_auc_mean"]))
