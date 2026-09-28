"""Model Spider selection baseline: tokens, exact CLS-only attention, Plackett-Luce, training, deployment."""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.moe.routergfm.baselines.selection import run as selection_run
from src.moe.routergfm.baselines.selection.common import QueryAccessError, QueryGuard, selection_metrics
from src.moe.routergfm.baselines.selection.model_spider import (
    ModelSpiderRanker,
    ModelSpiderSelector,
    ModelSpiderTrainer,
)
from src.moe.routergfm.baselines.selection.model_spider.data import SpecificCenterCache, SpiderTask
from src.moe.routergfm.baselines.selection.model_spider.loss import plackett_luce_loss
from src.moe.routergfm.baselines.selection.model_spider.tokens import (
    pad_token_sets,
    partition_weights,
    weighted_centers,
)
from src.moe.routergfm.applications import derive_seed
from src.moe.routergfm.common import AppSpec, enumerate_applications
from src.moe.routergfm.history import generate_history
from src.moe.routergfm.infra import RouterInfra
from src.moe.routergfm.router.trainer import validation_groups
from tests.routergfm.fixtures import SyntheticDataProvider, tiny_cfg
from tests.routergfm.test_selection_harness import _LoggingProvider

NAN = float("nan")


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
def test_partition_weights():
    w = partition_weights(torch.tensor([2, 0, 2, -1]), "node_cls")
    assert torch.equal(w, torch.tensor([[0.0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 0]]))
    assert partition_weights(torch.tensor([[1], [0], [1]]), "link").shape == (3, 2)

    reg = torch.tensor([[3.0, 9.0], [1.0, 0.0], [2.0, 0.0], [2.0, 5.0], [5.0, 1.0]])
    w = partition_weights(reg, "regression", regression_bins=5)  # singleton bins; tie (2.0) by support order
    assert torch.equal(w.argmax(1), torch.tensor([3, 0, 1, 2, 4])) and torch.equal(w.sum(0), torch.ones(5))
    w = partition_weights(torch.tensor([[float(v)] for v in (5, 1, NAN, 4, 2, 3)]), "regression", regression_bins=2)
    assert w[2].sum() == 0 and torch.equal(w.sum(0), torch.tensor([3.0, 2.0]))  # 5 valid rows -> bins of 3 / 2
    assert torch.equal(w[[1, 4, 5]].argmax(1), torch.zeros(3, dtype=torch.long))

    ml = torch.tensor([[1.0, 0.0, NAN], [NAN, NAN, NAN], [1.0, 1.0, 0.0]])
    assert torch.equal(partition_weights(ml, "multilabel"), torch.tensor([[1.0, 1.0], [0.0, 0.0], [2.0, 1.0]]))


def test_weighted_centers_and_padding():
    feats = torch.tensor([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0], [0.0, 1.0]])
    weights = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [0, 0, 1.0], [0, 0, 3.0]])  # column 1 is empty
    centers = weighted_centers(feats, weights)
    assert torch.allclose(centers, torch.tensor([[2.0, 1.0], [5.0 / 4, 7.0 / 4]]))
    padded, mask = pad_token_sets([centers, feats[:1], feats[:3]])
    assert padded.shape == (3, 3, 2) and mask.tolist() == [[True, True, False], [True, False, False], [True] * 3]
    assert torch.equal(padded[1, 0], feats[0]) and torch.equal(padded[0, 2], torch.zeros(2))


# --------------------------------------------------------------------------- #
# Exact CLS-only attention vs. the official full-sequence block
# --------------------------------------------------------------------------- #
def _official_score(model: ModelSpiderRanker, tokens: torch.Tensor, pad: int) -> torch.Tensor:
    """Official ``MultiHeadAttention`` (+ ``mlp_head`` on position 0) over a zero-padded, key-masked sequence."""
    z = torch.cat([tokens, torch.zeros(pad, tokens.size(1))])
    key_mask = torch.arange(z.size(0)) < tokens.size(0)
    h, d = model.num_heads, model.token_dim
    q, k, v = (lin(z).view(z.size(0), h, d).permute(1, 0, 2) for lin in (model.w_q, model.w_k, model.w_v))
    attn = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
    attn = torch.softmax(attn.masked_fill(~key_mask[None, None, :], float("-inf")), dim=2)
    out = torch.bmm(attn, v).permute(1, 0, 2).reshape(z.size(0), h * d)
    return model.head(model.layer_norm(model.fc(out) + z)[0]).squeeze(-1)


@pytest.mark.parametrize(
    "mode,type_prompts,num_heads", list(itertools.product(("append", "replace"), (False, True), (1, 2)))
)
def test_cls_only_matches_full_attention(mode, type_prompts, num_heads):
    torch.manual_seed(0)
    model = ModelSpiderRanker(
        [f"e{i}" for i in range(5)], 6, [4, 7], token_dim=8, num_heads=num_heads, dropout=0.1,
        type_prompts=type_prompts, specific_token_mode=mode,
    ).eval()
    sizes = [3, 1, 2]
    general, gmask = pad_token_sets([torch.randn(c, 6) for c in sizes])
    expert_idx = torch.tensor([[0, 2, 4, 1], [3, 1, 0, 0], [4, 4, 2, 0]])
    expert_mask = torch.tensor([[True] * 4, [True, True, True, False], [True, True, True, True]])
    specific = {(0, 1): torch.randn(2, 4), (1, 0): torch.randn(3, 7), (2, 1): torch.randn(1, 4), (2, 3): torch.randn(2, 7)}
    with torch.no_grad():
        scores = model.score(general, gmask, expert_idx, expert_mask, specific)
        assert torch.isinf(scores[1, 3]) and scores[1, 3] < 0
        for b, j in zip(*expert_mask.nonzero(as_tuple=True)):
            b, j = int(b), int(j)
            e = int(expert_idx[b, j])
            gen = model.general_proj(general[b, : sizes[b]]) + (model.p_gen if type_prompts else 0)
            parts = [model.theta[e][None]]
            if (b, j) in specific:
                s = specific[(b, j)]
                de = str(s.size(1))
                spec = s @ model.spec_weight[de][e] + model.spec_bias[de][e] + (model.p_spec if type_prompts else 0)
                parts += ([gen] if mode == "append" else []) + [spec]
            else:
                parts.append(gen)
            ref = _official_score(model, torch.cat(parts), pad=2)
            assert torch.allclose(scores[b, j], ref, atol=1e-5), (b, j)


# --------------------------------------------------------------------------- #
# Plackett-Luce
# --------------------------------------------------------------------------- #
def _pl_explicit(scores, order):
    s = scores[order]
    return sum(torch.logsumexp(s[m:], 0) - s[m] for m in range(len(order)))


def test_plackett_luce_bruteforce():
    s = torch.tensor([0.3, -1.2, 2.0, 0.5])
    t = torch.tensor([0.4, 0.1, 0.9, 0.2])  # best (lowest loss) first: 1, 3, 0, 2
    loss = plackett_luce_loss(s[None], t[None], torch.ones(1, 4, dtype=torch.bool))
    assert loss == pytest.approx(float(_pl_explicit(s, [1, 3, 0, 2])), abs=1e-6)

    s6 = torch.tensor([0.3, 9.0, -1.2, 2.0, -7.0, 0.5])  # items 1 and 4 invalid: same as deleting them
    t6 = torch.tensor([0.4, 0.0, 0.1, 0.9, NAN, 0.2])
    valid = torch.tensor([[True, False, True, True, True, True]])  # item 4 has a NaN target
    assert plackett_luce_loss(s6[None], t6[None], valid) == pytest.approx(float(loss), abs=1e-6)

    ordered = -torch.tensor([0.4, 0.1, 0.9, 0.2]) * 10
    ones = torch.ones(1, 4, dtype=torch.bool)
    assert plackett_luce_loss(ordered[None], t[None], ones) < plackett_luce_loss(-ordered[None], t[None], ones)

    tied = torch.tensor([[0.2, 0.2]])
    pair = torch.tensor([[1.0, -1.0]])
    first = plackett_luce_loss(pair, tied, torch.ones(1, 2, dtype=torch.bool), torch.tensor([[0, 1]]))
    second = plackett_luce_loss(pair, tied, torch.ones(1, 2, dtype=torch.bool), torch.tensor([[1, 0]]))
    assert first == pytest.approx(float(_pl_explicit(pair[0], [0, 1])))
    assert second == pytest.approx(float(_pl_explicit(pair[0], [1, 0])))

    batch = torch.stack([s, torch.full((4,), float("-inf"))]).requires_grad_(True)  # task 2: no valid item
    mask = torch.tensor([[True] * 4, [False] * 4])
    out = plackett_luce_loss(batch, torch.stack([t, t]), mask)
    out.backward()
    assert float(out) == pytest.approx(float(loss), abs=1e-6) and bool(torch.isfinite(batch.grad).all())


# --------------------------------------------------------------------------- #
# Synthetic zoo: three latent domains, experts best on their own domain
# --------------------------------------------------------------------------- #
class _FakeSpiderInfra:
    """Historical applications (one group each) in 3 domains; any query-side access raises."""

    D_Z, D_E, N_SUPPORT = 6, 8, 12

    def __init__(self, n_history=45, n_test=9, extra_experts=(), target_history=False):
        g = torch.Generator().manual_seed(0)
        self.expert_ids = [f"x{i:02d}" for i in range(12)] + list(extra_experts)
        self.catalog = [SimpleNamespace(expert_id=e) for e in self.expert_ids]
        domain_mean = 3.0 * torch.randn(3, self.D_Z, generator=g)
        class_offset = torch.randn(3, self.D_Z, generator=g)
        self.history = [AppSpec(f"h{i:02d}", "node", 5, 0) for i in range(n_history)]
        self.tests = [AppSpec("target", "node", 10 + i, 42) for i in range(n_test)]
        if target_history:  # the target's own group must never be used
            self.history.insert(0, AppSpec("target", "node", 5, 0))
        self.domain, self.labels, self.z, self.mu = {}, {}, {}, {}
        for i, app in enumerate(self.history + self.tests):
            dom = i % 3
            self.domain[app.key] = dom
            y = torch.arange(self.N_SUPPORT) % 3
            self.labels[app.key] = y
            self.z[app.key] = domain_mean[dom] + 0.5 * class_offset[y] + 0.3 * torch.randn(self.N_SUPPORT, self.D_Z, generator=g)
            for j, e in enumerate(self.expert_ids[:12]):
                self.mu[(app.key, e)] = 0.1 + 0.3 * (j // 4 != dom) + 0.02 * (j % 4) + 0.005 * float(torch.rand(1, generator=g))
        self.calls = []

    def risk(self, app):
        return {e: self.mu[(app.key, e)] for e in self.expert_ids[:12]}

    def historical_applications(self, target):
        return list(self.history)

    def task_family(self, app):
        return "node_cls"

    def compatible_pool(self, app):
        return list(self.expert_ids)

    def support_labels(self, app):
        return self.labels[app.key]

    def descriptors(self, app, split):
        assert split == "support", f"query-side descriptors of {app.key}"
        return self.z[app.key]

    def historical_mu(self, app, expert_id):
        assert app.group != "target", "target-group history read"
        value = self.mu.get((app.key, expert_id))
        return (NAN, 0) if value is None else (value, 5)

    def embeddings(self, app, expert_id, split):
        assert split == "support", f"query-side embeddings of {app.key}"
        self.calls.append((app.key, expert_id))
        g = torch.Generator().manual_seed(derive_seed(app.key, expert_id))
        return torch.randn(self.N_SUPPORT, self.D_E, generator=g)

    def __getattr__(self, name):
        raise AssertionError(f"Model Spider accessed infra.{name}")


def _spider_cfg(root: Path, **overrides):
    cfg = tiny_cfg(Path("/nonexistent"), write_checkpoints=False)
    cfg.moe.routergfm.output_root = str(root / "routergfm")
    cfg.moe.routergfm.baselines.topk = 3
    cfg.moe.routergfm.router.val_datasets = []
    cfg.moe.routergfm.router.num_val_datasets = 9
    m = cfg.moe.routergfm.baselines.model_spider
    m.token_dim, m.epochs, m.batch_size, m.lr = 32, 30, 4, 3e-3
    m.train_specific_max, m.rerank_topk_grid = 3, [0, 3]
    for key, value in overrides.items():
        setattr(m, key, value)
    return cfg


@pytest.fixture(scope="module")
def zoo(tmp_path_factory):
    root = tmp_path_factory.mktemp("spider")
    cfg = _spider_cfg(root)
    infra = _FakeSpiderInfra(target_history=True)
    guard = QueryGuard(infra)
    selector = ModelSpiderSelector(cfg, guard)
    outcomes = []
    for app in infra.tests:
        guard.target = app
        outcomes.append(selector.rank(app))
    guard.target = None
    target_keys = {a.key for a in infra.tests}
    target_calls = [c for c in infra.calls if c[0] in target_keys]
    return SimpleNamespace(cfg=cfg, infra=infra, selector=selector, outcomes=outcomes, root=root, target_calls=target_calls)


def test_training_learns_synthetic_zoo(zoo):
    hits = [selection_metrics(o.ranking, zoo.infra.risk(o.app), 3)["hit_at_k"] for o in zoo.outcomes]
    assert sum(hits) / len(hits) >= 0.8  # a random team of 3 hits ~0.25
    for o in zoo.outcomes:
        scores = [s for _, s in o.ranking]
        assert scores == sorted(scores, reverse=True) and o.team == [e for e, _ in o.ranking[:3]]
        assert o.num_target_executions == o.extras["rerank_topk"] and o.variants["k0"] == [e for e, _ in o.extras["coarse_ranking"][:3]]
        assert o.extras["num_untrained_tokens"] == 0
    info = zoo.outcomes[0].extras
    assert info["n_train_tasks"] == 36 and info["n_val_tasks"] == 9 and 1 <= info["best_epoch"] <= 30
    assert len(zoo.selector._trainers) == 1  # one ranker for the target group and seed, reused across budgets


def test_no_leakage_and_grouped_exclusion(zoo):
    train, val = zoo.selector.split(zoo.infra.tests[0])
    assert all(b.group != "target" for b in train + val)
    assert [b.dataset for b in val] == [f"h{i:02d}" for i in range(9)]  # router convention: first groups
    # Target support executions are exactly the re-ranked experts (each once, support split only).
    assert len(zoo.target_calls) == len(set(zoo.target_calls)) == sum(o.num_target_executions for o in zoo.outcomes)
    for o in zoo.outcomes:
        top = [e for e, _ in o.extras["coarse_ranking"][: o.num_target_executions]]
        assert sorted(e for k, e in zoo.target_calls if k == o.app.key) == sorted(top)
    guard = QueryGuard(zoo.infra)
    guard.target = zoo.infra.tests[0]
    with pytest.raises(QueryAccessError):
        guard.historical_mu(AppSpec("target", "node", 5, 0), "x00")


def test_untrained_tokens_reported(tmp_path):
    cfg = _spider_cfg(tmp_path, epochs=2)
    infra = _FakeSpiderInfra(n_history=12, extra_experts=("new",))
    cfg.moe.routergfm.router.num_val_datasets = 3
    out = ModelSpiderSelector(cfg, infra).rank(infra.tests[0])
    assert "new" in [e for e, _ in out.ranking] and len(out.ranking) == 13  # still scored, never dropped
    assert out.extras["num_untrained_tokens"] == 1 and out.extras["n_train_tasks"] == 9


def _world_trainer(cfg, infra, tmp_path, seed=3):
    centers = SpecificCenterCache(infra, tmp_path / "cache", 5)
    return ModelSpiderTrainer(cfg, infra, infra.expert_ids, infra.D_Z, [infra.D_E], seed, centers=centers)


def test_rerank_only_touches_topk(tmp_path):
    cfg = _spider_cfg(tmp_path)
    infra = _FakeSpiderInfra(n_history=0)
    app = infra.tests[0]
    task = SpiderTask(app, torch.randn(3, infra.D_Z), list(infra.expert_ids))
    trainer = _world_trainer(cfg, infra, tmp_path)
    final, coarse, n = trainer.rank(task, 0)
    assert n == 0 and infra.calls == [] and final == coarse
    final, coarse2, n = trainer.rank(task, 3)
    top = [e for e, _ in coarse[:3]]
    assert n == 3 and coarse2 == coarse and sorted(infra.calls) == sorted((app.key, e) for e in top)
    coarse_s, final_s = dict(coarse), dict(final)
    assert all(final_s[e] == coarse_s[e] for e in infra.expert_ids if e not in top)
    assert any(final_s[e] != coarse_s[e] for e in top)
    trainer.rank(task, 3)
    assert len(infra.calls) == 3  # specific centres are cached per data key


def test_non_finite_specific_centres_are_recomputed(tmp_path):
    infra = _FakeSpiderInfra(n_history=0)
    app, eid = infra.tests[0], infra.expert_ids[0]
    good = SpecificCenterCache(infra, tmp_path / "cache", 5).get(app, eid)
    cache = SpecificCenterCache(infra, tmp_path / "cache", 5)
    torch.save({eid: torch.full_like(good, float("nan")).half()}, cache._file(app.data_key))  # overflowed readouts
    assert torch.equal(cache.get(app, eid), good)


def test_checkpoint_roundtrip(zoo, tmp_path, monkeypatch):
    trainer = next(iter(zoo.selector._trainers.values()))
    app = zoo.infra.tests[1]
    task = SpiderTask(app, torch.randn(3, zoo.infra.D_Z), list(zoo.infra.expert_ids))
    before = trainer.rank(task, 3)
    trainer.save(tmp_path / "ms.pt")
    loaded = ModelSpiderTrainer.load(tmp_path / "ms.pt", zoo.cfg, zoo.infra)
    assert loaded.rank(task, 3) == before and loaded.info == trainer.info and loaded.trained_ids == trainer.trained_ids
    assert torch.equal(loaded.standardizer.mean, trainer.standardizer.mean)

    # A fresh selector reuses the saved checkpoint (no training) and reproduces the outcome.
    monkeypatch.setattr(ModelSpiderTrainer, "fit", lambda *a, **k: pytest.fail("retrained despite checkpoint"))
    again = ModelSpiderSelector(zoo.cfg, zoo.infra).rank(zoo.outcomes[0].app)
    assert again.ranking == zoo.outcomes[0].ranking and again.extras["rerank_topk"] == zoo.outcomes[0].extras["rerank_topk"]
    assert len(list((zoo.root / "routergfm" / "model_spider" / "checkpoints").glob("heldout-target_seed42_*.pt"))) == 1


# --------------------------------------------------------------------------- #
# End to end through the selection harness on the synthetic RouterGFM setup
# --------------------------------------------------------------------------- #
def test_run_selection_baseline_model_spider(tmp_path):
    cfg = tiny_cfg(
        tmp_path,
        targets=("nodea:node",),
        history_extra=("nodeb:node", "nodec:node", "srca:node", "grapha:graph", "linka:edge"),
        budgets=(3,),
        seeds=(42,),
    )
    cfg.save_results.output_dir = str(tmp_path / "results")
    m = cfg.moe.routergfm.baselines.model_spider
    m.token_dim, m.epochs, m.batch_size, m.rerank_topk_grid = 8, 3, 2, [0, 2]
    cfg.moe.routergfm.baselines.method = "model_spider"
    provider = SyntheticDataProvider()
    generate_history(cfg, provider, apps=enumerate_applications(cfg.moe.routergfm))
    logging = _LoggingProvider(provider)
    infra = RouterInfra(cfg, logging)
    assert selection_run.run_selection_baseline(cfg, infra=infra) == 0

    assert {split for _, split in logging.log} == {"support"}  # only support labels, never query/diag
    (path,) = (tmp_path / "baselines" / "model_spider").rglob("*.json")
    payload = json.loads(path.read_text())
    outcome, metrics = payload["outcome"], payload["metrics"]
    app = infra.application("nodea:node", 3, 42)
    assert sorted(e for e, _ in outcome["ranking"]) == sorted(infra.compatible_pool(app))
    assert outcome["num_target_executions"] == outcome["extras"]["rerank_topk"]
    assert {"test_hit_at_2", "test_regret_at_2", "test_hit_at_2_k0", "test_regret_at_2_k0"} <= set(metrics)
    ckpts = list((tmp_path / "routergfm" / "model_spider" / "checkpoints").glob("heldout-nodea_seed42_*.pt"))
    assert len(ckpts) == 1
    specific_dims = torch.load(ckpts[0])["specific_dims"]
    assert specific_dims == [8, 32]  # node/graph readouts (out_dim 8) and the link readout (4 x 8)
    with open(tmp_path / "results" / "moe_routergfm_selection.tsv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert rows[0]["moe.routergfm.baselines.method"] == "model_spider" and "test_hit_at_2_k0_mean" in rows[0]

    # Same validation groups as the router of this target; training covers every other group.
    history = [a for a in enumerate_applications(cfg.moe.routergfm) if a.group != "nodea" and infra.store.expert_ids(a.data_key)]
    families = {a.key: infra.task_family(a) for a in history}
    router_val = validation_groups(cfg.moe.routergfm.router, "nodea", history, families, {"node_cls"})
    train, val = ModelSpiderSelector(cfg, infra).split(app)
    assert sorted({b.group for b in val}) == sorted(router_val) == ["nodeb"]
    assert {b.group for b in train} == {"nodec", "srca", "grapha", "linka"}
