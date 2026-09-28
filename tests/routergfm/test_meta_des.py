"""META-DES matched-pool baseline: meta-features, lambda, selection, streaming, and harness wiring."""

from __future__ import annotations

import csv
from types import SimpleNamespace

import pytest
import torch

from src.moe.routergfm.baselines import load_runner_class
from src.moe.routergfm.baselines import run as matched_run
from src.moe.routergfm.baselines.meta_des import METADESRunner
from src.moe.routergfm.baselines.meta_des.meta_features import (
    build_meta_training_set,
    compute_meta_features,
    knn_indices,
    pool_consensus,
)
from src.moe.routergfm.baselines.meta_des.runner import select_competent, vote_with_tiebreak
from src.moe.routergfm.baselines.meta_des.selector import fit_meta_selector
from src.moe.routergfm.common import AppSpec, enumerate_applications
from src.moe.routergfm.history import generate_history
from src.moe.routergfm.infra import RouterInfra
from tests.routergfm.fixtures import SyntheticDataProvider, tiny_cfg

APP = AppSpec("toy", "node", 10, 42)


def _base_cfg(**meta):
    from src.config import cfg as base_cfg

    cfg = base_cfg.clone()
    cfg.moe.routergfm.baselines.candidate_rule = "eligible"
    for key, value in meta.items():
        setattr(cfg.moe.routergfm.baselines.meta_des, key, value)
    return cfg


# --------------------------------------------------------------------------- #
# Meta-features and the meta-training set
# --------------------------------------------------------------------------- #
def test_meta_feature_arithmetic():
    correct = torch.tensor([[1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [0, 1]], dtype=torch.bool)
    post_true = torch.tensor([[0.9, 0.2], [0.8, 0.7], [0.3, 0.6], [0.1, 0.2], [0.7, 0.3], [0.2, 0.9]])
    conf = torch.tensor([[0.6, 0.8], [0.5, 0.9]])
    theta = torch.tensor([[1, 2, 4], [0, 3, 5]])
    phi = torch.tensor([[2, 5], [1, 0]])
    feats = compute_meta_features(correct, post_true, conf, theta, phi)
    expected = torch.tensor([
        [1, 0, 1, 0.8, 0.3, 0.7, 2 / 3, 0, 0, 0.6],  # (sample 0, expert 0)
        [1, 1, 0, 0.7, 0.6, 0.3, 2 / 3, 1, 1, 0.8],  # (sample 0, expert 1)
        [1, 0, 0, 0.9, 0.1, 0.2, 1 / 3, 1, 1, 0.5],  # (sample 1, expert 0)
        [0, 0, 1, 0.2, 0.2, 0.9, 1 / 3, 1, 0, 0.9],  # (sample 1, expert 1)
    ])
    assert feats.shape == (4, 2 * 3 + 2 + 2)
    torch.testing.assert_close(feats, expected)


def test_knn_leave_one_out_by_index_with_duplicates():
    z = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [5.0, 5.0], [5.0, 5.0]])
    idx = knn_indices(z, z, 1, exclude=torch.arange(6))
    assert idx.view(-1).tolist() == [1, 0, 3, 2, 5, 4]  # the twin, never the sample itself

    oof = torch.softmax(z.repeat(1, 2).view(6, 2, 2), dim=-1)  # duplicated output profiles too
    meta = build_meta_training_set(z, oof, torch.tensor([0, 1, 0, 1, 0, 1]), k=3, kp=2, hc=1.1)
    assert meta.rows.tolist() == list(range(6)) and not meta.fallback
    for i, row in enumerate(meta.rows.tolist()):
        assert row not in meta.theta[i].tolist() and row not in meta.phi[i].tolist()
        assert meta.theta[i, 0].item() == row ^ 1  # duplicate descriptor is the nearest neighbour
    assert meta.X.shape == (6 * 2, 2 * 3 + 2 + 2) and meta.y.shape == (12,)


def _one_hot_posteriors(labels, num_classes=3):
    return torch.nn.functional.one_hot(labels, num_classes).float() * 0.8 + 0.1


def test_consensus_filter_and_fallbacks():
    H = pool_consensus(torch.tensor([[0, 0, 0, 1], [0, 1, 2, 1], [2, 2, 2, 2]]), 3)
    torch.testing.assert_close(H, torch.tensor([0.75, 0.5, 1.0]))

    z = torch.randn(6, 2, generator=torch.Generator().manual_seed(0))
    y = torch.tensor([0, 1, 2, 0, 2, 1])
    labels = torch.tensor([[0, 0, 1], [1, 1, 1], [2, 0, 1], [0, 0, 0], [1, 2, 2], [2, 2, 2]])
    meta = build_meta_training_set(z, _one_hot_posteriors(labels), y, k=2, kp=2, hc=0.7)
    assert meta.rows.tolist() == [0, 2, 4] and not meta.fallback
    assert meta.consensus_kept_frac == pytest.approx(0.5)
    assert meta.y.tolist() == [1, 1, 0, 1, 0, 0, 0, 1, 1]  # alpha in (sample, expert) order
    assert meta.correct.tolist() == (labels == y[:, None]).tolist()

    one_left = labels.clone()
    one_left[2], one_left[4] = torch.tensor([2, 2, 2]), torch.tensor([2, 2, 2])
    meta = build_meta_training_set(z, _one_hot_posteriors(one_left), y, k=2, kp=2, hc=0.7)
    assert meta.fallback and meta.rows.tolist() == list(range(6)) and meta.X.size(0) == 18

    all_wrong = torch.tensor([[1, 2, 1], [1, 1, 1], [1, 0, 0], [0, 0, 0], [2, 2, 2], [2, 2, 2]])
    meta = build_meta_training_set(z, _one_hot_posteriors(all_wrong), y, k=2, kp=2, hc=0.7)
    assert meta.fallback and meta.consensus_kept_frac == pytest.approx(2 / 6)


def test_meta_selector_learns_stops_early_and_is_deterministic():
    X = torch.rand(400, 10, generator=torch.Generator().manual_seed(1))
    y = (X[:, 6] > 0.5).float()  # label = local accuracy f3 > 0.5
    kwargs = dict(hidden=10, val_frac=0.25, patience=5, max_epochs=200, seed=3)
    model, info = fit_meta_selector(X, y, **kwargs)
    assert ((model(X) > 0.5).float() == y).float().mean() >= 0.95
    assert info["epochs"] < 200 and info["epochs"] - info["best_epoch"] == 5
    again, info2 = fit_meta_selector(X, y, **kwargs)
    assert info2 == info
    for a, b in zip(model.state_dict().values(), again.state_dict().values()):
        assert torch.equal(a, b)


def test_selection_and_voting():
    sel = select_competent(torch.tensor([[0.2, 0.5, 0.1], [0.6, 0.4, 0.9]]), 0.5)
    assert sel.tolist() == [[True, True, True], [True, False, True]]

    pred = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    post_sum = torch.tensor([[1.6, 2.0, 0.4], [2.4, 1.2, 0.4]])  # 2-2 tie: higher mean posterior wins
    scores = vote_with_tiebreak(pred, post_sum, torch.ones(2, 4, dtype=torch.bool), 3)
    assert scores.argmax(1).tolist() == [1, 0]
    torch.testing.assert_close(scores[0], torch.tensor([2 + 0.4e-3, 2 + 0.5e-3, 0.1e-3]))

    scores = vote_with_tiebreak(torch.tensor([[0, 0, 1]]), torch.tensor([[0.0, 3.0, 0.0]]), torch.ones(1, 3, dtype=torch.bool), 3)
    assert scores.argmax(1).tolist() == [0]  # the posterior term never overrides a vote margin of 1
    masked = vote_with_tiebreak(torch.tensor([[0, 1, 1]]), torch.tensor([[0.9, 0.0, 0.0]]), torch.tensor([[True, False, False]]), 3)
    assert masked.argmax(1).tolist() == [0]  # unselected experts do not vote


# --------------------------------------------------------------------------- #
# Runner on a toy pool with directly constructed posteriors
# --------------------------------------------------------------------------- #
class _GuardedLabels(dict):
    def __getitem__(self, key):
        if key != "support":
            raise AssertionError(f"META-DES read {key!r} labels")
        return super().__getitem__(key)


def _posteriors(labels, correct_prob, num_classes, g):
    n = labels.numel()
    correct = torch.rand(n, generator=g) < correct_prob
    wrong = (labels + torch.randint(1, num_classes, (n,), generator=g)) % num_classes
    decided = torch.where(correct, labels, wrong)
    peak = 0.5 + 0.4 * torch.rand(n, generator=g)
    post = ((1 - peak) / (num_classes - 1)).unsqueeze(1).repeat(1, num_classes)
    post[torch.arange(n), decided] = peak
    return post.half().float()  # the infra returns float16-stored predictions


class _ToyInfra:
    """Two descriptor clusters; expert A is expert in cluster 0, B in cluster 1, C-F guess."""

    def __init__(self, family="node_cls", n_q=600, num_classes=5, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.family, self.device, self.L = family, torch.device("cpu"), num_classes
        self.y_s = torch.arange(num_classes).repeat_interleave(10)
        n_s = self.y_s.numel()
        self.y_q = torch.randint(0, num_classes, (n_q,), generator=g)
        self.c_s, self.c_q = torch.randint(0, 2, (n_s,), generator=g), torch.randint(0, 2, (n_q,), generator=g)
        centers = torch.tensor([[-3.0, 0.0], [3.0, 0.0]])
        self.z = {
            "support": centers[self.c_s] + 0.5 * torch.randn(n_s, 2, generator=g),
            "query": centers[self.c_q] + 0.5 * torch.randn(n_q, 2, generator=g),
        }
        chance = 1.0 / num_classes
        skill = {"A": (0.95, chance), "B": (chance, 0.95), "C": (chance,) * 2, "D": (chance,) * 2, "E": (chance,) * 2, "F": (chance,) * 2}
        support_pos, query_pos = torch.arange(n_s), torch.arange(n_s, n_s + n_q)
        self.preds = {}
        for eid, (p0, p1) in skill.items():
            self.preds[eid] = {
                "support_pos": support_pos,
                "query_pos": query_pos,
                "pred": _posteriors(self.y_q, torch.where(self.c_q == 0, p0, p1), num_classes, g),
                "support_oof_pred": _posteriors(self.y_s, torch.where(self.c_s == 0, p0, p1), num_classes, g),
                "support_pred": torch.nn.functional.one_hot(self.y_s, num_classes).float(),  # in-sample: never used
            }
        self._data = SimpleNamespace(
            num_classes=num_classes, support_pos=support_pos, query_pos=query_pos, labels=_GuardedLabels(support=self.y_s)
        )
        self.evaluated = 0

    def task_family(self, app):
        return self.family

    def compatible_pool(self, app):
        return list(self.preds)

    def data(self, app):
        return self._data

    def support_labels(self, app):
        return self.y_s

    def descriptors(self, app, split):
        return self.z[split]

    def expert_predictions(self, app, expert_ids):
        return {e: dict(self.preds[e]) for e in expert_ids}

    def evaluate_outputs(self, app, pred):
        self.evaluated += 1
        assert pred.shape == (self.y_q.numel(), self.L)
        return {"acc": 100.0 * float((pred.argmax(1) == self.y_q).float().mean()), "risk": 0.0}

    def accuracy(self, eid):
        return 100.0 * float((self.preds[eid]["pred"].argmax(1) == self.y_q).float().mean())


def test_runner_beats_best_single_expert_with_local_competence():
    infra = _ToyInfra()
    runner = METADESRunner(_base_cfg(query_chunk_size=128), APP, infra)
    runner.fit()
    best_single = max(infra.accuracy(e) for e in infra.preds)
    assert runner.best_metrics["test_acc"] >= best_single + 10.0
    assert infra.evaluated == 1 and runner.best_epoch is None
    for key in ("test_mean_ensemble_size", "test_fallback_rate", "test_all_agree_rate", "meta_train_size", "meta_val_mse", "consensus_kept_frac"):
        assert key in runner.best_metrics
    # Meta-labels come from the out-of-fold posteriors, not the in-sample support predictions.
    oof_correct = torch.stack([infra.preds[e]["support_oof_pred"].argmax(1) == infra.y_s for e in runner.pool], dim=1)
    assert torch.equal(runner._meta.correct, oof_correct)
    # C'(x) contains A on cluster-0 queries (and B on cluster-1 queries) almost always.
    sel = runner.selection
    assert sel[infra.c_q == 0, 0].float().mean() >= 0.9 and sel[infra.c_q == 1, 1].float().mean() >= 0.9


def _dense_reference(runner, infra):
    """Algorithm 2 with the full [|Q|, M * L] query profiles and the runner's fitted lambda."""
    meta, pool = runner._meta, runner.pool
    F = torch.stack([infra.preds[e]["pred"] for e in pool], dim=1)  # [n_q, M, L]
    P = torch.stack([infra.preds[e]["support_oof_pred"] for e in pool], dim=1)
    n_q, M, L = F.shape
    phi = torch.cdist(F.reshape(n_q, -1), P.reshape(P.size(0), -1)).topk(meta.kp, dim=1, largest=False).indices
    theta = knn_indices(runner._scaler.transform(infra.z["query"]), runner._z_s, meta.k)
    conf, pred = F.max(dim=-1)
    with torch.no_grad():
        comp = runner.selector(compute_meta_features(meta.correct, meta.post_true, conf, theta, phi)).view(n_q, M)
    sel = select_competent(comp, runner.mcfg.selection_threshold)
    sel[(pred == pred[:, :1]).all(1)] = True
    scores = vote_with_tiebreak(pred, (sel.unsqueeze(-1) * F).sum(1), sel, L)
    return scores / scores.sum(1, keepdim=True)


def test_streaming_matches_dense_reference_across_chunk_sizes():
    infra = _ToyInfra(n_q=300, seed=1)
    small = METADESRunner(_base_cfg(query_chunk_size=7), APP, infra)
    small.fit()
    torch.testing.assert_close(small.predict_queries(), _dense_reference(small, infra))
    large = METADESRunner(_base_cfg(query_chunk_size=10**6), APP, infra)
    large.fit()
    torch.testing.assert_close(large.predict_queries(), small.predict_queries())
    assert large.best_metrics == small.best_metrics


@pytest.mark.parametrize("family", ["link", "multilabel", "regression"])
def test_scope_guard(family):
    with pytest.raises(NotImplementedError):
        METADESRunner(_base_cfg(), APP, SimpleNamespace(task_family=lambda app: family))


# --------------------------------------------------------------------------- #
# Harness wiring on the real infra (tiny experts, synthetic applications)
# --------------------------------------------------------------------------- #
def test_matched_harness_on_real_infra(tmp_path):
    cfg = tiny_cfg(
        tmp_path,
        targets=("nodea:node", "grapha:graph"),
        history_extra=("nodeb:node", "srca:node", "srcb:graph"),
        budgets=(3,),
    )
    cfg.save_results.output_dir = str(tmp_path / "results")
    cfg.moe.routergfm.baselines.method = "meta_des"
    provider = SyntheticDataProvider()
    history = [a for a in enumerate_applications(cfg.moe.routergfm) if a.dataset not in ("nodea", "grapha")]
    generate_history(cfg, provider, apps=history)
    infra = RouterInfra(cfg, provider)

    assert load_runner_class("meta_des") is METADESRunner
    assert matched_run.run_matched_baseline_from_cfg(cfg, infra=infra) == 0
    with open(tmp_path / "results" / "moe_meta_des.tsv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) == 2
    for row in rows:
        assert 0.0 <= float(row["test_acc_mean"]) <= 100.0
        assert float(row["test_mean_ensemble_size_mean"]) <= cfg.moe.routergfm.baselines.candidate_pool
        assert "meta_val_mse_mean" in row and "consensus_kept_frac_mean" in row

    app = infra.application("nodea:node", 3, 42)
    runner = METADESRunner(cfg, app, infra)
    runner.fit()
    assert len(runner.pool) == cfg.moe.routergfm.baselines.candidate_pool
    assert set(runner.pool) <= set(infra.compatible_pool(app))
    probs = runner.predict_queries()
    torch.testing.assert_close(probs.sum(1), torch.ones(probs.size(0)))
