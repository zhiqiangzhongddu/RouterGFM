"""Tests for the train-label-only LogME expert selection baseline."""

import pytest
import torch

logme = pytest.importorskip("src.moe.routergfm.baselines.logme")
logme_score = logme.logme_score
rank_experts_by_evidence = logme.rank_experts_by_evidence


def _make_classification(n_per_class=20, d=32, classes=4, noise=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(classes, d, generator=g) * 2
    feats, labels = [], []
    for c in range(classes):
        feats.append(centers[c] + noise * torch.randn(n_per_class, d, generator=g))
        labels.extend([c] * n_per_class)
    return torch.cat(feats), torch.tensor(labels)


class TestLogME:
    def test_informative_beats_noise_classification(self):
        feats, labels = _make_classification()
        noise = torch.randn_like(feats)
        assert logme_score(feats, labels) > logme_score(noise, labels)

    def test_informative_beats_shuffled_labels(self):
        feats, labels = _make_classification()
        perm = torch.randperm(len(labels), generator=torch.Generator().manual_seed(1))
        assert logme_score(feats, labels) > logme_score(feats, labels[perm])

    def test_regression_targets(self):
        g = torch.Generator().manual_seed(2)
        f = torch.randn(50, 16, generator=g)
        w = torch.randn(16, generator=g)
        y = f @ w + 0.05 * torch.randn(50, generator=g)
        assert logme_score(f, y) > logme_score(torch.randn_like(f), y)

    def test_vector_regression_with_nan(self):
        g = torch.Generator().manual_seed(3)
        f = torch.randn(40, 16, generator=g)
        w = torch.randn(16, 3, generator=g)
        y = f @ w
        y[::5, 1] = float("nan")  # missing entries must not poison the score
        s = logme_score(f, y)
        assert s == s  # finite, not nan
        assert s > logme_score(torch.randn_like(f), y)

    def test_few_shot_regime(self):
        # 5-shot x 4 classes = 20 rows, d=128 (d >> n): must stay finite and
        # still prefer informative features.
        feats, labels = _make_classification(n_per_class=5, d=128, noise=0.05)
        assert logme_score(feats, labels) > logme_score(torch.randn_like(feats), labels)

    def test_degenerate_rows(self):
        assert logme_score(torch.randn(1, 8), torch.tensor([0])) == float("-inf")


class TestRanking:
    def test_ranks_informative_expert_first(self):
        feats, labels = _make_classification()
        experts = {
            7: torch.randn_like(feats),  # noise expert
            3: feats,  # informative expert
            9: feats + 3.0 * torch.randn_like(feats),  # weak expert
        }
        ranked = rank_experts_by_evidence(experts, labels, order_hint=[7, 3, 9])
        assert ranked[0][0] == 3
        assert ranked[-1][0] == 7

    def test_tie_breaks_by_router_order(self):
        feats, labels = _make_classification()
        experts = {5: feats.clone(), 2: feats.clone()}
        ranked = rank_experts_by_evidence(experts, labels, order_hint=[2, 5])
        assert [e for e, _ in ranked] == [2, 5]
