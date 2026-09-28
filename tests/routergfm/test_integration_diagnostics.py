"""Integration rules (Eq. 1, 6-8; App. C/D.2 fixed-team rules) and evaluation diagnostics (Sec. 4.1, App. A.2, B.4)."""

from __future__ import annotations

import math

import pytest
import torch

from src.config import cfg as base_cfg
from src.moe.routergfm.applications import derive_seed
from src.moe.routergfm.archive import Archive, perturb_archive
from src.moe.routergfm.common import GRAPH_CLS, LINK, MULTILABEL, NODE_CLS, REGRESSION, AppSpec, CompatKey
from src.moe.routergfm.diagnostics import (
    cell_mass,
    cell_means,
    eval_cells,
    hit_at_k,
    mixture_risk,
    regret_at_k,
    routing_risks,
    specialization_index,
    task_metrics,
    winner_agreement,
    winner_coverage,
    worst_cell_risk,
)
from src.moe.routergfm.integration import (
    RULES,
    IntegrationContext,
    LocalEvidence,
    integration_weights,
    local_estimates,
    mix,
    retrieve,
)
from src.moe.routergfm.losses import RegressionNormalizer, routing_loss


class _KeyModel:
    """k_phi(z, v) = z + v, so keys and distances are easy to write down."""

    def keys(self, z, v):
        return z + v


def _cfg(**integration):
    cfg = base_cfg.clone()
    ic = cfg.moe.routergfm.integration
    ic.stacking_epochs, ic.local_mlp_epochs, ic.local_mlp_hidden = 150, 200, 8
    ic.stacking_lr, ic.local_mlp_lr, ic.local_mlp_weight_decay = 0.1, 0.05, 0.0
    for key, value in integration.items():
        setattr(ic, key, value)
    cfg.moe.routergfm.integration.eval_cells = 2
    cfg.moe.routergfm.archive.kmeans_iters = 20
    return cfg


def _archive(rep, r_local, mu_app, app, expert=None):
    n = rep.size(0)
    num_apps = int(app.max()) + 1
    compat = CompatKey(NODE_CLS, 5).as_tuple()
    return Archive(
        rep=rep, expert=torch.zeros(n, dtype=torch.long) if expert is None else expert, app=app,
        group=[f"g{i}" for i in range(num_apps)], compat=[compat] * num_apps,
        r_local=r_local, mu_app=mu_app, count=torch.ones(n), cell=torch.arange(n),
        family=torch.zeros(n, dtype=torch.long), apps=[AppSpec(f"g{i}", "node", 5, 0) for i in range(num_apps)],
    )


def _evidence(archive, team_v, *, j=8, cap=8, bandwidth=1.0):
    model = _KeyModel()
    v_rec = torch.zeros(len(archive), team_v.size(1))
    return LocalEvidence(model, team_v, archive, model.keys(archive.rep, v_rec), j, cap, bandwidth)


def _ctx(evidence=None, *, mu=(0.3, 0.5), z=None, rho=1.0, tau=0.1, family=NODE_CLS, **kw):
    z = torch.tensor([[0.0, 0.0], [1.0, 0.5], [-1.0, 2.0]]) if z is None else z
    return IntegrationContext(
        family=family, mu_hat=torch.tensor(mu), tau=tau, rho=rho, z_query=z, cfg=kw.pop("cfg", _cfg()), seed=7,
        evidence=evidence, **kw,
    )


@pytest.fixture
def evidence():
    rep = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [-1.0, 2.0]])
    r_local = torch.tensor([0.1, 0.6, 0.2, 0.9])
    mu_app = torch.tensor([0.3, 0.3, 0.5, 0.5])
    team_v = torch.tensor([[0.0, 0.0], [0.2, -0.1]])
    return _evidence(_archive(rep, r_local, mu_app, torch.tensor([0, 0, 1, 1])), team_v, bandwidth=0.8)


# --------------------------------------------------------------------------- #
# Eq. 1 and the application-level rules
# --------------------------------------------------------------------------- #
def test_mix_is_eq1_for_every_family():
    preds = torch.softmax(torch.randn(4, 3, 5), -1)
    alpha = torch.softmax(torch.randn(4, 3), -1)
    mixed = mix(preds, alpha, NODE_CLS)
    assert torch.allclose(mixed, torch.einsum("nk,nkc->nc", alpha, preds))
    assert torch.allclose(mixed.sum(-1), torch.ones(4))  # convex combination stays on the simplex
    shared = torch.tensor([0.2, 0.3, 0.5])
    assert torch.allclose(mix(preds, shared, REGRESSION), mix(preds, shared.expand(4, -1), REGRESSION))
    with pytest.raises(ValueError):
        mix(preds, alpha, "unknown")


def test_uniform_and_global_rules(evidence):
    ctx = _ctx(evidence, tau=0.2)
    assert torch.equal(integration_weights("uniform", ctx), torch.full((3, 2), 0.5))
    expected = torch.softmax(-torch.tensor([0.3, 0.5]) / 0.2, 0)
    alpha = integration_weights("global", ctx)
    assert alpha.shape == (3, 2) and torch.allclose(alpha, expected.expand(3, -1))
    with pytest.raises(ValueError):
        integration_weights("nope", ctx)
    assert set(RULES) == {"uniform", "global", "routergfm", "no_centering", "shuffled", "simplex_stacking", "local_mlp"}


# --------------------------------------------------------------------------- #
# Local transfer (Eq. 6-8) and its controls
# --------------------------------------------------------------------------- #
def test_routergfm_matches_eq6_to_eq8(evidence):
    ctx = _ctx(evidence, rho=0.7, tau=0.1)
    arc, h = evidence.archive, evidence.bandwidth
    expected = torch.zeros(3, 2)
    for n in range(3):
        for k in range(2):
            q = ctx.z_query[n] + evidence.team_v[k]
            w = torch.softmax(-((q - evidence.record_keys) ** 2).sum(-1) / h**2, 0)  # all 4 records retrieved
            expected[n, k] = ctx.mu_hat[k] + 0.7 * (w * (arc.r_local - arc.mu_app)).sum()
    assert torch.allclose(local_estimates(ctx, "routergfm"), expected, atol=1e-6)
    assert torch.allclose(integration_weights("routergfm", ctx), torch.softmax(-expected / 0.1, -1), atol=1e-6)


def test_rho_zero_or_empty_evidence_recovers_global(evidence):
    for ctx in (_ctx(evidence, rho=0.0), _ctx(None), _ctx(_evidence(evidence.archive.subset(torch.zeros(4, dtype=torch.bool)), evidence.team_v))):
        glob = integration_weights("global", ctx)
        for rule in ("routergfm", "no_centering", "shuffled"):
            assert torch.allclose(integration_weights(rule, ctx), glob, atol=1e-7), rule
    assert not torch.allclose(integration_weights("routergfm", _ctx(evidence)), integration_weights("global", _ctx(evidence)))


def test_retrieval_respects_j_and_per_application_cap(evidence):
    evidence.retrieval_j, evidence.per_app_cap = 3, 1
    idx, w = retrieve(evidence, torch.tensor([[0.1, 0.1]]))
    kept = idx[0, 0][w[0, 0] > 0]
    assert kept.numel() == 2  # one record per source application, although J = 3
    assert sorted(evidence.archive.app[kept].tolist()) == [0, 1]
    assert torch.allclose(w.sum(-1), torch.ones(1, 2))


def test_no_centering_uses_raw_local_losses(evidence):
    ctx = _ctx(evidence, rho=0.6)
    idx, w = ctx.retrieved()
    raw = (1 - 0.6) * ctx.mu_hat + 0.6 * (w * evidence.archive.r_local[idx]).sum(-1)
    assert torch.allclose(local_estimates(ctx, "no_centering"), raw, atol=1e-6)
    assert not torch.allclose(local_estimates(ctx, "no_centering"), local_estimates(ctx, "routergfm"))


def test_shuffled_permutes_residuals_on_the_same_retrieved_records(evidence):
    ctx = _ctx(evidence)
    idx, w = ctx.retrieved()
    shuffled = perturb_archive(evidence.archive, "shuffled", derive_seed(ctx.seed, "shuffled"))
    assert sorted(shuffled.residual.tolist()) == pytest.approx(sorted(evidence.archive.residual.tolist()))
    expected = ctx.mu_hat + (w * shuffled.residual[idx]).sum(-1)
    assert torch.allclose(local_estimates(ctx, "shuffled"), expected, atol=1e-6)
    assert torch.equal(integration_weights("shuffled", ctx), integration_weights("shuffled", _ctx(evidence)))
    # Constant residuals: shuffling cannot change anything.
    flat = _archive(evidence.archive.rep, torch.full((4,), 0.4), torch.full((4,), 0.3), evidence.archive.app)
    ctx = _ctx(_evidence(flat, evidence.team_v))
    assert torch.allclose(integration_weights("shuffled", ctx), integration_weights("routergfm", ctx))


# --------------------------------------------------------------------------- #
# Support-fitted rules
# --------------------------------------------------------------------------- #
def _regional_support(n=40):
    """Expert 0 is right where z0 > 0, expert 1 where z0 < 0; the label is always class 0."""
    z = torch.zeros(n, 2)
    z[:, 0] = torch.linspace(-1, 1, n)
    right, wrong = torch.tensor([0.9, 0.1]), torch.tensor([0.1, 0.9])
    pos = (z[:, 0] > 0).view(-1, 1)
    preds = torch.stack([torch.where(pos, right, wrong), torch.where(pos, wrong, right)], dim=1)
    return z, preds, torch.zeros(n, dtype=torch.long)


def test_simplex_stacking_minimizes_support_mixture_risk():
    # Expert 0 is right on 60% of the support, expert 1 on the rest. Mean mixture loss
    # 0.6 (0.9 - 0.8 w)^2 + 0.4 (0.1 + 0.8 w)^2 is minimized at w = 0.625 (risk 0.24; uniform 0.25).
    z, preds, y = _regional_support(50)
    preds = torch.stack([preds[-1]] * 30 + [preds[0]] * 20)  # rows where expert 0 / expert 1 is right
    ctx = _ctx(None, z=z[:5], support_oof=preds, support_target=y, support_z=z)
    alpha = integration_weights("simplex_stacking", ctx)
    assert alpha.shape == (5, 2) and torch.allclose(alpha.sum(-1), torch.ones(5)) and bool((alpha == alpha[0]).all())
    assert float(alpha[0, 0]) == pytest.approx(0.625, abs=0.02)
    risk = float(routing_loss(mix(preds, alpha[0], NODE_CLS), y, NODE_CLS).mean())
    assert risk == pytest.approx(0.24, abs=1e-3)


def test_local_mlp_learns_context_dependent_weights():
    z, preds, y = _regional_support()
    queries = torch.tensor([[0.8, 0.0], [-0.8, 0.0]])
    ctx = _ctx(None, z=queries, support_oof=preds, support_target=y, support_z=z)
    alpha = integration_weights("local_mlp", ctx)
    assert torch.allclose(alpha.sum(-1), torch.ones(2))
    assert alpha[0, 0] > 0.8 and alpha[1, 1] > 0.8
    assert torch.equal(alpha, integration_weights("local_mlp", _ctx(None, z=queries, support_oof=preds, support_target=y, support_z=z)))
    stack = integration_weights("simplex_stacking", ctx)
    assert torch.allclose(stack[0], stack[1]) and abs(float(stack[0, 0]) - 0.5) < 0.05  # no single best expert


def test_support_rules_normalize_regression_targets():
    y = torch.tensor([[10.0], [12.0], [14.0], [16.0]])
    normalizer = RegressionNormalizer().fit(y)
    y_norm = normalizer.transform(y)
    preds = torch.stack([y_norm + 0.05, y_norm + 3.0], dim=1)  # expert 0 accurate in normalized units
    ctx = _ctx(None, z=torch.zeros(2, 2), family=REGRESSION, support_oof=preds, support_target=y,
               support_z=torch.zeros(4, 2), normalizer=normalizer)
    assert integration_weights("simplex_stacking", ctx)[0, 0] > 0.9
    with pytest.raises(ValueError):
        integration_weights("simplex_stacking", _ctx(None))


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def test_hit_and_regret_at_k():
    mu = {"a": 0.30, "b": 0.20, "c": 0.20, "d": float("nan")}
    assert hit_at_k(["a", "c"], mu) == 1.0  # ties count as a best eligible expert
    assert hit_at_k(["a", "d"], mu) == 0.0
    assert regret_at_k(["a", "d"], mu) == pytest.approx(0.10)
    assert regret_at_k(["b"], mu) == 0.0
    assert math.isnan(regret_at_k(["d"], mu)) and math.isnan(hit_at_k(["a"], {"a": float("nan")}))


def test_mixture_risk_uses_mixed_predictions():
    preds = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # two confident, opposite experts
    target = torch.tensor([0])
    mixed = mix(preds, torch.tensor([0.5, 0.5]), NODE_CLS)
    assert mixture_risk(mixed, target, NODE_CLS) == pytest.approx(0.25)  # not 0.5 * (0 + 1)
    losses = routing_risks(preds, target, NODE_CLS)
    assert losses.shape == (1, 2) and torch.allclose(losses, torch.tensor([[0.0, 1.0]]))
    y = torch.tensor([[1.0], [3.0], [float("nan")]])
    norm = RegressionNormalizer().fit(torch.tensor([[1.0], [2.0], [3.0]]))
    risk = mixture_risk(norm.transform(torch.tensor([[1.0], [2.0], [0.0]])), y, REGRESSION, normalizer=norm)
    assert risk == pytest.approx(0.5)  # |0 - 0| and |1 - 2| in MAD units, NaN row dropped


def test_cells_worst_cell_and_mass():
    losses = torch.tensor([0.1, 0.3, 0.8, float("nan"), 0.4])
    cells = torch.tensor([0, 0, 1, 1, 2])
    means, counts = cell_means(losses, cells)
    assert torch.allclose(means, torch.tensor([0.2, 0.8, 0.4])) and counts.tolist() == [2, 1, 1]
    assert worst_cell_risk(losses, cells) == pytest.approx(0.8)
    assert cell_mass(losses.view(-1, 1), cells, 3).tolist() == [2, 1, 1]
    z = torch.cat([torch.randn(20, 3) - 4, torch.randn(20, 3) + 4])
    cfg = _cfg()
    assign = eval_cells(z, cfg, seed=3)
    assert torch.equal(assign, eval_cells(z, cfg, seed=3)) and assign.unique().numel() == 2
    assert assign[:20].unique().numel() == 1 and assign[20:].unique().numel() == 1


def test_winner_coverage_and_agreement():
    # Cell 0 (3 instances): expert 2 best; cell 1 (1 instance): expert 0 best.
    losses = torch.tensor([[0.5, 0.4, 0.1], [0.5, 0.4, 0.1], [0.5, float("nan"), 0.1], [0.0, 0.9, 0.9]])
    cells = torch.tensor([0, 0, 0, 1])
    assert winner_coverage(cells, losses, [2]) == pytest.approx(0.75)
    assert winner_coverage(cells, losses, [0, 2]) == pytest.approx(1.0)
    assert winner_coverage(cells, losses, [1]) == pytest.approx(0.0)
    team = losses[:, [0, 2]]  # cellwise best team expert: index 1 in cell 0, index 0 in cell 1
    alpha = torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.3, 0.7], [0.6, 0.4]])
    assert winner_agreement(alpha, cells, team) == pytest.approx(0.75)


def test_specialization_index_matches_prop2():
    a, b = 0.1, 0.7
    cell_risk = torch.tensor([[a, b], [b, a]])  # equal global risks, opposite local winners
    assert specialization_index(cell_risk, torch.tensor([5.0, 5.0])) == pytest.approx((b - a) / 2)
    dominated = torch.tensor([[0.1, 0.3], [0.2, 0.5]])
    assert specialization_index(dominated, torch.tensor([1.0, 3.0])) == pytest.approx(0.0)
    with_gap = torch.tensor([[a, b, float("nan")], [b, a, 0.0]])  # expert 2 lacks a cell: left out
    assert specialization_index(with_gap, torch.tensor([1.0, 1.0])) == pytest.approx((b - a) / 2)


def test_task_metrics_per_family():
    probs = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4], [0.9, 0.05, 0.05]])
    assert task_metrics(probs, torch.tensor([0, 1, 1, 0]), NODE_CLS)["acc"] == pytest.approx(0.75)
    assert task_metrics(probs[:, :2] / probs[:, :2].sum(-1, keepdim=True), torch.tensor([0, 1, 1, 0]), LINK)["auc"] == pytest.approx(1.0)
    assert task_metrics(probs, torch.tensor([0, 1, 2, 0]), GRAPH_CLS)["acc"] == pytest.approx(1.0)
    multi = torch.tensor([[0.9, 0.2], [0.1, 0.8], [0.7, 0.3]])
    labels = torch.tensor([[1.0, 0.0], [0.0, float("nan")], [1.0, 1.0]])
    assert task_metrics(multi, labels, MULTILABEL)["auc"] == pytest.approx(1.0)
    y = torch.tensor([[10.0], [20.0], [30.0]])
    norm = RegressionNormalizer().fit(y)
    pred = norm.transform(y + torch.tensor([[1.0], [-2.0], [3.0]]))
    assert task_metrics(pred, y, REGRESSION, norm)["mae"] == pytest.approx(2.0, rel=1e-5)  # raw units
