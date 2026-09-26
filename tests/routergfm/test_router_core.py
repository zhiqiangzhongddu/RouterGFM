"""Router core: encoder/scorer/keys (Eq. 3-4), retrieval (Eq. 6-8), objectives (Eq. 9-10)."""

import math
from types import SimpleNamespace

import pytest
import torch

from src.moe.routergfm.router import (
    RouterGFMModel,
    allowed_records,
    huber,
    kernel_weights,
    listmle,
    local_estimate,
    local_sq,
    mixture_weights,
    search,
)

CFG = SimpleNamespace(hidden_dim=16, num_layers=2, dropout=0.0, key_dim=8, key_hidden_dim=16)
DESC_DIM = 5
IN_DIMS = {"app": 7, "expert": 9, "arch": 4, "objective": 4}
NUM_NODES = {"app": 6, "expert": 8, "arch": 3, "objective": 2}
RELATIONS = [
    ("app", "evaluates", "expert"),
    ("expert", "rev_evaluates", "app"),
    ("expert", "uses_arch", "arch"),
    ("arch", "rev_uses_arch", "expert"),
    ("expert", "uses_objective", "objective"),
    ("objective", "rev_uses_objective", "expert"),
    ("expert", "pretrained_on", "app"),
    ("app", "rev_pretrained_on", "expert"),
]


def _toy_graph(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = {t: torch.randn(NUM_NODES[t], IN_DIMS[t], generator=g) for t in NUM_NODES}
    experts = torch.arange(NUM_NODES["expert"])
    eval_pairs = [(a, e) for a in range(4) for e in range(NUM_NODES["expert"]) if (a + e) % 3 != 0]
    ev = torch.tensor(eval_pairs).t()
    ev_attr = torch.stack([torch.rand(ev.shape[1], generator=g), torch.log1p(torch.randint(1, 50, (ev.shape[1],), generator=g).float())], dim=-1)
    forward = {
        ("app", "evaluates", "expert"): (ev, ev_attr),
        ("expert", "uses_arch", "arch"): (torch.stack([experts, experts % 3]), torch.zeros(len(experts), 2)),
        ("expert", "uses_objective", "objective"): (torch.stack([experts, experts % 2]), torch.zeros(len(experts), 2)),
        ("expert", "pretrained_on", "app"): (torch.stack([experts, 4 + experts % 2]), torch.zeros(len(experts), 2)),
    }
    edge_index, edge_attr = {}, {}
    for (src, rel, dst), (index, attr) in forward.items():
        edge_index[(src, rel, dst)] = index
        edge_attr[(src, rel, dst)] = attr
        edge_index[(dst, f"rev_{rel}", src)] = index.flip(0)
        edge_attr[(dst, f"rev_{rel}", src)] = attr
    return x, edge_index, edge_attr


def _model(seed=0):
    torch.manual_seed(seed)
    return RouterGFMModel(IN_DIMS, RELATIONS, CFG, DESC_DIM).eval()


def _relabel(x, edge_index, edge_attr, perms, seed=1):
    """Relabel nodes within each type (new node i = old node perms[t][i]) and shuffle edge order."""
    g = torch.Generator().manual_seed(seed)
    inv = {t: torch.argsort(p) for t, p in perms.items()}
    x2 = {t: x[t][perms[t]] for t in x}
    ei2, ea2 = {}, {}
    for (src, rel, dst), index in edge_index.items():
        shuffle = torch.randperm(index.shape[1], generator=g)
        ei2[(src, rel, dst)] = torch.stack([inv[src][index[0]], inv[dst][index[1]]])[:, shuffle]
        ea2[(src, rel, dst)] = edge_attr[(src, rel, dst)][shuffle]
    return x2, ei2, ea2


def _all_pair_scores(model, h):
    num_app, num_exp = h["app"].shape[0], h["expert"].shape[0]
    a = torch.arange(num_app).repeat_interleave(num_exp)
    e = torch.arange(num_exp).repeat(num_app)
    return model.score(h["app"][a], h["expert"][e]).view(num_app, num_exp)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_model_shapes_and_nonnegative_scores():
    model = _model()
    x, ei, ea = _toy_graph()
    h0 = model.project(x)
    h = model.encode(x, ei, ea)
    for t, n in NUM_NODES.items():
        assert h0[t].shape == (n, CFG.hidden_dim)
        assert h[t].shape == (n, CFG.hidden_dim)
    mu = model.score(h["app"][:3], h["expert"][:3])
    assert mu.shape == (3,) and bool((mu >= 0).all())
    k = model.keys(torch.randn(4, DESC_DIM), h0["expert"][:4])
    assert k.shape == (4, CFG.key_dim)


def test_encoder_is_equivariant_to_relabeling_within_types():
    """Prop. 4: relabeling nodes, relations, and evidence consistently relabels all scores."""
    model = _model()
    x, ei, ea = _toy_graph()
    g = torch.Generator().manual_seed(3)
    perms = {t: torch.randperm(n, generator=g) for t, n in NUM_NODES.items()}
    h = model.encode(x, ei, ea)
    h2 = model.encode(*_relabel(x, ei, ea, perms))
    for t in NUM_NODES:
        torch.testing.assert_close(h2[t], h[t][perms[t]], atol=1e-5, rtol=1e-5)
    mu = _all_pair_scores(model, h)
    mu2 = _all_pair_scores(model, h2)
    torch.testing.assert_close(mu2, mu[perms["app"]][:, perms["expert"]], atol=1e-5, rtol=1e-5)


def test_mean_aggregation_is_degree_stable():
    model = _model()
    x, ei, ea = _toy_graph()
    doubled_ei = {r: torch.cat([i, i], dim=1) for r, i in ei.items()}
    doubled_ea = {r: torch.cat([a, a], dim=0) for r, a in ea.items()}
    h = model.encode(x, ei, ea)
    h2 = model.encode(x, doubled_ei, doubled_ea)
    for t in NUM_NODES:
        torch.testing.assert_close(h2[t], h[t], atol=1e-5, rtol=1e-5)


def test_node_without_incoming_messages_keeps_projection():
    model = _model()
    x, ei, ea = _toy_graph()
    x["app"] = torch.cat([x["app"], torch.randn(1, IN_DIMS["app"])])  # inserted, no edges
    h = model.encode(x, ei, ea)
    torch.testing.assert_close(h["app"][-1], model.project(x)["app"][-1])


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def _brute_force(qk, rk, rec_app, allowed, J, cap):
    out = []
    for p in range(qk.shape[0]):
        d = (rk - qk[p]).pow(2).sum(-1)
        taken, counts = [], {}
        for r in torch.argsort(d).tolist():
            a = int(rec_app[r])
            if not allowed[r] or counts.get(a, 0) >= cap:
                continue
            counts[a] = counts.get(a, 0) + 1
            taken.append(r)
            if len(taken) == J:
                break
        out.append(taken)
    return out


@pytest.mark.parametrize("chunk_size", [None, 3])
def test_search_matches_exact_capped_top_j(chunk_size):
    g = torch.Generator().manual_seed(0)
    qk, rk = torch.randn(10, 4, generator=g), torch.randn(60, 4, generator=g)
    rec_app = torch.randint(0, 5, (60,), generator=g)
    allowed = torch.rand(60, generator=g) > 0.2
    J, cap = 7, 2
    idx, valid = search(qk, rk, rec_app, allowed, J, cap, chunk_size=chunk_size)
    assert idx.shape == (10, J) and valid.shape == (10, J)
    assert not idx.requires_grad
    expected = _brute_force(qk, rk, rec_app, allowed, J, cap)
    for p in range(10):
        assert idx[p][valid[p]].tolist() == expected[p]


def test_search_respects_per_application_cap():
    # Application 0 owns the six nearest records; the cap limits it to two.
    rk = torch.cat([0.1 * torch.arange(1, 7).float().unsqueeze(1), 5.0 + torch.arange(6).float().unsqueeze(1)])
    rec_app = torch.tensor([0] * 6 + [1, 1, 2, 2, 3, 3])
    idx, valid = search(torch.zeros(1, 1), rk, rec_app, torch.ones(12, dtype=torch.bool), J=5, cap=2)
    apps = rec_app[idx[0][valid[0]]]
    assert bool(valid.all())
    assert idx[0].tolist() == [0, 1, 6, 7, 8]
    assert int((apps == 0).sum()) == 2


def test_search_marks_slots_beyond_allowed_set_invalid():
    rk = torch.randn(6, 3)
    rec_app = torch.tensor([0, 0, 1, 1, 2, 2])
    allowed = torch.tensor([True, False, True, False, False, False])
    idx, valid = search(torch.randn(4, 3), rk, rec_app, allowed, J=5, cap=3)
    assert valid.sum(-1).tolist() == [2, 2, 2, 2]
    assert set(idx[valid].tolist()) <= {0, 2}


def test_allowed_mask_excludes_own_and_hidden_groups_and_incompatible_records():
    app_groups = ["cora", "cora", "pubmed", "photo", "photo", "computers"]
    app_compat = [
        ("node_cls", 5, "prob"),
        ("node_cls", 100, "prob"),
        ("node_cls", 5, "prob"),
        ("node_cls", 5, "prob"),
        ("link", 5, "prob"),
        ("node_cls", 5, "prob"),
    ]
    rec_app = torch.arange(6).repeat_interleave(4)
    mask = allowed_records(rec_app, app_groups, app_compat, group="photo", compat=("node_cls", 5, "prob"), hidden_groups={"pubmed"})
    assert mask.tolist() == [a in (0, 5) for a in rec_app.tolist()]

    g = torch.Generator().manual_seed(1)
    rk = torch.randn(24, 3, generator=g)
    qk = rk[12:16] + 1e-3  # queries sit on top of the photo records
    idx, valid = search(qk, rk, rec_app, mask, J=6, cap=3)
    groups = {app_groups[int(a)] for a in rec_app[idx[valid]].tolist()}
    assert groups <= {"cora", "computers"} and "photo" not in groups

    # Grouped masks: row query_group[p] applies to query p.
    other = allowed_records(rec_app, app_groups, app_compat, group="cora", compat=("node_cls", 5, "prob"))
    masks = torch.stack([mask, other])
    qgroup = torch.tensor([0, 1, 0, 1])
    idx, valid = search(qk, rk, rec_app, masks, J=6, cap=3, query_group=qgroup)
    for p in range(4):
        assert bool(masks[qgroup[p]][idx[p][valid[p]]].all())
    assert {app_groups[int(a)] for a in rec_app[idx[1][valid[1]]].tolist()} <= {"pubmed", "photo", "computers"}


def test_kernel_weights_match_eq6_and_ignore_invalid():
    g = torch.Generator().manual_seed(2)
    qk, rk_sel = torch.randn(3, 4, generator=g), torch.randn(3, 5, 4, generator=g)
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]], dtype=torch.bool)
    h = 1.7
    w = kernel_weights(qk, rk_sel, valid, h)
    for p in range(3):
        d2 = (rk_sel[p] - qk[p]).pow(2).sum(-1)
        e = torch.exp(-d2 / h**2) * valid[p]
        torch.testing.assert_close(w[p], e / e.sum())
    assert torch.allclose(w.sum(-1), torch.ones(3))


def test_empty_retrieval_gives_application_level_estimate():
    qk = torch.randn(4, 3, requires_grad=True)
    rk = torch.randn(10, 3)
    rec_app = torch.arange(10) % 3
    idx, valid = search(qk, rk, rec_app, torch.zeros(10, dtype=torch.bool), J=4, cap=2)
    assert not bool(valid.any())
    w = kernel_weights(qk, rk[idx], valid, 1.0)
    assert bool((w == 0).all())
    mu_hat = torch.rand(4)
    residual = torch.randn(4, 4)
    for mode in ("centered", "raw"):
        torch.testing.assert_close(local_estimate(mu_hat, w, residual, 0.7, mode=mode), mu_hat)
    local_estimate(mu_hat, w, residual, 0.7).sum().backward()
    assert torch.isfinite(qk.grad).all()


def test_local_estimate_centered_and_raw_formulas():
    mu_hat = torch.tensor([0.3, 0.5])
    w = torch.tensor([[0.25, 0.75], [1.0, 0.0]])
    r_local = torch.tensor([[0.2, 0.6], [0.1, 0.9]])
    mu_rec = torch.tensor([[0.4, 0.4], [0.3, 0.3]])
    rho = 0.5
    centered = local_estimate(mu_hat, w, r_local - mu_rec, rho)
    torch.testing.assert_close(centered, mu_hat + rho * (w * (r_local - mu_rec)).sum(-1))
    raw = local_estimate(mu_hat, w, r_local, rho, mode="raw")
    torch.testing.assert_close(raw, (1 - rho) * mu_hat + rho * (w * r_local).sum(-1))
    with pytest.raises(ValueError):
        local_estimate(mu_hat, w, r_local, rho, mode="other")


def test_rho_zero_recovers_global_weights():
    g = torch.Generator().manual_seed(4)
    num_queries, team = 6, 3
    mu_hat = torch.rand(team, generator=g)
    w = torch.softmax(torch.randn(num_queries, team, 5, generator=g), dim=-1)
    residual = torch.randn(num_queries, team, 5, generator=g)
    r_hat = local_estimate(mu_hat.expand(num_queries, team), w, residual, 0.0)
    alpha = mixture_weights(r_hat, 0.05)
    torch.testing.assert_close(alpha, mixture_weights(mu_hat, 0.05).expand(num_queries, team))
    assert torch.allclose(alpha.sum(-1), torch.ones(num_queries))
    # Lower estimated loss -> larger weight.
    assert alpha[0].argmax() == mu_hat.argmin()


def test_retrieval_is_equivariant_to_expert_and_record_relabeling():
    """Prop. 4 for retrieval: relabeling experts and archive records leaves r_hat unchanged."""
    model = _model()
    g = torch.Generator().manual_seed(5)
    x, ei, ea = _toy_graph()
    v = model.project(x)["expert"]
    num_rec, num_q = 40, 12
    c = torch.randn(num_rec, DESC_DIM, generator=g)
    rec_exp = torch.randint(0, NUM_NODES["expert"], (num_rec,), generator=g)
    rec_app = torch.randint(0, 6, (num_rec,), generator=g)
    residual = torch.randn(num_rec, generator=g)
    z = torch.randn(num_q, DESC_DIM, generator=g)
    q_exp = torch.randint(0, NUM_NODES["expert"], (num_q,), generator=g)
    mu_hat = torch.rand(num_q, generator=g)
    allowed = rec_app != 0

    def run(v, c, rec_exp, rec_app, residual, allowed, q_exp):
        with torch.no_grad():
            qk = model.keys(z, v[q_exp])
            rk = model.keys(c, v[rec_exp])
        idx, valid = search(qk, rk, rec_app, allowed, J=6, cap=2)
        w = kernel_weights(qk, rk[idx], valid, 1.0)
        return idx, valid, local_estimate(mu_hat, w, residual[idx], 0.8)

    idx, valid, r_hat = run(v, c, rec_exp, rec_app, residual, allowed, q_exp)
    perm_e = torch.randperm(NUM_NODES["expert"], generator=g)
    inv_e = torch.argsort(perm_e)
    perm_r = torch.randperm(num_rec, generator=g)
    perm_a = torch.randperm(6, generator=g)
    inv_a = torch.argsort(perm_a)
    idx2, valid2, r_hat2 = run(
        v[perm_e], c[perm_r], inv_e[rec_exp[perm_r]], inv_a[rec_app[perm_r]], residual[perm_r], allowed[perm_r], inv_e[q_exp]
    )
    torch.testing.assert_close(r_hat2, r_hat, atol=1e-5, rtol=1e-5)
    assert torch.equal(valid2, valid) and bool(valid.any())
    assert torch.equal(perm_r[idx2][valid], idx[valid])


# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #
def test_huber_matches_definition():
    mu_hat, mu_bar, delta = torch.tensor([0.1, 0.5, 0.2]), torch.tensor([0.12, 0.2, 0.2]), 0.1
    diff = (mu_hat - mu_bar).abs()
    ref = torch.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta)).mean()
    torch.testing.assert_close(huber(mu_hat, mu_bar, delta), ref)


def test_listmle_matches_plackett_luce_and_prefers_correct_order():
    scores = torch.tensor([0.3, -1.2, 2.0, 0.1])
    targets = torch.tensor([0.4, 0.9, 0.1, 0.5])  # ascending-loss order: 2, 0, 3, 1
    s = scores[[2, 0, 3, 1]]
    nll = -sum(float(s[i]) - math.log(float(torch.exp(s[i:]).sum())) for i in range(4))
    assert listmle(scores, targets).item() == pytest.approx(nll / 4, rel=1e-5)

    mu_bar = torch.rand(8, generator=torch.Generator().manual_seed(6))
    good = listmle(-mu_bar * 50, mu_bar)
    bad = listmle(mu_bar * 50, mu_bar)
    assert good < bad and good.item() < 0.1
    shuffle = torch.randperm(8, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(listmle(-mu_bar[shuffle], mu_bar[shuffle]), listmle(-mu_bar, mu_bar))
    assert listmle(torch.tensor([1.0]), torch.tensor([0.2])).item() == 0.0


def test_local_sq_masks_invalid_and_averages_per_expert():
    r_hat = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    loss = torch.tensor([0.0, float("nan"), 0.1, 0.4, 0.9])
    expert = torch.tensor([0, 0, 0, 1, 1])
    torch.testing.assert_close(local_sq(r_hat, loss), torch.tensor([0.01, 0.04, 0.0, 0.16]).mean())
    per_expert = (torch.tensor([0.01, 0.04]).mean() + torch.tensor([0.0, 0.16]).mean()) / 2
    torch.testing.assert_close(local_sq(r_hat, loss, expert), per_expert)
    assert local_sq(r_hat, torch.full((5,), float("nan"))).item() == 0.0


# --------------------------------------------------------------------------- #
# End-to-end gradient flow (one training-style step)
# --------------------------------------------------------------------------- #
def test_gradients_reach_scorer_encoder_and_key_net():
    model = _model()
    model.train()
    x, ei, ea = _toy_graph()
    g = torch.Generator().manual_seed(8)

    h = model.encode(x, ei, ea)
    experts = torch.arange(NUM_NODES["expert"])
    mu_hat = model.score(h["app"][0].expand(len(experts), -1), h["expert"][experts])
    mu_bar = torch.rand(len(experts), generator=g)
    l_glob = huber(mu_hat, mu_bar, 0.1) + 0.1 * listmle(-mu_hat, mu_bar)

    v = model.project(x)["expert"]
    num_rec, num_q = 30, 16
    c = torch.randn(num_rec, DESC_DIM, generator=g)
    rec_exp = torch.randint(0, len(experts), (num_rec,), generator=g)
    rec_app = torch.randint(1, 5, (num_rec,), generator=g)
    residual = 0.1 * torch.randn(num_rec, generator=g)
    z = torch.randn(num_q, DESC_DIM, generator=g)
    q_exp = torch.randint(0, len(experts), (num_q,), generator=g)
    with torch.no_grad():
        rk_all = model.keys(c, v[rec_exp])
    qk = model.keys(z, v[q_exp])
    idx, valid = search(qk.detach(), rk_all, rec_app, torch.ones(num_rec, dtype=torch.bool), J=5, cap=2)
    rk_sel = model.keys(c[idx], v[rec_exp[idx]])
    w = kernel_weights(qk, rk_sel, valid, 1.0)
    r_hat = local_estimate(mu_hat[q_exp], w, residual[idx], 1.0)
    l_loc = local_sq(r_hat, torch.rand(num_q, generator=g), q_exp)

    (l_glob + l_loc).backward()

    def grad_norm(module):
        return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)

    assert grad_norm(model.scorer) > 0
    assert grad_norm(model.key_net) > 0
    assert grad_norm(model.layers) > 0
    assert grad_norm(model.proj["expert"]) > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    # Key-net gradients come from L_loc only.
    model.zero_grad()
    l_glob_only = huber(model.score(h["app"][0].expand(len(experts), -1).detach(), h["expert"].detach()), mu_bar, 0.1)
    l_glob_only.backward()
    assert grad_norm(model.key_net) == 0 and grad_norm(model.scorer) > 0
