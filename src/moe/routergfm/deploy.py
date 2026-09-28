"""Deployment on a target application with the router fixed (paper Alg. 1 l.13-21, Sec. 3.4).

1. Load (or train) the leave-one-dataset-out router of (target group, budget[, seed]).
2. Rebuild H from the router's applications and insert the target from metadata
   only (no evaluation edges); encode with the target group's evaluation edges hidden.
3. Score E_a (Eq. 4); the team T_a is the K lowest ``mu_hat``.
4. Fit the team's heads once on S_a and predict Q_a (``history.predict_queries``).
5. Standardize the query contexts with the router's descriptor standardizer.
6. Weight the same predictions with every requested integration rule (Eq. 1, 6-8).
7. Metrics. Query labels and the target's recorded history are read only by
   :func:`evaluate_integration`, after every weight is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.utils.checkpoint import save_json_atomic

from .applications import derive_seed
from .archive import Archive, build_archive, perturb_archive
from .common import AppSpec, CompatKey, ExpertSpec, RouterPaths, reported_metric, stable_hash
from .context_graph import APP, EXPERT, ContextGraph, build_context_graph, insert_application, masked_edges
from .diagnostics import (
    cell_mass,
    cell_means,
    eval_cells,
    hit_at_k,
    regret_at_k,
    routing_risks,
    specialization_index,
    task_metrics,
    winner_agreement,
    winner_coverage,
    worst_cell_risk,
)
from .integration import LOCAL_RULES, RULES, IntegrationContext, LocalEvidence, integration_weights, mix
from .router.retrieval import allowed_records
from .router.trainer import RouterBundle, RouterTrainer, router_run_key, train_router

_LOG = "[RouterGFM deploy]"
_KEY_CHUNK = 1 << 16  # archive records per key-network pass


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
def router_seed(cfg, app: AppSpec) -> int:
    """Router seed of an application: its split seed with ``router.per_seed``, else ``router.seed``."""
    rt = cfg.moe.routergfm.router
    return int(app.seed) if bool(rt.per_seed) else int(rt.seed)


def router_directory(cfg, app: AppSpec) -> Path:
    return RouterPaths.from_cfg(cfg).router_dir(router_run_key(app.group, app.budget, router_seed(cfg, app)))


def load_router_bundle(cfg, app: AppSpec, provider=None, device=None) -> Tuple[RouterBundle, Path]:
    """The router of (app.group, app.budget, router seed), via :func:`train_router` (reused with ``router.skip_if_exists``)."""
    directory = train_router(cfg, app.group, app.budget, provider, router_seed(cfg, app))
    return RouterTrainer.load(directory, cfg, device), directory


@dataclass
class DeployRouter:
    """A router bundle with the H and M it was trained around (target group absent from both)."""

    bundle: RouterBundle
    catalog: List[ExpertSpec]  # index space of graph.expert_catalog_index and archive.expert
    graph: ContextGraph  # H over bundle.graph_apps, CPU, without the target
    archive: Archive  # M over bundle.archive_apps (archive.perturbation applied)
    directory: Optional[Path] = None


def prepare_router(cfg, app: AppSpec, infra, *, bundle: Optional[RouterBundle] = None) -> DeployRouter:
    """Load the router of *app* (unless *bundle* is given) and rebuild H and M as in training.

    H uses the bundle's graph applications and numeric standardization; M the
    bundle's archive applications and descriptor standardizer, with
    ``archive.perturbation`` (App. D.5 controls; default ``none``) applied.
    """
    directory = None
    if bundle is None:
        bundle, directory = load_router_bundle(cfg, app, infra.provider, infra.device)
    leaked = sorted({a.group for a in bundle.graph_apps + bundle.archive_apps if a.group == app.group})
    if leaked:
        raise ValueError(f"Router for {app.key} was trained with the target group {leaked} (leave-one-dataset-out).")
    unknown = [e for e in bundle.catalog_ids if e not in infra.expert_index]
    if unknown:
        raise ValueError(f"Router experts missing from the current catalog: {unknown[:5]} ({len(unknown)} total).")
    catalog = [infra.catalog[infra.expert_index[e]] for e in bundle.catalog_ids]
    stats = {a.key: infra.data(a).stats for a in bundle.graph_apps}
    families = {a.key: infra.task_family(a) for a in bundle.graph_apps}
    graph = build_context_graph(
        cfg, catalog, bundle.graph_apps, infra.store, infra.text_encoder, stats,
        families=families, numeric_stats=bundle.numeric_stats,
    )
    arc = cfg.moe.routergfm.archive
    archive = build_archive(bundle.archive_apps, infra.store, bundle.standardizer, cfg, catalog=catalog, provider=infra.provider)
    archive = perturb_archive(archive, str(arc.perturbation), int(arc.perturbation_seed))
    return DeployRouter(bundle=bundle, catalog=catalog, graph=graph, archive=archive, directory=directory)


def _model_device(router: DeployRouter) -> torch.device:
    return next(router.bundle.model.parameters()).device


@torch.no_grad()
def score_pool(cfg, app: AppSpec, infra, router: DeployRouter) -> Tuple[List[str], torch.Tensor, ContextGraph]:
    """E_a and ``mu_hat_{a,e}`` (Eq. 4) with the target inserted from metadata only (Alg. 1 l.13-14).

    Returns the pool (catalog order), its scores (CPU), and the device copy of H
    holding the target node.
    """
    data = infra.data(app)
    graph = router.graph.to(_model_device(router))
    node = insert_application(graph, app, data.stats, infra.text_encoder, cfg, family=data.task_family)
    model = router.bundle.model.eval()
    h = model.encode(graph.x, *masked_edges(graph, {app.group}))
    pool = [e for e in infra.compatible_pool(app) if e in graph.expert_index]
    if not pool:
        raise ValueError(f"{app.key}: no eligible expert of E_a is a node of the router graph.")
    nodes = torch.tensor([graph.expert_index[e] for e in pool], dtype=torch.long, device=h[EXPERT].device)
    mu = model.score(h[APP][node].expand(len(pool), -1), h[EXPERT][nodes])
    return pool, mu.float().cpu(), graph


@torch.no_grad()
def local_evidence(
    app: AppSpec, family: str, router: DeployRouter, graph: ContextGraph, team: Sequence[str]
) -> LocalEvidence:
    """Records of M the target may retrieve (other groups, same CompatKey) with their keys ``k_phi(c_i, v_{e_i})``."""
    bundle, rc = router.bundle, router.bundle.router_cfg
    model, device = bundle.model, _model_device(router)
    node_catalog = torch.as_tensor(graph.expert_catalog_index, dtype=torch.long)
    size = max(len(router.catalog), int(node_catalog.max()) + 1 if node_catalog.numel() else 0)
    node_of = torch.full((size,), -1, dtype=torch.long)
    node_of[node_catalog] = torch.arange(node_catalog.numel())
    archive = router.archive
    rec_node = node_of[archive.expert.cpu()]
    compat = CompatKey(family, app.budget).as_tuple()
    allowed = allowed_records(archive.app.cpu(), archive.group, archive.compat, group=app.group, compat=compat)
    allowed &= rec_node >= 0
    archive = archive.subset(allowed.to(archive.app.device))
    nodes = rec_node[allowed].to(device)
    v = model.project(graph.x)[EXPERT]
    rep = archive.rep.to(device)
    keys = [model.keys(rep[s:s + _KEY_CHUNK], v[nodes[s:s + _KEY_CHUNK]]) for s in range(0, len(archive), _KEY_CHUNK)]
    record_keys = torch.cat(keys) if keys else torch.zeros(0, int(rc["key_dim"]), device=device)
    team_nodes = torch.tensor([graph.expert_index[e] for e in team], dtype=torch.long, device=device)
    return LocalEvidence(
        model=model,
        team_v=v[team_nodes],
        archive=archive,
        record_keys=record_keys,
        retrieval_j=int(rc["retrieval_j"]),
        per_app_cap=int(rc["per_app_cap"]),
        bandwidth=float(bundle.bandwidth),
    )


# --------------------------------------------------------------------------- #
# Integration (label-free on the target query side)
# --------------------------------------------------------------------------- #
@dataclass
class Integration:
    """One deployment's team, fixed query predictions, and the weights and mixtures of every rule."""

    app: AppSpec
    family: str
    pool: List[str]  # E_a
    pool_mu_hat: torch.Tensor  # [|E_a|]
    team: List[str]  # T_a
    mu_hat: torch.Tensor  # [K]
    preds: torch.Tensor  # [N, K, C] team predictions on Q_a (family space)
    z_query: torch.Tensor  # [N, D] standardized query contexts
    alpha: Dict[str, torch.Tensor]  # rule -> [N, K]
    mixed: Dict[str, torch.Tensor]  # rule -> [N, C]
    rho: float
    tau: float
    bandwidth: float  # h of Eq. 6 (the bundle's validation-selected value)
    num_records: int  # compatible archive records available for retrieval
    context: Optional[IntegrationContext] = None  # the rules' shared inputs (evidence and retrieval included)


def _check_rules(rules: Sequence[str]) -> List[str]:
    rules = list(dict.fromkeys(str(r) for r in rules))
    unknown = [r for r in rules if r not in RULES]
    if unknown or not rules:
        raise ValueError(f"Unknown or empty integration rules {unknown or rules}; expected a subset of {RULES}.")
    return rules


def _stack_outputs(outputs: Dict[str, Dict[str, Any]], team: Sequence[str], key: str, positions, pos_key: str):
    for e in team:
        if not torch.equal(outputs[e][pos_key].cpu(), torch.as_tensor(positions).cpu()):
            raise ValueError(f"{e}: stored {pos_key} differ from the application's positions (stale predictions).")
    return torch.stack([outputs[e][key].float() for e in team], dim=1)


def integrate_application(
    cfg,
    app: AppSpec,
    infra,
    router: DeployRouter,
    rules: Sequence[str],
    *,
    team_override: Optional[Sequence[str]] = None,
) -> Integration:
    """Alg. 1 l.13-20 for every rule on one team. Reads support labels only (heads, fitted rules)."""
    rules = _check_rules(rules)
    data = infra.data(app)
    family = data.task_family
    bundle = router.bundle
    pool, pool_mu, graph = score_pool(cfg, app, infra, router)
    if team_override is None:
        order = torch.argsort(pool_mu, stable=True)[: int(bundle.router_cfg["topk"])]
    else:
        index = {e: i for i, e in enumerate(pool)}
        team_ids = list(dict.fromkeys(str(e) for e in team_override))
        missing = [e for e in team_ids if e not in index]
        if missing or not team_ids:
            raise ValueError(f"team_override must be a non-empty subset of E_a; not eligible: {missing}")
        order = torch.tensor([index[e] for e in team_ids], dtype=torch.long)
    team, mu_team = [pool[i] for i in order.tolist()], pool_mu[order]

    outputs = infra.expert_predictions(app, team)
    preds = _stack_outputs(outputs, team, "pred", data.query_pos, "query_pos")
    oof = _stack_outputs(outputs, team, "support_oof_pred", data.support_pos, "support_pos")
    standardizer = bundle.standardizer
    z_query = standardizer.transform(infra.descriptors(app, "query")).cpu()
    z_support = standardizer.transform(infra.descriptors(app, "support")).cpu()
    evidence = local_evidence(app, family, router, graph, team) if any(r in LOCAL_RULES for r in rules) else None
    ctx = IntegrationContext(
        family=family,
        mu_hat=mu_team,
        tau=float(bundle.tau),
        rho=float(bundle.rho),
        z_query=z_query,
        cfg=cfg,
        seed=derive_seed(app.seed, "integration", app.key),
        evidence=evidence,
        support_oof=oof,
        support_target=data.labels["support"],
        support_z=z_support,
        normalizer=infra.normalizer(app),
    )
    alpha = {rule: integration_weights(rule, ctx) for rule in rules}
    return Integration(
        app=app,
        family=family,
        pool=pool,
        pool_mu_hat=pool_mu,
        team=team,
        mu_hat=mu_team,
        preds=preds,
        z_query=z_query,
        alpha=alpha,
        mixed={rule: mix(preds, a, family) for rule, a in alpha.items()},
        rho=ctx.rho,
        tau=ctx.tau,
        bandwidth=float(bundle.bandwidth),
        num_records=len(evidence.archive) if evidence is not None else 0,
        context=ctx,
    )


# --------------------------------------------------------------------------- #
# Evaluation (the only reader of query labels and of the target's history)
# --------------------------------------------------------------------------- #
def _selection_diagnostics(infra, app: AppSpec, integ: Integration, query_pos: torch.Tensor, cells: torch.Tensor) -> Dict[str, float]:
    """hit@K / regret@K, local-winner coverage, and specialization index from the target's D_a history."""
    store = infra.store
    mu_true = {e: store.app_average(app, e)[0] for e in integ.pool}
    out = {
        "k": len(integ.team),
        "hit_at_k": hit_at_k(integ.team, mu_true),
        "regret_at_k": regret_at_k(integ.team, mu_true),
        "winner_coverage": float("nan"),
        "specialization_index": float("nan"),
    }
    if not store.expert_ids(app.data_key) or cells.numel() == 0:
        return out
    diag_pos = store.matrix(app.data_key)["diag_pos"]
    rows = torch.searchsorted(query_pos, diag_pos).clamp(max=query_pos.numel() - 1)
    if not torch.equal(query_pos[rows], diag_pos):
        raise ValueError(f"{app.key}: recorded diagnostic positions are not query positions (stale history).")
    losses = store.losses(app, integ.pool)  # [|D_a|, |E_a|]
    diag_cells = cells[rows]
    index = {e: i for i, e in enumerate(integ.pool)}
    out["winner_coverage"] = winner_coverage(diag_cells, losses, [index[e] for e in integ.team])
    b = int(cells.max()) + 1
    out["specialization_index"] = specialization_index(cell_means(losses, diag_cells, b)[0], cell_mass(losses, diag_cells, b))
    return out


def evaluate_integration(cfg, app: AppSpec, infra, integ: Integration) -> Dict[str, Any]:
    """Per-rule task metric, mixture routing risk, worst-cell risk, winner agreement; team diagnostics."""
    rg = cfg.moe.routergfm
    data = infra.data(app)
    family, reg_kind = integ.family, str(rg.loss.regression)
    target, normalizer = data.labels["query"], infra.normalizer(app)
    cells = eval_cells(integ.z_query, cfg, derive_seed(app.seed, "eval_cells", app.key))
    team_losses = routing_risks(integ.preds, target, family, normalizer=normalizer, reg_kind=reg_kind)
    per_rule: Dict[str, Dict[str, float]] = {}
    for rule, pred in integ.mixed.items():
        losses = routing_risks(pred, target, family, normalizer=normalizer, reg_kind=reg_kind)
        valid = torch.isfinite(losses)
        metrics = task_metrics(pred, target, family, normalizer)
        metrics["risk"] = float(losses[valid].mean()) if bool(valid.any()) else float("nan")
        metrics["worst_cell_risk"] = worst_cell_risk(losses, cells)
        metrics["winner_agreement"] = winner_agreement(integ.alpha[rule], cells, team_losses)
        per_rule[rule] = metrics
    return {
        "app": app.to_dict(),
        "app_key": app.key,
        "family": family,
        "metric": reported_metric(family),
        "rho": integ.rho,
        "tau": integ.tau,
        "bandwidth": integ.bandwidth,
        "team": list(integ.team),
        "mu_hat": integ.mu_hat.tolist(),
        "pool_mu_hat": dict(zip(integ.pool, integ.pool_mu_hat.tolist())),
        "num_queries": int(integ.preds.size(0)),
        "num_records": integ.num_records,
        "selection": _selection_diagnostics(infra, app, integ, data.query_pos, cells),
        "rules": per_rule,
    }


def deploy_application(
    cfg,
    app: AppSpec,
    *,
    provider=None,
    rules: Optional[Sequence[str]] = None,
    team_override: Optional[Sequence[str]] = None,
    bundle: Optional[RouterBundle] = None,
    infra=None,
    router: Optional[DeployRouter] = None,
) -> Dict[str, Any]:
    """Deploy the router on *app* and evaluate every rule (default ``integration.rules``).

    ``infra`` / ``router`` reuse a :class:`RouterInfra` / :func:`prepare_router`
    result across calls; ``bundle`` replaces the stored router. Returns
    ``{'team', 'mu_hat', 'selection': {...}, 'rules': {rule: {metric: value}}, ...}``
    and writes it to ``deploy_dir(app.key)/deploy[_team_<hash>].json``.
    """
    if infra is None:
        from .infra import RouterInfra

        infra = RouterInfra(cfg, provider)
    if router is None:
        router = prepare_router(cfg, app, infra, bundle=bundle)
    rules = list(rules) if rules is not None else list(cfg.moe.routergfm.integration.rules)
    integ = integrate_application(cfg, app, infra, router, rules, team_override=team_override)
    result = evaluate_integration(cfg, app, infra, integ)
    result["router"] = str(router.directory) if router.directory is not None else ""
    name = "deploy.json" if team_override is None else f"deploy_team_{stable_hash(sorted(result['team']))}.json"
    save_json_atomic(str(RouterPaths.from_cfg(cfg).deploy_dir(app.key) / name), result)
    metric = result["metric"]
    summary = ", ".join(f"{r} {metric}={m.get(metric, float('nan')):.4f} risk={m['risk']:.4f}" for r, m in result["rules"].items())
    print(f"{_LOG} {app.key}: team={result['team']} | {summary}", flush=True)
    return result


__all__ = [
    "DeployRouter",
    "Integration",
    "deploy_application",
    "evaluate_integration",
    "integrate_application",
    "load_router_bundle",
    "local_evidence",
    "prepare_router",
    "router_directory",
    "router_seed",
    "score_pool",
]
