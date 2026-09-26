"""Metadata MLP selection baseline (App. C, Table 9).

An MLP on concatenated application and expert metadata predicts the
application-average routing loss mu_bar_{b,e} (Eq. 2) and is trained on the
historical evaluations of A_tr(a) (every declared application outside the
target's base dataset). Inputs are the raw metadata vectors RouterGFM's
type-specific projections consume (``infra.app_metadata`` /
``infra.expert_metadata``: text ⊕ numeric), z-scored on the training
applications / experts. Loss: application-balanced Huber (Eq. 9 with
lambda_r = 0); early stopping on validation regret@K over held-out base
datasets. The target contributes metadata only: no labels, no expert runs.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common import AppSpec
from .common import ColumnScaler, SelectionOutcome, application_averages


class MetadataMLP(nn.Module):
    """``softplus(MLP([x_app ; x_expert]))`` with ``num_layers`` hidden ReLU layers.

    The first layer is split into an application and an expert block (the same
    function class as one Linear on the concatenation), so pair batches can
    index precomputed per-application / per-expert projections.
    """

    def __init__(self, app_dim: int, expert_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.app_in = nn.Linear(app_dim, hidden_dim)
        self.expert_in = nn.Linear(expert_dim, hidden_dim, bias=False)
        layers: List[nn.Module] = [nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(int(num_layers) - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, 1))
        self.body = nn.Sequential(*layers)

    def forward(
        self,
        x_app: torch.Tensor,
        x_expert: torch.Tensor,
        app_index: Optional[torch.Tensor] = None,
        expert_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``[N]`` non-negative mu_hat; rows pair ``x_app[app_index]`` with ``x_expert[expert_index]``."""
        h_app, h_exp = self.app_in(x_app), self.expert_in(x_expert)
        if app_index is not None:
            h_app = h_app[app_index]
        if expert_index is not None:
            h_exp = h_exp[expert_index]
        return F.softplus(self.body(h_app + h_exp)).squeeze(-1)


def application_balanced_huber(mu_hat: torch.Tensor, mu_bar: torch.Tensor, app_index: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Mean over applications of the per-application mean Huber loss."""
    per_pair = F.huber_loss(mu_hat, mu_bar, reduction="none", delta=float(delta))
    n_apps = int(app_index.max()) + 1
    sums = torch.zeros(n_apps, device=mu_hat.device, dtype=per_pair.dtype).index_add(0, app_index, per_pair)
    counts = torch.zeros(n_apps, device=mu_hat.device, dtype=per_pair.dtype).index_add(0, app_index, torch.ones_like(per_pair))
    present = counts > 0
    return (sums[present] / counts[present]).mean()


def validation_regret_at_k(mu_hat: torch.Tensor, mu_bar: torch.Tensor, app_index: torch.Tensor, k: int) -> float:
    """Mean over applications of ``min_{top-k by mu_hat} mu_bar - min mu_bar`` (evaluated experts only)."""
    regrets = []
    for a in torch.unique(app_index).tolist():
        sel = app_index == a
        pred, true = mu_hat[sel], mu_bar[sel]
        top = torch.argsort(pred, stable=True)[: int(k)]
        regrets.append(float(true[top].min() - true.min()))
    return sum(regrets) / len(regrets)


@dataclass
class PairTable:
    """Observed historical evaluations: pair p is ``(apps[app_index[p]], expert_ids[expert_index[p]])``."""

    apps: List[AppSpec]
    expert_ids: List[str]
    app_index: torch.Tensor
    expert_index: torch.Tensor
    mu_bar: torch.Tensor

    @property
    def group(self) -> List[str]:
        return [a.group for a in self.apps]


def build_pair_table(infra, apps: Sequence[AppSpec]) -> PairTable:
    """Every ``(b, e)`` with a valid Eq. 2 average; experts in catalog order, unobserved ones dropped."""
    catalog_ids = [s.expert_id for s in infra.catalog]
    mu, count = application_averages(infra, apps, catalog_ids)
    keep = (count > 0).any(dim=0)
    expert_ids = [e for e, k in zip(catalog_ids, keep.tolist()) if k]
    mu, count = mu[:, keep], count[:, keep]
    app_index, expert_index = (count > 0).nonzero(as_tuple=True)
    return PairTable(list(apps), expert_ids, app_index, expert_index, mu[app_index, expert_index].float())


def _validation_groups(groups: Sequence[str], frac: float, seed: int) -> set:
    """Held-out base datasets (at least one, never all); none when there is a single group."""
    unique = sorted(set(groups))
    if len(unique) < 2:
        return set()
    n_val = min(max(1, int(round(frac * len(unique)))), len(unique) - 1)
    perm = torch.randperm(len(unique), generator=torch.Generator().manual_seed(int(seed)))
    return {unique[i] for i in perm[:n_val].tolist()}


@dataclass
class _Fit:
    model: MetadataMLP
    app_scaler: ColumnScaler
    expert_scaler: ColumnScaler
    expert_ids: List[str]
    best_epoch: int
    val_regret: float
    n_train_pairs: int
    n_val_apps: int
    n_historical_apps: int


class MetadataMLPSelector:
    """Rank E_a by ascending predicted mu_hat(a, e); one model per (target base dataset, seed)."""

    name = "metadata_mlp"

    def __init__(self, cfg, infra):
        self.cfg = cfg
        self.infra = infra
        self.mcfg = cfg.moe.routergfm.baselines.metadata_mlp
        self.topk = int(cfg.moe.routergfm.baselines.topk)
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        self._fits: Dict[Tuple[str, int], _Fit] = {}
        self._app_x: Dict[str, torch.Tensor] = {}
        self._expert_x: Dict[str, torch.Tensor] = {}

    def _app_features(self, apps: Sequence[AppSpec]) -> torch.Tensor:
        for a in apps:
            if a.key not in self._app_x:
                self._app_x[a.key] = self.infra.app_metadata(a).float()
        return torch.stack([self._app_x[a.key] for a in apps])

    def _expert_features(self, expert_ids: Sequence[str]) -> torch.Tensor:
        for e in expert_ids:
            if e not in self._expert_x:
                self._expert_x[e] = self.infra.expert_metadata(e).float()
        return torch.stack([self._expert_x[e] for e in expert_ids])

    def fit_for(self, app: AppSpec) -> _Fit:
        key = (app.group, int(app.seed))
        if key in self._fits:
            return self._fits[key]
        m = self.mcfg
        history = [b for b in self.infra.historical_applications(app) if b.group != app.group]
        table = build_pair_table(self.infra, history)
        if table.mu_bar.numel() == 0:
            raise ValueError(f"{app.key}: no historical evaluations outside group {app.group!r}.")
        x_app_raw = self._app_features(table.apps)
        x_exp_raw = self._expert_features(table.expert_ids)
        app_scaler = ColumnScaler("standard").fit(x_app_raw)
        expert_scaler = ColumnScaler("standard").fit(x_exp_raw)
        dev = self.device
        x_app = app_scaler.transform(x_app_raw).to(dev)
        x_exp = expert_scaler.transform(x_exp_raw).to(dev)
        ai, ei, mu = table.app_index.to(dev), table.expert_index.to(dev), table.mu_bar.to(dev)

        val_groups = _validation_groups(table.group, float(m.val_group_frac), app.seed)
        is_val = torch.tensor([table.apps[i].group in val_groups for i in table.app_index.tolist()], device=dev)
        tr, va = ~is_val, is_val
        delta = float(m.huber_delta)
        best_crit, best_state, best_epoch, bad = (float("inf"), float("inf")), None, int(m.epochs), 0
        with torch.random.fork_rng(devices=[dev.index or 0] if dev.type == "cuda" else []):
            torch.manual_seed(int(app.seed))
            model = MetadataMLP(x_app.size(1), x_exp.size(1), int(m.hidden_dim), int(m.num_layers), float(m.dropout)).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=float(m.lr), weight_decay=float(m.weight_decay))
            for epoch in range(1, int(m.epochs) + 1):
                model.train()
                opt.zero_grad()
                loss = application_balanced_huber(model(x_app, x_exp, ai[tr], ei[tr]), mu[tr], ai[tr], delta)
                loss.backward()
                opt.step()
                if not bool(va.any()):
                    continue
                model.eval()
                with torch.no_grad():
                    pred = model(x_app, x_exp, ai[va], ei[va])
                    crit = (
                        validation_regret_at_k(pred, mu[va], ai[va], self.topk),
                        float(application_balanced_huber(pred, mu[va], ai[va], delta)),
                    )
                if crit < best_crit:
                    best_crit, best_state, best_epoch, bad = crit, copy.deepcopy(model.state_dict()), epoch, 0
                else:
                    bad += 1
                    if bad >= int(m.patience):
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        fit = _Fit(
            model=model,
            app_scaler=app_scaler,
            expert_scaler=expert_scaler,
            expert_ids=table.expert_ids,
            best_epoch=best_epoch,
            val_regret=best_crit[0] if best_state is not None else float("nan"),
            n_train_pairs=int(tr.sum()),
            n_val_apps=len({int(i) for i in table.app_index[is_val.cpu()].tolist()}),
            n_historical_apps=len(table.apps),
        )
        self._fits[key] = fit
        return fit

    def rank(self, app: AppSpec) -> SelectionOutcome:
        started = time.perf_counter()
        fit = self.fit_for(app)
        pool = self.infra.compatible_pool(app)
        x_app = fit.app_scaler.transform(self.infra.app_metadata(app).float()[None]).to(self.device)
        x_exp = fit.expert_scaler.transform(self._expert_features(pool)).to(self.device)
        with torch.no_grad():
            mu_hat = fit.model(x_app, x_exp, torch.zeros(len(pool), dtype=torch.long, device=self.device)).cpu()
        order = sorted(range(len(pool)), key=lambda j: (float(mu_hat[j]), j))  # ties: catalog order
        trained = set(fit.expert_ids)
        return SelectionOutcome.from_ranking(
            app,
            [(pool[j], -float(mu_hat[j])) for j in order],
            self.topk,
            num_target_executions=0,
            wall_time_sec=time.perf_counter() - started,
            extras={
                "best_epoch": fit.best_epoch,
                "val_regret": fit.val_regret,
                "n_train_pairs": fit.n_train_pairs,
                "n_val_apps": fit.n_val_apps,
                "n_historical_apps": fit.n_historical_apps,
                "n_unseen_experts": sum(e not in trained for e in pool),
            },
        )


__all__ = [
    "MetadataMLP",
    "MetadataMLPSelector",
    "PairTable",
    "application_balanced_huber",
    "build_pair_table",
    "validation_regret_at_k",
]
