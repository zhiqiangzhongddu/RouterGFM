"""Router learning with application-masked episodes (paper Sec. 3.5, Alg. 1 l.5-12, App. B.3).

An episode treats a historical training application b as new: the evaluation
edges (and reverses) of every application in b's group are removed before
encoding, and b's group is excluded from retrieval, so b's recorded losses act
only as supervision. One update minimizes Eq. 11 averaged over
``episodes_per_step`` episodes, ``L_glob(b) + lambda_l L_loc(b)``, where
``L_glob`` is the mean Huber loss plus ``lambda_r`` ListMLE on the application
averages of Omega_b (Eq. 9) and ``L_loc`` the squared error of the centered
local estimates (Eq. 7) on sampled (x, e) pairs of D_b x Omega_b (Eq. 10).
Retrieval searches record keys refreshed at the start of every update without
gradients; the kernel weights of the retrieved records are differentiable.
The checkpoint is selected on validation applications of held-out groups, and
rho / tau / the kernel bandwidth h on the mixture routing risk of their stored
diagnostic predictions.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from src.utils.checkpoint import cfg_to_dict, save_json_atomic, save_torch_atomic

from ..applications import derive_seed, instance_set_key
from ..archive import Archive, build_archive
from ..common import (
    AppSpec,
    CompatKey,
    RouterPaths,
    base_group,
    enumerate_applications,
    parse_dataset_spec,
    stable_hash,
)
from ..context_graph import APP, EDGE_DIM, EXPERT, RELATIONS, ContextGraph, build_context_graph, masked_edges
from ..descriptors import DescriptorStandardizer, descriptors_at, ensure_descriptors
from ..history import app_normalizer, instance_losses
from .model import RouterGFMModel
from .objectives import huber, listmle, local_sq
from .retrieval import allowed_records, kernel_weights, local_estimate, mixture_weights, search

BUNDLE_FILE = "bundle.pt"
LOG_FILE = "log.json"
_KEY_CHUNK = 1 << 16  # archive records per key-network pass


def _device(cfg) -> torch.device:
    return torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")


def _group_of(name: str) -> str:
    """Group of ``"dataset"`` or ``"dataset:level"``."""
    return base_group(parse_dataset_spec(name)[0])


def router_run_key(target_group: str, budget: int, seed: int) -> str:
    """``RouterPaths.router_dir`` key of the router trained for (target group, budget, seed)."""
    return f"{target_group}__b{int(budget)}__s{int(seed)}"


# Bookkeeping and deploy-time keys; the router seed is part of the run key.
_UNHASHED = {"router": ("skip_if_exists", "per_seed", "seed"), "archive": ("perturbation", "perturbation_seed")}


def router_cfg_hash(cfg) -> str:
    """Hash of the config a router bundle is trained under (``meta['cfg_hash']``)."""
    rg = cfg.moe.routergfm
    blocks = {k: cfg_to_dict(rg[k]) for k in ("router", "descriptors", "archive", "graph", "heads", "loss")}
    for block, keys in _UNHASHED.items():
        for key in keys:
            blocks[block].pop(key, None)
    return stable_hash(blocks)


def reusable_bundle(cfg, directory: Path) -> bool:
    """True when ``router.skip_if_exists`` and *directory* holds a bundle trained under the current config.

    A bundle trained under another config raises instead of being reused
    silently (or retrained implicitly): result rows are stamped with the
    current config.
    """
    path = Path(directory) / BUNDLE_FILE
    if not (bool(cfg.moe.routergfm.router.skip_if_exists) and path.is_file()):
        return False
    stored, current = torch.load(str(path), map_location="cpu")["meta"].get("cfg_hash"), router_cfg_hash(cfg)
    if stored != current:
        raise ValueError(
            f"Router bundle at {directory} was trained under another config (cfg_hash {stored} != {current}); "
            "use another output_root or set router.skip_if_exists False."
        )
    return True


@dataclass
class RouterBundle:
    """A trained router and what deployment needs to rebuild H and M around it."""

    model: RouterGFMModel  # eval mode, on the load device
    standardizer: DescriptorStandardizer  # fitted on the training applications' diagnostic descriptors
    numeric_stats: Dict[str, Dict[str, torch.Tensor]]  # context-graph numeric standardization (build_context_graph)
    rho: float
    tau: float
    bandwidth: float  # h of Eq. 6 at deployment (validation-selected unless router.select_bandwidth is False)
    catalog_ids: List[str]
    train_apps: List[AppSpec]
    val_apps: List[AppSpec]
    graph_apps: List[AppSpec]  # applications of H (training + validation groups)
    archive_apps: List[AppSpec]  # source applications of M
    router_cfg: Dict[str, Any]  # cfg.moe.routergfm.router at training time
    log: Dict[str, Any]
    meta: Dict[str, Any]  # target_group, budget, seed, run_key, val_groups, cfg_hash


@dataclass
class _Episode:
    app: AppSpec
    node: int  # application node of H
    omega: torch.Tensor  # [K_b] expert nodes with an evaluation edge (Omega_b)
    mu_bar: torch.Tensor  # [K_b] recorded application averages (Eq. 2)
    cols: torch.Tensor  # [K_b] columns of the data key's loss matrix
    compat: Tuple[str, int, str]


class RouterTrainer:
    """Episodic training of the scorer (Eq. 4) and key network (Eq. 6) on H and M.

    ``graph`` and ``archive`` cover the training and validation applications
    (never the deployment target's group); ``descriptors`` maps a data key to
    its raw descriptor cache; ``provider`` supplies the validation
    applications' diagnostic labels for rho / tau / bandwidth selection.
    """

    def __init__(
        self,
        cfg,
        catalog: Sequence,
        apps_train: Sequence[AppSpec],
        apps_val: Sequence[AppSpec],
        store,
        graph: ContextGraph,
        archive: Archive,
        standardizer: DescriptorStandardizer,
        descriptors: Mapping[str, Dict[str, Any]],
        device=None,
        *,
        provider=None,
        seed: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg
        self.rt = cfg.moe.routergfm.router
        self.device = torch.device(device) if device is not None else _device(cfg)
        self.seed = int(self.rt.seed if seed is None else seed)
        self.catalog = list(catalog)
        self.apps_train, self.apps_val = list(apps_train), list(apps_val)
        self.store, self.standardizer, self.descriptors, self.provider = store, standardizer, descriptors, provider
        self.meta = dict(meta or {})
        self.graph = graph.to(self.device)

        # Archive experts are catalog indices; keys need their expert node of H.
        node_of = torch.full((len(self.catalog),), -1, dtype=torch.long)
        node_of[torch.as_tensor(graph.expert_catalog_index, dtype=torch.long)] = torch.arange(
            len(graph.expert_catalog_index)
        )
        rec_node = node_of[archive.expert.cpu()]
        self.archive = archive.subset(rec_node >= 0).to(self.device)
        self._rec_node = rec_node[rec_node >= 0].to(self.device)
        self._residual = self.archive.residual

        self.desc_dim = int(standardizer.mean.numel())
        torch.manual_seed(self.seed)
        self.model = RouterGFMModel(self.graph.in_dims, RELATIONS, self.rt, self.desc_dim, edge_dim=EDGE_DIM).to(
            self.device
        )

        self._loss: Dict[str, torch.Tensor] = {}  # data key -> [|D|, E] recorded losses (NaN invalid)
        self._columns: Dict[str, Dict[str, int]] = {}
        self._z: Dict[str, torch.Tensor] = {}  # data key -> [|D|, D] standardized descriptors
        self._diag_pos: Dict[str, torch.Tensor] = {}
        self._family: Dict[str, str] = {}
        self._train = [ep for ep in map(self._episode, self.apps_train) if ep is not None]
        self._val = [ep for ep in map(self._episode, self.apps_val) if ep is not None]
        if not self._train or not self._val:
            raise ValueError("Router training needs training and validation applications with evaluated experts.")
        self._episodes = {ep.app.key: ep for ep in self._train + self._val}
        # Fixed validation pairs keep the criterion comparable across epochs.
        self._val_pairs = {
            ep.app.key: self._sample_pairs(
                ep, torch.Generator().manual_seed(derive_seed(self.seed, "val_pairs", ep.app.key))
            )
            for ep in self._val
        }
        self.rho = float(self.rt.rho) if float(self.rt.rho) >= 0 else None
        self.tau = float(self.rt.tau) if float(self.rt.tau) >= 0 else None
        self.bandwidth = None if bool(self.rt.select_bandwidth) else float(self.rt.bandwidth)
        self.log: Dict[str, Any] = {}

    # -- episodes --------------------------------------------------------------
    def _episode(self, app: AppSpec) -> Optional[_Episode]:
        graph = self.graph
        node = graph.app_index[app.key]
        mask = graph.eval_app == node
        omega = graph.eval_expert[mask]
        if omega.numel() == 0:
            return None
        key = app.data_key
        if key not in self._loss:
            ids, loss = self.store.loss_matrix(key)
            matrix = self.store.matrix(key)
            self._loss[key] = loss.to(self.device)
            self._columns[key] = {eid: i for i, eid in enumerate(ids)}
            z = descriptors_at(self.descriptors[key], matrix["diag_pos"])
            self._z[key] = self.standardizer.transform(z).to(self.device)
            self._diag_pos[key] = matrix["diag_pos"].clone()
            self._family[key] = str(matrix["family"])
        cols = [self._columns[key][graph.expert_ids[e]] for e in omega.tolist()]
        return _Episode(
            app=app,
            node=node,
            omega=omega,
            mu_bar=graph.eval_mu[mask],
            cols=torch.tensor(cols, dtype=torch.long, device=self.device),
            compat=CompatKey(self._family[key], app.budget).as_tuple(),
        )

    def _sample_pairs(self, ep: _Episode, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """(row of D_b, index into Omega_b) pairs with a valid recorded loss: all of them, or
        ``local_pairs_per_episode`` drawn uniformly with replacement."""
        valid = torch.isfinite(self._loss[ep.app.data_key][:, ep.cols])
        flat = valid.reshape(-1).nonzero().view(-1)
        num = int(self.rt.local_pairs_per_episode)
        if flat.numel() > num:
            flat = flat[torch.randint(flat.numel(), (num,), generator=generator).to(flat.device)]
        k = ep.cols.numel()
        return flat // k, flat % k

    def _allowed(self, ep: _Episode) -> torch.Tensor:
        """Records of other groups with the same CompatKey; the episode's group is hidden."""
        a = self.archive
        return allowed_records(
            a.app, a.group, a.compat, group=ep.app.group, compat=ep.compat, hidden_groups=(ep.app.group,)
        )

    def _encode(self, group: str) -> Dict[str, torch.Tensor]:
        """Eq. 3 on H without the evaluation edges (and reverses) of ``group``."""
        return self.model.encode(self.graph.x, *masked_edges(self.graph, {group}))

    def _scores(self, ep: _Episode, h: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``mu_hat_{b,e}`` for e in Omega_b (Eq. 4)."""
        return self.model.score(h[APP][ep.node].expand(ep.omega.numel(), -1), h[EXPERT][ep.omega])

    @torch.no_grad()
    def _record_keys(self) -> torch.Tensor:
        """``k_phi(c_i, v_{e_i})`` of every archive record with the current parameters."""
        v = self.model.project(self.graph.x)[EXPERT]
        rep, nodes = self.archive.rep, self._rec_node
        chunks = range(0, len(self.archive), _KEY_CHUNK)
        return torch.cat([self.model.keys(rep[s:s + _KEY_CHUNK], v[nodes[s:s + _KEY_CHUNK]]) for s in chunks])

    def _neighbours(
        self, z: torch.Tensor, nodes: torch.Tensor, v: torch.Tensor, record_keys: torch.Tensor, allowed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Query keys ``[P, d_k]``, keys ``[P, J, d_k]`` and validity ``[P, J]`` of the records retrieved for
        (z_p, e_p), and their centered residuals; the top-J search does not depend on the bandwidth."""
        query = self.model.keys(z, v[nodes])
        idx, valid = search(
            query.detach(), record_keys, self.archive.app, allowed, int(self.rt.retrieval_j), int(self.rt.per_app_cap)
        )
        flat = idx.reshape(-1)
        selected = self.model.keys(self.archive.rep[flat], v[self._rec_node[flat]]).view(*idx.shape, -1)
        return query, selected, valid, self._residual[idx]

    def _retrieve(
        self, z: torch.Tensor, nodes: torch.Tensor, v: torch.Tensor, record_keys: torch.Tensor, allowed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eq. 6 weights ``[P, J]`` at the training bandwidth and centered residuals of the retrieved records."""
        query, selected, valid, residual = self._neighbours(z, nodes, v, record_keys, allowed)
        return kernel_weights(query, selected, valid, float(self.rt.bandwidth)), residual

    def _losses(
        self, ep: _Episode, rows: torch.Tensor, cols: torch.Tensor, record_keys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(L_glob, L_loc) of one episode (Eq. 9-10) on the given pairs."""
        rt = self.rt
        mu = self._scores(ep, self._encode(ep.app.group))
        glob = huber(mu, ep.mu_bar, float(rt.huber_delta)) + float(rt.lambda_rank) * listmle(-mu, ep.mu_bar)
        if rows.numel() == 0:
            return glob, mu.sum() * 0.0
        key = ep.app.data_key
        v = self.model.project(self.graph.x)[EXPERT]  # v_e = h^(0)_e, before evaluation-edge messages
        w, residual = self._retrieve(self._z[key][rows], ep.omega[cols], v, record_keys, self._allowed(ep))
        r_hat = local_estimate(mu[cols], w, residual, float(rt.rho_train))
        return glob, local_sq(r_hat, self._loss[key][rows, ep.cols[cols]], expert=cols)

    def episode_losses(
        self, app: AppSpec, pairs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(L_glob, L_loc) of a training/validation application with fresh record keys.

        ``pairs`` = (rows of D_b, indices into Omega_b); default: every valid pair
        (exact Eq. 10). Runs in the model's current mode, with gradients.
        """
        ep = self._episodes[app.key]
        if pairs is None:
            valid = torch.isfinite(self._loss[app.data_key][:, ep.cols]).nonzero()
            pairs = (valid[:, 0], valid[:, 1])
        return self._losses(ep, pairs[0], pairs[1], self._record_keys())

    @torch.no_grad()
    def mu_hat(self, app: AppSpec) -> Tuple[torch.Tensor, torch.Tensor]:
        """(Omega_a expert nodes, mu_hat) of a training/validation application with its group masked."""
        self.model.eval()
        ep = self._episodes[app.key]
        return ep.omega.clone(), self._scores(ep, self._encode(app.group))

    # -- optimization ------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Validation criterion: mean over validation episodes of L_glob + lambda_l L_loc (fixed pairs)."""
        self.model.eval()
        record_keys = self._record_keys()
        terms = [self._losses(ep, *self._val_pairs[ep.app.key], record_keys) for ep in self._val]
        glob = float(torch.stack([g for g, _ in terms]).mean())
        loc = float(torch.stack([l for _, l in terms]).mean())
        return {"loss": glob + float(self.rt.lambda_local) * loc, "glob": glob, "loc": loc}

    def fit(self) -> Dict[str, Any]:
        """Alg. 1 l.6-12: episodic updates (Eq. 11), early stopping on the validation criterion.

        Restores the best validation state and returns the training log.
        """
        rt = self.rt
        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(derive_seed(self.seed, "router_episodes"))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(rt.lr), weight_decay=float(rt.weight_decay))
        per_step, lam = max(1, int(rt.episodes_per_step)), float(rt.lambda_local)
        best, best_epoch, best_state = math.inf, 0, copy.deepcopy(self.model.state_dict())
        epochs: List[Dict[str, float]] = []
        for epoch in range(1, int(rt.epochs) + 1):
            self.model.train()
            order = torch.randperm(len(self._train), generator=generator).tolist()
            totals = torch.zeros(3)
            for start in range(0, len(order), per_step):
                batch = [self._train[i] for i in order[start:start + per_step]]
                optimizer.zero_grad(set_to_none=True)
                record_keys = self._record_keys()  # refreshed once per update
                for ep in batch:
                    glob, loc = self._losses(ep, *self._sample_pairs(ep, generator), record_keys)
                    total = glob + lam * loc
                    # Backward per episode (one masked graph in memory); accumulates the step mean.
                    (total / len(batch)).backward()
                    totals += torch.stack([total, glob, loc]).detach().cpu()
                optimizer.step()
            train = (totals / len(order)).tolist()
            val = self.evaluate()
            epochs.append({
                "epoch": epoch, "train_loss": train[0], "train_glob": train[1], "train_loc": train[2],
                "val_loss": val["loss"], "val_glob": val["glob"], "val_loc": val["loc"],
            })
            print(f"[RouterGFM router] epoch {epoch}: train {train[0]:.5f} val {val['loss']:.5f}", flush=True)
            if val["loss"] < best:
                best, best_epoch, best_state = val["loss"], epoch, copy.deepcopy(self.model.state_dict())
            elif epoch - best_epoch >= int(rt.patience):
                break
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.log.update({
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best,
            "num_train_apps": len(self._train),
            "num_val_apps": len(self._val),
            "num_records": len(self.archive),
        })
        return self.log

    # -- rho / tau / bandwidth ---------------------------------------------------------
    @torch.no_grad()
    def _mixture_risks(
        self,
        ep: _Episode,
        bandwidths: Sequence[float],
        rhos: Sequence[float],
        taus: Sequence[float],
        record_keys: torch.Tensor,
    ) -> torch.Tensor:
        """Mean mixture routing loss on D_v of the top-K team for every (h, rho, tau):
        ``[len(bandwidths), len(rhos), len(taus)]``. One search per (instance, team expert) serves every h."""
        key = ep.app.data_key
        data = self.provider.load(ep.app)
        if not torch.equal(torch.as_tensor(data.diag_pos).cpu(), self._diag_pos[key]):
            raise ValueError(f"{key}: diagnostic positions differ from the history records (stale history).")
        mu = self._scores(ep, self._encode(ep.app.group))
        k = min(int(self.rt.topk), mu.numel())
        order = torch.argsort(mu, stable=True)[:k]  # T_v: K lowest mu_hat (Eq. 4)
        team, mu_team = ep.omega[order], mu[order]
        z = self._z[key]
        n = z.size(0)
        v = self.model.project(self.graph.x)[EXPERT]
        query, selected, valid, residual = self._neighbours(
            z.repeat_interleave(k, 0), team.repeat(n), v, record_keys, self._allowed(ep)
        )
        residual = residual.view(n, k, -1)
        preds = self.store.pred_matrix(key, [self.graph.expert_ids[e] for e in team.tolist()]).to(self.device)
        normalizer = app_normalizer(self.cfg, data)
        out = torch.zeros(len(bandwidths), len(rhos), len(taus))
        for b, h in enumerate(bandwidths):
            w = kernel_weights(query, selected, valid, float(h)).view(n, k, -1)  # Eq. 6
            for i, rho in enumerate(rhos):
                r_hat = local_estimate(mu_team.expand(n, k), w, residual, float(rho))  # Eq. 7
                for j, tau in enumerate(taus):
                    alpha = mixture_weights(r_hat, float(tau))  # Eq. 8
                    mixed = (alpha.unsqueeze(-1) * preds).sum(dim=1)  # Eq. 1
                    loss = instance_losses(self.cfg, mixed.cpu(), data.labels["diag"], data.task_family, normalizer)
                    out[b, i, j] = loss[torch.isfinite(loss)].mean()
        return out

    @torch.no_grad()
    def select_integration_params(self) -> Tuple[float, float, float]:
        """(rho, tau, bandwidth): the configured rho / tau if >= 0 and ``router.bandwidth`` unless
        ``router.select_bandwidth``; the rest jointly at the grid point minimizing the mean validation
        mixture risk (``bandwidth_grid`` x ``rho_grid`` x ``tau_grid``)."""
        rt = self.rt
        rhos = [float(rt.rho)] if float(rt.rho) >= 0 else [float(r) for r in rt.rho_grid]
        taus = [float(rt.tau)] if float(rt.tau) >= 0 else [float(t) for t in rt.tau_grid]
        select_h = bool(rt.select_bandwidth)
        bandwidths = [float(h) for h in rt.bandwidth_grid] if select_h else [float(rt.bandwidth)]
        if float(rt.rho) >= 0 and float(rt.tau) >= 0 and not select_h:
            self.rho, self.tau, self.bandwidth = rhos[0], taus[0], bandwidths[0]
            return self.rho, self.tau, self.bandwidth
        if self.provider is None:
            raise ValueError(
                "Selecting rho/tau/bandwidth needs a provider for the validation applications' diagnostic labels."
            )
        self.model.eval()
        record_keys = self._record_keys()
        risks = [self._mixture_risks(ep, bandwidths, rhos, taus, record_keys) for ep in self._val]
        risk = torch.stack(risks).mean(dim=0)
        b, rest = divmod(int(torch.argmin(risk)), len(rhos) * len(taus))
        i, j = divmod(rest, len(taus))
        self.rho, self.tau, self.bandwidth = rhos[i], taus[j], bandwidths[b]
        self.log["integration_grid"] = [
            {"bandwidth": h, "rho": r, "tau": t, "risk": float(risk[c, a, d])}
            for c, h in enumerate(bandwidths) for a, r in enumerate(rhos) for d, t in enumerate(taus)
        ]
        self.log.update({"rho": self.rho, "tau": self.tau, "bandwidth": self.bandwidth})
        return self.rho, self.tau, self.bandwidth

    # -- bundle --------------------------------------------------------------------------
    def save(self, directory) -> Path:
        """Write ``bundle.pt`` (model, standardizer, graph numeric stats, rho, tau, h, apps, log) and ``log.json``."""
        if self.rho is None or self.tau is None or self.bandwidth is None:
            raise RuntimeError("rho/tau/bandwidth are not set: call select_integration_params() before save().")
        directory = Path(directory)
        cpu = lambda value: value.detach().cpu() if torch.is_tensor(value) else value  # noqa: E731
        payload = {
            "model_state": {k: cpu(v) for k, v in self.model.state_dict().items()},
            "in_dims": dict(self.graph.in_dims),
            "desc_dim": self.desc_dim,
            "relations": [list(r) for r in self.model.relations],
            "edge_dim": EDGE_DIM,
            "router_cfg": cfg_to_dict(self.rt),
            "standardizer": {k: cpu(v) for k, v in self.standardizer.state_dict().items()},
            "numeric_stats": {t: {k: cpu(v) for k, v in s.items()} for t, s in self.graph.numeric_stats.items()},
            "rho": float(self.rho),
            "tau": float(self.tau),
            "bandwidth": float(self.bandwidth),
            "catalog_ids": [spec.expert_id for spec in self.catalog],
            "train_apps": [a.to_dict() for a in self.apps_train],
            "val_apps": [a.to_dict() for a in self.apps_val],
            "graph_apps": [n.to_dict() for n in self.graph.app_nodes if isinstance(n, AppSpec)],
            "archive_apps": [a.to_dict() for a in self.archive.apps],
            "log": self.log,
            "meta": self.meta,
        }
        save_torch_atomic(str(directory / BUNDLE_FILE), payload)
        save_json_atomic(
            str(directory / LOG_FILE),
            {"meta": self.meta, "rho": self.rho, "tau": self.tau, "bandwidth": self.bandwidth, **self.log},
        )
        return directory

    @classmethod
    def load(cls, directory, cfg, device=None) -> RouterBundle:
        """Read a bundle written by :meth:`save`; the model is rebuilt from the stored router config."""
        payload = torch.load(str(Path(directory) / BUNDLE_FILE), map_location="cpu")
        device = torch.device(device) if device is not None else _device(cfg)
        model = RouterGFMModel(
            payload["in_dims"],
            [tuple(r) for r in payload["relations"]],
            SimpleNamespace(**payload["router_cfg"]),
            payload["desc_dim"],
            edge_dim=payload["edge_dim"],
        )
        model.load_state_dict(payload["model_state"])
        apps = lambda name: [AppSpec.from_dict(d) for d in payload[name]]  # noqa: E731
        return RouterBundle(
            model=model.to(device).eval(),
            standardizer=DescriptorStandardizer().load_state_dict(payload["standardizer"]),
            numeric_stats=payload["numeric_stats"],
            rho=float(payload["rho"]),
            tau=float(payload["tau"]),
            bandwidth=float(payload["bandwidth"]),
            catalog_ids=list(payload["catalog_ids"]),
            train_apps=apps("train_apps"),
            val_apps=apps("val_apps"),
            graph_apps=apps("graph_apps"),
            archive_apps=apps("archive_apps"),
            router_cfg=dict(payload["router_cfg"]),
            log=payload["log"],
            meta=payload["meta"],
        )


# --------------------------------------------------------------------------- #
# Leave-one-dataset-out router for one target group
# --------------------------------------------------------------------------- #
def validation_groups(
    router_cfg,
    target_group: str,
    apps: Sequence[AppSpec],
    families: Mapping[str, str],
    target_families: set,
) -> List[str]:
    """Held-out validation groups.

    ``router.val_datasets`` (minus the target group) if given, else up to
    ``num_val_datasets`` groups in declaration order, groups with an
    application of a target task family first. An automatic choice never
    takes the last training group of a target task family (e.g. QM9 for a
    QM7b target), so the router is always trained on the target's loss family
    when history has one.
    """
    available = list(dict.fromkeys(a.group for a in apps))
    configured = [g for g in dict.fromkeys(_group_of(n) for n in router_cfg.val_datasets) if g != target_group]
    if configured:
        missing = [g for g in configured if g not in available]
        if missing:
            raise ValueError(f"Validation datasets without historical applications: {missing}")
        groups = configured
    else:
        family_groups = {f: {a.group for a in apps if families[a.key] == f} for f in target_families}
        shared = set().union(*family_groups.values()) if family_groups else set()
        groups = []
        for g in sorted(available, key=lambda g: g not in shared):
            if len(groups) == int(router_cfg.num_val_datasets):
                break
            if all(members - set(groups) - {g} for members in family_groups.values() if members):
                groups.append(g)
    if not groups or len(groups) >= len(available):
        raise ValueError(f"Cannot split {available} into training and validation groups (validation: {groups}).")
    return groups


def build_router_trainer(
    cfg,
    target_group: str,
    budget: int,
    provider=None,
    seed: Optional[int] = None,
    device=None,
    *,
    exclude_experts: Sequence[str] = (),
) -> RouterTrainer:
    """RouterTrainer for one target group and budget, built from the history of every other group.

    Training applications are the declared applications with history outside
    the target and validation groups (all budgets); validation episodes use the
    validation groups' applications at ``budget``. H and M contain the training
    and validation groups, never the target group. The descriptor standardizer
    is fitted on the training applications' diagnostic descriptors only.
    ``exclude_experts`` are left out of the catalog, so H has no node or edge
    of theirs (nor an arch/objective node used only by them) and M no record;
    the applications and the standardizer do not change.
    """
    from ..infra import RouterInfra

    rg = cfg.moe.routergfm
    target_group = _group_of(target_group)
    seed = int(rg.router.seed if seed is None else seed)
    infra = RouterInfra(cfg, provider, device)
    excluded = sorted({str(e) for e in exclude_experts})
    catalog = [s for s in infra.catalog if s.expert_id not in excluded]
    declared = enumerate_applications(rg)
    history = [a for a in declared if a.group != target_group and infra.store.expert_ids(a.data_key)]
    families = {a.key: infra.task_family(a) for a in history}
    target_families = {infra.task_family(a) for a in declared if a.group == target_group}
    val_groups = validation_groups(rg.router, target_group, history, families, target_families)
    apps_train = [a for a in history if a.group not in val_groups]
    val_context = [a for a in history if a.group in val_groups]
    apps_val = [a for a in val_context if a.budget == int(budget)]
    context = apps_train + val_context

    descriptors: Dict[str, Dict[str, Any]] = {}
    by_set: Dict[str, Dict[str, Any]] = {}  # one in-memory cache per instance set (shared by its data keys)
    for app in context:
        if app.data_key not in descriptors:
            cache = by_set.get(instance_set_key(app))
            if cache is None or app.data_key not in cache["data_keys"]:
                cache = by_set[instance_set_key(app)] = ensure_descriptors(cfg, infra.data(app))
            descriptors[app.data_key] = cache
    train_keys = {a.data_key: a for a in apps_train}
    Z = torch.cat([descriptors_at(descriptors[k], infra.data(a).diag_pos) for k, a in train_keys.items()])
    standardizer = DescriptorStandardizer(clip=float(rg.descriptors.clip)).fit(Z)
    archive = build_archive(context, infra.store, standardizer, cfg, catalog=catalog, descriptors=descriptors)
    graph = build_context_graph(
        cfg,
        catalog,
        context,
        infra.store,
        infra.text_encoder,
        {a.key: infra.data(a).stats for a in context},
        families=families,
    )
    meta = {
        "target_group": target_group,
        "budget": int(budget),
        "seed": seed,
        "run_key": router_run_key(target_group, budget, seed),
        "val_groups": list(val_groups),
        "cfg_hash": router_cfg_hash(cfg),
    }
    if excluded:
        meta["hidden_experts"] = excluded
    return RouterTrainer(
        cfg, catalog, apps_train, apps_val, infra.store, graph, archive, standardizer, descriptors,
        infra.device, provider=infra.provider, seed=seed, meta=meta,
    )


def train_router(cfg, target_group: str, budget: int, provider=None, seed: Optional[int] = None) -> Path:
    """Train (or reuse, with ``router.skip_if_exists``) the router of (target group, budget, seed).

    ``seed`` defaults to ``router.seed``. Returns ``RouterPaths.router_dir(run_key)``
    holding the bundle; load it with :meth:`RouterTrainer.load`.
    """
    rg = cfg.moe.routergfm
    group = _group_of(target_group)
    seed = int(rg.router.seed if seed is None else seed)
    directory = RouterPaths.from_cfg(cfg).router_dir(router_run_key(group, budget, seed))
    if reusable_bundle(cfg, directory):
        print(f"[RouterGFM router] reuse {directory}", flush=True)
        return directory
    trainer = build_router_trainer(cfg, group, budget, provider, seed)
    trainer.fit()
    trainer.select_integration_params()
    return trainer.save(directory)


__all__ = [
    "BUNDLE_FILE",
    "LOG_FILE",
    "RouterBundle",
    "RouterTrainer",
    "build_router_trainer",
    "reusable_bundle",
    "router_cfg_hash",
    "router_run_key",
    "train_router",
    "validation_groups",
]
