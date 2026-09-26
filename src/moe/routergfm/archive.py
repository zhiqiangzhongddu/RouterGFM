"""Local evaluation archive M (paper Sec. 3.4, Eq. 5; App. A.4, B.2, D.2, D.5).

For every historical application b, its standardized diagnostic descriptors
are clustered into cells C_{b,j} shared by all experts. For each expert e with
valid observations in a cell, one record stores the cell representative
c_{b,j}, the local loss r_{b,j,e} (Eq. 5), the application average mu_{b,e},
and the valid count n_{b,j,e}. Counts and averages use exactly the same valid
(finite-loss) observations, so sum_j n_j (mu - r_j) = 0 (App. A.4).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from src.moe.routergfm.common import AppSpec, CompatKey, stable_hash
from src.moe.routergfm.descriptors import DescriptorStandardizer, descriptors_at, ensure_descriptors

PERTURBATIONS = ("none", "half_cells", "missing_family", "reversed", "shuffled")


def _seed(*parts: Any) -> int:
    return int(stable_hash(list(parts)), 16) % (2**31 - 1)


def kmeans(
    embeddings: torch.Tensor,
    num_clusters: int,
    seed: int,
    num_iters: int = 100,
    tol: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lloyd k-means with seeded random initialization (port of the reference
    ``prediction/clustering.py::kmeans``).

    Returns ``(assign[n], centers[k, D])`` with ``k = min(num_clusters, n)``;
    the returned centers are the member means of the returned assignment
    (an empty cluster keeps its previous center).
    """
    if embeddings.numel() == 0:
        return torch.empty(0, dtype=torch.long), embeddings.new_zeros(embeddings.shape)
    x = embeddings.to(torch.float32)
    k = min(int(num_clusters), x.size(0))
    if k <= 0:
        raise ValueError("num_clusters must be positive")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    centers = x[torch.randperm(x.size(0), generator=generator)[:k].to(x.device)].clone()

    def _update(assign: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        sums = torch.zeros_like(centers).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k).to(x.dtype)
        return torch.where(counts[:, None] > 0, sums / counts.clamp(min=1)[:, None], centers)

    for _ in range(int(num_iters)):
        assign = torch.cdist(x, centers).argmin(dim=1)
        new_centers = _update(assign, centers)
        delta = torch.norm(new_centers - centers) / (torch.norm(centers) + 1e-8)
        centers = new_centers
        if delta.item() < tol:
            break
    assign = torch.cdist(x, centers).argmin(dim=1)
    return assign, _update(assign, centers)


def build_cells(Z_std_diag: torch.Tensor, cfg, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cells of one application: ``B_b = min(num_cells, max(1, |D_b| // min_cell_size))``."""
    arc = cfg.moe.routergfm.archive
    num_cells = min(int(arc.num_cells), max(1, Z_std_diag.size(0) // max(int(arc.min_cell_size), 1)))
    return kmeans(Z_std_diag, num_cells, seed, num_iters=int(arc.kmeans_iters))


@dataclass
class Archive:
    """Records ``i`` of M; per-application lists are indexed by ``app``."""

    rep: torch.Tensor  # [R, D] standardized cell representatives c_{b,j}
    expert: torch.Tensor  # [R] catalog index e_i
    app: torch.Tensor  # [R] index into ``apps`` (source application b)
    group: List[str]  # per app: base dataset group
    compat: List[Tuple[str, int, str]]  # per app: CompatKey tuple (task family, budget, norm)
    r_local: torch.Tensor  # [R] local loss r_{b,j,e} (Eq. 5)
    mu_app: torch.Tensor  # [R] application average mu_{b,e} on the same valid observations
    count: torch.Tensor  # [R] valid observations n_{b,j,e}
    cell: torch.Tensor  # [R] cell id j within its application
    family: torch.Tensor  # [R] coarse context family (k-means within the record's compat group)
    apps: List[AppSpec]

    def __len__(self) -> int:
        return int(self.expert.numel())

    @property
    def residual(self) -> torch.Tensor:
        """Centered residual r_i - mu_i transferred by Eq. 7."""
        return self.r_local - self.mu_app

    def _record_fields(self) -> List[str]:
        return [f.name for f in fields(self) if isinstance(getattr(self, f.name), torch.Tensor)]

    def subset(self, mask: torch.Tensor) -> "Archive":
        """Records selected by a boolean mask (or index tensor); app lists are kept."""
        mask = torch.as_tensor(mask, device=self.expert.device)
        return replace(self, **{name: getattr(self, name)[mask] for name in self._record_fields()})

    def to(self, device) -> "Archive":
        return replace(self, **{name: getattr(self, name).to(device) for name in self._record_fields()})

    def records_of_groups(self, groups: Iterable[str]) -> torch.Tensor:
        """Boolean mask of records whose source application's group is in *groups*."""
        groups = set(groups)
        app_mask = torch.tensor([g in groups for g in self.group], dtype=torch.bool)
        return app_mask.to(self.app.device)[self.app]

    def records_with_compat(self, compat: Tuple[str, int, str]) -> torch.Tensor:
        """Boolean mask of records whose source application matches a CompatKey tuple."""
        compat = tuple(compat)
        app_mask = torch.tensor([tuple(c) == compat for c in self.compat], dtype=torch.bool)
        return app_mask.to(self.app.device)[self.app]

    def compat_index(self) -> torch.Tensor:
        """Per-record id of its compat tuple (ids follow first appearance in ``compat``)."""
        ids: Dict[Tuple, int] = {}
        per_app = torch.tensor([ids.setdefault(tuple(c), len(ids)) for c in self.compat], dtype=torch.long)
        return per_app.to(self.app.device)[self.app]


def _catalog_index(catalog: Sequence[Any]) -> Dict[str, int]:
    return {getattr(spec, "expert_id", spec): i for i, spec in enumerate(catalog)}


def build_archive(
    apps: Sequence[AppSpec],
    store: Any,
    standardizer: DescriptorStandardizer,
    cfg,
    *,
    catalog: Optional[Sequence[Any]] = None,
    descriptors: Optional[Mapping[str, Dict[str, Any]]] = None,
    provider: Optional[Any] = None,
) -> Archive:
    """Build M from the historical applications *apps* (Eq. 5).

    ``store.loss_matrix(data_key) -> (expert_ids, loss[n_diag, n_exp])`` with
    NaN for invalid observations; rows follow the records' ``diag_pos`` and
    the task family comes from the records (``store.load(...)``). *catalog*
    (ExpertSpecs or expert ids, default ``build_expert_catalog(cfg)``) fixes
    the expert index; experts outside it are ignored. *descriptors* maps a
    data key to its descriptor cache (default ``ensure_descriptors``).
    """
    if catalog is None:
        from src.moe.routergfm.experts import build_expert_catalog

        catalog = build_expert_catalog(cfg)
    if not apps:
        raise ValueError("build_archive needs at least one historical application")
    expert_index = _catalog_index(catalog)
    rg = cfg.moe.routergfm

    per_key: Dict[str, Tuple] = {}
    groups: List[str] = []
    compats: List[Tuple[str, int, str]] = []
    chunks: Dict[str, List[torch.Tensor]] = {k: [] for k in ("rep", "expert", "app", "r_local", "mu_app", "count", "cell")}
    for a_idx, app in enumerate(apps):
        key = app.data_key
        if key not in per_key:
            expert_ids, loss = store.loss_matrix(key)
            record = store.load(key, expert_ids[0])
            keep = [i for i, eid in enumerate(expert_ids) if eid in expert_index]
            cache = descriptors[key] if descriptors is not None and key in descriptors else ensure_descriptors(cfg, app, provider)
            z_std = standardizer.transform(descriptors_at(cache, record["diag_pos"]))
            assign, centers = build_cells(z_std, cfg, _seed(app.seed, "archive_cells", key))
            loss = torch.as_tensor(loss, dtype=torch.float32)[:, keep]
            valid = torch.isfinite(loss)
            member = torch.nn.functional.one_hot(assign, centers.size(0)).to(torch.float32).t()  # [B, n]
            counts = member @ valid.to(torch.float32)  # [B, K]
            sums = member @ torch.where(valid, loss, torch.zeros_like(loss))
            mu = sums.sum(0) / counts.sum(0).clamp(min=1)  # same valid observations as the cells
            ids = torch.tensor([expert_index[expert_ids[i]] for i in keep], dtype=torch.long)
            per_key[key] = (str(record["family"]), centers, counts, sums, mu, ids)
        family, centers, counts, sums, mu, ids = per_key[key]
        groups.append(app.group)
        compats.append(CompatKey(family, int(app.budget)).as_tuple())
        j, e = torch.nonzero(counts > 0, as_tuple=True)
        chunks["rep"].append(centers[j])
        chunks["expert"].append(ids[e])
        chunks["app"].append(torch.full_like(j, a_idx))
        chunks["r_local"].append(sums[j, e] / counts[j, e])
        chunks["mu_app"].append(mu[e])
        chunks["count"].append(counts[j, e])
        chunks["cell"].append(j)

    tensors = {name: torch.cat(parts) for name, parts in chunks.items()}
    archive = Archive(
        rep=tensors["rep"], expert=tensors["expert"], app=tensors["app"], group=groups, compat=compats,
        r_local=tensors["r_local"], mu_app=tensors["mu_app"], count=tensors["count"], cell=tensors["cell"],
        family=torch.zeros_like(tensors["cell"]), apps=list(apps),
    )
    archive.family = _context_families(archive, int(rg.archive.num_families), int(rg.archive.kmeans_iters))
    return archive


def _context_families(archive: Archive, num_families: int, num_iters: int) -> torch.Tensor:
    """Coarse families: k-means over the unique (app, cell) representatives of each compat group.

    Families are formed within a compat group because only compatible records
    are ever retrieved together; a global clustering would mostly separate
    task families and make the missing-family control remove all evidence.
    """
    family = torch.zeros_like(archive.cell)
    if len(archive) == 0:
        return family
    compat_id = archive.compat_index()
    pair_key = archive.app * (int(archive.cell.max()) + 1) + archive.cell
    for gid in torch.unique(compat_id).tolist():
        rec = torch.nonzero(compat_id == gid, as_tuple=True)[0]
        pairs, inverse = torch.unique(pair_key[rec], return_inverse=True)
        first = torch.zeros(pairs.numel(), dtype=torch.long).scatter_reduce(
            0, inverse, torch.arange(rec.numel()), reduce="amin", include_self=False
        )
        compat = tuple(archive.compat[int(archive.app[rec[0]])])
        assign, _ = kmeans(archive.rep[rec[first]], num_families, _seed("archive_families", *compat), num_iters)
        family[rec] = assign[inverse]
    return family


def perturb_archive(archive: Archive, kind: str, seed: int) -> Archive:
    """Evidence controls of App. D.2 / D.5 (deterministic given *seed*).

    * ``half_cells``: drop a random half of the (application, cell) pairs.
    * ``missing_family``: in every compat group, drop all records of the
      coarse family holding most of the group's records (ties -> lowest id).
    * ``reversed``: residuals r - mu become -(r - mu).
    * ``shuffled``: permute residuals across records within each compat group
      (each record keeps its representative, expert, and mu).
    """
    if kind == "none" or len(archive) == 0:
        return archive
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    device = archive.app.device
    if kind == "half_cells":
        pair_key = (archive.app * (int(archive.cell.max()) + 1) + archive.cell).cpu()
        pairs, inverse = torch.unique(pair_key, return_inverse=True)
        dropped = torch.randperm(pairs.numel(), generator=generator)[: pairs.numel() // 2]
        return archive.subset((~torch.isin(inverse, dropped)).to(device))
    compat_id = archive.compat_index().cpu()
    if kind == "missing_family":
        keep = torch.ones(len(archive), dtype=torch.bool)
        family = archive.family.cpu()
        for gid in torch.unique(compat_id).tolist():
            rec = compat_id == gid
            top = int(torch.bincount(family[rec]).argmax())
            keep &= ~(rec & (family == top))
        return archive.subset(keep.to(device))
    if kind == "reversed":
        return replace(archive, r_local=archive.mu_app - archive.residual)
    if kind == "shuffled":
        residual = archive.residual.cpu().clone()
        for gid in torch.unique(compat_id).tolist():
            rec = torch.nonzero(compat_id == gid, as_tuple=True)[0]
            residual[rec] = residual[rec[torch.randperm(rec.numel(), generator=generator)]]
        return replace(archive, r_local=archive.mu_app + residual.to(device))
    raise ValueError(f"Unknown archive perturbation {kind!r}; expected one of {PERTURBATIONS}")


__all__ = [
    "Archive",
    "PERTURBATIONS",
    "build_archive",
    "build_cells",
    "kmeans",
    "perturb_archive",
]
