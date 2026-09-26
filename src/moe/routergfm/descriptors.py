"""Label-free context descriptors z_a(x) (paper Sec. 3.1, 3.4; App. B.2).

A context is the induced subgraph around a marked node, the enclosing subgraph
of a marked node pair, or the whole input graph. Its descriptor concatenates
four families -- structure, node features, marked-node/endpoint roles, task
indicator -- with a fixed length across task levels (inapplicable entries are
0), so records of every application live in one space. Labels are never read.

Link contexts are computed with the candidate link absent: induced edge
subgraphs already drop it, and it is removed again here so a descriptor can
never reveal whether the queried pair is connected.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from src.moe.routergfm.common import AppSpec, RouterPaths, TASK_FAMILIES
from src.utils.checkpoint import save_torch_atomic

DEFAULT_NUM_SPECTRAL = 6  # cfg.moe.routergfm.descriptors.num_spectral default
SHORTEST_PATH_CAP = 6  # endpoint distance cap (unreachable pairs get the cap)
_ZERO_EIG_TOL = 1e-8  # zero-eigenvalue threshold (float64) for component counts
_LOAD_CHUNK = 4096  # instances materialized from the dataset at once
_BATCH_ELEMENTS = 1 << 22  # budget on B * n_max^2 per vectorized batch


def _family_names(num_spectral: int) -> Dict[str, List[str]]:
    k = int(num_spectral)
    return {
        "structure": [
            "struct_log_nodes", "struct_log_edges", "struct_density",
            "struct_degree_mean", "struct_degree_std", "struct_degree_max",
            "struct_clustering", "struct_log_components",
        ] + [f"struct_lap_eig_{i}" for i in range(k)],
        "feature": ["feat_norm_mean", "feat_norm_std", "feat_marked_cosine"]
        + [f"feat_sv_{i}" for i in range(k)],
        "role": [
            "role_node_degree", "role_node_degree_pct", "role_node_clustering", "role_node_feat_norm",
            "role_edge_degree_min", "role_edge_degree_max", "role_edge_common_neighbors",
            "role_edge_jaccard", "role_edge_adamic_adar", "role_edge_shortest_path",
            "role_edge_feat_cosine",
        ],
        "task": [f"task_{family}" for family in TASK_FAMILIES],
    }


def descriptor_names(num_spectral: int = DEFAULT_NUM_SPECTRAL) -> List[str]:
    return [name for names in _family_names(num_spectral).values() for name in names]


def family_slices(num_spectral: int = DEFAULT_NUM_SPECTRAL) -> Dict[str, slice]:
    out, start = {}, 0
    for family, names in _family_names(num_spectral).items():
        out[family] = slice(start, start + len(names))
        start += len(names)
    return out


# Layout for the default ``num_spectral``; use the functions for other values.
DESCRIPTOR_NAMES = descriptor_names()
FAMILY_SLICES = family_slices()


def _top_desc(values: torch.Tensor, k: int) -> torch.Tensor:
    """Largest *k* entries of ascending-sorted rows, descending, zero-padded."""
    top = values.flip(-1)[:, :k]
    if top.size(1) < k:
        top = torch.cat([top, top.new_zeros(top.size(0), k - top.size(1))], dim=1)
    return top


def _masked_std(values: torch.Tensor, mean: torch.Tensor, maskf: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    return ((((values - mean[:, None]) ** 2) * maskf).sum(-1) / count).sqrt()


def _dense_batch(graphs: Sequence[Any], level: str, task_family: str, k: int, device) -> torch.Tensor:
    """Vectorized descriptors of graphs padded to one dense ``[B, n_max, n_max]`` batch.

    Padding nodes are isolated with zero features, so they add only zero
    Laplacian eigenvalues / singular values and never change the leading ones.
    """
    dt = torch.float64
    B = len(graphs)
    bi = torch.arange(B, device=device)
    n = torch.tensor([int(g.num_nodes) for g in graphs], device=device)
    n_max = max(int(n.max()), 1)
    ar = torch.arange(n_max, device=device)
    mask = ar[None, :] < n[:, None]
    maskf = mask.to(dt)
    nf = n.to(dt)
    count = nf.clamp(min=1)

    # Concatenate on CPU: CUDA rejects a cat of only empty tensors (a bucket of
    # isolated single-node contexts has no edges at all).
    edge_index = [g.edge_index.to(dtype=torch.long).cpu() for g in graphs]
    eb = torch.repeat_interleave(torch.arange(B), torch.tensor([e.size(1) for e in edge_index])).to(device)
    src, dst = torch.cat(edge_index, dim=1).to(device)
    A = torch.zeros(B, n_max, n_max, dtype=dt, device=device)
    if eb.numel() > 0:
        A[eb, src, dst] = 1.0
        A[eb, dst, src] = 1.0  # undirected, simple
    A[:, ar, ar] = 0.0
    if level == "edge":
        pair = torch.stack([g.edge_label_index.view(2, -1)[:, 0] for g in graphs]).to(device)
        u, v = pair[:, 0], pair[:, 1]
        A[bi, u, v] = 0.0  # candidate link absent
        A[bi, v, u] = 0.0
    elif level == "node":
        t = torch.stack([g.target_node_index.view(-1)[0] for g in graphs]).to(device)

    if graphs[0].x is None:
        X = torch.zeros(B, n_max, 1, dtype=dt, device=device)
    else:
        X = torch.zeros(B, n_max, graphs[0].x.size(1), dtype=dt, device=device)
        X[mask] = torch.cat([g.x for g in graphs], dim=0).to(device=device, dtype=dt)

    # ---- structure -------------------------------------------------------
    deg = A.sum(-1)
    m = deg.sum(-1) / 2
    density = torch.where(n > 1, 2 * m / (nf * (nf - 1)).clamp(min=1), torch.zeros_like(m))
    deg_mean = deg.sum(-1) / count
    deg_std = _masked_std(deg, deg_mean, maskf, count)
    triangles = ((A @ A) * A).sum(-1) / 2
    wedges = deg * (deg - 1) / 2
    clustering = torch.where(wedges > 0, triangles / wedges.clamp(min=1), torch.zeros_like(deg))
    dinv = torch.where(deg > 0, deg.clamp(min=1).rsqrt(), torch.zeros_like(deg))
    lap = torch.diag_embed((deg > 0).to(dt)) - dinv[:, :, None] * A * dinv[:, None, :]
    lap_eig = torch.linalg.eigvalsh(lap).clamp(min=0)
    # Zero eigenvalues of the normalized Laplacian = components (isolated nodes included).
    components = (lap_eig < _ZERO_EIG_TOL).sum(-1) - (n_max - n)
    structure = torch.cat([
        torch.stack([
            torch.log1p(nf), torch.log1p(m), density, deg_mean, deg_std, deg.max(-1).values,
            clustering.sum(-1) / count, torch.log1p(components.to(dt)),
        ], dim=1),
        _top_desc(lap_eig, k),
    ], dim=1)

    # ---- features --------------------------------------------------------
    norms = X.norm(dim=-1)
    norm_mean = norms.sum(-1) / count
    norm_std = _masked_std(norms, norm_mean, maskf, count)
    Xc = (X - (X.sum(1) / count[:, None])[:, None, :]) * maskf[..., None]
    gram = Xc @ Xc.transpose(1, 2) if n_max <= X.size(2) else Xc.transpose(1, 2) @ Xc
    sv = torch.linalg.eigvalsh(gram).clamp(min=0).sqrt()
    sv_total = sv.sum(-1, keepdim=True)
    sv_top = torch.where(sv_total > 0, _top_desc(sv, k) / sv_total.clamp(min=1e-12), torch.zeros(B, k, dtype=dt, device=device))
    Xn = X / norms.clamp(min=1e-12)[..., None]
    cos = Xn @ Xn.transpose(1, 2)
    if level == "graph":
        pair_w = maskf[:, :, None] * maskf[:, None, :] * (1 - torch.eye(n_max, dtype=dt, device=device))
        marked_cos = (cos * pair_w).sum((1, 2)) / pair_w.sum((1, 2)).clamp(min=1)
    else:
        rest = maskf.clone()
        if level == "node":
            row = cos[bi, t]
            rest[bi, t] = 0.0
        else:
            row = (cos[bi, u] + cos[bi, v]) / 2
            rest[bi, u] = 0.0
            rest[bi, v] = 0.0
        marked_cos = (row * rest).sum(-1) / rest.sum(-1).clamp(min=1)
    feature = torch.cat([torch.stack([norm_mean, norm_std, marked_cos], dim=1), sv_top], dim=1)

    # ---- roles -----------------------------------------------------------
    role = torch.zeros(B, 11, dtype=dt, device=device)
    if level == "node":
        d_t = deg[bi, t]
        less = ((deg < d_t[:, None]) & mask).sum(-1).to(dt)
        ties = ((deg == d_t[:, None]) & mask).sum(-1).to(dt) - 1
        pct = torch.where(n > 1, (less + 0.5 * ties) / (nf - 1).clamp(min=1), torch.full_like(nf, 0.5))
        role[:, :4] = torch.stack([d_t, pct, clustering[bi, t], norms[bi, t]], dim=1)
    elif level == "edge":
        du, dv = deg[bi, u], deg[bi, v]
        common = A[bi, u] * A[bi, v]
        cn = common.sum(-1)
        union = du + dv - cn
        jaccard = torch.where(union > 0, cn / union.clamp(min=1), torch.zeros_like(cn))
        inv_log = torch.where(deg > 1, 1.0 / torch.log(deg.clamp(min=2)), torch.zeros_like(deg))
        adamic_adar = (common * inv_log).sum(-1)
        dist = torch.full((B,), float(SHORTEST_PATH_CAP), dtype=dt, device=device)
        found = torch.zeros(B, dtype=torch.bool, device=device)
        reach = torch.zeros(B, n_max, dtype=dt, device=device)
        reach[bi, u] = 1.0
        for step in range(1, SHORTEST_PATH_CAP + 1):
            reach = ((A @ reach[..., None]).squeeze(-1) + reach).clamp(max=1)
            hit = (reach[bi, v] > 0) & ~found
            dist[hit] = float(step)
            found |= hit
            if bool(found.all()):
                break
        role[:, 4:] = torch.stack(
            [torch.minimum(du, dv), torch.maximum(du, dv), cn, jaccard, adamic_adar, dist, cos[bi, u, v]], dim=1
        )

    task = torch.zeros(B, len(TASK_FAMILIES), dtype=dt, device=device)
    task[:, TASK_FAMILIES.index(task_family)] = 1.0
    return torch.cat([structure, feature, role, task], dim=1).to(device="cpu", dtype=torch.float32)


@torch.no_grad()
def batch_descriptors(
    graphs: Sequence[Any],
    level: str,
    task_family: str,
    num_spectral: int = DEFAULT_NUM_SPECTRAL,
    device=None,
) -> torch.Tensor:
    """Descriptors ``[B, D]`` of instances of one application (size-bucketed batches)."""
    if level not in ("node", "edge", "graph"):
        raise ValueError(f"Unknown task level {level!r}")
    k = int(num_spectral)
    out = torch.zeros(len(graphs), len(descriptor_names(k)))
    sizes = [max(int(g.num_nodes), 1) for g in graphs]
    order = sorted(range(len(graphs)), key=sizes.__getitem__)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and (end - start + 1) * sizes[order[end]] ** 2 <= _BATCH_ELEMENTS:
            end += 1
        idx = order[start:end]
        out[idx] = _dense_batch([graphs[i] for i in idx], level, task_family, k, device or "cpu")
        start = end
    return out


def instance_descriptor(graph: Any, level: str, task_family: str, num_spectral: int = DEFAULT_NUM_SPECTRAL) -> torch.Tensor:
    """Descriptor ``z(x)`` of one instance: node / edge context subgraph or whole graph."""
    return batch_descriptors([graph], level, task_family, num_spectral)[0]


def _device(cfg) -> torch.device:
    return torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")


def compute_descriptors(data: Any, positions, cfg, *, device=None) -> torch.Tensor:
    """Descriptors ``[n, D]`` of ``data.dataset[positions]`` in the order of *positions*."""
    positions = torch.as_tensor(positions, dtype=torch.long).view(-1)
    num_spectral = int(cfg.moe.routergfm.descriptors.num_spectral)
    device = device if device is not None else _device(cfg)
    out = torch.zeros(positions.numel(), len(descriptor_names(num_spectral)))
    for start in range(0, positions.numel(), _LOAD_CHUNK):
        chunk = positions[start:start + _LOAD_CHUNK].tolist()
        graphs = [data.dataset[p] for p in chunk]
        out[start:start + len(chunk)] = batch_descriptors(graphs, data.level, data.task_family, num_spectral, device)
    return out


def ensure_descriptors(cfg, data: Any, provider: Optional[Any] = None) -> Dict[str, Any]:
    """Load or build the per-data-key descriptor cache ``{'positions', 'z', 'names'}``.

    *data* is an ``AppData`` or an ``AppSpec``. For an ``AppSpec`` an existing
    cache is trusted and the application is loaded (via *provider*, default
    ``RealDataProvider``) only on a miss. For an ``AppData`` the cache is
    extended with any missing support / diagnostic / query position.
    ``positions`` are sorted ascending and ``z`` is raw (unstandardized).
    """
    app = data if isinstance(data, AppSpec) else data.app
    names = descriptor_names(int(cfg.moe.routergfm.descriptors.num_spectral))
    path = RouterPaths.from_cfg(cfg).descriptor_file(app.data_key)
    cache = torch.load(path, map_location="cpu") if path.exists() else None
    if cache is not None and list(cache.get("names", [])) != names:
        cache = None  # stale layout (num_spectral changed)
    if cache is not None and isinstance(data, AppSpec):
        return cache
    if isinstance(data, AppSpec):
        if provider is None:
            from src.moe.routergfm.applications import RealDataProvider

            provider = RealDataProvider(cfg)
        data = provider.load(app)

    needed = torch.unique(torch.cat([
        torch.as_tensor(data.support_pos, dtype=torch.long).view(-1),
        torch.as_tensor(data.diag_pos, dtype=torch.long).view(-1),
        torch.as_tensor(data.query_pos, dtype=torch.long).view(-1),
    ]))
    have = cache["positions"] if cache is not None else torch.empty(0, dtype=torch.long)
    missing = needed[~torch.isin(needed, have)]
    if cache is not None and missing.numel() == 0:
        return cache
    positions = torch.cat([have, missing])
    z = torch.cat([cache["z"] if cache is not None else torch.zeros(0, len(names)), compute_descriptors(data, missing, cfg)])
    order = torch.argsort(positions)
    cache = {"positions": positions[order], "z": z[order], "names": names}
    save_torch_atomic(str(path), cache)
    return cache


def descriptors_at(cache: Dict[str, Any], positions) -> torch.Tensor:
    """Rows of a descriptor cache for dataset *positions* (in their order)."""
    positions = torch.as_tensor(positions, dtype=torch.long).view(-1)
    cached = cache["positions"]
    if positions.numel() == 0:
        return cache["z"][:0]
    idx = torch.searchsorted(cached, positions).clamp(max=max(cached.numel() - 1, 0))
    if cached.numel() == 0 or not bool((cached[idx] == positions).all()):
        raise KeyError("descriptor cache does not cover the requested positions")
    return cache["z"][idx]


class DescriptorStandardizer:
    """Per-dimension z-scoring fitted on historical training applications, then clipped."""

    def __init__(self, clip: float = 5.0):
        self.clip = float(clip)
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, Z: torch.Tensor) -> "DescriptorStandardizer":
        Z = Z.to(torch.float32)
        self.mean = Z.mean(0)
        std = Z.std(0, unbiased=False)
        self.std = torch.where(std > 1e-6, std, torch.ones_like(std))  # constant dims map to 0
        return self

    def transform(self, Z: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            raise RuntimeError("DescriptorStandardizer.transform called before fit")
        mean, std = self.mean.to(Z.device), self.std.to(Z.device)
        return ((Z.to(torch.float32) - mean) / std).clamp(-self.clip, self.clip)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "clip": self.clip}

    def load_state_dict(self, state: Dict[str, Any]) -> "DescriptorStandardizer":
        self.mean, self.std, self.clip = state["mean"], state["std"], float(state["clip"])
        return self


__all__ = [
    "DEFAULT_NUM_SPECTRAL",
    "DESCRIPTOR_NAMES",
    "DescriptorStandardizer",
    "FAMILY_SLICES",
    "SHORTEST_PATH_CAP",
    "batch_descriptors",
    "compute_descriptors",
    "descriptor_names",
    "descriptors_at",
    "ensure_descriptors",
    "family_slices",
    "instance_descriptor",
]
