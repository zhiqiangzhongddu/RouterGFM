"""Expert-conditioned archive retrieval and local loss estimates (Eq. 6-8).

A query is one (instance context, expert) pair with key ``k_phi(z, v_e)``;
archive record ``i`` has key ``k_phi(c_i, v_{e_i})``. Retrieval is exact: the
``J`` nearest allowed records with at most ``cap`` records from any one source
application. Search runs without gradients on keys refreshed by the caller;
``kernel_weights`` recomputes the weights of the selected records
differentiably so gradients reach the query key and the ``J`` record keys.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch

from ..common import CompatKey

# Distance entries materialised per query chunk (float32: 64 MB).
_CHUNK_ELEMENTS = 1 << 24


def _compat_tuple(compat) -> Tuple[str, int, str]:
    if isinstance(compat, CompatKey):
        return compat.as_tuple()
    return CompatKey(*compat).as_tuple()


def allowed_records(
    record_app: torch.Tensor,
    app_groups: Sequence[str],
    app_compat: Sequence,
    *,
    group: str,
    compat,
    hidden_groups: Iterable[str] = (),
) -> torch.Tensor:
    """Records a query of an application in ``group`` with ``compat`` may retrieve: ``BoolTensor[R]``.

    Allowed records come from applications of OTHER groups (never the query's
    group, nor any group hidden in the current training episode) whose
    CompatKey (task family, budget, normalization) matches. ``app_groups`` and
    ``app_compat`` are indexed by ``record_app``.
    """
    excluded = {str(group), *(str(g) for g in hidden_groups)}
    target = _compat_tuple(compat)
    app_ok = torch.tensor(
        [str(g) not in excluded and _compat_tuple(c) == target for g, c in zip(app_groups, app_compat)],
        dtype=torch.bool,
        device=record_app.device,
    )
    if app_ok.numel() == 0:
        return torch.zeros(record_app.shape[0], dtype=torch.bool, device=record_app.device)
    return app_ok[record_app]


def _app_blocks(apps: torch.Tensor) -> torch.Tensor:
    """Padded per-application blocks ``[A, L]`` of positions into ``apps`` (``-1`` = padding)."""
    _, inverse, counts = torch.unique(apps, return_inverse=True, return_counts=True)
    order = torch.argsort(inverse, stable=True)
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.arange(apps.shape[0], device=apps.device) - starts[inverse[order]]
    blocks = torch.full((counts.numel(), int(counts.max())), -1, dtype=torch.long, device=apps.device)
    blocks[inverse[order], slot] = order
    return blocks


@torch.no_grad()
def search(
    query_keys: torch.Tensor,
    record_keys: torch.Tensor,
    record_app: torch.Tensor,
    allowed: torch.Tensor,
    J: int,
    cap: int,
    *,
    query_group: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact top-``J`` allowed records per query with a per-application cap.

    ``allowed`` is ``BoolTensor[R]`` (shared by all queries) or ``[G, R]``; in
    the latter case row ``query_group[p]`` applies to query ``p`` (default: one
    row per query). Per source application the ``cap`` nearest records are
    kept, then the global ``J`` nearest among them, which equals the exact
    capped top-``J``. Returns ``idx [P, J]`` (record indices, ascending
    distance) and ``valid [P, J]``; slots beyond the allowed set are invalid
    (index 0). An empty allowed set gives an all-invalid row.
    """
    if int(cap) < 1:
        raise ValueError(f"per-application cap must be >= 1, got {cap}")
    num_queries, J, cap = query_keys.shape[0], int(J), int(cap)
    device = query_keys.device
    idx = torch.zeros(num_queries, J, dtype=torch.long, device=device)
    valid = torch.zeros(num_queries, J, dtype=torch.bool, device=device)
    if J <= 0 or num_queries == 0 or record_keys.shape[0] == 0:
        return idx, valid

    record_keys = record_keys.to(device=device, dtype=query_keys.dtype)
    record_app = record_app.to(device)
    allowed = allowed.to(device=device, dtype=torch.bool)
    if allowed.dim() == 1:
        allowed = allowed.unsqueeze(0)
        query_group = torch.zeros(num_queries, dtype=torch.long, device=device)
    elif query_group is None:
        if allowed.shape[0] != num_queries:
            raise ValueError("allowed [G, R] needs query_group unless G equals the number of queries")
        query_group = torch.arange(num_queries, device=device)
    else:
        query_group = query_group.to(device)

    for g in torch.unique(query_group).tolist():
        records = allowed[g].nonzero(as_tuple=True)[0]
        if records.numel() == 0:
            continue
        queries = (query_group == g).nonzero(as_tuple=True)[0]
        keys_g = record_keys[records]
        blocks = _app_blocks(record_app[records])
        pad = blocks < 0
        safe_blocks = blocks.clamp(min=0)
        num_apps, width = blocks.shape
        per_app = min(cap, width)
        step = int(chunk_size) if chunk_size else max(1, _CHUNK_ELEMENTS // (num_apps * width))
        for start in range(0, queries.numel(), step):
            q = queries[start:start + step]
            d2 = torch.cdist(query_keys[q], keys_g).pow(2)
            d2 = d2[:, safe_blocks].masked_fill(pad, float("inf"))  # [c, A, L]
            top_d, top_pos = d2.topk(per_app, dim=-1, largest=False)
            cand = safe_blocks.expand(q.numel(), -1, -1).gather(-1, top_pos)
            top_d = top_d.reshape(q.numel(), -1)
            cand = cand.reshape(q.numel(), -1)
            k = min(J, top_d.shape[1])
            best_d, best_pos = top_d.topk(k, dim=-1, largest=False)
            ok = torch.isfinite(best_d)
            chosen = records[cand.gather(1, best_pos)]
            idx[q, :k] = torch.where(ok, chosen, torch.zeros_like(chosen))
            valid[q, :k] = ok
    return idx, valid


def kernel_weights(
    query_keys: torch.Tensor,
    record_keys_sel: torch.Tensor,
    valid: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    """Eq. 6: ``w_i = exp(-d_i^2/h^2) / sum_{i'} exp(-d_{i'}^2/h^2)`` over valid retrieved records.

    Counts do not enter the weights. Rows without a valid record get all-zero
    weights (and zero gradients).
    """
    d2 = (query_keys.unsqueeze(1) - record_keys_sel).pow(2).sum(dim=-1)
    logits = (-d2 / float(bandwidth) ** 2).masked_fill(~valid, float("-inf"))
    logits = logits.masked_fill(~valid.any(dim=-1, keepdim=True), 0.0)
    return torch.softmax(logits, dim=-1) * valid.to(logits.dtype)


def local_estimate(
    mu_hat: torch.Tensor,
    w: torch.Tensor,
    residual_sel: torch.Tensor,
    rho: float,
    *,
    mode: str = "centered",
) -> torch.Tensor:
    """Local loss estimate ``r_hat [P]``.

    ``centered`` (Eq. 7): ``residual_sel`` holds ``r_i - mu_i`` and
    ``r_hat = mu_hat + rho * sum_i w_i (r_i - mu_i)``.
    ``raw`` (no-centering control): ``residual_sel`` holds the raw local losses
    ``r_i`` and ``r_hat = (1 - rho) * mu_hat + rho * sum_i w_i r_i``.
    Either way an empty retrieval set gives ``r_hat = mu_hat``.
    """
    transfer = (w * residual_sel.to(w.dtype)).sum(dim=-1)
    if mode == "centered":
        return mu_hat + rho * transfer
    if mode == "raw":
        mixed = (1.0 - rho) * mu_hat + rho * transfer
        return torch.where(w.sum(dim=-1) > 0, mixed, mu_hat)
    raise ValueError(f"Unknown local estimate mode {mode!r} (expected 'centered' or 'raw')")


def mixture_weights(r_hat: torch.Tensor, tau: float) -> torch.Tensor:
    """Eq. 8: ``alpha = softmax(-r_hat / tau)`` over the team (last dim)."""
    return torch.softmax(-r_hat / float(tau), dim=-1)


__all__ = ["allowed_records", "kernel_weights", "local_estimate", "mixture_weights", "search"]
