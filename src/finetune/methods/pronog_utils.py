"""Utility helpers for the ProNoG finetune method.

Pure functions mirroring ``graphprompt_utils.py``: the capped multi-hop
ego-network pair list and the similarity-weighted conditioning readout
(paper Eq. 7) used by ``FinetuneProNoG``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Hard ceiling on a single hop expansion's pre-cap pair count.  Converts the
#: unbounded ``neighbor_cap=0`` blow-up on dense graphs into an actionable
#: error instead of a CUDA OOM crash mid-epoch.
MAX_HOP_EXPANSION = 200_000_000


def build_hop_neighbor_pairs(
    edge_index: torch.Tensor,
    num_nodes: int,
    hops: int = 2,
    cap: int = 20,
) -> torch.Tensor:
    """(center, member) pairs of every node's capped multi-hop ego-network.

    Edges are symmetrized and coalesced first (the official ProNoG reads a
    symmetric adjacency), so directed inputs get undirected ego-networks and
    duplicate edges cannot burn cap slots; coalescing also sorts neighbors by
    index, matching the official adjacency-matrix iteration order.  Hop 1
    keeps each node's first ``cap`` neighbors; each further hop expands the
    previous frontier through the same capped adjacency and keeps the first
    ``cap`` members per center.  Unlike the official ``find_2hop_neighbors``
    (which expands through the *full* hop-1 adjacency and caps only the
    collected list), the capped-adjacency expansion bounds peak memory at
    ``cap^2`` candidates per center.  The self pair ``(v, v)`` is always
    included and duplicates are removed, matching the paper's set-valued
    ego-network ``S_v``.  ``cap <= 0`` disables capping (guarded by
    :data:`MAX_HOP_EXPANSION`).

    Returns a ``(2, P)`` LongTensor ``[centers, members]``.
    """
    device = edge_index.device
    num_nodes = int(num_nodes)
    cap = int(cap)
    sym_row = torch.cat([edge_index[0], edge_index[1]])
    sym_col = torch.cat([edge_index[1], edge_index[0]])
    no_loop = sym_row != sym_col
    sym_row, sym_col = sym_row[no_loop], sym_col[no_loop]
    edge_key = torch.unique(sym_row * num_nodes + sym_col)
    row, col = edge_key // num_nodes, edge_key % num_nodes
    if cap > 0 and row.numel() > 0:
        deg = torch.bincount(row, minlength=num_nodes)
        start = torch.cumsum(deg, dim=0) - deg
        pos = torch.arange(row.numel(), device=device) - start[row]
        keep = pos < cap
        row, col = row[keep], col[keep]
    deg = torch.bincount(row, minlength=num_nodes)
    start = torch.cumsum(deg, dim=0) - deg

    self_idx = torch.arange(num_nodes, device=device)
    centers = [self_idx, row]
    members = [self_idx, col]

    prev_c, prev_m = row, col
    for _hop in range(2, int(hops) + 1):
        reps = deg[prev_m]
        total = int(reps.sum())
        if total == 0:
            break
        if total > MAX_HOP_EXPANSION:
            raise RuntimeError(
                f"[ProNoG] hop expansion would materialize {total} candidate "
                f"pairs (> {MAX_HOP_EXPANSION}); set finetune.pronog.neighbor_cap "
                "to a positive value to bound the ego-network size on this graph."
            )
        # Expand each frontier pair (v, u) into (v, w) for u's capped neighbors.
        exp_c = prev_c.repeat_interleave(reps)
        offsets = torch.cumsum(reps, dim=0) - reps
        local = torch.arange(total, device=device) - offsets.repeat_interleave(reps)
        exp_m = col[start[prev_m].repeat_interleave(reps) + local]
        keep = exp_m != exp_c
        exp_c, exp_m = exp_c[keep], exp_m[keep]
        if cap > 0 and exp_c.numel() > 0:
            # exp_c stays sorted (prev_c is sorted; repeat_interleave preserves
            # order), so position-in-group capping is valid.
            cnt = torch.bincount(exp_c, minlength=num_nodes)
            first = torch.cumsum(cnt, dim=0) - cnt
            pos = torch.arange(exp_c.numel(), device=device) - first[exp_c]
            keep = pos < cap
            exp_c, exp_m = exp_c[keep], exp_m[keep]
        centers.append(exp_c)
        members.append(exp_m)
        prev_c, prev_m = exp_c, exp_m

    all_c = torch.cat(centers)
    all_m = torch.cat(members)
    key = torch.unique(all_c * num_nodes + all_m)
    return torch.stack([key // num_nodes, key % num_nodes])


def conditioned_subgraph_readout(
    node_repr: torch.Tensor,
    pairs: torch.Tensor,
    chunk_size: int = 1_000_000,
) -> torch.Tensor:
    """Similarity-weighted ego-network readout (paper Eq. 7).

    ``s_v = mean_{u in S_v} h_u * cos(h_u, h_v)`` computed over the pre-built
    ``(center, member)`` pair list, chunked to bound peak memory on large
    graphs.
    """
    normed = F.normalize(node_repr, p=2, dim=-1)
    readout = torch.zeros_like(node_repr)
    centers, members = pairs[0], pairs[1]
    for lo in range(0, centers.numel(), int(chunk_size)):
        c = centers[lo:lo + chunk_size]
        m = members[lo:lo + chunk_size]
        sim = (normed[c] * normed[m]).sum(dim=-1, keepdim=True)
        readout.index_add_(0, c, node_repr[m] * sim)
    # Exact integer member counts (never zero: every center has its self pair).
    counts = torch.bincount(centers, minlength=node_repr.size(0)).to(node_repr.dtype)
    return readout / counts.clamp_min(1.0).unsqueeze(-1)
