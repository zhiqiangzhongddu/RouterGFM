"""Merge-team selection: compatible groups and the AnyGraph competence score (Eq. 5, 7).

Routing is label-free and query-free: the score uses only the edges of the
support (sub)graphs, and one set of triplets is shared by every candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torch_geometric.data import Batch

_LOG = "[RouterGFM][kdem_ppem]"
GROUP_POLICIES = ("top1_arch", "fixed")
# cfg.model subtrees that change an architecture's parameters or forward pass.
_ARCH_SUBTREES = {"gat": "gat", "transformer": "gat", "fagcn": "fagcn", "h2gcn": "h2gcn", "nodeformer": "nodeformer"}


@dataclass(frozen=True)
class MergeTeam:
    compat_key: tuple
    expert_ids: List[str]  # ordered by competence, best first
    competence: List[float]  # psi_i of the team
    alpha: torch.Tensor  # softmax(psi_I), [k]
    all_scores: Dict[str, float]  # psi of every scored candidate


def compat_key(model_cfg) -> tuple:
    """Architecture and shape knobs under which two experts' parameters can be averaged.

    ``model_cfg`` is the checkpoint-restored cfg; dropout is excluded (the merged
    forward uses the top-1 expert's structure).
    """
    m = model_cfg.model
    name = str(m.name).lower()
    key = (
        name, int(m.in_dim or 0), int(m.hidden_dim), int(m.out_dim), int(m.num_layers),
        str(m.activation).lower(), str(m.graph_pooling).lower(), bool(m.use_batchnorm),
    )
    sub = _ARCH_SUBTREES.get(name)
    if sub is not None and sub in m:
        key += (tuple(sorted((k, v) for k, v in dict(m[sub]).items())),)
    return key


def assert_state_compatible(states: Sequence[Mapping[str, torch.Tensor]]) -> None:
    """Raise ``ValueError`` unless every state has the same keys and tensor shapes."""
    ref = {k: tuple(v.shape) for k, v in states[0].items()}
    for i, state in enumerate(states[1:], start=1):
        shapes = {k: tuple(v.shape) for k, v in state.items()}
        if shapes.keys() != ref.keys():
            diff = sorted(set(shapes) ^ set(ref))[:5]
            raise ValueError(f"Expert {i} is not merge-compatible with expert 0: key sets differ ({diff}).")
        bad = [k for k in ref if shapes[k] != ref[k]]
        if bad:
            raise ValueError(f"Expert {i} is not merge-compatible with expert 0: shape mismatch at {bad[:5]}.")


def team_candidates(expert_ids: Sequence[str], architecture: Mapping[str, str], group_policy: str, fixed_arch: str) -> List[str]:
    """Experts scored for the team: all of E_a (``top1_arch``) or those of ``fixed_arch``."""
    if group_policy == "top1_arch":
        out = list(expert_ids)
    elif group_policy == "fixed":
        if not fixed_arch:
            raise ValueError("kdem_ppem.group_policy 'fixed' needs kdem_ppem.fixed_arch.")
        out = [e for e in expert_ids if architecture[e] == fixed_arch]
    else:
        raise ValueError(f"Unknown kdem_ppem.group_policy {group_policy!r} (expected one of {GROUP_POLICIES}).")
    if not out:
        raise ValueError(f"No eligible experts for group_policy={group_policy!r} fixed_arch={fixed_arch!r}.")
    return out


def competence_batches(
    graphs: Sequence, *, max_triplets: int, seed: int, batch_size: int, device=None
) -> List[Tuple[Batch, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Support triplets ``(a, p, n)`` collated once for all experts: ``[(batch, a, p, n)]``.

    Positives are unique undirected non-self-loop edges of each (sub)graph,
    oriented at random; negatives are uniform nodes of the same (sub)graph
    (unfiltered, as AnyGraph's ``negs``). At most ``max_triplets`` edges are
    sampled. Triplets depend only on ``edge_index`` and node counts (the
    runner passes graphs without ``y``). Returns ``[]`` if no support graph
    has an edge.
    """
    from src.pretrain.methods.edge_pred import _unique_undirected_edges

    gen = torch.Generator().manual_seed(int(seed))
    pairs, owner = [], []
    for gi, graph in enumerate(graphs):
        edge_index = graph.edge_index
        if edge_index is None or edge_index.numel() == 0:
            continue
        unique = _unique_undirected_edges(edge_index.cpu())
        pairs.append(unique)
        owner.append(torch.full((unique.size(1),), gi, dtype=torch.long))
    if not owner or sum(o.numel() for o in owner) == 0:
        return []
    pairs, owner = torch.cat(pairs, dim=1), torch.cat(owner)  # owner ascending
    if owner.numel() > int(max_triplets):
        keep = torch.randperm(owner.numel(), generator=gen)[: int(max_triplets)].sort().values
        pairs, owner = pairs[:, keep], owner[keep]
    flip = torch.rand(owner.numel(), generator=gen) < 0.5
    anchor = torch.where(flip, pairs[1], pairs[0])
    positive = torch.where(flip, pairs[0], pairs[1])
    used = torch.unique(owner)
    num_nodes = torch.zeros(len(graphs), dtype=torch.long)
    num_nodes[used] = torch.tensor([int(graphs[g].num_nodes) for g in used.tolist()], dtype=torch.long)
    sizes = num_nodes[owner]
    negative = torch.minimum((torch.rand(owner.numel(), generator=gen, dtype=torch.float64) * sizes).long(), sizes - 1)

    out = []
    for start in range(0, used.numel(), int(batch_size)):
        chunk = used[start:start + int(batch_size)]
        batch = Batch.from_data_list([graphs[g] for g in chunk.tolist()])
        mask = (owner >= chunk[0]) & (owner <= chunk[-1])
        offset = batch.ptr[torch.searchsorted(chunk, owner[mask])]
        triplet = (anchor[mask] + offset, positive[mask] + offset, negative[mask] + offset)
        out.append((batch.to(device), *(t.to(device) for t in triplet)))
    return out


@torch.no_grad()
def competence_score(encoder, batches) -> float:
    """``psi = mean sigmoid(h_a.h_p - h_a.h_n)`` on raw node representations (AnyGraph Eq. 5); 0.5 without triplets."""
    encoder.eval()
    total, count = 0.0, 0
    for batch, anchor, positive, negative in batches:
        node_repr, _ = encoder(batch)
        h_a = node_repr[anchor]
        margin = (h_a * node_repr[positive]).sum(-1) - (h_a * node_repr[negative]).sum(-1)
        total += float(torch.sigmoid(margin.float()).sum())
        count += int(anchor.numel())
    return total / count if count else 0.5


def select_merge_team(scores: Mapping[str, float], keys: Mapping[str, tuple], k: int) -> MergeTeam:
    """Group of the argmax-psi expert, then its top-k by psi (ties keep candidate order); alpha = softmax(psi_I)."""
    order = sorted(scores, key=lambda e: -float(scores[e]))
    top_key = keys[order[0]]
    group = [e for e in order if keys[e] == top_key]
    if int(k) > len(group):
        print(f"{_LOG} k={k} exceeds the compatible group ({top_key[0]}, {len(group)} experts); using {len(group)}.")
    team = group[: int(k)]
    psi = torch.tensor([float(scores[e]) for e in team], dtype=torch.float32)
    return MergeTeam(
        compat_key=top_key,
        expert_ids=team,
        competence=psi.tolist(),
        alpha=torch.softmax(psi, dim=0),
        all_scores={e: float(s) for e, s in scores.items()},
    )


__all__ = [
    "GROUP_POLICIES",
    "MergeTeam",
    "assert_state_compatible",
    "compat_key",
    "competence_batches",
    "competence_score",
    "select_merge_team",
    "team_candidates",
]
