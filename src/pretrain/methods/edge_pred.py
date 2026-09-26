from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from src.utils.config_helpers import cfg_default, tag_if_nondefault, validate_probability

from ..task_base import PretrainTask
from ..registry import register
from .utils import make_zero_loss, resolve_ptr


def _unique_undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Deduplicate reciprocal directed edges into unique undirected pairs.

    Unlike the official ``::2`` stride (which assumes adjacent-pair storage
    produced by ``to_undirected``), this orients every edge as ``(min, max)``
    and keeps one copy per unordered pair, which is robust regardless of edge
    ordering and keeps directed-only edges (a ``src < dst`` filter would
    silently drop edges stored only as ``(u, v)`` with ``u > v``, as in the
    WebKB datasets). Self-loops are dropped (their dot-product score is
    trivially large).
    """
    if edge_index is None or edge_index.numel() == 0:
        return edge_index
    src, dst = edge_index[0], edge_index[1]
    mask = src != dst
    lo = torch.minimum(src[mask], dst[mask])
    hi = torch.maximum(src[mask], dst[mask])
    oriented = torch.stack([lo, hi], dim=0)
    return torch.unique(oriented, dim=1)


def _undirected_edge_closure(edge_index: torch.Tensor) -> torch.Tensor:
    """Return both orientations of every non-self-loop edge.

    ``torch_geometric.utils.negative_sampling`` treats its input as directed
    unless explicitly asked otherwise. EdgePrediction scores unordered
    positive pairs, so sampling against only the observed orientation can
    incorrectly label the reverse of a real edge as negative. Supplying this
    closure makes both orientations unavailable to the sampler while keeping
    its requested sample-count semantics unchanged.
    """
    unique = _unique_undirected_edges(edge_index)
    if unique is None or unique.numel() == 0:
        return unique
    return torch.cat([unique, unique.flip(0)], dim=1)


@register("edge_pred")
class EdgePrediction(PretrainTask):
    """Edge Prediction pretraining task.
    
    Reference: Hu et al. "Strategies for Pre-training Graph Neural Networks" ICLR 2020.
    """

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        task_cfg = cfg.pretrain.edge_pred
        neg_ratio = float(task_cfg.neg_ratio)
        if neg_ratio <= 0.0:
            raise ValueError(
                f"[EdgePrediction] pretrain.edge_pred.neg_ratio must be > 0; "
                f"got {neg_ratio}. Disabling the negative term produces a "
                "degenerate objective; use a small positive value instead."
            )
        validate_probability("pretrain.edge_pred.pos_edge_ratio", task_cfg.pos_edge_ratio, low=0.0, high=1.0)
        pos_edge_ratio = float(task_cfg.pos_edge_ratio)
        if pos_edge_ratio <= 0.0:
            raise ValueError(
                f"[EdgePrediction] pretrain.edge_pred.pos_edge_ratio must be in (0, 1]; "
                f"got {pos_edge_ratio}."
            )
        validate_probability("pretrain.edge_pred.forward_edge_ratio", task_cfg.forward_edge_ratio, low=0.0, high=1.0)
        forward_edge_ratio = float(task_cfg.forward_edge_ratio)
        if forward_edge_ratio <= 0.0:
            raise ValueError(
                f"[EdgePrediction] pretrain.edge_pred.forward_edge_ratio must be in (0, 1]; "
                f"got {forward_edge_ratio}."
            )
        # ``*_max == 0`` is the "disabled" sentinel; any positive value is
        # a literal cap. Negative values used to collapse to the same
        # disabled behavior silently, but variant_tag only emits a tag
        # for ``> 0`` so the situation was drift-safe in practice — still,
        # reject negatives so cfg errors surface at validate time.
        pos_edge_max = int(task_cfg.pos_edge_max)
        if pos_edge_max < 0:
            raise ValueError(
                f"[EdgePrediction] pretrain.edge_pred.pos_edge_max must be >= 0 "
                f"(0 disables the cap); got {pos_edge_max}."
            )
        forward_edge_max = int(task_cfg.forward_edge_max)
        if forward_edge_max < 0:
            raise ValueError(
                f"[EdgePrediction] pretrain.edge_pred.forward_edge_max must be >= 0 "
                f"(0 disables the cap); got {forward_edge_max}."
            )

    @classmethod
    def variant_tag(cls, cfg) -> str:
        _d = "pretrain.edge_pred"
        task_cfg = cfg.pretrain.edge_pred
        parts: list[str] = []
        parts.append(tag_if_nondefault("mlp", bool(task_cfg.use_mlp_scorer), cfg_default(f"{_d}.use_mlp_scorer")))
        parts.append(tag_if_nondefault("neg", float(task_cfg.neg_ratio), cfg_default(f"{_d}.neg_ratio")))
        parts.append(tag_if_nondefault("pe", float(task_cfg.pos_edge_ratio), cfg_default(f"{_d}.pos_edge_ratio")))
        pos_edge_max = int(task_cfg.pos_edge_max)
        if pos_edge_max > 0:
            parts.append(f"pm{pos_edge_max}")
        parts.append(tag_if_nondefault("fe", float(task_cfg.forward_edge_ratio), cfg_default(f"{_d}.forward_edge_ratio")))
        forward_edge_max = int(task_cfg.forward_edge_max)
        if forward_edge_max > 0:
            parts.append(f"fm{forward_edge_max}")
        return "-".join(p for p in parts if p)

    def __init__(self, cfg):
        super().__init__(cfg)
        task_cfg = cfg.pretrain.edge_pred
        self.pos_edge_ratio = float(task_cfg.pos_edge_ratio)
        self.pos_edge_max = int(task_cfg.pos_edge_max)
        self.neg_ratio = float(task_cfg.neg_ratio)
        self.forward_edge_ratio = float(task_cfg.forward_edge_ratio)
        self.forward_edge_max = int(task_cfg.forward_edge_max)
        self.use_mlp_scorer = bool(task_cfg.use_mlp_scorer)
        if self.use_mlp_scorer:
            hidden = cfg.model.out_dim
            self.scorer = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.scorer = None

    def score_edges(
        self, node_repr: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index
        if self.scorer is None:
            # Original pretrain-gnns formulation: dot-product edge score.
            return torch.sum(node_repr[src] * node_repr[dst], dim=-1)
        pairs = torch.cat([node_repr[src], node_repr[dst]], dim=-1)
        return self.scorer(pairs).view(-1)

    def step(
        self, model: nn.Module, data, device
    ) -> tuple[torch.Tensor, dict]:
        data = data.to(device)
        # ``original_edge_index`` always refers to the full graph, even when
        # ``forward_edge_ratio``/``forward_edge_max`` drop edges from the
        # message-passing input. Supervision (positives) and negative sampling
        # must both be defined against the full graph, otherwise dropped real
        # edges can reappear as "negatives" and sampled negatives can collide
        # with real-but-hidden edges.
        data, original_edge_index = self._sample_forward_edges(data)
        node_repr, _ = model(data)
        pos_edge = _unique_undirected_edges(original_edge_index)
        pos_edge = self._subsample_positive_edges(pos_edge)
        if pos_edge is None or pos_edge.numel() == 0 or pos_edge.size(1) == 0:
            zero = make_zero_loss(self, device, node_repr)
            return zero, {
                "pos_mean": 0.0,
                "neg_mean": 0.0,
            }
        neg_edge = self._sample_negatives_per_graph(
            data, pos_edge, existing_edge_index=original_edge_index,
        )
        pos_logits = self.score_edges(node_repr, pos_edge)
        # Negative-sample degeneracy (a rare per-graph distribution where
        # negative_sampling returns nothing) must be handled before the
        # BCE: F.binary_cross_entropy_with_logits on empty tensors reduces
        # 0/0 -> NaN and silently poisons the optimizer. Skip the batch
        # with a zero-loss anchored to the encoder so backward() is safe,
        # mirroring the pos-empty path. (neg_ratio == 0 is rejected by
        # validate_cfg, so the only path here is sampler exhaustion.)
        if neg_edge is None or neg_edge.numel() == 0 or neg_edge.size(1) == 0:
            zero = make_zero_loss(self, device, node_repr)
            return zero, {
                "pos_mean": float(torch.sigmoid(pos_logits).detach().mean().item()),
                "neg_mean": 0.0,
            }
        neg_logits = self.score_edges(node_repr, neg_edge)
        # Match official pretrain-gnns loss: sum of two independently
        # mean-reduced BCE terms (pos and neg), not a single mean over the
        # concatenation.  This preserves the effective gradient scale.
        loss = (
            F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits)
            )
            + F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits)
            )
        )
        return loss, {
            "pos_mean": torch.sigmoid(pos_logits).detach().mean().item(),
            "neg_mean": torch.sigmoid(neg_logits).detach().mean().item(),
        }

    def _subsample_positive_edges(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply ``pos_edge_ratio`` / ``pos_edge_max`` to a positive-edge set.

        Do **not** reuse this for negative edges — the knobs it reads are
        positive-edge specific. Use ``_random_select`` when an exact count is
        required (e.g. trimming an over-sized negative set).
        """
        if self.pos_edge_ratio >= 1.0 and self.pos_edge_max <= 0:
            return edge_index

        total = edge_index.size(1)
        target = total
        if self.pos_edge_ratio < 1.0:
            target = int(total * self.pos_edge_ratio)
        if self.pos_edge_max > 0:
            target = min(target, self.pos_edge_max)
        target = max(1, target)
        if target >= total:
            return edge_index

        return self._random_select(edge_index, target)

    @staticmethod
    def _random_select(edge_index: torch.Tensor, k: int) -> torch.Tensor:
        """Select exactly ``k`` edges uniformly at random (or all if ``k`` >= total)."""
        total = edge_index.size(1)
        if k >= total or k <= 0:
            return edge_index
        perm = torch.randperm(total, device=edge_index.device)[:k]
        return edge_index[:, perm]

    def _sample_negatives_per_graph(
        self, data, pos_edge: torch.Tensor, existing_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Sample negatives within each graph, not across the batch.

        Matches the official pretrain-gnns behaviour where ``NegativeEdge``
        samples per-graph *before* batching, so no cross-component negatives
        can appear.  For single-graph batches this falls back to a single
        ``negative_sampling`` call.

        ``existing_edge_index`` is the full-graph edge set used as the "edges
        to avoid" constraint.  It is passed explicitly instead of reading
        ``data.edge_index`` because ``data.edge_index`` may be a reduced
        message-passing subset (see ``_sample_forward_edges``); sampling
        against the subset would allow dropped real edges to masquerade as
        negatives.

        The per-graph budget is distributed proportionally to the per-graph
        positive count so that the total respects
        ``_num_neg_samples(pos_edge)``.
        """
        batch = getattr(data, "batch", None)
        all_edges = _undirected_edge_closure(existing_edge_index)
        total_neg = self._num_neg_samples(pos_edge)

        # Fast path: single graph in the batch.
        if batch is None or int(batch.max()) == 0:
            return negative_sampling(
                all_edges,
                num_nodes=data.num_nodes,
                num_neg_samples=total_neg,
            )

        # Per-graph negative sampling.
        ptr = resolve_ptr(data)
        if ptr is None:
            # Fallback: treat as single-graph batch.
            return negative_sampling(
                all_edges,
                num_nodes=data.num_nodes,
                num_neg_samples=total_neg,
            )

        num_graphs = int(ptr.numel() - 1)
        total_pos = int(pos_edge.size(1))
        if total_pos == 0 or total_neg == 0:
            return torch.empty((2, 0), dtype=torch.long, device=all_edges.device)

        neg_src_parts: list[torch.Tensor] = []
        neg_dst_parts: list[torch.Tensor] = []
        remaining_neg = total_neg
        remaining_pos = total_pos

        for g in range(num_graphs):
            node_start = int(ptr[g])
            node_end = int(ptr[g + 1])
            # Edges belonging to this graph (already offset in the mega-graph).
            edge_mask = (all_edges[0] >= node_start) & (all_edges[0] < node_end)
            g_edges = all_edges[:, edge_mask] - node_start  # local indices
            g_num_nodes = node_end - node_start

            # Proportional per-graph negative budget; the last graph with
            # positives absorbs any rounding remainder so the total matches
            # ``total_neg`` exactly.
            pos_mask = (pos_edge[0] >= node_start) & (pos_edge[0] < node_end)
            g_pos = int(pos_mask.sum().item())
            if g_pos == 0 or remaining_pos <= 0:
                continue
            if g_pos >= remaining_pos:
                g_num_neg = remaining_neg
            else:
                g_num_neg = int(round(total_neg * g_pos / total_pos))
                g_num_neg = min(g_num_neg, remaining_neg)
            if g_num_neg <= 0:
                remaining_pos -= g_pos
                continue

            g_neg = negative_sampling(
                g_edges,
                num_nodes=g_num_nodes,
                num_neg_samples=g_num_neg,
            )
            # Shift back to mega-graph indices.
            neg_src_parts.append(g_neg[0] + node_start)
            neg_dst_parts.append(g_neg[1] + node_start)

            remaining_neg -= g_num_neg
            remaining_pos -= g_pos

        if not neg_src_parts:
            return torch.empty((2, 0), dtype=torch.long, device=all_edges.device)
        return torch.stack([
            torch.cat(neg_src_parts, dim=0),
            torch.cat(neg_dst_parts, dim=0),
        ])

    def _num_neg_samples(self, pos_edge: torch.Tensor) -> int:
        count = int(pos_edge.size(1) * self.neg_ratio)
        if pos_edge.numel() > 0 and count == 0:
            count = 1
        return count

    def _sample_forward_edges(self, data):
        """Optionally subsample the message-passing edge set.

        Returns ``(data, original_edge_index)``. ``data.edge_index`` may be
        replaced with a strictly smaller subset when
        ``forward_edge_ratio``/``forward_edge_max`` are configured, but the
        *original* full edge index is always returned as the second element
        so supervision and negative sampling can still reference the true
        graph. When no reduction applies, ``data`` is returned unchanged and
        ``original_edge_index`` is just ``data.edge_index``.
        """
        original_edge_index = getattr(data, "edge_index", None)
        if original_edge_index is None:
            return data, original_edge_index
        if self.forward_edge_ratio >= 1.0 and self.forward_edge_max <= 0:
            return data, original_edge_index

        total = original_edge_index.size(1)
        target = total
        if self.forward_edge_ratio < 1.0:
            target = int(total * self.forward_edge_ratio)
        if self.forward_edge_max > 0:
            target = min(target, self.forward_edge_max)
        target = max(1, target)
        if target >= total:
            return data, original_edge_index

        perm = torch.randperm(total, device=original_edge_index.device)
        idx = perm[:target]
        sampled = original_edge_index[:, idx]

        data = data.clone()
        data.edge_index = sampled
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None and edge_attr.size(0) == total:
            data.edge_attr = edge_attr[idx]
        return data, original_edge_index
