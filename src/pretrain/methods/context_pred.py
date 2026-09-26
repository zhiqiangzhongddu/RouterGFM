from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import k_hop_subgraph, subgraph

from src.model import build_encoder_from_cfg
from src.utils.pool import POOLERS, normalize_pool_mode, pool_nodes

from src.utils.config_helpers import cfg_default, tag_if_nondefault, validate_choice

from ..task_base import PretrainTask
from ..registry import register
from .utils import iter_graph_slices, make_zero_loss


class _SamplePair(NamedTuple):
    sub_data: Data
    context_data: Data
    center_local: int
    overlap_local: torch.Tensor


@register("context_pred")
class ContextPred(PretrainTask):
    """Chem-style substructure-context prediction (CBOW / skipgram).

    Reference: Hu et al. "Strategies for Pre-training Graph Neural Networks"
    (ICLR 2020). This reproduces the chem variant
    (``pretrain-gnns/chem/pretrain_contextpred.py``); the PPI-specific bio
    variant from ``pretrain-gnns/bio/`` is intentionally not implemented.

    IcG adaptation:
    - For node / edge tasks with ``pretrain.dataset.induced=True``, root
      sampling happens inside each induced subgraph rather than the original
      full graph, matching how IcG handles induced-subgraph workflows.
    """

    requires_graph_batches = True
    min_graphs_per_batch = 2

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        task_cfg = cfg.pretrain.context_pred
        validate_choice("pretrain.context_pred.mode", task_cfg.mode, {"cbow", "skipgram"})
        context_pooling = normalize_pool_mode(task_cfg.context_pooling)
        validate_choice("pretrain.context_pred.context_pooling", context_pooling, frozenset(POOLERS))
        if int(task_cfg.neg_samples) < 1:
            raise ValueError(
                f"[ContextPred] pretrain.context_pred.neg_samples must be >= 1; "
                f"got {task_cfg.neg_samples}."
            )
        if int(task_cfg.context_size) < 1:
            raise ValueError(
                f"[ContextPred] pretrain.context_pred.context_size must be >= 1; "
                f"got {task_cfg.context_size}."
            )
        # ``substruct_hops == 0`` is the "inherit from model.num_layers"
        # sentinel; any positive value is a literal hop count. Negative
        # values would yield a distinct variant_tag (``h-1``, ``h-2``,
        # ...) while collapsing to the same runtime behavior as 0,
        # causing silent checkpoint-name drift.
        if int(task_cfg.substruct_hops) < 0:
            raise ValueError(
                f"[ContextPred] pretrain.context_pred.substruct_hops must be >= 0 "
                f"(0 means 'inherit from model.num_layers'); got {task_cfg.substruct_hops}."
            )

    @classmethod
    def variant_tag(cls, cfg) -> str:
        _d = "pretrain.context_pred"
        task_cfg = cfg.pretrain.context_pred
        parts: list[str] = []
        parts.append(tag_if_nondefault("", str(task_cfg.mode).lower(), str(cfg_default(f"{_d}.mode")).lower()))
        parts.append(tag_if_nondefault("c", int(task_cfg.context_size), cfg_default(f"{_d}.context_size")))
        parts.append(tag_if_nondefault("h", int(task_cfg.substruct_hops), cfg_default(f"{_d}.substruct_hops")))
        parts.append(tag_if_nondefault("neg", int(task_cfg.neg_samples), cfg_default(f"{_d}.neg_samples")))
        ctx_pool = normalize_pool_mode(task_cfg.context_pooling)
        default_pool = normalize_pool_mode(cfg_default(f"{_d}.context_pooling"))
        parts.append(tag_if_nondefault("", ctx_pool, default_pool))
        return "-".join(p for p in parts if p)

    def __init__(self, cfg):
        super().__init__(cfg)
        task_cfg = cfg.pretrain.context_pred
        self.mode = str(task_cfg.mode).lower()
        self.context_pooling = normalize_pool_mode(task_cfg.context_pooling)
        self.neg_samples = int(task_cfg.neg_samples)
        self.context_size = int(task_cfg.context_size)
        # pretrain-gnns convention: k = num_layer, l1 = k - 1, l2 = l1 + csize.
        # substruct_hops == 0 means "inherit from model.num_layers". We clamp
        # k_hops to >= 1, so the official k == 0 edge case (single-node
        # substruct) cannot be reproduced here — harmless in practice because
        # the inherited num_layers is always >= 1.
        substruct_hops = int(task_cfg.substruct_hops)
        self.k_hops = substruct_hops if substruct_hops > 0 else max(1, int(cfg.model.num_layers))
        self.l1 = max(0, self.k_hops - 1)
        self.l2 = self.l1 + max(1, self.context_size)
        self.criterion = nn.BCEWithLogitsLoss()
        # Auxiliary context encoder uses (l2 - l1) layers, matching the
        # official ``GNN(num_layer=l2-l1, ...)``. All other encoder knobs
        # (graph_pooling, dropout, activation, ...) are inherited from the
        # main model config via ``cfg.clone()`` — when swapping in a non-MPNN
        # backbone, only ``num_layers`` is overridden here.
        context_cfg = cfg.clone()
        context_cfg.model.num_layers = max(1, int(self.l2 - self.l1))
        self.context_encoder = build_encoder_from_cfg(cfg=context_cfg, in_dim=cfg.model.in_dim)

    # ------------------------------ entry --------------------------------- #
    def step(self, model, data, device):
        data = data.to(device)
        edge_index = getattr(data, "edge_index", None)
        edge_attr = getattr(data, "edge_attr", None)
        if edge_index is None or edge_index.numel() == 0:
            return self._zero_output(model=model, device=device)

        pairs = self._sample_pairs_for_batch(
            data=data,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        if not pairs:
            return self._zero_output(model=model, device=device)

        substruct_rep, overlapped_cat, pair_ids = self._encode_pairs(
            model=model,
            pairs=pairs,
            device=device,
        )
        # Need at least 2 valid pairs for cyclic negatives; otherwise
        # the shifted index would produce self-negatives.
        if substruct_rep.size(0) < 2:
            return self._zero_output(
                model=model,
                device=device,
                num_pairs=substruct_rep.size(0),
                valid_graphs=len(pairs),
            )

        # ``mode`` is validated in ``validate_cfg`` before construction.
        if self.mode == "cbow":
            pred_pos, pred_neg = self._score_cbow(
                substruct_rep=substruct_rep,
                overlapped_cat=overlapped_cat,
                pair_ids=pair_ids,
            )
        else:  # skipgram
            pred_pos, pred_neg = self._score_skipgram(
                substruct_rep=substruct_rep,
                overlapped_cat=overlapped_cat,
                pair_ids=pair_ids,
            )

        loss, acc = self._bce_loss_and_acc(pred_pos=pred_pos, pred_neg=pred_neg)
        return loss, {
            "train_acc": acc,
            "num_pairs_count": float(substruct_rep.size(0)),
            "valid_graphs_count": float(len(pairs)),
        }

    # ------------------------ pair construction --------------------------- #
    def _sample_pairs_for_batch(self, data, edge_index, edge_attr):
        x = data.x
        pairs: list[_SamplePair] = []
        for start, end in iter_graph_slices(data):
            num_graph_nodes = int(end - start)
            if num_graph_nodes <= 1:
                continue
            in_graph = (
                (edge_index[0] >= start)
                & (edge_index[0] < end)
                & (edge_index[1] >= start)
                & (edge_index[1] < end)
            )
            graph_edge_index = edge_index[:, in_graph] - start
            if graph_edge_index.numel() == 0:
                continue
            graph_edge_attr = edge_attr[in_graph] if edge_attr is not None else None
            x_graph = x[start:end]
            pair = self._sample_pair_for_graph(
                x_graph=x_graph,
                graph_edge_index=graph_edge_index,
                graph_edge_attr=graph_edge_attr,
                num_graph_nodes=num_graph_nodes,
            )
            if pair is not None:
                pairs.append(pair)
        return pairs

    def _sample_pair_for_graph(
        self,
        x_graph,
        graph_edge_index,
        graph_edge_attr,
        num_graph_nodes,
    ):
        root_idx = int(torch.randint(0, num_graph_nodes, (1,)).item())

        # k-hop rooted substructure
        sub_nodes, sub_edge_index, mapping, sub_edge_mask = k_hop_subgraph(
            node_idx=root_idx,
            num_hops=self.k_hops,
            edge_index=graph_edge_index,
            relabel_nodes=True,
            num_nodes=num_graph_nodes,
        )
        if sub_nodes.numel() == 0:
            return None
        sub_edge_attr = graph_edge_attr[sub_edge_mask] if graph_edge_attr is not None else None

        # Context nodes: symmetric difference between <=l1 and <=l2 neighborhoods.
        l1_nodes, _, _, _ = k_hop_subgraph(
            node_idx=root_idx,
            num_hops=self.l1,
            edge_index=graph_edge_index,
            relabel_nodes=False,
            num_nodes=num_graph_nodes,
        )
        l2_nodes, _, _, _ = k_hop_subgraph(
            node_idx=root_idx,
            num_hops=self.l2,
            edge_index=graph_edge_index,
            relabel_nodes=False,
            num_nodes=num_graph_nodes,
        )
        if l2_nodes.numel() == 0:
            return None
        if l1_nodes.numel() == 0:
            context_nodes = l2_nodes
        else:
            context_nodes = l2_nodes[~torch.isin(l2_nodes, l1_nodes)]
        if context_nodes.numel() == 0:
            return None
        context_nodes = torch.unique(context_nodes, sorted=True)

        context_edge_index, context_edge_attr = subgraph(
            subset=context_nodes,
            edge_index=graph_edge_index,
            edge_attr=graph_edge_attr,
            relabel_nodes=True,
            num_nodes=num_graph_nodes,
        )

        overlap_local = torch.nonzero(
            torch.isin(context_nodes, sub_nodes), as_tuple=False
        ).view(-1)
        if overlap_local.numel() == 0:
            return None

        sub_data_kwargs = {"x": x_graph[sub_nodes], "edge_index": sub_edge_index}
        if sub_edge_attr is not None:
            sub_data_kwargs["edge_attr"] = sub_edge_attr
        sub_data = Data(**sub_data_kwargs)

        context_data_kwargs = {"x": x_graph[context_nodes], "edge_index": context_edge_index}
        if context_edge_attr is not None:
            context_data_kwargs["edge_attr"] = context_edge_attr
        context_data = Data(**context_data_kwargs)

        return _SamplePair(
            sub_data=sub_data,
            context_data=context_data,
            center_local=int(mapping.view(-1)[0].item()),
            overlap_local=overlap_local,
        )

    # ------------------------------ forward ------------------------------- #
    def _encode_pairs(self, model, pairs, device):
        sub_batch = Batch.from_data_list([p.sub_data for p in pairs])
        context_batch = Batch.from_data_list([p.context_data for p in pairs])
        sub_node_rep, _ = model(sub_batch)
        context_node_rep, _ = self.context_encoder(context_batch)

        sub_ptr = sub_batch.ptr
        context_ptr = context_batch.ptr

        center_local = torch.as_tensor(
            [p.center_local for p in pairs], dtype=torch.long, device=device
        )
        center_global = sub_ptr[:-1] + center_local
        substruct_rep = sub_node_rep[center_global]

        overlap_sizes = torch.as_tensor(
            [int(p.overlap_local.numel()) for p in pairs],
            dtype=torch.long,
            device=device,
        )
        pair_ids = torch.arange(len(pairs), device=device).repeat_interleave(overlap_sizes)
        context_offsets = context_ptr[:-1]
        overlap_global = torch.cat(
            [pairs[i].overlap_local + context_offsets[i] for i in range(len(pairs))],
            dim=0,
        )
        overlapped_cat = context_node_rep[overlap_global]
        return substruct_rep, overlapped_cat, pair_ids

    # ------------------------------ scoring ------------------------------- #
    def _effective_neg_samples(self, population: int) -> int:
        # A cyclic shift of ``population`` (or a multiple) is the identity, so
        # such "negatives" would be the positives labelled 0. Only
        # ``population - 1`` distinct non-trivial shifts exist.
        return min(int(self.neg_samples), max(0, int(population) - 1))

    def _score_cbow(self, substruct_rep, overlapped_cat, pair_ids):
        context_rep = pool_nodes(
            x=overlapped_cat,
            batch=pair_ids,
            mode=self.context_pooling,
        )
        pred_pos = torch.sum(substruct_rep * context_rep, dim=1)
        effective_negs = self._effective_neg_samples(len(context_rep))
        if effective_negs == 0:
            return pred_pos, pred_pos.new_empty(0)
        neg_context_rep = torch.cat(
            [
                context_rep[self._cycle_index(len(context_rep), i + 1, context_rep.device)]
                for i in range(effective_negs)
            ],
            dim=0,
        )
        pred_neg = torch.sum(
            substruct_rep.repeat((effective_negs, 1)) * neg_context_rep,
            dim=1,
        )
        return pred_pos, pred_neg

    def _score_skipgram(self, substruct_rep, overlapped_cat, pair_ids):
        expanded_substruct = substruct_rep[pair_ids]
        pred_pos = torch.sum(expanded_substruct * overlapped_cat, dim=1)

        effective_negs = self._effective_neg_samples(len(substruct_rep))
        if effective_negs == 0:
            return pred_pos, pred_pos.new_empty(0)
        shifted_expanded = []
        for i in range(effective_negs):
            shifted = substruct_rep[
                self._cycle_index(len(substruct_rep), i + 1, substruct_rep.device)
            ]
            shifted_expanded.append(shifted[pair_ids])
        shifted_expanded = torch.cat(shifted_expanded, dim=0)
        pred_neg = torch.sum(
            shifted_expanded * overlapped_cat.repeat((effective_negs, 1)),
            dim=1,
        )
        return pred_pos, pred_neg

    def _bce_loss_and_acc(self, pred_pos, pred_neg):
        # Cast to float64 for numerical stability, matching the official impl.
        loss_pos = self.criterion(
            pred_pos.double(),
            torch.ones(len(pred_pos), device=pred_pos.device).double(),
        )
        if pred_neg.numel() == 0:
            loss = loss_pos
        else:
            loss_neg = self.criterion(
                pred_neg.double(),
                torch.zeros(len(pred_neg), device=pred_neg.device).double(),
            )
            # Weight by the realised negatives-per-positive ratio: when the
            # batch is too small for the configured ``neg_samples`` we draw
            # fewer shifts (see _effective_neg_samples).
            neg_weight = float(pred_neg.numel()) / float(max(1, pred_pos.numel()))
            loss = loss_pos + neg_weight * loss_neg

        acc_pos = float((pred_pos > 0).float().mean().item()) if pred_pos.numel() > 0 else 0.0
        acc_neg = float((pred_neg < 0).float().mean().item()) if pred_neg.numel() > 0 else 0.0
        acc = 0.5 * (acc_pos + acc_neg)
        return loss, acc

    # ------------------------------ helpers ------------------------------- #
    @staticmethod
    def _cycle_index(num: int, shift: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(num, device=device)
        if num <= 0:
            return idx
        shift = int(shift) % int(num)
        if shift == 0:
            return idx
        return torch.cat([idx[shift:], idx[:shift]], dim=0)

    def _zero_output(
        self,
        model,
        device,
        num_pairs: float = 0.0,
        valid_graphs: float = 0.0,
    ):
        main_param = next(model.parameters(), None)
        zero = make_zero_loss(self, device, main_param)
        return zero, {
            "train_acc": 0.0,
            "num_pairs_count": float(num_pairs),
            "valid_graphs_count": float(valid_graphs),
        }
