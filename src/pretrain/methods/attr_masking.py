from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config_helpers import cfg_default, tag_if_nondefault, validate_choice, validate_probability

from ..task_base import PretrainTask
from ..registry import register
from .utils import make_zero_loss, sample_masked_node_indices


@register("attr_masking")
class AttrMasking(PretrainTask):
    """Masked node-feature reconstruction pretraining task.

    Randomly masks ``mask_ratio`` of node features with a learnable mask
    token and trains the encoder to reconstruct them from neighborhood
    context. Default loss is MSE on the full feature vector; set
    ``node_loss='ce'`` with ``node_vocab_size >= 2`` to switch to
    cross-entropy classification on ``x[:, 0]``.

    Inspired by Hu et al. "Strategies for Pre-training Graph Neural
    Networks" (ICLR 2020), adapted for generic graph encoders with
    continuous node features.

    NOTE: This implementation only supports **node** attribute masking.
    Edge attribute masking from the original paper is intentionally not
    implemented -- the project's default encoder family does not consume
    ``edge_attr``, so an edge objective would not reach the encoder.
    """

    @classmethod
    def validate_cfg(cls, cfg) -> None:
        task_cfg = cfg.pretrain.attr_masking
        node_loss_mode = validate_choice(
            "pretrain.attr_masking.node_loss", task_cfg.node_loss, {"mse", "ce"},
        )
        node_vocab_size = int(task_cfg.node_vocab_size)
        if node_loss_mode == "ce" and node_vocab_size < 2:
            raise ValueError(
                "[AttrMasking] node_loss='ce' requires node_vocab_size >= 2; "
                f"got node_vocab_size={node_vocab_size}."
            )
        validate_probability(
            "pretrain.attr_masking.mask_ratio", task_cfg.mask_ratio,
            low=0.0, high=1.0,
        )
        mask_ratio = float(task_cfg.mask_ratio)
        if mask_ratio <= 0.0 or mask_ratio >= 1.0:
            raise ValueError(
                f"[AttrMasking] pretrain.attr_masking.mask_ratio must be in (0, 1) exclusive; "
                f"got {mask_ratio}."
            )

    @classmethod
    def variant_tag(cls, cfg) -> str:
        task_cfg = cfg.pretrain.attr_masking
        parts: list[str] = []
        parts.append(tag_if_nondefault(
            "mr", float(task_cfg.mask_ratio), cfg_default("pretrain.attr_masking.mask_ratio"),
        ))
        node_loss = str(task_cfg.node_loss).lower()
        default_node_loss = str(cfg_default("pretrain.attr_masking.node_loss")).lower()
        if node_loss != default_node_loss:
            vocab = int(task_cfg.node_vocab_size)
            parts.append(f"{node_loss}{vocab}" if vocab > 0 else node_loss)
        return "-".join(p for p in parts if p)

    def __init__(self, cfg):
        super().__init__(cfg)
        task_cfg = cfg.pretrain.attr_masking
        self.mask_ratio = float(task_cfg.mask_ratio)
        self.node_loss_mode = str(task_cfg.node_loss).lower()
        self.node_vocab_size = int(task_cfg.node_vocab_size)
        self.mask_token = nn.Parameter(torch.zeros(cfg.model.in_dim))

        if self.node_loss_mode == "mse":
            self.node_reg_head = nn.Linear(cfg.model.out_dim, cfg.model.in_dim)
            self.node_cls_head = None
        else:
            self.node_reg_head = None
            self.node_cls_head = nn.Linear(cfg.model.out_dim, self.node_vocab_size)

    @staticmethod
    def _extract_class_targets(values: torch.Tensor, vocab_size: int):
        if values.dim() > 1:
            values = values[:, 0]
        raw = values.float()
        target = torch.round(raw).long()
        valid = torch.isfinite(raw)
        valid = valid & (target >= 0) & (target < int(vocab_size))
        return target, valid

    def step(self, model: nn.Module, data, device):
        data = data.to(device)

        num_nodes = data.num_nodes
        if num_nodes <= 0:
            model_param = next(model.parameters(), None)
            return make_zero_loss(self, device, model_param), {"masked_count": 0.0}

        perm = sample_masked_node_indices(
            num_nodes=num_nodes,
            ptr=getattr(data, "ptr", None),
            batch=getattr(data, "batch", None),
            mask_ratio=self.mask_ratio,
            device=device,
        )
        if perm.numel() == 0:
            model_param = next(model.parameters(), None)
            return make_zero_loss(self, device, model_param), {"masked_count": 0.0}

        target = data.x[perm]

        corrupted = data.clone()
        corrupted.x[perm] = self.mask_token

        node_repr, _ = model(corrupted)
        masked_repr = node_repr[perm]

        logs: dict[str, float] = {"masked_count": float(perm.numel())}
        if self.node_loss_mode == "ce":
            node_target, node_valid = self._extract_class_targets(target, self.node_vocab_size)
            if bool(node_valid.any().item()):
                logits = self.node_cls_head(masked_repr[node_valid])
                loss = F.cross_entropy(logits, node_target[node_valid])
                logs["train_acc"] = float(
                    (logits.argmax(dim=-1) == node_target[node_valid]).float().mean().item()
                )
            else:
                loss = make_zero_loss(self, device, node_repr)
        else:
            pred = self.node_reg_head(masked_repr)
            loss = F.mse_loss(pred, target.float())
            # No extra diagnostic key: in MSE mode the scalar equals train_loss.

        return loss, logs
