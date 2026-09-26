"""GraphPrompt modules for finetuning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GraphPrompt(nn.Module):
    """Feature-weighted prompt used in official GraphPrompt finetuning."""

    def __init__(
        self,
        in_channels: int,
        init: str = "xavier",
        init_std: float = 0.02,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.init = str(init).lower()
        self.init_std = float(init_std)
        self.weight = nn.Parameter(torch.empty(1, self.in_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init in {"identity", "ones"}:
            with torch.no_grad():
                self.weight.fill_(1.0)
                if self.init_std > 0:
                    self.weight.add_(torch.randn_like(self.weight) * self.init_std)
        elif self.init in {"xavier", "xavier_uniform"}:
            nn.init.xavier_uniform_(self.weight)
        else:
            raise ValueError(f"Unknown GraphPrompt init mode: {self.init}")

    def forward(self, embeddings: Tensor) -> Tensor:
        return embeddings * self.weight


class GraphPromptPlusStageWise(nn.Module):
    """
    Stage-wise GraphPrompt+ prompt bank inspired by the extension code path.

    Stages:
    - 0: input features
    - 1: after first hidden layer
    - 2: after second hidden layer
    - 3: readout/final node representation

    The set of active stages and their per-stage dims are normally derived
    from ``num_layers`` (stack-style backbones), but ``stage_specs`` lets a
    backbone-specific GraphPrompt+ adapter override that — e.g. H2GCN has a
    fixed 2-hop structure, so only stages {0, 1, 3} apply.  When
    ``stage_specs`` is given the four ``in/hidden/out`` arguments are used
    only as fallback dims for stages whose dim is left as ``None``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        p_num: int = 4,
        init: str = "xavier",
        init_std: float = 0.02,
        *,
        stage_specs: list[tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)
        self.num_layers = int(num_layers)
        self.p_num = max(1, int(p_num))
        self.init = str(init).lower()
        self.init_std = float(init_std)

        if stage_specs is None:
            # Default stack-style layout: stage dim follows in/hidden/.../out.
            available_stages = [(0, self.in_channels)]
            if self.num_layers >= 2:
                available_stages.append((1, self.hidden_channels))
            if self.num_layers >= 3:
                available_stages.append((2, self.hidden_channels))
            available_stages.append((3, self.out_channels))
        else:
            available_stages = [(int(sid), int(dim)) for sid, dim in stage_specs]

        active = available_stages[: self.p_num]
        self.stage_masks = nn.ParameterDict(
            {str(sid): nn.Parameter(torch.empty(1, dim)) for sid, dim in active}
        )
        self._active_stage_ids = tuple(sid for sid, _dim in active)
        self.temp = nn.Parameter(torch.empty(len(self._active_stage_ids), 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for mask in self.stage_masks.values():
            if self.init in {"identity", "ones"}:
                with torch.no_grad():
                    mask.fill_(1.0)
                    if self.init_std > 0:
                        mask.add_(torch.randn_like(mask) * self.init_std)
            elif self.init in {"xavier", "xavier_uniform"}:
                nn.init.xavier_uniform_(mask)
            else:
                raise ValueError(f"Unknown GraphPromptPlusStageWise init mode: {self.init}")
        nn.init.uniform_(self.temp, a=0.0, b=0.1)

    @property
    def active_stage_ids(self):
        return self._active_stage_ids

    def has_stage(self, stage_id: int) -> bool:
        return int(stage_id) in self._active_stage_ids

    def stage_coefficients(self) -> Tensor:
        return F.softmax(self.temp, dim=0).view(-1)

    def iter_stage_coefficients(self):
        coeff = self.stage_coefficients()
        for idx, stage_id in enumerate(self._active_stage_ids):
            yield stage_id, coeff[idx]

    def apply_stage(self, stage_id: int, embeddings: Tensor) -> Tensor:
        stage_key = str(int(stage_id))
        if stage_key not in self.stage_masks:
            return embeddings
        mask = self.stage_masks[stage_key]
        if embeddings.size(-1) != mask.size(-1):
            raise ValueError(
                f"Stage-wise prompt dim mismatch at stage={stage_id}: "
                f"expected {int(mask.size(-1))}, got {int(embeddings.size(-1))}."
            )
        return embeddings * mask


def compute_class_centers(embeddings: Tensor, labels: Tensor, num_classes: int):
    """Compute per-class prototype centers and class counts."""
    labels = torch.as_tensor(labels).view(-1).long()
    num_classes = int(num_classes)
    if labels.numel() == 0:
        centers = embeddings.new_zeros((num_classes, embeddings.size(-1)))
        counts = embeddings.new_zeros((num_classes, 1))
        return centers, counts

    centers = embeddings.new_zeros((num_classes, embeddings.size(-1)))
    index = labels.unsqueeze(1).expand(-1, embeddings.size(-1))
    centers = centers.scatter_add_(dim=0, index=index, src=embeddings)

    counts = torch.bincount(labels, minlength=num_classes).to(
        dtype=embeddings.dtype,
        device=embeddings.device,
    ).unsqueeze(1)
    centers = centers / counts.clamp_min(1.0)
    return centers, counts
