"""Model Spider ranker (Zhang et al., NeurIPS 2023, Eq. 6/9; official ``LearnwareCAHeterogeneous``).

For expert e and task b the official model runs one post-LN attention block
(``MultiHeadAttention``, no feed-forward) over ``[theta_e; G_b (; S_{b,e})]``
and reads out position 0 with ``LayerNorm -> Linear``. Only row 0 of that
block is ever used, so it is computed exactly and alone: the task tokens'
keys/values are projected once per task and shared by every expert, costing
O(M C d) per task instead of M full self-attention sequences.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .tokens import pad_token_sets

SPECIFIC_TOKEN_MODES = ("append", "replace")


class ModelSpiderRanker(nn.Module):
    """Scores ``sim(theta_e, task tokens)`` for every expert of a batch of tasks (higher = better).

    * ``theta``: one N(0, 1) model token per expert id.
    * General tokens: ``general_proj`` (official ``uni_linear``) of descriptor centres.
    * Specific tokens: a per-expert ``Linear(d_e, d)`` (official ``hete_linears``) of
      expert-readout centres, stored stacked per readout width ``d_e``.
    * ``type_prompts``: learned offsets added to general / specific tokens (paper).
    * ``specific_token_mode``: ``append`` = ``[theta; G; S]`` (official code),
      ``replace`` = ``[theta; S]`` (paper).
    """

    def __init__(
        self,
        expert_ids: Sequence[str],
        descriptor_dim: int,
        specific_dims: Sequence[int],
        *,
        token_dim: int = 128,
        num_heads: int = 1,
        dropout: float = 0.1,
        type_prompts: bool = False,
        specific_token_mode: str = "append",
    ):
        super().__init__()
        if specific_token_mode not in SPECIFIC_TOKEN_MODES:
            raise ValueError(f"specific_token_mode must be one of {SPECIFIC_TOKEN_MODES}, got {specific_token_mode!r}")
        d, h = int(token_dim), int(num_heads)
        self.expert_ids = [str(e) for e in expert_ids]
        self.token_dim, self.num_heads, self.mode = d, h, specific_token_mode
        self.specific_dims = sorted({int(x) for x in specific_dims})
        m = len(self.expert_ids)

        self.theta = nn.Parameter(torch.randn(m, d))
        self.general_proj = nn.Linear(int(descriptor_dim), d)
        self.spec_weight = nn.ParameterDict()
        self.spec_bias = nn.ParameterDict()
        for de in self.specific_dims:  # nn.Linear default init, one projection per expert
            bound = 1.0 / math.sqrt(de)
            self.spec_weight[str(de)] = nn.Parameter(torch.empty(m, de, d).uniform_(-bound, bound))
            self.spec_bias[str(de)] = nn.Parameter(torch.empty(m, d).uniform_(-bound, bound))
        self.p_gen = nn.Parameter(torch.randn(d)) if type_prompts else None
        self.p_spec = nn.Parameter(torch.randn(d)) if type_prompts else None

        # Official MultiHeadAttention(n_head=h, d_model=d, d_k=d, d_v=d, dropout) + mlp_head.
        self.w_q = nn.Linear(d, h * d, bias=False)
        self.w_k = nn.Linear(d, h * d, bias=False)
        self.w_v = nn.Linear(d, h * d, bias=False)
        for lin in (self.w_q, self.w_k, self.w_v):
            nn.init.normal_(lin.weight, mean=0.0, std=math.sqrt(2.0 / (d + d)))
        self.temperature = math.sqrt(d)
        self.attn_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(h * d, d)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def _kv(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = tokens.shape[:-1] + (self.num_heads, self.token_dim)
        return self.w_k(tokens).view(shape), self.w_v(tokens).view(shape)

    def encode_general(self, centers: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Keys and values ``[B, C, heads, d]`` of the projected (+prompt) general tokens."""
        tokens = self.general_proj(centers)
        if self.p_gen is not None:
            tokens = tokens + self.p_gen
        return self._kv(tokens)

    def encode_specific(self, expert_idx: torch.Tensor, centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Keys and values ``[P, C, heads, d]`` of ``P_e^T mu_e`` for expert rows ``expert_idx [P]``."""
        de = str(int(centers.size(-1)))
        tokens = torch.bmm(centers, self.spec_weight[de][expert_idx]) + self.spec_bias[de][expert_idx][:, None]
        if self.p_spec is not None:
            tokens = tokens + self.p_spec
        return self._kv(tokens)

    def _attend(self, theta: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Position-0 output of the block for CLS tokens ``theta [B, M, d]`` over task tokens ``[B, L, heads, d]``."""
        b, m, _ = theta.shape
        h, d = self.num_heads, self.token_dim
        q = self.w_q(theta).view(b, m, h, d)
        k0, v0 = self._kv(theta)
        logit_self = (q * k0).sum(-1, keepdim=True)
        logit_tok = torch.einsum("bmhd,blhd->bmhl", q, keys).masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(torch.cat([logit_self, logit_tok], dim=-1) / self.temperature, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn[..., :1] * v0 + torch.einsum("bmhl,blhd->bmhd", attn[..., 1:], values)
        hidden = self.layer_norm(self.dropout(self.fc(out.reshape(b, m, h * d))) + theta)
        return self.head(hidden).squeeze(-1)

    def score(
        self,
        general: torch.Tensor,
        general_mask: torch.Tensor,
        expert_idx: torch.Tensor,
        expert_mask: torch.Tensor,
        specific: Optional[Mapping[Tuple[int, int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Scores ``[B, M]`` (masked experts: -inf).

        ``general [B, C, d_z]`` descriptor centres (``general_mask [B, C]``);
        ``expert_idx [B, M]`` rows of ``theta`` (``expert_mask`` marks real
        entries); ``specific``: ``{(task b, column j): centres [C, d_e]}`` for
        the pairs that also receive their expert-specific tokens.
        """
        kg, vg = self.encode_general(general, general_mask)
        scores = self._attend(self.theta[expert_idx], kg, vg, general_mask)
        by_dim = {}
        for (bi, j), centers in (specific or {}).items():
            by_dim.setdefault(int(centers.size(-1)), []).append((int(bi), int(j), centers))
        for items in by_dim.values():
            bi = torch.tensor([it[0] for it in items], device=scores.device)
            j = torch.tensor([it[1] for it in items], device=scores.device)
            e = expert_idx[bi, j]
            spec, spec_mask = pad_token_sets([it[2].to(general) for it in items])
            ks, vs = self.encode_specific(e, spec)
            if self.mode == "append":
                ks, vs = torch.cat([kg[bi], ks], dim=1), torch.cat([vg[bi], vs], dim=1)
                spec_mask = torch.cat([general_mask[bi], spec_mask], dim=1)
            pair_scores = self._attend(self.theta[e][:, None], ks, vs, spec_mask)[:, 0]
            scores = scores.index_put((bi, j), pair_scores)
        return scores.masked_fill(~expert_mask, float("-inf"))


__all__ = ["ModelSpiderRanker", "SPECIFIC_TOKEN_MODES"]
