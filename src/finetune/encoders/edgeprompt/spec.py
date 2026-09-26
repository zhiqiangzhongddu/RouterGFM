"""EdgePrompt per-backbone spec resolution.

The prompt dim schedule and self-loop policy must agree between the
encoder factory (builds the encoder) and ``FinetuneEdgePrompt`` (builds
the prompt module).  ``resolve_edgeprompt_prompt_spec(cfg)`` is the
single pure function that both call, so the two constructions cannot
drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class EdgePromptSpec:
    """Immutable per-cfg description of how EdgePrompt attaches to a backbone."""

    model_name: str
    dim_list: tuple[int, ...]
    add_self_loops: bool
    replace_self_loops: bool
    support: str  # "official" | "extension"
    formula: str


def _dims_concat_in_then_hidden(c) -> tuple[int, ...]:
    return tuple(
        [int(c.model.in_dim)]
        + [int(c.model.hidden_dim)] * (int(c.model.num_layers) - 1)
    )


def _dims_hidden_per_layer(c) -> tuple[int, ...]:
    return tuple([int(c.model.hidden_dim)] * int(c.model.num_layers))


def _dims_h2gcn(c) -> tuple[int, ...]:
    # Fixed 2-hop structure: first aggregation reads raw features,
    # second reads hidden.
    return (int(c.model.in_dim), int(c.model.hidden_dim))


_DIM_SCHEDULE: dict[str, Callable] = {
    "gcn":         _dims_concat_in_then_hidden,
    "gin":         _dims_concat_in_then_hidden,
    "gat":         _dims_concat_in_then_hidden,
    "transformer": _dims_concat_in_then_hidden,
    "fagcn":       _dims_hidden_per_layer,
    "nodeformer":  _dims_hidden_per_layer,
    "h2gcn":       _dims_h2gcn,
}


_SUPPORT: dict[str, str] = {
    "gcn": "official",
    "gin": "official",
    "gat": "extension",
    "transformer": "extension",
    "h2gcn": "extension",
    "fagcn": "extension",
    "nodeformer": "extension",
}


_FORMULA: dict[str, str] = {
    "gcn":         "normalized (h_j + p_ji) message",
    "gin":         "summed (h_j + p_ji) message, then MLP",
    "gat":         "attention from unprompted h; value = W(h_j + p_ji)",
    "transformer": "value = W_V(h_j + p_ji); query/key unchanged",
    "h2gcn":       "prompted per-hop aggregation; lin/act/dropout/concat preserved",
    "fagcn":       "adaptive gate from unprompted features; message on (h_j + p_ji)",
    "nodeformer":  "prompted relational-bias path (requires rb_order >= 1)",
}


# Auto defaults for ``add_self_loops``.  The prompt module and the
# prompt-aware conv must agree: if the encoder adds self-loops to the
# edge index before aggregation, the prompt must produce E+N rows;
# otherwise only E rows.  A mismatch surfaces as a shape error at
# runtime, so these defaults are part of the math contract, not a
# style choice.
_AUTO_ADD_SELF_LOOPS: dict[str, bool] = {
    "gcn":   True,   # PromptGCNConv adds self-loops in forward
    "gin":   False,
    "gat":   True,   # PyG GATConv adds self-loops by default; PromptGATConv mirrors
    "transformer": False,
    "h2gcn": True,   # repo H2GCN always uses add_self_loops in its aggregation
    "fagcn": True,   # PromptFAConv uses gcn_norm(add_self_loops=True)
    "nodeformer": False,
}

# Most supported operators remove existing loops before adding one per node.
# The repository's vanilla H2GCN appends loops without removing existing ones.
_REPLACE_SELF_LOOPS: dict[str, bool] = {
    name: name != "h2gcn" for name in _DIM_SCHEDULE
}


def supported_edgeprompt_backbones() -> tuple[str, ...]:
    return tuple(sorted(_DIM_SCHEDULE.keys()))


def resolve_edgeprompt_prompt_spec(cfg) -> EdgePromptSpec:
    """Return the per-backbone prompt spec for the given cfg.

    Raises ``ValueError`` for any model whose EdgePrompt support has
    not been implemented (notably ``mlp``, which has no edge-message
    path).
    """
    name = str(getattr(getattr(cfg, "model", None), "name", "") or "").lower()
    if name not in _DIM_SCHEDULE:
        supported = supported_edgeprompt_backbones()
        raise ValueError(
            f"EdgePrompt does not support model '{name}'. "
            f"Supported GNN backbones: {list(supported)}."
        )
    return EdgePromptSpec(
        model_name=name,
        dim_list=_DIM_SCHEDULE[name](cfg),
        add_self_loops=_AUTO_ADD_SELF_LOOPS[name],
        replace_self_loops=_REPLACE_SELF_LOOPS[name],
        support=_SUPPORT[name],
        formula=_FORMULA[name],
    )
