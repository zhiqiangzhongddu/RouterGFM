"""GraphPrompt+ per-backbone spec resolution.

Single source of truth for which GNN backbones GraphPrompt+ supports,
the support tier (``official``/``extension``), and a short formula
string used in error messages and the coverage table.

The dim list and per-stage layout are derived inside each adapter via
``iter_stage_specs`` because the available stages depend on
``num_layers``, not just ``model.name``.  ``GraphPromptPlusSpec`` here
captures only the cfg-time, model-name-only facts — the same role
``EdgePromptSpec`` plays for EdgePrompt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphPromptPlusSpec:
    """Cfg-time GraphPrompt+ description for a backbone."""

    model_name: str
    support: str  # "official" | "extension"
    formula: str
    supports_layer_concat: bool


_SUPPORT: dict[str, str] = {
    # Phase 1: stack-style backbones whose forward is a uniform
    # conv/act/bn/dropout pipeline.  These match the original
    # GraphPromptPlusStageWise injection points exactly.
    "gcn": "official",
    "gin": "official",
    "gat": "extension",
    "mlp": "extension",
    # Phase 2: TransformerConv stack — same {input, between-layers, final}
    # injection points but no batch norm and no per-layer cache.
    "transformer": "extension",
    # Phase 3: NodeFormer — input-projection + attention stack +
    # output-projection.  Stage 1/2 inject between conv layers in
    # hidden space (post-residual + post-bn + post-dropout).
    "nodeformer": "extension",
    # Phase 4: FAGCN — between-layer prompts modify only the current
    # ``x``; the residual reference ``x0`` stays unprompted.
    "fagcn": "extension",
    # Phase 5: H2GCN — fixed 2-hop architecture, only stages {0, 1, 3}
    # are meaningful (no third hop -> stage 2 has no injection point).
    "h2gcn": "extension",
}


_FORMULA: dict[str, str] = {
    "gcn": "elementwise prompt at stages {input, between-layers, final}",
    "gin": "elementwise prompt at stages {input, between-layers, final}",
    "gat": "elementwise prompt at stages {input, between-layers, final}",
    "mlp": "elementwise prompt at stages {input, between-layers, final}",
    "transformer": "elementwise prompt at stages {input, between-layers, final}; no BN; no layer_concat",
    "nodeformer": "stage prompts on input/between conv layers/final; respects use_residual / use_bn / use_jk",
    "fagcn": "stage prompts on x only; FAConv residual reference x0 unprompted",
    "h2gcn": "stages {0, 1, 3} only; stage-1 prompt on x1 flows to both 2-hop agg and final concat",
}


_SUPPORTS_LAYER_CONCAT: dict[str, bool] = {
    "gcn": True,
    "gin": True,
    "gat": True,
    "mlp": True,
    "transformer": False,
    "nodeformer": False,
    "fagcn": False,
    "h2gcn": False,
}


def supported_graphprompt_plus_backbones() -> tuple[str, ...]:
    """Sorted tuple of backbones with a registered GraphPrompt+ spec."""
    return tuple(sorted(_SUPPORT.keys()))


def resolve_graphprompt_plus_spec(cfg) -> GraphPromptPlusSpec:
    """Return the GraphPrompt+ spec for the cfg's model.

    Raises ``ValueError`` for any model whose GraphPrompt+ support has
    not been implemented yet (transformer/nodeformer/h2gcn/fagcn
    will land in later phases).
    """
    name = str(getattr(getattr(cfg, "model", None), "name", "") or "").lower()
    if name not in _SUPPORT:
        supported = supported_graphprompt_plus_backbones()
        raise ValueError(
            f"GraphPrompt+ does not support model '{name}'. "
            f"Supported backbones: {list(supported)}; use "
            f"graphprompt.plus=False for unsupported backbones."
        )
    return GraphPromptPlusSpec(
        model_name=name,
        support=_SUPPORT[name],
        formula=_FORMULA[name],
        supports_layer_concat=_SUPPORTS_LAYER_CONCAT[name],
    )
