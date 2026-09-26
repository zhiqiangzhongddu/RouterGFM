"""Generic config validation and variant-tag helpers.

These helpers are workflow-agnostic and can be used by pretrain, train,
and finetune method modules alike.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Frozen default config — single source of truth for variant-tag defaults.
#
# ``src.config.cfg`` is populated with all defaults at import time.
# We clone and freeze it once so method modules never need their own
# ``_DEFAULT_*`` constants that must "stay in sync" with config.py.
# ---------------------------------------------------------------------------
from src.config import cfg as _global_cfg

_DEFAULTS = _global_cfg.clone()
_DEFAULTS.freeze()


def validate_probability(name: str, value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    """Validate that *value* is in the range [*low*, *high*] and return it as float."""
    v = float(value)
    if v < low or v > high:
        raise ValueError(f"{name} must be in [{low}, {high}]; got {v}.")
    return v


def validate_choice(name: str, value: str, choices: frozenset[str] | set[str]) -> str:
    """Validate that *value* is one of *choices* and return the lowered string."""
    v = str(value).lower()
    if v not in choices:
        raise ValueError(f"{name}='{value}' is invalid; expected one of {sorted(choices)}.")
    return v


def cfg_default(dotpath: str):
    """Look up a default value from the frozen config by dotted path.

    Example::

        cfg_default("pretrain.graphcl.temperature")  # -> 0.1
    """
    node = _DEFAULTS
    for part in dotpath.split("."):
        node = getattr(node, part)
    return node


def tag_if_nondefault(tag_name: str, value, default, *, fmt: str = "g") -> str:
    """Return ``'{tag_name}{value}'`` when *value* differs from *default*, else ``""``."""
    if isinstance(value, float) and isinstance(default, float):
        if abs(value - default) <= 1e-9:
            return ""
        return f"{tag_name}{value:{fmt}}"
    if value == default:
        return ""
    if isinstance(value, bool):
        return tag_name
    return f"{tag_name}{value}"


def resolve_workflow_dataset_cfg(cfg, workflow: str):
    """Return the dataset config sub-node for *workflow*.

    Looks up ``cfg.{workflow}.dataset`` with a fallback to ``cfg.dataset``
    when the workflow-specific block is missing.  Standardises the
    scattered ``getattr(getattr(cfg, ...), ...)`` chains found in
    supervised method ``__init__`` blocks.
    """
    wf = getattr(cfg, workflow, None)
    ds = getattr(wf, "dataset", None) if wf is not None else None
    return ds if ds is not None else getattr(cfg, "dataset", None)


def resolve_method_optim_field(method_cfg, field: str, fallback):
    """Resolve an optional optimizer field from a method config block.

    Returns *fallback* when the field is absent, ``None``, or the
    string ``"none"``/``"null"``.  Otherwise coerces to ``float``.
    Shared across prompt methods to replace scattered None/none handling.
    """
    raw = getattr(method_cfg, field, None) if method_cfg is not None else None
    if raw is None:
        return fallback
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in {"none", "null", ""}:
            return fallback
    try:
        return float(raw)
    except (TypeError, ValueError):
        return fallback


_OPTIMIZER_TAG_FIELDS = (
    ("lr", "mlr"),
    ("weight_decay", "mwd"),
    ("prompt_lr", "plr"),
    ("prompt_weight_decay", "pwd"),
    ("head_lr", "hlr"),
    ("head_weight_decay", "hwd"),
    ("head_lr_scale", "hlrs"),
)


def optimizer_variant_tags(
    method_cfg,
    method_key: str,
    exclude: frozenset[str] | set[str] = frozenset(),
) -> list[str]:
    """Variant tags for the optimizer fields read by ``build_prompt_head_optimizer``.

    The shared finetune run name embeds ``finetune.lr``, which prompt methods
    do not read; without these tags two runs differing only in a method-local
    optimizer setting (e.g. ``finetune.gppt.lr``) collide to one run name and
    ``skip_if_exists`` silently reuses the wrong checkpoint. Tags appear only
    for non-default values, so all-default run names are unchanged.
    """
    tags: list[str] = []
    if method_cfg is None:
        return tags

    def _norm(value):
        # Mirror resolve_method_optim_field: None / "none" / "null" = unset.
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    for field, tag_name in _OPTIMIZER_TAG_FIELDS:
        if field in exclude:
            continue
        try:
            default_raw = cfg_default(f"finetune.{method_key}.{field}")
        except AttributeError:
            continue
        value = _norm(getattr(method_cfg, field, None))
        default = _norm(default_raw)
        if value is None:
            # Unset (or explicitly reset) falls back to the same chain the
            # optimizer builder uses; treat as default.
            continue
        if default is not None and abs(value - default) <= 1e-12:
            continue
        tags.append(f"{tag_name}{value:g}")
    return tags


def build_prompt_head_optimizer(
    *,
    method_cfg,
    prompt_params,
    head_params=None,
    base_lr: float,
    base_wd: float,
    head_lr_scale: float | None = None,
    optimizer_cls=None,
):
    """Build a single ``{"primary": optimizer}`` for prompt + optional head.

    Reads ``lr`` / ``weight_decay`` / ``prompt_lr`` / ``prompt_weight_decay`` /
    ``head_lr`` / ``head_weight_decay`` / ``head_lr_scale`` from *method_cfg*
    via :func:`resolve_method_optim_field` with the same fallback rules every
    prompt method previously open-coded.  ``head_params=None`` or an empty
    iterable yields a single parameter group (prompt only).

    The helper centralises GPF / GraphPrompt / GPPT / EdgePrompt's divergent
    optimizer construction and makes the choice of ``optimizer_cls``
    explicit — methods that need AdamW (GraphPrompt) pass it in rather than
    silently deviating from Adam.
    """
    import torch  # local import: utils package must stay torch-light for config-only consumers

    if optimizer_cls is None:
        optimizer_cls = torch.optim.Adam

    method_lr = resolve_method_optim_field(method_cfg, "lr", base_lr)
    method_wd = resolve_method_optim_field(method_cfg, "weight_decay", base_wd)
    prompt_lr = resolve_method_optim_field(method_cfg, "prompt_lr", method_lr)
    prompt_wd = resolve_method_optim_field(method_cfg, "prompt_weight_decay", method_wd)

    effective_head_lr_scale = (
        resolve_method_optim_field(method_cfg, "head_lr_scale", 1.0)
        if head_lr_scale is None
        else float(head_lr_scale)
    )
    head_lr = resolve_method_optim_field(method_cfg, "head_lr", method_lr * effective_head_lr_scale)
    head_wd = resolve_method_optim_field(method_cfg, "head_weight_decay", method_wd)

    prompt_params = list(prompt_params)
    param_groups = [
        {"params": prompt_params, "lr": prompt_lr, "weight_decay": prompt_wd},
    ]
    if head_params is not None:
        head_params = list(head_params)
        if head_params:
            param_groups.append(
                {"params": head_params, "lr": head_lr, "weight_decay": head_wd}
            )

    optimizer = optimizer_cls(param_groups)
    return {"primary": optimizer}


__all__ = [
    "build_prompt_head_optimizer",
    "cfg_default",
    "resolve_method_optim_field",
    "resolve_workflow_dataset_cfg",
    "tag_if_nondefault",
    "validate_choice",
    "validate_probability",
]
