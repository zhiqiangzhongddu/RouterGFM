"""Naming helpers for split and run identifiers."""

from __future__ import annotations

import hashlib
import os
import re
from numbers import Integral
from typing import Optional

from .config_helpers import cfg_default


DEFAULT_ARTIFACT_STEM_MAX_BYTES = 180


def compact_artifact_stem(
    stem: str,
    *,
    max_bytes: int = DEFAULT_ARTIFACT_STEM_MAX_BYTES,
) -> str:
    """Bound an artifact stem while preserving deterministic run identity.

    Linux filesystems commonly limit one path component to 255 bytes. Training
    artifacts append extensions and atomic-save suffixes, so long logical run
    names need a smaller fixed budget. Short stems remain byte-for-byte
    unchanged. Long stems retain readable head/tail fragments separated by a
    SHA-256 digest; the full logical name remains in artifact metadata.
    """
    text = str(stem)
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    if max_bytes < 48:
        raise ValueError("artifact stem byte budget must be at least 48")

    digest = hashlib.sha256(raw).hexdigest()[:16]
    marker = f"__h{digest}__"
    remaining = max_bytes - len(marker.encode("ascii"))
    head_budget = (remaining * 2) // 3
    tail_budget = remaining - head_budget
    head = raw[:head_budget].decode("utf-8", errors="ignore").rstrip("._-")
    tail = raw[-tail_budget:].decode("utf-8", errors="ignore").lstrip("._-")
    compact = f"{head}{marker}{tail}"
    if len(compact.encode("utf-8")) > max_bytes:
        raise AssertionError("compacted artifact stem exceeds its byte budget")
    return compact


def format_split_for_name(split) -> str:
    """
    Format a split tuple for inclusion in filenames, e.g., (0.8,0.1,0.1) -> "split80-10-10".
    Returns an empty string when split is None or not iterable.
    """
    try:
        parts = list(split)
    except TypeError:
        return ""

    if not parts:
        return ""

    first = parts[0]
    # Same few-shot detection rule as dataset_helpers.is_few_shot_split:
    # an integral first element (including float-integers like 5.0) means
    # (shots, val_ratio, test_ratio). Diverging here would give the same
    # logical split two different run names.
    is_shots = isinstance(first, Integral) and not isinstance(first, bool)
    if not is_shots and isinstance(first, float) and first.is_integer():
        is_shots = True
        first = int(first)
    if is_shots:
        val_ratio = parts[1] if len(parts) > 1 else 0.0
        test_ratio = parts[2] if len(parts) > 2 else 0.0
        try:
            val_pct = int(round(float(val_ratio) * 100))
            test_pct = int(round(float(test_ratio) * 100))
        except (TypeError, ValueError):
            val_pct, test_pct = 0, 0
        return f"fewshot{first}-{val_pct}-{test_pct}"

    try:
        numeric = [float(part) for part in parts]
    except ValueError:
        return ""
    suffix = "-".join(str(int(round(part * 100))) for part in numeric)
    return f"split{suffix}"


def _resolve_variant_tag(cfg) -> str:
    """Look up a method-variant tag for ``cfg`` via the pretrain registry.

    The registry import is lazy to keep this module free of
    pretrain-package dependencies at import time. Errors raised by
    ``variant_tag_for`` (e.g. a broken ``variant_tag`` classmethod) are
    **not** suppressed -- they propagate so run-name construction fails
    loudly instead of silently collapsing distinct variant runs.
    """
    try:
        from src.pretrain.registry import variant_tag_for
    except Exception:
        return ""
    return variant_tag_for(cfg) or ""


def _resolve_task_class(cfg):
    """Return the registered ``PretrainTask`` class for ``cfg.pretrain.method``.

    Returns ``None`` when the registry import fails or the method is not
    registered. Used to consult task-level capability flags without
    forcing an eager dependency on the pretrain package.
    """
    try:
        from src.pretrain.registry import get_pretrain_task_class
    except Exception:
        return None
    method = getattr(getattr(cfg, "pretrain", None), "method", "") or ""
    try:
        return get_pretrain_task_class(method)
    except Exception:
        return None


def _float_or(value, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _int_or(value, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _warm_start_tag(path: object) -> str:
    """Return a compact identity tag for an explicit warm-start path."""
    raw = str(path or "").strip()
    if not raw:
        return ""
    normalized = os.path.abspath(os.path.expanduser(raw))
    stem = os.path.splitext(os.path.basename(normalized))[0]
    stem = re.sub(r"[^A-Za-z0-9.-]+", "-", stem).strip("-.") or "checkpoint"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
    return f"ws{stem[:24]}-{digest}"


def model_variant_tag_for(cfg) -> str:
    """Encode non-default model-level knobs that affect weights or semantics.

    Covers:
    - Top-level: ``use_batchnorm`` (state_dict shape), ``graph_pooling``,
      ``activation`` (``prelu`` adds parameters), ``dropout``.
    - Backbone sub-blocks: GAT (``heads``), NodeFormer (``heads``,
      ``num_random_features``, ``tau``, layernorm/gumbel/jk toggles,
      ``rb_order``), FAGCN (``eps``, ``use_batchnorm``), H2GCN
      (``use_batchnorm``).

    Emit a tag component only when the field differs from its default
    in ``src/config/``, so runs that use all-default model settings
    keep their pre-variant-tag filenames unchanged. Baselines are read
    from ``cfg_default`` rather than hardcoded: a hardcoded literal
    silently drifts when a config default changes (this happened with
    ``nodeformer.rb_order`` 0 -> 1, which orphaned every all-default
    NodeFormer checkpoint from ``skip_if_exists`` lookup).

    This is intentionally separate from ``variant_tag_for`` (which
    covers method-specific knobs) so the two kinds of identity can be
    composed independently by the run-name builder.
    """
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None:
        return ""

    def _default(dotpath: str, fallback):
        try:
            return cfg_default(f"model.{dotpath}")
        except AttributeError:
            return fallback

    def _bool_tag(parts_list: list[str], value, default, on_tag: str, off_tag: str) -> None:
        if bool(value) != bool(default):
            parts_list.append(on_tag if bool(value) else off_tag)

    parts: list[str] = []
    # ---- top-level ---- #
    # use_batchnorm changes state_dict shape -- checkpoint reuse across
    # this setting must be disabled at the filename level, because
    # load_state_dict(strict=False) would silently drop BN params.
    _bool_tag(
        parts,
        getattr(model_cfg, "use_batchnorm", False),
        _default("use_batchnorm", False),
        "bn",
        "nobn",
    )
    default_pool = str(_default("graph_pooling", "mean")).lower()
    default_pool = "add" if default_pool == "sum" else default_pool
    pool = str(getattr(model_cfg, "graph_pooling", default_pool)).lower()
    pool = "add" if pool == "sum" else pool
    if pool != default_pool:
        parts.append(f"pool{pool}")
    default_act = str(_default("activation", "relu")).lower()
    act = str(getattr(model_cfg, "activation", default_act)).lower()
    if act != default_act:
        parts.append(act)
    default_dropout = _float_or(_default("dropout", 0.5), 0.5)
    dropout = _float_or(getattr(model_cfg, "dropout", default_dropout), default_dropout)
    if abs(dropout - default_dropout) > 1e-9:
        parts.append(f"d{dropout:g}")

    # ---- backbone-specific sub-blocks ---- #
    backbone = str(getattr(model_cfg, "name", "")).lower()
    if backbone == "gat":
        gat_cfg = getattr(model_cfg, "gat", None)
        if gat_cfg is not None:
            default_heads = _int_or(_default("gat.heads", 8), 8)
            heads = _int_or(getattr(gat_cfg, "heads", default_heads), default_heads)
            if heads != default_heads:
                parts.append(f"heads{heads}")
    elif backbone == "nodeformer":
        nf_cfg = getattr(model_cfg, "nodeformer", None)
        if nf_cfg is not None:
            default_heads = _int_or(_default("nodeformer.heads", 4), 4)
            heads = _int_or(getattr(nf_cfg, "heads", default_heads), default_heads)
            if heads != default_heads:
                parts.append(f"heads{heads}")
            default_nrf = _int_or(_default("nodeformer.num_random_features", 30), 30)
            nrf = _int_or(getattr(nf_cfg, "num_random_features", default_nrf), default_nrf)
            if nrf != default_nrf:
                parts.append(f"nrf{nrf}")
            default_tau = _float_or(_default("nodeformer.tau", 1.0), 1.0)
            tau = _float_or(getattr(nf_cfg, "tau", default_tau), default_tau)
            if abs(tau - default_tau) > 1e-9:
                parts.append(f"tau{tau:g}")
            _bool_tag(
                parts,
                getattr(nf_cfg, "use_layernorm", True),
                _default("nodeformer.use_layernorm", True),
                "ln",
                "noln",
            )
            _bool_tag(
                parts,
                getattr(nf_cfg, "use_gumbel", True),
                _default("nodeformer.use_gumbel", True),
                "gum",
                "nogum",
            )
            _bool_tag(
                parts,
                getattr(nf_cfg, "use_jk", False),
                _default("nodeformer.use_jk", False),
                "jk",
                "nojk",
            )
            default_rb = _int_or(_default("nodeformer.rb_order", 1), 1)
            rb_order = _int_or(getattr(nf_cfg, "rb_order", default_rb), default_rb)
            if rb_order != default_rb:
                parts.append(f"rb{rb_order}")
            default_rb_trans = str(_default("nodeformer.rb_trans", "sigmoid")).lower()
            rb_trans = str(getattr(nf_cfg, "rb_trans", default_rb_trans)).lower()
            if rb_trans != default_rb_trans:
                parts.append(f"rbx{rb_trans}")
            _bool_tag(
                parts,
                getattr(nf_cfg, "use_residual", True),
                _default("nodeformer.use_residual", True),
                "res",
                "nores",
            )
            _bool_tag(
                parts,
                getattr(nf_cfg, "use_activation", True),
                _default("nodeformer.use_activation", True),
                "nfact",
                "nonfact",
            )
            _bool_tag(
                parts,
                getattr(nf_cfg, "use_edge_loss", False),
                _default("nodeformer.use_edge_loss", False),
                "edgeloss",
                "noedgeloss",
            )
            default_gumbel_samples = _int_or(
                _default("nodeformer.num_gumbel_samples", 10), 10
            )
            gumbel_samples = _int_or(
                getattr(nf_cfg, "num_gumbel_samples", default_gumbel_samples),
                default_gumbel_samples,
            )
            if gumbel_samples != default_gumbel_samples:
                parts.append(f"gs{gumbel_samples}")
    elif backbone == "fagcn":
        fagcn_cfg = getattr(model_cfg, "fagcn", None)
        if fagcn_cfg is not None:
            default_eps = _float_or(_default("fagcn.eps", 0.1), 0.1)
            eps = _float_or(getattr(fagcn_cfg, "eps", default_eps), default_eps)
            if abs(eps - default_eps) > 1e-9:
                parts.append(f"eps{eps:g}")
            _bool_tag(
                parts,
                getattr(fagcn_cfg, "use_batchnorm", False),
                _default("fagcn.use_batchnorm", False),
                "fagcnbn",
                "fagcnnobn",
            )
    elif backbone == "h2gcn":
        h2_cfg = getattr(model_cfg, "h2gcn", None)
        if h2_cfg is not None:
            _bool_tag(
                parts,
                getattr(h2_cfg, "use_batchnorm", False),
                _default("h2gcn.use_batchnorm", False),
                "h2gcnbn",
                "h2gcnnobn",
            )

    return "-".join(parts)


def build_pretrain_run_name_from_cfg(
    cfg,
    include_split: Optional[bool] = None,
    variant_tag: Optional[str] = None,
) -> str:
    """
    Canonical pretrain run-name convention. Single source of truth --
    ``PretrainRunner`` and any downstream checkpoint-lookup code must
    route through this function so the two never drift.

    Args:
        include_split:
            - None: include split tag when the registered PretrainTask
              class declares ``uses_dataset_splits = True`` (today:
              supervised only). Falls back to omitting the split tag
              when the method is not registered.
            - True: always include split tag when available.
            - False: never include split tag.
        variant_tag:
            - None (default): resolve automatically by looking up the
              registered ``PretrainTask`` class for ``cfg.pretrain.method``
              and calling its ``variant_tag(cfg)`` classmethod.
            - str: use the provided tag verbatim (empty string disables).
    """
    dataset_cfg = getattr(getattr(cfg, "pretrain", None), "dataset", None) or getattr(cfg, "dataset", None)
    model_cfg = getattr(cfg, "model", None)
    pretrain_cfg = getattr(cfg, "pretrain", None)
    split = getattr(dataset_cfg, "fixed_split", None) if dataset_cfg else None

    dataset_name = getattr(dataset_cfg, "name", "dataset") if dataset_cfg else "dataset"
    induced_flag = int(getattr(dataset_cfg, "induced", False)) if dataset_cfg else 0
    task_level = getattr(dataset_cfg, "task_level", "") if dataset_cfg else ""
    model_name = getattr(model_cfg, "name", "model") if model_cfg else "model"
    hidden_dim = getattr(model_cfg, "hidden_dim", "")
    out_dim = getattr(model_cfg, "out_dim", "")
    num_layers = getattr(model_cfg, "num_layers", "")
    epochs = getattr(pretrain_cfg, "epochs", "")
    lr = getattr(pretrain_cfg, "lr", "")
    batch_size = getattr(pretrain_cfg, "batch_size", "")
    seed = getattr(cfg, "seed", "")
    method = getattr(pretrain_cfg, "method", "method") if pretrain_cfg else "method"
    if include_split is None:
        cls = _resolve_task_class(cfg)
        include_split_effective = bool(cls and getattr(cls, "uses_dataset_splits", False))
    else:
        include_split_effective = bool(include_split)
    split_tag = format_split_for_name(split) if include_split_effective else ""

    if variant_tag is None:
        variant_tag = _resolve_variant_tag(cfg)
    method_part = f"{method}-{variant_tag}" if variant_tag else method
    model_variant_tag = model_variant_tag_for(cfg)
    model_part = f"{model_name}-{model_variant_tag}" if (model_name and model_variant_tag) else model_name

    parts = [
        method_part,
        _warm_start_tag(getattr(pretrain_cfg, "input_checkpoint", "")),
        dataset_name,
        f"task{task_level}",
        f"induced{induced_flag}",
        split_tag,
        model_part,
        f"h{hidden_dim}",
        f"o{out_dim}",
        f"l{num_layers}",
        f"e{epochs}",
        f"lr{lr:g}" if isinstance(lr, (int, float)) else f"lr{lr}" if lr != "" else "",
        f"bs{batch_size}",
        f"seed{seed}",
    ]
    return "_".join(str(part) for part in parts if part not in ("", None))


def build_train_run_name_from_cfg(cfg, *, split, task_level_raw: str, task_cls) -> str:
    """Canonical train run-name builder.

    Extracted from ``TrainRunner._build_run_name`` so that checkpoint
    lookup and the runner never drift on the run-name convention.
    Produces a byte-for-byte identical string to the previous inline
    builder for all currently supported train configs.
    """
    dataset_cfg = cfg.train.dataset
    train_cfg = cfg.train
    model_cfg = cfg.model
    split_tag = format_split_for_name(split)
    model_name = getattr(model_cfg, "name", "")
    model_variant_tag = model_variant_tag_for(cfg)
    model_part = f"{model_name}-{model_variant_tag}" if (model_name and model_variant_tag) else model_name
    method = getattr(train_cfg, "method", "supervised") or "supervised"
    variant_tag = task_cls.variant_tag(cfg) if task_cls is not None else ""
    method_part = f"{method}-{variant_tag}" if variant_tag else method
    parts = [
        "train",
        method_part,
        dataset_cfg.name,
        f"induced{int(getattr(dataset_cfg, 'induced', False))}",
        split_tag,
        f"task{task_level_raw}",
        model_part,
        f"h{model_cfg.hidden_dim}",
        f"o{model_cfg.out_dim}",
        f"l{model_cfg.num_layers}",
        f"e{train_cfg.epochs}",
        f"lr{train_cfg.lr:g}" if isinstance(train_cfg.lr, (int, float)) else f"lr{train_cfg.lr}",
        f"bs{train_cfg.batch_size}",
        f"seed{cfg.seed}",
    ]
    return "_".join(str(p) for p in parts if p not in ("", None))


def build_finetune_run_name_from_cfg(
    cfg,
    *,
    split,
    task_level_raw: str,
    task_cls,
    finetune_method: str,
    pretrained_run_name: str,
    freeze_pretrained_effective: bool,
) -> str:
    """Canonical finetune run-name builder.

    Extracted from ``FinetuneRunner._build_run_name`` so that checkpoint
    lookup, ``skip_if_exists``, and the results TSV key share a single
    source of truth.  Produces a byte-for-byte identical string to the
    previous inline builder for all currently supported finetune configs.
    """
    dataset_cfg = cfg.finetune.dataset
    finetune_cfg = cfg.finetune
    split_tag = format_split_for_name(split)
    freeze_flag = int(bool(freeze_pretrained_effective))
    method_run_tag = task_cls.run_tag(cfg) if task_cls is not None else ""
    method_variant_tag = task_cls.variant_tag(cfg) if task_cls is not None else ""
    variant_parts = [method_variant_tag] if method_variant_tag else []
    task_type = str(getattr(dataset_cfg, "task_type", "") or "").strip().lower()
    if task_type == "regression":
        # Regression normalization changes the scale optimized by the head.
        # Tag both states so newly fair runs cannot silently reuse historical
        # baseline checkpoints that predate the shared policy.
        from src.finetune.regression import (
            NORMALIZED_MSE,
            resolve_regression_loss,
            resolve_regression_target_normalization,
        )

        normalized = int(resolve_regression_target_normalization(cfg))
        variant_parts.append(f"regnorm{normalized}")
        regression_loss = resolve_regression_loss(cfg)
        if regression_loss != NORMALIZED_MSE:
            variant_parts.append("reglossmetricmae")
    from src.finetune.multilabel import (
        MACRO_BALANCED_BCE,
        resolve_multilabel_loss,
    )

    if resolve_multilabel_loss(cfg) == MACRO_BALANCED_BCE:
        variant_parts.append("mlbce")
    parts = [
        "ft",
        finetune_method,
        method_run_tag,
        pretrained_run_name,
        f"to_{dataset_cfg.name}",
        f"induced{int(getattr(dataset_cfg, 'induced', False))}",
        split_tag,
        f"task{task_level_raw}",
        f"frz{freeze_flag}",
        *variant_parts,
        f"e{finetune_cfg.epochs}",
        f"lr{finetune_cfg.lr:g}" if isinstance(finetune_cfg.lr, (int, float)) else f"lr{finetune_cfg.lr}",
        f"bs{finetune_cfg.batch_size}",
        f"seed{cfg.seed}",
    ]
    return "_".join(str(p) for p in parts if p not in ("", None))


__all__ = [
    "build_finetune_run_name_from_cfg",
    "build_pretrain_run_name_from_cfg",
    "build_train_run_name_from_cfg",
    "compact_artifact_stem",
    "format_split_for_name",
    "model_variant_tag_for",
]
