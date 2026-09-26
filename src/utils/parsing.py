"""Shared parsing utilities used by pretrain, train, finetune, and other modules."""

from __future__ import annotations

import ast
import os
from typing import Any, Sequence, Tuple, Union

from src.utils.save_results import get_explicit_cfg_keys


def looks_bool(value: str) -> bool:
    """Return True when *value* looks like a boolean literal."""
    return str(value).strip().lower() in (
        "true", "false", "1", "0", "yes", "no", "y", "n",
    )


def looks_int(value: str) -> bool:
    """Return True when *value* can be parsed as an integer."""
    try:
        int(str(value).strip())
        return True
    except (TypeError, ValueError):
        return False


def looks_split_literal(value: str) -> bool:
    """Return True when *value* looks like a tuple/list literal, e.g. ``(0.8, 0.1, 0.1)``."""
    text = str(value or "").strip()
    return bool(text) and text[0] in "([" and text[-1] in ")]"


def parse_fixed_split(value: str) -> Tuple[Union[int, float], float, float]:
    """Parse a fixed-split literal like ``(0.8, 0.1, 0.1)`` or ``(5, 0.0, 1.0)``."""
    text = str(value or "").strip()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid fixed split literal: {text}") from exc
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
        raise ValueError(f"Fixed split must contain exactly three values: {text}")
    first_raw, val_raw, test_raw = parsed
    first_float = float(first_raw)
    first = int(round(first_float)) if first_float.is_integer() and first_float >= 1.0 else first_float
    return (first, float(val_raw), float(test_raw))


def resolve_workflow_split(
    raw_split,
    default: tuple = (0.8, 0.1, 0.1),
) -> tuple:
    """Resolve a fixed split from a config value.

    Shared by pretrain, train, and finetune runners.  Returns a 3-tuple
    of (train, val, test) ratios or (shots, val_weight, test_weight) for
    few-shot.  Returns *default* when *raw_split* is None.
    """
    if raw_split is None:
        return default
    parts = list(raw_split)
    if len(parts) != 3:
        raise ValueError(f"fixed_split must be a length-3 tuple/list, got: {raw_split}")
    return tuple(parts)


def has_valid_config_file(argv: Sequence[str]) -> bool:
    """Return True when *argv* contains a ``--config`` pointing to an existing file.

    Handles both ``--config path.yaml`` and ``--config=path.yaml`` forms.
    """
    for i, token in enumerate(argv):
        if token == "--config":
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                return os.path.isfile(argv[i + 1])
            return False
        if token.startswith("--config="):
            path = token.split("=", 1)[1]
            return os.path.isfile(path)
    return False


def validate_required_dataset_overrides(
    cfg,
    argv: Sequence[str],
    workflow: str,
    dataset_name_key: str,
    task_level_key: str,
) -> None:
    """Validate that dataset name and task level are explicitly provided.

    Raises ``ValueError`` when neither a ``--config`` file nor explicit CLI
    keys supply the required dataset overrides.  When a valid ``--config``
    file exists, it is trusted as intentional and validation is skipped —
    even if the file happens to set the base-default values.

    Parameters
    ----------
    cfg : CfgNode
        The fully-merged config (after ``update_cfg`` and ``set_explicit_cfg_keys``).
    argv : sequence of str
        The CLI argv used to build *cfg* (needed for ``--config`` detection).
    workflow : str
        Human-readable label for error messages (e.g. ``"Pretrain"``).
    dataset_name_key : str
        Dot-separated config key for the dataset name
        (e.g. ``"pretrain.dataset.name"``).
    task_level_key : str
        Dot-separated config key for the task level
        (e.g. ``"pretrain.dataset.task_level"``).
    """
    if has_valid_config_file(argv):
        return

    explicit = {k.lower() for k in get_explicit_cfg_keys(cfg)}

    if dataset_name_key.lower() not in explicit:
        raise ValueError(
            f"[{workflow}] Missing dataset override ({dataset_name_key}). "
            "Refusing to use the default dataset."
        )

    # Resolve the task level value from the nested config.
    parts = task_level_key.split(".")
    node = cfg
    for part in parts:
        node = getattr(node, part, None)
        if node is None:
            break
    task_level = str(node or "").strip().lower()

    if task_level_key.lower() not in explicit and task_level in ("", "none"):
        raise ValueError(
            f"[{workflow}] Missing task level override ({task_level_key}). "
            "Please specify node, edge, or graph."
        )


def to_bool(value: Any) -> bool:
    """Coerce *value* to a boolean, accepting strings like ``true``/``false``/``yes``/``no``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def resolve_task_type(value: Any = None, *, default: str = "classification") -> str:
    """Normalize a task-type value to a lowercase string, falling back to *default*.

    The sentinel value ``"none"`` (case-insensitive) is treated as unset and
    triggers the *default* fallback, matching the config convention where
    ``task_type = "none"`` means "not yet resolved".
    """
    if value is not None:
        token = str(value).strip().lower()
        if token and token != "none":
            return token
    return str(default).strip().lower()


# Backward-compatible alias — ``to_bool`` is the canonical implementation.
parse_bool = to_bool
