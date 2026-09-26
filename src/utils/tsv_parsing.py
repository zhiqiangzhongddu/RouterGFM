"""Shared TSV parsing utilities for workflow task files.

Provides generic infrastructure for reading header-based TSV experiment
files used by pretrain, train, and finetune modules.  Each module supplies
its own column set, required columns, and optional custom column handler;
this module handles header detection, standard column parsing, and dedup.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .parsing import (
    looks_bool,
    looks_int,
    looks_split_literal,
    parse_fixed_split,
    to_bool,
)

TASK_TYPES = {"classification", "regression", "none"}

# Standard column categories recognised by :func:`parse_standard_column`.
_BOOL_COLUMNS = {"induced", "skip_if_exists", "pretrain_induced"}
_INT_COLUMNS = {"epochs", "batch", "seed"}

# Type alias for a custom column parser callback.
# Signature: (col, val, line_no) -> (parsed_value, ok) or None to fall through.
CustomParser = Callable[[str, str, int], Optional[Tuple[Any, bool]]]


# ---------------------------------------------------------------------------
# Column-level parsing
# ---------------------------------------------------------------------------

def parse_standard_column(
    col: str,
    val: str,
    line_no: int,
    log_prefix: str,
) -> Tuple[Any, bool]:
    """Parse a column value using standard type rules.

    Returns ``(parsed_value, ok)``.  When *ok* is ``False`` the calling
    row parser should skip the entire row.
    """
    if col in _BOOL_COLUMNS:
        if not looks_bool(val):
            print(f"{log_prefix} Skipping malformed task row {line_no}: invalid {col} flag '{val}'")
            return None, False
        return to_bool(val), True

    if col == "task_type":
        val_lower = val.strip().lower()
        if val_lower not in TASK_TYPES:
            print(f"{log_prefix} Skipping malformed task row {line_no}: unknown task_type '{val_lower}'")
            return None, False
        return (val_lower if val_lower != "none" else None), True

    if col == "fixed_split":
        if looks_split_literal(val):
            try:
                return parse_fixed_split(val), True
            except ValueError as exc:
                print(f"{log_prefix} Skipping malformed task row {line_no}: {exc}")
                return None, False
        return None, True

    if col in _INT_COLUMNS:
        return (int(val) if looks_int(val) else None), True

    # Default: string pass-through.
    return val, True


# ---------------------------------------------------------------------------
# Header detection
# ---------------------------------------------------------------------------

def is_header_row(
    parts: List[str],
    known_columns: Set[str],
    min_columns: int,
) -> bool:
    """Return ``True`` if every token is a known column name and count >= *min_columns*."""
    return len(parts) >= min_columns and all(p.lower() in known_columns for p in parts)


# ---------------------------------------------------------------------------
# Row-level parsing
# ---------------------------------------------------------------------------

def parse_row_by_header(
    parts: List[str],
    columns: List[str],
    line_no: int,
    log_prefix: str,
    required_columns: Sequence[str],
    defaults: Dict[str, Any] | None = None,
    custom_parser: CustomParser | None = None,
) -> Optional[Dict[str, Any]]:
    """Parse a single data row against detected header columns.

    Parameters
    ----------
    custom_parser:
        Called **before** the standard parser for every column.  Return
        ``(value, ok)`` to override, or ``None`` to fall through to
        :func:`parse_standard_column`.
    """
    row: Dict[str, Any] = {}
    for idx, col in enumerate(columns):
        if idx >= len(parts):
            break
        val = parts[idx]

        # Try module-specific parser first.
        if custom_parser is not None:
            result = custom_parser(col, val, line_no)
            if result is not None:
                value, ok = result
                if not ok:
                    return None
                row[col] = value
                continue

        # Fall through to standard parsing.
        value, ok = parse_standard_column(col, val, line_no, log_prefix)
        if not ok:
            return None
        row[col] = value

    # Warn on extra trailing tokens that the header doesn't account for.
    if len(parts) > len(columns):
        extra = parts[len(columns):]
        print(
            f"{log_prefix} Ignoring {len(extra)} extra column(s) on row {line_no}: "
            f"{' '.join(extra)}"
        )

    for key in required_columns:
        if key not in row:
            # A silently vanishing row makes a batch "succeed" while running
            # fewer experiments than the TSV lists — always say why.
            print(
                f"{log_prefix} Skipping row {line_no}: required column "
                f"'{key}' is missing (row has {len(parts)} of {len(columns)} columns)."
            )
            return None

    if defaults:
        for key, default_val in defaults.items():
            row.setdefault(key, default_val)

    return row


# ---------------------------------------------------------------------------
# File-level reading with header detection
# ---------------------------------------------------------------------------

def read_tsv_rows(
    tsv_path: str,
    known_columns: Set[str],
    min_header_columns: int,
    log_prefix: str,
) -> Tuple[Optional[List[str]], List[Tuple[int, List[str]]]]:
    """Read a TSV file with automatic header detection.

    Header detection (both ``#``-comment and first data line):

    1. The first ``#``-prefixed comment line whose tokens (after stripping
       ``#``) are all in *known_columns* becomes the header.
    2. Otherwise the first non-comment data line is checked.

    Returns ``(header_columns_or_None, [(line_no, parts), ...])``.
    """
    if not os.path.isfile(tsv_path):
        print(f"{log_prefix} Tasks TSV not found: {tsv_path}")
        return None, []

    header_columns: Optional[List[str]] = None
    data_rows: List[Tuple[int, List[str]]] = []

    with open(tsv_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            # Check commented lines as potential headers.
            if stripped.startswith("#"):
                if header_columns is None:
                    comment_body = stripped.lstrip("#").strip()
                    if comment_body:
                        comment_parts = comment_body.split()
                        if is_header_row(comment_parts, known_columns, min_header_columns):
                            header_columns = [p.lower() for p in comment_parts]
                continue

            parts = stripped.split()

            # Detect header row on first non-comment data line.
            if header_columns is None and is_header_row(parts, known_columns, min_header_columns):
                header_columns = [p.lower() for p in parts]
                continue

            data_rows.append((line_no, parts))

    return header_columns, data_rows


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def dedup_tasks(
    tasks: List[Dict[str, Any]],
    identity_fn: Callable[[Dict[str, Any]], tuple],
) -> List[Dict[str, Any]]:
    """Remove duplicate tasks based on an identity function."""
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for task in tasks:
        key = identity_fn(task)
        if key not in seen:
            seen.add(key)
            unique.append(task)
    return unique


# ---------------------------------------------------------------------------
# Config-building helpers
# ---------------------------------------------------------------------------

def set_cfg_field_and_track(
    cfg_obj: object,
    attr: str,
    value: Any,
    key: str,
    explicit_keys: List[str],
) -> None:
    """Set a config attribute and record the key in *explicit_keys* when *value* is not ``None``."""
    if value is not None:
        setattr(cfg_obj, attr, value)
        explicit_keys.append(key)


__all__ = [
    "TASK_TYPES",
    "CustomParser",
    "dedup_tasks",
    "is_header_row",
    "parse_row_by_header",
    "parse_standard_column",
    "read_tsv_rows",
    "set_cfg_field_and_track",
]
