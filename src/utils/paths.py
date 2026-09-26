"""Path and filesystem helpers shared across the codebase."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Return the repository root directory."""
    return PROJECT_ROOT


def ensure_project_root_on_path() -> Path:
    """Ensure the repository root is importable as a top-level package location."""
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return PROJECT_ROOT


def project_path(*parts: str | os.PathLike[str]) -> Path:
    """Build an absolute path under the repository root."""
    path = PROJECT_ROOT
    for part in parts:
        path = path / Path(part)
    return path.resolve()


def resolve_project_path(
    path_str: str | os.PathLike[str] | None,
    *,
    default: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a repo-relative or absolute path against the project root."""
    raw_path = path_str
    if raw_path is None or str(raw_path).strip() == "":
        if default is None:
            raise ValueError("path_str cannot be empty when no default is provided.")
        raw_path = default
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Ensure that a directory exists."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


__all__ = [
    "PROJECT_ROOT",
    "ensure_dir",
    "ensure_project_root_on_path",
    "project_path",
    "project_root",
    "resolve_project_path",
]
