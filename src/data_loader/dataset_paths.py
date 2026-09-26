"""Path helpers for dataset-scoped caches, split artifacts, and feature files.

These helpers sanitize dataset names, avoid duplicate nesting, and recover the
base dataset name for split-specific artifact directories.
"""

from pathlib import Path


def _sanitize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))
    return cleaned or "dataset"


def _dataset_scoped_dir(base_dir: Path | str, dataset_name: str) -> Path:
    base = Path(base_dir)
    scoped = _sanitize_name(dataset_name)
    if base.name == scoped:
        return base
    return base / scoped


def _base_dataset_for_split_name(dataset_name: str) -> str:
    text = str(dataset_name)
    for marker in ("_node_seed", "_graph_seed", "_edge_seed", "_seed"):
        if marker in text:
            return text.split(marker, 1)[0]
    if text.endswith("_edge"):
        return text[: -len("_edge")]
    return text


def _split_dataset_dir(split_root_path: Path, dataset_name: str) -> Path:
    return _dataset_scoped_dir(split_root_path, _base_dataset_for_split_name(dataset_name))


def _scoped_root(root: str, subdir: str) -> str:
    """Place datasets without internal namespacing into their own subfolder."""
    return str(Path(root) / subdir)
