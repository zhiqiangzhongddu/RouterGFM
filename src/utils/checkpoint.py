"""Shared checkpoint and training-log persistence helpers.

Used by pretrain, train, and finetune runners.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from collections.abc import Mapping
from typing import Any

import torch

from src.utils.paths import ensure_dir

# Sweep orphaned atomic-save tmp siblings older than this (seconds). Short
# enough to clear stale files quickly, long enough that a concurrent peer
# writer is never disturbed mid-save.
_TMP_MAX_AGE_SEC = 3600


def _atomic_tmp_suffix() -> str:
    """Return a per-writer unique suffix for atomic-save tmp files.

    PID alone is not unique across SLURM nodes sharing the project filesystem
    (different nodes have independent PID namespaces), and PIDs are also
    recycled on a single host.  Appending an 8-char random token makes
    collision effectively impossible. The final tmp basename uses a separate
    short hash prefix so long destination names do not exceed ``NAME_MAX``.
    """
    return f"{os.getpid()}.{secrets.token_hex(4)}"


def _atomic_tmp_prefix(basename: str) -> str:
    """Return a short deterministic prefix for tmp siblings of *basename*."""
    digest = hashlib.sha256(str(basename).encode("utf-8")).hexdigest()[:16]
    return f".atomic-{digest}.tmp."


def _atomic_tmp_path(path: str) -> str:
    directory = os.path.dirname(path)
    basename = os.path.basename(path)
    return os.path.join(
        directory,
        f"{_atomic_tmp_prefix(basename)}{_atomic_tmp_suffix()}",
    )


def _sweep_stale_tmp(directory: str, basename: str) -> None:
    """Remove hashed tmp siblings for *basename* older than _TMP_MAX_AGE_SEC.

    Called opportunistically before each save so tmp siblings left behind
    by SIGKILL / OOM mid-``torch.save`` don't accumulate. Best-effort only:
    any error is swallowed so a missing dir or racing peer never breaks
    the save itself.
    """
    try:
        cutoff = time.time() - _TMP_MAX_AGE_SEC
        prefixes = (_atomic_tmp_prefix(basename), f"{basename}.tmp.")
        with os.scandir(directory) as it:
            for entry in it:
                if not entry.name.startswith(prefixes):
                    continue
                try:
                    if entry.stat().st_mtime < cutoff:
                        os.remove(entry.path)
                except OSError:
                    pass
    except OSError:
        pass


def _plain_config_value(
    value: Any,
    *,
    path: str,
    active_containers: set[int],
) -> Any:
    """Recursively copy config values into JSON/weights-only-safe containers."""
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_containers:
            raise TypeError(f"Cyclic configuration container at {path}.")
        active_containers.add(identity)
        try:
            copied: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Unsupported configuration key at {path}: "
                        f"expected str, got {type(key).__module__}.{type(key).__qualname__}."
                    )
                plain_key = str(key)
                copied[plain_key] = _plain_config_value(
                    item,
                    path=f"{path}.{plain_key}",
                    active_containers=active_containers,
                )
            return copied
        finally:
            active_containers.remove(identity)

    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active_containers:
            raise TypeError(f"Cyclic configuration container at {path}.")
        active_containers.add(identity)
        try:
            return [
                _plain_config_value(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active_containers.remove(identity)

    raise TypeError(
        f"Unsupported configuration value at {path}: "
        f"{type(value).__module__}.{type(value).__qualname__}. "
        "Expected mappings, lists/tuples, or scalar str/int/float/bool/None values."
    )


def cfg_to_dict(cfg) -> dict[str, Any]:
    """Return an independent recursively plain representation of ``cfg``.

    YACS ``CfgNode`` is a ``dict`` subclass, so a shallow ``isinstance`` check
    is insufficient: persisting it directly makes PyTorch checkpoints fail
    ``torch.load(weights_only=True)``. Normalizing plain dictionaries through
    the same path also prevents nested config subclasses from leaking into an
    otherwise ordinary mapping.
    """
    payload = _plain_config_value(
        cfg,
        path="cfg",
        active_containers=set(),
    )
    if not isinstance(payload, dict):
        raise TypeError(
            "Configuration root must be a mapping, got "
            f"{type(cfg).__module__}.{type(cfg).__qualname__}."
        )
    return payload


def save_json_atomic(path: str, payload: Any, *, indent: int | None = 2) -> None:
    """Write *payload* as JSON via a tmp sibling + ``os.replace``.

    Concurrent same-path writers each land a complete file (last one wins);
    readers never observe a partial or interleaved write.
    """
    ensure_dir(os.path.dirname(path))
    _sweep_stale_tmp(os.path.dirname(path), os.path.basename(path))
    tmp_path = _atomic_tmp_path(path)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_torch_atomic(path: str, payload: Any) -> None:
    """``torch.save`` via a tmp sibling + ``os.replace`` (see save_json_atomic)."""
    ensure_dir(os.path.dirname(path))
    _sweep_stale_tmp(os.path.dirname(path), os.path.basename(path))
    tmp_path = _atomic_tmp_path(path)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_checkpoint(
    path: str,
    model,
    optimizer,
    epoch: int,
    cfg,
    dataset_meta: dict[str, Any],
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg_payload = cfg_to_dict(cfg)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg_payload,
        "dataset": dataset_meta,
        "metrics": metrics,
    }
    if extra:
        payload["extra"] = extra
    _sweep_stale_tmp(os.path.dirname(path), os.path.basename(path))
    tmp_path = _atomic_tmp_path(path)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_training_log(
    path: str,
    cfg,
    dataset_meta: dict[str, Any],
    history: list[dict[str, Any]],
    best_info: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a JSON training log shared by pretrain, train, and finetune.

    *best_info* should contain ``epoch``, ``metric``, and ``monitor`` keys.
    *extra* is an optional dict merged into the top level (e.g. finetune's
    ``pretrained_from`` block).
    """
    ensure_dir(os.path.dirname(path))
    payload: dict[str, Any] = {
        "config": cfg_to_dict(cfg),
        "dataset_meta": dataset_meta,
    }
    if extra:
        payload.update(extra)
    payload["history"] = history
    payload["best"] = best_info
    _sweep_stale_tmp(os.path.dirname(path), os.path.basename(path))
    tmp_path = _atomic_tmp_path(path)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


__all__ = [
    "cfg_to_dict",
    "save_json_atomic",
    "save_checkpoint",
    "save_torch_atomic",
    "save_training_log",
]
