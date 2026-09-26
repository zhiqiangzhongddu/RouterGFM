"""Deterministic checkpoint-identity helpers for MoE methods."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.utils.checkpoint import cfg_to_dict


_OPERATIONAL_KEYS = {
    "checkpoint_dir",
    "log_dir",
    "num_runs",
    "run_tasks_tsv",
    "skip_if_exists",
    "tasks_tsv",
}


def behavior_fingerprint(
    method_cfg: Any,
    *,
    external_behavior: Any | None = None,
    length: int = 10,
) -> str:
    """Hash behavior-changing config and explicit external behavior inputs.

    ``method_cfg`` keeps excluding run-orchestration settings. Callers may
    supply behavior consumed from outside that method subtree (for example a
    shared data root) through the JSON-serializable ``external_behavior``
    payload.
    """
    payload = cfg_to_dict(method_cfg)
    if not isinstance(payload, dict):
        raise TypeError("MoE method config must serialize to a mapping.")
    normalized = {
        key: value
        for key, value in payload.items()
        if key not in _OPERATIONAL_KEYS
    }
    fingerprint_payload: Any = normalized
    if external_behavior is not None:
        try:
            json.dumps(external_behavior)
        except (TypeError, ValueError) as exc:
            raise TypeError("external_behavior must be JSON-serializable.") from exc
        fingerprint_payload = {
            "method_cfg": normalized,
            "external_behavior": external_behavior,
        }
    encoded = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[: int(length)]


__all__ = ["behavior_fingerprint"]
