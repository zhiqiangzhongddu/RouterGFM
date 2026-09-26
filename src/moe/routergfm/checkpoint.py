"""Checkpoint reconstruction helpers shared by RouterGFM consumers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from yacs.config import CfgNode as CN


class RouterCheckpointError(RuntimeError):
    """Raised when a RouterGFM expert checkpoint cannot be used safely."""

    def __init__(self, checkpoint: str, reason: str):
        self.checkpoint = checkpoint
        self.reason = reason
        super().__init__(f"Invalid RouterGFM expert checkpoint {checkpoint!r}: {reason}")


def restore_checkpoint_model_cfg(runtime_cfg, checkpoint: Mapping[str, Any], *, checkpoint_path: str):
    """Return a cfg clone whose complete model subtree comes from a checkpoint.

    Only ``cfg.model`` is merged. Runtime choices such as the target dataset,
    output paths, device, and router settings therefore remain owned by the
    current invocation.
    """
    checkpoint_cfg = checkpoint.get("cfg") or {}
    if not isinstance(checkpoint_cfg, Mapping):
        raise RouterCheckpointError(checkpoint_path, "cfg is not a mapping")
    model_payload = checkpoint_cfg.get("model")
    if not isinstance(model_payload, Mapping) or not model_payload:
        raise RouterCheckpointError(checkpoint_path, "cfg.model is missing or empty")

    restored = runtime_cfg.clone()
    was_frozen = restored.is_frozen()
    if was_frozen:
        restored.defrost()
    previous_new_allowed = restored.model.is_new_allowed()
    restored.model.set_new_allowed(True)
    try:
        restored.model.merge_from_other_cfg(CN(dict(model_payload)))
        # Older checkpoints sometimes omitted model.in_dim but did record the
        # effective feature width in dataset metadata.
        if not model_payload.get("in_dim"):
            dataset_payload = checkpoint.get("dataset") or {}
            if isinstance(dataset_payload, Mapping) and dataset_payload.get("num_node_features"):
                restored.model.in_dim = int(dataset_payload["num_node_features"])
    except (KeyError, TypeError, ValueError, AssertionError) as exc:
        raise RouterCheckpointError(
            checkpoint_path,
            f"cfg.model cannot be merged into the current schema: {exc}",
        ) from exc
    finally:
        restored.model.set_new_allowed(previous_new_allowed)
        if was_frozen:
            restored.freeze()
    return restored


def checkpoint_input_dim(checkpoint: Mapping[str, Any], *, fallback: int) -> int:
    """Resolve the feature width recorded by a checkpoint."""
    checkpoint_cfg = checkpoint.get("cfg") or {}
    model_payload = checkpoint_cfg.get("model", {}) if isinstance(checkpoint_cfg, Mapping) else {}
    dataset_payload = checkpoint.get("dataset") or {}
    value = model_payload.get("in_dim") if isinstance(model_payload, Mapping) else None
    if not value and isinstance(dataset_payload, Mapping):
        value = dataset_payload.get("num_node_features")
    return int(value or fallback)


def load_encoder_state_strict(model, checkpoint: Mapping[str, Any], *, checkpoint_path: str) -> None:
    """Load a complete encoder state, rejecting partial/random restoration."""
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping) or not model_state:
        raise RouterCheckpointError(checkpoint_path, "model_state is missing or empty")
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as exc:
        raise RouterCheckpointError(
            checkpoint_path,
            f"model_state is incompatible with the reconstructed encoder: {exc}",
        ) from exc


__all__ = [
    "RouterCheckpointError",
    "checkpoint_input_dim",
    "load_encoder_state_strict",
    "restore_checkpoint_model_cfg",
]
