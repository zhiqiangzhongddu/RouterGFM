"""Transactional checkpoint-pair persistence for embedded AnyGraph runners."""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import pickle
import secrets
import zipfile
from contextlib import contextmanager


_MANIFEST_VERSION = 1
_HASH_CHUNK_SIZE = 1024 * 1024


def _tmp_path(path: str) -> str:
    return f"{path}.tmp.{os.getpid()}.{secrets.token_hex(4)}"


def _manifest_path(model_path: str) -> str:
    return f"{model_path}.pair.json"


def _pending_path(model_path: str) -> str:
    return f"{model_path}.pending"


def _ensure_parent(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _file_identity(path: str, *, published_name: str | None = None) -> dict:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as file_fh:
        while True:
            chunk = file_fh.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return {
        "name": published_name or os.path.basename(path),
        "size": size,
        "sha256": digest.hexdigest(),
    }


def _write_json_temp(path: str, payload: dict) -> str:
    tmp_path = _tmp_path(path)
    with open(tmp_path, "w", encoding="utf-8") as file_fh:
        json.dump(payload, file_fh, sort_keys=True, separators=(",", ":"))
        file_fh.flush()
        os.fsync(file_fh.fileno())
    return tmp_path


def _torch_load(torch_module, model_path: str, map_location=None):
    kwargs = {"weights_only": False}
    if map_location is not None:
        kwargs["map_location"] = map_location
    try:
        return torch_module.load(model_path, **kwargs)
    except TypeError:  # torch 2.1 compatibility
        kwargs.pop("weights_only", None)
        return torch_module.load(model_path, **kwargs)


def _load_pair(torch_module, model_path: str, history_path: str, map_location=None):
    checkpoint = _torch_load(torch_module, model_path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Invalid AnyGraph model checkpoint: {model_path}")
    with open(history_path, "rb") as history_fh:
        history = pickle.load(history_fh)
    if not isinstance(history, dict):
        raise ValueError(f"Invalid AnyGraph history checkpoint: {history_path}")
    return checkpoint, history


def _verify_manifest_file(path: str, identity: object, *, label: str) -> None:
    if not isinstance(identity, dict):
        raise ValueError(f"Invalid AnyGraph pair manifest {label} entry.")
    expected_name = identity.get("name")
    expected_size = identity.get("size")
    expected_digest = identity.get("sha256")
    if expected_name != os.path.basename(path):
        raise ValueError(f"AnyGraph pair manifest {label} filename does not match {path}.")
    if not isinstance(expected_size, int) or expected_size < 0:
        raise ValueError(f"Invalid AnyGraph pair manifest {label} size.")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ValueError(f"Invalid AnyGraph pair manifest {label} SHA256.")
    actual = _file_identity(path)
    if actual["size"] != expected_size or not hmac.compare_digest(
        actual["sha256"], expected_digest
    ):
        raise ValueError(f"AnyGraph checkpoint {label} does not match its pair manifest: {path}")


def _verify_manifest_pair(model_path: str, history_path: str, manifest_path: str) -> None:
    try:
        with open(manifest_path, "r", encoding="utf-8") as manifest_fh:
            manifest = json.load(manifest_fh)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid AnyGraph pair manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != _MANIFEST_VERSION:
        raise ValueError(f"Unsupported AnyGraph pair manifest: {manifest_path}")
    generation = manifest.get("generation")
    if not isinstance(generation, str) or not generation:
        raise ValueError(f"AnyGraph pair manifest is missing a generation id: {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"AnyGraph pair manifest is missing file identities: {manifest_path}")
    _verify_manifest_file(model_path, files.get("model"), label="model")
    _verify_manifest_file(history_path, files.get("history"), label="history")


def _verify_legacy_pair(model_path: str, history_path: str) -> None:
    """Validate the pre-manifest two-file layout without importing model classes."""
    if not os.path.isfile(model_path) or not os.path.isfile(history_path):
        raise ValueError("AnyGraph checkpoint pair is incomplete.")
    if not zipfile.is_zipfile(model_path):
        raise ValueError(f"Invalid legacy AnyGraph model checkpoint: {model_path}")
    with zipfile.ZipFile(model_path, "r") as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Corrupt legacy AnyGraph model checkpoint: {model_path}")
    with open(history_path, "rb") as history_fh:
        history = pickle.load(history_fh)
    if not isinstance(history, dict):
        raise ValueError(f"Invalid legacy AnyGraph history checkpoint: {history_path}")


def _validate_published_pair_unlocked(model_path: str, history_path: str) -> str:
    """Validate one published pair and return ``manifest`` or ``legacy``."""
    manifest_path = _manifest_path(model_path)
    pending_path = _pending_path(model_path)
    if os.path.isfile(manifest_path):
        _verify_manifest_pair(model_path, history_path, manifest_path)
        return "manifest"
    # A writer publishes this marker before touching either final data path.
    # Without it, a first-generation crash after publishing the two data files
    # but before the manifest could be mistaken for a valid legacy checkpoint.
    if os.path.exists(pending_path):
        raise ValueError(f"Incomplete AnyGraph checkpoint publication: {pending_path}")
    _verify_legacy_pair(model_path, history_path)
    return "legacy"


@contextmanager
def _checkpoint_lock(model_path: str):
    lock_path = f"{model_path}.lock"
    with open(lock_path, "a", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def _save_checkpoint_pair_unlocked(
    *,
    torch_module,
    model_path: str,
    history_path: str,
    model_payload,
    history_payload,
) -> None:
    """Write and verify AnyGraph's ``.mod``/``.his`` pair transactionally.

    A pending marker is published before either final data path can change.
    The manifest, containing hashes for both files, is published last. This
    makes a crash between the two legacy path replacements detectable.
    """
    _ensure_parent(model_path)
    _ensure_parent(history_path)
    generation = secrets.token_hex(16)
    manifest_path = _manifest_path(model_path)
    pending_path = _pending_path(model_path)
    model_tmp = _tmp_path(model_path)
    history_tmp = _tmp_path(history_path)
    manifest_tmp = ""
    pending_tmp = ""
    published_data = False
    try:
        pending_tmp = _write_json_temp(
            pending_path,
            {"version": _MANIFEST_VERSION, "generation": generation},
        )
        os.replace(pending_tmp, pending_path)
        pending_tmp = ""

        with open(history_tmp, "wb") as history_fh:
            pickle.dump(history_payload, history_fh)
            history_fh.flush()
            os.fsync(history_fh.fileno())
        with open(model_tmp, "wb") as model_fh:
            torch_module.save(model_payload, model_fh)
            model_fh.flush()
            os.fsync(model_fh.fileno())

        _load_pair(torch_module, model_tmp, history_tmp, map_location="cpu")
        manifest = {
            "version": _MANIFEST_VERSION,
            "generation": generation,
            "files": {
                "model": _file_identity(
                    model_tmp, published_name=os.path.basename(model_path)
                ),
                "history": _file_identity(
                    history_tmp, published_name=os.path.basename(history_path)
                ),
            },
        }
        manifest_tmp = _write_json_temp(manifest_path, manifest)
        os.replace(model_tmp, model_path)
        published_data = True
        os.replace(history_tmp, history_path)
        os.replace(manifest_tmp, manifest_path)
        manifest_tmp = ""
        _verify_manifest_pair(model_path, history_path, manifest_path)
        _load_pair(torch_module, model_path, history_path, map_location="cpu")
        try:
            os.remove(pending_path)
        except FileNotFoundError:
            pass
    except BaseException:
        for tmp_path in (model_tmp, history_tmp, manifest_tmp, pending_tmp):
            if not tmp_path:
                continue
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
        # A normal pre-publication failure did not alter either final data
        # path, so legacy fallback remains safe. If publication began, retain
        # the marker: it prevents a partially replaced first generation from
        # being accepted as legacy.
        if not published_data:
            try:
                os.remove(pending_path)
            except FileNotFoundError:
                pass
        raise


def save_checkpoint_pair(
    *,
    torch_module,
    model_path: str,
    history_path: str,
    model_payload,
    history_payload,
) -> None:
    """Serialize concurrent writers and atomically publish one checkpoint."""
    _ensure_parent(model_path)
    with _checkpoint_lock(model_path):
        _save_checkpoint_pair_unlocked(
            torch_module=torch_module,
            model_path=model_path,
            history_path=history_path,
            model_payload=model_payload,
            history_payload=history_payload,
        )


def load_checkpoint_pair(*, torch_module, model_path: str, history_path: str):
    """Load one internally consistent checkpoint pair under the writer lock."""
    with _checkpoint_lock(model_path):
        _validate_published_pair_unlocked(model_path, history_path)
        return _load_pair(torch_module, model_path, history_path)


def checkpoint_pair_exists(*, model_path: str, history_path: str) -> bool:
    """Return whether a complete pair exists, serialized with writers/readers."""
    model_dir = os.path.dirname(model_path) or "."
    if not os.path.isdir(model_dir):
        return False
    try:
        with _checkpoint_lock(model_path):
            _validate_published_pair_unlocked(model_path, history_path)
        return True
    except Exception:
        return False


__all__ = ["checkpoint_pair_exists", "load_checkpoint_pair", "save_checkpoint_pair"]
