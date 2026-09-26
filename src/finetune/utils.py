from __future__ import annotations

import glob
import json
import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from src.finetune.finetuner import FinetuneRunner
from src.utils import build_pretrain_run_name_from_cfg
from src.utils.dataset_helpers import checkpoint_dataset_dir_name
from src.utils.parsing import looks_bool, to_bool
from src.utils.random import set_seed
from src.utils.run_helpers import (
    aggregate_run_metrics,
    collect_checkpoint_paths,
    collect_run_metrics,
    resolve_seeds,
    should_save_result,
    summarize_runs,
)
from src.utils.save_results import (
    append_workflow_result,
    get_explicit_cfg_keys,
    set_explicit_cfg_keys,
)
from src.utils.tsv_parsing import (
    dedup_tasks,
    parse_row_by_header,
    read_tsv_rows,
    set_cfg_field_and_track,
)

_FINETUNE_METHODS = {"supervised", "all_in_one", "edgeprompt", "gpf", "gppt", "graphprompt", "pronog"}
_SEED_IN_RUN_NAME_RE = re.compile(r"(?:^|_)seed(-?\d+)(?:_|$)")

# All recognised column names for header-based TSV parsing.
_HEADER_COLUMNS = {
    # Run control columns.
    "skip_if_exists",
    # Required finetune target columns.
    "dataset", "task_level", "induced",
    # Optional finetune columns.
    "task_type", "fixed_split", "pretrained_run_name", "finetune_method",
    # Pretrain source columns (used by test experiments).
    "model", "pretrain_dataset", "pretrain_task_level", "pretrain_induced", "pretrain_method",
    # Method variant columns.
    "graphprompt_plus", "gpf_plus", "edgeprompt_plus",
}

_REQUIRED_COLUMNS = ("dataset", "task_level", "induced")

_DEFAULTS = {
    "task_type": None,
    "fixed_split": None,
    "pretrained_run_name": None,
    "finetune_method": None,
}


def extract_few_shot(argv: list[str]) -> tuple[list[str], tuple[int, float, float] | None]:
    """Pull a `--fewshot shots val_ratio test_ratio` override out of argv."""
    cleaned: list[str] = []
    few_shot: tuple[int, float, float] | None = None
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == "--fewshot":
            if idx + 3 >= len(argv):
                raise ValueError("--fewshot requires: shots_per_class val_ratio test_ratio")
            shots = int(argv[idx + 1])
            val_ratio = float(argv[idx + 2])
            test_ratio = float(argv[idx + 3])
            if shots < 1:
                raise ValueError("--fewshot requires shots_per_class >= 1.")
            if val_ratio < 0.0 or test_ratio < 0.0:
                raise ValueError("--fewshot requires non-negative val_ratio and test_ratio.")
            if (val_ratio + test_ratio) <= 0.0:
                raise ValueError("--fewshot requires val_ratio + test_ratio > 0.")
            few_shot = (shots, val_ratio, test_ratio)
            idx += 4
            continue
        cleaned.append(token)
        idx += 1
    return cleaned, few_shot


def _normalize_method_name(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _task_identity(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        task["dataset"],
        task["task_level"],
        bool(task["induced"]),
        task.get("task_type"),
        task.get("fixed_split"),
        task.get("pretrained_run_name"),
        task.get("finetune_method"),
        task.get("model"),
        task.get("pretrain_dataset"),
        task.get("pretrain_task_level"),
        task.get("pretrain_induced"),
        task.get("pretrain_method"),
        task.get("graphprompt_plus"),
        task.get("gpf_plus"),
        task.get("edgeprompt_plus"),
        task.get("skip_if_exists"),
    )


def _finetune_custom_parser(col: str, val: str, line_no: int) -> tuple[Any, bool] | None:
    """Custom column parser for finetune-specific columns.

    Returns ``(value, ok)`` for handled columns, or ``None`` to fall
    through to :func:`parse_standard_column`.
    """
    if col == "finetune_method":
        normalized = _normalize_method_name(val)
        if normalized and normalized not in _FINETUNE_METHODS:
            print(f"[Finetune] Skipping malformed task row {line_no}: unknown finetune method '{val}'")
            return None, False
        return (normalized or None), True

    if col == "pretrained_run_name":
        return (None if val in ("-", "_") else val), True

    if col in ("graphprompt_plus", "gpf_plus", "edgeprompt_plus"):
        if val in ("-", "_", ""):
            return None, True
        if not looks_bool(val):
            print(f"[Finetune] Skipping malformed task row {line_no}: invalid {col} flag '{val}'")
            return None, False
        return to_bool(val), True

    # Fall through to standard parsing.
    return None


def parse_finetune_tasks(tsv_path: str) -> list[dict[str, Any]]:
    """Read finetune tasks from a header-based TSV.

    The file must begin with a header row whose tokens are all known
    column names.  Rows are parsed by column name.
    """
    header_columns, data_rows = read_tsv_rows(
        tsv_path, _HEADER_COLUMNS, min_header_columns=3, log_prefix="[Finetune]",
    )

    tasks: list[dict[str, Any]] = []
    if header_columns is None:
        if data_rows:
            print("[Finetune] Missing header row; expected a '#'-prefixed header with known column names.")
        return tasks

    for line_no, parts in data_rows:
        row = parse_row_by_header(
            parts, header_columns, line_no, "[Finetune]",
            required_columns=_REQUIRED_COLUMNS,
            defaults=_DEFAULTS,
            custom_parser=_finetune_custom_parser,
        )
        if row is not None:
            tasks.append(row)

    return dedup_tasks(tasks, _task_identity)


def _iter_checkpoint_files(root: str) -> list[str]:
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    return sorted(str(path) for path in root_path.rglob("*.pt") if path.is_file())


def _run_name_has_variant_tag(run_name: str, method: str, model_name: str) -> bool:
    """Detect whether a run name carries a method- or model-variant tag.

    ``build_pretrain_run_name_from_cfg`` encodes variants as ``method-<tag>`` and
    ``model-<tag>`` fragments. Naive tokenization by underscore is
    unsafe because method/model names can themselves contain
    underscores (e.g. ``edge_pred``). Instead, check for literal
    ``<name>-`` substrings adjacent to the expected underscore
    boundaries: the method appears at the run-name start (``method-``)
    and the model appears preceded by an underscore (``_model-``).
    """
    if not run_name:
        return False
    name_l = run_name.lower()
    method_l = str(method or "").lower()
    model_l = str(model_name or "").lower()
    if method_l and name_l.startswith(f"{method_l}-"):
        return True
    if model_l and f"_{model_l}-" in name_l:
        return True
    return False


def _cfg_dict_to_cfgnode(cfg_dict: dict[str, Any]):
    """Rehydrate a persisted cfg dict into a YACS ``CfgNode``.

    Used by the finetune fallback matcher to replay ``build_pretrain_run_name_from_cfg``
    on each candidate's stored config. Returns ``None`` on any failure so
    the caller can treat the candidate as unverifiable.
    """
    try:
        from yacs.config import CfgNode as CN
    except Exception:  # pragma: no cover - defensive
        return None
    if not isinstance(cfg_dict, dict):
        return None
    try:
        return CN(init_dict=cfg_dict)
    except Exception:
        return None


def _coerce_optional_seed(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _seed_from_cfg_dict(cfg_dict: dict[str, Any]) -> int | None:
    seed = _coerce_optional_seed(cfg_dict.get("seed"))
    if seed is not None:
        return seed
    raw_seeds = cfg_dict.get("seeds")
    if isinstance(raw_seeds, (list, tuple)) and len(raw_seeds) == 1:
        return _coerce_optional_seed(raw_seeds[0])
    return None


def _seed_from_run_name(run_name: str) -> int | None:
    match = _SEED_IN_RUN_NAME_RE.search(str(run_name or ""))
    if not match:
        return None
    return _coerce_optional_seed(match.group(1))


def _checkpoint_seed(checkpoint_meta: dict[str, Any]) -> int | None:
    seed = _coerce_optional_seed(checkpoint_meta.get("seed"))
    if seed is not None:
        return seed
    return _seed_from_run_name(str(checkpoint_meta.get("run_name") or ""))


def _extract_pretrain_meta_from_log(log_path: Path) -> dict[str, Any]:
    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)
    cfg_dict = log.get("config") or log.get("cfg") or {}
    pretrain_cfg = cfg_dict.get("pretrain") or {}
    dataset_cfg = pretrain_cfg.get("dataset") or cfg_dict.get("dataset") or {}
    model_cfg = cfg_dict.get("model") or {}
    # Return the full config dict alongside the coarse fields so the
    # fallback matcher can reconstruct the candidate's run name and
    # compare variant tags -- the coarse fields alone cannot
    # distinguish e.g. bn vs non-bn or graph_pooling=max runs that
    # differ only in model-level knobs captured by
    # ``model_variant_tag_for``.
    return {
        "dataset": dataset_cfg.get("name"),
        "task_level": dataset_cfg.get("task_level"),
        "induced": dataset_cfg.get("induced"),
        "method": pretrain_cfg.get("method"),
        "model": model_cfg.get("name"),
        "seed": _seed_from_cfg_dict(cfg_dict),
        "_cfg_dict": cfg_dict,
    }


def collect_pretrained_checkpoints(
    root: str,
    log_root: str | None = None,
) -> list[dict[str, Any]]:
    """List pretrained checkpoint paths and basic metadata under a root directory.

    Metadata is only read from the *exact* log file matching the checkpoint name
    (``<run_name>_log.json``).  Unrelated log files in the same directory are
    never consulted, so a checkpoint whose own log is missing will have
    ``None`` metadata rather than metadata borrowed from a sibling.

    Logs are resolved under ``<log_root>/<dataset>/<run_name>_log.json``
    (the layout produced by ``cfg.pretrain.log_dir``).
    """
    checkpoints: list[dict[str, Any]] = []
    if not os.path.isdir(root):
        print(f"[Finetune] Pretrained checkpoint dir not found: {root}")
        return checkpoints

    root_path = Path(root).resolve()
    log_root_path = Path(log_root).resolve() if log_root else None
    missing_log_warned = False
    for ckpt_path_str in _iter_checkpoint_files(root):
        ckpt_path = Path(ckpt_path_str).resolve()
        run_name = ckpt_path.stem
        dataset_hint = None
        try:
            rel_parts = ckpt_path.relative_to(root_path).parts
            if len(rel_parts) >= 2:
                candidate_hint = rel_parts[0]
                if candidate_hint != run_name:
                    dataset_hint = candidate_hint
        except Exception:
            dataset_hint = None
        meta = {
            "path": str(ckpt_path),
            "run_name": run_name,
            "dataset": dataset_hint,
            "task_level": None,
            "induced": None,
            "method": None,
            "model": None,
            "seed": _seed_from_run_name(run_name),
            "_cfg_dict": None,
        }

        # Only read the exact log file for this checkpoint — never borrow
        # metadata from sibling logs. The configured log_root is preferred;
        # a sidecar `<run_name>_log.json` next to the .pt is the legacy
        # fallback (still an exact-stem match, so the strictness holds).
        exact_log: Path | None = None
        if log_root_path is not None:
            try:
                rel_parent = ckpt_path.parent.relative_to(root_path)
                candidate = log_root_path / rel_parent / f"{run_name}_log.json"
                if candidate.is_file():
                    exact_log = candidate
            except Exception:
                exact_log = None
        if exact_log is None:
            sidecar = ckpt_path.parent / f"{run_name}_log.json"
            if sidecar.is_file():
                exact_log = sidecar

        if exact_log is not None:
            try:
                extracted_meta = _extract_pretrain_meta_from_log(exact_log)
                if extracted_meta.get("seed") is None:
                    extracted_meta["seed"] = meta.get("seed")
                meta.update(extracted_meta)
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[Finetune] Failed to read log for {run_name}: {exc}")
        elif not missing_log_warned:
            searched = (
                str(log_root_path / f"<dataset>/{run_name}_log.json")
                if log_root_path is not None
                else "<log_root not configured>"
            )
            print(
                f"[Finetune] WARN: No metadata log found for checkpoint "
                f"'{run_name}' (searched: {searched}). Metadata-based filtering "
                f"will treat this checkpoint as unverifiable."
            )
            missing_log_warned = True
        checkpoints.append(meta)
    return checkpoints


def resolve_pretrained_checkpoint(cfg) -> tuple[str | None, str | None]:
    """Resolve checkpoint path for a single finetune run using pretrain/model config.

    Resolution strategy (stops at the first hit):
    1. Exact ``<dataset_dir>/<run_name>.pt`` path.
    2. Recursive glob for ``<run_name>.pt`` under the checkpoint root.
    3. Metadata-based fallback — filters by dataset (from directory) plus
       any checkpoint-log fields that are available (task_level, induced,
       method, model).  Requires at least one log field to be verified and
       exactly one candidate to match.  Fields missing from the checkpoint
       log are skipped (not treated as mismatches), so the match may be
       partial when log metadata is incomplete. Seed metadata must match the
       requested seed; seedless candidates are rejected for seed-stamped
       requests. A warning lists any other unverified fields.
    """
    ckpt_root = getattr(getattr(cfg, "pretrain", None), "checkpoint_dir", "outputs/pretrained_models")

    run_name = build_pretrain_run_name_from_cfg(cfg)
    dataset_name = str(getattr(getattr(cfg.pretrain, "dataset", None), "name", "") or "")
    dataset_dir_name = checkpoint_dataset_dir_name(dataset_name)

    candidate = os.path.join(ckpt_root, dataset_dir_name, f"{run_name}.pt")
    if os.path.isfile(candidate):
        return candidate, run_name

    exact_matches = sorted(glob.glob(os.path.join(ckpt_root, "**", f"{run_name}.pt"), recursive=True))
    if exact_matches:
        return exact_matches[0], run_name

    # Metadata-based fallback — only accept if there is exactly one match.
    dataset_name_lower = dataset_name.lower()
    task_level = str(getattr(getattr(cfg.pretrain, "dataset", None), "task_level", "")).lower()
    induced = bool(getattr(getattr(cfg.pretrain, "dataset", None), "induced", False))
    method = str(getattr(cfg.pretrain, "method", "")).lower()
    model_name = str(getattr(getattr(cfg, "model", None), "name", "")).lower()
    requested_seed = _coerce_optional_seed(getattr(cfg, "seed", None))
    requested_run_seed = _seed_from_run_name(run_name)
    if requested_seed is None:
        requested_seed = requested_run_seed
    log_root = getattr(getattr(cfg, "pretrain", None), "log_dir", None) or None
    candidates = collect_pretrained_checkpoints(ckpt_root, log_root=log_root)
    def _task_level_matches(ckpt_tl: str, expected_tl: str, is_induced: bool) -> bool:
        if ckpt_tl == expected_tl:
            return True
        # Induced conversion: node/edge → graph in checkpoint log.
        return is_induced and ckpt_tl == "graph" and expected_tl in ("node", "edge")

    def _fallback_matches(ckpt: dict[str, Any]) -> bool:
        # Dataset: always available from directory heuristic.
        if str(ckpt.get("dataset") or "").lower() != dataset_name_lower:
            return False
        ckpt_seed = _checkpoint_seed(ckpt)
        if requested_seed is not None:
            if ckpt_seed is not None and int(ckpt_seed) != int(requested_seed):
                return False
            if ckpt_seed is None and requested_run_seed is not None:
                return False
        # Strict: a requested field must be present AND equal on the
        # checkpoint's log metadata.  Mirrors _select_checkpoints_for_task's
        # strict semantics so the TSV path and the single-run path cannot
        # diverge.  Missing metadata -> reject; the variant-tag check
        # below provides an additional net for candidates that do carry
        # a persisted cfg.
        verified = 0
        if task_level:
            ckpt_tl = ckpt.get("task_level")
            if ckpt_tl is None:
                return False
            verified += 1
            if not _task_level_matches(str(ckpt_tl).lower(), task_level, induced):
                return False
        ckpt_induced = ckpt.get("induced")
        if ckpt_induced is None:
            return False
        verified += 1
        if bool(ckpt_induced) != induced:
            return False
        if method:
            ckpt_method = ckpt.get("method")
            if ckpt_method is None:
                return False
            verified += 1
            if str(ckpt_method).lower() != method:
                return False
        if model_name:
            ckpt_model = ckpt.get("model")
            if ckpt_model is None:
                return False
            verified += 1
            if str(ckpt_model).lower() != model_name:
                return False
        # Require at least one metadata field beyond dataset to be verified,
        # otherwise the match is based solely on directory name.
        if verified <= 0:
            return False
        # Variant-tag verification: if the candidate persisted its cfg,
        # replay ``build_pretrain_run_name_from_cfg`` on it and require the
        # resulting (variant-tagged) name to match the request's run_name
        # exactly. This closes the "coarse fields match but use_batchnorm
        # / graph_pooling / method-variant differ" hole -- otherwise
        # ``load_state_dict(strict=False)`` would silently drop keys from
        # an incompatible checkpoint.
        cand_cfg_dict = ckpt.get("_cfg_dict")
        if cand_cfg_dict:
            cand_cfg = _cfg_dict_to_cfgnode(cand_cfg_dict)
            if cand_cfg is not None:
                try:
                    cand_run_name = build_pretrain_run_name_from_cfg(cand_cfg)
                except Exception:
                    # Rebuilding the name failed -- we can't verify, so
                    # treat the candidate as unverifiable and reject.
                    return False
                if cand_run_name != run_name:
                    return False
            else:
                # Couldn't rehydrate the stored cfg -- reject rather
                # than accept, to keep the fallback from tolerating
                # unverifiable candidates.
                return False
        # Candidate has no persisted cfg; accept only if the request's
        # run name carries no variant tag beyond the base method/model
        # fields (i.e. all-default run). If there IS a variant tag, the
        # candidate cannot be verified and we must reject.
        elif _run_name_has_variant_tag(run_name, method, model_name):
            return False
        return True

    matches = [ckpt for ckpt in candidates if _fallback_matches(ckpt)]
    if len(matches) == 1:
        chosen = matches[0]
        missing_fields = [f for f in ("task_level", "induced", "method", "model")
                          if chosen.get(f) is None]
        warn_suffix = ""
        if missing_fields:
            warn_suffix = f" (unverified fields: {', '.join(missing_fields)})"
        print(
            f"[Finetune] WARN: Exact checkpoint '{run_name}' not found; "
            f"falling back to unique metadata match: {chosen['run_name']}{warn_suffix}"
        )
        return chosen["path"], chosen.get("run_name")
    if len(matches) > 1:
        names = [m.get("run_name") for m in matches]
        print(
            f"[Finetune] Checkpoint '{run_name}' not found and metadata fallback "
            f"matched {len(matches)} candidates (ambiguous): {names}"
        )
        return None, None

    print(f"[Finetune] Could not locate checkpoint for run '{run_name}' under {ckpt_root}")
    return None, None


def _apply_checkpoint_metadata(run_cfg, checkpoint_meta: dict[str, Any]) -> list[str]:
    """Replay checkpoint provenance that is not represented by coarse metadata.

    Checkpoint discovery exposes dataset/model/method fields for matching, but
    those fields cannot distinguish method variants encoded in the persisted
    config.  Downstream task rows must retain such variants or a checkpoint can
    be loaded correctly while its result row is labelled as a different run.
    Return the replayed config keys so callers can persist them explicitly.
    """
    replayed: list[str] = []
    pretrain_ds = run_cfg.pretrain.dataset
    if checkpoint_meta.get("dataset"):
        pretrain_ds.name = checkpoint_meta["dataset"]
    if checkpoint_meta.get("task_level"):
        pretrain_ds.task_level = checkpoint_meta["task_level"]
    if checkpoint_meta.get("induced") is not None:
        pretrain_ds.induced = bool(checkpoint_meta["induced"])
    if checkpoint_meta.get("method"):
        run_cfg.pretrain.method = checkpoint_meta["method"]

    cfg_dict = checkpoint_meta.get("_cfg_dict")
    if not isinstance(cfg_dict, dict):
        return replayed
    saved_pretrain = cfg_dict.get("pretrain")
    if not isinstance(saved_pretrain, dict):
        return replayed

    # InfoGraph's layerwise/nolw choice is encoded in the run-name suffix but
    # not in the coarse checkpoint metadata used by TSV task grids.
    saved_infograph = saved_pretrain.get("infograph")
    if (
        str(checkpoint_meta.get("method") or "").lower() == "infograph"
        and isinstance(saved_infograph, dict)
        and "use_layerwise" in saved_infograph
    ):
        run_cfg.pretrain.infograph.use_layerwise = bool(saved_infograph["use_layerwise"])
        replayed.append("pretrain.infograph.use_layerwise")
    return replayed


def _select_checkpoints_for_task(checkpoints: list[dict[str, Any]], task: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter checkpoints to those matching the task's pretrain source columns.

    Priority:
    1. ``pretrained_run_name`` — exact match on checkpoint filename stem.
    2. Metadata filter — ``pretrain_dataset``, ``pretrain_task_level``,
       ``pretrain_induced``, ``pretrain_method`` from the TSV row are compared
       against each checkpoint's log metadata.
    3. If no source columns are specified at all, return all checkpoints.
    """
    # 1. Explicit run name takes priority.
    requested_run_name = str(task.get("pretrained_run_name") or "").strip()
    if requested_run_name:
        exact = [c for c in checkpoints if str(c.get("run_name") or "") == requested_run_name]
        if exact:
            return exact
        lowered = requested_run_name.lower()
        return [c for c in checkpoints if str(c.get("run_name") or "").lower() == lowered]

    # 2. Filter by pretrain source metadata from the TSV row.
    filters: list[tuple[str, str]] = []  # (checkpoint_meta_key, expected_value)
    pt_model = str(task.get("model") or "").strip().lower()
    if pt_model:
        filters.append(("model", pt_model))
    pt_ds = str(task.get("pretrain_dataset") or "").strip().lower()
    if pt_ds:
        filters.append(("dataset", pt_ds))
    pt_tl = str(task.get("pretrain_task_level") or "").strip().lower()
    pt_method = str(task.get("pretrain_method") or "").strip().lower()
    if pt_method:
        filters.append(("method", pt_method))
    pt_induced = task.get("pretrain_induced")

    if not filters and pt_induced is None and not pt_tl:
        return checkpoints  # no source columns → return all

    def _matches(ckpt: dict[str, Any]) -> bool:
        # Strict policy: when a TSV row specifies a field, the checkpoint
        # metadata for that field must be present and equal.  Missing
        # metadata means "unverifiable" -> reject, which prevents a
        # log-less checkpoint from matching every row for its dataset.
        # Callers that really want to pin an unverifiable checkpoint
        # should use ``pretrained_run_name`` (handled earlier).
        verified = 0
        for meta_key, expected in filters:
            ckpt_val = ckpt.get(meta_key)
            if ckpt_val is None:
                return False  # TSV specified this field; ckpt metadata missing
            verified += 1
            if str(ckpt_val).lower() != expected:
                return False
        if pt_induced is not None:
            ckpt_induced = ckpt.get("induced")
            if ckpt_induced is None:
                return False
            verified += 1
            if bool(ckpt_induced) != bool(pt_induced):
                return False
        # Task level comparison: when induced=True, the pretrain module
        # converts node/edge tasks to graph-level.  The checkpoint log
        # stores the effective level ("graph") while the TSV row uses the
        # raw level ("node"/"edge").  Accept either as a match.
        if pt_tl:
            ckpt_tl = ckpt.get("task_level")
            if ckpt_tl is None:
                return False
            verified += 1
            ckpt_tl_lower = str(ckpt_tl).lower()
            if ckpt_tl_lower != pt_tl:
                is_induced = bool(pt_induced) if pt_induced is not None else bool(ckpt.get("induced"))
                if not (is_induced and ckpt_tl_lower == "graph" and pt_tl in ("node", "edge")):
                    return False
        # Require at least one metadata field to be actually verified.
        return verified > 0

    return [c for c in checkpoints if _matches(c)]


def _checkpoint_label(checkpoint_meta: dict[str, Any]) -> str:
    return str(checkpoint_meta.get("run_name") or checkpoint_meta.get("path") or "<unknown>")


def _select_checkpoint_for_pretrain_seed(
    checkpoints: list[dict[str, Any]],
    seed: int,
    *,
    task: dict[str, Any],
) -> dict[str, Any]:
    """Return the single discovered checkpoint for the configured pretrain seed.

    TSV source-column discovery can see historical checkpoints from several
    pretrain seeds. The current contract uses cfg.seeds[0] unless a TSV row
    explicitly names a checkpoint with ``pretrained_run_name``.
    """
    requested_seed = int(seed)
    matches: list[dict[str, Any]] = []
    seedless: list[dict[str, Any]] = []
    for ckpt in checkpoints:
        ckpt_seed = _checkpoint_seed(ckpt)
        if ckpt_seed is None:
            seedless.append(ckpt)
            continue
        if int(ckpt_seed) == requested_seed:
            matches.append(ckpt)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = [_checkpoint_label(ckpt) for ckpt in matches]
        raise ValueError(
            f"Multiple pretrained checkpoints matched pretrain seed={requested_seed} "
            f"for {task['dataset']}: {names}. Narrow the TSV pretrain source columns."
        )

    available_seeds = sorted({
        int(ckpt_seed)
        for ckpt in checkpoints
        if (ckpt_seed := _checkpoint_seed(ckpt)) is not None
    })
    if seedless:
        seedless_names = [_checkpoint_label(ckpt) for ckpt in seedless]
        raise ValueError(
            f"Could not determine a unique pretrained checkpoint for pretrain seed={requested_seed} "
            f"for {task['dataset']}. Available checkpoint seeds={available_seeds}; "
            f"seedless checkpoints={seedless_names}."
        )
    raise ValueError(
        f"No pretrained checkpoint matched pretrain seed={requested_seed} for {task['dataset']}. "
        f"Available checkpoint seeds={available_seeds}."
    )


def _build_task_cfg(base_cfg, task: dict[str, Any], checkpoint_meta: dict[str, Any]):
    run_cfg = base_cfg.clone()
    explicit_keys = get_explicit_cfg_keys(run_cfg)

    run_cfg.finetune.dataset.name = task["dataset"]
    run_cfg.finetune.dataset.task_level = task["task_level"]
    run_cfg.finetune.dataset.induced = bool(task["induced"])
    run_cfg.finetune.dataset.num_classes = None
    run_cfg.finetune.dataset.label_dim = None
    run_cfg.model.in_dim = 0
    run_cfg.finetune.run_tasks_tsv = False
    explicit_keys.extend([
        "finetune.dataset.name",
        "finetune.dataset.task_level",
        "finetune.dataset.induced",
    ])

    set_cfg_field_and_track(run_cfg.finetune.dataset, "task_type", task.get("task_type"), "finetune.dataset.task_type", explicit_keys)
    set_cfg_field_and_track(run_cfg.finetune.dataset, "fixed_split", task.get("fixed_split"), "finetune.dataset.fixed_split", explicit_keys)
    set_cfg_field_and_track(run_cfg.finetune, "method", task.get("finetune_method"), "finetune.method", explicit_keys)
    if task.get("skip_if_exists") is not None:
        run_cfg.finetune.skip_if_exists = bool(task["skip_if_exists"])

    # Store provenance so result rows identify which pretrained checkpoint was used.
    run_cfg.finetune.pretrained_checkpoint = str(checkpoint_meta.get("path") or "")
    run_cfg.finetune.pretrained_run_name = str(checkpoint_meta.get("run_name") or "")
    explicit_keys.extend([
        "finetune.pretrained_run_name",
        "finetune.pretrained_checkpoint",
    ])
    explicit_keys.extend(_apply_checkpoint_metadata(run_cfg, checkpoint_meta))

    # Apply pretrain source columns from TSV, but only for fields the
    # checkpoint metadata left empty.  When a checkpoint was selected by
    # explicit run name, its own log metadata is authoritative — TSV
    # columns must not overwrite it with potentially conflicting values.
    if checkpoint_meta.get("model"):
        run_cfg.model.name = checkpoint_meta["model"]
    elif task.get("model"):
        run_cfg.model.name = task["model"]
    if task.get("pretrain_dataset") and not checkpoint_meta.get("dataset"):
        run_cfg.pretrain.dataset.name = task["pretrain_dataset"]
    if task.get("pretrain_task_level") and not checkpoint_meta.get("task_level"):
        run_cfg.pretrain.dataset.task_level = task["pretrain_task_level"]
    if task.get("pretrain_induced") is not None and checkpoint_meta.get("induced") is None:
        run_cfg.pretrain.dataset.induced = bool(task["pretrain_induced"])
    if task.get("pretrain_method") and not checkpoint_meta.get("method"):
        run_cfg.pretrain.method = task["pretrain_method"]

    # Apply method variant columns from TSV.
    if task.get("graphprompt_plus") is not None:
        run_cfg.finetune.graphprompt.plus = bool(task["graphprompt_plus"])
    if task.get("gpf_plus") is not None:
        run_cfg.finetune.gpf.plus = bool(task["gpf_plus"])
    if task.get("edgeprompt_plus") is not None:
        run_cfg.finetune.edgeprompt.plus = bool(task["edgeprompt_plus"])

    # Only mark pretrain provenance as explicit when the value actually came
    # from checkpoint metadata or TSV source columns — not from stale defaults.
    if checkpoint_meta.get("dataset") or task.get("pretrain_dataset"):
        explicit_keys.append("pretrain.dataset.name")
    if checkpoint_meta.get("task_level") or task.get("pretrain_task_level"):
        explicit_keys.append("pretrain.dataset.task_level")
    if checkpoint_meta.get("induced") is not None or task.get("pretrain_induced") is not None:
        explicit_keys.append("pretrain.dataset.induced")
    if checkpoint_meta.get("method") or task.get("pretrain_method"):
        explicit_keys.append("pretrain.method")
    if checkpoint_meta.get("model") or task.get("model"):
        explicit_keys.append("model.name")
    if task.get("graphprompt_plus") is not None:
        explicit_keys.append("finetune.graphprompt.plus")
    if task.get("gpf_plus") is not None:
        explicit_keys.append("finetune.gpf.plus")
    if task.get("edgeprompt_plus") is not None:
        explicit_keys.append("finetune.edgeprompt.plus")
    set_explicit_cfg_keys(run_cfg, explicit_keys)
    return run_cfg


def run_finetune_tasks(cfg) -> int:
    """Run fine-tuning tasks defined in cfg.finetune.tasks_tsv.

    Each task uses the pretrained checkpoint produced with ``cfg.seeds[0]`` and
    expands finetuning itself to ``num_runs`` seeds.
    """
    tasks = parse_finetune_tasks(getattr(cfg.finetune, "tasks_tsv", ""))
    if not tasks:
        print("[Finetune] No dataset definitions found in tasks_tsv.")
        return 1

    checkpoint_root = getattr(getattr(cfg, "pretrain", None), "checkpoint_dir", "outputs/pretrained_models")
    log_root = getattr(getattr(cfg, "pretrain", None), "log_dir", None) or None
    checkpoints = collect_pretrained_checkpoints(checkpoint_root, log_root=log_root)
    if not checkpoints:
        print(f"[Finetune] No pretrained checkpoints found under {checkpoint_root}")
        return 1

    requested_runs = int(getattr(getattr(cfg, "finetune", None), "num_runs", 0) or 0)
    try:
        seeds = resolve_seeds(cfg, requested_count=requested_runs)
    except ValueError as exc:
        print(f"[Finetune][Multi-run] {exc}")
        return 1
    pretrain_seed = int(resolve_seeds(cfg, requested_count=1)[0])

    results: list[bool] = []
    for task in tasks:
        selected_checkpoints = _select_checkpoints_for_task(checkpoints, task)
        if not selected_checkpoints:
            requested = str(task.get("pretrained_run_name") or "").strip()
            if requested:
                print(f"[Finetune] No pretrained checkpoint matched run '{requested}'")
            else:
                print(f"[Finetune] No pretrained checkpoints available for task {task['dataset']}")
            results.append(False)
            continue

        started_at = datetime.now().astimezone()
        run_metrics: list[dict[str, float]] = []
        runners: list[FinetuneRunner] = []
        try:
            requested_run_name = str(task.get("pretrained_run_name") or "").strip()
            if requested_run_name:
                if len(selected_checkpoints) != 1:
                    names = [_checkpoint_label(ckpt) for ckpt in selected_checkpoints]
                    raise ValueError(
                        f"Multiple pretrained checkpoints matched explicit run "
                        f"'{requested_run_name}' for {task['dataset']}: {names}."
                    )
                ckpt = selected_checkpoints[0]
                checkpoint_source = f"explicit run={requested_run_name}"
            else:
                ckpt = _select_checkpoint_for_pretrain_seed(selected_checkpoints, pretrain_seed, task=task)
                checkpoint_source = f"pretrain seed={pretrain_seed}"
            base_cfg = _build_task_cfg(cfg, task, ckpt)
            total_runs = len(seeds)
            print(
                f"[Finetune] Running {ckpt.get('run_name')} -> "
                f"{task['dataset']} (task level={task['task_level']}, induced={task['induced']}, "
                f"{checkpoint_source})"
            )
            for index, seed in enumerate(seeds, start=1):
                run_cfg = base_cfg.clone()
                run_cfg.seed = int(seed)
                if total_runs > 1:
                    print(f"[Finetune][Multi-run] Running {index}/{total_runs} with seed={seed}")
                set_seed(run_cfg.seed)
                runner = FinetuneRunner(
                    cfg=run_cfg,
                    pretrained_checkpoint=ckpt["path"],
                    pretrained_run_name=ckpt.get("run_name"),
                )
                runner.fit()
                runners.append(runner)
                run_metrics.append(collect_run_metrics(runner, log_prefix="[Finetune][Summary]"))

            summarize_runs(run_metrics, seeds, log_prefix="[Finetune][Summary]")
            ended_at = datetime.now().astimezone()
            if should_save_result(runners, base_cfg):
                summary = aggregate_run_metrics(run_metrics)
                append_workflow_result(
                    cfg=base_cfg,
                    workflow="finetune",
                    started_at=started_at,
                    ended_at=ended_at,
                    checkpoint_save_paths=collect_checkpoint_paths(runners),
                    seeds=seeds,
                    best_epochs=summary["epoch_values"].get("best_epoch"),
                    metric_summary=summary["metric_stats"],
                )
            results.append(True)
        except Exception as exc:
            print(
                f"[Finetune] Failed {task['dataset']} "
                f"(task level={task['task_level']}, induced={task['induced']}) seeds={seeds}:"
            )
            traceback.print_exc()
            results.append(False)
    return 0 if all(results) else 1
