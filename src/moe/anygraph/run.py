"""AnyGraph (MoE) method orchestration."""

from __future__ import annotations

import ast
import json
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.utils import parse_csv_list, project_path, read_name_list_file, resolve_project_path
from src.utils.run_helpers import resolve_seeds
from src.utils.save_results import append_workflow_result_rows, result_table_path

from .naming import build_anygraph_route_run_name_from_cfg
from .src.checkpoint_io import checkpoint_pair_exists

from .conversion import main as run_anygraph_conversion_cli
from .graph import main as run_anygraph_graph_cli
from .link import main as run_anygraph_link_cli
from .node import main as run_anygraph_node_cli
from .report import build_agae_eval_report, build_agae_oom_rows
from .runtime import (
    ANYGRAPH_GRAPH_MAIN,
    ANYGRAPH_HISTORY_DIR,
    ANYGRAPH_LINK_MAIN,
    ANYGRAPH_MODELS_DIR,
    ANYGRAPH_NODE_MAIN,
    ensure_anygraph_runtime_files_exist,
)


@dataclass(frozen=True)
class _AnyGraphPaths:
    dataset_root: Path
    split_root: Path
    out_root: Path
    outputs_dir: Path
    link_csv: Path
    node_csv: Path
    graph_csv: Path
    report_csv: Path
    index_path: Path


@dataclass(frozen=True)
class _AnyGraphIndexState:
    payload: dict
    link_datasets: List[str]
    node_datasets: List[str]
    graph_datasets: List[str]
    link_records: List[dict]
    node_records: List[dict]
    graph_records: List[dict]


@dataclass(frozen=True)
class _AnyGraphStageSelection:
    link_setting: str
    node_setting: str
    link_datasets: List[str]
    node_datasets: List[str]


def _print_header(title: str, char: str = "=") -> None:
    line = char * 72
    print(line, flush=True)
    print(title, flush=True)
    print(line, flush=True)


def _format_module_cmd(module_name: str, argv: Sequence[str]) -> str:
    """Human-readable equivalent command; the runner actually runs in-process."""
    parts = [sys.executable, "-m", module_name, *[str(item) for item in argv]]
    return " ".join(shlex.quote(str(part)) for part in parts)


def _run_module(module_name: str, argv: Sequence[str], runner) -> None:
    printable = _format_module_cmd(module_name, argv)
    print(f"[MoE][AnyGraph][Run][in-process] {printable}", flush=True)
    rc = int(runner(argv))
    if rc != 0:
        raise RuntimeError(f"Command failed with rc={rc}: {printable}")


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_csv(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ",".join(items)
    return str(value).strip()


def _with_seed_arg(extra: List[str], cfg) -> List[str]:
    """Append --seed cfg.seed unless the user already supplied one.

    The vendored AnyGraph mains run in a subprocess, so the parent's
    set_seed() does not reach their anchor sampling / negative sampling /
    svd_lowrank; without this, two "identical-seed" runs produce different
    checkpoints under the same run name.
    """
    if "--seed" in extra:
        return extra
    seed = getattr(cfg, "seed", None)
    if seed is None:
        return extra
    return [*extra, "--seed", str(int(seed))]


def _as_token_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return shlex.split(text) if text else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            token = str(item).strip()
            if token:
                out.append(token)
        return out
    token = str(value).strip()
    return [token] if token else []


def _ordered_unique(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        key = str(value).strip()
        if not key or key in seen:
            continue
        out.append(key)
        seen.add(key)
    return out


def _anygraph_expert_pools(cfg):
    """AnyGraph's own expert-pool registry (``cfg.moe.anygraph.expert_pools``)."""
    return getattr(getattr(getattr(cfg, "moe", None), "anygraph", None), "expert_pools", None)


def _resolve_expert_pool_path(cfg, pool_key: str) -> Path:
    """Resolve an expert-pool key (e.g. primary) to its TSV path.

    AnyGraph reads its own ``cfg.moe.anygraph.expert_pools`` (the
    data/moe_anygraph_*.tsv files, which carry the per-row task_level column),
    distinct from the shared ``cfg.moe.expert_pools`` used by other MoE methods.
    """
    pools = _anygraph_expert_pools(cfg)
    raw = getattr(pools, pool_key, None) if pools is not None else None
    if not raw:
        valid = sorted(pools.keys()) if pools is not None else []
        raise ValueError(
            f"Unknown moe.anygraph.conversion.expert_pool '{pool_key}'. "
            f"Valid keys (from cfg.moe.anygraph.expert_pools): {valid}"
        )
    return resolve_project_path(str(raw))


def _read_dataset_list_file(path: Path, *, cfg_key: str) -> List[str]:
    """Read dataset names from a TSV path, with a clear error if it is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"{cfg_key} points to a missing file: {path}")
    return read_name_list_file(path)


def _expand_dataset_tokens(tokens: Sequence[str], cfg) -> List[str]:
    """Expand dataset-selection tokens to concrete source-dataset names.

    Each token is one of: an expert-pool key (e.g. ``primary``, resolved
    via ``cfg.moe.anygraph.expert_pools``), a path to a
    comment-tolerant TSV/`.txt` list (e.g. ``data/moe_anygraph_test_datasets.tsv``,
    resolved relative to the project root), or a literal dataset name. This lets
    ``moe.anygraph.{train,eval}.dataset.name`` point at the same pool/test files
    used by conversion instead of spelling out every dataset.
    """
    pools = _anygraph_expert_pools(cfg)
    pool_keys = set(pools.keys()) if pools is not None else set()
    out: List[str] = []
    for raw in tokens:
        token = str(raw).strip()
        if not token:
            continue
        if token in pool_keys:
            out.extend(_read_dataset_list_file(_resolve_expert_pool_path(cfg, token), cfg_key=f"expert pool '{token}'"))
            continue
        resolved = resolve_project_path(token)
        if resolved.is_file():
            out.extend(_read_dataset_list_file(resolved, cfg_key=f"dataset list file '{token}'"))
            continue
        out.append(token)
    return _ordered_unique(out)


def _read_dataset_level_rows(path: Path) -> List[Tuple[str, str]]:
    """Read ``(name, level)`` rows from a dataset-list TSV.

    Column 1 is the dataset name; column 2 is the ``task_level``
    (``node``/``edge``/``graph``) when present, else ``auto``. Comment (``#``)
    and blank lines are skipped; 1-column lists yield level ``auto``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Dataset list file not found: {path}")
    rows: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            name = parts[0].strip()
            level = parts[1].strip().lower() if len(parts) > 1 else "auto"
            if name:
                rows.append((name, level))
    return rows


def _ordered_unique_pairs(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    seen = set()
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def _expand_dataset_token_rows(tokens: Sequence[str], cfg) -> List[Tuple[str, str]]:
    """Expand selection tokens to ``(name, level)`` rows.

    Expert-pool keys and TSV paths are read with their task_level column so a
    dataset listed at several levels (e.g. ``cora`` as both node and edge)
    yields multiple rows; literal dataset names get level ``auto``.
    """
    pools = _anygraph_expert_pools(cfg)
    pool_keys = set(pools.keys()) if pools is not None else set()
    rows: List[Tuple[str, str]] = []
    for raw in tokens:
        token = str(raw).strip()
        if not token:
            continue
        if token in pool_keys:
            rows.extend(_read_dataset_level_rows(_resolve_expert_pool_path(cfg, token)))
            continue
        resolved = resolve_project_path(token)
        if resolved.is_file():
            rows.extend(_read_dataset_level_rows(resolved))
            continue
        rows.append((token, "auto"))
    return _ordered_unique_pairs(rows)


def _normalize_int_list(value, default: Sequence[int]) -> List[int]:
    raw = value if value not in (None, "") else default
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raw = list(default)
        else:
            try:
                raw = ast.literal_eval(text)
            except Exception:
                raw = [int(item) for item in parse_csv_list(text)]
    if isinstance(raw, (list, tuple, set)):
        values = [int(item) for item in raw]
        return values if values else [int(item) for item in default]
    return [int(raw)]


def _normalize_split_def(value) -> Tuple[float, float, float]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Split definition cannot be empty.")
        try:
            value = ast.literal_eval(text)
        except Exception:
            value = [float(item) for item in parse_csv_list(text)]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Invalid split definition '{value}'. Expected 3 values.")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _normalize_split_list(value, default: Sequence[Sequence[float]]) -> List[Tuple[float, float, float]]:
    raw = value if value not in (None, "") else default
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raw = list(default)
        else:
            try:
                raw = ast.literal_eval(text)
            except Exception:
                raw = [chunk.strip() for chunk in text.split(";") if chunk.strip()] if ";" in text else text
    if isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            return []
        if len(raw) == 3 and not isinstance(raw[0], (list, tuple)):
            return [_normalize_split_def(raw)]
        return [_normalize_split_def(item) for item in raw]
    return [_normalize_split_def(raw)]


def _split_list_arg(split_defs: Sequence[Sequence[float]]) -> str:
    return json.dumps([[float(item) for item in split_def] for split_def in split_defs])


def _resolve_token(token: str, known_datasets: Sequence[str]) -> List[str]:
    token = str(token).strip()
    if token == "":
        return []
    if "," in token:
        items = parse_csv_list(token)
    else:
        items = [token]
    known = set(known_datasets)
    unknown = [item for item in items if item not in known]
    if unknown:
        raise ValueError(
            f"Unknown datasets in setting token '{token}': {unknown}. "
            "Use dataset names that exist in the conversion index."
        )
    return _ordered_unique(items)


def _parse_dataset_setting(setting: str, known_datasets: Sequence[str]) -> Tuple[List[str], List[str], str]:
    """Return (train_datasets, test_datasets, mode) where mode in {'same','plus','in'}."""
    raw = str(setting).strip()
    if raw == "":
        return [], [], "same"
    if "+" in raw:
        idx = raw.index("+")
        train = _resolve_token(raw[:idx], known_datasets)
        test = _resolve_token(raw[idx + 1 :], known_datasets)
        return train, test, "plus"
    if "_in_" in raw:
        idx = raw.index("_in_")
        left = _resolve_token(raw[:idx], known_datasets)
        right = set(_resolve_token(raw[idx + len("_in_") :], known_datasets))
        both = [item for item in left if item in right]
        return both, both, "in"
    same = _resolve_token(raw, known_datasets)
    return same, same, "same"


def _render_dataset_setting(train_list: Sequence[str], test_list: Sequence[str], mode: str) -> str:
    train_csv = ",".join(train_list)
    test_csv = ",".join(test_list)
    if mode == "plus":
        return f"{train_csv}+{test_csv}"
    if mode == "in":
        return train_csv
    return train_csv


def _filter_setting_by_task(
    setting: str,
    *,
    link_datasets: Sequence[str],
    node_datasets: Sequence[str],
) -> Tuple[str, str]:
    known_union = _ordered_unique(list(link_datasets) + list(node_datasets))
    if not known_union:
        raise ValueError("Conversion index has no datasets; cannot derive task-specific settings.")

    train_all, test_all, mode = _parse_dataset_setting(setting, known_union)
    if len(train_all) == 0 or len(test_all) == 0:
        raise ValueError(
            f"Dataset setting resolves to empty split: '{setting}'. "
            "Provide non-empty train/test dataset names."
        )

    link_set = set(link_datasets)
    node_set = set(node_datasets)
    link_train = [item for item in train_all if item in link_set]
    link_test = [item for item in test_all if item in link_set]
    node_train = [item for item in train_all if item in node_set]
    node_test = [item for item in test_all if item in node_set]

    link_setting = ""
    node_setting = ""
    if mode == "plus":
        if link_train and link_test:
            link_setting = _render_dataset_setting(link_train, link_test, mode)
        elif link_train:
            link_setting = ",".join(link_train)
        elif link_test:
            link_setting = ",".join(link_test)

        if node_train and node_test:
            node_setting = _render_dataset_setting(node_train, node_test, mode)
        elif node_train:
            node_setting = ",".join(node_train)
        elif node_test:
            node_setting = ",".join(node_test)
    else:
        if link_train:
            link_setting = _render_dataset_setting(link_train, link_test, mode)
        if node_train:
            node_setting = _render_dataset_setting(node_train, node_test, mode)

    if link_setting == "" and node_setting == "":
        raise ValueError(
            f"Dataset setting '{setting}' contains no datasets available in converted link/node outputs."
        )
    return link_setting, node_setting


def _load_conversion_index(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse conversion index JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid conversion index payload (expected object): {path}")
    return payload


def _extract_str_list(value) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _parse_stage_dataset_names(value) -> List[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        return parse_csv_list(value)
    if isinstance(value, (list, tuple, set)):
        return _ordered_unique([str(item).strip() for item in value if str(item).strip()])
    return [str(value).strip()]


def _normalize_optional_split(value) -> Optional[Tuple[float, float, float]]:
    if value in (None, "", []):
        return None
    return _normalize_split_def(value)


def _format_fixed_split(split) -> str:
    if split in (None, "", []):
        return "<all>"
    values = _normalize_split_def(split)
    parts: List[str] = []
    for value in values:
        if abs(value - round(value)) < 1e-8:
            if abs(value) <= 1.0:
                parts.append(f"{value:.1f}")
            else:
                parts.append(str(int(round(value))))
        else:
            parts.append(f"{value:g}")
    return f"({', '.join(parts)})"


def _split_matches(record: dict, fixed_split: Optional[Tuple[float, float, float]]) -> bool:
    if fixed_split is None:
        return True
    raw_split = record.get("split")
    if not isinstance(raw_split, (list, tuple)) or len(raw_split) != 3:
        return False
    try:
        record_split = tuple(float(item) for item in raw_split)
    except Exception:
        return False
    return all(abs(left - right) < 1e-8 for left, right in zip(record_split, fixed_split))


def _record_split_tag(record: dict) -> str:
    explicit = str(record.get("split_tag", "") or "").strip()
    if explicit:
        return explicit
    raw_split = record.get("split")
    if isinstance(raw_split, (list, tuple)) and len(raw_split) == 3:
        return _format_fixed_split(raw_split)
    return ""


def _pick_stage_dataset_value(stage_cfg, field: str, fallback_cfg=None):
    dataset_cfg = getattr(stage_cfg, "dataset", None)
    value = getattr(dataset_cfg, field, None) if dataset_cfg is not None else None
    if value not in (None, "", []):
        return value
    if fallback_cfg is None:
        return value
    fallback_dataset_cfg = getattr(fallback_cfg, "dataset", None)
    if fallback_dataset_cfg is None:
        return value
    return getattr(fallback_dataset_cfg, field, None)


def _checkpoint_exists(models_dir: Path, history_dir: Path, run_name: str) -> bool:
    if not run_name:
        return False
    model_path = models_dir / f"{run_name}.mod"
    history_path = history_dir / f"{run_name}.his"
    return checkpoint_pair_exists(
        model_path=str(model_path),
        history_path=str(history_path),
    )


def _checkpoint_paths(models_dir: Path, history_dir: Path, run_name: str) -> Tuple[Path, Path]:
    return models_dir / f"{run_name}.mod", history_dir / f"{run_name}.his"


def _preview_values(values: Sequence[str], *, limit: int = 4) -> str:
    items = [str(value).strip() for value in values if str(value).strip()]
    if not items:
        return "<none>"
    if len(items) <= limit:
        return ", ".join(items)
    return f"{', '.join(items[:limit])}, ... ({len(items)} total)"


def _normalize_mode(value, *, cfg_key: str) -> str:
    mode = str(value or "all").strip().lower()
    if mode not in {"link", "node", "both", "graph", "all"}:
        raise ValueError(f"Unsupported {cfg_key}='{mode}'. Use link/node/both/graph/all.")
    return mode


def _mode_runs_route(mode: str, route: str) -> bool:
    """Which AnyGraph routes a mode triggers. ``all`` = link + node + graph
    (the default: every level present in the converted data is handled)."""
    return {
        "link": route == "link",
        "node": route == "node",
        "graph": route == "graph",
        "both": route in {"link", "node"},
        "all": route in {"link", "node", "graph"},
    }[mode]


def _route_should_run(mode: str, route: str, *, available: bool) -> bool:
    """Gate a route by both mode and availability. Under ``all`` a route is
    silently skipped when its level has no converted data; explicit modes
    (link/node/both/graph) stay strict and surface the usual errors."""
    if not _mode_runs_route(mode, route):
        return False
    return available if mode == "all" else True


def _link_node_availability(stage_selection, index_state, has_scope):
    """(link_available, node_available): does each route have data to run on?
    Falls back to all of a level's datasets only for an unscoped run."""
    link = bool(stage_selection and (
        stage_selection.link_setting or stage_selection.link_datasets
        or (not has_scope and index_state.link_datasets)
    ))
    node = bool(stage_selection and (
        stage_selection.node_setting or stage_selection.node_datasets
        or (not has_scope and index_state.node_datasets)
    ))
    return link, node


def _route_display(mode, route, stage_selection, index_state, has_scope, available) -> str:
    """Human-readable dataset summary for a link/node route in the stage banner,
    consistent with whether the route will actually run."""
    if not stage_selection or not _route_should_run(mode, route, available=available):
        return "<not selected>"
    if route == "link":
        setting, matched, all_ds = stage_selection.link_setting, stage_selection.link_datasets, index_state.link_datasets
    else:
        setting, matched, all_ds = stage_selection.node_setting, stage_selection.node_datasets, index_state.node_datasets
    fallback = all_ds if not has_scope else []
    return setting or _preview_values(matched or fallback)


def _print_stage_info(stage_name: str, rows: Sequence[Tuple[str, str]]) -> None:
    prefix = f"[MoE][AnyGraph][{stage_name}]"
    for key, value in rows:
        print(f"{prefix} {key}: {value}", flush=True)


def _resolve_stage_dataset_settings(
    stage_cfg,
    *,
    link_datasets: Sequence[str],
    node_datasets: Sequence[str],
    stage_cfg_key: str,
    fallback_cfg=None,
) -> Tuple[str, str]:
    def _pick(name: str) -> str:
        value = str(getattr(stage_cfg, name, "") or "").strip()
        if value or fallback_cfg is None:
            return value
        return str(getattr(fallback_cfg, name, "") or "").strip()

    link_dataset_setting = _pick("link_dataset_setting")
    node_dataset_setting = _pick("node_dataset_setting")
    mixed_dataset_setting = _pick("mixed_dataset_setting")

    if mixed_dataset_setting:
        if link_dataset_setting or node_dataset_setting:
            raise ValueError(
                f"Do not combine {stage_cfg_key}.mixed_dataset_setting with "
                f"{stage_cfg_key}.link_dataset_setting/{stage_cfg_key}.node_dataset_setting."
            )
        link_dataset_setting, node_dataset_setting = _filter_setting_by_task(
            mixed_dataset_setting,
            link_datasets=link_datasets,
            node_datasets=node_datasets,
        )
        print(
            "[MoE][AnyGraph][SplitSetting]",
            f"source={stage_cfg_key}.mixed_dataset_setting",
            f"mixed='{mixed_dataset_setting}'",
            f"-> link='{link_dataset_setting or '<none>'}'",
            f"node='{node_dataset_setting or '<none>'}'",
            flush=True,
        )
    return link_dataset_setting, node_dataset_setting


def _stage_has_scope(stage_cfg, fallback_cfg=None) -> bool:
    """True iff the caller pinned a dataset.name or fixed_split. When False
    (the default), a route may fall back to all converted datasets of its level;
    when True, a route must use only the explicitly-matched subset (and skip if
    none match, rather than silently widening to every dataset)."""
    name = _pick_stage_dataset_value(stage_cfg, "name", fallback_cfg)
    split = _pick_stage_dataset_value(stage_cfg, "fixed_split", fallback_cfg)
    return bool(_parse_stage_dataset_names(name)) or _normalize_optional_split(split) is not None


def _resolve_stage_selected_datasets(
    stage_cfg,
    *,
    stage_cfg_key: str,
    task_name: str,
    records: Sequence[dict],
    cfg,
    fallback_cfg=None,
    strict_unknown: bool = True,
) -> List[str]:
    source_datasets = _expand_dataset_tokens(
        _parse_stage_dataset_names(_pick_stage_dataset_value(stage_cfg, "name", fallback_cfg)),
        cfg,
    )
    fixed_split = _normalize_optional_split(_pick_stage_dataset_value(stage_cfg, "fixed_split", fallback_cfg))
    required_seeds = resolve_seeds(cfg)
    has_structured_filters = bool(source_datasets or fixed_split is not None)
    has_explicit_setting = any(
        str(getattr(stage_cfg, field, "") or "").strip()
        for field in ("link_dataset_setting", "node_dataset_setting", "mixed_dataset_setting")
    )
    if has_structured_filters and has_explicit_setting:
        raise ValueError(
            f"Do not combine {stage_cfg_key}.dataset.name/{stage_cfg_key}.dataset.fixed_split with "
            f"{stage_cfg_key}.link_dataset_setting/{stage_cfg_key}.node_dataset_setting/"
            f"{stage_cfg_key}.mixed_dataset_setting."
        )
    if not has_structured_filters:
        return []

    available_sources = _ordered_unique(
        [str(record.get("source_dataset", "") or "").strip() for record in records if str(record.get("source_dataset", "") or "").strip()]
    )
    requested_sources = source_datasets or available_sources
    unknown_sources = [name for name in requested_sources if name not in set(available_sources)]
    if unknown_sources:
        if strict_unknown:
            raise ValueError(
                f"Unknown {task_name} source datasets in {stage_cfg_key}.dataset.name: {unknown_sources}. "
                f"Available datasets: {_preview_values(available_sources)}"
            )
        # tolerant (mode=all): names belonging to other levels are skipped here
        requested_sources = [name for name in requested_sources if name in set(available_sources)]
        if not requested_sources:
            return []

    selected_records: List[dict] = []
    for source_dataset in requested_sources:
        matched_records = [
            record
            for record in records
            if str(record.get("source_dataset", "") or "").strip() == source_dataset and _split_matches(record, fixed_split)
        ]
        if not matched_records:
            if strict_unknown:
                raise ValueError(
                    f"No {task_name} datasets found for source_dataset='{source_dataset}' "
                    f"and fixed_split={_format_fixed_split(fixed_split)}."
                )
            continue

        if fixed_split is None:
            selected_records.extend(matched_records)
            continue

        per_seed = {}
        for record in matched_records:
            try:
                record_seed = int(record.get("seed", 0))
            except Exception:
                record_seed = 0
            if record_seed in per_seed:
                raise ValueError(
                    f"Duplicate {task_name} converted dataset for source_dataset='{source_dataset}', "
                    f"fixed_split={_format_fixed_split(fixed_split)}, seed={record_seed}."
                )
            per_seed[record_seed] = record

        missing_seeds = [seed for seed in required_seeds if seed not in per_seed]
        if missing_seeds and strict_unknown:
            raise ValueError(
                f"Missing {task_name} converted datasets for source_dataset='{source_dataset}', "
                f"fixed_split={_format_fixed_split(fixed_split)}, seeds={missing_seeds}. "
                "Run step=conversion first."
            )
        selected_records.extend(per_seed[seed] for seed in required_seeds if seed in per_seed)

    selected = _ordered_unique([str(record.get("dataset", "") or "").strip() for record in selected_records])
    if not selected:
        if strict_unknown:
            raise ValueError(
                f"No {task_name} datasets matched {stage_cfg_key}.dataset.name/{stage_cfg_key}.dataset.fixed_split. "
                f"Available {task_name} datasets include: {_preview_values([str(r.get('dataset', '')) for r in records])}"
            )
        return []
    selected_split_tags = _ordered_unique([_record_split_tag(record) for record in selected_records if _record_split_tag(record)])
    selected_seeds = _ordered_unique([str(record.get("seed", "")).strip() for record in selected_records])
    print(
        f"[MoE][AnyGraph][Select][{task_name}]",
        f"dataset.name={_preview_values(requested_sources)}",
        f"dataset.fixed_split={_format_fixed_split(fixed_split)}",
        f"split_tags={_preview_values(selected_split_tags)}",
        f"seeds={_preview_values(selected_seeds)}",
        f"selected={_preview_values(selected)}",
        flush=True,
    )
    return selected


def _extend_dataset_args(
    args: List[str],
    *,
    dataset_setting: str,
    datasets: Sequence[str],
    stage_cfg_key: str,
    task_name: str,
) -> None:
    if dataset_setting:
        args.extend(["--dataset_setting", dataset_setting])
        return
    if datasets:
        args.extend(["--datasets", ",".join(datasets)])
        return
    raise ValueError(
        f"No {task_name} datasets resolved. Provide {stage_cfg_key}.{task_name}_dataset_setting "
        "or run conversion first."
    )


def _route_selection_fields(
    selection: _AnyGraphStageSelection, route: str
) -> Tuple[str, List[str]]:
    if route == "link":
        return selection.link_setting, list(selection.link_datasets)
    return selection.node_setting, list(selection.node_datasets)


def _selection_has_route(selection: _AnyGraphStageSelection, route: str) -> bool:
    setting, datasets = _route_selection_fields(selection, route)
    return bool(setting) or bool(datasets)


def _resolve_route_save_path(
    cfg,
    *,
    route: str,
    explicit_value: str,
    stage_selection: _AnyGraphStageSelection,
    seeds: Sequence[int],
) -> str:
    """Pick the checkpoint tag for an AnyGraph route.

    When the caller supplied a non-empty tag (SLURM override, TSV row,
    explicit CLI value), honor it verbatim. Otherwise route through the
    canonical builder so distinct dataset/split/seed/epoch combinations
    never collide on the same ``.mod`` file.
    """
    explicit = str(explicit_value or "").strip()
    if explicit:
        return explicit
    setting, datasets = _route_selection_fields(stage_selection, route)
    return build_anygraph_route_run_name_from_cfg(
        cfg,
        route=route,
        dataset_setting=setting,
        datasets=datasets,
        seeds=seeds,
    )


def _resolve_eval_route_save_path(
    cfg,
    *,
    route: str,
    explicit_value: str,
    train_stage_selection: _AnyGraphStageSelection,
    eval_stage_selection: _AnyGraphStageSelection,
    seeds: Sequence[int],
) -> str:
    """Pick the checkpoint tag at eval time.

    A checkpoint is a product of training, so when the train-side cfg
    supplies a dataset selection, the auto-tag derives from it -- this
    lets cross-dataset (zero-shot) eval find the right checkpoint when
    the user re-passes moe.anygraph.train.dataset.name / .fixed_split
    / .epoch alongside the eval overrides. When the train-side cfg is
    empty we fall back to the eval selection so that eval-only runs
    mirroring the training cfg keep working.
    """
    explicit = str(explicit_value or "").strip()
    if explicit:
        return explicit
    selection = (
        train_stage_selection
        if _selection_has_route(train_stage_selection, route)
        else eval_stage_selection
    )
    return _resolve_route_save_path(
        cfg,
        route=route,
        explicit_value="",
        stage_selection=selection,
        seeds=seeds,
    )


def _available_route_checkpoints(
    models_dir: Path, history_dir: Path, route_name: str
) -> List[str]:
    prefix = f"anygraph_{route_name}_"
    if not models_dir.is_dir():
        return []
    tags: List[str] = []
    for model_path in sorted(models_dir.glob(f"{prefix}*.mod")):
        tag = model_path.stem
        if _checkpoint_exists(models_dir, history_dir, tag):
            tags.append(tag)
    return tags


def _require_checkpoint_tag(
    run_name: str,
    *,
    models_dir: Path,
    history_dir: Path,
    route_name: str,
    cfg_key: str,
) -> None:
    if not _checkpoint_exists(models_dir, history_dir, run_name):
        model_path, history_path = _checkpoint_paths(models_dir, history_dir, run_name)
        available = _available_route_checkpoints(models_dir, history_dir, route_name)
        if available:
            listing = "\n    - " + "\n    - ".join(available)
            available_hint = (
                f"  * Existing {route_name} checkpoints that could be loaded via "
                f"{cfg_key}=<tag>:{listing}"
            )
        else:
            available_hint = (
                f"  * No {route_name} checkpoints on disk under {models_dir}; "
                "run step=train first."
            )
        raise ValueError(
            f"Missing {route_name} checkpoint '{run_name}'. Expected "
            f"{model_path} and {history_path}.\n"
            f"  * If this is a separate eval invocation, re-pass the training cfg "
            f"(moe.anygraph.train.dataset.name, .fixed_split, .epoch) so the "
            f"auto-tag matches the one produced at train time.\n"
            f"  * For cross-dataset (zero-shot) eval, set {cfg_key}=<training tag> "
            f"to bypass auto-resolution.\n"
            f"{available_hint}"
        )


def _resolve_cfg_seeds(cfg) -> List[int]:
    return resolve_seeds(cfg)


def _resolve_runtime_paths(paths_cfg, output_cfg) -> _AnyGraphPaths:
    dataset_root = resolve_project_path(
        getattr(paths_cfg, "dataset_root", ""),
        default="data/datasets",
    )
    split_root = resolve_project_path(
        getattr(paths_cfg, "split_root", ""),
        default="data/splits",
    )
    out_root = resolve_project_path(
        getattr(paths_cfg, "out_root", ""),
        default="data/anygraph_data",
    )
    outputs_dir = project_path("outputs", "anygraph")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    link_csv = resolve_project_path(
        getattr(output_cfg, "link_csv", ""),
        default=outputs_dir / "anygraph_link_eval.csv",
    )
    node_csv = resolve_project_path(
        getattr(output_cfg, "node_csv", ""),
        default=outputs_dir / "anygraph_node_eval.csv",
    )
    graph_csv = resolve_project_path(
        getattr(output_cfg, "graph_csv", ""),
        default=outputs_dir / "anygraph_graph_eval.csv",
    )
    report_csv = resolve_project_path(
        getattr(output_cfg, "report_csv", ""),
        default=outputs_dir / "anygraph_report.csv",
    )
    link_csv.parent.mkdir(parents=True, exist_ok=True)
    node_csv.parent.mkdir(parents=True, exist_ok=True)
    graph_csv.parent.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    return _AnyGraphPaths(
        dataset_root=dataset_root,
        split_root=split_root,
        out_root=out_root,
        outputs_dir=outputs_dir,
        link_csv=link_csv,
        node_csv=node_csv,
        graph_csv=graph_csv,
        report_csv=report_csv,
        index_path=out_root / "conversion_index.json",
    )


def _load_index_state(index_path: Path) -> _AnyGraphIndexState:
    payload = _load_conversion_index(index_path)
    link_datasets = _extract_str_list(payload.get("link_datasets"))
    node_datasets = _extract_str_list(payload.get("node_datasets"))
    graph_datasets = _extract_str_list(payload.get("graph_datasets"))
    index_results = payload.get("results", []) if isinstance(payload.get("results"), list) else []

    def _records_for(task: str) -> List[dict]:
        return [
            record
            for record in index_results
            if isinstance(record, dict)
            and str(record.get("task", "")).strip() == task
            and str(record.get("status", "")).strip() == "ok"
        ]

    return _AnyGraphIndexState(
        payload=payload,
        link_datasets=link_datasets,
        node_datasets=node_datasets,
        graph_datasets=graph_datasets,
        link_records=_records_for("link"),
        node_records=_records_for("node"),
        graph_records=_records_for("graph"),
    )


def _resolve_stage_selection(
    cfg,
    stage_cfg,
    *,
    stage_cfg_key: str,
    mode: str,
    index_state: _AnyGraphIndexState,
    fallback_cfg=None,
) -> _AnyGraphStageSelection:
    link_setting, node_setting = _resolve_stage_dataset_settings(
        stage_cfg,
        link_datasets=index_state.link_datasets,
        node_datasets=index_state.node_datasets,
        stage_cfg_key=stage_cfg_key,
        fallback_cfg=fallback_cfg,
    )
    # Under ``all`` an explicit dataset.name may legitimately list datasets from
    # other levels (e.g. a graph dataset alongside node ones); be tolerant of
    # names unknown to a given route instead of erroring.
    strict_unknown = mode != "all"
    link_datasets = (
        _resolve_stage_selected_datasets(
            stage_cfg,
            stage_cfg_key=stage_cfg_key,
            task_name="link",
            records=index_state.link_records,
            cfg=cfg,
            fallback_cfg=fallback_cfg,
            strict_unknown=strict_unknown,
        )
        if _mode_runs_route(mode, "link")
        else []
    )
    node_datasets = (
        _resolve_stage_selected_datasets(
            stage_cfg,
            stage_cfg_key=stage_cfg_key,
            task_name="node",
            records=index_state.node_records,
            cfg=cfg,
            fallback_cfg=fallback_cfg,
            strict_unknown=strict_unknown,
        )
        if _mode_runs_route(mode, "node")
        else []
    )
    return _AnyGraphStageSelection(
        link_setting=link_setting,
        node_setting=node_setting,
        link_datasets=link_datasets,
        node_datasets=node_datasets,
    )


def _run_conversion_step(cfg, paths: _AnyGraphPaths, conversion_cfg, conversion_task: str) -> None:
    _print_header("[MoE][AnyGraph][Step 1] conversion", char="=")
    if bool(getattr(conversion_cfg, "skip", False)):
        print("[MoE][AnyGraph][Conversion] status: skipped by moe.anygraph.conversion.skip=True")
        return

    data_prep_cfg = getattr(cfg, "data_preparation", None)
    configured_edge_splits = list(getattr(data_prep_cfg, "edge_task_splits", [(0.1, 0.05, 0.1)]))
    configured_node_splits = list(getattr(data_prep_cfg, "node_task_splits", [(0.8, 0.1, 0.1)]))
    configured_graph_splits = list(getattr(data_prep_cfg, "graph_task_splits", [(0.8, 0.1, 0.1)]))
    default_edge_splits = configured_edge_splits if configured_edge_splits else [(0.1, 0.05, 0.1)]
    default_node_splits = configured_node_splits if configured_node_splits else [(0.8, 0.1, 0.1)]
    default_graph_splits = configured_graph_splits if configured_graph_splits else [(0.8, 0.1, 0.1)]
    conversion_seed = resolve_seeds(cfg, requested_count=1)[0]
    default_split_seeds = _resolve_cfg_seeds(cfg)
    split_seeds = _normalize_int_list(
        getattr(conversion_cfg, "seeds", default_split_seeds),
        default=default_split_seeds,
    )
    edge_splits = _normalize_split_list(
        getattr(conversion_cfg, "edge_splits", default_edge_splits),
        default=default_edge_splits,
    )
    node_splits = _normalize_split_list(
        getattr(conversion_cfg, "node_splits", default_node_splits),
        default=default_node_splits,
    )
    graph_splits = _normalize_split_list(
        getattr(conversion_cfg, "graph_splits", default_graph_splits),
        default=default_graph_splits,
    )
    dataset_file_raw = str(getattr(conversion_cfg, "dataset_file", "") or "").strip()
    pool_raw = _as_csv(getattr(conversion_cfg, "expert_pool", ""))
    test_file_raw = str(getattr(conversion_cfg, "test_dataset_file", "") or "").strip()

    # `dataset` and `expert_pool` both accept a comma-separated mix of
    # expert-pool keys (e.g. primary), dataset-list TSV paths
    # (e.g. data/moe_anygraph_test_datasets.tsv), and literal dataset names. They expand
    # to (name, level) rows — the pool TSV's task_level column decides each
    # dataset's conversion level, so a dataset listed at several levels (e.g.
    # cora as node *and* edge) is converted for each. All sources are unioned
    # (plus the legacy test_dataset_file) and passed to the converter as a
    # name:level spec. dataset_file stays the fallback when nothing is set.
    selection_tokens = (
        _parse_stage_dataset_names(getattr(conversion_cfg, "dataset", ""))
        + _parse_stage_dataset_names(getattr(conversion_cfg, "expert_pool", ""))
    )
    resolved_rows = _expand_dataset_token_rows(selection_tokens, cfg)
    if test_file_raw:
        resolved_rows = _ordered_unique_pairs(
            resolved_rows + _read_dataset_level_rows(resolve_project_path(test_file_raw))
        )
    dataset_spec = ",".join(f"{name}:{level}" for name, level in resolved_rows)
    resolved_names = _ordered_unique([name for name, _ in resolved_rows])

    if resolved_rows:
        levels = _ordered_unique([level for _, level in resolved_rows])
        dataset_summary = f"{len(resolved_names)} datasets / {len(resolved_rows)} (name,level) tasks [levels: {','.join(levels)}]; {_preview_values(resolved_names)}"
    elif dataset_file_raw:
        dataset_summary = dataset_file_raw
    else:
        dataset_summary = "<all available>"

    _print_stage_info(
        "Conversion",
        [
            ("task", conversion_task),
            ("dataset_root", str(paths.dataset_root)),
            ("split_root", str(paths.split_root)),
            ("out_root", str(paths.out_root)),
            ("conversion_index", str(paths.index_path)),
            ("expert_pool", pool_raw or "-"),
            ("test_dataset_file", test_file_raw or "-"),
            ("datasets", dataset_summary),
            ("seed_count", str(len(split_seeds))),
            ("seeds", _preview_values([str(item) for item in split_seeds])),
            ("edge_split_count", str(len(edge_splits))),
            ("node_split_count", str(len(node_splits))),
        ],
    )

    convert_args = [
        "--task",
        conversion_task,
        "--dataset_root",
        str(paths.dataset_root),
        "--split_root",
        str(paths.split_root),
        "--out_root",
        str(paths.out_root),
        "--seed",
        str(conversion_seed),
        "--mask_col",
        str(_safe_int(getattr(conversion_cfg, "mask_col", 0), 0)),
        "--feat_dim",
        str(_safe_int(getattr(conversion_cfg, "feat_dim", 100), 100)),
        "--seeds",
        str(_as_csv(split_seeds) or str(conversion_seed)),
        "--edge_splits",
        _split_list_arg(edge_splits),
        "--node_splits",
        _split_list_arg(node_splits),
        "--graph_splits",
        _split_list_arg(graph_splits),
        "--edge_eval_payload_name",
        str(getattr(conversion_cfg, "edge_eval_payload_name", "agae_edge_eval_payload.pt")),
        "--node_output_feat_dim",
        str(_safe_int(getattr(conversion_cfg, "node_output_feat_dim", 128), 128)),
        "--graph_output_feat_dim",
        str(_safe_int(getattr(conversion_cfg, "graph_output_feat_dim", 128), 128)),
        "--graph_filter_dir",
        str(resolve_project_path(getattr(conversion_cfg, "graph_filter_dir", "") or "data/filters", default="data/filters")),
        "--max_graphs",
        str(_safe_int(getattr(conversion_cfg, "max_graphs", 0), 0)),
        "--max_total_nodes",
        str(_safe_int(getattr(conversion_cfg, "max_total_nodes", 0), 0)),
        "--index_out",
        str(paths.index_path),
    ]
    convert_args.append(
        "--feat_reduction" if bool(getattr(conversion_cfg, "feat_reduction", False)) else "--no-feat_reduction"
    )
    convert_args.append(
        "--l1_normalize_features"
        if bool(getattr(conversion_cfg, "l1_normalize_features", True))
        else "--no-l1_normalize_features"
    )
    convert_args.append(
        "--emit_node_val" if bool(getattr(conversion_cfg, "emit_node_val", True)) else "--no-emit_node_val"
    )
    if dataset_spec:
        convert_args.extend(["--dataset_spec", dataset_spec])
    elif dataset_file_raw:
        convert_args.extend(
            ["--dataset_file", str(resolve_project_path(dataset_file_raw, default="data/available_node_datasets.tsv"))]
        )

    _run_module("src.moe.anygraph.conversion", convert_args, run_anygraph_conversion_cli)
    print("[MoE][AnyGraph][Conversion] status: completed")
    print(f"[MoE][AnyGraph][Conversion] conversion_index: {paths.index_path}")


def _resolve_graph_save_path(cfg, *, explicit_value, dataset_setting, datasets, seeds) -> str:
    explicit = str(explicit_value or "").strip()
    if explicit:
        return explicit
    return build_anygraph_route_run_name_from_cfg(
        cfg,
        route="graph",
        dataset_setting=dataset_setting,
        datasets=datasets,
        seeds=seeds,
    )


def _graph_head_args(graph_cfg) -> List[str]:
    """Forward per-dataset head hyperparameters to the graph runner."""
    out: List[str] = []
    for flag, key, default in (
        ("--head_lr", "head_lr", 1e-3),
        ("--head_weight_decay", "head_weight_decay", 0.0),
        ("--head_epoch", "head_epoch", 200),
        ("--head_hidden", "head_hidden", 256),
        ("--head_layers", "head_layers", 2),
        ("--head_dropout", "head_dropout", 0.2),
    ):
        out.extend([flag, str(getattr(graph_cfg, key, default))])
    return out


def _run_graph_train_route(
    cfg, *, paths, index_state, any_cfg, train_cfg, prediction_cfg, train_device, train_epoch, mode
) -> bool:
    """Run the graph route. Returns True iff a training job was launched."""
    graph_cfg = getattr(prediction_cfg, "graph", None)
    seed_scope = _resolve_cfg_seeds(cfg)
    graph_datasets = _resolve_stage_selected_datasets(
        train_cfg,
        stage_cfg_key="moe.anygraph.train",
        task_name="graph",
        records=index_state.graph_records,
        cfg=cfg,
        strict_unknown=(mode != "all"),
    )
    graph_setting = str(getattr(train_cfg, "graph_dataset_setting", "") or "").strip()
    # only fall back to all graph datasets when the caller pinned no scope
    has_scope = _stage_has_scope(train_cfg)
    datasets = graph_datasets if has_scope else (graph_datasets or index_state.graph_datasets)
    if not graph_setting and not datasets:
        # Under `all`, a pool with no graph-level datasets simply skips this route.
        if mode == "all":
            print("[MoE][AnyGraph][Train][Graph] skip: no graph datasets in the conversion index.")
            return False
        raise ValueError(
            "No graph datasets resolved for moe.anygraph.train. Provide "
            "moe.anygraph.train.dataset.name / graph_dataset_setting or run conversion first."
        )
    save_graph_path = _resolve_graph_save_path(
        cfg,
        explicit_value=getattr(train_cfg, "save_graph_path", ""),
        dataset_setting=graph_setting,
        datasets=datasets,
        seeds=seed_scope,
    )
    load_graph_model = str(getattr(train_cfg, "load_graph_model", "") or "").strip()
    _print_stage_info(
        "Train",
        [
            ("route", "graph"),
            ("device", train_device),
            ("epoch", str(train_epoch)),
            ("graph_datasets", graph_setting or _preview_values(datasets)),
            ("graph_save_path", save_graph_path),
            ("models_dir", str(ANYGRAPH_MODELS_DIR)),
        ],
    )
    shared_extra = _with_seed_arg(_as_token_list(getattr(any_cfg, "extra_args", [])), cfg)
    # Mirror the link/node routes: a supplied load tag means "do not retrain,
    # the checkpoint will be consumed at eval". A skip_if_exists hit also skips.
    if load_graph_model:
        _require_checkpoint_tag(
            load_graph_model,
            models_dir=ANYGRAPH_MODELS_DIR,
            history_dir=ANYGRAPH_HISTORY_DIR,
            route_name="graph",
            cfg_key="moe.anygraph.train.load_graph_model",
        )
        print(
            "[MoE][AnyGraph][Train][Graph] load_graph_model set; skipping training "
            f"(checkpoint '{load_graph_model}' will be used at eval)."
        )
        return False
    if bool(getattr(train_cfg, "skip_if_exists", True)) and _checkpoint_exists(
        ANYGRAPH_MODELS_DIR, ANYGRAPH_HISTORY_DIR, save_graph_path
    ):
        model_path, history_path = _checkpoint_paths(ANYGRAPH_MODELS_DIR, ANYGRAPH_HISTORY_DIR, save_graph_path)
        print(
            "[MoE][AnyGraph][Train][Graph] skip:",
            f"checkpoint exists ({save_graph_path})",
            f"model={model_path}",
            f"history={history_path}",
        )
        return False
    train_graph_csv = paths.outputs_dir / "_train_graph_eval.csv"
    graph_args = [
        "--data_root",
        str((paths.out_root / "graph").resolve()),
        "--result_csv",
        str(train_graph_csv),
        "--save_path",
        save_graph_path,
        "--gpu",
        train_device,
        "--epoch",
        str(train_epoch),
        "--tst_epoch",
        str(_safe_int(getattr(graph_cfg, "tst_epoch", 1), 1)),
        "--assignment",
        str(getattr(graph_cfg, "assignment", "top1")),
    ]
    graph_args.extend(_graph_head_args(graph_cfg))
    _extend_dataset_args(
        graph_args,
        dataset_setting=graph_setting,
        datasets=datasets,
        stage_cfg_key="moe.anygraph.train",
        task_name="graph",
    )
    graph_args.extend(shared_extra)
    _run_module("src.moe.anygraph.graph", graph_args, run_anygraph_graph_cli)
    _require_checkpoint_tag(
        save_graph_path,
        models_dir=ANYGRAPH_MODELS_DIR,
        history_dir=ANYGRAPH_HISTORY_DIR,
        route_name="graph",
        cfg_key="moe.anygraph.train.save_graph_path",
    )
    return True


def _run_train_step(
    cfg,
    *,
    paths: _AnyGraphPaths,
    index_state: _AnyGraphIndexState,
    any_cfg,
    conversion_cfg,
    train_cfg,
    prediction_cfg,
) -> None:
    _print_header("[MoE][AnyGraph][Step 2] train", char="=")
    train_mode = _normalize_mode(getattr(train_cfg, "mode", "all"), cfg_key="moe.anygraph.train.mode")
    train_device = str(_safe_int(getattr(cfg, "device", 0), 0))
    train_epoch = _safe_int(getattr(train_cfg, "epoch", 100), 100)
    if train_epoch <= 0:
        raise ValueError("moe.anygraph.train.epoch must be > 0 for step=train.")

    seed_scope = _resolve_cfg_seeds(cfg)
    shared_extra = _with_seed_arg(_as_token_list(getattr(any_cfg, "extra_args", [])), cfg)
    ran_train_route = False

    # link / node routes share the conversion-index selection
    need_link_node = _mode_runs_route(train_mode, "link") or _mode_runs_route(train_mode, "node")
    # When the caller pins dataset.name/fixed_split, a route must use only its
    # matched subset; only an unscoped run falls back to all of a level's data.
    train_has_scope = _stage_has_scope(train_cfg)
    stage_selection = None
    link_available = False
    node_available = False
    if need_link_node:
        stage_selection = _resolve_stage_selection(
            cfg,
            train_cfg,
            stage_cfg_key="moe.anygraph.train",
            mode=train_mode,
            index_state=index_state,
        )
        link_available, node_available = _link_node_availability(stage_selection, index_state, train_has_scope)
        _print_stage_info(
            "Train",
            [
                ("mode", train_mode),
                ("device", train_device),
                ("epoch", str(train_epoch)),
                ("seed_scope", _preview_values([str(seed) for seed in seed_scope])),
                ("link_datasets", _route_display(train_mode, "link", stage_selection, index_state, train_has_scope, link_available)),
                ("node_datasets", _route_display(train_mode, "node", stage_selection, index_state, train_has_scope, node_available)),
                ("models_dir", str(ANYGRAPH_MODELS_DIR)),
                ("history_dir", str(ANYGRAPH_HISTORY_DIR)),
            ],
        )

    if _route_should_run(train_mode, "link", available=link_available):
        link_cfg = prediction_cfg.link
        save_link_path = _resolve_route_save_path(
            cfg,
            route="link",
            explicit_value=getattr(train_cfg, "save_link_path", ""),
            stage_selection=stage_selection,
            seeds=seed_scope,
        )
        load_link_model = str(getattr(train_cfg, "load_link_model", "") or "").strip()
        if load_link_model:
            _require_checkpoint_tag(
                load_link_model,
                models_dir=ANYGRAPH_MODELS_DIR,
                history_dir=ANYGRAPH_HISTORY_DIR,
                route_name="link",
                cfg_key="moe.anygraph.train.load_link_model",
            )
        elif bool(getattr(train_cfg, "skip_if_exists", True)) and _checkpoint_exists(
            ANYGRAPH_MODELS_DIR,
            ANYGRAPH_HISTORY_DIR,
            save_link_path,
        ):
            model_path, history_path = _checkpoint_paths(ANYGRAPH_MODELS_DIR, ANYGRAPH_HISTORY_DIR, save_link_path)
            print(
                "[MoE][AnyGraph][Train][Link] skip:",
                f"checkpoint exists ({save_link_path})",
                f"model={model_path}",
                f"history={history_path}",
            )
        else:
            train_link_csv = paths.outputs_dir / "_train_link_eval.csv"
            link_args = [
                "--data_root",
                str((paths.out_root / "link").resolve()),
                "--result_csv",
                str(train_link_csv),
                "--save_path",
                save_link_path,
                "--gpu",
                train_device,
                "--epoch",
                str(train_epoch),
                "--tst_epoch",
                str(_safe_int(getattr(link_cfg, "tst_epoch", 1), 1)),
                "--topk",
                str(_safe_int(getattr(link_cfg, "topk", 20), 20)),
                "--eval_protocol",
                str(getattr(link_cfg, "eval_protocol", "agae")),
                "--edge_eval_threshold_mode",
                str(getattr(link_cfg, "edge_eval_threshold_mode", "val_best_acc")),
                "--edge_eval_payload_name",
                str(getattr(conversion_cfg, "edge_eval_payload_name", "agae_edge_eval_payload.pt")),
                "--edge_eval_repeat_times",
                str(_safe_int(getattr(link_cfg, "edge_eval_repeat_times", 5), 5)),
            ]
            if load_link_model:
                link_args.extend(["--load_model", load_link_model])
            _extend_dataset_args(
                link_args,
                dataset_setting=stage_selection.link_setting,
                datasets=stage_selection.link_datasets or index_state.link_datasets,
                stage_cfg_key="moe.anygraph.train",
                task_name="link",
            )
            _print_stage_info(
                "Train",
                [
                    ("link_action", "train"),
                    ("link_save_path", save_link_path),
                    ("link_resume_from", load_link_model or "<none>"),
                    ("link_result_csv", str(train_link_csv)),
                ],
            )
            link_args.extend(shared_extra)
            _run_module("src.moe.anygraph.link", link_args, run_anygraph_link_cli)
            _require_checkpoint_tag(
                save_link_path,
                models_dir=ANYGRAPH_MODELS_DIR,
                history_dir=ANYGRAPH_HISTORY_DIR,
                route_name="link",
                cfg_key="moe.anygraph.train.save_link_path",
            )
            ran_train_route = True

    if _route_should_run(train_mode, "node", available=node_available):
        node_cfg = prediction_cfg.node
        save_node_path = _resolve_route_save_path(
            cfg,
            route="node",
            explicit_value=getattr(train_cfg, "save_node_path", ""),
            stage_selection=stage_selection,
            seeds=seed_scope,
        )
        load_node_model = str(getattr(train_cfg, "load_node_model", "") or "").strip()
        if load_node_model:
            _require_checkpoint_tag(
                load_node_model,
                models_dir=ANYGRAPH_MODELS_DIR,
                history_dir=ANYGRAPH_HISTORY_DIR,
                route_name="node",
                cfg_key="moe.anygraph.train.load_node_model",
            )
        elif bool(getattr(train_cfg, "skip_if_exists", True)) and _checkpoint_exists(
            ANYGRAPH_MODELS_DIR,
            ANYGRAPH_HISTORY_DIR,
            save_node_path,
        ):
            model_path, history_path = _checkpoint_paths(ANYGRAPH_MODELS_DIR, ANYGRAPH_HISTORY_DIR, save_node_path)
            print(
                "[MoE][AnyGraph][Train][Node] skip:",
                f"checkpoint exists ({save_node_path})",
                f"model={model_path}",
                f"history={history_path}",
            )
        else:
            train_node_csv = paths.outputs_dir / "_train_node_eval.csv"
            node_args = [
                "--data_root",
                str((paths.out_root / "node").resolve()),
                "--result_csv",
                str(train_node_csv),
                "--save_path",
                save_node_path,
                "--gpu",
                train_device,
                "--epoch",
                str(train_epoch),
                "--tst_epoch",
                str(_safe_int(getattr(node_cfg, "tst_epoch", 1), 1)),
                "--assignment",
                str(getattr(node_cfg, "assignment", "top1")),
            ]
            if load_node_model:
                node_args.extend(["--load_model", load_node_model])
            _extend_dataset_args(
                node_args,
                dataset_setting=stage_selection.node_setting,
                datasets=stage_selection.node_datasets or index_state.node_datasets,
                stage_cfg_key="moe.anygraph.train",
                task_name="node",
            )
            _print_stage_info(
                "Train",
                [
                    ("node_action", "train"),
                    ("node_save_path", save_node_path),
                    ("node_resume_from", load_node_model or "<none>"),
                    ("node_result_csv", str(train_node_csv)),
                ],
            )
            node_args.extend(shared_extra)
            _run_module("src.moe.anygraph.node", node_args, run_anygraph_node_cli)
            _require_checkpoint_tag(
                save_node_path,
                models_dir=ANYGRAPH_MODELS_DIR,
                history_dir=ANYGRAPH_HISTORY_DIR,
                route_name="node",
                cfg_key="moe.anygraph.train.save_node_path",
            )
            ran_train_route = True

    if _mode_runs_route(train_mode, "graph"):
        if _run_graph_train_route(
            cfg,
            paths=paths,
            index_state=index_state,
            any_cfg=any_cfg,
            train_cfg=train_cfg,
            prediction_cfg=prediction_cfg,
            train_device=train_device,
            train_epoch=train_epoch,
            mode=train_mode,
        ):
            ran_train_route = True

    if not ran_train_route:
        print("[MoE][AnyGraph][Train] status: no routes executed (all requested checkpoints already existed)")
    else:
        print("[MoE][AnyGraph][Train] status: completed")


def _oom_marker_path(result_csv: Path) -> Path:
    """Durable sidecar written by an oom_tolerant eval subprocess."""
    return result_csv.parent / f"{result_csv.name}.oom.json"


def _read_oom_marker_datasets(marker_path: Path) -> List[str]:
    """Read the converted dataset names recorded in an OOM marker sidecar."""
    if not marker_path.is_file():
        return []
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse OOM marker JSON: {marker_path}") from exc
    raw = payload.get("datasets", []) if isinstance(payload, dict) else []
    return [str(item).strip() for item in raw if str(item).strip()]


def _run_graph_eval_route(
    cfg, *, paths, index_state, any_cfg, train_cfg, eval_cfg, prediction_cfg, eval_device, mode
) -> Tuple[bool, List[Dict[str, str]]]:
    """Evaluate the graph route.

    Returns ``(ran, oom_rows)``: ``ran`` is True iff an evaluation was
    launched; ``oom_rows`` are the ``result_status=OOM`` rows for datasets the
    eval subprocess had to skip on CUDA OOM (empty when none OOMed).
    """
    graph_cfg = getattr(prediction_cfg, "graph", None)
    seed_scope = _resolve_cfg_seeds(cfg)
    eval_datasets = _resolve_stage_selected_datasets(
        eval_cfg,
        stage_cfg_key="moe.anygraph.eval",
        task_name="graph",
        records=index_state.graph_records,
        cfg=cfg,
        fallback_cfg=train_cfg,
        strict_unknown=(mode != "all"),
    )
    eval_setting = str(getattr(eval_cfg, "graph_dataset_setting", "") or "").strip()
    eval_has_scope = _stage_has_scope(eval_cfg, fallback_cfg=train_cfg)
    datasets = eval_datasets if eval_has_scope else (eval_datasets or index_state.graph_datasets)
    if not eval_setting and not datasets:
        if mode == "all":
            print("[MoE][AnyGraph][Eval][Graph] skip: no graph datasets in the conversion index.")
            return False, []
        raise ValueError(
            "No graph datasets resolved for moe.anygraph.eval. Provide "
            "moe.anygraph.eval.dataset.name / graph_dataset_setting or run conversion first."
        )

    load_graph_model = str(getattr(eval_cfg, "load_graph_model", "") or "").strip()
    if not load_graph_model:
        # The checkpoint is a product of training: derive its tag from the
        # train-side selection so cross-dataset (held-out) eval finds it.
        train_graph_datasets = _resolve_stage_selected_datasets(
            train_cfg,
            stage_cfg_key="moe.anygraph.train",
            task_name="graph",
            records=index_state.graph_records,
            cfg=cfg,
            strict_unknown=(mode != "all"),
        )
        train_setting = str(getattr(train_cfg, "graph_dataset_setting", "") or "").strip()
        train_has_scope = _stage_has_scope(train_cfg)
        train_datasets = train_graph_datasets if train_has_scope else (train_graph_datasets or index_state.graph_datasets)
        explicit_save = str(getattr(train_cfg, "save_graph_path", "") or "").strip()
        load_graph_model = explicit_save or _resolve_graph_save_path(
            cfg,
            explicit_value="",
            dataset_setting=train_setting,
            datasets=train_datasets,
            seeds=seed_scope,
        )
    _require_checkpoint_tag(
        load_graph_model,
        models_dir=ANYGRAPH_MODELS_DIR,
        history_dir=ANYGRAPH_HISTORY_DIR,
        route_name="graph",
        cfg_key="moe.anygraph.eval.load_graph_model",
    )
    shared_extra = _with_seed_arg(_as_token_list(getattr(any_cfg, "extra_args", [])), cfg)
    _print_stage_info(
        "Eval",
        [
            ("mode", "graph"),
            ("device", eval_device),
            ("graph_datasets", eval_setting or _preview_values(datasets)),
            ("graph_load_model", load_graph_model),
            ("graph_csv", str(paths.graph_csv)),
        ],
    )
    graph_args = [
        "--data_root",
        str((paths.out_root / "graph").resolve()),
        "--result_csv",
        str(paths.graph_csv),
        "--save_path",
        load_graph_model,
        "--load_model",
        load_graph_model,
        "--gpu",
        eval_device,
        "--epoch",
        "0",
        "--tst_epoch",
        str(_safe_int(getattr(graph_cfg, "tst_epoch", 1), 1)),
        "--assignment",
        str(getattr(graph_cfg, "assignment", "top1")),
    ]
    graph_args.extend(_graph_head_args(graph_cfg))
    _extend_dataset_args(
        graph_args,
        dataset_setting=eval_setting,
        datasets=datasets,
        stage_cfg_key="moe.anygraph.eval",
        task_name="graph",
    )
    # Eval-only OOM tolerance: a per-dataset CUDA OOM inside the subprocess is
    # recorded in the <graph_csv>.oom.json sidecar instead of aborting the run;
    # the train route never passes this flag. Clear any stale marker first so
    # a previous run's OOMs cannot leak into this run's results.
    marker_path = _oom_marker_path(paths.graph_csv)
    if marker_path.is_file():
        marker_path.unlink()
    # save_eval_csv skips the write when every dataset OOMed, so drop the
    # previous run's CSV too; otherwise its rows would be re-aggregated as
    # fresh results next to this run's OOM rows.
    if paths.graph_csv.is_file():
        paths.graph_csv.unlink()
    graph_args.extend(["--oom_tolerant", "1"])
    graph_args.extend(shared_extra)
    _run_module("src.moe.anygraph.graph", graph_args, run_anygraph_graph_cli)
    print(f"[MoE][AnyGraph][Eval][Graph] graph_csv: {paths.graph_csv}")
    oom_datasets = _read_oom_marker_datasets(marker_path)
    oom_rows = build_agae_oom_rows(
        oom_datasets,
        paths.index_path,
        task="graph",
        dataset_setting=eval_setting or ",".join(datasets),
        load_model=load_graph_model,
        save_path=load_graph_model,
        source_csv=str(paths.graph_csv),
    )
    if oom_rows:
        print(
            "[MoE][AnyGraph][Eval][Graph] OOM datasets:",
            f"{_preview_values(oom_datasets)} -> {len(oom_rows)} result_status=OOM row(s)",
        )
    return True, oom_rows


def _append_anygraph_results_tsv(cfg, aggregated_rows, *, started_at) -> None:
    """Publish the aggregated per-dataset eval rows to the shared results table.

    Writes outputs/results/moe_anygraph.tsv alongside the sibling MoE methods
    (gmoe/mowst/graphmore).  AnyGraph evaluates many held-out datasets per run,
    so unlike the single-model siblings it emits one row per (task, dataset,
    split) instead of a single aggregated row; the per-route eval CSVs and the
    merged outputs/anygraph/anygraph_report.csv are still written as before.
    """
    if not aggregated_rows:
        return
    written = append_workflow_result_rows(
        cfg=cfg,
        workflow="moe_anygraph",
        rows=aggregated_rows,
        started_at=started_at,
        ended_at=datetime.now().astimezone(),
    )
    if written:
        table_path = result_table_path(cfg, "moe_anygraph")
        print(f"[MoE][AnyGraph][Result] wrote {written} row(s) -> {table_path}")


def _run_eval_step(
    cfg,
    *,
    paths: _AnyGraphPaths,
    index_state: _AnyGraphIndexState,
    any_cfg,
    conversion_cfg,
    train_cfg,
    eval_cfg,
    prediction_cfg,
) -> None:
    _print_header("[MoE][AnyGraph][Step 3] eval", char="=")
    eval_started_at = datetime.now().astimezone()
    eval_mode = _normalize_mode(getattr(eval_cfg, "mode", "all"), cfg_key="moe.anygraph.eval.mode")
    eval_device = str(_safe_int(getattr(cfg, "device", 0), 0))
    seed_scope = _resolve_cfg_seeds(cfg)
    shared_extra = _with_seed_arg(_as_token_list(getattr(any_cfg, "extra_args", [])), cfg)
    ran_eval_link = False
    ran_eval_node = False
    ran_eval_graph = False
    graph_oom_rows: List[Dict[str, str]] = []

    eval_has_scope = _stage_has_scope(eval_cfg, fallback_cfg=train_cfg)
    need_link_node = _mode_runs_route(eval_mode, "link") or _mode_runs_route(eval_mode, "node")
    stage_selection = None
    train_stage_selection = None
    link_available = False
    node_available = False
    if need_link_node:
        stage_selection = _resolve_stage_selection(
            cfg,
            eval_cfg,
            stage_cfg_key="moe.anygraph.eval",
            mode=eval_mode,
            index_state=index_state,
            fallback_cfg=train_cfg,
        )
        # Derive the tag from train-side cfg fields when the caller supplies
        # them; the checkpoint is a product of training, so for cross-dataset
        # eval the tag must describe the training inputs, not the eval inputs.
        train_stage_selection = _resolve_stage_selection(
            cfg,
            train_cfg,
            stage_cfg_key="moe.anygraph.train",
            mode=eval_mode,
            index_state=index_state,
            fallback_cfg=None,
        )
        link_available, node_available = _link_node_availability(stage_selection, index_state, eval_has_scope)
        _print_stage_info(
            "Eval",
            [
                ("mode", eval_mode),
                ("device", eval_device),
                ("seed_scope", _preview_values([str(seed) for seed in seed_scope])),
                ("link_datasets", _route_display(eval_mode, "link", stage_selection, index_state, eval_has_scope, link_available)),
                ("node_datasets", _route_display(eval_mode, "node", stage_selection, index_state, eval_has_scope, node_available)),
                ("link_csv", str(paths.link_csv)),
                ("node_csv", str(paths.node_csv)),
                ("report_csv", str(paths.report_csv)),
            ],
        )

    if _route_should_run(eval_mode, "link", available=link_available):
        link_cfg = prediction_cfg.link
        load_link_model = str(getattr(eval_cfg, "load_link_model", "") or "").strip()
        if not load_link_model:
            load_link_model = _resolve_eval_route_save_path(
                cfg,
                route="link",
                explicit_value=getattr(train_cfg, "save_link_path", ""),
                train_stage_selection=train_stage_selection,
                eval_stage_selection=stage_selection,
                seeds=seed_scope,
            )
        _require_checkpoint_tag(
            load_link_model,
            models_dir=ANYGRAPH_MODELS_DIR,
            history_dir=ANYGRAPH_HISTORY_DIR,
            route_name="link",
            cfg_key="moe.anygraph.eval.load_link_model",
        )
        link_args = [
            "--data_root",
            str((paths.out_root / "link").resolve()),
            "--result_csv",
            str(paths.link_csv),
            "--save_path",
            load_link_model,
            "--load_model",
            load_link_model,
            "--gpu",
            eval_device,
            "--epoch",
            "0",
            "--tst_epoch",
            str(_safe_int(getattr(link_cfg, "tst_epoch", 1), 1)),
            "--topk",
            str(_safe_int(getattr(link_cfg, "topk", 20), 20)),
            "--eval_protocol",
            str(getattr(link_cfg, "eval_protocol", "agae")),
            "--edge_eval_threshold_mode",
            str(getattr(link_cfg, "edge_eval_threshold_mode", "val_best_acc")),
            "--edge_eval_payload_name",
            str(getattr(conversion_cfg, "edge_eval_payload_name", "agae_edge_eval_payload.pt")),
            "--edge_eval_repeat_times",
            str(_safe_int(getattr(link_cfg, "edge_eval_repeat_times", 5), 5)),
        ]
        _extend_dataset_args(
            link_args,
            dataset_setting=stage_selection.link_setting,
            datasets=stage_selection.link_datasets or index_state.link_datasets,
            stage_cfg_key="moe.anygraph.eval",
            task_name="link",
        )
        _print_stage_info(
            "Eval",
            [
                ("link_action", "evaluate"),
                ("link_load_model", load_link_model),
            ],
        )
        link_args.extend(shared_extra)
        _run_module("src.moe.anygraph.link", link_args, run_anygraph_link_cli)
        ran_eval_link = True

    if _route_should_run(eval_mode, "node", available=node_available):
        node_cfg = prediction_cfg.node
        load_node_model = str(getattr(eval_cfg, "load_node_model", "") or "").strip()
        if not load_node_model:
            load_node_model = _resolve_eval_route_save_path(
                cfg,
                route="node",
                explicit_value=getattr(train_cfg, "save_node_path", ""),
                train_stage_selection=train_stage_selection,
                eval_stage_selection=stage_selection,
                seeds=seed_scope,
            )
        _require_checkpoint_tag(
            load_node_model,
            models_dir=ANYGRAPH_MODELS_DIR,
            history_dir=ANYGRAPH_HISTORY_DIR,
            route_name="node",
            cfg_key="moe.anygraph.eval.load_node_model",
        )
        node_args = [
            "--data_root",
            str((paths.out_root / "node").resolve()),
            "--result_csv",
            str(paths.node_csv),
            "--save_path",
            load_node_model,
            "--load_model",
            load_node_model,
            "--gpu",
            eval_device,
            "--epoch",
            "0",
            "--tst_epoch",
            str(_safe_int(getattr(node_cfg, "tst_epoch", 1), 1)),
            "--assignment",
            str(getattr(node_cfg, "assignment", "top1")),
        ]
        _extend_dataset_args(
            node_args,
            dataset_setting=stage_selection.node_setting,
            datasets=stage_selection.node_datasets or index_state.node_datasets,
            stage_cfg_key="moe.anygraph.eval",
            task_name="node",
        )
        _print_stage_info(
            "Eval",
            [
                ("node_action", "evaluate"),
                ("node_load_model", load_node_model),
            ],
        )
        node_args.extend(shared_extra)
        _run_module("src.moe.anygraph.node", node_args, run_anygraph_node_cli)
        ran_eval_node = True

    if _mode_runs_route(eval_mode, "graph"):
        ran_eval_graph, graph_oom_rows = _run_graph_eval_route(
            cfg,
            paths=paths,
            index_state=index_state,
            any_cfg=any_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            prediction_cfg=prediction_cfg,
            eval_device=eval_device,
            mode=eval_mode,
        )

    if not ran_eval_link and not ran_eval_node and not ran_eval_graph:
        raise ValueError("No evaluation route was executed (no link/node/graph datasets matched).")

    # Aggregate every route that ran (link, node and graph) into the merged
    # report, then publish the same per-dataset rows to the shared results
    # table outputs/results/moe_anygraph.tsv, mirroring the sibling MoE methods.
    if ran_eval_link or ran_eval_node or ran_eval_graph:
        report_link_csv = paths.link_csv if ran_eval_link else paths.outputs_dir / "_missing_link_eval.csv"
        report_node_csv = paths.node_csv if ran_eval_node else paths.outputs_dir / "_missing_node_eval.csv"
        report_graph_csv = paths.graph_csv if ran_eval_graph else paths.outputs_dir / "_missing_graph_eval.csv"
        for path in (report_link_csv, report_node_csv, report_graph_csv):
            if path.exists() and path.name.startswith("_missing_"):
                path.unlink()

        report_args = [
            "--link_csv",
            str(report_link_csv),
            "--node_csv",
            str(report_node_csv),
            "--graph_csv",
            str(report_graph_csv),
            "--out_csv",
            str(paths.report_csv),
            "--index_json",
            str(paths.index_path),
        ]
        print(f"[MoE][AnyGraph][Run][in-process] {_format_module_cmd('src.moe.anygraph.report', report_args)}")
        aggregated_rows = build_agae_eval_report(
            report_link_csv,
            report_node_csv,
            paths.report_csv,
            paths.index_path,
            graph_csv=report_graph_csv,
        )
        # Datasets skipped on CUDA OOM still get a row (result_status=OOM,
        # -1 metrics), mirroring the finetune convention.
        _append_anygraph_results_tsv(cfg, aggregated_rows + graph_oom_rows, started_at=eval_started_at)
    print("[MoE][AnyGraph][Eval] status: completed")


def run_anygraph(cfg) -> int:
    any_cfg = cfg.moe.anygraph
    execution_cfg = any_cfg.execution
    paths_cfg = any_cfg.paths
    conversion_cfg = any_cfg.conversion
    train_cfg = any_cfg.train
    eval_cfg = any_cfg.eval
    prediction_cfg = any_cfg.prediction
    output_cfg = any_cfg.output

    step = str(getattr(execution_cfg, "step", "all") or "all").strip().lower()
    if step not in {"conversion", "train", "eval", "all"}:
        raise ValueError(
            f"Unsupported moe.anygraph.execution.step='{step}'. Use conversion/train/eval/all."
        )
    planned_steps = ["conversion", "train", "eval"] if step == "all" else [step]

    conversion_task = str(getattr(conversion_cfg, "task", "auto") or "auto").strip().lower()
    if "conversion" in planned_steps and conversion_task not in {"auto", "all", "node", "link", "graph"}:
        raise ValueError(
            f"Unsupported moe.anygraph.conversion.task='{conversion_task}'. Use auto/all/node/link/graph."
        )

    train_mode = _normalize_mode(getattr(train_cfg, "mode", "all"), cfg_key="moe.anygraph.train.mode")
    eval_mode = _normalize_mode(getattr(eval_cfg, "mode", "all"), cfg_key="moe.anygraph.eval.mode")
    paths = _resolve_runtime_paths(paths_cfg, output_cfg)

    _print_header(f"[MoE][AnyGraph] pipeline_steps={','.join(planned_steps)}", char="=")

    if "conversion" in planned_steps:
        _run_conversion_step(cfg, paths, conversion_cfg, conversion_task)

    if step == "conversion":
        return 0

    index_state = _load_index_state(paths.index_path)

    _ROUTE_MAIN = {
        "link": ANYGRAPH_LINK_MAIN,
        "node": ANYGRAPH_NODE_MAIN,
        "graph": ANYGRAPH_GRAPH_MAIN,
    }
    for step_name, step_mode in (("train", train_mode), ("eval", eval_mode)):
        if step_name not in planned_steps:
            continue
        for route, main_path in _ROUTE_MAIN.items():
            if _mode_runs_route(step_mode, route):
                ensure_anygraph_runtime_files_exist(main_path)
    ANYGRAPH_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ANYGRAPH_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    if "train" in planned_steps:
        _run_train_step(
            cfg,
            paths=paths,
            index_state=index_state,
            any_cfg=any_cfg,
            conversion_cfg=conversion_cfg,
            train_cfg=train_cfg,
            prediction_cfg=prediction_cfg,
        )

    if "eval" in planned_steps:
        _run_eval_step(
            cfg,
            paths=paths,
            index_state=index_state,
            any_cfg=any_cfg,
            conversion_cfg=conversion_cfg,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            prediction_cfg=prediction_cfg,
        )

    _print_header("[MoE][AnyGraph] completed", char="=")
    print(f"[MoE][AnyGraph] conversion_index: {paths.index_path}")
    print(f"[MoE][AnyGraph] link_csv: {paths.link_csv}")
    print(f"[MoE][AnyGraph] node_csv: {paths.node_csv}")
    print(f"[MoE][AnyGraph] graph_csv: {paths.graph_csv}")
    print(f"[MoE][AnyGraph] report_csv: {paths.report_csv}")
    return 0
