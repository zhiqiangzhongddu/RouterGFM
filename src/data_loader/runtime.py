"""Runtime orchestration for the `run_data_preparation.py` workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

from src.data_loader.dataset_prepare import prepare_datasets, read_datasets
from src.data_loader.datasets import infer_task_level
from src.data_loader.summary import run_data_summary
from src.utils.run_helpers import resolve_seeds

TargetSelection = Optional[Union[str, list[str]]]
SplitTuple = tuple[float, float, float]


@dataclass(frozen=True)
class _PrepStage:
    """One dataset-preparation stage with its execution parameters."""

    title: str
    datasets: Union[str, list[str]]
    task_levels: tuple[str, ...]
    induced: bool
    use_infer: bool


def _print_stage_header(title: str, char: str = "=") -> None:
    line = char * 72
    print(line)
    print(title)
    print(line)


def parse_target_value(raw: str) -> Union[str, list[str]]:
    """Parse CLI target value like `[cora,actor]`, `cora,actor`, or `path/to/list.tsv`."""
    text = str(raw).strip()
    if not text:
        return []

    is_bracket_list = text.startswith("[") and text.endswith("]")
    if is_bracket_list:
        text = text[1:-1]

    if "," not in text:
        single = text.strip().strip("'\"")
        if is_bracket_list:
            return [single] if single else []
        return single

    items: list[str] = []
    for part in text.split(","):
        cleaned = part.strip().strip("'\"")
        if cleaned:
            items.append(cleaned)
    return items


def _looks_like_dataset_list_path(value: str) -> bool:
    """Treat path-like strings as dataset-list files instead of singleton dataset names."""
    text = str(value).strip().strip("'\"")
    if not text:
        return False

    path = Path(text).expanduser()
    return (
        path.exists()
        or text.startswith((".", "/", "~"))
        or "/" in text
        or "\\" in text
        or path.suffix.lower() in {".tsv", ".txt", ".csv", ".list"}
    )


def normalize_targets(value) -> TargetSelection:
    """Normalize target input into either a list of dataset names or a dataset-list path."""
    if value is None:
        return None

    if isinstance(value, (list, tuple, set)):
        names = [str(item).strip().strip("'\"") for item in value if str(item).strip().strip("'\"")]
        return names or None

    text = str(value).strip()
    if not text:
        return None

    parsed = parse_target_value(text) if "," in text or (text.startswith("[") and text.endswith("]")) else text
    if isinstance(parsed, list):
        return parsed or None

    cleaned = str(parsed).strip().strip("'\"")
    if not cleaned:
        return None
    if _looks_like_dataset_list_path(cleaned):
        return cleaned
    return [cleaned]


def _same_path(path_a: str, path_b: str) -> bool:
    try:
        return Path(path_a).resolve() == Path(path_b).resolve()
    except Exception:
        return str(path_a) == str(path_b)


def _coerce_split_tuple(split_def) -> SplitTuple:
    if not isinstance(split_def, (list, tuple)) or len(split_def) < 3:
        raise ValueError(f"Invalid split definition: {split_def}")
    return (float(split_def[0]), float(split_def[1]), float(split_def[2]))


def _normalize_edge_split(split_def) -> SplitTuple:
    train_pos, val_pos, test_pos = _coerce_split_tuple(split_def)
    if min(train_pos, val_pos, test_pos) < 0.0:
        raise ValueError(f"Invalid edge split definition {split_def}: all values must be >= 0.")

    total_pos = train_pos + val_pos + test_pos
    if total_pos > 1.0 + 1e-6:
        raise ValueError(
            f"Invalid edge split definition {split_def}: train/val/test positive ratios must sum to <= 1.0."
        )
    return (train_pos, val_pos, test_pos)


def _split_defs_for_task(cfg, task_level: str) -> list[SplitTuple]:
    dp_cfg = cfg.data_preparation
    if task_level == "node":
        return [_coerce_split_tuple(split_def) for split_def in dp_cfg.node_task_splits]
    if task_level == "graph":
        return [_coerce_split_tuple(split_def) for split_def in dp_cfg.graph_task_splits]
    if task_level == "edge":
        return [_normalize_edge_split(split_def) for split_def in dp_cfg.edge_task_splits]
    return []


def _build_split_generation_plan(cfg) -> tuple[dict[str, list[SplitTuple]], list[int]]:
    ds_cfg = cfg.data_preparation.dataset
    num_splits = int(ds_cfg.num_splits or 0)
    if num_splits <= 0:
        num_splits = 5

    split_defs_by_task = {
        "node": _split_defs_for_task(cfg, "node"),
        "graph": _split_defs_for_task(cfg, "graph"),
        "edge": _split_defs_for_task(cfg, "edge"),
    }
    split_seeds = resolve_seeds(cfg, requested_count=num_splits)
    return split_defs_by_task, split_seeds


def _resolve_stage_plan(cfg, targets: TargetSelection) -> list[_PrepStage]:
    """Create stage plan based on selected targets and configured dataset lists."""
    dp_cfg = cfg.data_preparation
    ds_cfg = dp_cfg.dataset

    node_list = ds_cfg.available_node_datasets
    graph_list = ds_cfg.available_graph_datasets
    include_edge_level = bool(dp_cfg.generate_edge_level)
    node_task_levels = ("node", "edge") if include_edge_level else ("node",)
    task_level_override = str(
        getattr(dp_cfg, "task_level_override", "") or ""
    ).strip().lower()
    if task_level_override and task_level_override not in {"node", "edge", "graph"}:
        raise ValueError(
            "data_preparation.task_level_override must be node, edge, graph, "
            f"or empty; got {task_level_override!r}."
        )

    if task_level_override:
        return [
            _PrepStage(
                title=f"[DataPrep] Dataset preparation ({task_level_override})",
                datasets=targets,
                task_levels=(task_level_override,),
                induced=bool(ds_cfg.induced),
                use_infer=False,
            )
        ]

    if isinstance(targets, list):
        return [
            _PrepStage(
                title="[DataPrep] Dataset preparation",
                datasets=targets,
                task_levels=("node", "edge", "graph"),
                induced=bool(ds_cfg.induced),
                use_infer=True,
            )
        ]

    if node_list and _same_path(targets, node_list):
        return [
            _PrepStage(
                title="[DataPrep] Dataset preparation (node/edge)",
                datasets=targets,
                task_levels=node_task_levels,
                induced=bool(ds_cfg.induced),
                use_infer=False,
            )
        ]

    if graph_list and _same_path(targets, graph_list):
        return [
            _PrepStage(
                title="[DataPrep] Dataset preparation (graph)",
                datasets=targets,
                task_levels=("graph",),
                induced=False,
                use_infer=False,
            )
        ]

    return [
        _PrepStage(
            title="[DataPrep] Dataset preparation",
            datasets=targets,
            task_levels=("node", "edge", "graph"),
            induced=bool(ds_cfg.induced),
            use_infer=True,
        )
    ]


def _run_prepare_stage(
    cfg,
    stage: _PrepStage,
    split_defs_by_task: dict[str, list[SplitTuple]],
    split_seeds: list[int],
) -> int:
    """Run one preparation stage using precomputed split plan."""
    _print_stage_header(stage.title, char="=")

    ds_cfg = cfg.data_preparation.dataset
    dp_cfg = cfg.data_preparation

    return prepare_datasets(
        tsv_path=stage.datasets,
        root=Path(ds_cfg.root),
        task_levels=stage.task_levels,
        feat_reduction=ds_cfg.feat_reduction,
        feat_reduction_dim=ds_cfg.feat_reduction_svd_dim,
        feature_svd_dir=ds_cfg.feature_svd_dir,
        induced=stage.induced,
        induced_min_size=ds_cfg.induced_min_size,
        induced_max_size=ds_cfg.induced_max_size,
        induced_max_hops=ds_cfg.induced_max_hops,
        edge_max_size=ds_cfg.edge_max_size,
        edge_level_max_num_nodes=ds_cfg.edge_level_max_num_nodes,
        induced_root=ds_cfg.induced_root,
        graph_filter_dir=ds_cfg.graph_filter_dir,
        subgraph_svd=ds_cfg.subgraph_svd,
        subgraph_svd_feat_dim=ds_cfg.subgraph_svd_feat_dim,
        subgraph_svd_struct_dim=ds_cfg.subgraph_svd_struct_dim,
        subgraph_svd_matrix=ds_cfg.subgraph_svd_matrix,
        subgraph_svd_dir=ds_cfg.subgraph_svd_dir,
        use_infer=stage.use_infer,
        split_root=ds_cfg.split_root,
        generate_edge_level=bool(dp_cfg.generate_edge_level),
        split_defs_by_task=split_defs_by_task,
        split_seeds=split_seeds,
        split_batch_size=int(dp_cfg.batch_size),
        split_num_workers=int(dp_cfg.num_workers),
    )


def _read_stage_dataset_names(stage: _PrepStage) -> list[str]:
    if isinstance(stage.datasets, list):
        return [str(name).strip() for name in stage.datasets if str(name).strip()]
    return [name.strip() for name in read_datasets(str(stage.datasets)) if name.strip()]


def _dedupe_names(names: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen = set()
    for raw_name in names:
        name = str(raw_name).strip()
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        deduped.append(name)
        seen.add(key)
    return deduped


def _collect_summary_dataset_names(stages: list[_PrepStage]) -> tuple[list[str], list[str]]:
    node_names: list[str] = []
    graph_names: list[str] = []

    for stage in stages:
        try:
            dataset_names = _read_stage_dataset_names(stage)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[DataPrep][Summary] Skip stage '{stage.title}': failed to resolve datasets ({exc})")
            continue
        if not dataset_names:
            continue

        has_node_or_edge = "node" in stage.task_levels or "edge" in stage.task_levels
        has_graph = "graph" in stage.task_levels

        if has_node_or_edge and not has_graph and not stage.use_infer:
            node_names.extend(dataset_names)
            continue
        if has_graph and not has_node_or_edge and not stage.use_infer:
            graph_names.extend(dataset_names)
            continue

        for name in dataset_names:
            inferred = infer_task_level(name)
            if inferred == "graph":
                graph_names.append(name)
            else:
                # Edge datasets share node-level storage/loading conventions here.
                node_names.append(name)

    return _dedupe_names(node_names), _dedupe_names(graph_names)


def run_data_preparation_runtime(cfg) -> int:
    """Main orchestrator used by `run_data_preparation.py`."""
    dp_cfg = cfg.data_preparation
    targets = normalize_targets(dp_cfg.target_datasets)

    if not targets:
        print(
            "[DataPrep][Error] `data_preparation.target_datasets` is required. "
            "Pass a dataset name like `cora`, a comma-separated list like `cora,actor`, "
            "or a dataset list file like `data/available_node_datasets.tsv`."
        )
        return 1

    # Shift split roots are an exclusive stage on top of prepared datasets:
    # run the standard preparation first, then again with shift.build / shift.verify.
    shift_cfg = dp_cfg.shift
    if bool(shift_cfg.build) or bool(shift_cfg.verify):
        from src.data_loader.shift_splits import run_shift_preparation

        return run_shift_preparation(cfg)

    stages = _resolve_stage_plan(cfg, targets)
    if not stages:
        print("[DataPrep] No datasets to prepare. Check dataset list configuration.")
        return 0

    try:
        split_defs_by_task, split_seeds = _build_split_generation_plan(cfg)
    except ValueError as exc:
        print(f"[DataPrep][Split] {exc}")
        return 1
    status_codes = [
        _run_prepare_stage(cfg, stage, split_defs_by_task=split_defs_by_task, split_seeds=split_seeds)
        for stage in stages
    ]

    node_summary_names, graph_summary_names = _collect_summary_dataset_names(stages)
    summary_status = 0
    if node_summary_names or graph_summary_names:
        _print_stage_header("[DataPrep] Dataset summary", char="=")
        summary_status = run_data_summary(
            cfg,
            node_names=node_summary_names,
            graph_names=graph_summary_names,
        )
    else:
        print("[DataPrep][Summary] No datasets selected for summary generation.")

    prep_status = 0 if all(code == 0 for code in status_codes) else 1
    return 0 if prep_status == 0 and summary_status == 0 else 1
