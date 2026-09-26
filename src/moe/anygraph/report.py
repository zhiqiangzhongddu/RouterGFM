"""Code-owned AnyGraph report merger."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .runtime import REPO_ROOT


DEFAULT_LINK_CSV = REPO_ROOT / "outputs" / "anygraph" / "anygraph_link_eval.csv"
DEFAULT_NODE_CSV = REPO_ROOT / "outputs" / "anygraph" / "anygraph_node_eval.csv"
DEFAULT_GRAPH_CSV = REPO_ROOT / "outputs" / "anygraph" / "anygraph_graph_eval.csv"
DEFAULT_OUT_CSV = REPO_ROOT / "outputs" / "anygraph" / "anygraph_report.csv"
DEFAULT_INDEX_JSON = REPO_ROOT / "data" / "anygraph_data" / "conversion_index.json"
_DATASET_ALIAS_RE = re.compile(r"^(?P<source>.+)_seed(?P<seed>-?\d+)_(?P<split_tag>.+)$")
_REPORT_FIELDNAMES = [
    "method",
    "task",
    "dataset",
    "fixed_split",
    "split_tag",
    "seed_count",
    "seeds",
    "metric_1_name",
    "metric_1_mean",
    "metric_1_std",
    "metric_2_name",
    "metric_2_mean",
    "metric_2_std",
    "repeat_times",
    "tst_num",
    "dataset_setting",
    "load_model",
    "save_path",
    "source_csv",
]
_REPORT_KEY_FIELDS = [
    "method",
    "task",
    "dataset",
    "fixed_split",
    "split_tag",
    "dataset_setting",
    "load_model",
    "save_path",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_index_lookup(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse conversion index JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid conversion index payload (expected object): {path}")

    lookup: Dict[str, Dict[str, str]] = {}
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        return lookup
    for row in rows:
        if not isinstance(row, dict):
            continue
        dataset = str(row.get("dataset", "") or "").strip()
        if not dataset:
            continue
        raw_seed = row.get("seed", "")
        lookup[dataset] = {
            "source_dataset": str(row.get("source_dataset", "") or "").strip(),
            "fixed_split": _format_fixed_split(row.get("split")),
            "split_tag": str(row.get("split_tag", "") or "").strip(),
            # Integer seed 0 is valid metadata, not an empty value.
            "seed": str(raw_seed if raw_seed is not None else "").strip(),
        }
    return lookup


def _format_fixed_split(split) -> str:
    if not isinstance(split, (list, tuple)) or len(split) != 3:
        return ""
    parts: List[str] = []
    for raw_value in split:
        value = float(raw_value)
        if abs(value - round(value)) < 1e-8:
            if abs(value) <= 1.0:
                parts.append(f"{value:.1f}")
            else:
                parts.append(str(int(round(value))))
        else:
            parts.append(f"{value:g}")
    return f"({', '.join(parts)})"


def _fallback_dataset_meta(dataset: str) -> Dict[str, str]:
    match = _DATASET_ALIAS_RE.match(dataset)
    if not match:
        return {
            "source_dataset": dataset,
            "fixed_split": "",
            "split_tag": "",
            "seed": "",
        }
    split_tag = match.group("split_tag")
    return {
        "source_dataset": match.group("source"),
        "fixed_split": _fixed_split_from_split_tag(split_tag),
        "split_tag": split_tag,
        "seed": match.group("seed"),
    }


def _dataset_meta(dataset: str, lookup: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    meta = lookup.get(dataset)
    if meta is not None:
        if not meta.get("fixed_split") and meta.get("split_tag"):
            meta = dict(meta)
            meta["fixed_split"] = _fixed_split_from_split_tag(str(meta.get("split_tag", "")))
        return meta
    return _fallback_dataset_meta(dataset)


def _fixed_split_from_split_tag(split_tag: str) -> str:
    raw = str(split_tag or "").strip()
    if not raw:
        return ""
    if raw.startswith("split-"):
        values = raw[len("split-") :].split("-")
        if len(values) == 3:
            return _format_fixed_split([float(value) / 100.0 for value in values])
    if raw.startswith("fewshot"):
        values = raw[len("fewshot") :].split("-")
        if len(values) == 3:
            return _format_fixed_split([float(values[0]), float(values[1]) / 100.0, float(values[2]) / 100.0])
    return raw


def _as_float_strict(raw: Optional[str], *, field: str, source: Path, dataset: str) -> float:
    if raw is None or raw == "":
        raise ValueError(f"Missing required field '{field}' in {source}: dataset={dataset}")
    try:
        return float(raw)
    except Exception as exc:
        raise ValueError(
            f"Invalid numeric value for field '{field}' in {source}: dataset={dataset}, value={raw!r}"
        ) from exc


def _as_text(raw: Optional[str]) -> str:
    return str(raw or "").strip()


def _standardize_link_rows(
    rows: List[Dict[str, str]],
    source: Path,
    dataset_lookup: Dict[str, Dict[str, str]],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        dataset = _as_text(row.get("dataset"))
        if not dataset or dataset == "__OVERALL__":
            continue
        meta = _dataset_meta(dataset, dataset_lookup)
        # New link eval CSVs carry auc_mean (AUC is the LP metric used in the
        # paper tables); older CSVs predate it, so fall back to acc/loss there.
        has_auc = _as_text(row.get("auc_mean")) != ""
        if has_auc:
            metric_1 = ("auc", _as_float_strict(row.get("auc_mean"), field="auc_mean", source=source, dataset=dataset))
            metric_2 = ("acc", _as_float_strict(row.get("acc_mean"), field="acc_mean", source=source, dataset=dataset))
        else:
            metric_1 = ("acc", _as_float_strict(row.get("acc_mean"), field="acc_mean", source=source, dataset=dataset))
            metric_2 = ("loss", _as_float_strict(row.get("loss_mean"), field="loss_mean", source=source, dataset=dataset))
        out.append(
            {
                "method": "anygraph",
                "task": "link",
                "dataset": dataset,
                "source_dataset": meta.get("source_dataset", dataset),
                "fixed_split": meta.get("fixed_split", ""),
                "split_tag": meta.get("split_tag", ""),
                "seed": meta.get("seed", ""),
                "metric_1_name": metric_1[0],
                "metric_1_mean": metric_1[1],
                "metric_2_name": metric_2[0],
                "metric_2_mean": metric_2[1],
                "repeat_times": _as_text(row.get("repeat_times")),
                "tst_num": _as_text(row.get("tst_num")),
                "dataset_setting": _as_text(row.get("dataset_setting")),
                "load_model": _as_text(row.get("load_model")),
                "save_path": _as_text(row.get("save_path")),
                "source_csv": str(source),
            }
        )
    return out


def _standardize_node_rows(
    rows: List[Dict[str, str]],
    source: Path,
    dataset_lookup: Dict[str, Dict[str, str]],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        dataset = _as_text(row.get("dataset"))
        if not dataset or dataset == "__OVERALL__":
            continue
        meta = _dataset_meta(dataset, dataset_lookup)
        out.append(
            {
                "method": "anygraph",
                "task": "node",
                "dataset": dataset,
                "source_dataset": meta.get("source_dataset", dataset),
                "fixed_split": meta.get("fixed_split", ""),
                "split_tag": meta.get("split_tag", ""),
                "seed": meta.get("seed", ""),
                "metric_1_name": "acc",
                "metric_1_mean": _as_float_strict(row.get("acc_mean"), field="acc_mean", source=source, dataset=dataset),
                "metric_2_name": "f1_macro",
                "metric_2_mean": _as_float_strict(row.get("f1_mean"), field="f1_mean", source=source, dataset=dataset),
                "repeat_times": _as_text(row.get("repeat_times")),
                "tst_num": _as_text(row.get("tst_num")),
                "dataset_setting": _as_text(row.get("dataset_setting")),
                "load_model": _as_text(row.get("load_model")),
                "save_path": _as_text(row.get("save_path")),
                "source_csv": str(source),
            }
        )
    return out


def _standardize_graph_rows(
    rows: List[Dict[str, str]],
    source: Path,
    dataset_lookup: Dict[str, Dict[str, str]],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        dataset = _as_text(row.get("dataset"))
        if not dataset or dataset == "__OVERALL__":
            continue
        meta = _dataset_meta(dataset, dataset_lookup)
        # Graph eval emits generic primary/secondary metric columns (the metric
        # name varies by task family, e.g. acc/f1 for classification, rmse/mae
        # for regression), unlike link/node which hard-code acc + loss/f1.
        out.append(
            {
                "method": "anygraph",
                "task": "graph",
                "dataset": dataset,
                "source_dataset": meta.get("source_dataset", dataset),
                "fixed_split": meta.get("fixed_split", ""),
                "split_tag": meta.get("split_tag", ""),
                "seed": meta.get("seed", ""),
                "metric_1_name": _as_text(row.get("primary_metric")) or "primary",
                "metric_1_mean": _as_float_strict(
                    row.get("primary_mean"), field="primary_mean", source=source, dataset=dataset
                ),
                "metric_2_name": _as_text(row.get("secondary_metric")) or "secondary",
                "metric_2_mean": _as_float_strict(
                    row.get("secondary_mean"), field="secondary_mean", source=source, dataset=dataset
                ),
                "repeat_times": _as_text(row.get("repeat_times")),
                "tst_num": _as_text(row.get("tst_num")),
                "dataset_setting": _as_text(row.get("dataset_setting")),
                "load_model": _as_text(row.get("load_model")),
                "save_path": _as_text(row.get("save_path")),
                "source_csv": str(source),
            }
        )
    return out


def _mean_std(values: List[float]) -> Tuple[float, float]:
    mean_value = statistics.mean(values)
    std_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_value, std_value


def _summarize_text_values(values: List[str]) -> str:
    unique = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        unique.append(text)
        seen.add(text)
    return ",".join(unique)


def _aggregate_rows(rows: List[Dict[str, object]]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str, str, str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["task"]),
            str(row["source_dataset"]),
            str(row["fixed_split"]),
            str(row["split_tag"]),
            str(row["dataset_setting"]),
            str(row["load_model"]),
            str(row["save_path"]),
        )
        grouped[key].append(row)

    out: List[Dict[str, str]] = []
    for key in sorted(grouped.keys()):
        items = grouped[key]
        metric_1_values = [float(item["metric_1_mean"]) for item in items]
        metric_2_values = [float(item["metric_2_mean"]) for item in items]
        metric_1_mean, metric_1_std = _mean_std(metric_1_values)
        metric_2_mean, metric_2_std = _mean_std(metric_2_values)
        seed_values = _summarize_text_values([str(item.get("seed", "")) for item in items])
        repeat_times = _summarize_text_values([str(item.get("repeat_times", "")) for item in items])
        tst_num = _summarize_text_values([str(item.get("tst_num", "")) for item in items])
        dataset_setting = _summarize_text_values([str(item.get("dataset_setting", "")) for item in items])
        source_csv = _summarize_text_values([str(item.get("source_csv", "")) for item in items])

        out.append(
            {
                "method": "anygraph",
                "task": str(items[0]["task"]),
                "dataset": str(items[0]["source_dataset"]),
                "fixed_split": str(items[0]["fixed_split"]),
                "split_tag": str(items[0]["split_tag"]),
                # Count input rows, not unique seed strings -- duplicate seeds
                # must not under-report the sample size backing mean/std.
                "seed_count": str(len(items)),
                "seeds": seed_values,
                "metric_1_name": str(items[0]["metric_1_name"]),
                "metric_1_mean": f"{metric_1_mean:.8f}",
                "metric_1_std": f"{metric_1_std:.8f}",
                "metric_2_name": str(items[0]["metric_2_name"]),
                "metric_2_mean": f"{metric_2_mean:.8f}",
                "metric_2_std": f"{metric_2_std:.8f}",
                "repeat_times": repeat_times,
                "tst_num": tst_num,
                "dataset_setting": dataset_setting,
                "load_model": str(items[0]["load_model"]),
                "save_path": str(items[0]["save_path"]),
                "source_csv": source_csv,
            }
        )
    return out


def build_agae_oom_rows(
    oom_datasets: Sequence[str],
    index_json: Optional[Path],
    *,
    task: str,
    dataset_setting: str = "",
    load_model: str = "",
    save_path: str = "",
    source_csv: str = "",
) -> List[Dict[str, str]]:
    """Shape ``result_status=OOM`` rows for datasets an eval route skipped.

    Mirrors the finetune convention (``result_status`` column with ``OOM`` and
    ``-1`` metrics) and the :func:`_aggregate_rows` schema so the rows can be
    appended to ``outputs/results/moe_anygraph.tsv`` alongside the normal
    aggregated rows. ``oom_datasets`` holds converted dataset names (e.g.
    ``mnist_seed42_split-80-10-10``); markers for the same source dataset and
    split are grouped across seeds into one row.
    """
    dataset_lookup = _load_index_lookup(index_json)
    grouped: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for raw in oom_datasets:
        dataset = _as_text(raw)
        if not dataset:
            continue
        meta = _dataset_meta(dataset, dataset_lookup)
        key = (
            _as_text(meta.get("source_dataset")) or dataset,
            _as_text(meta.get("fixed_split")),
            _as_text(meta.get("split_tag")),
        )
        grouped[key].append(_as_text(meta.get("seed")))
    out: List[Dict[str, str]] = []
    for key in sorted(grouped.keys()):
        source_dataset, fixed_split, split_tag = key
        seeds = grouped[key]
        out.append(
            {
                "method": "anygraph",
                "task": task,
                "dataset": source_dataset,
                "fixed_split": fixed_split,
                "split_tag": split_tag,
                "seed_count": str(len(seeds)),
                "seeds": _summarize_text_values(seeds),
                "metric_1_name": "",
                "metric_1_mean": "-1",
                "metric_1_std": "-1",
                "metric_2_name": "",
                "metric_2_mean": "-1",
                "metric_2_std": "-1",
                "repeat_times": "",
                "tst_num": "",
                "dataset_setting": dataset_setting,
                "load_model": load_model,
                "save_path": save_path,
                "source_csv": source_csv,
                "result_status": "OOM",
            }
        )
    return out


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_REPORT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_existing_report_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for row in rows:
        normalized.append({field: _as_text(row.get(field)) for field in _REPORT_FIELDNAMES})
    return normalized


def _report_row_key(row: Dict[str, str]) -> Tuple[str, ...]:
    return tuple(_as_text(row.get(field)) for field in _REPORT_KEY_FIELDS)


def _merge_report_rows(existing_rows: List[Dict[str, str]], new_rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], int, int]:
    normalized_existing = _normalize_existing_report_rows(existing_rows)
    normalized_new = _normalize_existing_report_rows(new_rows)

    existing_keys = {_report_row_key(row) for row in normalized_existing}
    new_keys = {_report_row_key(row) for row in normalized_new}
    kept_rows = [row for row in normalized_existing if _report_row_key(row) not in new_keys]

    replaced_count = len(existing_keys & new_keys)
    inserted_count = len(new_keys - existing_keys)
    return kept_rows + normalized_new, inserted_count, replaced_count


def _print_summary(rows: List[Dict[str, str]]) -> None:
    for row in rows:
        print(
            "[MoE][AnyGraph][Report]",
            f"task={row['task']}",
            f"dataset={row['dataset']}",
            f"fixed_split={row['fixed_split'] or row['split_tag'] or '<all>'}",
            f"seeds={row['seeds'] or '<none>'}",
            f"{row['metric_1_name']}: mean={row['metric_1_mean']} std={row['metric_1_std']}",
            f"{row['metric_2_name']}: mean={row['metric_2_mean']} std={row['metric_2_std']}",
        )


def build_agae_eval_report(
    link_csv: Path,
    node_csv: Path,
    out_csv: Path,
    index_json: Optional[Path] = None,
    graph_csv: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Merge the per-route eval CSVs into the aggregated report.

    Reads the link, node and (when provided) graph eval CSVs, standardizes and
    aggregates them across seeds, merges into ``out_csv`` (replace-on-match),
    and returns the freshly aggregated per-(task, dataset, split) rows so the
    caller can also publish them to ``outputs/results/moe_anygraph.tsv``.
    Missing CSV paths are treated as empty, so passing ``graph_csv`` is safe
    even when the graph route did not run.
    """
    dataset_lookup = _load_index_lookup(index_json)
    rows: List[Dict[str, object]] = []
    rows.extend(_standardize_link_rows(_read_csv(link_csv), link_csv, dataset_lookup))
    rows.extend(_standardize_node_rows(_read_csv(node_csv), node_csv, dataset_lookup))
    if graph_csv is not None:
        rows.extend(_standardize_graph_rows(_read_csv(graph_csv), graph_csv, dataset_lookup))
    aggregated_rows = _aggregate_rows(rows)
    merged_rows, inserted_count, replaced_count = _merge_report_rows(_read_csv(out_csv), aggregated_rows)
    _write_csv(out_csv, merged_rows)
    print(
        f"[MoE][AnyGraph][Report] new_rows={len(aggregated_rows)} "
        f"inserted={inserted_count} replaced={replaced_count} total_rows={len(merged_rows)} out={out_csv}"
    )
    _print_summary(aggregated_rows)
    return aggregated_rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build merged AnyGraph report.")
    parser.add_argument("--link_csv", default=str(DEFAULT_LINK_CSV))
    parser.add_argument("--node_csv", default=str(DEFAULT_NODE_CSV))
    parser.add_argument("--graph_csv", default=str(DEFAULT_GRAPH_CSV))
    parser.add_argument("--out_csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--index_json", default=str(DEFAULT_INDEX_JSON))
    args = parser.parse_args(argv)
    index_json = Path(args.index_json).resolve() if str(args.index_json).strip() else None
    graph_csv = Path(args.graph_csv).resolve() if str(args.graph_csv).strip() else None
    build_agae_eval_report(
        Path(args.link_csv).resolve(),
        Path(args.node_csv).resolve(),
        Path(args.out_csv).resolve(),
        index_json,
        graph_csv=graph_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
