"""Expert pool (App. B.1): checkpoint catalog, frozen encoders, compatible pools."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.model import build_encoder_from_cfg
from src.utils.checkpoint import save_json_atomic
from src.utils.dataset_helpers import canonical_source_dataset_name, checkpoint_dataset_dir_name

from .checkpoint import checkpoint_input_dim, load_encoder_state_strict, restore_checkpoint_model_cfg
from .common import AppSpec, ExpertSpec, RouterPaths, is_same_source

# The layer-wise InfoGraph readout needs the encoder layer cache, which these
# backbones do not expose; their canonical InfoGraph run is the last-layer
# ("nolw") variant.
_CANONICAL_METHOD = {
    ("infograph", arch): "infograph-nolw" for arch in ("fagcn", "h2gcn", "nodeformer", "transformer")
}
_PREFERRED_SEED = 42

_WARM_START = re.compile(r"^ws[A-Za-z0-9.-]*-[0-9a-f]{8}_")
_STEM_TAIL = re.compile(
    r"^(?P<source>.+?)_task(?P<task_level>node|edge|graph)_induced(?P<induced>[01])"
    r"(?:_(?P<split>(?:split|fewshot)[0-9-]+))?"
    r"_(?P<arch>[a-z0-9]+)(?:-(?P<arch_variant>[^_]+))?"
    r"_h(?P<hidden_dim>\d+)_o(?P<out_dim>\d+)_l(?P<num_layers>\d+)_e(?P<epochs>\d+)"
    r"_lr(?P<lr>[^_]+)_bs(?P<batch_size>\d+)_seed(?P<seed>\d+)$"
)


def parse_checkpoint_stem(stem: str, objectives: Sequence[str], architectures: Sequence[str]) -> Optional[Dict[str, object]]:
    """Parse a pretrain run name (``build_pretrain_run_name_from_cfg``) into its fields.

    ``<objective>[-<variant>][_ws..]_<source>_task<lvl>_induced<0|1>[_split..]_<arch>[-<mv>]_h.._o.._l.._e.._lr.._bs.._seed..``.
    Objectives may contain ``_`` and are matched against the configured list
    (longest first). Returns ``None`` for names outside the configured grid.
    """
    for objective in sorted(objectives, key=len, reverse=True):
        if not stem.startswith(objective) or len(stem) <= len(objective) or stem[len(objective)] not in "_-":
            continue
        cut = stem.index("_", len(objective)) if "_" in stem[len(objective):] else -1
        if cut < 0:
            return None
        method, rest = stem[:cut], stem[cut + 1:]
        warm = _WARM_START.match(rest)
        if warm:
            rest = rest[warm.end():]
        match = _STEM_TAIL.match(rest)
        if match is None or match["arch"] not in architectures:
            return None
        return {
            "method": method,
            "objective": objective,
            "warm_start": bool(warm),
            "source": match["source"],
            "task_level": match["task_level"],
            "induced": match["induced"] == "1",
            "split": match["split"] or "",
            "architecture": match["arch"],
            "arch_variant": match["arch_variant"] or "",
            "seed": int(match["seed"]),
        }
    return None


def _preference(parsed: Dict[str, object], stem: str) -> Tuple:
    canonical = _CANONICAL_METHOD.get((parsed["objective"], parsed["architecture"]), parsed["objective"])
    seed = int(parsed["seed"])
    return (
        parsed["method"] != canonical,
        bool(parsed["arch_variant"]),
        bool(parsed["warm_start"]),
        seed != _PREFERRED_SEED,
        seed,
        stem,
    )


def _scan_catalog(cfg) -> List[ExpertSpec]:
    ecfg = cfg.moe.routergfm.experts
    root = Path(str(ecfg.checkpoint_root))
    objectives = [str(o) for o in ecfg.objectives]
    architectures = [str(a) for a in ecfg.architectures]
    best: Dict[Tuple[str, str, str], Tuple[Tuple, ExpertSpec]] = {}
    for source in ecfg.sources:
        source = str(source)
        source_dir = root / checkpoint_dataset_dir_name(source)
        if not source_dir.is_dir():
            continue
        for path in sorted(source_dir.glob("*.pt")):
            parsed = parse_checkpoint_stem(path.stem, objectives, architectures)
            if parsed is None:
                continue
            if canonical_source_dataset_name(parsed["source"]) != canonical_source_dataset_name(source):
                continue
            spec = ExpertSpec(
                expert_id=path.stem,
                architecture=str(parsed["architecture"]),
                objective=str(parsed["objective"]),
                objective_variant=str(parsed["method"]),
                source=source,
                source_task_level=str(parsed["task_level"]),
                checkpoint_path=str(path),
            )
            cell = (source, spec.objective, spec.architecture)
            rank = _preference(parsed, path.stem)
            if cell not in best or rank < best[cell][0]:
                best[cell] = (rank, spec)

    catalog: List[ExpertSpec] = []
    missing: List[str] = []
    for source in ecfg.sources:
        for objective in objectives:
            for arch in architectures:
                entry = best.get((str(source), objective, arch))
                if entry is None:
                    missing.append(f"{arch}/{objective}/{source}")
                else:
                    catalog.append(entry[1])
    if missing and bool(ecfg.strict):
        raise FileNotFoundError(
            f"{len(missing)} expert checkpoint(s) missing under {root} "
            f"(architecture/objective/source): {', '.join(missing)}"
        )
    return catalog


def _cached_catalog_matches(catalog: List[ExpertSpec], cfg) -> bool:
    ecfg = cfg.moe.routergfm.experts
    grid = [
        (str(s), str(o), str(a))
        for s in ecfg.sources for o in ecfg.objectives for a in ecfg.architectures
    ]
    order = {cell: i for i, cell in enumerate(grid)}
    cells = [(e.source, e.objective, e.architecture) for e in catalog]
    if any(cell not in order for cell in cells):
        return False
    positions = [order[cell] for cell in cells]
    if positions != sorted(set(positions)):
        return False
    return not bool(ecfg.strict) or len(cells) == len(grid)


def build_expert_catalog(cfg, *, refresh: bool = False) -> List[ExpertSpec]:
    """One checkpoint per (architecture, objective, source), ordered sources > objectives > architectures.

    Persisted to ``RouterPaths.catalog_file`` and reused unless ``refresh`` (or
    the cached list no longer fits the configured grid).
    """
    path = RouterPaths.from_cfg(cfg).catalog_file
    if path.is_file() and not refresh:
        with path.open("r", encoding="utf-8") as handle:
            cached = [ExpertSpec.from_dict(item) for item in json.load(handle)]
        if _cached_catalog_matches(cached, cfg):
            return cached
    catalog = _scan_catalog(cfg)
    save_json_atomic(str(path), [spec.to_dict() for spec in catalog])
    return catalog


def load_frozen_encoder(cfg, spec: ExpertSpec, device) -> Tuple[nn.Module, object]:
    """Rebuild an expert encoder from its checkpoint; frozen, eval mode, on ``device``.

    Returns ``(encoder, restored_cfg)``; ``restored_cfg.model`` holds the
    checkpoint's model config (``out_dim``, ``graph_pooling``, ``in_dim``).
    """
    payload = torch.load(spec.checkpoint_path, map_location="cpu")
    model_cfg = restore_checkpoint_model_cfg(cfg, payload, checkpoint_path=spec.checkpoint_path)
    in_dim = checkpoint_input_dim(payload, fallback=int(model_cfg.model.in_dim or 0))
    encoder = build_encoder_from_cfg(model_cfg, in_dim)
    load_encoder_state_strict(encoder, payload, checkpoint_path=spec.checkpoint_path)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder.to(device), model_cfg


def compatible_experts(app: AppSpec, catalog: Sequence[ExpertSpec], cfg) -> List[int]:
    """Indices of E_a: the catalog minus same-source experts when ``exclude_same_source``."""
    exclude = bool(cfg.moe.routergfm.experts.exclude_same_source)
    return [i for i, spec in enumerate(catalog) if not (exclude and is_same_source(app, spec))]


__all__ = [
    "build_expert_catalog",
    "compatible_experts",
    "load_frozen_encoder",
    "parse_checkpoint_stem",
]
