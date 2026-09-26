"""Shared AnyGraph runtime paths and environment helpers."""

from __future__ import annotations

import os
from pathlib import Path

from src.utils import PROJECT_ROOT, project_path


REPO_ROOT = PROJECT_ROOT
ANYGRAPH_PACKAGE_ROOT = Path(__file__).resolve().parent
ANYGRAPH_SRC_ROOT = ANYGRAPH_PACKAGE_ROOT / "src"
ANYGRAPH_NODE_SRC_ROOT = ANYGRAPH_SRC_ROOT / "node_classification"
ANYGRAPH_GRAPH_SRC_ROOT = ANYGRAPH_SRC_ROOT / "graph_prediction"
ANYGRAPH_LINK_MAIN = ANYGRAPH_SRC_ROOT / "main.py"
ANYGRAPH_NODE_MAIN = ANYGRAPH_NODE_SRC_ROOT / "main.py"
ANYGRAPH_GRAPH_MAIN = ANYGRAPH_GRAPH_SRC_ROOT / "main.py"
_runtime_root_override = os.environ.get("ANYGRAPH_RUNTIME_ROOT", "").strip()
if _runtime_root_override:
    _runtime_root_path = Path(_runtime_root_override).expanduser()
    if not _runtime_root_path.is_absolute():
        raise ValueError("ANYGRAPH_RUNTIME_ROOT must be an absolute path when set.")
    ANYGRAPH_RUNTIME_ROOT = _runtime_root_path.resolve()
else:
    ANYGRAPH_RUNTIME_ROOT = project_path("outputs", "trained_models", "anygraph")
ANYGRAPH_MODELS_DIR = ANYGRAPH_RUNTIME_ROOT / "Models"
ANYGRAPH_HISTORY_DIR = ANYGRAPH_RUNTIME_ROOT / "History"


def ensure_anygraph_runtime_files_exist(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "AnyGraph runtime sources are missing: " + ", ".join(missing)
        )


def build_anygraph_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env["ANYGRAPH_RUNTIME_ROOT"] = str(ANYGRAPH_RUNTIME_ROOT)
    return env
