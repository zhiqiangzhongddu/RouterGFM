"""Reuse the shared AnyGraph logger implementation for node classification."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_SHARED_LOGGER_PATH = Path(__file__).resolve().parents[2] / "utils" / "timelogger.py"
_SPEC = importlib.util.spec_from_file_location("_anygraph_shared_timelogger", _SHARED_LOGGER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load shared TimeLogger from {_SHARED_LOGGER_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
sys.modules[__name__] = _MODULE
globals().update(_MODULE.__dict__)
