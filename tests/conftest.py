"""Pytest bootstrap: put the repo root on sys.path.

The project has no installed package (imports are ``src.*`` resolved from
the repo root). ``python -m pytest`` / ``python -m unittest`` run from the
repo root work without this shim, and ``tests/__init__.py`` already lets
pytest's default import mode locate the repo root on its own; this file is
belt-and-braces for bare ``pytest`` invocations from other working
directories. ``unittest`` never reads conftest files.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
