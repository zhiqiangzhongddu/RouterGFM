"""Pytest bootstrap: put the repo root on sys.path and cap CPU threads.

The project has no installed package (imports are ``src.*`` resolved from
the repo root). ``python -m pytest`` / ``python -m unittest`` run from the
repo root work without this shim, and ``tests/__init__.py`` already lets
pytest's default import mode locate the repo root on its own; this file is
belt-and-braces for bare ``pytest`` invocations from other working
directories. ``unittest`` never reads conftest files.

The tests run tiny synthetic CPU workloads, which torch's default intra-op
pool (one thread per core) slows down by 5-10x on the shared many-core login
nodes. ``OMP_NUM_THREADS`` defaults to at most 4 (an explicit value wins) and
is applied to torch even if it was imported before this file.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("OMP_NUM_THREADS", str(min(4, os.cpu_count() or 1)))

import torch  # noqa: E402  (after the thread default, so torch's pool picks it up)

torch.set_num_threads(max(1, int(os.environ["OMP_NUM_THREADS"])))
