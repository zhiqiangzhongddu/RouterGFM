"""CLI entrypoint for mixture-of-experts (MoE) methods.

Select a method via ``moe.method``:

- ``anygraph`` — the AnyGraph pipeline (stage via ``moe.anygraph.execution.step``).
- ``routergfm`` — the RouterGFM router-based GFM pipeline.
- ``gmoe`` — the Graph Mixture of Experts encoder, trained end-to-end (config under ``moe.gmoe``).

Examples:
    python scripts/run_moe.py moe.method anygraph moe.anygraph.execution.step train \\
      moe.anygraph.train.dataset.name cora

    python scripts/run_moe.py moe.method gmoe moe.gmoe.dataset.name cora \\
      moe.gmoe.dataset.task_level node moe.gmoe.dataset.induced True
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Iterable, Optional

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.moe.run import run_moe_from_cli


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return run_moe_from_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
