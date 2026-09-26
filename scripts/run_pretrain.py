"""CLI entrypoint for pretraining."""

from __future__ import annotations

import os
import sys
from typing import Iterable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pretrain.run import run_pretrain_from_cli


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return run_pretrain_from_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
