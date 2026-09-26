"""CLI entrypoint for finetuning."""

from __future__ import annotations

import os
import sys
from typing import Iterable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.finetune.run import run_finetune_from_cli


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return run_finetune_from_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
