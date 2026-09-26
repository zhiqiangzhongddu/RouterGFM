"""Small IO helpers shared across the codebase."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from .paths import ensure_dir


def parse_csv_list(raw: str) -> list[str]:
    """Parse a comma-separated string into stripped, non-empty items."""
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def read_name_list_file(path: str | Path) -> list[str]:
    """Read the first column/token from a comment-tolerant text or TSV file."""
    names: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "\t" in stripped:
                stripped = stripped.split("\t", 1)[0].strip()
            else:
                stripped = stripped.split()[0]
            if stripped:
                names.append(stripped)
    return names


def append_csv_row(
    csv_path: str | Path,
    *,
    fieldnames: Sequence[str],
    row: dict,
) -> None:
    """Append one CSV row and emit the header when the file is empty."""
    path = Path(csv_path)
    ensure_dir(path.parent)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


__all__ = ["append_csv_row", "parse_csv_list", "read_name_list_file"]
