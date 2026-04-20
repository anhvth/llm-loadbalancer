#!/usr/bin/env python3
"""Assert that a JSONL file of SFT conversations contains no duplicates or substring prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _canonicalize_messages(messages: list[dict[str, Any]]) -> str:
    return json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _load_rows(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must decode to an object")
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"{path}:{line_number}: row is missing 'messages'")
            rows.append((line_number, _canonicalize_messages(messages)))
    return rows


def assert_unique(path: Path) -> int:
    rows = _load_rows(path)

    # Sort by canonical length descending for efficient substring checks
    sorted_rows = sorted(rows, key=lambda r: -len(r[1]))

    kept: list[tuple[int, str]] = []
    for line_number, canonical in sorted_rows:
        for kept_line, kept_canonical in kept:
            if canonical == kept_canonical:
                raise AssertionError(
                    f"duplicate conversations at lines {line_number} and {kept_line}"
                )
            if canonical in kept_canonical:
                raise AssertionError(
                    f"line {line_number} is a substring of line {kept_line}"
                )
        kept.append((line_number, canonical))

    print(f"Checked {len(rows)} rows in {path}")
    print("No duplicate or substring conversations found.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assert that a JSONL file contains unique conversations."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path(
            "/home/anhvth8/.cache/llm-proxy/training-data/collected.unique_sft.jsonl"
        ),
    )
    args = parser.parse_args(argv)
    return assert_unique(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
