#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"


def _normalize_for_key(value: Any) -> Any:
    if isinstance(value, list):
        normalized_items = []
        for item in value:
            normalized_item = _normalize_for_key(item)
            if normalized_item is not None:
                normalized_items.append(normalized_item)
        return normalized_items

    if not isinstance(value, dict):
        return value

    if value.get("type") == "thinking":
        return None

    return {
        key: _normalize_for_key(item)
        for key, item in value.items()
        if key not in {"cache_control", "signature"}
    }


def _strip_billing_header(message: Any) -> Any:
    if not isinstance(message, dict):
        return message

    content = message.get("content")
    if not isinstance(content, list) or not content:
        return message

    first_block = content[0]
    if not isinstance(first_block, dict):
        return message
    if first_block.get("type") != "text":
        return message

    text = first_block.get("text")
    if not isinstance(text, str) or not text.startswith(_BILLING_HEADER_PREFIX):
        return message

    normalized = dict(message)
    normalized["content"] = content[1:]
    return normalized


def _canonicalize_message(message: Any, *, normalize_first_message: bool) -> str:
    message = _normalize_for_key(message)
    if normalize_first_message:
        message = _strip_billing_header(message)

    return json.dumps(
        message,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _history_tokens(row: dict[str, Any], source: str) -> tuple[str, ...]:
    messages = row.get("input", {}).get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{source}: row is missing a list-valued 'input.messages' field")

    completed_messages = list(messages)
    output = row.get("output")
    if output:
        completed_messages.append(
            {
                "role": "assistant",
                "content": output.get("content", "") if isinstance(output, dict) else output,
            }
        )

    return tuple(
        _canonicalize_message(message, normalize_first_message=index == 0)
        for index, message in enumerate(completed_messages)
    )


def _load_rows(path: Path) -> list[tuple[int, str | None, tuple[str, ...]]]:
    rows: list[tuple[int, str | None, tuple[str, ...]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must decode to an object")

            rows.append(
                (
                    line_number,
                    row.get("id") if isinstance(row.get("id"), str) else None,
                    _history_tokens(row, f"{path}:{line_number}"),
                )
            )
    return rows


def assert_unique(path: Path) -> int:
    rows = _load_rows(path)

    for left_index, (left_line, left_id, left_tokens) in enumerate(rows):
        for right_line, right_id, right_tokens in rows[left_index + 1 :]:
            if left_tokens == right_tokens:
                raise AssertionError(
                    f"duplicate conversation chains at lines {left_line} and {right_line}"
                    + (f" ({left_id!r}, {right_id!r})" if left_id or right_id else "")
                )

            if len(left_tokens) <= len(right_tokens) and left_tokens == right_tokens[: len(left_tokens)]:
                raise AssertionError(
                    f"line {left_line} is a prefix of line {right_line}"
                    + (f" ({left_id!r} -> {right_id!r})" if left_id or right_id else "")
                )

            if len(right_tokens) <= len(left_tokens) and right_tokens == left_tokens[: len(right_tokens)]:
                raise AssertionError(
                    f"line {right_line} is a prefix of line {left_line}"
                    + (f" ({right_id!r} -> {left_id!r})" if left_id or right_id else "")
                )

    print(f"Checked {len(rows)} rows in {path}")
    print("No duplicate or prefix-related conversations found.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assert that a JSONL file contains unique conversations.")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("/home/anhvth8/.cache/llm-proxy/training-data/collected.unique_sft.jsonl"),
    )
    args = parser.parse_args(argv)
    return assert_unique(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
