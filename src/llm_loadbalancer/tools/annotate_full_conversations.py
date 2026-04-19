#!/usr/bin/env python3
"""Extract only full conversations from merged training JSONL files."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple, Sequence


_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"


class _LoadedRow(NamedTuple):
    row: dict[str, Any]
    history_tokens: tuple[str, ...]


class _TrieNode:
    __slots__ = ("children", "terminal_row_indices")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.terminal_row_indices: list[int] = []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract full conversations from training JSONL files",
    )
    parser.add_argument(
        "--input-glob",
        required=True,
        help="Glob pattern for JSONL input files",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to the merged full-conversation JSONL output",
    )
    return parser


def _strip_billing_header(message: Any) -> Any:
    if not isinstance(message, dict) or message.get("role") != "system":
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
    if normalize_first_message:
        message = _strip_billing_header(message)

    return json.dumps(
        message,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _history_tokens_for_row(row: dict[str, Any], source: str) -> tuple[str, ...]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{source}: row is missing a list-valued 'messages' field")

    return tuple(
        _canonicalize_message(message, normalize_first_message=index == 0)
        for index, message in enumerate(messages[:-1])
    )


def _load_rows(paths: Sequence[Path]) -> list[_LoadedRow]:
    loaded_rows: list[_LoadedRow] = []

    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc

                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: row must decode to an object")

                history_tokens = _history_tokens_for_row(row, f"{path}:{line_number}")
                loaded_rows.append(_LoadedRow(row=row, history_tokens=history_tokens))

    return loaded_rows


def _terminal_nodes_for_rows(loaded_rows: Sequence[_LoadedRow]) -> list[_TrieNode]:
    root = _TrieNode()
    terminal_nodes: list[_TrieNode] = []

    for row_index, loaded_row in enumerate(loaded_rows):
        node = root
        for token in loaded_row.history_tokens:
            node = node.children.setdefault(token, _TrieNode())
        node.terminal_row_indices.append(row_index)
        terminal_nodes.append(node)

    return terminal_nodes


def _select_full_rows(loaded_rows: Sequence[_LoadedRow]) -> list[dict[str, Any]]:
    terminal_nodes = _terminal_nodes_for_rows(loaded_rows)
    return [
        loaded_row.row
        for loaded_row, terminal_node in zip(loaded_rows, terminal_nodes, strict=True)
        if not terminal_node.children
    ]


def extract_full_conversations(input_glob: str, output_path: Path) -> tuple[int, int, int]:
    input_paths = [Path(path) for path in sorted(glob.glob(input_glob))]
    if not input_paths:
        raise FileNotFoundError(f"No files matched input glob: {input_glob}")

    loaded_rows = _load_rows(input_paths)
    full_rows = _select_full_rows(loaded_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in full_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(input_paths), len(loaded_rows), len(full_rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        matched_files, total_rows, full_rows = extract_full_conversations(
            args.input_glob,
            args.output,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    cutoff_rows = total_rows - full_rows
    print(
        f"Matched {matched_files} files, read {total_rows} rows, "
        f"kept {full_rows} full rows, dropped {cutoff_rows} cutoff rows."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
