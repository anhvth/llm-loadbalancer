#!/usr/bin/env python3
"""Build unique conversation dataset by deduplicating prefix chains."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple, Sequence


_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"


class _LoadedRow(NamedTuple):
    row: dict[str, Any]
    history_tokens: tuple[str, ...]


class _TrieNode:
    __slots__ = ("children", "is_terminal", "has_terminal_descendant")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.is_terminal = False
        self.has_terminal_descendant = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate conversations with conversation IDs from collected JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/anhvth8/.cache/llm-proxy/training-data/collected.unique.jsonl"),
        help="Path to the output JSONL (default: ~/.cache/llm-proxy/training-data/conversion.jsonl)",
    )
    return parser


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

    if value.get("type") == "text" and not str(value.get("text", "")).strip():
        return None

    normalized = {
        key: _normalize_for_key(item)
        for key, item in value.items()
        if key not in {"cache_control", "signature"}
    }

    if value.get("type") == "tool_use" and "partial_json" in normalized:
        try:
            normalized["input"] = json.loads(normalized["partial_json"])
            normalized.pop("partial_json", None)
        except json.JSONDecodeError:
            pass

    return normalized


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


def _history_tokens_for_row(row: dict[str, Any], source: str) -> tuple[str, ...]:
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


def _annotate_conversations(loaded_rows: Sequence[_LoadedRow]) -> list[dict[str, Any]]:
    root = _TrieNode()

    def _is_prefix_of_kept_row(tokens: tuple[str, ...]) -> bool:
        node = root
        for token in tokens:
            node = node.children.get(token)
            if node is None:
                return False
        return node.is_terminal or node.has_terminal_descendant

    def _insert_kept_row(tokens: tuple[str, ...]) -> None:
        node = root
        path = [root]
        for token in tokens:
            node = node.children.setdefault(token, _TrieNode())
            path.append(node)

        node.is_terminal = True
        for ancestor in path[:-1]:
            ancestor.has_terminal_descendant = True

    kept_rows: list[_LoadedRow] = []
    for _, loaded_row in sorted(
        enumerate(loaded_rows),
        key=lambda item: (-len(item[1].history_tokens), item[0]),
    ):
        if _is_prefix_of_kept_row(loaded_row.history_tokens):
            continue

        _insert_kept_row(loaded_row.history_tokens)
        kept_rows.append(loaded_row)

    result: list[dict[str, Any]] = []
    for conversation_id, loaded_row in enumerate(kept_rows):
        row = dict(loaded_row.row)
        row["key_conversation_id"] = conversation_id
        row["key_is_longest"] = True
        result.append(row)

    return result


def extract_full_conversations(output_path: Path) -> tuple[int, int]:
    input_path = Path("/home/anhvth8/.cache/llm-proxy/training-data/collected.jsonl")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    loaded_rows = _load_rows([input_path])
    annotated_rows = _annotate_conversations(loaded_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in annotated_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    unique_conversations = len({row["key_conversation_id"] for row in annotated_rows})
    return len(loaded_rows), unique_conversations


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        total_rows, unique_conversations = extract_full_conversations(args.output)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"Read {total_rows} rows, annotated {unique_conversations} conversation groups."
    )
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
