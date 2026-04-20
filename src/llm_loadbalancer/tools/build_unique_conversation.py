#!/usr/bin/env python3
"""Build unique SFT dataset by deduplicating conversations via rendered prompts.

Pipeline:
    1. Read collected.jsonl (raw request/response logs)
    2. Convert each record to SFT messages and render as a prompt string
    3. Deduplicate: drop any prompt that is a substring of a longer kept prompt
    4. Output collected.unique_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from llm_loadbalancer.tools.convert_to_sft_data import (
    DEFAULT_TOKENIZER,
    _load_tokenizer,
    _record_tools,
    convert_record,
    messages_for_chat_template,
    split_rendered_chat,
    tools_for_chat_template,
)
from llm_loadbalancer.tools.sft_settings import drop_prefix_snapshots_in_unique


def _render_prompt_string(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    rendered = tokenizer.apply_chat_template(
        messages_for_chat_template(messages, tokenizer),
        tools=tools_for_chat_template(tools, tokenizer),
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        raise ValueError("Tokenizer chat template must render to a string")
    return rendered


def _convert_row(
    record: dict[str, Any], tokenizer: Any
) -> tuple[list[dict[str, str]], str]:
    """Convert a raw record to SFT messages and its rendered prompt string."""
    converted = convert_record(record)
    tools = _record_tools(record)
    rendered = _render_prompt_string(converted["messages"], tokenizer, tools=tools)
    sft_messages = split_rendered_chat(rendered, tokenizer)
    return sft_messages, rendered


def _render_row_prompt(record: dict[str, Any], tokenizer: Any) -> str:
    converted = convert_record(record)
    tools = _record_tools(record)
    return _render_prompt_string(converted["messages"], tokenizer, tools=tools)


def _drop_prefix_prompt_entries(
    entries: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Drop prompt entries whose prompt is a strict prefix of a longer prompt.

    Keeps the latest timestamp for exact-equal prompts.
    """
    if not entries:
        return []

    latest_by_prompt: dict[str, str] = {}
    for prompt, timestamp in entries:
        prev = latest_by_prompt.get(prompt, "")
        latest_by_prompt[prompt] = max(prev, timestamp or "")

    unique_entries = [(prompt, ts) for prompt, ts in latest_by_prompt.items()]
    unique_entries.sort(key=lambda item: len(item[0]), reverse=True)

    kept: list[tuple[str, str]] = []
    for prompt, timestamp in unique_entries:
        if any(
            len(prompt) < len(existing_prompt) and existing_prompt.startswith(prompt)
            for existing_prompt, _ in kept
        ):
            continue
        kept.append((prompt, timestamp))
    return kept


def _deduplicate_items_by_rendered_prompt(
    items: list[tuple[list[dict[str, str]], str, str]],
) -> list[tuple[list[dict[str, str]], str, str]]:
    """Deduplicate rendered items while preserving latest timestamp per prompt."""
    if not items:
        return []

    latest_by_prompt: dict[str, tuple[list[dict[str, str]], str, str]] = {}
    for messages, prompt, timestamp in items:
        prev = latest_by_prompt.get(prompt)
        if prev is None or (timestamp or "") >= prev[2]:
            latest_by_prompt[prompt] = (messages, prompt, timestamp or "")

    unique = list(latest_by_prompt.values())
    unique.sort(key=lambda x: -len(x[1]))

    kept: list[tuple[list[dict[str, str]], str, str]] = []
    kept_prompts: list[str] = []
    for messages, prompt, timestamp in tqdm(unique, desc="Deduplicating"):
        if any(prompt in kp for kp in kept_prompts):
            continue
        kept.append((messages, prompt, timestamp))
        kept_prompts.append(prompt)
    return kept


def group_by_session_then_dedupe(
    records: list[dict[str, Any]],
    tokenizer: Any,
    include_timestamps: bool = False,
) -> list[list[dict[str, str]]] | list[tuple[list[dict[str, str]], str]]:
    """Convert every row to prompt first, drop prefix snapshots, then dedupe."""
    prompt_entries: list[tuple[str, str]] = []
    for record in tqdm(records, desc="Rows to prompts"):
        prompt_entries.append(
            (_render_row_prompt(record, tokenizer), str(record.get("timestamp", "") or ""))
        )

    if drop_prefix_snapshots_in_unique():
        prompt_entries = _drop_prefix_prompt_entries(prompt_entries)

    items: list[tuple[list[dict[str, str]], str, str]] = []
    for rendered, timestamp in prompt_entries:
        sft_messages = split_rendered_chat(rendered, tokenizer)
        items.append((sft_messages, rendered, timestamp))

    kept = _deduplicate_items_by_rendered_prompt(items)
    kept.sort(key=lambda item: item[2])
    if include_timestamps:
        return [(messages, timestamp) for messages, _, timestamp in kept]
    return [messages for messages, _, _ in kept]


def deduplicate_by_rendered_prompt(
    items: list[tuple[list[dict[str, str]], str]],
) -> list[list[dict[str, str]]]:
    """Drop items whose rendered prompt is a substring of another's.

    Algorithm:
        1. Exact dedup via dict:  O(n)
        2. Sort by length desc:   O(n log n)
        3. Substring filter:      O(n · k · L)  where k = kept count, L = avg length.
           k ≪ n after exact dedup, so practical cost is low.
    """
    normalized_items = [
        (messages, prompt, "") for messages, prompt in items
    ]
    kept = _deduplicate_items_by_rendered_prompt(normalized_items)
    return [messages for messages, _, _ in kept]


def build_unique_sft(
    input_path: Path,
    output_path: Path,
    tokenizer_name: str | None = None,
) -> tuple[int, int]:
    """Read raw logs, convert to SFT, deduplicate, write output."""
    tokenizer = _load_tokenizer(tokenizer_name)

    records: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as fh:
        for line in tqdm(fh, desc="Reading records", unit="lines"):
            if not line.strip():
                continue
            records.append(json.loads(line))

    kept = group_by_session_then_dedupe(records, tokenizer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for messages in kept:
            fh.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    return len(records), len(kept)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build unique SFT dataset from collected request logs",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("~/.cache/llm-proxy/training-data/collected.jsonl").expanduser(),
        help="Input collected JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/anhvth8/.cache/llm-proxy/training-data/collected.unique_sft.jsonl"
        ),
        help="Output unique SFT JSONL file",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer for chat template rendering "
            f"(default: TOKENIZER_PATH or {DEFAULT_TOKENIZER})"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        total, unique = build_unique_sft(args.input, args.output, args.tokenizer)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Read {total} rows, kept {unique} unique conversations.")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
