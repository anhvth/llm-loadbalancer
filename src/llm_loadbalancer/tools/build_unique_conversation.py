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
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from llm_loadbalancer.tools.convert_to_sft_data import (
    DEFAULT_TOKENIZER,
    _load_tokenizer,
    _record_tools,
    convert_record,
    split_rendered_chat,
)


def _render_prompt_string(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
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
    sft_messages = split_rendered_chat(rendered)
    return sft_messages, rendered


def extract_anthropic_session_id(record: dict[str, Any]) -> str | None:
    """Extract session_id from Anthropic metadata.user_id JSON string.

    Returns None if metadata is missing, malformed, or has no session_id.
    """
    source = record.get("input") if "input" in record else record
    if not isinstance(source, dict):
        return None
    metadata = source.get("metadata")
    if not isinstance(metadata, dict):
        return None
    user_id_raw = metadata.get("user_id")
    if not isinstance(user_id_raw, str):
        return None
    try:
        user_id = json.loads(user_id_raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(user_id, dict):
        return None
    session_id = user_id.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return None
    return session_id


def _select_best_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the best representative from a group sharing the same session_id.

    Prefers: longest messages list, then latest timestamp, then has output.
    """
    def _score(rec: dict[str, Any]) -> tuple[int, str, bool]:
        source = rec.get("input") if "input" in rec else rec
        msgs = source.get("messages", []) if isinstance(source, dict) else []
        msg_count = len(msgs) if isinstance(msgs, list) else 0
        ts = rec.get("timestamp", "")
        has_output = bool(rec.get("output"))
        return (msg_count, ts, has_output)

    return max(records, key=_score)


def group_by_session_then_dedupe(
    records: list[dict[str, Any]],
    tokenizer: Any,
) -> list[list[dict[str, str]]]:
    """Group raw records by Anthropic session_id, pick best per group, then dedupe.

    Records with a valid session_id are grouped; only the best representative
    per session is kept. Records without a session_id go through the existing
    rendered-prompt deduplication.
    """
    session_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    no_session: list[dict[str, Any]] = []

    for record in records:
        sid = extract_anthropic_session_id(record)
        if sid is not None:
            session_groups[sid].append(record)
        else:
            no_session.append(record)

    # Convert best-per-session to SFT items
    items: list[tuple[list[dict[str, str]], str]] = []
    for sid, group in tqdm(session_groups.items(), desc="Session groups"):
        best = _select_best_record(group)
        sft_messages, rendered = _convert_row(best, tokenizer)
        items.append((sft_messages, rendered))

    # Convert no-session records
    for record in tqdm(no_session, desc="No-session records"):
        sft_messages, rendered = _convert_row(record, tokenizer)
        items.append((sft_messages, rendered))

    return deduplicate_by_rendered_prompt(items)


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
    if not items:
        return []

    # 1. Exact dedup
    seen: dict[str, int] = {}
    unique: list[tuple[list[dict[str, str]], str]] = []
    for messages, prompt in items:
        if prompt not in seen:
            seen[prompt] = len(unique)
            unique.append((messages, prompt))

    # 2. Sort longest first — longer prompts cannot be substrings of shorter ones
    unique.sort(key=lambda x: -len(x[1]))

    # 3. Drop any prompt that is a substring of an already-kept (longer) prompt
    kept_messages: list[list[dict[str, str]]] = []
    kept_prompts: list[str] = []
    for messages, prompt in tqdm(unique, desc="Deduplicating"):
        if any(prompt in kp for kp in kept_prompts):
            continue
        kept_messages.append(messages)
        kept_prompts.append(prompt)

    return kept_messages


def build_unique_sft(
    input_path: Path,
    output_path: Path,
    tokenizer_name: str = DEFAULT_TOKENIZER,
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
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer for chat template rendering (default: {DEFAULT_TOKENIZER})",
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
