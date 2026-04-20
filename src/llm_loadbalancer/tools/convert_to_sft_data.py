import argparse
import json
import os
from pathlib import Path
from typing import Any

from llm_loadbalancer.tools import (
    convert_to_sft_data_anthropic,
    convert_to_sft_data_openai,
)


DEFAULT_TOKENIZER = "Qwen/Qwen3.5-27B"
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def _is_openai_chat_record(record: dict[str, Any]) -> bool:
    output = record.get("output")
    return isinstance(output, dict) and isinstance(output.get("choices"), list)


def convert_record(record: dict[str, Any]):
    if _is_openai_chat_record(record):
        return convert_to_sft_data_openai.openai_format_to_train_sft(record)
    return convert_to_sft_data_anthropic.anthropic_format_to_train_sft(record)


def _load_tokenizer(tokenizer_name: str):
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name)


def _record_tools(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    source = record.get("input")
    if not isinstance(source, dict):
        source = record

    tools = source.get("tools")
    if isinstance(tools, list) and tools:
        return tools
    return None


def split_rendered_chat(rendered: str) -> list[dict[str, str]]:
    """Parse rendered chat template into role/content dicts.

    Handles content that itself contains literal <|im_start|>/<|im_end|> tokens
    (e.g. code conversations about chat templates) by only recognising
    <|im_start|> when it appears at the very start of a segment after an
    <|im_end|> split boundary.
    """
    raw_segments = rendered.split(_IM_END)

    messages: list[dict[str, str]] = []
    pending: str | None = None

    for segment in raw_segments:
        segment = segment.lstrip("\n")
        if not segment.strip():
            continue

        if segment.startswith(_IM_START):
            # Flush any previous pending message
            if pending is not None:
                _flush_pending(pending, messages)
            pending = segment[len(_IM_START):]
        else:
            # This segment is a continuation — the content itself contained
            # a literal <|im_end|> token.  Re-attach it.
            if pending is not None:
                pending += _IM_END + "\n" + segment
            else:
                # Leading content before the first <|im_start|>; skip.
                continue

    if pending is not None:
        _flush_pending(pending, messages)

    return messages


def _flush_pending(payload: str, messages: list[dict[str, str]]) -> None:
    role, separator, content = payload.partition("\n")
    if not separator:
        raise ValueError("Rendered chat segment is missing role/content separator")
    messages.append({"role": role.strip(), "content": content})


def render_messages_for_sft(
    messages: list[dict[str, Any]],
    tokenizer,
    tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        raise ValueError("Tokenizer chat template must render to a string")
    return split_rendered_chat(rendered)


def convert_record_to_sft_messages(
    record: dict[str, Any],
    tokenizer,
) -> list[dict[str, str]]:
    converted = convert_record(record)
    return render_messages_for_sft(
        converted["messages"],
        tokenizer,
        tools=_record_tools(record),
    )


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_sft{input_path.suffix}")


def convert_jsonl(
    input_path: Path,
    output_path: Path,
    tokenizer_name: str = DEFAULT_TOKENIZER,
) -> int:
    converted_rows = 0
    tokenizer = _load_tokenizer(tokenizer_name)

    with input_path.open(encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_number, line in enumerate(input_handle, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{input_path}:{line_number}: row must decode to an object")

            row = {"messages": convert_record_to_sft_messages(record, tokenizer)}
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            converted_rows += 1

    return converted_rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_process",
        nargs="?",
        default=Path("~/.cache/llm-proxy/training-data/collected.unique.jsonl").expanduser(),
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write the converted JSONL file (default: <input>_sft.jsonl)",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer chat template target (default: {DEFAULT_TOKENIZER})",
    )
    args = parser.parse_args(argv)

    input_path = args.path_to_process
    output_path = args.output or _default_output_path(input_path)
    converted_rows = convert_jsonl(input_path, output_path, tokenizer_name=args.tokenizer)
    print(f"Wrote {converted_rows} converted rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
