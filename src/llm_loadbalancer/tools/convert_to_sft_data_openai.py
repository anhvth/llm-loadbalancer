import argparse
import json
from pathlib import Path
from typing import Any


def _text_from_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content)

    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue

        if not isinstance(part, dict):
            parts.append(str(part))
            continue

        part_type = part.get("type")
        if part_type in {"text", "input_text", "output_text", "refusal", "thinking"}:
            text = part.get("text")
            if text is None and part_type == "thinking":
                text = part.get("thinking")
            if text is not None:
                parts.append(str(text))

    return "\n".join(parts)


def _with_thinking(content: Any, reasoning: Any) -> str:
    text = _text_from_content(content).lstrip("\n")
    if not isinstance(reasoning, str) or not reasoning.strip():
        return text
    return f"<think>\n{reasoning.strip()}\n</think>\n\n{text}"


def _normalize_input_message(message: dict[str, Any]) -> dict[str, str]:
    role = str(message.get("role", "user"))
    if role == "assistant":
        return {
            "role": role,
            "content": _with_thinking(
                message.get("content"),
                message.get("reasoning") or message.get("reasoning_content"),
            ),
        }

    return {
        "role": role,
        "content": _text_from_content(message.get("content")),
    }


def _extract_assistant_output(output: Any) -> dict[str, str] | None:
    if not isinstance(output, dict):
        return None

    choices = output.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    choice = choices[0]
    if not isinstance(choice, dict):
        return None

    message = choice.get("message")
    if not isinstance(message, dict):
        return None

    return {
        "role": "assistant",
        "content": _with_thinking(
            message.get("content"),
            message.get("reasoning") or message.get("reasoning_content"),
        ),
    }


def convert_openai_chat_record(record: dict[str, Any], include_output: bool = True):
    source = record["input"] if "input" in record else record
    messages = source.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("OpenAI chat record is missing list-valued messages")

    out = {"messages": []}
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"Unexpected message type: {type(message).__name__}")
        out["messages"].append(_normalize_input_message(message))

    if include_output:
        assistant_output = _extract_assistant_output(record.get("output"))
        if assistant_output is not None:
            out["messages"].append(assistant_output)

    return out


def openai_format_to_train_sft(example_record: dict[str, Any]):
    return convert_openai_chat_record(example_record, include_output=True)


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_sft{input_path.suffix}")


def convert_jsonl(input_path: Path, output_path: Path) -> int:
    converted_rows = 0

    with input_path.open(encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_number, line in enumerate(input_handle, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{input_path}:{line_number}: row must decode to an object")

            converted = openai_format_to_train_sft(record)
            row = dict(record)
            row["messages"] = converted["messages"]
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
    args = parser.parse_args(argv)

    input_path = args.path_to_process
    output_path = args.output or _default_output_path(input_path)
    converted_rows = convert_jsonl(input_path, output_path)
    print(f"Wrote {converted_rows} converted rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
