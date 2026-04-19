import argparse
import json
from pathlib import Path
from typing import Any

from llm_loadbalancer.tools import (
    convert_to_sft_data_anthropic,
    convert_to_sft_data_openai,
)


def _is_openai_chat_record(record: dict[str, Any]) -> bool:
    output = record.get("output")
    return isinstance(output, dict) and isinstance(output.get("choices"), list)


def convert_record(record: dict[str, Any]):
    if _is_openai_chat_record(record):
        return convert_to_sft_data_openai.openai_format_to_train_sft(record)
    return convert_to_sft_data_anthropic.anthropic_format_to_train_sft(record)


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

            converted = convert_record(record)
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
