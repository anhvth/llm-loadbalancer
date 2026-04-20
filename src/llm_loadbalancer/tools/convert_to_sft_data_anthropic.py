import argparse
import json
from copy import deepcopy
from pathlib import Path

from llm_loadbalancer.tools.sft_settings import preserve_thinking_in_content

_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"


def _strip_billing_header(text: str) -> str:
    if text.startswith(_BILLING_HEADER_PREFIX):
        return text[len(_BILLING_HEADER_PREFIX):].lstrip()
    return text


def _inline_historical_reasoning_for_qwen(messages):
    """
    Qwen's chat template only renders `reasoning_content` for assistant turns
    after the last user query. Inline earlier assistant thinking into content so
    multi-turn SFT keeps every assistant's thinking span.
    """
    last_user_index = None
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user_index = index

    if last_user_index is None:
        return

    for index, message in enumerate(messages):
        if index > last_user_index or message.get("role") != "assistant":
            continue

        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning.strip():
            continue

        think_block = f"<think>\n{reasoning.strip()}\n</think>\n\n"
        content = message.get("content", "")
        if isinstance(content, str):
            if content.lstrip().startswith("<think>"):
                continue
            message["content"] = think_block + content.lstrip("\n")
        elif isinstance(content, list):
            if (
                content
                and isinstance(content[0], dict)
                and content[0].get("type") == "text"
                and str(content[0].get("text", "")).lstrip().startswith("<think>")
            ):
                continue
            normalized_content = list(content)
            if (
                normalized_content
                and isinstance(normalized_content[0], dict)
                and normalized_content[0].get("type") == "text"
                and isinstance(normalized_content[0].get("text"), str)
            ):
                normalized_content[0] = {
                    **normalized_content[0],
                    "text": normalized_content[0]["text"].lstrip("\n"),
                }
            message["content"] = [{"type": "text", "text": think_block}, *normalized_content]


def convert_claude_code_record(record, include_output=False):
    """
    Convert a Claude Code / Anthropic-style record into the format expected by
    the provided tokenizer chat template.

    Input can be either:
      1) a full record: {"input": {...}, "output": {...}}
      2) a plain request payload: {"messages": [...], "system": [...], "tools": [...]}

    Returns:
      {
        "messages": [...],   # ready for tokenizer.apply_chat_template(messages=...)
        "tools": [...],      # passthrough tool schemas
      }
    """

    source = record["input"] if "input" in record else record
    out = {
        "messages": [],
        "tools": deepcopy(source.get("tools", [])),
    }

    def text_item(text):
        return {"type": "text", "text": text}

    def normalize_renderable_content(content):
        """
        Keep only content types supported by render_content():
        - string
        - list of items containing text/image/video
        Drop Anthropic-specific blocks handled elsewhere:
        - thinking
        - tool_use
        - tool_result
        """
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            return str(content)

        items = []
        for item in content:
            if item is None:
                continue

            if isinstance(item, str):
                items.append(text_item(item))
                continue

            if not isinstance(item, dict):
                items.append(text_item(str(item)))
                continue

            t = item.get("type")

            if t == "text":
                items.append({"type": "text", "text": item.get("text", "")})
            elif t == "image":
                items.append(item)
            elif t == "video":
                items.append(item)
            elif "image" in item or "image_url" in item:
                items.append(item)
            elif "video" in item:
                items.append(item)
            # thinking / tool_use / tool_result are intentionally skipped here

        if not items:
            return ""

        # keep a single text block as a plain string if possible
        if len(items) == 1 and items[0].get("type") == "text":
            return items[0]["text"]

        return items

    def parse_tool_args(block):
        """
        Anthropic tool_use blocks usually store args under `input`.
        In streamed / partial outputs you may only have `partial_json`.
        """
        args = block.get("input", None)

        if args in (None, {}) and block.get("partial_json"):
            try:
                args = json.loads(block["partial_json"])
            except Exception:
                args = {"__raw_partial_json": block["partial_json"]}

        return args if args is not None else {}

    def append_system_message(system_blocks):
        if not system_blocks:
            return

        content = []
        for block in system_blocks:
            if isinstance(block, str):
                content.append(text_item(_strip_billing_header(block)))
            elif isinstance(block, dict) and block.get("type") == "text":
                content.append({"type": "text", "text": _strip_billing_header(block.get("text", ""))})
            elif isinstance(block, dict):
                # fallback: stringify unknown system items
                content.append(text_item(json.dumps(block, ensure_ascii=False)))
            else:
                content.append(text_item(str(block)))

        out["messages"].append(
            {
                "role": "system",
                "content": content if len(content) > 1 else content[0]["text"],
            }
        )

    def convert_user_message(msg):
        content = msg.get("content", "")
        if isinstance(content, str):
            out["messages"].append({"role": "user", "content": content})
            return

        if not isinstance(content, list):
            out["messages"].append({"role": "user", "content": str(content)})
            return

        pending_user_blocks = []

        def flush_user():
            nonlocal pending_user_blocks
            if pending_user_blocks:
                rendered = normalize_renderable_content(pending_user_blocks)
                out["messages"].append({"role": "user", "content": rendered})
                pending_user_blocks = []

        for block in content:
            if isinstance(block, str):
                pending_user_blocks.append(text_item(block))
                continue

            if not isinstance(block, dict):
                pending_user_blocks.append(text_item(str(block)))
                continue

            t = block.get("type")

            if t == "tool_result":
                flush_user()
                tool_content = normalize_renderable_content(block.get("content", ""))
                out["messages"].append({"role": "tool", "content": tool_content})
            else:
                # keep only renderable user items
                if t == "text":
                    pending_user_blocks.append(
                        {"type": "text", "text": block.get("text", "")}
                    )
                elif t in ("image", "video"):
                    pending_user_blocks.append(block)
                elif "image" in block or "image_url" in block or "video" in block:
                    pending_user_blocks.append(block)
                else:
                    # fallback: stringify unknown user blocks
                    pending_user_blocks.append(
                        text_item(json.dumps(block, ensure_ascii=False))
                    )

        flush_user()

    def convert_assistant_message(msg):
        content = msg.get("content", "")
        reasoning_parts = []
        renderable_blocks = []
        tool_calls = []

        if isinstance(content, str):
            renderable_blocks = [text_item(content)]
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    renderable_blocks.append(text_item(block))
                    continue

                if not isinstance(block, dict):
                    renderable_blocks.append(text_item(str(block)))
                    continue

                t = block.get("type")

                if t == "thinking":
                    if block.get("thinking"):
                        reasoning_parts.append(block["thinking"])
                elif t == "text":
                    renderable_blocks.append(
                        {"type": "text", "text": block.get("text", "")}
                    )
                elif t == "tool_use":
                    tool_calls.append(
                        {
                            "name": block.get("name", ""),
                            "arguments": parse_tool_args(block),
                        }
                    )
                elif t in ("image", "video"):
                    renderable_blocks.append(block)
                elif "image" in block or "image_url" in block or "video" in block:
                    renderable_blocks.append(block)
                else:
                    # fallback: stringify unknown assistant blocks
                    renderable_blocks.append(
                        text_item(json.dumps(block, ensure_ascii=False))
                    )
        else:
            renderable_blocks = [text_item(str(content))]

        assistant_msg = {
            "role": "assistant",
            "content": normalize_renderable_content(renderable_blocks),
        }

        reasoning = "\n\n".join(p for p in reasoning_parts if p).strip()
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning

        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        out["messages"].append(assistant_msg)

    # 1) top-level system -> first system message
    append_system_message(source.get("system", []))

    # 2) convert conversation messages
    for msg in source.get("messages", []):
        role = msg.get("role")

        if role == "system":
            # tokenizer requires system at the beginning only; merge late system into user text if encountered
            content = msg.get("content", "")
            if isinstance(content, str):
                content = _strip_billing_header(content)
            elif isinstance(content, list):
                content = [
                    {**block, "text": _strip_billing_header(block["text"])}
                    if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
                    else block
                    for block in content
                ]
            rendered = normalize_renderable_content(content)
            if out["messages"] and out["messages"][0]["role"] == "system":
                existing = out["messages"][0]["content"]
                if isinstance(existing, str):
                    existing_list = [text_item(existing)]
                else:
                    existing_list = existing
                if isinstance(rendered, str):
                    existing_list.append(text_item(rendered))
                elif isinstance(rendered, list):
                    existing_list.extend(rendered)
                out["messages"][0]["content"] = existing_list
            else:
                out["messages"].insert(0, {"role": "system", "content": rendered})

        elif role == "user":
            convert_user_message(msg)

        elif role == "assistant":
            convert_assistant_message(msg)

        elif role == "tool":
            out["messages"].append(
                {
                    "role": "tool",
                    "content": normalize_renderable_content(msg.get("content", "")),
                }
            )

        else:
            raise ValueError(f"Unexpected role: {role}")

    # 3) optionally append the sampled assistant output as the final target turn
    if include_output and "output" in record and record["output"]:
        output_obj = record["output"]
        if isinstance(output_obj, dict):
            convert_assistant_message(
                {"role": "assistant", "content": output_obj.get("content", "")}
            )

    if preserve_thinking_in_content():
        _inline_historical_reasoning_for_qwen(out["messages"])

    return out


def anthropic_format_to_train_sft(example_record):
    converted = convert_claude_code_record(example_record, include_output=True)
    return converted


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

            converted = anthropic_format_to_train_sft(record)
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
