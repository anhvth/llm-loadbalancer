import json
from pathlib import Path

from llm_loadbalancer.tools import convert_to_sft_data, convert_to_sft_data_anthropic


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
    ):
        self.calls.append({"messages": messages, "tools": tools})
        rendered = []
        if tools:
            rendered.append("<|im_start|>system\nTOOL DEFINITIONS<|im_end|>\n")
        for message in messages:
            content = message["content"]
            if not isinstance(content, str):
                content = "\n".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            reasoning = message.get("reasoning_content")
            if (
                message["role"] == "assistant"
                and isinstance(reasoning, str)
                and reasoning
                and not content.lstrip().startswith("<think>")
            ):
                body = content.lstrip("\n")
                content = f"<think>\n{reasoning}\n</think>\n\n{body}"
            rendered.append(f"<|im_start|>{message['role']}\n{content}<|im_end|>\n")
        return "".join(rendered)


def test_convert_jsonl_writes_messages_column(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "input": {
                    "messages": [{"role": "user", "content": "hi"}],
                },
                "output": {"content": [{"type": "text", "text": "hello"}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    converted_rows = convert_to_sft_data_anthropic.convert_jsonl(input_path, output_path)

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert converted_rows == 1
    assert "messages" in row
    assert "sft_messages" not in row
    assert row["messages"][-1] == {"role": "assistant", "content": "hello"}


def test_convert_inlines_historical_reasoning_for_qwen_template():
    converted = convert_to_sft_data_anthropic.convert_claude_code_record(
        {
            "input": {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "greet\n"},
                            {"type": "text", "text": "\n\nhi"},
                        ],
                    },
                    {"role": "user", "content": "next"},
                ],
            },
            "output": {
                "content": [
                    {"type": "thinking", "thinking": "answer\n"},
                    {"type": "text", "text": "done"},
                ],
            },
        },
        include_output=True,
    )

    historical_assistant = converted["messages"][1]
    final_assistant = converted["messages"][-1]

    assert historical_assistant["content"] == "<think>\ngreet\n</think>\n\nhi"
    assert historical_assistant["reasoning_content"] == "greet"
    assert final_assistant["content"] == "done"
    assert final_assistant["reasoning_content"] == "answer"


def test_dispatcher_writes_anthropic_rows_as_rendered_role_content_only(
    tmp_path: Path,
    monkeypatch,
):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    tools = [{"name": "Bash", "description": "run shell"}]
    input_path.write_text(
        json.dumps(
            {
                "input": {
                    "tools": tools,
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "thinking", "thinking": "greet\n"},
                                {"type": "text", "text": "\n\nhi"},
                            ],
                        },
                        {"role": "user", "content": "next"},
                    ],
                },
                "output": {
                    "content": [
                        {"type": "thinking", "thinking": "answer\n"},
                        {"type": "text", "text": "done"},
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(convert_to_sft_data, "_load_tokenizer", lambda _: tokenizer)

    convert_to_sft_data.convert_jsonl(input_path, output_path)

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert list(row) == ["messages"]
    assert tokenizer.calls[0]["tools"] == tools
    assert all(set(message) == {"role", "content"} for message in row["messages"])
    assert row["messages"][0] == {"role": "system", "content": "TOOL DEFINITIONS"}
    assert row["messages"][2]["content"] == "<think>\ngreet\n</think>\n\nhi"
    assert row["messages"][-1]["content"] == "<think>\nanswer\n</think>\n\ndone"
