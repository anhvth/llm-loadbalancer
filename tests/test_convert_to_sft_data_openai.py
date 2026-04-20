import json
from pathlib import Path

from llm_loadbalancer.tools import convert_to_sft_data, convert_to_sft_data_openai


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
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )


class _FakeMiniMaxTokenizer:
    bos_token = "]~!b["
    eos_token = "[e~["

    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
    ):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        rendered = []
        for index, message in enumerate(messages):
            role = "ai" if message["role"] == "assistant" else message["role"]
            bos = self.bos_token if index == 0 else ""
            rendered.append(f"{bos}]~b]{role}\n{message['content']}{self.eos_token}\n")
        return "".join(rendered)


def test_convert_openai_chat_completion_embeds_reasoning_in_content():
    converted = convert_to_sft_data_openai.convert_openai_chat_record(
        {
            "input": {
                "messages": [{"role": "user", "content": "hi openai"}],
                "model": "demo",
            },
            "output": {
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "\n\nHello! How can I help?",
                            "reasoning": "greet the user\n",
                            "tool_calls": [],
                        }
                    }
                ],
            },
        }
    )

    assert converted["messages"] == [
        {"role": "user", "content": "hi openai"},
        {
            "role": "assistant",
            "content": "Hello! How can I help?",
            "reasoning_content": "greet the user",
        },
    ]


def test_convert_openai_chat_completion_flattens_text_parts_like_vllm():
    converted = convert_to_sft_data_openai.convert_openai_chat_record(
        {
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello"},
                            {"type": "input_text", "text": "world"},
                        ],
                    }
                ],
            },
            "output": {
                "choices": [
                    {"message": {"role": "assistant", "content": "done"}}
                ],
            },
        }
    )

    assert converted["messages"][0] == {"role": "user", "content": "hello\nworld"}


def test_convert_openai_historical_assistant_keeps_reasoning_content():
    converted = convert_to_sft_data_openai.convert_openai_chat_record(
        {
            "input": {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "hello",
                        "reasoning": "greet",
                    },
                    {"role": "user", "content": "next"},
                ],
            },
            "output": {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
        }
    )

    historical = converted["messages"][1]
    assert historical["content"] == "<think>\ngreet\n</think>\n\nhello"
    assert historical["reasoning_content"] == "greet"


def test_dispatcher_routes_openai_rows(tmp_path: Path, monkeypatch):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "input": {"messages": [{"role": "user", "content": "hi"}]},
                "output": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "hello",
                                "reasoning": "say hello",
                            }
                        }
                    ],
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
    assert row["messages"][-1] == {"role": "assistant", "content": "hello"}
    assert "input" not in row
    assert "output" not in row
    assert "tools" not in row
    assert all(set(message) == {"role", "content"} for message in row["messages"])


def test_split_rendered_chat_returns_role_content_only():
    rendered = (
        "<|im_start|>system\nsys<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nreason\n</think>\n\nhello<|im_end|>\n"
    )

    assert convert_to_sft_data.split_rendered_chat(rendered) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>\nreason\n</think>\n\nhello"},
    ]


def test_dispatcher_writes_messages_from_minimax_rendered_template(
    tmp_path: Path,
    monkeypatch,
):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "input": {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "hi"},
                    ],
                },
                "output": {
                    "choices": [
                        {"message": {"role": "assistant", "content": "hello"}}
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    tokenizer = _FakeMiniMaxTokenizer()
    monkeypatch.setattr(convert_to_sft_data, "_load_tokenizer", lambda _: tokenizer)

    convert_to_sft_data.convert_jsonl(input_path, output_path)

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert list(row) == ["messages"]
    assert row["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
