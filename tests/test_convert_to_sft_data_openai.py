import json
from pathlib import Path

from llm_loadbalancer.tools import convert_to_sft_data, convert_to_sft_data_openai


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
            "content": "<think>\ngreet the user\n</think>\n\nHello! How can I help?",
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


def test_dispatcher_routes_openai_rows(tmp_path: Path):
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

    convert_to_sft_data.convert_jsonl(input_path, output_path)

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["messages"][-1] == {
        "role": "assistant",
        "content": "<think>\nsay hello\n</think>\n\nhello",
    }
