import json
from pathlib import Path

from llm_loadbalancer.tools import convert_to_sft_data_anthropic


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
