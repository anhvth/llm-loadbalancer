import importlib.util
import json
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parent.parent / "tools" / "collect_jsonl.py"
_SPEC = importlib.util.spec_from_file_location("collect_jsonl", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
collect_jsonl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collect_jsonl)


def _anthropic_payload() -> dict:
    return {
        "input": {
            "system": [{"type": "text", "text": "You are helpful."}],
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What files are in /tmp?"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "I should inspect the directory."},
                        {"type": "tool_use", "name": "ReadDir", "input": {"path": "/tmp"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "a.txt\nb.txt",
                            "is_error": False,
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "ReadDir",
                    "description": "List a directory",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
        },
        "output": {
            "content": [
                {"type": "thinking", "thinking": "I can answer now."},
                {"type": "text", "text": "The directory contains a.txt and b.txt."},
            ]
        },
    }


def test_convert_file_uses_anthropic_converter_when_endpoint_slug_is_v1_messages(tmp_path: Path):
    sample = tmp_path / "1777000000000000000-12345-ep_v1_messages-abcdef.json"
    sample.write_text(json.dumps(_anthropic_payload()), encoding="utf-8")

    status, _, payload = collect_jsonl._convert_file(str(sample))

    assert status == "ok"
    assert payload is not None
    row = json.loads(payload)
    assert isinstance(row["messages"], list)
    assert isinstance(row["tools"], list)
    assert row["tools"][0]["name"] == "ReadDir"


def test_convert_file_skips_non_anthropic_endpoint_slug(tmp_path: Path):
    sample = tmp_path / "1777000000000000001-12345-ep_v1_chat_completions-fedcba.json"
    sample.write_text(json.dumps(_anthropic_payload()), encoding="utf-8")

    status, _, reason = collect_jsonl._convert_file(str(sample))

    assert status == "skip:unsupported_endpoint_format"
    assert "v1_chat_completions" in str(reason)
