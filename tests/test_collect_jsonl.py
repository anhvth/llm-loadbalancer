import importlib.util
import json
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parent.parent / "tools" / "collect_jsonl.py"
_SPEC = importlib.util.spec_from_file_location("collect_jsonl", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
collect_jsonl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collect_jsonl)


def _payload_with_messages() -> dict:
    """A wrapped payload whose input contains a top-level 'messages' key."""
    return {
        "input": {
            "messages": [
                {"role": "user", "content": "What files are in /tmp?"},
            ],
        },
        "output": {
            "content": [{"type": "text", "text": "a.txt and b.txt"}],
        },
    }


def test_convert_file_exports_id_timestamp_input_output(tmp_path: Path):
    """A file whose input has 'messages' should produce an eligible row
    with id, timestamp, input, and output — no SFT conversion."""
    sample = tmp_path / "my-request-001.json"
    sample.write_text(json.dumps(_payload_with_messages()), encoding="utf-8")

    status, _, payload = collect_jsonl._convert_file(str(sample))

    assert status == "ok"
    assert payload is not None
    row = json.loads(payload)
    assert set(row.keys()) == {"id", "timestamp", "input", "output"}
    assert row["id"] == "my-request-001"
    assert row["input"]["messages"][0]["role"] == "user"
    assert row["output"]["content"][0]["text"] == "a.txt and b.txt"
    # No top-level SFT keys
    assert "messages" not in row or row.get("messages") is None or isinstance(row.get("input"), dict)
    assert "tools" not in row


def test_convert_file_skips_payload_without_messages(tmp_path: Path):
    """A file whose input lacks 'messages' should be skipped."""
    sample = tmp_path / "no-messages.json"
    sample.write_text(json.dumps({"input": {"prompt": "hi"}, "output": {}}), encoding="utf-8")

    status, _, _ = collect_jsonl._convert_file(str(sample))

    assert status == "skip:missing_messages"


def test_convert_file_bare_payload_with_messages(tmp_path: Path):
    """A bare request payload (no input/output wrapper) with 'messages'
    should be treated as input with output=None."""
    sample = tmp_path / "bare.json"
    bare = {"messages": [{"role": "user", "content": "hello"}], "model": "x"}
    sample.write_text(json.dumps(bare), encoding="utf-8")

    status, _, payload = collect_jsonl._convert_file(str(sample))

    assert status == "ok"
    row = json.loads(payload)
    assert row["input"] == bare
    assert row["output"] is None
