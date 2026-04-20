import json
from pathlib import Path

import pytest

from llm_loadbalancer.tools import collect_jsonl


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
    assert isinstance(payload, str)
    row = json.loads(payload)
    assert set(row.keys()) == {"id", "timestamp", "input", "output"}
    assert row["id"] == "my-request-001"
    assert row["timestamp"] is not None
    assert row["input"]["messages"][0]["role"] == "user"
    assert row["output"]["content"][0]["text"] == "a.txt and b.txt"
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
    assert isinstance(payload, str)
    row = json.loads(payload)
    assert row["input"] == bare
    assert row["output"] is None


def test_main_collects_request_files_from_package_path(tmp_path: Path):
    requests_dir = tmp_path / "logs" / "requests"
    export_dir = tmp_path / "training-data"
    sample = requests_dir / "request-001.json"
    requests_dir.mkdir(parents=True)
    sample.write_text(json.dumps(_payload_with_messages()), encoding="utf-8")

    result = collect_jsonl.main([
        "--requests-dir",
        str(requests_dir),
        "--export-dir",
        str(export_dir),
    ])

    export_path = export_dir / "collected.jsonl"
    lines = export_path.read_text(encoding="utf-8").splitlines()

    assert result == 0
    assert not sample.exists()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["id"] == "request-001"
    assert row["input"]["messages"][0]["content"] == "What files are in /tmp?"
    assert row["output"]["content"][0]["text"] == "a.txt and b.txt"


def test_collect_jsonl_help_has_no_state_db_option(capsys):
    with pytest.raises(SystemExit) as exc_info:
        collect_jsonl.main(["--help"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "--state-db" not in captured.out
