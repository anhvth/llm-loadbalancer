import importlib.util
import json
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parent.parent / "tools" / "annotate_full_conversations.py"
_SPEC = importlib.util.spec_from_file_location("annotate_full_conversations", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
annotate_full_conversations = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(annotate_full_conversations)


def _system_message(billing_header: str, text: str = "You are helpful.") -> dict:
    return {
        "role": "system",
        "content": [
            {"type": "text", "text": billing_header},
            {"type": "text", "text": text},
        ],
    }


def _user_message(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant_message(content: str, reasoning: str | None = None) -> dict:
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return message


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_select_full_rows_ignores_volatile_billing_header():
    short_row = {
        "messages": [
            _system_message("x-anthropic-billing-header: a"),
            _user_message("hello"),
            _assistant_message("draft"),
        ],
        "tools": [],
    }
    full_row = {
        "messages": [
            _system_message("x-anthropic-billing-header: b"),
            _user_message("hello"),
            _assistant_message("draft"),
            _user_message("continue"),
            _assistant_message("done"),
        ],
        "tools": [{"name": "preserved"}],
    }

    selected = annotate_full_conversations._select_full_rows(
        [
            annotate_full_conversations._LoadedRow(
                row=short_row,
                history_tokens=annotate_full_conversations._history_tokens_for_row(short_row, "short"),
            ),
            annotate_full_conversations._LoadedRow(
                row=full_row,
                history_tokens=annotate_full_conversations._history_tokens_for_row(full_row, "full"),
            ),
        ]
    )

    assert selected == [full_row]


def test_select_full_rows_uses_history_not_final_assistant_message():
    short_row = {
        "messages": [
            _system_message("x-anthropic-billing-header: a"),
            _user_message("start"),
            _assistant_message("\n\n\n", reasoning="same reasoning"),
        ],
        "tools": [],
    }
    full_row = {
        "messages": [
            _system_message("x-anthropic-billing-header: a"),
            _user_message("start"),
            _assistant_message("", reasoning="same reasoning"),
            _user_message("next"),
            _assistant_message("final answer"),
        ],
        "tools": [],
    }

    selected = annotate_full_conversations._select_full_rows(
        [
            annotate_full_conversations._LoadedRow(
                row=short_row,
                history_tokens=annotate_full_conversations._history_tokens_for_row(short_row, "short"),
            ),
            annotate_full_conversations._LoadedRow(
                row=full_row,
                history_tokens=annotate_full_conversations._history_tokens_for_row(full_row, "full"),
            ),
        ]
    )

    assert selected == [full_row]


def test_main_merges_multiple_files_and_preserves_original_order(tmp_path: Path, capsys):
    session_b_full = {
        "messages": [
            _system_message("x-anthropic-billing-header: session-b"),
            _user_message("beta"),
            _assistant_message("beta done"),
        ],
        "tools": [{"name": "beta-tool"}],
    }
    session_a_short = {
        "messages": [
            _system_message("x-anthropic-billing-header: short-a"),
            _user_message("alpha"),
            _assistant_message("first"),
        ],
        "tools": [],
    }
    session_a_full = {
        "messages": [
            _system_message("x-anthropic-billing-header: full-a"),
            _user_message("alpha"),
            _assistant_message("first"),
            _user_message("follow up"),
            _assistant_message("final"),
        ],
        "tools": [{"name": "alpha-tool"}],
    }

    first_path = tmp_path / "2026-04-18.jsonl"
    second_path = tmp_path / "2026-04-19.jsonl"
    _write_jsonl(first_path, [session_b_full, session_a_short])
    _write_jsonl(second_path, [session_a_full])

    output_path = tmp_path / "full.jsonl"
    result = annotate_full_conversations.main(
        [
            "--input-glob",
            str(tmp_path / "2026-04-1*.jsonl"),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    written_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert result == 0
    assert written_rows == [session_b_full, session_a_full]
    assert (
        "Matched 2 files, read 3 rows, kept 2 full rows, dropped 1 cutoff rows."
        in captured.out
    )


def test_main_fails_when_no_files_match(tmp_path: Path, capsys):
    output_path = tmp_path / "full.jsonl"

    result = annotate_full_conversations.main(
        [
            "--input-glob",
            str(tmp_path / "*.jsonl"),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()

    assert result == 1
    assert not output_path.exists()
    assert "No files matched input glob" in captured.err
