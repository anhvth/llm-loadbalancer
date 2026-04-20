"""Tests for Anthropic session_id-based conversation deduplication."""

import json

import pytest

from llm_loadbalancer.tools.build_unique_conversation import (
    extract_anthropic_session_id,
    _select_best_record,
    group_by_session_then_dedupe,
    deduplicate_by_rendered_prompt,
)


def _make_anthropic_record(
    session_id: str,
    messages: list[dict],
    timestamp: str = "2026-04-20T00:00:00+00:00",
    output: dict | None = None,
) -> dict:
    """Build a minimal Anthropic-style collected record with session_id."""
    metadata_user_id = json.dumps({
        "device_id": "test-device",
        "account_uuid": "",
        "session_id": session_id,
    })
    rec = {
        "id": "test",
        "timestamp": timestamp,
        "input": {
            "messages": messages,
            "metadata": {"user_id": metadata_user_id},
        },
    }
    if output is not None:
        rec["output"] = output
    return rec


class _FakeTokenizer:
    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        return "".join(
            f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n"
            for m in messages
        )


# --- extract_anthropic_session_id ---

def test_extract_session_id_valid():
    rec = _make_anthropic_record("abc-123", [{"role": "user", "content": "hi"}])
    assert extract_anthropic_session_id(rec) == "abc-123"


def test_extract_session_id_missing_metadata():
    rec = {"input": {"messages": [{"role": "user", "content": "hi"}]}}
    assert extract_anthropic_session_id(rec) is None


def test_extract_session_id_malformed_user_id():
    rec = {
        "input": {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"user_id": "not-json"},
        }
    }
    assert extract_anthropic_session_id(rec) is None


def test_extract_session_id_no_session_id_key():
    rec = {
        "input": {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"user_id": json.dumps({"device_id": "x"})},
        }
    }
    assert extract_anthropic_session_id(rec) is None


def test_extract_session_id_empty_string():
    rec = {
        "input": {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"user_id": json.dumps({"session_id": ""})},
        }
    }
    assert extract_anthropic_session_id(rec) is None


def test_extract_session_id_bare_payload():
    """Works when record is a bare payload (no wrapping 'input' key)."""
    rec = {
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"user_id": json.dumps({"session_id": "s1"})},
    }
    assert extract_anthropic_session_id(rec) == "s1"


# --- _select_best_record ---

def test_select_best_record_prefers_longest_messages():
    r1 = _make_anthropic_record("s1", [{"role": "user", "content": "a"}])
    r2 = _make_anthropic_record(
        "s1",
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ],
    )
    assert _select_best_record([r1, r2]) is r2


def test_select_best_record_breaks_tie_by_timestamp():
    r1 = _make_anthropic_record(
        "s1",
        [{"role": "user", "content": "a"}],
        timestamp="2026-04-20T00:00:01+00:00",
    )
    r2 = _make_anthropic_record(
        "s1",
        [{"role": "user", "content": "a"}],
        timestamp="2026-04-20T00:00:02+00:00",
    )
    assert _select_best_record([r1, r2]) is r2


def test_select_best_record_prefers_with_output():
    r1 = _make_anthropic_record("s1", [{"role": "user", "content": "a"}])
    r2 = _make_anthropic_record(
        "s1",
        [{"role": "user", "content": "a"}],
        output={"content": [{"type": "text", "text": "resp"}]},
    )
    assert _select_best_record([r1, r2]) is r2


# --- group_by_session_then_dedupe ---

def test_same_session_collapses_to_one():
    """Multiple records with the same session_id produce one SFT row."""
    records = [
        _make_anthropic_record("s1", [{"role": "user", "content": "a"}], timestamp="2026-04-20T00:00:01+00:00"),
        _make_anthropic_record("s1", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}], timestamp="2026-04-20T00:00:02+00:00"),
        _make_anthropic_record("s1", [{"role": "user", "content": "x"}], timestamp="2026-04-20T00:00:03+00:00"),
    ]
    kept = group_by_session_then_dedupe(records, _FakeTokenizer())
    assert len(kept) == 1


def test_different_sessions_remain_separate():
    """Records with different session_ids produce separate SFT rows."""
    records = [
        _make_anthropic_record("s1", [{"role": "user", "content": "hello"}]),
        _make_anthropic_record("s2", [{"role": "user", "content": "world"}]),
    ]
    kept = group_by_session_then_dedupe(records, _FakeTokenizer())
    assert len(kept) == 2


def test_no_session_falls_back_to_prompt_dedupe():
    """Records without session_id use rendered-prompt deduplication."""
    r1 = {"input": {"messages": [{"role": "user", "content": "hi"}]}}
    r2 = {"input": {"messages": [{"role": "user", "content": "hi"}]}}
    r3 = {"input": {"messages": [{"role": "user", "content": "bye"}]}}
    kept = group_by_session_then_dedupe([r1, r2, r3], _FakeTokenizer())
    # r1 and r2 are exact duplicates, r3 is different
    assert len(kept) == 2


def test_mixed_session_and_no_session():
    """Session-grouped and non-session records are both handled."""
    records = [
        _make_anthropic_record("s1", [{"role": "user", "content": "a"}]),
        _make_anthropic_record("s1", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]),
        {"input": {"messages": [{"role": "user", "content": "standalone"}]}},
    ]
    kept = group_by_session_then_dedupe(records, _FakeTokenizer())
    # 1 from session s1, 1 from standalone
    assert len(kept) == 2


def test_representative_picks_richest():
    """The representative from a session group is the one with the most messages."""
    r_short = _make_anthropic_record("s1", [{"role": "user", "content": "a"}])
    r_long = _make_anthropic_record(
        "s1",
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
            {"role": "user", "content": "e"},
        ],
    )
    kept = group_by_session_then_dedupe([r_short, r_long], _FakeTokenizer())
    assert len(kept) == 1
    # The kept conversation should have content from the longer record
    all_content = " ".join(m.get("content", "") for m in kept[0])
    assert "e" in all_content


def test_malformed_metadata_falls_back():
    """Malformed metadata.user_id does not crash; falls back to prompt dedupe."""
    records = [
        {
            "input": {
                "messages": [{"role": "user", "content": "hi"}],
                "metadata": {"user_id": "{invalid json"},
            }
        },
        {
            "input": {
                "messages": [{"role": "user", "content": "bye"}],
                "metadata": {"user_id": "not-even-json"},
            }
        },
    ]
    kept = group_by_session_then_dedupe(records, _FakeTokenizer())
    assert len(kept) == 2
