"""Regression tests for prompt-based conversation deduplication."""

from llm_loadbalancer.tools.build_unique_conversation import (
    group_by_session_then_dedupe,
    deduplicate_by_rendered_prompt,
)


def _make_record(
    messages: list[dict],
    timestamp: str = "2026-04-20T00:00:00+00:00",
) -> dict:
    return {
        "id": "test",
        "timestamp": timestamp,
        "input": {
            "messages": messages,
        },
        "output": {
            "content": [{"type": "text", "text": "resp"}],
        },
    }


class _FakeTokenizer:
    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        return "".join(
            f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n"
            for m in messages
        )


def test_prompt_prefix_snapshots_collapsed():
    """Strict prompt-prefix snapshots are collapsed to the longer one."""
    records = [
        {
            "timestamp": "2026-04-20T00:00:01+00:00",
            "input": {"messages": [{"role": "user", "content": "a"}]},
            "output": {"content": [{"type": "text", "text": "b"}]},
        },
        {
            "timestamp": "2026-04-20T00:00:02+00:00",
            "input": {
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                    {"role": "user", "content": "c"},
                ]
            },
            "output": {"content": [{"type": "text", "text": "d"}]},
        },
        _make_record([{"role": "user", "content": "x"}], timestamp="2026-04-20T00:00:03+00:00"),
    ]
    kept = group_by_session_then_dedupe(records, _FakeTokenizer())
    assert len(kept) == 2


def test_non_prefix_histories_remain_separate():
    records = [
        _make_record([{"role": "user", "content": "hello"}]),
        _make_record([{"role": "user", "content": "world"}]),
    ]
    kept = group_by_session_then_dedupe(records, _FakeTokenizer())
    assert len(kept) == 2


def test_exact_duplicate_rows_collapsed():
    r1 = {"input": {"messages": [{"role": "user", "content": "hi"}]}}
    r2 = {"input": {"messages": [{"role": "user", "content": "hi"}]}}
    r3 = {"input": {"messages": [{"role": "user", "content": "bye"}]}}
    kept = group_by_session_then_dedupe([r1, r2, r3], _FakeTokenizer())
    assert len(kept) == 2


def test_malformed_metadata_is_ignored():
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
