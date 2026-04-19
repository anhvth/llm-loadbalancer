from llm_loadbalancer.tools import build_unique_conversation


def _row(
    messages: list[dict],
    output: dict | None = None,
) -> build_unique_conversation._LoadedRow:
    row = {"input": {"messages": messages}, "output": output}
    return build_unique_conversation._LoadedRow(
        row=row,
        history_tokens=build_unique_conversation._history_tokens_for_row(row, "test"),
    )


def test_annotate_conversations_drops_prefix_rows():
    shorter = _row(
        [
            {"role": "user", "content": "hi"},
        ]
    )
    longer = _row(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "continue"},
        ]
    )

    annotated = build_unique_conversation._annotate_conversations([shorter, longer])

    assert len(annotated) == 1
    assert annotated[0]["input"]["messages"] == longer.row["input"]["messages"]
    assert annotated[0]["key_conversation_id"] == 0
    assert annotated[0]["key_is_longest"] is True


def test_annotate_conversations_uses_completed_normalized_turns():
    shorter = _row(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
            },
        ],
        {"content": [{"type": "text", "text": "hello"}]},
    )
    longer = _row(
        [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "greet", "signature": "abc"},
                    {"type": "text", "text": "hello"},
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "continue"}]},
        ],
        {"content": [{"type": "text", "text": "sure"}]},
    )

    annotated = build_unique_conversation._annotate_conversations([shorter, longer])

    assert len(annotated) == 1
    assert annotated[0]["input"]["messages"] == longer.row["input"]["messages"]


def test_annotate_conversations_matches_streamed_partial_tool_use_to_history():
    shorter = _row(
        [
            {"role": "user", "content": "touch hello"},
        ],
        {
            "content": [
                {"type": "text", "text": "\n\n\n"},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Bash",
                    "partial_json": '{"command": "touch \\"hello\\""}',
                },
            ],
        },
    )
    longer = _row(
        [
            {"role": "user", "content": "touch hello"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "Bash",
                        "input": {"command": 'touch "hello"'},
                    },
                ],
            },
            {"role": "user", "content": "done"},
        ],
        {"content": [{"type": "text", "text": "ok"}]},
    )

    annotated = build_unique_conversation._annotate_conversations([shorter, longer])

    assert len(annotated) == 1
    assert annotated[0]["input"]["messages"] == longer.row["input"]["messages"]
