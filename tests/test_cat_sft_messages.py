import json
from io import StringIO
from pathlib import Path

import pytest

from llm_loadbalancer.tools import cat_sft_messages


def test_load_sft_messages_reads_selected_jsonl_row(tmp_path: Path):
    path = tmp_path / "sft.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"messages": [{"role": "user", "content": "first"}]}),
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "hello"},
                            {"role": "assistant", "content": "world"},
                        ]
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert cat_sft_messages.load_sft_messages(path, index=1) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]


def test_load_sft_messages_rejects_missing_messages_list(tmp_path: Path):
    path = tmp_path / "sft.jsonl"
    path.write_text(json.dumps({"not_messages": []}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="messages list"):
        cat_sft_messages.load_sft_messages(path)


def test_message_view_state_moves_and_toggles_expansion():
    state = cat_sft_messages.MessageViewState(
        messages=[
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
    )

    state.move(1)
    state.toggle_selected()
    state.move(100)

    assert state.selected == 1
    assert state.expanded == {1}


def test_render_message_lines_marks_selected_and_expanded_content():
    state = cat_sft_messages.MessageViewState(
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "long answer"},
        ],
        selected=1,
        expanded={1},
    )

    lines = [line.text for line in cat_sft_messages.render_message_lines(state, width=80)]

    assert lines[0] == "  * user: hello"
    assert lines[1] == "> * assistant: long answer"
    assert lines[2] == "    long answer"


def test_move_visual_line_walks_through_expanded_message_body():
    state = cat_sft_messages.MessageViewState(
        messages=[
            {"role": "user", "content": "first\nsecond"},
            {"role": "assistant", "content": "done"},
        ],
        expanded={0},
    )

    cat_sft_messages.move_visual_line(state, 1, width=80)
    first_step = cat_sft_messages.render_message_lines(state, width=80)
    cat_sft_messages.move_visual_line(state, 1, width=80)
    second_step = cat_sft_messages.render_message_lines(state, width=80)
    cat_sft_messages.move_visual_line(state, 1, width=80)
    third_step = cat_sft_messages.render_message_lines(state, width=80)

    assert state.selected == 1
    assert first_step[1].text == ">   first"
    assert second_step[2].text == ">   second"
    assert third_step[3].text == "> * assistant: done"


def test_move_visual_line_walks_back_into_expanded_message_body():
    state = cat_sft_messages.MessageViewState(
        messages=[
            {"role": "user", "content": "first\nsecond"},
            {"role": "assistant", "content": "done"},
        ],
        selected=1,
        expanded={0},
    )

    cat_sft_messages.move_visual_line(state, -1, width=80)
    lines = cat_sft_messages.render_message_lines(state, width=80)

    assert state.selected == 0
    assert lines[2].text == ">   second"


def test_render_message_lines_keeps_role_metadata_for_colors():
    state = cat_sft_messages.MessageViewState(
        messages=[{"role": "assistant", "content": "hello"}],
        expanded={0},
    )

    lines = cat_sft_messages.render_message_lines(state, width=80)

    assert lines[0].role == "assistant"
    assert not lines[0].is_body
    assert lines[1].role == "assistant"
    assert lines[1].is_body


def test_format_text_preserves_explicit_newlines_and_blank_lines():
    assert cat_sft_messages.format_text("hello\n\nworld\r\nagain", width=80) == [
        "hello",
        "",
        "world",
        "again",
    ]


def test_format_text_wraps_each_physical_line_without_eating_indentation():
    assert cat_sft_messages.format_text("  alpha beta gamma", width=12) == [
        "  alpha beta",
        "  gamma",
    ]


def test_format_text_wraps_long_unbroken_tokens():
    assert cat_sft_messages.format_text("abcdef", width=3) == ["abc", "def"]


def test_clip_text_preserves_spaces_while_truncating():
    assert cat_sft_messages.clip_text("    alpha beta gamma", width=12) == "    alpha..."


def test_format_tagged_text_marks_think_section_until_close():
    assert cat_sft_messages.format_tagged_text(
        "<think>\nstep one\n</think>\nanswer",
        width=80,
    ) == [
        ("<think>", "think"),
        ("step one", "think"),
        ("</think>", "think"),
        ("answer", None),
    ]


def test_format_tagged_text_marks_tool_response_section_until_close():
    assert cat_sft_messages.format_tagged_text(
        "<tool_response>\nfile contents\n</tool_response>",
        width=80,
    ) == [
        ("<tool_response>", "tool_response"),
        ("file contents", "tool_response"),
        ("</tool_response>", "tool_response"),
    ]


def test_render_message_lines_marks_special_tag_sections():
    state = cat_sft_messages.MessageViewState(
        messages=[
            {
                "role": "assistant",
                "content": "<tool_call>\n{\"name\":\"Bash\"}\n</tool_call>",
            }
        ],
        expanded={0},
    )

    lines = cat_sft_messages.render_message_lines(state, width=80)

    assert [line.tag for line in lines[1:]] == ["tool_call", "tool_call", "tool_call"]


def test_render_message_lines_marks_leading_tag_on_summary():
    state = cat_sft_messages.MessageViewState(
        messages=[{"role": "assistant", "content": "<think>short</think>"}],
    )

    lines = cat_sft_messages.render_message_lines(state, width=80)

    assert lines[0].tag == "think"


def test_visible_lines_scrolls_to_selected_message():
    state = cat_sft_messages.MessageViewState(
        messages=[
            {"role": "user", "content": str(index)}
            for index in range(10)
        ],
        selected=9,
    )

    lines = cat_sft_messages.visible_lines(state, height=3, width=80)

    assert lines == [
        "  * user: 7",
        "  * user: 8",
        "> * user: 9",
    ]


def test_print_plain_expands_all_messages():
    output = StringIO()
    cat_sft_messages.print_plain(
        [{"role": "user", "content": "hello\nworld"}],
        output,
        width=80,
    )

    assert output.getvalue() == "> * user: hello world\n    hello\n    world\n"
