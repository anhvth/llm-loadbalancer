from typing import cast

from llm_loadbalancer.tools.build_unique_conversation import (
    deduplicate_by_rendered_prompt,
    _convert_row,
    group_by_session_then_dedupe,
)


def test_exact_duplicates_deduplicated():
    prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n"
    items = [
        (
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            prompt,
        ),
        (
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            prompt,
        ),
    ]
    kept = deduplicate_by_rendered_prompt(items)
    assert len(kept) == 1


def test_same_input_different_output_both_kept():
    prompt_a = (
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n"
    )
    prompt_b = (
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhey there<|im_end|>\n"
    )
    items = [
        (
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            prompt_a,
        ),
        (
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hey there"},
            ],
            prompt_b,
        ),
    ]
    kept = deduplicate_by_rendered_prompt(items)
    assert len(kept) == 2


def test_prefix_conversation_dropped():
    short_prompt = (
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n"
    )
    long_prompt = (
        short_prompt
        + "<|im_start|>user\ncontinue<|im_end|>\n<|im_start|>assistant\nsure<|im_end|>\n"
    )
    short_msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    long_msgs = short_msgs + [
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "sure"},
    ]
    items = [
        (short_msgs, short_prompt),
        (long_msgs, long_prompt),
    ]
    kept = deduplicate_by_rendered_prompt(items)
    assert len(kept) == 1
    assert len(kept[0]) == 4


def test_independent_conversations_both_kept():
    items = [
        ([{"role": "user", "content": "hello"}], "prompt_A"),
        ([{"role": "user", "content": "goodbye"}], "prompt_B"),
    ]
    kept = deduplicate_by_rendered_prompt(items)
    assert len(kept) == 2


def test_empty_input_returns_empty():
    assert deduplicate_by_rendered_prompt([]) == []


def test_convert_row_includes_openai_output_in_rendered_prompt():
    """OpenAI-format outputs must appear in the rendered prompt."""

    class _FakeTokenizer:
        def apply_chat_template(
            self, messages, tools=None, tokenize=False, add_generation_prompt=False
        ):
            return "".join(
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
            )

    record = {
        "input": {"messages": [{"role": "user", "content": "hi"}]},
        "output": {
            "choices": [
                {"message": {"role": "assistant", "content": "hello from openai"}}
            ],
        },
    }
    sft_messages, rendered = _convert_row(record, _FakeTokenizer())
    assert "hello from openai" in rendered
    assert sft_messages[-1]["content"] == "hello from openai"


def test_convert_row_parses_minimax_rendered_assistant_output():
    """MiniMax renders assistant as `ai`; export should keep assistant role/content."""

    class _FakeMiniMaxTokenizer:
        bos_token = "]~!b["
        eos_token = "[e~["

        def apply_chat_template(
            self, messages, tools=None, tokenize=False, add_generation_prompt=False
        ):
            rendered = []
            for index, message in enumerate(messages):
                role = "ai" if message["role"] == "assistant" else message["role"]
                bos = self.bos_token if index == 0 else ""
                rendered.append(f"{bos}]~b]{role}\n{message['content']}{self.eos_token}\n")
            return "".join(rendered)

    record = {
        "input": {"messages": [{"role": "user", "content": "hi"}]},
        "output": {
            "choices": [
                {"message": {"role": "assistant", "content": "hello from minimax"}}
            ],
        },
    }
    sft_messages, rendered = _convert_row(record, _FakeMiniMaxTokenizer())

    assert "]~b]ai\nhello from minimax[e~[" in rendered
    assert sft_messages[-1] == {
        "role": "assistant",
        "content": "hello from minimax",
    }


def test_convert_row_includes_anthropic_output_in_rendered_prompt():
    """Anthropic-format outputs must appear in the rendered prompt."""

    class _FakeTokenizer:
        def apply_chat_template(
            self, messages, tools=None, tokenize=False, add_generation_prompt=False
        ):
            return "".join(
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
            )

    record = {
        "input": {"messages": [{"role": "user", "content": "hi"}]},
        "output": {"content": [{"type": "text", "text": "hello from anthropic"}]},
    }
    sft_messages, rendered = _convert_row(record, _FakeTokenizer())
    assert "hello from anthropic" in rendered
    assert sft_messages[-1]["content"] == "hello from anthropic"


def test_two_openai_calls_same_input_different_output_both_kept():
    """The original bug: same input, different OpenAI outputs must produce 2 conversations."""

    class _FakeTokenizer:
        def apply_chat_template(
            self, messages, tools=None, tokenize=False, add_generation_prompt=False
        ):
            return "".join(
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
            )

    record_a = {
        "input": {"messages": [{"role": "user", "content": "hi ban"}]},
        "output": {
            "choices": [{"message": {"role": "assistant", "content": "xin chào!"}}]
        },
    }
    record_b = {
        "input": {"messages": [{"role": "user", "content": "hi ban"}]},
        "output": {
            "choices": [{"message": {"role": "assistant", "content": "chào bạn!"}}]
        },
    }
    tok = _FakeTokenizer()
    msgs_a, prompt_a = _convert_row(record_a, tok)
    msgs_b, prompt_b = _convert_row(record_b, tok)

    kept = deduplicate_by_rendered_prompt([(msgs_a, prompt_a), (msgs_b, prompt_b)])
    assert len(kept) == 2


def test_openai_prefix_snapshots_dropped_without_session_id():
    """No-session OpenAI snapshots with strict message-prefix should collapse."""

    class _FakeTokenizer:
        def apply_chat_template(
            self, messages, tools=None, tokenize=False, add_generation_prompt=False
        ):
            rendered = []
            last_user_index = None
            for index, message in enumerate(messages):
                if message.get("role") == "user":
                    last_user_index = index
            for index, message in enumerate(messages):
                content = message["content"]
                reasoning = message.get("reasoning_content")
                if (
                    message.get("role") == "assistant"
                    and isinstance(reasoning, str)
                    and reasoning
                    and not str(content).lstrip().startswith("<think>")
                    and last_user_index is not None
                    and index > last_user_index
                ):
                    content = f"<think>\n{reasoning}\n</think>\n\n{content}"
                rendered.append(
                    f"<|im_start|>{message['role']}\n{content}<|im_end|>\n"
                )
            return "".join(rendered)

    short = {
        "input": {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
        },
        "output": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                        "reasoning": "greet",
                    }
                }
            ]
        },
    }
    long = {
        "input": {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {
                    "role": "assistant",
                    "content": "hello",
                    "reasoning": "greet",
                },
                {"role": "user", "content": [{"type": "text", "text": "how are you"}]},
            ],
        },
        "output": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "great",
                        "reasoning": "small talk",
                    }
                }
            ]
        },
    }

    kept = group_by_session_then_dedupe([short, long], _FakeTokenizer())
    assert len(kept) == 1


def test_openai_same_length_variants_not_dropped_by_prefix_pruning():
    """Same-length histories are alternatives, not strict-prefix snapshots."""

    class _FakeTokenizer:
        def apply_chat_template(
            self, messages, tools=None, tokenize=False, add_generation_prompt=False
        ):
            payload = f"messages={len(messages)} last={messages[-1]['content']}"
            return f"<|im_start|>assistant\n{payload}<|im_end|>\n"

    a = {
        "input": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
        },
        "output": {"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
    }
    b = {
        "input": {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
        },
        "output": {"choices": [{"message": {"role": "assistant", "content": "hey"}}]},
    }

    kept = group_by_session_then_dedupe([a, b], _FakeTokenizer())
    assert len(kept) == 2


def test_openai_new_conversation_same_hi_keeps_two_conversations():
    """A new conversation that starts with same 'hi' should not be collapsed."""

    class _FakeTokenizer:
        def apply_chat_template(
            self, messages, tools=None, tokenize=False, add_generation_prompt=False
        ):
            rendered = []
            for message in messages:
                rendered.append(
                    f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                )
            return "".join(rendered)

    row0 = {
        "input": {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ]
        },
        "output": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hi! How can I help you today?",
                        "reasoning": "friendly greet v1",
                    }
                }
            ]
        },
    }
    row1 = {
        "input": {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {
                    "role": "assistant",
                    "content": "Hi! How can I help you today?",
                    "reasoning": "friendly greet v1",
                },
                {"role": "user", "content": [{"type": "text", "text": "how are you"}]},
            ]
        },
        "output": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I'm doing well.",
                        "reasoning": "small talk answer",
                    }
                }
            ]
        },
    }
    row2 = {
        "input": {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ]
        },
        "output": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hi! How can I help you today?",
                        "reasoning": "friendly greet v2",
                    }
                }
            ]
        },
    }

    kept = group_by_session_then_dedupe([row0, row1, row2], _FakeTokenizer())
    assert len(kept) == 2


def test_group_by_session_then_dedupe_returns_oldest_first_by_timestamp():
    class _FakeTokenizer:
        def apply_chat_template(
            self, messages, tools=None, tokenize=False, add_generation_prompt=False
        ):
            rendered = []
            for message in messages:
                rendered.append(
                    f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                )
            return "".join(rendered)

    old = {
        "timestamp": "2026-04-20T01:00:00+00:00",
        "input": {"messages": [{"role": "user", "content": "old"}]},
        "output": {"choices": [{"message": {"role": "assistant", "content": "a"}}]},
    }
    new = {
        "timestamp": "2026-04-20T02:00:00+00:00",
        "input": {"messages": [{"role": "user", "content": "new"}]},
        "output": {"choices": [{"message": {"role": "assistant", "content": "b"}}]},
    }

    kept = cast(list[list[dict[str, str]]], group_by_session_then_dedupe([new, old], _FakeTokenizer()))
    assert kept[0][0]["content"] == "old"
    assert kept[1][0]["content"] == "new"
