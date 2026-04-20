from llm_loadbalancer.tools.build_unique_conversation import (
    deduplicate_by_rendered_prompt,
    _convert_row,
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
