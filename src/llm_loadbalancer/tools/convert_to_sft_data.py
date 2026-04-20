import argparse
import json
import os
from pathlib import Path
from typing import Any

from jinja2 import BaseLoader
from jinja2.sandbox import ImmutableSandboxedEnvironment

from llm_loadbalancer.tools import (
    convert_to_sft_data_anthropic,
    convert_to_sft_data_openai,
)


DEFAULT_TOKENIZER = "Qwen/Qwen3.5-27B"
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_MINIMAX_BOS = "]~!b["
_MINIMAX_ROLE_PREFIX = "]~b]"
_MINIMAX_EOS = "[e~["

# Chat template extracted from Qwen/Qwen3.5-27B tokenizer_config.json.
# This avoids importing the heavy `transformers` library (~2.5s) and loading
# the tokenizer from disk (~2s).
_QWEN_CHAT_TEMPLATE = """{%- set image_count = namespace(value=0) %}\n{%- set video_count = namespace(value=0) %}\n{%- macro render_content(content, do_vision_count, is_system_content=false) %}\n    {%- if content is string %}\n        {{- content }}\n    {%- elif content is iterable and content is not mapping %}\n        {%- for item in content %}\n            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}\n                {%- if is_system_content %}\n                    {{- raise_exception('System message cannot contain images.') }}\n                {%- endif %}\n                {%- if do_vision_count %}\n                    {%- set image_count.value = image_count.value + 1 %}\n                {%- endif %}\n                {%- if add_vision_id %}\n                    {{- 'Picture ' ~ image_count.value ~ ': ' }}\n                {%- endif %}\n                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}\n            {%- elif 'video' in item or item.type == 'video' %}\n                {%- if is_system_content %}\n                    {{- raise_exception('System message cannot contain videos.') }}\n                {%- endif %}\n                {%- if do_vision_count %}\n                    {%- set video_count.value = video_count.value + 1 %}\n                {%- endif %}\n                {%- if add_vision_id %}\n                    {{- 'Video ' ~ video_count.value ~ ': ' }}\n                {%- endif %}\n                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}\n            {%- elif 'text' in item %}\n                {{- item.text }}\n            {%- else %}\n                {{- raise_exception('Unexpected item type in content.') }}\n            {%- endif %}\n        {%- endfor %}\n    {%- elif content is none or content is undefined %}\n        {{- '' }}\n    {%- else %}\n        {{- raise_exception('Unexpected content type.') }}\n    {%- endif %}\n{%- endmacro %}\n{%- if not messages %}\n    {{- raise_exception('No messages provided.') }}\n{%- endif %}\n{%- if tools and tools is iterable and tools is not mapping %}\n    {{- '<|im_start|>system\\n' }}\n    {{- "# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>" }}\n    {%- for tool in tools %}\n        {{- "\\n" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- "\\n</tools>" }}\n    {{- '\\n\\nIf you choose to call a function ONLY reply in the following format with NO suffix:\\n\\n<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>\\nvalue_1\\n</parameter>\\n<parameter=example_parameter_2>\\nThis is the value for the second parameter\\nthat can span\\nmultiple lines\\n</parameter>\\n</function>\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\\n- Required parameters MUST be specified\\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n</IMPORTANT>' }}\n    {%- if messages[0].role == 'system' %}\n        {%- set content = render_content(messages[0].content, false, true)|trim %}\n        {%- if content %}\n            {{- '\\n\\n' + content }}\n        {%- endif %}\n    {%- endif %}\n    {{- '<|im_end|>\\n' }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {%- set content = render_content(messages[0].content, false, true)|trim %}\n        {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == "user" %}\n        {%- set content = render_content(message.content, false)|trim %}\n        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}\n            {%- set ns.multi_step_tool = false %}\n            {%- set ns.last_query_index = index %}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if ns.multi_step_tool %}\n    {{- raise_exception('No user query found in messages.') }}\n{%- endif %}\n{%- for message in messages %}\n    {%- set content = render_content(message.content, true)|trim %}\n    {%- if message.role == "system" %}\n        {%- if not loop.first %}\n            {{- raise_exception('System message must be at the beginning.') }}\n        {%- endif %}\n    {%- elif message.role == "user" %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == "assistant" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- set reasoning_content = reasoning_content|trim %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if tool_call.function is defined %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {%- if loop.first %}\n                    {%- if content|trim %}\n                        {{- '\\n\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n                    {%- else %}\n                        {{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n                    {%- endif %}\n                {%- else %}\n                    {{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n                {%- endif %}\n                {%- if tool_call.arguments is defined %}\n                    {%- for args_name, args_value in tool_call.arguments|items %}\n                        {{- '<parameter=' + args_name + '>\\n' }}\n                        {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}\n                        {{- args_value }}\n                        {{- '\\n</parameter>\\n' }}\n                    {%- endfor %}\n                {%- endif %}\n                {{- '</function>\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == "tool" %}\n        {%- if loop.previtem and loop.previtem.role != "tool" %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if not loop.last and loop.nextitem.role != "tool" %}\n            {{- '<|im_end|>\\n' }}\n        {%- elif loop.last %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- else %}\n        {{- raise_exception('Unexpected message role.') }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- else %}\n        {{- '<think>\\n' }}\n    {%- endif %}\n{%- endif %}"""


def _raise_exception(message: str) -> str:
    raise ValueError(message)


def _tojson(value, ensure_ascii=True, indent=None, sort_keys=False):
    """Custom tojson filter matching transformers' behaviour."""
    separators = (",", ": ") if indent else (", ", ": ")
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def _build_jinja_env() -> ImmutableSandboxedEnvironment:
    env = ImmutableSandboxedEnvironment(
        loader=BaseLoader(),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals["raise_exception"] = _raise_exception
    env.filters["tojson"] = _tojson
    return env


_JINJA_ENV = _build_jinja_env()
_COMPILED_TEMPLATE = _JINJA_ENV.from_string(_QWEN_CHAT_TEMPLATE)


class _LightweightTokenizer:
    """Drop-in replacement for transformers AutoTokenizer.apply_chat_template.

    Reads the chat template (Jinja2) directly from the tokenizer config files,
    avoiding the heavy ``transformers`` import entirely.
    """

    def __init__(
        self,
        template: str = _QWEN_CHAT_TEMPLATE,
        bos_token: str = _IM_START,
        eos_token: str = _IM_END,
    ):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self._template = _JINJA_ENV.from_string(template)

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
    ) -> str:
        return self._template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            add_vision_id=False,
            enable_thinking=True,
        )

    @classmethod
    def from_local_dir(cls, path: str) -> "_LightweightTokenizer":
        """Load chat template from a local tokenizer directory."""
        tokenizer_dir = Path(path).expanduser()
        # Read bos/eos from tokenizer_config.json
        config_path = tokenizer_dir / "tokenizer_config.json"
        bos_token = _IM_START
        eos_token = _IM_END
        template_str: str | None = None

        if config_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(cfg.get("bos_token"), str):
                bos_token = cfg["bos_token"]
            if isinstance(cfg.get("eos_token"), str):
                eos_token = cfg["eos_token"]
            if isinstance(cfg.get("chat_template"), str):
                template_str = cfg["chat_template"]

        # Prefer chat_template.jinja file if it exists
        jinja_path = tokenizer_dir / "chat_template.jinja"
        if jinja_path.exists():
            template_str = jinja_path.read_text(encoding="utf-8")

        if template_str is None:
            raise FileNotFoundError(
                f"No chat_template found in {tokenizer_dir} "
                "(neither tokenizer_config.json nor chat_template.jinja)"
            )

        return cls(template=template_str, bos_token=bos_token, eos_token=eos_token)


def _is_openai_chat_record(record: dict[str, Any]) -> bool:
    output = record.get("output")
    return isinstance(output, dict) and isinstance(output.get("choices"), list)


def convert_record(record: dict[str, Any]):
    if _is_openai_chat_record(record):
        return convert_to_sft_data_openai.openai_format_to_train_sft(record)
    return convert_to_sft_data_anthropic.anthropic_format_to_train_sft(record)


def _selected_tokenizer_name(tokenizer_name: str | None = None) -> str:
    return tokenizer_name or os.environ.get("TOKENIZER_PATH") or DEFAULT_TOKENIZER


def _load_tokenizer(tokenizer_name: str | None = None):
    tokenizer_name = _selected_tokenizer_name(tokenizer_name)

    # Fast path: default Qwen tokenizer with pre-compiled template
    if tokenizer_name == DEFAULT_TOKENIZER:
        return _LightweightTokenizer()

    # Fast path: local directory with tokenizer config / chat_template.jinja
    expanded = os.path.expanduser(tokenizer_name)
    if os.path.isdir(expanded):
        try:
            return _LightweightTokenizer.from_local_dir(expanded)
        except FileNotFoundError:
            pass  # Fall through to transformers

    # Slow fallback: use transformers for HuggingFace Hub models
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(expanded)


def _tokenizer_text_attr(tokenizer: Any | None, attr: str) -> str | None:
    value = getattr(tokenizer, attr, None)
    return value if isinstance(value, str) and value else None


def _is_minimax_tokenizer(tokenizer: Any | None) -> bool:
    return (
        _tokenizer_text_attr(tokenizer, "bos_token") == _MINIMAX_BOS
        and _tokenizer_text_attr(tokenizer, "eos_token") == _MINIMAX_EOS
    )


def _role_from_rendered_label(role: str) -> str:
    if role == "ai":
        return "assistant"
    return role


def _record_tools(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    source = record.get("input")
    if not isinstance(source, dict):
        source = record

    tools = source.get("tools")
    if isinstance(tools, list) and tools:
        return tools
    return None


def tools_for_chat_template(
    tools: list[dict[str, Any]] | None,
    tokenizer: Any | None,
) -> list[dict[str, Any]] | None:
    if not tools or not _is_minimax_tokenizer(tokenizer):
        return tools
    return [_minimax_tool(tool) for tool in tools]


def _minimax_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    if isinstance(function, dict):
        return tool

    name = tool.get("name")
    if not isinstance(name, str) or not name:
        return {"type": "function", "function": tool}

    parameters = tool.get("parameters")
    if parameters is None:
        parameters = tool.get("input_schema")
    if parameters is None:
        parameters = {}

    normalized = {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.get("description", ""),
            "parameters": parameters,
        },
    }
    return normalized


def messages_for_chat_template(
    messages: list[dict[str, Any]],
    tokenizer: Any | None,
) -> list[dict[str, Any]]:
    if not _is_minimax_tokenizer(tokenizer):
        return messages

    normalized: list[dict[str, Any]] = []
    in_tool_sequence = False
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            normalized.append(message)
            tool_calls = message.get("tool_calls")
            in_tool_sequence = isinstance(tool_calls, list) and bool(tool_calls)
            continue

        if role == "tool":
            if in_tool_sequence:
                normalized.append(message)
                continue
            normalized.append(
                {
                    "role": "user",
                    "content": _minimax_orphan_tool_content(message.get("content", "")),
                }
            )
            in_tool_sequence = False
            continue

        normalized.append(message)
        in_tool_sequence = False

    return normalized


def _minimax_orphan_tool_content(content: Any) -> str:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        text = "\n".join(parts)
    else:
        text = str(content)
    return f"<response>{text}</response>"


def split_rendered_chat(
    rendered: str,
    tokenizer: Any | None = None,
) -> list[dict[str, str]]:
    """Parse rendered chat template into role/content dicts.

    Supports Qwen-style <|im_start|>/<|im_end|> and MiniMax-style
    ]~!b[]~b]role/[e~[ blocks. Content may contain literal template tokens;
    boundaries are only recognised at the start of a rendered segment.
    """
    bos_token = _tokenizer_text_attr(tokenizer, "bos_token")
    eos_token = _tokenizer_text_attr(tokenizer, "eos_token")

    if _IM_START in rendered or bos_token == _IM_START:
        return _split_qwen_rendered_chat(rendered)

    if (
        _MINIMAX_ROLE_PREFIX in rendered
        or (bos_token == _MINIMAX_BOS and eos_token == _MINIMAX_EOS)
    ):
        return _split_minimax_rendered_chat(
            rendered,
            bos_token=bos_token or _MINIMAX_BOS,
            eos_token=eos_token or _MINIMAX_EOS,
        )

    raise ValueError("Unsupported rendered chat template format")


def _split_qwen_rendered_chat(rendered: str) -> list[dict[str, str]]:
    raw_segments = rendered.split(_IM_END)

    messages: list[dict[str, str]] = []
    pending: str | None = None

    for segment in raw_segments:
        segment = segment.lstrip("\n")
        if not segment.strip():
            continue

        if segment.startswith(_IM_START):
            # Flush any previous pending message
            if pending is not None:
                _flush_pending(pending, messages)
            pending = segment[len(_IM_START):]
        else:
            # This segment is a continuation — the content itself contained
            # a literal <|im_end|> token.  Re-attach it.
            if pending is not None:
                pending += _IM_END + "\n" + segment
            else:
                # Leading content before the first <|im_start|>; skip.
                continue

    if pending is not None:
        _flush_pending(pending, messages)

    return messages


def _split_minimax_rendered_chat(
    rendered: str,
    *,
    bos_token: str,
    eos_token: str,
) -> list[dict[str, str]]:
    raw_segments = rendered.split(eos_token)

    messages: list[dict[str, str]] = []
    pending: str | None = None

    for segment in raw_segments:
        if not segment.strip():
            continue

        payload = _minimax_segment_payload(segment, bos_token)
        if payload is not None:
            if pending is not None:
                _flush_pending(pending, messages)
            pending = payload
            continue

        if pending is not None:
            pending += eos_token + segment

    if pending is not None:
        _flush_pending(pending, messages)

    return messages


def _minimax_segment_payload(segment: str, bos_token: str) -> str | None:
    candidate = segment.lstrip("\n")
    if candidate.startswith(bos_token):
        candidate = candidate[len(bos_token):]
    if not candidate.startswith(_MINIMAX_ROLE_PREFIX):
        return None
    return candidate[len(_MINIMAX_ROLE_PREFIX):]


def _flush_pending(payload: str, messages: list[dict[str, str]]) -> None:
    role, separator, content = payload.partition("\n")
    if not separator:
        raise ValueError("Rendered chat segment is missing role/content separator")
    messages.append({"role": _role_from_rendered_label(role.strip()), "content": content})


def render_messages_for_sft(
    messages: list[dict[str, Any]],
    tokenizer,
    tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    rendered = tokenizer.apply_chat_template(
        messages_for_chat_template(messages, tokenizer),
        tools=tools_for_chat_template(tools, tokenizer),
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        raise ValueError("Tokenizer chat template must render to a string")
    return split_rendered_chat(rendered, tokenizer)


def convert_record_to_sft_messages(
    record: dict[str, Any],
    tokenizer,
) -> list[dict[str, str]]:
    converted = convert_record(record)
    return render_messages_for_sft(
        converted["messages"],
        tokenizer,
        tools=_record_tools(record),
    )


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_sft{input_path.suffix}")


def convert_jsonl(
    input_path: Path,
    output_path: Path,
    tokenizer_name: str | None = None,
) -> int:
    converted_rows = 0
    tokenizer = _load_tokenizer(tokenizer_name)

    with input_path.open(encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line_number, line in enumerate(input_handle, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{input_path}:{line_number}: row must decode to an object")

            row = {"messages": convert_record_to_sft_messages(record, tokenizer)}
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            converted_rows += 1

    return converted_rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_process",
        nargs="?",
        default=Path("~/.cache/llm-proxy/training-data/collected.unique.jsonl").expanduser(),
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write the converted JSONL file (default: <input>_sft.jsonl)",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer chat template target "
            f"(default: TOKENIZER_PATH or {DEFAULT_TOKENIZER})"
        ),
    )
    args = parser.parse_args(argv)

    input_path = args.path_to_process
    output_path = args.output or _default_output_path(input_path)
    converted_rows = convert_jsonl(input_path, output_path, tokenizer_name=args.tokenizer)
    print(f"Wrote {converted_rows} converted rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
