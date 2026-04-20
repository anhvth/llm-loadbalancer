import argparse
import curses
import json
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO


DEFAULT_PATH = Path(
    "~/.cache/llm-proxy/training-data/collected.unique_sft.jsonl"
).expanduser()
DEFAULT_FOLD_AFTER = 8
ROLE_COLOR_PAIRS = {
    "system": 1,
    "user": 2,
    "assistant": 3,
    "tool": 4,
}
TAG_COLOR_PAIRS = {
    "think": 5,
    "tools": 6,
    "tool_call": 7,
    "tool_response": 8,
}
TAG_PATTERNS = {
    "think": (re.compile(r"<think\b[^>]*>"), re.compile(r"</think>")),
    "tools": (re.compile(r"<tools\b[^>]*>"), re.compile(r"</tools>")),
    "tool_call": (re.compile(r"<tool_call\b[^>]*>"), re.compile(r"</tool_call>")),
    "tool_response": (
        re.compile(r"<tool_response\b[^>]*>"),
        re.compile(r"</tool_response>"),
    ),
}


@dataclass(frozen=True)
class RenderedLine:
    message_index: int
    text: str
    role: str
    is_body: bool = False
    tag: str | None = None
    is_cursor: bool = False


@dataclass
class MessageViewState:
    messages: list[dict[str, str]]
    selected: int = 0
    cursor_line: int | None = None
    expanded: set[int] = field(default_factory=set)
    top: int = 0
    fold_after: int = DEFAULT_FOLD_AFTER

    @property
    def folded(self) -> bool:
        return len(self.messages) > self.fold_after

    def move(self, delta: int) -> None:
        if not self.messages:
            self.selected = 0
            return
        self.selected = max(0, min(len(self.messages) - 1, self.selected + delta))
        self.cursor_line = None

    def toggle_selected(self) -> None:
        if not self.messages:
            return
        if self.selected in self.expanded:
            self.expanded.remove(self.selected)
        else:
            self.expanded.add(self.selected)
        self.cursor_line = None


def load_sft_messages(path: Path, index: int = 0) -> list[dict[str, str]]:
    if index < 0:
        raise ValueError("line index must be >= 0")

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line_number != index:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{index + 1}: row must decode to an object")
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"{path}:{index + 1}: row is missing a messages list")
            return [_normalize_message(message, path, index + 1) for message in messages]

    raise IndexError(f"{path} has no JSONL row at index {index}")


def _normalize_message(message: Any, path: Path, line_number: int) -> dict[str, str]:
    if not isinstance(message, dict):
        raise ValueError(f"{path}:{line_number}: message must be an object")

    role = message.get("role")
    content = message.get("content", "")
    if not isinstance(role, str):
        raise ValueError(f"{path}:{line_number}: message is missing a string role")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return {"role": role, "content": content}


def summarize_content(content: str, width: int) -> str:
    content = " ".join(content.split())
    return clip_text(content, width)


def clip_text(content: str, width: int) -> str:
    if width <= 1:
        return ""
    if len(content) <= width:
        return content
    if width <= 3:
        return "." * width
    return content[: max(0, width - 3)].rstrip() + "..."


def format_text(text: str, width: int) -> list[str]:
    width = max(1, width)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if normalized == "":
        return [""]

    rendered: list[str] = []
    for physical_line in normalized.split("\n"):
        if physical_line == "":
            rendered.append("")
            continue

        leading_match = re.match(r"[ \t]*", physical_line)
        leading = leading_match.group(0) if leading_match else ""
        subsequent_indent = leading if len(leading.expandtabs()) < width - 1 else ""
        wrapped = textwrap.wrap(
            physical_line,
            width=width,
            replace_whitespace=False,
            drop_whitespace=True,
            break_long_words=True,
            break_on_hyphens=False,
            subsequent_indent=subsequent_indent,
        )
        rendered.extend(wrapped or [""])

    return rendered


def format_tagged_text(text: str, width: int) -> list[tuple[str, str | None]]:
    active_tag: str | None = None
    rendered: list[tuple[str, str | None]] = []

    for line in format_text(text, width):
        line_tag = active_tag or _tag_started_by(line)
        rendered.append((line, line_tag))

        closed_tag = _tag_closed_by(line, line_tag)
        opened_tag = _tag_started_by(line)
        if closed_tag == active_tag:
            active_tag = None
        if opened_tag is not None and _tag_closed_by(line, opened_tag) != opened_tag:
            active_tag = opened_tag

    return rendered


def _tag_started_by(line: str) -> str | None:
    for tag, (open_pattern, _) in TAG_PATTERNS.items():
        if open_pattern.search(line):
            return tag
    return None


def _tag_closed_by(line: str, tag: str | None) -> str | None:
    if tag is None:
        return None
    _, close_pattern = TAG_PATTERNS[tag]
    if close_pattern.search(line):
        return tag
    return None


def render_message_lines(
    state: MessageViewState,
    width: int,
) -> list[RenderedLine]:
    lines: list[RenderedLine] = []
    summary_width = max(8, width - 18)
    cursor_line = state.cursor_line

    for index, message in enumerate(state.messages):
        role = message["role"]
        content = message["content"]
        summary = summarize_content(content, summary_width)
        is_cursor = (
            len(lines) == cursor_line if cursor_line is not None else index == state.selected
        )
        marker = ">" if is_cursor else " "
        prefix = f"{marker} * {role}: "
        lines.append(
            RenderedLine(
                index,
                f"{prefix}{summary}",
                role,
                tag=_leading_tag(content),
                is_cursor=is_cursor,
            )
        )

        if index in state.expanded:
            for wrapped_line, tag in format_tagged_text(content, width=max(10, width - 6)):
                is_cursor = len(lines) == cursor_line if cursor_line is not None else False
                prefix = ">   " if is_cursor else "    "
                lines.append(
                    RenderedLine(
                        index,
                        f"{prefix}{wrapped_line}",
                        role,
                        True,
                        tag,
                        is_cursor,
                    )
                )

    return lines


def move_visual_line(state: MessageViewState, delta: int, width: int) -> None:
    lines = render_message_lines(state, width)
    if not lines:
        state.selected = 0
        state.cursor_line = None
        return

    current = _current_cursor_line(state, lines)
    state.cursor_line = max(0, min(len(lines) - 1, current + delta))
    state.selected = lines[state.cursor_line].message_index


def _current_cursor_line(state: MessageViewState, lines: list[RenderedLine]) -> int:
    if state.cursor_line is not None:
        return max(0, min(len(lines) - 1, state.cursor_line))

    for line_index, line in enumerate(lines):
        if line.message_index == state.selected and not line.is_body:
            return line_index
    return 0


def visible_rendered_lines(
    state: MessageViewState,
    height: int,
    width: int,
) -> list[RenderedLine]:
    lines = render_message_lines(state, width)
    body_height = max(1, height)
    selected_line = _current_cursor_line(state, lines) if lines else 0

    if selected_line < state.top:
        state.top = selected_line
    elif selected_line >= state.top + body_height:
        state.top = selected_line - body_height + 1

    state.top = max(0, min(state.top, max(0, len(lines) - body_height)))
    return lines[state.top : state.top + body_height]


def visible_lines(state: MessageViewState, height: int, width: int) -> list[str]:
    return [line.text for line in visible_rendered_lines(state, height, width)]


def print_plain(messages: list[dict[str, str]], output: TextIO, width: int = 100) -> None:
    state = MessageViewState(messages=messages, expanded=set(range(len(messages))))
    for line in visible_lines(state, height=10**9, width=width):
        output.write(line + "\n")


def run_curses(stdscr, messages: list[dict[str, str]], path: Path, index: int) -> int:
    _init_colors()
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.keypad(True)
    state = MessageViewState(messages=messages)

    while True:
        height, width = stdscr.getmaxyx()
        stdscr.erase()
        _draw(stdscr, state, path, index, height, width)
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), 27):
            return 0
        if key in (curses.KEY_UP, ord("k")):
            move_visual_line(state, -1, width)
        elif key in (curses.KEY_DOWN, ord("j")):
            move_visual_line(state, 1, width)
        elif key in (curses.KEY_ENTER, 10, 13):
            state.toggle_selected()


def _draw(
    stdscr,
    state: MessageViewState,
    path: Path,
    index: int,
    height: int,
    width: int,
) -> None:
    header = f"{path} [-i {index}]  up/down move  Enter fold/unfold  q quit"
    footer = f"({state.selected + 1}/{len(state.messages)})"
    if state.folded:
        footer += " folded"

    _addstr(stdscr, 0, 0, summarize_content(header, width), curses.A_BOLD)
    _addstr(stdscr, 1, 0, "-" * max(0, width - 1))

    body_height = max(1, height - 4)
    for row, line in enumerate(visible_rendered_lines(state, body_height, width), start=2):
        display_text = clip_text(line.text, width) if line.is_body else summarize_content(line.text, width)
        _addstr(stdscr, row, 0, display_text, _line_attr(line))

    _addstr(stdscr, height - 1, 0, summarize_content(footer, width), curses.A_DIM)


def _init_colors() -> None:
    if not curses.has_colors():
        return

    curses.start_color()
    try:
        curses.use_default_colors()
        background = -1
    except curses.error:
        background = curses.COLOR_BLACK

    color_map = {
        "system": curses.COLOR_WHITE,
        "user": curses.COLOR_GREEN,
        "assistant": curses.COLOR_YELLOW,
        "tool": curses.COLOR_MAGENTA,
    }
    for role, pair_id in ROLE_COLOR_PAIRS.items():
        curses.init_pair(pair_id, color_map[role], background)

    tag_color_map = {
        "think": curses.COLOR_CYAN,
        "tools": curses.COLOR_MAGENTA,
        "tool_call": curses.COLOR_RED,
        "tool_response": curses.COLOR_WHITE,
    }
    for tag, pair_id in TAG_COLOR_PAIRS.items():
        curses.init_pair(pair_id, tag_color_map[tag], background)


def _line_attr(line: RenderedLine) -> int:
    attr = curses.A_BOLD if not line.is_body else curses.A_NORMAL
    pair_id = TAG_COLOR_PAIRS.get(line.tag or "") or ROLE_COLOR_PAIRS.get(line.role)
    if pair_id is not None and curses.has_colors():
        attr |= curses.color_pair(pair_id)
    if line.tag is not None:
        attr |= curses.A_BOLD
    if line.is_cursor:
        attr |= curses.A_REVERSE
    return attr


def _leading_tag(text: str) -> str | None:
    return _tag_started_by(text.lstrip()[:128])


def _addstr(stdscr, y: int, x: int, text: str, attr: int = curses.A_NORMAL) -> None:
    try:
        stdscr.addstr(y, x, text, attr)
    except curses.error:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_jsonl", nargs="?", type=Path, default=DEFAULT_PATH)
    parser.add_argument("-i", "--index", type=int, default=0, help="zero-based JSONL row")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="print all messages without opening the interactive viewer",
    )
    args = parser.parse_args(argv)

    messages = load_sft_messages(args.path_to_jsonl.expanduser(), args.index)
    if args.plain:
        import sys

        print_plain(messages, sys.stdout)
        return 0

    return curses.wrapper(run_curses, messages, args.path_to_jsonl.expanduser(), args.index)


if __name__ == "__main__":
    raise SystemExit(main())
