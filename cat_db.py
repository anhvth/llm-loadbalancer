from __future__ import annotations

import argparse
import fcntl
import json
import os
import pathlib
import sys
import textwrap

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - not expected on Unix-like systems used here
    termios = None
    tty = None


REQUEST_LOGS_DIRNAME = "requests"


def default_log_dir() -> pathlib.Path:
    for config_path in (
        pathlib.Path(os.environ.get("LLM_LOADBALANCER_CONFIG", "~/.cache/llmup/config.yaml")).expanduser(),
        pathlib.Path("config.yaml"),
    ):
        if config_path.exists():
            try:
                for line in config_path.read_text().splitlines():
                    if line.startswith("log-dir:"):
                        log_dir = line.split("log-dir:", 1)[1].strip()
                        if log_dir:
                            return pathlib.Path(log_dir).expanduser()
            except Exception:
                pass
    return pathlib.Path("~/.cache/llmup/logs").expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print request log files as pretty JSON by default"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print each row on a single line instead of pretty-printed JSON",
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        type=pathlib.Path,
        default=default_log_dir(),
        help="Path to the request log directory. Defaults to ~/.cache/llmup/logs",
    )
    return parser


def iter_rows(log_dir: pathlib.Path):
    requests_dir = log_dir / REQUEST_LOGS_DIRNAME
    for index, path in enumerate(sorted(requests_dir.glob("*.json")), start=1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        yield (
            index,
            str(payload.get("input", "")),
            str(payload.get("output", "")),
            str(payload.get("endpoint_used", "")),
        )


def parse_json_text(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def indent_block(lines: list[str], prefix: str = "  ") -> list[str]:
    return [f"{prefix}{line}" if line else prefix.rstrip() for line in lines]


def render_message_content(content, indent: str = "      ") -> list[str]:
    if isinstance(content, str):
        wrapped = textwrap.wrap(content, width=88) or [""]
        return [f"{indent}content: {wrapped[0]}"] + [f"{indent}{line}" for line in wrapped[1:]]
    if isinstance(content, list):
        lines = [f"{indent}content:"]
        for item in content:
            if isinstance(item, dict):
                lines.extend(indent_block(json.dumps(item, indent=2).splitlines(), f"{indent}  "))
            else:
                lines.append(f"{indent}  - {item}")
        return lines
    if isinstance(content, dict):
        return [f"{indent}content:"] + indent_block(json.dumps(content, indent=2).splitlines(), f"{indent}  ")
    return [f"{indent}content: {content}"]


def render_messages(messages: list[dict]) -> list[str]:
    lines = ["  messages:"]
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "unknown")
        lines.append(f"    [{index}] {role}")
        if "name" in message:
            lines.append(f"      name: {message['name']}")
        if "content" in message:
            lines.extend(render_message_content(message["content"]))
        remaining = {k: v for k, v in message.items() if k not in {"role", "name", "content"}}
        if remaining:
            lines.append("      extra:")
            lines.extend(indent_block(json.dumps(remaining, indent=2).splitlines(), "        "))
    return lines


def render_json_section(label: str, raw_text: str) -> list[str]:
    parsed = parse_json_text(raw_text)
    if parsed is None:
        return [f"{label}:", f"  {raw_text}"]

    if isinstance(parsed, dict) and isinstance(parsed.get("messages"), list):
        remaining = {k: v for k, v in parsed.items() if k != "messages"}
        lines = [f"{label}:"]
        if remaining:
            lines.extend(indent_block(json.dumps(remaining, indent=2).splitlines()))
        lines.extend(render_messages(parsed["messages"]))
        return lines

    return [f"{label}:"] + indent_block(json.dumps(parsed, indent=2).splitlines())


def render_row_pretty(row: tuple[int, str, str, str]) -> str:
    lines = [
        f"Row {row[0]}",
        f"endpoint_used: {row[3]}",
        *render_json_section("input", row[1]),
        *render_json_section("output", row[2]),
    ]
    return "\n".join(lines)


def render_row_raw(row: tuple[int, str, str, str]) -> str:
    return json.dumps(
        {
            "id": row[0],
            "input": row[1],
            "output": row[2],
            "endpoint_used": row[3],
        }
    )


def supports_interactive_navigation() -> bool:
    return (
        termios is not None
        and tty is not None
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )


def read_key() -> str:
    file_descriptor = sys.stdin.fileno()
    old_settings = termios.tcgetattr(file_descriptor)  # pyright: ignore[reportOptionalMemberAccess]
    try:
        tty.setraw(file_descriptor)  # pyright: ignore[reportOptionalMemberAccess]
        first = sys.stdin.read(1)
        if first == "G":
            return "end"
        if first == "g":
            flags = fcntl.fcntl(file_descriptor, fcntl.F_GETFL)
            fcntl.fcntl(file_descriptor, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            try:
                next_char = sys.stdin.read(1)
            except (BlockingIOError, OSError):
                next_char = ""
            finally:
                fcntl.fcntl(file_descriptor, fcntl.F_SETFL, flags)
            if next_char == "g":
                return "home"
            return "g"
        if first != "\x1b":
            return first
        second = sys.stdin.read(1)
        third = sys.stdin.read(1)
        if second == "[" and third == "A":
            return "up"
        if second == "[" and third == "B":
            return "down"
        if second == "O" and third == "F":
            return "end"
        return first + second + third
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, old_settings)  # pyright: ignore[reportOptionalMemberAccess]


def browse_rows(rows: list[tuple[int, str, str, str]]) -> int:
    index = 0
    while True:
        os.system("clear")
        print(render_row_pretty(rows[index]))
        print()
        print(
            f"Navigation: {index + 1}/{len(rows)}"
            "  Enter/down/j next  up/k previous  gg top  G bottom  q quit"
        )
        key = read_key()
        if key in {"q", "Q", "\x03"}:
            break
        if key in {"\r", "\n", " ", "j", "J", "down"} and index < len(rows) - 1:
            index += 1
        if key in {"k", "K", "up"} and index > 0:
            index -= 1
        if key == "home":
            index = 0
        if key == "end":
            index = len(rows) - 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        rows = list(iter_rows(args.log_dir))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to read {args.log_dir}: {exc}", file=sys.stderr)
        return 1

    if args.raw:
        for row in rows:
            print(render_row_raw(row))
        return 0

    if supports_interactive_navigation() and rows:
        return browse_rows(rows)

    for index, row in enumerate(rows):
        if index:
            print()
        print(render_row_pretty(row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
