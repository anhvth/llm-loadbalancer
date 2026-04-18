from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

from keep_connection import iter_commands, launch_in_tmux, parse_config
from load_balancer import serve_forever

DEFAULT_CONFIG = """# Replace the worker host pattern and upstream port start for your cluster.
endpoints:
  - hosts: worker-[1,2]
  - port-start: 18000

port:
  - 8001

load-balancer:
  workers: 20
  worker-concurrency: 512
  max-connections: 20000
  max-keepalive-connections: 4096
  upstream-timeout: 300
  log-dir: ~/.cache/llmup/logs
  affinity-db: ~/.cache/llmup/affinity.sqlite3
"""


def default_config_path() -> pathlib.Path:
    return pathlib.Path("~/.cache/llmup/config.yaml").expanduser()


def ensure_config_exists(config_path: pathlib.Path) -> None:
    if config_path.exists():
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(DEFAULT_CONFIG)


def open_config_in_editor(config_path: pathlib.Path) -> int:
    ensure_config_exists(config_path)

    editor = os.environ.get("EDITOR")
    if editor:
        return subprocess.call([editor, str(config_path)])

    if sys.platform == "darwin":
        return subprocess.call(["open", "-t", str(config_path)])

    for fallback_editor in ("nano", "vi"):
        try:
            return subprocess.call([fallback_editor, str(config_path)])
        except FileNotFoundError:
            continue

    print(f"Config file is at {config_path}", file=sys.stderr)
    return 0


def start_everything(config_path: pathlib.Path, verbose: bool = False) -> int:
    ensure_config_exists(config_path)
    cfg = parse_config(config_path)
    commands = list(iter_commands(cfg))
    launch_in_tmux(cfg.tmux_session_name, commands)
    print(f"Started SSH tunnels in tmux session: {cfg.tmux_session_name}", file=sys.stderr)
    serve_forever(config_path, verbose=verbose)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the SSH tunnel pool and the load balancer"
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=default_config_path(),
        help="Path to config.yaml. Defaults to ~/.cache/llmup/config.yaml",
    )
    parser.add_argument(
        "--set-config",
        action="store_true",
        help="Open the config file in your editor and exit",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress printing JSON request/response files to the local cache directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.set_config:
        return open_config_in_editor(args.config)

    try:
        return start_everything(args.config, verbose=not args.silent)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
