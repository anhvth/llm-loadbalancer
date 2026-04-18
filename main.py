from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

from keep_connection import iter_commands, launch_in_tmux, parse_config
from load_balancer import serve_forever

DEFAULT_CONFIG = """# 1. ssh to the target server and forward the port
# ssh -L 8000:localhost:8000 user@target_server_ip
endpoints:
  - hosts: worker-[41,45,49,53-54,57,59,61] # copy from login node
  - port-start: 18000

tmux:
  session-name: keepssh

port:
  - 8001

load-balancer:
  workers: 20
  worker-concurrency: 512
  max-connections: 20000
  max-keepalive-connections: 4096
  upstream-timeout: 300
  db-path: llm_loadbalancer.sqlite3
"""


def default_config_path() -> pathlib.Path:
    return pathlib.Path.cwd() / "config.yaml"


def ensure_config_exists(config_path: pathlib.Path) -> None:
    if config_path.exists():
        return
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
        help="Path to config.yaml. Defaults to ./config.yaml in the current directory",
    )
    parser.add_argument(
        "--set-config",
        action="store_true",
        help="Open the config file in your editor and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each JSON request/response row when it is inserted into the local DB",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.set_config:
        return open_config_in_editor(args.config)

    try:
        return start_everything(args.config, verbose=args.verbose)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
