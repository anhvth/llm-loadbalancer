from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

from loguru import logger

from keep_connection import iter_commands, launch_in_tmux, parse_config
from load_balancer import serve_forever

logger.remove()


def _log_sink(message):
    sys.stderr.write(str(message))


logger.add(_log_sink, format="{message}", level="INFO")

DEFAULT_CONFIG = """# Replace the worker host entries with your actual SSH targets.
endpoints:
  - worker-[41,45,49,53-54,57,59]:8000

port:
  - 8001

load-balancer:
  workers: 20
  worker-concurrency: 512
  health-path: /models
  log-dir: ~/.cache/llm-proxy/logs
  affinity-db: ~/.cache/llm-proxy/affinity.sqlite3

port-start: 18000
"""


def default_config_path() -> pathlib.Path:
    return pathlib.Path("~/.config/llm-proxy.yaml").expanduser()


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

    logger.info("Config file is at {}", config_path)
    return 0


def format_config_table(config_path: pathlib.Path, cfg) -> str:
    endpoint_rows = [
        f"{host}:{remote_port}"
        for host, remote_port in zip(cfg.hosts, cfg.remote_ports or [cfg.remote_port] * len(cfg.hosts))
    ]
    rows = [
        ("config-path", str(config_path)),
        ("endpoints", ", ".join(endpoint_rows) if endpoint_rows else "-"),
        ("listen-port", str(cfg.listen_port)),
        ("port-start", str(cfg.port_start)),
        ("tmux-session", cfg.tmux_session_name),
        ("ssh-user", cfg.user or "-"),
        ("ssh-options", " ".join(cfg.ssh_options) if cfg.ssh_options else "-"),
        ("lb-workers", str(cfg.load_balancer_workers)),
        ("lb-worker-concurrency", str(cfg.load_balancer_worker_concurrency)),
        ("lb-health-path", cfg.load_balancer_health_path),
        ("lb-log-dir", str(cfg.load_balancer_log_dir)),
        ("lb-affinity-db", str(cfg.load_balancer_affinity_db_path)),
    ]
    key_width = max(len("key"), max(len(key) for key, _ in rows))
    value_width = max(len("value"), max(len(value) for _, value in rows))
    separator = f"+-{'-' * key_width}-+-{'-' * value_width}-+"
    lines = [
        separator,
        f"| {'key'.ljust(key_width)} | {'value'.ljust(value_width)} |",
        separator,
    ]
    for key, value in rows:
        lines.append(f"| {key.ljust(key_width)} | {value.ljust(value_width)} |")
    lines.append(separator)
    return "\n".join(lines)


def start_everything(config_path: pathlib.Path, verbose: bool = False) -> int:
    ensure_config_exists(config_path)
    cfg = parse_config(config_path)
    logger.info("Using config file: {}", config_path)
    logger.info("Loaded config:\n{}", format_config_table(config_path, cfg))
    commands = list(iter_commands(cfg))
    launch_in_tmux(cfg.tmux_session_name, commands)
    logger.info("Started SSH tunnels in tmux session: {}", cfg.tmux_session_name)
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
        help="Path to config file. Defaults to ~/.config/llm-proxy.yaml",
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
