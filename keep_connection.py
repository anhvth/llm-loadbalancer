#!/usr/bin/env python3
"""Keep SSH port-forward tunnels alive from a small YAML-like config.

The current config format is intentionally simple:

endpoints:
  - hosts: worker-[41,45,49,53-54,57,59,61]
  - port-start: 18000

Optional keys:
  - remote-port: 8000
  - user: myuser
  - ssh-options: -o ServerAliveInterval=30 -o ServerAliveCountMax=3
  - tmux.session-name: keepssh

The script can print the commands it would run, or launch them and restart
failed tunnels.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import shlex
import subprocess
import sys
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class TunnelConfig:
    hosts: list[str]
    port_start: int
    remote_port: int = 8000
    listen_port: int = 8001
    load_balancer_workers: int = 20
    load_balancer_worker_concurrency: int = 512
    load_balancer_max_connections: int = 20000
    load_balancer_max_keepalive_connections: int = 4096
    load_balancer_upstream_timeout: float = 300.0
    load_balancer_log_dir: pathlib.Path = pathlib.Path("~/.cache/llmup/logs").expanduser()
    load_balancer_affinity_db_path: pathlib.Path = pathlib.Path(
        "~/.cache/llmup/affinity.sqlite3"
    ).expanduser()
    user: str | None = None
    ssh_options: list[str] = dataclasses.field(default_factory=list)
    tmux_session_name: str = "keepssh"


def strip_comment(line: str) -> str:
    if "#" not in line:
        return line.rstrip()
    return line.split("#", 1)[0].rstrip()


def expand_host_pattern(pattern: str) -> list[str]:
    match = re.fullmatch(r"([A-Za-z0-9_.-]*?)\[(.+)\]", pattern)
    if not match:
        return [pattern]

    prefix, body = match.groups()
    hosts: list[str] = []
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            width = max(len(start_s), len(end_s))
            step = 1 if end >= start else -1
            for value in range(start, end + step, step):
                hosts.append(f"{prefix}{value:0{width}d}")
        else:
            hosts.append(f"{prefix}{part}")
    return hosts


def resolve_config_relative_path(config_path: pathlib.Path, raw_path: str) -> pathlib.Path:
    candidate = pathlib.Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return config_path.parent / candidate


def parse_config(path: pathlib.Path) -> TunnelConfig:
    lines = [strip_comment(line) for line in path.read_text().splitlines()]
    lines = [line for line in lines if line.strip()]

    section: str | None = None
    root: dict[str, str] = {}
    tmux: dict[str, str] = {}
    load_balancer: dict[str, str] = {}
    ports: list[str] = []
    endpoint_items: list[tuple[str, str]] = []

    for raw_line in lines:
        line = raw_line.rstrip()
        if re.fullmatch(r"[A-Za-z0-9_.-]+:", line):
            section = line[:-1]
            continue

        match = re.fullmatch(r"\s*-\s*([A-Za-z0-9_.-]+):\s*(.+)", line)
        if match and section == "endpoints":
            endpoint_items.append((match.group(1), match.group(2).strip()))
            continue

        match = re.fullmatch(r"\s*-\s*(.+)", line)
        if match and section == "port":
            ports.append(match.group(1).strip())
            continue

        match = re.fullmatch(r"\s*-\s*(.+)", line)
        if match and section == "endpoints":
            endpoint_items.append(("hosts", match.group(1).strip()))
            continue

        match = re.fullmatch(r"\s*([A-Za-z0-9_.-]+):\s*(.+)", line)
        if match:
            if section == "tmux":
                target = tmux
            elif section == "load-balancer":
                target = load_balancer
            else:
                target = root
            target[match.group(1)] = match.group(2).strip()
            continue

        raise ValueError(f"Cannot parse line: {raw_line!r}")

    merged: dict[str, str] = {}
    if endpoint_items:
        for key, value in endpoint_items:
            merged[key] = value
    merged.update(root)

    hosts_raw = merged.get("hosts")
    if not hosts_raw:
        raise ValueError("Config is missing endpoints.hosts")
    if hosts_raw.startswith("[") and hosts_raw.endswith("]"):
        hosts = [
            item.strip().strip("'\"")
            for item in hosts_raw[1:-1].split(",")
            if item.strip()
        ]
    else:
        hosts = expand_host_pattern(hosts_raw.strip().strip("'\""))

    port_start_raw = merged.get("port-start")
    if port_start_raw is None:
        raise ValueError("Config is missing endpoints.port-start")

    remote_port = int(merged.get("remote-port", "8000"))
    listen_port = int(ports[0]) if ports else int(merged.get("listen-port", "8001"))
    load_balancer_workers = int(load_balancer.get("workers", "20"))
    load_balancer_worker_concurrency = int(load_balancer.get("worker-concurrency", "512"))
    load_balancer_max_connections = int(load_balancer.get("max-connections", "20000"))
    load_balancer_max_keepalive_connections = int(
        load_balancer.get("max-keepalive-connections", "4096")
    )
    load_balancer_upstream_timeout = float(load_balancer.get("upstream-timeout", "300"))
    load_balancer_log_dir = resolve_config_relative_path(
        path,
        load_balancer.get("log-dir", "~/.cache/llmup/logs")
    )
    load_balancer_affinity_db_path = resolve_config_relative_path(
        path,
        load_balancer.get("affinity-db", "~/.cache/llmup/affinity.sqlite3")
    )
    user = merged.get("user")
    ssh_options = shlex.split(merged.get("ssh-options", ""))
    tmux_session_name = tmux.get("session-name", "keepssh")

    return TunnelConfig(
        hosts=hosts,
        port_start=int(port_start_raw),
        remote_port=remote_port,
        listen_port=listen_port,
        load_balancer_workers=load_balancer_workers,
        load_balancer_worker_concurrency=load_balancer_worker_concurrency,
        load_balancer_max_connections=load_balancer_max_connections,
        load_balancer_max_keepalive_connections=load_balancer_max_keepalive_connections,
        load_balancer_upstream_timeout=load_balancer_upstream_timeout,
        load_balancer_log_dir=load_balancer_log_dir,
        load_balancer_affinity_db_path=load_balancer_affinity_db_path,
        user=user,
        ssh_options=ssh_options,
        tmux_session_name=tmux_session_name,
    )


def build_command(host: str, local_port: int, remote_port: int, cfg: TunnelConfig) -> list[str]:
    target = f"{cfg.user}@{host}" if cfg.user else host
    return [
        "ssh",
        *cfg.ssh_options,
        "-o",
        "ExitOnForwardFailure=yes",
        "-N",
        "-L",
        f"{local_port}:localhost:{remote_port}",
        target,
    ]


def iter_commands(cfg: TunnelConfig) -> Iterable[list[str]]:
    for index, host in enumerate(cfg.hosts):
        yield build_command(host, cfg.port_start + index, cfg.remote_port, cfg)


def tmux_quoted_command(cmd: list[str]) -> str:
    return shlex.join(cmd)


def tmux_window_command(cmd: list[str]) -> list[str]:
    quoted = tmux_quoted_command(cmd)
    return [
        "bash",
        "-lc",
        f'while true; do {quoted}; rc=$?; printf "[keep_connection] ssh exited with %s, restarting in 2s...\\n" "$rc" >&2; sleep 2; done',
    ]


def launch_in_tmux(session_name: str, commands: list[list[str]]) -> None:
    if not commands:
        return

    try:
        has_session = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        if has_session:
            subprocess.run(
                ["tmux", "kill-session", "-t", session_name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            has_session = False
    except FileNotFoundError as exc:
        raise RuntimeError("tmux is not installed or not on PATH") from exc

    for index, cmd in enumerate(commands):
        window_name = f"tunnel-{index + 1}"
        window_cmd = tmux_window_command(cmd)
        if index == 0 and not has_session:
            subprocess.run(
                [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    session_name,
                    "-n",
                    window_name,
                    *window_cmd,
                ],
                check=True,
            )
            continue

        subprocess.run(
            [
                "tmux",
                "new-window",
                "-t",
                session_name,
                "-n",
                window_name,
                *window_cmd,
            ],
            check=True,
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Keep SSH tunnels alive from config.yaml")
    parser.add_argument(
        "-c",
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("~/.cache/llmup/config.yaml").expanduser(),
        help="Path to the tunnel config",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print the SSH commands without starting tmux",
    )
    args = parser.parse_args(argv)

    cfg = parse_config(args.config)
    commands = list(iter_commands(cfg))

    for cmd in commands:
        print(shlex.join(cmd))

    if args.print_only:
        return 0

    try:
        launch_in_tmux(cfg.tmux_session_name, commands)
        print(f"Started tunnels in tmux session: {cfg.tmux_session_name}", file=sys.stderr)
    except KeyboardInterrupt:
        return 130
    except subprocess.CalledProcessError as exc:
        print(f"Failed to start tmux command: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
