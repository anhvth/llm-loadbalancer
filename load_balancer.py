#!/usr/bin/env python3
"""Async streaming reverse proxy for OpenAI-like APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import queue
import random
import sqlite3
import sys
import threading
import time
from typing import Any
import uuid

import httpx
import uvicorn

from keep_connection import TunnelConfig, parse_config

HOP_BY_HOP_HEADERS = {
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}

MESSAGE_AFFINITY_CACHE_SIZE = 4096
MESSAGE_AFFINITY_IGNORED_KEYS = {"cache_control", "role", "signature", "type"}
REQUEST_LOGS_DIRNAME = "requests"


class AsyncFileLogWriter:
    def __init__(self, root_dir: pathlib.Path, verbose: bool = False):
        self.root_dir = root_dir
        self.verbose = verbose
        self.requests_dir = root_dir / REQUEST_LOGS_DIRNAME
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="request-log-writer", daemon=True)
        self._thread.start()

    def submit(self, payload: dict[str, Any]) -> None:
        self._queue.put_nowait(payload)

    def close(self) -> None:
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join()
        self._thread = None

    def _run(self) -> None:
        while True:
            payload = self._queue.get()
            try:
                if payload is None:
                    return
                path = self._write_payload(payload)
                if self.verbose:
                    self._print_logged_payload(path, payload)
            finally:
                self._queue.task_done()

    def _write_payload(self, payload: dict[str, Any]) -> pathlib.Path:
        file_name = f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex}.json"
        final_path = self.requests_dir / file_name
        tmp_path = final_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(final_path)
        return final_path

    def _print_logged_payload(self, path: pathlib.Path, payload: dict[str, Any]) -> None:
        print(
            "[request_response_log] "
            f"log_path={path} "
            f"endpoint={payload.get('endpoint_used', '')}",
            file=sys.stderr,
        )


class SqliteMessageAffinityStore:
    def __init__(self, path: pathlib.Path, max_entries: int):
        self.path = path
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        with self._lock:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS message_affinity (
                    prefix_key TEXT PRIMARY KEY,
                    upstream_port INTEGER NOT NULL,
                    last_access_ns INTEGER NOT NULL
                )
                """
            )

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def get(self, prefix_key: str) -> int | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT upstream_port FROM message_affinity WHERE prefix_key = ?",
                (prefix_key,),
            ).fetchone()
            if row is None:
                return None
            self.connection.execute(
                "UPDATE message_affinity SET last_access_ns = ? WHERE prefix_key = ?",
                (time.time_ns(), prefix_key),
            )
            return int(row[0])

    def set(self, prefix_key: str, upstream_port: int) -> None:
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO message_affinity(prefix_key, upstream_port, last_access_ns)
                VALUES(?, ?, ?)
                ON CONFLICT(prefix_key) DO UPDATE SET
                    upstream_port = excluded.upstream_port,
                    last_access_ns = excluded.last_access_ns
                """,
                (prefix_key, upstream_port, time.time_ns()),
            )
            self.connection.execute(
                """
                DELETE FROM message_affinity
                WHERE prefix_key IN (
                    SELECT prefix_key
                    FROM message_affinity
                    ORDER BY last_access_ns DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self.max_entries,),
            )


def build_upstream_ports(cfg: TunnelConfig) -> list[int]:
    return [cfg.port_start + index for index in range(len(cfg.hosts))]


class LoadBalancerApp:
    def __init__(self, cfg: TunnelConfig, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.upstream_ports = build_upstream_ports(cfg)
        self.client: httpx.AsyncClient | None = None
        self.message_affinity = SqliteMessageAffinityStore(
            cfg.load_balancer_affinity_db_path,
            MESSAGE_AFFINITY_CACHE_SIZE,
        )
        self.log_writer = AsyncFileLogWriter(cfg.load_balancer_log_dir, verbose=verbose)

    async def __call__(self, scope, receive, send) -> None:
        scope_type = scope["type"]
        if scope_type == "lifespan":
            await self._handle_lifespan(receive, send)
            return
        if scope_type != "http":
            await send({"type": "http.response.start", "status": 500, "headers": []})
            await send({"type": "http.response.body", "body": b"Unsupported scope"})
            return

        await self._handle_http(scope, receive, send)

    async def _handle_lifespan(self, receive, send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                self.client = self._build_client()
                self.log_writer.start()
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if self.client is not None:
                    await self.client.aclose()
                    self.client = None
                self.message_affinity.close()
                self.log_writer.close()
                await send({"type": "lifespan.shutdown.complete"})
                return

    def _build_client(self) -> httpx.AsyncClient:
        limits = httpx.Limits(
            max_connections=self.cfg.load_balancer_max_connections,
            max_keepalive_connections=self.cfg.load_balancer_max_keepalive_connections,
        )
        timeout = httpx.Timeout(
            connect=self.cfg.load_balancer_upstream_timeout,
            read=self.cfg.load_balancer_upstream_timeout,
            write=self.cfg.load_balancer_upstream_timeout,
            pool=self.cfg.load_balancer_upstream_timeout,
        )
        return httpx.AsyncClient(timeout=timeout, limits=limits)

    async def _handle_http(self, scope, receive, send) -> None:
        if not self.upstream_ports:
            await self._send_plain_error(send, 503, b"No upstream ports configured")
            return

        client = self.client or self._build_client()
        method = scope["method"]
        path = scope.get("raw_path", scope["path"].encode("utf-8")).decode("utf-8")
        query_string = scope.get("query_string", b"")
        if query_string:
            path = f"{path}?{query_string.decode('latin1')}"

        request_chunks = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                continue
            body = message.get("body", b"")
            if body:
                request_chunks.extend(body)
            if not message.get("more_body", False):
                break

        request_body = bytes(request_chunks)
        upstream_port = self._choose_upstream_port(request_body)
        upstream_url = f"http://127.0.0.1:{upstream_port}{path}"
        headers = self._build_upstream_headers(scope["headers"], upstream_port)
        request = client.build_request(method, upstream_url, headers=headers, content=request_body)

        try:
            response = await client.send(request, stream=True)
        except httpx.HTTPError as exc:
            await self._send_plain_error(send, 502, f"Upstream request failed: {exc}".encode("utf-8"))
            return

        response_headers = [
            (key.encode("latin1"), value.encode("latin1"))
            for key, value in response.headers.items()
            if key.lower().encode("latin1") not in HOP_BY_HOP_HEADERS
        ]
        response_headers.append((b"connection", b"close"))
        response_chunks = bytearray()

        await send(
            {
                "type": "http.response.start",
                "status": response.status_code,
                "headers": response_headers,
            }
        )

        try:
            async for chunk in response.aiter_raw():
                if chunk:
                    response_chunks.extend(chunk)
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
        finally:
            await response.aclose()

        await send({"type": "http.response.body", "body": b"", "more_body": False})
        self._remember_messages_affinity(request_body, upstream_port)
        self._log_exchange(
            request_body,
            bytes(response_chunks),
            upstream_url,
            response.headers.get("content-type"),
        )

    def _build_upstream_headers(self, headers, upstream_port: int) -> list[tuple[str, str]]:
        forwarded_headers: list[tuple[str, str]] = []
        for raw_key, raw_value in headers:
            key = raw_key.lower()
            if key in HOP_BY_HOP_HEADERS or key == b"host":
                continue
            forwarded_headers.append((raw_key.decode("latin1"), raw_value.decode("latin1")))
        forwarded_headers.append(("host", f"127.0.0.1:{upstream_port}"))
        return forwarded_headers

    async def _send_plain_error(self, send, status: int, body: bytes) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    def _decode_body_text(self, body: bytes) -> str:
        if not body:
            return ""
        return body.decode("utf-8")

    def _decode_json_value(self, body_text: str):
        try:
            return json.loads(body_text)
        except json.JSONDecodeError:
            return None

    def _choose_upstream_port(self, request_body: bytes) -> int:
        affinity_port = self._find_messages_affinity_port(request_body)
        if affinity_port is not None:
            return affinity_port
        return random.choice(self.upstream_ports)

    def _find_messages_affinity_port(self, request_body: bytes) -> int | None:
        payload = self._decode_json_value(self._decode_body_text(request_body))
        if not isinstance(payload, dict):
            return None
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return None
        if len(messages) <= 1:
            return None

        for prefix_key in reversed(self._messages_prefix_keys(messages[:-1])):
            cached_port = self.message_affinity.get(prefix_key)
            if cached_port is not None:
                return cached_port
        return None

    def _remember_messages_affinity(self, request_body: bytes, upstream_port: int) -> None:
        payload = self._decode_json_value(self._decode_body_text(request_body))
        if not isinstance(payload, dict):
            return
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return

        prefix_key = self._messages_prefix_keys(messages)[-1]
        self._cache_messages_affinity(prefix_key, upstream_port)

    def _cache_messages_affinity(self, prefix_key: str, upstream_port: int) -> None:
        self.message_affinity.set(prefix_key, upstream_port)

    def _messages_prefix_key(self, messages: list[Any]) -> str:
        return self._messages_prefix_keys(messages)[-1]

    def _messages_prefix_keys(self, messages: list[Any]) -> list[str]:
        hasher = hashlib.sha256()
        prefix_keys: list[str] = []
        for index, message in enumerate(messages):
            if index:
                hasher.update(b"\n")
            hasher.update(self._normalize_message_for_affinity(message).encode("utf-8"))
            prefix_keys.append(hasher.copy().hexdigest())
        return prefix_keys

    def _normalize_message_for_affinity(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        if isinstance(value, list):
            return "\x1e".join(
                part for item in value if (part := self._normalize_message_for_affinity(item))
            )
        if isinstance(value, dict):
            return "\x1f".join(
                part
                for key in sorted(value)
                if key not in MESSAGE_AFFINITY_IGNORED_KEYS
                if (part := self._normalize_message_for_affinity(value[key]))
            )
        return repr(value)

    def _normalize_sse_payload(self, response_text: str) -> str | None:
        parsed_events: list[dict[str, Any]] = []
        for chunk in response_text.split("\n\n"):
            chunk = chunk.strip()
            if not chunk:
                continue
            data_lines = [line[5:].lstrip() for line in chunk.splitlines() if line.startswith("data:")]
            if not data_lines:
                return None
            payload = "\n".join(data_lines).strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                parsed_events.append(json.loads(payload))
            except json.JSONDecodeError:
                return None

        if not parsed_events:
            return None

        anthropic_payload = self._normalize_anthropic_stream(parsed_events)
        if anthropic_payload is not None:
            return anthropic_payload

        openai_payload = self._normalize_openai_stream(parsed_events)
        if openai_payload is not None:
            return openai_payload

        return None

    def _normalize_anthropic_stream(self, events: list[dict[str, Any]]) -> str | None:
        message: dict[str, Any] | None = None
        content_blocks: list[dict[str, Any]] = []

        for event in events:
            event_type = event.get("type")
            if event_type == "message_start":
                raw_message = event.get("message")
                if not isinstance(raw_message, dict):
                    return None
                message = dict(raw_message)
                content_blocks = [dict(block) for block in raw_message.get("content", []) if isinstance(block, dict)]
                message["content"] = content_blocks
                continue

            if message is None:
                continue

            if event_type == "content_block_start":
                index = event.get("index")
                block = event.get("content_block")
                if not isinstance(index, int) or not isinstance(block, dict):
                    return None
                while len(content_blocks) <= index:
                    content_blocks.append({})
                content_blocks[index] = dict(block)
                continue

            if event_type == "content_block_delta":
                index = event.get("index")
                delta = event.get("delta")
                if not isinstance(index, int) or not isinstance(delta, dict):
                    return None
                while len(content_blocks) <= index:
                    content_blocks.append({})
                block = content_blocks[index]
                if not block and isinstance(event.get("content_block"), dict):
                    block.update(event["content_block"])
                for key, value in delta.items():
                    if key == "type":
                        continue
                    if isinstance(value, str) and isinstance(block.get(key), str):
                        block[key] += value
                    else:
                        block[key] = value
                continue

            if event_type == "message_delta":
                delta = event.get("delta")
                if isinstance(delta, dict):
                    for key, value in delta.items():
                        message[key] = value
                usage = event.get("usage")
                if isinstance(usage, dict):
                    merged_usage = dict(message.get("usage", {}))
                    merged_usage.update(usage)
                    message["usage"] = merged_usage
                continue

            if event_type == "message_stop":
                break

        if message is None:
            return None
        message["content"] = content_blocks
        return json.dumps(message)

    def _normalize_openai_stream(self, events: list[dict[str, Any]]) -> str | None:
        completion: dict[str, Any] | None = None
        choices_by_index: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] | None = None

        for event in events:
            choices = event.get("choices")
            if not isinstance(choices, list):
                continue

            if completion is None:
                completion = {key: value for key, value in event.items() if key != "choices"}
                if isinstance(completion.get("object"), str) and completion["object"].endswith(".chunk"):
                    completion["object"] = completion["object"][:-6]

            for choice in choices:
                if not isinstance(choice, dict):
                    return None
                index = choice.get("index")
                if not isinstance(index, int):
                    return None
                merged_choice = choices_by_index.setdefault(index, {"index": index, "message": {}})
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    message = merged_choice.setdefault("message", {})
                    for key, value in delta.items():
                        if isinstance(value, str) and isinstance(message.get(key), str):
                            message[key] += value
                        elif value is not None:
                            message[key] = value
                if choice.get("finish_reason") is not None:
                    merged_choice["finish_reason"] = choice["finish_reason"]

            if isinstance(event.get("usage"), dict):
                usage = dict(event["usage"])

        if completion is None:
            return None

        ordered_choices = [choices_by_index[index] for index in sorted(choices_by_index)]
        if not ordered_choices:
            return None
        completion["choices"] = ordered_choices
        if usage is not None:
            completion["usage"] = usage
        return json.dumps(completion)

    def _normalize_logged_response(self, response_text: str, content_type: str | None) -> str:
        if "text/event-stream" not in (content_type or "").lower():
            return response_text
        normalized = self._normalize_sse_payload(response_text)
        return normalized if normalized is not None else response_text

    def _log_exchange(
        self,
        request_body: bytes,
        response_body: bytes,
        endpoint_used: str,
        content_type: str | None = None,
    ) -> None:
        try:
            request_text = self._decode_body_text(request_body)
            response_text = self._decode_body_text(response_body)
        except UnicodeDecodeError:
            return
        response_text = self._normalize_logged_response(response_text, content_type)

        try:
            request_json = json.loads(request_text) if request_text else {}
        except json.JSONDecodeError:
            request_json = request_text

        try:
            response_json = json.loads(response_text) if response_text else {}
        except json.JSONDecodeError:
            response_json = response_text

        self.log_writer.submit(
            {
                "input": request_json,
                "output": response_json,
                "endpoint_used": endpoint_used,
            }
        )


def resolve_config_path(config_path: pathlib.Path | None = None) -> pathlib.Path:
    if config_path is not None:
        return config_path
    return pathlib.Path(os.environ.get("LLM_LOADBALANCER_CONFIG", "config.yaml"))


def create_app(config_path: pathlib.Path | None = None, verbose: bool | None = None) -> LoadBalancerApp:
    cfg = parse_config(resolve_config_path(config_path))
    if verbose is None:
        verbose = os.environ.get("LLM_LOADBALANCER_VERBOSE", "0") == "1"
    return LoadBalancerApp(cfg, verbose=verbose)


def serve_forever(config_path: pathlib.Path = pathlib.Path("config.yaml"), verbose: bool = True) -> None:
    cfg = parse_config(config_path)
    os.environ["LLM_LOADBALANCER_CONFIG"] = str(config_path)
    os.environ["LLM_LOADBALANCER_VERBOSE"] = "1" if verbose else "0"
    uvicorn.run(
        "load_balancer:create_app",
        factory=True,
        host="127.0.0.1",
        port=cfg.listen_port,
        workers=cfg.load_balancer_workers,
        limit_concurrency=cfg.load_balancer_worker_concurrency,
        backlog=4096,
        timeout_keep_alive=30,
        access_log=False,
        log_level="info",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the random worker load balancer")
    parser.add_argument(
        "-c",
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("config.yaml"),
        help="Path to the shared config file",
    )
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="Disable verbose logging to terminal",
    )
    args = parser.parse_args(argv)
    verbose = not args.silent and os.environ.get("LLM_LOADBALANCER_VERBOSE", "1") == "1"
    serve_forever(args.config, verbose=verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
