#!/usr/bin/env python3
"""Async streaming reverse proxy for OpenAI-like APIs."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import fcntl
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
from loguru import logger
from tabulate import tabulate
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
UPSTREAM_TIMEOUT_SECONDS = 300.0
HEALTHCHECK_TIMEOUT_SECONDS = 2.0
HEALTHCHECK_INTERVAL_SECONDS = 30.0
HEALTHCHECK_CONNECT_RETRIES = 3
HEALTHCHECK_CONNECT_RETRY_DELAY_SECONDS = 0.2
HEALTHCHECK_STATE_FILENAME = "health_state.json"
REQUEST_RESPONSE_LOG_MIN_INTERVAL_SECONDS = 2.0

logger.remove()


def _log_sink(message):
    sys.stderr.write(str(message))


logger.add(_log_sink, format="{message}", level="INFO")


@dataclasses.dataclass(frozen=True)
class EndpointCheckResult:
    host: str
    port: int
    models: tuple[str, ...] = ()
    error: str | None = None


class AsyncFileLogWriter:
    def __init__(self, root_dir: pathlib.Path, verbose: bool = False):
        self.root_dir = root_dir
        self.verbose = verbose
        self.requests_dir = root_dir / REQUEST_LOGS_DIRNAME
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._last_verbose_log_at: float | None = None

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
        now = time.monotonic()
        if (
            self._last_verbose_log_at is not None
            and now - self._last_verbose_log_at < REQUEST_RESPONSE_LOG_MIN_INTERVAL_SECONDS
        ):
            return
        self._last_verbose_log_at = now
        logger.info(
            "[request_response_log] log_path {} endpoint={} route={}",
            path,
            payload.get("endpoint_used", ""),
            payload.get("route_reason", ""),
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


def build_upstream_endpoints(cfg: TunnelConfig) -> list[tuple[str, int]]:
    return [(host, cfg.port_start + index) for index, host in enumerate(cfg.hosts)]


def build_upstream_endpoint_labels(cfg: TunnelConfig) -> dict[tuple[str, int], str]:
    remote_ports = cfg.remote_ports or [cfg.remote_port] * len(cfg.hosts)
    labels: dict[tuple[str, int], str] = {}
    for index, (host, remote_port) in enumerate(zip(cfg.hosts, remote_ports)):
        local_port = cfg.port_start + index
        key = (host, local_port)
        if local_port == remote_port:
            labels[key] = f"{host}:{local_port}"
        else:
            labels[key] = f"{host}:{remote_port} (local {local_port})"
    return labels


class LoadBalancerApp:
    def __init__(self, cfg: TunnelConfig, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.upstream_endpoints = build_upstream_endpoints(cfg)
        self.upstream_endpoint_labels = build_upstream_endpoint_labels(cfg)
        self.endpoint_request_counts: dict[str, int] = {
            label: 0 for label in self.upstream_endpoint_labels.values()
        }
        self._endpoint_label_by_port: dict[int, str] = {
            endpoint[1]: label for endpoint, label in self.upstream_endpoint_labels.items()
        }
        self._health_state_path = self.cfg.load_balancer_log_dir / HEALTHCHECK_STATE_FILENAME
        self._last_health_digest: str | None = None
        self._last_health_snapshot: dict[str, Any] | None = None
        self.valid_endpoints = self._initial_healthcheck()
        if not self.valid_endpoints:
            raise RuntimeError("No valid endpoints after healthcheck")
        self.client: httpx.AsyncClient | None = None
        self.message_affinity = SqliteMessageAffinityStore(
            cfg.load_balancer_affinity_db_path,
            MESSAGE_AFFINITY_CACHE_SIZE,
        )
        self.log_writer = AsyncFileLogWriter(cfg.load_balancer_log_dir, verbose=verbose)
        self._healthcheck_task: asyncio.Task[None] | None = None
        self._healthcheck_stop = asyncio.Event()

    def _endpoint_label(self, endpoint: EndpointCheckResult) -> str:
        return self.upstream_endpoint_labels.get(
            (endpoint.host, endpoint.port),
            f"{endpoint.host}:{endpoint.port}",
        )

    def _endpoint_url(self, endpoint: tuple[str, int]) -> str:
        _, port = endpoint
        return f"http://127.0.0.1:{port}{self.cfg.load_balancer_health_path}"

    def _endpoint_probe_urls(self, endpoint: tuple[str, int]) -> list[str]:
        _, port = endpoint
        base = f"http://127.0.0.1:{port}"
        paths = [self.cfg.load_balancer_health_path]
        if self.cfg.load_balancer_health_path != "/v1/models":
            paths.append("/v1/models")
        return [f"{base}{path}" for path in paths]

    def _models_from_payload(self, payload: Any) -> tuple[str, ...]:
        if not isinstance(payload, dict):
            return ()

        models: list[str] = []
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                model_id = item.get("id")
                if isinstance(model_id, str):
                    models.append(model_id)
                model_name = item.get("model")
                if isinstance(model_name, str):
                    models.append(model_name)
        if models:
            return tuple(models)

        raw_models = payload.get("models")
        if isinstance(raw_models, list):
            for item in raw_models:
                if isinstance(item, str):
                    models.append(item)
                elif isinstance(item, dict):
                    model_id = item.get("id")
                    if isinstance(model_id, str):
                        models.append(model_id)
                    model_name = item.get("model")
                    if isinstance(model_name, str):
                        models.append(model_name)
        return tuple(models)

    def _normalize_health_error(self, error: str) -> str:
        if "Connection refused" in error or "All connection attempts failed" in error:
            return "Connection refused"
        return error

    def _is_retryable_health_error(self, error: httpx.HTTPError) -> bool:
        if isinstance(error, httpx.ConnectError):
            return True
        text = str(error)
        return "Connection refused" in text or "All connection attempts failed" in text

    def _stateful_health_snapshot(
        self, results: list[EndpointCheckResult]
    ) -> tuple[dict[str, Any] | None, dict[str, Any], bool]:
        valid_results = [result for result in results if result.error is None]
        snapshot = {
            "connected": len(valid_results),
            "models": sorted({model for result in valid_results for model in result.models}),
            "endpoints": {},
        }
        for result in sorted(results, key=lambda item: (item.host, item.port)):
            label = self._endpoint_label(result)
            requests_served = self.endpoint_request_counts.get(label, 0)
            if result.error is None:
                snapshot["endpoints"][label] = {
                    "status": "up",
                    "models": sorted(set(result.models)),
                    "requests": requests_served,
                }
            else:
                snapshot["endpoints"][label] = {
                    "status": "down",
                    "requests": requests_served,
                    "error": self._normalize_health_error(result.error),
                }

        serialized = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        self._health_state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"digest": digest, "snapshot": snapshot}

        try:
            with self._health_state_path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.seek(0)
                raw = handle.read().strip()
                previous: dict[str, Any] = {}
                if raw:
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            previous = parsed
                    except json.JSONDecodeError:
                        previous = {}
                previous_snapshot = previous.get("snapshot")
                if (
                    isinstance(previous_snapshot, dict)
                    and previous.get("digest") == digest
                ):
                    return previous_snapshot, snapshot, False

                handle.seek(0)
                handle.truncate()
                handle.write(json.dumps(payload, sort_keys=True))
                handle.flush()
                os.fsync(handle.fileno())
                if isinstance(previous_snapshot, dict):
                    return previous_snapshot, snapshot, True
                return None, snapshot, True
        except OSError:
            if self._last_health_digest == digest and self._last_health_snapshot is not None:
                return self._last_health_snapshot, snapshot, False
            previous_snapshot = self._last_health_snapshot
            self._last_health_digest = digest
            self._last_health_snapshot = snapshot
            return previous_snapshot, snapshot, True

    def _probe_models_sync(self, client: httpx.Client, endpoint: tuple[str, int]) -> tuple[tuple[str, ...], str | None]:
        models: tuple[str, ...] = ()
        server_error: str | None = None
        for url in self._endpoint_probe_urls(endpoint):
            response = client.get(url)
            if response.status_code >= 500:
                server_error = f"Server error '{response.status_code}' for url '{response.request.url}'"
                continue
            server_error = None
            if response.status_code >= 400:
                continue
            try:
                models = self._models_from_payload(response.json())
            except (ValueError, json.JSONDecodeError):
                models = ()
            if models:
                break
        return models, server_error

    async def _probe_models_async(
        self, client: httpx.AsyncClient, endpoint: tuple[str, int]
    ) -> tuple[tuple[str, ...], str | None]:
        models: tuple[str, ...] = ()
        server_error: str | None = None
        for url in self._endpoint_probe_urls(endpoint):
            response = await client.get(url)
            if response.status_code >= 500:
                server_error = f"Server error '{response.status_code}' for url '{response.request.url}'"
                continue
            server_error = None
            if response.status_code >= 400:
                continue
            try:
                models = self._models_from_payload(response.json())
            except (ValueError, json.JSONDecodeError):
                models = ()
            if models:
                break
        return models, server_error

    def _check_endpoint_sync(self, client: httpx.Client, endpoint: tuple[str, int]) -> EndpointCheckResult:
        host, port = endpoint
        retries = max(0, HEALTHCHECK_CONNECT_RETRIES)
        for attempt in range(retries + 1):
            try:
                models, server_error = self._probe_models_sync(client, endpoint)
                if server_error is not None:
                    return EndpointCheckResult(
                        host=host,
                        port=port,
                        error=server_error,
                    )
                return EndpointCheckResult(host=host, port=port, models=models)
            except httpx.HTTPError as exc:
                if attempt >= retries or not self._is_retryable_health_error(exc):
                    return EndpointCheckResult(host=host, port=port, error=str(exc))
                time.sleep(max(0.0, HEALTHCHECK_CONNECT_RETRY_DELAY_SECONDS))
        return EndpointCheckResult(host=host, port=port, error="Healthcheck failed")

    async def _check_endpoint_async(
        self,
        client: httpx.AsyncClient,
        endpoint: tuple[str, int],
    ) -> EndpointCheckResult:
        host, port = endpoint
        retries = max(0, HEALTHCHECK_CONNECT_RETRIES)
        for attempt in range(retries + 1):
            try:
                models, server_error = await self._probe_models_async(client, endpoint)
                if server_error is not None:
                    return EndpointCheckResult(
                        host=host,
                        port=port,
                        error=server_error,
                    )
                return EndpointCheckResult(host=host, port=port, models=models)
            except httpx.HTTPError as exc:
                if attempt >= retries or not self._is_retryable_health_error(exc):
                    return EndpointCheckResult(host=host, port=port, error=str(exc))
                await asyncio.sleep(max(0.0, HEALTHCHECK_CONNECT_RETRY_DELAY_SECONDS))
        return EndpointCheckResult(host=host, port=port, error="Healthcheck failed")

    def _summarize_health(self, results: list[EndpointCheckResult]) -> None:
        _, current_snapshot, changed = self._stateful_health_snapshot(results)
        if not changed:
            return
        endpoints = current_snapshot["endpoints"]
        headers = ["endpoint", "status", "models", "requests", "error"]
        if not endpoints:
            logger.info("{}", tabulate([], headers=headers, tablefmt="simple"))
            return
        rows = [
            [
                endpoint,
                state["status"].upper(),
                ",".join(state.get("models", ())) if state.get("models") else "-",
                state.get("requests", 0),
                state.get("error", "-"),
            ]
            for endpoint, state in endpoints.items()
        ]
        table = tabulate(
            rows,
            headers=headers,
            tablefmt="simple",
        )
        logger.info("{}", table)

    def _initial_healthcheck(self) -> list[EndpointCheckResult]:
        with httpx.Client(timeout=HEALTHCHECK_TIMEOUT_SECONDS) as client:
            results = [self._check_endpoint_sync(client, endpoint) for endpoint in self.upstream_endpoints]
        self._summarize_health(results)
        return [result for result in results if result.error is None]

    async def refresh_health(self) -> None:
        async with httpx.AsyncClient(timeout=HEALTHCHECK_TIMEOUT_SECONDS) as client:
            results = await asyncio.gather(
                *(self._check_endpoint_async(client, endpoint) for endpoint in self.upstream_endpoints)
            )
        self.valid_endpoints = [result for result in results if result.error is None]
        self._summarize_health(list(results))

    async def _healthcheck_loop(self) -> None:
        try:
            while not self._healthcheck_stop.is_set():
                await self.refresh_health()
                try:
                    await asyncio.wait_for(
                        self._healthcheck_stop.wait(),
                        timeout=HEALTHCHECK_INTERVAL_SECONDS,
                    )
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            return

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
                self._healthcheck_stop.clear()
                self._healthcheck_task = asyncio.create_task(self._healthcheck_loop())
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                self._healthcheck_stop.set()
                if self._healthcheck_task is not None:
                    self._healthcheck_task.cancel()
                    try:
                        await self._healthcheck_task
                    except asyncio.CancelledError:
                        pass
                    self._healthcheck_task = None
                if self.client is not None:
                    await self.client.aclose()
                    self.client = None
                self.message_affinity.close()
                self.log_writer.close()
                await send({"type": "lifespan.shutdown.complete"})
                return

    def _build_client(self) -> httpx.AsyncClient:
        max_connections = self.cfg.load_balancer_workers * self.cfg.load_balancer_worker_concurrency
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )
        timeout = httpx.Timeout(
            connect=UPSTREAM_TIMEOUT_SECONDS,
            read=UPSTREAM_TIMEOUT_SECONDS,
            write=UPSTREAM_TIMEOUT_SECONDS,
            pool=UPSTREAM_TIMEOUT_SECONDS,
        )
        return httpx.AsyncClient(timeout=timeout, limits=limits)

    async def _handle_http(self, scope, receive, send) -> None:
        if not self.valid_endpoints:
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
        upstream_port, route_reason = self._choose_upstream_port(request_body)
        upstream_url = f"http://127.0.0.1:{upstream_port}{path}"
        headers = self._build_upstream_headers(scope["headers"], upstream_port)
        request = client.build_request(method, upstream_url, headers=headers, content=request_body)

        try:
            response = await client.send(request, stream=True)
        except httpx.HTTPError as exc:
            await self._send_plain_error(send, 502, f"Upstream request failed: {exc}".encode("utf-8"))
            return

        endpoint_label = self._endpoint_label_by_port.get(upstream_port, f"127.0.0.1:{upstream_port}")
        self.endpoint_request_counts[endpoint_label] = (
            self.endpoint_request_counts.get(endpoint_label, 0) + 1
        )

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
            route_reason=route_reason,
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

    def _choose_upstream_port(self, request_body: bytes) -> tuple[int, str]:
        valid_ports = {endpoint.port for endpoint in self.valid_endpoints}
        affinity_port = self._find_messages_affinity_port(request_body)
        if affinity_port is not None and affinity_port in valid_ports:
            return affinity_port, "affinity"
        return random.choice(list(valid_ports)), "random"

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
        route_reason: str = "unknown",
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
                "route_reason": route_reason,
            }
        )


def resolve_config_path(config_path: pathlib.Path | None = None) -> pathlib.Path:
    if config_path is not None:
        return config_path
    return pathlib.Path(
        os.environ.get("LLM_PROXY_CONFIG", "~/.config/llm-proxy.yaml")
    ).expanduser()


def create_app(config_path: pathlib.Path | None = None, verbose: bool | None = None) -> LoadBalancerApp:
    cfg = parse_config(resolve_config_path(config_path))
    if verbose is None:
        verbose = os.environ.get("LLM_PROXY_VERBOSE", "0") == "1"
    return LoadBalancerApp(cfg, verbose=verbose)


def serve_forever(
    config_path: pathlib.Path = pathlib.Path("~/.config/llm-proxy.yaml").expanduser(),
    verbose: bool = True,
) -> None:
    cfg = parse_config(config_path)
    os.environ["LLM_PROXY_CONFIG"] = str(config_path)
    os.environ["LLM_PROXY_VERBOSE"] = "1" if verbose else "0"
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
        default=pathlib.Path("~/.config/llm-proxy.yaml").expanduser(),
        help="Path to the shared config file",
    )
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="Disable verbose logging to terminal",
    )
    args = parser.parse_args(argv)
    verbose = not args.silent and os.environ.get("LLM_PROXY_VERBOSE", "1") == "1"
    serve_forever(args.config, verbose=verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
