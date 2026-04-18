#!/usr/bin/env python3
"""Async streaming reverse proxy for OpenAI-like APIs."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sqlite3
from typing import AsyncIterator

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

REQUEST_RESPONSE_LOG_TABLE = "request_response_log"


def build_upstream_ports(cfg: TunnelConfig) -> list[int]:
    return [cfg.port_start + index for index in range(len(cfg.hosts))]


class LoadBalancerApp:
    def __init__(self, cfg: TunnelConfig):
        self.cfg = cfg
        self.upstream_ports = build_upstream_ports(cfg)
        self.client: httpx.AsyncClient | None = None
        self.db: sqlite3.Connection | None = None

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
                self.db = self._open_log_db()
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if self.client is not None:
                    await self.client.aclose()
                    self.client = None
                if self.db is not None:
                    self.db.close()
                    self.db = None
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

    def _open_log_db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.cfg.load_balancer_db_path)
        db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {REQUEST_RESPONSE_LOG_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input TEXT NOT NULL,
                output TEXT NOT NULL,
                endpoint_used TEXT NOT NULL
            )
            """
        )
        db.commit()
        return db

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

        upstream_port = random.choice(self.upstream_ports)
        upstream_url = f"http://127.0.0.1:{upstream_port}{path}"
        headers = self._build_upstream_headers(scope["headers"], upstream_port)

        request_chunks = bytearray()

        async def request_body() -> AsyncIterator[bytes]:
            while True:
                message = await receive()
                if message["type"] != "http.request":
                    continue
                body = message.get("body", b"")
                if body:
                    request_chunks.extend(body)
                    yield body
                if not message.get("more_body", False):
                    break

        request = client.build_request(method, upstream_url, headers=headers, content=request_body())

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
        should_log_response = self._is_json_content_type(response.headers.get("content-type"))
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
                if should_log_response and chunk:
                    response_chunks.extend(chunk)
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
        finally:
            await response.aclose()

        await send({"type": "http.response.body", "body": b"", "more_body": False})
        self._log_exchange_if_json(bytes(request_chunks), bytes(response_chunks), upstream_url)

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

    def _is_json_content_type(self, content_type: str | None) -> bool:
        if not content_type:
            return False
        return content_type.split(";", 1)[0].strip().lower() == "application/json"

    def _decode_json_string(self, body: bytes) -> str | None:
        if not body:
            return None
        try:
            text = body.decode("utf-8")
            json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return text

    def _log_exchange_if_json(self, request_body: bytes, response_body: bytes, endpoint_used: str) -> None:
        if self.db is None:
            return
        request_json = self._decode_json_string(request_body)
        response_json = self._decode_json_string(response_body)
        if request_json is None or response_json is None:
            return
        self.db.execute(
            f"INSERT INTO {REQUEST_RESPONSE_LOG_TABLE} (input, output, endpoint_used) VALUES (?, ?, ?)",
            (request_json, response_json, endpoint_used),
        )
        self.db.commit()


def resolve_config_path(config_path: pathlib.Path | None = None) -> pathlib.Path:
    if config_path is not None:
        return config_path
    return pathlib.Path(os.environ.get("LLM_LOADBALANCER_CONFIG", "config.yaml"))


def create_app(config_path: pathlib.Path | None = None) -> LoadBalancerApp:
    cfg = parse_config(resolve_config_path(config_path))
    return LoadBalancerApp(cfg)


def serve_forever(config_path: pathlib.Path = pathlib.Path("config.yaml")) -> None:
    cfg = parse_config(config_path)
    os.environ["LLM_LOADBALANCER_CONFIG"] = str(config_path)
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
    args = parser.parse_args(argv)
    serve_forever(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
