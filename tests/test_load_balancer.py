import asyncio
import http.client
import json
import os
import socket
import sqlite3
import subprocess
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import httpx
import pytest
import uvicorn

from llm_loadbalancer.keep_connection import parse_config
import llm_loadbalancer.load_balancer as load_balancer
from llm_loadbalancer.load_balancer import create_app


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class StubHandler(BaseHTTPRequestHandler):
    server: ReusableThreadingHTTPServer  # pyright: ignore[reportIncompatibleVariableOverride]
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # noqa: A003
        return

    def do_GET(self):  # noqa: N802
        if self.path == "/models":
            payload = json.dumps(
                {
                    "object": "list",
                    "data": [{"id": f"backend-{self.server.server_port}"}],
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/v1/models":
            payload = json.dumps(
                {
                    "object": "list",
                    "data": [{"id": f"backend-{self.server.server_port}"}],
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/stream":
            chunks = [b"data: one\n\n", b"data: two\n\n", b"data: done\n\n"]
            payload = b"".join(chunks)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(chunk)
                self.wfile.flush()
                time.sleep(0.02)
            return

        self.send_error(404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if self.path == "/v1/messages":
            events = [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "model": "demo-model",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 12, "output_tokens": 0},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": " world"},
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 2},
                },
                {"type": "message_stop"},
            ]
            payload = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        payload = json.dumps(
            {
                "received_bytes": len(body),
                "content_type": self.headers.get("Content-Type"),
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextmanager
def run_stub_server(port: int):
    server = ReusableThreadingHTTPServer(("127.0.0.1", port), StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def run_load_balancer(config_path: Path, verbose: bool = False):
    app = create_app(config_path, verbose=verbose)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(2048)
    listen_port = sock.getsockname()[1]
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    wait_for_port(listen_port)
    try:
        yield listen_port
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


def wait_for_port(port: int, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for port {port}")


def find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def find_free_port_block(size: int) -> list[int]:
    start = find_free_port()
    ports = list(range(start, start + size))
    sockets = []
    try:
        for port in ports:
            sock = socket.socket()
            sock.bind(("127.0.0.1", port))
            sockets.append(sock)
    except OSError:
        for sock in sockets:
            sock.close()
        return find_free_port_block(size)

    for sock in sockets:
        sock.close()
    return ports


def write_config(
    config_path: Path,
    upstream_ports: list[int],
    listen_port: int,
    health_path: str = "/models",
):
    config_path.write_text(
        "\n".join(
            [
                "endpoints:",
                f"  - hosts: [{', '.join(f'worker-{i + 1}' for i in range(len(upstream_ports)))}]",
                f"  - port-start: {upstream_ports[0]}",
                "port:",
                f"  - {listen_port}",
                "load-balancer:",
                "  workers: 1",
                "  worker-concurrency: 512",
                f"  health-path: {health_path}",
                "",
            ]
        )
    )


def read_log_rows(db_path: Path) -> list[tuple[int, str, str, str]]:
    def encode_logged_value(value):
        if value == {}:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return value

    requests_dir = db_path / "requests"
    rows = []
    for index, path in enumerate(sorted(requests_dir.glob("*.json")), start=1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            (
                index,
                encode_logged_value(payload["input"]),
                encode_logged_value(payload["output"]),
                payload["endpoint_used"],
            )
        )
    return rows


def test_curl_localhost_8001_models(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)

    with run_stub_server(upstream_ports[0]), run_stub_server(upstream_ports[1]), run_load_balancer(
        config_path
    ) as listen_port:
        result = subprocess.run(
            ["curl", "-fsS", f"http://localhost:{listen_port}/models"],
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] in {f"backend-{upstream_ports[0]}", f"backend-{upstream_ports[1]}"}


def test_direct_setup_proxies_to_reachable_worker_hosts(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "request_logs"
    upstream_ports = find_free_port_block(2)
    config_path.write_text(
        "\n".join(
            [
                "endpoints:",
                f"  - 127.0.0.1:{upstream_ports[0]}",
                f"  - 127.0.0.1:{upstream_ports[1]}",
                '  - setup: "direct"',
                "port:",
                "  - 8001",
                "load-balancer:",
                "  workers: 1",
                "  worker-concurrency: 512",
                "  health-path: /models",
                f"  log-dir: {db_path}",
                "port-start: 18001",
                "",
            ]
        )
    )

    request_payload = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}

    with run_stub_server(upstream_ports[0]), run_stub_server(upstream_ports[1]), run_load_balancer(
        config_path
    ) as listen_port:
        conn = http.client.HTTPConnection("127.0.0.1", listen_port, timeout=30)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(request_payload),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        conn.close()

    assert response.status == 200
    assert json.loads(body)["received_bytes"] > 0
    rows = read_log_rows(db_path)
    assert len(rows) == 1
    assert rows[0][3] in {f"http://127.0.0.1:{port}/v1/chat/completions" for port in upstream_ports}


def test_create_app_raises_when_no_valid_endpoints(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    monkeypatch.setattr(load_balancer, "HEALTHCHECK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(load_balancer, "INITIAL_HEALTHCHECK_TIMEOUT_SECONDS", 0.0)

    with pytest.raises(RuntimeError, match="No valid endpoints after healthcheck"):
        load_balancer.create_app(config_path)


def test_create_app_waits_for_delayed_initial_endpoint(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    monkeypatch.setattr(load_balancer, "HEALTHCHECK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(load_balancer, "INITIAL_HEALTHCHECK_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(load_balancer, "INITIAL_HEALTHCHECK_RETRY_DELAY_SECONDS", 0.01)

    started = threading.Event()

    def start_server_later():
        time.sleep(0.1)
        with run_stub_server(upstream_ports[0]):
            started.set()
            time.sleep(0.3)

    thread = threading.Thread(target=start_server_later, daemon=True)
    thread.start()

    try:
        app = load_balancer.create_app(config_path)
        assert [endpoint.port for endpoint in app.valid_endpoints] == upstream_ports
        assert started.is_set()
    finally:
        thread.join(timeout=5)


def test_create_app_accepts_404_healthcheck_when_endpoint_is_reachable(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001, health_path="/does-not-exist")

    with run_stub_server(upstream_ports[0]):
        app = load_balancer.create_app(config_path)

    assert [endpoint.port for endpoint in app.valid_endpoints] == upstream_ports


def test_create_app_probes_v1_models_when_health_path_has_no_models(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001, health_path="/does-not-exist")

    with run_stub_server(upstream_ports[0]):
        app = load_balancer.create_app(config_path)

    assert app.valid_endpoints[0].models == (f"backend-{upstream_ports[0]}",)


def test_refresh_health_keeps_working_endpoints_and_warns_on_dead_ones(
    monkeypatch, tmp_path: Path
):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  log-dir: {tmp_path / 'logs-a'}\n")
    monkeypatch.setattr(load_balancer, "HEALTHCHECK_TIMEOUT_SECONDS", 0.05)
    logs: list[str] = []

    monkeypatch.setattr(
        load_balancer.logger,
        "info",
        lambda message, *args: logs.append(message.format(*args)),
    )

    def start_server(port: int):
        server = ReusableThreadingHTTPServer(("127.0.0.1", port), StubHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        wait_for_port(port)
        return server, thread

    server_one, thread_one = start_server(upstream_ports[0])
    server_two, thread_two = start_server(upstream_ports[1])
    try:
        app = load_balancer.create_app(config_path)
        assert [endpoint.port for endpoint in app.valid_endpoints] == upstream_ports
        logs.clear()

        server_two.shutdown()
        server_two.server_close()
        thread_two.join(timeout=5)

        asyncio.run(app.refresh_health())
        rendered = "\n".join(logs)

        # Compact grouped format: MODEL, UP/TOTAL, REQS, ERR, ENDPOINTS
        assert "MODEL" in rendered
        assert "UP/TOTAL" in rendered
        assert "REQS" in rendered
        assert "ERR" in rendered
        assert "ENDPOINTS" in rendered
        # Worker endpoints appear in ENDPOINTS column (unsorted within group)
        assert f"worker-1:8000 (local {upstream_ports[0]})" in rendered
        assert f"worker-2:8000 (local {upstream_ports[1]})" in rendered
        # One worker down, one up: shows 1/2
        assert "1/2" in rendered
        # One error (Connection refused)
        assert "1" in rendered
        assert [endpoint.port for endpoint in app.valid_endpoints] == [upstream_ports[0]]
    finally:
        server_one.shutdown()
        server_one.server_close()
        thread_one.join(timeout=5)
        try:
            server_two.shutdown()
            server_two.server_close()
        except OSError:
            pass
        thread_two.join(timeout=5)


def test_refresh_health_keeps_last_known_endpoints_when_all_checks_fail(
    monkeypatch, tmp_path: Path
):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  log-dir: {tmp_path / 'logs-brownout'}\n")
    logs: list[str] = []

    initial_endpoints = [
        load_balancer.EndpointCheckResult(host=f"worker-{index + 1}", port=port)
        for index, port in enumerate(upstream_ports)
    ]
    monkeypatch.setattr(
        load_balancer.logger,
        "info",
        lambda message, *args: logs.append(message.format(*args)),
    )
    monkeypatch.setattr(
        load_balancer.LoadBalancerApp,
        "_initial_healthcheck",
        lambda self: initial_endpoints,
    )

    async def failing_check(_self, _client, endpoint):
        host, port = endpoint
        return load_balancer.EndpointCheckResult(
            host=host,
            port=port,
            error="Healthcheck timeout",
        )

    monkeypatch.setattr(load_balancer.LoadBalancerApp, "_check_endpoint_async", failing_check)

    app = load_balancer.create_app(config_path)
    logs.clear()

    asyncio.run(app.refresh_health())
    rendered = "\n".join(logs)

    assert [endpoint.port for endpoint in app.valid_endpoints] == upstream_ports
    # Compact format: all workers show 0/2 UP, 2 total, 2 errors
    assert "0/2" in rendered
    assert "MODEL" in rendered


def test_refresh_health_logs_only_on_information_gain(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  log-dir: {tmp_path / 'logs-b'}\n")
    monkeypatch.setattr(load_balancer, "HEALTHCHECK_TIMEOUT_SECONDS", 0.05)
    logs: list[str] = []

    monkeypatch.setattr(
        load_balancer.logger,
        "info",
        lambda message, *args: logs.append(message.format(*args)),
    )

    server_one = ReusableThreadingHTTPServer(("127.0.0.1", upstream_ports[0]), StubHandler)
    thread_one = threading.Thread(target=server_one.serve_forever, daemon=True)
    thread_one.start()
    wait_for_port(upstream_ports[0])
    try:
        app = load_balancer.create_app(config_path)
        logs.clear()

        asyncio.run(app.refresh_health())
        assert logs == []
    finally:
        server_one.shutdown()
        server_one.server_close()
        thread_one.join(timeout=5)


def test_refresh_health_table_includes_request_counts(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  log-dir: {tmp_path / 'logs-c'}\n")
    logs: list[str] = []

    monkeypatch.setattr(
        load_balancer.logger,
        "info",
        lambda message, *args: logs.append(message.format(*args)),
    )

    with run_stub_server(upstream_ports[0]):
        app = load_balancer.create_app(config_path)
        logs.clear()
        endpoint_label = app.upstream_endpoint_labels[("worker-1", upstream_ports[0])]
        app.endpoint_request_counts[endpoint_label] = 5

        asyncio.run(app.refresh_health())
        rendered = "\n".join(logs)

    # Compact format uses REQS column
    assert "REQS" in rendered
    assert f"{endpoint_label}" in rendered
    assert "5" in rendered


def test_check_endpoint_sync_retries_transient_connection_errors(monkeypatch):
    app = load_balancer.LoadBalancerApp.__new__(load_balancer.LoadBalancerApp)
    endpoint = ("worker-1", 18000)
    request = httpx.Request("GET", "http://127.0.0.1:18000/models")
    calls = 0

    def fake_probe(_client, _endpoint):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("All connection attempts failed", request=request)
        return ("model-a",), None

    monkeypatch.setattr(app, "_probe_models_sync", fake_probe)
    monkeypatch.setattr(load_balancer, "HEALTHCHECK_CONNECT_RETRIES", 1)
    monkeypatch.setattr(load_balancer, "HEALTHCHECK_CONNECT_RETRY_DELAY_SECONDS", 0.0)

    result = app._check_endpoint_sync(cast(httpx.Client, object()), endpoint)

    assert result.error is None
    assert result.models == ("model-a",)
    assert calls == 2


def test_check_endpoint_async_retries_transient_connection_errors(monkeypatch):
    app = load_balancer.LoadBalancerApp.__new__(load_balancer.LoadBalancerApp)
    endpoint = ("worker-1", 18000)
    request = httpx.Request("GET", "http://127.0.0.1:18000/models")
    calls = 0

    async def fake_probe(_client, _endpoint):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("All connection attempts failed", request=request)
        return ("model-a",), None

    monkeypatch.setattr(app, "_probe_models_async", fake_probe)
    monkeypatch.setattr(load_balancer, "HEALTHCHECK_CONNECT_RETRIES", 1)
    monkeypatch.setattr(load_balancer, "HEALTHCHECK_CONNECT_RETRY_DELAY_SECONDS", 0.0)

    result = asyncio.run(app._check_endpoint_async(cast(httpx.AsyncClient, object()), endpoint))

    assert result.error is None
    assert result.models == ("model-a",)
    assert calls == 2


def test_proxies_large_post_payload(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)
    body = b"x" * (2 * 1024 * 1024)

    with run_stub_server(upstream_ports[0]), run_stub_server(upstream_ports[1]), run_load_balancer(
        config_path
    ) as listen_port:
        conn = http.client.HTTPConnection("127.0.0.1", listen_port, timeout=30)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read())
        conn.close()

    assert response.status == 200
    assert payload["received_bytes"] == len(body)


def test_logs_json_request_response_to_sqlite(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "request_logs"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  log-dir: {db_path}\n")
    request_payload = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}

    with run_stub_server(upstream_ports[0]), run_load_balancer(config_path) as listen_port:
        conn = http.client.HTTPConnection("127.0.0.1", listen_port, timeout=30)
        conn.request(
            "POST",
            "/v1/chat/completions?mode=test",
            body=json.dumps(request_payload),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        response_payload = json.loads(response.read())
        conn.close()

    assert response.status == 200
    assert response_payload["received_bytes"] > 0
    assert read_log_rows(db_path) == [
        (
            1,
            json.dumps(request_payload),
            json.dumps(response_payload),
            f"http://127.0.0.1:{upstream_ports[0]}/v1/chat/completions?mode=test",
        )
    ]


def test_logs_bodyless_json_response_to_sqlite(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "request_logs"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  log-dir: {db_path}\n")

    with run_stub_server(upstream_ports[0]), run_load_balancer(config_path) as listen_port:
        conn = http.client.HTTPConnection("127.0.0.1", listen_port, timeout=30)
        conn.request("GET", "/models")
        response = conn.getresponse()
        response_payload = response.read().decode("utf-8")
        conn.close()

    assert response.status == 200
    assert response_payload
    assert read_log_rows(db_path) == []


def test_logs_assign_incrementing_ids(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "request_logs"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  log-dir: {db_path}\n")

    with run_stub_server(upstream_ports[0]), run_load_balancer(config_path) as listen_port:
        for index in range(2):
            conn = http.client.HTTPConnection("127.0.0.1", listen_port, timeout=30)
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({"model": "demo", "messages": [{"role": "user", "content": str(index)}]}),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            response.read()
            conn.close()

    rows = read_log_rows(db_path)
    assert [row[0] for row in rows] == [1, 2]


def test_verbose_prints_pretty_logged_payload(monkeypatch, tmp_path: Path, capsys):
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "request_logs"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  log-dir: {db_path}\n")
    request_payload = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
    logs: list[str] = []

    monkeypatch.setattr(
        load_balancer.logger,
        "info",
        lambda message, *args: logs.append(message.format(*args)),
    )

    with run_stub_server(upstream_ports[0]), run_load_balancer(config_path, verbose=True) as listen_port:
        conn = http.client.HTTPConnection("127.0.0.1", listen_port, timeout=30)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(request_payload),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        response.read()
        conn.close()

    captured = capsys.readouterr()
    assert captured.out == ""
    rendered = "\n".join(logs)
    assert rendered.count("[request_response_log]") == 1
    assert str(db_path / "requests") in rendered
    assert f"endpoint=http://127.0.0.1:{upstream_ports[0]}/v1/chat/completions" in rendered
    assert '"input": {' not in rendered
    assert '"output": {' not in rendered


def test_verbose_request_response_log_is_rate_limited(monkeypatch, tmp_path: Path):
    writer = load_balancer.AsyncFileLogWriter(tmp_path, verbose=True)
    payload = {
        "endpoint_used": "http://127.0.0.1:18000/v1/messages",
        "route_reason": "affinity",
    }
    calls: list[tuple[str, tuple[object, ...]]] = []
    ticks = iter([10.0, 10.5, 12.1])

    monkeypatch.setattr(load_balancer.time, "monotonic", lambda: next(ticks))

    def fake_info(message: str, *args: object):
        calls.append((message, args))

    monkeypatch.setattr(load_balancer.logger, "info", fake_info)

    writer._print_logged_payload(tmp_path / "a.json", payload)
    writer._print_logged_payload(tmp_path / "b.json", payload)
    writer._print_logged_payload(tmp_path / "c.json", payload)

    assert len(calls) == 2
    assert calls[0][0] == "[request_response_log] log_path {} endpoint={} route={}"
    assert calls[1][0] == "[request_response_log] log_path {} endpoint={} route={}"


def test_request_log_filename_encodes_endpoint_slug(tmp_path: Path):
    writer = load_balancer.AsyncFileLogWriter(tmp_path, verbose=False)
    writer.start()
    writer.submit(
        {
            "input": {"messages": [{"role": "user", "content": "hi"}]},
            "output": {"choices": []},
            "endpoint_used": "http://127.0.0.1:18000/v1/messages",
            "route_reason": "least_requests",
        }
    )
    writer.close()

    files = sorted((tmp_path / "requests").glob("*.json"))
    assert len(files) == 1
    assert "-ep_v1_messages-" in files[0].name


def test_proxies_streaming_response(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "request_logs"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  log-dir: {db_path}\n")

    with run_stub_server(upstream_ports[0]), run_stub_server(upstream_ports[1]), run_load_balancer(
        config_path
    ) as listen_port:
        conn = http.client.HTTPConnection("127.0.0.1", listen_port, timeout=30)
        conn.request("GET", "/stream")
        response = conn.getresponse()
        body = response.read()
        conn.close()

    assert response.status == 200
    assert body == b"data: one\n\ndata: two\n\ndata: done\n\n"
    assert read_log_rows(db_path) == []


def test_logs_streaming_messages_as_final_json_shape(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "request_logs"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  log-dir: {db_path}\n")
    request_payload = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    with run_stub_server(upstream_ports[0]), run_load_balancer(config_path) as listen_port:
        conn = http.client.HTTPConnection("127.0.0.1", listen_port, timeout=30)
        conn.request(
            "POST",
            "/v1/messages",
            body=json.dumps(request_payload),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        conn.close()

    assert response.status == 200
    assert "message_start" in body
    assert read_log_rows(db_path) == [
        (
            1,
            json.dumps(request_payload),
            json.dumps(
                {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "model": "demo-model",
                    "content": [{"type": "text", "text": "Hello world"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 12, "output_tokens": 2},
                }
            ),
            f"http://127.0.0.1:{upstream_ports[0]}/v1/messages",
        )
    ]


def test_does_not_log_prompt_only_requests(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "request_logs"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  log-dir: {db_path}\n")
    request_payload = {
        "model": "demo-model",
        "prompt": "hello",
        "max_tokens": 5,
    }

    with run_stub_server(upstream_ports[0]), run_load_balancer(config_path) as listen_port:
        conn = http.client.HTTPConnection("127.0.0.1", listen_port, timeout=30)
        conn.request(
            "POST",
            "/v1/completions",
            body=json.dumps(request_payload),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        response.read()
        conn.close()

    assert response.status == 200
    assert read_log_rows(db_path) == []


def test_does_not_log_requests_with_empty_messages(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "request_logs"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  log-dir: {db_path}\n")
    request_payload = {
        "model": "demo-model",
        "messages": [],
    }

    with run_stub_server(upstream_ports[0]), run_load_balancer(config_path) as listen_port:
        conn = http.client.HTTPConnection("127.0.0.1", listen_port, timeout=30)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(request_payload),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        response.read()
        conn.close()

    assert response.status == 200
    assert read_log_rows(db_path) == []


def test_reuses_backend_for_matching_messages_prefix(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "request_logs"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  log-dir: {db_path}\n")

    first_payload = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
    second_payload = {
        "model": "demo",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "tell me more"},
        ],
    }

    with run_stub_server(upstream_ports[0]), run_stub_server(upstream_ports[1]), run_load_balancer(
        config_path
    ) as listen_port:
        conn = http.client.HTTPConnection("127.0.0.1", listen_port, timeout=30)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(first_payload),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        response.read()
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", listen_port, timeout=30)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(second_payload),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        response.read()
        conn.close()

    rows = read_log_rows(db_path)
    assert len(rows) == 2
    assert rows[0][3] == rows[1][3]


def test_shares_messages_affinity_across_load_balancer_workers_via_sqlite(
    tmp_path: Path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "request_logs"
    affinity_db_path = tmp_path / "affinity.sqlite3"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(
        config_path.read_text()
        + f"  log-dir: {db_path}\n"
        + f"  affinity-db: {affinity_db_path}\n"
    )

    first_payload = json.dumps(
        {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
    ).encode("utf-8")
    second_payload = json.dumps(
        {
            "model": "demo",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "tell me more"},
            ],
        }
    ).encode("utf-8")

    monkeypatch.setattr(
        load_balancer.LoadBalancerApp,
        "_initial_healthcheck",
        lambda self: [
            load_balancer.EndpointCheckResult(host=host, port=port)
            for host, port in self.upstream_endpoints
        ],
    )
    app_a = create_app(config_path)
    app_b = create_app(config_path)
    app_a._remember_messages_affinity(first_payload, upstream_ports[0])

    assert app_b._find_messages_affinity_port(second_payload) == upstream_ports[0]


def test_reuses_messages_affinity_when_message_shape_changes(
    tmp_path: Path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "request_logs"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  log-dir: {db_path}\n")

    first_payload = json.dumps(
        {
            "model": "demo",
            "messages": [
                {"content": "ok", "role": "user"},
                {
                    "content": [
                        {"cache_control": {"type": "ephemeral"}, "text": "hi", "type": "text"}
                    ],
                    "role": "user",
                },
            ],
        }
    ).encode("utf-8")
    second_payload = json.dumps(
        {
            "model": "demo",
            "messages": [
                {"content": "ok", "role": "user"},
                {"content": "hi", "role": "user"},
                {"content": "hello", "role": "assistant"},
                {"content": "tell me more", "role": "user"},
            ],
        }
    ).encode("utf-8")

    monkeypatch.setattr(
        load_balancer.LoadBalancerApp,
        "_initial_healthcheck",
        lambda self: [
            load_balancer.EndpointCheckResult(host=host, port=port)
            for host, port in self.upstream_endpoints
        ],
    )
    app = create_app(config_path)
    app._remember_messages_affinity(first_payload, upstream_ports[0])

    assert app._find_messages_affinity_port(second_payload) == upstream_ports[0]


def test_affinity_lookup_normalizes_each_candidate_message_once(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)

    prefix_messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]
    lookup_payload = json.dumps(
        {
            "model": "demo",
            "messages": [*prefix_messages, {"role": "user", "content": "five"}],
        }
    ).encode("utf-8")

    monkeypatch.setattr(
        load_balancer.LoadBalancerApp,
        "_initial_healthcheck",
        lambda self: [
            load_balancer.EndpointCheckResult(host=host, port=port)
            for host, port in self.upstream_endpoints
        ],
    )
    app = create_app(config_path)
    app._remember_messages_affinity(
        json.dumps({"model": "demo", "messages": prefix_messages}).encode("utf-8"),
        upstream_ports[0],
    )

    original_prefix_keys = app._messages_prefix_keys
    call_count = 0

    def counting_prefix_keys(messages):
        nonlocal call_count
        call_count += 1
        return original_prefix_keys(messages)

    monkeypatch.setattr(app, "_messages_prefix_keys", counting_prefix_keys)

    assert app._find_messages_affinity_port(lookup_payload) == upstream_ports[0]
    assert call_count == 1


def test_choose_upstream_uses_request_endpoint_snapshot(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)

    monkeypatch.setattr(
        load_balancer.LoadBalancerApp,
        "_initial_healthcheck",
        lambda self: [
            load_balancer.EndpointCheckResult(host=host, port=port)
            for host, port in self.upstream_endpoints
        ],
    )
    app = create_app(config_path)
    request_body = json.dumps(
        {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
    ).encode("utf-8")
    app._remember_messages_affinity(request_body, upstream_ports[0])

    endpoint_snapshot = list(app.valid_endpoints)
    app.valid_endpoints = []

    upstream_port, route_reason = app._choose_upstream_port(
        json.dumps(
            {
                "model": "demo",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "tell me more"},
                ],
            }
        ).encode("utf-8"),
        endpoint_snapshot,
    )

    assert upstream_port == upstream_ports[0]
    assert route_reason == "affinity"


def test_random_routing_ignores_messages_affinity(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + "  routing: random\n")

    monkeypatch.setattr(
        load_balancer.LoadBalancerApp,
        "_initial_healthcheck",
        lambda self: [
            load_balancer.EndpointCheckResult(host=host, port=port)
            for host, port in self.upstream_endpoints
        ],
    )
    monkeypatch.setattr(load_balancer.random, "choice", lambda values: max(values))
    app = create_app(config_path)
    request_body = json.dumps(
        {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
    ).encode("utf-8")
    app._remember_messages_affinity(request_body, upstream_ports[0])

    upstream_port, route_reason = app._choose_upstream_port(
        json.dumps(
            {
                "model": "demo",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "tell me more"},
                ],
            }
        ).encode("utf-8"),
        list(app.valid_endpoints),
    )

    assert upstream_port == max(upstream_ports)
    assert route_reason == "random"


def test_sqlite_affinity_store_retries_transient_init_lock(tmp_path: Path, monkeypatch):
    class FakeConnection:
        def __init__(self):
            self.queries: list[str] = []
            self._locked_once = False

        def execute(self, query, *args, **kwargs):
            self.queries.append(query)
            if query == "PRAGMA journal_mode=WAL" and not self._locked_once:
                self._locked_once = True
                raise sqlite3.OperationalError("database is locked")
            return self

        def close(self):
            return None

    fake_connection = FakeConnection()
    monkeypatch.setattr(load_balancer.sqlite3, "connect", lambda *args, **kwargs: fake_connection)

    store = load_balancer.SqliteMessageAffinityStore(tmp_path / "affinity.sqlite3", max_entries=16)

    assert fake_connection.queries.count("PRAGMA journal_mode=WAL") == 2
    assert any("CREATE TABLE IF NOT EXISTS message_affinity" in query for query in fake_connection.queries)
    store.close()


class FakeResourceModule:
    RLIMIT_NOFILE = 7
    RLIM_INFINITY = -1

    def __init__(self, soft_limit: int, hard_limit: int, setrlimit_error: Exception | None = None):
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit
        self.setrlimit_error = setrlimit_error
        self.setrlimit_calls: list[tuple[int, tuple[int, int]]] = []

    def getrlimit(self, kind):
        assert kind == self.RLIMIT_NOFILE
        return self.soft_limit, self.hard_limit

    def setrlimit(self, kind, limits):
        assert kind == self.RLIMIT_NOFILE
        if self.setrlimit_error is not None:
            raise self.setrlimit_error
        self.setrlimit_calls.append((kind, limits))
        self.soft_limit, self.hard_limit = limits


def test_fd_plan_raises_soft_limit_when_hard_limit_allows(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + "  worker-concurrency: 10\n")
    cfg = parse_config(config_path)
    fake_resource = FakeResourceModule(soft_limit=64, hard_limit=1024)
    monkeypatch.setattr(load_balancer, "resource_module", fake_resource)

    plan = load_balancer.plan_file_descriptor_limits(cfg)

    assert plan.effective_worker_concurrency == 10
    assert plan.raised_soft_limit is True
    assert fake_resource.setrlimit_calls == [
        (fake_resource.RLIMIT_NOFILE, (plan.required_soft_limit, 1024))
    ]


def test_fd_plan_caps_concurrency_when_hard_limit_is_too_low(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + "  worker-concurrency: 100\n")
    cfg = parse_config(config_path)
    fake_resource = FakeResourceModule(soft_limit=64, hard_limit=200)
    monkeypatch.setattr(load_balancer, "resource_module", fake_resource)

    plan = load_balancer.plan_file_descriptor_limits(cfg)

    assert plan.effective_worker_concurrency == 35
    assert plan.capped_by_limit is True
    assert plan.warning is not None
    assert "hard limit is too low" in plan.warning
    assert fake_resource.setrlimit_calls == []


def test_fd_plan_uses_conservative_concurrency_without_resource(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)
    cfg = parse_config(config_path)
    monkeypatch.setattr(load_balancer, "resource_module", None)

    plan = load_balancer.plan_file_descriptor_limits(cfg)

    assert plan.effective_worker_concurrency == 447
    assert plan.capped_by_limit is True
    assert plan.warning is not None
    assert "Could not inspect RLIMIT_NOFILE" in plan.warning


def test_fd_plan_caps_concurrency_when_setrlimit_fails(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + "  worker-concurrency: 100\n")
    cfg = parse_config(config_path)
    fake_resource = FakeResourceModule(
        soft_limit=160,
        hard_limit=1024,
        setrlimit_error=OSError("permission denied"),
    )
    monkeypatch.setattr(load_balancer, "resource_module", fake_resource)

    plan = load_balancer.plan_file_descriptor_limits(cfg)

    assert plan.effective_worker_concurrency == 15
    assert plan.capped_by_limit is True
    assert plan.warning is not None
    assert "Could not raise RLIMIT_NOFILE" in plan.warning


def test_parse_config_resolves_default_db_path_relative_to_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)

    cfg = parse_config(config_path)

    assert cfg.load_balancer_log_dir == Path("~/.cache/llm-proxy/logs").expanduser()
    assert cfg.load_balancer_affinity_db_path == Path("~/.cache/llm-proxy/affinity.sqlite3").expanduser()
    assert cfg.load_balancer_routing == "smart"


def test_parse_config_supports_random_routing(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + "  routing: random\n")

    cfg = parse_config(config_path)

    assert cfg.load_balancer_routing == "random"


def test_listen_backlog_scales_with_total_concurrency(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(
        config_path.read_text()
        + "  workers: 4\n"
        + "  worker-concurrency: 10204\n"
    )
    cfg = parse_config(config_path)

    assert load_balancer.listen_backlog_for_worker_concurrency(cfg, 10204) == 40816


def test_serve_forever_does_not_enable_reload(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8123)
    seen = {}

    def fake_run(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs

    monkeypatch.setattr(load_balancer.uvicorn, "run", fake_run)
    monkeypatch.setattr(
        load_balancer,
        "plan_file_descriptor_limits",
        lambda cfg: load_balancer.FileDescriptorPlan(
            configured_worker_concurrency=512,
            effective_worker_concurrency=123,
            required_soft_limit=1153,
            soft_limit=4096,
            hard_limit=4096,
        ),
    )

    load_balancer.serve_forever(config_path, verbose=True)

    assert seen["kwargs"]["workers"] == 1
    assert seen["kwargs"]["port"] == 8123
    assert seen["kwargs"]["limit_concurrency"] == 123
    assert seen["kwargs"]["backlog"] == 4096
    assert "reload" not in seen["kwargs"]
    assert os.environ[load_balancer.LLM_PROXY_EFFECTIVE_WORKER_CONCURRENCY_ENV] == "123"


def test_build_client_uses_effective_worker_concurrency(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(
        config_path.read_text()
        + "  workers: 3\n"
        + "  worker-concurrency: 5\n"
    )
    monkeypatch.delenv(load_balancer.LLM_PROXY_EFFECTIVE_WORKER_CONCURRENCY_ENV, raising=False)

    seen = {}

    class DummyAsyncClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(load_balancer.httpx, "AsyncClient", DummyAsyncClient)
    monkeypatch.setattr(
        load_balancer.LoadBalancerApp,
        "_initial_healthcheck",
        lambda self: [
            load_balancer.EndpointCheckResult(host=host, port=port)
            for host, port in self.upstream_endpoints
        ],
    )

    app = create_app(config_path)
    client = app._build_client()

    assert isinstance(client, DummyAsyncClient)
    assert seen["limits"].max_connections == 5
    assert seen["limits"].max_keepalive_connections == 5
