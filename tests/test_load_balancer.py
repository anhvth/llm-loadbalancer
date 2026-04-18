import http.client
import json
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import uvicorn

from keep_connection import parse_config
import load_balancer
from load_balancer import create_app


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


def write_config(config_path: Path, upstream_ports: list[int], listen_port: int):
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
                "  max-connections: 2048",
                "  max-keepalive-connections: 512",
                "  upstream-timeout: 30",
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
    assert read_log_rows(db_path) == [
        (
            1,
            "",
            response_payload,
            f"http://127.0.0.1:{upstream_ports[0]}/models",
        )
    ]


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
                body=json.dumps({"request": index}),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            response.read()
            conn.close()

    rows = read_log_rows(db_path)
    assert [row[0] for row in rows] == [1, 2]


def test_verbose_prints_pretty_logged_payload(tmp_path: Path, capsys):
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "request_logs"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  log-dir: {db_path}\n")
    request_payload = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}

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
    assert captured.err.count("[request_response_log]") == 1
    assert str(db_path / "requests") in captured.err
    assert f"endpoint=http://127.0.0.1:{upstream_ports[0]}/v1/chat/completions" in captured.err
    assert '"input": {' not in captured.err
    assert '"output": {' not in captured.err


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
    rows = read_log_rows(db_path)
    assert rows[0][:3] == (1, "", "data: one\n\ndata: two\n\ndata: done\n\n")
    assert rows[0][3] in {f"http://127.0.0.1:{port}/stream" for port in upstream_ports}


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


def test_shares_messages_affinity_across_load_balancer_workers_via_sqlite(tmp_path: Path):
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

    app_a = create_app(config_path)
    app_b = create_app(config_path)
    app_a._remember_messages_affinity(first_payload, upstream_ports[0])

    assert app_b._find_messages_affinity_port(second_payload) == upstream_ports[0]


def test_reuses_messages_affinity_when_message_shape_changes(tmp_path: Path):
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


def test_parse_config_resolves_default_db_path_relative_to_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)

    cfg = parse_config(config_path)

    assert cfg.load_balancer_log_dir == Path("~/.cache/llmup/logs").expanduser()
    assert cfg.load_balancer_affinity_db_path == Path("~/.cache/llmup/affinity.sqlite3").expanduser()


def test_serve_forever_does_not_enable_reload(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8123)
    seen = {}

    def fake_run(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs

    monkeypatch.setattr(load_balancer.uvicorn, "run", fake_run)

    load_balancer.serve_forever(config_path, verbose=True)

    assert seen["kwargs"]["workers"] == 1
    assert seen["kwargs"]["port"] == 8123
    assert "reload" not in seen["kwargs"]
