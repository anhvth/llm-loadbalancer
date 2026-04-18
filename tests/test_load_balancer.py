import http.client
import json
import sqlite3
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import uvicorn

from keep_connection import parse_config
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
    with sqlite3.connect(db_path) as connection:
        return list(
            connection.execute(
                "SELECT id, input, output, endpoint_used FROM request_response_log ORDER BY id"
            )
        )


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
    db_path = tmp_path / "requests.sqlite3"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  db-path: {db_path}\n")
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


def test_logs_assign_incrementing_ids(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "requests.sqlite3"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  db-path: {db_path}\n")

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
    db_path = tmp_path / "requests.sqlite3"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  db-path: {db_path}\n")
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
    assert captured.err == (
        "[request_response_log] inserted row:\n"
        + json.dumps(
            {
                "id": 1,
                "input": request_payload,
                "output": {
                    "received_bytes": len(json.dumps(request_payload)),
                    "content_type": "application/json",
                },
                "endpoint_used": f"http://127.0.0.1:{upstream_ports[0]}/v1/chat/completions",
            },
            indent=2,
        )
        + "\n"
    )


def test_proxies_streaming_response(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "requests.sqlite3"
    upstream_ports = find_free_port_block(2)
    write_config(config_path, upstream_ports, 8001)
    config_path.write_text(config_path.read_text() + f"  db-path: {db_path}\n")

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


def test_parse_config_resolves_default_db_path_relative_to_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    upstream_ports = find_free_port_block(1)
    write_config(config_path, upstream_ports, 8001)

    cfg = parse_config(config_path)

    assert cfg.load_balancer_db_path == tmp_path / "llm_loadbalancer.sqlite3"
