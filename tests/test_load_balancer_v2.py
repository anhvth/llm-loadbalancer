import json
import sqlite3
from pathlib import Path

import pytest

import cat_db
import load_balancer_v2
from llm_loadbalancer.tools import collect_jsonl


def write_v2_config(config_path: Path, state_db_path: Path, upstream_ports: list[int]) -> None:
    config_path.write_text(
        "\n".join(
            [
                "endpoints:",
                f"  - hosts: [{', '.join(f'worker-{index + 1}' for index in range(len(upstream_ports)))}]",
                f"  - port-start: {upstream_ports[0]}",
                "port:",
                "  - 8001",
                "load-balancer:",
                "  workers: 1",
                "  worker-concurrency: 64",
                "  health-path: /models",
                f"  log-dir: {config_path.parent / 'logs'}",
                f"  state-db: {state_db_path}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_app(monkeypatch, tmp_path: Path, upstream_ports: list[int]) -> load_balancer_v2.LoadBalancerAppV2:
    config_path = tmp_path / "config.yaml"
    state_db_path = tmp_path / "state.sqlite3"
    write_v2_config(config_path, state_db_path, upstream_ports)
    monkeypatch.setattr(
        load_balancer_v2.LoadBalancerAppV2,
        "_initial_healthcheck",
        lambda self: [
            load_balancer_v2.EndpointCheckResult(host=host, port=port)
            for host, port in self.upstream_endpoints
        ],
    )
    return load_balancer_v2.create_app(config_path)


def persist_payload(
    app: load_balancer_v2.LoadBalancerAppV2,
    payload: dict,
    endpoint_path: str = "/v1/messages",
    response_payload=None,
    valid_endpoints=None,
):
    prepared = load_balancer_v2.prepare_chat_request(payload)
    assert prepared is not None
    selected_port, route_reason, lookup = app._select_upstream_for_prepared_request(
        prepared,
        app.valid_endpoints if valid_endpoints is None else valid_endpoints,
    )
    endpoint_used = f"{app._base_url_for_upstream_port(selected_port)}{endpoint_path}"
    job = app._build_persistence_job(
        prepared,
        lookup,
        selected_port,
        endpoint_used,
        {} if response_payload is None else response_payload,
        route_reason,
        200,
    )
    app.state_store.persist_job(job)
    return prepared, lookup, selected_port, route_reason


def anthropic_request_payload() -> dict:
    return {
        "model": "demo-model",
        "system": [{"type": "text", "text": "You are helpful."}],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What files are in /tmp?"}],
            }
        ],
        "tools": [
            {
                "name": "ReadDir",
                "description": "List a directory",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
    }


def anthropic_response_payload() -> dict:
    return {
        "content": [
            {"type": "thinking", "thinking": "I should inspect the directory."},
            {"type": "text", "text": "The directory contains a.txt and b.txt."},
        ]
    }


def test_v2_reuses_conversation_for_continuation(monkeypatch, tmp_path: Path):
    app = build_app(monkeypatch, tmp_path, [18000, 18001])
    try:
        first_payload = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
        second_payload = {
            "model": "demo",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "tell me more"},
            ],
        }

        _, first_lookup, first_port, _ = persist_payload(app, first_payload)
        _, second_lookup, second_port, second_route_reason = persist_payload(app, second_payload)

        assert first_lookup.conversation_id == second_lookup.conversation_id
        assert second_port == first_port
        assert second_route_reason == "conversation"
    finally:
        app.state_store.close()


def test_v2_diverging_continuations_share_conversation_and_split_input_states(
    monkeypatch, tmp_path: Path
):
    app = build_app(monkeypatch, tmp_path, [18000, 18001])
    try:
        root_payload = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
        branch_a_payload = {
            "model": "demo",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "branch a"},
            ],
        }
        branch_b_payload = {
            "model": "demo",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "branch b"},
            ],
        }

        root_prepared, root_lookup, _, _ = persist_payload(app, root_payload)
        branch_a_prepared, branch_a_lookup, _, _ = persist_payload(app, branch_a_payload)
        branch_b_prepared, branch_b_lookup, _, _ = persist_payload(app, branch_b_payload)

        assert root_lookup.conversation_id == branch_a_lookup.conversation_id
        assert branch_a_lookup.conversation_id == branch_b_lookup.conversation_id
        assert branch_a_prepared.state_hash != branch_b_prepared.state_hash

        connection = sqlite3.connect(app.state_store.path)
        try:
            input_state_count = connection.execute("SELECT COUNT(*) FROM input_states").fetchone()[0]
        finally:
            connection.close()
        assert input_state_count == 3
        assert root_prepared.state_hash != branch_a_prepared.state_hash
    finally:
        app.state_store.close()


def test_v2_exact_identity_is_role_sensitive_but_loose_affinity_reuses_backend(
    monkeypatch, tmp_path: Path
):
    app = build_app(monkeypatch, tmp_path, [18000])
    try:
        first_payload = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
        second_payload = {"model": "demo", "messages": [{"role": "assistant", "content": "hi"}]}

        _, first_lookup, first_port, _ = persist_payload(app, first_payload)
        _, second_lookup, second_port, second_route_reason = persist_payload(app, second_payload)

        assert first_lookup.conversation_id != second_lookup.conversation_id
        assert second_port == first_port
        assert second_route_reason == "affinity"
    finally:
        app.state_store.close()


def test_v2_shares_state_across_app_instances(monkeypatch, tmp_path: Path):
    app_a = build_app(monkeypatch, tmp_path, [18000])
    app_b = build_app(monkeypatch, tmp_path, [18000])
    try:
        root_payload = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
        continuation_payload = {
            "model": "demo",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "continue"},
            ],
        }

        _, first_lookup, first_port, _ = persist_payload(app_a, root_payload)
        prepared = load_balancer_v2.prepare_chat_request(continuation_payload)
        assert prepared is not None
        lookup = app_b.state_store.lookup(prepared, {18000})

        assert lookup.conversation_id == first_lookup.conversation_id
        assert lookup.preferred_upstream_port == first_port
        assert lookup.route_reason == "conversation"
    finally:
        app_a.state_store.close()
        app_b.state_store.close()


def test_v2_ignores_stale_upstream_and_falls_back(monkeypatch, tmp_path: Path):
    app = build_app(monkeypatch, tmp_path, [18000, 18001])
    try:
        monkeypatch.setattr(load_balancer_v2.random, "choice", lambda values: sorted(values)[0])
        root_payload = {"model": "demo", "messages": [{"role": "user", "content": "hi"}]}
        continuation_payload = {
            "model": "demo",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "continue"},
            ],
        }

        persist_payload(app, root_payload)
        prepared = load_balancer_v2.prepare_chat_request(continuation_payload)
        assert prepared is not None
        surviving_endpoint = [endpoint for endpoint in app.valid_endpoints if endpoint.port == 18001]
        selected_port, route_reason, _ = app._select_upstream_for_prepared_request(
            prepared,
            surviving_endpoint,
        )

        assert selected_port == 18001
        assert route_reason == "random"
    finally:
        app.state_store.close()


def test_v2_repeated_identical_requests_reuse_state_and_messages(monkeypatch, tmp_path: Path):
    app = build_app(monkeypatch, tmp_path, [18000])
    try:
        payload = {"model": "demo", "messages": [{"role": "user", "content": "same"}]}

        persist_payload(app, payload)
        persist_payload(app, payload)

        connection = sqlite3.connect(app.state_store.path)
        try:
            request_count = connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            input_state_count = connection.execute("SELECT COUNT(*) FROM input_states").fetchone()[0]
            message_count = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        finally:
            connection.close()

        assert request_count == 2
        assert input_state_count == 1
        assert message_count == 1
    finally:
        app.state_store.close()


def test_cat_db_reads_v2_state_db(monkeypatch, tmp_path: Path, capsys):
    app = build_app(monkeypatch, tmp_path, [18000])
    try:
        persist_payload(
            app,
            anthropic_request_payload(),
            response_payload=anthropic_response_payload(),
        )

        result = cat_db.main(["--raw", str(app.state_store.path)])

        captured = capsys.readouterr()
        lines = [json.loads(line) for line in captured.out.splitlines()]
        assert result == 0
        assert captured.err == ""
        assert len(lines) == 1
        assert json.loads(lines[0]["input"])["messages"][0]["role"] == "user"
        assert json.loads(lines[0]["output"])["content"][-1]["text"] == "The directory contains a.txt and b.txt."
        assert lines[0]["endpoint_used"].endswith("/v1/messages")
    finally:
        app.state_store.close()


def test_collect_jsonl_exports_pending_v2_rows_incrementally(monkeypatch, tmp_path: Path):
    app = build_app(monkeypatch, tmp_path, [18000])
    export_dir = tmp_path / "training-data"
    try:
        persist_payload(
            app,
            anthropic_request_payload(),
            response_payload=anthropic_response_payload(),
        )

        result_first = collect_jsonl.main(
            [
                "--state-db",
                str(app.state_store.path),
                "--export-dir",
                str(export_dir),
            ]
        )
        result_second = collect_jsonl.main(
            [
                "--state-db",
                str(app.state_store.path),
                "--export-dir",
                str(export_dir),
            ]
        )

        export_path = export_dir / "collected.jsonl"
        lines = export_path.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])

        connection = sqlite3.connect(app.state_store.path)
        try:
            collected_at_ns = connection.execute(
                "SELECT collected_at_ns FROM requests ORDER BY request_id ASC"
            ).fetchone()[0]
        finally:
            connection.close()

        assert result_first == 0
        assert result_second == 0
        assert len(lines) == 1
        assert set(row.keys()) == {"id", "timestamp", "input", "output"}
        assert row["input"]["messages"][0]["role"] == "user"
        assert row["output"]["content"][-1]["text"] == "The directory contains a.txt and b.txt."
        assert collected_at_ns is not None
    finally:
        app.state_store.close()
