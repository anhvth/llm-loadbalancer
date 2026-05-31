import json

from starlette.testclient import TestClient

from llm_loadbalancer.monitor import server


def test_requests_endpoint_uses_global_cache(tmp_path, monkeypatch):
    requests_dir = tmp_path / "requests"
    requests_dir.mkdir()
    log_path = requests_dir / "123-test.json"
    log_path.write_text(
        json.dumps(
            {
                "endpoint_used": "http://127.0.0.1:8000/v1/chat/completions",
                "route_reason": "least_requests",
                "output": {"model": "demo", "usage": {"prompt_tokens": 1}},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(server, "LOG_DIR", requests_dir)
    monkeypatch.setattr(server, "_cache", {})
    monkeypatch.setattr(server, "_cache_loaded", False)

    response = TestClient(server.create_app()).get("/api/requests")

    assert response.status_code == 200
    assert response.json()["items"][0]["file"] == "123-test.json"
