from __future__ import annotations

import json
import pathlib
import sqlite3
import time
import urllib.error
import urllib.request
from typing import Any


BASE_URL = "http://127.0.0.1:8001"
MODEL = "/tmp/scratch/models/MiniMax-M2.7"
STATE_DB_PATH = pathlib.Path("~/.cache/llm-proxy/state.sqlite3").expanduser()
DB_WRITE_TIMEOUT_SECONDS = 10.0
DB_POLL_INTERVAL_SECONDS = 0.1
HTTP_RETRIES = 8
HTTP_RETRY_DELAY_SECONDS = 0.5


def post_chat(messages: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, str]]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 64,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(HTTP_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
                headers = {key.lower(): value for key, value in response.headers.items()}
                return body, headers
        except urllib.error.HTTPError as exc:
            if exc.code not in {502, 503, 504} or attempt == HTTP_RETRIES - 1:
                raise
            time.sleep(HTTP_RETRY_DELAY_SECONDS)
        except urllib.error.URLError:
            if attempt == HTTP_RETRIES - 1:
                raise
            time.sleep(HTTP_RETRY_DELAY_SECONDS)
    raise RuntimeError("unreachable")


def assistant_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def fetch_conversation_row(connection: sqlite3.Connection, conversation_id: str) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT conversation_id, request_count, last_upstream_port, last_base_url, last_seen_ns
        FROM conversations
        WHERE conversation_id = ?
        """,
        (conversation_id,),
    ).fetchone()


def fetch_request_count(connection: sqlite3.Connection, conversation_id: str) -> int:
    row = connection.execute(
        "SELECT COUNT(*) FROM requests WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    return int(row[0]) if row is not None else 0


def wait_for_request_count(
    connection: sqlite3.Connection,
    conversation_id: str,
    minimum_count: int,
) -> tuple[sqlite3.Row, int]:
    deadline = time.monotonic() + DB_WRITE_TIMEOUT_SECONDS
    while True:
        row = fetch_conversation_row(connection, conversation_id)
        request_count = fetch_request_count(connection, conversation_id)
        if row is not None and request_count >= minimum_count:
            return row, request_count
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Timed out waiting for conversation {conversation_id} "
                f"to reach {minimum_count} requests; saw {request_count}"
            )
        time.sleep(DB_POLL_INTERVAL_SECONDS)


def main() -> int:
    connection = sqlite3.connect(STATE_DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        before_by_conversation: dict[str, int] = {}
        turn1_body, turn1_headers = post_chat(
            [{"role": "user", "content": "hello"}]
        )
        conversation_id_1 = turn1_headers.get("x-llm-proxy-conversation-id")
        if not conversation_id_1:
            raise RuntimeError("Missing x-llm-proxy-conversation-id header on turn 1")

        first_assistant = assistant_text(turn1_body)
        turn1_count_before = before_by_conversation.setdefault(
            conversation_id_1,
            fetch_request_count(connection, conversation_id_1),
        )
        row_after_turn1, requests_after_turn1 = wait_for_request_count(
            connection,
            conversation_id_1,
            turn1_count_before + 1,
        )

        turn2_body, turn2_headers = post_chat(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": first_assistant},
                {"role": "user", "content": "this is new"},
            ]
        )
        conversation_id_2 = turn2_headers.get("x-llm-proxy-conversation-id")
        if not conversation_id_2:
            raise RuntimeError("Missing x-llm-proxy-conversation-id header on turn 2")

        turn2_count_before = (
            requests_after_turn1
            if conversation_id_2 == conversation_id_1
            else fetch_request_count(connection, conversation_id_2)
        )
        row_after_turn2, requests_after_turn2 = wait_for_request_count(
            connection,
            conversation_id_2,
            turn2_count_before + 1,
        )

        result = {
            "turn_1": {
                "conversation_id": conversation_id_1,
                "route_reason": turn1_headers.get("x-llm-proxy-route-reason"),
                "assistant_text": first_assistant,
                "db_conversation": dict(row_after_turn1),
                "db_request_count_before": turn1_count_before,
                "db_request_count": requests_after_turn1,
            },
            "turn_2": {
                "conversation_id": conversation_id_2,
                "route_reason": turn2_headers.get("x-llm-proxy-route-reason"),
                "assistant_text": assistant_text(turn2_body),
                "db_conversation": dict(row_after_turn2),
                "db_request_count_before": turn2_count_before,
                "db_request_count": requests_after_turn2,
            },
            "same_conversation_id": conversation_id_1 == conversation_id_2,
            "conversation_grew": (
                conversation_id_1 == conversation_id_2
                and requests_after_turn2 > requests_after_turn1
                and int(row_after_turn2["request_count"]) > int(row_after_turn1["request_count"])
            ),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
