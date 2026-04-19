from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
from typing import Any


DEFAULT_DB_PATH = pathlib.Path("~/.cache/llm-proxy/state.sqlite3").expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query conversation data from the SQLite state DB")
    parser.add_argument(
        "db_path",
        nargs="?",
        type=pathlib.Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to the state DB (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--last-conversation",
        action="store_true",
        help="Print the most recently seen conversation with its latest request and reconstructed messages",
    )
    return parser


def _connect(db_path: pathlib.Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _reconstruct_messages(connection: sqlite3.Connection, state_hash: str) -> list[Any]:
    state_row = connection.execute(
        "SELECT matched_prefix_hash FROM input_states WHERE state_hash = ?",
        (state_hash,),
    ).fetchone()
    if state_row is None:
        return []

    messages: list[Any] = []
    matched_prefix_hash = state_row["matched_prefix_hash"]
    if isinstance(matched_prefix_hash, str) and matched_prefix_hash:
        messages.extend(_reconstruct_messages(connection, matched_prefix_hash))

    rows = connection.execute(
        """
        SELECT m.raw_json
        FROM state_tail_messages AS stm
        JOIN messages AS m
          ON m.message_hash = stm.message_hash
        WHERE stm.state_hash = ?
        ORDER BY stm.ordinal ASC
        """,
        (state_hash,),
    ).fetchall()
    messages.extend(json.loads(row["raw_json"]) for row in rows)
    return messages


def _print_last_conversation(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        """
        SELECT
            r.request_id,
            r.conversation_id,
            r.input_state_hash,
            r.request_meta_json,
            r.response_json,
            r.endpoint_used,
            r.route_reason,
            r.status_code,
            r.created_ns,
            c.last_seen_ns,
            c.request_count,
            c.last_upstream_port,
            c.last_base_url
        FROM requests AS r
        JOIN conversations AS c
          ON c.conversation_id = r.conversation_id
        ORDER BY c.last_seen_ns DESC, r.request_id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        print("No conversations found.")
        return 0

    request_meta = json.loads(row["request_meta_json"])
    response_payload = json.loads(row["response_json"])
    messages = _reconstruct_messages(connection, row["input_state_hash"])

    output = {
        "conversation_id": row["conversation_id"],
        "request_id": row["request_id"],
        "request_count": row["request_count"],
        "last_seen_ns": row["last_seen_ns"],
        "created_ns": row["created_ns"],
        "status_code": row["status_code"],
        "route_reason": row["route_reason"],
        "endpoint_used": row["endpoint_used"],
        "last_upstream_port": row["last_upstream_port"],
        "last_base_url": row["last_base_url"],
        "request_meta": request_meta,
        "messages": messages,
        "response": response_payload,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    connection = _connect(args.db_path.expanduser())
    try:
        return _print_last_conversation(connection)
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
