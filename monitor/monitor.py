#!/usr/bin/env python3
"""Read-only live monitor for llm-loadbalancer state."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response


ROOT_DIR = Path(__file__).resolve().parents[1]
try:
    from keep_connection import parse_config
except Exception:  # pragma: no cover - fallback when the parent module is unavailable.
    parse_config = None


DEFAULT_CONFIG_PATH = Path("~/.config/llm-proxy.yaml").expanduser()
DEFAULT_STATE_DB_PATH = Path("~/.cache/llm-proxy/state.sqlite3").expanduser()
DEFAULT_LOG_DIR = Path("~/.cache/llm-proxy/logs").expanduser()
DEFAULT_HOST = os.environ.get("MONITOR_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("MONITOR_PORT", "4477"))
DB_TIMEOUT_SECONDS = 1.0
AUTO_REFRESH_MS = 3_000
MAX_CONVERSATIONS = 200


@dataclass(frozen=True)
class RuntimePaths:
    config_path: Path
    state_db_path: Path
    log_dir: Path
    health_state_path: Path
    endpoint_labels_by_port: dict[int, str]
    config_error: str | None = None


def resolve_config_path() -> Path:
    raw_path = os.environ.get("LLM_PROXY_CONFIG")
    if raw_path:
        return Path(raw_path).expanduser()
    return DEFAULT_CONFIG_PATH


def _build_endpoint_labels(cfg: Any) -> dict[int, str]:
    remote_ports = list(cfg.remote_ports or [cfg.remote_port] * len(cfg.hosts))
    labels: dict[int, str] = {}
    for index, (host, remote_port) in enumerate(zip(cfg.hosts, remote_ports, strict=True)):
        local_port = cfg.port_start + index
        if cfg.endpoint_setup == "direct":
            labels[local_port] = f"{host}:{remote_port}"
        elif local_port == remote_port:
            labels[local_port] = f"{host}:{local_port}"
        else:
            labels[local_port] = f"{host}:{remote_port} (local {local_port})"
    return labels


def resolve_runtime_paths() -> RuntimePaths:
    config_path = resolve_config_path()
    state_db_path = DEFAULT_STATE_DB_PATH
    log_dir = DEFAULT_LOG_DIR
    endpoint_labels_by_port: dict[int, str] = {}
    config_error: str | None = None

    if parse_config is not None and config_path.exists():
        try:
            cfg = parse_config(config_path)
        except Exception as exc:
            config_error = f"Could not parse config at {config_path}: {exc}"
        else:
            state_db_path = Path(cfg.load_balancer_state_db_path).expanduser()
            log_dir = Path(cfg.load_balancer_log_dir).expanduser()
            endpoint_labels_by_port = _build_endpoint_labels(cfg)

    return RuntimePaths(
        config_path=config_path,
        state_db_path=state_db_path,
        log_dir=log_dir,
        health_state_path=log_dir / "health_state.json",
        endpoint_labels_by_port=endpoint_labels_by_port,
        config_error=config_error,
    )


def now_ns() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1_000_000_000)


def ns_to_iso(value: int | None) -> str | None:
    if value is None:
        return None
    return (
        datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds")
    )


def file_mtime_iso(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).astimezone().isoformat(
        timespec="seconds"
    )


def truncate(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def normalize_piece(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [normalize_piece(item) for item in value]
        return "\n\n".join(part for part in parts if part)
    if isinstance(value, dict):
        kind = value.get("type")
        if kind in {"text", "input_text", "output_text"} and isinstance(value.get("text"), str):
            return value["text"]
        if kind == "thinking" and isinstance(value.get("thinking"), str):
            return value["thinking"]
        if kind == "image":
            media_type = value.get("source", {}).get("media_type", "image")
            return f"[{media_type}]"
        if kind == "tool_result":
            return normalize_piece(value.get("content"))
        if kind == "tool_use":
            name = value.get("name", "tool")
            return f"[tool:{name}]"
        if isinstance(value.get("content"), str):
            return value["content"]
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return repr(value)


def normalize_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        return {"role": "unknown", "content": normalize_piece(message), "tool_names": []}

    role = str(message.get("role", "unknown"))
    content = normalize_piece(message.get("content"))
    tool_names: list[str] = []
    raw_tool_calls = message.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        for item in raw_tool_calls:
            if not isinstance(item, dict):
                continue
            function_data = item.get("function")
            if isinstance(function_data, dict) and isinstance(function_data.get("name"), str):
                tool_names.append(function_data["name"])
    if role == "assistant" and tool_names:
        summary = f"Tool calls: {', '.join(tool_names)}"
        content = f"{summary}\n\n{content}".strip()
    return {"role": role, "content": content, "tool_names": tool_names}


def normalize_messages(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    messages: list[dict[str, Any]] = []
    system_prompt = payload.get("system")
    if system_prompt:
        messages.append({"role": "system", "content": normalize_piece(system_prompt), "tool_names": []})

    raw_messages = payload.get("messages")
    if isinstance(raw_messages, list):
        messages.extend(normalize_message(message) for message in raw_messages)
        return messages

    prompt = payload.get("prompt")
    if prompt:
        messages.append({"role": "user", "content": normalize_piece(prompt), "tool_names": []})
    return messages


def extract_preview(payload: Any) -> str:
    messages = normalize_messages(payload)
    for message in reversed(messages):
        role = message.get("role")
        if role in {"system", "tool"}:
            continue
        content = str(message.get("content", "")).strip()
        if content:
            return truncate(content)
    return ""


def extract_model(request_meta: Any, response_payload: Any) -> str | None:
    if isinstance(request_meta, dict):
        model = request_meta.get("model")
        if isinstance(model, str) and model:
            return model
    if isinstance(response_payload, dict):
        if isinstance(response_payload.get("model"), str) and response_payload["model"]:
            return response_payload["model"]
        choices = response_payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict) and isinstance(first.get("model"), str):
                return first["model"]
    return None


def extract_reasoning_text(response_payload: Any) -> str:
    if not isinstance(response_payload, dict):
        return ""
    content = response_payload.get("content")
    if isinstance(content, list):
        thinking_parts = [
            str(item.get("thinking", "")).strip()
            for item in content
            if isinstance(item, dict) and item.get("type") == "thinking" and item.get("thinking")
        ]
        if thinking_parts:
            return "\n\n".join(part for part in thinking_parts if part).strip()
    if isinstance(response_payload.get("reasoning_content"), str):
        return response_payload["reasoning_content"].strip()
    message = response_payload.get("message")
    if isinstance(message, dict) and isinstance(message.get("reasoning_content"), str):
        return message["reasoning_content"].strip()
    if isinstance(response_payload.get("full_reasoning_content"), str):
        return response_payload["full_reasoning_content"].strip()
    choices = response_payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("reasoning_content"), str):
                return message["reasoning_content"].strip()
    return ""


def extract_response_text(response_payload: Any) -> str:
    if response_payload is None:
        return ""
    if isinstance(response_payload, str):
        return response_payload
    if not isinstance(response_payload, dict):
        return json.dumps(response_payload, ensure_ascii=False, indent=2, sort_keys=True)

    message = response_payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        text = normalize_piece(content)
        if text:
            return text.strip()

    choices = response_payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            choice_message = first.get("message")
            if isinstance(choice_message, dict):
                text = normalize_piece(choice_message.get("content"))
                if text:
                    return text.strip()
            if isinstance(first.get("text"), str):
                return first["text"].strip()

    content = response_payload.get("content")
    if isinstance(content, list):
        text_parts = [
            normalize_piece(item)
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "input_text", "output_text"}
        ]
        if text_parts:
            return "\n\n".join(part for part in text_parts if part).strip()
    text = normalize_piece(content)
    if text:
        return text.strip()

    error = response_payload.get("error")
    if error is not None:
        return normalize_piece(error).strip()

    return json.dumps(response_payload, ensure_ascii=False, indent=2, sort_keys=True)


def load_health_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, dict):
        return {}
    return snapshot


class ReadOnlyStateDB:
    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=DB_TIMEOUT_SECONDS,
            isolation_level=None,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(f"PRAGMA busy_timeout = {int(DB_TIMEOUT_SECONDS * 1000)}")
        self._messages_cache: dict[str, tuple[Any, ...]] = {}

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ReadOnlyStateDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def reconstruct_messages(self, state_hash: str) -> tuple[Any, ...]:
        if state_hash in self._messages_cache:
            return self._messages_cache[state_hash]

        state_row = self.connection.execute(
            """
            SELECT matched_prefix_hash
            FROM input_states
            WHERE state_hash = ?
            """,
            (state_hash,),
        ).fetchone()
        if state_row is None:
            self._messages_cache[state_hash] = ()
            return ()

        messages: list[Any] = []
        matched_prefix_hash = state_row["matched_prefix_hash"]
        if isinstance(matched_prefix_hash, str) and matched_prefix_hash:
            messages.extend(self.reconstruct_messages(matched_prefix_hash))

        for row in self.connection.execute(
            """
            SELECT m.raw_json
            FROM state_tail_messages AS stm
            JOIN messages AS m
              ON m.message_hash = stm.message_hash
            WHERE stm.state_hash = ?
            ORDER BY stm.ordinal ASC
            """,
            (state_hash,),
        ):
            messages.append(json.loads(str(row["raw_json"])))

        result = tuple(messages)
        self._messages_cache[state_hash] = result
        return result

    def reconstruct_input_payload(self, request_meta: Any, state_hash: str) -> Any:
        if not isinstance(request_meta, dict):
            return request_meta
        payload = dict(request_meta)
        payload["messages"] = list(self.reconstruct_messages(state_hash))
        return payload

    def load_summary(self) -> dict[str, Any]:
        current_ns = now_ns()
        one_minute_ago = current_ns - 60 * 1_000_000_000
        five_minutes_ago = current_ns - 5 * 60 * 1_000_000_000
        one_hour_ago = current_ns - 60 * 60 * 1_000_000_000

        row = self.connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM conversations) AS total_conversations,
                (SELECT COUNT(*) FROM requests) AS total_requests,
                (SELECT COUNT(*) FROM requests WHERE created_ns >= ?) AS requests_last_minute,
                (SELECT COUNT(*) FROM requests WHERE created_ns >= ?) AS requests_last_5m,
                (SELECT COUNT(*) FROM requests WHERE created_ns >= ?) AS requests_last_hour,
                (SELECT COUNT(DISTINCT conversation_id) FROM requests WHERE created_ns >= ?) AS active_conversations_last_5m,
                (SELECT MAX(created_ns) FROM requests) AS last_request_ns,
                (SELECT MAX(last_seen_ns) FROM conversations) AS last_conversation_ns
            """,
            (one_minute_ago, five_minutes_ago, one_hour_ago, five_minutes_ago),
        ).fetchone()
        if row is None:
            return {}
        return {
            "total_conversations": int(row["total_conversations"] or 0),
            "total_requests": int(row["total_requests"] or 0),
            "requests_last_minute": int(row["requests_last_minute"] or 0),
            "requests_last_5m": int(row["requests_last_5m"] or 0),
            "requests_last_hour": int(row["requests_last_hour"] or 0),
            "active_conversations_last_5m": int(row["active_conversations_last_5m"] or 0),
            "last_request_at": ns_to_iso(
                int(row["last_request_ns"]) if row["last_request_ns"] is not None else None
            ),
            "last_conversation_at": ns_to_iso(
                int(row["last_conversation_ns"]) if row["last_conversation_ns"] is not None else None
            ),
        }

    def load_endpoint_rows(self) -> dict[int, dict[str, Any]]:
        current_ns = now_ns()
        one_minute_ago = current_ns - 60 * 1_000_000_000
        five_minutes_ago = current_ns - 5 * 60 * 1_000_000_000
        one_hour_ago = current_ns - 60 * 60 * 1_000_000_000

        rows = self.connection.execute(
            """
            SELECT
                upstream_port,
                COUNT(*) AS total_requests,
                SUM(CASE WHEN created_ns >= ? THEN 1 ELSE 0 END) AS requests_last_minute,
                SUM(CASE WHEN created_ns >= ? THEN 1 ELSE 0 END) AS requests_last_5m,
                SUM(CASE WHEN created_ns >= ? THEN 1 ELSE 0 END) AS requests_last_hour,
                SUM(CASE WHEN status_code BETWEEN 200 AND 299 THEN 1 ELSE 0 END) AS ok_responses,
                SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) AS error_responses,
                SUM(CASE WHEN route_reason = 'conversation' THEN 1 ELSE 0 END) AS conversation_routes,
                SUM(CASE WHEN route_reason = 'affinity' THEN 1 ELSE 0 END) AS affinity_routes,
                SUM(CASE WHEN route_reason = 'random' THEN 1 ELSE 0 END) AS random_routes,
                MAX(created_ns) AS last_request_ns
            FROM requests
            GROUP BY upstream_port
            ORDER BY total_requests DESC, upstream_port ASC
            """,
            (one_minute_ago, five_minutes_ago, one_hour_ago),
        )

        endpoint_rows: dict[int, dict[str, Any]] = {}
        for row in rows:
            upstream_port = int(row["upstream_port"])
            latest = self.connection.execute(
                """
                SELECT endpoint_used, base_url, status_code, route_reason
                FROM requests
                WHERE upstream_port = ?
                ORDER BY created_ns DESC, request_id DESC
                LIMIT 1
                """,
                (upstream_port,),
            ).fetchone()
            endpoint_rows[upstream_port] = {
                "upstream_port": upstream_port,
                "total_requests": int(row["total_requests"] or 0),
                "requests_last_minute": int(row["requests_last_minute"] or 0),
                "requests_last_5m": int(row["requests_last_5m"] or 0),
                "requests_last_hour": int(row["requests_last_hour"] or 0),
                "ok_responses": int(row["ok_responses"] or 0),
                "error_responses": int(row["error_responses"] or 0),
                "conversation_routes": int(row["conversation_routes"] or 0),
                "affinity_routes": int(row["affinity_routes"] or 0),
                "random_routes": int(row["random_routes"] or 0),
                "last_request_at": ns_to_iso(
                    int(row["last_request_ns"]) if row["last_request_ns"] is not None else None
                ),
                "base_url": str(latest["base_url"]) if latest is not None else None,
                "endpoint_used": str(latest["endpoint_used"]) if latest is not None else None,
                "last_status_code": int(latest["status_code"]) if latest is not None else None,
                "last_route_reason": str(latest["route_reason"]) if latest is not None else None,
            }
        return endpoint_rows

    def list_conversations(self, *, limit: int, search: str | None) -> list[dict[str, Any]]:
        candidate_limit = limit if not search else min(MAX_CONVERSATIONS, max(limit * 5, 60))
        rows = self.connection.execute(
            """
            SELECT
                c.conversation_id,
                c.created_ns,
                c.last_seen_ns,
                c.last_upstream_port,
                c.last_base_url,
                c.request_count,
                r.request_id,
                r.input_state_hash,
                r.request_meta_json,
                r.response_json,
                r.status_code,
                r.route_reason,
                r.endpoint_used,
                r.created_ns AS latest_request_ns
            FROM conversations AS c
            LEFT JOIN requests AS r
              ON r.request_id = (
                  SELECT request_id
                  FROM requests
                  WHERE conversation_id = c.conversation_id
                  ORDER BY created_ns DESC, request_id DESC
                  LIMIT 1
              )
            ORDER BY c.last_seen_ns DESC, c.created_ns DESC
            LIMIT ?
            """,
            (candidate_limit,),
        )

        lowered_search = search.lower().strip() if search else None
        items: list[dict[str, Any]] = []
        for row in rows:
            request_meta = json.loads(str(row["request_meta_json"])) if row["request_meta_json"] else {}
            response_payload = json.loads(str(row["response_json"])) if row["response_json"] else {}
            input_payload = self.reconstruct_input_payload(request_meta, str(row["input_state_hash"]))
            preview = extract_preview(input_payload)
            model = extract_model(request_meta, response_payload)
            item = {
                "conversation_id": str(row["conversation_id"]),
                "created_at": ns_to_iso(int(row["created_ns"])),
                "last_seen_at": ns_to_iso(int(row["last_seen_ns"])),
                "request_count": int(row["request_count"]),
                "latest_request_id": int(row["request_id"]) if row["request_id"] is not None else None,
                "latest_request_at": ns_to_iso(
                    int(row["latest_request_ns"]) if row["latest_request_ns"] is not None else None
                ),
                "latest_status_code": int(row["status_code"]) if row["status_code"] is not None else None,
                "latest_route_reason": str(row["route_reason"]) if row["route_reason"] is not None else None,
                "latest_endpoint_used": str(row["endpoint_used"]) if row["endpoint_used"] else None,
                "last_upstream_port": (
                    int(row["last_upstream_port"]) if row["last_upstream_port"] is not None else None
                ),
                "last_base_url": str(row["last_base_url"]) if row["last_base_url"] else None,
                "preview": preview,
                "model": model,
            }
            if lowered_search:
                haystack = " ".join(
                    part
                    for part in [
                        item["conversation_id"],
                        preview,
                        model or "",
                        item["latest_endpoint_used"] or "",
                    ]
                    if part
                ).lower()
                if lowered_search not in haystack:
                    continue
            items.append(item)
            if len(items) >= limit:
                break
        return items

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        conversation_row = self.connection.execute(
            """
            SELECT
                conversation_id,
                created_ns,
                last_seen_ns,
                last_upstream_port,
                last_base_url,
                request_count
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if conversation_row is None:
            return None

        request_rows = self.connection.execute(
            """
            SELECT
                request_id,
                input_state_hash,
                request_meta_json,
                response_json,
                endpoint_used,
                base_url,
                upstream_port,
                route_reason,
                status_code,
                created_ns
            FROM requests
            WHERE conversation_id = ?
            ORDER BY created_ns ASC, request_id ASC
            """,
            (conversation_id,),
        )

        requests: list[dict[str, Any]] = []
        for row in request_rows:
            request_meta = json.loads(str(row["request_meta_json"]))
            response_payload = json.loads(str(row["response_json"]))
            input_payload = self.reconstruct_input_payload(request_meta, str(row["input_state_hash"]))
            normalized_messages = normalize_messages(input_payload)
            requests.append(
                {
                    "request_id": int(row["request_id"]),
                    "created_at": ns_to_iso(int(row["created_ns"])),
                    "upstream_port": int(row["upstream_port"]),
                    "endpoint_used": str(row["endpoint_used"]),
                    "base_url": str(row["base_url"]),
                    "route_reason": str(row["route_reason"]),
                    "status_code": int(row["status_code"]),
                    "model": extract_model(request_meta, response_payload),
                    "preview": extract_preview(input_payload),
                    "messages": normalized_messages,
                    "assistant_text": extract_response_text(response_payload),
                    "assistant_reasoning": extract_reasoning_text(response_payload),
                    "input_payload": input_payload,
                    "output_payload": response_payload,
                }
            )

        return {
            "conversation_id": str(conversation_row["conversation_id"]),
            "created_at": ns_to_iso(int(conversation_row["created_ns"])),
            "last_seen_at": ns_to_iso(int(conversation_row["last_seen_ns"])),
            "last_upstream_port": (
                int(conversation_row["last_upstream_port"])
                if conversation_row["last_upstream_port"] is not None
                else None
            ),
            "last_base_url": (
                str(conversation_row["last_base_url"]) if conversation_row["last_base_url"] else None
            ),
            "request_count": int(conversation_row["request_count"]),
            "requests": requests,
        }


def collect_overview() -> dict[str, Any]:
    paths = resolve_runtime_paths()
    health_snapshot = load_health_snapshot(paths.health_state_path)
    config_meta = {
        "config_path": str(paths.config_path),
        "config_exists": paths.config_path.exists(),
        "config_error": paths.config_error,
    }
    file_meta = {
        "state_db_path": str(paths.state_db_path),
        "state_db_exists": paths.state_db_path.exists(),
        "state_db_size_bytes": paths.state_db_path.stat().st_size if paths.state_db_path.exists() else 0,
        "state_db_mtime": file_mtime_iso(paths.state_db_path),
        "health_state_path": str(paths.health_state_path),
        "health_state_exists": paths.health_state_path.exists(),
        "health_state_mtime": file_mtime_iso(paths.health_state_path),
    }

    summary: dict[str, Any] = {
        "total_conversations": 0,
        "total_requests": 0,
        "requests_last_minute": 0,
        "requests_last_5m": 0,
        "requests_last_hour": 0,
        "active_conversations_last_5m": 0,
        "last_request_at": None,
        "last_conversation_at": None,
    }
    endpoint_stats: list[dict[str, Any]] = []
    database_error: str | None = None

    endpoint_rows: dict[int, dict[str, Any]] = {}
    if paths.state_db_path.exists():
        try:
            with ReadOnlyStateDB(paths.state_db_path) as reader:
                summary = reader.load_summary()
                endpoint_rows = reader.load_endpoint_rows()
        except sqlite3.Error as exc:
            database_error = str(exc)

    health_endpoints = health_snapshot.get("endpoints", {}) if isinstance(health_snapshot, dict) else {}
    if not isinstance(health_endpoints, dict):
        health_endpoints = {}

    seen_labels: set[str] = set()
    ports = sorted(
        set(paths.endpoint_labels_by_port)
        | set(endpoint_rows),
        key=lambda port: (
            -(endpoint_rows.get(port, {}).get("total_requests", 0)),
            port,
        ),
    )

    for port in ports:
        row = endpoint_rows.get(port, {})
        label = paths.endpoint_labels_by_port.get(port, f"upstream {port}")
        seen_labels.add(label)
        health = health_endpoints.get(label, {})
        endpoint_stats.append(
            {
                "label": label,
                "upstream_port": port,
                "status": health.get("status", "unknown"),
                "models": health.get("models", []),
                "health_error": health.get("error"),
                "health_requests": health.get("requests"),
                **row,
            }
        )

    for label, health in health_endpoints.items():
        if label in seen_labels:
            continue
        endpoint_stats.append(
            {
                "label": label,
                "status": health.get("status", "unknown"),
                "models": health.get("models", []),
                "health_error": health.get("error"),
                "health_requests": health.get("requests"),
                "upstream_port": None,
                "total_requests": 0,
                "requests_last_minute": 0,
                "requests_last_5m": 0,
                "requests_last_hour": 0,
                "ok_responses": 0,
                "error_responses": 0,
                "conversation_routes": 0,
                "affinity_routes": 0,
                "random_routes": 0,
                "last_request_at": None,
                "base_url": None,
                "endpoint_used": None,
                "last_status_code": None,
                "last_route_reason": None,
            }
        )

    return {
        "generated_at": datetime.now(tz=timezone.utc).astimezone().isoformat(timespec="seconds"),
        "refresh_interval_ms": AUTO_REFRESH_MS,
        "config": config_meta,
        "files": file_meta,
        "summary": summary,
        "database_error": database_error,
        "health_snapshot": {
            "connected": health_snapshot.get("connected", 0) if isinstance(health_snapshot, dict) else 0,
            "models": health_snapshot.get("models", []) if isinstance(health_snapshot, dict) else [],
        },
        "endpoint_stats": endpoint_stats,
    }


app = FastAPI(title="llm-loadbalancer monitor")


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return HTML_TEMPLATE


@app.get("/favicon.ico")
def favicon() -> Response:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">
    <rect width="96" height="96" rx="22" fill="#0f1720"/>
    <path d="M16 30h64v36H16z" fill="#f5efe3"/>
    <path d="M26 22h12v10H26zm32 0h12v10H58z" fill="#ff8e3c"/>
    <path d="M25 48h12v10H25zm17 0h12v10H42zm17 0h12v10H59z" fill="#0f1720"/>
    <path d="M16 70h64" stroke="#ff8e3c" stroke-width="6" stroke-linecap="round"/>
    </svg>"""
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/overview")
def api_overview() -> dict[str, Any]:
    return collect_overview()


@app.get("/api/summary")
def api_summary() -> dict[str, Any]:
    return collect_overview()


@app.get("/api/conversations")
def api_conversations(
    limit: int = Query(default=60, ge=1, le=MAX_CONVERSATIONS),
    search: str | None = Query(default=None),
) -> dict[str, Any]:
    paths = resolve_runtime_paths()
    if not paths.state_db_path.exists():
        return {"items": [], "state_db_exists": False}
    try:
        with ReadOnlyStateDB(paths.state_db_path) as reader:
            items = reader.list_conversations(limit=limit, search=search)
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail=f"Could not read state DB: {exc}") from exc
    return {"items": items, "state_db_exists": True}


@app.get("/api/conversations/{conversation_id}")
def api_conversation_detail(conversation_id: str) -> dict[str, Any]:
    paths = resolve_runtime_paths()
    if not paths.state_db_path.exists():
        raise HTTPException(status_code=404, detail="State DB not found")

    try:
        with ReadOnlyStateDB(paths.state_db_path) as reader:
            detail = reader.get_conversation(conversation_id)
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail=f"Could not read state DB: {exc}") from exc

    if detail is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    labels = paths.endpoint_labels_by_port
    for request in detail["requests"]:
        request["endpoint_label"] = labels.get(request["upstream_port"], f"upstream {request['upstream_port']}")
    detail["endpoint_label"] = (
        labels.get(detail["last_upstream_port"], f"upstream {detail['last_upstream_port']}")
        if detail["last_upstream_port"] is not None
        else None
    )
    return detail


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>llm-loadbalancer monitor</title>
  <style>
    :root {
      --bg: #0b1116;
      --panel: rgba(19, 30, 39, 0.84);
      --panel-strong: rgba(12, 20, 27, 0.96);
      --line: rgba(255, 255, 255, 0.08);
      --line-strong: rgba(255, 142, 60, 0.38);
      --text: #f4efe7;
      --muted: #9fb0b8;
      --accent: #ff8e3c;
      --accent-soft: rgba(255, 142, 60, 0.16);
      --good: #74d39a;
      --warn: #f4c15d;
      --bad: #ff6b6b;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
      --radius: 22px;
      --radius-small: 14px;
      --mono: "IBM Plex Mono", "SFMono-Regular", "Menlo", monospace;
      --ui: "Avenir Next", "Gill Sans", "Trebuchet MS", sans-serif;
      --serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
    }

    * {
      box-sizing: border-box;
    }

    html, body {
      margin: 0;
      min-height: 100%;
      background:
        radial-gradient(circle at top left, rgba(255, 142, 60, 0.18), transparent 30%),
        radial-gradient(circle at 80% 0%, rgba(90, 188, 216, 0.14), transparent 28%),
        linear-gradient(180deg, #111a21 0%, #0b1116 100%);
      color: var(--text);
      font-family: var(--ui);
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background:
        linear-gradient(transparent 0, rgba(255,255,255,0.018) 1px, transparent 1px),
        linear-gradient(90deg, transparent 0, rgba(255,255,255,0.018) 1px, transparent 1px);
      background-size: 100% 26px, 26px 100%;
      mask-image: radial-gradient(circle at center, black 55%, transparent 100%);
      opacity: 0.55;
    }

    .shell {
      position: relative;
      z-index: 1;
      max-width: 1680px;
      margin: 0 auto;
      padding: 28px;
    }

    .masthead {
      display: grid;
      grid-template-columns: 1.3fr 0.7fr;
      gap: 18px;
      margin-bottom: 18px;
    }

    .hero, .paths, .summary-card, .table-panel, .list-panel, .detail-panel, .request-tab {
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }

    .hero {
      border-radius: var(--radius);
      padding: 24px 26px;
      overflow: hidden;
      position: relative;
    }

    .hero::after {
      content: "";
      position: absolute;
      inset: auto -6% -32% auto;
      width: 280px;
      height: 280px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(255,142,60,0.24), transparent 65%);
      pointer-events: none;
    }

    .hero h1 {
      margin: 0;
      font-family: var(--serif);
      font-size: clamp(2rem, 3vw, 3.4rem);
      letter-spacing: -0.04em;
      line-height: 0.96;
      max-width: 14ch;
    }

    .hero p {
      margin: 14px 0 0;
      max-width: 64ch;
      color: var(--muted);
      font-size: 0.98rem;
      line-height: 1.6;
    }

    .hero-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 18px;
    }

    .live-pill, .tiny-pill, .route-pill, .status-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      border: 1px solid var(--line);
      color: var(--text);
      background: rgba(255, 255, 255, 0.03);
    }

    .pulse {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--good);
      box-shadow: 0 0 0 rgba(116, 211, 154, 0.5);
      animation: pulse 1.9s infinite;
    }

    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(116, 211, 154, 0.45); }
      70% { box-shadow: 0 0 0 14px rgba(116, 211, 154, 0); }
      100% { box-shadow: 0 0 0 0 rgba(116, 211, 154, 0); }
    }

    .paths {
      border-radius: var(--radius);
      padding: 18px;
      display: grid;
      gap: 12px;
      align-content: start;
    }

    .eyebrow {
      font-size: 0.72rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .path-block {
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.035);
      border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .path-label {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--muted);
      margin-bottom: 6px;
    }

    .path-value {
      font-family: var(--mono);
      font-size: 0.84rem;
      color: var(--text);
      word-break: break-all;
      line-height: 1.45;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }

    .summary-card {
      border-radius: 20px;
      padding: 18px;
      min-height: 130px;
      display: grid;
      gap: 8px;
      align-content: start;
    }

    .summary-card .label {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--muted);
    }

    .summary-card .value {
      font-size: clamp(1.7rem, 2.3vw, 2.55rem);
      line-height: 0.95;
      font-family: var(--serif);
      letter-spacing: -0.04em;
    }

    .summary-card .meta {
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.45;
    }

    .workspace {
      display: grid;
      grid-template-columns: minmax(340px, 0.8fr) minmax(0, 1.7fr);
      gap: 18px;
      align-items: start;
    }

    .left-column, .right-column {
      display: grid;
      gap: 18px;
    }

    .table-panel, .list-panel, .detail-panel {
      border-radius: var(--radius);
      overflow: hidden;
    }

    .panel-head {
      padding: 18px 20px 14px;
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 14px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.03), transparent);
    }

    .panel-title {
      margin: 0;
      font-size: 1rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 600;
    }

    .panel-subtitle {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 0.85rem;
    }

    .panel-body {
      padding: 0 20px 20px;
    }

    .endpoint-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }

    .endpoint-table thead th {
      text-align: left;
      font-size: 0.72rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--muted);
      padding: 14px 0 10px;
      border-bottom: 1px solid var(--line);
    }

    .endpoint-table tbody td {
      padding: 16px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      vertical-align: top;
    }

    .endpoint-name {
      display: grid;
      gap: 8px;
    }

    .endpoint-name strong {
      font-size: 0.96rem;
    }

    .endpoint-secondary {
      color: var(--muted);
      font-family: var(--mono);
      font-size: 0.8rem;
      line-height: 1.45;
    }

    .status-chip {
      font-family: var(--mono);
      letter-spacing: 0.12em;
    }

    .status-up {
      color: var(--good);
      border-color: rgba(116, 211, 154, 0.28);
      background: rgba(116, 211, 154, 0.1);
    }

    .status-down {
      color: var(--bad);
      border-color: rgba(255, 107, 107, 0.28);
      background: rgba(255, 107, 107, 0.1);
    }

    .status-unknown {
      color: var(--warn);
      border-color: rgba(244, 193, 93, 0.28);
      background: rgba(244, 193, 93, 0.1);
    }

    .kpi-stack {
      display: grid;
      gap: 4px;
    }

    .kpi-main {
      font-size: 1.16rem;
      font-family: var(--mono);
    }

    .kpi-meta {
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.45;
    }

    .search-wrap {
      padding: 16px 20px 0;
    }

    .search-input {
      width: 100%;
      padding: 12px 14px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.04);
      color: var(--text);
      font-size: 0.94rem;
      outline: none;
    }

    .search-input:focus {
      border-color: var(--line-strong);
      box-shadow: 0 0 0 4px rgba(255, 142, 60, 0.08);
    }

    .conversation-list {
      padding: 14px;
      display: grid;
      gap: 10px;
      max-height: 820px;
      overflow: auto;
    }

    .conversation-card {
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 18px;
      padding: 14px;
      background: rgba(255,255,255,0.028);
      cursor: pointer;
      transition: transform 140ms ease, border-color 140ms ease, background 140ms ease;
    }

    .conversation-card:hover {
      transform: translateY(-1px);
      border-color: rgba(255, 142, 60, 0.24);
      background: rgba(255, 255, 255, 0.04);
    }

    .conversation-card.active {
      border-color: rgba(255, 142, 60, 0.42);
      background: rgba(255, 142, 60, 0.08);
    }

    .conversation-card-head {
      display: flex;
      gap: 8px;
      justify-content: space-between;
      align-items: start;
      margin-bottom: 8px;
    }

    .conversation-id {
      font-family: var(--mono);
      font-size: 0.78rem;
      color: var(--muted);
      word-break: break-all;
    }

    .conversation-preview {
      font-size: 0.94rem;
      line-height: 1.45;
      color: var(--text);
      min-height: 2.8em;
    }

    .conversation-meta {
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 0.78rem;
    }

    .detail-shell {
      padding: 18px;
      display: grid;
      gap: 16px;
    }

    .detail-empty {
      min-height: 520px;
      display: grid;
      place-items: center;
      color: var(--muted);
      text-align: center;
      padding: 28px;
    }

    .detail-header {
      display: grid;
      gap: 10px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
    }

    .detail-title {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
    }

    .detail-title h2 {
      margin: 0;
      font-size: clamp(1.4rem, 2vw, 2rem);
      font-family: var(--serif);
      letter-spacing: -0.03em;
    }

    .detail-id {
      font-family: var(--mono);
      color: var(--muted);
      font-size: 0.82rem;
      word-break: break-all;
    }

    .detail-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .request-tabs {
      display: flex;
      gap: 10px;
      overflow: auto;
      padding-bottom: 4px;
    }

    .request-tab {
      min-width: 210px;
      border-radius: 18px;
      padding: 12px 14px;
      border: 1px solid rgba(255, 255, 255, 0.06);
      background: rgba(255,255,255,0.03);
      color: var(--text);
      cursor: pointer;
      text-align: left;
      transition: transform 140ms ease, border-color 140ms ease, background 140ms ease;
    }

    .request-tab:hover {
      transform: translateY(-1px);
    }

    .request-tab.active {
      border-color: rgba(255, 142, 60, 0.42);
      background: rgba(255, 142, 60, 0.1);
    }

    .request-tab .request-title {
      font-size: 0.82rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }

    .request-tab .request-preview {
      font-size: 0.88rem;
      line-height: 1.45;
      min-height: 2.6em;
    }

    .message-stream {
      display: grid;
      gap: 12px;
    }

    .message-card {
      border-radius: 18px;
      padding: 14px 16px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.028);
    }

    .message-card.role-user {
      border-left: 4px solid rgba(90, 188, 216, 0.86);
    }

    .message-card.role-assistant {
      border-left: 4px solid rgba(116, 211, 154, 0.86);
    }

    .message-card.role-system {
      border-left: 4px solid rgba(244, 193, 93, 0.86);
    }

    .message-card.role-tool {
      border-left: 4px solid rgba(255, 142, 60, 0.86);
    }

    .message-card.role-reasoning {
      border-left: 4px solid rgba(200, 170, 255, 0.86);
      background: rgba(140, 112, 255, 0.08);
    }

    .message-head {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 10px;
    }

    .message-role {
      padding: 5px 9px;
      border-radius: 999px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--text);
    }

    .message-tools {
      color: var(--muted);
      font-size: 0.78rem;
      font-family: var(--mono);
    }

    .message-content, .json-box {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      font-size: 0.87rem;
      line-height: 1.6;
      color: var(--text);
    }

    details {
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.03);
      overflow: hidden;
    }

    details summary {
      cursor: pointer;
      padding: 14px 16px;
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      list-style: none;
    }

    details summary::-webkit-details-marker {
      display: none;
    }

    .json-wrap {
      padding: 0 16px 16px;
    }

    .error-banner {
      margin: 0 0 18px;
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255, 107, 107, 0.12);
      border: 1px solid rgba(255, 107, 107, 0.22);
      color: #ffd9d9;
      font-family: var(--mono);
      font-size: 0.86rem;
      line-height: 1.5;
      white-space: pre-wrap;
    }

    .footer-note {
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.8rem;
      text-align: right;
    }

    @media (max-width: 1320px) {
      .summary-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }
    }

    @media (max-width: 1080px) {
      .masthead, .workspace {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 820px) {
      .shell {
        padding: 16px;
      }

      .summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    @media (max-width: 560px) {
      .summary-grid {
        grid-template-columns: 1fr;
      }

      .request-tab {
        min-width: 180px;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="masthead">
      <div class="hero">
        <div class="eyebrow">Read-only database monitor</div>
        <h1>Live endpoint health and conversation state.</h1>
        <p>
          This dashboard only reads the SQLite state database and the load balancer health snapshot.
          It never writes to serving state, so we can watch traffic, endpoint drift, and full
          reconstructed conversations without touching the proxy path.
        </p>
        <div class="hero-row">
          <div class="live-pill"><span class="pulse"></span><span>Auto refresh every 3s</span></div>
          <div class="tiny-pill" id="refresh-pill">Waiting for first refresh</div>
          <div class="tiny-pill" id="health-pill">Health snapshot pending</div>
        </div>
      </div>
      <div class="paths" id="paths-panel"></div>
    </section>

    <div id="database-error"></div>

    <section class="summary-grid" id="summary-grid"></section>

    <section class="workspace">
      <div class="left-column">
        <section class="list-panel">
          <div class="panel-head">
            <div>
              <h2 class="panel-title">Conversations</h2>
              <p class="panel-subtitle">Latest reconstructed conversation snapshots from SQLite.</p>
            </div>
            <div class="tiny-pill" id="conversation-count-pill">0 loaded</div>
          </div>
          <div class="search-wrap">
            <input id="search-input" class="search-input" type="search" placeholder="Search conversation id, preview, model, endpoint">
          </div>
          <div class="conversation-list" id="conversation-list"></div>
        </section>
      </div>

      <div class="right-column">
        <section class="table-panel">
          <div class="panel-head">
            <div>
              <h2 class="panel-title">Endpoints</h2>
              <p class="panel-subtitle">Merged view of health_state.json and request history in the database.</p>
            </div>
            <div class="tiny-pill" id="endpoint-count-pill">0 endpoints</div>
          </div>
          <div class="panel-body" id="endpoint-table-wrap"></div>
        </section>

        <section class="detail-panel">
          <div class="detail-shell" id="detail-shell">
            <div class="detail-empty">
              <div>
                <div class="eyebrow">Conversation detail</div>
                <h2 style="margin: 10px 0 8px; font-family: var(--serif); font-size: 2rem;">Pick a conversation.</h2>
                <p style="max-width: 44ch; color: var(--muted); line-height: 1.6;">
                  The monitor will render the selected request timeline, the reconstructed message history,
                  the assistant response, and the raw request and response payloads.
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </section>

    <div class="footer-note">This page polls the backend in read-only mode.</div>
  </div>

  <script>
    const state = {
      overview: null,
      conversations: [],
      selectedConversationId: null,
      selectedConversation: null,
      selectedRequestId: null,
      search: "",
      refreshTimer: null,
      searchTimer: null,
      lastLoadedAt: null,
    };

    const nodes = {
      summaryGrid: document.getElementById("summary-grid"),
      pathsPanel: document.getElementById("paths-panel"),
      endpointTableWrap: document.getElementById("endpoint-table-wrap"),
      conversationList: document.getElementById("conversation-list"),
      detailShell: document.getElementById("detail-shell"),
      searchInput: document.getElementById("search-input"),
      refreshPill: document.getElementById("refresh-pill"),
      healthPill: document.getElementById("health-pill"),
      endpointCountPill: document.getElementById("endpoint-count-pill"),
      conversationCountPill: document.getElementById("conversation-count-pill"),
      databaseError: document.getElementById("database-error"),
    };

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function formatNumber(value) {
      return Number(value || 0).toLocaleString();
    }

    function formatWhen(value) {
      if (!value) {
        return "—";
      }
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) {
        return value;
      }
      return date.toLocaleString();
    }

    function relativeWhen(value) {
      if (!value) {
        return "No activity";
      }
      const date = new Date(value);
      const diffSeconds = Math.floor((Date.now() - date.getTime()) / 1000);
      if (diffSeconds < 60) return diffSeconds + "s ago";
      if (diffSeconds < 3600) return Math.floor(diffSeconds / 60) + "m ago";
      if (diffSeconds < 86400) return Math.floor(diffSeconds / 3600) + "h ago";
      return Math.floor(diffSeconds / 86400) + "d ago";
    }

    function jsonBox(value) {
      return "<pre class=\\"json-box\\">" + escapeHtml(JSON.stringify(value, null, 2)) + "</pre>";
    }

    function routePill(label, value) {
      return "<span class=\\"route-pill\\">" + escapeHtml(label) + ": " + escapeHtml(value) + "</span>";
    }

    function statusClass(status) {
      if (status === "up") return "status-up";
      if (status === "down") return "status-down";
      return "status-unknown";
    }

    function codeClass(code) {
      if (code == null) return "status-unknown";
      return Number(code) >= 400 ? "status-down" : "status-up";
    }

    async function fetchJson(url) {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || ("Request failed: " + response.status));
      }
      return response.json();
    }

    async function loadOverview() {
      state.overview = await fetchJson("/api/overview");
      state.lastLoadedAt = new Date();
      renderOverview();
    }

    async function loadConversations() {
      const params = new URLSearchParams();
      params.set("limit", "60");
      if (state.search.trim()) {
        params.set("search", state.search.trim());
      }
      const payload = await fetchJson("/api/conversations?" + params.toString());
      state.conversations = payload.items || [];
      if (!state.conversations.length) {
        state.selectedConversationId = null;
        state.selectedConversation = null;
        state.selectedRequestId = null;
      } else if (!state.selectedConversationId) {
        state.selectedConversationId = state.conversations[0].conversation_id;
      }
      nodes.conversationCountPill.textContent = state.conversations.length + " loaded";
      renderConversationList();
      if (state.selectedConversationId) {
        const stillVisible = state.conversations.some(
          (item) => item.conversation_id === state.selectedConversationId,
        );
        if (!stillVisible) {
          if (state.conversations.length) {
            state.selectedConversationId = state.conversations[0].conversation_id;
          } else {
            state.selectedConversationId = null;
            state.selectedConversation = null;
            state.selectedRequestId = null;
          }
        }
      }
    }

    async function loadSelectedConversation() {
      if (!state.selectedConversationId) {
        state.selectedConversation = null;
        renderConversationDetail();
        return;
      }
      state.selectedConversation = await fetchJson(
        "/api/conversations/" + encodeURIComponent(state.selectedConversationId),
      );
      const requests = state.selectedConversation.requests || [];
      if (!requests.length) {
        state.selectedRequestId = null;
      } else if (!requests.some((item) => item.request_id === state.selectedRequestId)) {
        state.selectedRequestId = requests[requests.length - 1].request_id;
      }
      renderConversationDetail();
    }

    async function refreshAll() {
      const errors = [];

      try {
        await loadConversations();
      } catch (error) {
        errors.push(error);
      }

      try {
        await loadSelectedConversation();
      } catch (error) {
        errors.push(error);
      }

      try {
        await loadOverview();
      } catch (error) {
        errors.push(error);
      }

      if (errors.length) {
        nodes.databaseError.innerHTML =
          "<div class=\\"error-banner\\">" +
          errors.map((error) => escapeHtml(error.message || String(error))).join("<br>") +
          "</div>";
        return;
      }

      nodes.databaseError.innerHTML = "";
    }

    function renderOverview() {
      const overview = state.overview;
      if (!overview) return;

      const summary = overview.summary || {};
      const files = overview.files || {};
      const config = overview.config || {};
      const health = overview.health_snapshot || {};

      nodes.refreshPill.textContent = state.lastLoadedAt
        ? "Last refresh " + state.lastLoadedAt.toLocaleTimeString()
        : "Waiting for first refresh";
      nodes.healthPill.textContent =
        "Connected " + formatNumber(health.connected || 0) +
        " endpoint" + ((health.connected || 0) === 1 ? "" : "s");
      nodes.endpointCountPill.textContent =
        (overview.endpoint_stats || []).length + " endpoints";

      const cards = [
        {
          label: "Requests total",
          value: formatNumber(summary.total_requests),
          meta: "1m " + formatNumber(summary.requests_last_minute) +
            " · 5m " + formatNumber(summary.requests_last_5m) +
            " · 1h " + formatNumber(summary.requests_last_hour),
        },
        {
          label: "Conversations total",
          value: formatNumber(summary.total_conversations),
          meta: "Active in last 5m: " + formatNumber(summary.active_conversations_last_5m),
        },
        {
          label: "Last request",
          value: summary.last_request_at ? relativeWhen(summary.last_request_at) : "none",
          meta: formatWhen(summary.last_request_at),
        },
        {
          label: "Last conversation",
          value: summary.last_conversation_at ? relativeWhen(summary.last_conversation_at) : "none",
          meta: formatWhen(summary.last_conversation_at),
        },
        {
          label: "State DB",
          value: files.state_db_exists ? formatNumber(files.state_db_size_bytes) + " B" : "missing",
          meta: files.state_db_exists ? formatWhen(files.state_db_mtime) : files.state_db_path,
        },
        {
          label: "Health snapshot",
          value: files.health_state_exists ? "present" : "missing",
          meta: files.health_state_exists ? formatWhen(files.health_state_mtime) : files.health_state_path,
        },
      ];

      nodes.summaryGrid.innerHTML = cards.map((card) => `
        <article class="summary-card">
          <div class="label">${escapeHtml(card.label)}</div>
          <div class="value">${escapeHtml(card.value)}</div>
          <div class="meta">${escapeHtml(card.meta)}</div>
        </article>
      `).join("");

      nodes.pathsPanel.innerHTML = `
        <div>
          <div class="eyebrow">Resolved runtime paths</div>
        </div>
        <div class="path-block">
          <div class="path-label">Config</div>
          <div class="path-value">${escapeHtml(config.config_path || "—")}</div>
        </div>
        <div class="path-block">
          <div class="path-label">State DB</div>
          <div class="path-value">${escapeHtml(files.state_db_path || "—")}</div>
        </div>
        <div class="path-block">
          <div class="path-label">Health State</div>
          <div class="path-value">${escapeHtml(files.health_state_path || "—")}</div>
        </div>
        ${
          config.config_error
            ? `<div class="error-banner">${escapeHtml(config.config_error)}</div>`
            : ""
        }
      `;

      if (overview.database_error) {
        nodes.databaseError.innerHTML =
          "<div class=\\"error-banner\\">Database read error\\n" +
          escapeHtml(overview.database_error) +
          "</div>";
      }

      renderEndpoints();
    }

    function renderEndpoints() {
      const endpoints = (state.overview && state.overview.endpoint_stats) || [];
      if (!endpoints.length) {
        nodes.endpointTableWrap.innerHTML =
          "<div class=\\"detail-empty\\" style=\\"min-height: 260px;\\">No endpoint data yet.</div>";
        return;
      }

      nodes.endpointTableWrap.innerHTML = `
        <table class="endpoint-table">
          <thead>
            <tr>
              <th>Endpoint</th>
              <th>Status</th>
              <th>Traffic</th>
              <th>Routing</th>
              <th>Recent</th>
            </tr>
          </thead>
          <tbody>
            ${endpoints.map((item) => `
              <tr>
                <td>
                  <div class="endpoint-name">
                    <strong>${escapeHtml(item.label || "unknown")}</strong>
                    <div class="endpoint-secondary">
                      ${item.base_url ? escapeHtml(item.base_url) + "<br>" : ""}
                      ${item.endpoint_used ? escapeHtml(item.endpoint_used) : ""}
                    </div>
                  </div>
                </td>
                <td>
                  <div class="kpi-stack">
                    <span class="status-chip ${statusClass(item.status)}">${escapeHtml(item.status || "unknown")}</span>
                    <div class="kpi-meta">
                      ${(item.models || []).length ? escapeHtml((item.models || []).join(", ")) : "No model list"}
                      ${item.health_error ? "<br>" + escapeHtml(item.health_error) : ""}
                    </div>
                  </div>
                </td>
                <td>
                  <div class="kpi-stack">
                    <div class="kpi-main">${escapeHtml(formatNumber(item.total_requests))}</div>
                    <div class="kpi-meta">
                      1m ${escapeHtml(formatNumber(item.requests_last_minute))}
                      · 5m ${escapeHtml(formatNumber(item.requests_last_5m))}
                      · 1h ${escapeHtml(formatNumber(item.requests_last_hour))}
                    </div>
                  </div>
                </td>
                <td>
                  <div class="kpi-stack">
                    <div class="kpi-main">
                      ok ${escapeHtml(formatNumber(item.ok_responses))}
                      · err ${escapeHtml(formatNumber(item.error_responses))}
                    </div>
                    <div class="kpi-meta">
                      conv ${escapeHtml(formatNumber(item.conversation_routes))}
                      · aff ${escapeHtml(formatNumber(item.affinity_routes))}
                      · rnd ${escapeHtml(formatNumber(item.random_routes))}
                    </div>
                  </div>
                </td>
                <td>
                  <div class="kpi-stack">
                    <div class="kpi-main">${escapeHtml(relativeWhen(item.last_request_at))}</div>
                    <div class="kpi-meta">
                      ${escapeHtml(formatWhen(item.last_request_at))}
                      ${item.last_status_code ? "<br>status " + escapeHtml(item.last_status_code) : ""}
                    </div>
                  </div>
                </td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    }

    function renderConversationList() {
      if (!state.conversations.length) {
        nodes.conversationList.innerHTML =
          "<div class=\\"detail-empty\\" style=\\"min-height: 240px;\\">No conversations matched.</div>";
        return;
      }

      nodes.conversationList.innerHTML = state.conversations.map((item) => `
        <article
          class="conversation-card ${item.conversation_id === state.selectedConversationId ? "active" : ""}"
          data-conversation-id="${escapeHtml(item.conversation_id)}"
        >
          <div class="conversation-card-head">
            <div class="conversation-id">${escapeHtml(item.conversation_id)}</div>
            <span class="status-chip ${codeClass(item.latest_status_code)}">
              ${escapeHtml(item.latest_status_code || "—")}
            </span>
          </div>
          <div class="conversation-preview">${escapeHtml(item.preview || "No preview text")}</div>
          <div class="conversation-meta">
            ${item.model ? routePill("model", item.model) : ""}
            ${routePill("reqs", item.request_count)}
            ${item.latest_route_reason ? routePill("route", item.latest_route_reason) : ""}
            ${item.last_upstream_port ? routePill("port", item.last_upstream_port) : ""}
            ${routePill("seen", relativeWhen(item.last_seen_at))}
          </div>
        </article>
      `).join("");

      for (const card of nodes.conversationList.querySelectorAll(".conversation-card")) {
        card.addEventListener("click", async () => {
          await selectConversation(card.dataset.conversationId);
        });
      }
    }

    async function selectConversation(conversationId) {
      if (!conversationId || conversationId === state.selectedConversationId) {
        return;
      }
      state.selectedConversationId = conversationId;
      renderConversationList();
      for (const card of nodes.conversationList.querySelectorAll(".conversation-card")) {
        if (card.dataset.conversationId === conversationId) {
          card.scrollIntoView({ block: "nearest" });
          break;
        }
      }
      await loadSelectedConversation();
    }

    async function selectConversationByOffset(offset) {
      if (!state.conversations.length) {
        return;
      }
      const currentIndex = state.conversations.findIndex(
        (item) => item.conversation_id === state.selectedConversationId,
      );
      const nextIndex =
        currentIndex === -1
          ? 0
          : (currentIndex + offset + state.conversations.length) % state.conversations.length;
      await selectConversation(state.conversations[nextIndex].conversation_id);
    }

    function isTypingTarget(target) {
      return target instanceof Element &&
        target.closest("input, textarea, select, [contenteditable='true']");
    }

    function renderConversationDetail() {
      const detail = state.selectedConversation;
      if (!detail) {
        nodes.detailShell.innerHTML = `
          <div class="detail-empty">
            <div>
              <div class="eyebrow">Conversation detail</div>
              <h2 style="margin: 10px 0 8px; font-family: var(--serif); font-size: 2rem;">Pick a conversation.</h2>
              <p style="max-width: 44ch; color: var(--muted); line-height: 1.6;">
                The monitor will render the selected request timeline, the reconstructed message history,
                the assistant response, and the raw request and response payloads.
              </p>
            </div>
          </div>
        `;
        return;
      }

      const requests = detail.requests || [];
      const activeRequest = requests.find((item) => item.request_id === state.selectedRequestId) || requests[requests.length - 1];

      nodes.detailShell.innerHTML = `
        <div class="detail-header">
          <div class="detail-title">
            <div>
              <div class="eyebrow">Conversation detail</div>
              <h2>${escapeHtml(activeRequest?.preview || "Conversation snapshot")}</h2>
            </div>
            <div class="detail-meta">
              ${routePill("requests", detail.request_count)}
              ${detail.endpoint_label ? routePill("endpoint", detail.endpoint_label) : ""}
              ${routePill("created", relativeWhen(detail.created_at))}
              ${routePill("seen", relativeWhen(detail.last_seen_at))}
            </div>
          </div>
          <div class="detail-id">${escapeHtml(detail.conversation_id)}</div>
        </div>

        <div class="request-tabs">
          ${requests.map((request) => `
            <button
              class="request-tab ${request.request_id === state.selectedRequestId ? "active" : ""}"
              data-request-id="${escapeHtml(request.request_id)}"
              type="button"
            >
              <div class="request-title">Request ${escapeHtml(request.request_id)}</div>
              <div class="request-preview">${escapeHtml(request.preview || "No preview text")}</div>
              <div class="conversation-meta" style="margin-top: 8px;">
                ${request.model ? routePill("model", request.model) : ""}
                ${routePill("route", request.route_reason)}
                ${routePill("status", request.status_code)}
              </div>
            </button>
          `).join("")}
        </div>

        ${activeRequest ? renderActiveRequest(activeRequest) : ""}
      `;

      for (const button of nodes.detailShell.querySelectorAll(".request-tab")) {
        button.addEventListener("click", () => {
          state.selectedRequestId = Number(button.dataset.requestId);
          renderConversationDetail();
        });
      }
    }

    function renderActiveRequest(request) {
      const messages = request.messages || [];
      const reasoning = request.assistant_reasoning || "";
      return `
        <div class="detail-meta">
          ${routePill("endpoint", request.endpoint_label || ("upstream " + request.upstream_port))}
          ${routePill("used", request.endpoint_used)}
          ${routePill("time", formatWhen(request.created_at))}
          ${routePill("route", request.route_reason)}
          ${routePill("status", request.status_code)}
        </div>

        <div class="message-stream">
          ${messages.map((message) => `
            <article class="message-card role-${escapeHtml(message.role || "unknown")}">
              <div class="message-head">
                <span class="message-role">${escapeHtml(message.role || "unknown")}</span>
                ${
                  message.tool_names && message.tool_names.length
                    ? `<span class="message-tools">${escapeHtml(message.tool_names.join(", "))}</span>`
                    : ""
                }
              </div>
              <pre class="message-content">${escapeHtml(message.content || "(empty)")}</pre>
            </article>
          `).join("")}

          ${
            reasoning
              ? `
                <article class="message-card role-reasoning">
                  <div class="message-head">
                    <span class="message-role">reasoning</span>
                  </div>
                  <pre class="message-content">${escapeHtml(reasoning)}</pre>
                </article>
              `
              : ""
          }

          <article class="message-card role-assistant">
            <div class="message-head">
              <span class="message-role">assistant</span>
            </div>
            <pre class="message-content">${escapeHtml(request.assistant_text || "(empty)")}</pre>
          </article>
        </div>

        <details>
          <summary>Request payload</summary>
          <div class="json-wrap">${jsonBox(request.input_payload)}</div>
        </details>
        <details>
          <summary>Response payload</summary>
          <div class="json-wrap">${jsonBox(request.output_payload)}</div>
        </details>
      `;
    }

    nodes.searchInput.addEventListener("input", () => {
      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(async () => {
        state.search = nodes.searchInput.value;
        await loadConversations();
        await loadSelectedConversation();
      }, 220);
    });

    document.addEventListener("keydown", async (event) => {
      if (
        !event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        isTypingTarget(event.target)
      ) {
        return;
      }
      if (event.key === "ArrowUp") {
        event.preventDefault();
        await selectConversationByOffset(-1);
      } else if (event.key === "ArrowDown") {
        event.preventDefault();
        await selectConversationByOffset(1);
      }
    });

    async function boot() {
      await refreshAll();
      state.refreshTimer = setInterval(refreshAll, ${AUTO_REFRESH_MS});
    }

    boot();
  </script>
</body>
</html>
"""


def main() -> None:
    paths = resolve_runtime_paths()
    print(f"Starting monitor at http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    print(f"Config path: {paths.config_path}")
    print(f"State DB: {paths.state_db_path}")
    print(f"Health snapshot: {paths.health_state_path}")
    if paths.config_error:
        print(f"Config warning: {paths.config_error}")
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
