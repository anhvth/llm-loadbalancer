#!/usr/bin/env python3
"""SQLite-backed reverse proxy with deduplicated conversation tracking."""

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
import traceback
import urllib.parse
import uuid
from typing import Any, Iterable, Iterator

import httpx
import uvicorn
from loguru import logger
from tabulate import tabulate

from keep_connection import TunnelConfig, parse_config

try:
    import resource as resource_module
except ImportError:  # pragma: no cover - resource is Unix-only.
    resource_module = None


EXACT_IDENTITY_IGNORED_KEYS = {"cache_control", "signature"}
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
MESSAGE_AFFINITY_IGNORED_KEYS = {"cache_control", "role", "signature", "type"}
MESSAGE_AFFINITY_INIT_LOCK_RETRY_TIMEOUT_SECONDS = 30.0
MESSAGE_AFFINITY_INIT_LOCK_RETRY_DELAY_SECONDS = 0.1
STATE_DB_INIT_LOCK_RETRY_TIMEOUT_SECONDS = MESSAGE_AFFINITY_INIT_LOCK_RETRY_TIMEOUT_SECONDS
STATE_DB_INIT_LOCK_RETRY_DELAY_SECONDS = MESSAGE_AFFINITY_INIT_LOCK_RETRY_DELAY_SECONDS
STATE_DB_BUSY_TIMEOUT_MILLISECONDS = 30_000
WRITE_BATCH_SIZE = 64
UPSTREAM_TIMEOUT_SECONDS = 300.0
HEALTHCHECK_TIMEOUT_SECONDS = 2.0
HEALTHCHECK_INTERVAL_SECONDS = 30.0
HEALTHCHECK_CONNECT_RETRIES = 3
HEALTHCHECK_CONNECT_RETRY_DELAY_SECONDS = 0.2
INITIAL_HEALTHCHECK_TIMEOUT_SECONDS = 15.0
INITIAL_HEALTHCHECK_RETRY_DELAY_SECONDS = 0.25
HEALTHCHECK_STATE_FILENAME = "health_state.json"
BUG_REPORT_PATH = pathlib.Path(".log/error/bug_report.json")
FD_BASE_OVERHEAD = 128
FD_PER_CONCURRENT_REQUEST = 2
FD_PER_ENDPOINT_HEALTHCHECK = 1
FD_FALLBACK_SOFT_LIMIT = 1024
MIN_LISTEN_BACKLOG = 4096
MAX_LISTEN_BACKLOG = 65535
LLM_PROXY_V2_EFFECTIVE_WORKER_CONCURRENCY_ENV = "LLM_PROXY_EFFECTIVE_WORKER_CONCURRENCY"

def _log_sink(message):
    sys.stderr.write(str(message))


# Default: INFO until first app instantiation overrides it.
logger.add(_log_sink, format="{message}", level="INFO")


def _configure_logger(verbose: bool) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(_log_sink, format="{message}", level=level)


@dataclasses.dataclass(frozen=True)
class EndpointCheckResult:
    host: str
    port: int
    models: tuple[str, ...] = ()
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class FileDescriptorPlan:
    configured_worker_concurrency: int
    effective_worker_concurrency: int
    required_soft_limit: int
    soft_limit: int | None
    hard_limit: int | None
    raised_soft_limit: bool = False
    capped_by_limit: bool = False
    warning: str | None = None


def fd_limit_required_for_worker_concurrency(cfg: TunnelConfig, worker_concurrency: int) -> int:
    return (
        FD_BASE_OVERHEAD
        + len(cfg.hosts) * FD_PER_ENDPOINT_HEALTHCHECK
        + worker_concurrency * FD_PER_CONCURRENT_REQUEST
    )


def worker_concurrency_for_fd_limit(cfg: TunnelConfig, fd_limit: int) -> int:
    reserved = FD_BASE_OVERHEAD + len(cfg.hosts) * FD_PER_ENDPOINT_HEALTHCHECK
    usable = max(0, fd_limit - reserved)
    return max(1, usable // FD_PER_CONCURRENT_REQUEST)


def _fd_limit_allows(limit: int, required: int) -> bool:
    assert resource_module is not None
    return limit == resource_module.RLIM_INFINITY or limit >= required


def plan_file_descriptor_limits(cfg: TunnelConfig) -> FileDescriptorPlan:
    configured_concurrency = max(1, cfg.load_balancer_worker_concurrency)
    required_soft_limit = fd_limit_required_for_worker_concurrency(cfg, configured_concurrency)
    if resource_module is None:
        effective_concurrency = min(
            configured_concurrency,
            worker_concurrency_for_fd_limit(cfg, FD_FALLBACK_SOFT_LIMIT),
        )
        return FileDescriptorPlan(
            configured_worker_concurrency=configured_concurrency,
            effective_worker_concurrency=effective_concurrency,
            required_soft_limit=required_soft_limit,
            soft_limit=None,
            hard_limit=None,
            capped_by_limit=effective_concurrency < configured_concurrency,
            warning="Could not inspect RLIMIT_NOFILE; using conservative worker concurrency",
        )

    soft_limit, hard_limit = resource_module.getrlimit(resource_module.RLIMIT_NOFILE)
    if _fd_limit_allows(soft_limit, required_soft_limit):
        return FileDescriptorPlan(
            configured_worker_concurrency=configured_concurrency,
            effective_worker_concurrency=configured_concurrency,
            required_soft_limit=required_soft_limit,
            soft_limit=soft_limit,
            hard_limit=hard_limit,
        )

    if _fd_limit_allows(hard_limit, required_soft_limit):
        try:
            resource_module.setrlimit(
                resource_module.RLIMIT_NOFILE,
                (required_soft_limit, hard_limit),
            )
        except (OSError, ValueError) as exc:
            effective_concurrency = min(
                configured_concurrency,
                worker_concurrency_for_fd_limit(cfg, soft_limit),
            )
            return FileDescriptorPlan(
                configured_worker_concurrency=configured_concurrency,
                effective_worker_concurrency=effective_concurrency,
                required_soft_limit=required_soft_limit,
                soft_limit=soft_limit,
                hard_limit=hard_limit,
                capped_by_limit=effective_concurrency < configured_concurrency,
                warning=f"Could not raise RLIMIT_NOFILE: {exc}",
            )
        return FileDescriptorPlan(
            configured_worker_concurrency=configured_concurrency,
            effective_worker_concurrency=configured_concurrency,
            required_soft_limit=required_soft_limit,
            soft_limit=required_soft_limit,
            hard_limit=hard_limit,
            raised_soft_limit=True,
        )

    effective_concurrency = min(
        configured_concurrency,
        worker_concurrency_for_fd_limit(cfg, hard_limit),
    )
    return FileDescriptorPlan(
        configured_worker_concurrency=configured_concurrency,
        effective_worker_concurrency=effective_concurrency,
        required_soft_limit=required_soft_limit,
        soft_limit=soft_limit,
        hard_limit=hard_limit,
        capped_by_limit=effective_concurrency < configured_concurrency,
        warning=(
            "RLIMIT_NOFILE hard limit is too low for configured worker concurrency; "
            f"using worker-concurrency={effective_concurrency}"
        ),
    )


def resolve_effective_worker_concurrency(cfg: TunnelConfig) -> int:
    raw_value = os.environ.get(LLM_PROXY_V2_EFFECTIVE_WORKER_CONCURRENCY_ENV)
    if raw_value is None:
        return max(1, cfg.load_balancer_worker_concurrency)
    try:
        return max(1, int(raw_value))
    except ValueError:
        return max(1, cfg.load_balancer_worker_concurrency)


def listen_backlog_for_worker_concurrency(cfg: TunnelConfig, worker_concurrency: int) -> int:
    desired_backlog = max(1, cfg.load_balancer_workers) * max(1, worker_concurrency)
    return max(MIN_LISTEN_BACKLOG, min(MAX_LISTEN_BACKLOG, desired_backlog))


def build_upstream_endpoints(cfg: TunnelConfig) -> list[tuple[str, int]]:
    return [(host, cfg.port_start + index) for index, host in enumerate(cfg.hosts)]


def build_upstream_targets(cfg: TunnelConfig) -> dict[int, tuple[str, int]]:
    remote_ports = cfg.remote_ports or [cfg.remote_port] * len(cfg.hosts)
    targets: dict[int, tuple[str, int]] = {}
    for index, (host, remote_port) in enumerate(zip(cfg.hosts, remote_ports)):
        endpoint_port = cfg.port_start + index
        if cfg.endpoint_setup == "direct":
            targets[endpoint_port] = (host, remote_port)
        else:
            targets[endpoint_port] = ("127.0.0.1", endpoint_port)
    return targets


def build_upstream_endpoint_labels(cfg: TunnelConfig) -> dict[tuple[str, int], str]:
    remote_ports = cfg.remote_ports or [cfg.remote_port] * len(cfg.hosts)
    labels: dict[tuple[str, int], str] = {}
    for index, (host, remote_port) in enumerate(zip(cfg.hosts, remote_ports)):
        local_port = cfg.port_start + index
        key = (host, local_port)
        if cfg.endpoint_setup == "direct":
            labels[key] = f"{host}:{remote_port}"
        elif local_port == remote_port:
            labels[key] = f"{host}:{local_port}"
        else:
            labels[key] = f"{host}:{remote_port} (local {local_port})"
    return labels


@dataclasses.dataclass(frozen=True)
class PreparedChatRequest:
    request_meta: dict[str, Any]
    message_jsons: tuple[str, ...]
    message_hashes: tuple[str, ...]
    exact_prefix_hashes: tuple[str, ...]
    loose_prefix_hashes: tuple[str, ...]
    session_id: str | None = None

    @property
    def message_count(self) -> int:
        return len(self.message_hashes)

    @property
    def state_hash(self) -> str:
        return self.exact_prefix_hashes[-1]

    @property
    def loose_state_hash(self) -> str:
        return self.loose_prefix_hashes[-1]


@dataclasses.dataclass(frozen=True)
class RouteLookupResult:
    conversation_id: str
    matched_state_hash: str | None
    matched_message_count: int
    preferred_upstream_port: int | None
    preferred_base_url: str | None
    route_reason: str


@dataclasses.dataclass(frozen=True)
class PersistenceJob:
    prepared_request: PreparedChatRequest
    conversation_id: str
    matched_state_hash: str | None
    matched_message_count: int
    endpoint_used: str
    base_url: str
    upstream_port: int
    route_reason: str
    status_code: int
    response_payload: Any
    created_ns: int


@dataclasses.dataclass(frozen=True)
class LoggedRequestRecord:
    request_id: int
    conversation_id: str
    input_state_hash: str
    input_payload: Any
    output_payload: Any
    endpoint_used: str
    route_reason: str
    status_code: int
    created_ns: int
    collected_at_ns: int | None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_exact_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_normalize_exact_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_exact_value(value[key])
            for key in sorted(value)
            if key not in EXACT_IDENTITY_IGNORED_KEYS
        }
    return repr(value)


def _normalize_loose_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\x1e".join(
            part for item in value if (part := _normalize_loose_value(item))
        )
    if isinstance(value, dict):
        return "\x1f".join(
            part
            for key in sorted(value)
            if key not in MESSAGE_AFFINITY_IGNORED_KEYS
            if (part := _normalize_loose_value(value[key]))
        )
    return repr(value)


def _extract_session_id(payload: dict[str, Any]) -> str | None:
    """Extract session_id from Anthropic metadata.user_id JSON string."""
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    user_id_raw = metadata.get("user_id")
    if not isinstance(user_id_raw, str):
        return None
    try:
        user_id = json.loads(user_id_raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(user_id, dict):
        return None
    session_id = user_id.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        return None
    return session_id


def prepare_chat_request(payload: Any) -> PreparedChatRequest | None:
    if not isinstance(payload, dict):
        return None
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    request_meta = {key: value for key, value in payload.items() if key != "messages"}
    message_jsons: list[str] = []
    message_hashes: list[str] = []
    exact_prefix_hashes: list[str] = []
    loose_prefix_hashes: list[str] = []
    exact_hasher = hashlib.sha256()
    loose_hasher = hashlib.sha256()

    for index, message in enumerate(messages):
        exact_json = _canonical_json(_normalize_exact_value(message))
        loose_text = _normalize_loose_value(message)

        if index:
            exact_hasher.update(b"\n")
            loose_hasher.update(b"\n")
        exact_hasher.update(exact_json.encode("utf-8"))
        loose_hasher.update(loose_text.encode("utf-8"))

        message_jsons.append(exact_json)
        message_hashes.append(hashlib.sha256(exact_json.encode("utf-8")).hexdigest())
        exact_prefix_hashes.append(exact_hasher.copy().hexdigest())
        loose_prefix_hashes.append(loose_hasher.copy().hexdigest())

    return PreparedChatRequest(
        request_meta=request_meta,
        message_jsons=tuple(message_jsons),
        message_hashes=tuple(message_hashes),
        exact_prefix_hashes=tuple(exact_prefix_hashes),
        loose_prefix_hashes=tuple(loose_prefix_hashes),
        session_id=_extract_session_id(payload),
    )


def _state_db_connection(
    path: pathlib.Path,
    *,
    read_only: bool = False,
) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if read_only:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=STATE_DB_BUSY_TIMEOUT_MILLISECONDS / 1000,
            isolation_level=None,
            check_same_thread=False,
        )
    else:
        connection = sqlite3.connect(
            path,
            timeout=STATE_DB_BUSY_TIMEOUT_MILLISECONDS / 1000,
            isolation_level=None,
            check_same_thread=False,
        )
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {STATE_DB_BUSY_TIMEOUT_MILLISECONDS}")
    if not read_only:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def _create_state_db_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            created_ns INTEGER NOT NULL,
            last_seen_ns INTEGER NOT NULL,
            last_upstream_port INTEGER,
            last_base_url TEXT,
            request_count INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS input_states (
            state_hash TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            matched_prefix_hash TEXT,
            message_count INTEGER NOT NULL,
            created_ns INTEGER NOT NULL,
            last_seen_ns INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS state_tail_messages (
            state_hash TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            message_hash TEXT NOT NULL,
            PRIMARY KEY(state_hash, ordinal)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            message_hash TEXT PRIMARY KEY,
            raw_json TEXT NOT NULL,
            created_ns INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            request_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            input_state_hash TEXT NOT NULL,
            request_meta_json TEXT NOT NULL,
            response_json TEXT NOT NULL,
            endpoint_used TEXT NOT NULL,
            base_url TEXT NOT NULL,
            upstream_port INTEGER NOT NULL,
            route_reason TEXT NOT NULL,
            status_code INTEGER NOT NULL,
            created_ns INTEGER NOT NULL,
            collected_at_ns INTEGER
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS route_affinity (
            loose_hash TEXT PRIMARY KEY,
            message_count INTEGER NOT NULL,
            conversation_id TEXT NOT NULL,
            base_url TEXT NOT NULL,
            upstream_port INTEGER NOT NULL,
            last_seen_ns INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS session_affinity (
            session_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            base_url TEXT NOT NULL,
            upstream_port INTEGER NOT NULL,
            last_seen_ns INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_input_states_conversation_id ON input_states(conversation_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_requests_collected ON requests(collected_at_ns, request_id)"
    )


def _initialize_state_db_schema_with_retry(path: pathlib.Path) -> None:
    deadline = time.monotonic() + STATE_DB_INIT_LOCK_RETRY_TIMEOUT_SECONDS
    while True:
        connection = _state_db_connection(path, read_only=False)
        try:
            _create_state_db_schema(connection)
            return
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc).lower():
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(STATE_DB_INIT_LOCK_RETRY_DELAY_SECONDS)
        finally:
            connection.close()


def _query_longest_prefix(
    connection: sqlite3.Connection,
    table: str,
    key_column: str,
    hashes: Iterable[str],
) -> sqlite3.Row | None:
    values = list(hashes)
    if not values:
        return None
    placeholders = ",".join("?" for _ in values)
    return connection.execute(
        f"""
        SELECT *
        FROM {table}
        WHERE {key_column} IN ({placeholders})
        ORDER BY message_count DESC
        LIMIT 1
        """,
        values,
    ).fetchone()


def _persist_job_with_connection(connection: sqlite3.Connection, job: PersistenceJob) -> None:
    prepared = job.prepared_request
    state_hash = prepared.state_hash
    now_ns = job.created_ns

    logger.debug(
        "[persist] conv={} state_hash={} msgs={} route={} port={}",
        job.conversation_id,
        state_hash,
        prepared.message_count,
        job.route_reason,
        job.upstream_port,
    )

    connection.execute("BEGIN")
    try:
        connection.execute(
            """
            INSERT OR IGNORE INTO conversations(
                conversation_id,
                created_ns,
                last_seen_ns,
                last_upstream_port,
                last_base_url,
                request_count
            )
            VALUES(?, ?, ?, ?, ?, 0)
            """,
            (
                job.conversation_id,
                now_ns,
                now_ns,
                job.upstream_port,
                job.base_url,
            ),
        )

        existing_state = connection.execute(
            "SELECT conversation_id FROM input_states WHERE state_hash = ?",
            (state_hash,),
        ).fetchone()
        actual_conversation_id = job.conversation_id
        if existing_state is None:
            suffix_start = job.matched_message_count
            suffix_message_jsons = prepared.message_jsons[suffix_start:]
            suffix_message_hashes = prepared.message_hashes[suffix_start:]
            logger.debug(
                "[persist] new state, storing {} new messages (suffix_start={})",
                len(suffix_message_hashes),
                suffix_start,
            )
            for message_hash, message_json in zip(suffix_message_hashes, suffix_message_jsons, strict=True):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO messages(message_hash, raw_json, created_ns)
                    VALUES(?, ?, ?)
                    """,
                    (message_hash, message_json, now_ns),
                )
            connection.execute(
                """
                INSERT INTO input_states(
                    state_hash,
                    conversation_id,
                    matched_prefix_hash,
                    message_count,
                    created_ns,
                    last_seen_ns
                )
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    state_hash,
                    job.conversation_id,
                    job.matched_state_hash,
                    prepared.message_count,
                    now_ns,
                    now_ns,
                ),
            )
            for ordinal, message_hash in enumerate(suffix_message_hashes):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO state_tail_messages(state_hash, ordinal, message_hash)
                    VALUES(?, ?, ?)
                    """,
                    (state_hash, ordinal, message_hash),
                )
        else:
            logger.debug(
                "[persist] existing state for state_hash={}, reusing conversation_id={}",
                state_hash,
                existing_state["conversation_id"],
            )
            actual_conversation_id = str(existing_state["conversation_id"])

        connection.execute(
            """
            UPDATE input_states
            SET last_seen_ns = ?
            WHERE state_hash = ?
            """,
            (now_ns, state_hash),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO conversations(
                conversation_id,
                created_ns,
                last_seen_ns,
                last_upstream_port,
                last_base_url,
                request_count
            )
            VALUES(?, ?, ?, ?, ?, 0)
            """,
            (
                actual_conversation_id,
                now_ns,
                now_ns,
                job.upstream_port,
                job.base_url,
            ),
        )
        connection.execute(
            """
            UPDATE conversations
            SET
                last_seen_ns = ?,
                last_upstream_port = ?,
                last_base_url = ?,
                request_count = request_count + 1
            WHERE conversation_id = ?
            """,
            (now_ns, job.upstream_port, job.base_url, actual_conversation_id),
        )
        connection.execute(
            """
            INSERT INTO requests(
                conversation_id,
                input_state_hash,
                request_meta_json,
                response_json,
                endpoint_used,
                base_url,
                upstream_port,
                route_reason,
                status_code,
                created_ns,
                collected_at_ns
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                actual_conversation_id,
                state_hash,
                _canonical_json(prepared.request_meta),
                _canonical_json(job.response_payload),
                job.endpoint_used,
                job.base_url,
                job.upstream_port,
                job.route_reason,
                job.status_code,
                now_ns,
            ),
        )
        connection.execute(
            """
            INSERT INTO route_affinity(
                loose_hash,
                message_count,
                conversation_id,
                base_url,
                upstream_port,
                last_seen_ns
            )
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(loose_hash) DO UPDATE SET
                message_count = excluded.message_count,
                conversation_id = excluded.conversation_id,
                base_url = excluded.base_url,
                upstream_port = excluded.upstream_port,
                last_seen_ns = excluded.last_seen_ns
            """,
            (
                prepared.loose_state_hash,
                prepared.message_count,
                actual_conversation_id,
                job.base_url,
                job.upstream_port,
                now_ns,
            ),
        )
        if prepared.session_id is not None:
            connection.execute(
                """
                INSERT INTO session_affinity(
                    session_id,
                    conversation_id,
                    base_url,
                    upstream_port,
                    last_seen_ns
                )
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    conversation_id = excluded.conversation_id,
                    base_url = excluded.base_url,
                    upstream_port = excluded.upstream_port,
                    last_seen_ns = excluded.last_seen_ns
                """,
                (
                    prepared.session_id,
                    actual_conversation_id,
                    job.base_url,
                    job.upstream_port,
                    now_ns,
                ),
            )
    except Exception:
        connection.execute("ROLLBACK")
        raise
    else:
        connection.execute("COMMIT")


class AsyncSqliteStateWriter:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self._queue: queue.Queue[PersistenceJob | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="sqlite-state-writer", daemon=True)
        self._thread.start()

    def submit(self, job: PersistenceJob) -> None:
        self._queue.put_nowait(job)

    def wait_for_idle(self) -> None:
        self._queue.join()

    def close(self) -> None:
        if self._thread is None:
            return
        self._queue.put(None)
        self._queue.join()
        self._thread.join()
        self._thread = None

    def _run(self) -> None:
        connection = _state_db_connection(self.path, read_only=False)
        try:
            while True:
                job = self._queue.get()
                batch: list[PersistenceJob] = []
                try:
                    if job is None:
                        return
                    batch.append(job)
                    while len(batch) < WRITE_BATCH_SIZE:
                        try:
                            next_job = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if next_job is None:
                            self._queue.put(None)
                            break
                        batch.append(next_job)
                    for item in batch:
                        _persist_job_with_connection(connection, item)
                finally:
                    self._queue.task_done()
                    for _ in batch[1:]:
                        self._queue.task_done()
        finally:
            connection.close()


class SqliteConversationStateStore:
    def __init__(self, path: pathlib.Path):
        self.path = path
        _initialize_state_db_schema_with_retry(self.path)
        self._read_connection = _state_db_connection(self.path, read_only=False)
        self._writer = AsyncSqliteStateWriter(self.path)

    def start(self) -> None:
        self._writer.start()

    def submit(self, job: PersistenceJob) -> None:
        self._writer.submit(job)

    def wait_for_idle(self) -> None:
        self._writer.wait_for_idle()

    def close(self) -> None:
        self._writer.close()
        self._read_connection.close()

    def lookup(
        self,
        prepared_request: PreparedChatRequest,
        valid_ports: set[int],
    ) -> RouteLookupResult:
        logger.debug(
            "[lookup] messages={}, exact_prefix_hashes={}, loose_prefix_hashes={}",
            prepared_request.message_count,
            prepared_request.exact_prefix_hashes,
            prepared_request.loose_prefix_hashes,
        )
        exact_row = _query_longest_prefix(
            self._read_connection,
            "input_states",
            "state_hash",
            prepared_request.exact_prefix_hashes,
        )
        matched_state_hash: str | None = None
        matched_message_count = 0
        conversation_id = uuid.uuid4().hex
        if exact_row is not None:
            matched_state_hash = str(exact_row["state_hash"])
            matched_message_count = int(exact_row["message_count"])
            conversation_id = str(exact_row["conversation_id"])
            logger.debug(
                "[lookup] exact match: state_hash={}, message_count={}, conversation_id={}",
                matched_state_hash,
                matched_message_count,
                conversation_id,
            )
            conversation_row = self._read_connection.execute(
                """
                SELECT last_upstream_port, last_base_url
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if conversation_row is not None:
                upstream_port = conversation_row["last_upstream_port"]
                base_url = conversation_row["last_base_url"]
                logger.debug(
                    "[lookup] conversation found: upstream_port={}, base_url={}, valid_ports={}",
                    upstream_port,
                    base_url,
                    valid_ports,
                )
                if isinstance(upstream_port, int) and upstream_port in valid_ports:
                    logger.info(
                        "[lookup] route=conversation conv={} port={} msgs={}",
                        conversation_id,
                        upstream_port,
                        matched_message_count,
                    )
                    return RouteLookupResult(
                        conversation_id=conversation_id,
                        matched_state_hash=matched_state_hash,
                        matched_message_count=matched_message_count,
                        preferred_upstream_port=upstream_port,
                        preferred_base_url=str(base_url) if isinstance(base_url, str) else None,
                        route_reason="conversation",
                    )
                else:
                    logger.debug(
                        "[lookup] upstream_port {} not in valid_ports {}",
                        upstream_port,
                        valid_ports,
                    )

        if prepared_request.session_id is not None:
            session_row = self._read_connection.execute(
                "SELECT conversation_id, upstream_port, base_url FROM session_affinity WHERE session_id = ?",
                (prepared_request.session_id,),
            ).fetchone()
            if session_row is not None:
                s_port = session_row["upstream_port"]
                s_base = session_row["base_url"]
                s_conv = str(session_row["conversation_id"])
                if isinstance(s_port, int) and s_port in valid_ports:
                    logger.info(
                        "[lookup] route=session conv={} port={} session={}",
                        s_conv,
                        s_port,
                        prepared_request.session_id,
                    )
                    return RouteLookupResult(
                        conversation_id=s_conv,
                        matched_state_hash=matched_state_hash,
                        matched_message_count=matched_message_count,
                        preferred_upstream_port=s_port,
                        preferred_base_url=str(s_base) if isinstance(s_base, str) else None,
                        route_reason="session",
                    )

        loose_row = _query_longest_prefix(
            self._read_connection,
            "route_affinity",
            "loose_hash",
            prepared_request.loose_prefix_hashes,
        )
        if loose_row is not None:
            upstream_port = loose_row["upstream_port"]
            base_url = loose_row["base_url"]
            loose_conv_id = str(loose_row["conversation_id"])
            loose_msg_count = int(loose_row["message_count"])
            logger.debug(
                "[lookup] loose match: loose_hash={}, conversation_id={}, message_count={}, upstream_port={}",
                loose_row["loose_hash"],
                loose_conv_id,
                loose_msg_count,
                upstream_port,
            )
            if isinstance(upstream_port, int) and upstream_port in valid_ports:
                logger.info(
                    "[lookup] route=affinity conv={} port={} msgs={}",
                    loose_conv_id,
                    upstream_port,
                    loose_msg_count,
                )
                return RouteLookupResult(
                    conversation_id=conversation_id,
                    matched_state_hash=matched_state_hash,
                    matched_message_count=matched_message_count,
                    preferred_upstream_port=upstream_port,
                    preferred_base_url=str(base_url) if isinstance(base_url, str) else None,
                    route_reason="affinity",
                )
            else:
                logger.debug(
                    "[lookup] loose upstream_port {} not in valid_ports {}",
                    upstream_port,
                    valid_ports,
                )

        logger.info(
            "[lookup] route=random new_conv={} msgs={}",
            conversation_id,
            prepared_request.message_count,
        )
        return RouteLookupResult(
            conversation_id=conversation_id,
            matched_state_hash=matched_state_hash,
            matched_message_count=matched_message_count,
            preferred_upstream_port=None,
            preferred_base_url=None,
            route_reason="random",
        )

    def persist_job(self, job: PersistenceJob) -> None:
        connection = _state_db_connection(self.path, read_only=False)
        try:
            _persist_job_with_connection(connection, job)
        finally:
            connection.close()


class StateDbReader:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.connection = _state_db_connection(self.path, read_only=True)
        self._messages_cache: dict[str, tuple[Any, ...]] = {}

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> StateDbReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def iter_requests(self, *, only_uncollected: bool = False) -> Iterator[LoggedRequestRecord]:
        query = """
            SELECT
                request_id,
                conversation_id,
                input_state_hash,
                request_meta_json,
                response_json,
                endpoint_used,
                route_reason,
                status_code,
                created_ns,
                collected_at_ns
            FROM requests
        """
        params: tuple[Any, ...] = ()
        if only_uncollected:
            query += " WHERE collected_at_ns IS NULL"
        query += " ORDER BY request_id ASC"

        for row in self.connection.execute(query, params):
            request_meta = json.loads(str(row["request_meta_json"]))
            response_payload = json.loads(str(row["response_json"]))
            input_payload = self.reconstruct_input_payload(
                request_meta,
                str(row["input_state_hash"]),
            )
            yield LoggedRequestRecord(
                request_id=int(row["request_id"]),
                conversation_id=str(row["conversation_id"]),
                input_state_hash=str(row["input_state_hash"]),
                input_payload=input_payload,
                output_payload=response_payload,
                endpoint_used=str(row["endpoint_used"]),
                route_reason=str(row["route_reason"]),
                status_code=int(row["status_code"]),
                created_ns=int(row["created_ns"]),
                collected_at_ns=(
                    int(row["collected_at_ns"])
                    if row["collected_at_ns"] is not None
                    else None
                ),
            )

    def reconstruct_input_payload(self, request_meta: Any, state_hash: str) -> Any:
        if not isinstance(request_meta, dict):
            return request_meta

        payload = dict(request_meta)
        payload["messages"] = list(self.reconstruct_messages(state_hash))
        return payload

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


def mark_requests_collected(
    path: pathlib.Path,
    request_ids: Iterable[int],
    *,
    collected_at_ns: int | None = None,
) -> None:
    ids = [int(request_id) for request_id in request_ids]
    if not ids:
        return
    timestamp = time.time_ns() if collected_at_ns is None else collected_at_ns
    connection = _state_db_connection(path, read_only=False)
    try:
        connection.executemany(
            """
            UPDATE requests
            SET collected_at_ns = ?
            WHERE request_id = ?
            """,
            [(timestamp, request_id) for request_id in ids],
        )
    finally:
        connection.close()


class LoadBalancerAppV2:
    def __init__(self, cfg, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        _configure_logger(verbose)
        self.effective_worker_concurrency = resolve_effective_worker_concurrency(cfg)
        self.upstream_endpoints = build_upstream_endpoints(cfg)
        self._upstream_targets = build_upstream_targets(cfg)
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
        self.valid_endpoints = self._wait_for_initial_endpoints()
        self.client: httpx.AsyncClient | None = None
        self.state_store = SqliteConversationStateStore(cfg.load_balancer_state_db_path)
        self._healthcheck_task = None
        self._healthcheck_stop = asyncio.Event()

    def _endpoint_label(self, endpoint: EndpointCheckResult) -> str:
        return self.upstream_endpoint_labels.get(
            (endpoint.host, endpoint.port),
            f"{endpoint.host}:{endpoint.port}",
        )

    def _endpoint_probe_urls(self, endpoint: tuple[str, int]) -> list[str]:
        _, port = endpoint
        target_host, target_port = self._upstream_targets.get(port, ("127.0.0.1", port))
        base = f"http://{target_host}:{target_port}"
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

    def _probe_models_sync(
        self,
        client: httpx.Client,
        endpoint: tuple[str, int],
    ) -> tuple[tuple[str, ...], str | None]:
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
        self,
        client: httpx.AsyncClient,
        endpoint: tuple[str, int],
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

    def _check_endpoint_sync(
        self,
        client: httpx.Client,
        endpoint: tuple[str, int],
    ) -> EndpointCheckResult:
        host, port = endpoint
        retries = max(0, HEALTHCHECK_CONNECT_RETRIES)
        for attempt in range(retries + 1):
            try:
                models, server_error = self._probe_models_sync(client, endpoint)
                if server_error is not None:
                    return EndpointCheckResult(host=host, port=port, error=server_error)
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
                    return EndpointCheckResult(host=host, port=port, error=server_error)
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
        logger.info("{}", tabulate(rows, headers=headers, tablefmt="simple"))

    def _initial_healthcheck(self) -> list[EndpointCheckResult]:
        with httpx.Client(timeout=HEALTHCHECK_TIMEOUT_SECONDS) as client:
            results = [self._check_endpoint_sync(client, endpoint) for endpoint in self.upstream_endpoints]
        self._summarize_health(results)
        return [result for result in results if result.error is None]

    def _wait_for_initial_endpoints(self) -> list[EndpointCheckResult]:
        deadline = time.monotonic() + max(0.0, INITIAL_HEALTHCHECK_TIMEOUT_SECONDS)
        while True:
            valid_results = self._initial_healthcheck()
            if valid_results:
                return valid_results
            if time.monotonic() >= deadline:
                break
            time.sleep(max(0.0, INITIAL_HEALTHCHECK_RETRY_DELAY_SECONDS))
        raise RuntimeError("No valid endpoints after healthcheck")

    async def refresh_health(self) -> None:
        async with httpx.AsyncClient(timeout=HEALTHCHECK_TIMEOUT_SECONDS) as client:
            results = await asyncio.gather(
                *(self._check_endpoint_async(client, endpoint) for endpoint in self.upstream_endpoints)
            )
        valid_results = [result for result in results if result.error is None]
        if valid_results or not self.valid_endpoints:
            self.valid_endpoints = valid_results
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
                self.state_store.start()
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
                self.state_store.close()
                await send({"type": "lifespan.shutdown.complete"})
                return

    def _build_client(self) -> httpx.AsyncClient:
        limits = httpx.Limits(
            max_connections=self.effective_worker_concurrency,
            max_keepalive_connections=self.effective_worker_concurrency,
        )
        timeout = httpx.Timeout(
            connect=UPSTREAM_TIMEOUT_SECONDS,
            read=UPSTREAM_TIMEOUT_SECONDS,
            write=UPSTREAM_TIMEOUT_SECONDS,
            pool=UPSTREAM_TIMEOUT_SECONDS,
        )
        return httpx.AsyncClient(timeout=timeout, limits=limits)

    def _prepare_request_state(self, request_body: bytes) -> PreparedChatRequest | None:
        try:
            request_text = self._decode_body_text(request_body)
        except UnicodeDecodeError:
            return None
        request_json = self._decode_json_value(request_text)
        return prepare_chat_request(request_json)

    def _select_upstream_for_prepared_request(
        self,
        prepared_request: PreparedChatRequest,
        valid_endpoints: list[EndpointCheckResult],
    ) -> tuple[int, str, RouteLookupResult]:
        valid_ports = {endpoint.port for endpoint in valid_endpoints}
        lookup = self.state_store.lookup(prepared_request, valid_ports)
        if (
            lookup.preferred_upstream_port is not None
            and lookup.preferred_upstream_port in valid_ports
        ):
            return lookup.preferred_upstream_port, lookup.route_reason, lookup
        return random.choice(list(valid_ports)), "random", lookup

    def _base_url_for_upstream_port(self, upstream_port: int) -> str:
        upstream_host, upstream_target_port = self._upstream_targets.get(
            upstream_port,
            ("127.0.0.1", upstream_port),
        )
        return f"http://{upstream_host}:{upstream_target_port}"

    def _build_persistence_job(
        self,
        prepared_request: PreparedChatRequest,
        lookup: RouteLookupResult,
        upstream_port: int,
        endpoint_used: str,
        response_payload: Any,
        route_reason: str,
        status_code: int,
        *,
        created_ns: int | None = None,
    ) -> PersistenceJob:
        return PersistenceJob(
            prepared_request=prepared_request,
            conversation_id=lookup.conversation_id,
            matched_state_hash=lookup.matched_state_hash,
            matched_message_count=lookup.matched_message_count,
            endpoint_used=endpoint_used,
            base_url=self._base_url_for_upstream_port(upstream_port),
            upstream_port=upstream_port,
            route_reason=route_reason,
            status_code=status_code,
            response_payload=response_payload,
            created_ns=time.time_ns() if created_ns is None else created_ns,
        )

    def _decoded_response_payload(self, response_body: bytes, content_type: str | None) -> Any:
        try:
            response_text = self._decode_body_text(response_body)
        except UnicodeDecodeError:
            return ""
        response_text = self._normalize_logged_response(response_text, content_type)
        try:
            return json.loads(response_text) if response_text else {}
        except json.JSONDecodeError:
            return response_text


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

    def _dump_bug_report(
        self,
        *,
        scope: dict[str, Any],
        request_body: bytes,
        response_body: bytes,
        upstream_url: str | None,
        response_status_code: int | None,
        route_reason: str | None,
        exc: BaseException,
    ) -> None:
        BUG_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "created_ns": time.time_ns(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "request": {
                "method": scope.get("method"),
                "path": scope.get("path"),
                "query_string": scope.get("query_string", b"").decode("latin1"),
                "body_text": request_body.decode("utf-8", errors="replace"),
            },
            "response": {
                "status_code": response_status_code,
                "body_text": response_body.decode("utf-8", errors="replace"),
            },
            "upstream_url": upstream_url,
            "route_reason": route_reason,
        }
        BUG_REPORT_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    async def _handle_http(self, scope, receive, send) -> None:
        request_body = b""
        response_chunks = bytearray()
        upstream_url: str | None = None
        response_status_code: int | None = None
        route_reason: str | None = None

        try:
            valid_endpoints = list(self.valid_endpoints)
            if not valid_endpoints:
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
            prepared_request = self._prepare_request_state(request_body)
            if prepared_request is not None:
                upstream_port, route_reason, lookup = self._select_upstream_for_prepared_request(
                    prepared_request,
                    valid_endpoints,
                )
            else:
                upstream_port = random.choice([endpoint.port for endpoint in valid_endpoints])
                route_reason = "random"
                lookup = None

            upstream_host, upstream_target_port = self._upstream_targets.get(
                upstream_port,
                ("127.0.0.1", upstream_port),
            )
            upstream_url = f"http://{upstream_host}:{upstream_target_port}{path}"
            headers = self._build_upstream_headers(scope["headers"], upstream_port)
            request = client.build_request(method, upstream_url, headers=headers, content=request_body)

            try:
                response = await client.send(request, stream=True)
            except httpx.HTTPError as exc:
                await self._send_plain_error(send, 502, f"Upstream request failed: {exc}".encode("utf-8"))
                return

            endpoint_label = self._endpoint_label_by_port.get(
                upstream_port,
                f"{upstream_host}:{upstream_target_port}",
            )
            self.endpoint_request_counts[endpoint_label] = (
                self.endpoint_request_counts.get(endpoint_label, 0) + 1
            )

            response_headers = [
                (key.encode("latin1"), value.encode("latin1"))
                for key, value in response.headers.items()
                if key.lower().encode("latin1") not in HOP_BY_HOP_HEADERS
            ]
            if lookup is not None:
                response_headers.append(
                    (b"x-llm-proxy-conversation-id", lookup.conversation_id.encode("ascii"))
                )
                response_headers.append(
                    (b"x-llm-proxy-route-reason", route_reason.encode("ascii"))
                )
            response_headers.append((b"connection", b"close"))
            response_status_code = response.status_code

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

            if prepared_request is not None and lookup is not None:
                response_payload = self._decoded_response_payload(
                    bytes(response_chunks),
                    response.headers.get("content-type"),
                )
                job = self._build_persistence_job(
                    prepared_request,
                    lookup,
                    upstream_port,
                    endpoint_label,
                    response_payload,
                    route_reason,
                    response.status_code,
                )
                logger.debug(
                    "[http] submitting persist job conv={} state_hash={}",
                    job.conversation_id,
                    job.prepared_request.state_hash,
                )
                self.state_store.submit(job)
        except Exception as exc:
            self._dump_bug_report(
                scope=scope,
                request_body=request_body,
                response_body=bytes(response_chunks),
                upstream_url=upstream_url,
                response_status_code=response_status_code,
                route_reason=route_reason,
                exc=exc,
            )
            raise

    def _build_upstream_headers(self, headers, upstream_port: int) -> list[tuple[str, str]]:
        forwarded_headers: list[tuple[str, str]] = []
        for raw_key, raw_value in headers:
            key = raw_key.lower()
            if key in HOP_BY_HOP_HEADERS or key == b"host":
                continue
            forwarded_headers.append((raw_key.decode("latin1"), raw_value.decode("latin1")))
        target_host, target_port = self._upstream_targets.get(upstream_port, ("127.0.0.1", upstream_port))
        forwarded_headers.append(("host", f"{target_host}:{target_port}"))
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

    def _decode_json_value(self, body_text: str) -> Any:
        try:
            return json.loads(body_text)
        except json.JSONDecodeError:
            return None


def resolve_config_path(config_path: pathlib.Path | None = None) -> pathlib.Path:
    if config_path is not None:
        return config_path
    return pathlib.Path(
        os.environ.get("LLM_PROXY_CONFIG", "~/.config/llm-proxy.yaml")
    ).expanduser()


def create_app(config_path: pathlib.Path | None = None, verbose: bool | None = None) -> LoadBalancerAppV2:
    cfg = parse_config(resolve_config_path(config_path))
    if verbose is None:
        verbose = os.environ.get("LLM_PROXY_VERBOSE", "0") == "1"
    return LoadBalancerAppV2(cfg, verbose=verbose)


def serve_forever(
    config_path: pathlib.Path = pathlib.Path("~/.config/llm-proxy.yaml").expanduser(),
    verbose: bool = True,
) -> None:
    cfg = parse_config(config_path)
    fd_plan = plan_file_descriptor_limits(cfg)
    os.environ["LLM_PROXY_CONFIG"] = str(config_path)
    os.environ["LLM_PROXY_VERBOSE"] = "1" if verbose else "0"
    os.environ[LLM_PROXY_V2_EFFECTIVE_WORKER_CONCURRENCY_ENV] = str(
        fd_plan.effective_worker_concurrency
    )
    if fd_plan.raised_soft_limit:
        logger.info(
            "Raised RLIMIT_NOFILE soft limit to {} for worker-concurrency={}",
            fd_plan.required_soft_limit,
            fd_plan.configured_worker_concurrency,
        )
    if fd_plan.warning:
        logger.warning(
            "{} (configured worker-concurrency={}, soft-limit={}, hard-limit={}, required={})",
            fd_plan.warning,
            fd_plan.configured_worker_concurrency,
            fd_plan.soft_limit if fd_plan.soft_limit is not None else "unknown",
            fd_plan.hard_limit if fd_plan.hard_limit is not None else "unknown",
            fd_plan.required_soft_limit,
        )
    uvicorn.run(
        "load_balancer_v2:create_app",
        factory=True,
        host="127.0.0.1",
        port=cfg.listen_port,
        workers=cfg.load_balancer_workers,
        limit_concurrency=fd_plan.effective_worker_concurrency,
        backlog=listen_backlog_for_worker_concurrency(
            cfg,
            fd_plan.effective_worker_concurrency,
        ),
        timeout_keep_alive=30,
        access_log=False,
        log_level="info",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SQLite-backed load balancer")
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
