"""Standalone web monitor server for llm-proxy request logs.

Usage:
    # As a subprocess (from cli.py):
    from llm_loadbalancer.monitor.server import start_monitor_process
    proc = start_monitor_process()

    # Standalone:
    uv run python -m src.llm_loadbalancer.monitor.server
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import starlette.applications
import starlette.responses
import starlette.routing
import starlette.requests

LOG_DIR = Path.home() / ".cache" / "llm-proxy" / "logs" / "requests"
HEALTH_FILE = Path.home() / ".cache" / "llm-proxy" / "logs" / "health_state.json"
STATIC_DIR = Path(__file__).parent / "static"
PORT = 5555

# In-memory cache: file_name -> (mtime, parsed_dict)
_cache: dict[str, tuple[float, dict]] = {}
_cache_loaded = False


def _refresh_cache() -> None:
    """Scan the log directory and update the cache with new/changed files."""
    global _cache, _cache_loaded
    if not LOG_DIR.is_dir():
        _cache = {}
        _cache_loaded = True
        return

    if not _cache_loaded:
        # First scan: load all files
        for p in LOG_DIR.iterdir():
            if not p.is_file():
                continue
            try:
                with open(p) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            data["_file"] = p.name
            data["_mtime"] = p.stat().st_mtime
            _cache[p.name] = (p.stat().st_mtime, data)
        _cache_loaded = True
    else:
        # Incremental: only read new/modified files
        seen = set()
        for p in LOG_DIR.iterdir():
            if not p.is_file():
                continue
            seen.add(p.name)
            cur_mtime = p.stat().st_mtime
            cached_mtime, _ = _cache.get(p.name, (0, None))
            if cur_mtime > cached_mtime:
                try:
                    with open(p) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                data["_file"] = p.name
                data["_mtime"] = cur_mtime
                _cache[p.name] = (cur_mtime, data)
        # Optionally drop deleted files (not critical)
        for name in list(_cache.keys()):
            if name not in seen:
                del _cache[name]


def _load_request_files(limit: int = 200):
    """Return the *limit* most recent request dicts, newest first, from cache."""
    _refresh_cache()
    if not _cache:
        return []
    # Sorted by filename (which starts with nanosecond timestamp), newest first
    sorted_names = sorted(_cache.keys(), reverse=True)[:limit]
    return [_cache[n][1] for n in sorted_names if n in _cache]


def _nanos_from_name(name: str) -> int:
    """Extract nanosecond timestamp from filename."""
    try:
        return int(name.split("-", 1)[0])
    except (ValueError, IndexError):
        return 0


def _endpoint_slug(endpoint_used: str) -> str:
    """Extract a short worker identifier from the endpoint URL."""
    # "http://worker-0:7777/v1/chat/completions" -> "worker-0"
    parts = endpoint_used.replace("http://", "").split(":")
    return parts[0] if parts else endpoint_used


def _parse_health() -> dict | None:
    """Return parsed health_state.json or None."""
    try:
        with open(HEALTH_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


async def handle_root(request: starlette.requests.Request):
    static = STATIC_DIR / "index.html"
    if not static.exists():
        return starlette.responses.PlainTextResponse("index.html not found", status_code=500)
    body = static.read_bytes()
    return starlette.responses.HTMLResponse(body)


async def handle_health(request: starlette.requests.Request):
    data = _parse_health()
    if data is None:
        return starlette.responses.JSONResponse({"error": "health_state.json not found"}, status_code=503)
    return starlette.responses.JSONResponse(data)


async def handle_stats(request: starlette.requests.Request):
    try:
        window = int(request.query_params.get("window", "60"))
    except ValueError:
        window = 60

    now = time.time()
    cutoff = now - window

    records = _load_request_files(limit=5000)
    recent = [r for r in records if r.get("_mtime", 0) >= cutoff]

    total = len(recent)
    per_endpoint: Counter = Counter()
    per_route: Counter = Counter()
    prompt_tokens = 0
    completion_tokens = 0
    latencies: list[float] = []

    for r in recent:
        ep = _endpoint_slug(r.get("endpoint_used", ""))
        per_endpoint[ep] += 1
        per_route[r.get("route_reason", "unknown")] += 1

        usage = r.get("output", {}).get("usage", {}) or {}
        prompt_tokens += usage.get("prompt_tokens", 0) or 0
        completion_tokens += usage.get("completion_tokens", 0) or 0

        # Estimate latency: output.created is unix timestamp, compare to file mtime
        created = r.get("output", {}).get("created")
        if created:
            latencies.append(abs(r["_mtime"] - created))

    mean_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
    median_lat = sorted(latencies)[len(latencies) // 2] if latencies else 0.0

    # Normalise endpoint names: isolate worker host from URL
    eps_sorted = per_endpoint.most_common()

    return starlette.responses.JSONResponse({
        "window": window,
        "total": total,
        "per_endpoint": [{"name": n, "count": c} for n, c in eps_sorted],
        "per_route": [{"name": n, "count": c} for n, c in per_route.most_common()],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "mean_latency": round(mean_lat, 2),
        "median_latency": round(median_lat, 2),
    })


async def handle_requests(request: starlette.requests.Request):
    try:
        limit = int(request.query_params.get("limit", "50"))
    except ValueError:
        limit = 50
    limit = min(limit, 500)

    endpoint_filter = request.query_params.get("endpoint", "").strip()

    records = _load_request_files(limit=1000)
    if endpoint_filter:
        records = [r for r in records if endpoint_filter in r.get("endpoint_used", "")]

    records = records[:limit]

    items = []
    for r in records:
        usage = r.get("output", {}).get("usage", {}) or {}
        items.append({
            "file": r["_file"],
            "timestamp_ns": _nanos_from_name(r["_file"]),
            "endpoint_used": r.get("endpoint_used", ""),
            "route_reason": r.get("route_reason", ""),
            "model": r.get("output", {}).get("model", ""),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        })

    return starlette.responses.JSONResponse({"items": items, "total": len(items)})


# ---------------------------------------------------------------------------
# Prometheus /metrics scraping for vLLM live endpoint data
# ---------------------------------------------------------------------------

# Regex patterns for vLLM Prometheus gauges (text format)
# Example: vllm:num_requests_running{model="foo"} 13.0
_METRIC_RE = re.compile(r"^vllm:(\w+)\{.*\}\s+([\d.]+)")

# Cache for scraped endpoint data: key -> (timestamp, data)
_endpoint_cache: dict[str, tuple[float, dict]] = {}
_ENDPOINT_CACHE_TTL = 2.0  # seconds


def _parse_prometheus_metrics(text: str) -> dict[str, float]:
    """Parse relevant vLLM Prometheus gauge values from `/metrics` text."""
    results: dict[str, float] = {}
    for line in text.splitlines():
        m = _METRIC_RE.match(line)
        if m:
            name, val = m.group(1), float(m.group(2))
            if name in (
                "num_requests_running",
                "num_requests_waiting",
                "kv_cache_usage_perc",
                "num_preemptions_created",
                "prompt_tokens_created",
                "gpu_cache_usage_perc",
            ):
                results[name] = val
    return results


async def _scrape_endpoint(
    client: httpx.AsyncClient, address: str, timeout: float = 3.0
) -> dict:
    """Scrape a single vLLM endpoint's /metrics and return parsed data."""
    url = f"http://{address}/metrics"
    try:
        r = await client.get(url, timeout=timeout)
        r.raise_for_status()
        metrics = _parse_prometheus_metrics(r.text)
        # requests_served = prompt_tokens_created -> we track via health_state
        return {
            "status": "up",
            "requests_running": metrics.get("num_requests_running", 0),
            "requests_waiting": metrics.get("num_requests_waiting", 0),
            "kv_cache_pct": round(metrics.get("kv_cache_usage_perc", 0) * 100, 2),
            "gpu_cache_pct": round(metrics.get("gpu_cache_usage_perc", 0) * 100, 2),
            "preemptions": metrics.get("num_preemptions_created", 0),
            "prompt_tokens": metrics.get("prompt_tokens_created", 0),
        }
    except Exception as e:
        return {
            "status": "down",
            "error": str(e),
        }


async def _scrape_all_endpoints(
    endpoints: dict[str, dict],
) -> dict[str, dict]:
    """Scrape /metrics from all endpoints concurrently."""
    async with httpx.AsyncClient() as client:
        tasks = {
            name: _scrape_endpoint(client, name)
            for name, info in endpoints.items()
        }
        results = {}
        for name, coro in tasks.items():
            info = endpoints[name]
            ep_info = _endpoint_info(info)
            try:
                data = await coro
            except Exception:
                data = {"status": "down", "error": "scrape failed"}
            data["model"] = ep_info["model"]
            data["requests_served"] = ep_info["requests"]
            results[name] = data
        return results


def _endpoint_info(endpoint_data: dict) -> dict:
    """Extract model and request count from health_state endpoint entry."""
    models = endpoint_data.get("models", [])
    model = models[0].split("/")[-1] if models else "unknown"
    return {"model": model, "requests": endpoint_data.get("requests", 0)}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


async def handle_endpoints_live(request: starlette.requests.Request):
    """Return per-endpoint live metrics scraped from vLLM /metrics.

    Results are cached for ~2s to avoid hammering endpoints on every poll.
    """
    now = time.time()
    health = _parse_health()
    if health is None:
        return starlette.responses.JSONResponse(
            {"error": "health_state.json not found"}, status_code=503
        )

    endpoints = (health.get("snapshot") or {}).get("endpoints") or {}

    # Check cache freshness
    cache_key = str(sorted(endpoints.keys()))
    cached_time, cached_data = _endpoint_cache.get(cache_key, (0, None))
    if cached_data and (now - cached_time) < _ENDPOINT_CACHE_TTL:
        return starlette.responses.JSONResponse({"endpoints": cached_data})

    data = await _scrape_all_endpoints(endpoints)
    _endpoint_cache[cache_key] = (now, data)
    return starlette.responses.JSONResponse({"endpoints": data})


def start_monitor_process() -> subprocess.Popen:
    """Launch the monitor server as a background subprocess.

    Spawns the monitor in a separate uv run process so it runs independently
    of the proxy's uvicorn worker pool.  Returns the Popen handle so the
    caller can terminate it on shutdown.
    """
    module = "llm_loadbalancer.monitor.server"
    cwd = Path(__file__).resolve().parent.parent.parent.parent
    proc = subprocess.Popen(
        ["uv", "run", "python", "-m", module],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def create_app() -> starlette.applications.Starlette:
    routes = [
        starlette.routing.Route("/", handle_root),
        starlette.routing.Route("/api/health", handle_health),
        starlette.routing.Route("/api/stats", handle_stats),
        starlette.routing.Route("/api/requests", handle_requests),
        starlette.routing.Route("/api/endpoints/live", handle_endpoints_live),
    ]
    return starlette.applications.Starlette(routes=routes)


app = create_app()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info", reload=True)


if __name__ == "__main__":
    main()
