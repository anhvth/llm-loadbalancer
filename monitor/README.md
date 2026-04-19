## llm-loadbalancer monitor

Read-only live monitor for the SQLite state database and health snapshot used by `load_balancer_v2.py`.

It starts without CLI flags and resolves paths from:

- `LLM_PROXY_CONFIG` when set
- `~/.config/llm-proxy.yaml` when present
- `~/.cache/llm-proxy/state.sqlite3` and `~/.cache/llm-proxy/logs/health_state.json` as fallback defaults

### Run

```bash
cd monitor
uv sync
uv run monitor.py
```

Optional environment variables:

- `MONITOR_HOST` defaults to `127.0.0.1`
- `MONITOR_PORT` defaults to `4477`

The monitor only opens the SQLite database in read-only mode and never writes to serving state.
