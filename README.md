# llm-loadbalancer

`llm-loadbalancer` keeps a pool of SSH port-forward tunnels alive and proxies
OpenAI-like HTTP traffic to a random local upstream.

## Quick Start

```bash
uvx --from llm-loadbalancer llmproxy
```

Or with `uv run` inside a clone of this repo:

```bash
uv run llmproxy
```

## Config

Example `config.yaml`:

```yaml
endpoints:
  - hosts: worker-[41,45,49,53-54,57,59,61]
  - port-start: 18000

tmux:
  session-name: keepssh

port:
  - 8001

load-balancer:
  workers: 20
  worker-concurrency: 512
  max-connections: 20000
  max-keepalive-connections: 4096
  upstream-timeout: 300
  db-path: llm_loadbalancer.sqlite3
```

`db-path` is optional. If omitted, the load balancer writes to
`llm_loadbalancer.sqlite3` in the same directory as the config file.

### Commands

- `uv run llmproxy` — start the load balancer (creates a default `config.yaml` if one does not exist)
- `uv run llmproxy --set-config` — open the config file in your editor
- `uv run llmproxy --verbose` — print each logged request/response row to stderr
- `uv run cat_db` — inspect the SQLite log in an interactive terminal
- `uv run cat_db --raw` — print one JSON object per line (for scripting)

## Request Logging

Completed JSON request/response pairs are stored in a local SQLite table named
`request_response_log` with these columns:

- `id` integer primary key, auto-incrementing
- `input` JSON request body as text
- `output` JSON response body as text
- `endpoint_used` full upstream URL selected for the request

Only complete JSON exchanges are logged. Streaming responses such as SSE and
non-JSON traffic are proxied normally but skipped in the database.

## Inspect Logged Rows

Use `cat_db` to inspect the SQLite rows with a readable pretty view by default:

```bash
uv run cat_db
uv run cat_db /path/to/custom.sqlite3
uv run cat_db --raw
```

Without an argument, `cat_db` reads `./llm_loadbalancer.sqlite3`.

- Default mode formats one row at a time and, in an interactive terminal, lets
  you move with `Enter`, arrow keys, `j`/`k`, `gg`/`G`, and quit with `q`.
- Use `--raw` to print one JSON object per line for scripts or piping.