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
  log-dir: ~/.cache/llmup/logs
  affinity-db: ~/.cache/llmup/affinity.sqlite3
```

`log-dir` and `affinity-db` are optional. If omitted, the load balancer writes
request log files to `~/.cache/llmup/logs` and stores shared message-affinity
state in `~/.cache/llmup/affinity.sqlite3`.

### Commands

- `uv run llmproxy` — start the load balancer (creates a default `config.yaml` if one does not exist)
- `uv run llmproxy --set-config` — open the config file in your editor
- `uv run llmproxy --silent` — disable printing each logged request/response file to stderr
- `uv run cat_db` — inspect the request log directory in an interactive terminal
- `uv run cat_db --raw` — print one JSON object per line (for scripting)

## Request Logging

Completed request/response pairs are written as individual JSON files under
`<log-dir>/requests`. Each file contains:

- `input` JSON request body as text
- `output` JSON response body as text
- `endpoint_used` full upstream URL selected for the request

The request path only enqueues the log record. A background thread writes the
files, so proxying is not blocked on disk I/O. Message affinity is shared across
workers through the SQLite database at `affinity-db`.

## Inspect Logged Requests

Use `cat_db` to inspect the request log files with a readable pretty view by default:

```bash
uv run cat_db
uv run cat_db /path/to/custom-log-dir
uv run cat_db --raw
```

Without an argument, `cat_db` reads `~/.cache/llmup/logs`.

- Default mode formats one request at a time and, in an interactive terminal, lets
  you move with `Enter`, arrow keys, `j`/`k`, `gg`/`G`, and quit with `q`.
- Use `--raw` to print one JSON object per line for scripts or piping.
