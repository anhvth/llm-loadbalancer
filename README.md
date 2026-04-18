# llm-loadbalancer

`llm-loadbalancer` keeps a pool of SSH port-forward tunnels alive and proxies
OpenAI-like HTTP traffic to a random local upstream.

## Quick Start

```bash
uvx --from git+https://github.com/anhvth/llm-loadbalancer llmproxy --set-config
```

Run that once on a new machine to create and open `~/.cache/llmup/config.yaml`.
After saving it, run the same command without `--set-config` to start the
service.

## Config

Example `config.yaml`:

```yaml
endpoints:
  - worker-[41,45,49,53-54,57,59]:8000

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
state in `~/.cache/llmup/affinity.sqlite3`. Local SSH tunnel ports are chosen
automatically from a free high port range, so you do not need `port-start`.

### Commands

- `uvx --from git+https://github.com/anhvth/llm-loadbalancer llmproxy --set-config` — create and open `~/.cache/llmup/config.yaml`
- `uvx --from git+https://github.com/anhvth/llm-loadbalancer llmproxy` — start the load balancer after config is set
- `uvx --from git+https://github.com/anhvth/llm-loadbalancer cat_db` — inspect the request log directory in an interactive terminal
- `uvx --from git+https://github.com/anhvth/llm-loadbalancer cat_db --raw` — print one JSON object per line for scripting

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
uvx --from git+https://github.com/anhvth/llm-loadbalancer cat_db
uvx --from git+https://github.com/anhvth/llm-loadbalancer cat_db /path/to/custom-log-dir
uvx --from git+https://github.com/anhvth/llm-loadbalancer cat_db --raw
```

Without an argument, `cat_db` reads `~/.cache/llmup/logs`.

- Default mode formats one request at a time and, in an interactive terminal, lets
  you move with `Enter`, arrow keys, `j`/`k`, `gg`/`G`, and quit with `q`.
- Use `--raw` to print one JSON object per line for scripts or piping.
