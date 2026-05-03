# llm-loadbalancer

`llm-loadbalancer` can either keep a pool of SSH port-forward tunnels alive or
connect directly to reachable workers, then proxy OpenAI-like HTTP traffic to a
random upstream.

## Quick Start

```bash
uvx --from git+https://github.com/anhvth/llm-loadbalancer llm-proxy --set-config
```

Run that once on a new machine to create and open `~/.config/llm-proxy.yaml`.
After saving it, run the same command without `--set-config` to start the
service.

## Config

Example [`example_config.yaml`](/Users/anhvth/projects/llm-loadbalancer/example_config.yaml):

```yaml
endpoints:
  - worker-[41,45,49,53-54,57,59]:8000
  - setup: "ssh"

port:
  - 8001

load-balancer:
  workers: 4
  worker-concurrency: 10204
  routing: smart
  health-path: /models
  log-dir: ~/.cache/llm-proxy/logs
  affinity-db: ~/.cache/llm-proxy/affinity.sqlite3

port-start: 18001
```

`endpoints.setup` defaults to `ssh`. Use `setup: "direct"` when the load
balancer can already reach the worker hosts without SSH tunneling. `routing`
defaults to `smart`, which keeps related chat turns on the same upstream with
message affinity. Use `routing: random` or `llm-proxy --routing random` to
disable affinity and choose a random upstream for every request. `health-path`,
`log-dir`, and `affinity-db` are optional. If omitted, the load balancer probes
`/models`, writes request log files to `~/.cache/llm-proxy/logs` and stores
shared message-affinity state in `~/.cache/llm-proxy/affinity.sqlite3`. Pass
`--no-cache` to disable filesystem-backed request logs, health state, and the
shared affinity database. Local SSH tunnel ports default to the fixed range
starting at `18000`, so you only need `port-start` if you want a different
range.
Per-worker upstream connection limits are derived automatically from
`worker-concurrency` and the process file descriptor limit.

### Commands

- `uvx --from git+https://github.com/anhvth/llm-loadbalancer llm-proxy --set-config` — create and open `~/.config/llm-proxy.yaml`
- `uvx --from git+https://github.com/anhvth/llm-loadbalancer llm-proxy` — start the load balancer after config is set
- `uvx --from git+https://github.com/anhvth/llm-loadbalancer llm-proxy --routing random` — start with pure random routing
- `uvx --from git+https://github.com/anhvth/llm-loadbalancer llm-proxy --no-cache` — run without filesystem-backed cache artifacts

## Request Logging

Completed request/response pairs are written as individual JSON files under
`<log-dir>/requests`. Each file contains:

- `input` JSON request body as text
- `output` JSON response body as text
- `endpoint_used` full upstream URL selected for the request

The request path only enqueues the log record. A background thread writes the
files, so proxying is not blocked on disk I/O. Message affinity is shared across
workers through the SQLite database at `affinity-db`. Use `--no-cache` to keep
that state in memory only and skip writing log and health-state files.
