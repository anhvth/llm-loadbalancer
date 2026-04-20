# Setup

This is a Python 3.13+ project using `uv` with a local editable dependency on `speedy-utils`.

## Prerequisites

- Python 3.13+
- `uv` package manager
- `speedy-utils` sibling directory (the `pyproject.toml` references `../speedy_utils`)

## Installation

```bash
# 1. Ensure speedy-utils exists at the sibling path
ls ../speedy_utils

# 2. Sync dependencies
uv sync

# 3. Verify install
uv run llm-proxy --help
```

## Initial Config Setup

```bash
uvx --from git+https://github.com/anhvth/llm-loadbalancer llm-proxy --set-config
```

This creates `~/.config/llm-proxy.yaml`. Edit it with your worker endpoints, then run:

```bash
uv run llm-proxy
```

For available commands, see the [README](../README.md).