#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/llm-proxy/training-data}"

echo "=== Step 1: collect ==="
uv run "$SCRIPT_DIR/src/llm_loadbalancer/tools/collect_jsonl.py"

echo ""
echo "=== Step 2: build unique ==="
uv run "$SCRIPT_DIR/src/llm_loadbalancer/tools/build_unique_conversation.py"

echo ""
echo "=== Step 3: convert to sft data ==="
PYTHONPATH="$SCRIPT_DIR/src" uv run python "$SCRIPT_DIR/src/llm_loadbalancer/tools/convert_to_sft_data.py"

echo ""
echo "Done."
