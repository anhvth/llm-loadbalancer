#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/llm-proxy/training-data}"
REQUESTS_DIR="$HOME/.cache/llm-proxy/logs/requests"

MINIMAX_TOKENIZER_PATH="${MINIMAX_TOKENIZER_PATH:-$HOME/ckpt/hf_models/MiniMaxAI/MiniMax-M2.7/}"
QWEN3_TOKENIZER_PATH="${QWEN3_TOKENIZER_PATH:-$HOME/ckpt/hf_models/Qwen/Qwen3.6-27B/}"

usage() {
    echo "Usage: $0 [--minimax|--qwen3]"
}

TOKENIZER_PATH="${TOKENIZER_PATH:-$QWEN3_TOKENIZER_PATH}"
SFT_TYPE="qwen3"
if [[ "$TOKENIZER_PATH" == "$MINIMAX_TOKENIZER_PATH" ]]; then
    SFT_TYPE="minimax"
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --minimax|minimax)
            TOKENIZER_PATH="$MINIMAX_TOKENIZER_PATH"
            SFT_TYPE="minimax"
            ;;
        --qwen3|qwen3)
            TOKENIZER_PATH="$QWEN3_TOKENIZER_PATH"
            SFT_TYPE="qwen3"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done
export TOKENIZER_PATH

TODAY_DIR=$(mktemp -d)
EXPORT_DIR=$(mktemp -d)
PROCESSED_DIR=$(mktemp -d)

echo "=== Filtering today's (2026-05-07) request files ==="
find "$REQUESTS_DIR" -newermt "2026-05-07 00:00" ! -newermt "2026-05-08 00:00" -name "*.json" -exec ln -s {} "$TODAY_DIR/" \;
FILE_COUNT=$(ls "$TODAY_DIR" | wc -l)
echo "Found $FILE_COUNT files from today"

if [[ $FILE_COUNT -eq 0 ]]; then
    echo "No files from today. Nothing to do."
    rm -rf "$TODAY_DIR" "$EXPORT_DIR" "$PROCESSED_DIR"
    exit 0
fi

echo ""
echo "=== Step 1: collect ==="
uv run "$SCRIPT_DIR/src/llm_loadbalancer/tools/collect_jsonl.py" \
    --requests-dir "$TODAY_DIR" \
    --export-dir "$EXPORT_DIR" \
    --processed-dir "$PROCESSED_DIR"

echo ""
echo "=== Step 2: build unique SFT (convert + dedup) ==="
echo "Tokenizer: $TOKENIZER_PATH"
OUTPUT_PATH="$CACHE_DIR/collected.unique_sft.$SFT_TYPE.jsonl"
echo "Output: $OUTPUT_PATH"
PYTHONPATH="$SCRIPT_DIR/src" uv run python "$SCRIPT_DIR/src/llm_loadbalancer/tools/build_unique_conversation.py" \
    --tokenizer "$TOKENIZER_PATH" \
    --output "$OUTPUT_PATH" \
    "$EXPORT_DIR/collected.jsonl"

echo ""
echo "Cleaning up temp dirs..."
rm -rf "$TODAY_DIR" "$EXPORT_DIR" "$PROCESSED_DIR"
echo "Done."
