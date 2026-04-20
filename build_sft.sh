#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/llm-proxy/training-data}"
MINIMAX_TOKENIZER_PATH="${MINIMAX_TOKENIZER_PATH:-$HOME/ckpt/hf_models/MiniMaxAI/MiniMax-M2.7/}"
QWEN3_TOKENIZER_PATH="${QWEN3_TOKENIZER_PATH:-$HOME/ckpt/hf_models/Qwen/Qwen3.5-27B/}"

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
OUTPUT_PATH="$CACHE_DIR/collected.unique_sft.$SFT_TYPE.jsonl"

echo "=== Step 1: collect ==="
uv run "$SCRIPT_DIR/src/llm_loadbalancer/tools/collect_jsonl.py"

echo ""
echo "=== Step 2: build unique SFT (convert + dedup) ==="
echo "Tokenizer: $TOKENIZER_PATH"
echo "Output: $OUTPUT_PATH"
PYTHONPATH="$SCRIPT_DIR/src" uv run python "$SCRIPT_DIR/src/llm_loadbalancer/tools/build_unique_conversation.py" \
    --tokenizer "$TOKENIZER_PATH" \
    --output "$OUTPUT_PATH"

echo ""
echo "Done."
