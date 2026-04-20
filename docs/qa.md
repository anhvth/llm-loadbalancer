# QA Tasks

## Core Load Balancer
- [ ] `load_balancer.py` — test routing strategies (random, smart), health check path, affinity db
- [ ] `keep_connection.py` — test SSH tunnel config parsing, host pattern expansion (`worker-[41,45,49-54]`), tmux launch, `setup: direct` skip

## CLI
- [ ] `cli.py` — test `--set-config`, config file creation, argument parsing

## Data Collection
- [ ] `collect_jsonl.py` — test session grouping, tool call normalization, dedup logic

## SFT Conversion
- [ ] `convert_to_sft_data_anthropic.py` — test reasoning_content preservation, leading newline stripping, prefix snapshot handling
- [ ] `convert_to_sft_data_openai.py` — test reasoning_content handling (no injected `<think>` into final assistant), session dedup
- [ ] `build_unique_conversation.py` — test JSON import, dedup order (raw → prompt string → drop prefix snapshots → split back), `partial_json` tool call parsing
- [ ] `cat_sft_messages.py` — test output formatting

## Integration
- [ ] End-to-end test: collect → dedup → convert → verify row counts and rendered samples
- [ ] `example_config.yaml` — validate it parses correctly with `keep_connection.py --print-only`