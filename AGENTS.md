# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 5. Training Data Conversion Notes

When working on SFT data conversion:
- Anthropic/Claude request logs need provider-specific conversion; keep that code named explicitly (for example `convert_to_sft_data_anthropic.py`) instead of implying a generic converter.
- Qwen chat templates may ignore `reasoning_content` on historical assistant turns before the last real user query. For Qwen SFT, inline historical assistant thinking into `content` as `<think>...</think>` while preserving the final target assistant turn's native `reasoning_content`.
- Anthropic assistant text often starts with leading newlines. When injecting `<think>...</think>\n\n`, strip only leading newlines from the assistant body so rendered prompts do not get extra blank lines.
- Deduplicating collected conversations must normalize streamed tool calls: whitespace-only text blocks should not split conversations, and `tool_use.partial_json` should be parsed to the same `input` shape used by completed history when possible.
- OpenAI-compatible logs from pi usually have no `session_id`; they are rolling snapshots in `input.messages`. Treating `input.messages` prefix alone as "same conversation" can incorrectly merge a brand-new chat that starts with the same first turn (for example another `"hi"`).
- For no-session dedupe, use this order: raw row -> convert to model prompt string -> drop strict prompt-prefix snapshots -> split back to `{role, content}` -> final rendered-prompt dedupe.
- Preserve thinking in content for both Anthropic and OpenAI export paths by default. Use project switches (for controlled overrides) instead of path-specific behavior drift.
- OpenAI gotcha: do not force `<think>...</think>` into the final target assistant `content` before rendering. Keep `reasoning_content` and inline historical thinking only; otherwise Qwen rendering can produce inconsistent/double-think prompts and break prefix matching.
- After changing conversion or dedupe logic, regenerate `collected.unique.jsonl` and `collected.unique_sft.jsonl`, then inspect row counts and at least one rendered chat-template sample.

