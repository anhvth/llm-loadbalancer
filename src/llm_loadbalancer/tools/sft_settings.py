"""Shared SFT export and dedup settings."""

from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def preserve_thinking_in_content() -> bool:
    """Whether exported assistant turns should keep think blocks in content."""
    return _env_bool("LLM_SFT_PRESERVE_THINKING_IN_CONTENT", True)


def drop_prefix_snapshots_in_unique() -> bool:
    """Whether no-session rows that are strict message-prefixes are dropped."""
    return _env_bool("LLM_SFT_DROP_PREFIX_SNAPSHOTS", True)

