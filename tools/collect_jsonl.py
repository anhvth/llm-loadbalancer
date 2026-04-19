#!/usr/bin/env python3
"""Collect request logs and export Anthropic-format rows to SFT JSONL."""

import argparse
import glob
import json
import logging
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

try:
    from tools.convert_anthropic_messages_to_sft import anthropic_format_to_train_sft
except ModuleNotFoundError:
    from convert_anthropic_messages_to_sft import anthropic_format_to_train_sft

logger = logging.getLogger("collect_jsonl")
_ENDPOINT_MARKER = "-ep_"


def _extract_endpoint_slug_from_filename(path: Path) -> str | None:
    stem = path.stem
    marker_index = stem.find(_ENDPOINT_MARKER)
    if marker_index < 0:
        return None
    start = marker_index + len(_ENDPOINT_MARKER)
    end = stem.find("-", start)
    if end < 0:
        return None
    slug = stem[start:end]
    return slug or None


def _is_anthropic_endpoint_slug(endpoint_slug: str | None) -> bool:
    if endpoint_slug is None:
        return False
    return endpoint_slug == "v1_messages" or endpoint_slug.endswith("_v1_messages")


def _convert_file(json_path: str) -> tuple[str, str, str | None]:
    path = Path(json_path)
    endpoint_slug = _extract_endpoint_slug_from_filename(path)
    if not _is_anthropic_endpoint_slug(endpoint_slug):
        return (
            "skip:unsupported_endpoint_format",
            json_path,
            f"endpoint_slug={endpoint_slug!r}",
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ("skip:unreadable_json", json_path, f"unreadable: {exc}")

    try:
        converted = anthropic_format_to_train_sft(payload)
    except Exception as exc:
        return ("skip:anthropic_convert_error", json_path, str(exc))

    if not isinstance(converted, dict):
        return ("skip:anthropic_convert_invalid", json_path, "converter returned non-dict")

    messages = converted.get("messages")
    if not isinstance(messages, list) or not messages:
        return ("skip:anthropic_convert_empty_messages", json_path, "messages missing or empty")

    tools = converted.get("tools", [])
    if tools is None:
        tools = []
    if not isinstance(tools, list):
        return ("skip:anthropic_convert_invalid_tools", json_path, "tools must be a list")

    row = {"messages": messages, "tools": tools}
    return ("ok", json_path, json.dumps(row, ensure_ascii=False))


def _collect(
    requests_dir: Path,
    export_dir: Path,
    processed_dir: Path | None,
    dry_run: bool = False,
) -> tuple[int, int]:
    requests_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    if processed_dir:
        processed_dir.mkdir(parents=True, exist_ok=True)

    today = time.strftime("%Y-%m-%d")
    export_path = export_dir / f"{today}.jsonl"

    collected = 0
    skipped = 0
    skip_reasons = Counter()

    json_paths = sorted(glob.glob(str(requests_dir / "*.json")))
    for status, json_path, payload in tqdm(map(_convert_file, json_paths), total=len(json_paths), desc="collect_jsonl"):
        json_file = Path(json_path)
        if status != "ok":
            skip_reasons[status.removeprefix("skip:")] += 1
            if payload:
                logger.debug("Skipping %s: %s", json_file.name, payload)
            skipped += 1
            continue

        if dry_run:
            logger.info("[DRY RUN] Would write to %s: %s", export_path, payload)
        else:
            assert payload is not None
            with export_path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")

        if processed_dir and not dry_run:
            shutil.move(str(json_file), str(processed_dir / json_file.name))
        elif not dry_run:
            json_file.unlink()

        collected += 1

    if skip_reasons:
        logger.info("Skip reasons: %s", json.dumps(dict(sorted(skip_reasons.items())), ensure_ascii=False))

    return collected, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect request logs and export Anthropic-format SFT JSONL")
    parser.add_argument(
        "--requests-dir",
        "--path-to-process",
        dest="path_to_process",
        type=Path,
        default=Path("~/.cache/llm-proxy/logs/requests").expanduser(),
        help="Directory containing raw .json request files",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("~/.cache/llm-proxy/training-data").expanduser(),
        help="Directory to write JSONL output files",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        help="Directory to move processed files (default: delete after export)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and log conversions without writing files or deleting sources",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    collected, skipped = _collect(
        args.path_to_process,
        args.export_dir,
        args.processed_dir,
        args.dry_run,
    )

    logger.info("Collected %d entries (%d skipped) to %s", collected, skipped, args.export_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()
