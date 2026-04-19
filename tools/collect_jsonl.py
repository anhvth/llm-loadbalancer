#!/usr/bin/env python3
"""Collect request logs and export eligible rows to JSONL.

Eligibility: the input payload must be a dict containing a top-level
"messages" key.  Each exported row is a simple JSON object with
id, timestamp, input, and output.  No SFT conversion is performed here.
"""

import argparse
import datetime
import glob
import importlib.util
import json
import logging
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger("collect_jsonl")


def _has_messages(input_payload: object) -> bool:
    """Return True if *input_payload* is a dict with a top-level 'messages' key."""
    return isinstance(input_payload, dict) and "messages" in input_payload


def _make_row(row_id: object, timestamp: str | None, input_payload: object, output_payload: object) -> str:
    """Serialize an export row to a JSON string."""
    row = {
        "id": row_id,
        "timestamp": timestamp,
        "input": input_payload,
        "output": output_payload,
    }
    return json.dumps(row, ensure_ascii=False)


def _convert_file(json_path: str) -> tuple[str, str, str | None]:
    path = Path(json_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ("skip:unreadable_json", json_path, f"unreadable: {exc}")

    if not isinstance(payload, dict):
        return ("skip:not_a_dict", json_path, None)

    # Support wrapped {"input": ..., "output": ...} or bare request payloads.
    if "input" in payload and "output" in payload:
        input_payload = payload["input"]
        output_payload = payload["output"]
    else:
        input_payload = payload
        output_payload = None

    if not _has_messages(input_payload):
        return ("skip:missing_messages", json_path, None)

    row_id = path.stem
    row_json = _make_row(row_id, None, input_payload, output_payload)
    return ("ok", json_path, row_json)


def _load_load_balancer_v2_module():
    try:
        import load_balancer_v2
    except ModuleNotFoundError:
        module_path = Path(__file__).resolve().parent.parent / "load_balancer_v2.py"
        spec = importlib.util.spec_from_file_location("load_balancer_v2", module_path)
        assert spec is not None and spec.loader is not None
        load_balancer_v2 = importlib.util.module_from_spec(spec)
        sys.modules["load_balancer_v2"] = load_balancer_v2
        spec.loader.exec_module(load_balancer_v2)
    return load_balancer_v2


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


def _collect_state_db(
    state_db: Path,
    export_dir: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    load_balancer_v2 = _load_load_balancer_v2_module()

    export_dir.mkdir(parents=True, exist_ok=True)
    today = time.strftime("%Y-%m-%d")
    export_path = export_dir / f"{today}.jsonl"

    collected = 0
    skipped = 0
    skip_reasons = Counter()

    with load_balancer_v2.StateDbReader(state_db) as reader:
        for record in reader.iter_requests(only_uncollected=True):
            input_payload = record.input_payload
            if not _has_messages(input_payload):
                skip_reasons["missing_messages"] += 1
                skipped += 1
                continue

            ts = datetime.datetime.fromtimestamp(
                record.created_ns / 1e9, tz=datetime.timezone.utc
            ).isoformat()
            payload = _make_row(
                record.request_id, ts, input_payload, record.output_payload
            )

            if dry_run:
                logger.info("[DRY RUN] Would write request %s to %s", record.request_id, export_path)
            else:
                assert payload is not None
                with export_path.open("a", encoding="utf-8") as handle:
                    handle.write(payload + "\n")
                load_balancer_v2.mark_requests_collected(state_db, [record.request_id])

            collected += 1

    if skip_reasons:
        logger.info("Skip reasons: %s", json.dumps(dict(sorted(skip_reasons.items())), ensure_ascii=False))

    return collected, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect request logs and export eligible rows to JSONL")
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
        "--state-db",
        type=Path,
        help="Export pending rows from a v2 SQLite state database instead of request files",
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
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.state_db is not None:
        collected, skipped = _collect_state_db(
            args.state_db,
            args.export_dir,
            args.dry_run,
        )
    else:
        collected, skipped = _collect(
            args.path_to_process,
            args.export_dir,
            args.processed_dir,
            args.dry_run,
        )

    logger.info("Collected %d entries (%d skipped) to %s", collected, skipped, args.export_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
