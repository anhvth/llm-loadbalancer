from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys


DEFAULT_DB_PATH = pathlib.Path("llm_loadbalancer.sqlite3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print request_response_log rows as JSONL")
    parser.add_argument(
        "db_path",
        nargs="?",
        type=pathlib.Path,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database. Defaults to ./llm_loadbalancer.sqlite3",
    )
    return parser


def iter_rows(db_path: pathlib.Path):
    with sqlite3.connect(db_path) as connection:
        yield from connection.execute(
            """
            SELECT id, input, output, endpoint_used
            FROM request_response_log
            ORDER BY id
            """
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        for row in iter_rows(args.db_path):
            print(
                json.dumps(
                    {
                        "id": row[0],
                        "input": row[1],
                        "output": row[2],
                        "endpoint_used": row[3],
                    }
                )
            )
    except sqlite3.Error as exc:
        print(f"Failed to read {args.db_path}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
