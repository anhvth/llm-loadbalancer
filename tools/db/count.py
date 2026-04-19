from __future__ import annotations

import argparse
import pathlib
import sqlite3


DEFAULT_DB_PATH = pathlib.Path("~/.cache/llm-proxy/state.sqlite3").expanduser()
TABLES = (
    "conversations",
    "input_states",
    "state_tail_messages",
    "messages",
    "requests",
    "route_affinity",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print row counts for the SQLite state DB")
    parser.add_argument(
        "db_path",
        nargs="?",
        type=pathlib.Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to the state DB (default: {DEFAULT_DB_PATH})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = args.db_path.expanduser()

    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        width = max(len(table) for table in TABLES)
        for table in TABLES:
            count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table.ljust(width)}  {count}")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
