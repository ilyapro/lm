#!/usr/bin/env python3
"""Import phase-2 transcript-grounding verdicts into the additive ledger.

    python scripts/transcript_ledger_import.py \
        --db ~/.local/share/living-memory/global.sqlite3 \
        --verdicts artifacts/transcript-grounding/grade/verdicts-local.jsonl \
        [--json OUT.json] [--dry-run]

Offline by design: the database is opened directly with ``sqlite3``, the one
new table is created if absent (the same shared DDL constant the server runs
at every open, so a pre-restart file gains exactly the objects it will carry
anyway), and *nothing else* is touched — this tool never constructs
``MemoryStore``, so none of the store's migrations or backfills run from
here.

Idempotent: ``UNIQUE(recall_event_id, node_id, method_version)`` makes a
re-run replay instead of duplicate, and the whole import is one transaction,
so an interrupted run leaves no partial state. An existing row is never
updated: a same-key verdict with different numbers is a method_version
discipline violation, reported under ``mismatched`` and not applied.

Exit codes: 0 — imported (or replayed) cleanly; 1 — mismatched same-key
verdicts were found (everything else still imported); 2 — malformed input or
unusable database, nothing committed.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sqlite3
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from living_memory.transcript_ledger import (  # noqa: E402
    TranscriptLedgerError,
    ensure_ledger_table,
    import_verdicts,
    read_verdicts,
)

#: A live server may hold the write lock for the moment its own transaction
#: runs; waiting beats failing for a tool whose input is offline data with no
#: urgency at all.
_BUSY_TIMEOUT_MS = 30_000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db", required=True, type=Path, help="Living Memory sqlite3 database file"
    )
    parser.add_argument(
        "--verdicts",
        required=True,
        type=Path,
        help="phase-2 per-event verdict JSONL file",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="also write the import report here"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and plan only; write nothing (no table creation either)",
    )
    args = parser.parse_args(argv)

    try:
        if not args.db.exists():
            raise TranscriptLedgerError(f"database file does not exist: {args.db}")
        verdicts = read_verdicts(args.verdicts)
        conn = sqlite3.connect(args.db, timeout=_BUSY_TIMEOUT_MS / 1000)
        try:
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            if not args.dry_run:
                ensure_ledger_table(conn)
            stats = import_verdicts(conn, verdicts, apply=not args.dry_run)
        finally:
            conn.close()
    except (TranscriptLedgerError, OSError, sqlite3.Error) as error:
        print(f"transcript_ledger_import: {error}", file=sys.stderr)
        return 2

    report = {
        "db": str(args.db),
        "verdicts": str(args.verdicts),
        "dry_run": bool(args.dry_run),
        **stats.as_dict(),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 1 if stats.mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
