#!/usr/bin/env python3
"""Verify the additive schema migration against a copy of a live database.

Copies the source database with the sqlite3 backup API (safe to run against a
database a live server is writing to — the source is opened read-only), opens
the copy with the current ``MemoryStore`` (running any pending schema
migration), and checks that the migration is purely additive:

* ``schema_version`` reaches the current ``SCHEMA_VERSION`` and
  ``recall_events`` gains ``transport_session_id`` via ``ALTER TABLE ADD
  COLUMN`` (the stored DDL is the old DDL with one column appended);
* node / connection / recall-event row counts and ``feedback_applied`` totals
  are identical before and after;
* every pre-existing recall event keeps ``transport_session_id`` NULL and the
  explicit/none identity partition is unchanged;
* ``PRAGMA integrity_check`` passes on the migrated copy;
* ``feedback_closure_metrics`` computes on the migrated copy (all-history and
  recent windows).

The copy is written to a scratch directory and never touches the source.

Usage::

    python scripts/verify_live_db_migration.py ~/.local/share/living-memory/global.sqlite3 \
        --json artifacts/transport_closure_live_db_check.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from living_memory.resources import feedback_closure_metrics
from living_memory.storage import SCHEMA_VERSION, MemoryStore

ALL_HISTORY_WINDOW_HOURS = 24 * 365 * 20


def _snapshot(conn: sqlite3.Connection) -> dict[str, Any]:
    """State that must survive the migration byte-identically."""

    conn.row_factory = sqlite3.Row
    columns = [
        str(row["name"]) for row in conn.execute("PRAGMA table_info(recall_events)")
    ]
    has_transport = "transport_session_id" in columns
    version_row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()
    ddl_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'recall_events'"
    ).fetchone()

    def count(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0])

    identity = dict(
        conn.execute(
            f"""
            SELECT
                CASE
                    WHEN agent IS NOT NULL AND task IS NOT NULL AND session_id IS NOT NULL
                        THEN 'explicit'
                    {"WHEN transport_session_id IS NOT NULL THEN 'transport_only'" if has_transport else ""}
                    ELSE 'none'
                END AS identity_class,
                COUNT(*) AS events
            FROM recall_events
            GROUP BY identity_class
            """
        ).fetchall()
    )
    return {
        "schema_version": str(version_row["value"]) if version_row else None,
        "recall_events_columns": columns,
        "recall_events_ddl": str(ddl_row["sql"]) if ddl_row else None,
        "nodes": count("SELECT COUNT(*) FROM nodes"),
        "connections": count("SELECT COUNT(*) FROM connections"),
        "recall_events": count("SELECT COUNT(*) FROM recall_events"),
        "feedback_applied": count(
            "SELECT COALESCE(SUM(feedback_applied), 0) FROM recall_events"
        ),
        "identity_classes": {
            name: int(identity.get(name, 0))
            for name in ("explicit", "transport_only", "none")
        },
    }


def _copy_database(source: Path, dest: Path) -> float:
    started = time.perf_counter()
    src_conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        dest_conn = sqlite3.connect(str(dest))
        try:
            src_conn.backup(dest_conn)
        finally:
            dest_conn.close()
    finally:
        src_conn.close()
    return round(time.perf_counter() - started, 3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the live SQLite database.")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Directory for the scratch copy (default: a fresh temp dir).",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Also write the report to this path."
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=168,
        help="Recent window for the feedback_closure metric (default 168).",
    )
    parser.add_argument(
        "--skip-integrity",
        action="store_true",
        help="Skip PRAGMA integrity_check (it scans the whole copy).",
    )
    args = parser.parse_args(argv)

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="lm-migration-check-"))
    workdir.mkdir(parents=True, exist_ok=True)
    copy_path = workdir / f"{args.source.stem}-migration-check.sqlite3"
    if copy_path.exists():
        copy_path.unlink()

    copy_seconds = _copy_database(args.source, copy_path)

    pre_conn = sqlite3.connect(str(copy_path))
    try:
        pre = _snapshot(pre_conn)
    finally:
        pre_conn.close()

    migrate_started = time.perf_counter()
    store = MemoryStore(copy_path)
    migrate_seconds = round(time.perf_counter() - migrate_started, 3)
    try:
        post = _snapshot(store.connection)
        legacy_transport_stamped = int(
            store.connection.execute(
                "SELECT COUNT(*) FROM recall_events WHERE transport_session_id IS NOT NULL"
            ).fetchone()[0]
        )
        integrity: str | None = None
        if not args.skip_integrity:
            integrity = str(
                store.connection.execute("PRAGMA integrity_check").fetchone()[0]
            )
        closure_all = feedback_closure_metrics(
            store, scope=None, window_hours=ALL_HISTORY_WINDOW_HOURS
        )
        closure_recent = feedback_closure_metrics(
            store, scope=None, window_hours=args.window_hours
        )
    finally:
        store.close()

    pre_columns = set(pre["recall_events_columns"])
    post_columns = set(post["recall_events_columns"])
    checks = {
        "schema_version_current": post["schema_version"] == str(SCHEMA_VERSION),
        "transport_column_present": "transport_session_id" in post_columns,
        "column_change_additive_only": (
            post_columns - pre_columns <= {"transport_session_id"}
            and pre_columns <= post_columns
        ),
        "ddl_extended_not_rebuilt": (
            pre["recall_events_ddl"] == post["recall_events_ddl"]
            if "transport_session_id" in pre_columns
            else str(post["recall_events_ddl"]).startswith(
                str(pre["recall_events_ddl"]).rstrip().removesuffix(")")
            )
        ),
        "nodes_preserved": pre["nodes"] == post["nodes"],
        "connections_preserved": pre["connections"] == post["connections"],
        "recall_events_preserved": pre["recall_events"] == post["recall_events"],
        "feedback_flags_preserved": pre["feedback_applied"] == post["feedback_applied"],
        "identity_partition_preserved": (
            pre["identity_classes"]["explicit"] == post["identity_classes"]["explicit"]
            and pre["identity_classes"]["none"] == post["identity_classes"]["none"]
        ),
        "no_legacy_row_stamped": legacy_transport_stamped
        == pre["identity_classes"]["transport_only"],
        "integrity_ok": args.skip_integrity or integrity == "ok",
        "closure_metric_computes": isinstance(
            closure_all.get("closure_ratio"), (int, float)
        ),
    }

    report = {
        "source": str(args.source),
        "copy": str(copy_path),
        "copy_seconds": copy_seconds,
        "migrate_open_seconds": migrate_seconds,
        "schema_version": {"pre": pre["schema_version"], "post": post["schema_version"]},
        "pre": pre,
        "post": post,
        "integrity_check": integrity,
        "feedback_closure_all_history": closure_all,
        "feedback_closure_recent": closure_recent,
        "checks": checks,
        "ok": all(checks.values()),
    }

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
