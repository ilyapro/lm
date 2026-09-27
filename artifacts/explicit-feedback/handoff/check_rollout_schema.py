#!/usr/bin/env python3
"""Read-only schema check for the explicit-feedback rollout.

Standard library only, so it runs with the system ``python3`` on alt, which has
no living_memory deps. It opens the database as ``file:...?mode=ro``, so it
never writes. That includes the live store while the server runs.

Checks:

* ``schema_version`` is still 8 (the rollout adds tables, it does not bump it);
* ``recall_feedback_marks``, ``recall_explicit_credit`` and
  ``idx_recall_feedback_marks_event`` exist (created by the first open of the
  new code, so they appear on restart);
* ``recall_credit_ledger`` keeps its ``CHECK (basis IN ('grounded','lookup'))``
  DDL (no rebuild happened);
* optionally, ``--expect absent`` asserts the pre-rollout state instead.

``query_irrelevance`` is reported but never required: it is created lazily by
the first accepted irrelevant mark under ``LM_EXPLICIT_FEEDBACK_POLICY=credit``.

Exit 0 when every check passes, 1 otherwise. Prints one JSON object.

    python3 check_rollout_schema.py ~/.local/share/living-memory/global.sqlite3
    python3 check_rollout_schema.py <db> --expect absent     # before restart
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys

NEW_TABLES = ("recall_feedback_marks", "recall_explicit_credit")
NEW_INDEX = "idx_recall_feedback_marks_event"
LAZY_TABLE = "query_irrelevance"
MARK_COLUMNS = {
    "id", "recall_event_id", "node_id", "mark", "accepted", "reject_reason",
    "via_tool", "source_id", "transport_session_id", "agent", "rank", "marked_at",
}


def inspect(path: str) -> dict:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        objects = {
            (row[0], row[1]): row[2]
            for row in conn.execute("SELECT type, name, sql FROM sqlite_master")
        }
        version = conn.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        counts = {}
        for table in NEW_TABLES + (LAZY_TABLE, "recall_credit_ledger"):
            if ("table", table) in objects:
                counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        mark_columns = (
            {row[1] for row in conn.execute("PRAGMA table_info(recall_feedback_marks)")}
            if ("table", "recall_feedback_marks") in objects
            else set()
        )
        by_mark = (
            [list(row) for row in conn.execute(
                "SELECT mark, accepted, COUNT(*) FROM recall_feedback_marks "
                "GROUP BY mark, accepted ORDER BY mark, accepted"
            )]
            if mark_columns
            else []
        )
        ledger_ddl = objects.get(("table", "recall_credit_ledger")) or ""
    finally:
        conn.close()
    return {
        "schema_version": version[0] if version else None,
        "tables": {name: ("table", name) in objects for name in NEW_TABLES + (LAZY_TABLE,)},
        "index": ("index", NEW_INDEX) in objects,
        "mark_columns_missing": sorted(MARK_COLUMNS - mark_columns) if mark_columns else None,
        "ledger_check_unchanged": "basis IN ('grounded','lookup')" in ledger_ddl.replace(", ", ","),
        "row_counts": counts,
        "marks_by_mark_accepted": by_mark,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db")
    parser.add_argument("--expect", choices=("present", "absent"), default="present")
    args = parser.parse_args(argv)
    state = inspect(args.db)
    present = all(state["tables"][t] for t in NEW_TABLES) and state["index"]
    absent = not any(state["tables"][t] for t in NEW_TABLES) and not state["index"]
    checks = {
        "schema_version_8": state["schema_version"] == "8",
        "ledger_check_unchanged": state["ledger_check_unchanged"],
    }
    if args.expect == "present":
        checks["new_tables_and_index_present"] = present
        checks["mark_columns_complete"] = state["mark_columns_missing"] == []
    else:
        checks["new_tables_and_index_absent"] = absent
    report = {"db": args.db, "expect": args.expect, **state, "checks": checks,
              "ok": all(checks.values())}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
