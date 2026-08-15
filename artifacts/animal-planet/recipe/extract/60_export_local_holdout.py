#!/usr/bin/env python3
"""Step 60: export holdout workloads from the LOCAL LM DB snapshot.

For each scope in $AP_HOLDOUT_SCOPES (non-animal-planet projects; defaults:
project:octopus, project:online, project:x) exports the most recent
$AP_HOLDOUT_LIMIT (default 1500) recall_events — written in chronological
order — plus every node referenced by their results/feedback_trace_id.
Same row format as the alt export (all columns; nodes minus `embedding`).

Files: $AP_STAGING/local-db/holdout_<scope_slug>.recall_events.jsonl,
       $AP_STAGING/local-db/holdout_<scope_slug>.nodes.jsonl,
       $AP_STAGING/local-db/summary.json
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

STAGING = Path(os.environ.get("AP_STAGING", "/home/sfx/.cache/ap-audit/staging"))
SNAPSHOT = Path(os.environ.get("AP_TMP", "/tmp/ap-audit")) / "local" / "global.sqlite3"
SCOPES = [
    scope.strip()
    for scope in os.environ.get(
        "AP_HOLDOUT_SCOPES", "project:octopus,project:online,project:x"
    ).split(",")
    if scope.strip()
]
LIMIT = int(os.environ.get("AP_HOLDOUT_LIMIT", "1500"))
OUT = STAGING / "local-db"
NODE_DROP_COLS = ("embedding",)


def open_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def dump_rows(rows, out_path: Path, drop_cols=()) -> int:
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            record = {key: row[key] for key in row.keys() if key not in drop_cols}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = open_ro(SNAPSHOT)
    summary: dict = {"snapshot": str(SNAPSHOT), "limit_per_scope": LIMIT, "scopes": {}}
    for scope in SCOPES:
        slug = scope.replace(":", "_").replace("/", "_")
        total = conn.execute(
            "SELECT COUNT(*) FROM recall_events WHERE scope = ?", (scope,)
        ).fetchone()[0]
        events = conn.execute(
            "SELECT * FROM recall_events WHERE scope = ?"
            " ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (scope, LIMIT),
        ).fetchall()
        events = list(reversed(events))
        events_written = dump_rows(events, OUT / f"holdout_{slug}.recall_events.jsonl")

        referenced: list[str] = []
        bad_json = 0
        for row in events:
            try:
                results = json.loads(row["results"] or "[]")
            except ValueError:
                bad_json += 1
                results = []
            for result in results:
                node_id = result.get("node_id")
                if node_id:
                    referenced.append(str(node_id))
            if row["feedback_trace_id"]:
                referenced.append(str(row["feedback_trace_id"]))
        ids = [node_id for node_id in dict.fromkeys(referenced) if node_id]
        node_rows = []
        for start in range(0, len(ids), 900):
            chunk = ids[start : start + 900]
            placeholders = ",".join("?" * len(chunk))
            node_rows.extend(
                conn.execute(f"SELECT * FROM nodes WHERE id IN ({placeholders})", chunk)
            )
        nodes_written = dump_rows(
            node_rows, OUT / f"holdout_{slug}.nodes.jsonl", drop_cols=NODE_DROP_COLS
        )
        summary["scopes"][scope] = {
            "recall_events_total_in_db": total,
            "recall_events_exported": events_written,
            "span": [events[0]["created_at"], events[-1]["created_at"]] if events else None,
            "referenced_node_ids": len(ids),
            "nodes_exported": nodes_written,
            "nodes_missing": len(ids) - nodes_written,
            "results_json_parse_errors": bad_json,
        }
    conn.close()
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["scopes"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
