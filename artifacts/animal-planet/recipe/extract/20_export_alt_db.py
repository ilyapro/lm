#!/usr/bin/env python3
"""Step 20: export the audit slice of the alt LM DB snapshot into staging.

Reads the checkpointed snapshot (step 10) strictly read-only (mode=ro URI)
and writes JSONL/JSON files under $AP_STAGING/alt-db/:

- recall_events.jsonl          all recall_events with created_at in the
                               export window (windows.json: W1.start..W2.end),
                               full rows, JSON columns kept as stored text.
- nodes.jsonl                  all nodes referenced by those events' results
                               (every candidate node_id at every rank) plus
                               their feedback_trace_id nodes. All columns
                               except `embedding` (excluded: bulky model
                               vector, irrelevant to the replay corpus —
                               BM25/vector re-execution is out of scope).
- connections_typed.jsonl      ALL supersedes + contradicts rows (full table
                               slice, not window-bound).
- nodes_typed_edges.jsonl      endpoint nodes of typed edges not already in
                               nodes.jsonl (for correction-ordering analysis).
- connections_among_exported.jsonl  remaining edges (related/caused/requires)
                               where BOTH endpoints are exported nodes.
- game_schema_nodes.jsonl      project:game level=schema nodes (mixed-era
                               consolidation evidence).
- game_schema_source_nodes.jsonl  their source_traces nodes.
- outcome_nodes.jsonl          all nodes whose content matches '%OUTCOME%'
                               (superset of the node.sh auto-OUTCOME
                               template; predicate pinning happens in step 70).
- retrieval_weights.json       full retrieval_weights table.
- schema.json                  PRAGMA table_info of the four tables.
- summary.json                 row counts and per-scope aggregates.

No node content, query text or any other raw text is ever printed to
stdout/stderr — only counts.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGING = Path(os.environ.get("AP_STAGING", "/home/sfx/.cache/ap-audit/staging"))
SNAPSHOT = Path(os.environ.get("AP_TMP", "/tmp/ap-audit")) / "alt" / "global.sqlite3"
OUT = STAGING / "alt-db"

NODE_DROP_COLS = ("embedding",)
TABLES = ("recall_events", "nodes", "connections", "retrieval_weights")


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


def fetch_nodes(conn, node_ids, drop_cols=NODE_DROP_COLS):
    ids = [nid for nid in dict.fromkeys(node_ids) if nid]
    found = []
    for start in range(0, len(ids), 900):
        chunk = ids[start : start + 900]
        placeholders = ",".join("?" * len(chunk))
        found.extend(
            conn.execute(f"SELECT * FROM nodes WHERE id IN ({placeholders})", chunk)
        )
    return ids, found


def main() -> int:
    windows = json.loads((HERE / "windows.json").read_text(encoding="utf-8"))
    export_start = windows["export"]["start"]
    export_end = windows["export"]["end"]

    OUT.mkdir(parents=True, exist_ok=True)
    conn = open_ro(SNAPSHOT)
    summary: dict = {"snapshot": str(SNAPSHOT), "export_window": [export_start, export_end]}

    schema = {
        table: [
            {"name": r[1], "type": r[2], "notnull": r[3], "pk": r[5]}
            for r in conn.execute(f"PRAGMA table_info({table})")
        ]
        for table in TABLES
    }
    (OUT / "schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    # --- recall_events in the export window -------------------------------
    events = conn.execute(
        "SELECT * FROM recall_events WHERE created_at >= ? AND created_at <= ?"
        " ORDER BY created_at, rowid",
        (export_start, export_end),
    ).fetchall()
    summary["recall_events_exported"] = dump_rows(events, OUT / "recall_events.jsonl")
    summary["recall_events_total_in_db"] = conn.execute(
        "SELECT COUNT(*) FROM recall_events"
    ).fetchone()[0]
    per_scope: dict[str, int] = {}
    for row in events:
        per_scope[row["scope"]] = per_scope.get(row["scope"], 0) + 1
    summary["recall_events_per_scope"] = dict(
        sorted(per_scope.items(), key=lambda kv: -kv[1])
    )

    # --- nodes referenced by results + feedback traces --------------------
    referenced: list[str] = []
    bad_results_json = 0
    for row in events:
        try:
            results = json.loads(row["results"] or "[]")
        except ValueError:
            bad_results_json += 1
            results = []
        for result in results:
            node_id = result.get("node_id")
            if node_id:
                referenced.append(str(node_id))
        if row["feedback_trace_id"]:
            referenced.append(str(row["feedback_trace_id"]))
    requested_ids, node_rows = fetch_nodes(conn, referenced)
    exported_node_ids = {r["id"] for r in node_rows}
    summary["results_json_parse_errors"] = bad_results_json
    summary["referenced_node_ids"] = len(requested_ids)
    summary["referenced_nodes_found"] = dump_rows(
        node_rows, OUT / "nodes.jsonl", drop_cols=NODE_DROP_COLS
    )
    summary["referenced_nodes_missing"] = len(requested_ids) - len(node_rows)

    # --- typed connections (full table slice) -----------------------------
    typed = conn.execute(
        "SELECT * FROM connections WHERE type IN ('supersedes','contradicts')"
        " ORDER BY created_at, rowid"
    ).fetchall()
    summary["connections_typed_exported"] = dump_rows(typed, OUT / "connections_typed.jsonl")
    summary["connections_by_type_total"] = {
        row["type"]: row["n"]
        for row in conn.execute("SELECT type, COUNT(*) AS n FROM connections GROUP BY type")
    }

    typed_endpoint_ids = {r["source_id"] for r in typed} | {r["target_id"] for r in typed}
    extra_ids = sorted(typed_endpoint_ids - exported_node_ids)
    _, extra_rows = fetch_nodes(conn, extra_ids)
    summary["typed_edge_extra_nodes"] = dump_rows(
        extra_rows, OUT / "nodes_typed_edges.jsonl", drop_cols=NODE_DROP_COLS
    )
    all_exported_ids = exported_node_ids | {r["id"] for r in extra_rows}

    # --- remaining edges among exported nodes -----------------------------
    among = [
        row
        for row in conn.execute(
            "SELECT * FROM connections WHERE type NOT IN ('supersedes','contradicts')"
        )
        if row["source_id"] in all_exported_ids and row["target_id"] in all_exported_ids
    ]
    summary["connections_among_exported"] = dump_rows(
        among, OUT / "connections_among_exported.jsonl"
    )

    # --- project:game schema nodes + their source traces ------------------
    schema_nodes = conn.execute(
        "SELECT * FROM nodes WHERE scope = 'project:game' AND level = 'schema'"
        " ORDER BY created_at"
    ).fetchall()
    summary["game_schema_nodes"] = dump_rows(
        schema_nodes, OUT / "game_schema_nodes.jsonl", drop_cols=NODE_DROP_COLS
    )
    summary["game_schema_nodes_active"] = sum(1 for r in schema_nodes if not r["decayed"])
    source_ids: list[str] = []
    for row in schema_nodes:
        try:
            source_ids.extend(str(x) for x in json.loads(row["source_traces"] or "[]"))
        except ValueError:
            pass
    requested_src, src_rows = fetch_nodes(conn, source_ids)
    summary["game_schema_source_ids"] = len(requested_src)
    summary["game_schema_source_nodes"] = dump_rows(
        src_rows, OUT / "game_schema_source_nodes.jsonl", drop_cols=NODE_DROP_COLS
    )

    # --- auto-OUTCOME candidate nodes (broad superset; pinned in step 70) --
    outcome_rows = conn.execute(
        "SELECT * FROM nodes WHERE content LIKE '%OUTCOME%' ORDER BY created_at"
    ).fetchall()
    summary["outcome_candidate_nodes"] = dump_rows(
        outcome_rows, OUT / "outcome_nodes.jsonl", drop_cols=NODE_DROP_COLS
    )

    # --- retrieval weights -------------------------------------------------
    weights = [dict(row) for row in conn.execute("SELECT * FROM retrieval_weights ORDER BY scope")]
    (OUT / "retrieval_weights.json").write_text(
        json.dumps(weights, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary["retrieval_weights_rows"] = len(weights)

    conn.close()
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    printable = {k: v for k, v in summary.items() if k != "recall_events_per_scope"}
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
