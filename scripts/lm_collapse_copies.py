#!/usr/bin/env python3
"""Collapse byte-identical copies with matching scope, level and provenance.

Recorded derivations must agree on their materializer, group and source inputs.
Each group keeps one node; every other member is decayed with reason
``duplicate_content`` behind a ``related`` edge from the kept node. This is
technical bookkeeping, not a semantic correction. The kept node absorbs usage:
``access_count`` is summed, ``last_accessed`` and ``usefulness_score`` take the
maximum, and query-anchor edges are merged (hits summed, weight maximum). No
row is deleted and no content is rewritten.

The kept node is, in order: one that can still be updated (no ``corrections``,
no incoming correction) -- a corrected or replaced node never comes back to
life, and consolidation would skip it and write a new copy next pass; then the
most accessed, so the node recall already credits keeps its id; then the most
recently updated; then the highest id.

Run with the owning Living Memory environment. Identity and correction policies
come from this checkout's shared models. ``--dry-run`` is the default and opens
the database read-only; ``--apply`` writes in one transaction. A second
``--apply`` finds nothing to do.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from living_memory.models import Node, content_identity  # noqa: E402
from living_memory.relations import correction_sql  # noqa: E402

PROGRAM = "lm_collapse_copies.py"
DECAY_REASON = "duplicate_content"
EDGE_KIND = "duplicate_content"
_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _ulid() -> str:
    value = (int(time.time() * 1000) << 80) | int.from_bytes(os.urandom(10), "big")
    chars = []
    for _ in range(26):
        chars.append(_ULID_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def connect(path: str, read_only: bool) -> sqlite3.Connection:
    if read_only:
        uri = "file:" + urllib.parse.quote(os.path.abspath(path)) + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
    else:
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA busy_timeout = 30000")
    connection.row_factory = sqlite3.Row
    return connection


def _json(raw, default):
    try:
        value = json.loads(raw) if raw else default
    except (TypeError, ValueError):
        return default
    return value if isinstance(value, type(default)) else default


def find_groups(connection: sqlite3.Connection) -> list[dict]:
    """Return copy groups: ``{"keep": row, "copies": [row...], "basis": [...]}``."""

    rows = connection.execute(
        """
        SELECT id, level, scope, content, context, provenance, source_traces, corrections,
               access_count, last_accessed, usefulness_score, updated_at
        FROM nodes WHERE decayed = 0
        """
    ).fetchall()
    superseded = {
        row[0]
        for row in connection.execute(
            f"SELECT DISTINCT target_id FROM connections WHERE {correction_sql()}"
        )
    }
    by_id = {row["id"]: row for row in rows}
    fingerprints: dict[str, str] = {}
    buckets: dict[tuple, list[str]] = {}
    for row in rows:
        fingerprint = hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
        fingerprints[row["id"]] = fingerprint
        node = Node(
            id=row["id"],
            level=row["level"],
            scope=row["scope"],
            content=row["content"],
            context=_json(row["context"], {}),
            provenance=_json(row["provenance"], {}),
            source_traces=_json(row["source_traces"], []),
        )
        identity = content_identity(node)
        if identity is not None:
            buckets.setdefault((node.level, identity), []).append(node.id)

    def rank(node_id: str) -> tuple:
        row = by_id[node_id]
        updatable = not _json(row["corrections"], []) and node_id not in superseded
        return (updatable, int(row["access_count"] or 0), row["updated_at"] or "", node_id)

    groups = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=rank, reverse=True)
        keep = by_id[ordered[0]]
        groups.append(
            {
                "keep": keep,
                "copies": [by_id[node_id] for node_id in ordered[1:]],
                "basis": ["content"],
                "fingerprints": fingerprints,
            }
        )
    groups.sort(key=lambda group: (-len(group["copies"]), group["keep"]["scope"], group["keep"]["id"]))
    return groups


def apply_groups(connection: sqlite3.Connection, groups: list[dict]) -> int:
    now = _utc_now()
    anchor_edges = bool(
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'query_anchor_edges'"
        ).fetchone()
    )
    decayed = 0
    with connection:
        for group in groups:
            keep = group["keep"]
            copies = group["copies"]
            for copy in copies:
                metadata = {
                    "kind": EDGE_KIND,
                    "basis": "content",
                    "fingerprint": group["fingerprints"][copy["id"]],
                    "collapsed_by": PROGRAM,
                }
                connection.execute(
                    """
                    INSERT OR IGNORE INTO connections
                        (id, source_id, target_id, type, weight, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, 'related', 1.0, ?, ?, ?)
                    """,
                    (_ulid(), keep["id"], copy["id"], json.dumps(metadata, sort_keys=True), now, now),
                )
                connection.execute(
                    "UPDATE nodes SET decayed = 1, decay_reason = ?, updated_at = ? WHERE id = ?",
                    (DECAY_REASON, now, copy["id"]),
                )
                decayed += 1
                if anchor_edges:
                    _merge_anchor_edges(connection, copy["id"], keep["id"], now)
            last_accessed = max(
                (row["last_accessed"] for row in (keep, *copies) if row["last_accessed"]),
                default=None,
            )
            connection.execute(
                """
                UPDATE nodes
                SET access_count = ?, last_accessed = ?, usefulness_score = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    sum(int(row["access_count"] or 0) for row in (keep, *copies)),
                    last_accessed,
                    max(float(row["usefulness_score"] or 0.0) for row in (keep, *copies)),
                    now,
                    keep["id"],
                ),
            )
    return decayed


def _merge_anchor_edges(connection: sqlite3.Connection, copy_id: str, keep_id: str, now: str) -> None:
    for edge in connection.execute(
        "SELECT anchor_id, weight, hits, created_at FROM query_anchor_edges WHERE target_id = ?",
        (copy_id,),
    ).fetchall():
        connection.execute(
            """
            INSERT INTO query_anchor_edges (anchor_id, target_id, weight, hits, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (anchor_id, target_id) DO UPDATE SET
                weight = max(weight, excluded.weight),
                hits = hits + excluded.hits,
                updated_at = excluded.updated_at
            """,
            (edge[0], keep_id, edge[1], edge[2], edge[3], now),
        )


def report(groups: list[dict], *, limit: int) -> dict:
    levels: dict[str, dict[str, int]] = {}
    for group in groups:
        entry = levels.setdefault(group["keep"]["level"], {"groups": 0, "copies": 0})
        entry["groups"] += 1
        entry["copies"] += len(group["copies"])
    summary = {
        "groups": len(groups),
        "copies": sum(len(group["copies"]) for group in groups),
        "by_level": levels,
    }
    print(f"{PROGRAM}: {summary['groups']} groups, {summary['copies']} copies to decay")
    for level, entry in sorted(levels.items()):
        print(f"  {level}: {entry['groups']} groups, {entry['copies']} copies")
    for group in groups[:limit]:
        keep = group["keep"]
        head = keep["content"].splitlines()[0][:90] if keep["content"] else ""
        print(
            f"  keep {keep['id']} {keep['level']} {keep['scope']} "
            f"+{len(group['copies'])} [{','.join(group['basis'])}] {head!r}"
        )
    return summary


def parse_args(argv):
    parser = argparse.ArgumentParser(prog=PROGRAM, description=__doc__.splitlines()[0])
    parser.add_argument("database", help="path to the Living Memory sqlite database")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True, help="report only (default)")
    mode.add_argument("--apply", action="store_true", help="decay the copies")
    parser.add_argument("--show", type=int, default=25, help="groups to list (default 25)")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON last")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not os.path.exists(args.database):
        print(f"{PROGRAM}: no such database: {args.database}", file=sys.stderr)
        return 2
    connection = connect(args.database, read_only=not args.apply)
    try:
        groups = find_groups(connection)
        summary = report(groups, limit=args.show)
        if args.apply:
            summary["decayed"] = apply_groups(connection, groups)
            print(f"applied: {summary['decayed']} copies decayed")
        else:
            print("dry run: nothing written (pass --apply to decay the copies)")
    finally:
        connection.close()
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
