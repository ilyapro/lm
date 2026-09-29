#!/usr/bin/env python3
"""Collapse copies that consolidation left behind: one active node per fact.

Two kinds of copy are one group each:

* nodes of the same level and scope with byte-identical content (the rule the
  write path now applies to every level);
* active schemas of the same scope and procedure key, whatever their content:
  consolidation updates exactly one schema per group, so every other one is a
  copy the next pass would never touch.

Each group keeps one node; every other member is decayed with reason
``duplicate_content`` behind a ``supersedes`` edge from the kept node, exactly
as the trace write path does. The kept node absorbs the usage of the copies:
``access_count`` is summed, ``last_accessed`` and ``usefulness_score`` take the
maximum, and query-anchor edges are merged (hits summed, weight maximum). No
row is deleted and no content is rewritten.

The kept node is, in order: one that can still be updated (no ``corrections``,
no incoming ``supersedes``) -- a corrected or replaced node never comes back to
life, and consolidation would skip it and write a new copy next pass; then the
most accessed, so the node recall already credits keeps its id; then the most
recently updated; then the highest id.

Like ``lm_collapse_near_dups.py`` it imports nothing from ``living_memory``: it
runs under whatever interpreter owns the database. ``--dry-run`` is the default
and opens the database read-only; ``--apply`` writes in one transaction. A
second ``--apply`` finds nothing to do.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import urllib.parse
from datetime import datetime, timezone

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


def normalize_trigger(value: str) -> str:
    """Mirror of ``living_memory.consolidation._normalize_trigger``."""

    cleaned = value.replace("_", " ").replace("-", " ").replace("/", " ")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def schema_group_id(context: dict, provenance: dict) -> str:
    """Mirror of ``living_memory.consolidation._schema_group_id``."""

    group = context.get("procedure_key") or provenance.get("procedure_key")
    if group is not None and str(group).strip():
        return normalize_trigger(str(group))
    for field in ("task_pattern", "procedure_id"):
        raw = str(context.get(field) or "").strip()
        if raw:
            return normalize_trigger(raw)
    return ""


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


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        self.parent[self.find(left)] = self.find(right)


def find_groups(connection: sqlite3.Connection) -> list[dict]:
    """Return copy groups: ``{"keep": row, "copies": [row...], "basis": [...]}``."""

    rows = connection.execute(
        """
        SELECT id, level, scope, content, context, provenance, corrections,
               access_count, last_accessed, usefulness_score, updated_at
        FROM nodes WHERE decayed = 0
        """
    ).fetchall()
    superseded = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT target_id FROM connections WHERE type = 'supersedes'"
        )
    }
    by_id = {row["id"]: row for row in rows}
    fingerprints: dict[str, str] = {}
    buckets: dict[tuple, list[str]] = {}
    for row in rows:
        fingerprint = hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
        fingerprints[row["id"]] = fingerprint
        buckets.setdefault(("content", row["level"], row["scope"], fingerprint), []).append(
            row["id"]
        )
        if row["level"] == "schema":
            group = schema_group_id(_json(row["context"], {}), _json(row["provenance"], {}))
            if group:
                buckets.setdefault(("procedure_key", row["scope"], group), []).append(row["id"])

    union = _UnionFind()
    bases: dict[str, set[str]] = {}
    for key, ids in buckets.items():
        if len(ids) < 2:
            continue
        for node_id in ids[1:]:
            union.union(node_id, ids[0])
        for node_id in ids:
            bases.setdefault(node_id, set()).add(key[0])

    components: dict[str, list[str]] = {}
    for node_id in bases:
        components.setdefault(union.find(node_id), []).append(node_id)

    def rank(node_id: str) -> tuple:
        row = by_id[node_id]
        updatable = not _json(row["corrections"], []) and node_id not in superseded
        return (updatable, int(row["access_count"] or 0), row["updated_at"] or "", node_id)

    groups = []
    for members in components.values():
        ordered = sorted(members, key=rank, reverse=True)
        keep = by_id[ordered[0]]
        groups.append(
            {
                "keep": keep,
                "copies": [by_id[node_id] for node_id in ordered[1:]],
                "basis": sorted(set().union(*(bases[node_id] for node_id in members))),
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
                same_content = group["fingerprints"][copy["id"]] == group["fingerprints"][keep["id"]]
                metadata = {
                    "kind": EDGE_KIND,
                    "basis": "content" if same_content else "procedure_key",
                    "fingerprint": group["fingerprints"][copy["id"]],
                    "collapsed_by": PROGRAM,
                }
                connection.execute(
                    """
                    INSERT OR IGNORE INTO connections
                        (id, source_id, target_id, type, weight, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, 'supersedes', 1.0, ?, ?, ?)
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
