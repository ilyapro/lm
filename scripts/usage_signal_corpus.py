#!/usr/bin/env python3
"""Build the usage-signal replay corpus: closed recall events with their closers.

The three usage-signal vectors (lookup credit, Cyrillic tokenization, the
grounding threshold) are all measured on the same population, so that
population is extracted once, here, and every measurement script replays the
same JSONL instead of re-deriving its own join against the database.

One record per **closed** recall event — ``feedback_trace_id IS NOT NULL`` —
created at or after ``--since``. A record carries everything a replay needs
without touching the database again:

* the event (id, query, scope, transport session, created_at) and its
  recorded results with the per-channel scores the live credit rule attributes
  by (``bm25_score``/``vector_score``/``graph_score``), rank, level, and the
  node's *current* content (what ``ground_results`` would tokenize);
* the closing trace: id, content, scope, agent, task, created_at, and the
  closure lag in seconds;
* a language class per result node and for the trace — ``cyr`` when at least
  a fifth of the letters are Cyrillic, ``lat`` when at most a twentieth are,
  ``mixed`` otherwise — so grounding rates can be split by script;
* relatedness, measured with the production multilingual encoder and therefore
  independent of the lexical tokenizer the containment measure runs on:
  ``query_trace_cosine`` (was the closing trace about what was asked?) and,
  per result, ``node_trace_cosine`` (is this delivered node about what the
  closing trace says?);
* lookups: every ``recall_lookup_events`` row naming one of the event's result
  nodes after the event, tagged with whether it shares the event's transport
  session and its lag from the event.

Read-only discipline: the database is opened ``file:...?mode=ro`` and never
through :class:`MemoryStore`. Point ``--db`` at a snapshot taken with the
SQLite backup API (``retrieval_harness.create_snapshot``), never at the live
file of a running server.

Usage::

    python3 scripts/usage_signal_corpus.py \
        --db ~/.cache/living-memory-harness/usage-signal/sfx-2026-09-07.sqlite3 \
        --since 2026-09-04T00:00:00Z --host sfx \
        --out ~/.cache/living-memory-harness/usage-signal/sfx-closures.jsonl
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity  # noqa: E402

CORPUS_VERSION = 1


def open_readonly(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _parse_instant(raw: str) -> datetime:
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _lag_seconds(earlier: str, later: str) -> float:
    return (_parse_instant(later) - _parse_instant(earlier)).total_seconds()


def script_class(text: str) -> str:
    """``cyr`` / ``lat`` / ``mixed`` by share of Cyrillic letters among letters."""

    letters = 0
    cyrillic = 0
    for char in text:
        if not char.isalpha():
            continue
        letters += 1
        if "Ѐ" <= char <= "ӿ":
            cyrillic += 1
    if letters == 0:
        return "lat"
    share = cyrillic / letters
    if share >= 0.2:
        return "cyr"
    if share <= 0.05:
        return "lat"
    return "mixed"


def pair_class(node_class: str, trace_class: str) -> str:
    """``lat/lat`` when both sides are Latin, otherwise ``cyr-any``."""

    if node_class == "lat" and trace_class == "lat":
        return "lat/lat"
    return "cyr-any"


class Encoder:
    """Memoized production encoder; one vector per distinct text."""

    def __init__(self, model: LocalEmbeddingModel | None = None) -> None:
        self.model = model or LocalEmbeddingModel()
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        vector = self._cache.get(text)
        if vector is None:
            vector = self.model.embed(text)
            self._cache[text] = vector
        return vector

    def cosine(self, left: str, right: str) -> float:
        if not left.strip() or not right.strip():
            return 0.0
        return round(cosine_similarity(self.embed(left), self.embed(right)), 6)


def _load_nodes(connection: sqlite3.Connection, node_ids: Iterable[str]) -> dict[str, sqlite3.Row]:
    ids = [node_id for node_id in dict.fromkeys(node_ids) if node_id]
    found: dict[str, sqlite3.Row] = {}
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
        placeholders = ",".join("?" for _ in batch)
        rows = connection.execute(
            f"""
            SELECT id, content, scope, level, agent, task, created_at, decayed
            FROM nodes WHERE id IN ({placeholders})
            """,
            batch,
        ).fetchall()
        for row in rows:
            found[str(row["id"])] = row
    return found


def _lookups_after(
    connection: sqlite3.Connection,
    node_ids: Sequence[str],
    *,
    after: str,
    transport_session_id: str | None,
) -> dict[str, list[dict[str, Any]]]:
    """Every id-fetch of one of ``node_ids`` at or after ``after``."""

    if not node_ids:
        return {}
    placeholders = ",".join("?" for _ in node_ids)
    rows = connection.execute(
        f"""
        SELECT lookup_event_id, node_id, occurred_at, transport_session_id
        FROM recall_lookup_events
        WHERE node_id IN ({placeholders}) AND occurred_at >= ?
        ORDER BY occurred_at ASC
        """,
        [*node_ids, after],
    ).fetchall()
    lookups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        node_id = str(row["node_id"])
        lookup_transport = row["transport_session_id"]
        lookups.setdefault(node_id, []).append(
            {
                "lookup_event_id": str(row["lookup_event_id"]),
                "occurred_at": str(row["occurred_at"]),
                "lag_seconds": round(_lag_seconds(after, str(row["occurred_at"])), 3),
                "same_transport": bool(
                    transport_session_id
                    and lookup_transport
                    and str(lookup_transport) == str(transport_session_id)
                ),
            }
        )
    return lookups


def build_corpus(
    connection: sqlite3.Connection,
    *,
    since: str,
    host: str,
    encoder: Encoder,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = connection.execute(
        """
        SELECT id, query, scope, requested_scope, transport_session_id, agent, task,
               feedback_trace_id, feedback_applied_at, created_at, results
        FROM recall_events
        WHERE created_at >= ? AND feedback_trace_id IS NOT NULL
        ORDER BY created_at ASC, rowid ASC
        """,
        (since,),
    ).fetchall()
    total_events = connection.execute(
        "SELECT COUNT(*) FROM recall_events WHERE created_at >= ?", (since,)
    ).fetchone()[0]

    wanted: list[str] = []
    for event in events:
        wanted.append(str(event["feedback_trace_id"]))
        for item in json.loads(event["results"] or "[]"):
            if isinstance(item, dict) and item.get("node_id"):
                wanted.append(str(item["node_id"]))
    nodes = _load_nodes(connection, wanted)

    records: list[dict[str, Any]] = []
    dropped_missing_trace = 0
    missing_result_nodes = 0
    for event in events:
        trace = nodes.get(str(event["feedback_trace_id"]))
        if trace is None:
            dropped_missing_trace += 1
            continue
        trace_content = str(trace["content"] or "")
        trace_class = script_class(trace_content)
        query = str(event["query"] or "")
        raw_results = [
            item for item in json.loads(event["results"] or "[]") if isinstance(item, dict)
        ]
        result_ids = [str(item["node_id"]) for item in raw_results if item.get("node_id")]
        lookups = _lookups_after(
            connection,
            result_ids,
            after=str(event["created_at"]),
            transport_session_id=event["transport_session_id"],
        )
        results: list[dict[str, Any]] = []
        for rank, item in enumerate(raw_results):
            node_id = str(item.get("node_id") or "")
            node = nodes.get(node_id)
            if node is None:
                missing_result_nodes += 1
                continue
            content = str(node["content"] or "")
            node_class = script_class(content)
            results.append(
                {
                    "node_id": node_id,
                    "rank": rank,
                    "level": str(node["level"] or item.get("level") or ""),
                    "scope": str(node["scope"] or ""),
                    "decayed": bool(node["decayed"]),
                    "bm25_score": float(item.get("bm25_score", 0.0) or 0.0),
                    "vector_score": float(item.get("vector_score", 0.0) or 0.0),
                    "graph_score": float(item.get("graph_score", 0.0) or 0.0),
                    "trigger_score": float(item.get("trigger_score", 0.0) or 0.0),
                    "score": float(item.get("score", 0.0) or 0.0),
                    "delivery": item.get("delivery"),
                    "content": content,
                    "script": node_class,
                    "pair_script": pair_class(node_class, trace_class),
                    "node_trace_cosine": encoder.cosine(content, trace_content),
                    "lookups": lookups.get(node_id, []),
                }
            )
        records.append(
            {
                "corpus_version": CORPUS_VERSION,
                "host": host,
                "event": {
                    "id": str(event["id"]),
                    "query": query,
                    "scope": str(event["scope"]),
                    "requested_scope": str(event["requested_scope"]),
                    "transport_session_id": event["transport_session_id"],
                    "agent": event["agent"],
                    "task": event["task"],
                    "created_at": str(event["created_at"]),
                    "feedback_applied_at": event["feedback_applied_at"],
                    "result_count": len(raw_results),
                },
                "trace": {
                    "id": str(trace["id"]),
                    "content": trace_content,
                    "chars": len(trace_content),
                    "scope": str(trace["scope"]),
                    "agent": trace["agent"],
                    "task": trace["task"],
                    "created_at": str(trace["created_at"]),
                    "script": trace_class,
                    "closure_lag_seconds": round(
                        _lag_seconds(str(event["created_at"]), str(trace["created_at"])), 3
                    ),
                    "same_scope": str(trace["scope"]) == str(event["scope"]),
                },
                "query_trace_cosine": encoder.cosine(query, trace_content),
                "results": results,
            }
        )

    summary = {
        "corpus_version": CORPUS_VERSION,
        "host": host,
        "since": since,
        "recall_events_since": int(total_events),
        "closed_events": len(events),
        "records": len(records),
        "dropped_missing_trace": dropped_missing_trace,
        "missing_result_nodes": missing_result_nodes,
        "pairs": sum(len(record["results"]) for record in records),
        "encoder": {
            "model": encoder.model.model_name,
            "backend": getattr(encoder.model, "_backend", None),
        },
    }
    return records, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", required=True, help="read-only snapshot path")
    parser.add_argument("--since", default="2026-09-04T00:00:00Z")
    parser.add_argument("--host", required=True, help="label for the source host, e.g. sfx")
    parser.add_argument("--out", required=True, help="JSONL corpus path")
    parser.add_argument(
        "--summary", default=None, help="JSON summary path (default: <out>.summary.json)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    connection = open_readonly(args.db)
    try:
        records, summary = build_corpus(
            connection, since=args.since, host=args.host, encoder=Encoder()
        )
    finally:
        connection.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary["db"] = str(Path(args.db).resolve())
    summary["out"] = str(out.resolve())
    summary_path = Path(args.summary) if args.summary else out.with_suffix(out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
