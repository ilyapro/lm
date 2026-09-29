#!/usr/bin/env python3
"""Census of query-relative demotion strength (goal recall-precision, P3).

Read-only. For one host store and a set of read-side valve settings
(``LM_QUERY_IRRELEVANCE_FULL_COSINE``, ``LM_QUERY_IRRELEVANCE_MARK_WEIGHT``,
``LM_QUERY_IRRELEVANCE_FACTOR``) it computes the multiplier
``living_memory.irrelevance.query_demotions`` would have applied, over the
*applicable cases* of measurement LM 01M3MXJWTPZ2H2HJ3ZSPKZXGTE:

    event x node, where the node already has a ``query_irrelevance`` row
    created strictly before the event, on an anchor the event's query matches
    at cosine >= 0.60 (the anchor channel's floor and top-5 limit, in-scope
    anchors that existed at the event).

Reported per setting: number of cases, share with multiplier > 0.9, share
<= 0.75, quantiles; the same for the subset where the node was actually
delivered by the event. The applicable set does not depend on the setting
(closeness > 0 iff cos > floor for any full-cosine), so the rows compare.

Row weight at event time is rebuilt from the row itself, which is exact for
the rows that exist today: the first mark is ``created_at``; a second mark
(saturation, so later ones do not matter) is ``updated_at`` when
``marks >= 2`` (for ``marks > 2`` this is late, i.e. conservative). A row
cancelled to weight 0 counts as live on ``[created_at, updated_at)``. A row
cancelled and marked again (weight 0.5, ``cancels > 0``) counts as one mark
from ``created_at``. The counts of these approximated rows are reported.

The store is opened only as ``file:...?mode=ro``. ``--snapshot`` first copies
a live database through the SQLite backup API
(``retrieval_harness.backup_database``) and reads the copy.

Usage::

    PYTHONPATH=src python3 scripts/query_demotion_strength_census.py \\
        --db ~/.local/share/living-memory/global.sqlite3 --snapshot /tmp/x.sqlite3 \\
        --host sfx --setting default: --setting fc080:full=0.80 \\
        --setting fc075_mw1:full=0.75,mw=1.0 --json-out out.json
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sqlite3
import sys
from typing import Any

import numpy as np

from living_memory.irrelevance import (
    DEFAULT_QUERY_IRRELEVANCE_FACTOR,
    IRRELEVANCE_MARK_WEIGHT,
    demotion_closeness,
    effective_row_weight,
)
from living_memory.query_anchors import (
    ANCHOR_MATCH_COSINE_THRESHOLD,
    ANCHOR_MATCH_LIMIT,
)
from living_memory.scope import normalize_scope

#: Settings reported when none are given: the default and the candidates the
#: docs table carries.
DEFAULT_SETTINGS = (
    "default:",
    "fc080:full=0.80",
    "fc075:full=0.75",
    "mw1:mw=1.0",
    "fc080_mw1:full=0.80,mw=1.0",
    "fc075_mw1:full=0.75,mw=1.0",
)

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


@dataclass(frozen=True)
class Setting:
    label: str
    full_cosine: float | None = None
    mark_weight: float | None = None
    factor: float = DEFAULT_QUERY_IRRELEVANCE_FACTOR

    def multiplier(self, stored_weight: float, similarity: float) -> float:
        strength = effective_row_weight(stored_weight, self.mark_weight) * demotion_closeness(
            similarity, ANCHOR_MATCH_COSINE_THRESHOLD, self.full_cosine
        )
        return 1.0 - (1.0 - self.factor) * strength


def parse_setting(text: str) -> Setting:
    """``label:full=0.8,mw=1.0,factor=0.5``; an empty body is the default."""

    label, _, body = text.partition(":")
    values: dict[str, float] = {}
    for part in filter(None, (item.strip() for item in body.split(","))):
        key, _, raw = part.partition("=")
        values[key.strip()] = float(raw)
    unknown = set(values) - {"full", "mw", "factor"}
    if unknown:
        raise ValueError(f"unknown setting keys {sorted(unknown)} in {text!r}")
    full = values.get("full")
    if full is not None and not ANCHOR_MATCH_COSINE_THRESHOLD < full <= 1.0:
        raise ValueError(f"full must be in (floor, 1]: {text!r}")
    mark_weight = values.get("mw")
    if mark_weight is not None and not 0.0 < mark_weight <= 1.0:
        raise ValueError(f"mw must be in (0, 1]: {text!r}")
    return Setting(
        label=label.strip() or "default",
        full_cosine=full,
        mark_weight=mark_weight,
        factor=values.get("factor", DEFAULT_QUERY_IRRELEVANCE_FACTOR),
    )


@dataclass
class _Row:
    anchor_id: str
    node_id: str
    #: ``(from, until, stored_weight)`` pieces; ``until`` None is open-ended.
    pieces: list[tuple[str, str | None, float]] = field(default_factory=list)

    def weight_at(self, moment: str) -> float:
        weight = 0.0
        for start, until, value in self.pieces:
            if start < moment and (until is None or moment < until):
                weight = max(weight, value)
        return weight


def _row_pieces(row: Mapping[str, Any]) -> tuple[list[tuple[str, str | None, float]], str | None]:
    """Weight timeline of one row plus an approximation tag (None = exact)."""

    created, updated = str(row["created_at"]), str(row["updated_at"])
    marks, cancels, weight = int(row["marks"]), int(row["cancels"]), float(row["weight"])
    one = IRRELEVANCE_MARK_WEIGHT
    if cancels == 0:
        if marks <= 1:
            return [(created, None, min(1.0, one))], None
        pieces = [(created, updated, min(1.0, one)), (updated, None, min(1.0, 2 * one))]
        return pieces, ("second_mark_late" if marks > 2 else None)
    if weight <= 0.0:
        return [(created, updated, min(1.0, one * max(1, marks)))], "cancelled_window"
    return [(created, None, min(1.0, weight))], "cancelled_then_remarked"


def _connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_present(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


def default_embedder() -> Embedder:
    from living_memory.embeddings import LocalEmbeddingModel

    model = LocalEmbeddingModel()
    return lambda texts: [model.embed(text) for text in texts]


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    return {
        f"p{q}": round(float(np.percentile(array, q)), 4) for q in (10, 25, 50, 75, 90)
    }


def _summary(values: Sequence[float]) -> dict[str, Any]:
    n = len(values)
    above = sum(1 for value in values if value > 0.9)
    strong = sum(1 for value in values if value <= 0.75)
    return {
        "cases": n,
        "share_gt_0_9": round(above / n, 4) if n else None,
        "share_le_0_75": round(strong / n, 4) if n else None,
        "mean": round(float(np.mean(values)), 4) if n else None,
        **_quantiles(values),
    }


def census(
    conn: sqlite3.Connection,
    settings: Sequence[Setting],
    embedder: Embedder,
    *,
    since: str | None = None,
    until: str | None = None,
) -> dict[str, Any]:
    """Multiplier distribution per setting over the applicable cases."""

    if not _table_present(conn, "query_irrelevance") or not _table_present(conn, "query_anchors"):
        return {"applicable": 0, "settings": {s.label: _summary([]) for s in settings}}

    approximations: dict[str, int] = {}
    rows_by_anchor: dict[str, list[_Row]] = {}
    first_row = None
    for raw in conn.execute("SELECT * FROM query_irrelevance"):
        pieces, tag = _row_pieces(raw)
        if tag:
            approximations[tag] = approximations.get(tag, 0) + 1
        row = _Row(str(raw["anchor_id"]), str(raw["node_id"]), pieces)
        rows_by_anchor.setdefault(row.anchor_id, []).append(row)
        first_row = min(first_row or raw["created_at"], raw["created_at"])
    if first_row is None:
        return {"applicable": 0, "settings": {s.label: _summary([]) for s in settings}}

    anchor_ids: list[str] = []
    anchor_scope: list[str] = []
    anchor_created: list[str] = []
    blobs: list[bytes] = []
    width = None
    for raw in conn.execute(
        "SELECT id, scope, dimensions, embedding, created_at FROM query_anchors "
        "WHERE decayed = 0 ORDER BY id"
    ):
        if width is None:
            width = int(raw["dimensions"])
        if int(raw["dimensions"]) != width:
            continue
        anchor_ids.append(str(raw["id"]))
        anchor_scope.append(normalize_scope(str(raw["scope"])))
        anchor_created.append(str(raw["created_at"]))
        blobs.append(bytes(raw["embedding"]))
    if not anchor_ids:
        return {"applicable": 0, "settings": {s.label: _summary([]) for s in settings}}
    matrix = np.frombuffer(b"".join(blobs), dtype="<f4").reshape(len(anchor_ids), width)
    norms = np.linalg.norm(matrix, axis=1)
    safe = np.where(norms > 0.0, norms, 1.0)
    scope_arr = np.asarray(anchor_scope, dtype=object)
    created_arr = np.asarray(anchor_created, dtype=object)

    start = since or str(first_row)
    params: list[Any] = [start]
    clause = "created_at > ?"
    if until:
        clause += " AND created_at < ?"
        params.append(until)
    events = conn.execute(
        f"SELECT id, query, scope, resolved_scopes, results, created_at FROM recall_events "
        f"WHERE {clause} ORDER BY created_at, id",
        params,
    ).fetchall()
    queries = sorted({str(event["query"]) for event in events})
    vectors = dict(zip(queries, embedder(queries), strict=True))

    per_setting: dict[str, list[float]] = {s.label: [] for s in settings}
    delivered: dict[str, list[float]] = {s.label: [] for s in settings}
    events_with_case = 0
    for event in events:
        vector = np.asarray(vectors[str(event["query"])], dtype="<f4")
        if vector.shape[0] != width:
            continue
        query_norm = float(np.linalg.norm(vector))
        if query_norm <= 0.0:
            continue
        try:
            scopes = [normalize_scope(str(s)) for s in json.loads(event["resolved_scopes"] or "[]")]
        except ValueError:
            scopes = []
        scopes = scopes or [normalize_scope(str(event["scope"]))]
        moment = str(event["created_at"])
        eligible = np.isin(scope_arr, scopes) & (created_arr < moment)
        scores = np.where(norms > 0.0, (matrix @ vector) / (safe * query_norm), 0.0)
        surviving = np.flatnonzero(eligible & (scores >= ANCHOR_MATCH_COSINE_THRESHOLD))
        if surviving.size == 0:
            continue
        ranked = sorted(
            ((anchor_ids[i], float(scores[i])) for i in surviving), key=lambda a: (-a[1], a[0])
        )[:ANCHOR_MATCH_LIMIT]
        claims: dict[str, list[tuple[float, float]]] = {}
        for anchor_id, similarity in ranked:
            for row in rows_by_anchor.get(anchor_id, ()):
                weight = row.weight_at(moment)
                if weight > 0.0:
                    claims.setdefault(row.node_id, []).append((weight, similarity))
        try:
            delivered_ids = {
                str(item.get("node_id")) for item in json.loads(event["results"] or "[]")
            }
        except (ValueError, AttributeError):
            delivered_ids = set()
        counted = False
        for node_id, pairs in claims.items():
            # Applicable iff the default reading demotes at all (cos > floor).
            if max(demotion_closeness(sim) * w for w, sim in pairs) <= 0.0:
                continue
            counted = True
            for setting in settings:
                value = min(setting.multiplier(w, sim) for w, sim in pairs)
                per_setting[setting.label].append(value)
                if node_id in delivered_ids:
                    delivered[setting.label].append(value)
        events_with_case += int(counted)

    return {
        "since": start,
        "until": until,
        "events": len(events),
        "events_with_case": events_with_case,
        "applicable": len(per_setting[settings[0].label]) if settings else 0,
        "rows": sum(len(rows) for rows in rows_by_anchor.values()),
        "row_approximations": approximations,
        "settings": {
            s.label: {
                "full_cosine": s.full_cosine,
                "mark_weight": s.mark_weight,
                "factor": s.factor,
                "all": _summary(per_setting[s.label]),
                "delivered": _summary(delivered[s.label]),
            }
            for s in settings
        },
    }


def _format(host: str, result: Mapping[str, Any]) -> str:
    lines = [
        f"host={host} events={result.get('events')} applicable={result.get('applicable')} "
        f"rows={result.get('rows')} approx={result.get('row_approximations')}",
        f"{'setting':<14}{'cases':>7}{'>0.9':>8}{'<=0.75':>8}{'p50':>8}"
        f"{'  delivered: n':>15}{'>0.9':>8}{'<=0.75':>8}",
    ]
    for label, body in result.get("settings", {}).items():
        a, d = body.get("all", body), body.get("delivered", {})
        lines.append(
            f"{label:<14}{a.get('cases', 0):>7}{_pct(a.get('share_gt_0_9')):>8}"
            f"{_pct(a.get('share_le_0_75')):>8}{a.get('p50', float('nan')):>8.3f}"
            f"{d.get('cases', 0):>15}{_pct(d.get('share_gt_0_9')):>8}{_pct(d.get('share_le_0_75')):>8}"
        )
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.1%}"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", required=True, type=Path, help="store to read (opened mode=ro)")
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="copy --db here through the SQLite backup API first and read the copy",
    )
    parser.add_argument("--host", default="sfx", help="label for the report")
    parser.add_argument("--setting", action="append", help="label:full=..,mw=..,factor=..")
    parser.add_argument("--since", help="first event instant (default: first irrelevance row)")
    parser.add_argument("--until", help="events strictly before this instant")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    path = args.db.expanduser()
    if args.snapshot:
        from living_memory.retrieval_harness import backup_database

        backup_database(path, args.snapshot)
        path = args.snapshot
    settings = [parse_setting(text) for text in (args.setting or DEFAULT_SETTINGS)]
    conn = _connect_ro(path)
    try:
        result = census(conn, settings, default_embedder(), since=args.since, until=args.until)
    finally:
        conn.close()
    result = {"host": args.host, **result}
    print(_format(args.host, result))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
