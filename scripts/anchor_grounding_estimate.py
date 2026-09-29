#!/usr/bin/env python3
"""Measure what query anchors recall history can actually yield — read-only.

Two questions, one script, because the second is meaningless without the first.

1. **Yield.** Grade every consumed ``recall_events`` row with the *live*
   grounding verdict (:func:`living_memory.grounding.ground_results`, per-event
   IDF — the same corpus the live write path holds) and count what a retro
   backfill would create: anchors after exact-fingerprint dedup, edges to the
   nodes that were actually used, how many of those edges point at nodes the
   decay policy has since retired, and the same figures split at candidate
   temporal cutoffs. Plus the cosine-similarity distribution between anchor
   queries inside one scope, which is what fixes the near-duplicate dedup
   threshold the anchor store will use.

2. **Reachability ceiling.** With anchors built strictly from events *before* a
   cutoff and the goldset evaluated *after* it, what share of goldset queries
   could an anchor entry point reach at all? Two arms: the grounding-filtered
   edge set that a real backfill would write, and the unfiltered "every recorded
   result is an edge" set that a parent-node probe used, so the cost of the
   filter is visible rather than argued. Each arm also reports an *oracle*
   number — could ANY eligible anchor reach a relevant node, ignoring the
   nearest-neighbour match entirely. Oracle separates "the edge does not exist"
   from "the match does not find it"; only the first is a structural ceiling.

Read-only by construction
-------------------------
The snapshot is opened ``file:...?mode=ro`` and never through ``MemoryStore``,
whose ``__init__`` migrates whatever it opens. The live database is refused
outright: freeze it first with

    python3 -m living_memory.retrieval_harness snapshot \\
        --source-db ~/.local/share/living-memory/global.sqlite3 \\
        --snapshot-out /tmp/anchor-est/snap.sqlite3

Label corpus, and why it differs from the goldset's
---------------------------------------------------
``ground_results`` builds its IDF from exactly the documents the live loop holds
(the consuming trace plus that event's results). ``build-goldset`` labels with
replay's whole-corpus IDF instead. The two agree on 96.2% of pairs at the shared
0.25 threshold (``artifacts/grounding/calibration.json``). The live view is used
here on purpose: it is the verdict the anchor write path will actually apply, so
this is an estimate of what gets built, not of what an offline judge would wish.

Usage::

    python3 scripts/anchor_grounding_estimate.py \\
        --snapshot /tmp/anchor-est/snap.sqlite3 \\
        --goldset artifacts/harness/goldset.jsonl \\
        --out-json artifacts/anchors/estimate.json \\
        --out-md artifacts/anchors/estimate.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import statistics
import subprocess
import sys
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from living_memory.grounding import (  # noqa: E402
    DEFAULT_MIN_CONTAINMENT,
    ground_results,
)
from living_memory.scope import resolve_scope  # noqa: E402
from living_memory.storage import recall_fingerprint  # noqa: E402

#: The database the MCP server writes to. Opening it here — even read-only —
#: is refused, because a stray tool change that swaps ``mode=ro`` for a
#: ``MemoryStore`` would silently migrate production.
LIVE_DB = Path("~/.local/share/living-memory/global.sqlite3").expanduser()

#: Cutoffs the acceptance run could pick. 2026-06-10 is the one the frozen
#: goldset was built at (``artifacts/harness/goldset.jsonl`` provenance).
DEFAULT_CUTOFFS = (
    "2026-06-10T00:00:00Z",
    "2026-07-01T00:00:00Z",
    "2026-07-15T00:00:00Z",
    "2026-08-01T00:00:00Z",
    "2026-08-10T00:00:00Z",
)

#: Nearest-anchor depths the ceiling is reported at.
TOP_K = (1, 3, 10)

#: Thresholds swept for near-duplicate anchor dedup.
DEDUP_THRESHOLDS = (0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99)

#: Similarity bands sampled for human inspection of the dedup decision.
EXAMPLE_BANDS = ((0.99, 1.01), (0.97, 0.99), (0.95, 0.97), (0.90, 0.95), (0.85, 0.90))

#: A token carrying a digit is an identifier (EZ-13871, 9806, NW-7). Two queries
#: that differ only in one are the same *class* of situation, not the same
#: situation — merging them under a dedup threshold loses the ticket.
IDENTIFIER_TOKEN = re.compile(r"[0-9]")

CYRILLIC = re.compile(r"[Ѐ-ӿ]")

#: Numbers the goal carried into this node, from a 600-event sample.
PRIOR_ESTIMATE = {
    "sample_events": 600,
    "grounded_share": 0.413,
    "anchors": 3760,
    "edges": 8550,
    "edges_per_anchor": 2.28,
    "decayed_edge_share": 0.046,
}

#: What the parent-node probe measured: real model, anchors = all pre-2026-06-10
#: consumed fingerprints, edges = every recorded result with NO grounding
#: filter, scope-gated.
PARENT_PROBE = {
    "setup": "all pre-cutoff consumed fingerprints; every recorded result as an edge; no grounding filter",
    "cutoff": "2026-06-10T00:00:00Z",
    "anchors": 4589,
    "content_grounded": {"top1": 0.031, "top3": 0.056, "top10": 0.113},
    "cross_lingual": {"top1": 0.000, "top3": 0.079, "top10": 0.158},
    "role_query": {"top1": 0.083, "top3": 0.111, "top10": 0.139},
}


# ---------------------------------------------------------------------------
# Snapshot access
# ---------------------------------------------------------------------------


def open_snapshot(path: Path) -> sqlite3.Connection:
    """Read-only connection to a frozen snapshot; refuses the live database."""

    resolved = path.expanduser().resolve()
    if LIVE_DB.exists() and resolved == LIVE_DB.resolve():
        raise SystemExit(
            f"refusing to open the live database {resolved}; freeze a snapshot first "
            "(python3 -m living_memory.retrieval_harness snapshot ...)"
        )
    if not resolved.exists():
        raise SystemExit(f"snapshot not found: {resolved}")
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


@dataclass(frozen=True, slots=True)
class NodeRow:
    content: str
    scope: str
    decayed: bool


@dataclass(slots=True)
class ConsumedEvent:
    """One consumed recall event with its live grounding verdict attached."""

    event_id: str
    query: str
    scope: str
    requested_scope: str
    fingerprint: str
    created_at: str
    trace_id: str
    result_ids: tuple[str, ...]
    grounded_ids: tuple[str, ...]
    resolvable: bool

    @property
    def grounded(self) -> bool:
        return bool(self.grounded_ids)


def load_nodes(connection: sqlite3.Connection) -> dict[str, NodeRow]:
    return {
        str(row["id"]): NodeRow(
            content=str(row["content"]),
            scope=str(row["scope"]),
            decayed=bool(row["decayed"]),
        )
        for row in connection.execute("SELECT id, content, scope, decayed FROM nodes")
    }


def load_consumed(
    connection: sqlite3.Connection,
    nodes: Mapping[str, NodeRow],
    *,
    min_containment: float,
) -> tuple[list[ConsumedEvent], dict[str, Any]]:
    """Grade every ``feedback_applied = 1`` event with the live grounding rule.

    An event is *unresolvable* when its consuming trace is gone from ``nodes``
    (forgotten or decayed away) — it can never be graded, so it is excluded
    from the grounded share instead of counted as a negative.
    """

    events: list[ConsumedEvent] = []
    missing_trace = 0
    missing_result_nodes = 0
    fingerprint_mismatch = 0
    rows = connection.execute(
        """
        SELECT id, query, scope, requested_scope, results, feedback_trace_id,
               created_at, fingerprint
        FROM recall_events
        WHERE feedback_applied = 1
        ORDER BY created_at, id
        """
    )
    for row in rows:
        raw_results = json.loads(row["results"] or "[]")
        result_ids = tuple(
            str(item["node_id"]) for item in raw_results if isinstance(item, dict) and item.get("node_id")
        )
        trace_id = str(row["feedback_trace_id"] or "")
        stored_fingerprint = row["fingerprint"]
        computed = recall_fingerprint(str(row["query"]), str(row["requested_scope"]))
        if stored_fingerprint and str(stored_fingerprint) != computed:
            fingerprint_mismatch += 1
        trace = nodes.get(trace_id)
        contents = {node_id: nodes[node_id].content for node_id in result_ids if node_id in nodes}
        missing_result_nodes += len(set(result_ids)) - len(contents)
        if trace is None:
            missing_trace += 1
            grounded_ids: tuple[str, ...] = ()
            resolvable = False
        else:
            resolvable = True
            verdicts = ground_results(
                trace.content, contents, min_containment=min_containment
            )
            grounded_ids = tuple(
                sorted(node_id for node_id, verdict in verdicts.items() if verdict.grounded)
            )
        events.append(
            ConsumedEvent(
                event_id=str(row["id"]),
                query=str(row["query"]),
                scope=str(row["scope"]),
                requested_scope=str(row["requested_scope"]),
                fingerprint=str(stored_fingerprint or computed),
                created_at=str(row["created_at"]),
                trace_id=trace_id,
                result_ids=result_ids,
                grounded_ids=grounded_ids,
                resolvable=resolvable,
            )
        )
    stats = {
        "consumed_events": len(events),
        "unresolvable_missing_trace_node": missing_trace,
        "missing_result_node_contents": missing_result_nodes,
        "stored_fingerprint_mismatch": fingerprint_mismatch,
    }
    return events, stats


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Anchor:
    """One anchor after exact-fingerprint dedup."""

    fingerprint: str
    query: str
    scope: str
    targets: set[str] = field(default_factory=set)
    events: int = 0
    first_seen: str = ""
    last_seen: str = ""
    scopes_seen: set[str] = field(default_factory=set)


def build_anchors(
    events: Iterable[ConsumedEvent],
    *,
    grounded_only: bool = True,
    until: str | None = None,
) -> dict[str, Anchor]:
    """Collapse events into anchors keyed by ``recall_fingerprint``.

    ``grounded_only`` is the real write path: an anchor exists only where the
    consuming trace actually used something, and its edges go only to what was
    used. With it off the function reproduces the unfiltered probe — every
    consumed fingerprint becomes an anchor and every recorded result an edge.
    ``until`` keeps events strictly before an ISO cutoff (the leak rule).
    """

    anchors: dict[str, Anchor] = {}
    for event in events:
        if until is not None and event.created_at >= until:
            continue
        targets = event.grounded_ids if grounded_only else event.result_ids
        if grounded_only and not event.grounded:
            continue
        if not targets:
            continue
        anchor = anchors.get(event.fingerprint)
        if anchor is None:
            anchor = Anchor(
                fingerprint=event.fingerprint,
                query=" ".join(event.query.split()),
                scope=event.scope,
                first_seen=event.created_at,
                last_seen=event.created_at,
            )
            anchors[event.fingerprint] = anchor
        anchor.targets.update(targets)
        anchor.events += 1
        anchor.scopes_seen.add(event.scope)
        anchor.first_seen = min(anchor.first_seen, event.created_at)
        anchor.last_seen = max(anchor.last_seen, event.created_at)
    return anchors


def anchor_figures(
    anchors: Mapping[str, Anchor], nodes: Mapping[str, NodeRow]
) -> dict[str, Any]:
    edges = sum(len(anchor.targets) for anchor in anchors.values())
    decayed = sum(
        1
        for anchor in anchors.values()
        for node_id in anchor.targets
        if nodes.get(node_id) is not None and nodes[node_id].decayed
    )
    unknown = sum(
        1
        for anchor in anchors.values()
        for node_id in anchor.targets
        if node_id not in nodes
    )
    cross_scope = sum(1 for anchor in anchors.values() if len(anchor.scopes_seen) > 1)
    return {
        "anchors": len(anchors),
        "edges": edges,
        "edges_per_anchor": _ratio(edges, len(anchors)),
        "decayed_target_edges": decayed,
        "decayed_target_share": _ratio(decayed, edges),
        "edges_to_missing_nodes": unknown,
        "anchors_with_multiple_scopes": cross_scope,
        "reinforced_anchors": sum(1 for anchor in anchors.values() if anchor.events > 1),
        "events_per_anchor": _ratio(
            sum(anchor.events for anchor in anchors.values()), len(anchors)
        ),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


# ---------------------------------------------------------------------------
# Per-scope and cutoff views
# ---------------------------------------------------------------------------


def per_scope_yield(
    events: Sequence[ConsumedEvent], nodes: Mapping[str, NodeRow]
) -> dict[str, Any]:
    by_scope: dict[str, list[ConsumedEvent]] = defaultdict(list)
    for event in events:
        by_scope[event.scope].append(event)
    out: dict[str, Any] = {}
    for scope, scope_events in by_scope.items():
        resolvable = [event for event in scope_events if event.resolvable]
        grounded = [event for event in resolvable if event.grounded]
        anchors = build_anchors(scope_events)
        figures = anchor_figures(anchors, nodes)
        out[scope] = {
            "consumed_events": len(scope_events),
            "resolvable_events": len(resolvable),
            "grounded_events": len(grounded),
            "grounded_share": _ratio(len(grounded), len(resolvable)),
            **figures,
        }
    return dict(sorted(out.items(), key=lambda item: -item[1]["consumed_events"]))


def cutoff_split(
    events: Sequence[ConsumedEvent],
    nodes: Mapping[str, NodeRow],
    cutoff: str,
) -> dict[str, Any]:
    before = [event for event in events if event.created_at < cutoff]
    after = [event for event in events if event.created_at >= cutoff]

    def side(subset: Sequence[ConsumedEvent]) -> dict[str, Any]:
        resolvable = [event for event in subset if event.resolvable]
        grounded = [event for event in resolvable if event.grounded]
        anchors = build_anchors(subset)
        unfiltered = build_anchors(subset, grounded_only=False)
        cyrillic = sum(1 for event in subset if CYRILLIC.search(event.query))
        days = _span_days(subset)
        return {
            "consumed_events": len(subset),
            "grounded_events": len(grounded),
            "grounded_share": _ratio(len(grounded), len(resolvable)),
            "cyrillic_queries": cyrillic,
            "cyrillic_share": _ratio(cyrillic, len(subset)),
            "span_days": days,
            "first_event": subset[0].created_at if subset else None,
            "last_event": subset[-1].created_at if subset else None,
            **anchor_figures(anchors, nodes),
            "unfiltered": anchor_figures(unfiltered, nodes),
        }

    return {"cutoff": cutoff, "before": side(before), "after": side(after)}


def _span_days(subset: Sequence[ConsumedEvent]) -> float:
    if not subset:
        return 0.0
    first = _parse_iso(subset[0].created_at)
    last = _parse_iso(subset[-1].created_at)
    return round((last - first).total_seconds() / 86400.0, 2)


def _parse_iso(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class Embedder:
    """Query embedder over the production model, with an L2-normalized cache.

    ``LocalEmbeddingModel.embed`` already normalizes, but it silently falls back
    to the hash backend when the sentence-transformer cannot load — and every
    similarity number here would then be meaningless. :meth:`require_real_model`
    makes that failure loud instead.
    """

    def __init__(self) -> None:
        from living_memory.embeddings import LocalEmbeddingModel

        self._model = LocalEmbeddingModel()
        self._model.warmup()
        self._cache: dict[str, Any] = {}
        self.model_name = self._model.model_name
        # The only honest signal for "the real weights loaded": the private
        # handle the fallback path leaves at None.
        self.backend = (
            "sentence-transformers" if getattr(self._model, "_model", None) is not None else "hash"
        )

    def require_real_model(self) -> None:
        if self.backend != "sentence-transformers":
            raise SystemExit(
                "the sentence-transformer did not load (backend=hash); similarity and "
                "reachability numbers would be meaningless. Unset "
                "LIVING_MEMORY_EMBEDDING_BACKEND or pass --allow-hash-backend."
            )

    def embed_many(self, texts: Sequence[str]) -> Any:
        import numpy as np

        for text in dict.fromkeys(texts):
            if text in self._cache:
                continue
            vector = np.asarray(self._model.embed(text), dtype="float32")
            norm = float(np.linalg.norm(vector))
            self._cache[text] = vector / norm if norm > 0 else vector
        if not texts:
            return np.zeros((0, self._model.dimensions), dtype="float32")
        return np.vstack([self._cache[text] for text in texts])


# ---------------------------------------------------------------------------
# Near-duplicate similarity between anchor queries inside one scope
# ---------------------------------------------------------------------------


def _identifier_tokens(query: str) -> frozenset[str]:
    return frozenset(
        token for token in re.split(r"[^\w-]+", query.lower()) if token and IDENTIFIER_TOKEN.search(token)
    )


def near_duplicate_similarity(
    anchors: Mapping[str, Anchor],
    embedder: Embedder,
    *,
    seed: int,
    floor: float = 0.80,
) -> dict[str, Any]:
    """Cosine between every pair of anchor queries that share a scope.

    Anchors never cross scope, so only same-scope pairs can ever be merged.
    Reports the distribution, the collapse curve at candidate thresholds under
    the *transitive* merge a similarity-upsert store actually performs, and how
    many merged pairs carry different identifiers — the failure mode that
    matters, because "поревьювь EZ-13871" and "поревьювь EZ-12826" are the same
    class of situation and different tickets.
    """

    import numpy as np

    by_scope: dict[str, list[Anchor]] = defaultdict(list)
    for anchor in anchors.values():
        by_scope[anchor.scope].append(anchor)

    sampled_pairs: list[float] = []
    max_per_anchor: list[float] = []
    pair_count = 0
    singleton_scopes = 0
    comparable_anchors = 0
    # Union-find over a flat anchor index; only same-scope pairs ever union.
    flat: list[Anchor] = []
    index_of: dict[tuple[str, int], int] = {}
    # (similarity, flat_left, flat_right) for every same-scope pair >= floor.
    close_pairs: list[tuple[float, int, int]] = []

    for scope, scope_anchors in sorted(by_scope.items()):
        scope_anchors.sort(key=lambda anchor: anchor.fingerprint)
        base = len(flat)
        for position, anchor in enumerate(scope_anchors):
            index_of[(scope, position)] = base + position
            flat.append(anchor)
        if len(scope_anchors) < 2:
            singleton_scopes += 1
            continue
        comparable_anchors += len(scope_anchors)
        matrix = embedder.embed_many([anchor.query for anchor in scope_anchors])
        similarity = np.asarray(matrix @ matrix.T, dtype="float64")
        np.fill_diagonal(similarity, -1.0)
        max_per_anchor.extend(float(value) for value in similarity.max(axis=1))
        rows, columns = np.triu_indices(len(scope_anchors), k=1)
        values = similarity[rows, columns]
        pair_count += int(values.size)
        # Percentiles over every pair would be hundreds of millions of floats.
        # Sample deterministically per scope instead (seed mixed with a stable
        # digest of the scope name, never the process-salted builtin hash).
        sampler = random.Random(seed ^ zlib.crc32(scope.encode("utf-8")))
        if values.size > 100_000:
            picks = sampler.sample(range(int(values.size)), 100_000)
            sampled_pairs.extend(float(values[pick]) for pick in picks)
        else:
            sampled_pairs.extend(float(value) for value in values)
        above = np.nonzero(values >= floor)[0]
        for position in above:
            close_pairs.append(
                (
                    float(values[position]),
                    base + int(rows[position]),
                    base + int(columns[position]),
                )
            )

    parent = list(range(len(flat)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    identifiers = [_identifier_tokens(anchor.query) for anchor in flat]
    curve: dict[str, Any] = {}
    for threshold in DEDUP_THRESHOLDS:
        parent[:] = range(len(flat))
        pairs = 0
        colliding = 0
        for value, left, right in close_pairs:
            if value < threshold:
                continue
            pairs += 1
            if identifiers[left] != identifiers[right]:
                colliding += 1
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_left] = root_right
        clusters = len({find(node) for node in range(len(flat))})
        curve[f"{threshold:.2f}"] = {
            "pairs_at_or_above": pairs,
            "pairs_with_differing_identifiers": colliding,
            "identifier_collision_share": _ratio(colliding, pairs),
            "clusters_after_transitive_merge": clusters,
            "anchors_absorbed": len(flat) - clusters,
            "absorbed_share": _ratio(len(flat) - clusters, len(flat)),
        }

    sampler = random.Random(seed)
    examples: dict[str, Any] = {}
    for low, high in EXAMPLE_BANDS:
        band = sorted(
            (pair for pair in close_pairs if low <= pair[0] < high),
            key=lambda pair: (-pair[0], flat[pair[1]].fingerprint, flat[pair[2]].fingerprint),
        )
        chosen = band if len(band) <= 6 else sampler.sample(band, 6)
        examples[f"{low:.2f}-{high:.2f}"] = [
            {
                "similarity": round(value, 4),
                "scope": flat[left].scope,
                "left": flat[left].query[:160],
                "right": flat[right].query[:160],
                "same_identifiers": identifiers[left] == identifiers[right],
            }
            for value, left, right in sorted(chosen, key=lambda pair: -pair[0])
        ]

    return {
        "anchors": len(flat),
        "scopes": len(by_scope),
        "scopes_with_a_single_anchor": singleton_scopes,
        "anchors_with_a_same_scope_neighbour": comparable_anchors,
        "same_scope_pairs": pair_count,
        "pairs_at_or_above_floor": len(close_pairs),
        "floor": floor,
        "sampled_pairs": len(sampled_pairs),
        "pair_similarity_percentiles": _percentiles(sampled_pairs),
        "nearest_neighbour_percentiles": _percentiles(max_per_anchor),
        "anchors_with_a_neighbour_above": {
            f"{threshold:.2f}": sum(1 for value in max_per_anchor if value >= threshold)
            for threshold in DEDUP_THRESHOLDS
        },
        "dedup_curve": curve,
        "examples": examples,
    }


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return round(ordered[index], 4)

    return {
        "min": round(ordered[0], 4),
        "p50": at(0.50),
        "p90": at(0.90),
        "p99": at(0.99),
        "p99.9": at(0.999),
        "max": round(ordered[-1], 4),
        "mean": round(statistics.fmean(ordered), 4),
    }


# ---------------------------------------------------------------------------
# Reachability ceiling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoldsetItem:
    query_id: str
    query: str
    scope: str | None
    ambient_context: Mapping[str, Any] | None
    stratum: str
    relevant: frozenset[str]
    source_event_id: str | None
    recorded_created_at: str | None


def load_goldset(path: Path, connection: sqlite3.Connection) -> list[GoldsetItem]:
    """Goldset items plus the event timestamp that binds the temporal cutoff.

    ``cross_lingual`` and ``role_query`` items are curated, not drawn from an
    event, so they carry no timestamp at all: no cutoff can bind them.
    """

    items: list[GoldsetItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        source_event_id = raw.get("source_event_id")
        created_at = None
        if source_event_id:
            row = connection.execute(
                "SELECT created_at FROM recall_events WHERE id = ?", (source_event_id,)
            ).fetchone()
            if row is not None:
                created_at = str(row["created_at"])
        if created_at is None:
            created_at = (raw.get("provenance") or {}).get("recorded_created_at")
        items.append(
            GoldsetItem(
                query_id=str(raw["query_id"]),
                query=str(raw["query"]),
                scope=raw.get("scope"),
                ambient_context=raw.get("ambient_context"),
                stratum=str(raw["stratum"]),
                relevant=frozenset(str(node_id) for node_id in raw["relevant_node_ids"]),
                source_event_id=source_event_id,
                recorded_created_at=created_at,
            )
        )
    items.sort(key=lambda item: item.query_id)
    return items


class ScopeShim:
    """Just enough of ``MemoryStore`` for ``resolve_scope``.

    The resolver reads ``store.config.default_scope`` for the deployment
    default; this satisfies it without ``MemoryStore.__init__`` and its
    migrations.
    """

    class _Config:
        default_scope = None

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.config = self._Config()


def reachability(
    items: Sequence[GoldsetItem],
    anchors: Mapping[str, Anchor],
    embedder: Embedder,
    shim: ScopeShim,
    *,
    cutoff: str,
) -> dict[str, Any]:
    """Share of goldset queries an anchor entry point could reach, per stratum.

    An item counts at depth *k* when one of the *k* nearest eligible anchors
    holds an edge to one of the item's relevant nodes. ``oracle`` drops the
    nearest-neighbour step entirely: it asks only whether such an edge exists
    anywhere in the item's scope, which is the structural ceiling the match
    quality is measured against.
    """

    import numpy as np

    anchor_list = sorted(anchors.values(), key=lambda anchor: anchor.fingerprint)
    if not anchor_list:
        return {"anchors": 0, "strata": {}}
    matrix = embedder.embed_many([anchor.query for anchor in anchor_list])
    by_scope: dict[str, list[int]] = defaultdict(list)
    for index, anchor in enumerate(anchor_list):
        by_scope[anchor.scope].append(index)

    buckets: dict[str, dict[str, Any]] = {}
    per_item: list[dict[str, Any]] = []
    for item in items:
        eligible_after_cutoff = (
            item.recorded_created_at is None or item.recorded_created_at >= cutoff
        )
        plan = resolve_scope(
            scope=item.scope,
            ambient_context=item.ambient_context,
            store=shim,
        )
        eligible: list[int] = []
        seen: set[int] = set()
        for scope in plan.scopes if plan.restricted else sorted(by_scope):
            # A plan lists each scope once today; deduplicate anyway, because a
            # repeated scope would silently give one anchor two top-k slots.
            for index in by_scope.get(scope, ()):
                if index not in seen:
                    seen.add(index)
                    eligible.append(index)
        record: dict[str, Any] = {
            "query_id": item.query_id,
            "stratum": item.stratum,
            "scopes": list(plan.scopes),
            "eligible_anchors": len(eligible),
            "after_cutoff": eligible_after_cutoff,
            "has_timestamp": item.recorded_created_at is not None,
        }
        if eligible:
            query_vector = embedder.embed_many([item.query])[0]
            similarity = matrix[eligible] @ query_vector
            order = np.argsort(-similarity)
            for depth in TOP_K:
                hit = any(
                    anchor_list[eligible[int(position)]].targets & item.relevant
                    for position in order[:depth]
                )
                record[f"top{depth}"] = hit
            record["oracle"] = any(
                anchor_list[index].targets & item.relevant for index in eligible
            )
            record["best_similarity"] = round(float(similarity.max()), 4)
        else:
            for depth in TOP_K:
                record[f"top{depth}"] = False
            record["oracle"] = False
            record["best_similarity"] = None
        per_item.append(record)

    def summarize(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        total = len(subset)
        out: dict[str, Any] = {"items": total}
        for depth in TOP_K:
            out[f"top{depth}"] = _ratio(
                sum(1 for record in subset if record[f"top{depth}"]), total
            )
        out["oracle_any_anchor"] = _ratio(
            sum(1 for record in subset if record["oracle"]), total
        )
        out["items_with_eligible_anchor"] = sum(
            1 for record in subset if record["eligible_anchors"] > 0
        )
        out["items_bound_by_cutoff"] = sum(1 for record in subset if record["has_timestamp"])
        out["items_after_cutoff"] = sum(1 for record in subset if record["after_cutoff"])
        return out

    for stratum in sorted({item.stratum for item in items}):
        subset = [record for record in per_item if record["stratum"] == stratum]
        buckets[stratum] = summarize(subset)
        leak_free = [record for record in subset if record["after_cutoff"]]
        buckets[stratum]["leak_free_subset"] = summarize(leak_free)
    buckets["overall"] = summarize(per_item)

    return {
        "anchors": len(anchor_list),
        "edges": sum(len(anchor.targets) for anchor in anchor_list),
        "strata": buckets,
        "per_item": per_item,
    }


def goldset_stability(
    shipped: Path, alternatives: Sequence[Path]
) -> dict[str, Any]:
    """How stable a goldset's labels are when it is rebuilt.

    ``content_grounded`` labels come from recorded events and node content, so
    a rebuild at the same cutoff reproduces them. ``cross_lingual`` labels are
    *the paraphrase query's own top-k on the snapshot being built from*
    (``retrieval_harness.py``, ``resolution: "paraphrase_top_k"``), so they move
    whenever retrieval or the corpus moves — including when anchors themselves
    change retrieval. This function measures that movement instead of asserting
    it: for each alternative build it counts curated items whose relevant node
    set differs from the shipped goldset's.
    """

    def load(path: Path) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                out[str(record["query_id"])] = record
        return out

    base = load(shipped)
    curated = {"cross_lingual", "role_query"}
    entries: list[dict[str, Any]] = []
    for path in alternatives:
        other = load(path)
        counts = Counter(record["stratum"] for record in other.values())
        changed: Counter[str] = Counter()
        present: Counter[str] = Counter()
        examples: list[dict[str, Any]] = []
        for query_id, record in sorted(base.items()):
            if record["stratum"] not in curated:
                continue
            present[record["stratum"]] += 1
            counterpart = other.get(query_id)
            if counterpart is None or tuple(counterpart["relevant_node_ids"]) != tuple(
                record["relevant_node_ids"]
            ):
                changed[record["stratum"]] += 1
                if len(examples) < 5:
                    examples.append(
                        {
                            "query_id": query_id,
                            "stratum": record["stratum"],
                            "query": record["query"][:120],
                            "shipped_relevant": list(record["relevant_node_ids"]),
                            "rebuilt_relevant": (
                                list(counterpart["relevant_node_ids"]) if counterpart else None
                            ),
                        }
                    )
        entries.append(
            {
                "path": str(path),
                "items": len(other),
                "per_stratum": dict(sorted(counts.items())),
                "curated_items_compared": dict(sorted(present.items())),
                "curated_items_changed": dict(sorted(changed.items())),
                "curated_change_share": {
                    stratum: _ratio(changed[stratum], present[stratum])
                    for stratum in sorted(present)
                },
                "examples": examples,
            }
        )
    return {
        "shipped": {
            "path": str(shipped),
            "items": len(base),
            "per_stratum": dict(sorted(Counter(r["stratum"] for r in base.values()).items())),
        },
        "rebuilds": entries,
    }


def headroom(
    baseline_report: Path,
    reach: Mapping[str, Any],
    *,
    depths: Sequence[int] = (1, 5),
) -> dict[str, Any]:
    """Upper bound on what anchors can add, per stratum, over a real baseline run.

    Joins an anchor-free ``retrieval_harness run`` report with the per-item
    anchor verdicts. For every item the baseline *misses* at depth k, it asks
    whether an anchor could have reached a relevant node at all:

    ``structural_ceiling`` = baseline hit rate + share of misses whose relevant
    node sits behind SOME eligible anchor edge (the oracle). It is a ceiling in
    the strongest sense: it assumes the anchor entry always surfaces that
    anchor, the injected seed always ranks into the top k, and nothing already
    hit ever regresses. ``matched_ceiling`` replaces the oracle with the
    measured top-10 nearest-anchor match, which is what the mechanism can
    actually see. A goal target above the structural ceiling is unreachable by
    anchors no matter how well the store, the matcher or the injection are
    built.
    """

    report = json.loads(baseline_report.read_text(encoding="utf-8"))
    runs = {str(run["query_id"]): run for run in report.get("runs", [])}
    records = {record["query_id"]: record for record in reach.get("per_item", [])}
    by_stratum: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for query_id, run in runs.items():
        record = records.get(query_id)
        if record is not None:
            by_stratum[str(run["stratum"])].append((run, record))

    def block(pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
        total = len(pairs)
        out: dict[str, Any] = {"items": total}
        for depth in depths:
            hits = []
            for run, record in pairs:
                relevant = set(run["relevant_node_ids"])
                ranked = list(run["ranked_node_ids"])[:depth]
                hits.append((bool(relevant & set(ranked)), record))
            baseline = sum(1 for hit, _ in hits if hit)
            missed = [record for hit, record in hits if not hit]
            oracle_rescuable = sum(1 for record in missed if record["oracle"])
            matched_rescuable = sum(1 for record in missed if record["top10"])
            out[f"depth{depth}"] = {
                "baseline_hit": _ratio(baseline, total),
                "missed": len(missed),
                "misses_anchor_reachable_oracle": oracle_rescuable,
                "misses_anchor_reachable_top10": matched_rescuable,
                "structural_ceiling": _ratio(baseline + oracle_rescuable, total),
                "matched_ceiling": _ratio(baseline + matched_rescuable, total),
            }
        return out

    out = {stratum: block(pairs) for stratum, pairs in sorted(by_stratum.items())}
    out["overall"] = block([pair for pairs in by_stratum.values() for pair in pairs])
    out["baseline_report"] = {
        "path": str(baseline_report),
        "hit@1": report.get("metrics", {}).get("overall", {}).get("hit@1"),
        "hit@5": report.get("metrics", {}).get("overall", {}).get("hit@5"),
        "mrr": report.get("metrics", {}).get("overall", {}).get("mrr"),
    }
    return out


def fingerprint_split(
    items: Sequence[GoldsetItem],
    anchors: Mapping[str, Anchor],
    reach: Mapping[str, Any],
) -> dict[str, Any]:
    """Exact-repeat items versus the fingerprint-disjoint holdout.

    An item is an *exact repeat* when its ``recall_fingerprint`` is already one
    of the pre-cutoff anchors: the anchor store has literally seen that query
    string in that scope. Everything else is the holdout that shows whether the
    mechanism generalizes rather than looks its own answer up.
    """

    known = set(anchors)
    records = {record["query_id"]: record for record in reach.get("per_item", [])}
    groups: dict[str, list[Mapping[str, Any]]] = {"exact_repeat": [], "fingerprint_disjoint": []}
    per_stratum: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: {"exact_repeat": [], "fingerprint_disjoint": []}
    )
    for item in items:
        fingerprint = recall_fingerprint(item.query, item.scope or "global")
        bucket = "exact_repeat" if fingerprint in known else "fingerprint_disjoint"
        record = records.get(item.query_id)
        if record is None:
            continue
        groups[bucket].append(record)
        per_stratum[item.stratum][bucket].append(record)

    def summarize(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        total = len(subset)
        out: dict[str, Any] = {"items": total}
        for depth in TOP_K:
            out[f"top{depth}"] = _ratio(sum(1 for r in subset if r[f"top{depth}"]), total)
        out["oracle_any_anchor"] = _ratio(sum(1 for r in subset if r["oracle"]), total)
        return out

    return {
        "overall": {bucket: summarize(records_) for bucket, records_ in groups.items()},
        "per_stratum": {
            stratum: {bucket: summarize(records_) for bucket, records_ in buckets.items()}
            for stratum, buckets in sorted(per_stratum.items())
        },
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def render_markdown(report: Mapping[str, Any]) -> str:
    history = report["history"]
    lines: list[str] = []
    add = lines.append
    add("# Query-anchor yield and reachability ceiling")
    add("")
    provenance = report["provenance"]
    add(
        f"Read-only measurement over snapshot `{provenance['snapshot_path']}` "
        f"(sha256 `{provenance['snapshot_sha256'][:16]}…`, {provenance['nodes']} nodes, "
        f"{provenance['recall_events']} recall events), embedding backend "
        f"`{provenance['embedding_backend']}`, grounding threshold "
        f"{provenance['min_containment']}."
    )
    add("")
    add("## 1. What retro grounded labeling yields")
    add("")
    figures = history["all_history"]
    add(f"- consumed events: **{history['consumed_events']}** "
        f"({history['first_event']} .. {history['last_event']})")
    add(f"- gradeable (consuming trace still present): **{history['resolvable_events']}**; "
        f"unresolvable: {history['unresolvable_missing_trace_node']}")
    add(f"- grounded events (>= 1 grounded result): **{history['grounded_events']}** "
        f"= **{history['grounded_share']:.1%}** of gradeable")
    add(f"- anchors after exact-fingerprint dedup: **{figures['anchors']}**")
    add(f"- edges: **{figures['edges']}**, **{figures['edges_per_anchor']}** per anchor")
    add(f"- edges into already-decayed targets: **{figures['decayed_target_edges']}** "
        f"= **{figures['decayed_target_share']:.1%}**")
    add(f"- anchors reinforced by more than one event: {figures['reinforced_anchors']} "
        f"({figures['events_per_anchor']} events per anchor)")
    add("")
    prior = report["prior_estimate_comparison"]
    add("### Against the goal's prior estimate (600-event sample)")
    add("")
    add("| figure | prior estimate | measured (full history) | delta |")
    add("| --- | --- | --- | --- |")
    for key, row in prior.items():
        add(f"| {key} | {row['prior']} | {row['measured']} | {row['delta']} |")
    add("")
    add("### Per scope (top scopes by consumed events)")
    add("")
    add("| scope | consumed | grounded | grounded share | anchors | edges | per anchor | decayed edges |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    # The JSON is key-sorted for reproducibility; rank by volume for reading.
    ranked = sorted(
        history["per_scope"].items(), key=lambda item: (-item[1]["consumed_events"], item[0])
    )
    for scope, row in ranked[:12]:
        add(
            f"| `{scope}` | {row['consumed_events']} | {row['grounded_events']} | "
            f"{row['grounded_share']:.1%} | {row['anchors']} | {row['edges']} | "
            f"{row['edges_per_anchor']} | {row['decayed_target_share']:.1%} |"
        )
    add("")
    add("### Split at candidate cutoffs (before = anchor training window)")
    add("")
    add("| cutoff | before: events | anchors | edges | per anchor | cyrillic | span days | after: events | anchors | cyrillic |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for entry in report["cutoff_splits"]:
        before, after = entry["before"], entry["after"]
        add(
            f"| {entry['cutoff'][:10]} | {before['consumed_events']} | {before['anchors']} | "
            f"{before['edges']} | {before['edges_per_anchor']} | "
            f"{before['cyrillic_queries']} ({before['cyrillic_share']:.1%}) | {before['span_days']} | "
            f"{after['consumed_events']} | {after['anchors']} | "
            f"{after['cyrillic_queries']} ({after['cyrillic_share']:.1%}) |"
        )
    add("")
    similarity = report["near_duplicate_similarity"]
    add("## 2. Near-duplicate anchor similarity (fixes the dedup threshold)")
    add("")
    add(
        f"{similarity['anchors']} anchors across {similarity['scopes']} scopes, "
        f"{similarity['same_scope_pairs']} same-scope pairs "
        f"({similarity['sampled_pairs']} sampled for percentiles)."
    )
    add("")
    add(f"- pair similarity: {json.dumps(similarity['pair_similarity_percentiles'])}")
    add(f"- nearest neighbour per anchor: {json.dumps(similarity['nearest_neighbour_percentiles'])}")
    add("")
    add("| threshold | pairs >= t | with differing identifiers | collision share | clusters after transitive merge | anchors absorbed |")
    add("| --- | --- | --- | --- | --- | --- |")
    for threshold, row in similarity["dedup_curve"].items():
        add(
            f"| {threshold} | {row['pairs_at_or_above']} | "
            f"{row['pairs_with_differing_identifiers']} | "
            f"{row['identifier_collision_share']:.1%} | "
            f"{row['clusters_after_transitive_merge']} | "
            f"{row['anchors_absorbed']} ({row['absorbed_share']:.1%}) |"
        )
    add("")
    for band, rows in similarity["examples"].items():
        if not rows:
            continue
        add(f"**Band {band}**")
        add("")
        for row in rows:
            mark = "same ids" if row["same_identifiers"] else "DIFFERENT ids"
            add(f"- {row['similarity']} [{mark}] `{row['left']}` ↔ `{row['right']}`")
        add("")
    add("## 3. Reachability ceiling")
    add("")
    for entry in report["ceiling"]:
        add(f"### Cutoff {entry['cutoff']}")
        add("")
        for arm in ("grounded", "unfiltered"):
            block = entry[arm]
            add(
                f"**{arm}** — anchors {block['anchors']}, edges {block['edges']}"
            )
            add("")
            add("| stratum | items | after cutoff | with eligible anchor | @1 | @3 | @10 | oracle (any anchor) |")
            add("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for stratum, row in block["strata"].items():
                add(
                    f"| {stratum} | {row['items']} | {row['items_after_cutoff']} | "
                    f"{row['items_with_eligible_anchor']} | {row['top1']:.3f} | "
                    f"{row['top3']:.3f} | {row['top10']:.3f} | {row['oracle_any_anchor']:.3f} |"
                )
                leak_free = row.get("leak_free_subset")
                if leak_free and leak_free["items"] != row["items"]:
                    add(
                        f"| ↳ {stratum} leak-free only | {leak_free['items']} | "
                        f"{leak_free['items']} | {leak_free['items_with_eligible_anchor']} | "
                        f"{leak_free['top1']:.3f} | {leak_free['top3']:.3f} | "
                        f"{leak_free['top10']:.3f} | {leak_free['oracle_any_anchor']:.3f} |"
                    )
            add("")
            leaked = sum(
                row["items"] - row["items_after_cutoff"]
                for stratum, row in block["strata"].items()
                if stratum != "overall"
            )
            if leaked:
                add(
                    f"> **{leaked} of {block['strata']['overall']['items']} items predate this "
                    "cutoff**: their own source events are inside the anchor training window, so "
                    "the un-suffixed rows above are leak-contaminated and are shown only to size "
                    "the leak. Read the `leak-free only` rows, or regenerate the goldset at this "
                    "cutoff."
                )
                add("")
        if "headroom_grounded" in entry:
            add("**Headroom over the anchor-free baseline run (grounded arm)**")
            add("")
            add(
                "`structural_ceiling` assumes every anchor-reachable miss converts to a hit "
                "and nothing regresses; `matched_ceiling` uses the measured top-10 anchor match."
            )
            add("")
            add("| stratum | depth | baseline | missed | oracle-rescuable | top10-rescuable | structural ceiling | matched ceiling |")
            add("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for stratum, block in entry["headroom_grounded"].items():
                if stratum == "baseline_report":
                    continue
                for depth_key, row in block.items():
                    if not depth_key.startswith("depth"):
                        continue
                    add(
                        f"| {stratum} | @{depth_key[5:]} | {row['baseline_hit']:.3f} | "
                        f"{row['missed']} | {row['misses_anchor_reachable_oracle']} | "
                        f"{row['misses_anchor_reachable_top10']} | "
                        f"{row['structural_ceiling']:.3f} | {row['matched_ceiling']:.3f} |"
                    )
            add("")
        split = entry["fingerprint_split"]
        add("**Exact-repeat vs fingerprint-disjoint holdout (grounded arm)**")
        add("")
        add("| subset | items | @1 | @3 | @10 | oracle |")
        add("| --- | --- | --- | --- | --- | --- |")
        for bucket, row in split["overall"].items():
            add(
                f"| {bucket} | {row['items']} | {row['top1']:.3f} | {row['top3']:.3f} | "
                f"{row['top10']:.3f} | {row['oracle_any_anchor']:.3f} |"
            )
        add("")
    add("## 4. Parent probe reproduction")
    add("")
    add("| stratum | probe @1 | probe @3 | probe @10 | reproduced @1 | @3 | @10 | grounded @1 | @3 | @10 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    baseline = next(
        (entry for entry in report["ceiling"] if entry["cutoff"] == PARENT_PROBE["cutoff"]),
        None,
    )
    if baseline is not None:
        for stratum in ("content_grounded", "cross_lingual", "role_query"):
            probe = PARENT_PROBE[stratum]
            unfiltered = baseline["unfiltered"]["strata"].get(stratum, {})
            grounded = baseline["grounded"]["strata"].get(stratum, {})
            add(
                f"| {stratum} | {probe['top1']:.3f} | {probe['top3']:.3f} | {probe['top10']:.3f} | "
                f"{unfiltered.get('top1', 0):.3f} | {unfiltered.get('top3', 0):.3f} | "
                f"{unfiltered.get('top10', 0):.3f} | {grounded.get('top1', 0):.3f} | "
                f"{grounded.get('top3', 0):.3f} | {grounded.get('top10', 0):.3f} |"
            )
    add("")
    stability = report.get("goldset_stability")
    if stability:
        add("## 5. Goldset label stability under rebuild")
        add("")
        add(
            "`content_grounded` labels come from recorded events; `cross_lingual` labels are the "
            "paraphrase query's own top-k on the snapshot being built from, so they move when "
            "retrieval moves."
        )
        add("")
        add("| rebuild | items | content_grounded | cross_lingual changed | role_query changed |")
        add("| --- | --- | --- | --- | --- |")
        for entry in stability["rebuilds"]:
            compared = entry["curated_items_compared"]
            changed = entry["curated_items_changed"]

            def cell(stratum: str) -> str:
                total = compared.get(stratum, 0)
                count = changed.get(stratum, 0)
                return f"{count} / {total} ({_ratio(count, total):.1%})"

            add(
                f"| `{Path(entry['path']).name}` | {entry['items']} | "
                f"{entry['per_stratum'].get('content_grounded', 0)} | "
                f"{cell('cross_lingual')} | {cell('role_query')} |"
            )
        add("")
    add("## 6. Goldset structure")
    add("")
    structure = report["goldset_structure"]
    add(
        f"{structure['items']} items: "
        + ", ".join(f"{stratum} {count}" for stratum, count in structure["per_stratum"].items())
    )
    add("")
    add(
        f"- items with a source event timestamp (a per-item cutoff can bind them): "
        f"**{structure['with_timestamp']}**"
    )
    add(
        f"- items with `source_event_id: null` (no cutoff binds them): "
        f"**{structure['without_timestamp']}** "
        + ", ".join(
            f"{stratum} {count}" for stratum, count in structure["without_timestamp_per_stratum"].items()
        )
    )
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--goldset", type=Path, default=Path("artifacts/harness/goldset.jsonl"))
    parser.add_argument("--out-json", type=Path, default=Path("artifacts/anchors/estimate.json"))
    parser.add_argument("--out-md", type=Path, default=Path("artifacts/anchors/estimate.md"))
    parser.add_argument("--min-containment", type=float, default=DEFAULT_MIN_CONTAINMENT)
    parser.add_argument("--cutoff", action="append", dest="cutoffs", default=None)
    parser.add_argument(
        "--ceiling-cutoff",
        action="append",
        dest="ceiling_cutoffs",
        default=None,
        help="cutoffs the reachability ceiling is measured at (default: 2026-06-10 and 2026-08-01)",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=None,
        help=(
            "anchor-free `retrieval_harness run` report over the SAME snapshot and goldset; "
            "enables the per-stratum headroom bound"
        ),
    )
    parser.add_argument(
        "--rebuilt-goldset",
        action="append",
        dest="rebuilt_goldsets",
        default=None,
        type=Path,
        help=(
            "goldset rebuilt at another cutoff, compared against --goldset to measure "
            "label stability (repeatable)"
        ),
    )
    parser.add_argument("--skip-ceiling", action="store_true")
    parser.add_argument(
        "--allow-hash-backend",
        action="store_true",
        help="proceed even when the real sentence-transformer did not load (numbers become meaningless)",
    )
    args = parser.parse_args(argv)

    cutoffs = tuple(args.cutoffs or DEFAULT_CUTOFFS)
    ceiling_cutoffs = tuple(args.ceiling_cutoffs or ("2026-06-10T00:00:00Z", "2026-08-01T00:00:00Z"))

    started = time.time()
    connection = open_snapshot(args.snapshot)
    nodes = load_nodes(connection)
    events, load_stats = load_consumed(
        connection, nodes, min_containment=args.min_containment
    )
    resolvable = [event for event in events if event.resolvable]
    grounded = [event for event in resolvable if event.grounded]
    anchors_all = build_anchors(events)
    figures_all = anchor_figures(anchors_all, nodes)

    history: dict[str, Any] = {
        **load_stats,
        "first_event": events[0].created_at if events else None,
        "last_event": events[-1].created_at if events else None,
        "resolvable_events": len(resolvable),
        "grounded_events": len(grounded),
        "grounded_share": _ratio(len(grounded), len(resolvable)),
        "grounded_share_of_all_consumed": _ratio(len(grounded), len(events)),
        "results_total": sum(len(set(event.result_ids)) for event in events),
        "grounded_results_total": sum(len(event.grounded_ids) for event in events),
        "distinct_fingerprints_all_consumed": len({event.fingerprint for event in events}),
        "distinct_fingerprints_grounded": len({event.fingerprint for event in grounded}),
        "all_history": figures_all,
        "per_scope": per_scope_yield(events, nodes),
        "cyrillic_queries": sum(1 for event in events if CYRILLIC.search(event.query)),
    }

    report: dict[str, Any] = {
        "provenance": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": git_commit(),
            "snapshot_path": str(args.snapshot),
            "snapshot_sha256": _sha256(args.snapshot),
            "nodes": len(nodes),
            "recall_events": connection.execute("SELECT COUNT(*) FROM recall_events").fetchone()[0],
            "min_containment": args.min_containment,
            "label_corpus": "per-event IDF (live write path), grounding.ground_results",
            "goldset": str(args.goldset),
            "seed": args.seed,
        },
        "history": history,
        "cutoff_splits": [cutoff_split(events, nodes, cutoff) for cutoff in cutoffs],
        "prior_estimate_comparison": _compare_prior(history, figures_all),
    }

    if not args.skip_ceiling:
        embedder = Embedder()
        if not args.allow_hash_backend:
            embedder.require_real_model()
        report["provenance"]["embedding_backend"] = embedder.backend
        report["provenance"]["embedding_model"] = embedder.model_name
        report["near_duplicate_similarity"] = near_duplicate_similarity(
            anchors_all, embedder, seed=args.seed
        )
        goldset = load_goldset(args.goldset, connection)
        shim = ScopeShim(connection)
        report["goldset_structure"] = _goldset_structure(goldset)
        if args.rebuilt_goldsets:
            report["goldset_stability"] = goldset_stability(
                args.goldset, args.rebuilt_goldsets
            )
        ceiling: list[dict[str, Any]] = []
        for cutoff in ceiling_cutoffs:
            grounded_anchors = build_anchors(events, until=cutoff)
            unfiltered_anchors = build_anchors(events, grounded_only=False, until=cutoff)
            grounded_reach = reachability(
                goldset, grounded_anchors, embedder, shim, cutoff=cutoff
            )
            unfiltered_reach = reachability(
                goldset, unfiltered_anchors, embedder, shim, cutoff=cutoff
            )
            entry = {
                "cutoff": cutoff,
                "grounded": _strip_items(grounded_reach),
                "unfiltered": _strip_items(unfiltered_reach),
                "grounded_per_item": grounded_reach["per_item"],
                "fingerprint_split": fingerprint_split(
                    goldset, grounded_anchors, grounded_reach
                ),
                "edge_strip_factor": _ratio(
                    unfiltered_reach["edges"], grounded_reach["edges"]
                ),
                "anchor_strip_factor": _ratio(
                    unfiltered_reach["anchors"], grounded_reach["anchors"]
                ),
            }
            if args.baseline_report is not None:
                entry["headroom_grounded"] = headroom(args.baseline_report, grounded_reach)
                entry["headroom_unfiltered"] = headroom(
                    args.baseline_report, unfiltered_reach
                )
            ceiling.append(entry)
        report["ceiling"] = ceiling
        report["parent_probe"] = PARENT_PROBE
    else:
        report["provenance"]["embedding_backend"] = "skipped"

    report["provenance"]["wall_seconds"] = round(time.time() - started, 1)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not args.skip_ceiling:
        args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"consumed {history['consumed_events']} events, grounded {history['grounded_events']} "
        f"({history['grounded_share']:.1%}), anchors {figures_all['anchors']}, "
        f"edges {figures_all['edges']} ({figures_all['edges_per_anchor']}/anchor), "
        f"decayed targets {figures_all['decayed_target_share']:.1%} -> {args.out_json}"
    )
    return 0


def _strip_items(reach: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in reach.items() if key != "per_item"}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _compare_prior(history: Mapping[str, Any], figures: Mapping[str, Any]) -> dict[str, Any]:
    measured = {
        "grounded_share": history["grounded_share"],
        "anchors": figures["anchors"],
        "edges": figures["edges"],
        "edges_per_anchor": figures["edges_per_anchor"],
        "decayed_edge_share": figures["decayed_target_share"],
    }
    out: dict[str, Any] = {}
    for key, value in measured.items():
        prior = PRIOR_ESTIMATE[key]
        out[key] = {
            "prior": prior,
            "measured": value,
            "delta": round(value - prior, 6),
            "ratio": _ratio(value, prior),
        }
    return out


def _goldset_structure(items: Sequence[GoldsetItem]) -> dict[str, Any]:
    per_stratum = Counter(item.stratum for item in items)
    without = Counter(item.stratum for item in items if item.recorded_created_at is None)
    return {
        "items": len(items),
        "per_stratum": dict(sorted(per_stratum.items())),
        "with_timestamp": sum(1 for item in items if item.recorded_created_at is not None),
        "without_timestamp": sum(1 for item in items if item.recorded_created_at is None),
        "without_timestamp_per_stratum": dict(sorted(without.items())),
        "earliest_timestamp": min(
            (item.recorded_created_at for item in items if item.recorded_created_at), default=None
        ),
        "latest_timestamp": max(
            (item.recorded_created_at for item in items if item.recorded_created_at), default=None
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
