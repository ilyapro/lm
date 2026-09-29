"""Query anchors: the memory graph's entry from query space.

Retrieval seeds its graph walk only from nodes that BM25, the vector channel,
or a schema trigger already found, so the graph has no entry of its own from
the space of *questions*. Anchors add one. An anchor is a remembered query —
the operator's own words, embedded by the same model as node content — carrying
weighted edges to the nodes that a grounded consumption of that query actually
used. A repeat of a situation is then found by "query <-> past query" plus one
hop, instead of by "query <-> node content".

Why that direction works is measured, not assumed (2026-08-18, live system):
one operator's queries share a language and a jargon, so
«поревьювь EZ-13871» ↔ «поревьювь EZ-12826» sits at 0.931 and
«посмотри мердж-реквест 9806» ↔ «глянь мердж-реквест 9443» at 0.916, while the
same queries against the English content of the node they should find sit at
0.100–0.269. «реквест» meets «реквест»; nothing has to be translated.

What lives here and what does not
---------------------------------
This module owns anchor *policy*: which anchor an incoming query reinforces,
how much one grounded consumption is worth, which anchors a query matches, when
an anchor has gone stale, and where an anchor edge goes when its target is
replaced. ``storage.py`` owns only the v7 migration and thin table access.

Anchors are not nodes and anchor edges are not connections. Both existing
tables carry baked-in CHECK constraints — ``nodes.level`` and
``connections.type`` — on a 500 MB live database, so a new level or edge type
would mean a table rebuild rather than an additive migration. Two new tables
cost one ``CREATE`` each and leave the existing DDL byte-identical. The
bipartite split is also what makes an anchor edge structurally incapable of
being a self-loop or of closing a cycle.

Edge migration
--------------
Nothing in this repository rewires edges when a node is superseded: the node is
soft-deleted and retrieval treats inbound edges as dead ends. That is tolerable
for node-to-node edges, which are read as evidence; it is not tolerable for an
anchor edge, which is read as an *answer* — a stale one would keep steering a
repeated question into the past. Observed live: a schema collapsing 5 sources
into 1. So anchor edges follow their target, through both routes a target can
be replaced:

* an inbound active ``supersedes`` edge — hooked in
  ``MemoryStore._insert_connection``, so ``memory_teach``, duplicate-content
  dedup, and ``edge_derivation.rule_r1c_content_correction`` are all covered in
  the same transaction that writes the edge;
* era displacement by consolidation — no edge exists to hook, so
  :func:`migrate_anchor_edges_for_exclusions` takes
  ``consolidation._assess_cluster_eras``'s exclusion mapping directly.

:func:`sweep_anchor_edge_migration` is the batch form of both, for the
retro-backfill and for repairing a database written before the hook existed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple
import math

from living_memory.embeddings import cosine_similarity
from living_memory.scope import ScopePlan, normalize_scope
from living_memory.storage import (
    MemoryStore,
    QueryAnchor,
    QueryAnchorEdge,
    recall_fingerprint,
    unpack_chunk_embedding,
)
from living_memory.temporal import parse_timestamp

try:
    import numpy as _np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - numpy is a normal runtime dep
    _np = None  # type: ignore[assignment]


#: Cosine at or above which an incoming query reinforces an existing in-scope
#: anchor instead of creating a new one.
#:
#: Deliberately *high*, and the reason is the same measurement that motivates
#: anchors at all: two genuinely different situations phrased in the operator's
#: house style score 0.916–0.931 («поревьювь EZ-13871» vs «поревьювь
#: EZ-12826»; «мердж-реквест 9806» vs «мердж-реквест 9443»). Those must stay
#: two anchors — each one's edges point at the nodes that *its* ticket needed,
#: and merging them would blur both into a generic "review something" anchor
#: whose edges answer neither. A dedup cut above that band merges only trivial
#: rewordings, while retrieval still reaches the sibling anchor at match time,
#: where the threshold is lower on purpose.
#:
#: This is a documented default, not a measured constant. The empirical value
#: is due from ``artifacts/anchors/estimate.json``; when it lands, change this
#: number. ``tests/test_query_anchors.py`` pins the *behaviour* — that a pair
#: at or above the constant merges and a pair below it does not — by reading
#: the constant, so retuning it does not require editing a test.
ANCHOR_DEDUP_COSINE_THRESHOLD = 0.95

#: Default floor for :func:`match_anchors`. **Measured, not chosen.**
#:
#: 0.80 shipped as a documented default picked to sit under the 0.95 dedup
#: constant, and the scored holdout found it admits almost nothing: 13 of 343
#: queries, 0 of 38 ``cross_lingual``, 1 of 36 ``role_query`` (``result.md``
#: §5). ``scripts/anchor_match_calibration.py`` then swept the floor on a
#: calibration set built strictly *inside* that run's anchor training window --
#: anchors from events before 2026-06-10, queries from grounded consumptions in
#: [2026-06-10, 2026-07-15), so the calibration queries are the events that
#: BUILD the scored corpus and are disjoint from the scored items by
#: construction. 234 items, 1,607 anchors, 36 cells from no floor at all to
#: 0.90 (``artifacts/anchors/calibration.json``).
#:
#: What the sweep found is that the floor is **not** what bounds retrieval
#: quality. hit@1, hit@5 and hit@10 come back byte-identical to the anchors-off
#: arm in *every one* of the 36 cells -- including at no floor, where all 234
#: queries match. Zero items improved anywhere; the only rank changes below
#: 0.55 are regressions. The funnel names why: at maximum firing only 23 of 234
#: matched anchors carry an edge to a relevant node, and of the 8 items whose
#: relevant node was both seeded and missing from the anchors-off top 5, none
#: was rescued. The leak is anchor *edge coverage*, not the match floor.
#:
#: So the floor is set by coverage against cost, with quality flat across the
#: range. 0.60 is the widest floor *certified* inside the hard +5 ms p50 budget.
#: The paired delta is heavy-tailed, so cost was measured twice: 0.60 came back
#: at +2.07 and +2.51 ms, inside on both replicates, while 0.55 gave +8.31 and
#: +4.17 ms -- opposite sides of the budget, so it cannot be certified -- and no
#: floor at all costs +25.93 ms. 0.60 lifts coverage 23x over 0.80 (1.3% ->
#: 29.5%), which is what gives the scored run the power ``result.md`` §5 said it
#: lacked, at 0 items improved and 0 regressed.
#:
#: Retrieval passes its own value; this is the default for callers that do not
#: care. Retuning this against the scored goldset is what the calibration
#: exists to prevent -- re-measure on a fresh in-window split instead.
ANCHOR_MATCH_COSINE_THRESHOLD = 0.60

#: Default number of anchors :func:`match_anchors` returns. Swept over {3, 5,
#: 10} beside the floor and left where it shipped: at 0.60 a limit of 3 buys
#: identical coverage at lower precision (0.128 vs 0.133 anchor-precision), and
#: 10 introduces a regression for no gain. Limit does not move coverage at all
#: -- coverage asks whether *any* anchor clears the floor, which the first one
#: already answers.
ANCHOR_MATCH_LIMIT = 5

#: Weight one grounded consumption contributes to an anchor -> node edge.
#: Four independent consumptions of the same (query, node) pair saturate it at
#: 1.0, so a link the operator's work keeps re-confirming reaches full strength
#: while a single accident stays weak.
ANCHOR_EDGE_WEIGHT = 0.25

#: Days without a match after which an anchor decays. Half of
#: ``MemoryConfig.trace_ttl_days`` (180): an anchor is a claim about *current*
#: working habits, which go stale faster than the facts they retrieve, and its
#: decay is soft — the edges survive and a later match revives the anchor.
ANCHOR_TTL_DAYS = 90

ANCHOR_TTL_DECAY_REASON = "anchor_ttl_expired"

#: Hard stop when walking a ``supersedes`` chain. Cycles are already impossible
#: to loop on (the walk carries a visited set); this only bounds a pathological
#: chain length.
MAX_SUPERSEDES_CHAIN_DEPTH = 32

#: Exclusion-reason prefixes ``consolidation._assess_cluster_eras`` emits that
#: name the node that replaced the excluded member.
_SUPERSEDED_BY_PREFIX = "superseded-by:"
_DISPLACED_VIA_PREFIX = "displaced-via:"


class AnchorMatch(NamedTuple):
    """One matched anchor: ``(anchor, similarity, [(target_id, weight), ...])``.

    A tuple rather than a dataclass because the retrieval caller destructures
    it, and its third element is already the shape a graph seed list wants.
    ``targets`` is ordered heaviest-first and holds live targets only.
    """

    anchor: QueryAnchor
    similarity: float
    targets: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class AnchorUpsert:
    """Outcome of one :func:`upsert_anchor` call.

    ``matched_by`` is ``"fingerprint"`` when the exact (query, scope) identity
    already existed, ``"cosine"`` when a near-identical in-scope anchor
    absorbed it, and ``None`` when the anchor is new — which is also exactly
    when ``created`` is true.
    """

    anchor: QueryAnchor
    created: bool
    matched_by: str | None
    similarity: float
    edges: tuple[QueryAnchorEdge, ...]
    skipped_targets: tuple[str, ...]

    @property
    def reinforced(self) -> bool:
        return not self.created


@dataclass(frozen=True)
class AnchorEdgeSweep:
    """What one :func:`sweep_anchor_edge_migration` pass moved."""

    targets_examined: int
    targets_migrated: int
    edges_moved: int
    #: ``old_target -> new_target`` for every re-point performed.
    migrations: tuple[tuple[str, str], ...]


# ----------------------------------------------------------------------
# Anchor upsert
# ----------------------------------------------------------------------


def upsert_anchor(
    store: MemoryStore,
    query: str,
    scope: str,
    embedding: Sequence[float],
    targets: Iterable[Any] = (),
    now: str | datetime | None = None,
    *,
    edge_weight: float = ANCHOR_EDGE_WEIGHT,
    dedup_threshold: float = ANCHOR_DEDUP_COSINE_THRESHOLD,
) -> AnchorUpsert:
    """Create or reinforce the anchor for ``query`` in ``scope``, with edges.

    Dedup runs in two stages, cheapest first:

    1. exact identity — :func:`storage.recall_fingerprint` over
       (whitespace-collapsed query, scope), the same key ``recall_events``
       already stamps, resolved by an indexed ``UNIQUE(scope, fingerprint)``
       probe;
    2. near-identity — cosine at or above ``dedup_threshold`` against the live
       in-scope anchors, highest first.

    An anchor never crosses scope: the fingerprint includes the scope and the
    cosine scan is scoped, so the same question asked in two scopes stays two
    anchors with two sets of edges.

    A reinforcing upsert keeps the stored query text, vector, and fingerprint of
    the anchor it matched. Rewriting them to the newest phrasing would let an
    anchor drift across a long tail of near-misses until it no longer matches
    the question it was built from.

    ``targets`` accepts node ids, ``(node_id, weight)`` pairs, or mappings with
    a ``node_id``/``id`` key and an optional ``weight``. Targets are resolved
    through :func:`resolve_replacement` before the edge is written, so a fresh
    anchor cannot be born pointing at a node that has already been replaced;
    ones that resolve to nothing live are reported in ``skipped_targets``
    rather than written, which keeps the edge table free of dead ends instead
    of relying on readers to filter them.
    """

    anchor_scope = normalize_scope(scope)
    stamp = _timestamp(now)
    vector = [float(value) for value in embedding]
    if not vector:
        raise ValueError("anchor embedding must not be empty")

    fingerprint = recall_fingerprint(query, anchor_scope)
    matched_by: str | None = None
    similarity = 0.0

    anchor = store.find_query_anchor(anchor_scope, fingerprint)
    if anchor is not None:
        matched_by = "fingerprint"
        similarity = 1.0
    else:
        near = _nearest_anchor(store, vector, anchor_scope, dedup_threshold)
        if near is not None:
            anchor, similarity = near
            matched_by = "cosine"

    if anchor is None:
        anchor = store.insert_query_anchor(
            scope=anchor_scope,
            query=query,
            fingerprint=fingerprint,
            embedding=vector,
            now=stamp,
        )
        created = True
    else:
        reinforced = store.reinforce_query_anchor(anchor.id, now=stamp)
        anchor = reinforced if reinforced is not None else anchor
        created = False

    edges, skipped = _write_anchor_edges(
        store, anchor.id, targets, weight=edge_weight, now=stamp
    )
    return AnchorUpsert(
        anchor=anchor,
        created=created,
        matched_by=matched_by,
        similarity=similarity,
        edges=edges,
        skipped_targets=skipped,
    )


def _write_anchor_edges(
    store: MemoryStore,
    anchor_id: str,
    targets: Iterable[Any],
    *,
    weight: float,
    now: str,
) -> tuple[tuple[QueryAnchorEdge, ...], tuple[str, ...]]:
    written: dict[str, QueryAnchorEdge] = {}
    skipped: list[str] = []
    for raw in targets:
        target_id, target_weight = _normalize_target(raw, weight)
        if not target_id:
            continue
        resolved = _live_target(store, target_id)
        if resolved is None:
            skipped.append(target_id)
            continue
        written[resolved] = store.upsert_query_anchor_edge(
            anchor_id, resolved, weight=target_weight, now=now
        )
    return tuple(written.values()), tuple(skipped)


def _normalize_target(raw: Any, default_weight: float) -> tuple[str, float]:
    if isinstance(raw, Mapping):
        node_id = raw.get("node_id") or raw.get("id") or raw.get("target_id")
        weight = raw.get("weight", default_weight)
        return ("" if node_id is None else str(node_id)), float(weight)
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        return str(raw[0]), float(raw[1])
    return str(raw), float(default_weight)


def resolve_query_anchor(
    store: MemoryStore,
    query: str,
    scope: str,
    embedding: Sequence[float],
    now: str | datetime | None = None,
    *,
    dedup_threshold: float = ANCHOR_DEDUP_COSINE_THRESHOLD,
) -> QueryAnchor:
    """The anchor ``query`` in ``scope`` belongs to, created if absent.

    The same two-stage identity :func:`upsert_anchor` uses — fingerprint, then
    in-scope cosine at or above ``dedup_threshold`` — so a caller keying data
    on the returned id lands on exactly the anchor a grounded consumption of
    the same question reinforces. Unlike :func:`upsert_anchor` it neither
    reinforces an existing anchor nor writes an edge: a negative signal
    (``living_memory.irrelevance``) must not count as a use of the anchor.
    A new anchor is born with no edges, so it seeds nothing.
    """

    anchor_scope = normalize_scope(scope)
    vector = [float(value) for value in embedding]
    if not vector:
        raise ValueError("anchor embedding must not be empty")
    fingerprint = recall_fingerprint(query, anchor_scope)
    anchor = store.find_query_anchor(anchor_scope, fingerprint)
    if anchor is not None:
        return anchor
    near = _nearest_anchor(store, vector, anchor_scope, dedup_threshold)
    if near is not None:
        return near[0]
    return store.insert_query_anchor(
        scope=anchor_scope,
        query=query,
        fingerprint=fingerprint,
        embedding=vector,
        now=_timestamp(now),
    )


def _live_target(store: MemoryStore, node_id: str) -> str | None:
    """The live node an edge to ``node_id`` should actually point at."""

    replacement = resolve_replacement(store, node_id)
    candidate = replacement or node_id
    node = store.get_node(candidate)
    if node is None or node.decayed:
        return None
    return candidate


# ----------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------


def match_anchors(
    store: MemoryStore,
    query_embedding: Sequence[float],
    scope_plan: ScopePlan | Sequence[str] | str | None = None,
    limit: int = ANCHOR_MATCH_LIMIT,
    *,
    min_similarity: float = ANCHOR_MATCH_COSINE_THRESHOLD,
    include_targets: bool = True,
) -> list[AnchorMatch]:
    """Match a query vector against the live anchors of the planned scopes.

    A full scan, not an index: an anchor corpus is one row per distinct
    remembered question — a few thousand — against tens of thousands of node
    chunks, so the whole thing is one small matmul. Vectors are compared only
    against anchors of the *same* recorded width; differing widths are points in
    different spaces (a corpus caught mid-re-embed, a fixture with 3-d vectors)
    and are skipped rather than silently truncated to a common prefix.

    **Cost, measured at the projected corpus size** (4,000 anchors x 384 dims,
    on-disk SQLite, this machine): ~12 ms p50 cold, of which ~6 ms is pulling
    4,000 BLOBs through the sqlite3 driver and the rest is streaming a fresh
    6.1 MB matrix past numpy twice. That is affordable on the write and
    backfill paths, and it is *not* affordable once per recall against a
    single-digit-millisecond latency budget. A caller on the read path must
    cache the matrix and re-validate it with
    :meth:`MemoryStore.query_anchor_revision` (~0.04 ms), the same shape
    ``retrieval._ScopeChunkIndex`` already uses for the chunk corpus. That
    revision is complete precisely because a reinforcement never rewrites a
    stored vector.

    Ranked by descending similarity, ties broken by anchor id so the order is
    stable. ``targets`` holds live targets only, heaviest first; an anchor whose
    every target has decayed still matches, with an empty target list, so a
    caller can tell "no anchor" from "anchor with nothing live to offer".

    Returns an empty list on a pre-v7 database, on an empty query vector, and
    when no anchor clears ``min_similarity``.
    """

    vector = [float(value) for value in query_embedding]
    if not vector or limit <= 0:
        return []
    scopes = _scope_list(scope_plan)
    if scopes is not None and not scopes:
        return []

    ranked = _rank_anchors(
        store, vector, scopes, min_similarity=min_similarity, limit=limit
    )

    matches: list[AnchorMatch] = []
    for anchor_id, similarity in ranked:
        anchor = store.get_query_anchor(anchor_id)
        if anchor is None:  # pragma: no cover - row vanished mid-scan
            continue
        targets: tuple[tuple[str, float], ...] = ()
        if include_targets:
            targets = tuple(
                (edge.target_id, edge.weight)
                for edge in store.list_query_anchor_edges(
                    anchor_id=anchor_id, active_targets_only=True
                )
            )
        matches.append(AnchorMatch(anchor=anchor, similarity=similarity, targets=targets))
        if len(matches) >= limit:
            break
    return matches


def _rank_anchors(
    store: MemoryStore,
    vector: Sequence[float],
    scopes: Sequence[str] | None,
    *,
    min_similarity: float,
    limit: int,
) -> list[tuple[str, float]]:
    """Top ``limit`` live anchors at or above ``min_similarity``, best first.

    Thresholding and truncation happen *inside* the vectorized step on purpose.
    Materializing one Python float and one tuple per anchor and then sorting
    the lot costs about four times the scan it decorates (measured at 4,000
    anchors x 384 dims: 19.8 ms that way against 4.7 ms this way), and the
    result is thrown away a line later — a caller wants five anchors, not four
    thousand scores. Only the survivors are converted.

    Ties break by anchor id so the order is stable across runs; that comparison
    only ever runs over the handful of survivors.
    """

    width = len(vector)
    ids: list[str] = []
    blobs: list[memoryview] = []
    for anchor_id, _scope, dimensions, blob in store.iter_query_anchor_vectors(scopes):
        if dimensions != width:
            continue
        ids.append(anchor_id)
        blobs.append(blob)
    if not ids:
        return []

    if _np is not None:
        query = _np.asarray(vector, dtype="<f4")
        query_norm = float(_np.linalg.norm(query))
        if query_norm <= 0.0:
            return []
        matrix = _np.frombuffer(b"".join(blobs), dtype="<f4").reshape(len(ids), width)
        norms = _np.linalg.norm(matrix, axis=1)
        # A zero-norm anchor has no direction to compare; scoring it 0.0 keeps
        # the numpy path agreeing with cosine_similarity instead of producing
        # nan, which would sort unpredictably.
        safe = _np.where(norms > 0.0, norms, 1.0)
        scores = _np.where(norms > 0.0, (matrix @ query) / (safe * query_norm), 0.0)
        surviving = _np.flatnonzero(scores >= min_similarity)
        if surviving.size == 0:
            return []
        if surviving.size > limit:
            kept = surviving[
                _np.argpartition(-scores[surviving], limit - 1)[:limit]
            ]
        else:
            kept = surviving
        candidates = [(ids[index], float(scores[index])) for index in kept]
    else:
        query_norm = math.sqrt(sum(value * value for value in vector))
        if query_norm <= 0.0:
            return []
        candidates = []
        for anchor_id, blob in zip(ids, blobs):
            similarity = cosine_similarity(vector, unpack_chunk_embedding(blob, width))
            if similarity >= min_similarity:
                candidates.append((anchor_id, similarity))

    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[:limit]


def _nearest_anchor(
    store: MemoryStore, vector: Sequence[float], scope: str, threshold: float
) -> tuple[QueryAnchor, float] | None:
    """Best live in-scope anchor at or above ``threshold``, if any."""

    ranked = _rank_anchors(
        store, vector, [scope], min_similarity=threshold, limit=1
    )
    if not ranked:
        return None
    anchor_id, similarity = ranked[0]
    anchor = store.get_query_anchor(anchor_id)
    return None if anchor is None else (anchor, similarity)


def _scope_list(
    scope_plan: ScopePlan | Sequence[str] | str | None,
) -> list[str] | None:
    if scope_plan is None:
        return None
    if isinstance(scope_plan, ScopePlan):
        return list(scope_plan.scopes) if scope_plan.restricted else None
    if isinstance(scope_plan, str):
        return [normalize_scope(scope_plan)]
    return [normalize_scope(str(scope)) for scope in scope_plan]


# ----------------------------------------------------------------------
# Decay
# ----------------------------------------------------------------------


def decay_stale_anchors(
    store: MemoryStore,
    *,
    now: str | datetime | None = None,
    ttl_days: int = ANCHOR_TTL_DAYS,
    scope: str | None = None,
) -> list[QueryAnchor]:
    """Soft-delete anchors whose last match has aged past ``ttl_days``.

    Age is measured from ``last_matched_at``, falling back to ``first_seen``
    for an anchor that has never been matched since it was written. Soft, like
    every other deletion in this store: the row and its edges stay, so a later
    :func:`upsert_anchor` on the same question revives the anchor with its
    learned edges intact rather than relearning them from zero.
    """

    if ttl_days < 0:
        raise ValueError("ttl_days must be non-negative")
    moment = _moment(now)
    cutoff = moment - timedelta(days=int(ttl_days))
    stamp = _iso(moment)

    decayed: list[QueryAnchor] = []
    for anchor in store.list_query_anchors(scope=scope, include_decayed=False):
        reference = _parse(anchor.last_matched_at) or _parse(anchor.first_seen)
        if reference is None or reference >= cutoff:
            continue
        updated = store.decay_query_anchor(
            anchor.id, ANCHOR_TTL_DECAY_REASON, now=stamp
        )
        if updated is not None:
            decayed.append(updated)
    return decayed


# ----------------------------------------------------------------------
# Edge migration
# ----------------------------------------------------------------------


def resolve_replacement(
    store: MemoryStore, node_id: str, *, max_depth: int = MAX_SUPERSEDES_CHAIN_DEPTH
) -> str | None:
    """Follow ``supersedes`` upward to the node that currently replaces ``node_id``.

    An edge ``S -supersedes-> T`` means S replaces T, so the walk climbs from
    target to source and repeats: A superseded by B superseded by C resolves to
    C in one call. Only live superseders are followed — a replacement that has
    itself been forgotten is not an improvement on what it replaced.

    Returns ``None`` when nothing supersedes ``node_id``, and stops (returning
    the last live node reached) on a ``supersedes`` cycle or at ``max_depth``.
    A cycle cannot be resolved into a "latest" node, so the walk refuses to
    guess rather than looping.
    """

    current = str(node_id)
    seen = {current}
    for _ in range(max_depth):
        successor = _direct_superseder(store, current)
        if successor is None or successor in seen:
            break
        seen.add(successor)
        current = successor
    return None if current == str(node_id) else current


def _direct_superseder(store: MemoryStore, node_id: str) -> str | None:
    """The live node that most strongly supersedes ``node_id``, if any."""

    edges = store.list_connections(target_id=node_id, relation_type="supersedes")
    for edge in edges:  # already ordered by weight DESC, created_at DESC
        if edge.source_id == node_id:
            continue
        source = store.get_node(edge.source_id)
        if source is not None and not source.decayed:
            return source.id
    return None


def migrate_anchor_edges(
    store: MemoryStore,
    from_target: str,
    to_target: str,
    *,
    now: str | datetime | None = None,
) -> int:
    """Re-point every anchor edge on ``from_target`` onto ``to_target``.

    The explicit entry point, for callers that know the replacement without
    having written a ``supersedes`` edge — era displacement, backfills, repair.
    The write paths that *do* write such an edge are covered automatically by
    ``MemoryStore._insert_connection``.

    Refuses a no-op or unsafe move rather than performing it: identical
    endpoints, a missing or decayed destination, and a destination that
    ``supersedes``-resolves back to the source (which would bounce the edges
    between two nodes forever) all return 0. Weights merge on collision — the
    contract of :meth:`MemoryStore.repoint_query_anchor_edges` — so an anchor
    already pointing at the replacement keeps both claims.

    Returns the number of edge rows moved.
    """

    source = str(from_target)
    destination = str(to_target)
    if source == destination:
        return 0
    node = store.get_node(destination)
    if node is None or node.decayed:
        return 0
    if _reaches(store, destination, source):
        return 0
    with store.connection:
        return store.repoint_query_anchor_edges(source, destination, now=_timestamp(now))


def _reaches(store: MemoryStore, start: str, goal: str) -> bool:
    """Whether climbing ``supersedes`` from ``start`` arrives at ``goal``."""

    current = str(start)
    seen = {current}
    for _ in range(MAX_SUPERSEDES_CHAIN_DEPTH):
        if current == str(goal):
            return True
        successor = _direct_superseder(store, current)
        if successor is None or successor in seen:
            return False
        seen.add(successor)
        current = successor
    return current == str(goal)


def migrate_anchor_edges_for_exclusions(
    store: MemoryStore,
    excluded: Mapping[str, str],
    *,
    now: str | datetime | None = None,
) -> int:
    """Follow era displacement, given ``_EraAssessment.excluded`` verbatim.

    Consolidation drops era-displaced members from derivation without writing
    any edge, so there is nothing for the ``supersedes`` hook to catch. This
    takes its exclusion mapping — ``{node_id: reason}`` — and moves the anchor
    edges of the members whose reason names a replacement:

    * ``superseded-by:<id>`` — the named node is the replacement;
    * ``displaced-via:<id>`` — the member was folded into a consolidated node
      that has since been superseded, so the replacement is whatever now
      replaces that consolidated node.

    ``corrected`` and ``obsolete-era`` are deliberately skipped: both mark a
    member as no longer speaking for the current era without naming anything
    that speaks for it instead, and inventing a destination would be worse than
    leaving the edge where a later supersedes or sweep can find it.

    Returns the number of edge rows moved.
    """

    stamp = _timestamp(now)
    moved = 0
    for node_id, reason in excluded.items():
        text = str(reason)
        if text.startswith(_SUPERSEDED_BY_PREFIX):
            destination = text[len(_SUPERSEDED_BY_PREFIX) :].strip()
        elif text.startswith(_DISPLACED_VIA_PREFIX):
            via = text[len(_DISPLACED_VIA_PREFIX) :].strip()
            destination = resolve_replacement(store, via) or ""
        else:
            continue
        if not destination:
            continue
        moved += migrate_anchor_edges(store, str(node_id), destination, now=stamp)
    return moved


def sweep_anchor_edge_migration(
    store: MemoryStore, *, now: str | datetime | None = None
) -> AnchorEdgeSweep:
    """Re-point every anchor edge whose target has since been replaced.

    The batch form, for the retro-backfill (which writes edges from historical
    consumptions, some of whose targets were superseded long ago) and for
    repairing a database written before the write-path hook existed.
    Idempotent: a second pass over a swept database moves nothing, because
    after the first pass no anchor edge has a live superseder.

    Drives off the distinct targets actually present in the edge table, so its
    cost scales with the anchor graph rather than with the node corpus.
    """

    stamp = _timestamp(now)
    targets = store.list_anchor_edge_targets()
    migrations: list[tuple[str, str]] = []
    edges_moved = 0
    for target_id in targets:
        replacement = resolve_replacement(store, target_id)
        if replacement is None:
            continue
        moved = migrate_anchor_edges(store, target_id, replacement, now=stamp)
        if moved:
            migrations.append((target_id, replacement))
            edges_moved += moved
    return AnchorEdgeSweep(
        targets_examined=len(targets),
        targets_migrated=len(migrations),
        edges_moved=edges_moved,
        migrations=tuple(migrations),
    )


# ----------------------------------------------------------------------
# Time helpers
# ----------------------------------------------------------------------


def _moment(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        parsed = _parse(value)
        if parsed is not None:
            return parsed
    return datetime.now(UTC)


def _timestamp(value: str | datetime | None) -> str:
    if isinstance(value, str) and value:
        return value
    return _iso(_moment(value))


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse(value: str | None) -> datetime | None:
    return parse_timestamp(value)
