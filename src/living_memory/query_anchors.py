"""Query anchors connect remembered questions to their grounded memory sources.

An anchor's identity is its exact whitespace-normalized query and scope.
Semantic matching ranks independent anchors at retrieval time. Positive
consumptions reinforce weighted edges; negative feedback resolves the same
identity without reinforcing it.

Edges follow genuine corrections and exact compatible duplicate replacements.
Consolidation can also supply an explicit source displacement. The batch sweep
uses the same resolution for existing anchor edges and historical backfills.
Storage owns the additive anchor tables and transactional edge migration hooks.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple, Protocol
import math

from living_memory.models import Connection, ConnectionType, Node
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


#: Retrieval-only cosine floor. Similarity ranks independent query anchors;
#: it never changes their exact identities.
ANCHOR_MATCH_COSINE_THRESHOLD = 0.60

#: Maximum number of anchors returned by default.
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

#: Hard stop when walking a replacement chain. Cycles are already impossible
#: to loop on (the walk carries a visited set); this only bounds a pathological
#: chain length.
MAX_SUPERSEDES_CHAIN_DEPTH = 32

#: Consolidation exclusion reasons that name a replacement for a source.
_DIRECT_REPLACEMENT_PREFIXES = ("superseded-by:", "replaced-by:")
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
    already existed, and ``None`` when the anchor is new — which is exactly
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
) -> AnchorUpsert:
    """Create or reinforce the anchor for ``query`` in ``scope``, with edges.

    Identity is :func:`storage.recall_fingerprint` over the whitespace-collapsed
    query and scope, resolved by an indexed ``UNIQUE(scope, fingerprint)``
    probe. Case, token order, conditions and identifiers remain significant.
    Similarity is used only by :func:`match_anchors` to rank retrieval seeds.

    A reinforcing upsert keeps the stored query text, vector, and fingerprint.
    The same question in two scopes keeps independent anchors and edges.

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
) -> QueryAnchor:
    """The anchor ``query`` in ``scope`` belongs to, created if absent.

    Uses the same exact (query, scope) fingerprint as :func:`upsert_anchor`,
    so a caller keying data on the returned id lands on exactly the anchor a
    grounded consumption of the same question reinforces. Unlike
    :func:`upsert_anchor` it neither reinforces an existing anchor nor writes an edge: a negative signal
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

    Read-path callers cache the prepared matrix and re-validate it with
    :meth:`MemoryStore.query_anchor_revision`, a transactional SQLite counter
    covering membership and vector changes. Reinforcement of a live anchor
    preserves this revision. Stores without it use the uncached scan.

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
        if anchor is None or anchor.decayed:
            continue
        if scopes is not None and anchor.scope not in scopes:
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

    if _np is not None:
        query = _np.asarray(vector, dtype="<f4")
        query_norm = float(_np.linalg.norm(query))
    else:
        query_squared_norm = 0.0
        for value in vector:
            query_squared_norm += float(value) * float(value)
        query_norm = math.sqrt(query_squared_norm)
    if query_norm <= 0.0:
        return []

    prepared = getattr(store, "query_anchor_corpus", None)
    if prepared is None:
        corpus = _prepare_anchor_vectors(
            store.iter_query_anchor_vectors(scopes, dimensions=len(vector)), len(vector),
        )
        indices = slice(None)
    else:
        corpus = prepared(len(vector), scopes)
        if corpus is None:
            return []
        indices = corpus.select(scopes)
    if not corpus.ids:
        return []

    if _np is not None:
        # A zero-norm anchor has no direction to compare; scoring it 0.0 keeps
        # the numpy path agreeing with cosine_similarity instead of producing
        # nan, which would sort unpredictably.
        scores = _np.where(
            corpus.nonzero[indices],
            (corpus.matrix[indices] @ query) / (corpus.safe_norms[indices] * query_norm),
            0.0,
        )
        surviving = _np.flatnonzero(scores >= min_similarity)
        if surviving.size == 0:
            return []
        if surviving.size > limit:
            kept = surviving[
                _np.argpartition(-scores[surviving], limit - 1)[:limit]
            ]
        else:
            kept = surviving
        selected_ids = (
            range(len(corpus.ids))[indices] if isinstance(indices, slice) else indices
        )
        candidates = [
            (corpus.ids[selected_ids[index]], float(scores[index])) for index in kept
        ]
    else:
        candidates = []
        selected_ids = (
            range(len(corpus.ids))[indices] if isinstance(indices, slice) else indices
        )
        for index in selected_ids:
            row = corpus.matrix[index]
            squared_norm = corpus.safe_norms[index]
            dot = 0.0
            for left, right in zip(vector, row):
                dot += float(left) * right
            similarity = (
                0.0 if squared_norm <= 0.0 else
                dot / math.sqrt(query_squared_norm * squared_norm)
            )
            if similarity >= min_similarity:
                candidates.append((corpus.ids[index], similarity))

    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[:limit]


@dataclass(frozen=True, slots=True)
class _PreparedAnchorVectors:
    """One width's ordered corpus, with reusable norms and scope row indices."""

    ids: tuple[str, ...]
    matrix: Any
    safe_norms: Any
    nonzero: Any
    by_scope: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def reserve(cls, layout: Sequence[tuple[str, str]], width: int) -> _PreparedAnchorVectors:
        """Allocate one width's corpus; only filled scope rows may be selected."""

        ids = tuple(anchor_id for anchor_id, _scope in layout)
        by_scope: dict[str, Any] = {}
        for index, (_anchor_id, scope) in enumerate(layout):
            by_scope.setdefault(scope, []).append(index)
        if _np is not None:
            return cls(
                ids, _np.empty((len(ids), width), dtype="<f4"),
                _np.empty(len(ids), dtype="<f4"), _np.empty(len(ids), dtype=bool),
                {scope: _np.asarray(indices, dtype=_np.intp) for scope, indices in by_scope.items()},
            )
        return cls(ids, [None] * len(ids), [0.0] * len(ids), (),
                   {scope: tuple(indices) for scope, indices in by_scope.items()})

    def fill(self, prepared: _PreparedAnchorVectors) -> None:
        """Place newly decoded scopes into their reserved scan-order rows."""

        for scope, source in prepared.by_scope.items():
            target = self.by_scope[scope]
            if _np is not None:
                self.matrix[target] = prepared.matrix[source]
                self.safe_norms[target] = prepared.safe_norms[source]
                self.nonzero[target] = prepared.nonzero[source]
            else:
                for left, right in zip(target, source, strict=True):
                    self.matrix[left] = prepared.matrix[right]
                    self.safe_norms[left] = prepared.safe_norms[right]

    def select(self, scopes: Sequence[str] | None) -> Any:
        """Select query scopes in plan order, using a view when contiguous.

        Disjoint rows need a temporary gather to preserve the exact matmul
        shape. Separate scope matmuls can round differently at thresholds.
        No selected matrix is retained alongside the prepared corpus.
        """

        if scopes is None:
            return slice(None)
        groups = [
            self.by_scope[scope]
            for scope in dict.fromkeys(normalize_scope(scope) for scope in scopes)
            if scope in self.by_scope
        ]
        if not groups:
            return slice(0, 0)
        start = stop = int(groups[0][0])
        for group in groups:
            if int(group[0]) != stop or int(group[-1]) - stop + 1 != len(group):
                break
            stop += len(group)
        else:
            return slice(start, stop)
        if len(groups) == 1:
            return groups[0]
        if _np is not None:
            return _np.concatenate(groups)
        return tuple(index for group in groups for index in group)


def _prepare_anchor_vectors(
    rows: Iterable[tuple[str, str, int, Any]],
    width: int,
    *,
    index_scopes: bool = False,
) -> _PreparedAnchorVectors:
    """Prepare one width in scan order, discarding the source BLOB rows."""

    ids: list[str] = []
    blobs: list[Any] = []
    by_scope: dict[str, Any] = {}
    for anchor_id, scope, dimensions, blob in rows:
        if dimensions == width:
            if index_scopes:
                by_scope.setdefault(scope, []).append(len(ids))
            ids.append(anchor_id)
            blobs.append(blob)
    if not ids:
        return _PreparedAnchorVectors((), (), (), ())
    if _np is not None:
        matrix = _np.frombuffer(b"".join(blobs), dtype="<f4").reshape(len(ids), width)
        norms = _np.linalg.norm(matrix, axis=1)
        nonzero = norms > 0.0
        return _PreparedAnchorVectors(
            tuple(ids), matrix, _np.where(nonzero, norms, 1.0), nonzero,
            {scope: _np.asarray(indices, dtype=_np.intp) for scope, indices in by_scope.items()},
        )
    matrix = tuple(array("f", unpack_chunk_embedding(blob, width)) for blob in blobs)
    squared_norms = []
    for row in matrix:
        squared_norm = 0.0
        for value in row:
            squared_norm += value * value
        squared_norms.append(squared_norm)
    return _PreparedAnchorVectors(
        tuple(ids), matrix, tuple(squared_norms), (),
        {scope: tuple(indices) for scope, indices in by_scope.items()},
    )


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


class _ReplacementReader(Protocol):
    def get_node(self, node_id: str) -> Node | None: ...

    def list_connections(
        self, *, target_id: str, relation_type: ConnectionType,
    ) -> list[Connection]: ...

    def list_duplicate_replacements(self, node_id: str) -> list[Node]: ...


def resolve_replacement(
    store: _ReplacementReader, node_id: str, *, max_depth: int = MAX_SUPERSEDES_CHAIN_DEPTH
) -> str | None:
    """Follow corrections and exact copies to the current replacement of ``node_id``.

    An edge ``S -supersedes-> T`` means S replaces T, so the walk climbs from
    target to source and repeats: A superseded by B superseded by C resolves to
    C in one call. If no live correction exists, compatible exact-copy
    bookkeeping can supply a replacement. Only live replacements are followed.

    Returns ``None`` when nothing replaces ``node_id``, and stops (returning
    the last live node reached) on a replacement cycle or at ``max_depth``.
    A cycle cannot be resolved into a "latest" node, so the walk refuses to
    guess rather than looping.
    """

    current = str(node_id)
    seen = {current}
    for _ in range(max_depth):
        successor = _direct_replacement(store, current)
        if successor is None or successor in seen:
            break
        seen.add(successor)
        current = successor
    return None if current == str(node_id) else current


def _direct_replacement(store: _ReplacementReader, node_id: str) -> str | None:
    """The strongest live correction, otherwise an exact compatible copy."""

    edges = store.list_connections(target_id=node_id, relation_type="supersedes")
    for edge in edges:  # already ordered by weight DESC, created_at DESC
        if edge.source_id == node_id:
            continue
        source = store.get_node(edge.source_id)
        if source is not None and not source.decayed:
            return source.id
    replacements = store.list_duplicate_replacements(node_id)
    return replacements[0].id if replacements else None


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
    """Whether following replacements from ``start`` arrives at ``goal``."""

    current = str(start)
    seen = {current}
    for _ in range(MAX_SUPERSEDES_CHAIN_DEPTH):
        if current == str(goal):
            return True
        successor = _direct_replacement(store, current)
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

    * ``superseded-by:<id>`` or ``replaced-by:<id>`` — the named node is
      the replacement, with exact-copy bookkeeping included only in the latter;
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
        if text.startswith(_DIRECT_REPLACEMENT_PREFIXES):
            destination = text.partition(":")[2].strip()
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
