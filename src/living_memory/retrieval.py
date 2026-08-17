"""Recall service combining scope, FTS5, embeddings, graph traversal, and feedback.

The vector channel
==================

A node's vector score is the **maximum cosine over its chunk vectors**, read as
float32 BLOBs from ``node_chunk_embeddings`` (schema v6) and kept as a cached
matrix per scope. The encoder is configured with ``max_seq_length = 128``, so a
single whole-node vector only ever saw the node's first window; max-pooling
over windows is what makes the rest of a node findable at all.

Length bias, measured
---------------------

Max over ``k`` draws is stochastically greater than one draw, so a node with
more windows gets more attempts to match. That is a real effect and it was
measured rather than assumed. Over all 12,862 active nodes of a snapshot of the
live database and all 234 frozen goldset queries, scoring every node three ways
-- the legacy single vector, the re-encoded *first* chunk alone, and the
max-pool -- isolates it from both selection and encoder drift:

===========  =====  ==============================  =====================
chunks (k)   nodes  max-pool minus first chunk      that gain / log2(k)
===========  =====  ==============================  =====================
1            4851   +0.000                          --
2            1627   +0.027                          0.027
3             980   +0.052                          0.033
4             912   +0.065                          0.033
6             737   +0.085                          0.033
8             413   +0.095                          0.032
12            165   +0.109                          0.030
16            155   +0.130                          0.032
===========  =====  ==============================  =====================

The gain is not noise and it is not content: re-encoding the same first window
moves every bucket by the same -0.005, while the extra-window gain is strictly
monotone in ``k`` and almost exactly proportional to ``log2(k)``. A
least-squares fit through the origin (a one-window node has no extra windows
and must take no correction) gives **0.0313 of cosine per doubling**. Left
uncorrected it moved real rankings: on a vector-only top-5 over the same fixed
population, one-chunk nodes fell from 29.7% of slots to 15.0% while 16+-chunk
nodes rose from 4.4% to 13.6%.

So the correction is applied, at the coefficient the measurement produced --
:data:`LENGTH_BIAS_LOG2_COEFFICIENT`. The frozen goldset then *checks* that
coefficient rather than choosing it, and agrees. All three rows are full
end-to-end harness runs over the same 234-query goldset and the same chunked
snapshot, so the vector channel is the only difference between them:

==========================  ======  ======  ======  ==============
vector channel              hit@1   hit@5   MRR     hit@5 on tail
==========================  ======  ======  ======  ==============
single vector (baseline)    0.175   0.526   0.324   0.077
max-pool, uncorrected       0.188   0.534   0.338   --
max-pool, beta = 0.031      0.188   0.577   0.354   0.385
==========================  ======  ======  ======  ==============

The last column is the 26 goldset items whose answer lies past the node's first
128 tokens -- the subset this whole change exists for. It goes from 0.077 to
0.385 hit@5 (0.046 to 0.204 MRR), and no stratum regresses: content_grounded
0.694 -> 0.719, role_query 0.222 -> 0.444, cross_lingual flat at 0.105 (that
one is a jargon-vocabulary problem, not a window problem).

A sensitivity sweep over the same goldset puts hit@5 at 0.573 for every beta in
[0.015, 0.045] and back down to 0.543 at 0.060, so the fitted value sits inside
a plateau rather than on a peak, and nothing was picked for scoring best -- the
sweep's own argmax on MRR is beta = 0.015, which is not what ships.

What the shift does to STRONG_VECTOR_MATCH
------------------------------------------

:data:`STRONG_VECTOR_MATCH` (0.65) is both the base-score override in
``rank_candidates`` and the cross-scope admission bar, so a shift in the
vector_score distribution moves how often either fires. Over every result the
goldset run returned, the share at or above 0.65 goes:

* baseline 30.1%, precision of those strong matches 0.284
* max-pool uncorrected 40.0%, precision 0.204 -- a third more results claiming
  "strong match", and materially less often right
* max-pool corrected **18.3%**, precision 0.259

The correction does not merely undo the inflation, it lands the distribution
below where it started (median vector_score 0.553 -> 0.515): a long node now
pays up to 0.12 for its windows, and long nodes were most of what sat above
0.65. So 0.65 fires *less* often than the calibration it was chosen under, in
the safe direction -- fewer overrides, fewer cross-scope admissions, and the
precision of what still qualifies is back within noise of baseline. The
relative gate (``CROSS_SCOPE_RELATIVE_VECTOR``) is scale-free and unaffected.
The constant is therefore left alone: the goldset improves with it unchanged,
and moving it would be a second untested intervention on top of this one.
Retuning the channel blend for the new distribution is the separate downstream
step (``weights-recalibration``); nothing here changes a weight or a floor.
"""

from __future__ import annotations

import sqlite3
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, replace
from math import log2
from typing import Any

from living_memory.edge_derivation import CONTENT_REFERENCE_KIND, DERIVED_FROM_KIND
from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity, tokenize
from living_memory.feedback import FeedbackService, feedback_weighted_score
from living_memory.models import (
    REJECTED_ALTERNATIVE_KIND,
    Connection,
    ConnectionType,
    Node,
)
from living_memory.scope import GLOBAL_SCOPE, ScopePlan, ScopeResolver, scope_family
from living_memory.storage import (
    CHUNK_EMBEDDING_DTYPE,
    CHUNK_EMBEDDING_ITEMSIZE,
    CHUNK_EMBEDDING_TABLE,
    MemoryStore,
    unpack_chunk_embedding,
)

try:
    import numpy as _np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - numpy is a normal runtime dep
    _np = None  # type: ignore[assignment]


DEFAULT_VECTOR_SCAN_LIMIT = 50_000
# Per-rank multiplier favouring narrower scopes in the plan. Must be large
# enough that a requested-scope node of comparable relevance outranks a
# broader-scope node carrying entrenched feedback boosts (compounded
# confidence/usefulness/access multiplier, capped at FEEDBACK_MULTIPLIER_CAP),
# yet stay a soft re-rank: a clearly stronger cross-scope precedent must
# remain retrievable, never filtered.
SCOPE_RANK_BOOST_STEP = 0.2
STRONG_VECTOR_MATCH = 0.65
# Admission gate for candidates outside the plan's narrow scopes: without
# it the broad tail of a narrow-scope plan crowds out max_results, because
# _collect_bm25 normalizes rank scores per scope and therefore mints
# bm25 = 1.0 for the broadest scope's top FTS hit however weak the lexical
# match really is — which is why bm25 evidence deliberately does NOT admit a
# cross-scope candidate. Deliberate evidence does: a schema trigger, a strong
# graph connection, or vector similarity that is strong in absolute terms or
# comparable to the best ungated match. Requested-scope candidates are never
# gated (structural retention guarantee), and under a session plan neither
# are candidates from the plan's own project scope — that scope enters the
# plan only through the caller's ambient project declaration, a deliberate
# association unlike the global fallback. When an event has no candidate in
# any ungated scope the gate stays open: a cross-scope answer is then the
# only answer, so the gate can never empty a recall on its own. This keeps
# the promise above: a clearly stronger cross-scope precedent remains
# retrievable; only the weak-evidence tail is dropped.
#
# Graph activation strong enough to witness a deliberate connection: about a
# first-hop edge over a high-weight relation from a solid seed. Multi-hop
# chains decay by 0.72 per hop and weak-seed fan-out starts far below 1.0,
# so both fall under the bar unless the path is genuinely strong.
CROSS_SCOPE_GRAPH_ADMIT = 0.75
# A vector match that could stand on raw similarity alone (the
# STRONG_VECTOR_MATCH override above) is deliberate evidence wherever the
# node lives; keep the two coupled so "strong match" means one thing.
CROSS_SCOPE_VECTOR_ADMIT = STRONG_VECTOR_MATCH
# Below the absolute bar, a cross-scope candidate must be comparably relevant
# to the best ungated vector match: within this share of it. Applies only
# when the ungated scopes have vector evidence at all — with none, weak
# cross-scope vectors would trivially clear a zero bar.
CROSS_SCOPE_RELATIVE_VECTOR = 0.9
SCHEMA_TRIGGER_OVERLAP_THRESHOLD = 0.5
SCHEMA_TRIGGER_BASE_SCORE = 0.95
SCHEMA_TRIGGER_BOOST = 1.8
VECTOR_MATCH_THRESHOLD = 0.08
GRAPH_SEED_LIMIT = 50
MAX_NEIGHBORS_PER_NODE = 200


@dataclass(frozen=True, slots=True)
class RecallResult:
    """Ranked recall result with scoring provenance."""

    node: Node
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    graph_score: float = 0.0
    trigger_score: float = 0.0
    scope_rank: int = 0
    methods: tuple[str, ...] = ()
    path: tuple[str, ...] = ()
    recall_event_id: str | None = None
    # True when some supersedes edge targets this node: a correction exists,
    # whether or not it qualified for this recall. Consumers get the staleness
    # signal even when the correction itself is decayed or absent.
    superseded: bool = False

    @property
    def node_id(self) -> str:
        return self.node.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node.to_dict(),
            "score": self.score,
            "bm25_score": self.bm25_score,
            "vector_score": self.vector_score,
            "graph_score": self.graph_score,
            "trigger_score": self.trigger_score,
            "methods": list(self.methods),
            "path": list(self.path),
            "recall_event_id": self.recall_event_id,
            "superseded": self.superseded,
        }


@dataclass(slots=True)
class _Candidate:
    node: Node
    bm25_score: float = 0.0
    vector_score: float = 0.0
    graph_score: float = 0.0
    trigger_score: float = 0.0
    path: tuple[str, ...] = ()

    def methods(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.bm25_score > 0.0:
            names.append("bm25")
        if self.vector_score > 0.0:
            names.append("vector")
        if self.graph_score > 0.0:
            names.append("graph")
        if self.trigger_score > 0.0:
            names.append("trigger")
        return tuple(names)


@dataclass(frozen=True, slots=True)
class _ChunkBlock:
    """One vector-width slice of a scope's chunk corpus, grouped by node.

    ``rows`` holds every chunk vector of ``node_ids`` back to back in scan
    order and ``starts[i]`` is where node ``i``'s run of chunks begins, which
    is what lets the max-pool be one ``np.maximum.reduceat`` instead of a
    Python loop over the corpus's 60,530 rows. Rows are stored unit-length, so
    a dot product with a unit query *is* the cosine — and so the numpy path and
    the numpy-free path compute the same number rather than two different ones.

    One block per vector width, because a corpus is allowed to hold more than
    one: a fixture storing 3-d vectors, a database caught mid-re-embed,
    whatever ``MemoryStore.chunk_embedding_dimensions`` would report as several
    values. Vectors of different widths are points in different spaces — they
    cannot share a matrix and must not be compared at all, so each width gets
    its own and only the one matching the query is ever multiplied.
    """

    dimension: int
    node_ids: tuple[str, ...]
    #: ``np.ndarray[intp]`` with numpy, ``tuple[int, ...]`` without.
    starts: Any
    #: ``(len(rows), dimension)`` float32 matrix with numpy, a tuple of
    #: per-chunk float tuples without.
    rows: Any
    row_count: int
    #: ``log2`` of each node's chunk count, aligned with ``node_ids``. Computed
    #: once at scan time because it is what the length-bias correction
    #: subtracts, on every query, from every node.
    log2_chunk_counts: Any


@dataclass(frozen=True, slots=True)
class _ScopeChunkIndex:
    """A scope's whole chunk corpus, reusable until ``revision`` moves."""

    revision: tuple[Any, ...]
    blocks: tuple[_ChunkBlock, ...]
    node_count: int
    row_count: int


class MemoryRecallService:
    """Callable retrieval policy service on top of MemoryStore."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        embedder: LocalEmbeddingModel | None = None,
        vector_scan_limit: int = DEFAULT_VECTOR_SCAN_LIMIT,
    ) -> None:
        self.store = store
        self.embedder = embedder or LocalEmbeddingModel(model_name=store.config.embedding_model)
        # Kept because it is part of this constructor's published signature, and
        # deliberately still not applied as a scan cap: truncating the chunk
        # corpus would drop vectors silently, and the live corpus (60.5k chunks)
        # already exceeds the old 50k default. What used to make a cap tempting
        # -- re-parsing every vector on every recall -- is what the cached
        # matrix below removes.
        self.vector_scan_limit = int(vector_scan_limit)
        self.scope_resolver = ScopeResolver()
        self.feedback = FeedbackService(store)
        self.last_recall_event_id: str | None = None
        self._chunk_index: dict[str, _ScopeChunkIndex] = {}
        self._write_probe: tuple[int, int] | None = None
        self._chunk_revision: tuple[Any, ...] | None = None
        self._store_embedder_shared = False

    def memory_recall(
        self,
        query: str,
        *,
        scope: str | None = None,
        ambient_context: dict[str, Any] | None = None,
        depth: int | str | None = 1,
        max_results: int = 10,
        log_access: bool = True,
        log_event: bool | None = None,
    ) -> list[RecallResult]:
        self.last_recall_event_id = None
        if max_results <= 0:
            return []

        query = query.strip()
        if not query:
            return []

        plan = self.scope_resolver.resolve(
            query=query,
            scope=scope,
            ambient_context=ambient_context,
            store=self.store,
        )
        candidates: dict[str, _Candidate] = {}

        self._collect_bm25(query, plan, candidates, max_results=max_results)
        self._collect_vector(query, plan, candidates, max_results=max_results)
        self._collect_schema_triggers(query, plan, candidates)

        graph_depth, causal_mode = _parse_depth(depth, query)
        decision_mode = _is_decision_depth(depth)
        if graph_depth > 0 and candidates:
            self._collect_graph(
                plan,
                candidates,
                max_depth=graph_depth,
                causal_mode=causal_mode,
                decision_mode=decision_mode,
            )

        ranked = self.rank_candidates(
            candidates,
            plan,
            causal_mode=causal_mode,
            decision_mode=decision_mode,
        )
        limited = ranked[:max_results]
        if log_access:
            limited = [self._record_result_access(result) for result in limited]
        if log_event is None:
            log_event = log_access
        if log_event:
            event = self.store.record_recall_event(
                query=query,
                scope=plan.requested_scope,
                requested_scope=plan.requested_scope,
                resolved_scopes=plan.scopes,
                ambient_context=ambient_context,
                depth=depth,
                max_results=max_results,
                results=[_recall_result_summary(index, result) for index, result in enumerate(limited)],
            )
            self.last_recall_event_id = event.id
            limited = [replace(result, recall_event_id=event.id) for result in limited]
        return limited

    def memory_connect(
        self,
        id_a: str,
        id_b: str,
        relation_type: ConnectionType,
        *,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> Connection:
        return self.store.create_connection(
            id_a,
            id_b,
            relation_type,
            weight=weight,
            metadata=metadata,
        )

    def rank_candidates(
        self,
        candidates: dict[str, _Candidate],
        plan: ScopePlan,
        *,
        causal_mode: bool = False,
        decision_mode: bool = False,
    ) -> list[RecallResult]:
        results: list[RecallResult] = []
        corrections_by_superseded, superseding_ids = self._supersedes_sets()
        ungated_scopes = _ungated_scopes(plan)
        narrow_present, best_narrow_vector = _ungated_scope_profile(
            candidates, ungated_scopes, decision_mode=decision_mode
        )
        for candidate in candidates.values():
            node = candidate.node
            if node.decayed or not plan.allows(node.scope):
                continue
            if _is_rejected_alternative_node(node) and not decision_mode:
                continue
            if (
                narrow_present
                and node.scope not in ungated_scopes
                and not _cross_scope_admissible(candidate, best_narrow_vector)
            ):
                continue

            weights = self.store.get_retrieval_weights(node.scope).normalized()
            bm25_weight = weights.bm25
            vector_weight = weights.vector
            graph_weight = weights.graph
            if candidate.graph_score > 0.0:
                minimum_graph = 0.75 if causal_mode else 0.25
                if graph_weight < minimum_graph:
                    deficit = minimum_graph - graph_weight
                    graph_weight = minimum_graph
                    remaining = max(0.0, bm25_weight + vector_weight)
                    if remaining > 0.0:
                        bm25_weight = max(0.0, bm25_weight - deficit * (bm25_weight / remaining))
                        vector_weight = max(0.0, vector_weight - deficit * (vector_weight / remaining))

            base_score = (
                bm25_weight * candidate.bm25_score
                + vector_weight * candidate.vector_score
                + graph_weight * candidate.graph_score
            )
            if candidate.vector_score >= STRONG_VECTOR_MATCH:
                base_score = max(base_score, candidate.vector_score)
            if node.level == "schema" and candidate.trigger_score > 0.0:
                base_score = max(base_score, candidate.trigger_score)
            if base_score <= 0.0:
                continue

            scope_boost = (
                1.0
                + max(0, len(plan.scopes) - plan.rank(node.scope) - 1) * SCOPE_RANK_BOOST_STEP
            )
            adjusted = feedback_weighted_score(
                node,
                base_score * scope_boost,
                superseded=node.id in corrections_by_superseded,
                superseding=node.id in superseding_ids,
            )
            if causal_mode and candidate.graph_score > 0.0:
                adjusted *= 1.5
            if node.level == "schema" and candidate.trigger_score > 0.0:
                adjusted *= SCHEMA_TRIGGER_BOOST
            results.append(
                RecallResult(
                    node=node,
                    score=adjusted,
                    bm25_score=candidate.bm25_score,
                    vector_score=candidate.vector_score,
                    graph_score=candidate.graph_score,
                    trigger_score=candidate.trigger_score,
                    scope_rank=plan.rank(node.scope),
                    methods=candidate.methods(),
                    path=candidate.path,
                    superseded=node.id in corrections_by_superseded,
                )
            )

        # The 0.2x/1.2x correction multipliers above are a soft prior only: a
        # stale node whose compounded feedback multiplier sits at the cap can
        # still outscore its own correction. The dominance pass below turns
        # the ordering into a structural guarantee.
        ordered = sorted(
            results,
            key=lambda result: (
                result.score,
                -result.scope_rank,
                result.node.confidence,
                result.node.usefulness_score,
                result.node.access_count,
            ),
            reverse=True,
        )
        return _enforce_correction_dominance(ordered, corrections_by_superseded)

    def submit_feedback(
        self,
        result: RecallResult,
        *,
        useful: bool = True,
        signal: float = 1.0,
        scope: str | None = None,
    ) -> Any:
        return self.feedback.apply(result, useful=useful, signal=signal, scope=scope)

    def _collect_bm25(
        self,
        query: str,
        plan: ScopePlan,
        candidates: dict[str, _Candidate],
        *,
        max_results: int,
    ) -> None:
        per_scope_limit = max(25, max_results * 8)
        expanded_query = _expanded_query(query)
        for scope in plan.scopes:
            rows = self.store.search_content(expanded_query, scope=scope, limit=per_scope_limit)
            for rank, (node, _raw_score) in enumerate(rows):
                if node.decayed or not plan.allows(node.scope):
                    continue
                score = 1.0 / (rank + 1)
                candidate = candidates.setdefault(node.id, _Candidate(node=node))
                candidate.bm25_score = max(candidate.bm25_score, score)

    def _collect_vector(
        self,
        query: str,
        plan: ScopePlan,
        candidates: dict[str, _Candidate],
        *,
        max_results: int,
    ) -> None:
        """Score nodes by the best cosine among their chunk vectors.

        A node's vector score is ``max`` over its chunks, not the mean and not
        a single whole-node vector. The encoder only ever sees the first 128
        tokens of whatever it is handed, so a single vector answers "does the
        *opening* of this node match?"; the max over windows answers "does
        *any part* of this node match?", which is the question recall is
        actually asking. Mean would answer a third question nobody asked --
        one relevant paragraph in a long node is a hit, not a 1/k-strength
        hit -- and would make long nodes systematically unfindable.

        Everything downstream is untouched: the pooled score goes into
        ``_Candidate.vector_score`` exactly where the old single-vector cosine
        went, and fusion, the STRONG_VECTOR_MATCH override and the cross-scope
        gate read it the same way. What the shift in that score's distribution
        does to those two thresholds is measured in the module notes above.
        """

        per_scope_limit = max(50, max_results * 12)
        query_embedding = self.embedder.embed(query)

        # Lazy backfill: ensure every active node in the searched scopes has an
        # embedding before the scan. In steady state these queries return zero
        # rows. The loop drains batches until none are left so first-time
        # recalls populate the full scope just like the previous code did.
        # Under schema v6 this also produces the node's chunks, because
        # update_node writes them in the same transaction as the vector -- so a
        # node the drain reaches is scannable by the very next statement here.
        for scope in plan.scopes:
            while True:
                unembedded = self.store.list_unembedded_nodes(scope=scope, limit=500)
                if not unembedded:
                    break
                for node in unembedded:
                    self._ensure_embedding(node)

        q_arr = _as_query_array(query_embedding)
        # After the drain, never before it: the drain writes chunks.
        revision = self._chunk_corpus_revision()
        scoped_scores: list[tuple[float, str]] = []
        for scope in plan.scopes:
            index = self._scope_chunk_index(scope, revision)
            for block in index.blocks:
                if block.dimension != len(query_embedding):
                    continue
                for similarity, node_id in _pooled_chunk_similarities(
                    block, query_embedding, q_arr
                ):
                    if similarity >= VECTOR_MATCH_THRESHOLD:
                        scoped_scores.append((similarity, node_id))

        if not scoped_scores:
            return

        scoped_scores.sort(key=lambda item: item[0], reverse=True)
        keep_n = per_scope_limit * max(1, len(plan.scopes))
        kept = scoped_scores[:keep_n]
        # One query for the whole kept set rather than one per node. Max-pool
        # puts more nodes over VECTOR_MATCH_THRESHOLD than a single vector did
        # -- a node now clears it if *any* window does -- so this list runs
        # closer to its `keep_n` ceiling than it used to, and the per-node
        # round trip is the part of it that is pure overhead.
        missing = [node_id for _score, node_id in kept if node_id not in candidates]
        fetched = self.store.get_nodes(missing) if missing else {}
        for similarity, node_id in kept:
            existing = candidates.get(node_id)
            if existing is None:
                node = fetched.get(node_id)
                if node is None or node.decayed or not plan.allows(node.scope):
                    continue
                existing = candidates.setdefault(node_id, _Candidate(node=node))
            existing.vector_score = max(existing.vector_score, similarity)

    # ------------------------------------------------------------------
    # Chunk corpus cache
    # ------------------------------------------------------------------

    def _chunk_corpus_revision(self) -> tuple[Any, ...]:
        """Identity of the chunk corpus, cheap enough to re-ask on every recall.

        The cached matrices are only as correct as this value, so it is worth
        being precise about why it is complete. Two tiers, because the exact
        signal and the cheap signal are different things:

        * ``(Connection.total_changes, PRAGMA data_version)`` is exact and
          costs no query. ``total_changes`` counts every row *this* connection
          has written; ``data_version`` changes whenever *another* connection
          commits. If neither moved, nothing anywhere has written since the
          last check, so every cached matrix is provably current. This is the
          path taken for the second and later scopes of one recall.
        * When they did move it was usually this recall's own bookkeeping --
          ``record_access`` and ``record_recall_event`` write on every recall
          and touch no chunk -- so tier two asks the chunk table itself rather
          than throwing away a 60k-row matrix for an access-count bump.

        Tier two is ``COUNT(*)`` plus ``MAX(id)``, and it is not a heuristic.
        Chunk rows are only ever deleted, or delete-then-inserted by
        ``_replace_node_chunks``; nothing updates one in place. A delete-only
        change (content edited without a new vector) moves ``COUNT(*)``. An
        insert mints fresh ULIDs whose leading 48 bits are the current
        millisecond, so ``MAX(id)`` rises above every id minted in an earlier
        millisecond. For both to sit still, two chunk-writing transactions
        would have to land in the same millisecond *with a recall between
        them* -- and the recall that reads this value costs milliseconds, so
        it does not fit in the gap. ``MAX(updated_at)`` would be the obvious
        third component and is deliberately absent: ``updated_at`` is in no
        index, so that aggregate scans the table itself, BLOB pages included.
        """

        connection = self.store.connection
        probe = (int(connection.total_changes), _data_version(connection))
        cached = self._chunk_revision
        if cached is not None and probe == self._write_probe:
            return cached
        revision = _chunk_table_revision(connection)
        self._write_probe = probe
        self._chunk_revision = revision
        return revision

    def _scope_chunk_index(self, scope: str, revision: tuple[Any, ...]) -> _ScopeChunkIndex:
        cached = self._chunk_index.get(scope)
        if cached is not None and cached.revision == revision:
            return cached
        index = self._build_chunk_index(scope, revision)
        self._chunk_index[scope] = index
        return index

    def _build_chunk_index(self, scope: str, revision: tuple[Any, ...]) -> _ScopeChunkIndex:
        """Read one scope's chunk BLOBs into reusable matrices.

        The whole point of the cache. Measured over the whole corpus of a
        snapshot of the live database (12,862 active nodes, 60,530 chunks,
        88.7 MiB of vectors): 146 ms once and nothing afterwards, against the
        767 ms the JSON column costs on the same box for the same nodes
        (100 ms to fetch, 666 ms to ``json.loads``) -- and *that* scan was
        re-issued on every recall, not cached. A recall touches only its plan's
        scopes, so the first recall of a two-scope plan pays ~69 ms of this,
        not the full 146.

        Rows are grouped by byte width rather than by the ``dimensions`` column
        storage records, and that is not the shortcut it looks like. Reading
        the column would mean a second full scan of the chunk table per scope
        for a fact the scorer does not need: a block is only ever multiplied by
        a query of *exactly* its own width, so a row whose bytes disagree with
        its recorded width lands in a width nothing queries and is skipped by
        construction, which is the same outcome and no scan. What the byte
        length must never do is reshape a *scored* corpus around itself, and it
        cannot -- the query width decides that.
        """

        accumulators: dict[int, _BlockAccumulator] = {}
        width = -1
        accumulator: _BlockAccumulator | None = None
        for node_id, _ordinal, view in self.store.iter_chunk_embedding_rows(scope=scope):
            row_width, remainder = divmod(len(view), CHUNK_EMBEDDING_ITEMSIZE)
            if remainder or not row_width:
                # Not a whole number of float32s, or none at all: not a vector.
                continue
            if row_width != width or accumulator is None:
                width = row_width
                accumulator = accumulators.setdefault(width, _BlockAccumulator())
            accumulator.add(node_id, view)

        blocks = tuple(accumulators[width].freeze(width) for width in sorted(accumulators))
        return _ScopeChunkIndex(
            revision=revision,
            blocks=blocks,
            node_count=sum(len(block.node_ids) for block in blocks),
            row_count=sum(block.row_count for block in blocks),
        )

    def _collect_schema_triggers(
        self,
        query: str,
        plan: ScopePlan,
        candidates: dict[str, _Candidate],
    ) -> None:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return
        for scope in plan.scopes:
            schemas = self.store.list_nodes(
                level="schema",
                scope=scope,
                include_decayed=False,
                limit=1_000,
            )
            for schema in schemas:
                trigger = str(schema.context.get("trigger") or "")
                if not trigger:
                    continue
                trigger_tokens = set(tokenize(trigger))
                if not trigger_tokens:
                    continue
                overlap = len(query_tokens & trigger_tokens) / len(trigger_tokens)
                if overlap < SCHEMA_TRIGGER_OVERLAP_THRESHOLD:
                    continue
                score = SCHEMA_TRIGGER_BASE_SCORE + 0.05 * overlap
                candidate = candidates.setdefault(schema.id, _Candidate(node=schema))
                candidate.trigger_score = max(candidate.trigger_score, score)

    def _collect_graph(
        self,
        plan: ScopePlan,
        candidates: dict[str, _Candidate],
        *,
        max_depth: int,
        causal_mode: bool,
        decision_mode: bool = False,
    ) -> None:
        queue: deque[tuple[str, float, int, tuple[str, ...]]] = deque()
        best_seen: dict[str, float] = {}

        # Seed BFS only from the top-K pre-graph candidates to avoid
        # fan-out from low-quality matches that would dilute the graph
        # signal and waste cycles on super-hub expansions.
        seed_candidates = sorted(
            candidates.values(),
            key=lambda c: max(c.bm25_score, c.vector_score, 0.0),
            reverse=True,
        )[:GRAPH_SEED_LIMIT]
        for candidate in seed_candidates:
            seed = max(candidate.bm25_score, candidate.vector_score, 0.05)
            queue.append((candidate.node.id, seed, 0, (candidate.node.id,)))
            best_seen[candidate.node.id] = seed

        # Cache neighbor lookups across the BFS so a node reached via multiple
        # paths is fetched at most once. A None entry marks "fetched and
        # filtered out" so we do not re-issue get_node on revisits.
        node_cache: dict[str, Node | None] = {}

        while queue:
            current_depth = queue[0][2]
            if current_depth >= max_depth:
                break
                
            level_items = []
            while queue and queue[0][2] == current_depth:
                level_items.append(queue.popleft())

            # Batch fetch connections for this level
            node_ids_at_depth = list({item[0] for item in level_items})
            connections_by_node = self.store.list_connections_for_nodes(node_ids_at_depth)

            # Pre-calculate which neighbor nodes we'll need to fetch.
            # Apply the same per-node neighbor bound as the processing pass
            # so super-hubs don't dominate even the pre-fetch phase.
            needed_neighbor_ids = set()
            for item in level_items:
                node_id, activation, depth, path = item
                node_connections = connections_by_node.get(node_id, [])
                if len(node_connections) > MAX_NEIGHBORS_PER_NODE:
                    node_connections = node_connections[:MAX_NEIGHBORS_PER_NODE]
                for connection in node_connections:
                    traversal = _traversal(
                        connection,
                        node_id,
                        causal_mode=causal_mode,
                        decision_mode=decision_mode,
                    )
                    if traversal is None:
                        continue
                    neighbor_id, relation_score = traversal
                    if neighbor_id in path:
                        continue

                    next_score = activation * relation_score * (0.72 ** depth)
                    if next_score <= best_seen.get(neighbor_id, 0.0):
                        continue

                    if neighbor_id not in node_cache and neighbor_id not in candidates:
                        needed_neighbor_ids.add(neighbor_id)
            
            # Batch fetch missing neighbor nodes
            if needed_neighbor_ids:
                fetched_nodes = self.store.get_nodes(needed_neighbor_ids)
                for nid in needed_neighbor_ids:
                    node_cache[nid] = fetched_nodes.get(nid)

            # Process the level. Bound per-node neighbor fan-out to avoid
            # super-hubs (nodes with thousands of edges) dominating the BFS.
            for item in level_items:
                node_id, activation, depth, path = item
                node_connections = connections_by_node.get(node_id, [])
                if len(node_connections) > MAX_NEIGHBORS_PER_NODE:
                    node_connections = node_connections[:MAX_NEIGHBORS_PER_NODE]
                for connection in node_connections:
                    traversal = _traversal(
                        connection,
                        node_id,
                        causal_mode=causal_mode,
                        decision_mode=decision_mode,
                    )
                    if traversal is None:
                        continue
                    neighbor_id, relation_score = traversal
                    if neighbor_id in path:
                        continue

                    next_score = activation * relation_score * (0.72 ** depth)
                    if next_score <= best_seen.get(neighbor_id, 0.0):
                        continue

                    if neighbor_id in node_cache:
                        neighbor = node_cache[neighbor_id]
                    else:
                        existing_candidate = candidates.get(neighbor_id)
                        if existing_candidate is not None:
                            neighbor = existing_candidate.node
                        else:
                            neighbor = self.store.get_node(neighbor_id)
                            node_cache[neighbor_id] = neighbor
                    
                    if neighbor is None or neighbor.decayed or not plan.allows(neighbor.scope):
                        continue

                    best_seen[neighbor_id] = next_score
                    candidate = candidates.setdefault(neighbor_id, _Candidate(node=neighbor))
                    candidate.graph_score = max(candidate.graph_score, min(1.5, next_score))
                    candidate.path = path + (neighbor_id,)
                    queue.append((neighbor_id, next_score, depth + 1, path + (neighbor_id,)))

    def _ensure_embedding(self, node: Node) -> list[float]:
        if node.embedding is not None:
            return node.embedding
        self._share_embedder_with_store()
        embedding = self.embedder.embed(node.content)
        updated = self.store.update_node(node.id, embedding=embedding)
        node.embedding = updated.embedding
        return updated.embedding or embedding

    def _share_embedder_with_store(self) -> None:
        """Lend the store this service's encoder for the chunks it writes.

        ``update_node(embedding=...)`` chunks the node in the same
        transaction, and content longer than one window needs every window
        encoded -- which the store would otherwise do by loading a *second*
        copy of the same model, doubling both load time and resident memory
        for no gain. Done here rather than in ``__init__`` because this is the
        first moment a real write is about to happen: the ranking-only store
        face that ``replay`` passes in has no such method and never gets here.
        """

        if self._store_embedder_shared:
            return
        self._store_embedder_shared = True
        share = getattr(self.store, "set_chunk_embedder", None)
        if share is None:
            return
        embedder = self.embedder
        share(lambda texts: [embedder.embed(text) for text in texts])

    def _record_result_access(self, result: RecallResult) -> RecallResult:
        updated_node = self.store.record_access(result.node.id)
        return replace(result, node=updated_node)

    def _supersedes_sets(self) -> tuple[dict[str, tuple[str, ...]], set[str]]:
        """Supersedes edges as (superseded -> corrections, superseding ids).

        One query feeds every ranking consumer: membership behind the soft
        0.2x/1.2x prior and the ``RecallResult.superseded`` flag (mapping
        keys / the superseding set), and the mapping the dominance pass
        enforces. The SQL text is a load-bearing contract — the replay shim
        (``replay._SchemeStore``) recognizes and serves exactly this
        statement with exactly these columns, which is how replay reranks
        recorded candidates through this same code path — so consume richer
        structure from the same rows rather than issuing new queries here.
        """

        rows = self.store.connection.execute(
            "SELECT source_id, target_id FROM connections WHERE type = 'supersedes'"
        ).fetchall()
        corrections_by_superseded: dict[str, set[str]] = {}
        superseding: set[str] = set()
        for row in rows:
            source = str(row["source_id"])
            target = str(row["target_id"])
            superseding.add(source)
            corrections_by_superseded.setdefault(target, set()).add(source)
        return (
            {
                target: tuple(sorted(sources))
                for target, sources in corrections_by_superseded.items()
            },
            superseding,
        )


MemoryRetrievalService = MemoryRecallService


def memory_recall(
    store: MemoryStore,
    query: str,
    *,
    scope: str | None = None,
    ambient_context: dict[str, Any] | None = None,
    depth: int | str | None = 1,
    max_results: int = 10,
    log_access: bool = True,
    log_event: bool | None = None,
) -> list[RecallResult]:
    return MemoryRecallService(store).memory_recall(
        query,
        scope=scope,
        ambient_context=ambient_context,
        depth=depth,
        max_results=max_results,
        log_access=log_access,
        log_event=log_event,
    )


def memory_connect(
    store: MemoryStore,
    id_a: str,
    id_b: str,
    relation_type: ConnectionType,
    *,
    weight: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> Connection:
    return store.create_connection(
        id_a,
        id_b,
        relation_type,
        weight=weight,
        metadata=metadata,
    )


def feedback_aware_rank(
    store: MemoryStore,
    candidates: dict[str, _Candidate],
    plan: ScopePlan,
) -> list[RecallResult]:
    return MemoryRecallService(store).rank_candidates(candidates, plan)


# DFS colors for the correction-dominance pass.
_UNSEEN, _ACTIVE, _DONE = 0, 1, 2


def _enforce_correction_dominance(
    ordered: list[RecallResult],
    corrections_by_superseded: dict[str, tuple[str, ...]],
) -> list[RecallResult]:
    """Reorder so every present correction outranks the node it supersedes.

    Invariant: for every supersedes edge whose correction and superseded node
    both appear in ``ordered``, the correction ends up strictly above the
    stale node — feedback multipliers, mode boosts, and scope boosts already
    happened and get no say. Presence in ``ordered`` is the liveness test
    (decayed, gated, and zero-scored candidates never reach it): with the
    correction absent, the stale node keeps surfacing at its scored position,
    flagged but never filtered. Chains lift transitively (A over B over C);
    a supersedes cycle among present nodes is bad data that cannot be fully
    satisfied, so exactly the edges that close a cycle are deterministically
    skipped instead of hanging the sort. Results on no enforced edge keep
    their relative order.

    Mechanically each correction inherits the best (smallest) base rank among
    the nodes it transitively supersedes, ties broken so deeper correction
    chains sort first and everything else stays in base order: a lifted
    correction lands directly above the best-ranked node it corrects, and
    unrelated results never trade places.
    """

    if not corrections_by_superseded:
        return ordered
    base_index = {result.node.id: position for position, result in enumerate(ordered)}
    successors: dict[str, list[str]] = {}
    for superseded_id, correction_ids in corrections_by_superseded.items():
        if superseded_id not in base_index:
            continue
        for correction_id in correction_ids:
            if correction_id != superseded_id and correction_id in base_index:
                successors.setdefault(correction_id, []).append(superseded_id)
    if not successors:
        return ordered
    for targets in successors.values():
        targets.sort(key=base_index.__getitem__)

    state: dict[str, int] = {}
    effective_rank: dict[str, int] = {}
    chain_depth: dict[str, int] = {}
    for seed in ordered:
        seed_id = seed.node.id
        if seed_id not in successors or state.get(seed_id, _UNSEEN) != _UNSEEN:
            continue
        state[seed_id] = _ACTIVE
        stack: list[tuple[str, Iterator[str]]] = [(seed_id, iter(successors[seed_id]))]
        while stack:
            node_id, pending = stack[-1]
            descended = False
            for successor_id in pending:
                if state.get(successor_id, _UNSEEN) == _UNSEEN and successor_id in successors:
                    state[successor_id] = _ACTIVE
                    stack.append((successor_id, iter(successors[successor_id])))
                    descended = True
                    break
            if descended:
                continue
            stack.pop()
            state[node_id] = _DONE
            # An _ACTIVE successor here is an ancestor still on the stack, so
            # this edge closes a cycle: skipping it (consistently with the
            # descent above) is the deterministic cycle break. Everything else
            # is final — _DONE, or a leaf with no outgoing edges of its own.
            lifted_rank = base_index[node_id]
            depth = 0
            for successor_id in successors[node_id]:
                if state.get(successor_id, _UNSEEN) == _ACTIVE:
                    continue
                lifted_rank = min(
                    lifted_rank, effective_rank.get(successor_id, base_index[successor_id])
                )
                depth = max(depth, chain_depth.get(successor_id, 0) + 1)
            effective_rank[node_id] = lifted_rank
            chain_depth[node_id] = depth

    return sorted(
        ordered,
        key=lambda result: (
            effective_rank.get(result.node.id, base_index[result.node.id]),
            -chain_depth.get(result.node.id, 0),
            base_index[result.node.id],
        ),
    )


def _recall_result_summary(index: int, result: RecallResult) -> dict[str, Any]:
    return {
        "rank": index + 1,
        "node_id": result.node.id,
        "level": result.node.level,
        "scope": result.node.scope,
        "score": result.score,
        "bm25_score": result.bm25_score,
        "vector_score": result.vector_score,
        "graph_score": result.graph_score,
        "trigger_score": result.trigger_score,
        "methods": list(result.methods),
        "path": list(result.path),
    }


_DEPTH_ALIASES = {"shallow": 1, "normal": 1, "medium": 2, "deep": 3}


def _parse_depth(depth: int | str | None, query: str) -> tuple[int, bool]:
    causal_query = _is_causal_query(query)
    if depth is None:
        return (1, causal_query)
    if isinstance(depth, str):
        lowered = depth.strip().lower()
        if lowered in {"none", "off", "0"}:
            return (0, causal_query)
        if lowered in {"decision", "decisions"}:
            return (1, False)
        if lowered in {"causal", "cause", "why"}:
            return (2, True)
        if lowered in _DEPTH_ALIASES:
            return (_DEPTH_ALIASES[lowered], causal_query)
        try:
            return (max(0, int(lowered)), causal_query)
        except ValueError:
            return (1, causal_query)
    return (max(0, int(depth)), causal_query)


def _is_decision_depth(depth: int | str | None) -> bool:
    return isinstance(depth, str) and depth.strip().lower() in {"decision", "decisions"}


def _is_causal_query(query: str) -> bool:
    lowered = query.lower()
    return any(
        marker in lowered
        for marker in (
            "why",
            "cause",
            "caused",
            "because",
            "reason",
            "root cause",
            "happen",
            "happened",
        )
    )


def _expanded_query(query: str) -> str:
    expanded_tokens = tokenize(query)
    if not expanded_tokens:
        return query
    return f"{query} {' '.join(dict.fromkeys(expanded_tokens))}"


def _traversal(
    connection: Connection,
    current_id: str,
    *,
    causal_mode: bool,
    decision_mode: bool = False,
) -> tuple[str, float] | None:
    rejected_alternative = _is_rejected_alternative_connection(connection)
    if decision_mode:
        if not rejected_alternative:
            return None
    elif rejected_alternative:
        return None

    if causal_mode and connection.type not in {"caused", "requires"}:
        return None

    if connection.source_id == current_id:
        neighbor = connection.target_id
        forward = True
    elif connection.target_id == current_id:
        neighbor = connection.source_id
        forward = False
    else:
        return None

    base = max(0.0, connection.weight)
    if rejected_alternative:
        factor = 1.0
    elif connection.type == "related" and connection.metadata.get("kind") == DERIVED_FROM_KIND:
        # Schema→source-trace provenance: an instance strongly surfaces the
        # distilled schema (backward), drilling schema→exemplar is weaker.
        factor = 0.5 if forward else 0.95
    elif connection.type == "related" and connection.metadata.get("kind") == CONTENT_REFERENCE_KIND:
        # Explicit citation: a matched referee prefers the newer trace that
        # built on it; the referee is context for the referrer.
        factor = 0.75 if forward else 0.9
    elif connection.type == "related":
        factor = 0.65
    elif connection.type == "contradicts":
        factor = 0.35
    elif connection.type == "caused":
        if causal_mode:
            factor = 0.25 if forward else 1.0
        else:
            factor = 0.75 if forward else 0.85
    elif connection.type == "requires":
        factor = 0.85 if forward else 0.45
    elif connection.type == "supersedes":
        factor = 0.35 if forward else 1.15
    else:
        factor = 0.0

    score = base * factor
    if score <= 0.0:
        return None
    return neighbor, score


def _ungated_scopes(plan: ScopePlan) -> tuple[str, ...]:
    """Scopes whose candidates bypass the cross-scope admission gate.

    The requested scope always does. A session plan additionally shields the
    project scope it carries: that scope is in the plan only because the
    caller's ambient context declared it as the current project — a
    deliberate association — and a session's few notes must not gate the
    project's knowledge behind similarity to themselves.

    A global-requested plan shields every scope it carries for the same
    reason: the resolver widens a global request only with the deployment's
    configured default project scope — a deliberate operator declaration,
    never a similarity inference — so such a plan holds no cross-scope guess
    to gate.
    """

    if plan.requested_scope == GLOBAL_SCOPE:
        return plan.scopes
    if scope_family(plan.requested_scope) != "session":
        return (plan.requested_scope,)
    return (plan.requested_scope,) + tuple(
        scope
        for scope in plan.scopes
        if scope != plan.requested_scope and scope_family(scope) == "project"
    )


def _ungated_scope_profile(
    candidates: dict[str, _Candidate],
    ungated_scopes: tuple[str, ...],
    *,
    decision_mode: bool = False,
) -> tuple[bool, float]:
    """Whether any rankable ungated-scope candidate exists, and its best vector.

    Only candidates that survive the same pre-score skips as the ranking loop
    (decayed, rejected-alternative outside decision mode) count: a candidate
    that can never rank must not arm the cross-scope gate, or the gate could
    empty a recall whose only real answers are cross-scope.
    """

    present = False
    best_vector = 0.0
    for candidate in candidates.values():
        node = candidate.node
        if node.scope not in ungated_scopes or node.decayed:
            continue
        if _is_rejected_alternative_node(node) and not decision_mode:
            continue
        present = True
        if candidate.vector_score > best_vector:
            best_vector = candidate.vector_score
    return present, best_vector


def _cross_scope_admissible(candidate: _Candidate, best_narrow_vector: float) -> bool:
    """Deliberate evidence that admits a candidate from outside the ungated scopes.

    bm25 rank scores never qualify (per-scope normalization makes them
    incomparable across scopes; see the gate constants above).
    """

    if candidate.trigger_score > 0.0:
        return True
    if candidate.graph_score >= CROSS_SCOPE_GRAPH_ADMIT:
        return True
    if candidate.vector_score >= CROSS_SCOPE_VECTOR_ADMIT:
        return True
    return (
        best_narrow_vector > 0.0
        and candidate.vector_score >= CROSS_SCOPE_RELATIVE_VECTOR * best_narrow_vector
    )


def _is_rejected_alternative_connection(connection: Connection) -> bool:
    return (
        connection.type == "contradicts"
        and connection.metadata.get("kind") == REJECTED_ALTERNATIVE_KIND
    )


def _is_rejected_alternative_node(node: Node) -> bool:
    return bool(node.context.get("is_rejected_alternative"))


def _as_query_array(query_embedding: list[float]) -> Any:
    """Convert query embedding to a normalized numpy array (or None without numpy)."""

    if _np is None or not query_embedding:
        return None
    arr = _np.asarray(query_embedding, dtype=_np.float32)
    norm = float(_np.linalg.norm(arr))
    if norm <= 0.0:
        return None
    if not 0.999 <= norm <= 1.001:
        arr = arr / norm
    return arr


def _data_version(connection: sqlite3.Connection) -> int:
    """``PRAGMA data_version``: bumped when another connection commits."""

    row = connection.execute("PRAGMA data_version").fetchone()
    return int(row[0]) if row is not None else 0


def _chunk_table_revision(connection: sqlite3.Connection) -> tuple[Any, ...]:
    """``(row count, greatest chunk id)``, or ``()`` on a pre-v6 database.

    Two statements rather than the one they obviously fold into, because the
    fold costs an order of magnitude: SQLite answers a bare ``COUNT(*)`` from
    the smallest index (0.14 ms over 60,530 rows) and a bare ``MAX`` of an
    indexed column by seeking one edge of it (0.002 ms), but asking for both in
    one SELECT gives up both fast paths and scans (1.8 ms). This runs on every
    recall whose predecessor wrote anything, which is most of them.

    A file with no chunk table has no chunk corpus to invalidate, and an empty
    tuple compares equal to itself, so such a store caches an empty index once
    instead of re-probing forever.
    """

    try:
        counted = connection.execute(
            f"SELECT COUNT(*) FROM {CHUNK_EMBEDDING_TABLE}"
        ).fetchone()
        greatest = connection.execute(
            f"SELECT MAX(id) FROM {CHUNK_EMBEDDING_TABLE}"
        ).fetchone()
    except sqlite3.OperationalError:
        return ()
    return (int(counted[0]), greatest[0])


#: Largest deviation from unit length that still counts as "the writer already
#: normalized this". The encoder returns unit vectors and storing them as
#: float32 rounds the norm by ~2e-7 (measured 2.4e-7 across the live corpus's
#: 60,530 chunks), so real vectors clear this by four orders of magnitude while
#: anything genuinely unnormalized -- a fixture, a hand-written vector -- does
#: not. Skipping the division when it holds saves ~54 ms and a second 89 MiB
#: allocation on the cold build; the error it can hide is bounded by the
#: tolerance itself, far below what any threshold in this module resolves.
UNIT_NORM_TOLERANCE = 1e-6

#: Cosine subtracted per doubling of a node's chunk count, cancelling the part
#: of a max-pooled score that comes from having more windows rather than better
#: ones. See ``Length bias`` in the module docstring for the derivation and the
#: numbers; ``0.031`` is the least-squares fit of the measured gain, not a
#: value tuned against the goldset.
LENGTH_BIAS_LOG2_COEFFICIENT = 0.031


class _BlockAccumulator:
    """Scan-time buffer for one vector width, frozen into a ``_ChunkBlock``.

    The BLOBs are concatenated into one ``bytearray`` as they arrive rather
    than collected into a list and joined at the end. Measured over the live
    corpus that is the difference between a 146 ms cold build and a 276 ms
    one: the list costs 60,530 Python objects to hold, and the join then
    copies all 89 MiB of them a second time.
    """

    __slots__ = ("payload", "node_ids", "starts", "rows")

    def __init__(self) -> None:
        self.payload = bytearray()
        self.node_ids: list[str] = []
        self.starts: list[int] = []
        self.rows = 0

    def add(self, node_id: str, view: Any) -> None:
        # iter_chunk_embedding_rows promises rows grouped by node, so a node id
        # that differs from the last one can only mean a new run.
        if not self.node_ids or self.node_ids[-1] != node_id:
            self.node_ids.append(node_id)
            self.starts.append(self.rows)
        self.payload += view
        self.rows += 1

    def freeze(self, dimension: int) -> _ChunkBlock:
        node_ids = tuple(self.node_ids)
        if _np is not None:
            # One reinterpretation of one buffer, no per-vector Python object:
            # this is the step that replaces 685 ms of json.loads. The
            # bytearray stays alive as the array's base and is buffer-locked
            # against resizing for exactly as long.
            matrix = _np.frombuffer(self.payload, dtype=CHUNK_EMBEDDING_DTYPE).reshape(
                self.rows, dimension
            )
            # einsum here is a fused row-wise self-dot: the row norms without a
            # temporary copy of the matrix.
            norms = _np.sqrt(_np.einsum("ij,ij->i", matrix, matrix))
            if self.rows and float(_np.abs(norms - 1.0).max()) > UNIT_NORM_TOLERANCE:
                norms[norms == 0.0] = 1.0
                matrix = matrix / norms[:, None]
            starts = _np.asarray(self.starts, dtype=_np.intp)
            counts = _np.diff(_np.append(starts, self.rows)).astype(_np.float32)
            return _ChunkBlock(
                dimension=dimension,
                node_ids=node_ids,
                starts=starts,
                rows=matrix,
                row_count=self.rows,
                log2_chunk_counts=_np.log2(counts),
            )
        payload = bytes(self.payload)
        stride = dimension * CHUNK_EMBEDDING_ITEMSIZE
        rows = tuple(
            tuple(unpack_chunk_embedding(payload[offset : offset + stride], dimension))
            for offset in range(0, len(payload), stride)
        )
        bounds = tuple(self.starts) + (self.rows,)
        return _ChunkBlock(
            dimension=dimension,
            node_ids=node_ids,
            starts=tuple(self.starts),
            rows=rows,
            row_count=self.rows,
            log2_chunk_counts=tuple(
                log2(bounds[index + 1] - bounds[index]) for index in range(len(node_ids))
            ),
        )


def _pooled_chunk_similarities(
    block: _ChunkBlock,
    query_embedding: list[float],
    q_arr: Any,
) -> list[tuple[float, str]]:
    """Max-pool one block's chunk cosines into one score per node.

    Negative cosines clip to 0 before pooling, as the single-vector path did;
    clipping and ``max`` commute, so the order costs nothing either way. The
    length-bias correction is then subtracted, and the result clipped again --
    a node's score is a cosine, and cosines do not go below zero here.
    """

    if _np is not None and isinstance(block.rows, _np.ndarray):
        if q_arr is None:
            # A query with no direction has no cosine to anything.
            return []
        sims = block.rows @ q_arr
        # `starts` is the run boundary of each node, so one reduceat pools
        # every node at once -- no Python loop over the corpus's 60k chunk rows.
        pooled = _np.clip(_np.maximum.reduceat(sims, block.starts), 0.0, None)
        if LENGTH_BIAS_LOG2_COEFFICIENT:
            pooled -= LENGTH_BIAS_LOG2_COEFFICIENT * block.log2_chunk_counts
            _np.clip(pooled, 0.0, None, out=pooled)
        return list(zip(pooled.tolist(), block.node_ids, strict=True))

    bounds = tuple(block.starts) + (block.row_count,)
    pooled_rows: list[tuple[float, str]] = []
    for index, node_id in enumerate(block.node_ids):
        best = 0.0
        for row in block.rows[bounds[index] : bounds[index + 1]]:
            best = max(best, cosine_similarity(query_embedding, list(row)))
        penalty = LENGTH_BIAS_LOG2_COEFFICIENT * block.log2_chunk_counts[index]
        pooled_rows.append((max(0.0, best - penalty), node_id))
    return pooled_rows
