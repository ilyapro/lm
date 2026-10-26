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

The graph channel's entry from query space
==========================================

BFS used to be seedable only from what BM25, the vector channel, or a schema
trigger had already found, so the graph had no entry of its own from the space
of *questions*: a query whose answer no other channel can reach never got a
walk at all. :mod:`living_memory.query_anchors` supplies the missing entry --
a remembered query, embedded by the same model as node content, carrying
weighted edges to the nodes a grounded consumption of that query actually used.
Matching "query <-> past query" and taking one hop finds a repeated situation
that "query <-> node content" cannot: the operator's own jargon meets itself
(«поревьювь EZ-13871» <-> «поревьювь EZ-12826» = 0.931) where the same query
against the English content of the node it should find sits at 0.100-0.269.

This is deliberately **not** a fifth channel. A matched anchor's targets enter
``_collect_graph`` as seeds, are scored as graph activation, and are blended by
the same learned per-scope ``graph`` weight as any other graph evidence -- so
the weights keep meaning what they meant. Three things change and nothing else:

* the graph guard runs when anchors produced seeds even with no other
  candidate, which is precisely the blind spot;
* an anchor's targets open a second walk with activation
  ``anchor cosine x edge weight``, landing in ``_Candidate.anchor_score`` under
  the same ``min(1.5, ...)`` cap as a BFS-discovered neighbour;
* nothing else. With no anchors, or on a query class no anchor resembles, no
  anchor walk runs, every ``anchor_score`` is 0.0, and the ranking is
  byte-identical to the pre-anchor code.

Anchors never demote what they seed
-----------------------------------

Extra evidence must not cost a candidate anything, and the first cut of the
above broke that. ``rank_candidates`` floors the per-scope normalized graph
weight at 0.25 (0.75 in causal mode) for any candidate with a graph activation,
taking the deficit proportionally out of ``bm25`` and ``vector``. That floor
predates anchors and is exactly what makes a graph-only candidate competitive
against lexical ones -- it is not the defect and is not touched here. But an
anchor is a *new* way for an already-strong lexical/vector candidate to acquire
a **small** graph activation, and crossing that step function moved up to a
quarter of the weight off the very evidence that was carrying it. Measured on
the frozen 234-item goldset: of the results that gained a graph score from an
anchor seed, 54 lost score against 18 that gained, median relative loss 8.1%;
two fell out of the returned list entirely. The anchor demoted the node it was
seeding (``result.md`` section 6).

The fix is to keep the anchors-off score computable and never go below it:

* ``_collect_graph`` runs the pre-anchor walk **first and alone**, before a
  single anchor target has been admitted as a candidate. ``graph_score``
  therefore still holds exactly what it would hold with anchors off -- an
  anchor can neither displace one of its seeds nor perturb one of its numbers.
* the anchor walk then runs separately into ``anchor_score``, and
  ``combined_graph_score`` is what the ranker blends.
* a candidate whose activation an anchor moved is scored **both** ways, through
  one ``_blend_candidate_score``, and keeps the better result. The floor
  applies identically in both, so the comparison is honest.

An anchor-*only* candidate has no anchors-off score to fall back to, so it
keeps the floored blend in full: the blind spot the floor was covering stays
covered, which a plain "exempt anchor activations from the floor" would have
closed again.

Both factors of the activation live in [0, 1] and the product is a confidence,
so an anchor seed lands in exactly the numeric range seeds already occupied
(a bm25 rank score, a cosine) rather than inflating the channel. One grounded
consumption is worth ``ANCHOR_EDGE_WEIGHT`` (0.25), so a link the operator's
work has confirmed once seeds weakly and one confirmed four times seeds at full
strength -- the calibration is the edge weight's, not a new constant here.

Cost, measured
--------------

``match_anchors`` is a full scan by design (a few thousand rows against the
chunk corpus's 60k), and its own docstring measures that scan at ~12 ms cold
and instructs read-path callers to cache the vectors and re-validate with
``MemoryStore.query_anchor_revision``. That is what :class:`_AnchorVectorCache`
below provides: policy stays in ``match_anchors``, while matrices, norms and
scope row indices are prepared once per embedding width. Each query selects
its scopes in their original order; contiguous selections use matrix views,
and disjoint selections use temporary gathers. The transactional SQLite
revision covers anchor membership and vectors across all writers.
Reinforcement of a live anchor never rewrites a stored vector, so it preserves
the cache.

End to end, on a snapshot of the live database (12,901 active nodes, 143,285
connections, 60,636 chunk vectors) carrying 3,760 anchors and 10,716 edges
built from real pre-cutoff consumptions, against real post-cutoff queries --
``artifacts/anchors/latency.json``:

============================  ==================  ==================
stratum                       paired delta p50    anchor stage p50
============================  ==================  ==================
holdout, no anchor matched     -0.6 ms             0.8 ms
holdout, anchor matched        +8.3 ms             2.9 ms
repeated query (worst case)    +4.4 ms             1.1 ms
============================  ==================  ==================

against a budget of +5 ms p50. The middle and bottom rows are the feature doing
work rather than overhead: a matched anchor opens a walk from seeds no other
channel produced, so the BFS explores a neighbourhood that would not have been
visited at all. That walk is a *second* one, separate from the pre-anchor walk
so it cannot perturb it (see below); it is bounded by the same
:data:`GRAPH_SEED_LIMIT` roots and the same depth, and it runs only when an
anchor actually matched -- which the table's top row, the common case, shows
costing nothing.

With zero live anchors the probe short-circuits and no scan happens: cold start
costs one indexed ``COUNT``.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from math import log2, sqrt
from typing import Any, Callable

from living_memory.edge_derivation import CONTENT_REFERENCE_KIND, DERIVED_FROM_KIND
from living_memory.embeddings import (
    LocalEmbeddingModel,
    cosine_similarity,
    cyrillic_prefix_terms,
    tokenize,
)
from living_memory.feedback import (
    FeedbackService,
    _explicit_feedback_policy,
    feedback_weighted_score,
)
from living_memory.hubs import hub_demotions
from living_memory.irrelevance import query_demotions
from living_memory.models import (
    REJECTED_ALTERNATIVE_KIND,
    Connection,
    ConnectionType,
    Node,
    RetrievalWeights,
    SourcePassage,
)
from living_memory.query_anchors import (
    ANCHOR_MATCH_COSINE_THRESHOLD,
    ANCHOR_MATCH_LIMIT,
    AnchorMatch,
    _PreparedAnchorVectors,
    _prepare_anchor_vectors,
    match_anchors,
)
from living_memory.schema_dedup import collapse_schema_duplicates
from living_memory.relations import source_relation_weight
from living_memory.scope import (
    ScopePlan,
    ScopeResolver,
    normalize_scope,
    scope_family,
)
from living_memory.score_gate import QueryRelevance, REFERENCE_WEIGHTS, apply_score_gate, min_score_from_env, passes_gate
from living_memory.storage import (
    CHUNK_EMBEDDING_DTYPE,
    CHUNK_EMBEDDING_ITEMSIZE,
    MemoryStore,
    unpack_chunk_embedding,
)
from living_memory.validity import supersedes_sets

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
# cross-scope candidate. Deliberate evidence does: an applicable schema trigger, a strong
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
#: Valve of goal schema-ranks-by-meaning (docs/recall-schema-trigger.md).
#: Unset: half the trigger's words in the query make the node available;
#: its rank then depends on content evidence, while the separate trigger gate
#: can preserve an instruction with sparse content evidence. ``name``: every node, schema or not, is
#: ranked on the gate's own score (``score_gate.gate_score``: REFERENCE_WEIGHTS, no
#: per-scope weights, no 1.2x correction prior), so order and gate agree; a schema's
#: trigger counts only when the query *is* the procedure's name (same token set):
#: then bm25 1.0, first if it passes the gate (:meth:`MemoryRecallService._named_schemas_first`).
#: The ordinary trigger and gate branches do not run under the valve.
SCHEMA_TRIGGER_ENV = "LM_RECALL_SCHEMA_TRIGGER"
SCHEMA_TRIGGER_BY_NAME = "name"
#: ``trigger_score`` of a schema the query names in ``name`` mode.
SCHEMA_NAME_TRIGGER_SCORE = 1.0


def schema_trigger_by_name() -> bool:
    """Read ``LM_RECALL_SCHEMA_TRIGGER``; True means ``name`` mode."""

    return os.environ.get(SCHEMA_TRIGGER_ENV, "").strip().lower() == SCHEMA_TRIGGER_BY_NAME

VECTOR_MATCH_THRESHOLD = 0.08
GRAPH_SEED_LIMIT = 50
MAX_NEIGHBORS_PER_NODE = 200
LAZY_CHUNK_NODE_LIMIT = 16
LEGACY_PASSAGE_MAX_CHARS = 32_768
LEGACY_PASSAGE_MAX_CHUNKS = 32

@dataclass(frozen=True, slots=True)
class RecallResult:
    """Ranked recall result with scoring provenance."""

    node: Node
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    graph_score: float = 0.0
    trigger_score: float = 0.0
    trigger_coverage: float = 0.0
    scope_rank: int = 0
    methods: tuple[str, ...] = ()
    path: tuple[str, ...] = ()
    recall_event_id: str | None = None
    # True when a genuine correction edge targets this node: a correction exists,
    # whether or not it qualified for this recall. Consumers get the staleness
    # signal even when the correction itself is decayed or absent.
    superseded: bool = False
    # Set by ``score_gate`` when a result is delivered as a stub instead of a
    # full slot (e.g. ``"below_threshold"``); ``None`` for a regular result.
    withheld: str | None = None
    # Direct relation endpoints, before the public reader's scope projection.
    correction_ids: tuple[str, ...] = ()
    # Transient source evidence from this query's winning chunk, not a score.
    semantic_passage: SourcePassage | None = None
    scored_chunk_id: str | None = None
    # Query activation through source relations, excluding anchors and
    # access/usefulness-derived edge strength. Never projected as wire history.
    source_graph_score: float = 0.0

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
            **({"withheld": self.withheld} if self.withheld else {}),
        }


@dataclass(slots=True)
class _Candidate:
    node: Node
    bm25_score: float = 0.0
    vector_score: float = 0.0
    chunk_id: str | None = None
    #: Activation from the traversal that would have happened with anchors
    #: switched off. Written by that walk only, so it stays usable as the
    #: anchors-off counterfactual the ranker needs.
    graph_score: float = 0.0
    trigger_score: float = 0.0
    trigger_coverage: float = 0.0
    path: tuple[str, ...] = ()
    #: Activation this candidate owes to a matched query anchor, kept apart
    #: from ``graph_score`` so the ranker can still see what the candidate was
    #: worth before the anchor touched it. See "Anchors never demote what they
    #: seed" in the module docstring.
    anchor_score: float = 0.0
    source_graph_score: float = 0.0

    @property
    def combined_graph_score(self) -> float:
        """The graph evidence this candidate actually carries, both sources.

        A max rather than a sum, matching every other writer of graph
        activation: two routes to the same node are one claim seen twice, not
        two claims worth adding up.
        """

        return max(self.graph_score, self.anchor_score)

    def methods(self, graph_score: float | None = None) -> tuple[str, ...]:
        """Channels that carried this candidate.

        ``graph_score`` overrides which graph activation counts, so a result
        the ranker scored on its anchors-off blend reports the channels that
        blend actually used rather than evidence it deliberately set aside.
        """

        effective_graph = (
            self.combined_graph_score if graph_score is None else graph_score
        )
        names: list[str] = []
        if self.bm25_score > 0.0:
            names.append("bm25")
        if self.vector_score > 0.0:
            names.append("vector")
        if effective_graph > 0.0:
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
    Python loop over chunk rows. Raw float32 rows and their norms are retained
    so scoped and broad queries can share them while applying normalization
    to the query's complete width block.

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
    #: per-chunk float tuples without. These are the stored, unnormalized rows.
    rows: Any
    row_count: int
    #: ``log2`` of each node's chunk count, aligned with ``node_ids``. Computed
    #: once at scan time because it is what the length-bias correction
    #: subtracts, on every query, from every node.
    log2_chunk_counts: Any
    norms: Any = None
    norm_deviation: float = 0.0
    chunk_ids: tuple[str, ...] = ()

    @property
    def needs_normalization(self) -> bool:
        return self.norm_deviation > UNIT_NORM_TOLERANCE


@dataclass(frozen=True, slots=True)
class _ScopeChunkIndex:
    """A scope's whole chunk corpus, reusable until ``revision`` moves."""

    revision: int | None
    blocks: tuple[_ChunkBlock, ...]
    node_count: int
    row_count: int


@dataclass(frozen=True, slots=True)
class _ChunkGatherPlan:
    """Reusable row placement and pooling metadata, without a second matrix."""

    blocks: tuple[_ChunkBlock, ...]
    positions: tuple[Any, ...]
    template: _ChunkBlock

    def gather(self) -> _ChunkBlock:
        if not self.positions:
            # Scalar rows are references to the immutable per-scope tuples.
            return self.template
        rows = _np.empty(
            (self.template.row_count, self.template.dimension), dtype=CHUNK_EMBEDDING_DTYPE,
        )
        for block, positions in zip(self.blocks, self.positions, strict=True):
            rows[positions] = block.rows
        return replace(self.template, rows=rows)


@dataclass(frozen=True, slots=True)
class _AnchorVectorCache:
    """The revision-guarded corpus, prepared once per embedding width.

    Scope row indices select from these matrices in the matcher's query path.
    Source BLOBs and selected matrices are not retained. Metadata and targets
    always read through to the store.
    """

    revision: int
    store: MemoryStore
    by_width: dict[int, _PreparedAnchorVectors]

    def query_anchor_corpus(self, width: int) -> _PreparedAnchorVectors | None:
        return self.by_width.get(width)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)


class MemoryRecallService:
    """Callable retrieval policy service on top of MemoryStore."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        embedder: LocalEmbeddingModel | None = None,
        vector_scan_limit: int = DEFAULT_VECTOR_SCAN_LIMIT,
        anchor_seeding: bool = True,
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
        # The ranked candidates beyond the max_results cut of the latest
        # memory_recall call. Downstream consumers (recall-map construction)
        # read it the same way server.py reads ``last_recall_event_id``; it
        # never feeds back into ranking, access logging, or recall events.
        self.last_residual: list[RecallResult] = []
        self.last_query_relevance: QueryRelevance | None = None
        # Ablation switch, not a feature gate: anchors are on by default and
        # this only turns them *off*, so the leak-free with/without evaluation
        # can hold one snapshot, one goldset, and one code path fixed and vary
        # nothing but the graph channel's entry from query space.
        self.anchor_seeding = bool(anchor_seeding)
        self._chunk_index: dict[str, _ScopeChunkIndex] = {}
        self._chunk_gathers: dict[int, _ChunkGatherPlan] = {}
        self._anchor_vectors: _AnchorVectorCache | None = None
        self._store_embedder_shared = False
        self._legacy_refresh_failed = False

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
        select_results: Callable[[list[RecallResult], list[RecallResult], int], tuple[list[RecallResult], list[RecallResult]]] | None = None,
    ) -> list[RecallResult]:
        self.last_recall_event_id = None
        self.last_residual = []
        self.last_query_relevance = None
        if max_results <= 0:
            return []

        query = query.strip()
        if not query:
            return []

        plan = self.scope_resolver.resolve(
            scope=scope,
            ambient_context=ambient_context,
            store=self.store,
        )
        candidates: dict[str, _Candidate] = {}

        self._collect_bm25(query, plan, candidates, max_results=max_results)
        query_embedding = self._collect_vector(
            query, plan, candidates, max_results=max_results
        )
        by_name = schema_trigger_by_name()
        if not by_name:
            self._collect_schema_triggers(query, plan, candidates)

        graph_depth, causal_mode = _parse_depth(depth, query)
        decision_mode = _is_decision_depth(depth)
        # Anchors are an entry *into* the graph channel, so they are asked for
        # exactly when that channel runs: with the graph off (depth=0) there is
        # nothing for a seed to open. The query vector is the one
        # ``_collect_vector`` already computed -- an anchor match embeds
        # nothing of its own.
        anchor_matches = (
            self._match_query_anchors(plan, query_embedding)
            if graph_depth > 0 and self.anchor_seeding else None
        )
        anchor_seeds = (
            self._collect_anchor_seeds(plan, query_embedding, matches=anchor_matches)
            if graph_depth > 0 else {}
        )
        # Anchors can open the graph even when no other channel found a seed.
        if graph_depth > 0 and (candidates or anchor_seeds):
            self._collect_graph(
                plan,
                candidates,
                max_depth=graph_depth,
                causal_mode=causal_mode,
                decision_mode=decision_mode,
                anchor_seeds=anchor_seeds,
            )

        demotions = _merge_demotions(
            self._collect_query_demotions(plan, query_embedding, matches=anchor_matches),
            self._collect_hub_demotions(),
        )
        supersedes = supersedes_sets(self.store, candidates)
        ranked = self.rank_candidates(
            candidates,
            plan,
            causal_mode=causal_mode,
            decision_mode=decision_mode,
            demotions=demotions,
            supersedes=supersedes,
        )
        # Precision stages (goal recall-precision), each behind its own env
        # valve and the identity while it is off: same-procedure schema
        # duplicates give up their slot, then the quality gate decides what is
        # delivered and what stays in the residual the recall map describes.
        if by_name:
            ranked = self._named_schemas_first(
                query,
                plan,
                ranked,
                demotions=demotions,
                causal_mode=causal_mode,
            )
        ranked, _collapsed = collapse_schema_duplicates(ranked)
        self.last_query_relevance = QueryRelevance(
            ranked, plan=plan, demotions=demotions, causal_mode=causal_mode,
            corrections_by_superseded=supersedes[0],
        )
        limited, self.last_residual = apply_score_gate(
            ranked,
            max_results,
            plan=plan,
            demotions=demotions,
            causal_mode=causal_mode,
            corrections_by_superseded=supersedes[0],
            relevance=self.last_query_relevance,
        )
        # Selection belongs before access and event writes: their result ids
        # must describe the answer that the caller actually delivers. A score
        # gate may leave rejected candidates in the residual, so the delivery
        # selector is used only with the plain, ungated cut.
        if select_results is not None and min_score_from_env() is None:
            limited, self.last_residual = select_results(
                limited, self.last_residual, max_results
            )
        if log_event is None:
            log_event = log_access
        if log_access or log_event:
            nodes, event = self.store.record_recall(
                nodes=[result.node for result in limited],
                log_access=log_access,
                log_event=log_event,
                query=query,
                scope=plan.requested_scope,
                requested_scope=plan.requested_scope,
                resolved_scopes=plan.scopes,
                ambient_context=ambient_context,
                depth=depth,
                max_results=max_results,
                results=[_recall_result_summary(index, result) for index, result in enumerate(limited)],
            )
            self.last_recall_event_id = event.id if event is not None else None
            limited = [
                replace(result, node=node, recall_event_id=self.last_recall_event_id)
                for result, node in zip(limited, nodes, strict=True)
            ]
        chunk_ids = {
            result.node_id: candidates[result.node_id].chunk_id
            for result in limited
            if not result.withheld and result.node.level != "schema"
            and candidates[result.node_id].chunk_id is not None
        }
        passages = self.store.get_chunk_passages(list(chunk_ids.values()))
        return [
            replace(result, semantic_passage=passages.get(chunk_ids.get(result.node_id)),
                    scored_chunk_id=chunk_ids.get(result.node_id))
            for result in limited
        ]

    def refresh_legacy_passages(self, snippets: Sequence[RecallResult]) -> None:
        """Prepare at most one delivered snippet for a future semantic recall.

        The caller supplies actual snippet bearers after delivery shaping. The
        scored generation remains this call's only evidence; replacing its index
        cannot give that old score an address in the replacement generation.
        """
        refresh = getattr(self.store, "refresh_legacy_chunks", None)
        if self._legacy_refresh_failed or not callable(refresh):
            return
        for result in snippets:
            if (result.semantic_passage is not None or not result.scored_chunk_id
                    or result.withheld or result.node.level == "schema"
                    or len(result.node.content) > LEGACY_PASSAGE_MAX_CHARS):
                continue
            try:
                self._share_embedder_with_store()
                refresh(
                    result.node_id, result.node.content, result.scored_chunk_id,
                    max_chars=LEGACY_PASSAGE_MAX_CHARS,
                    max_chunks=LEGACY_PASSAGE_MAX_CHUNKS,
                )
            except Exception:
                # Optional indexing must not charge every useful read for the
                # same broken encoder/store. A new service can try again.
                self._legacy_refresh_failed = True
                logging.getLogger(__name__).warning(
                    "could not refresh selected legacy chunk index", exc_info=True,
                )
            return

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
        demotions: Mapping[str, float] | None = None,
        supersedes: tuple[dict[str, tuple[str, ...]], set[str]] | None = None,
    ) -> list[RecallResult]:
        """Score and order candidates.

        ``demotions`` maps node id -> a multiplier in (0, 1] applied to that
        node's final score for this query only (query-relative irrelevance,
        ``living_memory.irrelevance``). Applied before sorting, so the
        correction-dominance pass still sees the demoted order.
        """

        results: list[RecallResult] = []
        corrections_by_superseded, superseding_ids = (
            supersedes if supersedes is not None else supersedes_sets(self.store, candidates)
        )
        ungated_scopes = _ungated_scopes(plan)
        narrow_present, best_narrow_vector = _ungated_scope_profile(
            candidates, ungated_scopes, decision_mode=decision_mode
        )
        # The same gate as it would have read with anchors switched off.
        # Anchors only ever *add* candidates, and an added candidate carries no
        # vector score of its own, so the single thing they can move here is
        # ``narrow_present`` False -> True -- which would gate out a
        # cross-scope candidate the anchor never touched. Recomputed only when
        # an anchor actually seeded something; otherwise it is the same walk
        # over the same dict for the same answer.
        anchor_seeded = any(
            candidate.anchor_score > 0.0 for candidate in candidates.values()
        )
        reference = REFERENCE_WEIGHTS if schema_trigger_by_name() else None
        anchor_free_narrow_present = (
            _ungated_scope_profile(
                candidates, ungated_scopes, decision_mode=decision_mode, anchor_free=True
            )[0]
            if anchor_seeded
            else narrow_present
        )
        for candidate in candidates.values():
            node = candidate.node
            if node.decayed or not plan.allows(node.scope):
                continue
            if _is_rejected_alternative_node(node) and not decision_mode:
                continue

            weights = (reference or self.store.get_retrieval_weights(node.scope)).normalized()
            graph_score = candidate.combined_graph_score
            ungated = not plan.restricted or node.scope in ungated_scopes
            admissible = (
                ungated
                or not narrow_present
                or _cross_scope_admissible(
                    candidate, best_narrow_vector, graph_score=graph_score
                )
            )
            adjusted = (
                _blend_candidate_score(
                    candidate,
                    graph_score,
                    weights,
                    plan=plan,
                    causal_mode=causal_mode,
                    superseded=node.id in corrections_by_superseded,
                    superseding=reference is None and node.id in superseding_ids,
                )
                if admissible
                else 0.0
            )
            effective_graph = graph_score

            # MONOTONICITY. An anchor is extra evidence, so it must not be able
            # to cost a candidate anything -- yet the graph-weight floor a few
            # lines down is a step function at graph_score > 0, and an anchor
            # is a brand new way for a strong lexical/vector candidate to
            # acquire a *small* graph score. Crossing that step moved up to a
            # quarter of the per-scope weight off the very evidence that was
            # carrying the candidate, demoting the node the anchor was seeding
            # (measured: 54 of 204 such results lost score, median -8.1%).
            #
            # The floor is not the defect and is not touched. Instead the
            # candidate is also scored the way it would have been scored with
            # anchors off -- which ``graph_score`` and ``anchor_free_*`` above
            # preserve exactly -- and keeps the better of the two. An anchor
            # can then only ever lift a candidate. Where the floor is what
            # makes an anchor-only candidate competitive at all, this branch
            # never runs: such a candidate has no anchors-off score to fall
            # back to, so it keeps the floored one in full.
            if _has_anchor_free_evidence(candidate) and (
                graph_score > candidate.graph_score or not admissible
            ):
                anchor_free_admissible = (
                    ungated
                    or not anchor_free_narrow_present
                    or _cross_scope_admissible(
                        candidate, best_narrow_vector, graph_score=candidate.graph_score
                    )
                )
                if anchor_free_admissible:
                    anchor_free_score = _blend_candidate_score(
                        candidate,
                        candidate.graph_score,
                        weights,
                        plan=plan,
                        causal_mode=causal_mode,
                        superseded=node.id in corrections_by_superseded,
                        superseding=reference is None and node.id in superseding_ids,
                    )
                    if anchor_free_score > adjusted:
                        adjusted = anchor_free_score
                        effective_graph = candidate.graph_score
            if adjusted <= 0.0:
                continue
            if demotions:
                adjusted *= demotions.get(node.id, 1.0)

            results.append(
                RecallResult(
                    node=node,
                    score=adjusted,
                    bm25_score=candidate.bm25_score,
                    vector_score=candidate.vector_score,
                    graph_score=effective_graph,
                    source_graph_score=candidate.source_graph_score,
                    trigger_score=candidate.trigger_score,
                    trigger_coverage=candidate.trigger_coverage,
                    scope_rank=plan.rank(node.scope),
                    methods=candidate.methods(effective_graph),
                    # A candidate scored without its graph activation has no
                    # graph route to report; with one, the walk recorded it.
                    path=candidate.path if effective_graph > 0.0 else (),
                    superseded=node.id in corrections_by_superseded,
                    correction_ids=corrections_by_superseded.get(node.id, ()),
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
        # A bm25 score is a rank within one query. The named scopes keep their
        # own ranking, as a restricted plan ranks them; a whole-store plan adds
        # one pass over everything, where an unnamed scope competes on the
        # corpus-wide ranking instead of getting a rank 1 of its own.
        scopes = plan.search_scopes if plan.restricted else (*plan.named_scopes, None)
        for scope in scopes:
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
    ) -> list[float]:
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

        Returns the query's embedding so the anchor match can reuse it. The
        encoder costs ~8.6 ms warm on this machine -- more than the whole
        anchor budget -- and the two channels want the same vector of the same
        query, so embedding twice would be paying that twice for one answer.
        """

        per_scope_limit = max(50, max_results * 12)
        query_embedding = self.embedder.embed(query)

        # One bounded queue covers new content, edits and migrated derived
        # chunks. Chunk presence, not the legacy representative, decides work.
        # Oldest-first progress prevents new writes from starving old repairs.
        pending = sorted(
            (node for scope in plan.search_scopes
             for node in self.store.list_unchunked_nodes(scope=scope, limit=LAZY_CHUNK_NODE_LIMIT)),
            key=lambda node: (node.timestamp, node.id),
        )[:LAZY_CHUNK_NODE_LIMIT]
        try:
            for node in pending:
                self._rechunk_node(node)
        except Exception as error:
            # Each node's index write is atomic. Stop this drain on failure;
            # missing chunks stay pending for the next recall. Required reads
            # and scoring below still run, with no invented semantic evidence.
            logging.getLogger(__name__).warning(
                "lazy chunk indexing deferred (%s); using available index",
                type(error).__name__,
            )

        q_arr = _as_query_array(query_embedding)
        # Resolve membership after the drain, which can write new chunks.
        scoped_scores: list[tuple[float, str]] = []
        matched_chunks: dict[str, str] = {}
        for block in self._chunk_blocks(plan, len(query_embedding)):
            for similarity, node_id in _pooled_chunk_similarities(
                block, query_embedding, q_arr, matched_chunks=matched_chunks,
            ):
                if similarity >= VECTOR_MATCH_THRESHOLD:
                    scoped_scores.append((similarity, node_id))

        if not scoped_scores:
            return query_embedding

        scoped_scores.sort(key=lambda item: item[0], reverse=True)
        keep_n = per_scope_limit * max(1, len(plan.scopes))
        # The limit controls vector discovery, not evidence on a node already
        # admitted lexically. Its content similarity was computed above too;
        # discarding it would turn a weaker semantic match into a false zero.
        # No additional candidate or embedding is introduced by this union.
        kept = [
            item for rank, item in enumerate(scoped_scores)
            if rank < keep_n or item[1] in candidates
        ]
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
            if similarity > existing.vector_score:
                existing.vector_score = similarity
                existing.chunk_id = matched_chunks.get(node_id)
        return query_embedding

    # ------------------------------------------------------------------
    # Chunk corpus cache
    # ------------------------------------------------------------------

    def _chunk_blocks(self, plan: ScopePlan, dimension: int) -> tuple[_ChunkBlock, ...]:
        """Share scope partitions, reading revisions and rows in one snapshot.

        The existing deferred transaction helper reserves no writer. Its read
        snapshot prevents a concurrent move from appearing in two partitions.
        An outer caller transaction may contain writes that roll back, so its
        rows are never retained. This snapshot covers only the vector read;
        metadata, corrections and source validity remain live recall inputs.

        Broad queries temporarily gather rows in node/chunk order. Keeping the
        full width block's normalization and matmul shape preserves scores and
        stable candidate ties. Row placement is retained until its source blocks
        change; there is no second persistent corpus matrix.
        """

        cacheable = not self.store.connection.in_transaction
        with self.store.transaction():
            revisions = self.store.chunk_corpus_revisions(
                plan.scopes if plan.restricted else None,
            )
            if revisions is None:
                # Old read-only snapshots have no trustworthy cache identity.
                self._chunk_index.clear()
                self._chunk_gathers.clear()
                indexes = [self._build_chunk_index(scope, None) for scope in plan.search_scopes]
            else:
                scopes = plan.scopes if plan.restricted else revisions
                indexes = [
                    self._scope_chunk_index(scope, revisions[scope], cacheable=cacheable)
                    for scope in scopes
                ]
        blocks = tuple(block for index in indexes for block in index.blocks
                       if block.dimension == dimension)
        if len(blocks) > 1 and not plan.restricted and revisions is not None:
            gather = self._chunk_gathers.get(dimension) if cacheable else None
            if gather is None or len(gather.blocks) != len(blocks) or any(
                old is not new for old, new in zip(gather.blocks, blocks)
            ):
                gather = _prepare_chunk_gather(blocks)
                if cacheable:
                    self._chunk_gathers[dimension] = gather
            return (gather.gather(),)
        return blocks

    def _scope_chunk_index(
        self, scope: str, revision: int, *, cacheable: bool,
    ) -> _ScopeChunkIndex:
        if not cacheable:
            return self._build_chunk_index(scope, None)
        cached = self._chunk_index.get(scope)
        if cached is not None and cached.revision == revision:
            return cached
        index = self._build_chunk_index(scope, revision)
        # Release layouts referencing replaced arrays, even on scoped queries.
        for block in (*cached.blocks, *index.blocks) if cached else index.blocks:
            self._chunk_gathers.pop(block.dimension, None)
        self._chunk_index[scope] = index
        return index

    def _build_chunk_index(self, scope: str | None, revision: int | None) -> _ScopeChunkIndex:
        """Read a scope's chunk BLOBs into reusable blocks by vector width.

        Byte width determines the block. Malformed or empty BLOBs are skipped;
        only the width matching the query is scored. The ordered storage scan
        keeps each node's chunks together for max-pooling.
        """

        accumulators: dict[int, _BlockAccumulator] = {}
        width = -1
        accumulator: _BlockAccumulator | None = None
        for node_id, _ordinal, view, chunk_id in self.store.iter_chunk_embedding_rows(
            scope=scope, include_identity=True,
        ):
            row_width, remainder = divmod(len(view), CHUNK_EMBEDDING_ITEMSIZE)
            if remainder or not row_width:
                # Not a whole number of float32s, or none at all: not a vector.
                continue
            if row_width != width or accumulator is None:
                width = row_width
                accumulator = accumulators.setdefault(width, _BlockAccumulator())
            accumulator.add(node_id, view, chunk_id)

        blocks = tuple(accumulators[width].freeze(width) for width in sorted(accumulators))
        return _ScopeChunkIndex(
            revision=revision,
            blocks=blocks,
            node_count=sum(len(block.node_ids) for block in blocks),
            row_count=sum(block.row_count for block in blocks),
        )

    # ------------------------------------------------------------------
    # Query anchors: the graph channel's entry from query space
    # ------------------------------------------------------------------

    def _match_query_anchors(
        self, plan: ScopePlan, query_embedding: list[float]
    ) -> list[AnchorMatch]:
        """Observe matches and graph targets for this recall only."""

        if not query_embedding:
            return []
        source = self._anchor_vector_source()
        if source is None:
            return []
        return match_anchors(
            source,
            query_embedding,
            plan,
            limit=ANCHOR_MATCH_LIMIT,
            min_similarity=ANCHOR_MATCH_COSINE_THRESHOLD,
        )

    def _collect_anchor_seeds(
        self, plan: ScopePlan, query_embedding: list[float],
        *, matches: list[AnchorMatch] | None = None,
    ) -> dict[str, float]:
        """Graph seeds contributed by the anchors this query matches.

        ``{node_id: activation}``, where activation is the matched anchor's
        cosine times the weight of its edge to that node. Both factors are in
        [0, 1] and their product is a confidence, so the value lands in the
        same range ``_collect_graph``'s existing seeds occupy -- a bm25 rank
        score, a cosine -- and blends through the same learned ``graph``
        weight. A node reachable from two matched anchors keeps the stronger
        claim rather than accumulating, so a query that happens to sit near
        several anchors cannot manufacture activation the edges do not carry.

        Scope is the ``ScopePlan``'s, handed to ``match_anchors`` so the SQL
        never returns an anchor from outside it: an anchor seeds only where its
        own scope is admitted. The targets it names are filtered against the
        same plan downstream, exactly as every other channel's candidates are.

        Returns ``{}`` on a pre-v7 database, with anchor seeding switched off,
        with no live anchor, and when nothing clears the match floor.
        """

        if not self.anchor_seeding or not query_embedding:
            return {}
        if matches is None:
            matches = self._match_query_anchors(plan, query_embedding)
        seeds: dict[str, float] = {}
        for match in matches:
            for target_id, weight in match.targets:
                activation = match.similarity * min(1.0, max(0.0, weight))
                if activation > seeds.get(target_id, 0.0):
                    seeds[target_id] = activation
        if len(seeds) <= GRAPH_SEED_LIMIT:
            return seeds
        # The same bound the seed block applies to every other seed, applied
        # before the node fetch rather than after it: a pathological anchor
        # with hundreds of edges must not turn into hundreds of get_nodes rows
        # that the sort would then discard anyway.
        return dict(
            sorted(seeds.items(), key=lambda item: (-item[1], item[0]))[:GRAPH_SEED_LIMIT]
        )

    def _collect_query_demotions(
        self, plan: ScopePlan, query_embedding: list[float],
        *, matches: list[AnchorMatch] | None = None,
    ) -> dict[str, float]:
        """Query-relative irrelevance multipliers for this recall. Never raises.

        Active only under ``LM_EXPLICIT_FEEDBACK_POLICY=credit`` -- the valve
        that writes the rows also gates reading them, so switching it back to
        ``audit`` or ``off`` restores the undemoted ranking without touching
        the store -- and only when ``LM_QUERY_IRRELEVANCE_FACTOR`` < 1. Anchor
        matching is ``match_anchors`` with the anchor channel's floor and
        limit. Matches already observed by graph seeding are reused within
        this recall, including an empty result. Otherwise matching is independent
        of graph depth and ``anchor_seeding``. The probe for any live row runs
        first, so a store that never saw a credited irrelevant mark pays one
        indexed lookup and no anchor scan.
        """

        if not query_embedding or _explicit_feedback_policy() != "credit":
            return {}
        try:
            if matches is not None:
                return query_demotions(self.store, query_embedding, plan, matches=matches)
            source = self._anchor_vector_source()
            if source is None:
                return {}
            return query_demotions(source, query_embedding, plan)
        except Exception:  # pragma: no cover - derived signal, never fatal
            return {}

    def _collect_hub_demotions(self) -> dict[str, float]:
        """Global hub multipliers (``living_memory.hubs``). Never raises."""

        try:
            return hub_demotions(self.store)
        except Exception:  # pragma: no cover - derived signal, never fatal
            return {}

    def _anchor_vector_source(self) -> _AnchorVectorCache | MemoryStore | None:
        """The live anchor corpus, cached until ``query_anchor_revision`` moves.

        One indexed revision lookup guards reuse, including an empty cache.
        The live count is checked only on rebuild, so an empty corpus never
        scans vectors. Unmigrated read-only stores use the uncached reader.
        Uncommitted inputs must not enter the cache: rollback can restore a
        revision that another mutation will reuse for different vectors.
        """

        revision = self.store.query_anchor_revision()
        if revision is None or self.store.connection.in_transaction:
            self._anchor_vectors = None
            return self.store
        cached = self._anchor_vectors
        if cached is None or cached.revision != revision:
            self._anchor_vectors = None
            cached = None
            grouped: dict[int, list[tuple[str, str, int, Any]]] = {}
            if self.store.count_query_anchors(include_decayed=False):
                for row in self.store.iter_query_anchor_vectors():
                    grouped.setdefault(row[2], []).append(row)
            by_width = {}
            while grouped:
                width, rows = grouped.popitem()
                by_width[width] = _prepare_anchor_vectors(rows, width, index_scopes=True)
            cached = _AnchorVectorCache(revision, self.store, by_width)
            self._anchor_vectors = cached
        return cached if cached.by_width else None

    def _admit_anchor_seeds(
        self,
        plan: ScopePlan,
        candidates: dict[str, _Candidate],
        anchor_seeds: dict[str, float],
    ) -> dict[str, float]:
        """Make every anchor-seeded target a candidate, and report what stuck.

        An anchor's target is usually a node no other channel found -- that is
        the point -- so it has to be fetched and admitted before the anchor
        walk can start from it. Admission uses the ranking loop's own liveness
        and scope rules, so a seed can never carry a decayed or out-of-plan
        node into the walk that the ranker would then drop.

        Called only *after* the pre-anchor walk has finished, so a target
        admitted here cannot compete for one of that walk's seed slots. That
        ordering is load-bearing: it is what keeps ``_Candidate.graph_score``
        equal to its anchors-off value, which is the number the ranker's
        monotonicity fallback compares against.
        """

        if not anchor_seeds:
            return {}
        missing = [node_id for node_id in anchor_seeds if node_id not in candidates]
        fetched = self.store.get_nodes(missing) if missing else {}
        admitted: dict[str, float] = {}
        for node_id, activation in anchor_seeds.items():
            existing = candidates.get(node_id)
            node = existing.node if existing is not None else fetched.get(node_id)
            if node is None or node.decayed or not plan.allows(node.scope):
                continue
            if existing is None:
                candidates[node_id] = _Candidate(node=node)
            admitted[node_id] = activation
        return admitted

    def _collect_schema_triggers(
        self,
        query: str,
        plan: ScopePlan,
        candidates: dict[str, _Candidate],
    ) -> None:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return
        scores: dict[str, float] = {}
        coverages: dict[str, float] = {}
        for scope in plan.search_scopes:
            for schema_id, trigger in self.store.schema_triggers(scope):
                trigger_tokens = set(tokenize(trigger))
                if not trigger_tokens:
                    continue
                matching = query_tokens & trigger_tokens
                overlap = len(matching) / len(trigger_tokens)
                # A trigger is an applicability hint, including when a short
                # instruction is one clause of a longer question. Ranking
                # separately asks whether the content channels support it.
                if overlap < SCHEMA_TRIGGER_OVERLAP_THRESHOLD:
                    continue
                scores[schema_id] = SCHEMA_TRIGGER_BASE_SCORE + 0.05 * overlap
                # The trigger is evidence about this question only to the
                # extent that it covers the question's meaningful terms.
                # tokenize() removes procedural filler such as "how to".
                coverage = len(matching) / len(query_tokens)
                coverages[schema_id] = max(coverages.get(schema_id, 0.0), coverage)
        for schema in self.store.get_nodes(scores).values():
            candidate = candidates.setdefault(schema.id, _Candidate(node=schema))
            candidate.trigger_score = max(candidate.trigger_score, scores[schema.id])
            candidate.trigger_coverage = max(candidate.trigger_coverage, coverages[schema.id])

    def _named_schemas_first(
        self,
        query: str,
        plan: ScopePlan,
        ranked: list[RecallResult],
        *,
        demotions: Mapping[str, float] | None,
        causal_mode: bool,
    ) -> list[RecallResult]:
        """``name`` mode: the schemas this query names, if they earned a slot, go first.

        A schema is named when the query's token set equals its trigger's:
        the whole query is its title, the lexical channel's best match (bm25
        1.0). Only schemas the meaning channels already ranked can be named,
        and a named schema moves to the front only if
        :func:`score_gate.passes_gate` passes it on that score (with the gate
        off, always). Named schemas keep their ranked order among themselves
        and are marked with ``trigger_score``/``"trigger"`` for the wire.
        """

        query_tokens = frozenset(tokenize(query))
        if not query_tokens or not ranked:
            return ranked
        named_ids = {
            node_id
            for scope in plan.search_scopes
            for node_id, trigger in self.store.schema_triggers(scope)
            if frozenset(tokenize(trigger)) == query_tokens
        }
        threshold = min_score_from_env()
        first: list[RecallResult] = []
        rest: list[RecallResult] = []
        for result in ranked:
            if result.node.id not in named_ids:
                rest.append(result)
                continue
            result = replace(
                result,
                bm25_score=1.0,
                trigger_score=SCHEMA_NAME_TRIGGER_SCORE,
                methods=(*result.methods, "trigger"),
            )
            if threshold is None or passes_gate(
                result, threshold, demotions, plan=plan, causal_mode=causal_mode
            ):
                first.append(result)
            else:
                rest.append(result)
        return first + rest

    def _collect_graph(
        self,
        plan: ScopePlan,
        candidates: dict[str, _Candidate],
        *,
        max_depth: int,
        causal_mode: bool,
        decision_mode: bool = False,
        anchor_seeds: dict[str, float] | None = None,
    ) -> None:
        """Score candidates by graph proximity, in two separable walks.

        The first walk is the one that existed before anchors: it seeds from
        the top pre-graph candidates and writes ``graph_score``. Nothing in it
        reads an anchor, and it runs *before* any anchor target is admitted as
        a candidate, so it computes the same numbers whether anchors are on or
        off. That is what makes ``graph_score`` the anchors-off counterfactual
        the ranker needs -- see "Anchors never demote what they seed" in the
        module docstring.

        The second walk is the anchors', seeded from the matched anchors'
        targets and writing ``anchor_score``. Keeping the two apart is the
        whole fix: merged into one seed block, an anchor target could displace
        a real seed, and an anchor-derived activation was indistinguishable
        from a traversed one, so the ranker had no way to tell what a
        candidate had been worth before the anchor named it.
        """

        # A bulk query's tie order depends on its entire frontier, so reuse
        # observations only for the same ordered request. Both walks share
        # these bounded, directional neighbors, never their score state.
        observations: dict[
            tuple[str, ...],
            tuple[dict[str, list[tuple[str, float]]], dict[str, list[tuple[str, float]]]],
        ] = {}
        observed_state: tuple[int, int] | None = None
        historical_edges = False

        def neighbors_for_nodes(
            node_ids: list[str], *, source_only: bool = False,
        ) -> dict[str, list[tuple[str, float]]]:
            nonlocal observed_state, historical_edges
            connection = self.store.connection
            state = (
                connection.total_changes,
                connection.execute("PRAGMA data_version").fetchone()[0],
            )
            # Transaction-local reads may be rolled back without changing
            # total_changes again. Never retain them for a later level/pass.
            if connection.in_transaction or state != observed_state:
                observations.clear()
            observed_state = None if connection.in_transaction else state
            key = tuple(node_ids)
            if key not in observations:
                connections = self.store.list_recall_connections(
                    node_ids, limit=MAX_NEIGHBORS_PER_NODE
                )
                ordinary = {node_id: [] for node_id in connections}
                source = {node_id: [] for node_id in connections}
                for node_id, edges in connections.items():
                    for edge in edges:
                        traversal = _traversal(
                            edge, node_id, causal_mode=causal_mode,
                            decision_mode=decision_mode,
                        )
                        if traversal is None:
                            continue
                        ordinary[node_id].append(traversal)
                        weight = source_relation_weight(edge.type, edge.weight, edge.metadata)
                        if weight == edge.weight:
                            source[node_id].append(traversal)
                        else:
                            historical_edges = True
                            if weight > 0.0:
                                neighbor, activation = traversal
                                source[node_id].append((neighbor, activation * weight / edge.weight))
                observations[key] = ordinary, source
            return observations[key][int(source_only)]

        # Seed BFS only from the top-K pre-graph candidates to avoid
        # fan-out from low-quality matches that would dilute the graph
        # signal and waste cycles on super-hub expansions.
        seed_candidates = sorted(
            candidates.values(),
            key=lambda c: max(c.bm25_score, c.vector_score, 0.0),
            reverse=True,
        )[:GRAPH_SEED_LIMIT]
        roots = [
            (candidate.node.id, max(candidate.bm25_score, candidate.vector_score))
            for candidate in seed_candidates
        ]
        self._walk_graph(
            plan, candidates, roots,
            max_depth=max_depth,
            neighbors_for_nodes=neighbors_for_nodes,
            anchor_pass=False,
        )

        if historical_edges:
            # Reuse the bounded walk with independent best-path state: a
            # stronger historical path must not hide a weaker source path.
            # Extra source-only discoveries do not change ranking membership.
            self._walk_graph(
                plan, dict(candidates), roots,
                max_depth=max_depth,
                neighbors_for_nodes=lambda ids: neighbors_for_nodes(ids, source_only=True),
                anchor_pass=False, source_pass=True,
            )
        else:
            for candidate in candidates.values():
                candidate.source_graph_score = candidate.graph_score

        # An anchor's targets are admitted only now, so they cannot compete
        # for a slot in the walk above or perturb a single one of its numbers.
        anchor_activations = self._admit_anchor_seeds(plan, candidates, anchor_seeds or {})
        if not anchor_activations:
            return
        anchor_roots: list[tuple[str, float]] = []
        for node_id, activation in sorted(
            anchor_activations.items(), key=lambda item: (-item[1], item[0])
        )[:GRAPH_SEED_LIMIT]:
            # The anchor edge is itself the evidence -- "this question used
            # this node" -- so the target scores in the graph channel rather
            # than merely opening a walk from it. Under the same cap as a
            # BFS-discovered neighbour, so both writers of a graph activation
            # agree on its ceiling.
            candidate = candidates[node_id]
            candidate.anchor_score = max(candidate.anchor_score, min(1.5, activation))
            anchor_roots.append((node_id, activation))
        self._walk_graph(
            plan,
            candidates,
            anchor_roots,
            max_depth=max_depth,
            neighbors_for_nodes=neighbors_for_nodes,
            anchor_pass=True,
        )

    def _walk_graph(
        self,
        plan: ScopePlan,
        candidates: dict[str, _Candidate],
        roots: list[tuple[str, float]],
        *,
        max_depth: int,
        neighbors_for_nodes: Callable[[list[str]], dict[str, list[tuple[str, float]]]],
        anchor_pass: bool,
        source_pass: bool = False,
    ) -> None:
        """One breadth-first activation spread from ``roots``.

        ``anchor_pass`` selects which field the activation lands in and
        nothing else: the traversal, its bounds, and its arithmetic are one
        implementation, so the anchor walk can never drift from the walk it is
        the counterfactual for.
        """

        queue: deque[tuple[str, float, int, tuple[str, ...]]] = deque()
        best_seen: dict[str, float] = {}
        for node_id, activation in roots:
            seed = max(activation, 0.05)
            queue.append((node_id, seed, 0, (node_id,)))
            best_seen[node_id] = seed

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

            node_ids_at_depth = list({item[0] for item in level_items})
            neighbors_by_node = neighbors_for_nodes(node_ids_at_depth)

            # Prepare each path-dependent transition once, in traversal order.
            # best_seen must also be checked when consuming the transitions:
            # an earlier item at this level can strengthen the same neighbor.
            transitions = []
            needed_neighbor_ids = set()
            for node_id, activation, depth, path in level_items:
                for neighbor_id, relation_score in neighbors_by_node[node_id]:
                    if neighbor_id in path:
                        continue

                    next_score = activation * relation_score * (0.72 ** depth)
                    if next_score <= best_seen.get(neighbor_id, 0.0):
                        continue

                    transitions.append((neighbor_id, next_score, depth + 1, path + (neighbor_id,)))
                    if neighbor_id not in node_cache and neighbor_id not in candidates:
                        needed_neighbor_ids.add(neighbor_id)

            # Batch fetch missing neighbor nodes
            if needed_neighbor_ids:
                fetched_nodes = self.store.get_nodes(needed_neighbor_ids)
                for nid in needed_neighbor_ids:
                    node_cache[nid] = fetched_nodes.get(nid)

            for neighbor_id, next_score, depth, path in transitions:
                if next_score <= best_seen.get(neighbor_id, 0.0):
                    continue

                neighbor = (
                    node_cache[neighbor_id] if neighbor_id in node_cache
                    else candidates[neighbor_id].node
                )
                if neighbor is None or neighbor.decayed or not plan.allows(neighbor.scope):
                    continue

                best_seen[neighbor_id] = next_score
                candidate = candidates.setdefault(neighbor_id, _Candidate(node=neighbor))
                if source_pass:
                    candidate.source_graph_score = max(candidate.source_graph_score, min(1.5, next_score))
                elif anchor_pass:
                    reached = min(1.5, next_score)
                    # Provenance belongs to the strongest explanation across
                    # the ordinary and anchor walks.
                    if reached > candidate.combined_graph_score:
                        candidate.path = path
                    candidate.anchor_score = max(candidate.anchor_score, reached)
                else:
                    candidate.graph_score = max(candidate.graph_score, min(1.5, next_score))
                    candidate.path = path
                queue.append((neighbor_id, next_score, depth, path))

    def _rechunk_node(self, node: Node) -> None:
        """Refresh content search without overwriting a cluster representative."""

        self._share_embedder_with_store()
        if node.materializer in ("cluster", "cross-scope-promotion") and node.embedding:
            self.store.update_node(node.id, representative_embedding=node.embedding)
        else:
            embedding = self.embedder.embed(node.content)
            updated = self.store.update_node(node.id, embedding=embedding)
            node.embedding = updated.embedding

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
        batch = getattr(embedder, "embed_batch", None)
        share(batch if callable(batch) else lambda texts: [embedder.embed(text) for text in texts])


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


def _merge_demotions(*sources: Mapping[str, float]) -> dict[str, float]:
    """Per-node multipliers from several sources; the strongest (smallest) wins."""

    merged: dict[str, float] = {}
    for source in sources:
        for node_id, multiplier in source.items():
            if multiplier < merged.get(node_id, 1.0):
                merged[node_id] = multiplier
    return merged


def _recall_result_summary(index: int, result: RecallResult) -> dict[str, Any]:
    summary = _recall_result_summary_fields(index, result)
    if result.withheld:
        summary["withheld"] = result.withheld
    return summary


def _recall_result_summary_fields(index: int, result: RecallResult) -> dict[str, Any]:
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
    """The BM25 query: the raw text plus its normalized tokens and prefix terms.

    The raw query keeps every surface form (``unicode61`` indexes surface
    forms), ``tokenize`` adds the canonical tokens the synonym table maps to
    ("миграции" -> "schema"), and :func:`cyrillic_prefix_terms` adds one FTS5
    prefix term per Russian content word ("миграц*") so that every inflection
    of the word in the index matches, not just the one spelled in the query.
    A Cyrillic stem that already has a prefix term is not repeated as an
    exact term -- the index never holds a bare stem. With
    ``LM_TOKENIZE_CYRILLIC_STEM=off`` there are no prefix terms and the query
    is exactly what it was before Cyrillic stemming existed.
    """

    expanded_tokens = tokenize(query)
    prefix_terms = cyrillic_prefix_terms(query)
    if not expanded_tokens and not prefix_terms:
        return query
    stems = {term[:-1] for term in prefix_terms}
    parts = [token for token in expanded_tokens if token not in stems] + prefix_terms
    return f"{query} {' '.join(dict.fromkeys(parts))}"


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

    Only a restricted plan gates at all: a whole-store plan asked for no
    narrower scope, so every scope in it is a peer ranked on its evidence.
    """

    if scope_family(plan.requested_scope) != "session":
        return (plan.requested_scope,)
    return (plan.requested_scope,) + tuple(
        scope
        for scope in plan.scopes
        if scope != plan.requested_scope and scope_family(scope) == "project"
    )


def _has_anchor_free_evidence(candidate: _Candidate) -> bool:
    """Whether this candidate would exist at all with anchors switched off.

    Every channel except the anchor entry, plus the traversal that runs before
    any anchor target is admitted. False means the candidate is in the running
    solely because an anchor named it -- it has no anchors-off score, so there
    is nothing for the monotonicity fallback to fall back to.
    """

    return (
        candidate.bm25_score > 0.0
        or candidate.vector_score > 0.0
        or candidate.trigger_score > 0.0
        or candidate.graph_score > 0.0
    )


def _blend_candidate_score(
    candidate: _Candidate,
    graph_score: float,
    weights: RetrievalWeights,
    *,
    plan: ScopePlan,
    causal_mode: bool,
    superseded: bool,
    superseding: bool,
    positive_history: bool = True,
) -> float:
    """A candidate's final score for one given graph activation.

    Takes ``graph_score`` as an argument rather than reading it off the
    candidate because the ranker scores some candidates twice -- once with the
    activation an anchor contributed and once without it -- and the two
    scorings have to be the same function of it, floor included, or the
    comparison between them proves nothing.

    Returns 0.0 for a candidate no channel scored, which is the caller's cue
    to drop it.
    """

    bm25_weight = weights.bm25
    vector_weight = weights.vector
    graph_weight = weights.graph
    if graph_score > 0.0:
        minimum_graph = 0.75 if causal_mode else 0.25
        if graph_weight < minimum_graph:
            deficit = minimum_graph - graph_weight
            graph_weight = minimum_graph
            remaining = max(0.0, bm25_weight + vector_weight)
            if remaining > 0.0:
                bm25_weight = max(0.0, bm25_weight - deficit * (bm25_weight / remaining))
                vector_weight = max(0.0, vector_weight - deficit * (vector_weight / remaining))

    node = candidate.node
    base_score = (
        bm25_weight * candidate.bm25_score
        + vector_weight * candidate.vector_score
        + graph_weight * graph_score
    )
    if candidate.vector_score >= STRONG_VECTOR_MATCH:
        base_score = max(base_score, candidate.vector_score)
    if node.level != "trace" and candidate.trigger_score > 0.0:
        # Keep a trigger-only instruction available, but do not award every
        # historical carrier a near-perfect score for a common label. Vector
        # evidence about the content supplies the discovery floor; coverage
        # of the meaningful query supplies a separate applicability claim.
        base_score = max(
            base_score,
            candidate.trigger_score * max(candidate.vector_score, weights.graph),
        )
        # A saved trigger that covers the whole meaningful query is direct
        # evidence of applicability. Extra query clauses rapidly reduce this
        # contribution unless content/graph evidence supports the carrier.
        # The channel weights keep it on the ordinary blend's scale. Taking
        # the maximum avoids counting the same lexical claim twice.
        # Schemas carry reusable recipes. Other procedural carriers earn
        # this discovery route only while citing source records; a matching
        # trigger on an arbitrary concept does not establish that support.
        if node.level == "schema" or (
            node.provenance.get("strategy") == "procedural" and node.source_traces
        ):
            base_score = max(
                base_score,
                candidate.trigger_score
                * candidate.trigger_coverage**2
                * (bm25_weight + vector_weight / 2),
            )
    if base_score <= 0.0:
        return 0.0

    scope_boost = (
        1.0 + plan.boost_steps(node.scope) * SCOPE_RANK_BOOST_STEP
    )
    adjusted = feedback_weighted_score(
        node,
        base_score * scope_boost,
        superseded=superseded,
        superseding=superseding,
        positive_history=positive_history,
    )
    if causal_mode and graph_score > 0.0:
        adjusted *= 1.5
    return adjusted


def _ungated_scope_profile(
    candidates: dict[str, _Candidate],
    ungated_scopes: tuple[str, ...],
    *,
    decision_mode: bool = False,
    anchor_free: bool = False,
) -> tuple[bool, float]:
    """Whether any rankable ungated-scope candidate exists, and its best vector.

    Only candidates that survive the same pre-score skips as the ranking loop
    (decayed, rejected-alternative outside decision mode) count: a candidate
    that can never rank must not arm the cross-scope gate, or the gate could
    empty a recall whose only real answers are cross-scope.

    ``anchor_free`` additionally ignores candidates that exist only because an
    anchor named them, which is what the gate looked like before the anchor
    matched -- an anchor must not arm a gate that drops somebody else.
    """

    present = False
    best_vector = 0.0
    for candidate in candidates.values():
        node = candidate.node
        if node.scope not in ungated_scopes or node.decayed:
            continue
        if _is_rejected_alternative_node(node) and not decision_mode:
            continue
        if anchor_free and not _has_anchor_free_evidence(candidate):
            continue
        present = True
        if candidate.vector_score > best_vector:
            best_vector = candidate.vector_score
    return present, best_vector


def _cross_scope_admissible(
    candidate: _Candidate,
    best_narrow_vector: float,
    *,
    graph_score: float,
) -> bool:
    """Deliberate evidence that admits a candidate from outside the ungated scopes.

    bm25 rank scores never qualify (per-scope normalization makes them
    incomparable across scopes; see the gate constants above). ``graph_score``
    is passed in for the same reason ``_blend_candidate_score`` takes it: the
    anchors-off counterfactual has to be able to ask this question about the
    activation it is scoring, not about the one the anchor added.
    """

    if candidate.trigger_score > 0.0:
        return True
    if graph_score >= CROSS_SCOPE_GRAPH_ADMIT:
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
    """Collect ordered BLOBs once, retaining one raw array per vector width."""

    __slots__ = ("payload", "node_ids", "starts", "rows", "chunk_ids")

    def __init__(self) -> None:
        self.payload = bytearray()
        self.node_ids: list[str] = []
        self.starts: list[int] = []
        self.rows = 0
        self.chunk_ids: list[str] = []

    def add(self, node_id: str, view: Any, chunk_id: str) -> None:
        # iter_chunk_embedding_rows promises rows grouped by node, so a node id
        # that differs from the last one can only mean a new run.
        if not self.node_ids or self.node_ids[-1] != node_id:
            self.node_ids.append(node_id)
            self.starts.append(self.rows)
        self.payload += view
        self.chunk_ids.append(chunk_id)
        self.rows += 1

    def freeze(self, dimension: int) -> _ChunkBlock:
        node_ids = tuple(self.node_ids)
        if _np is not None:
            # The bytearray remains alive as the array's base; no BLOB list or
            # second persistent matrix is retained.
            matrix = _np.frombuffer(self.payload, dtype=CHUNK_EMBEDDING_DTYPE).reshape(
                self.rows, dimension
            )
            # einsum here is a fused row-wise self-dot: the row norms without a
            # temporary copy of the matrix.
            norms = _np.sqrt(_np.einsum("ij,ij->i", matrix, matrix))
            norm_deviation = float(_np.abs(norms - 1.0).max()) if self.rows else 0.0
            norms[norms == 0.0] = 1.0
            starts = _np.asarray(self.starts, dtype=_np.intp)
            counts = _np.diff(_np.append(starts, self.rows)).astype(_np.float32)
            return _ChunkBlock(
                dimension=dimension,
                node_ids=node_ids,
                starts=starts,
                rows=matrix,
                row_count=self.rows,
                log2_chunk_counts=_np.log2(counts),
                norms=norms,
                norm_deviation=norm_deviation,
                chunk_ids=tuple(self.chunk_ids),
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
            chunk_ids=tuple(self.chunk_ids),
        )


def _prepare_chunk_gather(blocks: tuple[_ChunkBlock, ...]) -> _ChunkGatherPlan:
    """Prepare the storage scan's exact row order once for these source blocks."""

    runs = sorted(
        (node_id, block, index)
        for block in blocks for index, node_id in enumerate(block.node_ids)
    )
    node_ids = tuple(node_id for node_id, _, _ in runs)
    counts = [
        (int(block.starts[index + 1]) if index + 1 < len(block.node_ids) else block.row_count)
        - int(block.starts[index])
        for _, block, index in runs
    ]
    row_count = sum(counts)
    dimension = blocks[0].dimension
    chunk_ids = tuple(
        chunk_id
        for (_, block, index), count in zip(runs, counts, strict=True)
        for chunk_id in block.chunk_ids[int(block.starts[index]):int(block.starts[index]) + count]
    )
    if _np is not None and isinstance(blocks[0].rows, _np.ndarray):
        starts = _np.cumsum([0, *counts[:-1]], dtype=_np.intp)
        destinations = dict(zip(node_ids, starts, strict=True))
        norms = _np.empty(row_count, dtype=_np.float32)
        placements = []
        for block in blocks:
            offsets = _np.asarray([destinations[node_id] for node_id in block.node_ids])
            shifts = offsets - block.starts
            if _np.all(shifts == shifts[0]):
                positions = slice(int(shifts[0]), int(shifts[0]) + block.row_count)
            else:
                lengths = _np.diff(_np.append(block.starts, block.row_count))
                positions = _np.repeat(shifts, lengths) + _np.arange(block.row_count)
            placements.append(positions)
            norms[positions] = block.norms
        template = _ChunkBlock(
            dimension, node_ids, starts, None, row_count,
            _np.asarray([block.log2_chunk_counts[index] for _, block, index in runs]),
            norms, float(_np.max([block.norm_deviation for block in blocks])),
            chunk_ids=chunk_ids,
        )
        return _ChunkGatherPlan(blocks, tuple(placements), template)
    starts_list = []
    scalar_rows = []
    for (_, block, index), count in zip(runs, counts, strict=True):
        starts_list.append(len(scalar_rows))
        start = block.starts[index]
        scalar_rows.extend(block.rows[start:start + count])
    template = _ChunkBlock(
        dimension, node_ids, tuple(starts_list), tuple(scalar_rows), row_count,
        tuple(block.log2_chunk_counts[index] for _, block, index in runs),
        chunk_ids=chunk_ids,
    )
    return _ChunkGatherPlan(blocks, (), template)


def _pooled_chunk_similarities(
    block: _ChunkBlock,
    query_embedding: list[float],
    q_arr: Any,
    *,
    matched_chunks: dict[str, str] | None = None,
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
        rows = block.rows / block.norms[:, None] if block.needs_normalization else block.rows
        sims = rows @ q_arr
        # `starts` is the run boundary of each node, so one reduceat pools
        # every node at once -- no Python loop over the corpus's 60k chunk rows.
        maxima = _np.maximum.reduceat(sims, block.starts)
        pooled = _np.clip(maxima, 0.0, None)
        if matched_chunks is not None:
            counts = _np.diff(_np.append(block.starts, block.row_count))
            winners = _np.minimum.reduceat(
                _np.where(sims == _np.repeat(maxima, counts), _np.arange(block.row_count), block.row_count),
                block.starts,
            )
            matched_chunks.update(
                (node_id, block.chunk_ids[int(row)])
                for node_id, row in zip(block.node_ids, winners, strict=True)
                if row < block.row_count
            )
        if LENGTH_BIAS_LOG2_COEFFICIENT:
            pooled -= LENGTH_BIAS_LOG2_COEFFICIENT * block.log2_chunk_counts
            _np.clip(pooled, 0.0, None, out=pooled)
        return list(zip(pooled.tolist(), block.node_ids, strict=True))

    bounds = tuple(block.starts) + (block.row_count,)
    pooled_rows: list[tuple[float, str]] = []
    for index, node_id in enumerate(block.node_ids):
        best = 0.0
        winner = bounds[index]
        for position in range(bounds[index], bounds[index + 1]):
            similarity = cosine_similarity(query_embedding, list(block.rows[position]))
            if similarity > best:
                best, winner = similarity, position
        if matched_chunks is not None:
            matched_chunks[node_id] = block.chunk_ids[winner]
        penalty = LENGTH_BIAS_LOG2_COEFFICIENT * block.log2_chunk_counts[index]
        pooled_rows.append((max(0.0, best - penalty), node_id))
    return pooled_rows
