"""Recall service combining scope, FTS5, embeddings, graph traversal, and feedback.

The vector channel pools the best matching chunk per node and applies a
length correction proportional to the logarithm of the chunk count.

A query anchor links a remembered question to its grounded targets. Matching
a new question against anchor vectors lets «сверь патч LX-42018» reach a
procedure used for «сверь патч LX-42017», even across languages. Matched targets
seed a separate graph walk with activation equal to cosine times edge weight.
The ranker keeps the better of the blends with and without anchor evidence for
candidates reached independently. Anchor vectors are cached by store revision.
"""

from __future__ import annotations

import json
import logging
import os
import re
from bisect import bisect_left
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from heapq import heappush, heapreplace, nlargest, nsmallest
from itertools import groupby, zip_longest
from operator import itemgetter
from math import log2, sqrt
from typing import Any, Callable

from living_memory.consolidation import memory_connect
from living_memory.edge_derivation import CONTENT_REFERENCE_KIND, DERIVED_FROM_KIND
from living_memory.embeddings import (
    EncodedVector,
    LocalEmbeddingModel,
    cosine_similarity,
    cyrillic_prefix_terms,
    distinct_evidence_tokens,
    repair_search_terms,
    tokenize,
)
from living_memory.feedback import (
    FeedbackService,
    _explicit_feedback_policy,
    feedback_weighted_score,
)
from living_memory.hubs import hub_demotions
from living_memory.irrelevance import has_active_irrelevance, query_demotions, query_irrelevance_factor
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
    _rank_anchors,
    match_anchors,
)
from living_memory.schema_dedup import collapse_schema_duplicates
from living_memory.relations import is_semantic_relation, needs_source_projection
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
    _content_fingerprint,
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
# it the broad tail of a narrow-scope plan can crowd out max_results.
# Lexical overlap, including a matching task label, alone does not establish
# cross-scope applicability. Deliberate evidence does: an applicable schema trigger, a strong
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
LAZY_CHUNK_PLAN_LIMIT = 128
LEGACY_PASSAGE_MAX_CHARS = 32_768
LEGACY_PASSAGE_MAX_CHUNKS = 32

@dataclass(frozen=True, slots=True)
class RecallResult:
    """Ranked recall result with scoring provenance."""

    node: Node
    score: float
    # Query coverage in content/task for FTS discoveries. The channel name
    # is shared with stored retrieval weights and feedback event records.
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
    # Current query-term coverage of source text, independent of FTS position.
    # Transient qualification evidence, not a learned ranking channel.
    lexical_score: float = 0.0

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


_ResultSelector = Callable[
    [list[RecallResult], list[RecallResult], int, QueryRelevance],
    tuple[list[RecallResult], list[RecallResult]],
]


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
    #: Activation from the separate query-anchor walk.
    anchor_score: float = 0.0
    source_graph_score: float = 0.0
    lexical_score: float = 0.0
    # A prepared zero is valid content evidence, including task-only matches.
    lexical_score_prepared: bool = False

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
    """One encoder space and width within a scope, grouped by node.

    ``rows`` holds every chunk vector of ``node_ids`` back to back in scan
    order and ``starts[i]`` is where node ``i``'s run of chunks begins, which
    is what lets the max-pool be one ``np.maximum.reduceat`` instead of a
    Python loop over chunk rows. Raw float32 rows and their norms are retained
    so scoped and broad queries can share them while applying normalization
    to the query's complete width block.

    Each block contains a single encoder space and vector width. Only the
    matching partition participates in scoring a prepared query.
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
    source_versions: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    encoder_id: str | None = None

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


class _AnchorTextIndex:
    """Nominate foreign-space questions by shared terms, across vector widths."""

    def __init__(self, rows: Iterable[tuple[str, str]]):
        self.texts: dict[str, str] = {}
        self.term_counts: dict[str, int] = {}
        self.postings: dict[str, list[str]] = defaultdict(list)
        for anchor_id, text in rows:
            self.texts[anchor_id] = text
            terms = set(tokenize(text, evidence=True))
            self.term_counts[anchor_id] = len(terms)
            for term in terms:
                self.postings[term].append(anchor_id)
            self.postings["=" + " ".join(text.split())].append(anchor_id)

    def nearest(self, query: str, limit: int) -> list[str]:
        terms = set(tokenize(query, evidence=True))
        overlap = Counter(key for term in terms for key in self.postings.get(term, ()))
        scores = {key: count / (len(terms) + self.term_counts[key]) for key, count in overlap.items()}
        # Exact text also covers punctuation and stopword-only queries.
        scores.update((key, 1.0) for key in self.postings.get("=" + " ".join(query.split()), ()))
        return nsmallest(limit, scores, key=lambda key: (-scores[key], key))


_AnchorBlockKey = tuple[int, tuple[str | None, bool] | None]


@dataclass(frozen=True, slots=True)
class _AnchorVectorCache:
    """One revision's prepared rows, grown lazily by query width and scope.

    Layouts retain only identities and scopes. Numerical storage covers only
    prepared rows, merged in scan order when another scope is admitted.
    Warm broad queries reuse that single matrix without a full gather.
    Source BLOBs and selected matrices are not retained. Metadata and targets
    always read through to the store.
    """

    revision: int
    store: MemoryStore
    by_width: dict[_AnchorBlockKey, _PreparedAnchorVectors]
    has_anchors: bool
    encoder_id: str | None = None
    layouts: dict[_AnchorBlockKey, list[tuple[str, str]]] = field(default_factory=dict)
    pending: dict[_AnchorBlockKey, set[str]] = field(default_factory=dict)

    def query_anchor_corpus(
        self, width: int, scopes: Sequence[str] | None = None, *, space: tuple[str | None, bool] | None = None,
    ) -> _PreparedAnchorVectors:
        key = (width, space)
        corpus = self.by_width.get(key)
        requested = None if scopes is None else set(map(normalize_scope, scopes))
        if corpus is not None and (
            not self.pending[key] if requested is None else
            not requested.intersection(self.pending[key])
        ):
            return corpus

        with self.store.read_snapshot():
            cacheable = self.store.cacheable_read
            # Layout and newly prepared rows must belong to the same revision,
            # even if another writer committed after the source was selected.
            if not cacheable or self.store.query_anchor_revision() != self.revision:
                return _prepare_anchor_vectors(
                    self.store.iter_query_anchor_vectors(scopes, dimensions=width, space=space),
                    width, index_scopes=True,
                )
            if key not in self.layouts:
                self.layouts[key] = self.store.query_anchor_vector_layout(width, space=space)
                self.pending[key] = {scope for _anchor_id, scope in self.layouts[key]}
            if corpus is None:
                corpus = _PreparedAnchorVectors((), (), (), ())
            missing = self.pending[key] if requested is None else requested & self.pending[key]
            if missing:
                prepared = _prepare_anchor_vectors(
                    self.store.iter_query_anchor_vectors(sorted(missing), dimensions=width, space=space),
                    width, index_scopes=True,
                )
                corpus = corpus.extend(prepared, self.layouts[key])
                self.pending[key].difference_update(missing)
            self.by_width[key] = corpus
        return corpus

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)


@dataclass(frozen=True, slots=True)
class _PreparedQuery:
    """Normalized query, producing space, and optional source-verified anchor scores."""

    text: str
    embedding: tuple[float, ...]
    encoder_id: str | None = None
    historical_scores: Mapping[str, tuple[str, float]] | None = None
    prompt_discovery: _PromptDiscovery | None = None


@dataclass(frozen=True, slots=True)
class _PromptDiscovery:
    """Channel seeds and source coverage for one query inside the prompt's snapshot."""

    plan: ScopePlan
    by_limit: dict[int, dict[str, _Candidate]]
    coverage: dict[str, float]

    def lexical_support(self, query_terms: set[str], content: str) -> float:
        version = _content_fingerprint(content)
        if version not in self.coverage:
            self.coverage[version] = _lexical_support(query_terms, content)
        return self.coverage[version]


_VectorMatch = tuple[float, int, str, str | None, str]


def _vector_keep_n(plan: ScopePlan, max_results: int) -> int:
    return max(50, max_results * 12) * max(1, len(plan.scopes))


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
        self._anchor_text_index: tuple[
            str, tuple[str, ...] | None, int | tuple[int, int], _AnchorTextIndex
        ] | None = None
        self._anchor_encodings: OrderedDict[tuple[str, str], Sequence[float]] = OrderedDict()
        self._store_embedder_shared = False
        self._unavailable_legacy_chunks: set[str] = set()
        self._pending_chunk_positions: OrderedDict[
            tuple[str | None, ...], tuple[str, str]
        ] = OrderedDict()
        self._complete_chunk_plans: OrderedDict[
            tuple[str | None, ...], tuple[str | None, int]
        ] = OrderedDict()

    def prepare_query(self, query: str, *, max_results: int = 10) -> _PreparedQuery:
        """Encode immutable input through the shared encoder, without store access."""

        if max_results <= 0:
            return _PreparedQuery("", ())
        query = query.strip()
        vector = self.embedder.embed(query) if query else ()
        encoder_id = vector.encoder_id if isinstance(vector, EncodedVector) else None
        return _PreparedQuery(query, tuple(vector), encoder_id)

    def _historical_anchor_queries(
        self, source: MemoryStore | _AnchorVectorCache, query: _PreparedQuery, plan: ScopePlan,
    ) -> dict[str, str]:
        encoder = query.encoder_id
        assert encoder is not None
        scopes = plan.scopes if plan.restricted else None
        revision: int | tuple[int, int] | None = None
        if isinstance(source, _AnchorVectorCache):
            revision = self.store.foreign_anchor_text_revision(encoder)
            if revision is None:
                # Older read-only schemas may also lack a trigger for query edits.
                data_version = 0 if self.store._anchor_has_encoder() else self.store.connection.execute(
                    "PRAGMA data_version"
                ).fetchone()[0]
                revision = (source.revision, data_version)
        cached = self._anchor_text_index
        if revision is not None and cached is not None and cached[:3] == (encoder, scopes, revision):
            index = cached[3]
        else:
            index = _AnchorTextIndex(self.store.iter_foreign_anchor_queries(encoder, scopes))
            if revision is not None:
                self._anchor_text_index = (encoder, scopes, revision, index)
        if len(index.texts) <= ANCHOR_MATCH_LIMIT:
            return index.texts

        # Neither a lexical hit nor a historical cosine contributes a score.
        # The two shortlists cover changed spaces and unlabelled compatible rows.
        candidates = index.nearest(query.text, ANCHOR_MATCH_LIMIT)
        nominations = _rank_anchors(
            source, query.embedding, scopes, min_similarity=ANCHOR_MATCH_COSINE_THRESHOLD,
            limit=ANCHOR_MATCH_LIMIT, space=(None, True),
        )
        candidates.extend(anchor_id for anchor_id, _score in nominations)
        return {anchor_id: index.texts[anchor_id] for anchor_id in candidates if anchor_id in index.texts}

    def prepare_anchor_history(
        self, query: _PreparedQuery, *, scope: str | None, ambient_context: dict[str, Any] | None,
        depth: int | str | None = 1, for_annotation: bool = False,
        guard: Callable[[], AbstractContextManager[Any]] = nullcontext,
    ) -> _PreparedQuery:
        """Re-score a bounded set of historical questions before response selection."""
        encoder = query.encoder_id
        if not encoder or not query.embedding or query.historical_scores is not None:
            return query
        with guard():
            graph = self.anchor_seeding and _parse_depth(depth, query.text)[0] > 0
            if not (graph or for_annotation) and not (
                _explicit_feedback_policy() == "credit" and query_irrelevance_factor() < 1
                and has_active_irrelevance(self.store)
            ):
                return replace(query, historical_scores={})
            try:
                with self.store.read_snapshot():
                    source = self._anchor_vector_source(encoder)
                    if source is None:
                        return replace(query, historical_scores={})
                    plan = self.scope_resolver.resolve(scope=scope, ambient_context=ambient_context, store=self.store)
                    texts = self._historical_anchor_queries(source, query, plan)
            except Exception:
                if graph or for_annotation:
                    raise
                # Query-relative demotion is optional, including its historical read.
                return replace(query, historical_scores={})
            vectors = {}
            for text in dict.fromkeys(texts.values()):
                if text == query.text:
                    vectors[text] = query.embedding
                elif (encoder, text) in self._anchor_encodings:
                    vectors[text] = self._anchor_encodings[encoder, text]
                    self._anchor_encodings.move_to_end((encoder, text))
        missing = [text for text in dict.fromkeys(texts.values()) if text not in vectors]
        if missing:
            batch = getattr(self.embedder, "embed_batch", None)
            try:
                fresh = batch(missing) if batch else [self.embedder.embed(text) for text in missing]
                encoded = list(zip(missing, fresh, strict=True))
            except Exception as error:
                logging.getLogger(__name__).warning(
                    "historical anchor encoding deferred for %d questions (%s)", len(missing), type(error).__name__,
                )
                encoded = []
            with guard():
                for text, vector in encoded:
                    if (isinstance(vector, EncodedVector) and vector.encoder_id == encoder
                            and len(vector) == len(query.embedding)):
                        vectors[text] = vector
                        self._anchor_encodings[encoder, text] = vector
                while len(self._anchor_encodings) > 256:
                    self._anchor_encodings.popitem(last=False)
        scores = {key: (text, cosine_similarity(query.embedding, vectors[text]))
                  for key, text in texts.items() if text in vectors}
        return replace(query, historical_scores=scores)

    def memory_recall(
        self,
        query: str | _PreparedQuery,
        *,
        scope: str | None = None,
        ambient_context: dict[str, Any] | None = None,
        depth: int | str | None = 1,
        max_results: int = 10,
        log_access: bool = True,
        log_event: bool | None = None,
        repair_pending: bool = True,
        include_passages: bool = True,
        select_results: _ResultSelector | None = None,
    ) -> list[RecallResult]:
        """Retrieve and commit a complete direct response."""
        with self.recall_response(
            query, scope=scope, ambient_context=ambient_context, depth=depth,
            max_results=max_results, log_access=log_access, log_event=log_event,
            repair_pending=repair_pending, select_results=select_results, include_passages=include_passages,
        ) as results:
            return results

    @contextmanager
    def recall_response(
        self,
        query: str | _PreparedQuery,
        *,
        scope: str | None = None,
        ambient_context: dict[str, Any] | None = None,
        depth: int | str | None = 1,
        max_results: int = 10,
        log_access: bool = True,
        log_event: bool | None = None,
        repair_pending: bool = True,
        include_passages: bool = True,
        repeatable_selection: bool = False,
        select_results: _ResultSelector | None = None,
    ) -> Iterator[list[RecallResult]]:
        """Form a response within the short consumption rollback boundary.

        Search, encoding and selection precede the transaction. The caller may
        read required metadata and shape the selected results before leaving
        this context; failure rolls back access and event/history together.
        Optional work and transport belong after it. A repeatable read-only
        selector observes one snapshot through passage and validity reads.
        Logging flags independently control the consumption transaction.
        """
        self.last_recall_event_id = None
        self.last_residual = []
        self.last_query_relevance = None
        if max_results <= 0:
            yield []
            return

        prepared = query if isinstance(query, _PreparedQuery) else self.prepare_query(query)
        prepared = self.prepare_anchor_history(prepared, scope=scope, ambient_context=ambient_context, depth=depth)
        query = prepared.text
        if not query:
            yield []
            return

        if repair_pending:
            self.refresh_pending_chunks(scope=scope, ambient_context=ambient_context, query=prepared)

        with self.store.read_snapshot() if repeatable_selection else nullcontext():
            plan, limited = self._select_recall(
                prepared, scope=scope, ambient_context=ambient_context, depth=depth,
                max_results=max_results, select_results=select_results,
                repeatable_selection=repeatable_selection, include_passages=include_passages,
            )

        if log_event is None:
            log_event = log_access
        try:
            with self.store.transaction() if log_access or log_event else nullcontext():
                if log_access or log_event:
                    nodes, event_id = self.store.record_recall(
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
                        event_id_only=True,
                    )
                    assert event_id is None or isinstance(event_id, str)
                    self.last_recall_event_id = event_id
                    limited = [
                        replace(result, node=node, recall_event_id=self.last_recall_event_id)
                        for result, node in zip(limited, nodes, strict=True)
                    ]
                yield limited
        except BaseException:
            self.last_recall_event_id = None
            raise

    def _prepare_prompt_discovery(
        self, query: _PreparedQuery, plan: ScopePlan, limits: Sequence[int],
    ) -> _PreparedQuery:
        """Prepare bounded seeds for the active selectors within their shared snapshot."""
        groups: dict[int, dict[str, _Candidate]] = {}
        matched_by_version: dict[str, set[str]] = {}
        scope_limits = {limit: max(25, limit * 8) for limit in limits}
        rows = (self.store.recall_content_ids_by_limit(_expanded_query(query.text), plan, list(scope_limits.values()))
                if len(scope_limits) > 1 else {})
        for limit, per_scope in scope_limits.items():
            candidates = groups[limit] = {}
            self._collect_bm25(query.text, plan, candidates, max_results=limit,
                               matched_by_version=matched_by_version, rows=rows.get(per_scope))
        del rows
        if query.encoder_id is not None:
            lexical_ids = set().union(*(group.keys() for group in groups.values()))
            discovery, lexical = self._vector_winners(
                query.embedding, plan, query.encoder_id, _vector_keep_n(plan, max(limits)), lexical_ids,
            )
            if len(groups) > 1:
                discovery.sort(reverse=True)
            for limit, candidates in groups.items():
                keep = _vector_keep_n(plan, limit)
                winners = discovery[:keep] if keep < len(discovery) else discovery
                self._admit_vector(plan, candidates, winners, lexical)
        term_count = len(_lexical_terms(query.text))
        coverage = {version: len(matched) / term_count for version, matched in matched_by_version.items()}
        return replace(query, prompt_discovery=_PromptDiscovery(plan, groups, coverage))

    def _select_recall(
        self, prepared: _PreparedQuery, *, scope: str | None,
        ambient_context: dict[str, Any] | None, depth: int | str | None, max_results: int,
        select_results: _ResultSelector | None, repeatable_selection: bool, include_passages: bool,
    ) -> tuple[ScopePlan, list[RecallResult]]:
        query, query_embedding = prepared.text, prepared.embedding
        plan = self.scope_resolver.resolve(
            scope=scope,
            ambient_context=ambient_context,
            store=self.store,
        )
        discovery = prepared.prompt_discovery
        lexical_support = _lexical_support
        if discovery is not None and discovery.plan == plan and max_results in discovery.by_limit:
            lexical_support = discovery.lexical_support
            # Graph walks and trigger collection mutate evidence independently for each depth.
            candidates = {key: replace(seed) for key, seed in discovery.by_limit[max_results].items()}
        else:
            candidates = {}
            self._collect_bm25(query, plan, candidates, max_results=max_results)
            self._collect_vector(query_embedding, plan, candidates, max_results=max_results,
                                 encoder_id=prepared.encoder_id)
        by_name = schema_trigger_by_name()
        if not by_name:
            self._collect_schema_triggers(query, plan, candidates)

        query_terms = _lexical_terms(query)
        for candidate in candidates.values():
            if not candidate.lexical_score_prepared:
                candidate.lexical_score = lexical_support(query_terms, candidate.node.content)
        lexical_ids = set(candidates)

        graph_depth, causal_mode = _parse_depth(depth, query)
        decision_mode = _is_decision_depth(depth)
        # Anchors are an entry *into* the graph channel, so they are asked for
        # exactly when that channel runs: with the graph off (depth=0) there is
        # nothing for a seed to open. Vector scoring and anchor matching
        # reuse the same prepared query embedding.
        anchor_matches = (
            self._match_query_anchors(plan, prepared)
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

        for node_id in candidates.keys() - lexical_ids:
            candidate = candidates[node_id]
            candidate.lexical_score = lexical_support(query_terms, candidate.node.content)

        demotions = _merge_demotions(
            self._collect_query_demotions(plan, query_embedding, matches=anchor_matches, encoder_id=prepared.encoder_id,
                                          historical_scores=prepared.historical_scores),
            self._collect_hub_demotions(candidates),
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
                repeatable_selection=repeatable_selection,
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
        # must describe the answer that the caller actually delivers. Share
        # the gate's evidence so replacements obey the same qualification,
        # including relevance inherited by present corrections.
        if select_results is not None:
            limited, self.last_residual = select_results(
                limited, self.last_residual, max_results, self.last_query_relevance
            )
        if not include_passages:
            return plan, limited

        chunk_ids = {
            result.node_id: candidates[result.node_id].chunk_id
            for result in limited
            if not result.withheld and result.node.level != "schema"
            and candidates[result.node_id].chunk_id is not None
        }
        passages = self.store.get_chunk_passages(list(chunk_ids.values()))
        limited = [
            replace(result, semantic_passage=passages.get(chunk_ids.get(result.node_id)),
                    scored_chunk_id=chunk_ids.get(result.node_id))
            for result in limited
        ]
        return plan, limited

    def refresh_pending_chunks(
        self, *, scope: str | None = None,
        encoder_id: str | None = None,
        query: _PreparedQuery | None = None,
        ambient_context: dict[str, Any] | None = None,
        guard: Callable[[], AbstractContextManager[Any]] = nullcontext,
    ) -> None:
        """Repair query candidates and advance through pending sources in cyclic order.

        Positions belong to this service and search-scope set, across clients.
        Restart or LRU eviction starts at the oldest pending source again;
        positions never certify index validity or suppress later retries.
        """
        if query is not None:
            encoder_id = query.encoder_id
        with guard():
            plan = self.scope_resolver.resolve(scope=scope, ambient_context=ambient_context,
                                               store=self.store)
            scopes = plan.search_scopes
            after = self._pending_chunk_positions.get(scopes)
            pending = self._list_pending_chunks(plan, encoder_id, after)
            if not pending:
                self._pending_chunk_positions.pop(scopes, None)
                return
            # Advance under the selection guard before model work releases it,
            # even when a selected source fails or changes before application.
            candidates = []
            if (len(pending) > LAZY_CHUNK_NODE_LIMIT and query is not None
                    and encoder_id and query.embedding and LAZY_CHUNK_NODE_LIMIT > 1):
                candidates = self._repair_candidates(query, scopes, LAZY_CHUNK_NODE_LIMIT - 1)
            selected = dict.fromkeys(candidates)
            for position in pending:
                selected[position[1]] = None
                if len(selected) == LAZY_CHUNK_NODE_LIMIT:
                    break
            self._pending_chunk_positions[scopes] = position
            self._pending_chunk_positions.move_to_end(scopes)
            if len(self._pending_chunk_positions) > LAZY_CHUNK_PLAN_LIMIT:
                self._pending_chunk_positions.popitem(last=False)
            self._share_embedder_with_store()
        deferred_errors: list[str] = []
        for node_id in selected:
            try:
                with guard():
                    node = self.store.get_node(node_id)
            except (TypeError, ValueError) as error:
                # Malformed source data is local; storage read failures remain
                # explicit, just like failures in pending discovery.
                deferred_errors.append(type(error).__name__)
                continue
            try:
                if node is not None and not node.decayed and plan.allows(node.scope):
                    self._rechunk_node(node, guard=guard, encoder_id=encoder_id)
            except Exception as error:
                deferred_errors.append(type(error).__name__)
        if deferred_errors:
            logging.getLogger(__name__).warning(
                "lazy chunk indexing deferred for %d/%d nodes (first error: %s); using available index",
                len(deferred_errors), len(selected), deferred_errors[0],
            )

    def _list_pending_chunks(
        self, plan: ScopePlan, encoder_id: str | None, after: tuple[str, str] | None,
    ) -> list[tuple[str, str]]:
        """Remember only emptiness proven in the same snapshot as its revisions."""
        scopes = plan.search_scopes
        # A caller can roll back its own writes and reuse their revision numbers.
        cacheable = not self.store.connection.in_transaction
        with self.store.read_snapshot():
            revision = self.store.pending_chunk_revision() if cacheable else None
            if revision is not None and self._complete_chunk_plans.get(scopes) == (encoder_id, revision):
                self._complete_chunk_plans.move_to_end(scopes)
                return []
            pending = sorted(
                (position for item in scopes for position in self.store.list_unchunked_node_positions(
                    scope=item, limit=LAZY_CHUNK_NODE_LIMIT + 1, after=after, encoder_id=encoder_id,
                )),
                key=lambda position: (after is not None and position <= after, *position),
            )
            if not pending and revision is not None:
                self._complete_chunk_plans[scopes] = (encoder_id, revision)
                self._complete_chunk_plans.move_to_end(scopes)
                if len(self._complete_chunk_plans) > LAZY_CHUNK_PLAN_LIMIT:
                    self._complete_chunk_plans.popitem(last=False)
            return pending

    def _repair_candidates(
        self, query: _PreparedQuery, scopes: tuple[str | None, ...], limit: int,
    ) -> list[str]:
        """Nominate sources only. Freshly encoded chunks supply all vector evidence."""
        def historical_scores() -> Iterator[tuple[float, str]]:
            for scope in scopes:
                batches = self.store.unknown_chunk_batches(scope, len(query.embedding))
                rows = _historical_window_scores(batches, query.embedding)
                for node_id, windows in groupby(rows, key=itemgetter(0)):
                    best = max(score for _, score in windows)
                    if best >= VECTOR_MATCH_THRESHOLD:
                        yield best, node_id

        historical = [node_id for _, node_id in nlargest(limit, historical_scores())]
        # When the historical coordinates come from another backend, the
        # fallback encoder's aliases can still find the source through FTS.
        terms = repair_search_terms(query.text)
        lexical = []
        if terms:
            matches = sorted(
                (score, node_id) for scope in scopes
                for node_id, score in self.store.search_content_ids(
                    terms, scope=scope, limit=limit, pending_encoder=query.encoder_id,
                )
            )
            lexical = [node_id for _, node_id in matches]
        return list(dict.fromkeys(node_id for pair in zip_longest(historical, lexical)
                                  for node_id in pair if node_id is not None))[:limit]

    def refresh_legacy_passages(
        self, snippets: Sequence[RecallResult], *,
        guard: Callable[[], AbstractContextManager[Any]] = nullcontext,
    ) -> None:
        """Prepare at most one delivered snippet for a future semantic recall.

        The caller supplies actual snippet bearers after delivery shaping. The
        scored generation remains this call's only evidence; replacing its index
        cannot give that old score an address in the replacement generation.
        The shared-store guard protects selection and storage reads/writes, while
        the selected source's tokenization and encoding run outside it.
        """
        refresh = getattr(self.store, "refresh_legacy_chunks", None)
        if not callable(refresh):
            return
        with guard():
            for result in snippets:
                if (result.semantic_passage is not None or not result.scored_chunk_id
                        or result.scored_chunk_id in self._unavailable_legacy_chunks
                        or result.withheld or result.node.level == "schema"
                        or len(result.node.content) > LEGACY_PASSAGE_MAX_CHARS):
                    continue
                node_id, content, scored_chunk_id = (
                    result.node_id, result.node.content, result.scored_chunk_id,
                )
                # Reserve the same immutable row while preparing it. A failure
                # retains the reservation; successful/no-op attempts release it.
                self._unavailable_legacy_chunks.add(scored_chunk_id)
                break
            else:
                return
        try:
            with guard():
                self._share_embedder_with_store()
            refresh(
                node_id, content, scored_chunk_id,
                max_chars=LEGACY_PASSAGE_MAX_CHARS,
                max_chunks=LEGACY_PASSAGE_MAX_CHUNKS,
                guard=guard,
            )
        except Exception:
            # Immutable scored rows bound repeated failure work without
            # disabling independent sources. Replacement rows can retry;
            # an unchanged eligible index has at most max_chunks winners.
            logging.getLogger(__name__).warning(
                "could not refresh selected legacy chunk index", exc_info=True,
            )
        else:
            with guard():
                self._unavailable_legacy_chunks.discard(scored_chunk_id)

    def memory_connect(
        self,
        id_a: str,
        id_b: str,
        relation_type: ConnectionType,
        *,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
        guard: Callable[[], AbstractContextManager[Any]] = nullcontext,
    ) -> Connection:
        return memory_connect(
            self.store,
            id_a,
            id_b,
            relation_type,
            weight=weight,
            metadata=metadata,
            guard=guard,
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
        reference = REFERENCE_WEIGHTS.normalized() if schema_trigger_by_name() else None
        weights_by_scope: dict[str, RetrievalWeights] = {}
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

            weights = reference or weights_by_scope.get(node.scope)
            if weights is None:
                weights = self.store.get_retrieval_weights(node.scope).normalized()
                weights_by_scope[node.scope] = weights
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
                    lexical_score=candidate.lexical_score,
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
        matched_by_version: dict[str, set[str]] | None = None,
        rows: Iterable[tuple[str, float]] | None = None,
    ) -> None:
        query_terms = _lexical_terms(query)
        # FTS order selects the bounded discovery set. Score admitted sources
        # by query coverage, independently of that order or corpus statistics.
        # Named scopes retain their own discovery limits; a whole-store plan
        # also searches scopes that were not explicitly named.
        # Keep discovery and admitted bodies in one short read snapshot. IDs
        # carry membership only; the current scoped/live reader admits bodies.
        with self.store.transaction():
            if rows is None:
                rows = self.store.iter_recall_content_ids(_expanded_query(query), plan, limit=max(25, max_results * 8))
            discovered = dict.fromkeys(node_id for node_id, _raw_score in rows)
            fetched = self.store.get_recall_nodes(
                (node_id for node_id in discovered if node_id not in candidates), plan,
            )
        for node_id in discovered:
            candidate = candidates.get(node_id)
            if candidate is None:
                node = fetched.get(node_id)
                if node is None:
                    continue
                candidate = candidates[node_id] = _Candidate(node=node)
            # Prompt limits share only the query terms matched by the body.
            # Task terms extend discovery after content qualification is fixed.
            score = 0.0
            candidate.lexical_score = 0.0
            if query_terms:
                content = candidate.node.content
                if matched_by_version is None:
                    matched = query_terms.intersection(_lexical_terms(content))
                else:
                    version = _content_fingerprint(content)
                    if version not in matched_by_version:
                        matched_by_version[version] = query_terms.intersection(_lexical_terms(content))
                    matched = matched_by_version[version]
                candidate.lexical_score = len(matched) / len(query_terms)
                if candidate.node.task:
                    matched = matched.union(query_terms.intersection(_lexical_terms(candidate.node.task)))
                score = len(matched) / len(query_terms)
            candidate.lexical_score_prepared = True
            candidate.bm25_score = max(candidate.bm25_score, score)

    def _collect_vector(
        self,
        query_embedding: Sequence[float],
        plan: ScopePlan,
        candidates: dict[str, _Candidate],
        *,
        max_results: int,
        encoder_id: str | None = None,
    ) -> None:
        """Score nodes by the best cosine among their chunk vectors.

        The caller supplies the prepared query vector, also used by anchors.
        Pending repair precedes selection; scoring reads the current index.
        """

        if encoder_id is None:
            return

        discovery, lexical = self._vector_winners(
            query_embedding, plan, encoder_id, _vector_keep_n(plan, max_results), candidates,
        )
        self._admit_vector(plan, candidates, discovery, lexical)

    def _vector_winners(
        self, query_embedding: Sequence[float], plan: ScopePlan, encoder_id: str,
        keep_n: int, lexical_ids: Collection[str],
    ) -> tuple[list[_VectorMatch], list[_VectorMatch]]:
        q_arr = _as_query_array(query_embedding)
        discovery: list[_VectorMatch] = []
        lexical: list[_VectorMatch] = []
        order = 0
        for block in self._chunk_blocks(plan, len(query_embedding), encoder_id):
            order = _select_block_matches(block, query_embedding, q_arr, order, keep_n, lexical_ids,
                                          discovery, lexical)
        return discovery, lexical

    def _admit_vector(
        self, plan: ScopePlan, candidates: dict[str, _Candidate],
        discovery: list[_VectorMatch], lexical: list[_VectorMatch],
    ) -> None:
        # Keep vector evidence for this selector's lexical tail, including
        # winners below its discovery cutoff, before current-source admission.
        kept = [item for item in discovery if item[2] not in candidates]
        kept.extend(item for item in lexical if item[2] in candidates)
        kept.sort(reverse=True)
        # One query for the whole kept set rather than one per node. Max-pool
        # puts more nodes over VECTOR_MATCH_THRESHOLD than a single vector did
        # -- a node now clears it if *any* window does -- so this list runs
        # closer to its `keep_n` ceiling than it used to, and the per-node
        # round trip is the part of it that is pure overhead.
        missing = [item[2] for item in kept if item[2] not in candidates]
        fetched = self.store.get_recall_nodes(missing, plan) if missing else {}
        for similarity, _order, node_id, chunk_id, version in kept:
            existing = candidates.get(node_id)
            node = existing.node if existing is not None else fetched.get(node_id)
            # Hydration can span a concurrent edit. Historical source rows can
            # also retain a fingerprint predating their text. Check only the
            # bounded winners, using the version from the scored block.
            if node is None or _content_fingerprint(node.content) != version:
                continue
            if existing is None:
                existing = candidates[node_id] = _Candidate(node=node)
            if similarity > existing.vector_score:
                existing.vector_score = similarity
                existing.chunk_id = chunk_id

    # ------------------------------------------------------------------
    # Chunk corpus cache
    # ------------------------------------------------------------------

    def _chunk_blocks(self, plan: ScopePlan, dimension: int, encoder_id: str) -> tuple[_ChunkBlock, ...]:
        """Share scope partitions, reading revisions and rows in one snapshot.

        The existing deferred transaction helper reserves no writer. Its read
        snapshot prevents a concurrent move from appearing in two partitions.
        An outer caller transaction may contain writes that roll back, so its
        rows are never retained. MCP selection shares its snapshot with this read.

        Broad queries temporarily gather rows in node/chunk order. Keeping the
        full width block's normalization and matmul shape preserves scores and
        stable candidate ties. Row placement is retained until its source blocks
        change; there is no second persistent corpus matrix.
        """

        with self.store.read_snapshot():
            cacheable = self.store.cacheable_read
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
                    for scope in scopes if scope in revisions
                ]
        blocks = tuple(block for index in indexes for block in index.blocks
                       if block.dimension == dimension and block.encoder_id == encoder_id)
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
        """Group known chunk vectors by encoder space and byte width.

        The ordered storage scan keeps each node's compatible chunks together
        for max-pooling. Rows without producer identity remain unscored.
        """

        accumulators: dict[tuple[int, str], _BlockAccumulator] = {}
        for node_id, _ordinal, view, chunk_id, encoder_id, version in self.store.iter_chunk_embedding_rows(
            scope=scope, include_identity=True, include_encoder=True, known_only=True, current_source_version=True,
        ):
            width, remainder = divmod(len(view), CHUNK_EMBEDDING_ITEMSIZE)
            if remainder or not width or not encoder_id:
                continue
            key = (width, encoder_id)
            if key not in accumulators:
                accumulators[key] = _BlockAccumulator()
            accumulators[key].add(node_id, view, chunk_id, version)

        blocks = tuple(
            replace(accumulator.freeze(width), encoder_id=encoder_id)
            for (width, encoder_id), accumulator in sorted(accumulators.items())
        )
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
        self, plan: ScopePlan, query: _PreparedQuery | Sequence[float], *,
        limit: int | None = None, min_similarity: float = ANCHOR_MATCH_COSINE_THRESHOLD,
        include_targets: bool = True,
    ) -> list[AnchorMatch]:
        """Observe live matches and their current targets in this query's space."""
        if isinstance(query, _PreparedQuery):
            vector, encoder_id, scores = query.embedding, query.encoder_id, query.historical_scores
        else:
            vector = query
            encoder_id = vector.encoder_id if isinstance(vector, EncodedVector) else None
            scores = None
        if not vector or encoder_id is None:
            return []
        source = self._anchor_vector_source(encoder_id)
        if source is None:
            return []
        return match_anchors(
            source, vector, plan, limit=ANCHOR_MATCH_LIMIT if limit is None else limit,
            min_similarity=min_similarity, include_targets=include_targets,
            encoder_id=encoder_id, historical_scores=scores,
        )

    def _collect_anchor_seeds(
        self, plan: ScopePlan, query_embedding: Sequence[float],
        *, matches: list[AnchorMatch] | None = None,
    ) -> dict[str, float]:
        """Graph seeds contributed by the anchors this query matches.

        ``{node_id: activation}``, where activation is the matched anchor's
        cosine times the weight of its edge to that node. Both factors are in
        [0, 1] and their product is a confidence, so the value lands in the
        same range ``_collect_graph``'s existing seeds occupy -- lexical
        coverage, a cosine -- and blends through the same learned ``graph``
        weight. A node reachable from two matched anchors keeps the stronger
        claim rather than accumulating, so a query that happens to sit near
        several anchors cannot manufacture activation the edges do not carry.

        Scope is the ``ScopePlan``'s, handed to ``match_anchors`` so the SQL
        never returns an anchor from outside it: an anchor seeds only where its
        own scope is admitted. Target scopes are checked before the seed cap,
        so only targets admitted by the plan compete for its slots. Downstream
        admission still checks current scope and liveness before body use.

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
        if seeds and plan.restricted:
            scopes = self.store.get_node_scopes(seeds)
            seeds = {
                target_id: activation for target_id, activation in seeds.items()
                if target_id in scopes and plan.allows(scopes[target_id])
            }
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
        self, plan: ScopePlan, query_embedding: Sequence[float],
        *, matches: list[AnchorMatch] | None = None, encoder_id: str | None = None,
        historical_scores: Mapping[str, tuple[str, float]] | None = None,
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
            source = self._anchor_vector_source(encoder_id)
            if source is None:
                return {}
            return query_demotions(source, query_embedding, plan, encoder_id=encoder_id, historical_scores=historical_scores)
        except Exception:  # pragma: no cover - derived signal, never fatal
            return {}

    def _collect_hub_demotions(self, candidate_ids: Iterable[str]) -> dict[str, float]:
        """Global hub multipliers (``living_memory.hubs``). Never raises."""

        try:
            return hub_demotions(self.store, candidate_ids=candidate_ids)
        except Exception:  # pragma: no cover - derived signal, never fatal
            return {}

    def _anchor_vector_source(self, encoder_id: str | None = None) -> _AnchorVectorCache | MemoryStore | None:
        """The live anchor corpus, cached until ``query_anchor_revision`` moves.

        One indexed revision lookup guards reuse, including an empty corpus.
        Matching prepares only the query's scope/width rows. An empty layout
        never scans vectors. Unmigrated read-only stores use the uncached reader.
        Uncommitted inputs must not enter the cache: rollback can restore a
        revision that another mutation will reuse for different vectors.
        """

        revision = self.store.query_anchor_revision()
        if revision is None or not self.store.cacheable_read:
            self._anchor_vectors = None
            return self.store
        cached = self._anchor_vectors
        if cached is None or cached.revision != revision or (encoder_id and cached.encoder_id != encoder_id):
            cached = _AnchorVectorCache(
                revision, self.store, {},
                self.store.has_live_query_anchors(), encoder_id,
            )
            self._anchor_vectors = cached
        return cached if cached.has_anchors else None

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
        fetched = self.store.get_recall_nodes(missing, plan) if missing else {}
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
        query_tokens = set(tokenize(query, evidence=True))
        if not query_tokens:
            return
        scores: dict[str, float] = {}
        coverages: dict[str, float] = {}
        for scope in plan.search_scopes:
            for schema_id, trigger in self.store.schema_triggers(scope):
                trigger_tokens = set(tokenize(trigger, evidence=True))
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
        # Keep surviving identities in the reader's order without reloading
        # bodies already selected by another channel in this recall.
        identities = self.store.get_node_scopes(scores)
        missing = self.store.get_nodes(node_id for node_id in identities if node_id not in candidates)
        for node_id in identities:
            candidate = candidates.get(node_id)
            if candidate is None:
                node = missing.get(node_id)
                if node is None:
                    continue
                candidate = candidates[node_id] = _Candidate(node=node)
            candidate.trigger_score = max(candidate.trigger_score, scores[node_id])
            candidate.trigger_coverage = max(candidate.trigger_coverage, coverages[node_id])

    def _named_schemas_first(
        self,
        query: str,
        plan: ScopePlan,
        ranked: list[RecallResult],
        *,
        demotions: Mapping[str, float] | None,
        causal_mode: bool,
        repeatable_selection: bool = False,
    ) -> list[RecallResult]:
        """``name`` mode: the schemas this query names, if they earned a slot, go first.

        A schema is named when the query's evidence terms equal its trigger's:
        the whole query is its title, with complete lexical support and a
        discovery score of 1.0. Only schemas already ranked can be named,
        and a named schema moves to the front only if
        :func:`score_gate.passes_gate` passes it on that score (with the gate
        off, always). Named schemas keep their ranked order among themselves
        and are marked with ``trigger_score``/``"trigger"`` for the wire.
        """

        query_tokens = frozenset(tokenize(query, evidence=True))
        if not query_tokens or not ranked:
            return ranked
        # Only a repeatable selection can rule out a level change since ranking.
        ranked_ids = [
            result.node.id for result in ranked
            if not repeatable_selection or result.node.level in ("schema", "concept")
        ]
        if not ranked_ids:
            return ranked
        named_ids = {
            node_id
            for scope in plan.search_scopes
            for node_id, trigger in self.store.schema_triggers(scope, node_ids=ranked_ids)
            if frozenset(tokenize(trigger, evidence=True)) == query_tokens
        }
        if not named_ids:
            return ranked
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
                lexical_score=1.0,
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
        """Score graph discovery and retain its independent source contribution.

        Ordinary candidate seeds contribute to ``graph_score``. A separate
        walk from matched anchor targets contributes to ``anchor_score``,
        so the ranker can also compute each candidate's anchors-off score.

        Source qualification seeds by current lexical/vector support and
        excludes history-derived relation weights. The same bounded walker
        evaluates source paths independently when either differs from discovery.
        Neither source pass nor anchors change the ordinary ranking walk.
        """

        # A bulk query's tie order depends on its entire frontier, so reuse
        # observations only for the same ordered request and contribution.
        # Walks share bounded neighbors, never their best-path state.
        observations: dict[
            tuple[bool, tuple[str, ...]], dict[str, list[tuple[str, float]]],
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
            if not self.store.cacheable_read or state != observed_state:
                observations.clear()
            observed_state = state if self.store.cacheable_read else None
            key = source_only, tuple(node_ids)
            if key not in observations:
                connections = self.store.list_recall_connections(
                    node_ids, limit=MAX_NEIGHBORS_PER_NODE, source_only=source_only,
                )
                neighbors = {node_id: [] for node_id in connections}
                for node_id, edges in connections.items():
                    for edge in edges:
                        if not source_only and needs_source_projection(edge.type, edge.weight, edge.metadata):
                            historical_edges = True
                        traversal = _traversal(
                            edge, node_id, causal_mode=causal_mode,
                            decision_mode=decision_mode,
                        )
                        if traversal is not None:
                            neighbors[node_id].append(traversal)
                observations[key] = neighbors
            return observations[key]

        # Seed BFS only from the top-K pre-graph candidates to avoid
        # fan-out from low-quality matches that would dilute the graph
        # signal and waste cycles on super-hub expansions.
        seed_candidates = sorted(
            candidates.values(),
            key=lambda c: (-max(c.bm25_score, c.vector_score, 0.0), c.node.id),
        )[:GRAPH_SEED_LIMIT]
        roots = [
            (candidate.node.id, max(candidate.bm25_score, candidate.vector_score))
            for candidate in seed_candidates
        ]
        source_roots = sorted(
            ((candidate.node.id, max(candidate.lexical_score, candidate.vector_score))
             for candidate in candidates.values()
             if max(candidate.lexical_score, candidate.vector_score) > 0.0),
            key=lambda item: (-item[1], item[0]),
        )[:GRAPH_SEED_LIMIT]
        self._walk_graph(
            plan, candidates, roots,
            max_depth=max_depth,
            neighbors_for_nodes=neighbors_for_nodes,
            anchor_pass=False,
        )

        if historical_edges or source_roots != roots or any(score < 0.05 for _, score in roots):
            # Reuse the bounded walk with independent roots and best-path
            # state: discovery strength must not hide a source path.
            # Extra source-only discoveries do not change ranking membership.
            self._walk_graph(
                plan, candidates, source_roots,
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

        The selected pass controls its score field and provenance updates.
        Traversal, bounds and arithmetic are shared. The source pass supplies
        history-free relation contributions and never changes ranking paths
        or membership; its intermediate identities need admission only.
        """

        queue: deque[tuple[str, float, int, tuple[str, ...]]] = deque()
        best_seen: dict[str, float] = {}
        for node_id, activation in roots:
            seed = activation if source_pass else max(activation, 0.05)
            queue.append((node_id, seed, 0, (node_id,)))
            best_seen[node_id] = seed

        # Each pass retains its own admission observations, including misses,
        # so multiple paths fetch an identity at most once. Only discovery
        # needs the bodies of newly admitted neighbors.
        node_cache: dict[str, Node | None] = {}
        source_admission: dict[str, bool] = {}

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
            final_source_level = source_pass and current_depth + 1 == max_depth
            transitions = []
            needed_neighbor_ids = set()
            for node_id, activation, depth, path in level_items:
                for neighbor_id, relation_score in neighbors_by_node[node_id]:
                    if neighbor_id in path:
                        continue

                    next_score = activation * relation_score * (0.72 ** depth)
                    if next_score <= best_seen.get(neighbor_id, 0.0):
                        continue

                    if final_source_level:
                        candidate = candidates.get(neighbor_id)
                        if candidate is not None and not candidate.node.decayed and plan.allows(candidate.node.scope):
                            candidate.source_graph_score = max(candidate.source_graph_score, min(1.5, next_score))
                        continue

                    transitions.append((neighbor_id, next_score, depth + 1, path + (neighbor_id,)))
                    if (neighbor_id not in node_cache and neighbor_id not in source_admission
                            and neighbor_id not in candidates):
                        needed_neighbor_ids.add(neighbor_id)

            # Admit missing neighbors before materializing their bodies.
            if needed_neighbor_ids:
                if source_pass:
                    admitted = self.store.get_recall_node_ids(needed_neighbor_ids, plan)
                    source_admission.update((nid, nid in admitted) for nid in needed_neighbor_ids)
                else:
                    fetched_nodes = self.store.get_recall_nodes(needed_neighbor_ids, plan)
                    for nid in needed_neighbor_ids:
                        node_cache[nid] = fetched_nodes.get(nid)

            for neighbor_id, next_score, depth, path in transitions:
                if next_score <= best_seen.get(neighbor_id, 0.0):
                    continue

                candidate = candidates.get(neighbor_id)
                if source_pass and candidate is None:
                    if not source_admission[neighbor_id]:
                        continue
                else:
                    neighbor = candidate.node if candidate is not None else node_cache[neighbor_id]
                    if neighbor is None or neighbor.decayed or not plan.allows(neighbor.scope):
                        continue
                    if candidate is None:
                        candidate = candidates[neighbor_id] = _Candidate(node=neighbor)

                best_seen[neighbor_id] = next_score
                if source_pass:
                    if candidate is not None:
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

    def _rechunk_node(
        self, node: Node, *, encoder_id: str | None = None,
        guard: Callable[[], AbstractContextManager[Any]] = nullcontext,
    ) -> None:
        """Refresh content search without overwriting a cluster representative."""

        with guard():
            self._share_embedder_with_store()
        self.store.refresh_pending_chunks(
            node, self.embedder.embed, encoder_id=encoder_id,
            embed_batch=getattr(self.embedder, "embed_batch", None), guard=guard,
        )

    def _share_embedder_with_store(self) -> None:
        """Share one encoder for queries, source chunks and feedback anchors.

        Multi-window sources need every window encoded in the query's vector
        space. The ranking-only replay store has no encoder setter.
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


_LEXICAL_IDENTIFIER_RE = re.compile(r"\w+(?:[-/._]\w+)+")


def _lexical_terms(text: str) -> set[str]:
    """Normalized terms plus exact compound identifiers as distinct evidence.

    Keeping the compound spelling distinguishes a path or task identifier
    from its component words scattered through unrelated text. Individual
    terms still admit partial matches and retain synonym/stem support.
    """

    terms = distinct_evidence_tokens(text)
    terms.update(identifier.casefold() for identifier in _LEXICAL_IDENTIFIER_RE.findall(text))
    return terms


def _lexical_support(query_terms: set[str], content: str) -> float:
    """Query coverage by source terms, without corpus counts or discovery rank."""

    if not query_terms:
        return 0.0
    return len(query_terms.intersection(_lexical_terms(content))) / len(query_terms)


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
    if not is_semantic_relation(connection.type, connection.metadata):
        return None
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
    qualification: bool = False,
) -> float:
    """Blend one graph activation with the candidate's other query channels.

    Takes ``graph_score`` as an argument rather than reading it off the
    candidate because the ranker scores some candidates twice -- once with the
    activation an anchor contributed and once without it -- and the two
    scorings have to be the same function of it, floor included, or the
    comparison between them proves nothing.

    Returns 0.0 for a candidate no channel scored, which is the caller's cue
    to drop it. Qualification reads lexical coverage, source activation and
    source restrictions; ranking uses coverage of FTS-indexed content/task,
    effective activation and feedback rewards. FTS position is not evidence
    on either scale.
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
        bm25_weight * (candidate.lexical_score if qualification else candidate.bm25_score)
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
        qualification=qualification,
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

    Lexical overlap alone does not qualify (see the gate constants above).
    ``graph_score``
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


def _historical_window_scores(
    batches: Iterable[Sequence[Sequence[Any]]], query: Sequence[float],
) -> Iterator[tuple[str, float]]:
    """Stream provisional cosines with bounded scratch space."""
    direction = _as_query_array(query)
    for batch in batches:
        if _np is None:
            for node_id, blob in batch:
                yield node_id, cosine_similarity(query, unpack_chunk_embedding(blob, len(query)))
        elif direction is not None:
            matrix = _np.frombuffer(b"".join(row[1] for row in batch),
                                    dtype=CHUNK_EMBEDDING_DTYPE).reshape(-1, len(query))
            norms = _np.sqrt(_np.einsum("ij,ij->i", matrix, matrix))
            scores = _np.einsum("ij,j->i", matrix, direction) / _np.where(norms > 0, norms, 1)
            for row, score in zip(batch, scores, strict=True):
                yield row[0], float(score)


def _as_query_array(query_embedding: Sequence[float]) -> Any:
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

#: Cosine subtracted per doubling of a node's chunk count after max pooling.
LENGTH_BIAS_LOG2_COEFFICIENT = 0.031


class _BlockAccumulator:
    """Collect ordered BLOBs once, retaining one raw array per vector width."""

    __slots__ = ("payload", "node_ids", "starts", "rows", "chunk_ids", "source_versions")

    def __init__(self) -> None:
        self.payload = bytearray()
        self.node_ids: list[str] = []
        self.starts: list[int] = []
        self.rows = 0
        self.chunk_ids: list[str] = []
        self.source_versions: list[str] = []

    def add(self, node_id: str, view: Any, chunk_id: str, version: str) -> None:
        # iter_chunk_embedding_rows promises rows grouped by node, so a node id
        # that differs from the last one can only mean a new run.
        if not self.node_ids or self.node_ids[-1] != node_id:
            self.node_ids.append(node_id)
            self.source_versions.append(version)
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
                source_versions=tuple(self.source_versions), chunk_ids=tuple(self.chunk_ids),
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
            source_versions=tuple(self.source_versions), chunk_ids=tuple(self.chunk_ids),
        )


def _prepare_chunk_gather(blocks: tuple[_ChunkBlock, ...]) -> _ChunkGatherPlan:
    """Prepare the storage scan's exact row order once for these source blocks."""

    runs = sorted(
        (node_id, block, index)
        for block in blocks for index, node_id in enumerate(block.node_ids)
    )
    node_ids = tuple(node_id for node_id, _, _ in runs)
    versions = tuple(block.source_versions[index] for _, block, index in runs)
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
            source_versions=versions, chunk_ids=chunk_ids, encoder_id=blocks[0].encoder_id,
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
        source_versions=versions, chunk_ids=chunk_ids, encoder_id=blocks[0].encoder_id,
    )
    return _ChunkGatherPlan(blocks, (), template)


def _pooled_contenders(
    pooled: Any, node_ids: tuple[str, ...], keep_n: int, lexical_ids: Collection[str], order: int,
) -> tuple[dict[int, int], int]:
    """Return block winners and lexical positions with their original qualifying ranks."""
    lexical = []
    for node_id in lexical_ids:
        index = bisect_left(node_ids, node_id)
        if index < len(node_ids) and node_ids[index] == node_id:
            lexical.append(index)
    lexical.sort()
    ranks: dict[int, int] = {}
    winners = _np.empty(0, dtype=_np.intp)
    winner_ranks = _np.empty(0, dtype=_np.intp)
    # Bound sort and rank scratch independently of corpus size. Earlier
    # winners precede new rows so stable sorting retains the first equal score.
    tile_size = max(4096, keep_n)
    for start in range(0, len(pooled), tile_size):
        tile = pooled[start:start + tile_size]
        # A Python threshold can lie between two adjacent float32 scores.
        eligible = _np.flatnonzero(tile.astype(_np.float64) >= VECTOR_MATCH_THRESHOLD)
        first = bisect_left(lexical, start)
        stop = bisect_left(lexical, start + len(tile), first)
        for index in lexical[first:stop]:
            if float(pooled[index]) >= VECTOR_MATCH_THRESHOLD:
                ranks[index] = order - int(eligible.searchsorted(index - start))

        indices = _np.concatenate((winners, start + eligible))
        ordinals = _np.concatenate((winner_ranks, order - _np.arange(len(eligible))))
        if len(indices) > keep_n:
            top = _np.argsort(-pooled[indices], kind="stable")[:keep_n]
            winners, winner_ranks = indices[top], ordinals[top]
        else:
            winners, winner_ranks = indices, ordinals
        order -= len(eligible)
    ranks.update(zip(winners.tolist(), winner_ranks.tolist()))
    return ranks, order


def _select_block_matches(
    block: _ChunkBlock, query_embedding: Sequence[float], q_arr: Any, order: int,
    keep_n: int, lexical_ids: Collection[str], discovery: list[_VectorMatch], lexical: list[_VectorMatch],
) -> int:
    scores, chunk_id = _pooled_chunk_similarities(block, query_embedding, q_arr)
    ranks = None
    if _np is not None and isinstance(scores, _np.ndarray):
        # A dense lexical set or discovery budget already needs most of the
        # block. Stream it directly without building positions or sorting.
        if len(scores) > max(2 * keep_n, 8 * len(lexical_ids)):
            ranks, order = _pooled_contenders(scores, block.node_ids, keep_n, lexical_ids, order)
            rows = ((i, (float(scores[i]), i)) for i in sorted(ranks))
        else:
            rows = enumerate(zip(map(float, scores), range(len(scores))))
    else:
        rows = enumerate(scores)
    pending: dict[int, int] = {}
    for index, (similarity, position) in rows:
        if not similarity >= VECTOR_MATCH_THRESHOLD:
            continue
        if ranks is None:
            rank = order
            order -= 1
        else:
            rank = ranks[index]
        node_id = block.node_ids[index]
        lexical_hit = node_id in lexical_ids
        # Later ties cannot displace the earliest matching node.
        discover = len(discovery) < keep_n or similarity > discovery[0][0]
        if not (discover or lexical_hit):
            continue
        item = (similarity, rank, node_id, chunk_id(position) if lexical_hit else None, block.source_versions[index])
        if lexical_hit:
            lexical.append(item)
        if discover:
            if len(discovery) < keep_n:
                heappush(discovery, item)
            else:
                evicted = heapreplace(discovery, item)
                pending.pop(evicted[1], None)
            if not lexical_hit:
                pending[rank] = position

    # Only this block's surviving nonlexical discoveries still need a chunk.
    # Updating the payload preserves heap order because ranks are unique.
    if pending:
        for slot, (score, rank, node_id, _, version) in enumerate(discovery):
            if rank in pending:
                discovery[slot] = (score, rank, node_id, chunk_id(pending[rank]), version)
    return order


def _pooled_chunk_similarities(
    block: _ChunkBlock,
    query_embedding: Sequence[float],
    q_arr: Any,
) -> tuple[_np.ndarray | Iterable[tuple[float, int]], Callable[[int], str | None]]:
    """Return pooled scores with positions that resolve in this block."""
    if _np is not None and isinstance(block.rows, _np.ndarray):
        if q_arr is None:
            return (), block.chunk_ids.__getitem__
        rows = block.rows / block.norms[:, None] if block.needs_normalization else block.rows
        sims = rows @ q_arr
        # `starts` is the run boundary of each node, so one reduceat pools
        # every node at once -- no Python loop over the corpus's 60k chunk rows.
        maxima = _np.maximum.reduceat(sims, block.starts)
        pooled = _np.clip(maxima, 0.0, None)
        if LENGTH_BIAS_LOG2_COEFFICIENT:
            pooled -= LENGTH_BIAS_LOG2_COEFFICIENT * block.log2_chunk_counts
            _np.clip(pooled, 0.0, None, out=pooled)

        def chunk_id(index: int) -> str | None:
            start = int(block.starts[index])
            end = int(block.starts[index + 1]) if index + 1 < len(block.starts) else block.row_count
            row = start if end == start + 1 else start + int(sims[start:end].argmax())
            return None if sims[row] != sims[row] else block.chunk_ids[row]

        return pooled, chunk_id

    return _scalar_chunk_scores(block, query_embedding), block.chunk_ids.__getitem__


def _scalar_chunk_scores(block: _ChunkBlock, query_embedding: Sequence[float]) -> Iterator[tuple[float, int]]:
    for index in range(len(block.node_ids)):
        best = 0.0
        winner = block.starts[index]
        end = block.starts[index + 1] if index + 1 < len(block.starts) else block.row_count
        for position in range(winner, end):
            similarity = cosine_similarity(query_embedding, list(block.rows[position]))
            if similarity > best:
                best, winner = similarity, position
        penalty = LENGTH_BIAS_LOG2_COEFFICIENT * block.log2_chunk_counts[index]
        yield max(0.0, best - penalty), winner
