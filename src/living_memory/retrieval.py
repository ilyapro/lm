"""Recall service combining scope, FTS5, embeddings, graph traversal, and feedback."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, replace
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
from living_memory.scope import ScopePlan, ScopeResolver
from living_memory.storage import MemoryStore

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
        self.vector_scan_limit = int(vector_scan_limit)
        self.scope_resolver = ScopeResolver()
        self.feedback = FeedbackService(store)
        self.last_recall_event_id: str | None = None
        self._embedding_cache: dict[str, Any] = {}

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
        superseded_ids, superseding_ids = self._supersedes_sets()
        for candidate in candidates.values():
            node = candidate.node
            if node.decayed or not plan.allows(node.scope):
                continue
            if _is_rejected_alternative_node(node) and not decision_mode:
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
                superseded=node.id in superseded_ids,
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
                )
            )

        return sorted(
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
        per_scope_limit = max(50, max_results * 12)
        query_embedding = self.embedder.embed(query)

        # Lazy backfill: ensure every active node in the searched scopes has an
        # embedding before the scan. In steady state these queries return zero
        # rows. The loop drains batches until none are left so first-time
        # recalls populate the full scope just like the previous code did.
        for scope in plan.scopes:
            while True:
                unembedded = self.store.list_unembedded_nodes(scope=scope, limit=500)
                if not unembedded:
                    break
                for node in unembedded:
                    self._ensure_embedding(node)

        q_arr = _as_query_array(query_embedding)
        scoped_scores: list[tuple[float, str]] = []
        for scope in plan.scopes:
            ids, vectors = self._scan_scope_embeddings(scope)
            if not ids:
                continue
            sims = _batch_similarity(q_arr, query_embedding, vectors)
            for sim, node_id in zip(sims, ids, strict=True):
                if sim >= VECTOR_MATCH_THRESHOLD:
                    scoped_scores.append((sim, node_id))

        if not scoped_scores:
            return

        scoped_scores.sort(key=lambda item: item[0], reverse=True)
        keep_n = per_scope_limit * max(1, len(plan.scopes))
        for similarity, node_id in scoped_scores[:keep_n]:
            existing = candidates.get(node_id)
            if existing is None:
                node = self.store.get_node(node_id)
                if node is None or node.decayed or not plan.allows(node.scope):
                    continue
                existing = candidates.setdefault(node_id, _Candidate(node=node))
            existing.vector_score = max(existing.vector_score, similarity)

    def _scan_scope_embeddings(self, scope: str) -> tuple[list[str], list[Any]]:
        """Return cached (ids, vectors) for active embedded nodes in ``scope``."""

        ids: list[str] = []
        vectors: list[Any] = []
        cache = self._embedding_cache
        for node_id, embedding_json in self.store.iter_embedding_rows(scope=scope):
            vector = cache.get(node_id)
            if vector is None:
                try:
                    parsed = json.loads(embedding_json)
                except (TypeError, ValueError):
                    continue
                if not parsed:
                    continue
                vector = _np.asarray(parsed, dtype=_np.float32) if _np is not None else parsed
                cache[node_id] = vector
            ids.append(node_id)
            vectors.append(vector)
        return ids, vectors

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
        embedding = self.embedder.embed(node.content)
        updated = self.store.update_node(node.id, embedding=embedding)
        node.embedding = updated.embedding
        return updated.embedding or embedding

    def _record_result_access(self, result: RecallResult) -> RecallResult:
        updated_node = self.store.record_access(result.node.id)
        return replace(result, node=updated_node)

    def _supersedes_sets(self) -> tuple[set[str], set[str]]:
        rows = self.store.connection.execute(
            "SELECT source_id, target_id FROM connections WHERE type = 'supersedes'"
        ).fetchall()
        superseding = {str(row["source_id"]) for row in rows}
        superseded = {str(row["target_id"]) for row in rows}
        return superseded, superseding


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


def _batch_similarity(
    q_arr: Any,
    query_embedding: list[float],
    vectors: list[Any],
) -> list[float]:
    """Vectorized cosine for unit-length stored embeddings, with a pure-Python fallback."""

    if _np is not None and q_arr is not None and vectors and isinstance(vectors[0], _np.ndarray):
        mat = _np.vstack(vectors)
        sims = mat @ q_arr
        sims = _np.clip(sims, 0.0, None)
        return sims.tolist()
    return [max(0.0, cosine_similarity(query_embedding, vec)) for vec in vectors]
