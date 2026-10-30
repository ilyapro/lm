"""Vector discovery admits current source identities before decoding bodies."""

from dataclasses import replace
import sqlite3
import sys

import pytest

from living_memory import retrieval, storage
from living_memory.embeddings import cosine_similarity
from living_memory.retrieval import MemoryRecallService, _Candidate
from living_memory.scope import ScopePlan, resolve_scope
from living_memory.storage import MemoryStore
from test_retrieval_chunk_maxpool import put_chunks


QUERY = "orbital telemetry drift"
VECTOR = [1.0, 0.0, 0.0]
SCOPE = "project:vector"


class Encoder:
    def embed(self, text):
        return list(VECTOR)


def put(store, node_id, *, scope=SCOPE):
    return store.create_node(
        node_id=node_id, level="concept", content=f"Cobalt prism specimen {node_id}",
        context={"scope": scope}, provenance={"detail": node_id}, embedding=VECTOR,
        stats={"unique_agents": 2},
    )


def recall(service, scope=SCOPE):
    return service.memory_recall(
        QUERY, scope=scope, depth=0, max_results=2,
        log_access=False, log_event=False, repair_pending=False,
    )


def full_pool_oracle(blocks, query):
    """Unbounded reference arithmetic and first-row winners, without selection."""
    result = []
    np = retrieval._np
    for block in blocks:
        if np is not None:
            q = retrieval._as_query_array(query)
            if q is None:
                continue
            rows = block.rows / block.norms[:, None] if block.needs_normalization else block.rows
            similarities = rows @ q
            maxima = np.maximum.reduceat(similarities, block.starts)
            scores = np.clip(maxima, 0.0, None)
            scores -= retrieval.LENGTH_BIAS_LOG2_COEFFICIENT * block.log2_chunk_counts
            np.clip(scores, 0.0, None, out=scores)
        for index, node_id in enumerate(block.node_ids):
            start = int(block.starts[index])
            end = int(block.starts[index + 1]) if index + 1 < len(block.starts) else block.row_count
            if np is not None:
                winner = next((row for row in range(start, end) if similarities[row] == maxima[index]), None)
                score = float(scores[index])
            else:
                best, winner = 0.0, start
                for row in range(start, end):
                    similarity = cosine_similarity(query, list(block.rows[row]))
                    if similarity > best:
                        best, winner = similarity, row
                score = max(0.0, best - retrieval.LENGTH_BIAS_LOG2_COEFFICIENT * block.log2_chunk_counts[index])
            result.append((score, node_id, block.chunk_ids[winner] if winner is not None else None))
    return result


def selection_oracle(store, blocks, query, plan, initial, max_results=2):
    # Stable full sort is independent of the bounded heap and includes lexical
    # evidence only after establishing the discovery cutoff over all matches.
    scores = [row for row in full_pool_oracle(blocks, query)
              if row[0] >= retrieval.VECTOR_MATCH_THRESHOLD]
    scores.sort(key=lambda row: row[0], reverse=True)
    keep_n = max(50, max_results * 12) * max(1, len(plan.scopes))
    expected = {key: replace(value) for key, value in initial.items()}
    for rank, (score, node_id, chunk_id) in enumerate(scores):
        if rank >= keep_n and node_id not in initial:
            continue
        if node_id not in expected:
            expected[node_id] = _Candidate(node=store.get_node(node_id))
        if score > expected[node_id].vector_score:
            expected[node_id].vector_score = score
            expected[node_id].chunk_id = chunk_id
    return expected


def measure_vector_state(call):
    """Peak Python containers in both producer/selector frames, including yields.

    Cached blocks, numeric arrays, input bodies/query and referenced strings
    are outside this measurement. Tuples, floats and order integers are counted
    once by identity, even when both the lexical list and heap reference them.
    """
    codes = {retrieval._pooled_chunk_similarities.__code__, MemoryRecallService._collect_vector.__code__}
    peak = {"container_entries": 0, "records": 0, "bytes": 0, "observations": 0}

    def observe(frame, event, arg):
        if frame.f_code not in codes:
            return None
        if event not in {"line", "return"}:
            return observe
        seen = set()
        entries = records = size = 0

        def visit(value):
            nonlocal entries, records, size
            if type(value) not in (list, tuple, dict, set, float, int) or id(value) in seen:
                return
            seen.add(id(value))
            size += sys.getsizeof(value)
            if type(value) in (float, int):
                return
            entries += len(value)
            if isinstance(value, tuple) and value and isinstance(value[0], float):
                records += 1
            for item in value.values() if isinstance(value, dict) else value:
                visit(item)

        current = frame
        while current is not None and current.f_code in codes:
            for name, value in current.f_locals.items():
                if name not in {"self", "query_embedding", "q_arr", "plan", "candidates", "block", "fetched"}:
                    visit(value)
            current = current.f_back
        if event == "return":
            visit(arg)
        peak["container_entries"] = max(peak["container_entries"], entries)
        peak["records"] = max(peak["records"], records)
        peak["bytes"] = max(peak["bytes"], size)
        peak["observations"] += 1
        return observe

    previous = sys.gettrace()
    try:
        sys.settrace(observe)
        call()
    finally:
        sys.settrace(previous)
    return peak


@pytest.mark.parametrize("entry", ["collection", "recall"])
@pytest.mark.parametrize("mutation", ["scope", "decay", "remove", "admitted"])
def test_current_admission_after_chunk_selection(tmp_path, monkeypatch, entry, mutation):
    path = tmp_path / "admission.sqlite3"
    with MemoryStore(path) as store, MemoryStore(path) as writer:
        put(store, "changed")
        survivor = put(store, "survivor")
        service = MemoryRecallService(store, embedder=Encoder())
        plan = resolve_scope(scope=SCOPE)
        blocks = service._chunk_blocks
        decode = storage._node_from_row
        decoded = []

        def change_after_selection(plan, dimension):
            selected = blocks(plan, dimension)
            assert not store.connection.in_transaction
            with writer.transaction():
                if mutation == "remove":
                    writer.delete_node_chunks("changed")
                    writer.connection.execute("DELETE FROM nodes WHERE id = 'changed'")
                else:
                    writer.connection.execute("UPDATE nodes SET provenance = '[]' WHERE id = 'changed'")
                    if mutation == "scope":
                        writer.connection.execute(
                            "UPDATE nodes SET scope = 'project:excluded' WHERE id = 'changed'",
                        )
                    elif mutation == "decay":
                        writer.connection.execute("UPDATE nodes SET decayed = 1 WHERE id = 'changed'")
            return selected

        def count(row):
            decoded.append(row["id"])
            return decode(row)

        def collect():
            candidates = {}
            service._collect_vector(VECTOR, plan, candidates, max_results=2)
            return candidates

        with monkeypatch.context() as patch:
            patch.setattr(service, "_chunk_blocks", change_after_selection)
            patch.setattr(storage, "_node_from_row", count)
            if mutation == "admitted":
                with pytest.raises(TypeError):
                    collect() if entry == "collection" else recall(service)
                assert "changed" in decoded
                return
            actual = collect() if entry == "collection" else recall(service)
        assert decoded == [survivor.id]
        if entry == "collection":
            chunk_id = next(store.iter_chunk_embedding_rows(scope=SCOPE, include_identity=True))[3]
            assert actual == {survivor.id: _Candidate(node=survivor, vector_score=1.0, chunk_id=chunk_id)}
        else:
            assert [result.node.id for result in actual] == [survivor.id]
            assert actual[0].bm25_score == 0.0
            assert actual[0].vector_score == pytest.approx(1.0)
            assert actual == recall(MemoryRecallService(store, embedder=Encoder()))


@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "scalar"])
@pytest.mark.parametrize("plan", [
    ScopePlan(SCOPE, (SCOPE, "global")),
    ScopePlan(SCOPE, ("global", SCOPE)),
    ScopePlan("global", (SCOPE, "global", "*")),
    ScopePlan("global", ("*",)),
])
def test_healthy_candidates_keep_scores_order_chunks_limits_and_existing_bodies(tmp_path, monkeypatch, plan, numpy):
    if not numpy:
        monkeypatch.setattr(retrieval, "_np", None)
    with MemoryStore(tmp_path / "healthy.sqlite3") as store:
        for scope in (SCOPE, "global", "project:foreign"):
            for index in range(55):
                put(store, f"{scope}-{index:02}", scope=scope)
        tail = put(store, "tail")
        put_chunks(store, tail.id, [[0.0, 1.0, 0.0], [0.8, 0.6, 0.0], [0.8, 0.6, 0.0]])
        strongest = put(store, "strongest")
        put_chunks(store, strongest.id, [[0.8, 0.6, 0.0]])
        initial = {
            tail.id: _Candidate(node=tail, vector_score=0.1, chunk_id="previous", bm25_score=0.7,
                                graph_score=0.4, trigger_score=0.6, trigger_coverage=0.5,
                                path=("root",), anchor_score=0.3, source_graph_score=0.2,
                                lexical_score=0.7),
            strongest.id: _Candidate(node=strongest, vector_score=1.1, chunk_id="retained"),
        }
        service = MemoryRecallService(store, embedder=Encoder())
        expected = selection_oracle(store, service._chunk_blocks(plan, 3), VECTOR, plan, initial)
        actual = {key: replace(value) for key, value in initial.items()}
        decoded = []
        decode = storage._node_from_row
        blocks = service._chunk_blocks

        def count(row):
            decoded.append(row["id"])
            return decode(row)

        def change_after_selection(plan, dimension):
            selected = blocks(plan, dimension)
            with store.transaction():
                store.connection.execute(
                    "UPDATE nodes SET provenance = '[]', decayed = 1 WHERE id IN (?, ?)",
                    tuple(initial),
                )
            return selected

        with monkeypatch.context() as patch:
            patch.setattr(storage, "_node_from_row", count)
            patch.setattr(service, "_chunk_blocks", change_after_selection)
            service._collect_vector(VECTOR, plan, actual, max_results=2)
        # Existing candidates retain their bodies and vector evidence even when
        # the source changes after chunk selection; only new bodies are admitted.
        assert actual == expected
        assert list(actual) == list(expected)
        assert all(actual[key].node is initial[key].node for key in initial)
        assert set(decoded) == actual.keys() - initial.keys()
        assert len(decoded) == len(set(decoded))
        assert len(actual) == 50 * len(plan.scopes) + 2
        assert actual[tail.id].chunk_id != "previous"
        assert actual[strongest.id].chunk_id == "retained"


@pytest.mark.parametrize("scope", [None, SCOPE])
def test_healthy_vector_only_recall_matches_full_body_reader(tmp_path, monkeypatch, scope):
    with MemoryStore(tmp_path / "recall.sqlite3") as store:
        for item_scope in (SCOPE, "global", "project:foreign"):
            for index in range(3):
                put(store, f"{item_scope}-{index}", scope=item_scope)
        service = MemoryRecallService(store, embedder=Encoder())
        actual = recall(service, scope)
        residual = service.last_residual
        assert actual and residual
        assert all(item.bm25_score == 0 and item.vector_score > 0 for item in [*actual, *residual])
        monkeypatch.setattr(store, "get_recall_nodes", lambda ids, plan: store.get_nodes(ids))
        assert recall(service, scope) == actual
        assert service.last_residual == residual


@pytest.mark.parametrize("table,column", [
    ("nodes", "provenance"), (storage.CHUNK_EMBEDDING_TABLE, "embedding"),
])
def test_vector_collection_propagates_body_and_index_sql_errors(tmp_path, table, column):
    with MemoryStore(tmp_path / "errors.sqlite3") as store:
        put(store, "source")
        service = MemoryRecallService(store, embedder=Encoder())

        def deny_read(action, name, field, *_):
            if action == sqlite3.SQLITE_READ and (name, field) == (table, column):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        store.connection.set_authorizer(deny_read)
        try:
            with pytest.raises(sqlite3.DatabaseError, match="prohibited"):
                service._collect_vector(VECTOR, resolve_scope(scope=SCOPE), {}, max_results=2)
        finally:
            store.connection.set_authorizer(None)


@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "scalar"])
def test_vector_discovery_retains_bounded_scores_and_winners(tmp_path, monkeypatch, numpy):
    if not numpy:
        monkeypatch.setattr(retrieval, "_np", None)
    with MemoryStore(tmp_path / "bounded.sqlite3") as store:
        with store.transaction():
            for number in range(10000):
                put(store, f"n{number:05}")
        service = MemoryRecallService(store, embedder=Encoder())
        plan = ScopePlan(SCOPE, (SCOPE,))
        blocks = service._chunk_blocks(plan, 3)
        tail = store.get_node("n09999")
        candidates = {tail.id: _Candidate(node=tail, bm25_score=0.7)}
        expected = selection_oracle(store, blocks, VECTOR, plan, candidates)
        # Every rejected body is malformed: the bounded discovery must never
        # fetch or decode it, even though its vector is fully scored.
        with store.transaction():
            store.connection.execute(
                "UPDATE nodes SET provenance = '[]' WHERE id >= 'n00050' AND id != 'n09999'",
            )
        peak = measure_vector_state(lambda: service._collect_vector(VECTOR, plan, candidates, max_results=2))
        assert candidates == expected
        assert list(candidates) == list(expected)
        assert len(candidates) == 51
        assert peak["observations"] > 10000
        # At most K discovery records, L lexical records and transient producer
        # /selector tuples; container slots include the final union and ID list.
        assert peak["records"] <= 50 + 1 + 3, peak
        assert peak["container_entries"] <= 12 * (50 + 1) + 64, peak


@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "scalar"])
def test_streamed_pool_matches_full_arithmetic_and_first_winners(tmp_path, monkeypatch, numpy):
    if not numpy:
        monkeypatch.setattr(retrieval, "_np", None)
    with MemoryStore(tmp_path / "arithmetic.sqlite3") as store:
        vectors = [
            [[0.0, 1.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 0.0]],
            [[1.00005, 0.0, 0.0]], [[0.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [-0.8, 0.6, 0.0]],
            [[0.1, 0.99, 0.0]], [[0.2, 0.98, 0.0]], [[0.3, 0.95, 0.0]],
        ]
        for index, chunks in enumerate(vectors):
            node = put(store, f"node-{index}", scope=(SCOPE, "global")[index % 2])
            put_chunks(store, node.id, chunks + [[0.6, 0.8]])
        service = MemoryRecallService(store, embedder=Encoder())
        for plan in (ScopePlan(SCOPE, (SCOPE, "global")), ScopePlan("global", ("*",))):
            for query in (VECTOR, [0.3, 0.9, 0.2], [0.6, 0.8], [0.0, 0.0, 0.0]):
                blocks = service._chunk_blocks(plan, len(query))
                expected = full_pool_oracle(blocks, query)
                actual = [row for block in blocks for row in retrieval._pooled_chunk_similarities(
                    block, query, retrieval._as_query_array(query),
                )]
                assert actual == expected
                # Exercise the admission boundary exactly at a computed score.
                if expected:
                    monkeypatch.setattr(retrieval, "VECTOR_MATCH_THRESHOLD", expected[0][0])
                candidates = {}
                service._collect_vector(query, plan, candidates, max_results=2)
                assert candidates == selection_oracle(store, blocks, query, plan, {})


def test_vector_scoring_error_after_discovery_cutoff_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieval, "_np", None)
    with MemoryStore(tmp_path / "scoring-error.sqlite3") as store:
        for number in range(51):
            put(store, f"n{number:03}")
        cosine = retrieval.cosine_similarity
        calls = 0

        def fail_late(*args):
            nonlocal calls
            calls += 1
            if calls == 51:
                raise ArithmeticError("required scoring failed")
            return cosine(*args)

        monkeypatch.setattr(retrieval, "cosine_similarity", fail_late)
        with pytest.raises(ArithmeticError, match="required scoring failed"):
            MemoryRecallService(store)._collect_vector(VECTOR, ScopePlan(SCOPE, (SCOPE,)), {}, max_results=2)
        assert calls == 51
