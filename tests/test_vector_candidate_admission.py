"""Vector discovery admits current source identities before decoding bodies."""

from dataclasses import replace
import sqlite3

import pytest

from living_memory import storage
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
            chunk_id = next(store.iter_chunk_embedding_rows(include_identity=True))[3]
            assert actual == {survivor.id: _Candidate(node=survivor, vector_score=1.0, chunk_id=chunk_id)}
        else:
            assert [result.node.id for result in actual] == [survivor.id]
            assert actual[0].bm25_score == 0.0
            assert actual[0].vector_score == pytest.approx(1.0)
            assert actual == recall(MemoryRecallService(store, embedder=Encoder()))


@pytest.mark.parametrize("plan", [
    ScopePlan(SCOPE, (SCOPE, "global")),
    ScopePlan("global", (SCOPE, "global", "*")),
    ScopePlan("global", ("*",)),
])
def test_healthy_candidates_keep_scores_order_chunks_limits_and_existing_bodies(tmp_path, monkeypatch, plan):
    with MemoryStore(tmp_path / "healthy.sqlite3") as store:
        for scope in (SCOPE, "global", "project:foreign"):
            for index in range(55):
                put(store, f"{scope}-{index:02}", scope=scope)
        tail = put(store, "tail")
        put_chunks(store, tail.id, [[0.0, 1.0, 0.0], [0.8, 0.6, 0.0]])
        strongest = put(store, "strongest")
        initial = {
            tail.id: _Candidate(node=tail, vector_score=0.1, chunk_id="previous", bm25_score=0.7,
                                graph_score=0.4, trigger_score=0.6, trigger_coverage=0.5,
                                path=("root",), anchor_score=0.3, source_graph_score=0.2,
                                lexical_score=0.7),
            strongest.id: _Candidate(node=strongest, vector_score=1.1, chunk_id="retained"),
        }
        service = MemoryRecallService(store, embedder=Encoder())
        expected = {key: replace(value) for key, value in initial.items()}
        with monkeypatch.context() as patch:
            patch.setattr(store, "get_recall_nodes", lambda ids, plan: {
                key: node for key, node in store.get_nodes(ids).items()
                if not node.decayed and plan.allows(node.scope)
            })
            service._collect_vector(VECTOR, plan, expected, max_results=2)
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
