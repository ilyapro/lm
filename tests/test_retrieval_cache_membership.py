"""Warm vector discovery follows committed membership before candidate limits."""

from contextlib import ExitStack
from math import sqrt

import pytest

from living_memory import retrieval
from living_memory.decay import memory_forget
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore


QUERY = "orbital telemetry drift"
VECTOR = [1.0, 0.0, 0.0]


class FixedEmbedder:
    def embed(self, text):
        return list(VECTOR)


@pytest.fixture(params=[False, True], ids=["same-connection", "second-connection"])
def stores(tmp_path, request):
    with ExitStack() as stack:
        path = tmp_path / "memory.sqlite3"
        reader = stack.enter_context(MemoryStore(path))
        writer = stack.enter_context(MemoryStore(path)) if request.param else reader
        yield reader, writer


def add_node(store, content, *, scope="global", similarity=1.0):
    return store.create_node(
        level="concept",
        content=content,
        context={"scope": scope},
        embedding=[similarity, sqrt(1.0 - similarity**2), 0.0],
    ).id


def service(store):
    return MemoryRecallService(store, embedder=FixedEmbedder())


def recall(retriever, scope="global"):
    return retriever.memory_recall(QUERY, scope=scope, depth=0, max_results=1)


def assert_same_knowledge(warm, fresh, expected):
    assert [r.node.id for r in warm] == [r.node.id for r in fresh] == expected
    assert [r.node.content for r in warm] == [r.node.content for r in fresh]
    assert [r.vector_score for r in warm] == pytest.approx([r.vector_score for r in fresh])
    assert all(r.bm25_score == 0.0 and not r.node.decayed for r in warm)


@pytest.mark.parametrize("mutation", ["forget", "move"])
@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "python"])
def test_membership_changes_precede_vector_candidate_limit(stores, mutation, numpy, monkeypatch):
    if not numpy:
        monkeypatch.setattr(retrieval, "_np", None)
    reader, writer = stores
    retired = [add_node(writer, f"obsolete fact {i}") for i in range(50)]
    survivor = add_node(writer, "cobalt prism remains useful", similarity=0.95)
    warm = service(reader)
    assert recall(warm)[0].node.id in retired

    for node_id in retired:
        if mutation == "forget":
            memory_forget(writer, node_id)
        else:
            writer.update_node(node_id, context={"scope": "project:elsewhere"})

    result = recall(warm)
    assert_same_knowledge(result, recall(service(reader)), [survivor])
    assert result[0].vector_score == pytest.approx(0.95)


def test_scope_move_populates_warm_empty_destination(stores):
    reader, writer = stores
    node_id = add_node(writer, "cobalt prism", scope="project:source")
    warm = service(reader)
    assert recall(warm, "project:destination") == []
    assert recall(warm, "project:source")[0].node.id == node_id
    writer.update_node(node_id, context={"scope": "project:destination"})

    assert_same_knowledge(
        recall(warm, "project:destination"),
        recall(service(reader), "project:destination"),
        [node_id],
    )
    assert recall(warm, "project:source") == []
    assert_same_knowledge(recall(warm, None), recall(service(reader), None), [node_id])


def test_revival_populates_warm_empty_corpus(stores):
    reader, writer = stores
    node_id = add_node(writer, "cobalt prism")
    memory_forget(writer, node_id)
    warm = service(reader)
    assert recall(warm) == []
    with writer.connection:
        writer.connection.execute("UPDATE nodes SET decayed = 0 WHERE id = ?", (node_id,))
    assert_same_knowledge(recall(warm), recall(service(reader)), [node_id])


def test_bookkeeping_keeps_cached_matrices(stores):
    reader, writer = stores
    node_id = add_node(writer, "cobalt prism")
    warm = service(reader)
    assert recall(warm)[0].node.id == node_id
    cached = dict(warm._chunk_index)

    writer.record_access(node_id)
    writer.update_node(node_id, stats={"usefulness_score": 0.75})
    writer.add_correction(node_id, old="old wording", new="current wording", by="test")
    writer.update_node(node_id, context={"scope": "global", "task": "bookkeeping"})
    for _ in range(3):
        assert recall(warm)[0].node.id == node_id
        assert all(warm._chunk_index[scope] is index for scope, index in cached.items())
