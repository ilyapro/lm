"""Derivative refresh preserves its outcome without repeated source preparation."""

from collections import Counter

import pytest

from living_memory import consolidation as c
from living_memory.storage import MemoryStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    with MemoryStore(tmp_path / "memory.sqlite3") as instance:
        yield instance


def cluster(store, traces):
    group = c._TraceCluster("project:harbor", traces, Counter({"harbor": len(traces)}))
    node, _ = c._merge_cluster_into_concept(store, group, c._assess_cluster_eras(store, traces))
    c._update_edge_weights_from_co_access(store, node, traces)
    return node


def sources(store):
    return [
        store.append_trace(text, {"scope": "project:harbor", "agent": "engineer"})
        for text in (
            "Harbor crane requires the paper permit.",
            "Harbor crane inspection includes the brake assembly.",
            "Harbor crane maintenance records the cable tension.",
        )
    ]


def test_teach_prepares_each_current_cluster_once(store, monkeypatch):
    traces = sources(store)
    concept = cluster(store, traces)
    counts = Counter()
    for name in ("_current_derivation", "_synthesize_digest"):
        original = getattr(c, name)

        def observe(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(c, name, observe)
    correction = "Harbor crane requires the signed electronic manifest."
    taught = c.memory_teach(store, traces[0].id, correction)
    current = store.get_node(concept.id)
    assert correction in current.content
    assert traces[0].content not in current.content
    assert set(current.source_traces) == {taught.corrective_trace.id, traces[1].id, traces[2].id}
    assert counts == {"_current_derivation": 1, "_synthesize_digest": 1}
    assert any(edge.target_id == taught.corrective_trace.id for edge in store.list_connections(
        source_id=concept.id, relation_type="related"
    ))


def test_cluster_edge_failure_rolls_back_the_entire_correction(store, monkeypatch):
    traces = sources(store)
    cluster(store, traces)
    before = list(store.connection.iterdump())
    update = c._update_edge_weights_from_co_access

    def fail_after_edges(*args):
        update(*args)
        raise RuntimeError("required derived edge failure")

    monkeypatch.setattr(c, "_update_edge_weights_from_co_access", fail_after_edges)
    with pytest.raises(RuntimeError, match="required derived edge failure"):
        c.memory_teach(store, traces[0].id, "Harbor crane uses an electronic manifest.")
    assert list(store.connection.iterdump()) == before
    assert not store.connection.in_transaction


def test_connections_validate_identity_without_loading_endpoint_documents(store, monkeypatch):
    traces = sources(store)
    concept = cluster(store, traces)
    store.soft_delete_node(traces[0].id, "historical source")

    def no_documents(*args):
        raise AssertionError("edge existence checks must not load node documents")

    monkeypatch.setattr(store, "get_node", no_documents)
    edge = store.create_connection(concept.id, traces[0].id, "related", weight=0.75,
                                   metadata={"basis": "cluster"})
    assert edge.source_id == concept.id and edge.target_id == traces[0].id
    assert edge.weight == 0.75 and edge.metadata == {"basis": "cluster"}
    updated = store.create_connection(concept.id, traces[0].id, "related", weight=0.8)
    assert updated.id == edge.id and updated.weight == 0.8
    assert store.create_connection(concept.id, concept.id, "related").target_id == concept.id
    for source, target, missing in (("absent-source", "absent-target", "absent-source"),
                                    (concept.id, "absent-target", "absent-target")):
        with pytest.raises(KeyError) as error:
            store.create_connection(source, target, "related")
        assert error.value.args == (missing,)
    with pytest.raises(ValueError):
        store.create_connection("absent-source", "absent-target", "invalid")
