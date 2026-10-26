"""Bounded graph evidence keeps traversal order, freshness and score provenance."""

from collections import Counter
from dataclasses import asdict

import pytest

import living_memory.retrieval as retrieval
import living_memory.storage as storage
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore


class Encoder:
    def embed(self, text):
        return [1.0, 0.0] if "needle" in text else [0.0, 1.0]


def node(store, name, *, scope="project:graph", content=None):
    text = content or name
    return store.create_node(
        level="trace", node_id=name, content=text,
        context={"scope": scope}, embedding=Encoder().embed(text),
    )


def anchor(store, *targets):
    value = store.insert_query_anchor(
        scope="project:graph", query="needle", fingerprint="needle",
        embedding=[1.0, 0.0],
    )
    for target in targets:
        store.upsert_query_anchor_edge(value.id, target, weight=1.0)


def recall(service, depth=1):
    delivered = service.memory_recall(
        "needle", scope="project:graph", depth=depth, max_results=5,
        log_access=False, log_event=False,
    )
    return [asdict(r) for r in delivered], [asdict(r) for r in service.last_residual]


class RecordingService(MemoryRecallService):
    def rank_candidates(self, candidates, *args, **kwargs):
        self.evidence = {
            key: (value.graph_score, value.anchor_score, value.path)
            for key, value in candidates.items()
        }
        return super().rank_candidates(candidates, *args, **kwargs)


@pytest.mark.parametrize("size", [12, 430])
@pytest.mark.parametrize("limit", [0, 1, 5, 200])
def test_reader_preserves_bulk_order_ties_self_edges_and_batch_boundaries(tmp_path, size, limit):
    with MemoryStore(tmp_path / "graph.sqlite3") as store:
        with store.transaction():
            for i in range(size):
                node(store, str(i))
            for i in range(size):
                store.create_connection(str(i), str((i + 1) % size), "related")
                store.create_connection(str(i), str(i), "requires", weight=0.5)
                store.create_connection(str(i), "0", "caused")
            store.connection.execute("UPDATE connections SET created_at = '2000-01-01'")
        ids = [str(i) for i in range(size)] + ["missing", "0"]
        complete = store.list_connections_for_nodes(ids)
        bounded = store.list_recall_connections(ids, limit=limit)
        assert bounded == {key: edges[:limit] for key, edges in complete.items()}
        assert store.list_connections_for_nodes(ids) == complete
        assert store.list_recall_connections([], limit=limit) == {}


def test_hub_bound_precedes_materialization_and_traversal_filters(tmp_path, monkeypatch):
    with MemoryStore(tmp_path / "graph.sqlite3") as store:
        with store.transaction():
            node(store, "root", content="needle")
            for i in range(260):
                name = f"leaf-{i:03}"
                node(store, name, scope="project:other" if i == 0 else "project:graph")
                store.create_connection("root", name, "related", weight=1 - i / 1000)
            store.soft_delete_node("leaf-001", "inactive")
            anchor(store, "root")
        service = MemoryRecallService(store, embedder=Encoder())
        counts = Counter()
        original = storage._connection_from_row
        traversal = retrieval._traversal

        def convert(row):
            counts["objects"] += 1
            return original(row)

        def traverse(*args, **kwargs):
            counts["traversals"] += 1
            return traversal(*args, **kwargs)

        def trace(sql):
            if "FROM connections WHERE (source_id IN" in sql:
                counts["adjacency_queries"] += 1

        monkeypatch.setattr(storage, "_connection_from_row", convert)
        monkeypatch.setattr(retrieval, "_traversal", traverse)
        store.connection.set_trace_callback(trace)
        delivered, residual = recall(service)
        store.connection.set_trace_callback(None)
        ids = {r["node"]["id"] for r in delivered + residual}
        assert counts == {"objects": 200, "traversals": 200, "adjacency_queries": 1}
        assert "leaf-199" in ids
        assert not {"leaf-000", "leaf-001", "leaf-200"} & ids
        assert len(ids) == 199


@pytest.mark.parametrize("second_connection", [False, True])
def test_graph_changes_between_walks_and_calls_are_visible(tmp_path, monkeypatch, second_connection):
    path = tmp_path / "graph.sqlite3"
    with MemoryStore(path) as store, MemoryStore(path) as other:
        node(store, "root", content="needle")
        node(store, "leaf")
        edge = store.create_connection("root", "leaf", "related", weight=0.8)
        anchor(store, "root")
        service = RecordingService(store, embedder=Encoder())
        original = service._admit_anchor_seeds
        writer = other if second_connection else store

        def change(*args, **kwargs):
            writer.update_connection(edge.id, weight=0.2)
            return original(*args, **kwargs)

        monkeypatch.setattr(service, "_admit_anchor_seeds", change)
        recall(service)
        graph, anchored, path = service.evidence["leaf"]
        assert graph == pytest.approx(0.8 * 0.65)
        assert anchored == pytest.approx(0.2 * 0.65)
        assert path == ("root", "leaf")
        monkeypatch.setattr(service, "_admit_anchor_seeds", original)
        first = recall(service)
        assert first == recall(RecordingService(store, embedder=Encoder()))
        writer.delete_connection(edge.id)
        second = recall(service)
        assert second != first
        assert second == recall(RecordingService(store, embedder=Encoder()))
        assert "leaf" not in service.evidence


def test_transaction_observations_do_not_survive_rollback_between_walks(tmp_path, monkeypatch):
    with MemoryStore(tmp_path / "graph.sqlite3") as store:
        node(store, "root", content="needle")
        node(store, "leaf")
        edge = store.create_connection("root", "leaf", "related", weight=0.8)
        anchor(store, "root")
        service = RecordingService(store, embedder=Encoder())
        original = service._admit_anchor_seeds
        store.connection.execute("BEGIN")
        store.update_connection(edge.id, weight=0.2)

        def rollback(*args, **kwargs):
            store.connection.rollback()
            return original(*args, **kwargs)

        monkeypatch.setattr(service, "_admit_anchor_seeds", rollback)
        recall(service)
        graph, anchored, path = service.evidence["leaf"]
        assert graph == pytest.approx(0.2 * 0.65)
        assert anchored == pytest.approx(0.8 * 0.65)
        assert path == ("root", "leaf")
        monkeypatch.setattr(service, "_admit_anchor_seeds", original)
        assert recall(service) == recall(RecordingService(store, embedder=Encoder()))


def test_same_level_best_score_recheck_preserves_strongest_path(tmp_path):
    with MemoryStore(tmp_path / "graph.sqlite3") as store:
        for name in ("root", "strong", "weak", "shared", "tail"):
            node(store, name, content="needle" if name == "root" else name)
        for source, target, weight in (
            ("root", "strong", 1.0), ("root", "weak", 0.8),
            ("strong", "shared", 1.0), ("weak", "shared", 0.8),
            ("shared", "tail", 0.9), ("tail", "root", 0.1),
        ):
            store.create_connection(source, target, "requires", weight=weight)
        anchor(store, "root")
        service = RecordingService(store, embedder=Encoder())
        recall(service, depth=3)
        graph, anchored, path = service.evidence["shared"]
        assert graph == pytest.approx(0.85 * 0.85 * 0.72)
        assert anchored == graph
        assert path == ("root", "strong", "shared")
        assert service.evidence["tail"][2] == ("root", "strong", "shared", "tail")


def test_frontier_dependent_ties_keep_the_bulk_readers_neighbors(tmp_path, monkeypatch):
    with MemoryStore(tmp_path / "graph.sqlite3") as store:
        for name in ("a", "c", "d"):
            node(store, name, content="needle" if name == "c" else name)
        store.create_connection("c", "d", "related")
        store.create_connection("a", "c", "related")
        store.connection.execute("UPDATE connections SET created_at = '2000-01-01'")
        anchor(store, "a", "c")
        monkeypatch.setattr(retrieval, "MAX_NEIGHBORS_PER_NODE", 1)
        service = RecordingService(store, embedder=Encoder())
        recall(service)
        # Ordinary root c selects c->d; the anchor frontier a,c selects a->c.
        assert service.evidence["d"][0] == pytest.approx(0.65)
        assert service.evidence["d"][1] == 0
        assert service.evidence["d"][2] == ("c", "d")
        assert service.evidence["a"][0] == 0
        assert service.evidence["a"][1] == 1


def test_missing_neighbor_after_observation_is_filtered(tmp_path, monkeypatch):
    with MemoryStore(tmp_path / "graph.sqlite3") as store:
        node(store, "root", content="needle")
        node(store, "leaf")
        store.create_connection("root", "leaf", "related")
        original = store.list_recall_connections

        def read(*args, **kwargs):
            result = original(*args, **kwargs)
            store.delete_node("leaf")
            return result

        monkeypatch.setattr(store, "list_recall_connections", read)
        delivered, residual = recall(MemoryRecallService(store, embedder=Encoder()))
        assert [r["node"]["id"] for r in delivered + residual] == ["root"]
