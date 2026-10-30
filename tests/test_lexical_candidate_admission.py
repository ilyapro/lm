"""Lexical discovery preserves membership while admitting each new body once."""

from collections import Counter
from dataclasses import replace
from functools import partial
import sqlite3

import pytest

from living_memory import storage
from living_memory.retrieval import MemoryRecallService, _Candidate, _expanded_query
from living_memory.scope import ScopePlan, resolve_scope
from living_memory.storage import MemoryStore


QUERY = "amber recall boundary"
SCOPE = "project:lexical"


class Encoder:
    def embed(self, text):
        return [1.0, 0.0]

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]


def put(store, node_id, *, scope=SCOPE, **kwargs):
    return store.create_node(
        node_id=node_id, level="concept", content=f"{QUERY} {node_id}",
        context={"scope": scope, "detail": "x" * 1024},
        provenance={"detail": "y" * 1024}, embedding=[1.0, 0.0],
        stats={"unique_agents": 2, **kwargs.pop("stats", {})}, **kwargs,
    )


def collect_full_bodies(service, query, plan, candidates, *, max_results):
    """Full-node discovery control for this fixture's complete coverage."""
    assert query == QUERY
    scopes = plan.search_scopes if plan.restricted else (*plan.named_scopes, None)
    for scope in scopes:
        rows = service.store.search_content(
            _expanded_query(query), scope=scope, limit=max(25, max_results * 8),
        )
        for node, _ in rows:
            if not node.decayed and plan.allows(node.scope):
                assert node.content.startswith(QUERY)
                candidate = candidates.setdefault(node.id, _Candidate(node=node))
                candidate.bm25_score = max(candidate.bm25_score, 1.0)
                candidate.lexical_score = 1.0
                candidate.lexical_score_prepared = True


def test_search_projections_share_order_filters_task_and_validation(tmp_path):
    with MemoryStore(tmp_path / "search.sqlite3") as store:
        low = put(store, "low", stats={"confidence": 0.2})
        high = put(store, "high", stats={"confidence": 0.9})
        useful = put(store, "useful", stats={"confidence": 0.9, "usefulness_score": 0.8})
        put(store, "foreign", scope="project:foreign")
        dead = put(store, "dead")
        store.soft_delete_node(dead.id)
        task = store.create_node(level="trace", content="Recorded task evidence.",
                                 context={"scope": SCOPE, "task": "zircon-914"})
        store.create_node(level="trace", content="Other metadata.",
                          context={"scope": SCOPE, "procedure": "zircon-914"})
        for query in (QUERY, "zircon-914", '"', ""):
            for level in (None, "trace", "concept"):
                for scope in (None, SCOPE, "project:foreign"):
                    for limit in (0, 2, 25):
                        options = dict(level=level, scope=scope, limit=limit)
                        full = store.search_content(query, **options)
                        assert store.search_content_ids(query, **options) == [
                            (node.id, score) for node, score in full
                        ]
        assert [node.id for node, _ in store.search_content(QUERY, scope=SCOPE)] == [
            useful.id, high.id, low.id,
        ]
        assert [node.id for node, _ in store.search_content("zircon-914")] == [task.id]
        for search in (store.search_content, store.search_content_ids):
            assert search("", level="invalid") == []
            with pytest.raises(ValueError):
                search(QUERY, level="invalid")
        with store.transaction():
            store.connection.execute("UPDATE nodes SET provenance = '[]' WHERE id = ?", (low.id,))
        assert low.id in dict(store.search_content_ids(QUERY, scope=SCOPE))
        with pytest.raises(TypeError):
            store.search_content(QUERY, scope=SCOPE)


@pytest.mark.parametrize("plan", [
    ScopePlan(SCOPE, (SCOPE, "global")),
    ScopePlan("global", (SCOPE, "global", "*")),
    ScopePlan("global", ("*",)),
])
def test_overlapping_passes_keep_candidates_order_limits_and_load_new_bodies_once(
    tmp_path, monkeypatch, plan,
):
    with MemoryStore(tmp_path / "discovery.sqlite3") as store:
        for scope in ("project:foreign", "global", SCOPE):
            for index in range(28):
                put(store, f"{scope}-{index:02}", scope=scope,
                    stats={"confidence": 0.9 - (index // 2) * 0.01,
                           "usefulness_score": float(index % 2)})
        existing = _Candidate(
            node=store.get_node(f"{SCOPE}-01"), bm25_score=0.01, vector_score=0.8,
            chunk_id="selected-chunk", graph_score=0.4, trigger_score=0.6,
            trigger_coverage=0.5, path=("root",), anchor_score=0.3,
            source_graph_score=0.2, lexical_score=0.7,
        )
        stronger = _Candidate(node=store.get_node("global-01"), bm25_score=2.0)
        initial = {item.node.id: item for item in (existing, stronger)}
        service = MemoryRecallService(store, embedder=Encoder())
        expected = {key: replace(value) for key, value in initial.items()}
        collect_full_bodies(service, QUERY, plan, expected, max_results=2)
        actual = {key: replace(value) for key, value in initial.items()}
        decoded = Counter()
        passes = []
        decode = storage._node_from_row
        search = store.search_content_ids

        def count(row):
            decoded[row["id"]] += 1
            return decode(row)

        def discover(query, **options):
            passes.append((options["scope"], options["limit"]))
            return search(query, **options)

        monkeypatch.setattr(storage, "_node_from_row", count)
        monkeypatch.setattr(store, "search_content_ids", discover)
        service._collect_bm25(QUERY, plan, actual, max_results=2)
        assert actual == expected
        assert list(actual) == list(expected)
        assert actual[existing.node.id].node is existing.node
        assert decoded == Counter({key: 1 for key in expected.keys() - initial.keys()})
        scopes = plan.search_scopes if plan.restricted else (*plan.named_scopes, None)
        assert passes == [(scope, 25) for scope in scopes]
        assert not store.connection.in_transaction


@pytest.mark.parametrize("scope", [None, SCOPE])
def test_ordinary_recall_and_residual_match_full_body_discovery_with_vectors(
    tmp_path, monkeypatch, scope,
):
    with MemoryStore(tmp_path / "recall.sqlite3", chunk_embedder=Encoder().embed_batch) as store:
        for item_scope in (SCOPE, "global", "project:foreign"):
            for index in range(3):
                put(store, f"{item_scope}-{index}", scope=item_scope)
        service = MemoryRecallService(store, embedder=Encoder(), anchor_seeding=False)
        options = dict(scope=scope, ambient_context={"project": "lexical"}, depth=0,
                       max_results=2, log_access=False, log_event=False, repair_pending=False)
        actual = service.memory_recall(QUERY, **options)
        residual = service.last_residual
        assert actual and residual
        assert all(item.vector_score > 0 for item in [*actual, *residual])
        monkeypatch.setattr(service, "_collect_bm25", partial(collect_full_bodies, service))
        assert service.memory_recall(QUERY, **options) == actual
        assert service.last_residual == residual


@pytest.mark.parametrize("mutation", ["decay", "remove", "move"])
def test_body_admission_rechecks_live_scope_and_scores_surviving_source(tmp_path, monkeypatch, mutation):
    with MemoryStore(tmp_path / "admission.sqlite3") as store:
        rejected = put(store, "rejected", stats={"confidence": 0.9})
        survivor = put(store, "survivor", stats={"confidence": 0.2})
        service = MemoryRecallService(store, embedder=Encoder())
        admit = store.get_recall_nodes
        decode = storage._node_from_row
        decoded = []

        def change_then_admit(ids, plan):
            ids = list(ids)
            assert ids == [rejected.id, survivor.id]
            if mutation == "remove":
                store.delete_node_chunks(rejected.id)
                store.connection.execute("DELETE FROM nodes WHERE id = ?", (rejected.id,))
            else:
                column, value = ("decayed", 1) if mutation == "decay" else ("scope", "project:foreign")
                store.connection.execute(
                    f"UPDATE nodes SET {column} = ?, provenance = '[]' WHERE id = ?",
                    (value, rejected.id),
                )
            return admit(ids, plan)

        def count(row):
            decoded.append(row["id"])
            return decode(row)

        monkeypatch.setattr(store, "get_recall_nodes", change_then_admit)
        monkeypatch.setattr(storage, "_node_from_row", count)
        candidates = {}
        service._collect_bm25(QUERY, resolve_scope(scope=SCOPE), candidates, max_results=2)
        assert candidates == {survivor.id: _Candidate(node=survivor, bm25_score=1.0,
                                                       lexical_score=1.0, lexical_score_prepared=True)}
        assert decoded == [survivor.id]


@pytest.mark.parametrize("mutation", ["content", "decay", "remove", "move"])
def test_discovery_and_bodies_share_snapshot_but_next_collection_reads_current_source(
    tmp_path, monkeypatch, mutation,
):
    path = tmp_path / "snapshot.sqlite3"
    with MemoryStore(path) as store, MemoryStore(path) as writer:
        original = put(store, "source")
        service = MemoryRecallService(store, embedder=Encoder())
        plan = resolve_scope(scope=SCOPE)
        admit = store.get_recall_nodes

        def concurrent_write(ids, plan):
            assert store.connection.in_transaction
            with writer.transaction():
                if mutation == "remove":
                    writer.delete_node_chunks(original.id)
                    writer.connection.execute("DELETE FROM nodes WHERE id = ?", (original.id,))
                else:
                    column, value = {
                        "content": ("content", QUERY + " revised"),
                        "decay": ("decayed", 1),
                        "move": ("scope", "project:foreign"),
                    }[mutation]
                    writer.connection.execute(f"UPDATE nodes SET {column} = ? WHERE id = ?",
                                              (value, original.id))
            return admit(ids, plan)

        with monkeypatch.context() as patch:
            patch.setattr(store, "get_recall_nodes", concurrent_write)
            candidates = {}
            service._collect_bm25(QUERY, plan, candidates, max_results=2)
        assert candidates == {original.id: _Candidate(node=original, bm25_score=1.0,
                                                       lexical_score=1.0, lexical_score_prepared=True)}
        assert not store.connection.in_transaction
        fresh = {}
        service._collect_bm25(QUERY, plan, fresh, max_results=2)
        if mutation == "content":
            assert fresh[original.id].node.content == QUERY + " revised"
        else:
            assert fresh == {}


def test_lexical_snapshot_accepts_read_only_connection(tmp_path):
    path = tmp_path / "readonly.sqlite3"
    with MemoryStore(path) as store:
        original = put(store, "source")
        store.close()
        store._conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        store.connection.row_factory = sqlite3.Row
        candidates = {}
        MemoryRecallService(store, embedder=Encoder())._collect_bm25(
            QUERY, resolve_scope(scope=SCOPE), candidates, max_results=2,
        )
        assert candidates == {original.id: _Candidate(node=original, bm25_score=1.0,
                                                       lexical_score=1.0, lexical_score_prepared=True)}
        assert not store.connection.in_transaction


def test_lexical_snapshot_preserves_outer_writes_rollback_and_failure(tmp_path):
    with MemoryStore(tmp_path / "nested.sqlite3") as store:
        original = put(store, "source")
        service = MemoryRecallService(store, embedder=Encoder())
        plan = resolve_scope(scope=SCOPE)
        store.connection.execute("BEGIN")
        store.connection.execute("UPDATE nodes SET content = ? WHERE id = ?",
                                 (QUERY + " uncommitted", original.id))
        candidates = {}
        service._collect_bm25(QUERY, plan, candidates, max_results=2)
        assert store.connection.in_transaction
        assert candidates[original.id].node.content == QUERY + " uncommitted"
        store.connection.execute("UPDATE nodes SET provenance = '[]' WHERE id = ?", (original.id,))
        with pytest.raises(TypeError):
            service._collect_bm25(QUERY, plan, {}, max_results=2)
        assert store.connection.in_transaction
        assert store.connection.execute("SELECT content FROM nodes").fetchone()[0] == QUERY + " uncommitted"
        store.connection.rollback()
        candidates = {}
        service._collect_bm25(QUERY, plan, candidates, max_results=2)
        assert candidates == {original.id: _Candidate(node=original, bm25_score=1.0,
                                                       lexical_score=1.0, lexical_score_prepared=True)}
        assert not store.connection.in_transaction
