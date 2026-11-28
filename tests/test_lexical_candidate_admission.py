"""Lexical discovery preserves membership while admitting each new body once."""

from collections import Counter
from dataclasses import replace
from functools import partial
import sqlite3

import pytest

from fixture_vectors import fixture_vector
from living_memory import storage
from living_memory.retrieval import MemoryRecallService, _Candidate, _expanded_query
from living_memory.scope import ScopePlan, resolve_scope
from living_memory.storage import MemoryStore


QUERY = "amber recall boundary"
SCOPE = "project:lexical"


class Encoder:
    def embed(self, text):
        return fixture_vector([1.0, 0.0])

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]


def put(store, node_id, *, scope=SCOPE, **kwargs):
    return store.create_node(
        node_id=node_id, level="concept", content=f"{QUERY} {node_id}",
        context={"scope": scope, "detail": "x" * 1024},
        provenance={"detail": "y" * 1024}, embedding=fixture_vector([1.0, 0.0]),
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
            passes.append((options.get("scope"), options["limit"]))
            return search(query, **options)

        monkeypatch.setattr(storage, "_node_from_row", count)
        monkeypatch.setattr(store, "search_content_ids", discover)
        service._collect_bm25(QUERY, plan, actual, max_results=2)
        assert actual == expected
        assert list(actual) == list(expected)
        assert actual[existing.node.id].node is existing.node
        assert decoded == Counter({key: 1 for key in expected.keys() - initial.keys()})
        if plan.restricted or not plan.named_scopes:
            assert passes == [(scope, 25) for scope in plan.search_scopes]
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

        def change_then_admit(ids, plan, **kwargs):
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
            return admit(ids, plan, **kwargs)

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
@pytest.mark.parametrize("broad", [False, True])
def test_discovery_and_bodies_share_snapshot_but_next_collection_reads_current_source(
    tmp_path, monkeypatch, mutation, broad,
):
    path = tmp_path / "snapshot.sqlite3"
    with MemoryStore(path) as store, MemoryStore(path) as writer:
        original = put(store, "source")
        service = MemoryRecallService(store, embedder=Encoder())
        plan = resolve_scope(scope=None if broad else SCOPE, ambient_context={"project": "lexical"})
        admit = store.get_recall_nodes

        def concurrent_write(ids, plan, **kwargs):
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
            return admit(ids, plan, **kwargs)

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
        elif broad and mutation == "move":
            assert fresh[original.id].node.scope == "project:foreign"
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


@pytest.mark.parametrize("scope", [None, SCOPE])
def test_lexical_snapshot_preserves_outer_writes_rollback_and_failure(tmp_path, scope):
    with MemoryStore(tmp_path / "nested.sqlite3") as store:
        original = put(store, "source")
        service = MemoryRecallService(store, embedder=Encoder())
        plan = resolve_scope(scope=scope, ambient_context={"project": "lexical"})
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


def test_empty_named_scope_skips_fts_until_first_live_source(tmp_path):
    path = tmp_path / "first-live.sqlite3"
    with MemoryStore(path) as store, MemoryStore(path) as writer:
        scope = "session:first-live"
        dead = put(store, "dead", scope=scope)
        store.soft_delete_node(dead.id)
        put(store, "outside")
        service = MemoryRecallService(store, embedder=Encoder())
        plan = ScopePlan(scope, (scope,))

        def collect():
            statements = []
            candidates = {}
            store.connection.set_trace_callback(statements.append)
            try:
                service._collect_bm25(QUERY, plan, candidates, max_results=2)
            finally:
                store.connection.set_trace_callback(None)
            reads_postings = any("nodes_fts_idx" in sql for sql in statements)
            return list(candidates), reads_postings

        with store.transaction():
            assert collect() == ([], False)
            writer.create_node(node_id="first", level="trace", content=QUERY, context={"scope": scope})
            assert collect() == ([], False)
        assert collect() == (["first"], True)


@pytest.mark.parametrize("analyzed", [False, True])
def test_broad_discovery_matches_each_top_k_before_deduplication(tmp_path, analyzed):
    with MemoryStore(tmp_path / "broad.sqlite3") as store:
        for scope in (SCOPE, "global", "project:foreign"):
            for index in range(8):
                put(store, f"{scope}-{8-index}", scope=scope,
                    stats={"confidence": (8 - index // 2) / 10})
        if analyzed:
            store.connection.execute("ANALYZE")
        scopes = ("session:empty", SCOPE, "global", SCOPE)
        for plan in (ScopePlan(SCOPE, scopes), ScopePlan("global", (*scopes, "*"))):
            partitions = plan.search_scopes if plan.restricted else (*plan.named_scopes, None)
            for query in (QUERY, "nonmatching", ""):
                with store.read_snapshot():
                    expected = {
                        limit: [row for scope in partitions
                                for row in store.search_content_ids(query, scope=scope, limit=limit)]
                        for limit in (3, 25, 1, 4)
                    }
                    for limit, rows in expected.items():
                        assert list(store.iter_recall_content_ids(query, plan, limit=limit)) == rows
                    assert store.recall_content_ids_by_limit(query, plan, tuple(expected)) == expected


def test_broad_discovery_scans_distinct_winners_once(tmp_path):
    with MemoryStore(tmp_path / "distinct.sqlite3") as store:
        for index in range(30):
            put(store, f"item-{index:02}", scope=(SCOPE, "global", "project:foreign")[index % 3],
                stats={"confidence": index / 30})
        plan = ScopePlan("global", ("session:empty", SCOPE, "global", "*"))
        expected = [row for scope in (*plan.named_scopes, None)
                    for row in store.search_content_ids(QUERY, scope=scope, limit=3)]
        statements = []
        store.connection.set_trace_callback(statements.append)
        try:
            with store.transaction():
                assert store.recall_content_ids_by_limit(QUERY, plan, (1, 3))[3] == expected
        finally:
            store.connection.set_trace_callback(None)
        assert sum("nodes_fts MATCH" in sql for sql in statements) == 1


def test_broad_discovery_preserves_storage_order_and_projection_errors(tmp_path):
    with MemoryStore(tmp_path / "projection.sqlite3") as store:
        plan = ScopePlan("global", (SCOPE, "global", "*"))
        put(store, "low", stats={"confidence": 0.1})
        put(store, "top", stats={"confidence": 0.9})
        for assignment in ("confidence = 'invalid'", "usefulness_score = CAST(x'ff' AS TEXT)"):
            store.connection.execute(f"UPDATE nodes SET {assignment} WHERE id = 'low'")
            expected = [row for scope in (*plan.named_scopes, None)
                        for row in store.search_content_ids(QUERY, scope=scope, limit=1)]
            assert list(store.iter_recall_content_ids(QUERY, plan, limit=1)) == expected
            assert store.recall_content_ids_by_limit(QUERY, plan, (1, 2))[1] == expected
        store.connection.execute("DROP TABLE nodes_fts")
        with pytest.raises(sqlite3.OperationalError, match="no such table: nodes_fts"):
            list(store.iter_recall_content_ids(QUERY, plan, limit=1))
        with pytest.raises(sqlite3.OperationalError, match="no such table: nodes_fts"):
            store.recall_content_ids_by_limit(QUERY, plan, (1, 2))


def test_scoped_prefix_checks_score_ties_inside_and_across_limit(tmp_path, monkeypatch):
    plan = ScopePlan(SCOPE, (SCOPE,))
    with MemoryStore(tmp_path / "scoped.sqlite3") as store:
        # Increasing document lengths give distinct scores, except this pair.
        for i, words in enumerate((1, 2, 3, 3, 4, 5)):
            store.create_node(node_id=str(i), level="trace", content=f"amber item{i}" + " mineral" * words,
                              context={"scope": SCOPE})
        limits = (1, 2, 3, 4, 6)
        expected = {k: store.search_content_ids("amber", scope=SCOPE, limit=k) for k in limits}
        assert expected[6][2][1] == expected[6][3][1]
        calls = []
        search = store.search_content_ids

        def read(query, *, scope, limit):
            calls.append(limit)
            return search(query, scope=scope, limit=limit)

        monkeypatch.setattr(store, "search_content_ids", read)
        with store.read_snapshot():
            assert store.recall_content_ids_by_limit("amber", plan, limits) == expected
        assert calls == [6, 3, 4]
