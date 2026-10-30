"""Trigger contributions reuse selected bodies and retain current identity reads."""

from collections import Counter
from threading import RLock
import sqlite3

import pytest

from living_memory import storage
from living_memory.feedback import prepared_feedback
from living_memory.retrieval import MemoryRecallService, SCHEMA_TRIGGER_BASE_SCORE, _Candidate
from living_memory.scope import ScopePlan
from living_memory.storage import MemoryStore


SCOPE = "project:harbor"
QUERY = "harbor inspection"


class Encoder:
    def embed(self, text):
        return []


@pytest.fixture(autouse=True)
def ordinary_recall(monkeypatch):
    monkeypatch.delenv("LM_RECALL_SCHEMA_TRIGGER", raising=False)
    monkeypatch.delenv("LM_RECALL_SCHEMA_DEDUP", raising=False)


def node(store, name, *, level="schema", scope=SCOPE, trigger=QUERY, content=None):
    return store.create_node(
        node_id=name, level=level, content=content if content is not None else f"{QUERY} {name}",
        context={"scope": scope, "trigger": trigger},
    )


def service(store):
    return MemoryRecallService(store, embedder=Encoder())


def recall(retriever, **kwargs):
    return retriever.memory_recall(
        QUERY, scope=SCOPE, depth=0, max_results=10,
        log_access=False, log_event=False, repair_pending=False, **kwargs,
    )


@pytest.mark.parametrize("existing", [(), ("concept", "schema"), ("concept",)])
@pytest.mark.parametrize("scopes", [(SCOPE, "global", SCOPE), (SCOPE, "global", "*")])
def test_collector_hydrates_only_missing_bodies_and_keeps_channel_evidence(
    tmp_path, monkeypatch, existing, scopes,
):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        nodes = {level: node(store, level, level=level) for level in ("concept", "schema")}
        node(store, "unmatched", trigger="cedar quartz")
        node(store, "decayed")
        store.soft_delete_node("decayed")
        node(store, "trace", level="trace")
        candidates = {
            name: _Candidate(node=nodes[name], bm25_score=0.4, vector_score=0.7)
            for name in existing
        }
        selected = dict(candidates)
        hydrated = []
        hydrate = storage._node_from_row

        def read(row):
            hydrated.append(row["id"])
            return hydrate(row)

        monkeypatch.setattr(storage, "_node_from_row", read)
        service(store)._collect_schema_triggers(
            QUERY + " weather", ScopePlan(SCOPE, scopes), candidates,
        )
        assert Counter(hydrated) == Counter(set(nodes) - set(existing))
        assert set(candidates) == set(nodes)
        for name, candidate in candidates.items():
            assert candidate.trigger_score == pytest.approx(SCHEMA_TRIGGER_BASE_SCORE + 0.05)
            assert candidate.trigger_coverage == pytest.approx(2 / 3)
            if name in selected:
                assert candidate is selected[name]
                assert candidate.node is nodes[name]
                assert (candidate.bm25_score, candidate.vector_score) == (0.4, 0.7)


@pytest.mark.parametrize("query", ["", "cedar quartz"])
def test_no_matching_triggers_read_no_bodies(tmp_path, monkeypatch, query):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node(store, "schema")

        def unexpected(row):
            pytest.fail("unmatched trigger hydrated a body")

        monkeypatch.setattr(storage, "_node_from_row", unexpected)
        candidates = {}
        service(store)._collect_schema_triggers(query, ScopePlan(SCOPE, (SCOPE,)), candidates)
        assert candidates == {}


def test_trigger_only_admission_keeps_reader_order_across_sql_batches(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        with store.transaction():
            for index in range(1005):
                node(store, f"n{index:04}")
        discovered = dict(store.schema_triggers(SCOPE))
        assert len(discovered) == 1005
        ordered = store.get_nodes(discovered)
        existing = dict(list(ordered.items())[::2])
        candidates = {name: _Candidate(node=body) for name, body in existing.items()}
        service(store)._collect_schema_triggers(QUERY, ScopePlan(SCOPE, (SCOPE,)), candidates)
        assert list(candidates) == [*existing, *(name for name in ordered if name not in existing)]


@pytest.mark.parametrize("existing", [False, True])
def test_external_deletion_after_discovery_suppresses_trigger_contribution(tmp_path, monkeypatch, existing):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store, MemoryStore(path) as writer:
        selected = node(store, "removed")
        candidates = {selected.id: _Candidate(node=selected, bm25_score=0.8)} if existing else {}
        discover = store.schema_triggers
        guard = RLock()

        def discover_then_delete(scope):
            assert guard._is_owned()
            assert not store.connection.in_transaction
            found = discover(scope)
            with writer.connection:
                writer.connection.execute("DELETE FROM nodes WHERE id = ?", (selected.id,))
            return found

        monkeypatch.setattr(store, "schema_triggers", discover_then_delete)
        with prepared_feedback(store, guard=lambda: guard):
            service(store)._collect_schema_triggers(QUERY, ScopePlan(SCOPE, (SCOPE,)), candidates)
        if existing:
            assert candidates[selected.id].node is selected
            assert candidates[selected.id].trigger_score == 0.0
            assert candidates[selected.id].trigger_coverage == 0.0
            assert candidates[selected.id].bm25_score == 0.8
        else:
            assert candidates == {}


def test_trigger_only_deletion_after_identity_read_does_not_admit_a_body(tmp_path, monkeypatch):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store, MemoryStore(path) as writer:
        node(store, "removed")
        identities = store.get_node_scopes

        def read_then_delete(ids):
            found = identities(ids)
            with writer.connection:
                writer.connection.execute("DELETE FROM nodes WHERE id = 'removed'")
            return found

        monkeypatch.setattr(store, "get_node_scopes", read_then_delete)
        candidates = {}
        service(store)._collect_schema_triggers(QUERY, ScopePlan(SCOPE, (SCOPE,)), candidates)
        assert candidates == {}


@pytest.mark.parametrize("mutation", ["content", "scope", "decay"])
def test_trigger_identity_read_keeps_the_already_selected_body(tmp_path, monkeypatch, mutation):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store, MemoryStore(path) as writer:
        selected = node(store, "selected")
        candidates = {selected.id: _Candidate(node=selected)}
        discover = store.schema_triggers

        def discover_then_update(scope):
            found = discover(scope)
            if mutation == "content":
                writer.update_node(selected.id, content="Updated instruction")
            elif mutation == "scope":
                writer.update_node(selected.id, context={"scope": "project:elsewhere"})
            else:
                writer.soft_delete_node(selected.id)
            return found

        monkeypatch.setattr(store, "schema_triggers", discover_then_update)
        service(store)._collect_schema_triggers(QUERY, ScopePlan(SCOPE, (SCOPE,)), candidates)
        assert candidates[selected.id].node is selected
        assert candidates[selected.id].trigger_score > 0


def test_actual_recall_hydrates_each_body_once_and_reads_changes_next_call(tmp_path, monkeypatch):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store, MemoryStore(path) as writer:
        for index in range(4):
            node(store, f"selected-{index}", level="concept" if index % 2 else "schema")
        node(store, "trigger-only", content="Zebra quartz xylophone")
        retriever = service(store)
        hydrated = Counter()
        hydrate = storage._node_from_row

        def read(row):
            hydrated[row["id"]] += 1
            return hydrate(row)

        monkeypatch.setattr(storage, "_node_from_row", read)
        before = list(store.connection.iterdump())
        results = recall(retriever)
        assert hydrated == Counter({**{f"selected-{i}": 1 for i in range(4)}, "trigger-only": 1})
        assert {result.node.id for result in results} == set(hydrated)
        assert all(result.trigger_score > 0 for result in results)
        assert list(store.connection.iterdump()) == before
        writer.update_node("selected-0", content=QUERY + " revised instruction", context={"trigger": "cedar quartz"})
        current = {result.node.id: result for result in recall(retriever)}
        assert current["selected-0"].node.content == QUERY + " revised instruction"
        assert current["selected-0"].trigger_score == 0.0


@pytest.mark.parametrize("column,existing", [("scope", True), ("content", False)])
def test_required_trigger_read_errors_abort_before_consumption(tmp_path, monkeypatch, column, existing):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node(store, "target", content=QUERY if existing else "Zebra quartz xylophone")
        discover = store.schema_triggers

        def discover_then_deny(scope):
            found = discover(scope)
            if scope == "global":
                store.connection.set_authorizer(
                    lambda action, table, field, *_: sqlite3.SQLITE_DENY
                    if action == sqlite3.SQLITE_READ and table == "nodes" and field == column
                    else sqlite3.SQLITE_OK
                )
            return found

        monkeypatch.setattr(store, "schema_triggers", discover_then_deny)
        before = list(store.connection.iterdump())
        retriever = service(store)
        try:
            with pytest.raises(sqlite3.DatabaseError):
                retriever.memory_recall(QUERY, scope=SCOPE, depth=0, repair_pending=False)
        finally:
            store.connection.set_authorizer(None)
        assert retriever.last_recall_event_id is None
        assert not store.connection.in_transaction
        assert list(store.connection.iterdump()) == before
