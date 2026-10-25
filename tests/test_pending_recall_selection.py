from __future__ import annotations

from pathlib import Path

import pytest

import living_memory.storage as storage_module
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore
from test_lookup import FakeMCP


SCOPE = "project:pending"
TEXT = "cobalt routing uses stable endpoint ownership"


@pytest.mark.parametrize("context", [
    {"session_id": "own"},
    {"task": "own"},
    {"transport_session_id": "own"},
    {"session_id": "own", "task": "own", "transport_session_id": "own"},
])
def test_foreign_events_do_not_displace_current_work(context: dict[str, str]) -> None:
    with MemoryStore(":memory:") as store:
        own = store.record_recall_event(query=TEXT, scope=SCOPE, ambient_context=context)
        for i in range(120):
            store.record_recall_event(
                query=TEXT, scope=SCOPE,
                ambient_context={key: f"foreign-{i}" for key in context},
            )
        assert [event.id for event in store.pending_recall_events(
            scope=SCOPE, context=context, limit=5,
        )] == [own.id]


@pytest.mark.parametrize("event_context,context,content,scope,expected", [
    ({"session_id": "s"}, {"session": "s"}, "unrelated", SCOPE, 2),
    ({"task": "t"}, {"task": "t"}, "unrelated", SCOPE, 2),
    ({"session_id": "s", "task": "a"}, {"session_id": "s", "task": "b"}, TEXT, SCOPE, 0),
    ({"session_id": "a", "task": "t"}, {"session_id": "b", "task": "t"}, TEXT, SCOPE, 0),
    ({"session_id": "s", "agent": "a", "transport_session_id": "a"},
     {"session_id": "s", "agent": "b", "transport_session_id": "b"}, "unrelated", SCOPE, 2),
    ({"task": "t", "transport_session_id": "a"},
     {"task": "t", "transport_session_id": "b"}, "unrelated", SCOPE, 2),
    ({"session_id": "a", "transport_session_id": "t"},
     {"session_id": "b", "transport_session_id": "t"}, TEXT, SCOPE, 0),
    ({"task": "a", "transport_session_id": "t"},
     {"task": "b", "transport_session_id": "t"}, TEXT, SCOPE, 0),
    ({"agent": "a", "transport_session_id": "t"},
     {"agent": "b", "transport_session_id": "t"}, TEXT, SCOPE, 0),
    ({"transport_session_id": "t"}, {"transport_session_id": "t"}, "unrelated", SCOPE, 2),
    ({"transport_session_id": "a"}, {"transport_session_id": "b"}, TEXT, SCOPE, 0),
    ({"transport_session_id": "t"}, {}, TEXT, SCOPE, 0),
    ({}, {"transport_session_id": "t"}, TEXT, SCOPE, 0),
    ({}, {}, TEXT, SCOPE, 0),
    ({}, {}, None, SCOPE, 0),
    ({}, {}, "unrelated", SCOPE, 0),
    ({"agent": "a"}, {"agent": "a"}, TEXT, SCOPE, 0),
    ({"agent": "a"}, {"agent": "a"}, "unrelated", SCOPE, 0),
    ({"agent": "a"}, {"agent": "b"}, TEXT, SCOPE, 0),
    ({"session_id": "s"}, {}, TEXT, SCOPE, 0),
    ({}, {"session_id": "s"}, TEXT, SCOPE, 0),
    ({"transport_session_id": "t"}, {"transport_session_id": "t"}, "unrelated", "global", 2),
    ({"task": "a", "transport_session_id": "t"},
     {"task": "b", "transport_session_id": "t"}, TEXT, "global", 0),
    ({"session_id": "s"}, {"session_id": "s"}, TEXT, "global", 0),
    ({}, {}, TEXT, "global", 0),
])
def test_shared_identity_and_conflict_precedence(
    event_context, context, content, scope, expected,
) -> None:
    with MemoryStore(":memory:") as store:
        events = [store.record_recall_event(
            query=TEXT, scope=scope, ambient_context=event_context,
            resolved_scopes=[SCOPE],
        ) for _ in range(2)]
        selected = store.pending_recall_events(
            scope=SCOPE, context=context, limit=5,
        )
        assert [event.id for event in selected] == [event.id for event in reversed(events)][:expected]


def test_newest_first_limit_includes_only_shared_identity() -> None:
    with MemoryStore(":memory:") as store:
        contexts = [{"task": "own"}, {}, {}, {"task": "own"}, {}, {"task": "foreign"}]
        events = [store.record_recall_event(
            query=TEXT, scope=SCOPE, ambient_context=context,
        ) for context in contexts]
        # Equal timestamps still use descending insertion order.
        store.connection.execute("UPDATE recall_events SET created_at = '2026-01-01T00:00:00Z'")
        for limit, positions in [(1, [3]), (2, [3, 0]), (5, [3, 0])]:
            assert [event.id for event in store.pending_recall_events(
                scope=SCOPE, context={"task": "own"}, limit=limit,
            )] == [events[i].id for i in positions]


def test_anonymous_population_does_not_hide_older_compatible_events() -> None:
    with MemoryStore(":memory:") as store:
        strong = [store.record_recall_event(
            query=TEXT, scope=SCOPE, ambient_context={"task": "own"},
        ) for _ in range(8)]
        for _ in range(120):
            store.record_recall_event(query=TEXT, scope=SCOPE)
        selected = store.pending_recall_events(
            scope=SCOPE, context={"task": "own"}, limit=5,
        )
        assert [event.id for event in selected] == [event.id for event in reversed(strong[-5:])]


def test_remember_consumes_own_event_once_amid_foreign_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    context = {"scope": SCOPE, "task": "own", "session_id": "own", "transport_session_id": "own"}
    try:
        node = store.append_trace(TEXT, {"scope": SCOPE})
        recalled = mcp.tools["memory_recall"](
            TEXT, scope=SCOPE, ambient_context=context, max_results=1,
        )
        event_id = recalled["recall_event_id"]
        foreign = [store.record_recall_event(
            query=TEXT, scope=SCOPE, ambient_context={**context, "task": f"foreign-{i}"},
            results=[{"node_id": node.id, "bm25_score": 1.0}],
        ) for i in range(120)]
        result = mcp.tools["memory_remember"](TEXT, context, used=[node.id])
        assert result["implicit_feedback"]["recall_event_ids"] == [event_id]
        trace = store.get_node(result["node"]["id"])
        assert [item["id"] for item in trace.provenance["prior_recalls"]] == [event_id]
        assert store.get_recall_event(event_id).feedback_trace_id == trace.id
        assert store.credited_node_ids(event_id) == {node.id}
        assert all(not store.get_recall_event(event.id).feedback_applied for event in foreign)
        score = store.get_node(node.id).usefulness_score
        repeated = mcp.tools["memory_remember"](TEXT + " verified", context)
        assert repeated["implicit_feedback"]["recall_event_ids"] == []
        assert store.get_node(node.id).usefulness_score == score
        assert store.connection.execute("SELECT COUNT(*) FROM recall_credit_ledger").fetchone()[0] == 1
    finally:
        store.close()


@pytest.mark.parametrize("context,foreign_context,foreign_scope", [
    ({"session_id": "own"}, {"session_id": "other"}, SCOPE),
    ({"task": "own"}, {"task": "other"}, SCOPE),
    ({"transport_session_id": "own"}, {"transport_session_id": "other"}, SCOPE),
    ({"session_id": "own"}, {}, SCOPE),
    ({"task": "own"}, {}, SCOPE),
    ({"transport_session_id": "own"}, {}, SCOPE),
    ({"task": "own", "session_id": "own"}, {"task": "own", "session_id": "other"}, SCOPE),
    ({"task": "own", "session_id": "own"}, {"task": "other", "session_id": "own"}, SCOPE),
    ({"task": "own"}, {"task": "own"}, "project:other"),
])
def test_pending_selection_reads_current_identity_without_materializing_foreign_rows(
    monkeypatch: pytest.MonkeyPatch, context, foreign_context, foreign_scope,
) -> None:
    with MemoryStore(":memory:") as store:
        own = store.record_recall_event(query=TEXT, scope=SCOPE, ambient_context=context)
        decoded = []
        original = storage_module._recall_event_from_row

        def decode(row):
            decoded.append(row["id"])
            return original(row)

        monkeypatch.setattr(storage_module, "_recall_event_from_row", decode)

        def select_steps():
            decoded.clear()
            steps = 0

            def progress():
                nonlocal steps
                steps += 1
                return 0

            store.connection.set_progress_handler(progress, 1)
            try:
                result = store.pending_recall_events(scope=SCOPE, context=context, limit=5)
            finally:
                store.connection.set_progress_handler(None, 0)
            assert [event.id for event in result] == [own.id]
            assert decoded == [own.id]
            return steps

        small = select_steps()
        # Both current-scope strangers and closed history must stay out of the
        # identity read; compare VM work, not a timing-sensitive speed claim.
        store.connection.executemany(
            """INSERT INTO recall_events(id, query, scope, requested_scope,
                   session_id, task, transport_session_id, agent, feedback_applied, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(f"foreign-{i}", TEXT, foreign_scope, foreign_scope,
              *(foreign_context.get(field) for field in ("session_id", "task", "transport_session_id", "agent")), i % 2,
              "2099-01-01T00:00:00Z") for i in range(4000)],
        )
        large = select_steps()
        assert large <= small * 2 + 100
