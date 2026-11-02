"""Contracts for any future followup result-identity projection.

The oracle retains the original full-result read independently of the production
reader. Unused fields may still carry required JSON/text errors.
"""

from collections.abc import Mapping, Sequence
from functools import partial
import json
import sqlite3
from typing import Any

import pytest

from living_memory import storage
from living_memory.recall_map import RecallMapBuilder
from living_memory.storage import (
    MemoryStore, RecallWork, RECALL_LOOKUP_EVENT_TABLE, RECALL_HISTORY_EVENT_TABLE,
    _normalize_recall_history_instant,
)

SCOPE = "project:followup-projection"
CONTEXT = {"task": "repair", "transport_session_id": "original"}
SINCE = "2026-01-01T00:00:00Z"
BODY = "unused-retained-delivery-body-" + "x" * 16000


def _json_loads(value, default):
    return default if value is None or value == "" else json.loads(value)


def _recall_result_node_ids(results):
    return [str(item["node_id"]) for item in results
            if isinstance(item, Mapping) and item.get("node_id")]


def full_followups(
    self, work: RecallWork, *, since: str, medoid_ids: Sequence[str],
    limit: int = 200,
) -> list[dict[str, Any]] | None:
    """Independent pre-projection reader: full JSON, original selection/lookup SQL."""

    predicate, values = work.predicate()
    recalls = self._conn.execute(
        f"""SELECT id, created_at, query, results FROM recall_events
            WHERE {predicate} AND created_at >= ?
            ORDER BY created_at DESC, rowid DESC LIMIT ?""",
        (*values, since, limit + 1),
    ).fetchall()
    if len(recalls) > limit:
        return None
    evidence = [
        {"at": _normalize_recall_history_instant(row["created_at"]),
         "query": row["query"],
         "node_ids": _recall_result_node_ids(_json_loads(row["results"], [])),
         "event_id": row["id"]}
        for row in recalls
    ]
    if not medoid_ids:
        return evidence
    marks = ",".join("?" for _ in medoid_ids)
    consumer_predicate, consumer_values = work.predicate("consumer")
    tied_predicate, tied_values = work.predicate("tied")
    lookups = self._conn.execute(
        f"""SELECT lookup.node_id, lookup.occurred_at
            FROM {RECALL_LOOKUP_EVENT_TABLE} AS lookup
            JOIN recall_events AS consumer ON consumer.rowid = (
                SELECT e.rowid FROM recall_events AS e
                JOIN {RECALL_HISTORY_EVENT_TABLE} AS h
                  ON h.recall_event_id = e.id
                WHERE e.transport_session_id = lookup.transport_session_id
                  AND h.occurred_at <= lookup.occurred_at
                ORDER BY e.created_at DESC, e.rowid DESC LIMIT 1
            )
            WHERE lookup.node_id IN ({marks}) AND lookup.occurred_at > ?
              AND ({consumer_predicate})
              AND NOT EXISTS (
                  SELECT 1 FROM recall_events AS tied
                  WHERE tied.transport_session_id = consumer.transport_session_id
                    AND tied.created_at = consumer.created_at
                    AND NOT COALESCE(({tied_predicate}), 0)
              )
            ORDER BY lookup.occurred_at DESC LIMIT ?""",
        (*medoid_ids, _normalize_recall_history_instant(since),
         *consumer_values, *tied_values, limit + 1),
    ).fetchall()
    if len(lookups) > limit:
        return None
    evidence.extend(
        {"at": row["occurred_at"], "query": "", "node_ids": [row["node_id"]],
         "event_id": None}
        for row in lookups
    )
    return evidence


@pytest.fixture
def sample(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    clock = [SINCE]
    monkeypatch.setattr(storage, "_utc_now", lambda: clock[0])
    with MemoryStore(tmp_path / "followups.sqlite3") as store:
        yield store, clock, RecallWork.resolve(SCOPE, task="repair")


def event(store, *, query="idle", context=CONTEXT, scope=SCOPE, **kwargs):
    return store.record_recall_event(query=query, scope=scope, ambient_context=context, **kwargs)


def offers(store, clock):
    for minute in range(3):
        clock[0] = f"2026-01-01T00:0{minute}:00Z"
        event(store, recall_map={"clusters": [{"label": "zebra", "ask_hint": "zebra",
                                              "medoid": {"node_id": "medoid"}}]})
    clock[0] = "2026-01-01T00:03:00Z"


def outcome(read):
    try:
        return read()
    except (ValueError, TypeError, RecursionError, sqlite3.Error) as error:
        return type(error), str(error)


def assert_consumer_matches_full(store, work, monkeypatch):
    actual = outcome(lambda: RecallMapBuilder(store)._read_curtailment(work))
    with monkeypatch.context() as patch:
        patch.setattr(store, "recall_work_followups", partial(full_followups, store))
        expected = outcome(lambda: RecallMapBuilder(store)._read_curtailment(work))
    assert actual == expected
    return actual


RAW_RESULTS = [
    json.dumps([{"node_id": value} for value in [
        "same", "same", None, False, True, 0, -0.0, 1.5, 2**100, "", [], {},
        [1, False], {"b": 2, "a": True}, "雪", "a\x00b",
    ]] + [{"other": "missing"}, None, 42, "medoid", [["node_id", "medoid"]]]),
    '[{"node_id":"first","node_id":"medoid"}]',
    '[{"node_id":"first","\\u006eode_id":"medoid"}]',
    '[{"node_id":{"key":1,"key":2}}]',
    '[{"\\u006eode_id":"medoid"}]',
    '[{"node_id\\u0000suffix":"wrong","node_id":"medoid"}]',
    '[{"node_id":NaN},{"node_id":Infinity},{"node_id":-Infinity}]',
    '[{"node_id":1e999},{"node_id":-1e999},{"extra":NaN}]',
    '[]', '[[]]', '[{}]', '{}', '{"node_id":"medoid"}', '"medoid"', '""', '',
    '[null,3,"abc",[["node_id","medoid"]]]',
    '{', '[{"node_id":"medoid"},]', '/* comment */ []', '[{"extra":"\x00"}]',
    'null', '42', 'true', 'NaN', 'Infinity',
    '[{"node_id":"medoid","unused":' + '1' * 5000 + '}]',
    '[{"node_id":"medoid","unused":' + '[' * 1100 + '0' + ']' * 1100 + '}]',
    '[{"node_id":"medoid","unused":' + '[' * 990 + '0' + ']' * 990 + '}]',
    b'[{"node_id":"medoid"}]', b'[{"node_id":"medoid","unused":"\xff"}]',
    '[{"node_id":"medoid","delivery":{"content":"сохранённый текст"}}]',
    '[{"node_id":"medoid","extra":"\\ud800"}]',
    '[{"node_id":"\\ud800"}]',
    ' [\n {"node_id":"medoid"}\n ] ',
]


@pytest.mark.parametrize("raw", RAW_RESULTS, ids=lambda value: repr(value)[:65])
def test_legacy_and_errors_match_full_reader_and_consumer(sample, monkeypatch, raw):
    store, clock, work = sample
    offers(store, clock)
    selected = event(store)
    with store.transaction():
        store.connection.execute("UPDATE recall_events SET results=? WHERE id=?", (raw, selected.id))
    options = {"since": SINCE, "medoid_ids": ["medoid"]}
    assert outcome(lambda: store.recall_work_followups(work, **options)) == outcome(
        lambda: full_followups(store, work, **options))
    assert_consumer_matches_full(store, work, monkeypatch)
    assert not store.connection.in_transaction


def test_overflow_precedes_all_result_decoding(sample, monkeypatch):
    store, clock, work = sample
    offers(store, clock)
    for _ in range(198):
        event(store)
    with store.transaction():
        store.connection.execute("UPDATE recall_events SET results=?", ("{" + BODY,))

    original_decode = storage._json_loads

    def decode(value, default):
        assert BODY not in str(value)
        return original_decode(value, default)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_json_loads", decode)
        assert store.recall_work_followups(work, since=SINCE, medoid_ids=[]) is None
        verdict = RecallMapBuilder(store)._read_curtailment(work)
        assert verdict.streak == verdict.offers == 0


def test_ordering_duplicate_ids_queries_and_timestamps(sample, monkeypatch):
    store, clock, work = sample
    offers(store, clock)
    rows = []
    for stamp in ("2026-01-01T01:01:00+01:00", "2026-01-01T00:04:00Z",
                  "2026-01-01T00:04:00Z", "invalid"):
        clock[0] = stamp
        rows.append(event(store, query="zebra" if len(rows) == 1 else "idle", results=[
            {"node_id": "b"}, {"node_id": "a"}, {"node_id": "b"}]))
    options = {"since": SINCE, "medoid_ids": []}
    actual = store.recall_work_followups(work, **options)
    assert actual == full_followups(store, work, **options)
    assert [row["event_id"] for row in actual[:4]] == [rows[i].id for i in (3, 0, 2, 1)]
    assert actual[0]["at"] is None
    assert actual[1]["at"] == "2026-01-01T00:01:00.000000Z"
    assert actual[1]["node_ids"] == ["b", "a", "b"]
    assert_consumer_matches_full(store, work, monkeypatch)


@pytest.mark.parametrize("tie", ["same", "other-work", "other-scope", "anonymous", "later"])
def test_work_identity_and_lookup_ties_match_complete_consumer(sample, monkeypatch, tie):
    store, clock, work = sample
    offers(store, clock)
    event(store, context={"task_pattern": "repair", "task": "turn", "transport_session_id": "new"})
    context = {"transport_session_id": "new", "task": "repair"}
    if tie in ("other-work", "later"):
        context["task"] = "foreign"
    if tie == "anonymous":
        context.pop("task")
    if tie == "later":
        clock[0] = "2026-01-01T00:04:00Z"
    event(store, context=context, scope="project:foreign" if tie == "other-scope" else SCOPE)
    store.record_lookup_event(["medoid"], transport_session_id="new", occurred_at="2026-01-01T00:05:00Z")
    options = {"since": SINCE, "medoid_ids": ["medoid"]}
    actual = store.recall_work_followups(work, **options)
    assert actual == full_followups(store, work, **options)
    assert sum(row["event_id"] is None for row in actual) == (tie == "same")
    verdict = assert_consumer_matches_full(store, work, monkeypatch)
    assert verdict.collapse == (tie != "same")


def test_same_read_snapshot_and_next_call_observe_other_connection(sample):
    store, clock, work = sample
    older = event(store, query="old", results=[{"node_id": "before"}])
    newest = event(store, query="new", results=[{"node_id": "before"}])
    options = {"since": SINCE, "medoid_ids": []}
    expected = full_followups(store, work, **options)
    changed = False
    with sqlite3.connect(store.db_path) as writer:
        def row_factory(cursor, row):
            nonlocal changed
            if not changed:
                changed = True
                writer.execute("UPDATE recall_events SET query='changed', results=? WHERE id=?",
                               ('[{"node_id":"after"}]', older.id))
                writer.execute("UPDATE recall_events SET task='foreign' WHERE id=?", (newest.id,))
                writer.commit()
            return sqlite3.Row(cursor, row)

        store.connection.row_factory = row_factory
        try:
            assert store.recall_work_followups(work, **options) == expected
        finally:
            store.connection.row_factory = sqlite3.Row
    assert changed and not store.connection.in_transaction
    current = store.recall_work_followups(work, **options)
    assert current == full_followups(store, work, **options)
    assert len(current) == 1 and current[0]["node_ids"] == ["after"]
    assert current[0]["query"] == "changed"


@pytest.mark.parametrize("encoded", [b"\xff", b"\xc0\xa0", b"\xed\xa0\x80"])
def test_invalid_utf8_text_preserves_read_errors(sample, encoded):
    store, clock, work = sample
    selected = event(store)
    raw = b'[{"node_id":"ok","unused":"' + encoded + b'"}]'
    with store.transaction():
        store.connection.execute("UPDATE recall_events SET results=CAST(? AS TEXT) WHERE id=?",
                                 (raw, selected.id))
    options = {"since": SINCE, "medoid_ids": []}
    expected = outcome(lambda: full_followups(store, work, **options))
    assert expected[0] is sqlite3.OperationalError
    assert outcome(lambda: store.recall_work_followups(work, **options)) == expected


def test_lookup_overflow_and_missing_storage_keep_consumer_unavailable(sample, monkeypatch):
    store, clock, work = sample
    offers(store, clock)
    for _ in range(201):
        store.record_lookup_event(["medoid"], transport_session_id="original")
    options = {"since": SINCE, "medoid_ids": ["medoid"]}
    assert store.recall_work_followups(work, **options) is None
    assert full_followups(store, work, **options) is None
    assert assert_consumer_matches_full(store, work, monkeypatch).streak == 0
    store.connection.execute("DROP TABLE recall_lookup_events")
    with pytest.raises(sqlite3.OperationalError):
        store.recall_work_followups(work, **options)
    assert assert_consumer_matches_full(store, work, monkeypatch).streak == 0


def test_only_bounded_matching_events_are_decoded(sample):
    store, clock, work = sample
    event(store, results=[{"node_id": "old"}])
    clock[0] = "2026-01-01T00:01:00Z"
    event(store, results=[{"node_id": "current"}])
    for _ in range(5):
        event(store, context={"task": "foreign"})
    store.connection.execute("UPDATE recall_events SET results='{' WHERE task='foreign'")
    rows = store.recall_work_followups(work, since=clock[0], medoid_ids=[], limit=1)
    assert len(rows) == 1
    assert rows[0]["node_ids"] == ["current"]
    assert store.recall_work_followups(work, since=SINCE, medoid_ids=[], limit=1) is None


@pytest.mark.parametrize("limit", [0, -1, -2, 1, 2, 200])
def test_limits_preserve_existing_behavior(sample, limit):
    store, clock, work = sample
    for _ in range(2):
        event(store)
    options = {"since": SINCE, "medoid_ids": [], "limit": limit}
    assert store.recall_work_followups(work, **options) == full_followups(store, work, **options)
