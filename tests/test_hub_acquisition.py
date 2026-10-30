"""Grouped evidence keeps its values and ends before the anchor observation."""

from contextlib import closing
import sqlite3
from types import SimpleNamespace

import pytest

from living_memory import hubs


class GroupedCursor(sqlite3.Cursor):
    closed = False
    exhausted = False
    read = 0

    def __next__(self):
        if self.connection.failure == "iteration" and self.read == 1:
            raise sqlite3.OperationalError("grouped read failed")
        try:
            row = super().__next__()
        except StopIteration:
            self.exhausted = True
            raise
        self.read += 1
        return row

    def close(self):
        super().close()
        self.closed = True


class ObservedConnection(sqlite3.Connection):
    failure = None
    grouped = None
    anchor_reads = 0
    before_anchors = None

    def execute(self, sql, parameters=()):
        if "GROUP BY m.node_id, m.recall_event_id" in sql:
            self.grouped = self.cursor(factory=GroupedCursor)
            return self.grouped.execute(sql, parameters)
        if "SELECT scope, fingerprint, id FROM query_anchors" in sql:
            assert self.grouped.closed and self.grouped.exhausted
            self.anchor_reads += 1
            if self.before_anchors is not None:
                self.before_anchors()
            if self.failure == "anchor":
                raise sqlite3.OperationalError("anchor read failed")
        return super().execute(sql, parameters)


def seed(conn, *, events=True, anchors=True, empty=False):
    conn.execute("CREATE TABLE recall_feedback_marks(node_id, recall_event_id, marked_at, accepted, mark)")
    if not empty:
        conn.executemany("INSERT INTO recall_feedback_marks VALUES (?, ?, ?, 1, 'irrelevant')", [
            (node, event, stamp)
            for node in ("n2", "n1")
            for event, stamp in (("e1", "2026-01-01"), ("e1", "2026-01-02"),
                                 ("e2", "2026-01-02"), ("missing", "2026-01-03"))
        ])
    if events:
        conn.execute("CREATE TABLE recall_events(id, query, scope, requested_scope)")
        conn.executemany("INSERT INTO recall_events VALUES (?, ?, ?, ?)", [
            ("e1", " Shared\t Вопрос ", " repo:one ", "project:requested"),
            ("e2", "Shared Вопрос", "project:one", None),
        ])
    if anchors:
        conn.execute("CREATE TABLE query_anchors(scope, fingerprint, id)")
        conn.execute("INSERT INTO query_anchors VALUES (?, ?, 'a1')", (
            "project:one", hubs.recall_fingerprint("Shared Вопрос", "project:one"),
        ))
    conn.commit()


@pytest.mark.parametrize("events,anchors", [(True, True), (False, True), (True, False)])
@pytest.mark.parametrize("before", [None, "2026-01-03"])
def test_grouped_values_order_cutoff_and_optional_tables(events, anchors, before):
    with closing(sqlite3.connect(":memory:", factory=ObservedConnection)) as conn:
        seed(conn, events=events, anchors=anchors)
        count = (2 if events else 3) - (before is not None)
        assert hubs.hub_counts(conn, min_queries=2, before=before) == [
            hubs.HubCount(node, count, count, 4 if before is None else 3,
                          "2026-01-01", 0, 0, count >= 2)
            for node in ("n1", "n2")
        ]
        assert conn.grouped.closed and conn.grouped.exhausted
        assert conn.anchor_reads == int(events and anchors)
        assert not conn.in_transaction


@pytest.mark.parametrize("missing_marks,events", [(True, True), (False, True), (False, False)])
def test_empty_skips_anchor_observation(missing_marks, events):
    with closing(sqlite3.connect(":memory:", factory=ObservedConnection)) as conn:
        seed(conn, events=events, empty=True)
        if missing_marks:
            conn.execute("DROP TABLE recall_feedback_marks")
        conn.failure = "anchor"
        assert hubs.hub_counts(conn) == []
        assert conn.anchor_reads == 0
        assert conn.grouped is None or conn.grouped.closed


@pytest.mark.parametrize("events,failure", [(True, "iteration"), (False, "iteration"), (True, "anchor")])
def test_cursor_closes_on_read_error_and_cache_fails_open(monkeypatch, events, failure):
    with closing(sqlite3.connect(":memory:", factory=ObservedConnection)) as conn:
        seed(conn, events=events)
        conn.failure = failure
        with pytest.raises(sqlite3.OperationalError, match="read failed"):
            hubs.hub_counts(conn)
        assert conn.grouped.closed
        monkeypatch.setenv("LM_HUB_SUPPRESSION_FACTOR", "0.2")
        store = SimpleNamespace(connection=conn)
        assert hubs.hub_demotions(store) == {}
        assert not hasattr(store, "_hub_suppression_cache")
        assert conn.grouped.closed
        conn.failure = None
        assert hubs.hub_counts(conn)


def test_raw_values_reach_fingerprint_after_anchors(monkeypatch):
    values = [" Equal\té ", " Equal\té ", "Equal é", "equal é", "Equal e\u0301", 17, 2.5, b"bytes", None]
    with closing(sqlite3.connect(":memory:", factory=ObservedConnection)) as conn:
        seed(conn, empty=True)
        conn.execute("DELETE FROM recall_events")
        for index, query in enumerate(values):
            conn.execute("INSERT INTO recall_events VALUES (?, ?, 'global', 'project:requested')", (index, query))
            conn.execute("INSERT INTO recall_feedback_marks VALUES ('n', ?, 'stamp', 1, 'irrelevant')", (index,))
        conn.commit()
        seen = []
        original = hubs.recall_fingerprint

        def fingerprint(query, scope):
            assert conn.anchor_reads == 1 and conn.grouped.closed
            seen.append(query)
            assert scope == "global"
            return original(query, scope)

        monkeypatch.setattr(hubs, "recall_fingerprint", fingerprint)
        assert hubs.hub_counts(conn) == [hubs.HubCount("n", 7, 7, 9, "stamp", 0, 0, True)]
        assert seen == values[:-1]
        assert [type(value) for value in seen] == [type(value) for value in values[:-1]]
        assert seen[0] is seen[1]


@pytest.mark.parametrize("custom", [False, True])
def test_string_subclasses_and_custom_values_are_not_hashed(monkeypatch, custom):
    class Text(str):
        __hash__ = None

    class Value:
        __hash__ = None

        def __str__(self):
            return "Shared Вопрос"

    value = Value() if custom else Text("Shared Вопрос")
    with closing(sqlite3.connect(":memory:", factory=ObservedConnection)) as conn:
        seed(conn)

        def row_factory(cursor, row):
            if len(row) == 6 and row[4] is not None:
                return (*row[:4], value, row[5])
            return row

        conn.row_factory = row_factory
        original = hubs.recall_fingerprint
        seen = []

        def fingerprint(query, scope):
            seen.append(query)
            return original(query, scope)

        monkeypatch.setattr(hubs, "recall_fingerprint", fingerprint)
        assert len(hubs.hub_counts(conn)) == 2
        assert seen and all(query is value for query in seen)


def test_anchor_error_precedes_scope_normalization():
    with closing(sqlite3.connect(":memory:", factory=ObservedConnection)) as conn:
        seed(conn)
        conn.execute("UPDATE recall_events SET scope = 'unsupported:scope'")
        conn.failure = "anchor"
        with pytest.raises(sqlite3.OperationalError, match="anchor read failed"):
            hubs.hub_counts(conn)
        conn.failure = None
        with pytest.raises(ValueError, match="unsupported scope"):
            hubs.hub_counts(conn)
        assert conn.grouped.closed


def test_wal_anchor_drop_between_statements_is_observed(tmp_path):
    path = tmp_path / "hubs.sqlite3"
    with closing(sqlite3.connect(path, factory=ObservedConnection)) as conn:
        seed(conn)
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        conn.execute("PRAGMA query_only=1")

        def drop_anchors():
            with closing(sqlite3.connect(path, timeout=1)) as writer:
                writer.execute("DROP TABLE query_anchors")
                writer.commit()

        conn.before_anchors = drop_anchors
        with pytest.raises(sqlite3.OperationalError, match="no such table: query_anchors"):
            hubs.hub_counts(conn)
        assert conn.grouped.closed and conn.grouped.exhausted
        assert conn.anchor_reads == 1 and not conn.in_transaction
