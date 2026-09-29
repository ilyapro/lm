"""Global hub suppression (goal recall-precision, P2).

A node marked ``irrelevant`` (accepted) by at least ``LM_HUB_MIN_QUERIES``
distinct query anchors and without positive evidence gets the
``LM_HUB_SUPPRESSION_FACTOR`` multiplier for every query:

* valve unset or invalid: ``{}`` and a byte-identical ranking;
* the threshold counts distinct questions, not marks: repeats of one query
  (whitespace variants included) count once, a shared near-duplicate anchor
  counts once;
* an accepted ``used`` mark ever, or grounded / lookup / explicit credit at or
  after the first irrelevant mark, lifts it (also through the cache);
* absent tables are tolerated;
* the hub multiplier merges with query-relative demotions, smallest wins.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

import pytest

import living_memory.hubs as hubs
from living_memory.hubs import (
    DEFAULT_HUB_MIN_QUERIES,
    hub_counts,
    hub_demotions,
    hub_ids,
    hub_min_queries,
    hub_suppression_factor,
)
from living_memory.retrieval import MemoryRecallService, _merge_demotions
from living_memory.storage import MemoryStore

SCOPE = "project:hubs"
BODIES = [
    "kappa ledger reconciliation drift found in the nightly batch",
    "kappa ledger export uses the vendor csv dialect",
    "kappa ledger totals are cached for five minutes",
    "kappa ledger archive moves to cold storage after a year",
]
QUERIES = ["kappa ledger drift", "kappa ledger export", "kappa ledger cache", "kappa ledger archive"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    for name in (
        "LM_HUB_SUPPRESSION_FACTOR",
        "LM_HUB_MIN_QUERIES",
        "LM_EXPLICIT_FEEDBACK_POLICY",
        "LM_QUERY_IRRELEVANCE_FACTOR",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed(store: MemoryStore) -> list[str]:
    return [store.append_trace(body, {"scope": SCOPE, "agent": "a"}).id for body in BODIES]


def _event(store: MemoryStore, query: str, scope: str = SCOPE) -> str:
    return store.record_recall_event(query=query, scope=scope, requested_scope=scope).id


def _mark(
    store: MemoryStore,
    event_id: str,
    node_id: str,
    mark: str = "irrelevant",
    *,
    at: str = "2026-09-28T10:00:00Z",
    accepted: bool = True,
) -> None:
    store.record_feedback_marks(
        [
            {
                "recall_event_id": event_id,
                "node_id": node_id,
                "mark": mark,
                "accepted": accepted,
                "via_tool": "memory_recall",
                "marked_at": at,
            }
        ]
    )


def _credit(store: MemoryStore, node_id: str, basis: str, at: str) -> None:
    conn = store.connection
    with conn:
        if basis == "explicit":
            conn.execute(
                "INSERT INTO recall_explicit_credit (recall_event_id, node_id, source_id, credited_at)"
                " VALUES (?, ?, 'src', ?)",
                (f"ev-{basis}-{at}", node_id, at),
            )
        else:
            conn.execute(
                "INSERT INTO recall_credit_ledger (recall_event_id, node_id, basis, source_id, credited_at)"
                " VALUES (?, ?, ?, 'src', ?)",
                (f"ev-{basis}-{at}", node_id, basis, at),
            )


def _mark_distinct(store: MemoryStore, node_id: str, queries: list[str]) -> None:
    for query in queries:
        _mark(store, _event(store, query), node_id)


def _recall(store: MemoryStore, query: str = "kappa ledger") -> list[tuple[str, float]]:
    service = MemoryRecallService(store)
    results = service.memory_recall(
        query, scope=SCOPE, max_results=4, log_access=False, log_event=False
    )
    return [(result.node.id, round(result.score, 12)) for result in results]


# ---------------------------------------------------------------------------
# Valves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["", "  ", "abc", "1.0", "1.5", "-0.1", "nan", "inf"])
def test_invalid_or_unset_factor_is_off(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("LM_HUB_SUPPRESSION_FACTOR", raw)
    assert hub_suppression_factor() is None


@pytest.mark.parametrize("raw,expected", [("0", 0.0), ("0.2", 0.2), ("0.999", 0.999)])
def test_valid_factor(monkeypatch: pytest.MonkeyPatch, raw: str, expected: float) -> None:
    monkeypatch.setenv("LM_HUB_SUPPRESSION_FACTOR", raw)
    assert hub_suppression_factor() == expected


@pytest.mark.parametrize("raw,expected", [(None, 3), ("", 3), ("x", 3), ("0", 3), ("-2", 3), ("5", 5), ("1", 1)])
def test_min_queries_valve(monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: int) -> None:
    if raw is not None:
        monkeypatch.setenv("LM_HUB_MIN_QUERIES", raw)
    assert DEFAULT_HUB_MIN_QUERIES == 3
    assert hub_min_queries() == expected


# ---------------------------------------------------------------------------
# Off means identical ranking
# ---------------------------------------------------------------------------


def test_off_is_identity_on_ranking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with MemoryStore(tmp_path / "m.sqlite3") as store:
        ids = _seed(store)
        _recall(store)  # drain the lazy embedding backfill before the baseline
        baseline = _recall(store)
        _mark_distinct(store, ids[0], QUERIES[:3])
        assert hub_demotions(store) == {}
        assert _recall(store) == baseline
        monkeypatch.setenv("LM_HUB_SUPPRESSION_FACTOR", "1.0")  # out of range -> off
        assert hub_demotions(store) == {}
        assert _recall(store) == baseline


def test_on_demotes_hub_for_every_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with MemoryStore(tmp_path / "m.sqlite3") as store:
        ids = _seed(store)
        _recall(store)
        baseline = dict(_recall(store))
        _mark_distinct(store, ids[0], QUERIES[:3])
        monkeypatch.setenv("LM_HUB_SUPPRESSION_FACTOR", "0.1")
        assert hub_demotions(store) == {ids[0]: 0.1}
        demoted = dict(_recall(store))
        assert demoted[ids[0]] == pytest.approx(baseline[ids[0]] * 0.1, rel=1e-9)
        for other in ids[1:]:
            assert demoted[other] == baseline[other]
        # A query never marked for it demotes it too: the multiplier is global.
        unrelated = dict(_recall(store, "vendor csv dialect cold storage"))
        if ids[0] in unrelated:
            assert unrelated[ids[0]] < max(unrelated.values())


# ---------------------------------------------------------------------------
# Threshold on distinct questions
# ---------------------------------------------------------------------------


def test_repeats_of_one_query_do_not_count(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.sqlite3") as store:
        ids = _seed(store)
        # Five events, one question (whitespace variants collapse to it).
        for query in ["kappa ledger drift", "kappa  ledger drift", " kappa ledger drift ", "kappa ledger drift", "kappa ledger\tdrift"]:
            _mark(store, _event(store, query), ids[0])
        [row] = hub_counts(store.connection)
        assert (row.irrelevant_marks, row.distinct_queries, row.distinct_anchors) == (5, 1, 1)
        assert row.hub is False
        assert hub_ids(store.connection) == {}


def test_threshold_counts_distinct_queries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with MemoryStore(tmp_path / "m.sqlite3") as store:
        ids = _seed(store)
        _mark_distinct(store, ids[0], QUERIES[:2])
        assert hub_ids(store.connection) == {}
        _mark_distinct(store, ids[0], QUERIES[2:3])
        assert hub_ids(store.connection) == {ids[0]: 3}
        assert hub_ids(store.connection, min_queries=4) == {}
        # The same question in another scope is another question.
        _mark(store, _event(store, QUERIES[0], scope="project:other"), ids[0])
        assert hub_ids(store.connection, min_queries=4) == {ids[0]: 4}
        monkeypatch.setenv("LM_HUB_SUPPRESSION_FACTOR", "0.2")
        monkeypatch.setenv("LM_HUB_MIN_QUERIES", "5")
        assert hub_demotions(store) == {}
        monkeypatch.setenv("LM_HUB_MIN_QUERIES", "4")
        assert hub_demotions(store) == {ids[0]: 0.2}


def test_rejected_marks_do_not_count(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.sqlite3") as store:
        ids = _seed(store)
        for query in QUERIES[:3]:
            _mark(store, _event(store, query), ids[0], accepted=False)
        assert hub_counts(store.connection) == []


def test_near_duplicate_anchor_counts_once(tmp_path: Path) -> None:
    """Two raw queries the write path resolved to one anchor are one question."""

    with MemoryStore(tmp_path / "m.sqlite3") as store:
        ids = _seed(store)
        events = [_event(store, query) for query in QUERIES[:3]]
        for event_id in events:
            _mark(store, event_id, ids[0])
        conn = store.connection
        with conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS query_irrelevance (anchor_id TEXT, node_id TEXT,"
                " weight REAL, marks INTEGER, cancels INTEGER, last_event_id TEXT,"
                " created_at TEXT, updated_at TEXT)"
            )
            for event_id in events[:2]:
                conn.execute(
                    "INSERT INTO query_irrelevance VALUES ('A1', ?, 1.0, 1, 0, ?, 'x', 'x')",
                    (ids[0], event_id),
                )
        [row] = hub_counts(conn)
        assert (row.distinct_queries, row.distinct_anchors, row.hub) == (3, 2, False)


def test_exact_anchor_is_the_identity(tmp_path: Path) -> None:
    """A fingerprint hit in query_anchors keys the mark by that anchor."""

    with MemoryStore(tmp_path / "m.sqlite3") as store:
        from living_memory.storage import recall_fingerprint

        ids = _seed(store)
        _mark_distinct(store, ids[0], QUERIES[:3])
        anchor = store.insert_query_anchor(
            scope=SCOPE,
            query=QUERIES[0],
            fingerprint=recall_fingerprint(QUERIES[0], SCOPE),
            embedding=[1.0, 0.0],
        )
        [row] = hub_counts(store.connection)
        assert row.distinct_anchors == 3 and row.hub
        assert anchor.id


def test_before_cutoff(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.sqlite3") as store:
        ids = _seed(store)
        for day, query in zip(("01", "02", "03"), QUERIES[:3]):
            _mark(store, _event(store, query), ids[0], at=f"2026-09-{day}T00:00:00Z")
        assert hub_ids(store.connection) == {ids[0]: 3}
        assert hub_ids(store.connection, before="2026-09-03T00:00:00Z") == {}
        _credit(store, ids[0], "grounded", "2026-09-05T00:00:00Z")
        assert hub_ids(store.connection) == {}
        assert hub_ids(store.connection, before="2026-09-04T00:00:00Z") == {ids[0]: 3}


# ---------------------------------------------------------------------------
# Positive evidence lifts
# ---------------------------------------------------------------------------


def test_used_mark_ever_lifts(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.sqlite3") as store:
        ids = _seed(store)
        # used BEFORE the first irrelevant mark still lifts: "ever".
        _mark(store, _event(store, "kappa ledger used"), ids[0], "used", at="2026-09-01T00:00:00Z")
        _mark_distinct(store, ids[0], QUERIES[:3])
        [row] = [item for item in hub_counts(store.connection) if item.node_id == ids[0]]
        assert row.used_marks == 1 and row.hub is False


def test_rejected_used_mark_does_not_lift(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "m.sqlite3") as store:
        ids = _seed(store)
        _mark(store, _event(store, "kappa ledger used"), ids[0], "used", accepted=False)
        _mark_distinct(store, ids[0], QUERIES[:3])
        assert hub_ids(store.connection) == {ids[0]: 3}


@pytest.mark.parametrize("basis", ["grounded", "lookup", "explicit"])
def test_credit_at_or_after_first_mark_lifts(tmp_path: Path, basis: str) -> None:
    with MemoryStore(tmp_path / "m.sqlite3") as store:
        ids = _seed(store)
        _mark_distinct(store, ids[0], QUERIES[:3])  # all at 2026-09-28T10:00:00Z
        _credit(store, ids[0], basis, "2026-09-28T09:59:59Z")
        assert hub_ids(store.connection) == {ids[0]: 3}, "credit before first mark"
        _credit(store, ids[0], basis, "2026-09-28T10:00:00Z")
        [row] = hub_counts(store.connection)
        assert row.credits_after_first == 1 and row.hub is False


def test_cached_hub_lifts_immediately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM_HUB_SUPPRESSION_FACTOR", "0.3")
    monkeypatch.setattr(hubs, "HUB_REFRESH_SECONDS", 3600.0)
    with MemoryStore(tmp_path / "m.sqlite3") as store:
        ids = _seed(store)
        _mark_distinct(store, ids[0], QUERIES[:3])
        _mark_distinct(store, ids[1], QUERIES[:3])
        _mark_distinct(store, ids[2], QUERIES[:3])
        assert hub_demotions(store) == {ids[0]: 0.3, ids[1]: 0.3, ids[2]: 0.3}
        # Positive evidence arrives inside the refresh window: no recount, lifted anyway.
        _credit(store, ids[0], "lookup", "2026-09-28T11:00:00Z")
        _mark(store, _event(store, "kappa ledger yes"), ids[1], "used", at="2026-09-28T11:00:00Z")
        assert hub_demotions(store) == {ids[2]: 0.3}
        # A new hub waits for the refresh window ...
        _mark_distinct(store, ids[3], QUERIES[:3])
        assert hub_demotions(store) == {ids[2]: 0.3}
        # ... and appears once it has passed.
        monkeypatch.setattr(hubs, "HUB_REFRESH_SECONDS", 0.0)
        assert hub_demotions(store) == {ids[2]: 0.3, ids[3]: 0.3}


def test_cache_serves_unchanged_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM_HUB_SUPPRESSION_FACTOR", "0.3")
    with MemoryStore(tmp_path / "m.sqlite3") as store:
        ids = _seed(store)
        _mark_distinct(store, ids[0], QUERIES[:3])
        calls: list[int] = []
        original = hubs.hub_counts
        monkeypatch.setattr(hubs, "hub_counts", lambda *a, **k: calls.append(1) or original(*a, **k))
        assert hub_demotions(store) == {ids[0]: 0.3}
        assert hub_demotions(store) == {ids[0]: 0.3}
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Absent tables
# ---------------------------------------------------------------------------


def test_empty_database_without_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = sqlite3.connect(":memory:")
    assert hub_counts(conn) == []
    assert hub_ids(conn) == {}
    monkeypatch.setenv("LM_HUB_SUPPRESSION_FACTOR", "0.2")

    class Bare:
        connection = conn

    assert hub_demotions(Bare()) == {}
    assert hub_demotions(object()) == {}


def test_marks_only_database(tmp_path: Path) -> None:
    """No events, anchors, credit or irrelevance tables: event ids stand in for questions."""

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE recall_feedback_marks (id INTEGER PRIMARY KEY, recall_event_id TEXT,"
        " node_id TEXT, mark TEXT, accepted INTEGER, marked_at TEXT)"
    )
    for event_id in ("e1", "e2", "e3"):
        conn.execute(
            "INSERT INTO recall_feedback_marks (recall_event_id, node_id, mark, accepted, marked_at)"
            " VALUES (?, 'n1', 'irrelevant', 1, '2026-09-28T00:00:00Z')",
            (event_id,),
        )
    assert hub_ids(conn) == {"n1": 3}


def test_read_only_snapshot_without_credit_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "m.sqlite3"
    with MemoryStore(db) as store:
        ids = _seed(store)
        _mark_distinct(store, ids[0], QUERIES[:3])
        with store.connection as conn:
            conn.execute("DROP TABLE recall_credit_ledger")
            conn.execute("DROP TABLE recall_explicit_credit")
            conn.execute("DROP TABLE query_anchor_edges")
            conn.execute("DROP TABLE query_anchors")
    ro = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    assert hub_ids(ro) == {ids[0]: 3}

    class Snapshot:
        connection = ro

    monkeypatch.setenv("LM_HUB_SUPPRESSION_FACTOR", "0.25")
    assert hub_demotions(Snapshot()) == {ids[0]: 0.25}


# ---------------------------------------------------------------------------
# Merge with query-relative demotions
# ---------------------------------------------------------------------------


def test_merge_takes_the_smaller_multiplier() -> None:
    assert _merge_demotions({"a": 0.8, "b": 0.2}, {"a": 0.3, "c": 0.5}) == {
        "a": 0.3,
        "b": 0.2,
        "c": 0.5,
    }


def test_service_merges_hub_and_query_demotions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM_HUB_SUPPRESSION_FACTOR", "0.4")
    with MemoryStore(tmp_path / "m.sqlite3") as store:
        ids = _seed(store)
        _mark_distinct(store, ids[0], QUERIES[:3])
        service = MemoryRecallService(store)
        monkeypatch.setattr(
            service,
            "_collect_query_demotions",
            lambda *_a, **_k: {ids[0]: 0.9, ids[1]: 0.2},
        )
        seen: dict[str, Any] = {}
        original = service.rank_candidates

        def capture(*args: Any, **kwargs: Any) -> Any:
            seen.update(kwargs.get("demotions") or {})
            return original(*args, **kwargs)

        monkeypatch.setattr(service, "rank_candidates", capture)
        service.memory_recall("kappa ledger", scope=SCOPE, max_results=4, log_access=False, log_event=False)
        assert seen == {ids[0]: 0.4, ids[1]: 0.2}
