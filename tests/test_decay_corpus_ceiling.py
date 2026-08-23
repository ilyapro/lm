"""The live working set gets a ceiling: a trace's TTL counts from creation.

Anchored on ``last_accessed`` — the behaviour that shipped first — a trace that
recall keeps returning is immortal, because the read that returns it is also
the read that renews it. So the corpus only ever grew. Every test here is
written as a discriminator between the two anchors: the same trace, read
yesterday and created long ago, must decay under the shipped default and must
survive under ``LM_DECAY_TTL_ANCHOR=last_accessed``.

The second rule in this file is the one the operator owns. Coverage is
recorded as a concept's ``provenance.source_traces``, so "decay traces a
concept already covers" and "never decay a trace a live concept cites" are the
same set read two ways. Only the guard ships on; the rule sits behind
``LM_DECAY_CONCEPT_COVERED`` and is proven here to be off until asked for.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from living_memory.config import MemoryConfig
from living_memory.consolidation import memory_consolidate, memory_teach
from living_memory.decay import (
    CONCEPT_COVERED_ENV,
    CONCEPT_COVERED_REASON,
    DEFAULT_MAX_EXPIRED_PER_SWEEP,
    MAX_PER_SWEEP_ENV,
    TTL_ANCHOR_CREATED,
    TTL_ANCHOR_ENV,
    TTL_ANCHOR_LAST_ACCESSED,
    TTL_EXPIRED_REASON,
    apply_decay,
    max_expired_per_sweep,
    resolve_ttl_anchor,
)
from living_memory.server import _decay_sweep_if_due
from living_memory.storage import MemoryStore

SCOPE = "project:ceiling"
TOPIC = "deploy rollback requires a migration guard before release"

#: Every test states its own "now" so the fixtures read as ages, not dates.
NOW = datetime(2026, 8, 23, 12, 0, tzinfo=UTC)
TTL_DAYS = 180

#: The age of the oldest active trace on the live corpus the day the anchor
#: shipped (measured read-only 2026-08-23: created 2026-05-14, 100.1 days).
#: Pinned here so the "this default decays nothing on ship day" claim in
#: ``decay.py`` is a test, not a comment.
LIVE_CORPUS_OLDEST_DAYS = 101


@pytest.fixture(autouse=True)
def _no_ambient_valves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test states the valves it wants; none inherits the operator's."""

    for name in (TTL_ANCHOR_ENV, CONCEPT_COVERED_ENV, MAX_PER_SWEEP_ENV):
        monkeypatch.delenv(name, raising=False)


def _open(tmp_path: Path, *, ttl_days: int = TTL_DAYS) -> MemoryStore:
    return MemoryStore(
        MemoryConfig(db_path=tmp_path / "memory.sqlite3", trace_ttl_days=ttl_days)
    )


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _trace(
    store: MemoryStore,
    content: str,
    *,
    created: datetime,
    read: datetime | None = None,
    scope: str = SCOPE,
) -> str:
    """Append one trace with an explicit creation moment and read history."""

    node = store.append_trace(
        content,
        {"scope": scope, "agent": "agent-a", "timestamp": _iso(created)},
    )
    if read is not None:
        store.update_node(node.id, stats={"last_accessed": _iso(read)})
    return node.id


def _cover(
    store: MemoryStore,
    trace_ids: list[str],
    *,
    covered_at: datetime,
    scope: str = SCOPE,
) -> str:
    """Write the concept-shaped node that records coverage of those traces."""

    concept = store.create_node(
        level="concept",
        content=f"digest of {len(trace_ids)} traces about {TOPIC}",
        context={"scope": scope, "agent": "consolidator", "timestamp": _iso(covered_at)},
        provenance={
            "source_traces": list(trace_ids),
            "consolidated_at": _iso(covered_at),
        },
    )
    return concept.id


def _alive(store: MemoryStore, node_id: str) -> bool:
    node = store.get_node(node_id)
    assert node is not None, "decay is soft: the row must survive"
    return not node.decayed


# ---------- the anchor ----------


def test_creation_anchor_retires_a_constantly_read_trace_and_spares_a_young_one(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as store:
        immortal = _trace(
            store,
            "read yesterday, created long before the ttl",
            created=NOW - timedelta(days=200),
            read=NOW - timedelta(days=1),
        )
        young = _trace(
            store,
            "created three days ago and never read once",
            created=NOW - timedelta(days=3),
        )
        ship_day_oldest = _trace(
            store,
            "as old as the oldest live trace on the day the anchor shipped",
            created=NOW - timedelta(days=LIVE_CORPUS_OLDEST_DAYS),
            read=NOW - timedelta(days=1),
        )

        result = apply_decay(store, now=NOW)

        assert [node.id for node in result.expired] == [immortal]
        assert store.get_node(immortal).decay_reason == TTL_EXPIRED_REASON
        assert _alive(store, young)
        # The documented margin: with trace_ttl_days=180 nothing the age of the
        # live corpus decays on ship day. The ceiling arrives later, by itself.
        assert _alive(store, ship_day_oldest)


def test_legacy_env_value_restores_last_accessed_anchoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TTL_ANCHOR_ENV, TTL_ANCHOR_LAST_ACCESSED)
    assert resolve_ttl_anchor() == TTL_ANCHOR_LAST_ACCESSED

    with _open(tmp_path) as store:
        renewed_by_reading = _trace(
            store,
            "read yesterday, created long before the ttl",
            created=NOW - timedelta(days=200),
            read=NOW - timedelta(days=1),
        )
        never_read = _trace(
            store,
            "created long before the ttl and never read",
            created=NOW - timedelta(days=200),
        )

        result = apply_decay(store, now=NOW)

        # Legacy anchoring, byte for byte: last read, falling back to creation.
        assert [node.id for node in result.expired] == [never_read]
        assert _alive(store, renewed_by_reading)


def test_an_unrecognised_anchor_value_falls_back_to_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TTL_ANCHOR_ENV, "whenever-feels-right")
    assert resolve_ttl_anchor() == TTL_ANCHOR_CREATED

    with _open(tmp_path) as store:
        immortal = _trace(
            store,
            "read yesterday, created long before the ttl",
            created=NOW - timedelta(days=200),
            read=NOW - timedelta(days=1),
        )

        result = apply_decay(store, now=NOW)

        assert [node.id for node in result.expired] == [immortal]


# ---------- the invariant under shipped defaults ----------


def test_a_live_concepts_source_traces_survive_a_default_sweep(tmp_path: Path) -> None:
    """The acceptance invariant, over a concept built by real consolidation."""

    with _open(tmp_path) as store:
        for index in range(8):
            _trace(
                store,
                f"{TOPIC} sample {index}",
                created=NOW - timedelta(days=200 + index),
                read=NOW - timedelta(days=1),
            )

        report = memory_consolidate(store, scope=SCOPE)
        assert report.concepts_created, "fixture must promote a live concept"
        concept = report.concepts_created[0]
        assert concept.source_traces, "a concept without sources proves nothing"

        # Through the server's own sweep, not just apply_decay: the valves are
        # read at call time, so _decay_sweep_if_due needs no edit to honour them.
        summary = _decay_sweep_if_due(store, force=True, now=NOW)
        assert summary is not None and summary["swept"] is True

        assert not store.get_node(concept.id).decayed
        for trace_id in concept.source_traces:
            assert _alive(store, trace_id), f"{trace_id} is provenance for a live concept"


def test_the_coverage_rule_is_off_until_the_operator_turns_it_on(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as store:
        unread = _trace(store, "covered and never read", created=NOW - timedelta(days=10))
        read_before = _trace(
            store,
            "covered and last read before coverage",
            created=NOW - timedelta(days=10),
            read=NOW - timedelta(days=9),
        )
        _cover(store, [unread, read_before], covered_at=NOW - timedelta(days=5))

        result = apply_decay(store, now=NOW)

        assert result.expired == []
        assert _alive(store, unread)
        assert _alive(store, read_before)


def test_the_coverage_rule_retires_covered_traces_nobody_read_since(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CONCEPT_COVERED_ENV, "1")

    with _open(tmp_path) as store:
        unread = _trace(store, "covered and never read", created=NOW - timedelta(days=10))
        read_before = _trace(
            store,
            "covered and last read before coverage",
            created=NOW - timedelta(days=10),
            read=NOW - timedelta(days=9),
        )
        read_since = _trace(
            store,
            "covered but still being read",
            created=NOW - timedelta(days=10),
            read=NOW - timedelta(days=1),
        )
        uncovered = _trace(
            store, "young and covered by nothing", created=NOW - timedelta(days=10)
        )
        _cover(store, [unread, read_before, read_since], covered_at=NOW - timedelta(days=5))

        result = apply_decay(store, now=NOW)

        assert {node.id for node in result.expired} == {unread, read_before}
        assert store.get_node(unread).decay_reason == CONCEPT_COVERED_REASON
        assert _alive(store, read_since)
        assert _alive(store, uncovered)


def test_a_covered_trace_still_being_read_falls_through_to_the_creation_ttl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CONCEPT_COVERED_ENV, "1")

    with _open(tmp_path) as store:
        old = _trace(
            store,
            "covered, read yesterday, created before the ttl",
            created=NOW - timedelta(days=200),
            read=NOW - timedelta(days=1),
        )
        young = _trace(
            store,
            "covered, read yesterday, created inside the ttl",
            created=NOW - timedelta(days=10),
            read=NOW - timedelta(days=1),
        )
        _cover(store, [old, young], covered_at=NOW - timedelta(days=5))

        result = apply_decay(store, now=NOW)

        assert [node.id for node in result.expired] == [old]
        assert store.get_node(old).decay_reason == TTL_EXPIRED_REASON
        assert _alive(store, young)


def test_a_correction_still_retires_a_covered_trace(tmp_path: Path) -> None:
    """Coverage guards the TTL path only — supersedes must keep winning."""

    with _open(tmp_path) as store:
        original = _trace(
            store, "release channel is beta", created=NOW - timedelta(days=10)
        )
        _cover(store, [original], covered_at=NOW - timedelta(days=5))
        taught = memory_teach(
            store, original, "release channel is stable", context={"agent": "agent-b"}
        )

        result = apply_decay(store, now=NOW)

        assert [node.id for node in result.superseded] == [original]
        assert not _alive(store, original)
        assert _alive(store, taught.corrective_trace.id)


# ---------- the per-sweep cap ----------


def test_the_cap_bounds_one_pass_and_the_next_sweep_resumes_oldest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MAX_PER_SWEEP_ENV, "2")

    with _open(tmp_path) as store:
        ages = [204, 203, 202, 201, 200]
        by_age = {
            age: _trace(
                store, f"expired trace aged {age} days", created=NOW - timedelta(days=age)
            )
            for age in ages
        }

        first = apply_decay(store, now=NOW)
        second = apply_decay(store, now=NOW)
        third = apply_decay(store, now=NOW)
        fourth = apply_decay(store, now=NOW)

        assert [node.id for node in first.expired] == [by_age[204], by_age[203]]
        assert [node.id for node in second.expired] == [by_age[202], by_age[201]]
        assert [node.id for node in third.expired] == [by_age[200]]
        assert fourth.expired == []


def test_the_cap_valve_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert max_expired_per_sweep() == DEFAULT_MAX_EXPIRED_PER_SWEEP

    monkeypatch.setenv(MAX_PER_SWEEP_ENV, "not-a-number")
    assert max_expired_per_sweep() == DEFAULT_MAX_EXPIRED_PER_SWEEP

    monkeypatch.setenv(MAX_PER_SWEEP_ENV, "7")
    assert max_expired_per_sweep() == 7

    monkeypatch.setenv(MAX_PER_SWEEP_ENV, "0")
    assert max_expired_per_sweep() == 0


def test_cap_zero_means_uncapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MAX_PER_SWEEP_ENV, "0")

    with _open(tmp_path) as store:
        expired = [
            _trace(store, f"expired trace {index}", created=NOW - timedelta(days=200))
            for index in range(5)
        ]

        result = apply_decay(store, now=NOW)

        assert {node.id for node in result.expired} == set(expired)


# ---------- soft delete, always ----------


def test_a_sweep_never_removes_a_row(tmp_path: Path) -> None:
    with _open(tmp_path) as store:
        doomed = _trace(
            store, "expired but not forgotten", created=NOW - timedelta(days=200)
        )
        before = store.connection.execute("SELECT COUNT(*) AS c FROM nodes").fetchone()["c"]

        apply_decay(store, now=NOW)

        after = store.connection.execute("SELECT COUNT(*) AS c FROM nodes").fetchone()["c"]
        assert after == before
        node = store.get_node(doomed)
        assert node is not None
        assert node.decayed is True
        assert node.content == "expired but not forgotten"
        assert doomed in {
            found.id for found in store.list_nodes(scope=SCOPE, include_decayed=True)
        }
