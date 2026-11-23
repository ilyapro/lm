"""Decay admission through the time-based maintenance sweep on a shared WAL."""

from contextlib import contextmanager
from datetime import UTC, datetime
import sqlite3

import pytest

from living_memory import decay
from living_memory.config import MemoryConfig
from living_memory.server import _DECAY_SWEEP_KV_KEY, _decay_sweep_if_due
from living_memory.storage import MemoryStore


NOW = datetime(2026, 10, 23, tzinfo=UTC)


@pytest.mark.parametrize(
    "change,anchor,opt_in,expected_reason",
    [
        ("stable", "created", False, decay.TTL_EXPIRED_REASON),
        ("cover", "created", False, None),
        ("read", "last_accessed", False, None),
        ("read", "created", False, decay.TTL_EXPIRED_REASON),
        ("read-covered", "created", True, None),
        ("read-covered-old", "created", True, decay.TTL_EXPIRED_REASON),
        ("uncover", "created", True, None),
        ("forget", "created", False, "manual"),
    ],
)
def test_sweep_admits_current_ttl_reason(
    tmp_path, monkeypatch, change, anchor, opt_in, expected_reason,
):
    monkeypatch.setenv(decay.TTL_ANCHOR_ENV, anchor)
    monkeypatch.setenv(decay.CONCEPT_COVERED_ENV, str(int(opt_in)))
    monkeypatch.setenv(decay.MAX_PER_SWEEP_ENV, "2")
    config = MemoryConfig(db_path=tmp_path / "memory.sqlite3", trace_ttl_days=7)
    with MemoryStore(config) as store, MemoryStore(config) as peer:
        assert peer.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        fresh = change in {"uncover", "read-covered"}
        for node_id, stamp in [
            ("first", "2019-01-01"), ("subject", "2026-10-20" if fresh else "2020-01-01"),
            ("next", "2026-10-21"),
        ]:
            store.create_node(level="trace", content=node_id, node_id=node_id,
                              context={"timestamp": stamp})
        if opt_in:
            store.create_node(level="schema", content="digest", node_id="owner",
                              provenance={"source_traces": ["subject", "next"],
                                          "source_coverage": {"subject": "2026-10-20"}})
        elif change != "stable":
            peer.connection.execute("UPDATE nodes SET timestamp='2021-01-01' WHERE id='next'")
            peer.connection.commit()

        reservations = 0
        transaction = store.transaction

        @contextmanager
        def interleave(*, immediate=False):
            nonlocal reservations
            if immediate:
                reservations += 1
                if reservations == 2:
                    assert peer.get_node("first").decayed
                    if change == "cover":
                        peer.create_node(level="concept", content="digest in another scope",
                                         context={"scope": "project:other"},
                                         provenance={"source_traces": ["subject"]})
                    elif change.startswith("read"):
                        peer.update_node("subject", stats={"last_accessed": NOW.isoformat()})
                    elif change == "uncover":
                        peer.update_node("owner", provenance={"source_traces": ["next"]})
                    elif change == "forget":
                        peer.soft_delete_node("subject", "manual")
            with transaction(immediate=immediate):
                yield

        scans = []

        def observe_sql(sql):
            if "SELECT source_traces" in sql or "SELECT id, timestamp, created_at" in sql:
                scans.append((sql, store.connection.in_transaction))

        monkeypatch.setattr(store, "transaction", interleave)
        store.connection.set_trace_callback(observe_sql)
        result = _decay_sweep_if_due(store, force=True, now=NOW)
        assert peer.get_node("subject").decay_reason == expected_reason
        assert peer.get_node("subject").decayed == (expected_reason is not None)
        assert peer.get_node("next").decayed == (expected_reason in (None, "manual"))
        assert result["expired_count"] == 2
        assert all(not in_transaction for _, in_transaction in scans)
        assert len(scans) == (2 if change == "stable" else 4)


def test_sweep_preserves_committed_prefix_on_write_failure(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for node_id in ("a", "b", "c"):
            store.create_node(level="trace", content=node_id, node_id=node_id,
                              context={"timestamp": "2000-01-01"})
        store.connection.execute("""
            CREATE TRIGGER fail_b BEFORE UPDATE OF decayed ON nodes WHEN OLD.id='b'
            BEGIN SELECT RAISE(ABORT, 'failed retirement'); END
        """)
        with pytest.raises(sqlite3.IntegrityError, match="failed retirement"):
            _decay_sweep_if_due(store, force=True, now=NOW)
        assert [store.get_node(i).decayed for i in ("a", "b", "c")] == [True, False, False]
        assert store.get_kv(_DECAY_SWEEP_KV_KEY) is None


def test_repeated_external_changes_bound_rescans_and_keep_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv(decay.CONCEPT_COVERED_ENV, "0")
    db = tmp_path / "memory.sqlite3"
    with MemoryStore(db) as store, MemoryStore(db) as peer:
        for node_id in ("a", "b", "c"):
            store.create_node(level="trace", content=node_id, node_id=node_id,
                              context={"timestamp": "2000-01-01"})
        transaction = store.transaction
        writes = iter([None, "b", "c"])

        @contextmanager
        def interleave(*, immediate=False):
            if immediate:
                source = next(writes)
                if source:
                    peer.create_node(level="concept", content="live digest",
                                     provenance={"source_traces": [source]})
            with transaction(immediate=immediate):
                yield

        monkeypatch.setattr(store, "transaction", interleave)
        result = _decay_sweep_if_due(store, force=True, now=NOW)
        assert result["expired_count"] == 1
        assert [peer.get_node(i).decayed for i in ("a", "b", "c")] == [True, False, False]


def test_retiring_covering_owner_invalidates_its_source_in_same_sweep(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.create_node(level="trace", content="recent source", node_id="source",
                          context={"timestamp": "2026-10-22"})
        store.create_node(level="schema", content="expired owner", node_id="owner",
                          context={"timestamp": "2000-01-01"},
                          provenance={"source_traces": ["source"]})
        retired = decay.soft_delete_expired(
            store, now=NOW, ttl_days=7, levels=iter(["schema", "trace"]),
            decay_concept_covered=True, ttl_anchor="created",
        )
        assert [(node.id, node.decay_reason) for node in retired] == [
            ("owner", decay.TTL_EXPIRED_REASON),
        ]
        assert not store.get_node("source").decayed


@pytest.mark.parametrize("change, expected", [
    ("stable", "superseded"),
    ("forget-original", "manual"),
    ("forget-correction", None),
    ("second-correction", "superseded"),
    ("reclassify", None),
])
def test_sweep_admits_current_supersession(tmp_path, monkeypatch, change, expected):
    db = tmp_path / "memory.sqlite3"
    ids = ("original", "correction", "other", "unrelated")
    with MemoryStore(db) as store, MemoryStore(db) as peer:
        assert peer.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        for node_id in ids:
            store.create_node(level="trace", node_id=node_id, content=node_id)
        store.create_connection("correction", "original", "supersedes")
        store.create_connection("other", "original", "supersedes", metadata=(
            {} if change == "second-correction" else {"kind": "duplicate_content"}
        ))
        transaction = store.transaction
        reservations = 0

        @contextmanager
        def interleave(*, immediate=False):
            nonlocal reservations
            if immediate:
                reservations += 1
                if reservations == 1:
                    if change == "forget-original":
                        decay.memory_forget(peer, "original", "manual")
                    elif change in {"forget-correction", "second-correction"}:
                        decay.memory_forget(peer, "correction", "manual")
                    elif change == "reclassify":
                        with peer.transaction():
                            peer.connection.execute("""
                                UPDATE connections SET metadata='{"kind":"duplicate_content"}'
                                WHERE source_id='correction'
                            """)
            with transaction(immediate=immediate):
                yield

        monkeypatch.setattr(store, "transaction", interleave)
        scans = []
        store.connection.set_trace_callback(lambda sql: scans.append(store.connection.in_transaction)
                                            if "FROM connections c" in sql else None)
        result = _decay_sweep_if_due(store, now=NOW)
        count = int(expected == "superseded")
        assert result["decayed_count"] == result["superseded_count"] == count
        assert scans == [False] * (1 if change == "stable" else 2)
        for node_id in ids:
            node = peer.get_node(node_id)
            reason = expected if node_id == "original" else None
            if node_id == "correction" and change in {"forget-correction", "second-correction"}:
                reason = "manual"
            assert (node.content, node.decayed, node.decay_reason) == (node_id, reason is not None, reason)
        assert peer.connection.execute("SELECT count(*) FROM connections").fetchone()[0] == 2


@pytest.mark.parametrize("interruption", ["none", "unrelated", "manual", "repeated", "failure"])
def test_sweep_keeps_its_own_supersession_basis_and_committed_prefix(tmp_path, monkeypatch, interruption):
    db = tmp_path / "memory.sqlite3"
    with MemoryStore(db) as store, MemoryStore(db) as peer:
        # Either first retirement takes the other candidate's only live basis.
        for node_id in ("left", "right", "unrelated"):
            store.create_node(level="trace", node_id=node_id, content=node_id)
        store.create_connection("left", "right", "supersedes")
        store.create_connection("right", "left", "supersedes")
        transaction = store.transaction
        reservations = 0

        @contextmanager
        def interleave(*, immediate=False):
            nonlocal reservations
            if immediate:
                reservations += 1
                if reservations == 2:
                    retired, = [i for i in ("left", "right") if peer.get_node(i).decayed]
                    if interruption == "manual":
                        decay.memory_forget(peer, retired, "manual")
                    elif interruption == "failure":
                        peer.connection.execute("""
                            CREATE TRIGGER fail_retirement BEFORE UPDATE OF decayed ON nodes
                            BEGIN SELECT RAISE(ABORT, 'failed retirement'); END
                        """)
                if reservations >= 2 and interruption in {"unrelated", "repeated"}:
                    if reservations == 2 or interruption == "repeated":
                        peer.set_kv("unrelated-change", str(reservations))
            with transaction(immediate=immediate):
                yield

        monkeypatch.setattr(store, "transaction", interleave)
        if interruption == "failure":
            with pytest.raises(sqlite3.IntegrityError, match="failed retirement"):
                _decay_sweep_if_due(store, now=NOW)
            assert peer.get_kv(_DECAY_SWEEP_KV_KEY) is None
        else:
            result = _decay_sweep_if_due(store, now=NOW)
            assert result["superseded_count"] == (1 if interruption in {"manual", "repeated"} else 2)
        reasons = [peer.get_node(i).decay_reason for i in ("left", "right")]
        if interruption in {"none", "unrelated"}:
            assert reasons == ["superseded", "superseded"]
        else:
            assert set(reasons) == {None, "manual" if interruption == "manual" else "superseded"}
        assert not peer.get_node("unrelated").decayed


def test_scoped_supersession_preserves_chain_and_cross_scope_basis(tmp_path):
    db = tmp_path / "memory.sqlite3"
    with MemoryStore(db) as store, MemoryStore(db) as reader:
        for node_id, scope in [
            ("original", "project:a"), ("middle", "project:a"), ("newest", "project:b"),
            ("outside", "project:b"), ("forgotten", "project:a"), ("survivor", "project:a"),
        ]:
            store.create_node(level="trace", node_id=node_id, content=node_id,
                              context={"scope": scope})
        for source, target in [
            ("newest", "middle"), ("middle", "original"), ("newest", "outside"),
            ("forgotten", "survivor"),
        ]:
            store.create_connection(source, target, "supersedes")
        decay.memory_forget(store, "forgotten", "manual")

        result = decay.apply_decay(store, scope="project:a", now=NOW)
        assert sorted(node.id for node in result.nodes) == ["middle", "original"]
        assert [(node.id, node.decay_reason) for node in
                (reader.get_node(i) for i in ("original", "middle", "newest", "outside",
                                               "forgotten", "survivor"))] == [
            ("original", "superseded"), ("middle", "superseded"), ("newest", None),
            ("outside", None), ("forgotten", "manual"), ("survivor", None),
        ]
