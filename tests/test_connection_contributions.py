"""Graph contributions preserve knowledge, row identity and transactional effects."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
import sqlite3

import pytest

from living_memory import query_anchors, storage
from living_memory.edge_derivation import DerivedEdge, upsert_derived_edge
from living_memory.resources import connection_to_dict
from living_memory.storage import MemoryStore


@pytest.fixture(params=["writer", "typed"])
def contribute(request):
    def apply(store, source, target, relation="related", *, weight=0.7, metadata=None):
        if request.param == "typed":
            return upsert_derived_edge(
                store, DerivedEdge(source, target, relation, weight, metadata or {}, "fixture")
            )
        return store.create_connection(
            source, target, relation, weight=weight, metadata=metadata, merge=True
        )
    return apply


@pytest.fixture
def graph(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        source = store.append_trace("North entrance marker").id
        target = store.append_trace("South exit marker").id
        yield store, source, target


def test_contributions_keep_identity_and_merge_shallowly(graph, contribute, monkeypatch):
    store, source, target = graph
    monkeypatch.setattr(storage, "_utc_now", lambda: "2026-01-01T00:00:00Z")
    first = contribute(store, source, target, weight=.6543214, metadata={
        "keep": "value", "overlap": "old", "nested": {"old": 1, "keep": 2}, "null": "old",
    })
    monkeypatch.setattr(storage, "_utc_now", lambda: "2026-01-02T00:00:00Z")
    merged = contribute(store, source, target, weight=.2, metadata={
        "overlap": "new", "nested": {"new": 3}, "null": None,
    })
    assert (merged.id, merged.created_at) == (first.id, first.created_at)
    assert (merged.source_id, merged.target_id, merged.type) == (source, target, "related")
    assert merged.weight == .654321
    assert merged.metadata == {"keep": "value", "overlap": "new", "nested": {"new": 3}, "null": None}
    assert merged.updated_at == "2026-01-02T00:00:00Z"
    monkeypatch.setattr(storage, "_utc_now", lambda: "2026-01-03T00:00:00Z")
    before = store.connection.total_changes
    repeated = contribute(store, source, target, weight=.3, metadata={"nested": {"new": 3}})
    assert repeated == merged
    assert store.connection.total_changes == before
    assert connection_to_dict(repeated)["updated_at"] == merged.updated_at
    reverse = contribute(store, target, source)
    different_type = contribute(store, source, target, "requires")
    assert len({first.id, reverse.id, different_type.id}) == 3


@pytest.mark.parametrize("weight,expected", [(-2., 0.), (2., 1.), (.1234565, .123456), (.1234567, .123457)])
def test_weight_bounds_and_rounding(graph, contribute, weight, expected):
    store, source, target = graph
    assert contribute(store, source, target, weight=weight).weight == expected
    assert contribute(store, source, target, weight=-5.).weight == expected
    # Ordinary rows can contain values outside contribution bounds.
    store.create_connection(source, target, "related", weight=3.)
    assert contribute(store, source, target, weight=.5).weight == 1.


@pytest.mark.parametrize("before,after", [(True, 1), (1, 1.0), (False, 0), ({"x": True}, {"x": 1})])
def test_metadata_json_types_are_significant(graph, contribute, before, after):
    store, source, target = graph
    first = contribute(store, source, target, metadata={"value": before})
    changes = store.connection.total_changes
    result = contribute(store, source, target, metadata={"value": after})
    assert result.id == first.id
    assert storage._json_dumps(result.metadata) == storage._json_dumps({"value": after})
    assert store.connection.total_changes == changes + 1


@pytest.mark.parametrize("private", [False, True])
def test_replacement_keeps_its_policy_and_timestamp(graph, private, monkeypatch):
    store, source, target = graph
    first = store.create_connection(source, target, "related", weight=.9, metadata={"old": True})
    monkeypatch.setattr(storage, "_utc_now", lambda: "2099-01-01T00:00:00Z")
    call = store._insert_connection if private else store.create_connection
    with store.transaction():
        result = call(source, target, "related", weight=-2., metadata={"new": None})
    assert result.id == first.id and result.created_at == first.created_at
    assert result.weight == -2. and result.metadata == {"new": None}
    assert result.updated_at == "2099-01-01T00:00:00Z"
    monkeypatch.setattr(storage, "_utc_now", lambda: "2099-01-02T00:00:00Z")
    with store.transaction():
        repeated = call(source, target, "related", weight=-2., metadata={"new": None})
    assert repeated.updated_at == "2099-01-02T00:00:00Z"


@pytest.mark.parametrize("missing", ["source", "target", "both"])
def test_missing_endpoints_leave_no_transaction_or_edge(graph, contribute, missing):
    store, source, target = graph
    source = "missing-source" if missing in ("source", "both") else source
    target = "missing-target" if missing in ("target", "both") else target
    with pytest.raises(KeyError) as exc:
        contribute(store, source, target)
    assert exc.value.args == (source if missing != "target" else target,)
    assert store.list_connections() == []
    assert not store.connection.in_transaction


def test_invalid_relation_leaves_no_edge(graph, contribute):
    store, source, target = graph
    with pytest.raises(ValueError):
        contribute(store, source, target, "invalid")
    assert store.list_connections() == []
    assert not store.connection.in_transaction


@pytest.mark.parametrize("existing", [False, True])
def test_outer_rollback_keeps_connections_and_other_writes(graph, contribute, existing):
    store, source, target = graph
    if existing:
        contribute(store, source, target, weight=.2, metadata={"keep": True})
    before = list(store.connection.iterdump())
    with pytest.raises(RuntimeError, match="abort"):
        with store.transaction():
            store.append_trace("Uncommitted knowledge")
            contribute(store, source, target, weight=.9, metadata={"new": True})
            raise RuntimeError("abort")
    assert list(store.connection.iterdump()) == before
    assert not store.connection.in_transaction


@pytest.mark.parametrize("fail", [False, True])
def test_repeated_supersedes_moves_new_anchors_atomically(graph, contribute, monkeypatch, fail):
    store, source, target = graph
    original = contribute(store, source, target, "supersedes")
    anchor = query_anchors.upsert_anchor(store, "where is the marker", "global", [1., 0.], []).anchor
    store.upsert_query_anchor_edge(anchor.id, target, weight=.2)
    assert [e.target_id for e in store.list_query_anchor_edges(anchor_id=anchor.id)] == [target]
    store.upsert_query_anchor_edge(anchor.id, source, weight=.4)
    anchors_before = store.list_query_anchor_edges(anchor_id=anchor.id)
    rows_before = [tuple(row) for row in store.connection.execute("SELECT * FROM connections")]
    revision = store.query_anchor_revision()
    if fail:
        follow = store._follow_supersedes_with_anchor_edges
        def failing_hook(*args, **kwargs):
            follow(*args, **kwargs)
            raise RuntimeError("anchor fault")
        monkeypatch.setattr(store, "_follow_supersedes_with_anchor_edges", failing_hook)
        with pytest.raises(RuntimeError, match="anchor fault"):
            contribute(store, source, target, "supersedes")
        assert store.query_anchor_revision() == revision
        assert store.list_query_anchor_edges(anchor_id=anchor.id) == anchors_before
    else:
        result = contribute(store, source, target, "supersedes")
        assert result == original
        assert store.query_anchor_revision() == revision
        edges = store.list_query_anchor_edges(anchor_id=anchor.id)
        assert [e.target_id for e in edges] == [source]
        assert edges[0].hits == 2
        assert edges[0].weight == pytest.approx(.6)
    assert [tuple(row) for row in store.connection.execute("SELECT * FROM connections")] == rows_before
    assert not store.connection.in_transaction


@pytest.mark.parametrize("existing", [False, True])
def test_concurrent_contributions_read_after_acquiring_writer(graph, contribute, existing):
    left, source, target = graph
    if existing:
        contribute(left, source, target, weight=.2, metadata={"seed": True})
    with MemoryStore(left.db_path) as right:
        read, release, right_started, right_read = Event(), Event(), Event(), Event()
        def pause_read(sql):
            if sql.lstrip().startswith("SELECT * FROM connections") and not read.is_set():
                read.set()
                assert release.wait(10)
        def observe_right(sql):
            if sql.startswith("BEGIN IMMEDIATE"):
                right_started.set()
            if sql.lstrip().startswith("SELECT * FROM connections"):
                right_read.set()
        left.connection.set_trace_callback(pause_read)
        right.connection.set_trace_callback(observe_right)
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(contribute, left, source, target, weight=.5, metadata={"left": True})
            try:
                assert read.wait(10)
                second = workers.submit(contribute, right, source, target, weight=.9, metadata={"right": True})
                assert right_started.wait(10)
                assert not right_read.is_set()
            finally:
                release.set()
            one, two = first.result(timeout=10), second.result(timeout=10)
        left.connection.set_trace_callback(None)
        assert one.id == two.id
        assert two.weight == .9
        assert two.metadata == {**({"seed": True} if existing else {}), "left": True, "right": True}
        assert left.get_connection(one.id) == two


def test_stale_outer_snapshot_cannot_overwrite_concurrent_contribution(graph, contribute):
    left, source, target = graph
    contribute(left, source, target, weight=.2)
    with MemoryStore(left.db_path) as right:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            with left.transaction():
                left.list_connections()
                committed = contribute(right, source, target, weight=.9, metadata={"right": True})
                contribute(left, source, target, weight=.5, metadata={"left": True})
        assert left.get_connection(committed.id) == committed
        assert not left.connection.in_transaction
