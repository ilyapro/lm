"""Coverage age survives ordinary maintenance without aging new sources."""

from datetime import UTC, datetime
import json

import pytest

from living_memory import consolidation as c, decay, storage
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.storage import MemoryStore


SCOPE = "project:harbor"


def stamp(hour):
    return f"2026-10-29T{hour:02d}:00:00Z"


def moment(hour):
    return datetime(2026, 10, 29, hour, tzinfo=UTC)


@pytest.fixture
def clock(monkeypatch):
    hour = [9]
    monkeypatch.setattr(c, "_utc_now", lambda: stamp(hour[0]))
    monkeypatch.setattr(storage, "_utc_now", lambda: stamp(hour[0]))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment(hour[0])

    monkeypatch.setattr(decay, "datetime", Clock)
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv(decay.CONCEPT_COVERED_ENV, "0")
    monkeypatch.delenv(decay.TTL_ANCHOR_ENV, raising=False)
    monkeypatch.delenv(decay.MAX_PER_SWEEP_ENV, raising=False)
    return hour


@pytest.fixture
def store(tmp_path, clock):
    encoder = LocalEmbeddingModel(backend="hash", dimensions=3)
    with MemoryStore(tmp_path / "memory.sqlite3", chunk_embedder=encoder.embed_batch) as opened:
        yield opened


def source(store, index, *, kind="cluster", scope=SCOPE):
    context = {"scope": scope, "agent": f"inspector-{index}"}
    if kind != "cluster":
        context["procedure_id"] = "harbor_departure"
    if kind == "schema":
        context["step_order"] = index + 1
    return store.create_node(
        level="trace", content=f"Harbor departure requires manifest inspection step {index}.",
        context=context, embedding=[1., 0., 0.],
    )


def consolidate(store, kind="cluster", *, scope=SCOPE, recent_limit=10000):
    return c.memory_consolidate(
        store, scope=scope, min_cluster_size=1 if kind == "cluster" else 100,
        recent_limit=recent_limit,
    )


def setup_owner(store, clock, kind):
    traces = [source(store, i, kind=kind) for i in range(3)]
    clock[0] = 10
    result = consolidate(store, kind)
    owners = result.concepts_created if kind == "cluster" else result.schemas_created
    owner, = owners
    assert owner.source_traces == sorted(trace.id for trace in traces)
    assert owner.provenance["source_coverage"] == {trace.id: stamp(10) for trace in traces}
    return traces, owner


@pytest.mark.parametrize("kind", ["cluster", "schema", "evidence"])
@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("opt_in", [False, True])
def test_read_sources_survive_direct_decay_and_identical_consolidation(
    store, clock, monkeypatch, kind, repeat, opt_in,
):
    traces, owner = setup_owner(store, clock, kind)
    clock[0] = 11
    for trace in traces:
        store.record_access(trace.id)
    clock[0] = 12
    monkeypatch.setenv(decay.CONCEPT_COVERED_ENV, "1" if opt_in else "0")
    retired = consolidate(store, kind).decayed if repeat else decay.apply_decay(store).nodes

    assert retired == []
    current = store.get_node(owner.id)
    assert current.content == owner.content
    assert current.source_traces == owner.source_traces
    assert current.provenance["consolidated_at"] == stamp(12 if repeat else 10)
    assert decay.concept_coverage(store) == {trace.id: moment(10) for trace in traces}
    assert all(store.get_node(trace.id).last_accessed == stamp(11) for trace in traces)
    with MemoryStore(store.db_path) as reopened:
        assert decay.concept_coverage(reopened) == decay.concept_coverage(store)


@pytest.mark.parametrize("kind", ["cluster", "schema", "evidence"])
def test_expansion_keeps_old_sources_and_dates_only_the_new_input(store, clock, monkeypatch, kind):
    traces, owner = setup_owner(store, clock, kind)
    clock[0] = 11
    added = source(store, 3, kind=kind)
    for trace in [*traces, added]:
        store.record_access(trace.id)
    clock[0] = 12
    # The owner retains sources outside the visited prefix.
    consolidate(store, kind, recent_limit=1)
    expected = {**{trace.id: stamp(10) for trace in traces}, added.id: stamp(12)}
    assert store.get_node(owner.id).provenance["source_coverage"] == expected

    monkeypatch.setenv(decay.CONCEPT_COVERED_ENV, "1")
    retired = decay.apply_decay(store).expired
    assert [(node.id, node.decay_reason) for node in retired] == [(added.id, decay.CONCEPT_COVERED_REASON)]
    assert all(not store.get_node(trace.id).decayed for trace in traces)


@pytest.mark.parametrize("kind", ["cluster", "schema", "evidence"])
@pytest.mark.parametrize("change", ["teach", "supersede", "forget", "missing"])
def test_current_membership_drops_old_age_and_dates_replacements(store, clock, kind, change):
    traces, owner = setup_owner(store, clock, kind)
    clock[0] = 12
    replacement = None
    if change == "teach":
        replacement = c.memory_teach(store, traces[0].id, "Harbor departure checks a signed manifest.").corrective_trace
    elif change == "supersede":
        replacement = source(store, 3, kind=kind)
        store.create_connection(replacement.id, traces[0].id, "supersedes")
        consolidate(store, kind)
    elif change == "forget":
        c.memory_forget(store, traces[0].id)
    else:
        # A missing citation is valid old provenance, not a fabricated live row.
        previous = dict(owner.provenance)
        previous["source_traces"] = [*owner.source_traces, "absent-source"]
        previous["source_coverage"] = {**previous["source_coverage"], "absent-source": stamp(8)}
        store.update_node(owner.id, provenance=previous)
        consolidate(store, kind)

    retained = traces if change == "missing" else traces[1:]
    expected = {trace.id: stamp(10) for trace in retained}
    if replacement:
        expected[replacement.id] = stamp(12)
    current = store.get_node(owner.id)
    assert current.provenance["source_coverage"] == expected
    assert set(current.source_traces) == set(expected)
    clock[0] = 13
    consolidate(store, kind)
    assert store.get_node(owner.id).provenance["source_coverage"] == expected


@pytest.mark.parametrize("kind", ["cluster", "schema", "evidence"])
def test_readmission_starts_a_new_coverage_interval(store, clock, kind):
    traces, owner = setup_owner(store, clock, kind)
    clock[0] = 11
    c.memory_forget(store, traces[0].id)
    assert traces[0].id not in store.get_node(owner.id).provenance["source_coverage"]
    # Operator restoration of a forgotten trace, in a disposable fixture.
    with store.transaction():
        store.connection.execute("UPDATE nodes SET decayed=0, decay_reason=NULL WHERE id=?", (traces[0].id,))
    clock[0] = 12
    consolidate(store, kind)
    assert store.get_node(owner.id).provenance["source_coverage"] == {
        traces[0].id: stamp(12), **{trace.id: stamp(10) for trace in traces[1:]},
    }


@pytest.mark.parametrize("kind", ["cluster", "schema", "evidence"])
@pytest.mark.parametrize("history", ["legacy", "partial", "malformed"])
def test_unknown_existing_age_survives_maintenance_and_does_not_age_new_sources(
    store, clock, monkeypatch, kind, history,
):
    traces, owner = setup_owner(store, clock, kind)
    provenance = dict(owner.provenance)
    provenance.pop("source_coverage")
    if history == "partial":
        provenance["source_coverage"] = {traces[0].id: stamp(10), traces[1].id: "invalid"}
    elif history == "malformed":
        provenance["source_coverage"] = [stamp(10)]
    store.update_node(owner.id, provenance=provenance)
    # These reads precede consolidation; unknown age must not invent an order.
    for trace in traces:
        store.update_node(trace.id, stats={"last_accessed": stamp(9)})
    clock[0] = 11
    added = source(store, 3, kind=kind)
    clock[0] = 12
    consolidate(store, kind)
    expected = {trace.id: None for trace in traces}
    if history == "partial":
        expected[traces[0].id] = stamp(10)
    expected[added.id] = stamp(12)
    assert store.get_node(owner.id).provenance["source_coverage"] == expected
    clock[0] = 13
    consolidate(store, kind)
    assert store.get_node(owner.id).provenance["source_coverage"] == expected
    monkeypatch.setenv(decay.CONCEPT_COVERED_ENV, "1")
    assert {node.id for node in decay.apply_decay(store).expired} == (
        {added.id, traces[0].id} if history == "partial" else {added.id}
    )


def test_promotion_preserves_its_own_coverage_through_repeat_expansion_and_teach(store, clock):
    scopes = [f"project:harbor-{i}" for i in range(3)]
    groups, local_owners = [], []
    for i, scope in enumerate(scopes):
        groups.append([source(store, j, scope=scope) for j in range(3)])
        clock[0] = 10 + i
        result = consolidate(store, scope=scope)
        local_owners.extend(result.concepts_created)
    owner, = store.list_nodes(level="concept", scope="global")
    original_ids = {trace.id for group in groups for trace in group}
    assert owner.provenance["source_coverage"] == dict.fromkeys(original_ids, stamp(12))
    assert decay.concept_coverage(store)[groups[0][0].id] == moment(10)
    clock[0] = 13
    consolidate(store, scope=scopes[0])
    assert store.get_node(owner.id).provenance["source_coverage"] == owner.provenance["source_coverage"]
    added = source(store, 3, scope=scopes[0])
    clock[0] = 14
    consolidate(store, scope=scopes[0])
    expected = {**dict.fromkeys(original_ids, stamp(12)), added.id: stamp(14)}
    assert store.get_node(owner.id).provenance["source_coverage"] == expected
    clock[0] = 15
    taught = c.memory_teach(store, groups[0][0].id, "Harbor departure requires a signed manifest.")
    expected.pop(groups[0][0].id)
    expected[taught.corrective_trace.id] = stamp(15)
    assert store.get_node(owner.id).provenance["source_coverage"] == expected
    assert decay.concept_coverage(store)[taught.corrective_trace.id] == moment(15)
    clock[0] = 16
    c.memory_forget(store, added.id)
    expected.pop(added.id)
    assert store.get_node(owner.id).provenance["source_coverage"] == expected


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("unknown", [False, True])
def test_multiple_live_owners_use_earliest_or_unknown_age(store, clock, reverse, unknown):
    trace = source(store, 0)
    store.update_node(trace.id, stats={"last_accessed": stamp(9 if unknown else 11)})
    owners = [("concept", stamp(10)), ("schema", None if unknown else stamp(12))]
    for level, start in reversed(owners) if reverse else owners:
        store.create_node(
            level=level, content=f"Independent {level} covers a source.",
            context={"scope": "project:another-owner"},
            provenance={"source_traces": [trace.id], "source_coverage": {trace.id: start}},
        )
    assert decay.concept_coverage(store) == {trace.id: None if unknown else moment(10)}
    assert decay.apply_decay(store, scope=SCOPE, decay_concept_covered=True).nodes == []


@pytest.mark.parametrize("provenance", [
    None, [], {}, {"consolidated_at": stamp(10)}, {"source_coverage": []},
    {"source_coverage": {"trace": "invalid"}}, {"source_coverage": {"trace": [stamp(10)]}},
])
def test_consumer_never_infers_age_from_owner_dates(store, provenance):
    trace = store.create_node(level="trace", node_id="trace", content="A source with a recorded read.")
    store.update_node(trace.id, stats={"last_accessed": stamp(8)})
    owner = store.create_node(level="concept", content="A legacy digest.", provenance={"source_traces": [trace.id]})
    with store.transaction():
        store.connection.execute("UPDATE nodes SET provenance=? WHERE id=?", (json.dumps(provenance), owner.id))
    assert decay.concept_coverage(store) == {trace.id: None}
    assert decay.apply_decay(store, decay_concept_covered=True).nodes == []


@pytest.mark.parametrize("read_hour,retired", [(None, True), (9, True), (10, True), (11, False)])
def test_access_boundary_is_strictly_after_coverage(store, read_hour, retired):
    trace = source(store, 0)
    if read_hour is not None:
        store.update_node(trace.id, stats={"last_accessed": stamp(read_hour)})
    store.create_node(
        level="schema", content="Inspection procedure.",
        provenance={"source_traces": [trace.id], "source_coverage": {trace.id: stamp(10)}},
    )
    result = decay.apply_decay(store, decay_concept_covered=True)
    assert [node.id for node in result.expired] == ([trace.id] if retired else [])
