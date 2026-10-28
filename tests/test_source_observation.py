"""Shared current inputs keep read order, validity and atomic writer outcomes."""

from collections import Counter

import pytest

from living_memory import consolidation as c, storage
from living_memory.storage import MemoryStore
from test_derivative_preparation import sources


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    with MemoryStore(tmp_path / "memory.sqlite3") as instance:
        yield instance


def derived(store, name, members, *, strategy="token-jaccard", scope="project:harbor"):
    provenance = {"strategy": strategy, "cluster_key": name,
                  "source_traces": [member.id for member in members]}
    if strategy == "cross-scope-promotion":
        provenance["source_concepts"] = [member.id for member in members]
    return store.create_node(
        level="concept", content=f"Harbor {name} records.",
        context={"scope": scope, "procedure_id": "harbor inspection"},
        provenance=provenance,
    )


def test_union_preserves_each_callers_sql_order_across_batches(store):
    with store.transaction():
        nodes = [store.append_trace(f"Harbor inspection record {i}.") for i in range(2017)]
    ids = [node.id for node in nodes]
    groups = [ids[:1501], ids[401:], ids[::2] + [ids[0], "missing"], list(reversed(ids))]
    observation = c._SourceObservation(store, (source for group in groups for source in group))
    for group in groups:
        # Direct SQL reproduces the store's bounded IN reads independently of
        # the shared reader's projection and the explicit batch-order helper.
        unique = list(set(group))
        expected = []
        for start in range(0, len(unique), 999):
            batch = unique[start:start + 999]
            expected.extend(row["id"] for row in store.connection.execute(
                f"SELECT * FROM nodes WHERE id IN ({','.join('?' for _ in batch)})", batch,
            ))
        assert list(store.get_nodes(group)) == expected
        observed = observation.get_nodes(group)
        assert list(observed) == expected
        assert observed == store.get_nodes(group)


@pytest.mark.parametrize("strategy", ["token-jaccard", "procedural"])
@pytest.mark.parametrize("prepared", [False, True])
def test_real_writers_decode_shared_leaves_once_per_phase(store, monkeypatch, strategy, prepared):
    traces = sources(store)
    owners = [derived(store, name, traces, strategy=strategy) for name in ("left", "right")]
    shared_ids = {trace.id for trace in traces[1:]}
    hydrated = Counter()
    decode = storage._node_from_row

    def counted(row):
        if row["id"] in shared_ids:
            hydrated[row["id"]] += 1
        return decode(row)

    monkeypatch.setattr(storage, "_node_from_row", counted)
    correction = "Harbor crane requires the signed electronic manifest."
    preparation = c.prepare_teach(store, traces[0].id, correction) if prepared else None
    assert hydrated == ({source: 1 for source in shared_ids}
                        if prepared and strategy == "token-jaccard" else {})
    hydrated.clear()
    result = c.memory_teach(store, traces[0].id, correction, preparation=preparation)
    assert hydrated == {source: 1 for source in shared_ids}
    for owner in owners:
        current = store.get_node(owner.id)
        assert not current.decayed
        assert correction in current.content and traces[0].content not in current.content
        assert set(current.source_traces) == shared_ids | {result.corrective_trace.id}


def test_shared_resolution_preserves_lineage_displacement_and_validity(store):
    traces = sources(store)
    rejected = derived(store, "rejected", traces[:2])
    replacement = store.append_trace("Harbor inspection uses the calibrated sensor.")
    latest = store.append_trace("Harbor inspection requires two calibrated sensors.")
    store.create_connection(replacement.id, rejected.id, "supersedes")
    store.create_connection(latest.id, replacement.id, "supersedes")
    store.soft_delete_node(traces[2].id, "expired observation")
    corrected = store.append_trace("Harbor inspector records the obsolete limit.")
    store.add_correction(corrected.id, old=corrected.content, new="New limit.", by="inspector")
    groups = [
        [rejected.id, *[trace.id for trace in traces], corrected.id, "missing"],
        [replacement.id, rejected.id, traces[0].id],
        [latest.id, corrected.id],
    ]
    observation = c._SourceObservation(store, (source for group in groups for source in group))
    for group in groups:
        assert c._current_sources(store, group, observation=observation) == c._current_sources(store, group)


def test_recorded_related_inputs_are_shared_in_store_order(store, monkeypatch):
    traces = sources(store)
    rejected = store.create_node(level="concept", content="Harbor inspection records.")
    replacement = store.append_trace("Harbor inspection uses calibrated sensors.")
    incidental = store.append_trace("Harbor inspection schedule.")
    for index, (trace, weight) in enumerate(zip(traces, (0.6, 0.8, 0.8))):
        monkeypatch.setattr(storage, "_utc_now", lambda i=index: f"2026-01-0{i + 1}T00:00:00Z")
        store.create_connection(rejected.id, trace.id, "related", weight=weight,
                                metadata={"kind": c.DERIVED_FROM_KIND})
    store.create_connection(rejected.id, incidental.id, "related", weight=1.0)
    store.create_connection(replacement.id, rejected.id, "supersedes")
    owners = [derived(store, name, [replacement, *traces, incidental]) for name in ("left", "right")]
    expected = [c._current_derivation(store, owner) for owner in owners]
    expected_inputs = c._current_derivation(store, rejected)
    queries = Counter()
    read = store.list_connections

    def counted(**kwargs):
        queries[tuple(sorted(kwargs.items()))] += 1
        return read(**kwargs)

    monkeypatch.setattr(store, "list_connections", counted)
    observation = c._SourceObservation(store, [replacement.id, incidental.id, *[t.id for t in traces]])
    for _ in range(2):
        assert c._derivation_sources(observation, rejected) == [trace.id for trace in reversed(traces)]
        assert c._current_derivation(store, rejected, observation=observation) == expected_inputs
        assert [c._current_derivation(store, owner, observation=observation) for owner in owners] == expected
    related_reads = sum(count for query, count in queries.items()
                        if dict(query).get("relation_type") == "related")
    assert related_reads == 1
    assert {node.id for node in expected[0].live} == {replacement.id, incidental.id}


@pytest.mark.parametrize("strategy", ["token-jaccard", "procedural"])
def test_source_first_writers_refresh_related_reads_for_owners_and_inputs(store, monkeypatch, strategy):
    traces = sources(store)
    for trace in traces:
        store.record_access(trace.id)
    left = derived(store, "left", traces, strategy=strategy)
    independent = derived(store, "independent", traces[1:])
    promotion = derived(store, "promotion", [left, independent],
                        strategy="cross-scope-promotion", scope="global")
    initialize = c._SourceObservation.__init__
    update = c._SourceObservation.updated
    changed = {}

    def prime(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        for source_id in self.nodes.keys() | self.changing:
            self.list_connections(source_id=source_id, relation_type="related")

    def updated(self, node):
        inputs = {node.id, *node.materializer_sources}
        before = {source: self.list_connections(source_id=source, relation_type="related")
                  for source in inputs}
        current = {source: store.list_connections(source_id=source, relation_type="related")
                   for source in inputs}
        changed[node.id] = {source for source in inputs if before[source] != current[source]}
        update(self, node)
        for source in inputs:
            assert self.list_connections(source_id=source, relation_type="related") == current[source]

    monkeypatch.setattr(c._SourceObservation, "__init__", prime)
    monkeypatch.setattr(c._SourceObservation, "updated", updated)
    taught = c.memory_teach(store, traces[0].id, "Harbor crane uses the electronic manifest.")
    assert left.id in changed[left.id]
    if strategy == "token-jaccard":
        assert changed[left.id] & {trace.id for trace in traces[1:]}
    assert changed[promotion.id] == {left.id, independent.id}
    assert store.get_node(independent.id) == independent
    assert set(store.get_node(left.id).source_traces) == {taught.corrective_trace.id, *[t.id for t in traces[1:]]}
    assert store.get_node(left.id).content in store.get_node(promotion.id).content


class SeparateReads:
    """Reference execution uses the store directly for every owner."""

    def __init__(self, store, source_ids, *, changing=(), refresh=None):
        self.store = store
        self.changing = set(changing)
        self.refresh = refresh

    def __getattr__(self, name):
        return getattr(self.store, name)

    def get_node(self, node_id):
        if node_id in self.changing and self.refresh is not None:
            self.refresh(node_id)
        return self.store.get_node(node_id)

    def get_nodes(self, source_ids):
        return {node_id: self.get_node(node_id) for node_id in self.store.get_nodes(source_ids)}

    def updated(self, node):
        self.changing.discard(node.id)


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("shape", ["diamond", "cycle", "lineage", "evidence"])
def test_complete_writer_outcome_matches_separate_current_reads(store, monkeypatch, shape, prepared):
    traces = sources(store)
    strategy = "procedural" if shape == "evidence" else "token-jaccard"
    left = derived(store, "left", traces[:2], strategy=strategy)
    right = derived(store, "right", traces, strategy=strategy)
    joined = derived(store, "joined", [left, right])
    derived(store, "promotion", [left, right], strategy="cross-scope-promotion", scope="global")
    derived(store, "independent", traces[1:])
    if shape == "cycle":
        store.update_node(left.id, provenance={
            **left.provenance, "source_traces": [traces[0].id, joined.id],
        })
    elif shape == "lineage":
        successor = store.append_trace("Harbor crane temporarily uses the interim permit.")
        store.create_connection(successor.id, traces[0].id, "supersedes")
        traces[0] = successor

    observation = c._SourceObservation
    outcomes = []

    class Rollback(Exception):
        pass

    for reader in (SeparateReads, observation):
        identifiers = iter(f"synthetic-id-{i:06}" for i in range(1000))
        monkeypatch.setattr(c, "_SourceObservation", reader)
        monkeypatch.setattr(c, "new_ulid", lambda: next(identifiers))
        monkeypatch.setattr(storage, "new_ulid", lambda: next(identifiers))
        monkeypatch.setattr(c, "_utc_now", lambda: "2026-01-02T00:00:00Z")
        monkeypatch.setattr(storage, "_utc_now", lambda: "2026-01-02T00:00:00Z")
        with pytest.raises(Rollback), store.transaction():
            correction = "Harbor crane requires the signed electronic manifest."
            preparation = c.prepare_teach(store, traces[0].id, correction) if prepared else None
            c.memory_teach(store, traces[0].id, correction, preparation=preparation)
            # Compare the complete durable result: digest, stats, provenance,
            # source unions, validity, edges, search indexes and correction.
            outcomes.append(list(store.connection.iterdump()))
            raise Rollback
    assert outcomes[0] == outcomes[1]


def order_consumers(store, monkeypatch, first):
    read = store.nodes_referencing_source
    monkeypatch.setattr(store, "nodes_referencing_source", lambda source: sorted(
        read(source), key=lambda node: (node.id != first.id, node.id),
    ))


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("consumer_first", [False, True])
@pytest.mark.parametrize("chain_length", [0, 2])
def test_cycle_decay_changes_a_shared_superseders_validity(
    store, monkeypatch, prepared, consumer_first, chain_length,
):
    traces = sources(store)
    cyclic = derived(store, "cyclic", [traces[0]])
    store.update_node(cyclic.id, provenance={
        **cyclic.provenance, "source_traces": [traces[0].id, cyclic.id],
    })
    fallback = traces[1]
    for index in range(chain_length):
        replacement = store.append_trace(f"Harbor inspection uses sensor {index}.")
        store.create_connection(replacement.id, fallback.id, "supersedes")
        fallback = replacement
    store.create_connection(cyclic.id, fallback.id, "supersedes")
    consumer = derived(store, "consumer", [traces[0], traces[1]])
    order_consumers(store, monkeypatch, consumer if consumer_first else cyclic)
    assert next(iter(c._affected_derivatives(store, traces[0].id))) == (
        consumer.id if consumer_first else cyclic.id
    )
    # Observe the replacement before the cycle decays it. Cached edges must
    # still use its current validity when the later consumer is refreshed.
    original = c._SourceObservation.__init__

    def observe(self, *args, **kwargs):
        original(self, *args, **kwargs)
        # Prime cached supersedes edges without triggering the writer before
        # construction has returned its observation to the traversal.
        assert self.list_connections(target_id=fallback.id, relation_type="supersedes")[0].source_id == cyclic.id

    monkeypatch.setattr(c._SourceObservation, "__init__", observe)
    correction = "Harbor crane uses the electronic manifest."
    preparation = c.prepare_teach(store, traces[0].id, correction) if prepared else None
    taught = c.memory_teach(store, traces[0].id, correction, preparation=preparation)
    with MemoryStore(store.db_path) as reader:
        assert reader.get_node(cyclic.id).decayed
        current = reader.get_node(consumer.id)
        assert not current.decayed
        assert set(current.source_traces) == {taught.corrective_trace.id, fallback.id}
        assert correction in current.content and fallback.content in current.content
        assert cyclic.content not in current.content and traces[0].content not in current.content
        assert reader.get_node(traces[1].id) == traces[1]


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("consumer_first", [False, True])
def test_replacement_chain_diamond_consumes_current_source_content(
    store, monkeypatch, prepared, consumer_first,
):
    traces = sources(store)
    predecessor = store.append_trace("Harbor inspection uses an earlier record.")
    interim = store.append_trace("Harbor inspection uses an interim record.")
    replacement = derived(store, "replacement", traces[:2])
    store.create_connection(interim.id, predecessor.id, "supersedes")
    store.create_connection(replacement.id, interim.id, "supersedes")
    consumers = [derived(store, name, [traces[0], predecessor, traces[2]])
                 for name in ("left", "right")]
    joined = derived(store, "joined", [*consumers, replacement])
    order_consumers(store, monkeypatch, consumers[0] if consumer_first else replacement)
    correction = "Harbor crane uses the electronic manifest."
    preparation = c.prepare_teach(store, traces[0].id, correction) if prepared else None
    taught = c.memory_teach(store, traces[0].id, correction, preparation=preparation)
    with MemoryStore(store.db_path) as reader:
        current_replacement = reader.get_node(replacement.id)
        assert not current_replacement.decayed
        assert set(current_replacement.source_traces) == {taught.corrective_trace.id, traces[1].id}
        for consumer in consumers:
            current = reader.get_node(consumer.id)
            assert not current.decayed
            assert set(current.source_traces) == {taught.corrective_trace.id, replacement.id, traces[2].id}
            assert all(text in current.content for text in (correction, traces[1].content, traces[2].content))
            assert replacement.content not in current.content
            assert traces[0].content not in current.content
        current_joined = reader.get_node(joined.id)
        assert not current_joined.decayed
        assert set(current_joined.source_traces) == {node.id for node in [*consumers, replacement]}
        assert all(text in current_joined.content for text in (correction, traces[1].content, traces[2].content))
        assert traces[0].content not in current_joined.content


@pytest.mark.parametrize("prepared", [False, True])
def test_replacement_cycle_decay_and_consumer_write_roll_back_together(store, monkeypatch, prepared):
    traces = sources(store)
    cyclic = derived(store, "cyclic", [traces[0]])
    store.update_node(cyclic.id, provenance={
        **cyclic.provenance, "source_traces": [traces[0].id, cyclic.id],
    })
    store.create_connection(cyclic.id, traces[1].id, "supersedes")
    consumer = derived(store, "consumer", traces[:2])
    order_consumers(store, monkeypatch, consumer)
    correction = "Harbor crane uses the electronic manifest."
    preparation = c.prepare_teach(store, traces[0].id, correction) if prepared else None
    before = list(store.connection.iterdump())
    write = c._refresh_derived_node

    def fail(store, node, **kwargs):
        result = write(store, node, **kwargs)
        if node.id == consumer.id:
            assert store.get_node(cyclic.id).decayed
            raise RuntimeError("consumer write failed")
        return result

    monkeypatch.setattr(c, "_refresh_derived_node", fail)
    with pytest.raises(RuntimeError, match="consumer write failed"):
        c.memory_teach(store, traces[0].id, correction, preparation=preparation)
    assert list(store.connection.iterdump()) == before
    assert not store.connection.in_transaction
