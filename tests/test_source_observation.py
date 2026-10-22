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
            expected.extend(row[0] for row in store.connection.execute(
                f"SELECT id FROM nodes WHERE id IN ({','.join('?' for _ in batch)})", batch,
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


class SeparateReads:
    """Reference execution uses the store directly for every owner."""

    def __init__(self, store, source_ids, *, changing=()):
        self.store = store

    def __getattr__(self, name):
        return getattr(self.store, name)

    def updated(self, node):
        pass


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


def test_cycle_decay_changes_a_shared_superseders_validity(store, monkeypatch):
    traces = sources(store)
    cyclic = derived(store, "cyclic", [traces[0]])
    store.update_node(cyclic.id, provenance={
        **cyclic.provenance, "source_traces": [traces[0].id, cyclic.id],
    })
    store.create_connection(cyclic.id, traces[1].id, "supersedes")
    consumer = derived(store, "consumer", [traces[0], traces[1]])
    # Observe the replacement before the cycle decays it. Cached edges must
    # still use its current validity when the later consumer is refreshed.
    original = c._SourceObservation.__init__

    def observe(self, *args, **kwargs):
        original(self, *args, **kwargs)
        assert c.resolve_replacement(self, traces[1].id) == cyclic.id

    monkeypatch.setattr(c._SourceObservation, "__init__", observe)
    taught = c.memory_teach(store, traces[0].id, "Harbor crane uses the electronic manifest.")
    assert store.get_node(cyclic.id).decayed
    assert set(store.get_node(consumer.id).source_traces) == {taught.corrective_trace.id, traces[1].id}
