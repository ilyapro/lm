"""Promotion counts current original authors across overlapping derivations."""

import pytest

from living_memory import consolidation as c
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore


TOPIC = "Database timeout connection pool retry budget requires bounded attempts"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    with MemoryStore(tmp_path / "memory.sqlite3") as opened:
        yield opened


def append_scope(store, scope, authors):
    traces = [store.append_trace(
        f"{TOPIC}; observation {index}.", {"scope": scope, "agent": author},
        feedback={"confidence": .9, "usefulness_score": .2},
    ) for index, author in enumerate(authors)]
    c.memory_consolidate(store, scope=scope, force=True, min_cluster_size=3)
    return traces


def assert_recalled_consensus(store, promoted, count):
    recalled = MemoryRecallService(store).memory_recall(
        TOPIC, scope="global", max_results=3, log_access=False, log_event=False,
    )
    delivered, = [result.node for result in recalled if result.node.id == promoted.id]
    assert delivered.unique_agents == count
    assert delivered.confidence == promoted.confidence


@pytest.mark.parametrize("authors,count", [
    ([("A", "A", "A")] * 3, 1),
    ([("A", "A", "A"), ("B", "B", "B"), ("C", "C", "C")], 3),
    ([("A", "B", "A"), ("B", "C", "B"), ("A", "B", "A")], 3),
])
def test_public_consolidation_and_recall_union_authors_through_repeat_and_expansion(store, authors, count):
    traces = []
    for index, writers in enumerate(authors):
        traces.extend(append_scope(store, f"project:scope-{index}", writers))
    original, = store.list_nodes(level="concept", scope="global")
    assert original.unique_agents == count
    assert original.confidence == .5 if count == 1 else original.confidence > .5
    assert set(original.source_traces) == {trace.id for trace in traces}
    assert set(original.provenance["source_coverage"]) == set(original.source_traces)
    assert_recalled_consensus(store, original, count)

    c.memory_consolidate(store, scope="project:scope-2", force=True, min_cluster_size=3)
    repeated, = store.list_nodes(level="concept", scope="global")
    assert repeated.id == original.id
    assert repeated.unique_agents == count
    assert repeated.confidence == original.confidence
    assert repeated.provenance["source_coverage"] == original.provenance["source_coverage"]
    assert_recalled_consensus(store, repeated, count)

    traces.extend(append_scope(store, "project:scope-3", authors[0]))
    expanded, = store.list_nodes(level="concept", scope="global")
    assert expanded.id == original.id
    assert expanded.unique_agents == count
    assert expanded.confidence == .5 if count == 1 else expanded.confidence > original.confidence
    assert set(expanded.source_traces) == {trace.id for trace in traces}
    assert_recalled_consensus(store, expanded, count)


def project_concept(store, scope, source_ids, *, count=1, strategy="embedding-cosine"):
    return store.create_node(
        level="concept", content=f"{TOPIC}. Evidence from {scope}.",
        context={"scope": scope, "agent": "memory_consolidate"}, embedding=[1., 0.],
        stats={"unique_agents": count, "confidence": .8},
        provenance={"strategy": strategy, "source_traces": source_ids},
    )


@pytest.mark.parametrize("before_promotion", [False, True])
def test_original_identities_follow_correction_and_decay_in_nested_sources(store, before_promotion):
    a = store.append_trace(f"{TOPIC}. Keep the guard.", {"agent": "A"})
    b = store.append_trace(f"{TOPIC}. Permit unbounded retries.", {"agent": "B"})
    obsolete = store.append_trace(f"{TOPIC}. Use the obsolete timer.", {"agent": "C"})
    shared = project_concept(store, "project:shared", [a.id, b.id, obsolete.id], count=3)
    projects = [project_concept(store, f"project:scope-{i}", [shared.id], count=3) for i in range(3)]
    promoted = None
    if not before_promotion:
        promoted, = c._cross_scope_promotion(store, projects[0], candidates=projects[1:])
        assert promoted.unique_agents == 3
    correction = f"{TOPIC}. Do not permit unbounded retries."
    taught = c.memory_teach(store, b.id, correction, context={"agent": "A"})
    c.memory_forget(store, obsolete.id, "obsolete evidence")
    if promoted is None:
        promoted, = c._cross_scope_promotion(store, projects[0], candidates=projects[1:])
    current = store.get_node(promoted.id)
    assert current.unique_agents == 1
    assert current.confidence == .5
    assert correction in current.content
    assert obsolete.content not in current.content
    assert store.get_node(b.id).corrections
    assert store.get_node(obsolete.id).decayed
    assert store.list_connections(source_id=taught.corrective_trace.id, relation_type="supersedes")
    c.memory_consolidate(store, scope="project:unrelated", force=True)
    refreshed = store.get_node(promoted.id)
    assert refreshed.unique_agents == 1 and refreshed.confidence == .5
    assert refreshed.source_traces == current.source_traces


@pytest.mark.parametrize("declared,known,expected", [
    (0, (), 1), (1, (), 1), (2, (), 2),
    (2, ("A",), 2), (2, ("A", "B", "C"), 3),
    (5, ("A", "B"), 5),
])
def test_missing_legacy_identities_retain_maximum_support_without_assuming_disjointness(
    store, declared, known, expected,
):
    traces = [store.append_trace(f"{TOPIC}. Source {i}.", {"agent": author})
              for i, author in enumerate(known)]
    projects = [project_concept(
        store, f"project:scope-{i}", [*[trace.id for trace in traces], f"unavailable-{i}"],
        count=declared,
    ) for i in range(3)]
    promoted, = c._cross_scope_promotion(store, projects[0])
    assert promoted.unique_agents == expected
    c.memory_consolidate(store, scope="project:unrelated", force=True)
    current = store.get_node(promoted.id)
    assert current.unique_agents == expected
    assert current.source_traces == promoted.source_traces
    assert current.confidence == .5 if expected == 1 else current.confidence > .5


def test_available_identities_replace_legacy_bounds_and_ignore_materializers(store):
    a = store.append_trace(f"{TOPIC}. Original evidence.", {"agent": "A"})
    projects = [project_concept(store, f"project:scope-{i}", [a.id, "unavailable"], count=4)
                for i in range(3)]
    promoted, = c._cross_scope_promotion(store, projects[0])
    assert promoted.unique_agents == 4
    store.append_trace(f"{TOPIC}. Recovered evidence.", {"agent": "A"}, node_id="unavailable")
    c.memory_consolidate(store, scope="project:unrelated", force=True)
    current = store.get_node(promoted.id)
    assert current.unique_agents == 1 and current.confidence == .5


@pytest.mark.parametrize("strategy", [None, "embedding-cosine"])
def test_source_free_materializer_identity_does_not_add_an_author(store, strategy):
    a = store.append_trace(f"{TOPIC}. Original evidence.", {"agent": "A"})
    projects = [project_concept(store, "project:known", [a.id])]
    projects.extend(project_concept(store, f"project:scope-{i}", [], strategy=strategy) for i in range(2))
    promoted, = c._cross_scope_promotion(store, projects[0])
    assert promoted.unique_agents == 1 and promoted.confidence == .5


def test_declared_leaf_support_is_a_maximum_not_a_sum(store):
    traces = [store.append_trace(
        f"{TOPIC}. Observation {i}.", {"agent": author}, feedback={"unique_agents": 2},
    ) for i, author in enumerate(("A", "B", "C"))]
    projects = [project_concept(store, f"project:scope-{i}", [traces[i].id, traces[(i + 1) % 3].id], count=2)
                for i in range(3)]
    promoted, = c._cross_scope_promotion(store, projects[0])
    assert promoted.unique_agents == 3


def test_current_leaf_validity_overrides_stale_derived_counts(store):
    traces = [store.append_trace(f"{TOPIC}. Observation {i}.", {"agent": author})
              for i, author in enumerate(("A", "B", "C"))]
    projects = [project_concept(store, f"project:scope-{i}", [trace.id for trace in traces], count=3)
                for i in range(3)]
    promoted, = c._cross_scope_promotion(store, projects[0])
    assert promoted.unique_agents == 3
    store.soft_delete_node(traces[1].id, "obsolete source")
    store.add_correction(traces[2].id, old=traces[2].content, new="Unsupported.", by="A")
    c.memory_consolidate(store, scope="project:unrelated", force=True)
    current = store.get_node(promoted.id)
    assert current.unique_agents == 1 and current.confidence == .5


def test_original_trace_recall_citations_do_not_supply_authors(store):
    cited = store.append_trace("Unrelated recalled evidence.", {"agent": "other-author"})
    original = store.create_node(
        level="trace", content=f"{TOPIC}. Original evidence.", context={"agent": "A"},
        provenance={"source_traces": [cited.id]},
    )
    projects = [project_concept(store, f"project:scope-{i}", [original.id]) for i in range(3)]
    promoted, = c._cross_scope_promotion(store, projects[0])
    assert promoted.unique_agents == 1 and promoted.confidence == .5
