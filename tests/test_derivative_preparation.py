"""Derivative refresh preserves its outcome without repeated source preparation."""

from collections import Counter

import pytest

from living_memory import consolidation as c
from living_memory.models import Node
from living_memory.storage import MemoryStore
from living_memory.source_text import passage_spans


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    with MemoryStore(tmp_path / "memory.sqlite3") as instance:
        yield instance


def cluster(store, traces):
    group = c._TraceCluster("project:harbor", traces, Counter({"harbor": len(traces)}))
    (node, _), = c._materialize_cluster(store, group, min_cluster_size=1)
    return node


def sources(store):
    return [
        store.append_trace(text, {"scope": "project:harbor", "agent": "engineer"})
        for text in (
            "Harbor crane requires the paper permit.",
            "Harbor crane inspection includes the brake assembly.",
            "Harbor crane maintenance records the cable tension.",
        )
    ]


def test_teach_prepares_each_current_cluster_once(store, monkeypatch):
    traces = sources(store)
    concept = cluster(store, traces)
    counts = Counter()
    for name in ("_current_derivation", "_synthesize_digest"):
        original = getattr(c, name)

        def observe(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(c, name, observe)
    correction = "Harbor crane requires the signed electronic manifest."
    taught = c.memory_teach(store, traces[0].id, correction)
    current = store.get_node(concept.id)
    assert correction in current.content
    assert traces[0].content not in current.content
    assert set(current.source_traces) == {taught.corrective_trace.id, traces[1].id, traces[2].id}
    assert counts == {"_current_derivation": 1, "_synthesize_digest": 1}
    assert any(edge.target_id == taught.corrective_trace.id for edge in store.list_connections(
        source_id=concept.id, relation_type="related"
    ))


def test_cluster_edge_failure_rolls_back_the_entire_correction(store, monkeypatch):
    traces = sources(store)
    cluster(store, traces)
    before = list(store.connection.iterdump())
    update = c._update_edge_weights_from_co_access

    def fail_after_edges(*args):
        update(*args)
        raise RuntimeError("required derived edge failure")

    monkeypatch.setattr(c, "_update_edge_weights_from_co_access", fail_after_edges)
    with pytest.raises(RuntimeError, match="required derived edge failure"):
        c.memory_teach(store, traces[0].id, "Harbor crane uses an electronic manifest.")
    assert list(store.connection.iterdump()) == before
    assert not store.connection.in_transaction


def test_connections_validate_identity_without_loading_endpoint_documents(store, monkeypatch):
    traces = sources(store)
    concept = cluster(store, traces)
    store.soft_delete_node(traces[0].id, "historical source")

    def no_documents(*args):
        raise AssertionError("edge existence checks must not load node documents")

    monkeypatch.setattr(store, "get_node", no_documents)
    edge = store.create_connection(concept.id, traces[0].id, "related", weight=0.75,
                                   metadata={"basis": "cluster"})
    assert edge.source_id == concept.id and edge.target_id == traces[0].id
    assert edge.weight == 0.75 and edge.metadata == {"basis": "cluster"}
    updated = store.create_connection(concept.id, traces[0].id, "related", weight=0.8)
    assert updated.id == edge.id and updated.weight == 0.8
    assert store.create_connection(concept.id, concept.id, "related").target_id == concept.id
    for source, target, missing in (("absent-source", "absent-target", "absent-source"),
                                    (concept.id, "absent-target", "absent-target")):
        with pytest.raises(KeyError) as error:
            store.create_connection(source, target, "related")
        assert error.value.args == (missing,)
    with pytest.raises(ValueError):
        store.create_connection("absent-source", "absent-target", "invalid")


@pytest.mark.parametrize("content", [
    " ".join(f"lexeme{i}" for i in range(8)) + ".\n"
    + " ".join(f"lexeme{i}" for i in range(8, 16)),
    "Ёлка _cranes_ 12 а calibration.\nКраны проверены; loading loaded loaders! final_step",
    "Repeated crane inspection. Repeated crane inspection! cable tension? cable tension",
    "\n\t __ 123 и the \n. ! ; \n",
])
def test_preparation_keeps_whole_source_token_order_and_passage_signals(content):
    source = Node("source", "trace", content)
    prepared, = c._prepare_digest_sources([source])
    expected = c._significant_tokens(content)
    assert list(prepared.tokens) == list(expected)
    assert Counter(prepared.tokens).most_common(12) == Counter(expected).most_common(12)
    assert prepared.passages == [
        (content[start:end], c._significant_tokens(content[start:end]))
        for start, end, _ in passage_spans(content)
    ]


def test_existing_cluster_renders_only_its_current_source_union(store, monkeypatch):
    traces = sources(store)
    concept = cluster(store, traces)
    group = c._TraceCluster("project:harbor", traces[:2], Counter({"harbor": 2}))
    rendered = []
    render = c._synthesize_digest

    def observe(members, **kwargs):
        rendered.append({member.id for member in members})
        return render(members, **kwargs)

    monkeypatch.setattr(c, "_synthesize_digest", observe)
    (updated, created), = c._materialize_cluster(store, group, min_cluster_size=1)
    assert not created and updated.id == concept.id
    assert set(updated.source_traces) == {trace.id for trace in traces}
    assert rendered == [{trace.id for trace in traces}]


def test_shared_leaves_prepare_once_and_diamond_consumes_updated_concepts(store, monkeypatch):
    traces = sources(store)

    def derived(name, members):
        return store.create_node(
            level="concept", content=c._synthesize_digest(members, best=members[0]),
            context={"scope": "project:harbor"},
            provenance={"strategy": "token-jaccard", "cluster_key": name,
                        "source_traces": [member.id for member in members]},
        )

    left = derived("left", traces[:2])
    right = derived("right", [traces[0], traces[2]])
    joined = derived("joined", [left, right])
    prepared = []
    prepare = c._prepare_digest_sources

    def observe(members):
        prepared.extend((member.id, member.content) for member in members)
        return prepare(members)

    monkeypatch.setattr(c, "_prepare_digest_sources", observe)
    correction = "Harbor crane requires the signed electronic manifest."
    taught = c.memory_teach(store, traces[0].id, correction)
    counts = Counter(node_id for node_id, _ in prepared)
    assert counts == {taught.corrective_trace.id: 1, traces[1].id: 1, traces[2].id: 1,
                      left.id: 1, right.id: 1}
    for node in (left, right, joined):
        current = store.get_node(node.id)
        assert correction in current.content and traces[0].content not in current.content
    assert set(store.get_node(joined.id).source_traces) == {left.id, right.id}
    assert dict(prepared)[left.id] == store.get_node(left.id).content
    assert dict(prepared)[right.id] == store.get_node(right.id).content


@pytest.mark.parametrize("materializer", ["cluster", "cross-scope-promotion"])
def test_overlapping_owners_share_text_preparation_only_within_one_operation(store, monkeypatch, materializer):
    if materializer == "cluster":
        members = sources(store)
        groups = [members[:2], members[1:]]
        scope = "project:harbor"
        provenance_key = "source_traces"
        strategy = "token-jaccard"

        def materialize():
            group = c._TraceCluster(scope, members, Counter())
            return [node for node, _ in c._materialize_cluster(store, group, min_cluster_size=1)]
    else:
        members = [store.create_node(
            level="concept", content=text, context={"scope": f"project:harbor-{i}"},
            embedding=[1.0, 0.0],
        ) for i, text in enumerate([
            "Harbor crane checks the paper permit.",
            "Harbor crane inspects the brake assembly.",
            "Harbor crane measures cable tension.",
            "Harbor crane records the departure time.",
        ])]
        groups = [members[:3], members[1:]]
        scope = "global"
        provenance_key = "source_concepts"
        strategy = materializer

        def materialize():
            return c._cross_scope_promotion(store, members[0], phase_number=4)

    owners = [store.create_node(
        level="concept", content=f"Harbor owner {i} awaiting materialization.",
        context={"scope": scope},
        provenance={"strategy": strategy, provenance_key: [member.id for member in group]},
    ) for i, group in enumerate(groups)]
    prepared = Counter()
    prepare = c._prepare_digest_sources

    def observe(members):
        prepared.update(member.id for member in members)
        return prepare(members)

    monkeypatch.setattr(c, "_prepare_digest_sources", observe)
    assert {node.id for node in materialize()} == {owner.id for owner in owners}
    assert prepared == {member.id: 1 for member in members}
    for owner, group in zip(owners, groups):
        current = store.get_node(owner.id)
        assert set(current.materializer_sources) == {member.id for member in group}
        assert not current.decayed
        best = (max(group, key=lambda member: (c._trace_quality(member), member.id))
                if materializer == "cluster" else c._highest_confidence_source(group))
        assert current.content == c._synthesize_digest(group, best=best, prepared_sources=prepare(group))
    materialize()
    assert prepared == {member.id: 2 for member in members}
