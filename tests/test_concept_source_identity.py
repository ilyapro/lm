"""Source identity survives correction, clustering changes and lossy digests."""

from collections import Counter
from dataclasses import asdict

import pytest

from living_memory import consolidation as c
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.retrieval import MemoryRecallService
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore
from test_transport_identity import FakeMCP


SCOPE = "project:source-identity"
TEXTS = [
    "Deploy release requires migration guard. Archive backups preserve audit evidence.",
    "Deploy release requires migration guard. Restore transaction records before validation.",
    "Deploy release requires migration guard. Encrypt transport packets using rotating credentials.",
]
CORRECTION = "Deploy release requires migration guard. Archive backups preserve verified audit evidence."


class Encoder:
    def embed(self, text):
        return [1.0, 0.0]


@pytest.fixture(autouse=True)
def embeddings(monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setattr(LocalEmbeddingModel, "embed", lambda self, text: Encoder().embed(text))


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3", chunk_embedder=lambda texts: [[1.0, 0.0] for _ in texts]) as instance:
        yield instance


def trace(store, content, **context):
    return store.append_trace(content, {"scope": SCOPE, "agent": "writer", **context})


def materialize(store, members, floor=1):
    cluster = c._TraceCluster(members[0].scope, members, Counter())
    return c._materialize_cluster(store, cluster, min_cluster_size=floor)


def active(store):
    return store.list_nodes(level="concept", scope=SCOPE, limit=1000)


def assert_sources(store, node, ids):
    assert set(node.source_traces) == set(ids)
    sources = list(store.get_nodes(ids).values())
    assert all(not source.decayed and not source.corrections for source in sources)
    best = max(sources, key=lambda source: (c._trace_quality(source), source.id))
    assert node.content == c._synthesize_digest(sources, best=best)
    tokens = Counter()
    for source in sources:
        tokens.update(c._significant_tokens(source.content))
    assert node.provenance["cluster_key"] == c._cluster_key(c._TraceCluster(node.scope, sources, tokens))
    assert set(ids) <= {edge.target_id for edge in store.list_connections(source_id=node.id, relation_type="related")}


def test_repeated_digest_is_independent_but_its_pass_keeps_its_owner_after_teach(store):
    a, b = [trace(store, text) for text in TEXTS[:2]]
    (original, _), = materialize(store, [a, b])
    reflected = trace(store, original.content)
    fresh = trace(store, TEXTS[2])
    (second, created), = materialize(store, [reflected, fresh])
    assert created and second.id != original.id
    assert set(second.source_traces) == {reflected.id, fresh.id}
    c.memory_teach(store, a.id, CORRECTION)
    before_repeat = {node.id for node in active(store)}
    (repeated, created), = materialize(store, [reflected, fresh])
    assert not created and repeated.id == second.id
    assert {node.id for node in active(store)} == before_repeat
    assert set(repeated.source_traces) == {reflected.id, fresh.id}
    assert CORRECTION in store.get_node(original.id).content


def test_recorded_union_keeps_all_sources_and_owner_after_teach_and_subset_repeat(store):
    a, b = [trace(store, text) for text in TEXTS[:2]]
    (owner, _), = materialize(store, [a, b])
    reflected, fresh = trace(store, owner.content), trace(store, TEXTS[2])
    # A retained derivation may already own a union assembled by older passes.
    members = [a, b, reflected, fresh]
    store.update_node(owner.id, provenance={**owner.provenance, "source_traces": [source.id for source in members]})
    taught = c.memory_teach(store, a.id, CORRECTION)
    (repeated, created), = materialize(store, [reflected, fresh])
    assert not created and repeated.id == owner.id
    assert [node.id for node in active(store)] == [owner.id]
    assert_sources(store, repeated, [taught.corrective_trace.id, b.id, reflected.id, fresh.id])
    # Re-remembering a source uses the existing duplicate-content supersedes chain.
    replacement = trace(store, fresh.content)
    (again, created), = materialize(store, [reflected, replacement])
    assert not created and again.id == owner.id
    assert_sources(store, again, [taught.corrective_trace.id, b.id, reflected.id, replacement.id])


def test_ordinary_consolidate_teach_repeat_and_new_knowledge(tmp_path):
    mcp = create_mcp_server(tmp_path / "ordinary.sqlite3", mcp_factory=FakeMCP)
    with mcp.memory_store as store:
        store.set_chunk_embedder(lambda texts: [[1.0, 0.0] for _ in texts])
        a, b = [trace(store, text) for text in TEXTS[:2]]
        first = c.memory_consolidate(store, scope=SCOPE, force=True, min_cluster_size=2)
        owner, = first.concepts_created
        taught = c.memory_teach(store, a.id, CORRECTION)
        second = c.memory_consolidate(store, scope=SCOPE, force=True, min_cluster_size=2)
        assert not second.concepts_created
        assert {node.id for node in second.concepts_updated} == {owner.id}
        assert_sources(store, store.get_node(owner.id), [taught.corrective_trace.id, b.id])
        fresh = trace(store, TEXTS[2])
        third = c.memory_consolidate(store, scope=SCOPE, force=True, min_cluster_size=2)
        assert not third.concepts_created
        current = store.get_node(owner.id)
        assert_sources(store, current, [taught.corrective_trace.id, b.id, fresh.id])
        assert "rotating credentials" in current.content
        service = MemoryRecallService(store, embedder=Encoder())
        recalled = service.memory_recall("Deploy release migration guard", scope=SCOPE, max_results=10,
                                         log_access=False, log_event=False)
        assert any(result.node.id == owner.id and result.node.content == current.content for result in recalled)
        lookup = mcp.tools["memory_lookup"](node_id=owner.id)["results"][0]
        assert lookup["content"] == current.content and not lookup["decayed"]
        old = mcp.tools["memory_lookup"](node_id=a.id)["results"][0]
        assert old["content"] == a.content and old["provenance"]["corrections"]
        assert store.get_node(current.id).embedding is not None
    mcp.auto_consolidation.close()


def test_new_semantic_groups_with_same_lexical_key_remain_separate(store, monkeypatch):
    common = "amber birch cedar dune elm fir grove hazel ivy juniper kelp larch"
    texts = [f"{common}. {index} " + " ".join(f"distinct{index}{chr(97+j)}" for j in range(16))
             for index in range(4)]
    # Two semantic groups share the same common vocabulary but have independent facts.
    vectors = {text: ([1.0, 0.0] if index < 2 else [-1.0, 0.0]) for index, text in enumerate(texts)}
    monkeypatch.setattr(LocalEmbeddingModel, "embed", lambda self, text: vectors.get(text, [0.0, 1.0]))
    traces = [trace(store, text) for text in texts]
    first = c.memory_consolidate(store, scope=SCOPE, force=True, min_cluster_size=2)
    assert len(first.concepts_created) == 2
    assert len({node.provenance["cluster_key"] for node in first.concepts_created}) == 1
    expected = {frozenset(source.id for source in traces[:2]), frozenset(source.id for source in traces[2:])}
    assert {frozenset(node.source_traces) for node in first.concepts_created} == expected
    second = c.memory_consolidate(store, scope=SCOPE, force=True, min_cluster_size=2)
    assert not second.concepts_created
    assert {frozenset(node.source_traces) for node in active(store)} == expected


def test_overlapping_owners_are_never_collapsed_or_given_each_others_sources(store):
    a, b, extra = [trace(store, text) for text in TEXTS]
    (left, _), = materialize(store, [a, b])
    right = store.create_node(level="concept", content="Retained overlapping derivation.", context={"scope": SCOPE},
                             provenance={"strategy": "token-jaccard", "cluster_key": left.provenance["cluster_key"],
                                         "source_traces": [b.id, extra.id]})
    independent = trace(store, "Deploy release checks the signed manifest.")
    outputs = materialize(store, [a, b, extra, independent])
    assert len(outputs) == 3
    assert set(store.get_node(left.id).source_traces) == {a.id, b.id}
    assert set(store.get_node(right.id).source_traces) == {b.id, extra.id}
    new = next(node for node, created in outputs if created)
    assert new.source_traces == [independent.id]
    again = materialize(store, [a, b, extra, independent])
    assert not any(created for _, created in again)
    assert {node.id for node in active(store)} == {left.id, right.id, new.id}


def test_same_labels_and_digest_do_not_replace_independent_derivations(store, monkeypatch):
    sources = [trace(store, text, task_pattern=f"group-{i}") for i, text in enumerate(TEXTS)]
    monkeypatch.setattr(c, "_synthesize_digest", lambda *args, **kwargs: "A lossy shared digest.")
    owners = [materialize(store, [source])[0][0] for source in sources]
    assert {node.id for node in active(store)} == {node.id for node in owners}
    c.memory_teach(store, sources[0].id, CORRECTION)
    assert all(not store.get_node(owner.id).decayed for owner in owners)
    assert set(store.get_node(owners[1].id).source_traces) == {sources[1].id}
    generic = store.create_node(level="concept", content=owners[0].content, context={"scope": SCOPE})
    assert all(not store.get_node(owner.id).decayed for owner in owners)
    assert not store.get_node(generic.id).decayed


def test_cluster_and_procedural_groups_keep_shared_titles_separate(store):
    for group in ("north", "south"):
        for index in range(3):
            trace(store, f"Deploy release {group} confirms checkpoint {index}.",
                  task_pattern=group, procedure_id="release", step_order=index)
    first = c.memory_consolidate(store, scope=SCOPE, force=True, min_cluster_size=2)
    assert len(first.schemas_created) == 2
    assert len(first.concepts_created) == 2
    for node in [*first.schemas_created, *first.concepts_created]:
        group = node.context["task_pattern"]
        assert all(store.get_node(source).context["task_pattern"] == group for source in node.source_traces)
    second = c.memory_consolidate(store, scope=SCOPE, force=True, min_cluster_size=2)
    assert not second.schemas_created and not second.concepts_created


def test_stale_pass_resolves_corrections_even_before_first_materialization(store):
    a, b = [trace(store, text) for text in TEXTS[:2]]
    taught = c.memory_teach(store, a.id, CORRECTION)
    (owner, created), = materialize(store, [a, b])
    assert created
    assert_sources(store, owner, [taught.corrective_trace.id, b.id])
    final = c.memory_teach(store, taught.corrective_trace.id, "Deploy release uses the electronic archive.")
    (repeated, created), = materialize(store, [a, b])
    assert not created and repeated.id == owner.id
    assert_sources(store, repeated, [final.corrective_trace.id, b.id])


def test_legacy_owner_follows_recorded_ancestor_and_drops_deleted_inputs(store):
    original, removed = [trace(store, text) for text in TEXTS[:2]]
    intermediate = c.memory_teach(store, original.id, CORRECTION).corrective_trace
    current = c.memory_teach(store, intermediate.id, "Deploy release uses the electronic archive.").corrective_trace
    store.delete_node(removed.id)
    owner = store.create_node(
        level="concept", content="Recorded archive evidence.", context={"scope": SCOPE},
        provenance={"cluster_key": "archive", "source_traces": [original.id, removed.id, "missing-source"]},
    )
    (updated, created), = materialize(store, [current])
    assert not created and updated.id == owner.id
    assert_sources(store, updated, [current.id])


@pytest.mark.parametrize("invalid", ["decayed", "corrected", "superseded"])
def test_invalid_recorded_owner_does_not_claim_current_inputs(store, invalid):
    members = [trace(store, text) for text in TEXTS[:2]]
    owner = store.create_node(
        level="concept", content="Retired release evidence.", context={"scope": SCOPE},
        provenance={"strategy": "token-jaccard", "source_traces": [member.id for member in members]},
    )
    if invalid == "decayed":
        store.soft_delete_node(owner.id)
    elif invalid == "corrected":
        store.update_node(owner.id, provenance={**owner.provenance, "corrections": [{"reason": "rejected"}]})
    else:
        replacement = trace(store, "Retired release evidence was rejected.")
        store.create_connection(replacement.id, owner.id, "supersedes")
    before = asdict(store.get_node(owner.id))
    (current, created), = materialize(store, members)
    assert created and current.id != owner.id
    assert_sources(store, current, [member.id for member in members])
    assert asdict(store.get_node(owner.id)) == before


@pytest.mark.parametrize("existing", [False, True])
def test_ordinary_materialization_edge_failure_rolls_back_sources_digest_and_chunks(store, monkeypatch, existing):
    a, b, fresh = [trace(store, text) for text in TEXTS]
    if existing:
        materialize(store, [a, b])
    before = list(store.connection.iterdump())
    write = c._update_edge_weights_from_co_access

    def fail(*args):
        write(*args)
        raise RuntimeError("source edge failure")

    monkeypatch.setattr(c, "_update_edge_weights_from_co_access", fail)
    with pytest.raises(RuntimeError, match="source edge failure"):
        materialize(store, [a, b, fresh])
    assert list(store.connection.iterdump()) == before
    assert not store.connection.in_transaction


def test_unknown_legacy_and_unrelated_materializers_are_not_owners(store):
    sources = [trace(store, text) for text in TEXTS]
    nodes = [
        store.create_node(level="concept", content=source.content, context={"scope": SCOPE},
                          provenance=provenance)
        for source, provenance in zip(sources, [
            {"cluster_key": "release"},
            {"strategy": "procedural", "procedure_key": "release", "source_traces": [sources[0].id]},
            {"source_traces": [sources[0].id]},
        ])
    ]
    before = {node.id: asdict(node) for node in nodes}
    (created, is_new), = materialize(store, sources)
    assert is_new and created.id not in before
    assert {node_id: asdict(store.get_node(node_id)) for node_id in before} == before


@pytest.mark.parametrize("changed", [{"task_pattern": "south"}, {"scope": "project:elsewhere"}])
def test_new_materialization_groups_current_replacements_not_stale_snapshot(store, changed):
    a, b = [trace(store, text, task_pattern="north") for text in TEXTS[:2]]
    taught = c.memory_teach(store, a.id, CORRECTION, context=changed)
    outputs = materialize(store, [a, b])
    assert len(outputs) == 2
    for owner, created in outputs:
        assert created and len(owner.source_traces) == 1
        source = store.get_node(owner.source_traces[0])
        assert c._cluster_group(owner) == c._cluster_group(source)
    assert {source for owner, _ in outputs for source in owner.source_traces} == {b.id, taught.corrective_trace.id}


def test_explicit_derivative_replacement_does_not_displace_independent_older_input(store):
    obsolete, independent = [trace(store, text, timestamp="2025-01-01T00:00:00Z") for text in TEXTS[:2]]
    (retired, _), = materialize(store, [obsolete])
    # A distinct retained derivation depends on the same old input plus an
    # independently valid older source. Only the explicit target is rejected.
    other = store.create_node(level="concept", content="Other retained derivation.", context={"scope": SCOPE},
                              provenance={"strategy": "token-jaccard", "cluster_key": "release",
                                          "source_traces": [obsolete.id, independent.id]})
    correction = trace(store, CORRECTION)
    store.create_connection(correction.id, retired.id, "supersedes")
    store.update_node(other.id, provenance={**other.provenance,
                                           "source_traces": [obsolete.id, independent.id, correction.id]})
    # The target's actual recorded inputs delimit the rejection.
    store.update_node(retired.id, provenance={**retired.provenance, "source_traces": [obsolete.id]})
    current = c._refresh_derived_node(store, store.get_node(other.id))
    assert_sources(store, current, [independent.id, correction.id])
    outputs = materialize(store, [obsolete, independent, correction])
    assert any(node.id == other.id and set(node.source_traces) == {independent.id, correction.id}
               for node, _ in outputs)


def test_promotions_use_recorded_owners_and_keep_other_global_knowledge(store):
    unrelated = store.create_node(level="concept", content="Independent global release policy.",
                                  context={"scope": "global"}, embedding=[1.0, 0.0])
    groups = {}
    for group in ("north", "south"):
        groups[group] = [
            store.create_node(level="concept", content=f"Release {group} evidence {scope}.",
                              context={"scope": f"project:{scope}", "task_pattern": group},
                              embedding=[1.0, 0.0])
            for scope in ("alpha", "beta", "gamma")
        ]
    before = asdict(store.get_node(unrelated.id))
    north, = c._cross_scope_promotion(store, groups["north"][0], phase_number=4)
    south, = c._cross_scope_promotion(store, groups["south"][0], phase_number=4)
    assert north.id != south.id
    assert set(north.provenance["source_concepts"]) == {node.id for node in groups["north"]}
    assert set(south.provenance["source_concepts"]) == {node.id for node in groups["south"]}
    assert c._cross_scope_promotion(store, groups["north"][0], phase_number=4) == []
    assert asdict(store.get_node(unrelated.id)) == before
    fourth = store.create_node(level="concept", content="Release north fourth project evidence.",
                               context={"scope": "project:delta", "task_pattern": "north"}, embedding=[1.0, 0.0])
    updated, = c._cross_scope_promotion(store, fourth, phase_number=4)
    assert updated.id == north.id
    assert set(updated.provenance["source_concepts"]) == {node.id for node in groups["north"]} | {fourth.id}
    assert store.get_node(south.id).source_traces == south.source_traces


def test_promotion_edge_failure_is_atomic(store, monkeypatch):
    sources = [store.create_node(level="concept", content=f"Release evidence {scope}.",
                                 context={"scope": f"project:{scope}"}, embedding=[1.0, 0.0])
               for scope in ("alpha", "beta", "gamma")]
    before = list(store.connection.iterdump())
    write = c._connect_project_concepts_to_global

    def fail(*args, **kwargs):
        write(*args, **kwargs)
        raise RuntimeError("promotion source edge failure")

    monkeypatch.setattr(c, "_connect_project_concepts_to_global", fail)
    with pytest.raises(RuntimeError, match="promotion source edge failure"):
        c._cross_scope_promotion(store, sources[0], phase_number=4)
    assert list(store.connection.iterdump()) == before


def test_promotion_rereads_project_sources_and_refreshes_same_input_ids(store):
    original = trace(store, "Release requires the obsolete paper permit.", scope="project:alpha")
    (stale, _), = materialize(store, [original])
    store.update_node(stale.id, embedding=[1.0, 0.0])
    stale = store.get_node(stale.id)
    for scope in ("beta", "gamma"):
        store.create_node(level="concept", content=f"Release {scope} verifies the checksum.",
                          context={"scope": f"project:{scope}"}, embedding=[1.0, 0.0])
    replacement = "Release requires the electronic manifest."
    c.memory_teach(store, original.id, replacement)
    promoted, = c._cross_scope_promotion(store, stale, phase_number=4)
    assert replacement in promoted.content and "obsolete paper permit" not in promoted.content
    source_ids = promoted.provenance["source_concepts"]
    current = store.get_node(stale.id)
    fresh = trace(store, "Release encrypts transport packets.", scope=current.scope)
    materialize(store, [*store.get_nodes(current.source_traces).values(), fresh])
    updated, = c._cross_scope_promotion(store, stale, phase_number=4)
    assert updated.id == promoted.id and updated.provenance["source_concepts"] == source_ids
    assert fresh.id in updated.source_traces
    assert "transport packets" in updated.content
    assert c._cross_scope_promotion(store, stale, phase_number=4) == []
