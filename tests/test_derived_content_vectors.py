"""Derived search describes current content independently of clustering inputs."""

from dataclasses import asdict

import pytest

from living_memory.chunking import chunk_text
from living_memory.config import MemoryConfig
from living_memory.consolidation import memory_consolidate, memory_teach
from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity
from living_memory.retrieval import LAZY_CHUNK_NODE_LIMIT, MemoryRecallService
from living_memory.storage import MemoryStore


CORE = (
    "Deploy rollback requires migration guard checklist verification covering "
    "database schema state plus release gating rules across regional failover clusters gateway."
)
FACTS = [
    "Ledger corrupted.", "Predeploy enforcement.", "Nightly audit.", "Canary validates.",
    "Oncall paged.", "Lockfile ordering.", "Rehearsal quarterly.", "Postmortem overrides.",
    "Flags decouple.", "Manifest signed.",
]
SCOPE = "project:release"


@pytest.fixture(autouse=True)
def hash_backend(monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")


class Encoder:
    def __init__(self):
        self.model = LocalEmbeddingModel(backend="hash")
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return self.model.embed(text)

    def batch(self, texts):
        return [self.embed(text) for text in texts]


def seed(store, encoder, scope=SCOPE, *, long=False):
    core = CORE if not long else " ".join([CORE] * 5)
    return [store.create_node(
        node_id=f"{scope}-{index:02d}", level="trace", content=f"{core} {fact}",
        context={"scope": scope, "agent": f"agent-{index % 2}"},
        embedding=encoder.embed(f"{core} {fact}"),
        stats={"confidence": .4, "usefulness_score": .2},
    ) for index, fact in enumerate(FACTS)]


def assert_content_vectors(store, node, encoder):
    windows = chunk_text(node.content)
    chunks = store.list_node_chunks(node.id)
    assert len(chunks) == len(windows)
    for chunk, window in zip(chunks, windows, strict=True):
        assert chunk.embedding == pytest.approx(encoder.model.embed(window.text), abs=1e-7)
    assert store.node_chunks_are_current(node.id)


def recall(service, query, scope=SCOPE):
    return service.memory_recall(
        query, scope=scope, max_results=30, log_access=False, log_event=False,
    )


@pytest.mark.parametrize("long", [False, True])
def test_consolidate_recall_and_updates_keep_content_vectors_and_sources(tmp_path, long):
    encoder = Encoder()
    with MemoryStore(tmp_path / "memory.sqlite3", chunk_embedder=encoder.batch) as store:
        sources = seed(store, encoder, long=long)
        preserved = {s.id: asdict(s) for s in sources}
        encoder.calls.clear()
        concept, = memory_consolidate(store, scope=SCOPE, min_cluster_size=10).concepts_created
        assert all(concept.content != s.content for s in sources)
        assert (len(chunk_text(concept.content)) > 1) is long
        assert encoder.calls == [w.text for w in chunk_text(concept.content)]
        assert_content_vectors(store, concept, encoder)
        assert concept.embedding == sources[-1].embedding
        assert set(concept.source_traces) == set(preserved)
        assert {s.id: asdict(store.get_node(s.id)) for s in sources} == preserved

        service = MemoryRecallService(store, embedder=encoder)
        result = next(r for r in recall(service, concept.content) if r.node.id == concept.id)
        assert result.node.content == concept.content
        if not long:
            assert result.vector_score == pytest.approx(1.0, abs=1e-6)
            assert result.vector_score > cosine_similarity(
                encoder.model.embed(concept.content), concept.embedding,
            )

        before = store.list_node_chunks(concept.id)
        encoder.calls.clear()
        again, = memory_consolidate(store, scope=SCOPE, min_cluster_size=10).concepts_updated
        assert again.id == concept.id
        assert store.list_node_chunks(concept.id) == before
        assert encoder.calls == []

        content = f"{CORE} Release requires a signed certificate from the harbor authority."
        added = store.create_node(
            node_id="z-new", level="trace", content=content,
            context={"scope": SCOPE, "agent": "reviewer"},
            embedding=encoder.embed(content), stats={"confidence": .5, "usefulness_score": .9},
        )
        encoder.calls.clear()
        updated, = memory_consolidate(store, scope=SCOPE, min_cluster_size=10).concepts_updated
        assert updated.id == concept.id
        assert updated.content != concept.content
        assert "signed certificate" in updated.content
        assert set(updated.source_traces) == {*preserved, added.id}
        assert updated.embedding == added.embedding
        assert encoder.calls == [w.text for w in chunk_text(updated.content)]
        assert_content_vectors(store, updated, encoder)
        current = next(r for r in recall(service, updated.content) if r.node.id == updated.id)
        assert current.node.content == updated.content


def test_cross_scope_create_update_and_teach_reach_warm_recall(tmp_path):
    encoder = Encoder()
    config = MemoryConfig(
        db_path=tmp_path / "memory.sqlite3",
        phase_thresholds={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 100_000},
    )
    with MemoryStore(config, chunk_embedder=encoder.batch) as store:
        all_sources = [seed(store, encoder, scope) for scope in (SCOPE, "project:beta", "project:gamma")]
        for sources in all_sources:
            memory_consolidate(store, scope=sources[0].scope, min_cluster_size=10)
        promoted, = store.list_nodes(level="concept", scope="global")
        assert promoted.materializer == "cross-scope-promotion"
        assert set(promoted.provenance["active_promoted_from"]) == {SCOPE, "project:beta", "project:gamma"}
        assert len(promoted.materializer_sources) == 3
        assert_content_vectors(store, promoted, encoder)
        service = MemoryRecallService(store, embedder=encoder)
        assert promoted.id in {r.node.id for r in recall(service, promoted.content)}

        old = all_sources[0][-1]
        corrected_text = f"{CORE} Certificates verified before deployment."
        taught = memory_teach(store, old.id, corrected_text)
        current = store.get_node(promoted.id)
        assert current.content != promoted.content
        assert "Certificates verified" in current.content
        assert taught.corrective_trace.id in current.source_traces
        assert old.id not in current.source_traces
        results = recall(service, current.content)
        delivered = next(r for r in results if r.node.id == current.id)
        assert delivered.node.content == current.content
        assert_content_vectors(store, current, encoder)
        assert all(r.node.scope in (SCOPE, "global") for r in results)
        positions = {r.node.id: i for i, r in enumerate(results)}
        if old.id in positions:
            assert positions[taught.corrective_trace.id] < positions[old.id]
            assert results[positions[old.id]].superseded
        assert store.get_node(old.id).content == old.content
        assert store.list_connections(source_id=taught.corrective_trace.id, target_id=old.id, relation_type="supersedes")


def test_representative_change_does_not_reencode_unchanged_content_and_failure_rolls_back(tmp_path):
    encoder = Encoder()
    with MemoryStore(tmp_path / "memory.sqlite3", chunk_embedder=encoder.batch) as store:
        node = store.create_node(
            level="concept", content=CORE, representative_embedding=[1., 0.],
            provenance={"strategy": "token-jaccard", "source_traces": ["source"]},
        )
        chunks = store.list_node_chunks(node.id)
        encoder.calls.clear()
        changed = store.update_node(node.id, representative_embedding=[0., 1.])
        assert changed.embedding == [0., 1.]
        assert store.list_node_chunks(node.id) == chunks
        assert encoder.calls == []
        before = asdict(changed)

        def unavailable(texts):
            raise RuntimeError("encoder unavailable")

        store.set_chunk_embedder(unavailable)
        with pytest.raises(RuntimeError, match="encoder unavailable"):
            store.update_node(node.id, content=CORE + " New fact.", representative_embedding=[1., 0.])
        assert asdict(store.get_node(node.id)) == before
        assert store.list_node_chunks(node.id) == chunks
        store.set_chunk_embedder(encoder.batch)
        for content in ("Signed release manifest.", " ".join([CORE] * 8)):
            encoder.calls.clear()
            updated = store.update_node(node.id, content=content, representative_embedding=[1., 0.])
            assert encoder.calls == [w.text for w in chunk_text(content)]
            assert_content_vectors(store, updated, encoder)


@pytest.mark.parametrize("drop_legacy", [False, True])
def test_existing_single_chunk_derivatives_are_repaired_by_bounded_ordinary_recall(tmp_path, drop_legacy):
    encoder = Encoder()
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path, chunk_embedder=encoder.batch) as store:
        source = store.create_node(level="trace", content=CORE, embedding=encoder.embed(CORE))
        affected = [store.create_node(
            level="concept", node_id=f"derived-{i:03d}", content=f"{CORE} Stored detail {i}.",
            context={"scope": SCOPE}, embedding=source.embedding,
            provenance={"strategy": "embedding-cosine", "source_traces": [source.id]},
        ) for i in range(LAZY_CHUNK_NODE_LIMIT + 3)]
        # Historical rows were written through the content-vector API with a
        # representative. Reopening runs the ordinary storage migration.
        assert all(store.node_chunks_are_current(n.id) for n in affected)
        multi = store.create_node(
            level="concept", content=" ".join([CORE] * 8), embedding=source.embedding,
            provenance={"strategy": "embedding-cosine", "source_traces": [source.id]},
        )
        unrelated = store.create_node(level="concept", content="Independent source.", embedding=[1., 0.])
        procedural = store.create_node(
            level="schema", content="1. Verify deployment.", embedding=[1., 0.],
            provenance={"strategy": "procedural", "source_traces": [source.id]},
        )
        untouched = {n.id: store.list_node_chunks(n.id) for n in (source, multi, unrelated, procedural)}
        if drop_legacy:
            store.drop_node_embedding_column()
        nodes_before = {n.id: asdict(store.get_node(n.id)) for n in affected}
        store.connection.execute("DELETE FROM metadata WHERE key='derived_content_chunks'")
        store.connection.commit()

    encoder.calls.clear()
    with MemoryStore(path, chunk_embedder=encoder.batch) as store:
        assert encoder.calls == []
        assert all(not store.node_chunks_are_current(n.id) for n in affected)
        assert {n.id: asdict(store.get_node(n.id)) for n in affected} == nodes_before
        assert {node_id: store.list_node_chunks(node_id) for node_id in untouched} == untouched
        service = MemoryRecallService(store, embedder=encoder)
        query = affected[0].content
        delivered = next(r for r in recall(service, query) if r.node.id == affected[0].id)
        assert delivered.vector_score == pytest.approx(1., abs=1e-6)
        assert sum(store.node_chunks_are_current(n.id) for n in affected) == LAZY_CHUNK_NODE_LIMIT
        assert len(encoder.calls) == LAZY_CHUNK_NODE_LIMIT + 1  # content plus query
        recall(service, affected[-1].content)
        for node in affected:
            assert_content_vectors(store, store.get_node(node.id), encoder)
            assert store.get_node(node.id).embedding == (None if drop_legacy else source.embedding)
        fixed = {n.id: store.list_node_chunks(n.id) for n in affected}

    with MemoryStore(path, chunk_embedder=encoder.batch) as store:
        assert {n.id: store.list_node_chunks(n.id) for n in affected} == fixed
