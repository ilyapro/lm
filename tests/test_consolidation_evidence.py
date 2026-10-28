"""Consolidation decisions depend on selected evidence and ready resources."""

import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from threading import Event

import pytest

from living_memory import consolidation as c, server
from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity
from living_memory.storage import MemoryStore
from test_mcp_server import FakeMCP


SCOPE = "project:target"
TEXTS = ["authentication timeout in production", "auth slow in prod", "authentication slow production"]


@pytest.fixture(autouse=True)
def local_policy(monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "fixed")
    monkeypatch.delenv(c.MERGE_FLOOR_ENV, raising=False)


async def native_evidence_probe(path):
    from fastmcp import Client

    mcp = server.create_mcp_server(path)
    store = mcp.memory_store
    model = LocalEmbeddingModel(backend="hash")
    vectors = model.embed_batch(TEXTS)
    assert all(cosine_similarity(vectors[0], vector) == pytest.approx(1) for vector in vectors)
    assert max(c._jaccard(c._significant_tokens(left), c._significant_tokens(right))
               for i, left in enumerate(TEXTS) for right in TEXTS[i+1:]) <= .5
    try:
        async with Client(mcp) as client:
            async def call(name, **arguments):
                result = await client.call_tool(name, arguments)
                assert not result.is_error
                return result.data

            sources = []
            for i, text in enumerate(TEXTS):
                response = await call("memory_remember", content=text,
                                      context={"scope": SCOPE, "agent": f"writer-{i}"})
                sources.append(response["node"]["id"])
            before_sources = {key: asdict(node) for key, node in store.get_nodes(sources).items()}
            status = await call("memory_status", scope=SCOPE)
            before = await call("memory_consolidate", scope=SCOPE)
            concept, = before["concepts_created"]
            assert set(concept["provenance"]["source_traces"]) == set(sources)
            for i in range(97):
                store.append_trace(f"Unrelated garden observation {i}", {"scope": "project:unrelated"})
            after = await call("memory_consolidate", scope=SCOPE)
            assert after["concepts_created"] == []
            updated, = after["concepts_updated"]
            assert updated["id"] == concept["id"]
            assert updated["content"] == concept["content"]
            assert updated["provenance"]["source_traces"] == concept["provenance"]["source_traces"]
            assert before["traces_considered"] == after["traces_considered"] == 3
            assert before["clusters_considered"] == after["clusters_considered"] == 1
            assert before_sources == {key: asdict(node) for key, node in store.get_nodes(sources).items()}
            assert "phase" not in status
            assert status["counts"]["by_level"]["trace"]["total"] == 3
            stats, = await client.read_resource("memory://stats")
            assert '"phase"' not in stats.text
            assert '"retrieval_weights"' in stats.text
            recalled = await call("memory_recall", query=TEXTS[0], scope=SCOPE, view="full")
            assert recalled["results"]
            return {
                "backend": model.encoder_id(), "sources_unchanged": True,
                "selected_traces": [before["traces_considered"], after["traces_considered"]],
                "clusters": [before["clusters_considered"], after["clusters_considered"]],
                "created": [len(before["concepts_created"]), len(after["concepts_created"])],
                "updated": [len(before["concepts_updated"]), len(after["concepts_updated"])],
                "same_owner_content_and_sources": True, "phase_in_status": "phase" in status,
            }
    finally:
        mcp.auto_consolidation.close()
        store.close()


def test_native_mcp_same_sources_survive_unrelated_population_change(tmp_path):
    asyncio.run(native_evidence_probe(tmp_path / "memory.sqlite3"))


def seed(store, scope=SCOPE, count=3):
    return [store.append_trace(text, {"scope": scope, "agent": f"writer-{i}"})
            for i, text in enumerate(TEXTS[:count])]


def test_same_scope_floor_similarity_scope_and_consensus_still_apply(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        sources = seed(store, count=2)
        seed(store, "project:other")
        assert c.memory_consolidate(store, scope=SCOPE).concepts_created == []
        third = store.append_trace(TEXTS[2], {"scope": SCOPE, "agent": "writer-2"})
        assert c.memory_consolidate(store, scope=SCOPE, min_cluster_size=4, force=True).concepts_created == []
        concept, = c.memory_consolidate(store, scope=SCOPE).concepts_created
        assert set(concept.source_traces) == {*(node.id for node in sources), third.id}
        assert concept.unique_agents == 3 and concept.confidence > .5
        solo = seed(store, "project:solo")
        for node in solo:
            store.update_node(node.id, context={"scope": node.scope, "agent": "one"})
        single_agent, = c.memory_consolidate(store, scope="project:solo").concepts_created
        assert single_agent.unique_agents == 1 and single_agent.confidence == .5
        for text in ("orchard pears fertilizer", "invoice currency rounding", "volcano magma basalt"):
            store.append_trace(text, {"scope": "project:different"})
        assert c.memory_consolidate(store, scope="project:different").concepts_created == []


@pytest.mark.parametrize("invalid", ["corrected", "superseded", "decayed"])
def test_invalid_traces_never_enter_new_concept(tmp_path, invalid):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        sources = seed(store)
        rejected = store.append_trace("authentication timeout production", {"scope": SCOPE})
        if invalid == "corrected":
            store.add_correction(rejected.id, old=rejected.content, new="obsolete", by="reviewer")
        elif invalid == "decayed":
            store.soft_delete_node(rejected.id)
        else:
            store.create_connection(sources[0].id, rejected.id, "supersedes")
        concept, = c.memory_consolidate(store, scope=SCOPE, min_cluster_size=3).concepts_created
        assert set(concept.source_traces) == {node.id for node in sources}


def peer(store, scope, *, embedding=(1., 0.), group=None):
    return store.create_node(level="concept", content=f"Authentication timeout evidence for {scope}.",
                             context={"scope": scope, **({"task_pattern": group} if group else {})},
                             embedding=list(embedding) if embedding else None)


@pytest.mark.parametrize("invalid", [None, "same_scope", "session", "group", "dissimilar",
                                    "missing_vector", "corrected", "superseded", "decayed"])
def test_cross_scope_admission_through_consolidation(tmp_path, invalid):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        model = LocalEmbeddingModel(backend="hash")
        vector = model.embed(TEXTS[0])
        seed(store)
        beta = peer(store, "project:beta", embedding=vector)
        gamma = peer(store, "project:gamma", embedding=None if invalid == "missing_vector" else vector)
        if invalid == "same_scope":
            store.update_node(gamma.id, context={"scope": beta.scope})
        elif invalid == "session":
            store.update_node(gamma.id, context={"scope": "session:gamma"})
        elif invalid == "group":
            store.update_node(gamma.id, context={"scope": gamma.scope, "task_pattern": "other"})
        elif invalid == "dissimilar":
            store.update_node(gamma.id, embedding=[-v for v in vector])
        elif invalid == "corrected":
            store.add_correction(gamma.id, old=gamma.content, new="obsolete", by="reviewer")
        elif invalid == "superseded":
            store.create_connection(beta.id, gamma.id, "supersedes")
        elif invalid == "decayed":
            store.soft_delete_node(gamma.id)
        result = c.memory_consolidate(store, scope=SCOPE)
        assert len(result.concepts_created) == 1
        assert len(result.concepts_promoted) == (1 if invalid is None else 0)
        if invalid is None:
            promoted, = result.concepts_promoted
            assert set(promoted.materializer_sources) == {beta.id, gamma.id, result.concepts_created[0].id}


@pytest.mark.parametrize("change", ["scope", "group", "vector", "correction", "supersedes", "decay"])
def test_cross_scope_writer_rechecks_prepared_candidates(tmp_path, monkeypatch, change):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        vector = LocalEmbeddingModel(backend="hash").embed(TEXTS[0])
        seed(store)
        beta = peer(store, "project:beta", embedding=vector)
        gamma = peer(store, "project:gamma", embedding=vector)
        prepare = c._promotion_candidates

        def changed(*args, **kwargs):
            snapshot = list(prepare(*args, **kwargs))
            if change == "scope":
                store.update_node(gamma.id, context={"scope": beta.scope})
            elif change == "group":
                store.update_node(gamma.id, context={"scope": gamma.scope, "task_pattern": "different"})
            elif change == "vector":
                store.update_node(gamma.id, embedding=[-v for v in vector])
            elif change == "correction":
                store.add_correction(gamma.id, old=gamma.content, new="obsolete", by="reviewer")
            elif change == "supersedes":
                store.create_connection(beta.id, gamma.id, "supersedes")
            else:
                store.soft_delete_node(gamma.id)
            return snapshot

        monkeypatch.setattr(c, "_promotion_candidates", changed)
        result = c.memory_consolidate(store, scope=SCOPE)
        assert len(result.concepts_created) == 1
        assert result.concepts_promoted == []
        assert store.list_nodes(level="concept", scope="global") == []


def test_pass_reuses_candidates_and_cached_trace_encodings(tmp_path, monkeypatch):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        traces = seed(store)
        for i in range(3):
            traces.append(store.append_trace(f"orchard pear fertilizer sample {i}", {"scope": SCOPE}))
        model = LocalEmbeddingModel(backend="hash")
        for scope in ("project:beta", "project:gamma"):
            peer(store, scope, embedding=model.embed(TEXTS[0]))
        counts = Counter()
        encoded = []
        original_batch = LocalEmbeddingModel.embed_batch
        original_page = store.promotion_candidate_page

        def batch(self, texts):
            encoded.extend(text for text in texts if text in {t.content for t in traces})
            return original_batch(self, texts)

        def page(**kwargs):
            counts["candidate_reads"] += 1
            return original_page(**kwargs)

        monkeypatch.setattr(LocalEmbeddingModel, "embed_batch", batch)
        monkeypatch.setattr(store, "promotion_candidate_page", page)
        result = c.memory_consolidate(store, scope=SCOPE)
        assert len(result.concepts_created) == 2
        assert counts["candidate_reads"] == 2
        assert Counter(encoded) == Counter(t.content for t in traces)
        encoded.clear()
        c.memory_consolidate(store, scope=SCOPE)
        assert counts["candidate_reads"] == 4
        assert encoded == []


def test_manual_mcp_releases_store_during_cross_scope_preparation(tmp_path, monkeypatch):
    mcp = server.create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    sources = seed(store)
    prepared, release = Event(), Event()
    prepare = c._promotion_candidates

    def held(*args, **kwargs):
        snapshot = list(prepare(*args, **kwargs))
        prepared.set()
        assert release.wait(15)
        return snapshot

    monkeypatch.setattr(c, "_promotion_candidates", held)
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            work = workers.submit(mcp.tools["memory_consolidate"], scope=SCOPE)
            try:
                assert prepared.wait(10)
                read = workers.submit(mcp.tools["memory_lookup"], node_id=sources[0].id)
                assert read.result(timeout=5)["results"][0]["id"] == sources[0].id
                assert not work.done()
            finally:
                release.set()
            assert len(work.result(timeout=10)["concepts_created"]) == 1
    finally:
        mcp.auto_consolidation.close()
        store.close()


def test_later_scopes_remain_eligible_after_full_candidate_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "_PROMOTION_READ_BATCH", 2)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        vector = LocalEmbeddingModel(backend="hash").embed(TEXTS[0])
        seed(store)
        # Multiple early concepts are dissimilar and cannot be three scopes.
        for i in range(7):
            peer(store, f"project:aaa-{i}", embedding=[-v for v in vector])
        beta = peer(store, "project:zzz-beta", embedding=vector)
        gamma = peer(store, "project:zzz-gamma", embedding=vector)
        pages = []
        page = store.promotion_candidate_page

        def observed(**kwargs):
            nodes, ceiling = page(**kwargs)
            assert len(nodes) <= 2
            pages.append([node.id for node in nodes])
            return nodes, ceiling

        monkeypatch.setattr(store, "promotion_candidate_page", observed)
        result = c.memory_consolidate(store, scope=SCOPE)
        promoted, = result.concepts_promoted
        assert set(promoted.materializer_sources) == {result.concepts_created[0].id, beta.id, gamma.id}
        assert len(pages) == 6
        assert len({node_id for batch in pages for node_id in batch}) == 10
        assert pages[-1] == []
        assert c.memory_consolidate(store, scope=SCOPE).concepts_promoted == []


def test_promotion_pages_do_not_expand_to_new_insertions(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "_PROMOTION_READ_BATCH", 1)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        vector = LocalEmbeddingModel(backend="hash").embed(TEXTS[0])
        seed(store)
        peer(store, "project:beta", embedding=vector)
        page = store.promotion_candidate_page
        added = []

        def insert_after_selection(**kwargs):
            result = page(**kwargs)
            if not added:
                added.append(peer(store, "project:zzz-gamma", embedding=vector))
            return result

        monkeypatch.setattr(store, "promotion_candidate_page", insert_after_selection)
        assert c.memory_consolidate(store, scope=SCOPE).concepts_promoted == []
        assert len(c.memory_consolidate(store, scope=SCOPE).concepts_promoted) == 1
