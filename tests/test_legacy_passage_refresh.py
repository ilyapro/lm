"""Only needed legacy snippets pay for a bounded, generation-checked index refresh."""

from dataclasses import replace

import pytest

from living_memory import retrieval, server, storage
from living_memory.chunking import chunk_text
from living_memory.health_audit import ReadOnlyAuditStore
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore
from test_mcp_server import FakeMCP

SCOPE = "project:legacy-passage"
QUERY = "почему закрыт шлюз"
FACT = "Only during maintenance. Do not open the amber hatch unless the indicator is green."
CONTENT = ("Routine schedules and meeting notes. " * 30 + "\n\n") * 3 + FACT


class Encoder:
    def __init__(self):
        self.queries = []
        self.batches = []
        self.fail = False

    def embed(self, text):
        self.queries.append(text)
        return [1.0, 0.0, 0.0]

    def embed_batch(self, texts):
        self.batches.append(list(texts))
        if self.fail:
            raise RuntimeError("fixture encoder failure")
        return [[1.0, 0.0, 0.0] if "amber" in text else [0.0, 1.0, 0.0] for text in texts]


def legacy(store, content=CONTENT, *, level="concept", materializer=None):
    provenance = ({"cluster_key": "fixture"} if materializer == "cluster" else
                  {"strategy": materializer} if materializer else None)
    node = store.create_node(
        level=level, content=content, context={"scope": SCOPE},
        representative_embedding=[0.0, 0.0, 1.0],
        provenance=provenance,
    )
    with store.transaction():
        store.connection.execute(
            "UPDATE node_chunk_embeddings SET char_start=NULL, char_end=NULL WHERE node_id=?",
            (node.id,),
        )
    return node


def rows(store, node_id):
    return [tuple(row) for row in store.connection.execute(
        "SELECT * FROM node_chunk_embeddings WHERE node_id=? ORDER BY chunk_index", (node_id,),
    )]


def refresh(store, node, chunk_id=None, **limits):
    return store.refresh_legacy_chunks(
        node.id, node.content, chunk_id or store.list_node_chunks(node.id)[0].id,
        max_chars=limits.get("max_chars", retrieval.LEGACY_PASSAGE_MAX_CHARS),
        max_chunks=limits.get("max_chunks", retrieval.LEGACY_PASSAGE_MAX_CHUNKS),
    )


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "off")
    monkeypatch.setenv("LM_DELIVERY_SNIPPET_LADDER", "2400,1000,700,500,300,200")


@pytest.fixture
def app(tmp_path, monkeypatch):
    encoder = Encoder()
    services = []

    def build(store):
        service = MemoryRecallService(store, embedder=encoder)
        services.append(service)
        return service

    monkeypatch.setattr(server, "MemoryRecallService", build)
    mcp = server.create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    mcp.memory_store.set_chunk_embedder(encoder.embed_batch)
    yield mcp, encoder, services[0]
    mcp.memory_store.close()


def call(mcp, **kwargs):
    return mcp.tools["memory_recall"](QUERY, scope=SCOPE, depth=0, max_results=1, **kwargs)


def test_first_fallback_next_passage_one_query_search_and_no_repeated_encoding(app, monkeypatch):
    mcp, encoder, service = app
    store = mcp.memory_store
    node = legacy(store)
    untouched = legacy(store, CONTENT.replace("amber", "violet"))
    untouched_rows = rows(store, untouched.id)
    original = rows(store, node.id)
    encoder.queries.clear()
    encoder.batches.clear()
    searches = []
    scans = []
    pool = retrieval._pooled_chunk_similarities
    scan = store.iter_chunk_embedding_rows

    def counted_pool(*args, **kwargs):
        searches.append(1)
        return pool(*args, **kwargs)

    def counted_scan(**kwargs):
        scans.append(kwargs.get("scope"))
        return scan(**kwargs)

    monkeypatch.setattr(retrieval, "_pooled_chunk_similarities", counted_pool)
    monkeypatch.setattr(store, "iter_chunk_embedding_rows", counted_scan)
    first = call(mcp)["results"][0]
    assert first["id"] == node.id and first["delivery"] == "snippet"
    assert FACT not in first["content"]
    assert len(encoder.batches) == 1
    assert 1 < len(encoder.batches[0]) <= retrieval.LEGACY_PASSAGE_MAX_CHUNKS
    assert encoder.queries == [QUERY] and len(searches) == 1
    refreshed = rows(store, node.id)
    assert refreshed != original
    assert all(chunk.char_start is not None for chunk in store.list_node_chunks(node.id))
    assert rows(store, untouched.id) == untouched_rows
    assert store.get_node(node.id).embedding == node.embedding
    second = call(mcp)["results"][0]
    assert FACT in second["content"] and len(second["content"]) <= 2400
    third = call(mcp)["results"][0]
    assert FACT in third["content"]
    assert encoder.queries == [QUERY] * 3 and len(searches) == 3
    # The plan also initializes its empty global scope. Only the changed scope
    # is rescanned after refresh; the third recall reuses both cached scopes.
    assert scans.count(SCOPE) == 2 and scans.count("global") == 1
    assert len(encoder.batches) == 1 and rows(store, node.id) == refreshed
    assert rows(store, untouched.id) == untouched_rows
    assert mcp.tools["memory_lookup"](node_id=node.id)["results"][0]["content"] == CONTENT


@pytest.mark.parametrize("case", ["current", "small", "chars", "chunks", "schema", "full"])
def test_ineligible_reads_do_not_tokenize_or_encode(app, monkeypatch, case):
    mcp, encoder, service = app
    store = mcp.memory_store
    content = FACT if case == "small" else CONTENT
    if case == "chars":
        content = "Routine notes. " * 2400 + FACT
    node = legacy(store, content, level="schema" if case == "schema" else "concept")
    if case == "current":
        assert refresh(store, node)
    if case == "chunks":
        # An old tokenizer can have more windows than the current one.
        chunk = chunk_text(content)[0]
        store.replace_node_chunks(node.id, [
            (replace(chunk, chunk_index=i), [1.0, 0.0, 0.0]) for i in range(33)
        ])
        with store.transaction():
            store.connection.execute("UPDATE node_chunk_embeddings SET char_start=NULL, char_end=NULL")
    if case == "full":
        monkeypatch.setenv("LM_DELIVERY_SNIPPET_LADDER", "full")
    before = rows(store, node.id)
    encoder.batches.clear()
    monkeypatch.setattr(storage, "chunk_text", lambda *_: pytest.fail("ineligible tokenization"))
    for _ in range(2):
        assert call(mcp)["results"][0]["id"] == node.id
    assert encoder.batches == [] and rows(store, node.id) == before


def test_one_record_limit_after_selection_and_delivery(app, monkeypatch):
    mcp, encoder, service = app
    store = mcp.memory_store
    nodes = [legacy(store, CONTENT + f"\nRecord {i}.") for i in range(3)]
    results = service.memory_recall(QUERY, scope=SCOPE, depth=0, max_results=3, log_access=False)
    monkeypatch.setattr(service, "memory_recall", lambda *args, **kwargs: results)
    before = {node.id: rows(store, node.id) for node in nodes}
    encoder.batches.clear()
    call(mcp)
    assert sum(rows(store, node.id) != before[node.id] for node in nodes) == 1
    assert len(encoder.batches) == 1
    # A withheld result and a duplicate stub cannot consume the refresh slot.
    fresh = service.store.get_node(results[1].node_id)
    hidden = replace(results[1], withheld="below_threshold")
    monkeypatch.setattr(service, "memory_recall", lambda *args, **kwargs: [hidden])
    encoder.batches.clear()
    assert call(mcp)["results"][0]["delivery"] == "below_threshold"
    assert encoder.batches == [] and rows(store, fresh.id) == before[fresh.id]


@pytest.mark.parametrize("materializer", [None, "cluster", "cross-scope-promotion"])
def test_chunk_only_write_preserves_whole_node(tmp_path, materializer):
    encoder = Encoder()
    with MemoryStore(tmp_path / "memory.sqlite3", chunk_embedder=encoder.embed_batch) as store:
        node = legacy(store, materializer=materializer)
        assert node.materializer == materializer
        before = store.get_node(node.id).to_dict()
        assert refresh(store, node)
        assert store.get_node(node.id).to_dict() == before
        assert all(chunk.char_start is not None for chunk in store.list_node_chunks(node.id))


@pytest.mark.parametrize("change", ["content", "generation"])
@pytest.mark.parametrize("when", ["before", "during"])
def test_stale_source_or_generation_cannot_overwrite_a_concurrent_writer(tmp_path, change, when):
    encoder = Encoder()
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path, chunk_embedder=encoder.embed_batch) as store, MemoryStore(path) as writer:
        node = legacy(store)
        scored = store.list_node_chunks(node.id)[0].id
        replacement_rows = []

        def mutate():
            if change == "content":
                writer.update_node(node.id, content=FACT + " changed")
            else:
                writer.replace_node_chunks(node.id, [(c, [0.0, 1.0, 0.0]) for c in chunk_text(node.content)])
            replacement_rows[:] = rows(writer, node.id)

        def encode(texts):
            mutate()
            return encoder.embed_batch(texts)

        if when == "before":
            mutate()
            store.set_chunk_embedder(lambda _: pytest.fail("stale source encoded"))
        else:
            store.set_chunk_embedder(encode)
        assert not refresh(store, node, scored)
        assert rows(store, node.id) == replacement_rows
        assert store.get_chunk_passages([scored]) == {}


def test_new_tokenizer_chunk_bound_is_checked_before_encoding(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3", chunk_embedder=Encoder().embed_batch) as store:
        node = legacy(store)
        first = chunk_text(node.content)[0]
        store.replace_node_chunks(node.id, [(replace(first, char_start=-1), [1.0, 0.0, 0.0])])
        before = rows(store, node.id)
        store.set_chunk_embedder(lambda _: pytest.fail("oversized new generation encoded"))
        assert not refresh(store, node, max_chunks=1)
        assert rows(store, node.id) == before


def test_changed_encoder_cannot_readdress_the_old_score(app, monkeypatch):
    mcp, encoder, service = app
    node = legacy(mcp.memory_store)
    scored = service.memory_recall(QUERY, scope=SCOPE, depth=0, max_results=1, log_access=False)[0]
    monkeypatch.setattr(encoder, "embed_batch", lambda texts: [
        [0.6, 0.8, 0.0] if "amber" in text else [0.0, 1.0, 0.0] for text in texts
    ])
    mcp.memory_store.set_chunk_embedder(encoder.embed_batch)
    first = call(mcp, view="full")["results"][0]
    assert first["vector_score"] == round(scored.vector_score, 6)
    assert FACT not in first["node"]["content"]
    assert scored.semantic_passage is None
    assert mcp.memory_store.get_chunk_passages([scored.scored_chunk_id]) == {}
    fresh = service.memory_recall(QUERY, scope=SCOPE, depth=0, max_results=1, log_access=False)[0]
    assert fresh.node_id == node.id and fresh.vector_score < scored.vector_score
    assert fresh.semantic_passage is not None
    assert fresh.scored_chunk_id != scored.scored_chunk_id
    assert FACT in call(mcp)["results"][0]["content"]


@pytest.mark.parametrize("failure", ["encoder", "empty_vector", "write"])
def test_failure_rolls_back_and_does_not_repeat_on_each_read(app, monkeypatch, failure):
    mcp, encoder, service = app
    store = mcp.memory_store
    node = legacy(store)
    before = rows(store, node.id)
    encoder.batches.clear()
    if failure == "encoder":
        encoder.fail = True
    elif failure == "empty_vector":
        monkeypatch.setattr(encoder, "embed_batch", lambda texts: [[] for _ in texts])
        store.set_chunk_embedder(encoder.embed_batch)
    else:
        original = store._replace_node_chunks

        def fail_write(*args):
            original(*args)
            raise RuntimeError("fixture write failure")

        monkeypatch.setattr(store, "_replace_node_chunks", fail_write)
    first = call(mcp)["results"][0]
    assert FACT not in first["content"] and rows(store, node.id) == before
    monkeypatch.setattr(store, "refresh_legacy_chunks", lambda *a, **kw: pytest.fail("repeated failure"))
    assert call(mcp)["results"][0]["content"] == first["content"]
    assert store.get_node(node.id).content == CONTENT


def test_failed_winners_are_bounded_and_do_not_consume_the_next_source_slot(app):
    mcp, encoder, service = app
    store = mcp.memory_store
    node = legacy(store)
    other = legacy(store, CONTENT + "\nIndependent source.")
    results = service.memory_recall(QUERY, scope=SCOPE, depth=0, max_results=2, log_access=False)
    by_id = {result.node_id: result for result in results}
    chunks = store.list_node_chunks(node.id)
    assert 1 < len(chunks) <= retrieval.LEGACY_PASSAGE_MAX_CHUNKS
    winners = [replace(by_id[node.id], scored_chunk_id=chunk.id) for chunk in chunks]
    encoder.batches.clear()
    encoder.fail = True
    for _ in range(3):
        for winner in winners:
            service.refresh_legacy_passages([winner])
    assert len(encoder.batches) == len(chunks)
    assert all(len(batch) <= retrieval.LEGACY_PASSAGE_MAX_CHUNKS for batch in encoder.batches)
    encoder.fail = False
    service.refresh_legacy_passages([*winners, by_id[other.id]])
    assert len(encoder.batches) == len(chunks) + 1
    assert all(chunk.char_start is not None for chunk in store.list_node_chunks(other.id))
    assert all(chunk.char_start is None for chunk in store.list_node_chunks(node.id))


@pytest.mark.parametrize("change", ["content", "generation"])
def test_failed_row_does_not_suppress_replacement_source_or_index(app, change):
    mcp, encoder, service = app
    store = mcp.memory_store
    node = legacy(store)
    original_ids = {chunk.id for chunk in store.list_node_chunks(node.id)}
    encoder.fail = True
    assert FACT not in call(mcp)["results"][0]["content"]
    encoder.fail = False
    if change == "content":
        store.update_node(node.id, content=CONTENT + "\nUpdated source.", embedding=node.embedding)
    else:
        store.replace_node_chunks(node.id, [
            (chunk, vector) for chunk, vector in zip(
                chunk_text(CONTENT), encoder.embed_batch([c.text for c in chunk_text(CONTENT)]),
                strict=True,
            )
        ])
    with store.transaction():
        store.connection.execute(
            "UPDATE node_chunk_embeddings SET char_start=NULL, char_end=NULL WHERE node_id=?",
            (node.id,),
        )
    assert original_ids.isdisjoint(chunk.id for chunk in store.list_node_chunks(node.id))
    encoder.batches.clear()
    assert FACT not in call(mcp)["results"][0]["content"]
    assert len(encoder.batches) == 1
    assert FACT in call(mcp)["results"][0]["content"]
    assert len(encoder.batches) == 1


def test_readonly_and_unsupported_paths_do_no_model_work(tmp_path):
    encoder = Encoder()
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path, chunk_embedder=encoder.embed_batch) as store:
        node = legacy(store)
        scored = store.list_node_chunks(node.id)[0].id
        before = rows(store, node.id)
        store.connection.execute("PRAGMA query_only=ON")
        assert not refresh(store, node, scored)
    with ReadOnlyAuditStore.open_path(path) as store:
        assert not refresh(store, node, scored)
        assert rows(store, node.id) == before
    service = MemoryRecallService(object(), embedder=encoder)
    service.refresh_legacy_passages([])
