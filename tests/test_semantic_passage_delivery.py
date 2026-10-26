"""Semantic evidence reaches bounded delivery without another vector search."""

from dataclasses import replace
import hashlib
import sqlite3

import pytest

from living_memory import retrieval
from living_memory.chunking import TextChunk, chunk_text
from living_memory.delivery import DeliveryHistory, recall_delivery_records, shape_recall_results
from living_memory.health_audit import ReadOnlyAuditStore
from living_memory.models import SourcePassage
from living_memory.retrieval import MemoryRecallService, RecallResult
from living_memory.storage import MemoryStore


QUERY = "почему закрыт шлюз"
FACT = "Only during maintenance. Do not open the amber hatch unless the indicator is green."
BACKGROUND = "The archive contains ordinary meeting schedules. " * 24
SCOPE = "project:semantic-passage"


def shape(results, budget=300, history=None, **kwargs):
    return shape_recall_results(
        results, delivery_history=history or DeliveryHistory(),
        snippet_max_chars=budget, snippet_ladder=None, context_value_max_chars=160,
        session_dedup=True, query=QUERY, view="knowledge", **kwargs,
    )


class QueryEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return [1.0, 0.0, 0.0]


def put_match(store, content, start, end, *, scope=SCOPE, level="trace"):
    store.set_chunk_embedder(lambda texts: [[0.0, 1.0, 0.0] for _ in texts])
    node = store.create_node(level=level, content=content, context={"scope": scope})
    # Overlapping windows, with the winning window starting inside the clause.
    spans = [(0, min(200, len(content))), (start, end), (max(0, start - 10), end)]
    store.replace_node_chunks(node.id, [
        (TextChunk(content[a:b], index, index * 30, index * 30 + 60, a, b), vector)
        for index, ((a, b), vector) in enumerate(zip(
            spans, ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.8, 0.6, 0.0]), strict=True,
        ))
    ])
    return node


def recall(service, scope=SCOPE):
    return service.memory_recall(
        QUERY, scope=scope, depth=0, max_results=10, log_access=False, log_event=False,
    )


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "off")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")


@pytest.mark.parametrize("numpy", [True, False])
@pytest.mark.parametrize("position", [0, 1, 3])
@pytest.mark.parametrize("budget", [200, 700, 2400])
def test_semantic_position_budget_and_qualifiers(tmp_path, monkeypatch, numpy, position, budget):
    if not numpy:
        monkeypatch.setattr(retrieval, "_np", None)
    paragraphs = [BACKGROUND] * 4
    paragraphs.insert(position, FACT)
    content = "\n\n".join(paragraphs)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        start = content.index("open the amber")
        node = put_match(store, content, start, start + len("open the amber hatch"))
        embedder = QueryEmbedder()
        service = MemoryRecallService(store, embedder=embedder)
        results = recall(service)
        result = next(item for item in results if item.node_id == node.id)
        assert result.bm25_score == 0
        assert result.semantic_passage is not None
        # Warm recall and delivery reuse the prepared corpus and query evidence.
        monkeypatch.setattr(store, "iter_chunk_embedding_rows", lambda **kw: pytest.fail("corpus reread"))
        monkeypatch.setattr(service, "_rechunk_node", lambda node: pytest.fail("re-embedding"))
        warm = recall(service)
        assert warm == results
        before = list(embedder.calls)
        body = shape(warm, budget)[0]["content"]
        assert FACT in body
        assert len(body) <= budget
        assert embedder.calls == before == [QUERY, QUERY]


@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_semantic_source_record_and_adjacent_condition(tmp_path, view):
    heading = "Evidence: hatch cases (case records, not steps)\n"
    prefix = "- 01SEMANTICSOURCE0000000001: "
    # The paragraph exceeds the budget; its previous sentence carries a condition.
    content = heading + prefix + BACKGROUND + FACT + " Routine archive notes." * 60
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        start = content.index("open the amber")
        node = put_match(store, content, start, start + 20)
        results = recall(MemoryRecallService(store, embedder=QueryEmbedder()))
        shaped = shape_recall_results(
            results, delivery_history=DeliveryHistory(), snippet_max_chars=350,
            snippet_ladder=None, context_value_max_chars=160, session_dedup=False,
            query=QUERY, view=view,
        )
        entry = shaped[0]
        body = entry.get("node", entry)["content"]
        assert body.startswith(heading)
        assert prefix + FACT in body
        assert len(body) <= 350
        assert entry["content_ref"]["node_id"] == node.id


@pytest.mark.parametrize("condition_before", [False, True])
@pytest.mark.parametrize("budget", [1000, 700, 500, 300, 200, 160])
def test_cross_paragraph_condition_precedes_remote_context(tmp_path, condition_before, budget):
    permission = "The dashboard may restart automatically after maintenance."
    condition = ("On the operator workstation, this permission requires an explicit operator "
                 "request; otherwise the dashboard must remain running.")
    paragraphs = [condition, permission] if condition_before else [permission, condition]
    background = "Routine dashboard maintenance reports describe the archive. " * 25
    content = "\n\n".join([background, *paragraphs, background])
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        start = content.index(permission)
        node = put_match(store, content, start, start + len(permission))
        results = recall(MemoryRecallService(store, embedder=QueryEmbedder()))
        entry = shape(results, budget)[0]
        body = entry["content"]
        assert permission in body
        assert len(body) <= budget
        if budget >= 300:
            assert condition in body
            assert (body.index(condition) < body.index(permission)) == condition_before
        else:
            assert condition not in body
        # The other adjacent paragraph cannot fit even at 1000. Its omission
        # remains explicit; retaining one nearby condition proves no completeness.
        assert body.endswith("\n[Incomplete source context]")
        assert entry["content_ref"]["node_id"] == node.id
        assert store.get_node(node.id).content == content


def test_exact_generation_and_content_binding(tmp_path, monkeypatch):
    content = BACKGROUND + "\n\n" + FACT
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        start = content.index("open the amber")
        node = put_match(store, content, start, start + 20, level="concept")
        service = MemoryRecallService(store, embedder=QueryEmbedder())
        result = recall(service)[0]
        original = store.get_chunk_passages

        def replace_generation(ids):
            # Same content, new windows and row ids between scoring and resolving.
            store.replace_node_chunks(node.id, [(c, [1.0, 0.0, 0.0]) for c in chunk_text(content)])
            return original(ids)

        monkeypatch.setattr(store, "get_chunk_passages", replace_generation)
        assert recall(service)[0].semantic_passage is None
        changed = store.update_node(node.id, content=FACT + "\n\n" + BACKGROUND)
        stale = replace(result, node=changed)
        assert shape([stale]) == shape([replace(stale, semantic_passage=None)])
        # A later recall has fresh coordinates for the changed content.
        monkeypatch.setattr(store, "get_chunk_passages", original)
        fresh = recall(service)[0]
        assert fresh.semantic_passage is not None
        assert fresh.semantic_passage.content_fingerprint != result.semantic_passage.content_fingerprint


def test_missing_or_invalid_addresses_keep_lexical_fallback(tmp_path):
    content = BACKGROUND + "\n\n" + FACT
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = put_match(store, content, content.index(FACT), len(content))
        result = recall(MemoryRecallService(store, embedder=QueryEmbedder()))[0]
        fallback = shape([replace(result, semantic_passage=None)])
        for passage in (
            SourcePassage("stale", 0, 100),
            SourcePassage(hashlib.sha256(content.encode()).hexdigest(), -1, 100),
            SourcePassage(hashlib.sha256(content.encode()).hexdigest(), 0, len(content) + 1),
        ):
            assert shape([replace(result, semantic_passage=passage)]) == fallback
        store.connection.execute("UPDATE node_chunk_embeddings SET char_start=NULL, char_end=NULL")
        store.connection.commit()
        legacy = recall(MemoryRecallService(store, embedder=QueryEmbedder()))[0]
        assert legacy.semantic_passage is None
        assert shape([legacy]) == fallback
        assert store.get_node(node.id).content == content


def test_legacy_migration_preserves_vectors_and_readonly_fallback(tmp_path):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store:
        node = put_match(store, BACKGROUND + FACT, len(BACKGROUND), len(BACKGROUND + FACT))
        before = [(row[0], row[1], bytes(row[2]), row[3]) for row in store.iter_chunk_embedding_rows(include_identity=True)]
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE node_chunk_embeddings DROP COLUMN char_start")
        conn.execute("ALTER TABLE node_chunk_embeddings DROP COLUMN char_end")
    with ReadOnlyAuditStore.open_path(path) as store:
        assert store.get_chunk_passages([before[0][3]]) == {}
    with MemoryStore(path) as store:
        after = [(row[0], row[1], bytes(row[2]), row[3]) for row in store.iter_chunk_embedding_rows(include_identity=True)]
        assert after == before
        assert all(chunk.char_start is None for chunk in store.list_node_chunks(node.id))
        assert store.node_chunks_are_current(node.id)


def test_semantic_excerpt_session_coverage_validity_and_recipe(tmp_path):
    content = BACKGROUND + "\n\n" + FACT + "\n\n" + BACKGROUND
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        start = content.index(FACT)
        node = put_match(store, content, start, start + len(FACT))
        result = recall(MemoryRecallService(store, embedder=QueryEmbedder()))[0]
        result = replace(result, superseded=True, correction_ids=("visible", "hidden"))
        first = shape([result], correction_refs={node.id: ["visible"]})
        assert first[0]["superseded"] is True
        assert first[0]["superseded_by"] == ["visible"]
        records = recall_delivery_records([result], first)
        history = DeliveryHistory([{"node_id": node.id, "delivery": records[node.id]}])
        assert shape([result], history=history)[0]["delivery"] == "session_duplicate"
        different = replace(result, semantic_passage=SourcePassage(
            result.semantic_passage.content_fingerprint, 0, 60,
        ))
        assert shape([different], history=history)[0]["delivery"] == "snippet"
        recipe = replace(result, node=replace(node, level="schema"))
        assert shape([recipe], budget=80)[0]["content"] == content
        withheld = replace(result, withheld="below_threshold")
        assert shape([withheld])[0]["content"] == ""
        assert store.get_node(node.id).content == content


@pytest.mark.parametrize("scope", [None, SCOPE])
@pytest.mark.parametrize("numpy", [True, False])
def test_scoped_gather_keeps_winning_address(tmp_path, monkeypatch, scope, numpy):
    if not numpy:
        monkeypatch.setattr(retrieval, "_np", None)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for item_scope in (SCOPE, "global", "project:hidden"):
            content = BACKGROUND + "\n\n" + item_scope + "\n\n" + FACT
            put_match(store, content, content.index(FACT), len(content), scope=item_scope)
        service = MemoryRecallService(store, embedder=QueryEmbedder())
        for result in recall(service, scope):
            assert result.semantic_passage is not None
            assert FACT in shape([result])[0]["content"]
            if scope is not None:
                assert result.node.scope != "project:hidden"


@pytest.mark.parametrize("view", ["knowledge", "full"])
@pytest.mark.parametrize("budget", [350, 2400])
@pytest.mark.parametrize("case", [
    "long_predecessor", "short_condition", "long_successor", "paragraph_break",
    "overlapped_paragraphs", "evidence_records", "oversized_sentence",
    "oversized_overlap", "lexical_only",
])
def test_source_aware_budget_through_mcp(tmp_path, monkeypatch, view, budget, case):
    import asyncio

    from fastmcp import Client
    from living_memory import server
    from test_transport_identity import _structured

    statement = "Do not open the amber hatch unless the indicator is green."
    condition = "Only during maintenance."
    caveat = "Never bypass the pressure interlock."
    long_sentence = "The archive records " + "ordinary maintenance context " * 105 + "before proceeding."
    heading = "Evidence: hatch cases (case records, not steps)\n"
    source_a = "- 01SEMANTICSOURCE0000000001: "
    source_b = "- 01SEMANTICSOURCE0000000002: "
    partial = case in {"oversized_sentence", "oversized_overlap"}
    required = [statement]
    if case == "long_predecessor":
        content = long_sentence + " " + statement + " Routine closing notes."
    elif case in {"short_condition", "lexical_only"}:
        content = long_sentence + " " + condition + " " + statement + " " + caveat
        required = [condition, statement, caveat]
    elif case == "long_successor":
        content = condition + " " + statement + " " + long_sentence
        required = [condition, statement]
    elif case == "paragraph_break":
        content = long_sentence + "\n\n" + statement
    elif case == "overlapped_paragraphs":
        content = long_sentence + "\n\n" + condition + "\n\n" + statement + "\n\n" + caveat
        required = [condition, statement, caveat]
    elif case == "evidence_records":
        content = heading + source_a + long_sentence + "\n\n" + condition + "\n" + source_b + statement + " " + caveat
        required = [source_a + condition, source_b + statement, caveat]
    elif case == "oversized_sentence":
        content = "Do not " + "while inspection is still pending " * 105 + statement.removeprefix("Do not ")
    else:
        content = condition + " " + statement + " " + long_sentence

    start = content.index("open the amber")
    end = start + len("open the amber hatch")
    if case in {"overlapped_paragraphs", "evidence_records", "oversized_overlap"}:
        start = content.index(condition)
        end = len(content) - 1

    class LexicalEmbedder(QueryEmbedder):
        def embed(self, text):
            return [0.0, 0.0, 0.0]

    embedder = LexicalEmbedder() if case == "lexical_only" else QueryEmbedder()
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.delenv("LM_DELIVERY_SNIPPET_CHARS", raising=False)
    if budget == 2400:
        monkeypatch.delenv("LM_DELIVERY_SNIPPET_LADDER", raising=False)
    else:
        monkeypatch.setenv("LM_DELIVERY_SNIPPET_LADDER", str(budget))
    monkeypatch.setattr(server, "MemoryRecallService", lambda store: MemoryRecallService(store, embedder=embedder))
    mcp = server.create_mcp_server(tmp_path / "memory.sqlite3")
    node = put_match(mcp.memory_store, content, start, end)
    query = "open amber hatch indicator green" if case == "lexical_only" else QUERY
    direct = MemoryRecallService(mcp.memory_store, embedder=embedder).memory_recall(
        query, scope=SCOPE, depth=0, max_results=1, log_access=False, log_event=False,
    )[0]
    if case == "lexical_only":
        assert direct.semantic_passage is None and direct.bm25_score > 0
    else:
        assert direct.bm25_score == 0 and direct.vector_score > 0
        assert direct.semantic_passage.char_start == start
        assert direct.semantic_passage.char_end == end

    async def scenario():
        async with Client(mcp) as client:
            recalled = _structured(await client.call_tool("memory_recall", {
                "query": query, "scope": SCOPE, "depth": 0, "max_results": 1, "view": view,
            }))
            entry = recalled["results"][0]
            body = entry.get("node", entry)["content"]
            assert entry["delivery"] == "snippet"
            assert len(body) <= budget
            if partial:
                assert body.endswith("…\n[Incomplete passage]")
                prefix = body.removesuffix("\n[Incomplete passage]")[:-1]
                assert content.startswith(prefix)
                if case == "oversized_sentence":
                    assert body.startswith("Do not ")
                    assert "open the amber hatch" not in body
            else:
                assert all(text in body for text in required)
                notice = ("[Incomplete source context]" if case in {
                    "paragraph_break", "overlapped_paragraphs", "evidence_records",
                } else "[Context omitted]")
                assert body.endswith("\n" + notice)
                assert "[Incomplete passage]" not in body
                if case == "evidence_records":
                    assert body.startswith(heading)
                    assert source_a + statement not in body
            assert entry["content_ref"] == {"node_id": node.id, "full_content_chars": len(content)}
            lookup = _structured(await client.call_tool("memory_lookup", {"node_id": node.id}))
            assert lookup["results"][0]["content"] == content

    try:
        asyncio.run(scenario())
    finally:
        mcp.auto_consolidation.close()
        mcp.memory_store.close()
