"""Optional recall context keeps source paragraphs through ordinary MCP reads."""

import asyncio
import hashlib
import json

from fastmcp import Client
import pytest

from living_memory import delivery, server
from living_memory.chunking import TextChunk
from living_memory.retrieval import MemoryRecallService


QUERY = "rollback archive inspection"
SCOPE = "project:optional-context"
CONDITION = (
    "Only after the incident owner signs the recovery acceptance, confirms that all "
    "downstream readers have moved to the replacement snapshot, and verifies the "
    "independent restore copy may the following instruction be performed."
)
ACTION = "Delete the rollback archive."
ANCHOR = "For rollback archive checks, first compare the manifest checksum and inspect the recovery log."
COMPLEMENT = "The rollback archive checksum is stored in the recovery manifest."
PRECEDING = "The catalog lists completed maintenance windows."
FOLLOWING = "The index records the current archive identifier."


class Encoder:
    def embed(self, text):
        return [1.0, 0.0, 0.0]

    def embed_batch(self, texts):
        return [[0.0, 1.0, 0.0] for _ in texts]


def put_match(store, content, anchor):
    node = store.create_node(level="trace", content=content, context={"scope": SCOPE})
    start = content.index(anchor)
    end = start + len(anchor)
    spans = [(0, min(200, len(content))), (start, end), (max(0, start - 10), end)]
    store.replace_node_chunks(node.id, [
        (TextChunk(content[left:right], index, index * 30, index * 30 + 60, left, right), vector)
        for index, ((left, right), vector) in enumerate(zip(
            spans, ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.8, 0.6, 0.0]), strict=True,
        ))
    ])
    return node


@pytest.mark.parametrize("view", ["knowledge", "full"])
@pytest.mark.parametrize("placement", ["explicit_300", "default_first", "default_fifth"])
@pytest.mark.parametrize("remote", [CONDITION + " " + ACTION, COMPLEMENT], ids=["conditional", "complement"])
def test_optional_context_keeps_source_paragraph(monkeypatch, view, placement, remote):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_AUTO_CONSOLIDATE_POLICY": "off",
        "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_RECALL_MAP": "0",
        "LM_RECALL_MIN_SCORE": "off",
        "LM_DELIVERY_SESSION_DEDUP": "0",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("LM_DELIVERY_SNIPPET_CHARS", raising=False)
    monkeypatch.delenv("LM_DELIVERY_SNIPPET_LADDER", raising=False)
    if placement == "explicit_300":
        monkeypatch.setenv("LM_DELIVERY_SNIPPET_LADDER", "300")
    else:
        assert delivery.snippet_ladder_from_env() == (2400, 1000, 700, 500, 300, 200)
    encoder = Encoder()
    monkeypatch.setattr(server, "MemoryRecallService", lambda store: MemoryRecallService(store, embedder=encoder))
    mcp = server.create_mcp_server(":memory:")
    store = mcp.memory_store
    content = "\n\n".join([
        remote, PRECEDING, ANCHOR, FOLLOWING,
        "The quarterly catalog contains " + "ordinary maintenance background " * 9 + "for completeness.",
    ])
    position = 4 if placement == "default_fifth" else 0
    try:
        store.set_chunk_embedder(encoder.embed_batch)
        node = put_match(store, content, ANCHOR)
        leaders = [put_match(store, f"Rollback archive inspection checklist {index}.", "Rollback archive inspection")
                   for index in range(position)]
        direct = MemoryRecallService(store, embedder=encoder).memory_recall(
            QUERY, scope=SCOPE, depth=0, max_results=position + 1, log_access=False, log_event=False,
        )
        assert direct[position].node_id == node.id
        assert {result.node_id for result in direct[:position]} == {leader.id for leader in leaders}
        selected = direct[position]
        assert selected.semantic_passage.char_start == content.index(ANCHOR)
        assert selected.semantic_passage.char_end == content.index(ANCHOR) + len(ANCHOR)

        async def scenario():
            async with Client(mcp) as client:
                response = (await client.call_tool("memory_recall", {
                    "query": QUERY, "scope": SCOPE, "depth": 0,
                    "max_results": position + 1, "view": view,
                })).data
                entries = response["results"]
                assert [entry.get("node", entry)["id"] for entry in entries] == [result.node_id for result in direct]
                assert all(entry["delivery"] == "full" for entry in entries[:position])
                lookup = (await client.call_tool("memory_lookup", {"node_id": node.id})).data
                looked_up, = lookup["results"]
                assert looked_up["content"] == content
                return entries[position], looked_up

        entry, looked_up = asyncio.run(scenario())
        body = entry.get("node", entry)["content"]
        print(json.dumps({
            "view": view, "placement": placement, "conditional": remote != COMPLEMENT,
            "source_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "anchor": [selected.semantic_passage.char_start, selected.semantic_passage.char_end],
            "scores": {"bm25": selected.bm25_score, "vector": selected.vector_score},
            "body": body, "body_chars": len(body), "delivery": entry["delivery"],
            "lookup_exact": looked_up["content"] == content,
        }, sort_keys=True))
        assert looked_up["scope"] == SCOPE and not looked_up["decayed"] and not looked_up["superseded"]
        assert ANCHOR in body
        if placement == "default_first":
            assert entry["delivery"] == "full"
            assert body == content
        else:
            assert entry["delivery"] == "snippet"
            assert len(body) <= 300
            assert entry["content_ref"] == {"node_id": node.id, "full_content_chars": len(content)}
            paragraphs = ([COMPLEMENT] if remote == COMPLEMENT else []) + [PRECEDING, ANCHOR, FOLLOWING]
            assert body == "\n\n".join(paragraphs) + "\n[Context omitted]"
    finally:
        mcp.auto_consolidation.close()
        store.close()
