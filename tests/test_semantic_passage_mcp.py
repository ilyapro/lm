"""Local MiniLM and a real MCP client deliver a semantic-only late answer."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_chunk_tail_regression import QUERY, TAIL_ANSWER, TAIL_HEAD


PROBE = r'''
import asyncio, json, sys, tempfile
from pathlib import Path
from fastmcp import Client
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore
from living_memory.server import create_mcp_server

payload = json.load(sys.stdin)
model = LocalEmbeddingModel(backend="auto")
model.warmup()
if model._sentence_transformer() is None:
    print(json.dumps({"model_available": False}))
    raise SystemExit(0)

with tempfile.TemporaryDirectory(prefix="lm-semantic-mcp-") as directory:
    db = Path(directory) / "memory.sqlite3"
    with MemoryStore(db) as store:
        store.set_chunk_embedder(model.embed_batch)
        node = store.create_node(level="trace", content=payload["content"], context={"scope": "project:semantic-mcp"})
        result = MemoryRecallService(store, embedder=model).memory_recall(
            payload["query"], scope=node.scope, depth=0, max_results=1,
            log_access=False, log_event=False,
        )[0]
        bm25_score = result.bm25_score
        vector_score = result.vector_score
        if payload.get("legacy"):
            with store.transaction():
                store.connection.execute("UPDATE node_chunk_embeddings SET char_start=NULL, char_end=NULL")
    mcp = create_mcp_server(db)
    def structured(reply):
        value = reply.structured_content
        return value["result"] if set(value) == {"result"} else value
    async def scenario():
        cases = []
        first = None
        if payload.get("legacy"):
            async with Client(mcp) as client:
                first = structured(await client.call_tool("memory_recall", {
                    "query": payload["query"], "scope": node.scope, "depth": 0, "max_results": 1,
                }))["results"][0]
            assert all(c.char_start is not None for c in mcp.memory_store.list_node_chunks(node.id))
        for view in ("knowledge", "full"):
            for query in (payload["query"], "registry certificate certbot mirror expired"):
                async with Client(mcp) as client:
                    response = structured(await client.call_tool("memory_recall", {
                        "query": query, "scope": node.scope, "depth": 0,
                        "max_results": 1, "view": view,
                    }))
                    entry = response["results"][0]
                    body = entry.get("node", entry)
                    looked = structured(await client.call_tool("memory_lookup", {"node_id": node.id}))
                    cases.append({
                        "view": view, "query": query, "delivery": entry["delivery"],
                        "content": body["content"], "content_ref": entry["content_ref"],
                        "lookup_equal": looked["results"][0]["content"] == payload["content"],
                    })
        return cases, first
    try:
        cases, first = asyncio.run(scenario())
    finally:
        mcp.memory_store.close()
print(json.dumps({"model_available": True, "bm25_score": bm25_score, "vector_score": vector_score, "cases": cases, "first": first}))
'''


@pytest.mark.parametrize("legacy", [False, True])
def test_local_model_semantic_and_lexical_mcp_delivery(legacy):
    environment = dict(os.environ)
    environment.pop("LIVING_MEMORY_EMBEDDING_BACKEND", None)
    environment.pop("LIVING_MEMORY_CHUNK_TOKENIZER", None)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["LM_DELIVERY_SNIPPET_LADDER"] = "2400,1000,700,500,300,200"
    environment["LM_RECALL_MIN_SCORE"] = "off"
    root = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = os.pathsep.join([str(root / "src"), environment.get("PYTHONPATH", "")])
    content = TAIL_HEAD * 3 + TAIL_ANSWER
    completed = subprocess.run(
        [sys.executable, "-B", "-c", PROBE],
        input=json.dumps({"content": content, "query": QUERY, "legacy": legacy}),
        capture_output=True, text=True, env=environment, timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    if not report["model_available"]:
        pytest.skip("The real local MiniLM is unavailable; hash vectors cannot establish this control")
    assert report["bm25_score"] == 0.0
    assert report["vector_score"] > 0
    if legacy:
        assert report["first"]["delivery"] == "snippet"
        assert TAIL_ANSWER not in report["first"]["content"]
    for case in report["cases"]:
        assert case["delivery"] == "snippet"
        assert TAIL_ANSWER in case["content"]
        assert len(case["content"]) <= 2400
        assert case["content_ref"]["full_content_chars"] == len(content)
        assert case["lookup_equal"]
