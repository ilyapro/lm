"""Ordinary MCP reads isolate optional refresh failures by scored generation."""

import asyncio
import json
import logging
import sqlite3

from fastmcp import Client
import pytest

from living_memory import server
from living_memory.retrieval import MemoryRecallService


SCOPE = "project:legacy-failure"
# The final condition shares an oversized paragraph with the unrelated prefix.
PREFIX = "Routine schedules and meeting notes. " * 90
FACTS = {
    "A": "Only during maintenance. Do not open the amber hatch unless the indicator is green.",
    "B": "Only during maintenance. Do not open the cobalt hatch unless the indicator is green.",
}


class Encoder:
    def embed(self, text):
        return [1.0, 0.0, 0.0] if text.endswith(" A") else [0.0, 1.0, 0.0]

    def embed_batch(self, texts):
        return [[1.0, 0.0, 0.0] if "amber" in text else
                [0.0, 1.0, 0.0] if "cobalt" in text else
                [0.0, 0.0, 1.0] for text in texts]


def chunk_rows(store, node_id):
    return [tuple(row) for row in store.connection.execute(
        "SELECT * FROM node_chunk_embeddings WHERE node_id=? ORDER BY chunk_index",
        (node_id,),
    )]


@pytest.mark.parametrize("fault", ["none", "encoder", "write", "common"])
def test_failed_a_preserves_independent_b_and_bounds_repeated_work(monkeypatch, caplog, fault):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_AUTO_CONSOLIDATE_POLICY": "off",
        "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_RECALL_MAP": "0",
        "LM_RECALL_MIN_SCORE": "off",
        "LM_DELIVERY_SESSION_DEDUP": "0",
        "LM_DELIVERY_SNIPPET_LADDER": "2400,1000,700,500,300,200",
    }.items():
        monkeypatch.setenv(key, value)
    encoder = Encoder()
    monkeypatch.setattr(server, "MemoryRecallService", lambda store: MemoryRecallService(
        store, embedder=encoder,
    ))
    mcp = server.create_mcp_server(":memory:")
    store = mcp.memory_store
    try:
        nodes = {label: store.create_node(
            level="concept", content=PREFIX + fact, context={"scope": SCOPE},
            representative_embedding=[0.0, 0.0, 1.0],
        ) for label, fact in FACTS.items()}
        old = store.append_trace("Open the cobalt hatch unconditionally.", {"scope": SCOPE})
        store.connect_nodes(nodes["B"].id, old.id, "supersedes")
        store.soft_delete_node(old.id, "replaced")
        foreign = store.create_node(
            level="concept", content=PREFIX + FACTS["B"],
            context={"scope": "project:excluded"}, representative_embedding=[0.0, 0.0, 1.0],
        )
        with store.transaction():
            store.connection.execute("UPDATE node_chunk_embeddings SET char_start=NULL, char_end=NULL")
        before = {label: chunk_rows(store, node.id) for label, node in nodes.items()}
        foreign_before = chunk_rows(store, foreign.id)
        attempts, batches, inserts, refreshing = [], [], [], []
        original_refresh = store.refresh_legacy_chunks

        def refresh(node_id, *args, **kwargs):
            attempts.append(next(label for label, node in nodes.items() if node.id == node_id))
            refreshing.append(attempts[-1])
            try:
                return original_refresh(node_id, *args, **kwargs)
            finally:
                refreshing.pop()

        def encode(texts):
            if not refreshing:
                return encoder.embed_batch(texts)
            label = refreshing[-1]
            batches.append({"source": label, "chunks": len(texts)})
            if fault == "common" or (fault == "encoder" and label == "A"):
                raise RuntimeError("synthetic encoder failure")
            return encoder.embed_batch(texts)

        monkeypatch.setattr(store, "refresh_legacy_chunks", refresh)
        # The store captured a bound encoder at registration; replace its actual writer.
        store.set_chunk_embedder(encode)
        if fault == "write":
            store.connection.create_function("observe_insert", 1, lambda i: inserts.append(i) or 0)
            with store.transaction():
                store.connection.execute("CREATE TABLE denied_index (id TEXT PRIMARY KEY)")
                store.connection.execute("INSERT INTO denied_index VALUES (?)", (nodes["A"].id,))
                store.connection.execute("""
                    CREATE TRIGGER deny_refresh BEFORE INSERT ON node_chunk_embeddings
                    WHEN NEW.node_id IN (SELECT id FROM denied_index)
                    BEGIN
                        SELECT observe_insert(NEW.chunk_index);
                        SELECT CASE WHEN NEW.chunk_index = 1
                            THEN RAISE(ABORT, 'synthetic write failure') END;
                    END
                """)

        async def scenario():
            observations = []
            async with Client(mcp) as client:
                for label in ["A", "B", "B"] + ["A", "B"] * 12:
                    response = (await client.call_tool("memory_recall", {
                        "query": "почему закрыт шлюз " + label,
                        "scope": SCOPE, "depth": 0, "max_results": 1,
                    })).data
                    entry, = response["results"]
                    assert entry["id"] == nodes[label].id
                    assert entry["scope"] == SCOPE and not entry["decayed"]
                    assert entry["delivery"] == "snippet"
                    assert entry["content_ref"]["node_id"] == nodes[label].id
                    assert not entry["superseded"]
                    observations.append({"source": label, "late_condition": FACTS[label] in entry["content"]})
                for label, node in nodes.items():
                    lookup = (await client.call_tool("memory_lookup", {
                        "node_id": node.id,
                    })).data
                    assert lookup["results"][0]["content"] == PREFIX + FACTS[label]
                lookup = (await client.call_tool("memory_lookup", {"node_id": old.id})).data
                assert lookup["results"][0]["decayed"]
                assert lookup["results"][0]["superseded_by"] == [nodes["B"].id]
            return observations

        with caplog.at_level(logging.WARNING, logger="living_memory.retrieval"):
            observations = asyncio.run(scenario())
        print(json.dumps({"fault": fault, "recalls": observations, "attempts": attempts,
                          "batches": batches, "insert_ordinals": inserts}, sort_keys=True))
        assert attempts == ["A", "B"]
        assert [batch["source"] for batch in batches] == ["A", "B"]
        assert all(1 < batch["chunks"] <= 32 for batch in batches)
        assert not observations[0]["late_condition"]
        assert not observations[1]["late_condition"]
        assert observations[2]["late_condition"] == (fault != "common")
        for observation in observations[3:]:
            assert observation["late_condition"] == (
                fault == "none" or (observation["source"] == "B" and fault != "common")
            )
        for label, node in nodes.items():
            failed = fault == "common" or (label == "A" and fault != "none")
            assert (chunk_rows(store, node.id) == before[label]) == failed
        assert chunk_rows(store, foreign.id) == foreign_before
        warnings = [record for record in caplog.records if "could not refresh selected legacy" in record.message]
        assert len(warnings) == (0 if fault == "none" else 2 if fault == "common" else 1)
        if fault == "write":
            assert inserts == [0, 1]
            assert isinstance(warnings[0].exc_info[1], sqlite3.IntegrityError)
    finally:
        mcp.auto_consolidation.close()
        store.close()
