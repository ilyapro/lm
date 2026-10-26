#!/usr/bin/env python3
"""Small ordinary-MCP write/first-read comparison on disposable databases.

Run the same script with each checkout's src on PYTHONPATH. It reports observed
outcomes, including misses, without changing ranking or draining maintenance.
The local model and tokenizer are warmed separately from the measured calls.
"""

import asyncio
import json
import os
from pathlib import Path
import tempfile
import time

os.environ.update(
    LIVING_MEMORY_EMBEDDING_BACKEND="auto", HF_HUB_OFFLINE="1",
    LM_AUTO_CONSOLIDATE_POLICY="off", LM_DECAY_SWEEP_INTERVAL_SEC="0", LM_RECALL_MAP="0",
)

from fastmcp import Client

from living_memory.chunking import chunk_text
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.retrieval import LAZY_CHUNK_NODE_LIMIT
from living_memory.server import create_mcp_server


SCOPE = "project:synthetic-backlog"
REGISTRY = "The container registry rejects image pulls because its TLS certificate expired. Renew it with certbot on the registry mirror."
QUERY = "Почему реестр контейнеров отвергает загрузку образов и как исправить истёкший сертификат?"
CASES = [
    ("registry", REGISTRY, QUERY),
    ("backup", "The database backup cannot be restored because the archive is encrypted. Retrieve the decryption key from the offline vault before restoring.",
     "Почему невозможно восстановить резервную копию базы и где взять ключ расшифровки архива?"),
    ("sensor", "The warehouse temperature sensor disconnects whenever its battery voltage drops. Replace the battery before pairing the sensor again.",
     "Почему отключается датчик температуры на складе и что заменить перед повторным подключением?"),
    ("long_registry", "Botanical field notes describe violet plants beside an amber pond. " * 160 + "\n\n" + REGISTRY, QUERY),
]


def stamp():
    return time.perf_counter(), time.process_time()


def elapsed(start):
    end = stamp()
    return {"wall_seconds": end[0] - start[0], "cpu_seconds": end[1] - start[1]}


def pending(store):
    return len(store.list_unchunked_nodes(scope=SCOPE, limit=100))


async def measure_case(name, content, query, *, teach=False):
    with tempfile.TemporaryDirectory(prefix="lm-write-read-") as directory:
        mcp = create_mcp_server(Path(directory) / "memory.sqlite3")
        store = mcp.memory_store
        report = {"name": name, "operation": "teach" if teach else "remember",
                  "content_chars": len(content), "expected_chunks": len(chunk_text(content)),
                  "backlog_writes": LAZY_CHUNK_NODE_LIMIT}
        write_responses = []
        try:
            async with Client(mcp) as client:
                async def remember(text):
                    response = (await client.call_tool("memory_remember", {
                        "content": text, "context": {"scope": SCOPE, "agent": "synthetic-write"},
                    })).data
                    write_responses.append(response)
                    return response["node"]["id"]

                start = stamp()
                original = await remember("The container registry certificate is current. Investigate image tags.") if teach else None
                for number in range(LAZY_CHUNK_NODE_LIMIT):
                    await remember(f"Botanical observation {number}: violet plants grow beside the amber pond.")
                report["preceding_writes"] = elapsed(start)
                start = stamp()
                if teach:
                    response = (await client.call_tool("memory_teach", {
                        "trace_id": original, "correction": content,
                        "context": {"scope": SCOPE, "agent": "synthetic-write"},
                    })).data
                    write_responses.append(response)
                    target = response["corrective_trace"]["id"]
                else:
                    target = await remember(content)
                report["target_write"] = elapsed(start)
                report["before_read"] = {"pending": pending(store), "target_chunks": store.count_node_chunks(target)}
                start = stamp()
                recalled = (await client.call_tool("memory_recall", {
                    "query": query, "scope": SCOPE, "depth": 0, "max_results": 3,
                })).data
                report["first_read"] = elapsed(start)
                event = store.get_recall_event(recalled["recall_event_id"])
                report["first_target_found"] = any(node["id"] == target for node in recalled["results"])
                report["first_scores"] = [
                    {key: row[key] for key in ("rank", "bm25_score", "vector_score", "methods")} | {"is_target": row["node_id"] == target}
                    for row in event.results
                ]
                report["after_read"] = {"pending": pending(store), "target_chunks": store.count_node_chunks(target)}
                report["all_writes_plus_first_read"] = {
                    metric: sum(report[stage][metric] for stage in ("preceding_writes", "target_write", "first_read"))
                    for metric in ("wall_seconds", "cpu_seconds")
                }
                report["response_bytes"] = {
                    "writes": len(json.dumps(write_responses, ensure_ascii=False).encode()),
                    "first_read": len(json.dumps(recalled, ensure_ascii=False).encode()),
                }
                looked = (await client.call_tool("memory_lookup", {"node_id": target})).data
                report["lookup_equal"] = looked["results"][0]["content"] == content
                if teach:
                    old = (await client.call_tool("memory_lookup", {"node_id": original})).data["results"][0]
                    report["old_superseded"] = old["superseded"] and target in old["superseded_by"]
                # Separate control after the measured first read. It cannot
                # retroactively turn that first outcome into a success.
                lexical = (await client.call_tool("memory_recall", {
                    "query": "certbot registry mirror" if "registry" in name else content,
                    "scope": SCOPE, "depth": 0, "max_results": 3,
                })).data
                lexical_event = store.get_recall_event(lexical["recall_event_id"])
                report["lexical_target_found"] = any(node["id"] == target for node in lexical["results"])
                report["lexical_target_bm25"] = next((row["bm25_score"] for row in lexical_event.results if row["node_id"] == target), None)
                windows = chunk_text(content)
                chunks = store.list_node_chunks(target)
                report["complete_current_chunks"] = store.node_chunks_are_current(target) and len(chunks) == len(windows)
                report["current_chunk_addresses"] = len(chunks) == len(windows) and all(
                    (chunk.char_start, chunk.char_end) == (window.char_start, window.char_end)
                    and content[chunk.char_start:chunk.char_end] == window.text
                    for chunk, window in zip(chunks, windows)
                )
                report["integrity"] = store.connection.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            mcp.auto_consolidation.close()
            store.close()
        return report


async def main():
    start = stamp()
    model = LocalEmbeddingModel()
    model.warmup()
    chunk_text(REGISTRY)
    report = {"encoder": model.encoder_id(), "warmup": elapsed(start),
              "lazy_limit": LAZY_CHUNK_NODE_LIMIT, "cases": []}
    if not report["encoder"].startswith("st:"):
        raise RuntimeError("This comparison requires the locally cached real encoder")
    for name, content, query in CASES:
        report["cases"].append(await measure_case(name, content, query))
    report["cases"].append(await measure_case("teach_registry", REGISTRY, QUERY, teach=True))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
