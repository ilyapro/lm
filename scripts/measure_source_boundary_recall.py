#!/usr/bin/env python3
"""Ordinary MCP recall and write/first-read cost on public authored sources.

Run with the checkout being measured on PYTHONPATH. Uses the cached real model,
default retrieval/delivery policy and disposable databases; never drains an index
or changes ranking, scope credit, thresholds or production memory. Reports misses
as well as matches. Model/tokenizer warmup is outside the measured calls.
"""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

os.environ.update(HF_HUB_OFFLINE="1", LIVING_MEMORY_EMBEDDING_BACKEND="auto", LIVING_MEMORY_CHUNK_TOKENIZER="model", OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
from fastmcp import Client
from living_memory.chunking import chunk_text, get_tokenizer, count_tokens
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.server import create_mcp_server

query = "Почему реестр контейнеров отвергает загрузку образов и как исправить истёкший сертификат?"

answer = "The container registry rejects image pulls because its TLS certificate expired. Renew it with certbot on the registry mirror."

ops = [
    "The deployment report records several independent observations from the maintenance window. The build cache moved to a dedicated SSD, reducing repeated dependency downloads. Cache contents are disposable and must never be used as the only copy of generated release packages.",
    "Database backups completed before the schema migration. The restore drill used an isolated instance and checked both row counts and application login. Keep the previous backup until the new release has served normal traffic without errors for a full day.",
    "The image processing queue briefly grew after a customer uploaded a large album. Worker concurrency remained unchanged because CPU use was already high. The backlog cleared after the batch finished; increasing parallelism would have competed with the interactive API.",
    "Monitoring showed intermittent packet loss between the office network and the staging host. Public health checks from two other networks remained green. Keep this observation separate from the application error rate until a trace connects the failed requests to that route.",
    "The documentation site now has working keyboard navigation and a visible focus outline. The accessibility check covered the search dialog and language switcher. Larger font settings still need to be included whenever a shared navigation component changes.",
    "A storage alert came from retained debug logs rather than customer uploads. The log rotation job was disabled during incident analysis and was restored after the relevant evidence was copied. Production attachments and database files were not removed.",
    "Release notes should list the user-visible change and the supported upgrade path. Internal investigation transcripts belong in the incident archive. The next operator will need the current rollback command and the exact artifact identifier, not a chronological account of every experiment.",
]

botanical = "Botanical observation 0: violet plants grow beside the amber pond."

related = "The registry accepts image pulls after its TLS certificate was renewed. Current upload failures come from expired authentication credentials, which must be replaced in the worker environment."

controls = {
    "short_two_topics": ops[0] + "\n\n" + answer,
    "maintenance_report_tail": "\n\n".join(ops + [answer]),
    "maintenance_report_middle": "\n\n".join(ops[:3] + [answer] + ops[3:]),
    "same_paragraph_constraint": "The deployment worker may restart automatically after a successful health check. This permission does not cover the dashboard: do not restart the dashboard on the operator workstation without an explicit operator request.",
    "cross_paragraph_constraint": "The dashboard may restart after the maintenance window.\n\nThis permission does not apply to the operator workstation; there the dashboard must remain running until the operator explicitly requests a restart.",
}
constants = {name: globals()[name] for name in ("query", "answer", "ops", "botanical", "related", "controls")}
SCOPE = "project:source-boundary-probe"

def stamp(): return (time.perf_counter(), time.process_time())
def elapsed(start):
    end = stamp()
    return {"wall_ms": (end[0]-start[0])*1000, "cpu_ms": (end[1]-start[1])*1000}

async def case(name, content, query, distractor):
    with tempfile.TemporaryDirectory(prefix="lm-boundaries-") as directory:
        mcp = create_mcp_server(Path(directory) / "memory.sqlite3")
        store = mcp.memory_store
        labels = {}
        counts = []
        original_embed = store._chunk_embedder
        def embed(texts):
            counts.append({"texts": len(texts), "tokens": sum(count_tokens(text) for text in texts)})
            return original_embed(texts)
        store.set_chunk_embedder(embed)
        report = {"name": name, "source": content, "query": query}
        try:
            async with Client(mcp) as client:
                for label, text in [("related_wrong", distractor), ("botanical", constants["botanical"]), ("target", content)]:
                    start = stamp()
                    response = (await client.call_tool("memory_remember", {"content": text, "context": {"scope": SCOPE, "source": label}})).data
                    labels[response["node"]["id"]] = label
                    if label == "target":
                        report["write"] = elapsed(start)
                        report["write_receipt"] = response
                        target = response["node"]["id"]
                report["chunk_embedding_work_before_recall"] = counts.copy()
                start = stamp()
                recalled = (await client.call_tool("memory_recall", {"query": query, "scope": SCOPE, "max_results": 3})).data
                report["first_recall"] = elapsed(start)
                report["write_plus_first_recall"] = {key: report["write"][key] + report["first_recall"][key] for key in report["write"]}
                event = store.get_recall_event(recalled["recall_event_id"])
                report["scores"] = [dict(row) | {"label": labels.get(row["node_id"], "other")} for row in event.results]
                report["delivered"] = [row | {"label": labels.get(row["id"], "other")} for row in recalled["results"]]
                lookup = (await client.call_tool("memory_lookup", {"node_id": target})).data
                report["exact_lookup"] = lookup["results"][0]["content"] == content
            async with Client(mcp) as client:
                lexical_query = "certbot registry mirror" if "constraint" not in name else "operator workstation explicit request dashboard"
                recalled = (await client.call_tool("memory_recall", {"query": lexical_query, "scope": SCOPE, "max_results": 3})).data
                report["lexical"] = [row | {"label": labels.get(row["id"], "other")} for row in recalled["results"]]
            windows = chunk_text(content)
            rows = store.list_node_chunks(target)
            report["chunks"] = [{"text": w.text, "tokens": w.token_count, "standalone_tokens": count_tokens(w.text), "token_start": w.token_start, "token_end": w.token_end, "char_start": w.char_start, "char_end": w.char_end} for w in windows]
            report["stored_coordinates"] = len(windows) == len(rows) and all((r.token_start,r.token_end,r.char_start,r.char_end)==(w.token_start,w.token_end,w.char_start,w.char_end) for r,w in zip(rows,windows))
            report["coverage"] = set().union(*(set(range(w.token_start,w.token_end)) for w in windows)) == set(range(count_tokens(content)))
            report["untruncated"] = all(count_tokens(w.text)+2 <=128 for w in windows)
            report["integrity"] = store.connection.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            mcp.auto_consolidation.close()
            store.close()
        return report

async def main():
    model = LocalEmbeddingModel()
    model.warmup()
    chunk_text(constants["answer"])
    assert model.encoder_id().startswith("st:")
    report = {"head": subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(), "encoder": model.encoder_id(), "tokenizer": get_tokenizer().name, "cases": []}
    for name, content in constants["controls"].items():
        constraint = "constraint" in name
        report["cases"].append(await case(name, content, "Можно ли автоматически перезапускать панель оператора на рабочей станции после обслуживания?" if constraint else constants["query"], "The dashboard restarts automatically after the maintenance window, including on the operator workstation." if constraint else constants["related"]))
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
