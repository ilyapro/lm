"""No digest without a possible delivery proof; encoding errors remain required."""

import asyncio
from dataclasses import replace
import hashlib
import json
import shutil
from types import SimpleNamespace

from fastmcp import Client
import pytest

from living_memory import delivery, server
from living_memory.models import Node
from living_memory.retrieval import RecallResult
from living_memory.storage import MemoryStore


class EagerHistory(delivery.DeliveryHistory):
    """Pre-shortcut oracle, independent of the optimized membership methods."""

    def has_content(self, node):
        return (node.id, delivery._content_digest(node.content)) in self._versions

    def covers(self, node, excerpt):
        source = delivery._content_digest(node.content)
        bodies = self._versions.get((node.id, source), ())
        return source in bodies or delivery._content_digest(excerpt) in bodies

    def uncovered_excerpt(self, node, budget, query, passage):
        excerpt = delivery._relevant_excerpt(node.content, budget, query, passage)
        return None if self.covers(node, excerpt) else excerpt


def evidence(node, excerpt=None):
    return {"node_id": node.id, "delivery": {
        "kind": "full" if excerpt is None else "snippet",
        "content_sha256": hashlib.sha256(node.content.encode("utf-8")).hexdigest(),
        "delivered_sha256": hashlib.sha256(
            (node.content if excerpt is None else excerpt).encode("utf-8")
        ).hexdigest(),
    }}


def count_hashes(patch):
    inputs = []

    def sha256(data):
        inputs.append(data)
        return hashlib.sha256(data)

    # Count actual SHA calls, not entry into the digest helper. Other modules'
    # hashing and the independent receipt oracle keep their original hashlib.
    patch.setattr(delivery, "hashlib", SimpleNamespace(sha256=sha256))
    return inputs


def without_event_ids(value):
    if isinstance(value, dict):
        return {key: without_event_ids(item) for key, item in value.items()
                if key != "recall_event_id"}
    if isinstance(value, list):
        return [without_event_ids(item) for item in value]
    return value


@pytest.fixture(scope="module")
def native_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("history-hashing")
    scope = "project:synthetic"
    query = "cobalt release signature"
    with pytest.MonkeyPatch.context() as patch:
        for key, value in {
            "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
            "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
            "LM_AUTO_CONSOLIDATE_POLICY": "off", "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
            "LM_RECALL_MAP": "0", "LM_RECALL_MIN_SCORE": "off",
            "LM_DELIVERY_SESSION_DEDUP": "1",
            "LM_DELIVERY_SNIPPET_LADDER": "off", "LM_DELIVERY_SNIPPET_CHARS": "1000",
        }.items():
            patch.setenv(key, value)
        seed = root / "seed.sqlite3"
        store = MemoryStore(seed)
        try:
            nodes = [store.create_node(
                level="concept",
                content=(f"Cobalt release signature item {i}. Independent inspection proof {i}. " * 65),
                context={"scope": scope},
            ) for i in range(8)]
        finally:
            store.close()

        async def run(path, history):
            mcp = server.create_mcp_server(path)
            store = mcp.memory_store
            answers, receipts, counts = [], [], []
            try:
                with pytest.MonkeyPatch.context() as run_patch:
                    run_patch.setattr(server, "DeliveryHistory", history)
                    inputs = count_hashes(run_patch)
                    async with Client(mcp) as client:
                        for index, current_query in enumerate([
                            query, query, "cobalt release inspection", query,
                            "amber hatch countersignature",
                        ]):
                            if index == 3:
                                store.update_node(nodes[0].id, content=nodes[0].content + "\n\n" + (
                                    "Amber hatch countersignature requires the violet seal. " * 50
                                ))
                            inputs.clear()
                            response = (await client.call_tool("memory_recall", {
                                "query": current_query, "scope": scope, "depth": 0, "max_results": 8,
                            })).data
                            counts.append({"calls": len(inputs), "bytes": sum(map(len, inputs))})
                            answers.append(without_event_ids(response))
                            event = store.get_recall_event(response["recall_event_id"])
                            receipts.append(without_event_ids(event.results))
                            # Receipts independently name each current source and actual body.
                            for entry in response["results"]:
                                row = next(row for row in event.results if row["node_id"] == entry["id"])
                                if entry["delivery"] in ("full", "snippet"):
                                    expected = evidence(store.get_node(entry["id"]), entry["content"])["delivery"]
                                    expected["kind"] = entry["delivery"]
                                    assert row["delivery"] == expected
                                else:
                                    assert row["delivery"] == {"kind": entry["delivery"]}
            finally:
                mcp.auto_consolidation.close()
                store.close()
            return {"answers": answers, "receipts": receipts, "counts": counts}

        runs = []
        for label, history in [("eager", EagerHistory), ("current", delivery.DeliveryHistory)]:
            path = root / f"{label}.sqlite3"
            shutil.copy2(seed, path)
            runs.append(asyncio.run(run(path, history)))
    return *runs, nodes[0].id


def test_native_recall_hashes_each_source_and_excerpt_once(native_runs):
    eager, current, _ = native_runs
    print(json.dumps({"eager": eager["counts"], "current": current["counts"]}))
    assert eager["counts"][0]["calls"] == 40
    for count in current["counts"][:3]:
        assert count == {"calls": 16, "bytes": 41776}


def test_native_answers_and_receipts_match_independently(native_runs):
    eager, current, changed_id = native_runs
    assert current["answers"] == eager["answers"]
    assert current["receipts"] == eager["receipts"]
    answers = current["answers"]
    assert [entry["delivery"] for entry in answers[0]["results"]] == ["snippet"] * 8
    for answer in answers[1:3]:
        assert [entry["delivery"] for entry in answer["results"]] == ["session_duplicate"] * 8
    changed = [next(entry for entry in answer["results"] if entry["id"] == changed_id)
               for answer in answers[3:]]
    assert all(entry["delivery"] == "snippet" for entry in changed)
    assert changed[0]["content"] != changed[1]["content"]
    assert "violet seal" in changed[1]["content"]


def test_mutable_content_and_exact_excerpt_proofs(monkeypatch):
    node = Node(id="source", level="trace", content="Original source.")
    history = delivery.DeliveryHistory([evidence(node, "First excerpt.")])
    inputs = count_hashes(monkeypatch)
    assert history.has_content(node)
    assert history.covers(node, "First excerpt.")
    assert not history.covers(node, "Different excerpt.")
    assert inputs == [node.content.encode(), b"First excerpt.", b"Different excerpt."]
    node.content = "Updated source."
    inputs.clear()
    assert not history.has_content(node)
    assert not history.covers(node, "First excerpt.")
    assert inputs == [b"Updated source."]
    inputs.clear()
    assert not history.covers(replace(node, id="other"), "First excerpt.")
    assert inputs == []
    node.content = "Original source."
    assert history.covers(node, "First excerpt.")


@pytest.mark.parametrize("content", ["", "Valid café 🌿"])
def test_empty_history_has_no_hash_consumer(monkeypatch, content):
    history = delivery.DeliveryHistory()
    inputs = count_hashes(monkeypatch)
    node = Node(id="source", level="trace", content=content)
    assert not history.has_content(node)
    assert not history.covers(node, content)
    assert inputs == []


@pytest.mark.parametrize("proof", ["empty", "other-version", "snippet", "full"])
@pytest.mark.parametrize("bad_source", [False, True])
def test_utf8_errors_and_full_proof_short_circuit(proof, bad_source):
    node = Node(id="source", level="trace", content="Current source.")
    records = {
        "empty": [], "other-version": [evidence(replace(node, content="Old source."))],
        "snippet": [evidence(node, "Known excerpt.")], "full": [evidence(node)],
    }[proof]
    if bad_source:
        node.content = "invalid\ud800source"
    excerpt = "invalid\udfffexcerpt"
    for history in (EagerHistory(records), delivery.DeliveryHistory(records)):
        if proof == "full" and not bad_source:
            assert history.covers(node, excerpt)
        else:
            with pytest.raises(UnicodeEncodeError) as error:
                history.covers(node, excerpt)
            assert error.value.object == (node.content if bad_source else excerpt)
        if bad_source:
            with pytest.raises(UnicodeEncodeError):
                history.has_content(node)


@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_in_memory_shaping_retains_source_encoding_error(view):
    node = Node(id="source", level="trace", content="invalid\ud800source")
    with pytest.raises(UnicodeEncodeError):
        delivery.shape_recall_results(
            [RecallResult(node=node, score=1)], delivery_history=delivery.DeliveryHistory(),
            session_dedup=True, snippet_max_chars=0, context_value_max_chars=160, view=view,
        )


@pytest.mark.parametrize("view", ["full", "knowledge"])
def test_whole_body_proof_skips_excerpt_without_spending_a_ladder_slot(monkeypatch, view):
    rows = [
        RecallResult(node=Node(id=str(i), level="trace", content=(f"Detail {i}. " * 220)[:1984]), score=1)
        for i in range(6)
    ]
    history = delivery.DeliveryHistory(evidence(row.node) for row in rows[1:5])
    calls = []
    select = delivery._relevant_excerpt

    def excerpt(content, budget, query, passage, **kwargs):
        calls.append((content, budget))
        return select(content, budget, query, passage, **kwargs)

    monkeypatch.setattr(delivery, "_relevant_excerpt", excerpt)
    shaped = delivery.shape_recall_results(
        rows, delivery_history=history, session_dedup=True, snippet_max_chars=1000,
        context_value_max_chars=160, query="detail", view=view,
    )
    assert [entry["delivery"] for entry in shaped] == ["full", *(["session_duplicate"] * 4), "snippet"]
    assert calls == [(rows[5].node.content, 1000)]


@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_native_full_source_digests_across_replacement(tmp_path, monkeypatch, view):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash", "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_AUTO_CONSOLIDATE_POLICY": "off", "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_RECALL_MAP": "0", "LM_RECALL_MIN_SCORE": "off", "LM_DELIVERY_SESSION_DEDUP": "1",
        "LM_DELIVERY_SNIPPET_LADDER": "off", "LM_DELIVERY_SNIPPET_CHARS": "0",
    }.items():
        monkeypatch.setenv(key, value)
    mcp = server.create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    content = "Quick recap\n" + "x" * 1024
    node = store.create_node(level="concept", content=content, context={"scope": "project:synthetic"})
    inputs = count_hashes(monkeypatch)

    async def recall(client, source, kind):
        inputs.clear()
        response = (await client.call_tool("memory_recall", {
            "query": "Quick recap", "scope": node.scope, "depth": 0, "max_results": 1, "view": view,
        })).data
        entry, = response["results"]
        body = entry.get("node", entry)
        assert (body["id"], entry["delivery"]) == (node.id, kind)
        assert body["content"] == (source if kind == "full" else "")
        receipt, = store.get_recall_event(response["recall_event_id"]).results
        assert receipt["node_id"] == node.id
        expected = {"kind": kind}
        encoded = source.encode("utf-8")
        if kind == "full":
            digest = hashlib.sha256(encoded).hexdigest()
            expected.update(content_sha256=digest, delivered_sha256=digest)
        assert receipt["delivery"] == expected
        assert inputs == [encoded]

    async def scenario():
        async with Client(mcp) as client:
            await recall(client, content, "full")
            monkeypatch.setenv("LM_DELIVERY_SNIPPET_CHARS", "500")
            await recall(client, content, "session_duplicate")
            updated = content[:-1] + "é"
            store.update_node(node.id, content=updated)
            monkeypatch.setenv("LM_DELIVERY_SNIPPET_CHARS", "0")
            await recall(client, updated, "full")
            await recall(client, updated, "session_duplicate")
        async with Client(mcp) as independent:
            await recall(independent, updated, "full")

    try:
        asyncio.run(scenario())
    finally:
        mcp.auto_consolidation.close()
        store.close()
