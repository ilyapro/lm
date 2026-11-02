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


def test_native_first_recall_omits_unusable_hashes(native_runs):
    eager, current, _ = native_runs
    print(json.dumps({"eager": eager["counts"], "current": current["counts"]}))
    assert eager["counts"][0]["calls"] == 40
    assert current["counts"][0]["calls"] == 24
    assert eager["counts"][0]["bytes"] - current["counts"][0]["bytes"] == 41776
    assert current["counts"][1:3] == eager["counts"][1:3]


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
    assert inputs == [node.content.encode()] * 2 + [b"First excerpt.", node.content.encode(), b"Different excerpt."]
    node.content = "Updated source."
    inputs.clear()
    assert not history.has_content(node)
    assert not history.covers(node, "First excerpt.")
    assert inputs == [b"Updated source."] * 2
    inputs.clear()
    assert not history.covers(replace(node, id="other"), "First excerpt.")
    assert inputs == [b"Updated source."]
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
