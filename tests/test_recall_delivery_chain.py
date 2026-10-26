"""Synthetic MCP checks for cost through the last necessary reading step.

The oracle below checks literal task facts, not the delivery module's passage
rank or excerpt positions. All bodies and queries in this file are invented.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastmcp")
from fastmcp import Client

from living_memory.consolidation import DIGEST_MAX_CHARS
from living_memory.delivery import DEFAULT_SNIPPET_LADDER
from living_memory.server import create_mcp_server
from test_transport_identity import _structured


@pytest.mark.parametrize("statement", [
    "Do not\nrestart the dashboard.",
    "Only after operator approval\nrestart the dashboard.",
    "Only after operator approval; restart the dashboard.",
    "Restart the dashboard; only after operator approval.",
    pytest.param(
        "Do not\n" + "under the dashboard maintenance policy " * 100 + "; restart the dashboard.",
        id="oversized-qualified-passage",
    ),
])
def test_consolidated_qualifications_survive_persistence_and_mcp_recall(tmp_path, statement):
    scope = "project:synthetic-consolidation"
    topic = "Service availability depends on operator scheduling for dashboard maintenance release policy safety checklist."
    complement = "Audit records expire after thirty days."
    db = tmp_path / "memory.sqlite3"
    mcp = create_mcp_server(db)
    store = mcp.memory_store
    sources = [
        store.append_trace(
            text, {"scope": scope, "agent": f"source-{index}"},
            feedback={"usefulness_score": 0.5 if index == 0 else 0.0},
        )
        for index, text in enumerate((topic, topic + " " + statement, topic + " " + complement))
    ]

    async def consolidate(server):
        async with Client(server) as client:
            return _structured(await client.call_tool("memory_consolidate", {"scope": scope}))

    async def read(server, concept_id):
        async with Client(server) as client:
            response = _structured(await client.call_tool("memory_recall", {
                "query": "dashboard maintenance operator scheduling", "scope": scope,
                "depth": 0, "max_results": 4,
            }))
            entry = next(item for item in response["results"] if item["id"] == concept_id)
            assert entry["delivery"] == "full"
            assert len(entry["content"]) <= DIGEST_MAX_CHARS
            if len(statement) > DIGEST_MAX_CHARS:
                assert "restart the dashboard" not in entry["content"]
                assert "Do not" not in entry["content"]
                assert "…" not in entry["content"]
                source_entry = next(item for item in response["results"] if item["id"] == sources[1].id)
                assert source_entry["delivery"] == "snippet"
                assert source_entry["content"].endswith("…\n[Incomplete passage]")
                assert "restart the dashboard" not in source_entry["content"]
            else:
                assert statement in entry["content"]
            assert complement in entry["content"]
            assert not entry["superseded"]
            lookup = _structured(await client.call_tool("memory_lookup", {
                "node_id": entry["content_ref"]["node_id"], "view": "full",
            }))
            saved = lookup["results"][0]
            assert saved["content"] == entry["content"]
            assert set(saved["provenance"]["source_traces"]) == {source.id for source in sources}
            originals = _structured(await client.call_tool("memory_lookup", {
                "node_ids": [source.id for source in sources],
            }))
            assert {item["id"]: item["content"] for item in originals["results"]} == {
                source.id: source.content for source in sources
            }
            assert all(not item["superseded"] for item in originals["results"])

    try:
        created = asyncio.run(consolidate(mcp))["concepts_created"]
        assert len(created) == 1
        assert created[0]["content"].startswith(topic + "\n")
        concept_id = created[0]["id"]
        # Read from a reopened store: delivery must consume the persisted concept.
        store.close()
        mcp = create_mcp_server(db)
        asyncio.run(read(mcp, concept_id))

        # An existing derived record changes when its ordinary materializer runs.
        previous = "Stored digest pending ordinary refresh."
        mcp.memory_store.update_node(concept_id, content=previous)
        mcp.memory_store.close()
        mcp = create_mcp_server(db)
        assert mcp.memory_store.get_node(concept_id).content == previous
        refreshed = asyncio.run(consolidate(mcp))
        assert refreshed["concepts_created"] == []
        assert [node["id"] for node in refreshed["concepts_updated"]] == [concept_id]
        asyncio.run(read(mcp, concept_id))
    finally:
        mcp.memory_store.close()


@pytest.mark.parametrize("view", ["knowledge", "full"])
@pytest.mark.parametrize("evidence", [False, True])
@pytest.mark.parametrize("qualifier", ["Do not", "Only if the indicator is green,"])
def test_mcp_excerpt_keeps_qualifier_and_lookup_restores_original(
    tmp_path, monkeypatch, view, evidence, qualifier,
):
    monkeypatch.setenv("LM_DELIVERY_SNIPPET_LADDER", "200")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    statement = qualifier + " open the amber hatch."
    lead = "Archive " + "x" * (700 - len(qualifier) - len("Archive . ")) + ". "
    heading = "Evidence: hatch cases (case records, not steps)\n" if evidence else ""
    prefix = "- 01SYNTHETICSOURCE0000000001: " if evidence else ""
    content = heading + prefix + lead + statement + " Routine background." * 100
    scope = "project:synthetic-excerpt"
    node = mcp.memory_store.append_trace(content, {"scope": scope})

    async def scenario():
        async with Client(mcp) as client:
            answer = _structured(await client.call_tool("memory_recall", {
                "query": "open amber hatch", "scope": scope, "max_results": 1,
                "depth": 0, "view": view,
            }))
            entry = answer["results"][0]
            body = entry.get("node", entry)
            assert body["id"] == node.id
            assert heading + prefix + statement in body["content"]
            assert len(body["content"]) <= 200
            assert entry["delivery"] == "snippet"
            assert entry["content_ref"] == {"node_id": node.id, "full_content_chars": len(content)}
            lookup = _structured(await client.call_tool("memory_lookup", {
                "node_id": entry["content_ref"]["node_id"],
            }))
            assert lookup["results"][0]["content"] == content

    asyncio.run(scenario())


@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_mcp_indivisible_excerpt_is_marked_and_refetchable(tmp_path, monkeypatch, view):
    monkeypatch.setenv("LM_DELIVERY_SNIPPET_LADDER", "200")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    content = "x" * 694 + "do not open the amber hatch" + " background" * 1000
    scope = "project:synthetic-indivisible"
    node = mcp.memory_store.append_trace(content, {"scope": scope})

    async def scenario():
        async with Client(mcp) as client:
            answer = _structured(await client.call_tool("memory_recall", {
                "query": "open amber hatch", "scope": scope, "max_results": 1,
                "depth": 0, "view": view,
            }))
            entry = answer["results"][0]
            excerpt = entry.get("node", entry)["content"]
            assert entry["delivery"] == "snippet"
            assert len(excerpt) <= 200 and excerpt.endswith("…\n[Incomplete passage]")
            prefix = excerpt.removesuffix("\n[Incomplete passage]")
            assert content.startswith(prefix[:-1])
            assert "open the amber hatch" not in excerpt
            assert entry["content_ref"]["full_content_chars"] == len(content)
            lookup = _structured(await client.call_tool("memory_lookup", {
                "node_id": entry["content_ref"]["node_id"],
            }))
            assert lookup["results"][0]["id"] == node.id
            assert lookup["results"][0]["content"] == content

    asyncio.run(scenario())


@dataclass
class Reading:
    responses: list[dict[str, Any]]
    sufficient: bool
    recall_sufficient: bool

    @property
    def calls(self) -> int:
        return len(self.responses)

    @property
    def chars(self) -> int:
        return sum(len(json.dumps(response, sort_keys=True, ensure_ascii=False)) for response in self.responses)


def _facts_present(responses: list[dict[str, Any]], facts: tuple[str, ...]) -> bool:
    """Independent answer oracle: every exact fact must be in delivered text."""
    text = "\n".join(
        node["content"]
        for response in responses
        for item in response.get("results", [])
        if isinstance(node := item, dict)
        if isinstance(node.get("content"), str)
    )
    return all(fact in text for fact in facts)


def _case_records(*, important: str, important_at: int = 16) -> tuple[str, str]:
    source_ids = [f"01SYNTH{i:018d}" for i in range(24)]
    records = [
        f"- {source}: Archive entry {i}: ordinary equipment ledger. " + "Archive context. " * 85
        for i, source in enumerate(source_ids)
    ]
    records[important_at] = (
        f"- {source_ids[important_at]}: {important} " + "Context for this case. " * 40
    )
    return "Evidence: synthetic equipment cases (case records, not steps)\n" + "\n".join(records), source_ids[important_at]


async def _read_chain(
    mcp: Any, query: str, scope: str, facts: tuple[str, ...],
    *, followup_id: str | None = None,
) -> Reading:
    async with Client(mcp) as client:
        recall = _structured(await client.call_tool(
            "memory_recall", {"query": query, "scope": scope, "max_results": 5}
        ))
        responses = [recall]
        recall_sufficient = _facts_present(responses, facts)
        if not recall_sufficient and followup_id is not None:
            entry = next(item for item in recall["results"] if item["id"] == followup_id)
            assert entry["content_ref"]["node_id"] == followup_id
            responses.append(_structured(await client.call_tool(
                "memory_lookup", {"node_id": followup_id}
            )))
        return Reading(responses, _facts_present(responses, facts), recall_sufficient)


def _paired(
    monkeypatch: pytest.MonkeyPatch, mcp: Any, query: str, scope: str,
    facts: tuple[str, ...], *, followup_id: str | None = None,
) -> tuple[Reading, Reading]:
    monkeypatch.delenv("LM_DELIVERY_SNIPPET_LADDER", raising=False)
    monkeypatch.delenv("LM_DELIVERY_SNIPPET_CHARS", raising=False)
    candidate = asyncio.run(_read_chain(mcp, query, scope, facts, followup_id=followup_id))
    # Same ranked store and query with the uniform full-content mode. Separate
    # MCP sessions prevent session dedup from changing either first delivery.
    monkeypatch.setenv("LM_DELIVERY_SNIPPET_LADDER", "off")
    monkeypatch.setenv("LM_DELIVERY_SNIPPET_CHARS", "0")
    baseline = asyncio.run(_read_chain(mcp, query, scope, facts, followup_id=followup_id))
    assert [item["id"] for item in candidate.responses[0]["results"]] == [
        item["id"] for item in baseline.responses[0]["results"]
    ]
    return candidate, baseline


def test_buried_knowledge_and_competing_passages_cost_one_bounded_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = "project:synthetic-carrier"
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    fact = "For opal valve calibration, the verified pressure is 31 psi."
    content, source = _case_records(important=fact)
    # A competing passage matches much of the query, but does not answer it.
    content = content.replace(
        "Archive entry 2: ordinary equipment ledger.",
        "Archive entry 2: opal valve equipment calibration discussion.",
    )
    carrier = mcp.memory_store.create_node(level="concept", content=content, context={"scope": scope})
    assert fact in carrier.content and f"- {source}: {fact}" in carrier.content
    candidate, baseline = _paired(
        monkeypatch, mcp, "opal valve calibration verified pressure", scope,
        (fact, f"- {source}:"), followup_id=carrier.id,
    )
    entry = candidate.responses[0]["results"][0]
    assert entry["id"] == carrier.id
    assert entry["delivery"] == "snippet"
    assert entry["content"].startswith("Evidence: synthetic equipment cases (case records, not steps)")
    assert len(entry["content"]) <= DEFAULT_SNIPPET_LADDER[0]
    assert candidate.recall_sufficient and candidate.sufficient
    assert candidate.calls == baseline.calls == 1
    assert candidate.chars < baseline.chars / 4


def test_buried_applicable_instruction_is_delivered_as_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = "project:synthetic-instruction"
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    fact = "For jade press maintenance, disconnect the amber lead before loosening the clamp."
    content, source = _case_records(important=fact, important_at=19)
    carrier = mcp.memory_store.create_node(level="concept", content=content, context={"scope": scope})
    assert fact in carrier.content
    candidate, baseline = _paired(
        monkeypatch, mcp, "jade press maintenance amber lead clamp", scope,
        (fact, f"- {source}:"), followup_id=carrier.id,
    )
    assert candidate.recall_sufficient and candidate.calls == baseline.calls == 1
    assert candidate.chars < baseline.chars
    assert "case records, not steps" in candidate.responses[0]["results"][0]["content"]


def test_large_carrier_lookup_cost_covers_the_whole_needed_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = "project:synthetic-large-lookup"
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    fact = "The archived jade press exception uses the amber isolation lead."
    content, _source = _case_records(important=fact, important_at=21)
    carrier = mcp.memory_store.create_node(level="concept", content=content, context={"scope": scope})
    candidate, baseline = _paired(
        monkeypatch, mcp, "synthetic equipment cases archive", scope,
        (fact,), followup_id=carrier.id,
    )
    assert candidate.sufficient and baseline.sufficient
    assert candidate.calls == 2 and baseline.calls == 1
    assert candidate.responses[1]["results"][0]["content"] == content
    # Count the lookup's full bytes: the snippet adds only bounded overhead
    # when the needed fact was outside the first delivery.
    assert candidate.chars <= baseline.chars * 1.2


def test_short_knowledge_and_binding_schema_stay_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = "project:synthetic-short"
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    short = mcp.memory_store.append_trace("Silver dial target: 17 rotations.", {"scope": scope})
    schema = mcp.memory_store.create_node(
        level="schema", content="Binding silver dial instruction: check the stop pin.\n" + "Check each seal.\n" * 400,
        context={"scope": scope},
    )
    facts = (short.content, schema.content)
    candidate, baseline = _paired(monkeypatch, mcp, "silver dial target instruction", scope, facts)
    assert candidate.recall_sufficient and candidate.calls == baseline.calls == 1
    entries = {entry["id"]: entry for entry in candidate.responses[0]["results"]}
    assert entries[short.id]["delivery"] == "full"
    assert entries[schema.id]["delivery"] == "full"
    assert entries[schema.id]["content"] == schema.content


def test_superseded_rule_does_not_become_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = "project:synthetic-correction"
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    old = mcp.memory_store.append_trace("Copper gate: open before venting pressure.", {"scope": scope})

    async def teach() -> dict[str, Any]:
        async with Client(mcp) as client:
            return _structured(await client.call_tool("memory_teach", {
                "trace_id": old.id,
                "correction": "Copper gate: vent pressure before opening.",
                "context": {"scope": scope},
            }))

    taught = asyncio.run(teach())
    candidate, baseline = _paired(
        monkeypatch, mcp, "copper gate pressure opening", scope,
        ("Copper gate: vent pressure before opening.",),
    )
    assert taught["supersedes"]["type"] == "supersedes"
    assert candidate.recall_sufficient and candidate.calls == baseline.calls == 1
    delivered = "\n".join(item["content"] for item in candidate.responses[0]["results"])
    assert old.content not in delivered


def test_missing_detail_requires_lookup_but_unknown_memory_stays_discoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = "project:synthetic-missing"
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    hidden = "The hidden iris rotor exception code is NOVA-82."
    content = "Iris rotor handbook.\n\n" + "Iris rotor handbook background. " * 170 + "\n\n" + hidden
    carrier = mcp.memory_store.create_node(level="concept", content=content, context={"scope": scope})
    assert hidden in carrier.content
    candidate, baseline = _paired(
        monkeypatch, mcp, "iris rotor handbook", scope, (hidden,), followup_id=carrier.id,
    )
    assert not candidate.recall_sufficient
    assert candidate.sufficient and baseline.sufficient
    assert candidate.calls == 2 and baseline.calls == 1
    assert candidate.responses[1]["results"][0]["content"] == carrier.content
    assert candidate.chars > len(json.dumps(
        candidate.responses[0], sort_keys=True, ensure_ascii=False
    ))  # The cost includes the necessary lookup response.

    async def absent() -> dict[str, Any]:
        async with Client(mcp) as client:
            return _structured(await client.call_tool("memory_recall", {
                "query": "unrecorded quartz sail requirement", "scope": "project:synthetic-absent"
            }))

    missing = asyncio.run(absent())
    assert missing["count"] == 0
    assert missing["results"] == []


@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_same_session_delivers_current_content_and_preserves_feedback(tmp_path, monkeypatch, view):
    monkeypatch.setenv("LM_RECALL_MAP", "0")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    scope = "project:synthetic-current"
    lead = ("Harbour deployment checklist covers the equipment inspection and routine crew briefing. "
            "The loading record retains its descriptive introduction for the harbour crew. ")
    node = store.create_node(level="concept", content=lead + "ZEPHYR authorization: gate ALPHA.",
                             context={"scope": scope})
    args = {"query": "ZEPHYR authorization", "scope": scope, "max_results": 1, "depth": 0, "view": view}

    async def scenario():
        async with Client(mcp) as client:
            async def recall(**extra):
                answer = _structured(await client.call_tool("memory_recall", {**args, **extra}))
                entry = answer["results"][0]
                body = entry.get("node", entry)
                assert body["id"] == node.id
                event = store.get_recall_event(answer["recall_event_id"])
                assert event.result_ids == [node.id]
                assert event.results[0]["delivery"]["kind"] == entry["delivery"]
                return answer, entry, body

            first, entry, body = await recall()
            assert entry["delivery"] == "full" and "ALPHA" in body["content"]
            repeat, entry, body = await recall()
            assert entry["delivery"] == "session_duplicate"
            # Access metadata changes do not constitute new knowledge.
            store.record_access(node.id)
            _, entry, _ = await recall()
            assert entry["delivery"] == "session_duplicate"
            store.update_node(node.id, content=lead + "ZEPHYR authorization: gate OMEGA.")
            changed, entry, body = await recall(used=[node.id])
            assert entry["delivery"] == "full" and "OMEGA" in body["content"]
            assert "ALPHA" not in body["content"]
            assert changed["feedback_marks"]["accepted"] == 1
            _, entry, _ = await recall(irrelevant=[node.id])
            assert entry["delivery"] == "session_duplicate"
            remembered = _structured(await client.call_tool("memory_remember", {
                "content": "ZEPHYR authorization outcome: gate OMEGA checked.",
                "context": {"scope": scope},
            }))
            assert remembered["implicit_feedback"]["recall_event_ids"]
            assert store.get_recall_event(first["recall_event_id"]).feedback_applied
            lookup = _structured(await client.call_tool("memory_lookup", {"node_id": node.id}))
            assert "OMEGA" in lookup["results"][0]["content"]
        async with Client(mcp) as fresh:
            answer = _structured(await fresh.call_tool("memory_recall", args))
            entry = answer["results"][0]
            assert entry["delivery"] == "full"
            assert "OMEGA" in entry.get("node", entry)["content"]

    asyncio.run(scenario())


def test_new_query_can_deliver_a_previously_unseen_excerpt(tmp_path, monkeypatch):
    monkeypatch.setenv("LM_RECALL_MAP", "0")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    scope = "project:synthetic-new-excerpt"
    fact = "The hidden iris rotor exception code is NOVA-82."
    content = "Iris rotor handbook.\n\n" + "Iris rotor handbook background. " * 170 + "\n\n" + fact
    node = mcp.memory_store.create_node(level="concept", content=content, context={"scope": scope})

    async def scenario():
        async with Client(mcp) as client:
            async def recall(query):
                return _structured(await client.call_tool("memory_recall", {
                    "query": query, "scope": scope, "max_results": 1, "depth": 0,
                }))["results"][0]
            first = await recall("iris rotor handbook")
            assert first["id"] == node.id and first["delivery"] == "snippet"
            assert fact not in first["content"]
            assert (await recall("iris rotor handbook"))["delivery"] == "session_duplicate"
            changed = await recall("hidden iris rotor exception code NOVA-82")
            assert changed["delivery"] == "snippet" and fact in changed["content"]
            assert (await recall("hidden iris rotor exception code NOVA-82"))["delivery"] == "session_duplicate"
            lookup = _structured(await client.call_tool("memory_lookup", {"node_id": changed["content_ref"]["node_id"]}))
            assert lookup["results"][0]["content"] == content
    asyncio.run(scenario())


def test_legacy_selection_and_receipt_failure_do_not_suppress_content(tmp_path, monkeypatch):
    from test_lookup import FakeMCP
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    scope = "project:synthetic-legacy"
    node = store.create_node(level="concept", content="Harbour authorization: use gate OMEGA.", context={"scope": scope})
    args = {"query": "Harbour authorization", "scope": scope, "depth": 0,
            "ambient_context": {"transport_session_id": "reader"}}
    legacy = store.record_recall_event(query=args["query"], scope=scope,
                                      ambient_context=args["ambient_context"], results=[{"node_id": node.id}])
    assert "delivery" not in legacy.results[0]
    store.connection.execute("""
        CREATE TEMP TRIGGER reject_receipt BEFORE UPDATE OF results ON recall_events
        BEGIN SELECT RAISE(ABORT, 'synthetic receipt failure'); END
    """)
    failed = mcp.tools["memory_recall"](**args)
    assert failed["results"][0]["content"] == node.content
    assert "delivery" not in store.get_recall_event(failed["recall_event_id"]).results[0]
    store.connection.execute("DROP TRIGGER reject_receipt")
    retried = mcp.tools["memory_recall"](**args)
    assert retried["results"][0]["delivery"] == "full"
    assert mcp.tools["memory_recall"](**args)["results"][0]["delivery"] == "session_duplicate"
    def fail(*args, **kwargs):
        raise OSError("synthetic history read failure")
    monkeypatch.setattr(store, "recall_delivery_history", fail)
    assert mcp.tools["memory_recall"](**args)["results"][0]["content"] == node.content
