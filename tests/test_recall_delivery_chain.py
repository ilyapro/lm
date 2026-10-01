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

from living_memory.delivery import DEFAULT_SNIPPET_LADDER
from living_memory.server import create_mcp_server
from test_transport_identity import _structured


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
        if isinstance(node := item.get("node", item), dict)
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
            entry = next(item for item in recall["results"] if item["node"]["id"] == followup_id)
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
    assert [item["node"]["id"] for item in candidate.responses[0]["results"]] == [
        item["node"]["id"] for item in baseline.responses[0]["results"]
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
    assert entry["node"]["id"] == carrier.id
    assert entry["delivery"] == "snippet"
    assert entry["node"]["content"].startswith("Evidence: synthetic equipment cases (case records, not steps)")
    assert len(entry["node"]["content"]) <= DEFAULT_SNIPPET_LADDER[0]
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
    assert "case records, not steps" in candidate.responses[0]["results"][0]["node"]["content"]


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
    entries = {entry["node"]["id"]: entry for entry in candidate.responses[0]["results"]}
    assert entries[short.id]["delivery"] == "full"
    assert entries[schema.id]["delivery"] == "full"
    assert entries[schema.id]["node"]["content"] == schema.content


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
    delivered = "\n".join(item["node"]["content"] for item in candidate.responses[0]["results"])
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
