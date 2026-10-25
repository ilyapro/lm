"""Bounded recall must retain a buried rule with its exact source record."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastmcp import Client
import pytest

import living_memory.delivery as delivery
from living_memory.server import create_mcp_server

QUERY = "cedar vault relay current archival rotation audit runtime ledger"
FACT = "keep signed receipts in the local ledger before any archive rotation"
SOURCE = "01SYNTH00000000000000008"
SCOPE = "project:invented-cedar"


def _assert_sourced_answer(content: str) -> None:
    assert any(
        line.startswith(f"- {SOURCE}: ") and FACT in line
        for line in content.splitlines()
    ), "recall must contain the complete rule under its exact source"


@pytest.mark.parametrize("control", ["ordinary", "record_leads", "wrong_source"])
def test_buried_rule_keeps_its_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str,
) -> None:
    monkeypatch.delenv("LM_DELIVERY_SNIPPET_CHARS", raising=False)
    monkeypatch.delenv("LM_DELIVERY_SNIPPET_LADDER", raising=False)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    target = (
        "Cedar vault relay rotation audit ledger covers the current archival runtime workflow. "
        + "Background dates and context for the cedar vault relay. " * 12
        + " The active rule for cedar vault relay is: " + FACT + ". "
        + "Further background. " * 16
    )
    decoy = (
        "Cedar vault relay current archival rotation audit runtime ledger had several unrelated scheduling notes. "
        + "General meeting notes. " * 33
    )
    other = (
        "Cedar vault relay current archival rotation audit was discussed without a rule. "
        + "Routine meeting notes. " * 33
    )
    records = [f"- 01SYNTH{i:020d}: {decoy if i == 0 else other}" for i in range(8)]
    records.append(f"- {SOURCE}: {target}")
    content = "Evidence: invented archive records, not procedures\n" + "\n".join(records)
    carrier = mcp.memory_store.create_node(
        level="concept", content=content, context={"scope": SCOPE}
    )
    assert content.index(f"- {SOURCE}: ") > 2400
    _assert_sourced_answer(content)

    if control == "record_leads":
        passages = delivery._passages
        # Retaining only each record's introduction loses its applicable rule.
        monkeypatch.setattr(delivery, "_passages", lambda body: passages(body)[:1])
    elif control == "wrong_source":
        excerpt = delivery._relevant_excerpt
        monkeypatch.setattr(
            delivery, "_relevant_excerpt",
            lambda body, budget, query: excerpt(body, budget, query).replace(
                f"- {SOURCE}: ", "- 01SYNTH00000000000000000: "
            ),
        )

    async def read() -> dict:
        async with Client(mcp) as client:
            recall = (await client.call_tool("memory_recall", {
                "query": QUERY, "scope": SCOPE, "max_results": 5,
            })).structured_content
            assert recall is not None
            assert [item["id"] for item in recall["results"]] == [carrier.id]
            entry = recall["results"][0]
            assert entry["delivery"] == "snippet"
            assert entry["content_ref"]["node_id"] == carrier.id
            assert entry["content_ref"]["full_content_chars"] == len(content)
            snippet = entry["content"]
            assert snippet.startswith(content.splitlines()[0] + "\n")
            assert len(snippet) <= 2400
            if control == "ordinary":
                # Sufficiency is established before making any lookup call.
                _assert_sourced_answer(snippet)
            else:
                if control == "record_leads":
                    assert f"- {SOURCE}: " in snippet
                    assert FACT not in snippet
                else:
                    assert FACT in snippet
                    assert f"- {SOURCE}: " not in snippet
                with pytest.raises(AssertionError, match="complete rule under its exact source"):
                    _assert_sourced_answer(snippet)
            lookup = (await client.call_tool("memory_lookup", {
                "node_id": entry["content_ref"]["node_id"],
            })).structured_content
            return lookup

    lookup = asyncio.run(read())
    assert lookup is not None
    assert [item["id"] for item in lookup["results"]] == [carrier.id]
    assert lookup["results"][0]["content"] == content
    _assert_sourced_answer(lookup["results"][0]["content"])
