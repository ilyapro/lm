"""A bounded ordinary recall must answer beyond topical source-record decoys."""
from __future__ import annotations

import asyncio
from pathlib import Path
import re

from fastmcp import Client
import pytest

import living_memory.delivery as delivery
from living_memory.server import create_mcp_server

QUERY = "opal valve calibration verified pressure"
FACT = "For opal valve calibration, the verified pressure is 31 psi."
SOURCE = "01SYNTH00000000000019"
SCOPE = "project:synthetic-opal"


def _assert_sourced_answer(content: str) -> None:
    assert f"- {SOURCE}: {FACT}" in content, "recall must contain the sourced pressure answer"


def _word_overlap_excerpt(content: str, budget: int, query: str | None) -> str:
    """Negative selector: word coverage alone cannot distinguish these records."""
    heading, *records = content.splitlines()
    words = set(re.findall(r"\w+", (query or "").casefold()))
    ranked = sorted(
        records,
        key=lambda record: -len(words.intersection(re.findall(r"\w+", record.casefold()))),
    )
    selected = heading
    for record in ranked:
        if len(selected) + 1 + len(record) > budget:
            break
        selected += "\n" + record
    return selected


@pytest.mark.parametrize("control", ["ordinary", "word_overlap", "wrong_source"])
def test_answer_survives_topical_decoys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str,
) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    decoy = (
        "Opal records discuss the valve inventory and calibration schedule. "
        "A verified inspector logged a pressure question without an answer. "
        + "Routine equipment notes. " * 24
    )
    records = [f"- 01SYNTH{i:020d}: {decoy}" for i in range(19)]
    records.append(f"- {SOURCE}: {FACT} " + "Applicable test context. " * 12)
    records.extend(f"- 01SYNTH{i:020d}: unrelated archive notes." for i in range(20, 24))
    content = "Evidence: invented equipment case records, not procedures\n" + "\n".join(records)
    carrier = mcp.memory_store.create_node(
        level="concept", content=content, context={"scope": SCOPE}
    )
    assert content.index(f"- {SOURCE}: ") > 2400
    _assert_sourced_answer(content)

    if control == "word_overlap":
        monkeypatch.setattr(delivery, "_relevant_excerpt", _word_overlap_excerpt)
    elif control == "wrong_source":
        excerpt = delivery._relevant_excerpt
        monkeypatch.setattr(
            delivery, "_relevant_excerpt",
            lambda body, budget, query: excerpt(body, budget, query).replace(
                f"- {SOURCE}: ", "- 01SYNTH00000000000000000: "
            ),
        )

    async def read() -> tuple[dict, dict]:
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
                if control == "word_overlap":
                    assert FACT not in snippet
                    assert "- 01SYNTH00000000000000000: " in snippet
                else:
                    assert FACT in snippet
                    assert f"- {SOURCE}: " not in snippet
                with pytest.raises(AssertionError, match="sourced pressure answer"):
                    _assert_sourced_answer(snippet)
            lookup = (await client.call_tool("memory_lookup", {
                "node_id": entry["content_ref"]["node_id"],
            })).structured_content
            return recall, lookup

    _, lookup = asyncio.run(read())
    assert lookup is not None
    assert [item["id"] for item in lookup["results"]] == [carrier.id]
    assert lookup["results"][0]["content"] == content
    _assert_sourced_answer(lookup["results"][0]["content"])
