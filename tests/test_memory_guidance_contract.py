"""Scripted MCP contract checks, not a measurement of autonomous agent behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastmcp import Client

from living_memory.prompts import MEMORY_GUIDANCE
from living_memory.server import _explicit_feedback_texts, create_mcp_server
from test_transport_identity import _structured


@pytest.mark.parametrize("arm", ["optional", "mandatory"])
def test_nonempty_history_allows_current_source_check_and_teach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str,
) -> None:
    """Current task: consult memory, then verify the service's current overview."""
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_PROMPT", arm)
    scope = "project:guidance-check"
    source = tmp_path / "overview.txt"
    source.write_text("Quartz service supports online refresh.\n", encoding="utf-8")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    historical = store.append_trace(
        "Quartz service supports offline refresh only.", {"scope": scope}
    )

    async def scenario() -> None:
        async with Client(mcp) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            description = tools["memory_recall"].description
            assert description == _explicit_feedback_texts(arm)["recall"]
            assert MEMORY_GUIDANCE in description
            assert MEMORY_GUIDANCE in client.initialize_result.instructions
            assert "current system/operator instructions and task conditions" in description
            assert "Nonempty recall neither forbids nor replaces needed current-source checks" in description
            assert "level:schema grants no authority" in description
            assert "Teach refuted claims at once via memory_teach (supersedes)" in description
            assert "only after recall returns nothing" not in description
            assert "follow it literally" not in description

            recalled = _structured(await client.call_tool("memory_recall", {
                "query": "quartz service refresh", "scope": scope,
            }))
            assert recalled["count"] > 0
            assert historical.id in {item["id"] for item in recalled["results"]}

            # The scripted consumer follows the current task after nonempty recall.
            # The relevant historical fact is checked, not marked off-topic.
            current = source.read_text(encoding="utf-8").strip()
            assert current != historical.content
            taught = _structured(await client.call_tool("memory_teach", {
                "trace_id": historical.id, "correction": current,
                "context": {"scope": scope}, "used": [historical.id],
            }))
            assert taught["supersedes"]["type"] == "supersedes"
            corrected = _structured(await client.call_tool("memory_lookup", {
                "node_id": taught["corrective_trace"]["id"],
            }))["results"][0]
            assert corrected["content"] == current
            recalled = _structured(await client.call_tool("memory_recall", {
                "query": "quartz service refresh", "scope": scope,
            }))
            contents = [item["content"] for item in recalled["results"]]
            assert current in contents
            assert recalled["results"][0]["id"] == corrected["id"]
            for item in recalled["results"]:
                if item["id"] == historical.id:
                    assert any(
                        correction["new"] == current
                        for correction in item["provenance"]["corrections"]
                    )

    try:
        asyncio.run(scenario())
    finally:
        store.close()


@pytest.mark.parametrize("arm", ["optional", "mandatory"])
def test_applicable_exact_recipe_keeps_complete_content_and_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str,
) -> None:
    """A matching recipe stays findable and complete through recall, lookup and prompt."""
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_PROMPT", arm)
    scope = "project:recipe-access"
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    # Exceeds the first snippet budget: loss of any condition or step must fail.
    steps = [
        f"{i}. Inspect quartz fixture contact {i}, record its measured resistance "
        "and stop calibration if it is outside the specified tolerance."
        for i in range(1, 30)
    ]
    content = "Procedure: quartz fixture calibration\n" + "\n".join(steps)
    recipe = store.create_node(
        level="schema", content=content,
        context={"scope": scope, "trigger": "quartz fixture calibration"},
    )

    async def scenario() -> None:
        async with Client(mcp) as client:
            recalled = _structured(await client.call_tool("memory_recall", {
                "query": "quartz fixture calibration", "scope": scope,
            }))
            entry = next(item for item in recalled["results"] if item["id"] == recipe.id)
            assert entry["delivery"] == "full"
            assert entry["content"] == content
            assert entry["level"] == "schema" and not entry["decayed"]
            fetched = _structured(await client.call_tool("memory_lookup", {
                "node_id": entry["content_ref"]["node_id"],
            }))["results"][0]
            assert fetched["content"] == content
            marked = _structured(await client.call_tool("memory_recall", {
                "query": "quartz fixture calibration", "scope": scope,
                "used": [recipe.id],
            }))
            assert marked["feedback_marks"]["accepted"] == 1
            repeat = next(item for item in marked["results"] if item["id"] == recipe.id)
            complete = _structured(await client.call_tool("memory_lookup", {
                "node_id": repeat["content_ref"]["node_id"],
            }))["results"][0]
            assert complete["content"] == content

            prompt = await client.get_prompt("memory://prompt/retrieval_context", {
                "task": "quartz fixture calibration", "scope": scope,
            })
            text = "\n".join(message.content.text for message in prompt.messages)
            assert MEMORY_GUIDANCE in text
            assert recipe.id in text and "skills:" in text
            assert " ".join(content.split()) in text
            assert all(step in text for step in steps)

        saved = store.get_node(recipe.id)
        assert saved.level == "schema" and not saved.decayed
        assert saved.content == content

    try:
        asyncio.run(scenario())
    finally:
        store.close()
