from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from living_memory.server import create_mcp_server


def test_real_fastmcp_server_registers_and_exercises_all_tools(tmp_path: Path) -> None:
    pytest.importorskip("fastmcp")
    asyncio.run(_exercise_real_fastmcp_server(tmp_path))


async def _exercise_real_fastmcp_server(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")

    tools = {tool.name for tool in await mcp.list_tools()}
    resources = {str(resource.uri) for resource in await mcp.list_resources()}
    templates = {str(template.uri_template) for template in await mcp.list_resource_templates()}
    prompts = {prompt.name for prompt in await mcp.list_prompts()}

    assert tools == {
        "memory_remember",
        "memory_teach",
        "memory_connect",
        "memory_recall",
        "memory_consolidate",
        "memory_forget",
        "memory_status",
        "memory_health",
    }
    assert resources == {
        "memory://global/concepts",
        "memory://stats",
        "memory://recent",
    }
    assert templates == {"memory://project/{name}/concepts"}
    assert prompts == {"memory://prompt/retrieval_context"}

    status = _structured(await mcp.call_tool("memory_status", {"scope": "project:runtime"}))
    assert status["scope"] == "project:runtime"

    first = _structured(
        await mcp.call_tool(
            "memory_remember",
            {
                "content": "real FastMCP runtime smoke cause",
                "context": {"scope": "project:runtime", "agent": "test-a"},
            },
        )
    )
    second = _structured(
        await mcp.call_tool(
            "memory_remember",
            {
                "content": "real FastMCP runtime smoke effect",
                "context": {"scope": "project:runtime", "agent": "test-b"},
            },
        )
    )
    first_id = first["node"]["id"]
    second_id = second["node"]["id"]

    connected = _structured(
        await mcp.call_tool(
            "memory_connect",
            {"id_a": first_id, "id_b": second_id, "relation_type": "caused"},
        )
    )
    assert connected["connection"]["type"] == "caused"

    recalled = _structured(
        await mcp.call_tool(
            "memory_recall",
            {
                "query": "why did runtime smoke effect happen",
                "scope": "project:runtime",
                "depth": "causal",
                "max_results": 5,
            },
        )
    )
    assert recalled["count"] >= 1

    taught = _structured(
        await mcp.call_tool(
            "memory_teach",
            {
                "trace_id": first_id,
                "correction": "real FastMCP runtime smoke corrected cause",
                "context": {"agent": "test-c"},
            },
        )
    )
    assert taught["supersedes"]["type"] == "supersedes"

    consolidated = _structured(
        await mcp.call_tool("memory_consolidate", {"scope": "project:runtime", "force": True})
    )
    assert "concepts_created" in consolidated
    assert "decayed" in consolidated

    forgotten = _structured(
        await mcp.call_tool("memory_forget", {"id": second_id, "reason": "runtime smoke"})
    )
    assert forgotten["node"]["decayed"] is True

    prompt = await mcp.render_prompt(
        "memory://prompt/retrieval_context",
        {"task": "runtime smoke", "scope": "project:runtime", "max_concepts": "2"},
    )
    assert "BEGIN ACTIVE MEMORY CONTEXT" in str(prompt)

    stats = await mcp.read_resource("memory://stats")
    assert "phase" in str(stats)


def _structured(result: Any) -> dict[str, Any]:
    payload = getattr(result, "structured_content", None)
    assert isinstance(payload, dict)
    if set(payload) == {"result"} and isinstance(payload["result"], dict):
        return payload["result"]
    return payload
