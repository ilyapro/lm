"""Scope-less ``memory_recall`` over the real MCP tool path.

Synthetic memory only. Omitting ``scope`` must reach every project's facts: a
project name hiding inside a query word and ambient session metadata must not
hide them, an explicit scope still restricts, and the wider search must not
fill the answer with unrelated memory under the deployed score gate.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client

from living_memory.server import create_mcp_server

FACT = "Proof sidecars leave the repo; the transfer proof refers to them by reference"
QUERY = "proof-refers-not-copies transfer proof by reference sidecars leave repo"
# "mm" hides inside "immediate" and "commit"; "repo" is a whole word of QUERY.
DECOYS = {
    "project:mm": "The mm widget renders the release calendar",
    "project:repo": "Repository mirrors sync every night at two",
}


def _structured(result: Any) -> dict[str, Any]:
    payload = result.structured_content
    if set(payload) == {"result"} and isinstance(payload["result"], dict):
        return payload["result"]
    return payload


async def _remember(client: Client, content: str, scope: str) -> str:
    result = _structured(
        await client.call_tool("memory_remember", {"content": content, "context": {"scope": scope}})
    )
    return result["node"]["id"]


async def _recall(client: Client, query: str, **arguments: Any) -> list[dict[str, Any]]:
    payload = _structured(
        await client.call_tool("memory_recall", {"query": query, "max_results": 5, **arguments})
    )
    return payload["results"]


def test_scopeless_mcp_recall_reaches_other_projects_and_explicit_scope_restricts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "0.35")  # the deployed gate

    async def scenario() -> None:
        mcp = create_mcp_server(tmp_path / "memory.sqlite3")
        async with Client(mcp) as client:
            fact = await _remember(client, FACT, "project:ae")
            for scope, content in DECOYS.items():
                await _remember(client, content, scope)
            fillers = {
                await _remember(client, f"Team {index} lunch rota and parking badges", f"project:f{index}")
                for index in range(20)
            }

            for ambient in (None, {"session_id": "s-1064"}, {"project": "repo"}):
                arguments = {"ambient_context": ambient} if ambient else {}
                for query in (QUERY, "private snapshot immediate commit proof sidecars"):
                    results = await _recall(client, query, **arguments)
                    ids = [entry["node"]["id"] for entry in results]
                    assert ids[0] == fact, (ambient, query)
                    assert not set(ids) & fillers

            restricted = await _recall(client, QUERY, scope="project:repo")
            assert fact not in {entry["node"]["id"] for entry in restricted}
            assert {entry["node"]["scope"] for entry in restricted} <= {"project:repo", "global"}

    asyncio.run(scenario())
