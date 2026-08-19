"""A small synchronous MCP client for the post-session extraction runner.

The runner is an *ordinary client* of the live Living Memory host: it talks to
the deployed unit over MCP exactly the way ``scripts/check_deployed_protocol.py``
does — streamable HTTP at ``http://127.0.0.1:8765/mcp/`` with a bearer token
resolved from ``--token``, then ``$LM_AUTH_TOKEN``, then
``~/.config/living-memory/env`` — and it NEVER opens the server's SQLite file.
That rule is structural here: this module imports neither ``sqlite3`` nor
``living_memory.storage``, so a write path around the server cannot appear by
accident.

Two things live here:

:class:`LiveMemoryClient`
    one MCP call per method, run to completion synchronously. A fresh
    connection per call is deliberate: the runner stamps the *source* session's
    ``transport_session_id`` explicitly into every write context, so nothing
    depends on the extractor's own connection identity, and a dropped
    connection cannot poison a batch.
:class:`McpMemoryIndex`
    the :class:`~living_memory.postsession.dedup.MemoryIndex` protocol served
    over MCP — ``memory_recall`` finds lexically/semantically plausible
    neighbours and ``memory_lookup`` re-fetches full contents for entries the
    delivery diet snippeted. This is what makes the runner's near-duplicate
    check work even when its local ledger is lost: the server itself is asked
    what already exists. There is no embedding channel over MCP, so
    ``embedding()`` returns ``None`` and the containment channel carries the
    verdict.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .dedup import NodeSnapshot

__all__ = [
    "DEFAULT_ENV_FILE",
    "DEFAULT_URL",
    "ClientError",
    "LiveMemoryClient",
    "McpMemoryIndex",
    "resolve_token",
]

DEFAULT_URL = "http://127.0.0.1:8765/mcp/"
DEFAULT_ENV_FILE = Path.home() / ".config" / "living-memory" / "env"
DEFAULT_TIMEOUT_SECONDS = 30.0


class ClientError(RuntimeError):
    """One MCP call failed; the message says which tool and why."""


def resolve_token(
    explicit: str | None = None,
    *,
    env_file: Path | None = DEFAULT_ENV_FILE,
    environ: dict[str, str] | None = None,
) -> str | None:
    """``--token``, else ``$LM_AUTH_TOKEN``, else the EnvironmentFile.

    The same resolution order as ``scripts/check_deployed_protocol.py``.
    Returns ``None`` when no token is configured anywhere, so an
    unauthenticated throwaway instance can be used too. The value is never
    logged by anything in this package.
    """

    if explicit:
        return explicit.strip() or None
    env = os.environ if environ is None else environ
    from_env = (env.get("LM_AUTH_TOKEN") or "").strip()
    if from_env:
        return from_env
    if env_file is None:
        return None
    try:
        raw = env_file.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("LM_AUTH_TOKEN="):
            continue
        value = line.split("=", 1)[1].strip().strip('"').strip("'")
        if value:
            return value
    return None


def _payload(result: Any) -> dict[str, Any]:
    """The tool's dict return, however the MCP layer wrapped it."""

    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        if set(structured) == {"result"} and isinstance(structured["result"], dict):
            return structured["result"]
        return structured
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    raise ClientError(f"tool returned no structured payload: {result!r:.200}")


class LiveMemoryClient:
    """Synchronous facade over one MCP host. All writes go through here."""

    def __init__(
        self,
        url: str = DEFAULT_URL,
        *,
        token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.url = url
        self.token = token
        self.timeout = timeout
        self.calls: list[str] = []

    async def _call_async(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            from fastmcp import Client
        except ImportError as exc:  # pragma: no cover - environment without fastmcp
            raise ClientError(f"fastmcp is required to reach the server: {exc}") from exc

        kwargs: dict[str, Any] = {"timeout": self.timeout, "init_timeout": self.timeout}
        if self.token:
            kwargs["auth"] = self.token
        client = Client(self.url, **kwargs)
        async with client:
            result = await client.call_tool(tool, arguments)
        return _payload(result)

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one tool call to completion. Raises :class:`ClientError`."""

        self.calls.append(tool)
        try:
            return asyncio.run(self._call_async(tool, arguments))
        except ClientError:
            raise
        except Exception as exc:  # noqa: BLE001 - any transport/tool failure
            raise ClientError(f"{tool} against {self.url} failed: {exc}") from exc

    def ping(self) -> dict[str, Any]:
        """Cheapest liveness probe that proves auth works: memory_status."""

        return self.call("memory_status", {})

    def remember(self, content: str, context: dict[str, Any]) -> dict[str, Any]:
        return self.call("memory_remember", {"content": content, "context": context})

    def teach(
        self,
        trace_id: str,
        correction: str | dict[str, Any],
        *,
        confidence: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {"trace_id": trace_id, "correction": correction}
        if confidence is not None:
            arguments["confidence"] = confidence
        if context is not None:
            arguments["context"] = context
        return self.call("memory_teach", arguments)

    def attest(
        self,
        recall_event_id: str,
        evidence: Sequence[str],
        *,
        context: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "recall_event_id": recall_event_id,
            "evidence": list(evidence),
        }
        if context is not None:
            arguments["context"] = context
        if trace_id is not None:
            arguments["trace_id"] = trace_id
        return self.call("memory_attest", arguments)

    def recall(
        self,
        query: str,
        *,
        scope: str | None = None,
        max_results: int = 8,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {"query": query, "max_results": max_results}
        if scope is not None:
            arguments["scope"] = scope
        return self.call("memory_recall", arguments)

    def lookup_nodes(self, node_ids: Sequence[str]) -> dict[str, Any]:
        return self.call("memory_lookup", {"node_ids": list(node_ids)})


def _entry_node(entry: Any) -> dict[str, Any] | None:
    """The node dict inside one shaped recall result, tolerant of dieting."""

    if not isinstance(entry, dict):
        return None
    node = entry.get("node")
    if isinstance(node, dict) and node.get("id"):
        return node
    # Sparse entries may carry only ids at the top level.
    if entry.get("id") or entry.get("node_id"):
        return {"id": entry.get("id") or entry.get("node_id")}
    return None


class McpMemoryIndex:
    """Near-duplicate candidates served by the live host over MCP.

    ``candidates`` returns *full* node contents: the delivery diet may snippet
    or stub a recall result, and grading containment against a snippet would
    under-count exactly the long traces most worth deduplicating, so anything
    not delivered ``full`` is re-fetched through ``memory_lookup``.

    Scope matters here: the server resolves a scope-less recall to the global
    scope only, so callers must pass the scope the write is aimed at — the
    server then searches that scope plus global, which is the same view a
    reader recalling in that project would get. The last
    fetch's node contexts are kept on :attr:`last_contexts` so the runner can
    additionally recognise its own earlier extraction of the same span
    (transcript sha + source span match) without a second round trip.
    """

    available = True

    def __init__(self, client: LiveMemoryClient, *, max_results: int = 8) -> None:
        self.client = client
        self.max_results = max_results
        self.last_contexts: dict[str, dict[str, Any]] = {}

    def candidates(
        self, content: str, *, scope: str | None = None, limit: int = 20
    ) -> Sequence[NodeSnapshot]:
        wanted = min(limit, self.max_results)
        try:
            delivered = self.client.recall(
                content[:600], scope=scope, max_results=wanted
            )
        except ClientError:
            return ()
        entries = delivered.get("results") or []
        by_id: dict[str, dict[str, Any]] = {}
        needs_full: list[str] = []
        for entry in entries:
            node = _entry_node(entry)
            if node is None:
                continue
            node_id = str(node["id"])
            by_id[node_id] = node
            if entry.get("delivery") != "full" or not node.get("content"):
                needs_full.append(node_id)
        if needs_full:
            try:
                fetched = self.client.lookup_nodes(needs_full)
            except ClientError:
                fetched = {}
            for full in fetched.get("results") or []:
                if isinstance(full, dict) and full.get("id"):
                    by_id[str(full["id"])] = full
        self.last_contexts = {
            node_id: dict(node.get("context") or {}) for node_id, node in by_id.items()
        }
        return [
            NodeSnapshot(
                node_id=node_id,
                content=str(node.get("content") or ""),
                scope=node.get("scope"),
                created_at=node.get("created_at"),
            )
            for node_id, node in by_id.items()
            if node.get("content")
        ]

    def embedding(self, content: str) -> Sequence[float] | None:
        return None

    def similar_by_embedding(
        self,
        embedding: Sequence[float],
        *,
        scope: str | None = None,
        threshold: float = 0.95,
        limit: int = 10,
    ) -> Sequence[tuple[NodeSnapshot, float]]:
        return ()
