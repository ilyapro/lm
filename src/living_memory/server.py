"""FastMCP server surface for Living Memory."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any
import sys

from living_memory.config import MemoryConfig, load_config
from living_memory.consolidation import (
    DEFAULT_MIN_CLUSTER_SIZE,
    ConsolidationResult,
    ConsolidationService,
    TeachResult,
)
from living_memory.prompts import retrieval_context_prompt
from living_memory.resources import (
    connection_to_dict,
    global_concepts,
    memory_stats,
    memory_status as status_view,
    node_to_dict,
    project_concepts,
    recent_interactions,
)
from living_memory.retrieval import MemoryRecallService, RecallResult
from living_memory.feedback import apply_pending_recall_feedback
from living_memory.storage import MemoryStore


def create_mcp_server(
    db_path: str | Path | None = None,
    *,
    config: MemoryConfig | None = None,
    config_path: str | Path | None = None,
    name: str = "Living Memory",
    mcp_factory: Any | None = None,
) -> Any:
    """Create a FastMCP server bound to one SQLite store."""

    store = MemoryStore(_resolve_config(db_path=db_path, config=config, config_path=config_path))
    mcp_cls = mcp_factory or _load_fastmcp()
    mcp = mcp_cls(
        name,
        instructions=(
            "Living Memory is your persistent knowledge store across tasks and sessions.\n"
            "\n"
            "## When to use\n"
            "1. **Start of task** — call `memory_recall` with the task description to "
            "retrieve relevant past context (patterns, pitfalls, architecture notes).\n"
            "2. **During work — recall** — call `memory_recall` when you have questions "
            "about the project domain, environment, conventions, or tooling before "
            "searching the codebase. Memory is faster and may already contain the answer "
            "(e.g. 'how are experiments configured?', 'what test runner is used?', "
            "'known issues with tooltip components').\n"
            "   - Use `depth: 'causal'` when debugging failures — it follows cause→effect "
            "chains to find root causes.\n"
            "3. **During work — remember** — call `memory_remember` whenever you discover "
            "something worth preserving:\n"
            "   - Unexpected behaviour or wrong assumptions about the codebase\n"
            "   - Non-obvious architectural constraints or coupling\n"
            "   - Failure root causes and what fixed them\n"
            "   - Newly discovered conventions, configs, or env details\n"
            "4. **Correct mistakes** — if you recall a fact that turns out to be wrong, "
            "call `memory_teach` with the original trace ID and the correction. "
            "This supersedes the bad fact and prevents it from resurfacing.\n"
            "5. **End of task** — call `memory_remember` with a concise task outcome "
            "summary (what was done, key files, anything surprising).\n"
            "\n"
            "## Scope\n"
            "Always set `context.scope` to `'project:<name>'` (e.g. `'project:online'`) "
            "when the fact is project-specific. The scope is inferred from your "
            "environment when omitted, but explicit is better.\n"
            "\n"
            "## Structured context\n"
            "Pass structured metadata in the `context` parameter of `memory_remember`:\n"
            "```json\n"
            '{"scope": "project:online", "task": "EZ-13771", "agent": "codex"}\n'
            "```\n"
            "This enables precise recall by task or project later.\n"
            "\n"
            "## How to write good traces\n"
            "- Be **concrete and factual**: prefer 'file X exports Y, not Z' over vague notes.\n"
            "- Include **file paths, function names, config keys** — specifics enable recall.\n"
            "- One fact per trace. Short is better than long.\n"
            "\n"
            "## What NOT to store\n"
            "- Routine actions ('I ran the tests') without new insight.\n"
            "- Copies of code — reference file paths instead.\n"
            "- Speculation or plans — only store verified facts.\n"
        ),
    )
    runtime_lock = RLock()
    _attach(mcp, "memory_store", store)
    _register_tools(mcp, store, runtime_lock)
    _register_resources(mcp, store, runtime_lock)
    _register_prompts(mcp, store, runtime_lock)
    return mcp


def run_server(
    db_path: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    mcp = create_mcp_server(db_path=db_path, config_path=config_path)
    if transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=transport, host=host, port=port)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    db_path = args.db or args.sqlite_file
    try:
        run_server(
            db_path=db_path,
            config_path=args.config,
            transport=args.transport,
            host=args.host,
            port=args.port,
        )
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _register_tools(mcp: Any, store: MemoryStore, runtime_lock: Any) -> None:
    recall_service = MemoryRecallService(store)
    consolidation_service = ConsolidationService(store)

    @mcp.tool
    def memory_remember(
        content: str,
        context: dict[str, Any] | None = None,
        feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a new append-only trace."""

        with runtime_lock:
            node = store.append_trace(content, context, feedback=feedback)
            implicit_feedback = apply_pending_recall_feedback(store, node)
            node = implicit_feedback.trace
            auto_consolidation = _auto_consolidate_if_due(
                store,
                consolidation_service,
                node.scope,
            )
            return {
                "node": node_to_dict(node),
                "implicit_feedback": _implicit_feedback_to_dict(implicit_feedback),
                "auto_consolidation": auto_consolidation,
            }

    @mcp.tool
    def memory_teach(
        trace_id: str,
        correction: str | dict[str, Any],
        confidence: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a corrective trace and connect it to the original."""

        with runtime_lock:
            taught = consolidation_service.memory_teach(
                trace_id,
                correction,
                confidence=confidence,
                context=context,
            )
            return _teach_result_to_dict(taught)

    @mcp.tool
    def memory_connect(
        id_a: str,
        id_b: str,
        relation_type: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Connect two existing memory nodes."""

        with runtime_lock:
            connection = recall_service.memory_connect(
                id_a,
                id_b,
                relation_type,
                weight=weight,
                metadata=metadata,
            )
            return {"connection": connection_to_dict(connection)}

    @mcp.tool
    def memory_recall(
        query: str,
        scope: str | None = None,
        depth: int | str | None = 1,
        max_results: int = 10,
        ambient_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant memory nodes using scope, text, vector, and graph signals."""

        with runtime_lock:
            results = recall_service.memory_recall(
                query,
                scope=scope,
                ambient_context=ambient_context,
                depth=depth,
                max_results=max_results,
            )
            return {
                "query": query,
                "scope": scope,
                "recall_event_id": recall_service.last_recall_event_id,
                "count": len(results),
                "results": [_recall_result_to_dict(result) for result in results],
            }

    @mcp.tool
    def memory_consolidate(scope: str | None = None, force: bool = False) -> dict[str, Any]:
        """Run one consolidation and decay maintenance pass."""

        with runtime_lock:
            result = consolidation_service.memory_consolidate(scope=scope, force=force)
            return _consolidation_result_to_dict(result)

    @mcp.tool
    def memory_forget(id: str, reason: str | None = None) -> dict[str, Any]:
        """Soft-delete a memory node while preserving the stored record."""

        with runtime_lock:
            node = consolidation_service.memory_forget(id, reason)
            return {"node": node_to_dict(node)}

    @mcp.tool
    def memory_status(scope: str | None = None) -> dict[str, Any]:
        """Describe current memory phase, coverage, confidence, and policy."""

        with runtime_lock:
            return status_view(store, scope=scope)


def _register_resources(mcp: Any, store: MemoryStore, runtime_lock: Any) -> None:
    @mcp.resource("memory://global/concepts")
    def resource_global_concepts() -> dict[str, Any]:
        with runtime_lock:
            return global_concepts(store)

    @mcp.resource("memory://project/{name}/concepts")
    def resource_project_concepts(name: str) -> dict[str, Any]:
        with runtime_lock:
            return project_concepts(store, name)

    @mcp.resource("memory://stats")
    def resource_stats() -> dict[str, Any]:
        with runtime_lock:
            return memory_stats(store)

    @mcp.resource("memory://recent")
    def resource_recent() -> dict[str, Any]:
        with runtime_lock:
            return recent_interactions(store)


def _register_prompts(mcp: Any, store: MemoryStore, runtime_lock: Any) -> None:
    @mcp.prompt(name="memory://prompt/retrieval_context")
    def retrieval_context(
        task: str = "",
        scope: str | None = None,
        max_concepts: int = 5,
        min_confidence: float = 0.0,
        retrieval_policy: str = "balanced",
        agent: str | None = None,
        ambient_context: dict[str, Any] | None = None,
    ) -> str:
        """Return a formatted active memory context block."""

        with runtime_lock:
            return retrieval_context_prompt(
                store,
                task=task,
                scope=scope,
                max_concepts=max_concepts,
                min_confidence=min_confidence,
                retrieval_policy=retrieval_policy,
                agent=agent,
                ambient_context=ambient_context,
            )


def _auto_consolidate_if_due(
    store: MemoryStore,
    consolidation_service: ConsolidationService,
    scope: str,
) -> dict[str, Any] | None:
    row = store.connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM nodes
        WHERE level = 'trace' AND scope = ? AND decayed = 0
        """,
        (scope,),
    ).fetchone()
    trace_count = int(row["count"])
    if trace_count < DEFAULT_MIN_CLUSTER_SIZE or trace_count % DEFAULT_MIN_CLUSTER_SIZE != 0:
        return None
    return _consolidation_result_to_dict(
        consolidation_service.memory_consolidate(scope=scope, force=False)
    )


def _resolve_config(
    *,
    db_path: str | Path | None,
    config: MemoryConfig | None,
    config_path: str | Path | None,
) -> MemoryConfig:
    if config is not None and config_path is not None:
        raise ValueError("provide config or config_path, not both")
    if config_path is not None:
        resolved = load_config(config_path)
    else:
        resolved = config or MemoryConfig()
    if db_path is None:
        return resolved
    return replace(resolved, db_path=Path(db_path))


def _load_fastmcp() -> Any:
    try:
        from fastmcp import FastMCP
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "FastMCP is required to run the MCP server. Install project dependencies first."
        ) from exc
    return FastMCP


def _recall_result_to_dict(result: RecallResult) -> dict[str, Any]:
    return {
        "node": node_to_dict(result.node),
        "score": result.score,
        "bm25_score": result.bm25_score,
        "vector_score": result.vector_score,
        "graph_score": result.graph_score,
        "scope_rank": result.scope_rank,
        "methods": list(result.methods),
        "path": list(result.path),
        "recall_event_id": result.recall_event_id,
    }


def _consolidation_result_to_dict(result: ConsolidationResult) -> dict[str, Any]:
    return {
        "concepts_created": [node_to_dict(node) for node in result.concepts_created],
        "concepts_updated": [node_to_dict(node) for node in result.concepts_updated],
        "decayed": [node_to_dict(node) for node in result.decayed],
        "clusters_considered": result.clusters_considered,
        "traces_considered": result.traces_considered,
    }


def _teach_result_to_dict(result: TeachResult) -> dict[str, Any]:
    return {
        "corrective_trace": node_to_dict(result.corrective_trace),
        "supersedes": connection_to_dict(result.supersedes),
        "original": node_to_dict(result.original),
        "implicit_feedback": _implicit_feedback_to_dict(result.implicit_feedback),
    }


def _implicit_feedback_to_dict(feedback: Any) -> dict[str, Any]:
    if feedback is None:
        return {"recall_event_ids": [], "linked_node_ids": [], "feedback_applied": False}
    return {
        "recall_event_ids": [event.id for event in feedback.events],
        "linked_node_ids": list(feedback.linked_node_ids),
        "feedback_applied": feedback.feedback_applied,
    }


def _attach(target: Any, name: str, value: Any) -> None:
    try:
        setattr(target, name, value)
    except Exception:
        return


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run the Living Memory MCP server over one SQLite file.")
    parser.add_argument(
        "sqlite_file",
        nargs="?",
        help="SQLite database file. Defaults to the configured path.",
    )
    parser.add_argument(
        "--db",
        dest="db",
        help="SQLite database file. Overrides the optional positional path.",
    )
    parser.add_argument(
        "--config",
        help="Optional TOML config path.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "http", "sse"),
        default="stdio",
        help="MCP transport to run.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP or SSE transport.")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP or SSE transport.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
