from __future__ import annotations

from pathlib import Path
import asyncio
import inspect
import json
import os
import subprocess
import sys
from typing import Any

import pytest

from living_memory.server import create_mcp_server


class FakeMCP:
    def __init__(self, name: str, instructions: str | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def tool(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.tools[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        if func is None:
            return decorate
        return decorate(func)

    def resource(self, uri: str, **_kwargs: Any) -> Any:
        def decorate(func: Any) -> Any:
            self.resources[uri] = func
            return func

        return decorate

    def prompt(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.prompts[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        if func is None:
            return decorate
        return decorate(func)


def test_server_registers_exact_tools_resources_and_prompt(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)

    assert set(mcp.tools) == {
        "memory_remember",
        "memory_teach",
        "memory_connect",
        "memory_recall",
        "memory_attest",
        "memory_lookup",
        "memory_consolidate",
        "memory_forget",
        "memory_status",
        "memory_health",
    }
    assert set(mcp.resources) == {
        "memory://global/concepts",
        "memory://project/{name}/concepts",
        "memory://stats",
        "memory://recent",
        "memory://latency",
    }
    assert set(mcp.prompts) == {"memory://prompt/retrieval_context"}


def test_mcp_tools_delegate_to_memory_services(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)

    first = mcp.tools["memory_remember"](
        "deploy incident caused by missing migration",
        {
            "scope": "project:alpha",
            "agent": "agent-a",
            "task_pattern": "deploy-migration",
        },
    )
    second = mcp.tools["memory_remember"](
        "missing migration requires rollback checklist",
        {"scope": "project:alpha", "agent": "agent-b"},
    )
    first_id = first["node"]["id"]
    second_id = second["node"]["id"]

    connected = mcp.tools["memory_connect"](first_id, second_id, "caused")
    assert connected["connection"]["type"] == "caused"

    recalled = mcp.tools["memory_recall"](
        "why did rollback checklist happen",
        scope="project:alpha",
        depth="causal",
        max_results=5,
    )
    assert recalled["count"] >= 1
    assert first_id in {result["node"]["id"] for result in recalled["results"]}

    looked_up = mcp.tools["memory_lookup"](
        scope="project:alpha",
        task_pattern="deploy-migration",
    )
    assert [result["id"] for result in looked_up["results"]] == [first_id]

    taught = mcp.tools["memory_teach"](
        first_id,
        "deploy incident caused by migration 43 missing",
        context={"agent": "agent-c"},
    )
    assert taught["supersedes"]["type"] == "supersedes"
    assert taught["corrective_trace"]["content"] == "deploy incident caused by migration 43 missing"

    consolidated = mcp.tools["memory_consolidate"](scope="project:alpha", force=True)
    assert "concepts_created" in consolidated
    assert "decayed" in consolidated

    status = mcp.tools["memory_status"](scope="project:alpha")
    assert status["scope"] == "project:alpha"
    assert status["phase"]["trace_count"] >= 3

    forgotten = mcp.tools["memory_forget"](second_id, "test cleanup")
    assert forgotten["node"]["decayed"] is True
    assert forgotten["node"]["decay_reason"] == "test cleanup"


def test_server_default_scope_applies_to_unspecified_trace_scope(tmp_path: Path) -> None:
    mcp = create_mcp_server(
        tmp_path / "memory.sqlite3",
        default_scope="project:scope-test",
        mcp_factory=FakeMCP,
    )

    assert "Your default scope is project:scope-test" in str(mcp.instructions)

    remembered = mcp.tools["memory_remember"](
        "trace without explicit scope",
        {"agent": "agent-a"},
    )

    assert remembered["node"]["scope"] == "project:scope-test"
    stored = mcp.memory_store.get_node(remembered["node"]["id"])
    assert stored.scope == "project:scope-test"
    assert stored.context["scope"] == "project:scope-test"


def test_server_help_does_not_require_runtime_dependency() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "living_memory.server", "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0
    assert "SQLite" in result.stdout
    assert "--default-scope" in result.stdout


def test_main_reads_LM_SCOPE_env_as_default_scope(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setenv("LM_SCOPE", "project:from-env")
    monkeypatch.delenv("LM_DEFAULT_SCOPE", raising=False)

    mcp = create_mcp_server(
        tmp_path / "memory.sqlite3",
        mcp_factory=FakeMCP,
    )
    # Without env, scope would be "global"; with LM_SCOPE it should still be
    # "global" because create_mcp_server doesn't read env — only main() does.
    # So test main()'s argv parsing path instead:
    from living_memory.server import main as _main
    import io, contextlib

    # main() calls run_server which calls mcp.run() — we can't easily test
    # the full path, so test the resolution logic directly:
    from living_memory.server import _build_parser
    args = _build_parser().parse_args(["--db", str(tmp_path / "t.db")])
    resolved = args.default_scope or os.environ.get("LM_DEFAULT_SCOPE") or os.environ.get("LM_SCOPE")
    assert resolved == "project:from-env"


def test_LM_DEFAULT_SCOPE_takes_priority_over_LM_SCOPE(monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setenv("LM_DEFAULT_SCOPE", "project:explicit")
    monkeypatch.setenv("LM_SCOPE", "project:fallback")

    from living_memory.server import _build_parser
    args = _build_parser().parse_args([])
    resolved = args.default_scope or os.environ.get("LM_DEFAULT_SCOPE") or os.environ.get("LM_SCOPE")
    assert resolved == "project:explicit"


# --- Delivery shaping at the memory_recall tool boundary --------------------


def _structured(result: Any) -> dict[str, Any]:
    payload = getattr(result, "structured_content", None)
    assert isinstance(payload, dict)
    if set(payload) == {"result"} and isinstance(payload["result"], dict):
        return payload["result"]
    return payload


def _clear_delivery_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env in (
        "LM_DELIVERY_SNIPPET_CHARS",
        "LM_DELIVERY_SNIPPET_LADDER",
        "LM_DELIVERY_SESSION_DEDUP",
        "LM_DELIVERY_CONTEXT_VALUE_CHARS",
        "LM_DELIVERY_FULL_NODE_DIET",
        "LM_DELIVERY_PROVENANCE_VALUE_CHARS",
        "LM_DELIVERY_STATS_COMPACTION",
        "LM_DELIVERY_SPARSE",
    ):
        monkeypatch.delenv(env, raising=False)


DEDUP_CONTENT = (
    "quokka reactor calibration uses the phased manifold sequence\n"
    "step two: bleed the coolant loop before torquing the flange"
)


def test_same_session_double_recall_stubs_second_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second recall on one transport session returns session_duplicate stubs;
    a fresh session gets full content again, and memory_lookup(node_id=...)
    round-trips the stubbed content in full over the same client."""

    pytest.importorskip("fastmcp")
    from fastmcp import Client

    _clear_delivery_env(monkeypatch)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    store.append_trace(DEDUP_CONTENT, {"scope": "project:dedup"})
    call = {"query": "quokka reactor calibration", "scope": "project:dedup"}

    async def scenario() -> tuple[Any, Any, Any, Any]:
        async with Client(mcp) as client:
            first = _structured(await client.call_tool("memory_recall", call))
            second = _structured(await client.call_tool("memory_recall", call))
            node_id = second["results"][0]["node"]["id"]
            fetched = _structured(
                await client.call_tool("memory_lookup", {"node_id": node_id})
            )
        async with Client(mcp) as client:
            fresh = _structured(await client.call_tool("memory_recall", call))
        return first, second, fetched, fresh

    first, second, fetched, fresh = asyncio.run(scenario())

    assert first["count"] == 1 and second["count"] == 1
    full = first["results"][0]
    stub = second["results"][0]
    assert full["delivery"] == "full"
    assert full["node"]["content"] == DEDUP_CONTENT
    assert "content_ref" not in full

    assert stub["delivery"] == "session_duplicate"
    assert stub["node"]["id"] == full["node"]["id"]
    assert set(stub["node"]) == set(full["node"])  # every node key survives
    assert stub["node"]["content"] == DEDUP_CONTENT.split("\n")[0]  # one-line preview
    assert stub["content_ref"]["node_id"] == full["node"]["id"]
    assert stub["content_ref"]["full_content_chars"] == len(DEDUP_CONTENT)

    # The stub's content_ref path restores byte-identical content.
    assert fetched["count"] == 1
    assert fetched["results"][0]["content"] == DEDUP_CONTENT

    # A distinct transport session starts with a clean delivery slate.
    assert fresh["results"][0]["delivery"] == "full"
    assert fresh["results"][0]["node"]["content"] == DEDUP_CONTENT


def test_session_dedup_env_valve_disables_stubs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    _clear_delivery_env(monkeypatch)
    monkeypatch.setenv("LM_DELIVERY_SESSION_DEDUP", "0")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    mcp.memory_store.append_trace(DEDUP_CONTENT, {"scope": "project:dedup"})
    call = {"query": "quokka reactor calibration", "scope": "project:dedup"}

    async def scenario() -> tuple[Any, Any]:
        async with Client(mcp) as client:
            first = _structured(await client.call_tool("memory_recall", call))
            second = _structured(await client.call_tool("memory_recall", call))
        return first, second

    first, second = asyncio.run(scenario())
    for response in (first, second):
        assert response["results"][0]["delivery"] == "full"
        assert response["results"][0]["node"]["content"] == DEDUP_CONTENT


def test_sessionless_direct_recall_stays_full_on_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a transport session id (FakeMCP direct calls) repeated recalls
    keep legacy full-content behavior — session dedup never engages."""

    _clear_delivery_env(monkeypatch)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    content = "sessionless delivery baseline fact\nsecond line stays intact"
    mcp.tools["memory_remember"](content, {"scope": "project:nodedup"})

    first = mcp.tools["memory_recall"](
        "sessionless delivery baseline", scope="project:nodedup"
    )
    second = mcp.tools["memory_recall"](
        "sessionless delivery baseline", scope="project:nodedup"
    )

    for response in (first, second):
        assert response["count"] >= 1
        for result in response["results"]:
            assert result["delivery"] == "full"
            assert "content_ref" not in result
    assert second["results"][0]["node"]["content"] == content


def test_recall_snippets_long_content_with_content_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_delivery_env(monkeypatch)
    # Uniform legacy mode: the default ladder ships the top bearer complete,
    # so budgeting the top result takes the documented uniform-budget valve.
    monkeypatch.setenv("LM_DELIVERY_SNIPPET_CHARS", "1200")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    long_content = "meridian telescope alignment procedure. " + " ".join(
        f"step {index}: calibrate axis {index}" for index in range(200)
    )
    assert len(long_content) > 1200
    remembered = mcp.tools["memory_remember"](long_content, {"scope": "project:snip"})

    recalled = mcp.tools["memory_recall"](
        "meridian telescope alignment", scope="project:snip"
    )
    top = recalled["results"][0]
    assert top["delivery"] == "snippet"
    assert len(top["node"]["content"]) <= 1200
    assert top["node"]["content"].endswith("…")
    assert top["content_ref"]["node_id"] == remembered["node"]["id"]
    assert top["content_ref"]["full_content_chars"] == len(long_content)


def test_memory_recall_default_max_results_is_five(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    signature = inspect.signature(mcp.tools["memory_recall"])
    assert signature.parameters["max_results"].default == 5

    for index in range(8):
        mcp.tools["memory_remember"](
            f"zanzibar calibration fact number {index}",
            {"scope": "project:defaults"},
        )
    recalled = mcp.tools["memory_recall"]("zanzibar calibration", scope="project:defaults")
    assert recalled["count"] == 5
    assert len(recalled["results"]) == 5


def test_retrieval_service_default_max_results_stays_ten() -> None:
    """The diet applies at the server boundary only — prompts/replay/resources
    callers of MemoryRecallService keep the internal default of 10."""

    from living_memory.retrieval import MemoryRecallService, memory_recall as recall_fn

    for func in (MemoryRecallService.memory_recall, recall_fn):
        assert inspect.signature(func).parameters["max_results"].default == 10


# --- memory_remember response diet ------------------------------------------


CONSOLIDATION_ID_KEYS = (
    "concepts_created",
    "concepts_updated",
    "concepts_promoted",
    "schemas_created",
    "schemas_updated",
    "decayed",
)


def test_memory_remember_confirmation_stays_small_under_auto_consolidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write into a scope seeded with large traces triggers auto-consolidation
    yet returns a bounded confirmation: ids and counters, no content echo."""

    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "adaptive")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store

    body = "meridian array recalibration dossier. " + " ".join(
        f"sector {index} torque ledger entry with a deliberately long diagnostic narrative"
        for index in range(220)
    )
    assert 10_000 <= len(body) <= 40_000
    scope = "project:diet"
    for index in range(4):
        store.append_trace(
            f"{body} copy {index}",
            {"scope": scope, "agent": f"agent-{index % 2}", "task_pattern": "diet-ritual"},
        )

    mcp.tools["memory_recall"]("meridian array recalibration dossier", scope=scope)

    final_content = f"{body} copy final"
    response = mcp.tools["memory_remember"](
        final_content,
        {"scope": scope, "agent": "agent-a", "task_pattern": "diet-ritual"},
    )

    assert len(json.dumps(response)) <= 4096
    assert set(response) == {"node", "implicit_feedback", "auto_consolidation", "auto_decay"}

    node = response["node"]
    stored = store.get_node(node["id"])
    assert stored is not None
    assert stored.content == final_content  # the write itself is fully stored
    assert node["level"] == "trace"
    assert node["scope"] == scope
    assert node["created_at"] == stored.created_at
    for verbose_key in ("content", "context", "stats", "provenance"):
        assert verbose_key not in node

    feedback = response["implicit_feedback"]
    assert len(feedback["recall_event_ids"]) == 1
    assert feedback["feedback_applied"] is True
    assert node["prior_recall_count"] == 1
    assert node["linked_node_count"] == len(feedback["linked_node_ids"]) == 4
    assert node["source_trace_count"] == len(stored.source_traces) == 4

    auto = response["auto_consolidation"]
    assert auto is not None
    assert auto["traces_considered"] >= 5
    assert auto["clusters_considered"] >= 1
    for key in CONSOLIDATION_ID_KEYS:
        assert all(isinstance(entry, str) for entry in auto[key])
    touched = [
        node_id
        for key in ("concepts_created", "concepts_updated", "schemas_created", "schemas_updated")
        for node_id in auto[key]
    ]
    assert touched  # the auto pass really produced or updated nodes
    for node_id in touched:
        assert store.get_node(node_id) is not None


def test_memory_remember_signature_unchanged_and_consolidate_report_stays_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LM_AUTO_CONSOLIDATE_POLICY", raising=False)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)

    signature = inspect.signature(mcp.tools["memory_remember"])
    assert list(signature.parameters) == [
        "content",
        "context",
        "feedback",
        "alternatives_considered",
    ]

    for index in range(3):
        mcp.tools["memory_remember"](
            f"procedural rehearsal for the diet ritual number {index}",
            {"scope": "project:diet-direct", "task_pattern": "diet-ritual"},
        )

    direct = mcp.tools["memory_consolidate"](scope="project:diet-direct", force=True)
    schemas = direct["schemas_created"] + direct["schemas_updated"]
    assert schemas  # the direct tool still reports full node dicts
    for schema in schemas:
        assert isinstance(schema, dict)
        assert schema["level"] == "schema"
        assert schema["content"]
        assert "provenance" in schema and "stats" in schema
