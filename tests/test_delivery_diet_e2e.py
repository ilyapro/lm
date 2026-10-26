"""End-to-end proof of the recall delivery diet over a real MCP client.

Every scenario drives the real FastMCP in-memory client (``fastmcp.Client``)
against ``create_mcp_server`` so transport-session identity, delivery shaping,
and the compact write confirmation are exercised exactly as a client sees
them on the wire:

* Session dedup — the first recall on a connection delivers full content, the
  repeat delivers ``session_duplicate`` stubs and a substantially smaller
  serialized response; a fresh connection gets full content again, and calls
  without any transport identity (direct tool-function calls, the degradation
  control) never stub.
* Source identity — a concept and a trace with identical bytes retain their
  own content when their recorded sources differ.
* Snippet ladder — the top-ranked content-bearer has a finite default budget;
  a configured ``LM_DELIVERY_SNIPPET_LADDER`` can tighten it further, and
  ``memory_lookup(node_id=...)`` returns the stored bytes unchanged (the
  re-fetch path).
* Context diet — a procedural schema node whose ``context.procedure``
  duplicates its multi-KB content arrives with that value compacted to
  ``{"count", "chars"}`` on every delivery class, full included (the
  full-node diet), keeping whole responses bounded, while
  ``memory_lookup(node_id=..., context_key="procedure")`` returns that one
  original field, and ``view="full"`` returns the complete stored node.
* Remember diet — the ``memory_remember`` wire response stays bounded while
  the trace itself is durably stored and recallable.
* Feedback closure — recall -> stub recall -> remember on one connection
  still closes both events (``feedback_applied`` 0 -> 1) and lands them in
  the new trace's provenance: shaping the response never touches the
  recorded event. The remember content shares zero tokens with the recall
  query, so transport identity is the only available matching rule.
* Digest consolidation — ``memory_consolidate`` over a seeded near-identical
  cluster returns a concept byte-distinct from every source trace.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client

from living_memory.delivery import DEFAULT_SNIPPET_LADDER, SNIPPET_LADDER_ENV
from living_memory.embeddings import tokenize
from living_memory.resources import node_to_dict
from living_memory.server import create_mcp_server

from test_transport_identity import FakeMCP, _structured

SNIPPET_LIMIT = 1200  # default LM_DELIVERY_SNIPPET_CHARS (uniform ladder-off mode)

# Node fields every delivery class keeps, whatever the sparse diet drops.
CORE_NODE_KEYS = {"id", "level", "content", "scope", "provenance", "decayed", "decay_reason"}

RECALL_SCOPE = "project:dietprobe"
RECALL_STEM = "meridian dossier torque calibration"


@pytest.fixture(autouse=True)
def _default_delivery_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every scenario under default delivery/consolidation settings."""

    for env in (
        "LM_DELIVERY_SNIPPET_CHARS",
        "LM_DELIVERY_SNIPPET_LADDER",
        "LM_DELIVERY_SESSION_DEDUP",
        "LM_DELIVERY_CONTEXT_VALUE_CHARS",
        "LM_DELIVERY_FULL_NODE_DIET",
        "LM_DELIVERY_PROVENANCE_VALUE_CHARS",
        "LM_DELIVERY_STATS_COMPACTION",
        "LM_DELIVERY_SPARSE",
        "LM_AUTO_CONSOLIDATE_POLICY",
    ):
        monkeypatch.delenv(env, raising=False)


def _server(tmp_path: Path) -> tuple[Any, Any]:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    return mcp, mcp.memory_store


async def _recall(client: Client, query: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool("memory_recall", {"query": query, **arguments})
    return _structured(result)


def _wire_bytes(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, sort_keys=True))


def _by_id(response: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["id"]: entry for entry in response["results"]}


def _bulk_content(tag: str, clauses: int) -> str:
    """A body heavy enough to measure, sized in ~115-char clause lines.

    Callers pick ``clauses`` so every body stays below the smallest ladder
    budget in play for their result count — the scenarios here probe dedup
    and stubbing, not snippeting.
    """

    lines = [f"{RECALL_STEM} {tag} ledger"]
    for index in range(clauses):
        lines.append(
            f"{tag} clause {index}: torque calibration paragraph keeping the "
            f"ledger dense enough to make full delivery measurably heavy"
        )
    return "\n".join(lines)


# --- Session dedup: full once per transport session, stubs on repeat --------


def test_session_dedup_full_then_stub_then_fresh_session_full_again(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, str], dict[str, Any], dict[str, Any], dict[str, Any]]:
        mcp, store = _server(tmp_path)
        contents: dict[str, str] = {}
        for tag in ("alpha", "beta", "gamma"):
            node = store.append_trace(_bulk_content(tag, clauses=5), {"scope": RECALL_SCOPE})
            contents[node.id] = node.content

        async with Client(mcp) as client:
            first = await _recall(client, RECALL_STEM, scope=RECALL_SCOPE)
            second = await _recall(client, RECALL_STEM, scope=RECALL_SCOPE)
        async with Client(mcp) as client:
            fresh = await _recall(client, RECALL_STEM, scope=RECALL_SCOPE)
        return contents, first, second, fresh

    contents, first, second, fresh = asyncio.run(scenario())

    # Below the smallest ladder budget for three bearers (complete, 1000, 700):
    # every first delivery is full inline, whatever the ranking order.
    smallest_budget = DEFAULT_SNIPPET_LADDER[2]
    for content in contents.values():
        assert 500 < len(content) < smallest_budget
    assert set(_by_id(first)) == set(_by_id(second)) == set(_by_id(fresh)) == set(contents)

    for node_id, entry in _by_id(first).items():
        assert entry["delivery"] == "full"
        assert entry["content_ref"] == {"node_id": node_id}
        assert entry["content"] == contents[node_id]

    for node_id, entry in _by_id(second).items():
        assert entry["delivery"] == "session_duplicate"
        stub = entry
        assert stub["content"] != contents[node_id]
        assert len(stub["content"]) <= 200  # a one-line preview, not the body
        assert contents[node_id].startswith(stub["content"])
        assert set(stub) >= CORE_NODE_KEYS  # stubs keep the core node shape
        ref = entry["content_ref"]
        assert ref["node_id"] == node_id
        assert ref["full_content_chars"] == len(contents[node_id])
        assert "fetch" not in ref  # sparse diet: the recall docstring names the lookup

    # The repeat response is strictly and substantially smaller on the wire.
    first_bytes, second_bytes = _wire_bytes(first), _wire_bytes(second)
    assert second_bytes < first_bytes
    assert second_bytes <= 0.7 * first_bytes

    # A new connection is a new transport session: full content again.
    for node_id, entry in _by_id(fresh).items():
        assert entry["delivery"] == "full"
        assert entry["content"] == contents[node_id]


def test_direct_calls_without_transport_identity_never_stub(tmp_path: Path) -> None:
    """Degradation control: no request context means no transport session id,
    so repeated identical recalls keep legacy full delivery every time."""

    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    expected: dict[str, str] = {}
    for tag in ("alpha", "beta"):
        # Below the two-bearer ladder budgets (complete, 1000): the control
        # isolates stubbing from snippeting.
        node = store.append_trace(_bulk_content(tag, clauses=7), {"scope": RECALL_SCOPE})
        expected[node.id] = node.content

    responses = [mcp.tools["memory_recall"](RECALL_STEM, scope=RECALL_SCOPE) for _ in range(2)]

    for response in responses:
        by_id = _by_id(response)
        assert set(by_id) == set(expected)
        for node_id, entry in by_id.items():
            assert entry["delivery"] == "full"
            assert entry["content_ref"] == {"node_id": node_id}
            assert entry["content"] == expected[node_id]

    events = store.list_recall_events()
    assert len(events) == 2
    assert all(event.transport_session_id is None for event in events)


# --- Source identity: identical content with different provenance ----------


def test_different_sources_deliver_their_own_content(tmp_path: Path) -> None:
    scope = "project:dietwin"
    body = (
        "obsidian relay charter checklist\n"
        "grounding verification, antenna sweep, and a signed handover entry "
        "are required before the relay charter is considered active."
    )

    async def scenario() -> tuple[str, str, dict[str, Any]]:
        mcp, store = _server(tmp_path)
        trace = store.append_trace(body, {"scope": scope})
        # Pre-digest consolidation copied trace bytes verbatim into concepts;
        # seed that exact legacy shape through direct store writes.
        concept = store.create_node(
            level="concept",
            content=body,
            context={"scope": scope, "agent": "memory_consolidate"},
            provenance={"source_traces": [trace.id]},
        )
        assert store.get_node(trace.id).decayed is False  # the pair coexists
        async with Client(mcp) as client:
            response = await _recall(client, "obsidian relay charter", scope=scope)
        return trace.id, concept.id, response

    trace_id, concept_id, response = asyncio.run(scenario())

    by_id = _by_id(response)
    assert set(by_id) == {trace_id, concept_id}

    assert all(entry["delivery"] == "full" for entry in by_id.values())
    assert all(entry["content"] == body for entry in by_id.values())
    assert all("duplicate_of" not in entry["content_ref"] for entry in by_id.values())


# --- Snippet ladder: bounded default, tighter budget + refetch --------------


def test_long_content_top_bounded_by_default_then_tighter_snippet_and_refetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope = "project:dietlong"
    paragraphs = ["glacier archive manifest overview"]
    for index in range(12):
        paragraphs.append(
            f"glacier archive manifest section {index}: "
            + "retention policy detail sentence for the archived manifest. " * 3
        )
    long_content = "\n\n".join(paragraphs)

    async def scenario() -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
        mcp, store = _server(tmp_path)
        node = store.append_trace(long_content, {"scope": scope})
        async with Client(mcp) as client:
            default_recall = await _recall(client, "glacier archive manifest", scope=scope)
        # A configured uniform ladder budgets even the top bearer; the fresh
        # connection below sidesteps session dedup.
        monkeypatch.setenv(SNIPPET_LADDER_ENV, str(SNIPPET_LIMIT))
        async with Client(mcp) as client:
            ladder_recall = await _recall(client, "glacier archive manifest", scope=scope)
            looked_up = _structured(
                await client.call_tool("memory_lookup", {"node_id": node.id})
            )
        return node.id, default_recall, ladder_recall, looked_up

    node_id, default_recall, ladder_recall, looked_up = asyncio.run(scenario())
    assert len(long_content) > SNIPPET_LIMIT

    # Default ladder bounds oversized content while retaining a lookup path.
    top_entry = _by_id(default_recall)[node_id]
    assert top_entry["delivery"] == "snippet"
    assert len(top_entry["content"]) <= DEFAULT_SNIPPET_LADDER[0]
    assert top_entry["content_ref"]["node_id"] == node_id

    entry = _by_id(ladder_recall)[node_id]
    assert entry["delivery"] == "snippet"
    snippet = entry["content"]
    assert snippet != long_content
    assert len(snippet) <= SNIPPET_LIMIT
    source = snippet.removesuffix("\n[Context omitted]").removesuffix("\n[Incomplete passage]")
    assert source.endswith((".", "…"))
    assert all(passage.removesuffix("…") in long_content for passage in source.splitlines())
    assert "glacier archive manifest" in snippet
    assert entry["content_ref"] == {
        "node_id": node_id,
        "full_content_chars": len(long_content),
    }

    # The advertised re-fetch returns the stored bytes unchanged.
    assert looked_up["count"] == 1
    assert looked_up["missing"] == []
    assert looked_up["results"][0]["id"] == node_id
    assert looked_up["results"][0]["content"] == long_content


# --- Context diet: bulky context values compacted on every delivery class ---


def test_bulky_context_procedure_compacted_on_all_delivery_classes(tmp_path: Path) -> None:
    scope = "project:dietcontext"
    steps = [
        f"voltaic pergola rite step {index}: "
        + "hold the calibration clause ledger for this rehearsal pass. " * 25
        for index in range(4)
    ]
    # Procedural schemas store their content twice: joined as ``content`` and
    # step-by-step as ``context.procedure`` — the exact shape that leaked.
    body = "Procedure: voltaic pergola rite\n" + "\n".join(
        f"{index + 1}. {step}" for index, step in enumerate(steps)
    )
    context = {
        "scope": scope,
        "agent": "memory_consolidate",
        "procedure": steps,
        "procedure_key": "voltaic pergola rite",
        "task_pattern": "voltaic-pergola-rite",
        "trigger": "voltaic pergola rite",
    }

    async def scenario() -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        mcp, store = _server(tmp_path)
        node = store.create_node(
            level="schema",
            content=body,
            context=dict(context),
            provenance={"strategy": "procedural"},
        )
        async with Client(mcp) as client:
            first = await _recall(client, "voltaic pergola rite", scope=scope, view="full")
            second = await _recall(client, "voltaic pergola rite", scope=scope, view="full")
            looked_up = _structured(
                await client.call_tool("memory_lookup", {"node_id": node.id})
            )
            procedure = _structured(await client.call_tool(
                "memory_lookup", {"node_id": node.id, "context_key": "procedure"}
            ))
            full = _structured(await client.call_tool(
                "memory_lookup", {"node_id": node.id, "view": "full"}
            ))
        return node.id, first, second, looked_up, procedure, full

    node_id, first, second, looked_up, procedure, full = asyncio.run(scenario())

    procedure_chars = len(json.dumps(steps, ensure_ascii=False))
    assert procedure_chars > 4096  # the leak this diet closes dwarfs the bounds below

    assert looked_up["results"][0]["content"] == body
    assert "context" not in looked_up["results"][0]
    assert procedure["results"][0]["context"] == {"procedure": steps}
    stored_context = full["results"][0]["context"]
    assert stored_context["procedure"] == steps

    first_entry = {entry["node"]["id"]: entry for entry in first["results"]}[node_id]
    second_entry = {entry["node"]["id"]: entry for entry in second["results"]}[node_id]
    # Top-ranked bearer: complete content, yet the full-node diet still
    # compacts the context that would re-ship that same content.
    assert first_entry["delivery"] == "full"
    assert first_entry["node"]["content"] == body
    assert first_entry["content_ref"] == {"node_id": node_id}  # lookup returns more
    assert second_entry["delivery"] == "session_duplicate"

    for entry in (first_entry, second_entry):
        shaped = entry["node"]["context"]
        assert set(shaped) == set(stored_context)  # every context key survives
        assert shaped["procedure"] == {"count": len(steps), "chars": procedure_chars}
        assert shaped["trigger"] == "voltaic pergola rite"
        assert shaped["task_pattern"] == "voltaic-pergola-rite"

    # With context.procedure compacted, the full delivery costs its content
    # plus a bounded envelope — not content shipped twice — and the stub
    # response stays small outright.
    assert _wire_bytes(first) < len(json.dumps(body)) + 2048
    assert _wire_bytes(second) < 4096


def test_exact_ref_growth_keeps_knowledge_and_validity_but_not_history(tmp_path: Path) -> None:
    scope = "project:exact-ref-diet"
    body = "Procedure: copper tide ledger\n1. Keep the exact binding instruction.\n"
    procedure = ["step α " * 800, "step β " * 800]
    prior = ["earlier recall " * 800]
    correction = {"by": "01CURRENTFACT", "supersedes": "01OLDFACT", "text": "Use current rule"}
    context = {"scope": scope, "procedure": procedure, "neighbor": "leave me out"}
    provenance = {"prior_recalls": prior, "corrections": [correction]}

    async def scenario() -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        mcp, store = _server(tmp_path)
        node = store.create_node(level="schema", content=body, context=context, provenance=provenance)
        store.soft_delete_node(node.id, "superseded")
        async with Client(mcp) as client:
            default = _structured(await client.call_tool("memory_lookup", {"node_id": node.id}))
            store.update_node(
                node.id,
                context={**context, "procedure": procedure * 6},
                provenance={**provenance, "prior_recalls": prior * 8},
            )
            grown = _structured(await client.call_tool("memory_lookup", {"node_id": node.id}))
            selected = _structured(await client.call_tool("memory_lookup", {
                "node_id": node.id, "context_key": "procedure",
            }))
            full = _structured(await client.call_tool("memory_lookup", {
                "node_id": node.id, "view": "full",
            }))
            missing = _structured(await client.call_tool("memory_lookup", {
                "node_ids": ["01MISSINGEXACTREF", node.id], "scope": "project:wrong",
            }))
            invalid = _structured(await client.call_tool("memory_lookup", {
                "node_id": node.id, "view": "brief",
            }))
            combined = _structured(await client.call_tool("memory_lookup", {
                "node_id": node.id, "view": "full", "context_key": "procedure",
            }))
        return store, default, grown, selected, full, missing, invalid, combined

    store, default, grown, selected, full, missing, invalid, combined = asyncio.run(scenario())
    node = store.get_node(default["results"][0]["id"])
    assert node is not None
    assert default == grown
    assert _wire_bytes(default) == _wire_bytes(grown)
    assert _wire_bytes(grown) < _wire_bytes(full)
    knowledge = grown["results"][0]
    assert knowledge["content"] == body and knowledge["level"] == "schema"
    assert knowledge["scope"] == scope
    assert knowledge["decayed"] is True and knowledge["decay_reason"] == "superseded"
    assert knowledge["provenance"] == {"corrections": [correction]}
    assert "context" not in knowledge and "prior_recalls" not in knowledge["provenance"]
    assert selected["results"][0]["context"] == {"procedure": procedure * 6}
    assert selected["results"][0]["content"] == body
    assert "neighbor" not in selected["results"][0]["context"]
    assert full["results"][0] == {**node_to_dict(node), "superseded": False, "superseded_by": []}
    assert missing["missing"] == ["01MISSINGEXACTREF"]
    assert missing["scope_mismatches"] == [{"id": node.id, "scope": scope}]
    assert missing["results"][0] == knowledge
    assert "error" in invalid and invalid["results"] == []
    assert "error" in combined and combined["results"] == []


# --- Remember diet: bounded confirmation, durable and recallable write ------


def test_remember_response_bounded_while_write_durable_and_recallable(
    tmp_path: Path,
) -> None:
    scope = "project:dietwrite"
    body = "quenched manifold registry decision record\n" + "\n".join(
        f"registry stanza {index}: the manifold decision holds because of a "
        f"deliberately verbose justification paragraph repeated for bulk"
        for index in range(80)
    )

    async def scenario() -> tuple[Any, dict[str, Any], dict[str, Any]]:
        mcp, store = _server(tmp_path)
        async with Client(mcp) as client:
            remembered = _structured(
                await client.call_tool(
                    "memory_remember",
                    {"content": body, "context": {"scope": scope}},
                )
            )
            recalled = await _recall(client, "quenched manifold registry", scope=scope)
        return store, remembered, recalled

    store, remembered, recalled = asyncio.run(scenario())
    assert len(body) > 4096  # the stored payload alone dwarfs the response bound

    assert _wire_bytes(remembered) <= 4096
    assert set(remembered) == {"node", "implicit_feedback", "auto_consolidation", "auto_decay"}
    confirmation = remembered["node"]
    assert confirmation["level"] == "trace"
    assert confirmation["scope"] == scope
    for verbose_key in ("content", "context", "stats", "provenance"):
        assert verbose_key not in confirmation

    stored = store.get_node(confirmation["id"])
    assert stored is not None
    assert stored.content == body  # durably stored, byte-equal

    entry = _by_id(recalled)[confirmation["id"]]  # and recallable over the wire
    assert entry["delivery"] == "snippet"
    assert len(entry["content"]) <= DEFAULT_SNIPPET_LADDER[0]
    assert entry["content_ref"]["node_id"] == confirmation["id"]


# --- Feedback closure: stub deliveries never break pending-recall matching --

CLOSURE_QUERY = "boreal auroral quasar lattice drift"
CLOSURE_SEED = "boreal auroral quasar lattice drift beacon calibration ledger"
CLOSURE_REMEMBER = "carbide flywheel gearset manifold spline"


def test_closure_links_recall_to_remember_despite_stub_delivery(
    tmp_path: Path,
) -> None:
    # Dissimilar content must still close through transport identity.
    assert not set(tokenize(CLOSURE_REMEMBER)) & set(tokenize(CLOSURE_QUERY))
    scope = "project:dietclosure"

    async def scenario() -> tuple[Any, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
        mcp, store = _server(tmp_path)
        seeded = store.append_trace(CLOSURE_SEED, {"scope": scope})
        async with Client(mcp) as client:
            first = await _recall(client, CLOSURE_QUERY, scope=scope)
            second = await _recall(client, CLOSURE_QUERY, scope=scope)
            assert all(not event.feedback_applied for event in store.list_recall_events())
            remembered = _structured(
                await client.call_tool(
                    "memory_remember",
                    {"content": CLOSURE_REMEMBER, "context": {"scope": scope}},
                )
            )
        return store, seeded.id, first, second, remembered

    store, seeded_id, first, second, remembered = asyncio.run(scenario())

    # Precondition of the invariant: the second response really was a stub.
    assert _by_id(first)[seeded_id]["delivery"] == "full"
    assert _by_id(second)[seeded_id]["delivery"] == "session_duplicate"

    trace_id = remembered["node"]["id"]
    events = {event.id: event for event in store.list_recall_events()}
    assert set(events) == {first["recall_event_id"], second["recall_event_id"]}
    for event in events.values():
        assert event.feedback_applied  # 0 -> 1 on both events
        assert event.feedback_trace_id == trace_id
        assert event.result_ids == [seeded_id]  # the recorded event stays unshaped

    stored = store.get_node(trace_id)
    prior = stored.provenance["prior_recalls"]
    assert {entry["id"] for entry in prior} == set(events)
    assert remembered["node"]["prior_recall_count"] == 2
    assert set(remembered["implicit_feedback"]["recall_event_ids"]) == set(events)
    # Linkage is exhaustive and reinforcement is not: the closure content
    # shares zero tokens with the seeded node (asserted above), so it grounds
    # nothing and earns no credit, while the provenance link still lands.
    # Both halves matter — a regression that re-broadened credit to every
    # delivered result would flip `feedback_applied` back to True here.
    assert stored.provenance["recalled_nodes"] == [seeded_id]
    assert remembered["implicit_feedback"]["feedback_applied"] is False


# --- Digest consolidation: concepts stop being verbatim trace copies --------


def test_consolidate_over_wire_returns_digest_distinct_from_every_source(
    tmp_path: Path,
) -> None:
    scope = "project:dietdigest"

    async def scenario() -> tuple[list[Any], dict[str, Any]]:
        mcp, store = _server(tmp_path)
        sources = [
            store.append_trace(
                f"deploy rollback requires migration guard before release {index}",
                {
                    "scope": scope,
                    "agent": "agent-a" if index % 2 == 0 else "agent-b",
                },
                feedback={"confidence": 0.4, "usefulness_score": 0.2},
            )
            for index in range(100)
        ]
        async with Client(mcp) as client:
            response = _structured(
                await client.call_tool(
                    "memory_consolidate", {"scope": scope, "force": True}
                )
            )
        return sources, response

    sources, response = asyncio.run(scenario())
    source_contents = {source.content for source in sources}
    assert len(source_contents) == 100  # a genuinely multi-distinct cluster

    assert len(response["concepts_created"]) == 1
    concept = response["concepts_created"][0]
    assert concept["level"] == "concept"
    assert concept["scope"] == scope
    # memory_consolidate's direct report keeps full node dicts...
    assert set(concept["provenance"]["source_traces"]) == {s.id for s in sources}
    # ...and the concept content is a digest, byte-distinct from every source.
    assert concept["content"]
    assert concept["content"] not in source_contents
