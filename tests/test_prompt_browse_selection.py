"""Empty-task prompts rank the bounded prefix and hydrate only selected sources."""

import asyncio
import json
import sqlite3
import tracemalloc

from fastmcp import Client
from mcp.shared.exceptions import McpError
import pytest

from living_memory import prompts, server, storage
from living_memory.scope import ScopePlan, resolve_scope
from living_memory.storage import MemoryStore


PROMPT = "memory://prompt/retrieval_context"
SCOPE = "project:browse"


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")


def concept(store, identity, *, scope=SCOPE, timestamp="2026-01-01", **stats):
    return store.create_node(
        node_id=identity, level="concept", content=f"Full source {identity}\n  details.",
        context={"scope": scope, "nested": {"values": [1, True, "source"]}},
        timestamp=timestamp, stats={"unique_agents": 2, **stats},
        provenance={"detail": {"source": identity}, "source_traces": ["trace-source"],
                    "corrections": [{"content": "stored correction"}]},
    )


def full_node_oracle(store, plan, *, max_concepts=5, min_confidence=0.0,
                     retrieval_policy="balanced"):
    """Independent full-source selection: no scalar reader or production scorer."""
    limit = max(0, int(max_concepts))
    threshold = max(0.0, min(1.0, float(min_confidence)))
    selected = {}
    for scope in plan.named_scopes if limit else ():
        weights = store.get_retrieval_weights(scope)
        total = weights.bm25 + weights.vector + weights.graph
        bm25, vector, graph = ((weights.bm25 / total, weights.vector / total, weights.graph / total)
                               if total > 0 else (1.0, 0.0, 0.0))
        for node in store.list_nodes(level="concept", scope=scope, limit=1000):
            if node.confidence < threshold:
                continue
            weighted = bm25 * 0.7 + vector * 0.625 + graph * min(1.0, 0.1 * node.access_count)
            boost = 1.0 + (len(plan.named_scopes) - plan.named_scopes.index(scope) - 1) * 0.2
            confidence = 0.5 + node.confidence
            policy_name = str(retrieval_policy or "balanced").lower()
            if policy_name in ("confidence", "trusted"):
                confidence += 0.35 * node.confidence
            elif policy_name in ("project", "local") and scope == plan.requested_scope:
                boost += 0.2
            elif policy_name in ("recent", "fresh"):
                weighted += 0.02 * min(10, node.access_count)
            score = weighted * boost * confidence * (1.0 + max(-0.5, min(0.5, node.usefulness_score)))
            selected[node.id] = prompts.ContextConcept(node, score, ("policy",))
    return sorted(selected.values(), key=lambda entry: (
        entry.score, entry.node.confidence, entry.node.usefulness_score,
        entry.node.access_count, entry.node.timestamp,
    ), reverse=True)[:limit]


@pytest.mark.parametrize("policy_name", ["balanced", "confidence", "trusted", "project", "local",
                                         "recent", "fresh", "CONFIDENCE", "unknown", ""])
@pytest.mark.parametrize("scope,ambient", [
    ("global", {}), (SCOPE, {}), ("session:browse", {"project": "browse"}),
    (None, {"project": "browse", "session_id": "browse"}),
])
def test_identity_score_full_nodes_and_output_match_oracle(tmp_path, scope, ambient, policy_name):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for index, item_scope in enumerate(("global", SCOPE, "session:browse", "project:foreign")):
            for number, (confidence, usefulness, accesses) in enumerate([
                (0.9, 0.6, 20), (0.9, 0.6, 20), (0.7, -0.7, 3),
                (0.2, 0.3, -1), (1.0, 0.0, 0), (0.6, 0.5, 10),
            ]):
                concept(store, f"{index}-{number}", scope=item_scope, confidence=confidence,
                        usefulness_score=usefulness, access_count=accesses)
            expired = concept(store, f"{index}-expired", scope=item_scope, confidence=1.0)
            store.soft_delete_node(expired.id, "expired")
            store.set_retrieval_weights(item_scope, bm25=index, vector=3-index, graph=index+1)
        # Exercise the nonpositive-total weight fallback as well as per-scope weights.
        store.connection.execute(
            "UPDATE retrieval_weights SET bm25=0, vector=0, graph=0 WHERE scope='session:browse'"
        )
        for name, correction_scope, decayed in [
            ("visible", SCOPE, False), ("expired", "global", True),
            ("foreign", "project:foreign", False),
        ]:
            correction = store.create_node(
                node_id=name, level="trace", content=f"Correction {name}",
                context={"scope": correction_scope},
            )
            store.connect_nodes(correction.id, "1-1", "supersedes")
            if decayed:
                store.soft_delete_node(correction.id, "expired correction")
        plan = resolve_scope(scope=scope, ambient_context=ambient, store=store)
        before = list(store.connection.iterdump())
        for threshold in (-1.0, 0.6, 1.5):
            for limit in (-2, 1, 5, 10000):
                arguments = dict(max_concepts=limit, min_confidence=threshold,
                                 retrieval_policy=policy_name)
                expected = full_node_oracle(store, plan, **arguments)
                actual = prompts.select_context_concepts(store, task=" \n", plan=plan, **arguments)
                assert actual == expected
                assert all(entry.node.scope in plan.named_scopes for entry in actual)
                expected_text = prompts.format_retrieval_context(
                    expected, task="", plan=plan, store=store, min_confidence=threshold,
                    retrieval_policy=policy_name,
                )
                assert prompts.retrieval_context_prompt(
                    store, task=" \n", scope=scope, ambient_context=ambient, **arguments,
                ) == expected_text
        assert list(store.connection.iterdump()) == before


def test_prefix_precedes_confidence_filter_and_preserves_stable_ties(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        with store.transaction():
            for index in range(1001):
                concept(store, f"{index:04}", confidence=0.9 if index in (0, 999, 1000) else 0.1)
            concept(store, "newer-low", timestamp="2026-01-02", confidence=0.1)
            expired = concept(store, "newest-expired", timestamp="2026-01-03", confidence=1.0)
            store.soft_delete_node(expired.id)
        plan = resolve_scope(scope=SCOPE)
        actual = prompts.select_context_concepts(store, task="", plan=plan, min_confidence=0.5,
                                                  max_concepts=10000)
        assert [entry.node.id for entry in actual] == ["1000", "0999"]
        assert actual == full_node_oracle(store, plan, min_confidence=0.5, max_concepts=10000)
        # Equal scores across scopes must retain scope iteration order, not global id order.
        concept(store, "z-global", scope="global", confidence=0.9)
        zero = ScopePlan(SCOPE, (SCOPE, "global", "*"))
        for item_scope in zero.named_scopes:
            store.set_retrieval_weights(item_scope, bm25=0, vector=0, graph=1)
        actual = prompts.select_context_concepts(store, task="", plan=zero, min_confidence=0.5)
        assert [entry.node.id for entry in actual] == ["1000", "0999", "z-global"]


@pytest.fixture
def native(tmp_path, monkeypatch):
    guards = []
    register = server._register_prompts

    def capture(mcp, store, guard):
        guards.append(guard)
        register(mcp, store, guard)

    monkeypatch.setattr(server, "_register_prompts", capture)
    mcp = server.create_mcp_server(tmp_path / "memory.sqlite3")
    try:
        yield mcp, guards[0]
    finally:
        mcp.auto_consolidation.close()
        mcp.memory_store.close()


def test_native_hydrates_only_winners_independent_of_discarded_payload(native, monkeypatch):
    mcp, guard = native
    store = mcp.memory_store
    with store.transaction():
        for index in range(1000):
            concept(store, f"{index:04}", confidence=0.9 if index >= 995 else 0.1)
    expected = prompts.format_retrieval_context(
        full_node_oracle(store, resolve_scope(scope=SCOPE)), task="", plan=resolve_scope(scope=SCOPE),
        store=store, min_confidence=0, retrieval_policy="balanced",
    )
    decode = storage._node_from_row
    decoded = []

    def counted(row):
        assert guard._is_owned()
        assert store.connection.in_transaction
        decoded.append((row["id"], len(row["content"]), len(row["context"]), len(row["provenance"])))
        return decode(row)

    monkeypatch.setattr(storage, "_node_from_row", counted)

    async def scenario():
        async with Client(mcp) as client:
            # Warm native transport before comparing allocation peaks.
            await client.get_prompt(PROMPT, {"scope": SCOPE, "task": "", "max_concepts": "0"})
            observations = []
            for payload_size in (0, 16384, 16384):
                store.connection.execute(
                    "UPDATE nodes SET provenance=? WHERE confidence < 0.5",
                    (json.dumps({"unused": "x" * payload_size}),),
                )
                changes = store.connection.total_changes
                decoded.clear()
                tracemalloc.start()
                try:
                    response = await client.get_prompt(PROMPT, {"scope": SCOPE, "task": ""})
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                assert response.messages[0].content.text == expected
                assert [row[0] for row in decoded] == [f"{index:04}" for index in range(995, 1000)]
                assert store.connection.total_changes == changes
                assert not store.connection.in_transaction
                observations.append({"discarded_payload_bytes": payload_size * 995,
                                     "hydrated": len(decoded), "peak_bytes": peak,
                                     "hydrated_payload_bytes": sum(sum(row[1:]) for row in decoded)})
            assert len({item["hydrated_payload_bytes"] for item in observations}) == 1
            # The unused 16 MB must not reappear as request allocations or retained bodies.
            assert max(item["peak_bytes"] for item in observations) < 995 * 16384 / 4
            print(json.dumps({"native_browse": observations}))

    asyncio.run(scenario())


@pytest.mark.parametrize("field,value,error", [
    ("provenance", "{", json.JSONDecodeError), ("provenance", "[]", TypeError),
    ("context", "{", json.JSONDecodeError), ("source_traces", "{", json.JSONDecodeError),
    ("corrections", "{", json.JSONDecodeError),
])
def test_discarded_payload_is_unread_but_selected_payload_errors_propagate(native, field, value, error):
    mcp, _ = native
    store = mcp.memory_store
    concept(store, "winner", confidence=0.9)
    concept(store, "loser", confidence=0.1)
    plan = resolve_scope(scope=SCOPE)
    store.connection.execute(f"UPDATE nodes SET {field}=? WHERE id='loser'", (value,))
    assert [entry.node.id for entry in prompts.select_context_concepts(
        store, task="", plan=plan, max_concepts=1,
    )] == ["winner"]
    # Other list_nodes consumers retain strict decoding of that same prefix.
    with pytest.raises(error):
        store.list_nodes(level="concept", scope=SCOPE)

    async def scenario():
        async with Client(mcp) as client:
            await client.get_prompt(PROMPT, {"scope": SCOPE, "max_concepts": "1"})
            store.connection.execute("UPDATE nodes SET confidence=1 WHERE id='loser'")
            with pytest.raises(error):
                prompts.select_context_concepts(store, task="", plan=plan, max_concepts=1)
            with pytest.raises(McpError):
                await client.get_prompt(PROMPT, {"scope": SCOPE, "max_concepts": "1"})
            assert not store.connection.in_transaction

    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["confidence", "access_count", "usefulness_score"])
def test_required_scalar_errors_propagate_even_below_confidence_threshold(native, field):
    mcp, _ = native
    store = mcp.memory_store
    concept(store, "winner", confidence=0.9)
    concept(store, "loser", confidence=0.1)
    store.connection.execute(f"UPDATE nodes SET {field}='invalid' WHERE id='loser'")
    with pytest.raises(ValueError):
        prompts.select_context_concepts(store, task="", plan=resolve_scope(scope=SCOPE),
                                        max_concepts=1, min_confidence=0.5)

    async def scenario():
        async with Client(mcp) as client:
            with pytest.raises(McpError):
                await client.get_prompt(PROMPT, {"scope": SCOPE, "min_confidence": "0.5"})
            response = await client.get_prompt(PROMPT, {"scope": SCOPE, "max_concepts": "0"})
            assert "concepts:\n- none" in response.messages[0].content.text

    asyncio.run(scenario())


def test_selection_and_hydration_share_snapshot_and_next_call_is_fresh(tmp_path, monkeypatch):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store, sqlite3.connect(path, isolation_level=None) as writer:
        original = concept(store, "winner", confidence=0.9)
        runner_up = concept(store, "runner-up", confidence=0.5)
        read = store.get_nodes

        def mutate_then_read(ids):
            assert store.connection.in_transaction
            writer.execute(
                "UPDATE nodes SET content='changed', scope='project:foreign', decayed=1 WHERE id='winner'"
            )
            return read(ids)

        plan = resolve_scope(scope=SCOPE)
        monkeypatch.setattr(store, "get_nodes", mutate_then_read)
        selected = prompts.select_context_concepts(store, task="", plan=plan, max_concepts=1)
        assert selected[0].node == original
        monkeypatch.setattr(store, "get_nodes", read)
        assert prompts.select_context_concepts(store, task="", plan=plan, max_concepts=1)[0].node == runner_up
        assert not store.connection.in_transaction


def test_missing_selected_source_is_not_silently_dropped(tmp_path, monkeypatch):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        concept(store, "required")
        monkeypatch.setattr(store, "get_nodes", lambda ids: {})
        with pytest.raises(KeyError, match="required"):
            prompts.select_context_concepts(store, task="", plan=resolve_scope(scope=SCOPE))
        assert not store.connection.in_transaction
