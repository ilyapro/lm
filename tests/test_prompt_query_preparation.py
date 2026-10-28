"""Prompt selectors share immutable query input while reading current memory."""

import asyncio
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

from fastmcp import Client
from mcp.shared.exceptions import McpError
import pytest

from living_memory import embeddings, prompts, server
from living_memory.retrieval import MemoryRecallService
from living_memory.scope import resolve_scope
from living_memory.storage import MemoryStore


TASK = "cobalt release rollback"
SCOPE = "project:prompt-query"
PROMPT = "memory://prompt/retrieval_context"


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_AUTO_CONSOLIDATE_POLICY": "off",
        "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_RECALL_MIN_SCORE": "off",
        "LM_RECALL_SCHEMA_TRIGGER": "",
    }.items():
        monkeypatch.setenv(key, value)
    state = SimpleNamespace(entered=Event(), release=Event(), hold=False, fail=False, encoded=[])
    register = server._register_prompts

    def register_prompts(mcp, store, guard):
        state.guard = guard
        register(mcp, store, guard)

    monkeypatch.setattr(server, "_register_prompts", register_prompts)
    state.mcp = server.create_mcp_server(tmp_path / "memory.sqlite3")
    store = state.store = state.mcp.memory_store

    def node(content, *, level="concept", scope=SCOPE, **context):
        return store.create_node(
            level=level, content=content, context={"scope": scope, **context},
            stats={"confidence": 0.9, "unique_agents": 2},
        )

    state.concept = node("Cobalt release rollback requires a migration dry run.")
    state.old = node("Cobalt release rollback may skip migration validation.")
    state.correction = node("Cobalt release rollback must validate migrations.", level="trace")
    store.connect_nodes(state.correction.id, state.old.id, "supersedes")
    store.soft_delete_node(state.correction.id, "expired correction")
    state.decayed = node("Cobalt release rollback decayed concept sentinel.")
    store.soft_delete_node(state.decayed.id, "expired concept")
    state.foreign = node("Cobalt release rollback foreign sentinel.", scope="project:foreign")
    state.schema = node(
        "Check the migration, perform the cobalt rollback, then verify release health.",
        level="schema", trigger=TASK, procedure_id="cobalt-rollback",
    )
    state.untriggered = node("Cobalt release rollback without a trigger.", level="schema")
    state.primary, _ = store.append_trace_with_rejected_alternatives(
        "Cobalt release rollback uses the verified migration snapshot.",
        {"scope": SCOPE},
        alternatives_considered=[{
            "approach": "Rebuild the cobalt release rollback state by hand",
            "rejected_because": "manual reconstruction loses migration ordering",
        }],
    )
    MemoryRecallService(store).refresh_pending_chunks()
    encode = embeddings._embed_hash

    def counted_encode(text, dimensions):
        if text == TASK:
            state.encoded.append(text)
            if state.hold:
                state.entered.set()
                assert state.release.wait(10), "prompt encoder was not released"
            if state.fail:
                raise RuntimeError("prompt query encoding failed")
        return encode(text, dimensions)

    monkeypatch.setattr(embeddings, "_embed_hash", counted_encode)
    try:
        yield state
    finally:
        state.release.set()
        state.mcp.auto_consolidation.close()
        store.close()


def standalone_context(store, *, task=TASK, scope=SCOPE, max_concepts=5, max_schemas=3):
    """Compose the independent string selector APIs as a comparison consumer."""
    task = task.strip()
    plan = resolve_scope(scope=scope, store=store)
    concepts = prompts.select_context_concepts(store, task=task, plan=plan, max_concepts=max_concepts)
    schemas = prompts.select_context_schemas(store, task=task, plan=plan, max_schemas=max_schemas)
    decisions = prompts.select_decision_history(store, task=task, plan=plan, max_items=max_concepts)
    return prompts.format_retrieval_context(
        concepts, schemas=schemas, decision_history=decisions, task=task, plan=plan,
        store=store, min_confidence=0.0, retrieval_policy="balanced",
    )


async def get_prompt(client, **arguments):
    result = await client.get_prompt(PROMPT, {"task": TASK, "scope": SCOPE, **arguments})
    return result.messages[0].content.text


@pytest.mark.parametrize("gate", ["off", "0.45"])
def test_native_prompt_matches_independent_selectors_and_encodes_once(runtime, monkeypatch, gate):
    r = runtime
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", gate)
    expected = standalone_context(r.store)
    assert r.encoded == [TASK] * 3
    assert r.concept.id in expected and r.schema.id in expected and r.primary.id in expected
    assert 'trigger="cobalt release rollback"' in expected
    assert "Alternatives rejected: 1 (manual reconstruction loses migration ordering)" in expected
    assert f'superseded=true superseded_by=["{r.correction.id}"]' in expected
    assert r.decayed.id not in expected and r.foreign.id not in expected
    assert r.untriggered.id not in expected
    before = list(r.store.connection.iterdump())
    r.encoded.clear()

    async def scenario():
        async with Client(r.mcp) as client:
            actual = await get_prompt(client, task=" \n" + TASK + " \t")
            assert actual == expected

    asyncio.run(scenario())
    assert r.encoded == [TASK]
    assert list(r.store.connection.iterdump()) == before


@pytest.mark.parametrize("limits", [(1, 1), (7, 4), (0, 3), (5, 0), (0, 0), (-1, -1)])
@pytest.mark.parametrize("task", [TASK, " \n\t"])
def test_direct_builder_preserves_selector_limits_and_empty_behavior(runtime, monkeypatch, limits, task):
    r = runtime
    concepts, schemas = limits
    expected = standalone_context(r.store, task=task, max_concepts=concepts, max_schemas=schemas)
    r.encoded.clear()
    recalls = []
    recall = MemoryRecallService.memory_recall

    def observe(self, query, **kwargs):
        recalls.append((kwargs.get("depth", 1), kwargs["max_results"],
                        kwargs["log_access"], kwargs.get("log_event")))
        return recall(self, query, **kwargs)

    monkeypatch.setattr(MemoryRecallService, "memory_recall", observe)
    actual = prompts.retrieval_context_prompt(
        r.store, task=task, scope=SCOPE, max_concepts=concepts, max_schemas=schemas,
    )
    assert actual == expected
    expected_calls = []
    if task.strip():
        if concepts > 0:
            expected_calls.append((1, max(concepts * 6, 20), False, None))
        if schemas > 0:
            expected_calls.append((1, max(schemas * 4, 12), False, False))
        if concepts > 0:
            expected_calls.append(("decision", max(concepts * 6, 20), False, None))
    assert recalls == expected_calls
    assert r.encoded == ([TASK] if expected_calls else [])
    assert r.store.list_recall_events() == []


@pytest.mark.parametrize("task,max_concepts", [("", 5), (" \n\t", 0), (TASK, 0)])
def test_native_empty_task_and_zero_concepts_keep_schema_behavior(runtime, task, max_concepts):
    r = runtime
    expected = standalone_context(r.store, task=task, max_concepts=max_concepts)
    r.encoded.clear()

    async def scenario():
        async with Client(r.mcp) as client:
            assert await get_prompt(client, task=task, max_concepts=max_concepts) == expected

    asyncio.run(scenario())
    assert r.encoded == ([TASK] if task.strip() else [])


def test_native_lookup_finishes_with_encoder_held_and_store_reads_guarded(runtime, monkeypatch):
    r = runtime
    expected = standalone_context(r.store)
    r.encoded.clear()
    r.hold = True
    for name in ("resolve_scope", "read_validity"):
        original = getattr(prompts, name)

        def guarded(*args, _original=original, **kwargs):
            assert r.guard._is_owned()
            assert r.release.is_set()
            return _original(*args, **kwargs)

        monkeypatch.setattr(prompts, name, guarded)
    recall = MemoryRecallService.memory_recall

    def guarded_recall(self, *args, **kwargs):
        assert r.guard._is_owned()
        assert r.release.is_set()
        return recall(self, *args, **kwargs)

    monkeypatch.setattr(MemoryRecallService, "memory_recall", guarded_recall)

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as peer:
            pending = asyncio.create_task(get_prompt(owner))
            try:
                assert await asyncio.to_thread(r.entered.wait, 5)
                assert not r.store.connection.in_transaction
                lookup = await asyncio.wait_for(peer.call_tool("memory_lookup", {"node_id": r.concept.id}), 3)
                assert lookup.data["results"][0]["content"] == r.concept.content
                assert not pending.done() and not r.release.is_set()
            finally:
                r.release.set()
            assert await asyncio.wait_for(pending, 5) == expected

    asyncio.run(scenario())
    assert r.encoded == [TASK]


def test_native_prompt_reads_scope_content_and_validity_after_preparation(runtime):
    r = runtime
    r.hold = True
    new_scope = "project:current-prompt"
    updated = "Cobalt release rollback now requires a verified countersignature."

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as peer:
            pending = asyncio.create_task(get_prompt(owner, scope=None))
            try:
                assert await asyncio.to_thread(r.entered.wait, 5)
                await asyncio.wait_for(peer.call_tool("memory_forget", {"id": r.schema.id}), 3)
                await asyncio.wait_for(peer.call_tool("memory_connect", {
                    "id_a": r.correction.id, "id_b": r.concept.id, "relation_type": "supersedes",
                }), 3)
                with MemoryStore(r.store.db_path) as writer:
                    writer.update_node(r.concept.id, content=updated, context={"scope": new_scope})
                with r.guard:
                    r.store.config = replace(r.store.config, default_scope=new_scope)
                assert not pending.done()
            finally:
                r.release.set()
            actual = await asyncio.wait_for(pending, 5)
            assert r.encoded == [TASK]
            assert f"scope_plan: {new_scope} > global > *" in actual
            assert updated in actual and r.concept.content not in actual
            assert r.schema.id not in actual
            concept_line = next(line for line in actual.splitlines() if f"id={r.concept.id} " in line)
            assert f"scope={new_scope}" in concept_line
            assert f'superseded=true superseded_by=["{r.correction.id}"]' in concept_line
            r.hold = False
            assert actual == standalone_context(r.store, scope=None)

    asyncio.run(scenario())


def test_preparation_preserves_recall_state_and_encoding_errors_propagate(runtime):
    r = runtime
    service = MemoryRecallService(r.store)
    service.last_recall_event_id = "previous-event"
    residual, relevance = [object()], object()
    service.last_residual = residual
    service.last_query_relevance = relevance
    prepared = service.prepare_query(" \n" + TASK + " \t")
    assert prepared.text == TASK and isinstance(prepared.embedding, tuple)
    assert service.last_recall_event_id == "previous-event"
    assert service.last_residual is residual and service.last_query_relevance is relevance
    r.fail = True
    before = list(r.store.connection.iterdump())
    with pytest.raises(RuntimeError, match="prompt query encoding failed"):
        prompts.retrieval_context_prompt(r.store, task=TASK, scope=SCOPE)

    async def scenario():
        async with Client(r.mcp) as client:
            with pytest.raises(McpError, match="prompt query encoding failed"):
                await get_prompt(client)
            r.fail = False
            assert r.concept.content in await get_prompt(client)

    asyncio.run(scenario())
    assert list(r.store.connection.iterdump()) == before


@pytest.mark.parametrize("arguments,error", [
    ({"scope": "project:"}, "scope must include an id"),
    ({"min_confidence": "invalid"}, "could not convert string to float"),
    ({"max_concepts": "invalid"}, "invalid literal for int"),
    ({"max_schemas": "invalid"}, "invalid literal for int"),
])
def test_direct_prompt_rejects_invalid_arguments(runtime, arguments, error):
    with pytest.raises(ValueError, match=error):
        prompts.retrieval_context_prompt(runtime.store, task=TASK, **arguments)
