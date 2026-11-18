"""Prompt selectors share immutable query input while reading current memory."""

import asyncio
import sqlite3
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

from fastmcp import Client
from mcp.shared.exceptions import McpError
import pytest

from living_memory import embeddings, prompts, retrieval, server
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

    def register_prompts(mcp, store, guard, recall_service):
        state.guard = guard
        register(mcp, store, guard, recall_service)

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


def standalone_context(store, *, task=TASK, scope=SCOPE, max_concepts=5, max_schemas=3,
                       min_confidence=0.0, retrieval_policy="balanced", ambient_context=None):
    """Compose the independent string selector APIs as a comparison consumer."""
    task = task.strip()
    plan = resolve_scope(scope=scope, ambient_context=ambient_context, store=store)
    arguments = dict(task=task, plan=plan, ambient_context=ambient_context)
    concepts = prompts.select_context_concepts(
        store, **arguments, max_concepts=max_concepts, min_confidence=min_confidence,
        retrieval_policy=retrieval_policy,
    )
    schemas = prompts.select_context_schemas(
        store, **arguments, max_schemas=max_schemas, min_confidence=min_confidence,
    )
    decisions = prompts.select_decision_history(store, **arguments, max_items=max_concepts)
    return prompts.format_retrieval_context(
        concepts, schemas=schemas, decision_history=decisions, task=task, plan=plan,
        store=store, min_confidence=min_confidence, retrieval_policy=retrieval_policy,
    )


async def get_prompt(client, **arguments):
    result = await client.get_prompt(PROMPT, {"task": TASK, "scope": SCOPE, **arguments})
    return result.messages[0].content.text


@pytest.mark.parametrize("gate", ["off", "0.45"])
def test_native_prompt_matches_independent_selectors_and_encodes_once(runtime, monkeypatch, gate):
    r = runtime
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", gate)
    recall = MemoryRecallService.memory_recall

    def recall_with_passages(self, query, **kwargs):
        kwargs["include_passages"] = True
        return recall(self, query, **kwargs)

    with monkeypatch.context() as reference:
        reference.setattr(MemoryRecallService, "memory_recall", recall_with_passages)
        expected = standalone_context(r.store)

    def unavailable(chunk_ids):
        raise sqlite3.OperationalError("prompt read semantic passages")

    monkeypatch.setattr(r.store, "get_chunk_passages", unavailable)
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


def test_native_schema_limit_is_discoverable_and_controls_only_skills(runtime):
    r = runtime
    for index, scope in enumerate([SCOPE, SCOPE, SCOPE, "project:foreign"]):
        node = r.store.create_node(
            level="schema", content=f"Cobalt release rollback procedure {index}.",
            context={"scope": scope, "trigger": TASK}, stats={"confidence": 0.9},
        )
    foreign_schema = node

    async def scenario():
        async with Client(r.mcp) as client:
            prompt = next(p for p in await client.list_prompts() if p.name == PROMPT)
            argument = next(arg for arg in prompt.arguments if arg.name == "max_schemas")
            assert not argument.required
            for limit in (None, 0, 1, 4):
                args = {} if limit is None else {"max_schemas": str(limit)}
                expected_count = 3 if limit is None else limit
                actual = await get_prompt(client, **args)
                assert actual.count(f'trigger="{TASK}"') == expected_count
                assert ("skills:" in actual) == bool(expected_count)
                assert r.concept.id in actual and r.primary.id in actual
                assert foreign_schema.id not in actual
                assert actual == standalone_context(r.store, max_schemas=expected_count)

    asyncio.run(scenario())


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_native_disabled_sections_skip_preparation_and_recall(runtime, monkeypatch, limit):
    r = runtime
    expected = standalone_context(r.store, max_concepts=0, max_schemas=0)

    def unexpected(*args, **kwargs):
        pytest.fail("disabled prompt sections started retrieval work")

    for name in ("prepare_query", "refresh_pending_chunks", "prepare_anchor_history", "memory_recall"):
        monkeypatch.setattr(MemoryRecallService, name, unexpected)

    async def scenario():
        async with Client(r.mcp) as client:
            assert await get_prompt(client, max_concepts=limit, max_schemas=limit) == expected

    asyncio.run(scenario())


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


@pytest.fixture
def pending_content(runtime, monkeypatch):
    r = runtime
    r.pending = r.store.create_node(
        level="schema", content="Cobalt release rollback pending checklist.",
        context={"scope": SCOPE, "trigger": TASK}, stats={"confidence": 0.9},
    )
    assert not r.store.node_chunks_are_current(r.pending.id)
    r.content_encodes = []
    r.encoder_guard_owned = []
    encode = embeddings._embed_hash

    def held_content(text, dimensions):
        if text != TASK:
            r.content_encodes.append(text)
            r.encoder_guard_owned.append(r.guard._is_owned())
        if text == r.pending.content:
            r.entered.set()
            assert r.release.wait(10), "pending content encoder was not released"
        return encode(text, dimensions)

    monkeypatch.setattr(embeddings, "_embed_hash", held_content)
    return r


@pytest.mark.parametrize("max_concepts", [0, 5])
def test_native_pending_content_keeps_exact_lookup_available(pending_content, monkeypatch, max_concepts):
    r = pending_content
    recalls = []
    selection_states = []
    recall = MemoryRecallService.memory_recall

    def guarded_recall(self, *args, **kwargs):
        selection_states.append((r.guard._is_owned(), r.release.is_set(),
                                 kwargs.get("repair_pending", True)))
        recalls.append((kwargs.get("depth", 1), kwargs["max_results"],
                        kwargs["log_access"], kwargs.get("log_event")))
        return recall(self, *args, **kwargs)

    monkeypatch.setattr(MemoryRecallService, "memory_recall", guarded_recall)

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as peer:
            pending = asyncio.create_task(get_prompt(owner, max_concepts=max_concepts))
            try:
                assert await asyncio.to_thread(r.entered.wait, 5)
                assert not r.store.connection.in_transaction
                lookup = await asyncio.wait_for(peer.call_tool("memory_lookup", {
                    "node_id": r.pending.id,
                }), 3)
                assert lookup.data["results"][0]["content"] == r.pending.content
                assert not pending.done() and not r.release.is_set()
            finally:
                r.release.set()
            actual = await asyncio.wait_for(pending, 5)
            assert r.pending.content in actual
            assert r.store.node_chunks_are_current(r.pending.id)
            return actual

    actual = asyncio.run(scenario())
    assert recalls == ([(1, 30, False, None)] if max_concepts else []) + [
        (1, 12, False, False),
    ] + ([("decision", 30, False, None)] if max_concepts else [])
    assert selection_states == [(True, True, False)] * len(recalls)
    assert r.encoded == [TASK]
    assert r.content_encodes == [r.pending.content]
    assert r.encoder_guard_owned == [False]
    monkeypatch.setattr(MemoryRecallService, "memory_recall", recall)
    assert actual == standalone_context(r.store, max_concepts=max_concepts)


@pytest.mark.parametrize("change", ["content", "scope", "decay", "correction", "default_scope"])
def test_native_pending_gap_reads_current_sources_without_fallback(pending_content, change):
    r = pending_content
    updated = "Cobalt release rollback requires a current countersignature."
    new_scope = "project:current-prompt"
    scope = None if change == "default_scope" else SCOPE

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as peer:
            pending = asyncio.create_task(get_prompt(owner, scope=scope))
            try:
                assert await asyncio.to_thread(r.entered.wait, 5)
                lookup = await asyncio.wait_for(peer.call_tool("memory_lookup", {
                    "node_id": r.pending.id,
                }), 3)
                assert lookup.data["results"][0]["content"] == r.pending.content
                with r.guard:
                    if change == "content":
                        r.store.update_node(r.pending.id, content=updated)
                    elif change in {"scope", "default_scope"}:
                        r.store.update_node(r.pending.id, context={"scope": new_scope, "trigger": TASK})
                        if change == "default_scope":
                            r.store.config = replace(r.store.config, default_scope=new_scope)
                    elif change == "decay":
                        r.store.soft_delete_node(r.pending.id, "expired during preparation")
                    else:
                        r.store.connect_nodes(r.correction.id, r.pending.id, "supersedes")
                    late = r.store.create_node(
                        level="schema", content="Cobalt release rollback late checklist.",
                        context={"scope": scope or new_scope, "trigger": TASK},
                        stats={"confidence": 0.9},
                    )
                assert not pending.done()
            finally:
                r.release.set()
            actual = await asyncio.wait_for(pending, 5)
            assert late.content in actual
            assert not r.store.node_chunks_are_current(late.id)
            if change in {"scope", "decay"}:
                assert r.pending.id not in actual
            elif change == "content":
                assert updated in actual and r.pending.content not in actual
            elif change == "default_scope":
                assert f"scope_plan: {new_scope} > global > *" in actual
                assert f"id={r.pending.id} scope={new_scope}" in actual
            else:
                line = next(line for line in actual.splitlines() if f"id={r.pending.id} " in line)
                assert f'superseded=true superseded_by=["{r.correction.id}"]' in line
            assert r.store.node_chunks_are_current(r.pending.id) == (change == "correction")
            assert r.content_encodes == [r.pending.content]
            assert r.encoder_guard_owned == [False]
            assert r.encoded == [TASK]

    asyncio.run(scenario())


def test_empty_native_prompt_does_not_prepare_pending_content(pending_content):
    r = pending_content

    async def scenario():
        async with Client(r.mcp) as client:
            assert r.concept.content in await get_prompt(client, task=" \n\t")

    asyncio.run(scenario())
    assert not r.entered.is_set() and not r.content_encodes and not r.encoded
    assert not r.store.node_chunks_are_current(r.pending.id)


@pytest.mark.parametrize("consumer", [
    prompts.retrieval_context_prompt, prompts.select_context_concepts,
    prompts.select_context_schemas, prompts.select_decision_history,
])
def test_standalone_prompt_consumers_still_repair_pending_without_runtime_guard(pending_content, consumer):
    r = pending_content
    r.release.set()
    arguments = ({"scope": SCOPE} if consumer is prompts.retrieval_context_prompt else
                 {"plan": resolve_scope(scope=SCOPE, store=r.store)})
    consumer(r.store, task=TASK, **arguments)
    assert r.store.node_chunks_are_current(r.pending.id)
    assert r.content_encodes == [r.pending.content]
    assert r.encoder_guard_owned == [False]
    assert r.store.list_recall_events() == []


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


@pytest.mark.parametrize("native", [False, True], ids=["builder", "native"])
def test_prompt_shares_chunk_and_anchor_preparation(runtime, monkeypatch, native):
    r = runtime
    vector = MemoryRecallService(r.store).embedder.embed(TASK)
    r.store.insert_query_anchor(
        anchor_id="rollback-anchor", scope=SCOPE, query=TASK,
        fingerprint="rollback-anchor", embedding=vector,
    )
    r.store.upsert_query_anchor_edge("rollback-anchor", r.concept.id, weight=1.0)
    counts = dict(builds=0, chunk_reads=0, rows=0, vector_bytes=0, anchor_builds=0)
    build = MemoryRecallService._build_chunk_index
    rows = r.store.iter_chunk_embedding_rows
    anchors = retrieval._prepare_anchor_vectors

    def counted_build(*args, **kwargs):
        counts["builds"] += 1
        return build(*args, **kwargs)

    def counted_rows(*args, **kwargs):
        counts["chunk_reads"] += 1
        for row in rows(*args, **kwargs):
            counts["rows"] += 1
            counts["vector_bytes"] += len(row[2])
            yield row

    def counted_anchors(*args, **kwargs):
        counts["anchor_builds"] += 1
        return anchors(*args, **kwargs)

    monkeypatch.setattr(MemoryRecallService, "_build_chunk_index", counted_build)
    monkeypatch.setattr(r.store, "iter_chunk_embedding_rows", counted_rows)
    monkeypatch.setattr(retrieval, "_prepare_anchor_vectors", counted_anchors)
    before = list(r.store.connection.iterdump())
    expected = standalone_context(r.store)
    independent_counts = counts.copy()
    counts.update(dict.fromkeys(counts, 0))

    async def scenario():
        async with Client(r.mcp) as client:
            return await get_prompt(client)

    actual = (asyncio.run(scenario()) if native else
              prompts.retrieval_context_prompt(r.store, task=TASK, scope=SCOPE))
    assert actual == expected
    assert counts["builds"] == counts["chunk_reads"] == 2
    assert counts["anchor_builds"] > 0 and counts["rows"] > 0
    assert independent_counts == {key: 3 * count for key, count in counts.items()}
    assert list(r.store.connection.iterdump()) == before


@pytest.mark.parametrize("concepts,schemas,fts_limits", [
    (5, 3, [240, 96]), (2, 5, [160]), (1, 12, [160, 384]),
    (0, 3, [96]), (5, 0, [240]), (0, 0, []),
])
def test_native_prompt_discovery_passes(runtime, monkeypatch, concepts, schemas, fts_limits):
    r = runtime
    expected = standalone_context(r.store, max_concepts=concepts, max_schemas=schemas).encode()
    blocks, searches = [], []
    pool = retrieval._pooled_chunk_similarities
    search = r.store.search_content_ids

    def count_pool(block, *args):
        blocks.append(block.node_ids)
        return pool(block, *args)

    def count_search(query, **kwargs):
        searches.append((kwargs["scope"], kwargs["limit"]))
        return search(query, **kwargs)

    monkeypatch.setattr(retrieval, "_pooled_chunk_similarities", count_pool)
    monkeypatch.setattr(r.store, "search_content_ids", count_search)

    async def scenario():
        async with Client(r.mcp) as client:
            for _ in range(2):
                actual = await get_prompt(client, max_concepts=concepts, max_schemas=schemas)
                assert actual.encode() == expected
                assert [len(ids) for ids in blocks] == ([6] if fts_limits else [])
                assert searches == [(scope, limit) for limit in fts_limits for scope in (SCOPE, "global")]
                blocks.clear()
                searches.clear()

    asyncio.run(scenario())


def test_shared_service_keeps_scope_query_and_selector_parameters_live(runtime):
    r = runtime
    service = MemoryRecallService(r.store)
    for arguments in (
        dict(scope=SCOPE, max_concepts=1, max_schemas=1),
        dict(scope="project:foreign", max_concepts=7, min_confidence=0.95),
        dict(scope="global", max_schemas=0),
        dict(scope=None, ambient_context={"project": "foreign"}, retrieval_policy="project"),
        dict(scope=None, task="migration snapshot", min_confidence=0.4),
        dict(scope=SCOPE, max_concepts=0, max_schemas=0),
        dict(scope=SCOPE, task="", retrieval_policy="recent"),
        dict(scope=SCOPE),
    ):
        options = dict(task=TASK, **{k: v for k, v in arguments.items() if k != "task"})
        options["task"] = arguments.get("task", TASK)
        assert prompts.retrieval_context_prompt(
            r.store, recall_service=service, **options,
        ) == standalone_context(r.store, **options)


@pytest.mark.parametrize("change", ["content", "scope", "decay", "correction", "default_scope"])
def test_native_warm_prompt_rechecks_changed_corpus(runtime, change):
    r = runtime
    updated = "Cobalt release rollback uses a new countersignature."

    async def scenario():
        async with Client(r.mcp) as client:
            before = await get_prompt(client)
            assert r.concept.content in before
            with MemoryStore(r.store.db_path) as writer:
                if change == "content":
                    writer.update_node(r.concept.id, content=updated)
                elif change == "scope":
                    writer.update_node(r.concept.id, context={"scope": "project:foreign"})
                elif change == "decay":
                    writer.soft_delete_node(r.concept.id, "expired")
                elif change == "correction":
                    writer.connect_nodes(r.correction.id, r.concept.id, "supersedes")
            scope = SCOPE
            if change == "default_scope":
                r.store.config = replace(r.store.config, default_scope="project:foreign")
                scope = None
            actual = await get_prompt(client, scope=scope)
            assert actual != before
            assert actual == standalone_context(r.store, scope=scope)
            if change in {"scope", "decay"}:
                assert r.concept.id not in actual
            elif change == "content":
                assert updated in actual and r.concept.content not in actual
            elif change == "correction":
                line = next(line for line in actual.splitlines() if f"id={r.concept.id} " in line)
                assert f'superseded=true superseded_by=["{r.correction.id}"]' in line
            else:
                assert "scope_plan: project:foreign > global > *" in actual

    asyncio.run(scenario())


@pytest.mark.parametrize("native", [False, True], ids=["builder", "get_prompt"])
@pytest.mark.parametrize("task", ["", TASK], ids=["browse", "recall"])
def test_whole_prompt_shares_wal_snapshot(runtime, monkeypatch, native, task):
    r = runtime
    expected = standalone_context(r.store, task=task)
    new_content = "Cobalt release rollback requires two migration checks."
    with MemoryStore(r.store.db_path) as writer:
        def publish_correction():
            with writer.transaction():
                new = writer.create_node(level="concept", content=new_content, context={"scope": SCOPE})
                writer.connect_nodes(new.id, r.concept.id, "supersedes")
                writer.update_node(r.schema.id, content="Use the revised migration checklist.")
            return new

        boundary = "select_context_schemas" if task else "read_validity"
        read = getattr(prompts, boundary)
        corrections = []

        def commit_then_read(*args, **kwargs):
            if not corrections:
                corrections.append(publish_correction())
            return read(*args, **kwargs)

        monkeypatch.setattr(prompts, boundary, commit_then_read)

        async def fetch_twice():
            async with Client(r.mcp) as client:
                return await get_prompt(client, task=task), await get_prompt(client, task=task)

        if native:
            first, second = asyncio.run(fetch_twice())
        else:
            first, second = [prompts.retrieval_context_prompt(r.store, task=task, scope=SCOPE) for _ in range(2)]

    assert first == expected
    new, = corrections
    assert new_content in second
    old_line = next(line for line in second.splitlines() if f"id={r.concept.id} " in line)
    assert f'superseded=true superseded_by=["{new.id}"]' in old_line
    if task:
        assert "Use the revised migration checklist." in second
    assert not r.store.connection.in_transaction
