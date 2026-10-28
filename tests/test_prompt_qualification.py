"""Prompt consumers preserve retrieval qualification and source reachability."""

import asyncio
from dataclasses import replace

from fastmcp import Client
import pytest

from living_memory import prompts
from living_memory.retrieval import MemoryRecallService
from living_memory.scope import resolve_scope
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore


TASK = "release database transaction WAL checkpoint deadlock"
UNRELATED = "quasar neutrino spectroscopy orbital telescope"
BALLOON = "Release the helium balloon into clear summer skies."
SCOPE = "project:prompt-qualification"
PROMPT = "memory://prompt/retrieval_context"


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    for name, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_AUTO_CONSOLIDATE_POLICY": "off",
        "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_RECALL_MIN_SCORE": "0.45",
        "LM_RECALL_GATE_FORM": "drop",
        "LM_RECALL_SCHEMA_TRIGGER": "",
    }.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def runtime(tmp_path):
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    try:
        yield mcp
    finally:
        mcp.auto_consolidation.close()
        mcp.memory_store.close()


def concept(store, content=BALLOON, **kwargs):
    return store.create_node(
        level="concept", content=content, context={"scope": SCOPE}, **kwargs,
    )


def assert_reference(text, node):
    line = next(line for line in text.splitlines() if f"id={node.id} " in line)
    assert "delivery=below_threshold" in line
    assert f'content_ref={{"node_id": "{node.id}"}}' in text
    assert "Unqualified source reference" in text
    assert "memory_lookup" in text
    assert node.content not in text


@pytest.mark.parametrize("form", ["drop", "stub"])
def test_native_withheld_reference_is_refetchable_without_becoming_knowledge(runtime, monkeypatch, form):
    monkeypatch.setenv("LM_RECALL_GATE_FORM", form)
    store = runtime.memory_store
    node = concept(store, stats={"confidence": 0.9, "access_count": 50, "usefulness_score": 0.8})
    service = MemoryRecallService(store)
    raw = service.memory_recall(TASK, scope=SCOPE, log_access=False)
    assert len(raw) == 1 and raw[0].withheld == "below_threshold"
    assert raw[0].lexical_score == pytest.approx(1 / 6)

    async def scenario():
        async with Client(runtime) as client:
            recall = (await client.call_tool("memory_recall", {
                "query": TASK, "scope": SCOPE, "view": "knowledge",
            })).data
            entry = next(entry for entry in recall["results"] if entry["id"] == node.id)
            assert entry["delivery"] == "below_threshold" and entry["content"] == ""
            response = await client.get_prompt(PROMPT, {"task": TASK, "scope": SCOPE})
            assert_reference(response.messages[0].content.text, node)
            lookup = (await client.call_tool("memory_lookup", {"node_id": node.id})).data
            assert lookup["results"][0]["content"] == node.content
            repeated = await client.get_prompt(PROMPT, {"task": TASK, "scope": SCOPE})
            assert_reference(repeated.messages[0].content.text, node)
            qualified = await client.get_prompt(PROMPT, {
                "task": "release helium balloon clear summer skies", "scope": SCOPE,
            })
            assert node.content in qualified.messages[0].content.text
            assert "below_threshold" not in qualified.messages[0].content.text

    asyncio.run(scenario())


@pytest.mark.parametrize("policy_name", ["balanced", "confidence", "project", "recent"])
def test_native_no_query_candidates_cannot_fall_back_to_scoped_browse(runtime, policy_name):
    store = runtime.memory_store
    node = concept(store, stats={"confidence": 0.9, "access_count": 50, "usefulness_score": 0.8})
    assert MemoryRecallService(store).memory_recall(UNRELATED, scope=SCOPE, log_access=False) == []

    async def scenario():
        async with Client(runtime) as client:
            for task in (UNRELATED, " \n\t"):
                response = await client.get_prompt(PROMPT, {
                    "task": task, "scope": SCOPE, "retrieval_policy": policy_name,
                })
                text = response.messages[0].content.text
                assert (node.content in text) == (not task.strip())
                assert (node.id in text) == (not task.strip())

    asyncio.run(scenario())


@pytest.mark.parametrize("form", ["drop", "stub"])
def test_all_selectors_keep_qualification_and_do_not_expose_withheld_reasons(tmp_path, monkeypatch, form):
    monkeypatch.setenv("LM_RECALL_GATE_FORM", form)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        weak = concept(store)
        strong = concept(store, TASK + " requires a diagnostic snapshot.")
        schema = store.create_node(
            level="schema", content="Inspect the lock owner before retrying.",
            context={"scope": SCOPE, "trigger": TASK},
        )
        old_schema = store.create_node(
            level="schema", content="Discard the diagnostic journal.",
            context={"scope": SCOPE, "trigger": "release database transaction"},
        )
        correction = store.append_trace("Retain the journal.", {"scope": SCOPE})
        store.connect_nodes(correction.id, old_schema.id, "supersedes")
        store.soft_delete_node(correction.id, "expired correction")
        untriggered = store.create_node(
            level="schema", content=TASK + " untriggered procedure.", context={"scope": SCOPE},
        )
        primary, alternatives = store.append_trace_with_rejected_alternatives(
            TASK + " uses bounded retries.", {"scope": SCOPE},
            alternatives_considered=[
                {"approach": "Unbounded retries", "rejected_because": "Holds locks indefinitely"},
                {"approach": "Foreign approach", "rejected_because": "Foreign reason sentinel"},
                {"approach": "Expired approach", "rejected_because": "Expired reason sentinel"},
            ],
        )
        store.update_node(alternatives[1].id, context={
            **alternatives[1].context, "scope": "project:foreign",
        })
        store.soft_delete_node(alternatives[2].id, "expired alternative")
        weak_primary, _ = store.append_trace_with_rejected_alternatives(
            "Release bright helium balloons.", {"scope": SCOPE},
            alternatives_considered=[{
                "approach": "Confetti", "rejected_because": "Withheld reason sentinel",
            }],
        )
        plan = resolve_scope(scope=SCOPE, store=store)
        service = MemoryRecallService(store)
        prepared = service.prepare_query(TASK)
        raw = {r.node.id: r for r in service.memory_recall(
            prepared, scope=SCOPE, max_results=30, depth="decision", log_access=False,
        )}
        for node in (strong, schema, primary):
            assert raw[node.id].withheld is None
        for node in (weak, old_schema, weak_primary):
            assert raw[node.id].withheld == "below_threshold"
        assert raw[schema.id].lexical_score == 0 and "trigger" in raw[schema.id].methods

        concepts = prompts.select_context_concepts(store, task=prepared, plan=plan)
        schemas = prompts.select_context_schemas(store, task=prepared, plan=plan)
        decisions = prompts.select_decision_history(store, task=prepared, plan=plan)
        for item in (*concepts, *schemas, *decisions):
            node = item.primary if isinstance(item, prompts.DecisionHistory) else item.node
            assert item.withheld == raw[node.id].withheld
        assert next(item for item in decisions if item.primary.id == weak_primary.id).reasons == ()
        block = prompts.retrieval_context_prompt(store, task=prepared, scope=SCOPE)
        for node in (strong, schema, primary):
            assert node.content in block
        for node in (weak, old_schema, weak_primary):
            assert_reference(block, node)
        assert f'superseded=true superseded_by=["{correction.id}"]' in block
        assert old_schema.context["trigger"] + '"' not in block
        assert untriggered.id not in block
        assert "Holds locks indefinitely" in block
        for reason in ("Withheld reason sentinel", "Foreign reason sentinel", "Expired reason sentinel"):
            assert reason not in block


@pytest.mark.parametrize("selector,level", [
    (prompts.select_context_concepts, "concept"),
    (prompts.select_context_schemas, "schema"),
    (prompts.select_decision_history, "trace"),
])
def test_reference_rank_does_not_displace_qualified_selector_result(tmp_path, monkeypatch, selector, level):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        nodes = []
        for text in (BALLOON, TASK + " requires a diagnostic snapshot."):
            if level == "trace":
                node, _ = store.append_trace_with_rejected_alternatives(
                    text, {"scope": SCOPE}, alternatives_considered=[{
                        "approach": "Alternative", "rejected_because": "Reason",
                    }],
                )
            else:
                node = store.create_node(level=level, content=text, context={"scope": SCOPE})
            nodes.append(node)
        service = MemoryRecallService(store)
        raw = {r.node.id: r for r in service.memory_recall(
            TASK, scope=SCOPE, max_results=20, log_access=False,
        )}
        weak, strong = (raw[node.id] for node in nodes)
        assert weak.withheld == "below_threshold" and strong.withheld is None
        if level == "schema":
            # Trigger eligibility is independent of the inherited gate verdict.
            weak = replace(weak, node=replace(weak.node, context={"trigger": TASK}), methods=("trigger",))
            strong = replace(strong, node=replace(strong.node, context={"trigger": TASK}), methods=("trigger",))
        monkeypatch.setattr(MemoryRecallService, "memory_recall", lambda *a, **kw: [
            replace(weak, score=100), replace(strong, score=0.1),
        ])
        limit_key = {"concept": "max_concepts", "schema": "max_schemas", "trace": "max_items"}[level]
        items = selector(store, task=TASK, plan=resolve_scope(scope=SCOPE, store=store), **{limit_key: 1})
        assert len(items) == 1
        node = items[0].primary if level == "trace" else items[0].node
        assert node.id == strong.node.id and items[0].withheld is None


def test_nonlexical_qualified_vector_support_reaches_direct_prompt(tmp_path):
    class Encoder:
        def embed(self, text):
            return [1.0, 0.0, 0.0]

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = concept(store, "Bounded retries prevent lock contention.", embedding=[1.0, 0.0, 0.0])
        service = MemoryRecallService(store, embedder=Encoder())
        prepared = service.prepare_query(UNRELATED)
        raw = service.memory_recall(prepared, scope=SCOPE, log_access=False)
        assert len(raw) == 1 and raw[0].withheld is None
        assert raw[0].lexical_score == 0 and raw[0].methods == ("vector",)
        block = prompts.retrieval_context_prompt(store, task=prepared, scope=SCOPE)
        assert node.content in block and "below_threshold" not in block


def test_min_confidence_scope_and_decay_restrict_query_and_empty_browse(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        qualified = concept(store, TASK + " valid result.", stats={"confidence": 0.9, "unique_agents": 2})
        low = concept(store, TASK + " low confidence.", stats={"confidence": 0.1})
        decayed = concept(store, TASK + " expired result.")
        store.soft_delete_node(decayed.id, "expired")
        foreign = store.create_node(
            level="concept", content=TASK + " foreign result.", context={"scope": "project:foreign"},
        )
        for task in (TASK, ""):
            block = prompts.retrieval_context_prompt(store, task=task, scope=SCOPE, min_confidence=0.7)
            assert qualified.content in block
            assert all(node.id not in block for node in (low, decayed, foreign))
