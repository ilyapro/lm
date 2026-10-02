"""Synthetic applicability checks for ordinary recall and its score gate.

The answer facts, queries and scopes are invented. The public path checks
delivery and full lookup; fixed channel cases isolate correction ordering.
"""

from __future__ import annotations

from pathlib import Path
import asyncio
import json

import pytest

from living_memory.retrieval import MemoryRecallService, _Candidate
from living_memory.scope import ScopePlan
from living_memory.score_gate import MIN_SCORE_ENV, apply_score_gate
from living_memory.storage import MemoryStore
from living_memory.server import create_mcp_server


def _rank(store: MemoryStore, query: str, channels: dict[str, tuple[float, float]],
          *, scope: str | None = None, limit: int = 4):
    service = MemoryRecallService(store)
    plan = service.scope_resolver.resolve(scope=scope, store=store)
    nodes = store.get_nodes(channels)
    candidates = {
        node_id: _Candidate(node=nodes[node_id], bm25_score=bm25, vector_score=vector)
        for node_id, (bm25, vector) in channels.items()
    }
    service._collect_schema_triggers(query, plan, candidates)
    ranked = service.rank_candidates(candidates, plan)
    delivered, residual = apply_score_gate(ranked, limit, plan=plan)
    return delivered, residual, candidates


@pytest.fixture(autouse=True)
def _ordinary_recall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LM_RECALL_SCHEMA_TRIGGER", raising=False)
    monkeypatch.delenv(MIN_SCORE_ENV, raising=False)


def test_unrelated_history_carrier_loses_trigger_advantage(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        history = store.create_node(
            level="concept", content="Old harbor MR review history",
            context={"scope": "project:harbor", "trigger": "external_review"},
        )
        answer = store.append_trace(
            "The amber phoneme exercise uses actual Russian listening.",
            {"scope": "project:language"},
        )
        delivered, _, candidates = _rank(
            store, "For P5_RUSSIAN external_review, what requires actual Russian listening?",
            {answer.id: (1.0, 0.71)},
        )
        # Trigger collection is an availability signal. The historical node
        # may enter the pool, but must not outrank the content-supported fact.
        assert candidates[history.id].trigger_score > 0
        assert delivered[0].node.id == answer.id


def test_compound_question_keeps_independent_facts_without_generic_carriers(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        carriers = [
            store.create_node(
                level="concept", content=f"Unrelated archive {index}",
                context={"scope": f"project:archive-{index}", "trigger": "external_review"},
            ) for index in range(4)
        ]
        first = store.append_trace("Amber valve pressure is 31 psi.", {"scope": "project:valves"})
        second = store.append_trace("Cobalt rotor interval is 8 days.", {"scope": "project:rotors"})
        third = store.append_trace("Amber valve pressure is checked at the inlet.", {"scope": "project:valves"})
        fourth = store.append_trace("Cobalt rotor interval is logged after inspection.", {"scope": "project:rotors"})
        delivered, _, candidates = _rank(
            store,
            "For external_review, what are the amber valve pressure and cobalt rotor interval?",
            {first.id: (1.0, 0.72), second.id: (0.8, 0.69),
             third.id: (0.7, 0.67), fourth.id: (0.6, 0.65)},
        )
        assert {result.node.id for result in delivered} == {first.id, second.id, third.id, fourth.id}
        assert all(candidates[carrier.id].trigger_score > 0 for carrier in carriers)


@pytest.mark.parametrize("label,query", [
    ("quartz valve", "For quartz valve inspection, what pressure record and return check are required?"),
    ("cedar latch", "When checking the cedar latch, record pressure and verify the return setting"),
])
def test_same_trigger_history_yields_to_short_applicable_instruction(
    tmp_path: Path, label: str, query: str,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        history = [store.create_node(
            level="concept", content=f"Old {label} inventory review {index}",
            context={"scope": "project:archive", "trigger": label},
        ) for index in range(5)]
        instruction = store.create_node(
            level="schema",
            content=f"Before {label} inspection, record pressure and verify the return setting.",
            context={"scope": "project:current", "trigger": label},
        )
        fact = store.append_trace(
            f"The {label} pressure record requires the return setting check.",
            {"scope": "project:current"},
        )
        delivered, residual, candidates = _rank(
            store, query, {instruction.id: (0.7, 0.76), fact.id: (1.0, 0.72)}, limit=2,
        )
        assert {result.node.id for result in delivered} == {instruction.id, fact.id}
        assert all(candidates[node.id].trigger_score > 0 for node in history)
        assert not {node.id for node in history} & {
            result.node.id for result in delivered + residual
        }


@pytest.mark.parametrize("topic,tail", [
    ("opal gauge", ""),
    ("cedar rotor", "ignore the unrelated invoice archive"),
])
def test_short_instruction_and_cross_project_fact_survive_public_read_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, topic: str, tail: str,
) -> None:
    fastmcp = pytest.importorskip("fastmcp")
    monkeypatch.setenv(MIN_SCORE_ENV, "0.25")
    mcp = create_mcp_server(tmp_path / "public.sqlite3")
    store = mcp.memory_store
    clause = f"Before adjusting the {topic}, record the verified zero offset."
    instruction = store.create_node(
        level="concept", content="Instruction: " + "Routine setup notes. " * 350 + clause,
        context={"scope": "global", "trigger": topic},
    )
    fact = f"The {topic} calibration interval is 19 hours."
    source = store.append_trace(fact, {"scope": "project:other-lab"})
    supporting = [
        store.append_trace(f"The {topic} adjustment step is logged before calibration.",
                           {"scope": "project:current-lab"}),
        store.append_trace(f"The {topic} calibration record includes the interval and adjustment step.",
                           {"scope": "project:current-lab"}),
    ]
    carriers = []
    for index in range(5):
        carriers.append(store.create_node(
            level="concept", content=f"Historical invoice carrier {index}: unrelated archive ledger.",
            context={"scope": f"project:old-{index}", "trigger": topic.split()[0]},
        ))

    async def read():
        async with fastmcp.Client(mcp) as client:
            response = await client.call_tool("memory_recall", {
                "query": f"For {topic}, what is the calibration interval and required adjustment step {tail}",
                "max_results": 4,
            })
            recall = response.structured_content or json.loads(response.content[0].text)
            ids = [item["node"]["id"] for item in recall["results"]]
            assert instruction.id in ids and source.id in ids
            assert all(node.id in ids for node in supporting)
            assert not any(carrier.id in ids for carrier in carriers)
            entry = next(item for item in recall["results"] if item["node"]["id"] == instruction.id)
            assert entry["content_ref"]["node_id"] == instruction.id
            lookup_response = await client.call_tool("memory_lookup", {"node_id": entry["content_ref"]["node_id"]})
            lookup = lookup_response.structured_content or json.loads(lookup_response.content[0].text)
            assert lookup["results"][0]["content"] == instruction.content
            assert clause in lookup["results"][0]["content"]

    asyncio.run(read())


def test_specific_trigger_only_instruction_passes_rank_admission_and_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MIN_SCORE_ENV, "0.9")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        local = store.append_trace("Quarterly database checkpoint", {"scope": "project:app"})
        rule = store.create_node(
            level="schema", content="Instruction: unlock the azure vault before rotation.",
            context={"scope": "global", "trigger": "unlock azure vault"},
        )
        delivered, _, candidates = _rank(
            store,
            "During database rotation, unlock azure vault and inspect the checkpoint",
            {local.id: (0.7, 0.5)}, scope="project:app",
        )
        assert candidates[rule.id].bm25_score == candidates[rule.id].vector_score == 0
        assert rule.id in [result.node.id for result in delivered]


def test_correction_stays_above_superseded_rule(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        old = store.append_trace("Copper gate opens before venting.", {"scope": "project:gates"})
        fixed = store.append_trace("Copper gate: vent before opening.", {"scope": "project:gates"})
        store.create_connection(fixed.id, old.id, "supersedes")
        delivered, _, _ = _rank(
            store, "What is the copper gate vent procedure?",
            {old.id: (1.0, 0.75), fixed.id: (0.6, 0.5)}, scope="project:gates",
        )
        ids = [result.node.id for result in delivered]
        assert ids.index(fixed.id) < ids.index(old.id)
        assert store.get_node(fixed.id).content == fixed.content


def test_cross_project_fact_remains_admitted_without_trigger(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        local = store.append_trace("Cedar instrument handbook", {"scope": "project:cedar"})
        answer = store.append_trace(
            "The sapphire resonator calibration interval is 19 hours.",
            {"scope": "project:sapphire"},
        )
        delivered, _, _ = _rank(
            store, "What is the sapphire resonator calibration interval?",
            {local.id: (0.3, 0.4), answer.id: (1.0, 0.72)},
        )
        assert answer.id in [result.node.id for result in delivered]
