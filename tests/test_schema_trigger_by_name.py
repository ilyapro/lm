"""``LM_RECALL_SCHEMA_TRIGGER=name``: a schema is ranked by meaning, not by trigger words.

Pins docs/recall-schema-trigger.md: under the valve the trigger finds nothing
and scores nothing; words shared with a trigger neither raise a schema nor
carry it past the quality gate; a query that *is* the procedure's name puts
that schema first (so it ships in full) when it passes the gate on its own
score. Unset, the legacy channel is untouched (tests/test_recall_score_gate.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from living_memory.delivery import shape_recall_results
from living_memory.retrieval import (
    SCHEMA_NAME_TRIGGER_SCORE,
    SCHEMA_TRIGGER_ENV,
    MemoryRecallService,
    RecallResult,
    schema_trigger_by_name,
)
from living_memory.score_gate import GATE_FORM_ENV, MIN_SCORE_ENV, passes_gate, trigger_gate_score
from living_memory.storage import MemoryStore

SCOPE = "project:names"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (MIN_SCORE_ENV, GATE_FORM_ENV, SCHEMA_TRIGGER_ENV, "LM_RECALL_SCHEMA_DEDUP"):
        monkeypatch.delenv(key, raising=False)


def _seed(store: MemoryStore) -> None:
    store.append_trace(
        "Database migration rollback after deploy failure: restore snapshot first",
        {"scope": SCOPE, "agent": "a"},
    )
    store.append_trace("Migration rollback checklist for the database", {"scope": SCOPE, "agent": "a"})
    store.append_trace(
        "Vault keys: rotate vault keys every quarter, vault keys live in the rotate store",
        {"scope": SCOPE, "agent": "b"},
    )
    for index in range(30):
        store.append_trace(f"unrelated note {index} about lunch menus", {"scope": SCOPE, "agent": "c"})


def _schema(store: MemoryStore):  # type: ignore[no-untyped-def]
    return store.create_node(
        level="schema",
        content="Procedure: rotate vault keys\n1. Quarterly key rotation for vault tokens.",
        context={"scope": SCOPE, "trigger": "rotate vault keys"},
    )


def _recall(store: MemoryStore, query: str, max_results: int = 5) -> tuple[list[RecallResult], list[RecallResult]]:
    service = MemoryRecallService(store)
    delivered = service.memory_recall(
        query, scope=SCOPE, max_results=max_results, log_access=False, log_event=False
    )
    return delivered, service.last_residual


@pytest.mark.parametrize("raw,on", [(None, False), ("", False), ("1", False), ("name", True), (" NAME ", True)])
def test_valve_reads_only_name(monkeypatch: pytest.MonkeyPatch, raw: str | None, on: bool) -> None:
    if raw is not None:
        monkeypatch.setenv(SCHEMA_TRIGGER_ENV, raw)
    assert schema_trigger_by_name() is on


def test_schema_triggers_reads_live_schemas_of_the_scope(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        live = _schema(store)
        store.create_node(level="schema", content="Procedure: no trigger", context={"scope": SCOPE})
        store.create_node(
            level="schema",
            content="Procedure: elsewhere",
            context={"scope": "project:other", "trigger": "elsewhere"},
        )
        assert store.schema_triggers(SCOPE) == [(live.id, "rotate vault keys")]


def test_shared_words_neither_raise_a_schema_nor_pass_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MIN_SCORE_ENV, "0.9")
    query = "database migration rollback: rotate vault keys too"
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed(store)
        schema = _schema(store)
        legacy, _ = _recall(store, query, max_results=10)
        monkeypatch.setenv(SCHEMA_TRIGGER_ENV, "name")
        delivered, residual = _recall(store, query, max_results=10)
    # Legacy: the trigger words alone carry the schema into the delivery.
    assert schema.id in [r.node.id for r in legacy]
    # Name mode: no trigger score, no slot of its own on a 0.9 gate.
    assert schema.id not in [r.node.id for r in delivered[1:]]
    for result in delivered + residual:
        assert result.trigger_score == 0.0
        assert "trigger" not in result.methods


def test_query_naming_the_procedure_puts_it_first_and_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SCHEMA_TRIGGER_ENV, "name")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed(store)
        schema = _schema(store)
        delivered, _ = _recall(store, "rotate_vault_keys")
    first = delivered[0]
    assert first.node.id == schema.id
    assert first.trigger_score == SCHEMA_NAME_TRIGGER_SCORE
    assert "trigger" in first.methods
    shaped = shape_recall_results(
        delivered,
        already_delivered_ids=set(),
        snippet_max_chars=1000,
        context_value_max_chars=300,
        session_dedup=False,
    )
    assert shaped[0]["delivery"] == "full"
    assert shaped[0]["node"]["content"] == schema.content


def test_a_named_schema_must_pass_the_gate_to_go_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SCHEMA_TRIGGER_ENV, "name")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        schema = _schema(store)
        other = store.append_trace("vault key rotation note", {"scope": SCOPE})
        service = MemoryRecallService(store)
        plan = service.scope_resolver.resolve(
            query="rotate vault keys", scope=SCOPE, ambient_context=None, store=store
        )
        strong = RecallResult(node=other, score=0.9, bm25_score=1.0, methods=("bm25",))
        weak = RecallResult(node=schema, score=0.1, bm25_score=0.1, methods=("bm25",))

        def reorder(threshold: str | None) -> list[RecallResult]:
            if threshold is None:
                monkeypatch.delenv(MIN_SCORE_ENV, raising=False)
            else:
                monkeypatch.setenv(MIN_SCORE_ENV, threshold)
            return service._named_schemas_first(
                "rotate vault keys", plan, [strong, weak], demotions=None, causal_mode=False
            )

        # Gate at 0.3: the schema's own score (0.4 * 0.1) fails -- it stays put.
        gated = reorder("0.3")
        assert [r.node.id for r in gated] == [other.id, schema.id]
        assert gated[1].trigger_score == SCHEMA_NAME_TRIGGER_SCORE
        assert not passes_gate(gated[1], 0.3)
        # Gate off: nothing to fail, the named schema goes first.
        assert [r.node.id for r in reorder(None)] == [schema.id, other.id]
        # A query that is not the name reorders nothing.
        assert service._named_schemas_first(
            "rotate vault keys quarterly", plan, [strong, weak], demotions=None, causal_mode=False
        ) == [strong, weak]


def test_no_trigger_scale_in_name_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        schema = _schema(store)
    hit = RecallResult(node=schema, score=1.7, trigger_score=0.975, methods=("trigger",))
    assert trigger_gate_score(hit) is not None
    monkeypatch.setenv(SCHEMA_TRIGGER_ENV, "name")
    assert trigger_gate_score(hit) is None
    assert not passes_gate(hit, 0.3)
