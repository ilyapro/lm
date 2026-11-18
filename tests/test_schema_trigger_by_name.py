"""``LM_RECALL_SCHEMA_TRIGGER=name``: a schema is ranked by meaning, not by trigger words.

Pins docs/recall-schema-trigger.md: under the valve the trigger finds nothing
and scores nothing; words shared with a trigger neither raise a schema nor
carry it past the quality gate; a query that *is* the procedure's name is
that schema's complete lexical match and puts it first (so it ships in
full) when it passes the gate on that score. Unset, ordinary trigger recall
uses query applicability (tests/test_recall_applicability.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from living_memory.delivery import DeliveryHistory, shape_recall_results
from living_memory.retrieval import (
    SCHEMA_NAME_TRIGGER_SCORE,
    SCHEMA_TRIGGER_ENV,
    MemoryRecallService,
    RecallResult,
    _Candidate,
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
    # Ordinary trigger availability qualifies its content.
    assert any(r.node.id == schema.id and r.withheld is None for r in legacy)
    # Name mode retains a bounded reference, without qualifying the content.
    assert any(r.node.id == schema.id and r.withheld == "below_threshold" for r in delivered)
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
        delivery_history=DeliveryHistory(),
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
            scope=SCOPE, ambient_context=None, store=store
        )
        strong = RecallResult(node=other, score=0.9, bm25_score=1.0, methods=("bm25",))
        weak = RecallResult(node=schema, score=0.1, bm25_score=0.1, methods=("bm25",))

        def reorder(threshold: str | None, demotion: float = 1.0) -> list[RecallResult]:
            if threshold is None:
                monkeypatch.delenv(MIN_SCORE_ENV, raising=False)
            else:
                monkeypatch.setenv(MIN_SCORE_ENV, threshold)
            return service._named_schemas_first(
                "rotate vault keys", plan, [strong, weak], demotions={schema.id: demotion}, causal_mode=False
            )

        # The name is complete lexical support: 1.0 on the reference blend
        # is 0.4, so a 0.3 gate passes it on the same scale as everyone.
        gated = reorder("0.3")
        assert [r.node.id for r in gated] == [schema.id, other.id]
        assert gated[0].bm25_score == 1.0
        assert gated[0].trigger_score == SCHEMA_NAME_TRIGGER_SCORE
        # Negative evidence still takes it out: the query's own demotion, or
        # a gate the lexical match alone does not reach. It stays put.
        for threshold, demotion in (("0.3", 0.5), ("0.9", 1.0)):
            gated = reorder(threshold, demotion)
            assert [r.node.id for r in gated] == [other.id, schema.id]
            assert not passes_gate(gated[1], float(threshold), {schema.id: demotion})
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


def test_name_mode_orders_on_the_gate_blend_not_scope_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Order and gate read one score: learned per-scope weights do not reorder under the valve."""

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        vector_only = store.append_trace("vector only evidence", {"scope": SCOPE, "agent": "a"})
        procedure = store.create_node(
            level="schema",
            content="Procedure: mixed evidence\n1. Step.",
            context={"scope": SCOPE, "trigger": "mixed evidence"},
        )
        store.set_retrieval_weights(SCOPE, bm25=0.05, vector=0.9, graph=0.05)
        service = MemoryRecallService(store)
        plan = service.scope_resolver.resolve(scope=SCOPE, ambient_context=None, store=store)

        def order() -> list[str]:
            candidates = {
                vector_only.id: _Candidate(node=vector_only, vector_score=0.57),
                procedure.id: _Candidate(node=procedure, bm25_score=0.5, vector_score=0.52, graph_score=0.3),
            }
            return [result.node.id for result in service.rank_candidates(candidates, plan)]

        assert order()[0] == vector_only.id
        monkeypatch.setenv(SCHEMA_TRIGGER_ENV, "name")
        assert order()[0] == procedure.id


@pytest.mark.parametrize("repeatable", [False, True])
def test_ranked_trigger_read_observes_selection_boundary(tmp_path, monkeypatch, repeatable):
    query = "rotate vault keys"
    monkeypatch.setenv(SCHEMA_TRIGGER_ENV, "name")
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store, MemoryStore(path) as writer:
        ids = ["promoted", "renamed", "moved", "demoted", "decayed", "deleted"]
        for node_id in ids:
            store.create_node(
                node_id=node_id, level="trace" if node_id == "promoted" else "schema",
                content=query, context={"scope": SCOPE, "trigger": query},
            )
        service = MemoryRecallService(store)
        rank = service.rank_candidates

        def rank_then_edit(*args, **kwargs):
            ranked = rank(*args, **kwargs)
            assert {r.node.id for r in ranked} == set(ids)
            with writer.transaction():
                writer.connection.execute("UPDATE nodes SET level='concept' WHERE id='promoted'")
                writer.connection.execute("UPDATE nodes SET context='{}' WHERE id='renamed'")
                writer.connection.execute("UPDATE nodes SET scope='elsewhere' WHERE id='moved'")
                writer.connection.execute("UPDATE nodes SET level='trace' WHERE id='demoted'")
                writer.connection.execute("UPDATE nodes SET decayed=1 WHERE id='decayed'")
                writer.connection.execute("DELETE FROM nodes WHERE id='deleted'")
                writer.create_node(
                    node_id="unranked", level="schema", content=query,
                    context={"scope": SCOPE, "trigger": query},
                )
            return ranked

        monkeypatch.setattr(service, "rank_candidates", rank_then_edit)
        read = store.schema_triggers
        rows = []

        def observe(*args, **kwargs):
            selected = read(*args, **kwargs)
            rows.extend(selected)
            return selected

        monkeypatch.setattr(store, "schema_triggers", observe)
        with service.recall_response(
            query, scope=SCOPE, depth=0, max_results=10, repeatable_selection=repeatable,
            log_access=False, repair_pending=False,
        ) as delivered:
            expected = set(ids[1:]) if repeatable else {"promoted"}
            assert sorted(rows) == sorted((node_id, query) for node_id in expected)
            assert {r.node.id for r in delivered if "trigger" in r.methods} == expected


@pytest.mark.parametrize("scope", [None, SCOPE])
def test_ranked_trigger_projection_preserves_sqlite_values(tmp_path, scope):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        contexts = [
            '{"trigger":true}', '{"trigger":false}', '{"trigger":0}', '{"trigger":0.5}',
            '{"trigger":99999999999999999999}', '{"trigger":[]}', '{"trigger":{"key": 1}}',
            '{"trigger":""}', '{"trigger":null}', '{}', '{"trigger":"first","trigger":"last"}',
        ]
        for context in contexts:
            node = store.create_node(level="concept", content="Stored procedure", context={"scope": SCOPE})
            store.connection.execute("UPDATE nodes SET context=? WHERE id=?", (context, node.id))
        unranked = _schema(store)
        ranked = [row[0] for row in store.connection.execute("SELECT id FROM nodes WHERE id != ?", (unranked.id,))]
        assert sorted(store.schema_triggers(scope, node_ids=ranked)) == sorted(
            row for row in store.schema_triggers(scope) if row[0] != unranked.id
        )
