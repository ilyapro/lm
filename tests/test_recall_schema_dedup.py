from pathlib import Path

import pytest

from living_memory.models import Node
from living_memory.retrieval import MemoryRecallService, RecallResult
from living_memory.schema_dedup import (
    SCHEMA_DEDUP_ENV,
    collapse_schema_duplicates,
    schema_title,
)
from living_memory.storage import MemoryStore

ON = {SCHEMA_DEDUP_ENV: "1"}


def _result(node_id: str, level: str, content: str, score: float) -> RecallResult:
    return RecallResult(node=Node(id=node_id, level=level, content=content), score=score)


def _ranked() -> list[RecallResult]:
    return [
        _result("s1", "schema", "Procedure: verify pass\n1. a", 0.9),
        _result("t1", "trace", "Procedure: verify pass\n1. a", 0.8),
        _result("s2", "schema", "Procedure:  Verify   Pass\n1. b", 0.7),
        _result("s3", "schema", "Procedure: task outcome\n1. c", 0.6),
        _result("t2", "trace", "Procedure: verify pass\n1. a", 0.5),
        _result("s4", "schema", "Procedure: verify pass\n1. d", 0.4),
    ]


def test_unset_is_identity() -> None:
    ranked = _ranked()
    kept, collapsed = collapse_schema_duplicates(ranked, env={})
    assert kept == ranked
    assert collapsed == []


@pytest.mark.parametrize("value", ["", "0", "false", "off"])
def test_non_enabling_values_are_identity(value: str) -> None:
    ranked = _ranked()
    kept, collapsed = collapse_schema_duplicates(ranked, env={SCHEMA_DEDUP_ENV: value})
    assert kept == ranked and collapsed == []


def test_on_keeps_highest_ranked_schema_per_title_and_never_touches_traces() -> None:
    kept, collapsed = collapse_schema_duplicates(_ranked(), env=ON)
    assert [r.node_id for r in kept] == ["s1", "t1", "s3", "t2"]
    assert [r.node_id for r in collapsed] == ["s2", "s4"]


def test_untitled_schemas_are_not_collapsed() -> None:
    ranked = [
        _result("a", "schema", "Procedure\n1. x", 0.9),
        _result("b", "schema", "Procedure\n1. y", 0.8),
        _result("c", "schema", "", 0.7),
        _result("d", "schema", "   ", 0.6),
    ]
    kept, collapsed = collapse_schema_duplicates(ranked, env=ON)
    assert kept == ranked and collapsed == []


def test_schema_title_definition() -> None:
    node = Node(id="x", level="schema", content="  Procedure: Reopen  lesson \n1. step")
    assert schema_title(node) == "procedure: reopen lesson"
    assert schema_title(Node(id="y", level="concept", content="Procedure: reopen lesson")) is None
    carrier = Node(id="z", level="concept", content="Evidence: Reopen  lesson\n- a", context={"trigger": "reopen lesson"})
    assert schema_title(carrier) == "evidence: reopen lesson"


def test_group_carriers_sharing_a_title_keep_one_slot() -> None:
    """A group carrier holds its records whole, so the best-matching carrier
    of a title wins its slot as the legacy schema it replaced did; the
    others give up theirs instead of crowding out records."""

    def carrier(node_id: str, score: float) -> RecallResult:
        node = Node(id=node_id, level="concept", content="Evidence: verify pass\n- x", context={"trigger": "verify pass"})
        return RecallResult(node=node, score=score)

    ranked = [carrier("c1", 0.9), _result("t1", "trace", "x", 0.8), carrier("c2", 0.7)]
    kept, collapsed = collapse_schema_duplicates(ranked, env=ON)
    assert [r.node_id for r in kept] == ["c1", "t1"]
    assert [r.node_id for r in collapsed] == ["c2"]


def _seed(store: MemoryStore) -> list[str]:
    ids = []
    for index in range(3):
        node = store.create_node(
            level="schema",
            content=(
                "Procedure: deploy rollback\n"
                f"1. restore release variant {index}"
            ),
            context={"scope": "project:alpha", "procedure_id": "deploy_rollback"},
            stats={"confidence": 0.9 - index * 0.1},
        )
        ids.append(node.id)
    for index in range(4):
        store.append_trace(
            f"Release checklist note {index} for a deploy: record the owner, "
            "then confirm backups before starting the rollback to restore service.",
            {"scope": "project:alpha", "agent": "agent-a"},
        )
    return ids


def test_live_path_frees_slots_for_next_ranked(tmp_path: Path, monkeypatch) -> None:
    # Exercise real retrieval with a deterministic local backend. The concise
    # schemas deliberately match more closely than the supporting notes, so
    # duplicates occupy baseline slots regardless of an installed model.
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        schema_ids = set(_seed(store))
        service = MemoryRecallService(store)

        monkeypatch.delenv(SCHEMA_DEDUP_ENV, raising=False)
        baseline = service.memory_recall(
            "deploy rollback restore release",
            scope="project:alpha",
            max_results=4,
            log_access=False,
            log_event=False,
        )
        base_schemas = [r for r in baseline if r.node_id in schema_ids]
        assert len(base_schemas) >= 2, "fixture must deliver duplicates without the valve"
        expected_ids = [
            r.node_id
            for r in baseline + service.last_residual
            if r.node_id not in schema_ids or r.node_id == base_schemas[0].node_id
        ][:4]

        monkeypatch.setenv(SCHEMA_DEDUP_ENV, "1")
        deduped = service.memory_recall(
            "deploy rollback restore release",
            scope="project:alpha",
            max_results=4,
            log_access=False,
            log_event=False,
        )
        schemas = [r for r in deduped if r.node_id in schema_ids]
        assert len(schemas) == 1
        # The kept one is the highest-ranked schema of the baseline answer.
        assert schemas[0].node_id == base_schemas[0].node_id
        # Freed slots go to the next ranked results, not left empty.
        assert len(deduped) == len(baseline) == 4
        assert [r.node_id for r in deduped] == expected_ids
        assert set(expected_ids) - {r.node_id for r in baseline}
        assert all(r.node.level == "trace" for r in deduped if r.node_id not in schema_ids)
