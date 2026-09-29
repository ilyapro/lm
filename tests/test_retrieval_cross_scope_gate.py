"""Cross-scope admission gate: weak out-of-requested-scope candidates are
dropped from ranking unless they carry deliberate evidence.

Deliberate evidence is a schema trigger, a strong graph connection, or vector
similarity that is strong in absolute terms or comparable to the best ungated
match. bm25 rank scores never qualify: `_collect_bm25` normalizes ranks per
scope, so the broadest scope's top FTS hit always carries bm25 = 1.0 however
weak the lexical match is. Requested-scope candidates are never gated, and a
recall whose only candidates are cross-scope is served in full — the gate can
never empty a recall on its own.
"""

from __future__ import annotations

from pathlib import Path

from living_memory.models import Node
from living_memory.retrieval import (
    CROSS_SCOPE_GRAPH_ADMIT,
    CROSS_SCOPE_RELATIVE_VECTOR,
    CROSS_SCOPE_VECTOR_ADMIT,
    MemoryRecallService,
    _Candidate,
)
from living_memory.scope import ScopePlan
from living_memory.storage import MemoryStore

PROJECT_SCOPE = "project:app"
PROJECT_PLAN = ScopePlan(requested_scope=PROJECT_SCOPE, scopes=(PROJECT_SCOPE, "global"))
SESSION_PLAN = ScopePlan(
    requested_scope="session:s1",
    scopes=("session:s1", PROJECT_SCOPE, "global"),
)


def _node(node_id: str, scope: str, *, level: str = "trace", decayed: bool = False) -> Node:
    return Node(
        id=node_id,
        level=level,  # type: ignore[arg-type]
        content=f"content of {node_id}",
        scope=scope,
        decayed=decayed,
    )


def _candidate(
    node_id: str,
    scope: str,
    *,
    level: str = "trace",
    decayed: bool = False,
    **scores: float,
) -> _Candidate:
    return _Candidate(node=_node(node_id, scope, level=level, decayed=decayed), **scores)


def _rank_ids(store: MemoryStore, candidates: dict[str, _Candidate], plan: ScopePlan) -> list[str]:
    ranked = MemoryRecallService(store).rank_candidates(candidates, plan)
    return [result.node.id for result in ranked]


def test_weak_global_tail_is_gated_under_project_plan(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rank_ids(
            store,
            {
                "p1": _candidate("p1", PROJECT_SCOPE, bm25_score=0.5, vector_score=0.5),
                "g1": _candidate("g1", "global", bm25_score=1.0, vector_score=0.2),
            },
            PROJECT_PLAN,
        )
    assert "p1" in ids
    assert "g1" not in ids


def test_bm25_rank_score_alone_never_admits_cross_scope(tmp_path: Path) -> None:
    # The broadest scope's top FTS hit always records bm25 = 1.0 by rank
    # normalization; that manufactured score must not count as evidence.
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rank_ids(
            store,
            {
                "p1": _candidate("p1", PROJECT_SCOPE, bm25_score=0.2, vector_score=0.3),
                "g1": _candidate("g1", "global", bm25_score=1.0),
            },
            PROJECT_PLAN,
        )
    assert "p1" in ids
    assert "g1" not in ids


def test_strong_absolute_vector_admits_cross_scope(tmp_path: Path) -> None:
    # 0.7 clears the absolute floor even though it is below the relative bar
    # against the requested-scope best (0.9).
    assert CROSS_SCOPE_VECTOR_ADMIT <= 0.7 < CROSS_SCOPE_RELATIVE_VECTOR * 0.9
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rank_ids(
            store,
            {
                "p1": _candidate("p1", PROJECT_SCOPE, vector_score=0.9),
                "g1": _candidate("g1", "global", vector_score=0.7),
            },
            PROJECT_PLAN,
        )
    assert ids and set(ids) == {"p1", "g1"}


def test_comparable_relative_vector_admits_cross_scope(tmp_path: Path) -> None:
    # 0.38 vs a requested-scope best of 0.40 is comparable relevance; it is
    # admitted despite sitting far below the absolute floor.
    assert CROSS_SCOPE_RELATIVE_VECTOR * 0.40 <= 0.38 < CROSS_SCOPE_VECTOR_ADMIT
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rank_ids(
            store,
            {
                "p1": _candidate("p1", PROJECT_SCOPE, vector_score=0.40),
                "g1": _candidate("g1", "global", vector_score=0.38),
            },
            PROJECT_PLAN,
        )
    assert set(ids) == {"p1", "g1"}


def test_below_comparable_vector_is_gated(tmp_path: Path) -> None:
    assert 0.30 < CROSS_SCOPE_RELATIVE_VECTOR * 0.40
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rank_ids(
            store,
            {
                "p1": _candidate("p1", PROJECT_SCOPE, vector_score=0.40),
                "g1": _candidate("g1", "global", vector_score=0.30, bm25_score=1.0),
            },
            PROJECT_PLAN,
        )
    assert ids == ["p1"]


def test_strong_graph_connection_admits_cross_scope_weak_graph_does_not(
    tmp_path: Path,
) -> None:
    assert 0.4 < CROSS_SCOPE_GRAPH_ADMIT <= 0.9
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rank_ids(
            store,
            {
                "p1": _candidate("p1", PROJECT_SCOPE, bm25_score=1.0, vector_score=0.5),
                "g_strong": _candidate("g_strong", "global", graph_score=0.9),
                "g_weak": _candidate("g_weak", "global", graph_score=0.4),
            },
            PROJECT_PLAN,
        )
    assert "g_strong" in ids
    assert "g_weak" not in ids


def test_schema_trigger_admits_cross_scope(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rank_ids(
            store,
            {
                "p1": _candidate("p1", PROJECT_SCOPE, vector_score=0.5),
                "g_schema": _candidate(
                    "g_schema", "global", trigger_score=0.97, level="schema"
                ),
            },
            PROJECT_PLAN,
        )
    assert "g_schema" in ids


def test_cross_scope_only_recall_is_fully_served(tmp_path: Path) -> None:
    # No requested-scope candidate at all: cross-scope is the only answer and
    # the gate must not drop any of it, however weak the evidence.
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rank_ids(
            store,
            {
                "g1": _candidate("g1", "global", bm25_score=1.0, vector_score=0.2),
                "g2": _candidate("g2", "global", bm25_score=0.5, vector_score=0.1),
            },
            PROJECT_PLAN,
        )
    assert set(ids) == {"g1", "g2"}


def test_requested_scope_candidates_are_never_gated(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rank_ids(
            store,
            {
                "p_weak": _candidate("p_weak", PROJECT_SCOPE, bm25_score=0.05),
                "g_strong": _candidate("g_strong", "global", vector_score=0.9),
            },
            PROJECT_PLAN,
        )
    assert "p_weak" in ids
    assert "g_strong" in ids


def test_decayed_requested_candidate_does_not_arm_the_gate(tmp_path: Path) -> None:
    # A candidate that can never rank must not arm the gate, or the gate
    # could empty a recall whose only live answers are cross-scope.
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rank_ids(
            store,
            {
                "p_dead": _candidate(
                    "p_dead", PROJECT_SCOPE, vector_score=0.9, decayed=True
                ),
                "g1": _candidate("g1", "global", bm25_score=1.0, vector_score=0.2),
            },
            PROJECT_PLAN,
        )
    assert ids == ["g1"]


def test_zero_vector_evidence_in_requested_scope_disables_relative_rule(
    tmp_path: Path,
) -> None:
    # With no vector evidence in the requested scope the relative rule would
    # otherwise degenerate to a zero bar and admit every weak vector match.
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rank_ids(
            store,
            {
                "p1": _candidate("p1", PROJECT_SCOPE, bm25_score=1.0),
                "g_weak": _candidate("g_weak", "global", vector_score=0.3),
                "g_strong": _candidate("g_strong", "global", vector_score=0.7),
            },
            PROJECT_PLAN,
        )
    assert "p1" in ids
    assert "g_weak" not in ids
    assert "g_strong" in ids


def test_session_plan_shields_ambient_project_scope(tmp_path: Path) -> None:
    # The project scope of a session triple comes from the caller's own
    # ambient project declaration: its candidates bypass the gate like the
    # session's, while weak global candidates are still gated.
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rank_ids(
            store,
            {
                "s1": _candidate("s1", "session:s1", bm25_score=0.5, vector_score=0.5),
                "p_weak": _candidate("p_weak", PROJECT_SCOPE, bm25_score=0.4, vector_score=0.1),
                "g_weak": _candidate("g_weak", "global", bm25_score=1.0, vector_score=0.2),
            },
            SESSION_PLAN,
        )
    assert "s1" in ids
    assert "p_weak" in ids
    assert "g_weak" not in ids


def test_session_plan_project_candidates_arm_the_gate_without_session_notes(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rank_ids(
            store,
            {
                "p1": _candidate("p1", PROJECT_SCOPE, vector_score=0.5),
                "g_weak": _candidate("g_weak", "global", bm25_score=1.0, vector_score=0.2),
                "g_comparable": _candidate("g_comparable", "global", vector_score=0.48),
            },
            SESSION_PLAN,
        )
    assert "p1" in ids
    assert "g_weak" not in ids
    assert "g_comparable" in ids  # within the relative bar of the project best


def test_session_plan_with_only_global_candidates_serves_them_all(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rank_ids(
            store,
            {
                "g1": _candidate("g1", "global", bm25_score=1.0, vector_score=0.2),
                "g2": _candidate("g2", "global", bm25_score=0.5, vector_score=0.12),
            },
            SESSION_PLAN,
        )
    assert set(ids) == {"g1", "g2"}


# A scope-less call searches the whole store; the configured default project
# only ranks first. Nothing in such a plan was asked to be narrower, so no
# scope in it gates another -- named or not.
SCOPELESS_PLAN = ScopePlan(
    requested_scope="global", scopes=(PROJECT_SCOPE, "global", "*")
)


def test_scopeless_plan_never_gates_a_weak_project_candidate(tmp_path: Path) -> None:
    # The project candidates are weak by every gate criterion (bm25-only); with
    # strong global candidates present they must still be served.
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rank_ids(
            store,
            {
                "p_weak": _candidate("p_weak", PROJECT_SCOPE, bm25_score=1.0),
                "o_weak": _candidate("o_weak", "project:other", bm25_score=1.0),
                "g_strong": _candidate("g_strong", "global", vector_score=0.9),
            },
            SCOPELESS_PLAN,
        )
    assert set(ids) == {"p_weak", "o_weak", "g_strong"}


def test_scopeless_plan_never_gates_weak_global_either(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = _rank_ids(
            store,
            {
                "p_strong": _candidate("p_strong", PROJECT_SCOPE, vector_score=0.9),
                "g_weak": _candidate("g_weak", "global", bm25_score=1.0, vector_score=0.1),
            },
            SCOPELESS_PLAN,
        )
    assert set(ids) == {"p_strong", "g_weak"}


# --- End-to-end: the gate is wired into memory_recall -----------------------


class _PlantedEmbeddingModel:
    """Returns planted unit vectors so recall similarities are exact."""

    def __init__(self, planted: dict[str, list[float]]) -> None:
        self._planted = planted

    def embed(self, text: str) -> list[float]:
        return self._planted[text]


def test_memory_recall_gates_weak_global_and_keeps_strong_global(tmp_path: Path) -> None:
    query = "release checklist for the app deploy"
    project_content = "App deploy release checklist with sign-off steps"
    strong_global = "Deploy release checklist template used across teams"
    weak_global = "Grafana palette conventions for dashboard tiles"
    planted = {
        query: [1.0, 0.0, 0.0],
        project_content: [0.5, 0.8660254, 0.0],
        strong_global: [0.8, 0.6, 0.0],
        weak_global: [0.25, 0.9682458, 0.0],
    }
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        project_node = store.create_node(
            level="trace",
            content=project_content,
            context={"scope": PROJECT_SCOPE},
            embedding=planted[project_content],
        )
        strong_node = store.create_node(
            level="trace",
            content=strong_global,
            context={"scope": "global"},
            embedding=planted[strong_global],
        )
        weak_node = store.create_node(
            level="trace",
            content=weak_global,
            context={"scope": "global"},
            embedding=planted[weak_global],
        )
        service = MemoryRecallService(store, embedder=_PlantedEmbeddingModel(planted))

        results = service.memory_recall(
            query,
            scope=PROJECT_SCOPE,
            depth=0,
            max_results=10,
            log_access=False,
            log_event=False,
        )

        ids = [result.node.id for result in results]
        assert project_node.id in ids
        assert strong_node.id in ids  # clearly strong cross-scope match survives
        assert weak_node.id not in ids  # weak global tail is gated
