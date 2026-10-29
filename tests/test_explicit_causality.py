"""Work context and chronology cannot supply causal knowledge."""

import json
import sqlite3

import pytest

from living_memory import consolidation
from living_memory.edge_backfill import backfill_edges
from living_memory.relations import (
    is_semantic_relation, semantic_relation_sql,
    source_relation_weight, source_relation_weight_sql,
)
from living_memory.replay import snapshot_evidence
from living_memory.retrieval import _traversal
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore
from test_lookup import FakeMCP
from test_mine_typed_edges import miner


SCOPE = "project:synthetic-causality"
PROBLEM = "Root cause: copper pump startup failure is due to a cracked fuse holder."
SOLUTION = "Fix: violet calendar previews render correctly after escaping ampersands in their HTML templates."
QUERY = "violet calendar previews render correctly after escaping ampersands in their HTML templates"
INFERRED = {
    "rule": "R2a", "kind": "failure_resolution", "gap_seconds": 60,
    "group_key": "task=maintain-several-independent-projects",
}
GROUPS = [
    ({"task": "maintain-several-independent-projects"}, {"task": "maintain-several-independent-projects"}),
    ({"session_id": "synthetic-multi-project-session"}, {"session_id": "synthetic-multi-project-session"}),
    ({"task": "repair-copper-pump"}, {"task": "repair-violet-calendar"}),
]


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "0.35")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    try:
        yield mcp
    finally:
        mcp.auto_consolidation.close()
        mcp.memory_store.close()


def write_pair(mcp, groups=GROUPS[0], *, problem_scope=SCOPE):
    remember = mcp.tools["memory_remember"]
    problem = remember(PROBLEM, context={
        "scope": problem_scope, **groups[0], "type": "root_cause",
        "timestamp": "2026-10-28T20:00:00Z",
    })["node"]["id"]
    solution = remember(SOLUTION, context={
        "scope": SCOPE, **groups[1], "type": "working_fix",
        "timestamp": "2026-10-28T20:01:00Z",
    })["node"]["id"]
    return problem, solution


def recall(mcp, depth):
    query = f"Why do {QUERY}?" if depth == "causal" else QUERY
    result = mcp.tools["memory_recall"](
        query=query, scope=SCOPE, depth=depth, max_results=5, view="full",
    )
    return {entry["node"]["id"]: entry for entry in result["results"]}


@pytest.mark.parametrize("groups", GROUPS, ids=["same-task", "same-session", "unshared"])
@pytest.mark.parametrize("depth", [1, "causal"])
def test_native_remember_does_not_infer_causality(native, groups, depth, tmp_path):
    problem, solution = write_pair(native, groups)
    store = native.memory_store
    assert store.connection.execute("SELECT count(*) FROM connections").fetchone()[0] == 0
    results = recall(native, depth)
    assert set(results) == {solution}
    assert results[solution]["graph_score"] == 0
    assert results[solution]["delivery"] == "full"
    assert store.get_node(problem).corrections == []
    assert not store.get_node(problem).decayed

    # The actual backfill and offline proposal entry point cannot reintroduce it.
    report = backfill_edges(store, apply=True)
    assert report["totals"]["new_rows"] == 0
    assert store.list_connections() == []
    report_path = tmp_path / "mining.json"
    assert miner.main([
        "--db", str(store.db_path), "--report", str(report_path),
        "--samples", str(tmp_path / "samples.md"),
    ]) == 0
    mined = json.loads(report_path.read_text())
    assert mined["edges_total"] == 0
    assert not any(rule["rule"].startswith("R2a") for rule in mined["rules"])
    assert all(rule["candidates_directed"] == 0 for rule in mined["rules"])


@pytest.mark.parametrize("depth", [1, "causal"])
@pytest.mark.parametrize("relation", ["caused", "requires"])
@pytest.mark.parametrize("sequence", ["inferred-then-explicit", "explicit-then-inferred"])
def test_native_explicit_connections_survive_history(native, depth, relation, sequence):
    problem, solution = write_pair(native)
    store = native.memory_store
    connect = native.tools["memory_connect"]
    if sequence == "inferred-then-explicit":
        history = store.create_connection(problem, solution, "caused", weight=.99, metadata=INFERRED)
        assert set(recall(native, depth)) == {solution}
        # The public writer replaces metadata, including when none is supplied.
        result = connect(problem, solution, relation, weight=.8)
        edge = store.get_connection(result["connection"]["id"])
        assert edge.metadata == {}
        assert edge.weight == .8
        if relation == "caused":
            assert edge.id == history.id
    else:
        result = connect(problem, solution, relation, weight=.8, metadata={"basis": "explicit"})
        edge = store.create_connection(problem, solution, relation, weight=.9, metadata=INFERRED, merge=True)
        assert edge.id == result["connection"]["id"]
        assert edge.metadata == {"basis": "explicit", **INFERRED}

    results = recall(native, depth)
    assert problem in results and solution in results
    assert results[problem]["graph_score"] > 0
    if depth == "causal":
        assert results[problem]["delivery"] == "full"
        assert results[problem]["node"]["content"] == PROBLEM
    else:
        assert results[problem]["content_ref"]["node_id"] == problem
    assert not store.get_node(problem).decayed
    assert store.get_node(problem).corrections == []


@pytest.mark.parametrize("depth", [1, "causal"])
def test_unmarked_historical_merge_is_indistinguishable(native, depth):
    problem, solution = write_pair(native)
    store = native.memory_store
    native.tools["memory_connect"](problem, solution, "caused", weight=.9)
    # The old producer's shallow merge retained no evidence of this explicit write.
    mixed = store.create_connection(problem, solution, "caused", weight=.9, metadata=INFERRED, merge=True)
    assert mixed.metadata == INFERRED
    before = store.get_connection(mixed.id)
    assert set(recall(native, depth)) == {solution}
    assert store.get_connection(mixed.id) == before


@pytest.mark.parametrize("basis", [None, "explicit"])
def test_semantic_merge_replaces_inferred_occupancy(native, basis):
    problem, solution = write_pair(native)
    store = native.memory_store
    old = store.create_connection(problem, solution, "caused", weight=.99, metadata=INFERRED)
    metadata = {"basis": basis} if basis else {}
    current = store.create_connection(problem, solution, "caused", weight=.6, metadata=metadata, merge=True)
    assert current.id == old.id
    assert current.weight == .6
    assert current.metadata == metadata
    assert store.list_connections() == [current]


@pytest.mark.parametrize("metadata,semantic", [
    ({}, True),
    ({"rule": "unknown", "kind": "unknown"}, True),
    (INFERRED, False),
    ({"rule": "R2a"}, False),
    ({"kind": "failure_resolution"}, False),
    ({**INFERRED, "basis": "provenance_derivation"}, False),
    ({**INFERRED, "basis": None}, False),
    ({**INFERRED, "basis": True}, False),
    ({**INFERRED, "basis": ["explicit"]}, False),
    ({**INFERRED, "basis": {"explicit": True}}, False),
    ({**INFERRED, "correction": "This does not assert a cause."}, False),
    ({**INFERRED, "basis": "explicit"}, True),
])
@pytest.mark.parametrize("relation", ["caused", "requires", "related", "contradicts", "supersedes"])
def test_python_sql_causal_policy_preserves_other_relations(metadata, semantic, relation):
    expected = semantic if relation == "caused" else True
    assert is_semantic_relation(relation, metadata) is expected
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE connections (type TEXT, metadata TEXT, weight REAL)")
        db.execute("INSERT INTO connections VALUES (?, ?, .9)", (relation, json.dumps(metadata)))
        for alias in ("", "c"):
            actual = db.execute(
                f"SELECT {semantic_relation_sql(alias)}, {source_relation_weight_sql(alias)} "
                "FROM connections AS c"
            ).fetchone()
            assert actual == (int(expected), source_relation_weight(relation, .9, metadata))


@pytest.mark.parametrize("explicit", [False, True])
def test_runtime_offline_and_consolidation_share_semantic_boundary(native, explicit):
    problem, solution = write_pair(native)
    store = native.memory_store
    metadata = {**INFERRED, **({"basis": "explicit"} if explicit else {})}
    edge = store.create_connection(problem, solution, "caused", weight=.9, metadata=metadata)
    expected = [edge] if explicit else []
    assert store.list_connections(node_id=solution) == expected
    assert store.list_connections_for_nodes([solution])[solution] == expected
    for source_only in (False, True):
        assert store.list_recall_connections([solution], limit=1, source_only=source_only)[solution] == expected
    for causal_mode in (False, True):
        assert (_traversal(edge, solution, causal_mode=causal_mode) is not None) is explicit
    assert store._has_graph_evidence(SCOPE) is explicit
    assert snapshot_evidence(store.connection)(SCOPE)[1] is explicit
    assert bool(miner.load_edges(store.connection)) is explicit
    for apply in (False, True):
        report = backfill_edges(store, apply=apply)
        assert report["edges_by_type"]["before"]["caused"] == int(explicit)
        assert report["typed_share"]["after"]["strict_typed"] == int(explicit)
    members = list(store.get_nodes([problem, solution]).values())
    selection = consolidation._current_sources(store, (node.id for node in members))
    assert not selection.excluded
    assert not consolidation._is_superseded(store, problem)
    assert not consolidation._is_superseded(store, solution)
    assert store.get_connection(edge.id) == edge


def test_history_is_filtered_before_graph_limits(native):
    problem, solution = write_pair(native)
    store = native.memory_store
    history = store.create_connection(problem, solution, "caused", weight=1, metadata=INFERRED)
    source = store.append_trace("Template parser must escape XML entities.", {"scope": SCOPE})
    genuine = store.create_connection(source.id, solution, "requires", weight=.5)
    for source_only in (False, True):
        assert store.list_recall_connections([solution], limit=1, source_only=source_only)[solution] == [genuine]
    assert store.get_connection(history.id) == history


@pytest.mark.parametrize("state", ["foreign", "decayed", "rejected"])
@pytest.mark.parametrize("depth", [1, "causal"])
def test_explicit_causality_preserves_scope_decay_and_eligibility(native, state, depth):
    problem, solution = write_pair(native, problem_scope="project:foreign" if state == "foreign" else SCOPE)
    store = native.memory_store
    native.tools["memory_connect"](problem, solution, "caused", metadata={"basis": "explicit", **INFERRED})
    if state == "decayed":
        store.soft_delete_node(problem, "expired evidence")
    elif state == "rejected":
        store.update_node(problem, context={"is_rejected_alternative": True})
    assert set(recall(native, depth)) == {solution}
    result = native.tools["memory_lookup"](node_id=solution, scope=SCOPE)["results"][0]
    assert not result["superseded"]
    assert not result["decayed"]


def test_offline_projection_matches_semantic_updates_and_ignores_history(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        a = store.append_trace("Copper pump fuse diagnosis.", {"scope": SCOPE})
        b = store.append_trace(f"Violet calendar reference to {a.id}.", {"scope": SCOPE})
        history = store.create_connection(a.id, b.id, "caused", metadata=INFERRED)
        technical = store.create_connection(b.id, a.id, "related", metadata={"basis": "implicit_recall_feedback"})
        plan = backfill_edges(store)
        applied = backfill_edges(store, apply=True)
        assert plan["totals"]["semantic_updates"] == applied["totals"]["semantic_updates"] == 1
        assert plan["edges_by_type"] == applied["edges_by_type"]
        expected_share = {key: value for key, value in plan["typed_share"]["after"].items() if key != "projected"}
        assert expected_share == applied["typed_share"]["after"]
        assert store.get_connection(history.id) == history
        assert store.get_connection(technical.id).metadata["kind"] == "content_reference"


def test_inferred_history_does_not_count_as_existing_semantic_connectivity(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        a = store.append_trace("Copper pump fuse diagnosis.", {"scope": SCOPE})
        b = store.append_trace(f"Violet calendar reference to {a.id}.", {"scope": SCOPE})
        history = store.create_connection(a.id, b.id, "caused", metadata=INFERRED)
        report = backfill_edges(store, apply=True)
        assert report["totals"]["new_rows"] == report["totals"]["net_new_connectivity"] == 1
        assert report["edges_by_type"]["after"]["caused"] == 0
        assert store.get_connection(history.id) == history
