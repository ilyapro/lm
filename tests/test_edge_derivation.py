"""Write-path typed-edge derivation (edge_derivation.py go-rules).

Synthetic fixtures in temp stores exercise reference and provenance rules,
shared caps/exclusions, re-derivation idempotence, memory_remember wiring,
consolidation annotation and traversal factors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from living_memory.consolidation import memory_consolidate
from living_memory.decay import memory_forget
from living_memory.edge_derivation import (
    CONTENT_REFERENCE_KIND,
    DERIVED_FROM_ANNOTATION,
    DERIVED_FROM_KIND,
    SAME_COMMIT_BASIS,
    SHARED_FILES_BASIS,
    derive_edges_for_new_trace,
    rule_r5a_derived_from_annotations,
)
from living_memory.models import Connection, Node
from living_memory.retrieval import _traversal, memory_recall
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore

SCOPE = "project:edge-derivation-tests"
# Syntactically valid Crockford ULID that no store ever issued.
UNKNOWN_ULID = "01" + "Z" * 24


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.sqlite3")


def _remember(store: MemoryStore, content: str, **context: Any) -> Node:
    context.setdefault("scope", SCOPE)
    return store.append_trace(content, context)


def _related(store: MemoryStore, source_id: str, target_id: str) -> list[Connection]:
    return store.list_connections(
        source_id=source_id, target_id=target_id, relation_type="related"
    )


@pytest.mark.parametrize("content,context", [
    ("CORRECTION to trace {id}: use the other lock file.", {}),
    ("This is not a correction to {id}; the source remains valid.", {}),
    ('Quoted example: "supersedes {id}".', {}),
    ("Correction: change the retry delay; source {id} stays valid.", {}),
    ("Source {id} stays valid.", {"type": "memory_correction"}),
    ("Source {id} stays valid.", {"type": "correction"}),
    ("Source {id} stays valid.", {"type": "decomposition_correction"}),
    ("Source {id} stays valid.", {"lesson_kind": "correction"}),
    ("Source {id} stays valid.", {}),
])
def test_content_and_labels_only_derive_references(tmp_path, content, context):
    with _store(tmp_path) as store:
        target = _remember(store, "The runner lock file lives in /tmp/runner.lock")
        note = _remember(store, content.format(id=target.id), **context)

        edges = derive_edges_for_new_trace(store, note)

        assert len(edges) == 1
        assert edges[0].type == "related"
        assert edges[0].metadata == {"rule": "R6", "kind": CONTENT_REFERENCE_KIND}
        assert edges[0].source_id == note.id and edges[0].target_id == target.id
        assert store.connection.execute(
            "SELECT count(*) FROM connections WHERE type = 'supersedes'"
        ).fetchone()[0] == 0
        assert store.get_node(target.id) == target


def test_derivation_never_links_to_decayed_nodes(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        stale = _remember(store, "Old fact that has since decayed")
        memory_forget(store, stale.id, "test decay")
        correction = _remember(
            store, f"CORRECTION to trace {stale.id}: the old fact is obsolete"
        )

        assert derive_edges_for_new_trace(store, correction) == []
        assert store.list_connections(source_id=correction.id) == []


# --- R4a same commit -------------------------------------------------------------


def test_r4a_links_same_commit_prefix_case_insensitively(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        impl = _remember(store, "Implementation note for the change", commit="ABC1234def999")
        short = _remember(store, "Note with a too-short commit", commit="abc12")
        verification = _remember(
            store, "Verification failure on the same change", commit="abc1234555"
        )

        derive_edges_for_new_trace(store, verification)

        source_id, target_id = sorted((impl.id, verification.id))
        edges = _related(store, source_id, target_id)
        assert len(edges) == 1
        assert edges[0].weight == 0.9
        assert edges[0].metadata == {
            "rule": "R4a",
            "basis": SAME_COMMIT_BASIS,
            "commit": "abc1234",
        }
        assert store.list_connections(node_id=short.id) == []


def test_r4a_group_larger_than_ten_emits_nothing(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        for index in range(10):
            _remember(store, f"Change note {index} on one commit", commit="feed1234")
        eleventh = _remember(store, "Eleventh note on the same commit", commit="feed1234")

        derive_edges_for_new_trace(store, eleventh)

        assert store.list_connections(node_id=eleventh.id) == []


def test_r4a_skips_both_rejected_alternative_pairs(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        existing_ra = _remember(
            store,
            "Rejected approach A for the change",
            commit="beef1234",
            is_rejected_alternative=True,
        )
        normal = _remember(store, "The chosen implementation", commit="beef1234")
        new_ra = _remember(
            store,
            "Rejected approach B for the change",
            commit="beef1234",
            is_rejected_alternative=True,
        )

        derive_edges_for_new_trace(store, new_ra)

        assert _related(store, *sorted((new_ra.id, existing_ra.id))) == []
        ra_to_normal = _related(store, *sorted((new_ra.id, normal.id)))
        assert len(ra_to_normal) == 1
        assert ra_to_normal[0].metadata["basis"] == SAME_COMMIT_BASIS


# --- R4b shared files ------------------------------------------------------------


def test_r4b_weight_grows_with_shared_files(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        one_shared = _remember(store, "Parser gotcha", files=["src/parser.py"])
        two_shared = _remember(
            store, "Parser+lexer regression", files=["src/parser.py", "src/lexer.py"]
        )
        new = _remember(
            store, "Fix touching parser and lexer", files=["src/parser.py", "src/lexer.py"]
        )

        derive_edges_for_new_trace(store, new)

        single = _related(store, *sorted((one_shared.id, new.id)))
        assert len(single) == 1
        assert single[0].weight == pytest.approx(0.5 + 0.15 * 1)
        assert single[0].metadata == {
            "rule": "R4b",
            "basis": SHARED_FILES_BASIS,
            "files": ["src/parser.py"],
        }

        double = _related(store, *sorted((two_shared.id, new.id)))
        assert len(double) == 1
        assert double[0].weight == pytest.approx(0.5 + 0.15 * 2)
        assert sorted(double[0].metadata["files"]) == ["src/lexer.py", "src/parser.py"]


def test_r4b_weight_caps_at_095(tmp_path: Path) -> None:
    files = [f"src/mod_{index}.py" for index in range(4)]
    with _store(tmp_path) as store:
        sibling = _remember(store, "Four-file refactor note", files=files)
        new = _remember(store, "Follow-up across the same four files", files=files)

        derive_edges_for_new_trace(store, new)

        edges = _related(store, *sorted((sibling.id, new.id)))
        assert len(edges) == 1
        assert edges[0].weight == pytest.approx(0.95)  # min(0.95, 0.5 + 0.15*4)


def test_r4b_skips_over_fanout_files(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        for index in range(21):
            _remember(store, f"Hub file note {index}", files=["src/hub.py"])
        new = _remember(store, "Yet another hub note", files=["src/hub.py"])

        derive_edges_for_new_trace(store, new)

        assert store.list_connections(node_id=new.id) == []


def test_r4b_skips_decayed_citers_and_both_rejected_pairs(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        decayed = _remember(store, "Decayed note citing the file", files=["src/x.py"])
        memory_forget(store, decayed.id, "test decay")
        existing_ra = _remember(
            store,
            "Rejected approach citing the file",
            files=["src/x.py"],
            is_rejected_alternative=True,
        )
        new_ra = _remember(
            store,
            "Another rejected approach citing the file",
            files=["src/x.py"],
            is_rejected_alternative=True,
        )

        derive_edges_for_new_trace(store, new_ra)
        assert store.list_connections(node_id=new_ra.id) == []

        normal = _remember(store, "Accepted note citing the file", files=["src/x.py"])
        derive_edges_for_new_trace(store, normal)

        assert store.list_connections(node_id=decayed.id) == []
        assert len(_related(store, *sorted((normal.id, existing_ra.id)))) == 1
        assert len(_related(store, *sorted((normal.id, new_ra.id)))) == 1


# --- R5a derived-from annotation ---------------------------------------------------


def test_r5a_selector_targets_only_unannotated_schema_source_edges() -> None:
    schema = Node(
        id="SCHEMA-1",
        level="schema",
        content="Procedure: deploy rollback",
        source_traces=["TRACE-1", "TRACE-2"],
    )
    concept = Node(
        id="CONCEPT-1",
        level="concept",
        content="Cluster summary",
        source_traces=["TRACE-1"],
    )

    def connection(cid: str, source: str, target: str, ctype: str = "related", **metadata: Any) -> Connection:
        return Connection(
            id=cid, source_id=source, target_id=target, type=ctype, weight=0.6, metadata=metadata
        )

    eligible = connection("c-el", schema.id, "TRACE-1", basis="procedural")
    already = connection(
        "c-done", schema.id, "TRACE-2", basis="procedural", **DERIVED_FROM_ANNOTATION
    )
    non_source = connection("c-ns", schema.id, "TRACE-9", basis="procedural")
    wrong_type = connection("c-wt", schema.id, "TRACE-1", ctype="supersedes")
    incoming = connection("c-in", "TRACE-1", schema.id, basis="procedural")
    edges = [eligible, already, non_source, wrong_type, incoming]

    assert rule_r5a_derived_from_annotations(schema, edges) == [eligible]
    # Concepts are excluded outright (mega-cluster provenance scored 0/16).
    concept_edge = connection("c-co", concept.id, "TRACE-1", basis="cluster")
    assert rule_r5a_derived_from_annotations(concept, [concept_edge]) == []


def test_consolidation_annotates_existing_schema_edges_without_new_rows(
    tmp_path: Path,
) -> None:
    with _store(tmp_path) as store:
        for index, step in enumerate(("drain", "flip", "verify")):
            _remember(
                store,
                f"Deploy rollback step {index + 1}: {step}",
                procedure_id="deploy_rollback",
                step_order=index + 1,
            )
        memory_consolidate(store, scope=SCOPE)

        schemas = store.list_nodes(level="schema", scope=SCOPE)
        assert len(schemas) == 1
        schema = schemas[0]
        edges = [
            edge
            for edge in store.list_connections(source_id=schema.id, relation_type="related")
            if edge.target_id in set(schema.source_traces)
        ]
        assert len(edges) == len(schema.source_traces) == 3
        for edge in edges:
            assert edge.metadata["kind"] == DERIVED_FROM_KIND
            assert edge.metadata["rule"] == "R5a"
            assert edge.metadata["basis"] == "procedural"

        # Simulate live-DB edges created before R5a existed, then re-run
        # consolidation: the metadata merge annotates them in place.
        for edge in edges:
            stripped = {
                key: value
                for key, value in edge.metadata.items()
                if key not in DERIVED_FROM_ANNOTATION
            }
            store.update_connection(edge.id, metadata=stripped)

        memory_consolidate(store, scope=SCOPE)

        re_edges = store.list_connections(source_id=schema.id, relation_type="related")
        assert sorted(edge.id for edge in re_edges) == sorted(edge.id for edge in edges)
        for edge in re_edges:
            assert edge.metadata["kind"] == DERIVED_FROM_KIND
            assert edge.metadata["rule"] == "R5a"
            assert edge.metadata["basis"] == "procedural"


# --- R6 content reference ----------------------------------------------------------


def test_r6_reference_is_directed_and_skips_unknown_ulids(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        referee = _remember(store, "Baseline replay metrics recorded")
        referrer = _remember(
            store,
            f"Re-ran the replay gate; builds on {referee.id} and mentions the"
            f" never-issued id {UNKNOWN_ULID}.",
        )

        derive_edges_for_new_trace(store, referrer)

        edges = store.list_connections(source_id=referrer.id, relation_type="related")
        assert [edge.target_id for edge in edges] == [referee.id]
        assert edges[0].weight == 0.9
        assert edges[0].metadata == {"rule": "R6", "kind": CONTENT_REFERENCE_KIND}
        # Stored referrer -> referee only; no reverse row.
        assert store.list_connections(source_id=referee.id, target_id=referrer.id) == []


# --- idempotence and upsert semantics ----------------------------------------------


def _connection_rows(store: MemoryStore) -> list[tuple[Any, ...]]:
    rows = store.connection.execute(
        "SELECT source_id, target_id, type, weight, metadata FROM connections"
        " ORDER BY source_id, target_id, type"
    ).fetchall()
    return [tuple(row) for row in rows]


def test_re_derivation_is_idempotent_and_multi_rule_pairs_merge(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        other = _remember(
            store, "First note on the change", commit="cafe1234", files=["src/a.py"]
        )
        new = _remember(
            store, "Second note on the change", commit="cafe1234", files=["src/a.py"]
        )

        derive_edges_for_new_trace(store, new)
        rows_first = _connection_rows(store)
        derive_edges_for_new_trace(store, new)
        assert _connection_rows(store) == rows_first

        # R4a and R4b hit the same (source, target, 'related') row; the upsert
        # merges metadata (derivation order: R4a then R4b) instead of adding rows.
        edges = _related(store, *sorted((other.id, new.id)))
        assert len(edges) == 1
        assert edges[0].weight == 0.9
        assert edges[0].metadata["basis"] == SHARED_FILES_BASIS
        assert edges[0].metadata["commit"] == "cafe123"  # R4a's 7-char prefix survives


def test_derived_evidence_replaces_recall_history_without_inheriting_weight(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        cited = _remember(store, "Earlier note about the module", files=["src/z.py"])
        new = _remember(store, "New note about the module", files=["src/z.py"])
        source_id, target_id = sorted((cited.id, new.id))
        store.create_connection(
            source_id,
            target_id,
            "related",
            weight=0.97,
            metadata={"basis": "implicit_recall_feedback", "recall_id": "evt-1"},
        )

        derive_edges_for_new_trace(store, new)

        edges = _related(store, source_id, target_id)
        assert len(edges) == 1
        assert edges[0].weight == 0.65
        assert edges[0].metadata["basis"] == SHARED_FILES_BASIS  # new keys win
        assert "recall_id" not in edges[0].metadata


# --- memory_remember wiring ---------------------------------------------------------


class _FakeMCP:
    def __init__(self, name: str, instructions: str | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, Any] = {}

    def tool(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.tools[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        if func is None:
            return decorate
        return decorate(func)

    def resource(self, uri: str, **_kwargs: Any) -> Any:
        def decorate(func: Any) -> Any:
            return func

        return decorate

    def prompt(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            return inner

        if func is None:
            return decorate
        return decorate(func)


def test_memory_remember_derives_typed_edges_end_to_end(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    mcp = create_mcp_server(db_path, mcp_factory=_FakeMCP)
    remember = mcp.tools["memory_remember"]

    first = remember(
        "Root cause: worker pool exhausted by orphaned jobs",
        {
            "scope": SCOPE,
            "task": "queue-stall",
            "type": "root_cause",
            "timestamp": "2026-07-08T10:00:00Z",
        },
    )
    root_cause_id = first["node"]["id"]
    second = remember(
        f"Fix: reap orphaned jobs on boot; builds on {root_cause_id}.",
        {
            "scope": SCOPE,
            "task": "queue-stall",
            "type": "working_fix",
            "timestamp": "2026-07-08T10:05:00Z",
            "commit": "0ddba11c0ffee",
            "files": ["src/queue.py"],
        },
    )
    fix_id = second["node"]["id"]

    with MemoryStore(db_path) as store:
        caused = store.list_connections(
            source_id=root_cause_id, target_id=fix_id, relation_type="caused"
        )
        assert caused == []

        references = _related(store, fix_id, root_cause_id)
        assert len(references) == 1
        assert references[0].metadata == {"rule": "R6", "kind": CONTENT_REFERENCE_KIND}


# --- traversal factors ---------------------------------------------------------------


def _edge(source: str, target: str, ctype: str, weight: float, metadata: dict[str, Any]) -> Connection:
    return Connection(
        id="conn-test",
        source_id=source,
        target_id=target,
        type=ctype,
        weight=weight,
        metadata=metadata,
    )


def test_traversal_factors_for_derived_from_and_content_reference() -> None:
    schema_edge = _edge(
        "SCHEMA", "TRACE", "related", 1.0, {"basis": "procedural", **DERIVED_FROM_ANNOTATION}
    )
    neighbor, score = _traversal(schema_edge, "SCHEMA", causal_mode=False)
    assert (neighbor, score) == ("TRACE", pytest.approx(0.5))
    neighbor, score = _traversal(schema_edge, "TRACE", causal_mode=False)
    assert (neighbor, score) == ("SCHEMA", pytest.approx(0.95))

    reference = _edge(
        "REFERRER", "REFEREE", "related", 0.9, {"rule": "R6", "kind": CONTENT_REFERENCE_KIND}
    )
    neighbor, score = _traversal(reference, "REFERRER", causal_mode=False)
    assert (neighbor, score) == ("REFEREE", pytest.approx(0.9 * 0.75))
    neighbor, score = _traversal(reference, "REFEREE", causal_mode=False)
    assert (neighbor, score) == ("REFERRER", pytest.approx(0.9 * 0.9))


def test_traversal_basis_rules_stay_generic_related() -> None:
    for metadata in (
        {"rule": "R4a", "basis": SAME_COMMIT_BASIS, "commit": "abc1234"},
        {"rule": "R4b", "basis": SHARED_FILES_BASIS, "files": ["src/a.py"]},
    ):
        edge = _edge("A", "B", "related", 0.9, metadata)
        for current, expected_neighbor in (("A", "B"), ("B", "A")):
            neighbor, score = _traversal(edge, current, causal_mode=False)
            assert neighbor == expected_neighbor
            assert score == pytest.approx(0.9 * 0.65)


def test_traversal_new_kinds_stay_excluded_from_causal_and_decision_modes() -> None:
    for metadata in (dict(DERIVED_FROM_ANNOTATION), {"kind": CONTENT_REFERENCE_KIND}):
        edge = _edge("A", "B", "related", 1.0, metadata)
        assert _traversal(edge, "A", causal_mode=True) is None
        assert _traversal(edge, "B", causal_mode=True) is None
        assert _traversal(edge, "A", causal_mode=False, decision_mode=True) is None


def test_traversal_factors_for_explicit_typed_edges() -> None:
    caused = _edge(
        "RC",
        "FIX",
        "caused",
        0.9,
        {},
    )
    expectations = [
        ("RC", False, "FIX", 0.9 * 0.75),
        ("FIX", False, "RC", 0.9 * 0.85),
        ("RC", True, "FIX", 0.9 * 0.25),
        ("FIX", True, "RC", 0.9 * 1.0),
    ]
    for current, causal, expected_neighbor, expected_score in expectations:
        neighbor, score = _traversal(caused, current, causal_mode=causal)
        assert neighbor == expected_neighbor
        assert score == pytest.approx(expected_score)

    supersedes = _edge(
        "CORRECTION",
        "STALE",
        "supersedes",
        0.9,
        {"by": "teacher", "correction": "Use the current lock path."},
    )
    neighbor, score = _traversal(supersedes, "CORRECTION", causal_mode=False)
    assert (neighbor, score) == ("STALE", pytest.approx(0.9 * 0.35))
    neighbor, score = _traversal(supersedes, "STALE", causal_mode=False)
    assert (neighbor, score) == ("CORRECTION", pytest.approx(0.9 * 1.15))


def test_causal_recall_traverses_explicit_caused_edge(tmp_path: Path) -> None:
    """Explicit causes support causal recall independently of matching text."""

    with _store(tmp_path) as store:
        root_cause = _remember(
            store,
            "Diagnosis: an orphaned lock artifact was left behind by the crashed daemon",
            task="fix-runner",
            type="root_cause",
            timestamp="2026-07-08T09:55:00Z",
        )
        fix = _remember(
            store,
            "Fix: remove the stale pid file during runner startup",
            task="fix-runner",
            type="working_fix",
            timestamp="2026-07-08T10:00:00Z",
        )
        store.create_connection(root_cause.id, fix.id, "caused", weight=.9)

        results = memory_recall(
            store,
            "why does the runner fail to start",
            scope=SCOPE,
            depth="causal",
            max_results=5,
        )

        by_id = {result.node.id: result for result in results}
        assert root_cause.id in by_id
        # Reached by traversing the explicit caused edge backwards from the fix,
        # not by matching the (lexically disjoint) query directly.
        assert by_id[root_cause.id].graph_score > 0
