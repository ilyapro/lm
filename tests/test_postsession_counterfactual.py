"""Tests for ``living_memory.postsession.counterfactual`` and its CLI.

Everything here runs against a real ``MemoryStore`` database built in
``tmp_path`` — real FTS, real chunk embeddings, real connections and anchor
edges — because the properties under test are exactly the ones a mock would
paper over. The live database is never opened, not even read-only.

The recorded ``recall_events`` in the fixture are produced by *running* the
retrieval code once against the pristine fixture and storing what it returned.
That makes the end-to-end expectation exact rather than guessed: if the harness
removes a cohort, puts it back losslessly and replays the same requests against
the same corpus, it must reproduce those deliveries node for node. Any gap is
then a defect in the harness, not a quirk of a hand-written ranking.

What is asserted:

* removal is referentially complete — nothing in ``nodes_fts``,
  ``node_chunk_embeddings``, ``connections``, ``query_anchor_edges`` or
  ``recall_events.feedback_trace_id`` survives it, and the ordering it uses is
  the one ``PRAGMA foreign_keys = ON`` forces;
* the round trip is lossless, and a lossy one is *detected* rather than
  silently scored;
* ``fresh`` re-insertion drops accumulated history but keeps ``created_at``,
  without which consumption ("delivered by a strictly later event") would
  silently measure zero;
* the replay writes no ``recall_events`` — it goes through
  ``retrieval_harness.run_item``, which pins ``log_event=False``;
* the source database is byte-identical afterwards and ``MemoryStore`` is only
  ever constructed on the disposable working copy;
* the grounded variant carries its own denominator, and the published report is
  aggregates only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from living_memory.config import MemoryConfig
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.postsession.counterfactual import (
    SELFCHECK_MAX_GAP,
    CoherenceError,
    CounterfactualConfig,
    assert_cohort_absent,
    assert_cohort_restored,
    assert_database_coherent,
    capture_cohort,
    check_privacy,
    foreign_key_violations,
    query_digest,
    reinsert_cohort,
    remove_cohort,
    run_counterfactual,
)
from living_memory.replay import open_readonly
from living_memory.retrieval import MemoryRecallService
from living_memory.retrieval_harness import working_copy
from living_memory.storage import MemoryStore

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "living_memory" / "postsession" / "counterfactual.py"
SCRIPT_PATH = REPO_ROOT / "scripts" / "counterfactual_consumption.py"

WINDOW = ("2026-08-12T00:00:00Z", "2026-08-15T00:00:00Z")
REPLAY_SINCE = "2026-08-12T00:00:00Z"  # = the window start: a COMPLETE replay
LATE_REPLAY_SINCE = "2026-08-15T00:00:00Z"  # skips event-quiet: a PARTIAL replay
AS_OF = "2026-08-19T00:00:00Z"
EVENT_AT = "2026-08-16T09:00:00Z"
EARLY_EVENT_AT = "2026-08-14T00:00:00Z"
COHORT_RULE = "nodes.created_at in [2026-08-12,2026-08-15) UTC; all levels; test fixture"

#: Cohort members. The first four are each the obvious answer to one recorded
#: query; ``cohort-orphan`` is never queried at all, so it stays unconsumed and
#: keeps the measured rate off 1.0 — a self-check that can only ever come out at
#: 100% tests nothing.
COHORT_CONTENT = {
    "cohort-writer": "restart the sqlite writer before backfilling chunk embeddings",
    "cohort-anchor": "query anchor edges migrate off the superseded node not the anchor",
    "cohort-wal": "a plain cp of the live database loses the write ahead log entirely",
    "cohort-quiet": "kubernetes ingress certificate rotation needs a webhook restart",
    "cohort-orphan": "the grafana dashboard panel legend truncates long series names",
}

#: Background corpus, created long before the window, so it is competition for
#: the replayed queries but never part of the cohort.
BACKGROUND_CONTENT = {
    "bg-latency": "recall latency percentiles are dominated by the vector scan stage",
    "bg-scope": "scope resolution prefers the narrower project scope over global",
    "bg-decay": "unreinforced query anchors decay out of the live set over time",
}

QUERIES = {
    "cohort-writer": "restart the sqlite writer before backfilling chunk embeddings",
    "cohort-anchor": "query anchor edges migrate off the superseded node",
    "cohort-wal": "a plain cp of the live database loses the write ahead log",
    # Recorded *before* LATE_REPLAY_SINCE, so a partial replay never gets the
    # chance to reproduce this consumption while a complete one does.
    "cohort-quiet": "kubernetes ingress certificate rotation webhook",
}

#: When each query's event was recorded.
EVENT_TIMES = {
    "cohort-writer": EVENT_AT,
    "cohort-anchor": EVENT_AT,
    "cohort-wal": EVENT_AT,
    "cohort-quiet": EARLY_EVENT_AT,
}

#: Repeats the target node's content verbatim, so its IDF containment is 1.0 and
#: the grounded variant must count it.
GROUNDING_TRACE = (
    "closure note: restart the sqlite writer before backfilling chunk embeddings, "
    "otherwise the second restart is skipped and the drain stalls"
)
#: Shares no token with anything, so it closes an event without grounding it.
SILENT_TRACE = "unrelated observation about a jenkins agent label typo"


def _node_context(agent: str) -> dict[str, Any]:
    return {"scope": "global", "agent": agent, "task": "fixture"}


def _set_created_at(connection: sqlite3.Connection, node_id: str, created_at: str) -> None:
    # `create_node` stamps created_at with *now*; the metric is defined on it, so
    # the fixture backdates it exactly as the harness does on re-insertion.
    connection.execute(
        "UPDATE nodes SET created_at = ?, updated_at = ?, timestamp = ? WHERE id = ?",
        (created_at, created_at, created_at, node_id),
    )


@pytest.fixture
def fixture_db(tmp_path: Path) -> Path:
    """A real store: nodes, chunks, edges, anchors, and recorded recall events."""

    path = tmp_path / "fixture.sqlite3"
    embedder = LocalEmbeddingModel()
    store = MemoryStore(MemoryConfig(db_path=path))
    try:
        for node_id, content in BACKGROUND_CONTENT.items():
            store.create_node(
                level="trace",
                content=content,
                context=_node_context("background"),
                embedding=embedder.embed(content),
                node_id=node_id,
            )
            _set_created_at(store.connection, node_id, "2026-07-01T00:00:00Z")
        for node_id, content in COHORT_CONTENT.items():
            store.create_node(
                level="trace",
                content=content,
                context=_node_context("fixture-agent"),
                embedding=embedder.embed(content),
                node_id=node_id,
                stats={"access_count": 4, "usefulness_score": 0.6},
            )
            _set_created_at(store.connection, node_id, "2026-08-13T12:00:00Z")
        for node_id, content in {"trace-ground": GROUNDING_TRACE, "trace-silent": SILENT_TRACE}.items():
            store.create_node(
                level="trace",
                content=content,
                context=_node_context("closer"),
                embedding=embedder.embed(content),
                node_id=node_id,
            )
            _set_created_at(store.connection, node_id, EVENT_AT)
        store.connection.commit()

        # Graph and query-anchor references, so removal has something real to
        # release in every table the goal enumerates.
        store.create_connection("cohort-writer", "bg-latency", "related", weight=0.8)
        store.create_connection("bg-scope", "cohort-anchor", "related", weight=0.5)
        anchor = store.insert_query_anchor(
            scope="global",
            fingerprint="fixture-anchor",
            query="restart the sqlite writer",
            embedding=embedder.embed("restart the sqlite writer"),
        )
        store.upsert_query_anchor_edge(anchor.id, "cohort-writer", weight=0.25)

        # Record what the retrieval code actually returns, so the end-to-end
        # expectation is exact rather than a guess about ranking.
        service = MemoryRecallService(store)
        recorded: list[tuple[str, str, list[dict[str, Any]]]] = []
        for node_id, query in QUERIES.items():
            results = service.memory_recall(
                query, scope="global", max_results=1, log_access=False, log_event=False
            )
            recorded.append(
                (
                    node_id,
                    query,
                    [
                        {
                            "node_id": result.node_id,
                            "rank": rank + 1,
                            "level": str(result.node.level),
                            "scope": str(result.node.scope),
                            "score": float(result.score),
                            "bm25_score": float(result.bm25_score),
                            "vector_score": float(result.vector_score),
                            "graph_score": float(result.graph_score),
                            "trigger_score": float(result.trigger_score),
                            "methods": [str(method) for method in result.methods],
                        }
                        for rank, result in enumerate(results)
                    ],
                )
            )

        closers = {
            "cohort-writer": "trace-ground",
            "cohort-anchor": "trace-silent",
            "cohort-wal": None,
            "cohort-quiet": None,
        }
        for index, (node_id, query, results) in enumerate(recorded):
            store.connection.execute(
                """
                INSERT INTO recall_events (
                    id, query, scope, requested_scope, resolved_scopes, ambient_context,
                    depth, max_results, results, agent, feedback_applied,
                    feedback_trace_id, created_at
                ) VALUES (?, ?, 'global', 'global', '["global"]', '{}', '1', 1, ?, 'tester', ?, ?, ?)
                """,
                (
                    f"event-{index}",
                    query,
                    json.dumps(results),
                    int(closers[node_id] is not None),
                    closers[node_id],
                    EVENT_TIMES[node_id],
                ),
            )
        # A second event for the *same* request as event-0 — it must dedupe into
        # one replayed request — and closed by a cohort node, so removal has a
        # `recall_events.feedback_trace_id` reference to release and restore.
        store.connection.execute(
            """
            INSERT INTO recall_events (
                id, query, scope, requested_scope, resolved_scopes, ambient_context,
                depth, max_results, results, agent, feedback_applied,
                feedback_trace_id, created_at
            ) VALUES ('event-dup', ?, 'global', 'global', '["global"]', '{}', '1', 1, ?,
                      'tester', 1, 'cohort-quiet', ?)
            """,
            (recorded[0][1], json.dumps(recorded[0][2]), EVENT_AT),
        )
        store.connection.commit()
    finally:
        store.close()
    return path


def _config(**overrides: Any) -> CounterfactualConfig:
    kwargs: dict[str, Any] = {
        "window": WINDOW,
        "replay_since": REPLAY_SINCE,
        "as_of": AS_OF,
        "cohort_rule": COHORT_RULE,
    }
    kwargs.update(overrides)
    return CounterfactualConfig(**kwargs)


def _cohort_ids() -> list[str]:
    return sorted(COHORT_CONTENT)


# ---------------------------------------------------------------------------
# Surgery
# ---------------------------------------------------------------------------


def test_removal_releases_every_referencing_table(fixture_db: Path) -> None:
    with working_copy(fixture_db) as working_db:
        store = MemoryStore(MemoryConfig(db_path=working_db))
        try:
            readonly = open_readonly(working_db)
            try:
                snapshots = capture_cohort(readonly, _cohort_ids())
                assert any(snap.chunks for snap in snapshots.values())
                assert any(snap.connections for snap in snapshots.values())
                assert any(snap.anchor_edges for snap in snapshots.values())
                assert any(snap.closed_events for snap in snapshots.values())

                surgery = remove_cohort(store, snapshots)
                assert surgery.chunk_rows > 0
                assert surgery.connection_rows == 2
                assert surgery.anchor_edge_rows == 1
                assert surgery.closed_event_refs == 1

                assert assert_cohort_absent(readonly, _cohort_ids()) == {
                    "nodes": 0,
                    "nodes_fts": 0,
                    "node_chunk_embeddings": 0,
                    "connections": 0,
                    "query_anchor_edges": 0,
                    "recall_events_feedback_trace_id": 0,
                }
                coherence = assert_database_coherent(
                    readonly, stage="after_remove", fts_check_connection=store.connection
                )
                assert coherence["nodes"] == coherence["nodes_fts"]
                assert coherence["fts_integrity_checked"] is True
            finally:
                readonly.close()
        finally:
            store.close()


def test_deleting_a_node_before_its_references_violates_the_foreign_key(
    fixture_db: Path,
) -> None:
    """The removal order is forced, not stylistic — this is what forces it."""

    with working_copy(fixture_db) as working_db:
        store = MemoryStore(MemoryConfig(db_path=working_db))
        try:
            with pytest.raises(sqlite3.IntegrityError):
                with store.connection:
                    store.connection.execute("DELETE FROM nodes WHERE id = 'cohort-writer'")
        finally:
            store.close()


def test_preexisting_dangling_references_are_tolerated_but_new_ones_are_not(
    fixture_db: Path,
) -> None:
    """The field database has 35 of these; the harness must add none of its own.

    ``PRAGMA foreign_keys=ON`` only polices new writes, so references orphaned
    before the constraint existed survive forever. A gate that demanded zero
    would be unusable on the database it exists to measure — and one that
    ignored the check entirely would miss a lossy removal.
    """

    with working_copy(fixture_db) as working_db:
        store = MemoryStore(MemoryConfig(db_path=working_db))
        try:
            readonly = open_readonly(working_db)
            try:
                with store.connection:
                    store.connection.execute("PRAGMA foreign_keys = OFF")
                    store.connection.execute(
                        "UPDATE recall_events SET feedback_trace_id = 'ghost' WHERE id = 'event-2'"
                    )
                    store.connection.execute("PRAGMA foreign_keys = ON")
                baseline = foreign_key_violations(readonly)
                assert len(baseline) == 1

                report = assert_database_coherent(
                    readonly, stage="before", baseline_violations=baseline
                )
                assert report["preexisting_foreign_key_violations"] == 1
                assert report["foreign_key_violations_introduced"] == 0

                with pytest.raises(CoherenceError, match="introduced 1 foreign-key"):
                    assert_database_coherent(readonly, stage="before", baseline_violations=set())
            finally:
                readonly.close()
        finally:
            store.close()


def test_faithful_round_trip_is_lossless(fixture_db: Path) -> None:
    with working_copy(fixture_db) as working_db:
        store = MemoryStore(MemoryConfig(db_path=working_db))
        try:
            readonly = open_readonly(working_db)
            try:
                snapshots = capture_cohort(readonly, _cohort_ids())
                surgery = remove_cohort(store, snapshots)
                reinsert_cohort(store, snapshots, mode="faithful", surgery=surgery)
                assert assert_cohort_restored(readonly, snapshots) == {
                    "nodes_compared": len(snapshots),
                    "mismatched": 0,
                }
                assert_database_coherent(
                    readonly, stage="after_reinsert", fts_check_connection=store.connection
                )
            finally:
                readonly.close()
        finally:
            store.close()


def test_a_lossy_round_trip_is_detected(fixture_db: Path) -> None:
    """A dropped chunk vector must fail loudly, not quietly deflate the score."""

    with working_copy(fixture_db) as working_db:
        store = MemoryStore(MemoryConfig(db_path=working_db))
        try:
            readonly = open_readonly(working_db)
            try:
                snapshots = capture_cohort(readonly, _cohort_ids())
                surgery = remove_cohort(store, snapshots)
                reinsert_cohort(store, snapshots, mode="faithful", surgery=surgery)
                store.delete_node_chunks("cohort-writer")
                with pytest.raises(CoherenceError, match="did not round-trip"):
                    assert_cohort_restored(readonly, snapshots)
            finally:
                readonly.close()
        finally:
            store.close()


def test_fresh_reinsertion_drops_history_but_keeps_created_at(fixture_db: Path) -> None:
    with working_copy(fixture_db) as working_db:
        store = MemoryStore(MemoryConfig(db_path=working_db))
        try:
            readonly = open_readonly(working_db)
            try:
                snapshots = capture_cohort(readonly, _cohort_ids())
                surgery = remove_cohort(store, snapshots)
                reinsert_cohort(
                    store,
                    snapshots,
                    mode="fresh",
                    baseline_state=None,
                    baseline_connection_ids=set(),
                    surgery=surgery,
                )
                row = readonly.execute(
                    "SELECT created_at, access_count, usefulness_score FROM nodes WHERE id = ?",
                    ("cohort-writer",),
                ).fetchone()
                assert row["created_at"] == "2026-08-13T12:00:00Z"
                assert int(row["access_count"]) == 0
                assert float(row["usefulness_score"]) == 0.0
                edges = readonly.execute(
                    "SELECT count(*) FROM connections WHERE source_id = ? OR target_id = ?",
                    ("cohort-writer", "cohort-writer"),
                ).fetchone()[0]
                assert int(edges) == 0
                # The event back-reference is a property of the event, not of the
                # node's history, so both modes restore it.
                closed = readonly.execute(
                    "SELECT count(*) FROM recall_events WHERE feedback_trace_id = 'cohort-quiet'"
                ).fetchone()[0]
                assert int(closed) == 1
                # Chunk vectors are content-derived and come back in both modes.
                chunks = readonly.execute(
                    "SELECT count(*) FROM node_chunk_embeddings WHERE node_id = ?",
                    ("cohort-writer",),
                ).fetchone()[0]
                assert int(chunks) > 0
            finally:
                readonly.close()
        finally:
            store.close()


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _report_cache() -> dict[str, Any]:
    return {}


def _run(db: Path, **overrides: Any) -> dict[str, Any]:
    return run_counterfactual(db, _config(**overrides))


def test_faithful_replay_reproduces_the_recorded_deliveries(fixture_db: Path) -> None:
    """Nothing changed but the round trip, so both arms must agree exactly."""

    report = _run(fixture_db)
    check = report["selfcheck"]
    retro = report["arms"]["retrospective"]
    counter = report["arms"]["counterfactual_faithful"]

    assert check["cohort_kind"] == "organic_holdout"
    assert check["cohort_nodes"] == len(COHORT_CONTENT)
    # Non-degenerate: some of the cohort is consumed and some is not, so a
    # harness that answered "all" or "none" could not match.
    assert 0 < retro["consumed_later"] < len(COHORT_CONTENT)
    assert check["gap"] == 0.0
    assert check["passed"] is True
    assert counter["rate"] == retro["rate"]
    assert check["agreement"]["jaccard"] == 1.0
    assert check["agreement"]["counterfactual_only"] == 0
    assert counter["round_trip"] == {"nodes_compared": len(COHORT_CONTENT), "mismatched": 0}
    assert counter["replay"]["fidelity"]["reproduced_share_of_recorded"] == 1.0


def test_a_complete_replay_leaves_no_recorded_consumption_unreplayed(
    fixture_db: Path,
) -> None:
    """Replaying from the window start covers every event that could consume.

    An event recorded before the earliest cohort node cannot consume one, so a
    replay that starts at the window start has been given the chance to
    reproduce *every* recorded consumption — which is exactly when the compared
    retrospective rate is the cohort's real one rather than a restricted slice.
    """

    report = _run(fixture_db)
    assert report["protocol"]["replay_is_complete"] is True
    assert report["selfcheck"]["replay_is_complete"] is True
    assert (
        report["arms"]["retrospective"]["consumed_later"]
        == report["arms"]["retrospective_all_events"]["consumed_later"]
    )
    assert report["selfcheck"]["retrospective_rate"] == (
        report["selfcheck"]["retrospective_rate_all_events"]
    )
    assert any("complete replay" in note for note in report["notes"])


def test_a_partial_replay_is_refused_unless_asked_for_and_then_disclosed(
    fixture_db: Path,
) -> None:
    """Silently skipping possible consumers would deflate the counterfactual."""

    with pytest.raises(ValueError, match="would skip events"):
        _run(fixture_db, replay_since=LATE_REPLAY_SINCE)

    report = _run(fixture_db, replay_since=LATE_REPLAY_SINCE, allow_partial_replay=True)
    assert report["protocol"]["replay_is_complete"] is False
    assert any(note.startswith("cutoff design") for note in report["notes"])
    # `cohort-quiet`'s only consumption was recorded before the late cutoff, so
    # the restricted arm sees one fewer — and, being restricted, both arms do.
    restricted = report["arms"]["retrospective"]["consumed_later"]
    everything = report["arms"]["retrospective_all_events"]["consumed_later"]
    assert everything == restricted + 1
    assert report["selfcheck"]["gap"] == 0.0


def test_grounded_variant_reports_its_own_denominator(fixture_db: Path) -> None:
    check = _run(fixture_db)["selfcheck"]
    grounded = check["grounded"]
    assert grounded["denominator_basis"] == "nodes_with_closed_consumer"
    # Not the cohort: only the nodes some closing trace could possibly have
    # grounded are gradeable at all, which is the measured trap this whole
    # denominator exists to avoid.
    assert 0 < grounded["denominator"] < check["cohort_nodes"]
    assert grounded["retrospective_denominator"] == grounded["counterfactual_denominator"]
    assert grounded["gap"] == 0.0
    # One closing trace repeats its node's content and one shares nothing with
    # it, so the grounded rate must sit strictly between "nobody" and "everyone".
    assert 0 < grounded["retrospective_consumed_later"] < grounded["retrospective_denominator"]
    assert grounded["retrospective_rate"] == grounded["counterfactual_rate"]


def test_fresh_arm_runs_on_its_own_working_copy(fixture_db: Path) -> None:
    report = _run(fixture_db, also_fresh_arm=True)
    assert "counterfactual_fresh" in report["arms"]
    fresh = report["arms"]["counterfactual_fresh"]
    assert fresh["surgery"]["reinsert_mode"] == "fresh"
    assert fresh["round_trip"]["mismatched"] is None
    # The self-check is still decided by the primary (faithful) arm.
    assert report["selfcheck"]["reinsert_mode"] == "faithful"


def test_alternate_replay_window_is_published_not_buried(fixture_db: Path) -> None:
    """How the answer moves with the replayed traffic belongs in the artifact.

    The gate's verdict depends on which recorded events it is handed. Running a
    second window and publishing it is what stops that dependence from being
    something only the person who ran it twice knows about.
    """

    report = _run(
        fixture_db,
        replay_since=LATE_REPLAY_SINCE,
        allow_partial_replay=True,
        diagnostic_replay_since=REPLAY_SINCE,
    )
    alternate = report["diagnostics"]["alternate_replay"]
    assert alternate is not None
    assert alternate["replay_since"] == REPLAY_SINCE
    assert alternate["replay_is_complete"] is True
    assert report["protocol"]["replay_is_complete"] is False
    # The wider window sees the consumption the narrow one could not.
    assert alternate["events_replayed"] > report["protocol"]["request_stats"]["events_replayed"]
    assert alternate["retrospective_consumed_later"] > (
        report["arms"]["retrospective"]["consumed_later"]
    )
    assert "gap" in alternate and "within_max_gap" in alternate
    assert "denominator" in str(alternate["grounded"]) or "gap" in alternate["grounded"]
    assert any("alternate_replay" in note for note in report["notes"])


def test_no_alternate_window_means_no_diagnostic_block(fixture_db: Path) -> None:
    assert _run(fixture_db)["diagnostics"]["alternate_replay"] is None


def test_a_capped_replay_caps_the_retrospective_arm_too(fixture_db: Path) -> None:
    """Both arms always score the same events, cap or no cap.

    A full recorded history compared against a truncated replay would make the
    counterfactual look empty for a reason that has nothing to do with
    retrieval — the exact way this measurement can lie.
    """

    report = _run(fixture_db, max_requests=1)
    replayed = report["protocol"]["request_stats"]["events_replayed"]
    assert 0 < replayed < report["protocol"]["request_stats"]["events"]
    assert report["arms"]["counterfactual_faithful"]["replay"]["events"] == replayed
    assert report["arms"]["retrospective"]["consumed_later"] < len(COHORT_CONTENT)
    assert (
        report["arms"]["retrospective"]["consumed_later"]
        == report["arms"]["counterfactual_faithful"]["consumed_later"]
    )
    assert report["selfcheck"]["gap"] == 0.0
    assert any("capped the replay" in note for note in report["notes"])


def test_empty_cohort_is_refused(fixture_db: Path) -> None:
    with pytest.raises(ValueError, match="matched no nodes"):
        _run(fixture_db, agent="nobody-wrote-this")


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_database_is_untouched(fixture_db: Path) -> None:
    """Byte-identical main file, and an empty WAL — no committed change leaks.

    Attaching to a WAL database *at all*, even through a ``mode=ro`` URI,
    materialises the shared-memory index and a zero-length ``-wal``: that is
    SQLite's coordination, not a write, and it is the only trace the run may
    leave. A non-empty ``-wal`` would mean frames were committed against the
    source, which is precisely the failure this harness exists to avoid.
    """

    before = _digest(fixture_db)
    _run(fixture_db)
    assert _digest(fixture_db) == before
    wal = fixture_db.with_name(fixture_db.name + "-wal")
    assert not wal.exists() or wal.stat().st_size == 0
    assert not fixture_db.with_name(fixture_db.name + "-journal").exists()


def test_replay_logs_no_recall_events(fixture_db: Path) -> None:
    """`run_item` pins log_event=False; a replay that logged would feed itself."""

    with sqlite3.connect(fixture_db) as connection:
        before = connection.execute("SELECT count(*) FROM recall_events").fetchone()[0]
    report = _run(fixture_db)
    with sqlite3.connect(fixture_db) as connection:
        after = connection.execute("SELECT count(*) FROM recall_events").fetchone()[0]
    assert after == before
    assert report["arms"]["counterfactual_faithful"]["replay"]["requests"] == len(QUERIES)


def test_migrating_store_is_only_ever_pointed_at_the_working_copy() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    constructions = re.findall(r"MemoryStore\(([^)]*)\)", source)
    assert constructions, "expected the module to construct a store on the working copy"
    for arguments in constructions:
        assert "working_db" in arguments, arguments
    assert "working_copy(" in source
    # The acceptance contract's literal guard, kept honest from inside the suite.
    assert not re.search(r"MemoryStore\(.*(global\.sqlite3|live_db|args\.db)", source)


def test_report_is_aggregates_only(fixture_db: Path) -> None:
    report = _run(fixture_db)
    check_privacy(report)
    payload = json.dumps(report)
    for content in (*COHORT_CONTENT.values(), *QUERIES.values(), GROUNDING_TRACE):
        assert content not in payload
    for node_id in COHORT_CONTENT:
        assert node_id not in payload
    assert report["protocol"]["request_stats"]["requests"] == len(QUERIES)


def test_query_digest_is_stable_and_short() -> None:
    assert query_digest("restart the writer") == query_digest("restart the writer")
    assert query_digest("a") != query_digest("b")
    assert len(query_digest("a")) == 16


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_cli() -> Any:
    spec = importlib.util.spec_from_file_location("counterfactual_cli", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["counterfactual_cli"] = module
    spec.loader.exec_module(module)
    return module


def test_cli_writes_the_selfcheck_contract(fixture_db: Path, tmp_path: Path) -> None:
    cli = _load_cli()
    out = tmp_path / "counterfactual-selfcheck.json"
    code = cli.main(
        [
            "--db",
            str(fixture_db),
            "--as-of",
            AS_OF,
            "--window",
            "2026-08-12..2026-08-15",
            "--replay-since",
            REPLAY_SINCE,
            "--cohort-rule",
            COHORT_RULE,
            "--out",
            str(out),
            "--quiet",
        ]
    )
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    check = report["selfcheck"]
    # Exactly the acceptance contract's assertion.
    assert check["cohort_kind"] == "organic_holdout"
    assert abs(check["counterfactual_rate"] - check["retrospective_rate"]) <= SELFCHECK_MAX_GAP
    assert "grounded" in check and "denominator" in check["grounded"]
    assert check["cohort_rule"] == COHORT_RULE


def test_cli_rejects_an_unpublishable_cohort_rule(fixture_db: Path) -> None:
    cli = _load_cli()
    with pytest.raises(SystemExit, match="printable ASCII"):
        cli.main(
            [
                "--db",
                str(fixture_db),
                "--as-of",
                AS_OF,
                "--window",
                "2026-08-12..2026-08-15",
                "--replay-since",
                REPLAY_SINCE,
                "--cohort-rule",
                "x" * 240,
                "--quiet",
            ]
        )


def test_cli_rejects_a_missing_database(tmp_path: Path) -> None:
    cli = _load_cli()
    with pytest.raises(SystemExit, match="database not found"):
        cli.main(
            [
                "--db",
                str(tmp_path / "nope.sqlite3"),
                "--as-of",
                AS_OF,
                "--window",
                "2026-08-12..2026-08-15",
                "--replay-since",
                REPLAY_SINCE,
                "--cohort-rule",
                COHORT_RULE,
                "--quiet",
            ]
        )
