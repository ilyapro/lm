"""Focused fixture-DB tests for the health audit metric engine.

Each test exercises a single audit contract from
``artifacts/health_audit_contract.md`` so that a regression in one metric
fails an obviously-named test. The fixture is small and explicit, with
counts deliberately distinct from the live-baseline numbers in
``artifacts/baseline.md`` (the parent ``_critique`` warned that the audit
must not look like a hardcoded answer key).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from living_memory.health_audit import (
    DEFAULT_LATENCY_QUERY,
    DEFAULT_TOP_CANDIDATE_SCOPES,
    ReadOnlyAuditStore,
    compute_health_audit,
    open_read_only_store,
    scope_hygiene_metrics,
)
from living_memory.resources import (
    feedback_closure_metrics,
    memory_health,
    recall_events_summary,
)
from living_memory.storage import MemoryStore


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------

_FIXTURE_TRACES: tuple[tuple[str, str, int], ...] = (
    # (scope, content, access_count)
    ("project:alpha", "alpha-shared-text", 3),
    ("project:alpha", "alpha-shared-text", 0),
    ("project:alpha", "alpha-shared-text", 0),
    ("project:alpha", "alpha-unique-text", 2),
    ("project:beta", "beta-task-text", 5),
    ("project:beta", "beta-task-text", 1),
    ("project:beta", "beta-other-text", 0),
    ("rise/foo", "leakage-rise-foo", 0),
    ("project:breakthrough", "leakage-breakthrough", 0),
    ("ocpa-generative-action-substrate-v1", "leakage-ocpa", 4),
)
# Distinct contents: 7 → 10 active traces, duplicate_excess = 3, density = 0.3.
# Active concept (+1) and decayed trace (excluded from active counts) added below.


def _build_fixture(path: Path) -> MemoryStore:
    """Construct a small, deterministic legacy-shaped DB fixture.

    The health audit must be able to inspect historical DBs that already
    contain duplicate active traces and raw leakage scopes. Insert those rows
    directly so the fixture measures audit behavior instead of the current
    write-path guards that now canonicalize scopes and supersede duplicates.
    """

    store = MemoryStore(path)
    timestamp = "2026-05-22T00:00:00Z"

    def insert_node(
        node_id: str,
        *,
        level: str,
        content: str,
        scope: str,
        access_count: int = 0,
        decayed: int = 0,
        decay_reason: str | None = None,
    ) -> None:
        store.connection.execute(
            """
            INSERT INTO nodes (
                id, level, content, content_fingerprint, embedding, scope,
                agent, task, context, timestamp, decayed, decay_reason,
                access_count, last_accessed, usefulness_score, confidence,
                unique_agents, temporal_hint, source_traces, corrections,
                provenance, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, NULL, ?, ?, NULL, ?, ?, ?, ?, ?, ?, 0.0, 0.5,
                    1, NULL, '[]', '[]', '{}', ?, ?)
            """,
            (
                node_id,
                level,
                content,
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
                scope,
                "fixture-agent",
                json.dumps(
                    {"scope": scope, "agent": "fixture-agent", "timestamp": timestamp},
                    sort_keys=True,
                ),
                timestamp,
                decayed,
                decay_reason,
                access_count,
                timestamp if access_count > 0 else None,
                timestamp,
                timestamp,
            ),
        )

    with store.connection:
        for index, (scope, content, access_count) in enumerate(_FIXTURE_TRACES, start=1):
            insert_node(
                f"fixture-trace-{index:02d}",
                level="trace",
                content=content,
                scope=scope,
                access_count=access_count,
            )

        insert_node(
            "fixture-concept-01",
            level="concept",
            content="alpha-concept-text",
            scope="project:alpha",
        )
        insert_node(
            "fixture-decayed-trace-01",
            level="trace",
            content="decayed-trace-text",
            scope="project:alpha",
            decayed=1,
            decay_reason="fixture-decay",
        )

        # 8 recall events spanning project + leakage scopes; 2 are marked
        # feedback_applied.
        for index, (query, scope) in enumerate(
            (
                ("q1", "project:alpha"),
                ("q2", "project:alpha"),
                ("q3", "project:alpha"),
                ("q4", "project:beta"),
                ("q5", "project:beta"),
                ("q6", "rise/foo"),
                ("q7", "project:rise/extra"),
                ("q8", "global"),
            ),
            start=1,
        ):
            feedback_applied = 1 if index in {3, 4} else 0
            store.connection.execute(
                """
                INSERT INTO recall_events (
                    id, query, scope, requested_scope, resolved_scopes,
                    ambient_context, depth, max_results, results, agent, task,
                    session_id, feedback_applied, feedback_trace_id,
                    feedback_applied_at, created_at
                )
                VALUES (?, ?, ?, ?, ?, '{}', NULL, 10, '[]', NULL, NULL, NULL,
                        ?, ?, ?, ?)
                """,
                (
                    f"fixture-event-{index:02d}",
                    query,
                    scope,
                    scope,
                    json.dumps([scope]),
                    feedback_applied,
                    "fixture-trace-01" if feedback_applied else None,
                    timestamp if feedback_applied else None,
                    timestamp,
                ),
            )

    return store


# ---------------------------------------------------------------------------
# Expected values, derived from the fixture (not hardcoded answer-key lookups)
# ---------------------------------------------------------------------------

_EXPECTED_ACTIVE_TRACES = 10
_EXPECTED_DISTINCT_CONTENT = 7
_EXPECTED_DUPLICATE_EXCESS = 3
_EXPECTED_DUPLICATE_DENSITY = 0.3

_EXPECTED_RECALL_EVENTS = 8
_EXPECTED_FEEDBACK_APPLIED = 2
_EXPECTED_FEEDBACK_RATIO = 0.25

_EXPECTED_ACTIVE_NODES = 11
_EXPECTED_NEVER_ACCESSED_NODES = 6
_EXPECTED_NEVER_ACCESSED_TRACES = 5
_EXPECTED_NEVER_ACCESSED_TRACE_RATIO = 0.5
_EXPECTED_NEVER_ACCESSED_NODE_RATIO = _EXPECTED_NEVER_ACCESSED_NODES / _EXPECTED_ACTIVE_NODES

_EXPECTED_CANDIDATE_SCOPES = 4
_EXPECTED_ACTIVE_CANDIDATE_NODES = 3
_EXPECTED_ACTIVE_CANDIDATE_TRACES = 3
_EXPECTED_NEVER_ACCESSED_CANDIDATE_TRACES = 2
_EXPECTED_CANDIDATE_ZERO_RECALL = 2
_EXPECTED_CANDIDATE_SCOPE_SET = frozenset(
    {
        "rise/foo",
        "project:breakthrough",
        "ocpa-generative-action-substrate-v1",
        "project:rise/extra",
    }
)

# Live-baseline values (from artifacts/baseline.md) used as anti-lookup guards.
_LIVE_DUPLICATE_DENSITY = 0.129648
_LIVE_FEEDBACK_RATIO = 0.207274
_LIVE_NEVER_ACCESSED_NODE_RATIO = 0.723479
_LIVE_NEVER_ACCESSED_TRACE_RATIO = 0.731955
_LIVE_CANDIDATE_SCOPES = 54
_LIVE_ACTIVE_CANDIDATE_TRACES = 111
_LIVE_CANDIDATE_ZERO_RECALL = 36
_LIVE_DB_SIZE_BYTES = 100339712
_LIVE_CLOSURE_RATIO = 0.16
_LIVE_EXPLICIT_CLASS_CLOSURE = 0.79
_LIVE_IDENTITYLESS_CLOSURE = 0.12


def _strip_time_varying_fields(report: dict[str, Any]) -> dict[str, Any]:
    """Drop wall-clock fields that legitimately vary between repeated calls."""

    stripped = json.loads(json.dumps(report))
    stripped.pop("generated_at", None)
    latency = stripped.get("latency")
    if latency:
        latency.pop("warmup_ms", None)
        latency.pop("samples_ms", None)
        latency.pop("summary", None)
    return stripped


def _md5_of_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Duplicate density
# ---------------------------------------------------------------------------


def test_duplicate_density_global_counts_match_fixture(tmp_path: Path) -> None:
    with _build_fixture(tmp_path / "f.sqlite3") as store:
        dedup = compute_health_audit(store)["dedup"]

    assert dedup["total_traces"] == _EXPECTED_ACTIVE_TRACES
    assert dedup["distinct_contents"] == _EXPECTED_DISTINCT_CONTENT
    assert dedup["duplicate_excess"] == _EXPECTED_DUPLICATE_EXCESS
    assert dedup["duplicate_density"] == pytest.approx(
        _EXPECTED_DUPLICATE_DENSITY, abs=1e-9
    )
    assert dedup["duplicate_density"] != pytest.approx(
        _LIVE_DUPLICATE_DENSITY, abs=1e-3
    )


def test_duplicate_density_honors_scope_filter(tmp_path: Path) -> None:
    with _build_fixture(tmp_path / "f.sqlite3") as store:
        alpha = compute_health_audit(store, scope="project:alpha")["dedup"]
        beta = compute_health_audit(store, scope="project:beta")["dedup"]

    assert alpha["total_traces"] == 4
    assert alpha["distinct_contents"] == 2
    assert alpha["duplicate_excess"] == 2
    assert alpha["duplicate_density"] == pytest.approx(0.5, abs=1e-9)

    assert beta["total_traces"] == 3
    assert beta["distinct_contents"] == 2
    assert beta["duplicate_excess"] == 1
    assert beta["duplicate_density"] == pytest.approx(1 / 3, abs=1e-9)


def test_duplicate_density_excludes_decayed_traces(tmp_path: Path) -> None:
    with _build_fixture(tmp_path / "f.sqlite3") as store:
        decayed_count = store.connection.execute(
            "SELECT COUNT(*) AS c FROM nodes WHERE level = 'trace' AND decayed = 1"
        ).fetchone()["c"]
        dedup = compute_health_audit(store)["dedup"]

    assert decayed_count == 1
    assert dedup["total_traces"] == _EXPECTED_ACTIVE_TRACES


# ---------------------------------------------------------------------------
# Feedback applied ratio
# ---------------------------------------------------------------------------


def test_feedback_applied_ratio_global_matches_fixture(tmp_path: Path) -> None:
    with _build_fixture(tmp_path / "f.sqlite3") as store:
        feedback = compute_health_audit(store)["feedback"]

    assert feedback["recall_events"] == _EXPECTED_RECALL_EVENTS
    assert feedback["feedback_applied"] == _EXPECTED_FEEDBACK_APPLIED
    assert (
        feedback["feedback_missing"]
        == _EXPECTED_RECALL_EVENTS - _EXPECTED_FEEDBACK_APPLIED
    )
    assert feedback["feedback_applied_ratio"] == pytest.approx(
        _EXPECTED_FEEDBACK_RATIO, abs=1e-9
    )
    assert feedback["feedback_applied_ratio"] != pytest.approx(
        _LIVE_FEEDBACK_RATIO, abs=1e-3
    )


def test_feedback_applied_ratio_filters_by_scope_only(tmp_path: Path) -> None:
    """When scope is set, only ``recall_events.scope = ?`` matches.

    A ``project:rise/extra`` recall event with ``scope == "project:rise/extra"``
    must not bleed into ``project:alpha`` results just because the global
    ``requested_scope`` overlap was relaxed.
    """

    with _build_fixture(tmp_path / "f.sqlite3") as store:
        alpha = compute_health_audit(store, scope="project:alpha")["feedback"]
        beta = compute_health_audit(store, scope="project:beta")["feedback"]

    assert alpha["recall_events"] == 3
    assert alpha["feedback_applied"] == 1
    assert alpha["feedback_applied_ratio"] == pytest.approx(1 / 3, abs=1e-9)

    assert beta["recall_events"] == 2
    assert beta["feedback_applied"] == 1
    assert beta["feedback_applied_ratio"] == pytest.approx(0.5, abs=1e-9)


def test_feedback_applied_ratio_is_none_on_zero_events(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "empty.sqlite3") as store:
        feedback = compute_health_audit(store)["feedback"]

    assert feedback["recall_events"] == 0
    assert feedback["feedback_applied"] == 0
    assert feedback["feedback_missing"] == 0
    assert feedback["feedback_applied_ratio"] is None


# ---------------------------------------------------------------------------
# Never-accessed ratio
# ---------------------------------------------------------------------------


def test_never_accessed_ratio_active_nodes_and_traces(tmp_path: Path) -> None:
    with _build_fixture(tmp_path / "f.sqlite3") as store:
        access = compute_health_audit(store)["access"]

    nodes = access["active_nodes"]
    traces = access["active_traces"]

    assert nodes["total"] == _EXPECTED_ACTIVE_NODES
    assert nodes["never_accessed"] == _EXPECTED_NEVER_ACCESSED_NODES
    assert nodes["never_accessed_ratio"] == pytest.approx(
        _EXPECTED_NEVER_ACCESSED_NODE_RATIO, abs=1e-9
    )

    assert traces["total"] == _EXPECTED_ACTIVE_TRACES
    assert traces["never_accessed"] == _EXPECTED_NEVER_ACCESSED_TRACES
    assert traces["never_accessed_ratio"] == pytest.approx(
        _EXPECTED_NEVER_ACCESSED_TRACE_RATIO, abs=1e-9
    )

    assert nodes["never_accessed_ratio"] != pytest.approx(
        _LIVE_NEVER_ACCESSED_NODE_RATIO, abs=1e-3
    )
    assert traces["never_accessed_ratio"] != pytest.approx(
        _LIVE_NEVER_ACCESSED_TRACE_RATIO, abs=1e-3
    )


def test_never_accessed_ratio_is_none_on_empty_db(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "empty.sqlite3") as store:
        access = compute_health_audit(store)["access"]

    assert access["active_nodes"]["total"] == 0
    assert access["active_nodes"]["never_accessed_ratio"] is None
    assert access["active_traces"]["total"] == 0
    assert access["active_traces"]["never_accessed_ratio"] is None


# ---------------------------------------------------------------------------
# Scope leakage candidates
# ---------------------------------------------------------------------------


def test_scope_leakage_candidate_counts_match_fixture(tmp_path: Path) -> None:
    with _build_fixture(tmp_path / "f.sqlite3") as store:
        sh = compute_health_audit(store)["scope_hygiene"]

    assert sh["candidate_scopes"] == _EXPECTED_CANDIDATE_SCOPES
    assert sh["active_candidate_nodes"] == _EXPECTED_ACTIVE_CANDIDATE_NODES
    assert sh["active_candidate_traces"] == _EXPECTED_ACTIVE_CANDIDATE_TRACES
    assert (
        sh["active_candidate_traces_never_accessed"]
        == _EXPECTED_NEVER_ACCESSED_CANDIDATE_TRACES
    )
    assert sh["active_candidate_trace_never_accessed_ratio"] == pytest.approx(
        _EXPECTED_NEVER_ACCESSED_CANDIDATE_TRACES
        / _EXPECTED_ACTIVE_CANDIDATE_TRACES,
        abs=1e-9,
    )
    assert (
        sh["candidate_scopes_with_zero_recall_events"]
        == _EXPECTED_CANDIDATE_ZERO_RECALL
    )

    assert sh["candidate_scopes"] != _LIVE_CANDIDATE_SCOPES
    assert sh["active_candidate_traces"] != _LIVE_ACTIVE_CANDIDATE_TRACES
    assert (
        sh["candidate_scopes_with_zero_recall_events"]
        != _LIVE_CANDIDATE_ZERO_RECALL
    )


def test_scope_leakage_predicate_unions_all_three_columns(tmp_path: Path) -> None:
    """Candidate scopes are the union of node.scope, recall_events.scope,
    and recall_events.requested_scope under the verbatim baseline predicate."""

    with _build_fixture(tmp_path / "f.sqlite3") as store:
        sh = compute_health_audit(store)["scope_hygiene"]

    candidate_scopes_in_top = {row["scope"] for row in sh["top_candidate_scopes"]}
    assert candidate_scopes_in_top == _EXPECTED_CANDIDATE_SCOPE_SET


def test_scope_hygiene_ignores_scope_filter(tmp_path: Path) -> None:
    """``scope_hygiene`` is always global; ``memory_health(scope=...)`` must
    not filter it (the contract: an audit run scoped to one project must
    still see leakage in other-project scopes)."""

    with _build_fixture(tmp_path / "f.sqlite3") as store:
        unscoped = compute_health_audit(store)["scope_hygiene"]
        scoped = compute_health_audit(store, scope="project:alpha")["scope_hygiene"]

    assert scoped["candidate_scopes"] == unscoped["candidate_scopes"]
    assert scoped["candidate_scopes"] == _EXPECTED_CANDIDATE_SCOPES
    assert scoped["top_candidate_scopes"] == unscoped["top_candidate_scopes"]


def test_scope_hygiene_top_candidate_scopes_ordering_is_deterministic(
    tmp_path: Path,
) -> None:
    """Ordering: ``(active_traces DESC, recall_events_as_scope ASC, scope ASC)``."""

    with _build_fixture(tmp_path / "f.sqlite3") as store:
        top = compute_health_audit(store)["scope_hygiene"]["top_candidate_scopes"]

    assert len(top) == _EXPECTED_CANDIDATE_SCOPES
    for prev, curr in zip(top, top[1:]):
        if prev["active_traces"] != curr["active_traces"]:
            assert prev["active_traces"] >= curr["active_traces"]
        elif prev["recall_events_as_scope"] != curr["recall_events_as_scope"]:
            assert prev["recall_events_as_scope"] <= curr["recall_events_as_scope"]
        else:
            assert prev["scope"] <= curr["scope"]


def test_scope_hygiene_top_candidate_scopes_respects_limit(tmp_path: Path) -> None:
    with _build_fixture(tmp_path / "f.sqlite3") as store:
        bounded = scope_hygiene_metrics(store, top_candidate_scopes=2)

    assert len(bounded["top_candidate_scopes"]) == 2
    assert bounded["candidate_scopes"] == _EXPECTED_CANDIDATE_SCOPES


def test_scope_hygiene_default_limit_matches_contract() -> None:
    assert DEFAULT_TOP_CANDIDATE_SCOPES == 20


# ---------------------------------------------------------------------------
# DB size
# ---------------------------------------------------------------------------


def test_storage_db_size_matches_filesystem_and_pragma_pages(tmp_path: Path) -> None:
    db_path = tmp_path / "size.sqlite3"
    with _build_fixture(db_path) as store:
        store.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        storage = compute_health_audit(store)["storage"]

    assert storage["db_path"] == str(db_path)
    assert storage["db_size_bytes"] == db_path.stat().st_size
    assert storage["page_count"] > 0
    assert storage["page_size"] > 0
    assert storage["page_bytes"] == storage["page_count"] * storage["page_size"]
    assert storage["db_size_bytes"] != _LIVE_DB_SIZE_BYTES


def test_storage_db_size_is_none_for_in_memory_store() -> None:
    with MemoryStore(":memory:") as store:
        storage = compute_health_audit(store)["storage"]

    assert storage["db_path"] == ":memory:"
    assert storage["db_size_bytes"] is None
    assert storage["page_count"] >= 0
    assert storage["page_size"] > 0


# ---------------------------------------------------------------------------
# Hot recall latency baseline shape
# ---------------------------------------------------------------------------


def test_hot_recall_latency_is_none_by_default(tmp_path: Path) -> None:
    """Default ``compute_health_audit`` does not invoke the recall probe."""

    with _build_fixture(tmp_path / "f.sqlite3") as store:
        assert compute_health_audit(store)["latency"] is None


def test_hot_recall_latency_emits_samples_and_summary(tmp_path: Path) -> None:
    with _build_fixture(tmp_path / "f.sqlite3") as store:
        result = compute_health_audit(store, latency_samples=3)

    latency = result["latency"]
    assert latency is not None
    assert latency["query"] == DEFAULT_LATENCY_QUERY
    assert latency["scope"] == "all"
    assert latency["max_results"] == 1
    assert latency["depth"] == 1
    assert latency["log_access"] is False
    assert latency["log_event"] is False
    assert latency["samples"] == 3
    assert isinstance(latency["warmup_ms"], float)
    assert latency["warmup_ms"] >= 0.0

    samples = latency["samples_ms"]
    assert isinstance(samples, list)
    assert len(samples) == 3
    assert all(isinstance(s, float) and s >= 0.0 for s in samples)

    summary = latency["summary"]
    assert summary["min_ms"] == min(samples)
    assert summary["max_ms"] == max(samples)
    assert summary["min_ms"] <= summary["median_ms"] <= summary["max_ms"]
    assert summary["min_ms"] <= summary["mean_ms"] <= summary["max_ms"]


def test_hot_recall_latency_probe_does_not_write_recall_events_or_access(
    tmp_path: Path,
) -> None:
    """Latency probe must use ``log_access=False, log_event=False`` so neither
    ``recall_events`` rows nor ``nodes.access_count`` are mutated."""

    with _build_fixture(tmp_path / "f.sqlite3") as store:
        before_events = store.connection.execute(
            "SELECT COUNT(*) AS c FROM recall_events"
        ).fetchone()["c"]
        before_access_sum = store.connection.execute(
            "SELECT COALESCE(SUM(access_count), 0) AS s FROM nodes"
        ).fetchone()["s"]

        result = compute_health_audit(store, latency_samples=3)
        assert result["latency"]["samples"] == 3

        after_events = store.connection.execute(
            "SELECT COUNT(*) AS c FROM recall_events"
        ).fetchone()["c"]
        after_access_sum = store.connection.execute(
            "SELECT COALESCE(SUM(access_count), 0) AS s FROM nodes"
        ).fetchone()["s"]

    assert after_events == before_events
    assert after_access_sum == before_access_sum


def test_hot_recall_latency_passes_scope_through(tmp_path: Path) -> None:
    with _build_fixture(tmp_path / "f.sqlite3") as store:
        scoped = compute_health_audit(
            store, scope="project:alpha", latency_samples=1
        )["latency"]
    assert scoped["scope"] == "project:alpha"
    assert scoped["samples"] == 1


# ---------------------------------------------------------------------------
# Deterministic output
# ---------------------------------------------------------------------------


def test_health_audit_deterministic_across_repeated_calls(tmp_path: Path) -> None:
    """Two calls against the same fixture produce byte-identical JSON for
    every non-wall-clock field."""

    with _build_fixture(tmp_path / "f.sqlite3") as store:
        first = compute_health_audit(store, latency_samples=2)
        second = compute_health_audit(store, latency_samples=2)

    first_clean = _strip_time_varying_fields(first)
    second_clean = _strip_time_varying_fields(second)
    assert json.dumps(first_clean, sort_keys=True) == json.dumps(
        second_clean, sort_keys=True
    )


def test_health_audit_top_candidate_scopes_ordering_deterministic(
    tmp_path: Path,
) -> None:
    with _build_fixture(tmp_path / "f.sqlite3") as store:
        first = compute_health_audit(store)["scope_hygiene"]["top_candidate_scopes"]
        second = compute_health_audit(store)["scope_hygiene"]["top_candidate_scopes"]
    assert first == second


# ---------------------------------------------------------------------------
# Read-only safety
# ---------------------------------------------------------------------------


def test_audit_against_read_only_path_does_not_modify_db_file(tmp_path: Path) -> None:
    fixture_path = tmp_path / "readonly.sqlite3"
    with _build_fixture(fixture_path) as store:
        store.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    md5_before = _md5_of_file(fixture_path)

    ro_store = open_read_only_store(fixture_path)
    try:
        assert isinstance(ro_store, ReadOnlyAuditStore)
        result = compute_health_audit(ro_store, latency_samples=3)
    finally:
        ro_store.close()

    md5_after = _md5_of_file(fixture_path)

    assert md5_before == md5_after
    for key in ("dedup", "feedback", "access", "scope_hygiene", "storage", "latency"):
        assert key in result, f"missing audit section: {key}"
    assert result["latency"]["samples"] == 3


def test_read_only_path_rejects_writes(tmp_path: Path) -> None:
    """The ``mode=ro`` URI must make SQLite refuse INSERT/UPDATE attempts."""

    fixture_path = tmp_path / "readonly-write-attempt.sqlite3"
    with _build_fixture(fixture_path) as store:
        store.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    ro_store = open_read_only_store(fixture_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            ro_store.connection.execute(
                """
                INSERT INTO nodes (
                    id, level, content, scope, timestamp, decayed,
                    access_count, usefulness_score, confidence, unique_agents,
                    source_traces, corrections, provenance,
                    created_at, updated_at, context
                ) VALUES (
                    'WRITE-ATTEMPT', 'trace', 'x', 'global',
                    '2026-05-22T00:00:00Z', 0, 0, 0.0, 0.5, 0,
                    '[]', '[]', '{}',
                    '2026-05-22T00:00:00Z', '2026-05-22T00:00:00Z', '{}'
                )
                """
            )
    finally:
        ro_store.close()


def test_audit_engine_path_input_opens_read_only(tmp_path: Path) -> None:
    """Calling ``compute_health_audit(path)`` should open ``mode=ro`` itself
    and close the connection on exit."""

    fixture_path = tmp_path / "path-input.sqlite3"
    with _build_fixture(fixture_path) as store:
        store.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    md5_before = _md5_of_file(fixture_path)
    result = compute_health_audit(fixture_path, latency_samples=2)
    md5_after = _md5_of_file(fixture_path)

    assert md5_before == md5_after
    assert result["dedup"]["total_traces"] == _EXPECTED_ACTIVE_TRACES
    assert result["latency"]["samples"] == 2


# ---------------------------------------------------------------------------
# Completeness + surface integration
# ---------------------------------------------------------------------------


def test_compute_health_audit_emits_every_required_metric_section(
    tmp_path: Path,
) -> None:
    with _build_fixture(tmp_path / "f.sqlite3") as store:
        result = compute_health_audit(store)

    required_sections = {
        "dedup",
        "feedback",
        "access",
        "scope_hygiene",
        "storage",
        "latency",
        "instructions",
    }
    assert required_sections.issubset(set(result.keys()))

    # Six required metrics from the parent SPEC map to these sub-keys.
    assert "duplicate_density" in result["dedup"]
    assert "feedback_applied_ratio" in result["feedback"]
    assert "candidate_scopes" in result["scope_hygiene"]
    assert "never_accessed_ratio" in result["access"]["active_nodes"]
    assert "never_accessed_ratio" in result["access"]["active_traces"]
    assert "db_size_bytes" in result["storage"]
    # `latency` may be None by default; the key itself must be present.


def test_memory_health_surface_includes_audit_sections(tmp_path: Path) -> None:
    """``memory_health`` (the existing MCP/CLI surface) must surface the audit
    sections without breaking pre-existing fields."""

    with _build_fixture(tmp_path / "f.sqlite3") as store:
        report = memory_health(store)
        direct = compute_health_audit(store)

    existing_fields = (
        "scope",
        "window_hours",
        "generated_at",
        "counts",
        "activity",
        "dedup",
        "staleness",
        "retrieval_policy",
    )
    for key in existing_fields:
        assert key in report, f"existing memory_health field missing: {key}"

    audit_fields = ("feedback", "access", "scope_hygiene", "storage", "latency", "instructions")
    for key in audit_fields:
        assert key in report, f"audit section missing from memory_health: {key}"

    assert report["feedback"] == direct["feedback"]
    assert report["access"] == direct["access"]
    assert report["scope_hygiene"] == direct["scope_hygiene"]
    assert report["storage"]["db_size_bytes"] == direct["storage"]["db_size_bytes"]
    assert report["storage"]["page_count"] == direct["storage"]["page_count"]


# ---------------------------------------------------------------------------
# Feedback closure (windowed closure ratio per identity class)
# ---------------------------------------------------------------------------


def _record_closure_event(
    store: MemoryStore,
    *,
    scope: str,
    ambient: dict[str, Any],
    closed_by: str | None = None,
    query: str = "closure-query",
) -> str:
    """Record one recall event through the real write path; optionally close it."""

    event = store.record_recall_event(
        query=query,
        scope=scope,
        ambient_context=ambient,
        results=[],
    )
    if closed_by is not None:
        store.mark_recall_event_feedback(event.id, closed_by)
    return event.id


def _rewrite_event_created_at(store: MemoryStore, event_id: str, *, days_ago: int) -> None:
    stale_created = (
        (datetime.now(UTC) - timedelta(days=days_ago))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    with store.connection:
        store.connection.execute(
            "UPDATE recall_events SET created_at = ? WHERE id = ?",
            (stale_created, event_id),
        )


def test_feedback_closure_counts_and_ratios_per_identity_class(tmp_path: Path) -> None:
    """Each event lands in exactly one class by precedence, with exact ratios.

    explicit: 4 events / 3 closed (one also carries a transport id, proving
    explicit precedence). transport_only: 3 events / 1 closed (one carries a
    partial agent-only identity, proving partial identity is not explicit).
    none: 2 events / 0 closed (one carries agent-only without transport).
    """

    explicit_ambient = {"agent": "agent-x", "task": "task-x", "session_id": "sess-x"}
    with MemoryStore(tmp_path / "closure.sqlite3") as store:
        target = store.append_trace(
            "closure feedback target", {"scope": "project:alpha"}
        )
        _record_closure_event(
            store,
            scope="project:alpha",
            ambient={**explicit_ambient, "transport_session_id": "tsid-precedence"},
            closed_by=target.id,
        )
        for index in range(2):
            _record_closure_event(
                store,
                scope="project:alpha",
                ambient=explicit_ambient,
                closed_by=target.id,
                query=f"explicit-{index}",
            )
        _record_closure_event(store, scope="project:alpha", ambient=explicit_ambient)

        _record_closure_event(
            store,
            scope="project:alpha",
            ambient={"transport_session_id": "tsid-1"},
            closed_by=target.id,
        )
        _record_closure_event(
            store, scope="project:alpha", ambient={"transport_session_id": "tsid-1"}
        )
        _record_closure_event(
            store,
            scope="project:alpha",
            ambient={"agent": "agent-x", "transport_session_id": "tsid-2"},
        )

        _record_closure_event(store, scope="project:alpha", ambient={})
        _record_closure_event(store, scope="project:alpha", ambient={"agent": "only-agent"})

        block = feedback_closure_metrics(store, window_hours=24)

    assert block["window_hours"] == 24
    assert block["recall_events_in_window"] == 9
    assert block["feedback_applied_in_window"] == 4
    assert block["closure_ratio"] == pytest.approx(4 / 9, abs=1e-9)

    coverage = block["identity_coverage"]
    assert set(coverage) == {"explicit", "transport_only", "none"}
    assert coverage["explicit"]["events"] == 4
    assert coverage["explicit"]["closed"] == 3
    assert coverage["explicit"]["closure_ratio"] == pytest.approx(0.75, abs=1e-9)
    assert coverage["transport_only"]["events"] == 3
    assert coverage["transport_only"]["closed"] == 1
    assert coverage["transport_only"]["closure_ratio"] == pytest.approx(1 / 3, abs=1e-9)
    assert coverage["none"]["events"] == 2
    assert coverage["none"]["closed"] == 0
    assert coverage["none"]["closure_ratio"] == 0.0

    # The classes partition the window: per-class counts sum to the totals.
    assert sum(item["events"] for item in coverage.values()) == 9
    assert sum(item["closed"] for item in coverage.values()) == 4

    assert block["closure_ratio"] != pytest.approx(_LIVE_CLOSURE_RATIO, abs=1e-2)
    assert coverage["explicit"]["closure_ratio"] != pytest.approx(
        _LIVE_EXPLICIT_CLASS_CLOSURE, abs=1e-2
    )
    assert coverage["none"]["closure_ratio"] != pytest.approx(
        _LIVE_IDENTITYLESS_CLOSURE, abs=1e-2
    )


def test_feedback_closure_excludes_events_older_than_window(tmp_path: Path) -> None:
    """A closed event rewritten to 10 days ago drops out of a 24h window."""

    with MemoryStore(tmp_path / "closure-window.sqlite3") as store:
        target = store.append_trace("closure window target", {"scope": "project:alpha"})
        _record_closure_event(
            store, scope="project:alpha", ambient={}, closed_by=target.id
        )
        _record_closure_event(store, scope="project:alpha", ambient={})
        stale_id = _record_closure_event(
            store, scope="project:alpha", ambient={}, closed_by=target.id
        )
        _rewrite_event_created_at(store, stale_id, days_ago=10)

        block = feedback_closure_metrics(store, window_hours=24)

    assert block["recall_events_in_window"] == 2
    assert block["feedback_applied_in_window"] == 1
    assert block["closure_ratio"] == pytest.approx(0.5, abs=1e-9)
    assert block["identity_coverage"]["none"]["events"] == 2
    assert block["identity_coverage"]["none"]["closed"] == 1


def test_feedback_closure_honors_scope_filter(tmp_path: Path) -> None:
    """Only ``recall_events.scope = ?`` matches; requested_scope must not bleed."""

    with MemoryStore(tmp_path / "closure-scope.sqlite3") as store:
        target = store.append_trace("closure scope target", {"scope": "project:alpha"})
        _record_closure_event(
            store,
            scope="project:alpha",
            ambient={"transport_session_id": "tsid-a"},
            closed_by=target.id,
        )
        _record_closure_event(store, scope="project:alpha", ambient={})
        _record_closure_event(store, scope="project:beta", ambient={})
        store.record_recall_event(
            query="beta-scope-alpha-requested",
            scope="project:beta",
            requested_scope="project:alpha",
            ambient_context={},
            results=[],
        )

        alpha = feedback_closure_metrics(store, scope="project:alpha", window_hours=24)
        beta = feedback_closure_metrics(store, scope="project:beta", window_hours=24)
        overall = feedback_closure_metrics(store, window_hours=24)

    assert alpha["recall_events_in_window"] == 2
    assert alpha["feedback_applied_in_window"] == 1
    assert alpha["closure_ratio"] == pytest.approx(0.5, abs=1e-9)
    assert alpha["identity_coverage"]["transport_only"] == {
        "events": 1,
        "closed": 1,
        "closure_ratio": 1.0,
    }

    assert beta["recall_events_in_window"] == 2
    assert beta["feedback_applied_in_window"] == 0
    assert beta["closure_ratio"] == 0.0

    assert overall["recall_events_in_window"] == 4
    assert overall["feedback_applied_in_window"] == 1


def test_feedback_closure_ratio_is_none_on_empty_window(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "closure-empty.sqlite3") as store:
        empty = feedback_closure_metrics(store, window_hours=24)

        target = store.append_trace("closure empty target", {"scope": "project:alpha"})
        stale_id = _record_closure_event(
            store, scope="project:alpha", ambient={}, closed_by=target.id
        )
        _rewrite_event_created_at(store, stale_id, days_ago=10)
        aged_out = feedback_closure_metrics(store, window_hours=24)

    for block in (empty, aged_out):
        assert block["recall_events_in_window"] == 0
        assert block["feedback_applied_in_window"] == 0
        assert block["closure_ratio"] is None
        for name in ("explicit", "transport_only", "none"):
            assert block["identity_coverage"][name] == {
                "events": 0,
                "closed": 0,
                "closure_ratio": None,
            }


def test_memory_health_payload_includes_feedback_closure_block(tmp_path: Path) -> None:
    """``memory_health`` surfaces the block; scope and window_hours flow through."""

    with MemoryStore(tmp_path / "closure-health.sqlite3") as store:
        target = store.append_trace("closure health target", {"scope": "project:alpha"})
        _record_closure_event(
            store,
            scope="project:alpha",
            ambient={"agent": "a", "task": "t", "session_id": "s"},
            closed_by=target.id,
        )
        _record_closure_event(
            store, scope="project:alpha", ambient={"transport_session_id": "tsid-h"}
        )

        report = memory_health(store, scope="project:alpha", window_hours=24, top_stale=0)
        direct = feedback_closure_metrics(store, scope="project:alpha", window_hours=24)
        default_report = memory_health(store, top_stale=0)

    closure = report["feedback_closure"]
    assert closure == direct
    assert closure["window_hours"] == 24
    assert closure["recall_events_in_window"] == 2
    assert closure["feedback_applied_in_window"] == 1
    assert closure["closure_ratio"] == pytest.approx(0.5, abs=1e-9)
    assert closure["identity_coverage"]["explicit"]["events"] == 1
    assert closure["identity_coverage"]["explicit"]["closed"] == 1
    assert closure["identity_coverage"]["transport_only"]["events"] == 1
    assert closure["identity_coverage"]["transport_only"]["closed"] == 0

    assert default_report["feedback_closure"]["window_hours"] == 168


def test_recall_events_summary_exposes_transport_session_id(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "closure-summary.sqlite3") as store:
        store.record_recall_event(
            query="with-transport",
            scope="project:alpha",
            ambient_context={"transport_session_id": "tsid-visible"},
            results=[],
        )
        store.record_recall_event(
            query="without-transport",
            scope="project:alpha",
            ambient_context={},
            results=[],
        )

        summary = recall_events_summary(store, scope="project:alpha")

    recent = {item["query"]: item for item in summary["recent"]}
    assert recent["with-transport"]["transport_session_id"] == "tsid-visible"
    assert recent["without-transport"]["transport_session_id"] is None
