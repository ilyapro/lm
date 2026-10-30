"""Decay reads the coverage facts required by the selected policy."""

from datetime import UTC, datetime
import json
import sqlite3

import pytest

from living_memory import decay
from living_memory.storage import MemoryStore


NOW = datetime(2026, 10, 23, tzinfo=UTC)
SCOPE = "project:decay"


def owner(store, node_id, sources, provenance, *, level="concept", decayed=False):
    store.create_node(
        level=level, node_id=node_id, content="Source digest.",
        context={"scope": "project:other", "timestamp": "2026-10-23"},
    )
    with store.transaction():
        store.connection.execute(
            "UPDATE nodes SET source_traces=?, provenance=?, decayed=? WHERE id=?",
            (sources, provenance, int(decayed), node_id),
        )


@pytest.mark.parametrize("policy", ["default", "env-off", "explicit-off", "env-on", "explicit-on"])
def test_ordinary_sweep_projects_and_parses_only_required_coverage(tmp_path, monkeypatch, policy):
    enabled = policy.endswith("-on")
    monkeypatch.delenv(decay.CONCEPT_COVERED_ENV, raising=False)
    options = {}
    if policy.startswith("env-"):
        monkeypatch.setenv(decay.CONCEPT_COVERED_ENV, "1" if enabled else "0")
    elif policy.startswith("explicit-"):
        monkeypatch.setenv(decay.CONCEPT_COVERED_ENV, "0" if enabled else "1")
        options["decay_concept_covered"] = enabled

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for node_id in ("covered", "dated"):
            store.create_node(
                level="trace", node_id=node_id, content="Read source.",
                context={"scope": SCOPE, "timestamp": "2026-10-20"},
                stats={"last_accessed": "2026-10-22"},
            )
        bodies = [
            json.dumps({"source_coverage": dict.fromkeys(["covered", "dated"], "2026-10-20"),
                        "unused": "x" * 262144}),
            json.dumps({"source_coverage": dict.fromkeys(["covered", "dated"], "2026-10-18")}),
            "{malformed provenance",
            json.dumps({"source_coverage": {"covered": ["invalid"]}}),
            json.dumps({"source_coverage": {"phantom": "2026-10-20"}}),
        ]
        for index, body in enumerate(bodies):
            sources = ["covered", "dated"] if index < 2 else ["covered"]
            owner(store, f"owner-{index}",
                  "{malformed sources" if index == 4 else json.dumps(sources), body,
                  level="schema" if index == 1 else "concept")
        excluded = json.dumps({"unused": "decayed body" * 30000})
        owner(store, "inactive", '["inactive-source"]', excluded, decayed=True)
        owner(store, "non-owner", '["non-owner-source"]', excluded, level="trace")
        assert decay.concept_coverage(store) == {
            "covered": None, "dated": datetime(2026, 10, 18, tzinfo=UTC),
        }

        parsed = []
        projected = []
        queries = []
        loads = json.loads

        def record_loads(raw, *args, **kwargs):
            if isinstance(raw, str) and raw in [*bodies, excluded]:
                parsed.append(raw)
            return loads(raw, *args, **kwargs)

        def record_rows(cursor, values):
            row = sqlite3.Row(cursor, values)
            if "source_traces" in row.keys():
                projected.append(dict(row))
            return row

        def required_reads(action, table, column, *rest):
            if (not enabled and action == sqlite3.SQLITE_READ and table == "nodes"
                    and column in {"content", "provenance"}):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        monkeypatch.setattr(json, "loads", record_loads)
        original_factory = store.connection.row_factory
        store.connection.row_factory = record_rows
        store.connection.set_authorizer(required_reads)
        store.connection.set_trace_callback(queries.append)
        try:
            assert decay.soft_delete_expired(
                store, scope=SCOPE, now=NOW, ttl_days=7,
                ttl_anchor="created", max_expired=0, **options,
            ) == []
        finally:
            store.connection.set_authorizer(None)
            store.connection.set_trace_callback(None)
            store.connection.row_factory = original_factory

        assert len(projected) == len(bodies)
        assert all(set(row) == ({"source_traces", "provenance"} if enabled
                                else {"source_traces"}) for row in projected)
        assert len([query for query in queries if "SELECT" in query.upper()
                    and "source_traces" in query]) == 1
        assert sorted(parsed) == sorted(bodies if enabled else [])
        assert sum(len(row.get("provenance", "")) for row in projected) == (
            sum(map(len, bodies)) if enabled else 0
        )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("anchor", ["created", "last_accessed"])
@pytest.mark.parametrize("cap", [0, 2])
def test_coverage_policy_preserves_selection_reasons_order_and_validity(tmp_path, enabled, anchor, cap):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        dates = [
            ("uncovered", "2020-01-01", "2026-10-22"),
            ("inactive-source", "2021-01-01", None),
            ("ttl-fallback", "2022-01-01", "2026-10-22"),
            ("earliest", "2026-10-18", "2026-10-19"),
            ("tie-b", "2026-10-19", None),
            ("tie-a", "2026-10-19", "2026-10-19"),
            ("unknown-read", "2026-10-20", "2026-10-19"),
            ("unknown-unread", "2026-10-20", None),
        ]
        for node_id, timestamp, read in dates:
            store.create_node(
                level="trace", node_id=node_id, content=node_id,
                context={"scope": SCOPE, "timestamp": timestamp},
                stats={"last_accessed": read},
            )
        store.create_node(
            level="trace", node_id="foreign", content="Outside candidate scope.",
            context={"scope": "project:other", "timestamp": "2020-01-01"},
        )
        covered = [node_id for node_id, _, _ in dates[2:]]
        owner(store, "dated-owner", json.dumps(covered), json.dumps({
            "source_coverage": dict.fromkeys(covered, "2026-10-20"),
        }))
        owner(store, "earlier-owner", '["earliest"]', json.dumps({
            "source_coverage": {"earliest": "2026-10-18"},
        }), level="schema")
        owner(store, "unknown-owner", '["unknown-read", "unknown-unread"]', "{bad provenance")
        owner(store, "inactive-owner", '["inactive-source"]', "{}", decayed=True)
        assert decay.concept_coverage(store) == {
            **dict.fromkeys(covered, datetime(2026, 10, 20, tzinfo=UTC)),
            "earliest": datetime(2026, 10, 18, tzinfo=UTC),
            "unknown-read": None, "unknown-unread": None,
        }

        expected = [("inactive-source", decay.TTL_EXPIRED_REASON)]
        if anchor == "created":
            expected.insert(0, ("uncovered", decay.TTL_EXPIRED_REASON))
        if enabled:
            if anchor == "created":
                expected.append(("ttl-fallback", decay.TTL_EXPIRED_REASON))
            expected.extend((node_id, decay.CONCEPT_COVERED_REASON)
                            for node_id in ["tie-a", "tie-b", "unknown-unread"])
        if cap:
            expected = expected[:cap]
        retired = decay.soft_delete_expired(
            store, scope=SCOPE, now=NOW, ttl_days=7, ttl_anchor=anchor,
            decay_concept_covered=enabled, max_expired=cap,
        )
        assert [(node.id, node.decay_reason) for node in retired] == expected
        assert [(row["id"], row["decay_reason"]) for row in store.connection.execute(
            "SELECT id, decay_reason FROM nodes WHERE decayed=1 AND level='trace' ORDER BY id"
        )] == sorted(expected)
        assert not store.get_node("foreign").decayed


def test_same_membership_with_different_dates_changes_opted_in_retirement(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.create_node(
            level="trace", node_id="source", content="Read between coverage dates.",
            context={"timestamp": "2026-10-18"}, stats={"last_accessed": "2026-10-19"},
        )
        owner(store, "digest", '["source"]', '{"source_coverage":{"source":"2026-10-18"}}')
        options = dict(now=NOW, ttl_days=7, ttl_anchor="created", max_expired=0)
        assert decay.soft_delete_expired(store, decay_concept_covered=True, **options) == []
        with store.transaction():
            store.connection.execute(
                "UPDATE nodes SET provenance=? WHERE id='digest'",
                ('{"source_coverage":{"source":"2026-10-20"}}',),
            )
        assert decay.concept_coverage(store) == {"source": datetime(2026, 10, 20, tzinfo=UTC)}
        assert decay.soft_delete_expired(store, decay_concept_covered=False, **options) == []
        assert [(node.id, node.decay_reason) for node in decay.soft_delete_expired(
            store, decay_concept_covered=True, **options,
        )] == [("source", decay.CONCEPT_COVERED_REASON)]
