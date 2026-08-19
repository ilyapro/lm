"""Tests for ``living_memory.postsession.usage_metric`` and its CLI.

Every test runs against a fixture database built in ``tmp_path``. The live
database at ``~/.local/share/living-memory/global.sqlite3`` is never opened
here, not even read-only — the published baseline is reproduced by the artifact
run, not by the unit suite.

The properties under test are the ones the metric would be worthless without:

* consumption is *strictly* later — an event sharing the node's own timestamp is
  the recall that produced the write, not evidence the write was used;
* the cohort is every row in the window at every level, decayed included, which
  is the definition the published field baseline was measured under;
* ``--as-of`` cuts both ``nodes`` and ``recall_events``, so a run is pinned;
* the grounded variant divides by the events that could possibly be graded
  (those closed by a remember trace), never by the whole cohort — the measured
  trap this module exists to avoid;
* the whole run is read-only: the database file comes out byte-identical and no
  journal or WAL sidecar is left behind.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Sequence

import pytest

from living_memory.grounding import DEFAULT_MIN_CONTAINMENT
from living_memory.postsession.usage_metric import (
    PrivacyGuardError,
    check_privacy,
    ground_delivered_results,
    parse_context_key,
    parse_window,
    run_metric,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "living_memory" / "postsession" / "usage_metric.py"
SCRIPT_PATH = REPO_ROOT / "scripts" / "trace_usage_linkage.py"

WINDOW = "2026-07-01..2026-08-01"
AS_OF = "2026-08-19T00:00:00Z"

# Two disjoint vocabularies. The "grounding" trace repeats the schema node's
# content verbatim, so its IDF containment is 1.0; the "silent" trace shares no
# token with anything, so its containment is 0.0. Both go through the real
# tokenizer — the threshold is never stubbed.
SCHEMA_CONTENT = "restart the sqlite writer before backfilling chunk embeddings"
GROUNDING_TRACE = (
    "closure note: restart the sqlite writer before backfilling chunk embeddings, "
    "otherwise the migration deadlocks"
)
SILENT_TRACE = "unrelated observation about kubernetes ingress certificate rotation"


NODE_COLUMNS = (
    "id",
    "level",
    "content",
    "scope",
    "agent",
    "task",
    "context",
    "decayed",
    "usefulness_score",
    "created_at",
)


def _make_db(path: Path, nodes: Sequence[dict[str, Any]], events: Sequence[dict[str, Any]]) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            level TEXT,
            content TEXT,
            scope TEXT,
            agent TEXT,
            task TEXT,
            context TEXT,
            decayed INTEGER DEFAULT 0,
            usefulness_score REAL DEFAULT 0.0,
            created_at TEXT
        );
        CREATE TABLE recall_events (
            id TEXT PRIMARY KEY,
            query TEXT,
            results TEXT,
            feedback_trace_id TEXT,
            created_at TEXT
        );
        """
    )
    for node in nodes:
        row = {
            "level": "trace",
            "content": "",
            "scope": "global",
            "agent": None,
            "task": None,
            "context": None,
            "decayed": 0,
            "usefulness_score": 0.0,
            **node,
        }
        connection.execute(
            "INSERT INTO nodes ({}) VALUES ({})".format(
                ", ".join(NODE_COLUMNS), ", ".join("?" * len(NODE_COLUMNS))
            ),
            [row[column] for column in NODE_COLUMNS],
        )
    for event in events:
        connection.execute(
            "INSERT INTO recall_events (id, query, results, feedback_trace_id, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                event["id"],
                event.get("query", "q"),
                json.dumps(
                    [
                        {"node_id": node_id, "rank": index + 1, "level": "trace", "scope": "global"}
                        for index, node_id in enumerate(event["results"])
                    ]
                ),
                event.get("feedback_trace_id"),
                event["created_at"],
            ),
        )
    connection.commit()
    connection.close()
    return path


@pytest.fixture()
def fixture_db(tmp_path: Path) -> Path:
    """A window with one node per behaviour the definition has to distinguish."""

    nodes = [
        # Closing traces, written before the window so they never join a cohort.
        {"id": "t_ground", "content": GROUNDING_TRACE, "created_at": "2026-06-20T00:00:00Z"},
        {"id": "t_silent", "content": SILENT_TRACE, "created_at": "2026-06-20T00:00:00Z"},
        # Before the window.
        {"id": "n_before", "content": "older note", "created_at": "2026-06-30T12:00:00Z"},
        # In the window.
        {
            "id": "n_trace",
            "level": "trace",
            "agent": "ae",
            "content": "trace level node about postgres connection pooling",
            "created_at": "2026-07-02T00:00:00Z",
        },
        {
            "id": "n_concept",
            "level": "concept",
            "content": "concept level node about caching",
            "context": json.dumps({"session_id": "S1", "source": "extractor"}),
            "created_at": "2026-07-03T00:00:00Z",
        },
        {
            "id": "n_schema",
            "level": "schema",
            "decayed": 1,
            "content": SCHEMA_CONTENT,
            "created_at": "2026-07-04T00:00:00Z",
        },
        {
            "id": "n_same_ts",
            "agent": "codex",
            "content": "node recalled only by the event that produced it",
            "created_at": "2026-07-05T00:00:00Z",
        },
        {
            "id": "n_late",
            "content": "node recalled again only after ten days",
            "context": json.dumps({"source": "organic"}),
            "created_at": "2026-07-06T00:00:00Z",
        },
        # After the window.
        {"id": "n_after", "content": "later note", "created_at": "2026-08-05T00:00:00Z"},
    ]
    events = [
        {"id": "e1", "results": ["n_trace", "n_concept"], "created_at": "2026-07-08T00:00:00Z"},
        # Same instant as n_same_ts: the recall that preceded the write.
        {"id": "e2", "results": ["n_same_ts"], "created_at": "2026-07-05T00:00:00Z"},
        {"id": "e3", "results": ["n_late"], "created_at": "2026-07-16T00:00:00Z"},
        {
            "id": "e4",
            "results": ["n_schema"],
            "feedback_trace_id": "t_ground",
            "created_at": "2026-07-05T00:00:00Z",
        },
        {
            "id": "e5",
            "results": ["n_trace"],
            "feedback_trace_id": "t_silent",
            "created_at": "2026-07-20T00:00:00Z",
        },
        # Past every --as-of the tests pin.
        {"id": "e_future", "results": ["n_same_ts"], "created_at": "2026-08-25T00:00:00Z"},
    ]
    return _make_db(tmp_path / "fixture.sqlite3", nodes, events)


def _cohorts(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {cohort["id"]: cohort for cohort in report["cohorts"]}


def _run(db: Path, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("window", WINDOW)
    kwargs.setdefault("as_of", AS_OF)
    return run_metric(db, **kwargs)


# ---------------------------------------------------------------------------
# The definition
# ---------------------------------------------------------------------------


def test_consumption_counts_only_strictly_later_events(fixture_db: Path) -> None:
    cohort = _cohorts(_run(fixture_db))["all"]

    # n_trace, n_concept, n_schema, n_late — but not n_same_ts, whose only
    # in-range event shares its timestamp.
    assert cohort["nodes"] == 5
    assert cohort["consumed_later"] == 4
    assert cohort["rate"] == 0.8


def test_cohort_keeps_every_level_and_decayed_rows(fixture_db: Path) -> None:
    cohort = _cohorts(_run(fixture_db))["all"]

    # trace + concept + schema, and n_schema is decayed=1. Dropping either would
    # shrink the cohort below 5 — the exact mistake that fails to reproduce the
    # published July baseline.
    assert cohort["nodes"] == 5
    assert cohort["selector"]["window_start"] == "2026-07-01T00:00:00Z"
    assert cohort["selector"]["window_end"] == "2026-08-01T00:00:00Z"


def test_within_7d_horizon_excludes_a_later_first_recall(fixture_db: Path) -> None:
    cohort = _cohorts(_run(fixture_db))["all"]

    # n_trace (+6d), n_concept (+5d), n_schema (+1d) land inside the horizon;
    # n_late is first recalled 10 days after it was written.
    assert cohort["consumed_within_7d"] == 3
    assert cohort["within_7d_rate"] == 0.6
    assert cohort["within_days"] == 7


def test_window_is_half_open(fixture_db: Path) -> None:
    edge = _cohorts(_run(fixture_db, window="2026-07-02..2026-07-05"))["all"]
    next_window = _cohorts(_run(fixture_db, window="2026-07-05..2026-07-06"))["all"]

    # n_trace (07-02, the inclusive start), n_concept and n_schema; n_same_ts at
    # the exclusive end belongs to the next window instead, so adjacent windows
    # partition the timeline without double-counting.
    assert edge["nodes"] == 3
    assert next_window["nodes"] == 1


def test_as_of_cuts_recall_events(fixture_db: Path) -> None:
    pinned = _cohorts(_run(fixture_db))["all"]
    open_ended = _cohorts(_run(fixture_db, as_of="2026-09-01T00:00:00Z"))["all"]

    # e_future (2026-08-25) delivers n_same_ts. Only the later cutoff sees it.
    assert pinned["consumed_later"] == 4
    assert open_ended["consumed_later"] == 5


def test_as_of_cuts_nodes(fixture_db: Path) -> None:
    late_window = {"window": "2026-08-01..2026-09-01"}
    before = _cohorts(_run(fixture_db, as_of="2026-08-03T00:00:00Z", **late_window))["all"]
    after = _cohorts(_run(fixture_db, **late_window))["all"]

    # n_after was written 2026-08-05.
    assert before["nodes"] == 0
    assert after["nodes"] == 1


def test_missing_as_of_is_disclosed_as_unreproducible(fixture_db: Path) -> None:
    report = _run(fixture_db, as_of=None)

    assert report["meta"]["as_of"] is None
    assert any("not reproducible" in note for note in report["notes"])


# ---------------------------------------------------------------------------
# Cohort selection
# ---------------------------------------------------------------------------


def test_agent_cohorts_including_unattributed(fixture_db: Path) -> None:
    cohorts = _cohorts(_run(fixture_db, agents=["ae", "unattributed", "codex"]))

    assert cohorts["agent=ae"]["nodes"] == 1
    assert cohorts["agent=ae"]["consumed_later"] == 1
    assert cohorts["agent=codex"]["nodes"] == 1
    # n_concept, n_schema, n_late carry no agent.
    assert cohorts["agent=unattributed"]["nodes"] == 3
    assert cohorts["agent=unattributed"]["selector"]["agent_is_null"] is True
    # The window-wide cohort is always kept as the comparison population.
    assert cohorts["all"]["nodes"] == 5


def test_context_key_predicate_selects_by_value_and_by_presence(fixture_db: Path) -> None:
    cohorts = _cohorts(
        _run(
            fixture_db,
            context_keys=[
                parse_context_key("source=extractor"),
                parse_context_key("source=organic"),
                parse_context_key("session_id="),
                parse_context_key("source=absent"),
            ],
        )
    )

    assert cohorts["context.source=extractor"]["nodes"] == 1
    assert cohorts["context.source=organic"]["nodes"] == 1
    assert cohorts["context.session_id=*"]["nodes"] == 1
    assert cohorts["context.source=absent"]["nodes"] == 0
    assert cohorts["context.source=absent"]["rate"] is None


def test_parse_helpers_reject_malformed_input() -> None:
    with pytest.raises(ValueError):
        parse_window("2026-07-01")
    with pytest.raises(ValueError):
        parse_window("2026-08-01..2026-07-01")
    with pytest.raises(ValueError):
        parse_context_key("source")
    assert parse_window("2026-07-01..2026-08-01") == (
        "2026-07-01T00:00:00Z",
        "2026-08-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# The grounded variant and its denominator
# ---------------------------------------------------------------------------


def test_grounded_divides_by_closed_consumers_not_by_the_cohort(fixture_db: Path) -> None:
    grounded = _cohorts(_run(fixture_db))["all"]["grounded"]

    # Only e4 and e5 were closed by a remember trace, reaching n_schema and
    # n_trace. n_concept and n_late were re-delivered but never by a closed
    # event, so they cannot be graded at all.
    assert grounded["nodes_with_closed_consumer"] == 2
    assert grounded["consumed_later"] == 1
    assert grounded["denominator"] == "nodes_with_closed_consumer"
    assert grounded["rate"] == 0.5
    # The same numerator over the whole cohort is 2.5x smaller — the trap this
    # split exists to keep out of the published number.
    assert grounded["rate_over_cohort"] == 0.2
    assert grounded["rate"] > grounded["rate_over_cohort"]


def test_grounded_requires_containment_not_merely_a_closing_trace(fixture_db: Path) -> None:
    grounded = _cohorts(_run(fixture_db))["all"]["grounded"]

    # e5 closes on a trace that shares no vocabulary with n_trace, so n_trace
    # sits in the denominator and not in the numerator.
    assert grounded["consuming_events_closed_by_trace"] == 2
    assert grounded["grounded_consuming_events"] == 1
    assert grounded["consumed_within_7d"] == 1
    assert grounded["min_containment"] == DEFAULT_MIN_CONTAINMENT


def test_grounded_bulk_pass_agrees_with_the_live_entry_point(fixture_db: Path) -> None:
    report = _run(fixture_db)
    crosscheck = report["grounding"]["live_path_crosscheck"]

    assert crosscheck["sampled_pairs"] > 0
    assert crosscheck["agreement_rate"] == 1.0
    assert report["grounding"]["diagnostics"]["min_containment"] == DEFAULT_MIN_CONTAINMENT


def test_ground_delivered_results_uses_the_shared_threshold() -> None:
    grounded = ground_delivered_results(
        GROUNDING_TRACE, {"n_schema": SCHEMA_CONTENT, "n_other": SILENT_TRACE}
    )

    assert grounded == {"n_schema"}
    assert ground_delivered_results(SILENT_TRACE, {"n_schema": SCHEMA_CONTENT}) == set()


def test_grounded_variant_can_be_skipped(fixture_db: Path) -> None:
    grounded = _cohorts(_run(fixture_db, grounded=False))["all"]["grounded"]

    assert grounded["skipped"] is True
    assert grounded["rate"] is None


def test_empty_grounded_denominator_is_disclosed(tmp_path: Path) -> None:
    db = _make_db(
        tmp_path / "unclosed.sqlite3",
        [{"id": "n1", "content": "solo node", "created_at": "2026-07-02T00:00:00Z"}],
        [{"id": "e1", "results": ["n1"], "created_at": "2026-07-03T00:00:00Z"}],
    )
    report = _run(db)
    grounded = _cohorts(report)["all"]["grounded"]

    # No closed consumer anywhere: the rate is undefined, not zero.
    assert grounded["nodes_with_closed_consumer"] == 0
    assert grounded["rate"] is None
    assert any("empty denominator" in note for note in report["notes"])


# ---------------------------------------------------------------------------
# Read-only safety and privacy
# ---------------------------------------------------------------------------


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_run_never_writes_to_the_database(fixture_db: Path) -> None:
    before = _digest(fixture_db)

    _run(fixture_db, agents=["ae", "unattributed"])

    assert _digest(fixture_db) == before
    for sidecar in ("-wal", "-shm", "-journal"):
        assert not Path(str(fixture_db) + sidecar).exists()


def test_module_never_names_the_migrating_store(fixture_db: Path) -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    # MemoryStore.__init__ migrates and writes whatever file it opens, so the
    # metric must not be able to reach it even by accident.
    assert "MemoryStore" not in source
    assert "mode=ro" in Path(
        REPO_ROOT / "src" / "living_memory" / "replay.py"
    ).read_text(encoding="utf-8")


def test_privacy_guard_accepts_aggregates_and_rejects_content() -> None:
    check_privacy({"nodes": 1129, "rate": 0.4942, "label": "agent=ae", "note": None})

    with pytest.raises(PrivacyGuardError):
        check_privacy({"content": "x" * 201})
    with pytest.raises(PrivacyGuardError):
        check_privacy({"content": "трасса из памяти"})
    with pytest.raises(PrivacyGuardError):
        check_privacy({"content": "two\nlines"})
    with pytest.raises(PrivacyGuardError):
        check_privacy({"content": {1, 2}})


def test_report_carries_no_node_content(fixture_db: Path) -> None:
    report = _run(fixture_db, agents=["ae"])
    payload = json.dumps(report)

    check_privacy(report)
    for secret in (SCHEMA_CONTENT, GROUNDING_TRACE, SILENT_TRACE):
        assert secret not in payload
    assert "n_schema" not in payload


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def _load_cli() -> Any:
    spec = importlib.util.spec_from_file_location("trace_usage_linkage_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_writes_sorted_indented_json_with_the_contract_keys(
    fixture_db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "report.json"
    exit_code = _load_cli().main(
        [
            "--db",
            str(fixture_db),
            "--as-of",
            AS_OF,
            "--window",
            WINDOW,
            "--agent",
            "ae",
            "--context-key",
            "source=extractor",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    text = out.read_text(encoding="utf-8")
    report = json.loads(text)
    assert text == json.dumps(report, indent=2, sort_keys=True) + "\n"

    assert isinstance(report["cohorts"], list)
    for cohort in report["cohorts"]:
        for key in ("nodes", "consumed_later", "rate", "consumed_within_7d", "grounded"):
            assert key in cohort
    ids = [cohort["id"] for cohort in report["cohorts"]]
    assert ids == ["all", "agent=ae", "context.source=extractor"]
    assert report["meta"]["as_of"] == AS_OF
    assert "wrote" in capsys.readouterr().out


def test_cli_requires_a_pinned_as_of(fixture_db: Path) -> None:
    with pytest.raises(SystemExit):
        _load_cli().main(["--db", str(fixture_db), "--window", WINDOW])


def test_cli_rejects_a_missing_database(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _load_cli().main(
            ["--db", str(tmp_path / "nope.sqlite3"), "--as-of", AS_OF, "--window", WINDOW]
        )
