"""The follow-signal census: does it classify, and does it stay honest?

Two things can go wrong with a measurement script, and they are not the same
thing.  It can classify wrongly — call a re-delivery a follow, or an
unobservable window an unfollowed one — and it can classify rightly while
overstating what the classification supports.  These tests cover both.

* **Discrimination** — a corpus built through the production write paths, with
  one window deliberately in each class, comes back with one window in each
  class, including the case where a lookup and a re-delivery land in the same
  window and the contingency has to keep both bits.
* **Tri-state honesty** — windows that closed before the first recorded lookup
  are NULL, are excluded from the ``known`` denominator, and still contribute
  their observable arms.  A database on ledger format 1 yields all-NULL and
  says why, rather than reading a missing column as "nobody followed anything".
* **Fidelity to the production readers** — the census's M/C/K is checked
  against :meth:`MemoryStore.matured_recall_history` on the same corpus, and
  its inverted-index echo probe against ``recall_map._echoes`` itself.  Both
  are reimplementations forced by the read-only boundary, and an unchecked
  reimplementation is where a census quietly stops describing production.
* **The boundary** — the corpus is opened read-only, and reading it does not
  move a byte of it.
* **The artifact** — the committed census and procedure exist, agree with each
  other, and set no default.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import importlib.util
import json
import sqlite3
import sys

import pytest

from living_memory.recall_map import _echoes
from living_memory.storage import (
    RECALL_DELIVERY_HISTORY_STATE_TABLE,
    MemoryStore,
)

REPO = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO / "artifacts" / "recall-map" / "follow-signal"


def _load_census_module() -> Any:
    """Import the script by path; ``scripts/`` is not an importable package."""

    name = "recall_map_follow_signal_census"
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered *before* execution: ``@dataclass`` resolves annotations
    # through ``sys.modules[cls.__module__]``, which is absent for a module
    # loaded by path alone.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


census = _load_census_module()

#: Every synthetic window matures long before this, so maturation is never the
#: thing under test when a class count is wrong.
DECISION_AT = datetime(2026, 3, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    path = tmp_path_factory.mktemp("census") / "corpus.sqlite3"
    return census.build_synthetic_corpus(path)


@pytest.fixture(scope="module")
def arm(synthetic: dict[str, Any]) -> dict[str, Any]:
    spec = census.CorpusSpec(
        label="synthetic", provenance="synthetic", path=Path(synthetic["path"])
    )
    return census.census_corpus(spec, decision_at=DECISION_AT)


# ----------------------------------------------------------------------
# Discrimination
# ----------------------------------------------------------------------


def test_every_class_is_represented_and_nothing_leaks_between_them(
    arm: dict[str, Any]
) -> None:
    known = arm["distribution"]["overall"]["known"]["by_ask_correlation"][
        "any_transport"
    ]["classes"]
    assert known["lookup_followed"] > 0
    assert known["ask_follow"] > 0
    assert known["redelivered_only"] > 0
    assert known["nothing"] > 0
    # The four classes partition the known windows exactly once each.
    assert sum(known.values()) == arm["volume"]["known_windows"]


def test_lookup_outranks_redelivery_in_the_class_but_not_in_the_contingency(
    arm: dict[str, Any], synthetic: dict[str, Any]
) -> None:
    """A window that is both must be classed as looked up and counted as both.

    The whole weighting question is a 2x2, and a class assignment that dropped
    the re-delivery bit of a looked-up window would silently empty the cell the
    likelihood ratio is computed from.
    """

    contingency = arm["distribution"]["overall"]["known"]["lookup_by_redelivery"]
    assert contingency["signal_1__redelivered_1"] >= 1
    classes = arm["distribution"]["overall"]["known"]["by_ask_correlation"][
        "any_transport"
    ]["classes"]
    # ...and that window is not also sitting in redelivered_only.
    assert classes["redelivered_only"] == 1


def test_ask_follow_probe_is_not_applied_to_windows_without_a_label(
    arm: dict[str, Any]
) -> None:
    probe = arm["distribution"]["overall"]["ask_follow_probe"]
    overall = arm["distribution"]["overall"]
    assert probe["applicable_windows"] == overall["map_medoid_windows"]
    assert probe["not_applicable_windows"] == overall["organic_only_windows"]
    assert probe["not_applicable_windows"] > 0


def test_map_medoid_view_is_the_map_limb_alone(arm: dict[str, Any]) -> None:
    medoid = arm["distribution"]["map_medoid_windows_only"]
    assert medoid["organic_only_windows"] == 0
    assert medoid["windows"] == arm["distribution"]["overall"]["map_medoid_windows"]


# ----------------------------------------------------------------------
# Tri-state honesty
# ----------------------------------------------------------------------


def test_pre_epoch_windows_are_null_and_stay_out_of_the_known_denominator(
    arm: dict[str, Any]
) -> None:
    volume = arm["volume"]
    assert volume["null_windows"] > 0
    assert volume["known_windows"] + volume["null_windows"] == volume["matured_windows"]
    known = arm["distribution"]["overall"]["known"]
    assert known["total"] == volume["known_windows"]
    assert sum(known["by_ask_correlation"]["any_transport"]["classes"].values()) == (
        volume["known_windows"]
    )


def test_null_windows_still_report_the_arms_that_were_observed(
    arm: dict[str, Any]
) -> None:
    """An unobservable lookup does not make the re-delivery bit unobservable.

    The pre-epoch scenario delivers a labelled cluster and then echoes its
    label, so the NULL partition must carry an ask-follow — a NULL bucket that
    reported only a total would throw away every arm the corpus did observe.
    """

    null = arm["distribution"]["overall"]["null"]["by_ask_correlation"][
        "any_transport"
    ]["classes"]
    assert "lookup_followed" not in null
    assert null["ask_follow"] >= 1
    assert sum(null.values()) == arm["volume"]["null_windows"]


def test_format_one_corpus_is_all_null_and_says_why(
    synthetic: dict[str, Any], tmp_path: Path
) -> None:
    """A ledger without the column reads as unobservable, never as unfollowed."""

    legacy = tmp_path / "legacy.sqlite3"
    legacy.write_bytes(Path(synthetic["path"]).read_bytes())
    connection = sqlite3.connect(legacy)
    connection.execute("ALTER TABLE recall_delivery_history DROP COLUMN lookup_consumed")
    connection.execute("DROP TABLE recall_lookup_events")
    connection.execute(
        f"UPDATE {RECALL_DELIVERY_HISTORY_STATE_TABLE} SET format_version = 1"
    )
    connection.commit()
    connection.close()

    arm = census.census_corpus(
        census.CorpusSpec(label="legacy", provenance="snapshot", path=legacy),
        decision_at=DECISION_AT,
    )
    assert arm["schema"]["lookup_column_present"] is False
    assert arm["schema"]["lookup_table_present"] is False
    assert arm["volume"]["known_windows"] == 0
    assert arm["volume"]["null_windows"] == arm["volume"]["matured_windows"] > 0
    assert arm["volume"]["lookup_signal"]["recorded"] is False
    assert "predates the lookup signal" in arm["volume"]["lookup_signal"]["reason"]
    # The observable arms survive the missing column.
    assert arm["distribution"]["overall"]["redelivered"] > 0


def test_zero_known_windows_never_reports_a_decided_rate(arm: dict[str, Any]) -> None:
    empty = census.wilson_interval(0, 0)
    assert empty == {"point": None, "low": None, "high": None, "half_width": None}
    # Wilson, not the normal approximation: 0/3 must not read as 0 +/- 0.
    tiny = census.wilson_interval(0, 3)
    assert tiny["point"] == 0.0 and tiny["high"] > 0.4
    assert arm["volume"]["power"]["powered"] is False
    assert census.windows_for_half_width(0.10) == 97


# ----------------------------------------------------------------------
# Fidelity to the production readers
# ----------------------------------------------------------------------


def test_census_mck_matches_the_stores_own_matured_recall_history(
    synthetic: dict[str, Any], arm: dict[str, Any]
) -> None:
    """The read-only copy of the ledger reader must agree with the reader.

    ``matured_recall_history`` needs a writable store, which the live-database
    boundary forbids, so the census reimplements the ordering and the trailing
    runs.  This is the check that keeps the copy a copy.
    """

    node_ids = sorted(synthetic["nodes"].values())
    with MemoryStore(Path(synthetic["path"])) as store:
        expected = store.matured_recall_history(node_ids, DECISION_AT)

    rows = {row["node_id"]: row["matured_history"] for row in arm["top_sticky_rows"]}
    checked = 0
    for node_id in node_ids:
        if node_id not in rows:
            continue
        theirs, ours = expected[node_id], rows[node_id]
        assert ours["available"] is theirs.available
        assert ours["matured"] == theirs.matured
        assert ours["consumed"] == theirs.consumed
        assert ours["trailing_nonconsumed"] == theirs.trailing_nonconsumed
        assert ours["lookup_consumed"] == theirs.lookup_consumed
        assert ours["lookup_known"] == theirs.lookup_known
        assert ours["lookup_trailing_absent"] == theirs.lookup_trailing_absent
        checked += 1
    assert checked == len(node_ids), "every scenario node must appear in the rows"


def test_inverted_index_probe_agrees_with_the_modules_own_echo(
    synthetic: dict[str, Any]
) -> None:
    """Exhaustive, not sampled: every phrasing against every query in range."""

    connection = census.open_readonly(Path(synthetic["path"]))
    try:
        index = census.load_events(connection)
    finally:
        connection.close()

    compared = 0
    for phrasings in index.maps.values():
        for label, hint in phrasings.values():
            for phrasing in (label, hint):
                fast = set(census.echo_matches(index, phrasing, 0, index.total))
                slow = {
                    ordinal
                    for ordinal in range(index.total)
                    if _echoes(phrasing, index.queries[ordinal])
                }
                assert fast == slow
                compared += 1
    assert compared > 0


def test_run_reports_zero_echo_disagreements(arm: dict[str, Any]) -> None:
    agreement = arm["diagnostics"]["echo_agreement"]
    assert agreement["probe_calls_replayed"] > 0
    assert agreement["disagreements"] == 0


def test_frozen_scorer_probe_records_that_the_k_tail_does_not_demote() -> None:
    """A measured claim the demotion valve's design rests on.

    If a future refit of the frozen evaluator ever gives K a demoting sign,
    this test fails and the procedure's section 2.2 has to be rewritten — which
    is the point of pinning it.
    """

    result = census.frozen_scorer_probe()
    scores = [point["score"] for point in result["curves"]["schema"]]
    assert scores == sorted(scores), "K is expected to be monotonically increasing"
    assert result["k_tail_is_demoting"] is False


# ----------------------------------------------------------------------
# The read-only boundary
# ----------------------------------------------------------------------


def test_corpora_are_opened_read_only_and_are_not_written_to(
    synthetic: dict[str, Any]
) -> None:
    path = Path(synthetic["path"])
    before = (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())

    connection = census.open_readonly(path)
    try:
        assert int(connection.execute("PRAGMA query_only").fetchone()[0]) == 1
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM recall_delivery_history")
    finally:
        connection.close()

    census.census_corpus(
        census.CorpusSpec(label="ro", provenance="snapshot", path=path),
        decision_at=DECISION_AT,
    )
    after = (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())
    assert after == before


def test_a_ledgerless_database_is_reported_unusable_rather_than_empty(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bare.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE recall_events (id TEXT)")
    connection.commit()
    connection.close()

    arm = census.census_corpus(
        census.CorpusSpec(label="bare", provenance="snapshot", path=path),
        decision_at=DECISION_AT,
    )
    assert arm["usable"] is False
    assert "recall_delivery_history" in arm["reason"]


def test_corpus_spec_requires_a_declared_provenance() -> None:
    parsed = census.CorpusSpec.parse("field:live=/tmp/x.sqlite3#a note")
    assert (parsed.provenance, parsed.label, parsed.note) == ("field", "live", "a note")
    for bad in ("live=/tmp/x.sqlite3", "guesswork:live=/tmp/x.sqlite3", "field:live="):
        with pytest.raises(Exception):
            census.CorpusSpec.parse(bad)


# ----------------------------------------------------------------------
# The committed artifact
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    return json.loads((ARTIFACT_DIR / "census.json").read_text(encoding="utf-8"))


def test_committed_artifact_is_present_and_non_empty(committed: dict[str, Any]) -> None:
    assert committed["arms"], "the census must carry at least one arm"
    procedure = (ARTIFACT_DIR / "procedure.md").read_text(encoding="utf-8")
    assert len(procedure) > 2000


def test_every_arm_declares_its_provenance_and_only_field_arms_claim_evidence(
    committed: dict[str, Any]
) -> None:
    assert any(arm["provenance"] == "field" for arm in committed["arms"])
    for arm in committed["arms"]:
        assert arm["provenance"] in {"field", "snapshot", "synthetic"}
        if arm.get("usable"):
            assert arm["field_evidence"] is (arm["provenance"] == "field")
        if arm["provenance"] == "synthetic":
            assert "NOT field evidence" in arm["note"]


def test_committed_artifact_states_its_volume_honestly(
    committed: dict[str, Any]
) -> None:
    """Under-powered is a permitted verdict; a silent one is not."""

    field_arms = [arm for arm in committed["arms"] if arm["provenance"] == "field"]
    assert field_arms
    for arm in field_arms:
        volume = arm["volume"]
        assert (
            volume["known_windows"] + volume["null_windows"]
            == volume["matured_windows"]
        )
        if not volume["power"]["powered"]:
            assert "under-powered" in volume["power"]["verdict"]
    assert committed["field_known_windows"] == sum(
        arm["volume"]["known_windows"] for arm in field_arms
    )


def test_committed_artifact_sets_no_valve_default(committed: dict[str, Any]) -> None:
    """Nothing in the artifact may read as a switched-on valve.

    The env names belong to the pool-gates node.  This artifact never mentions
    one, so that naming a valve here cannot be mistaken for enabling it.
    """

    assert committed["sets_no_defaults"] is True
    blob = json.dumps(committed) + (ARTIFACT_DIR / "procedure.md").read_text(
        encoding="utf-8"
    )
    assert "LM_RECALL_MAP_USEFULNESS" not in blob
    assert "LM_RECALL_MAP_DEMOTE" not in blob


def test_procedure_pins_the_thresholds_the_census_reports(
    committed: dict[str, Any]
) -> None:
    """The two documents must not drift apart on the numbers they share."""

    procedure = (ARTIFACT_DIR / "procedure.md").read_text(encoding="utf-8")
    assert str(census.windows_for_half_width(0.10)) in procedure
    assert str(census.windows_for_half_width(0.10, 0.1)) in procedure
    assert str(committed["frozen_scorer_probe"]["admission_threshold"]) in procedure
    for arm in committed["arms"]:
        if arm["provenance"] == "field" and arm.get("usable"):
            assert f"{arm['volume']['matured_windows']:,}".replace(",", " ") in (
                procedure
            )


def test_the_pinned_cohort_exemplar_is_reported_when_the_corpus_has_it(
    committed: dict[str, Any]
) -> None:
    for arm in committed["arms"]:
        if arm["provenance"] != "field" or not arm.get("usable"):
            continue
        pinned = [row for row in arm["top_sticky_rows"] if row["pinned"]]
        assert pinned, "the goal's named cohort node must carry its own row"
        for row in pinned:
            assert row["node_id"] in census.DEFAULT_PINNED_NODES
            # Both stickiness readings, so neither can be quoted alone.
            assert row["rank"]["by_redelivery_count"] is not None
            assert row["rank"]["by_map_offer_count"] is not None
            assert row["node"]["returned_by_recalls"] >= row["matured_history"]["rows"]
