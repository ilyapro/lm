"""Tests for ``scripts/drain_near_dup_simulate.py`` and its committed artifact.

Two subjects, and they are different in kind.

The first is the harness's ONE load-bearing claim: that intercepting
``store.create_connection`` records what the real drain pass decided without
writing it. That is tested against the real pass over a real ``MemoryStore``,
not against a mock -- a recorder that silently changed the pass's behaviour
would make every number in the artifact fiction, so the test drives
``_supersede_drained_near_dups`` itself and asserts both halves: the pair came
back, and the ``connections`` table did not move.

The second is the artifact. It is committed evidence, and these tests are what
stop it drifting away from the script that made it: every pair carries a
verdict that came from ``VERDICTS``, every band summary agrees with the pair
list it summarizes, and the drain flag is enabled nowhere in the tree. The
survivor count is asserted as a *consistency* property (``automatable`` is
false exactly when a read-as-different pair survives), never pinned to a
number, so re-running the simulation after the veto changes updates the
artifact instead of breaking a test that encodes today's answer.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

from living_memory.near_dup import IDENTIFIER_VETO_ENV
from living_memory.retrieval import (
    DRAIN_NEAR_DUP_COSINE_ENV,
    DRAIN_NEAR_DUP_ENV,
    DRAIN_NEAR_DUP_KIND,
    MemoryRecallService,
)
from living_memory.storage import MemoryStore


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "drain_near_dup_simulate.py"
ARTIFACT_JSON = REPO_ROOT / "artifacts" / "near-dup" / "drain-simulation.json"
ARTIFACT_MD = REPO_ROOT / "artifacts" / "near-dup" / "drain-simulation.md"


def _load_script_module() -> Any:
    spec = importlib.util.spec_from_file_location("drain_near_dup_simulate", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


simulate = _load_script_module()


SCOPE = "project:drainsim"
ALPHA = (1.0, 0.0, 0.0)
#: cos(ALPHA, .) == 0.995: inside the 0.99 band the drain ships at.
NEAR_995 = (0.995, math.sqrt(1.0 - 0.995**2), 0.0)


class FixedEmbedder:
    """Content keyword -> a direction this test wrote down."""

    def embed(self, text: str) -> list[float]:
        return list(NEAR_995 if "repeat" in text.lower() else ALPHA)


@pytest.fixture(autouse=True)
def neutral_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test inherits a gate, a threshold or a valve from the shell."""

    monkeypatch.delenv(DRAIN_NEAR_DUP_ENV, raising=False)
    monkeypatch.delenv(DRAIN_NEAR_DUP_COSINE_ENV, raising=False)
    monkeypatch.delenv(IDENTIFIER_VETO_ENV, raising=False)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    with MemoryStore(tmp_path / "memory.sqlite3") as opened:
        yield opened


def _node(store: MemoryStore, content: str) -> str:
    node = store.create_node(
        level="trace",
        content=content,
        context={"scope": SCOPE, "agent": "tester"},
        embedding=list(FixedEmbedder().embed(content)),
    )
    return node.id


def _connection_count(store: MemoryStore) -> int:
    return int(store.connection.execute("SELECT COUNT(*) FROM connections").fetchone()[0])


# --------------------------------------------------------------------------
# The interception, against the real pass
# --------------------------------------------------------------------------


def test_recorder_captures_the_real_pass_without_writing_an_edge(
    store: MemoryStore,
) -> None:
    """The harness's whole claim, driven through the shipped drain method.

    Both halves matter. If the recorder missed a write, the artifact would
    under-report; if a write escaped it, the simulation would have mutated a
    corpus. Asserting the returned pair AND the unchanged row count is what
    distinguishes "intercepted" from "did nothing".
    """

    first = _node(store, "alpha drift measured at three units")
    second = _node(store, "alpha drift repeat measured at three units")
    # The pass offers candidates greatest-id-first, so the LEXICOGRAPHICALLY
    # smaller id bears. That is not the same as "created first": two ULIDs
    # minted inside one millisecond share their timestamp prefix and are
    # ordered by their random tail, which happens often enough in a fast test
    # to make an assertion on creation order flake.
    bearer, arrival = sorted((first, second))
    before = _connection_count(store)

    recorder = simulate.ConnectionRecorder()
    store.create_connection = recorder  # type: ignore[method-assign]
    try:
        service = MemoryRecallService(store, embedder=FixedEmbedder())
        index = service._scope_chunk_index(SCOPE, service._chunk_corpus_revision())
        nodes = [store.get_node(first), store.get_node(second)]
        written = service._supersede_drained_near_dups(SCOPE, nodes, index)
    finally:
        del store.create_connection

    assert written == [(bearer, arrival)]
    assert [(edge.bearer_id, edge.candidate_id) for edge in recorder.edges] == [
        (bearer, arrival)
    ]
    assert recorder.edges[0].relation_type == "supersedes"
    assert recorder.edges[0].metadata["kind"] == DRAIN_NEAR_DUP_KIND
    assert recorder.edges[0].metadata["scope"] == SCOPE
    assert recorder.edges[0].metadata["cosine"] == pytest.approx(0.995, abs=1e-3)
    assert _connection_count(store) == before


def test_write_guard_denies_an_edge_that_escapes_the_recorder(
    store: MemoryStore,
) -> None:
    """Belt to the recorder's braces: SQLite itself refuses the row.

    The recorder is an assignment on one object; this is the guarantee that
    holds even if some other path tried to write. It has to raise rather than
    no-op, because a silent refusal would leave the simulation reporting fewer
    pairs than the pass decided.
    """

    source = _node(store, "alpha one")
    target = _node(store, "alpha two")
    simulate.install_connection_write_guard(store)

    with pytest.raises(sqlite3.DatabaseError):
        store.create_connection(source, target, "supersedes")

    store.connection.set_authorizer(None)
    assert _connection_count(store) == 0


def test_write_guard_leaves_other_tables_alone(store: MemoryStore) -> None:
    """The guard is about ``connections``; the scratch copy still works."""

    simulate.install_connection_write_guard(store)
    try:
        node = _node(store, "alpha three")
        assert store.get_node(node) is not None
    finally:
        store.connection.set_authorizer(None)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["global.sqlite3", "global-alt.sqlite3", "alt.sqlite3"])
def test_refuses_to_name_a_live_database(tmp_path: Path, name: str) -> None:
    with pytest.raises(simulate.SimulationError, match="live database"):
        simulate.refuse_live_database(tmp_path / name)


def test_accepts_the_pre_hygiene_backup_by_name(tmp_path: Path) -> None:
    """The substrate is a backup, and a backup is not a live corpus."""

    simulate.refuse_live_database(tmp_path / "global.pre-neardup-hygiene-20260823.sqlite3")


def test_refuses_a_workdir_inside_the_live_directory(tmp_path: Path) -> None:
    substrate = tmp_path / "share" / "global.pre-neardup-hygiene-20260823.sqlite3"
    substrate.parent.mkdir(parents=True)
    substrate.touch()

    simulate.refuse_live_directory(tmp_path / "elsewhere", substrate)
    with pytest.raises(simulate.SimulationError, match="live database's directory"):
        simulate.refuse_live_directory(substrate.parent / "work", substrate)


def test_refuses_to_run_with_the_drain_gate_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process that could really drain is not allowed to simulate one."""

    monkeypatch.setenv(DRAIN_NEAR_DUP_ENV, "1")
    with pytest.raises(simulate.SimulationError, match=DRAIN_NEAR_DUP_ENV):
        simulate.refuse_enabled_drain_flag()


# --------------------------------------------------------------------------
# Band arithmetic
# --------------------------------------------------------------------------


def test_band_boundary_follows_the_pass_not_the_prose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_duplicate_map`` needs cosine STRICTLY above the threshold.

    So "the >=0.99 band" is, in code, everything the pass admits at threshold
    0.99 -- and 0.99 exactly is not admitted. The band split has to agree with
    that or the artifact would file a pair under a band the drain would never
    have produced it in.
    """

    assert simulate.band_of(0.99) == simulate.BAND_CONTEXT
    assert simulate.band_of(0.990001) == simulate.BAND_HIGH
    assert simulate.band_of(0.9899) == simulate.BAND_CONTEXT


# --------------------------------------------------------------------------
# The verdict table
# --------------------------------------------------------------------------


def test_every_verdict_is_a_known_verdict_with_a_reason() -> None:
    assert simulate.VERDICTS, "the verdict table must not be empty"
    for pair_id, (verdict, note) in simulate.VERDICTS.items():
        assert verdict in simulate.VERDICTS_ALLOWED, pair_id
        assert "->" in pair_id, pair_id
        assert len(note) > 20, f"{pair_id}: a verdict needs a stated reason"


# --------------------------------------------------------------------------
# The classes the veto structurally cannot see
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Tree 'x' completed (PASS); over 525.1s", ("completed(PASS)",)),
        ("Tree 'x' completed (FAIL); over 632.4s", ("completed(FAIL)",)),
        ("OUTCOME pass: node/path — do a thing.", ("OUTCOME pass",)),
        ("OUTCOME fail: node/path — do a thing.", ("OUTCOME fail",)),
        # The critique boilerplate quotes both words as a template. It is one
        # string, identical on both sides of every such pair, so it must not
        # read as a flip.
        ("Output VERDICT: PASS|FAIL with FIXES", ()),
        ("the acceptance pass did not fail", ()),
    ],
)
def test_outcome_polarity_marks_are_shape_based(
    text: str, expected: tuple[str, ...]
) -> None:
    assert simulate.outcome_polarity_marks(text) == expected


def test_ordinal_marks_find_hash_prefixed_numbers() -> None:
    assert simulate.ordinal_marks("Latency benchmark probe #2 at 2026-05-24") == ("#2",)
    assert simulate.ordinal_marks("Latency benchmark probe at 2026-05-24") == ()
    assert simulate.ordinal_marks("run #12 then #3") == ("#12", "#3")


def test_the_residual_classes_really_are_invisible_to_the_shipped_extractor() -> None:
    """The claim under the whole section, checked against `extract_identifiers`.

    Not a restatement of the prose: this runs the SHIPPED extractor over texts
    that differ only in a polarity flip and only in a `#N` ordinal, and asserts
    the two identifier sets come back equal. If a future widening of the
    identifier rule ever made either difference visible, this fails and the
    section stops being true before the artifact can claim it is.
    """

    from living_memory.near_dup import extract_identifiers, identifiers_absent_from

    passed = "Tree 'probe' completed (PASS); decomposed into 1x0 over 525.1s."
    failed = "Tree 'probe' completed (FAIL); decomposed into 1x0 over 525.1s."
    assert extract_identifiers(passed) == extract_identifiers(failed)
    assert identifiers_absent_from(failed, passed) == ()

    plain = "Latency benchmark probe at 2026-05-24 — synthetic timing trace."
    second = "Latency benchmark probe #2 at 2026-05-24 — synthetic timing trace."
    assert extract_identifiers(plain) == extract_identifiers(second)
    assert identifiers_absent_from(second, plain) == ()

    # And the boundary that keeps the ordinal claim exact rather than blanket:
    # from four digits on the bare number IS an identifier, so `#1234` is seen.
    wide = "Latency benchmark probe #1234 at 2026-05-24 — synthetic timing trace."
    assert identifiers_absent_from(wide, plain) == ("1234",)


def _pair_record(bearer_full: str, candidate_full: str, **overrides: Any) -> Any:
    """A ``PairRecord`` whose PREVIEWS are deliberately blank.

    Several records in the real corpus run to thousands of characters and the
    artifact only carries a 1600-character preview of each. Blanking the
    preview here is how the test proves the residual scan reads the full text:
    if it ever went back to the preview, every assertion below would find
    nothing.
    """

    fields: dict[str, Any] = {
        "pair_id": "A->B",
        "bearer_id": "A",
        "candidate_id": "B",
        "scope": SCOPE,
        "level": "trace",
        "cosine": 0.995,
        "band": simulate.BAND_HIGH,
        "bearer_chars": len(bearer_full),
        "candidate_chars": len(candidate_full),
        "bearer_text": "",
        "candidate_text": "",
        "bearer_text_truncated": True,
        "candidate_text_truncated": True,
        "identifiers_lost": (),
        "identifiers_bearer_only": (),
        "identifier_veto_blocks": False,
        "arms": [simulate.ARM_HIGH_VETO_ON],
        "bearer_full_text": bearer_full,
        "candidate_full_text": candidate_full,
    }
    fields.update(overrides)
    return simulate.PairRecord(**fields)


def test_the_residual_scan_reads_the_full_text_not_the_preview() -> None:
    tail = "filler. " * 400
    hits = simulate._residual_hits(
        _pair_record(
            f"{tail}Tree 'probe' completed (PASS); over 525.1s.",
            f"{tail}Tree 'probe' completed (FAIL); over 525.1s.",
        )
    )
    assert simulate.RESIDUAL_OUTCOME_POLARITY in hits
    assert hits[simulate.RESIDUAL_OUTCOME_POLARITY]["invisible"] is True

    hits = simulate._residual_hits(
        _pair_record(f"{tail}probe at 2026-05-24.", f"{tail}probe #2 at 2026-05-24.")
    )
    assert hits[simulate.RESIDUAL_ORDINAL]["differing_ordinals"] == ["#2"]
    assert hits[simulate.RESIDUAL_ORDINAL]["invisible"] is True


def test_a_one_sided_outcome_marker_is_not_counted_as_a_flip() -> None:
    """Only a FLIP belongs in this class, not any marker asymmetry.

    A record that states an outcome against one that never does is two
    differently-shaped records; the pair tables already cover that. Counting
    it here would pad the class an operator is asked to take seriously.
    """

    hits = simulate._residual_hits(
        _pair_record(
            "NW-9 build-plastic-core DONE (CPU reference for the plastic mode).",
            "OUTCOME pass: nw9-architecture-build/build-plastic-core — Build it.",
        )
    )
    assert simulate.RESIDUAL_OUTCOME_POLARITY not in hits


def test_a_visible_ordinal_is_reported_as_visible() -> None:
    """`#1234` extracts as an identifier, so the class must not claim it hides.

    The value of the section is that it is exact. A blanket "ordinals are
    invisible" would be wrong from four digits on, and the artifact says so
    with a measured field rather than a caveat in prose.
    """

    hits = simulate._residual_hits(
        _pair_record("probe at 2026-05-24.", "probe #1234 at 2026-05-24.")
    )
    hit = hits[simulate.RESIDUAL_ORDINAL]
    assert hit["digits_visible_to_extractor"] == ["1234"]
    assert hit["invisible"] is False


# --------------------------------------------------------------------------
# The substrate is left untouched, and that is checked rather than promised
# --------------------------------------------------------------------------


def _fingerprint(**overrides: Any) -> dict[str, Any]:
    base = {
        "path": "/tmp/backup.sqlite3",
        "size_bytes": 10,
        "mtime_ns": 1,
        "mtime": "2026-08-23T14:35:02+00:00",
        "wal_present": False,
        "shm_present": False,
    }
    base.update(overrides)
    return base


def test_substrate_fingerprint_sees_a_journal_beside_the_backup(
    tmp_path: Path,
) -> None:
    backup = tmp_path / "global.pre-neardup-hygiene-20260823.sqlite3"
    backup.write_bytes(b"x" * 16)
    assert simulate.substrate_fingerprint(backup)["wal_present"] is False
    (tmp_path / "global.pre-neardup-hygiene-20260823.sqlite3-wal").write_bytes(b"")
    assert simulate.substrate_fingerprint(backup)["wal_present"] is True


@pytest.mark.parametrize(
    "after",
    [
        _fingerprint(mtime_ns=2),
        _fingerprint(size_bytes=11),
        _fingerprint(wal_present=True),
        _fingerprint(shm_present=True),
    ],
)
def test_a_touched_substrate_aborts_the_run(after: dict[str, Any]) -> None:
    with pytest.raises(simulate.SimulationError):
        simulate.verify_substrate_untouched(_fingerprint(), after)


def test_an_untouched_substrate_passes() -> None:
    verified = simulate.verify_substrate_untouched(_fingerprint(), _fingerprint())
    assert verified["mtime_unchanged"] is True
    assert verified["wal_absent_after_run"] is True


# --------------------------------------------------------------------------
# The committed artifact
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    assert ARTIFACT_JSON.exists(), "run scripts/drain_near_dup_simulate.py"
    return json.loads(ARTIFACT_JSON.read_text("utf-8"))


def test_artifacts_exist_and_are_not_empty(artifact: dict[str, Any]) -> None:
    assert artifact["pairs"], "the simulation found no pair at all"
    assert ARTIFACT_MD.exists() and ARTIFACT_MD.stat().st_size > 0


def test_every_pair_was_read(artifact: dict[str, Any]) -> None:
    """A pair with no verdict means nobody looked at the two texts."""

    assert artifact["complete"] is True
    for pair in artifact["pairs"]:
        assert pair["verdict"] in simulate.VERDICTS_ALLOWED, pair["pair_id"]
        assert pair["verdict_note"], pair["pair_id"]
        assert simulate.VERDICTS[pair["pair_id"]][0] == pair["verdict"]


def test_every_pair_carries_the_evidence_the_goal_asks_for(
    artifact: dict[str, Any],
) -> None:
    for pair in artifact["pairs"]:
        assert pair["bearer_id"] and pair["candidate_id"]
        assert pair["scope"] and pair["level"] == "trace"
        assert 0.0 < pair["cosine"] <= 1.0
        assert pair["bearer_text"] and pair["candidate_text"]
        assert isinstance(pair["identifiers_lost_by_collapse"], list)
        assert pair["identifier_veto_blocks"] == bool(
            pair["identifiers_lost_by_collapse"]
        )


def test_band_summaries_agree_with_the_pairs_they_summarize(
    artifact: dict[str, Any],
) -> None:
    """The headline numbers are recomputed here from the pair list itself."""

    survivor_arm = {
        simulate.BAND_HIGH: simulate.ARM_HIGH_VETO_ON,
        simulate.BAND_CONTEXT: simulate.ARM_CONTEXT_VETO_ON,
    }
    for band, summary in artifact["bands"].items():
        in_band = [pair for pair in artifact["pairs"] if pair["band"] == band]
        different = [
            pair for pair in in_band if pair["verdict"] == simulate.VERDICT_DIFFERENT
        ]
        surviving = [
            pair
            for pair in different
            if survivor_arm[band] in pair["arms"]
        ]
        assert summary["would_collapse_pairs_veto_off"] == len(in_band)
        assert summary["verdict_different_facts"] == len(different)
        assert summary["different_facts_surviving_veto"] == len(surviving)
        assert summary["different_facts_surviving_ids"] == [
            pair["pair_id"] for pair in surviving
        ]
        assert summary["unverdicted"] == 0


def test_automatable_is_true_exactly_when_no_read_different_pair_survives(
    artifact: dict[str, Any],
) -> None:
    """The band decision, stated as the rule rather than as today's answer.

    Deliberately not ``assert survivors == 0``. The measured answer belongs in
    the artifact, which a re-run updates; what belongs in a test is that the
    artifact cannot claim a band is automatable while a pair someone read as
    two different facts still collapses.
    """

    for summary in artifact["bands"].values():
        expected = (
            summary["different_facts_surviving_veto"] == 0
            and summary["unverdicted"] == 0
            and summary["would_collapse_pairs_veto_off"] > 0
        )
        assert summary["automatable"] is expected


def test_the_verdict_table_describes_exactly_this_run(
    artifact: dict[str, Any],
) -> None:
    """No verdict survives that no pair in this run carries.

    Blocking a collapse leaves the candidate a root, which changes bearer
    resolution downstream, so the pair SET moves when the veto's rule moves --
    a pair id can vanish, reverse direction, or appear for the first time. A
    ledger entry with no pair behind it is a verdict about a pair this run
    never produced, i.e. a verdict carried over instead of read.
    """

    produced = {pair["pair_id"] for pair in artifact["pairs"]}
    assert set(simulate.VERDICTS) == produced


def test_the_residual_class_section_is_recomputed_from_the_pairs(
    artifact: dict[str, Any],
) -> None:
    """Counts, band membership and the invisibility claim, all re-derived.

    "Zero different_facts pairs survived" is only honest next to a statement
    of what the veto cannot see at all, so the section is checked the same way
    the bands are: recomputed from the pair list rather than trusted.
    """

    residual = artifact["residual_classes"]
    keys = {entry["key"] for entry in residual["classes"]}
    assert keys == {
        simulate.RESIDUAL_OUTCOME_POLARITY,
        simulate.RESIDUAL_ORDINAL,
    }
    by_id = {pair["pair_id"]: pair for pair in artifact["pairs"]}
    survivor_arm = {
        simulate.BAND_HIGH: simulate.ARM_HIGH_VETO_ON,
        simulate.BAND_CONTEXT: simulate.ARM_CONTEXT_VETO_ON,
    }
    for entry in residual["classes"]:
        # Every class states presence for BOTH bands, which is the thing an
        # operator reads: "absent at >=0.99" has to be an assertion, not a gap.
        assert set(entry["bands"]) == {simulate.BAND_HIGH, simulate.BAND_CONTEXT}
        # Consistency, not today's answer: the headline flag must agree with
        # the per-match evidence. Whether the classes ARE invisible is pinned
        # against the shipped extractor above, on synthetic texts, where a
        # widening of the identifier rule fails the test instead of quietly
        # flipping a number in the artifact.
        assert entry["invisibility_verified"] is all(
            match["evidence"]["invisible"]
            for summary in entry["bands"].values()
            for match in summary["matches"]
        )
        seen = 0
        for band, summary in entry["bands"].items():
            assert summary["present"] is bool(summary["pairs"])
            assert summary["pairs"] == len(summary["matches"])
            assert summary["pair_ids"] == [m["pair_id"] for m in summary["matches"]]
            seen += summary["pairs"]
            surviving = 0
            for match in summary["matches"]:
                pair = by_id[match["pair_id"]]
                assert pair["band"] == band
                assert match["verdict"] == pair["verdict"]
                assert match["identifier_veto_blocks"] == pair["identifier_veto_blocks"]
                assert match["survives_veto"] == (survivor_arm[band] in pair["arms"])
                # A block is never ON an invisible difference, so wherever the
                # evidence says invisible the block must be on some other
                # token -- which is the field the operator reads.
                if match["evidence"]["invisible"] and match["identifier_veto_blocks"]:
                    assert match["blocked_on"], match["pair_id"]
                surviving += bool(match["survives_veto"])
            assert summary["surviving_veto"] == surviving
            assert summary["blocked_by_veto_on_an_unrelated_token"] == len(
                [m for m in summary["matches"] if m["identifier_veto_blocks"]]
            )
            assert summary["surviving_different_facts"] == len(
                [
                    m
                    for m in summary["matches"]
                    if m["survives_veto"]
                    and m["verdict"] == simulate.VERDICT_DIFFERENT
                ]
            )
        assert entry["total_pairs"] == seen


def test_the_markdown_carries_the_residual_caveat_next_to_the_env_line() -> None:
    """The operator must not be able to read the env line without the caveat.

    "Zero different-facts pairs measured" and "no different-facts pair can get
    through" are different claims, and the handoff is exactly where confusing
    them would be expensive.
    """

    text = ARTIFACT_MD.read_text("utf-8")
    assert "## What the identifier veto structurally cannot see" in text
    handoff = text.split("## Operator handoff", 1)[1]
    assert f"{DRAIN_NEAR_DUP_ENV}=1" in handoff
    assert "structurally cannot see" in handoff
    assert "not a claim that no different-facts pair can get through" in handoff


def test_nothing_was_written_to_the_corpus(artifact: dict[str, Any]) -> None:
    safety = artifact["safety"]
    assert safety["connections_delta"] == 0
    assert safety["connections_before"] == safety["connections_after"]
    assert safety["edges_intercepted"] > 0
    assert safety["live_databases_opened"] == []
    assert safety["drain_gate_in_process"] == "<unset>"


def test_the_backup_was_verified_untouched_after_the_run(
    artifact: dict[str, Any],
) -> None:
    integrity = artifact["safety"]["substrate_integrity"]
    assert integrity["before"]["mtime_ns"] == integrity["after"]["mtime_ns"]
    assert integrity["before"]["size_bytes"] == integrity["after"]["size_bytes"]
    assert integrity["after"]["wal_present"] is False
    assert integrity["after"]["shm_present"] is False
    assert integrity["mtime_unchanged"] is True
    assert integrity["wal_absent_after_run"] is True
    # The substrate is the backup, never a live corpus.
    assert Path(integrity["after"]["path"]).name not in simulate.LIVE_DB_NAMES


def test_the_arms_are_the_four_the_method_describes(artifact: dict[str, Any]) -> None:
    names = [arm["name"] for arm in artifact["arms"]]
    assert names == [name for name, _threshold, _veto in simulate.ARMS]
    for arm in artifact["arms"]:
        if arm["pairs"]:
            assert arm["edge_relation"] == ["supersedes"]
            assert arm["edge_kind"] == [DRAIN_NEAR_DUP_KIND]


# --------------------------------------------------------------------------
# The flag is enabled nowhere
# --------------------------------------------------------------------------


def test_no_code_config_or_env_file_enables_the_drain() -> None:
    """The postcondition clause this node CAN close, checked by grep.

    Mentioning the variable is not enabling it -- the constant lives in
    ``retrieval.py``, the drain's own tests set it, and this simulation
    documents it. An enablement is an assignment to a truthy value outside a
    test, and there must be none.
    """

    grep = simulate.grep_drain_flag(REPO_ROOT)
    assert grep["mentions"] > 0, "the variable should at least still exist"
    assert grep["enabling_assignments_outside_tests"] == []
    assert grep["enabled_anywhere"] is False


def test_the_artifact_says_this_node_did_not_flip_the_flag(
    artifact: dict[str, Any],
) -> None:
    handoff = artifact["operator_handoff"]
    assert handoff["flag_flipped_by_this_node"] is False
    assert handoff["env_line"] == f"{DRAIN_NEAR_DUP_ENV}=1"
    assert DRAIN_NEAR_DUP_COSINE_ENV in handoff["env_block"]
    assert IDENTIFIER_VETO_ENV in handoff["env_block"]
    assert artifact["flag_grep"]["enabled_anywhere"] is False
