"""Tests for the offline recall-replay evaluation harness (fixture DBs only)."""

from __future__ import annotations

import itertools
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from living_memory import replay
from living_memory.config import MemoryConfig
from living_memory.feedback import _method_signals
from living_memory.storage import MemoryStore

SCOPE = "project:fixture"
TRAJ_SCOPE = "project:traj"
CUTOFF = "2026-01-15T00:00:00Z"
REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "artifacts" / "replay" / "baseline.json"


def result_dict(
    node_id: str,
    rank: int,
    *,
    bm25: float = 0.0,
    vector: float = 0.0,
    graph: float = 0.0,
    trigger: float = 0.0,
    level: str = "trace",
    scope: str = SCOPE,
) -> dict:
    methods = [
        name
        for name, value in (("bm25", bm25), ("vector", vector), ("graph", graph), ("trigger", trigger))
        if value > 0.0
    ]
    return {
        "rank": rank,
        "node_id": node_id,
        "level": level,
        "scope": scope,
        "score": bm25 + vector + graph,
        "bm25_score": bm25,
        "vector_score": vector,
        "graph_score": graph,
        "trigger_score": trigger,
        "methods": methods,
        "path": [],
    }


def _set_times(store: MemoryStore, event_id: str, created_at: str, applied_at: str | None) -> None:
    with store.connection as conn:
        conn.execute(
            "UPDATE recall_events SET created_at = ?, feedback_applied_at = ? WHERE id = ?",
            (created_at, applied_at, event_id),
        )


@pytest.fixture(scope="module")
def fixture_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Small corpus with grounded/ungrounded results, graph-only evidence,
    a holdout event, an unlabeled event, and two unusable events."""

    db_path = tmp_path_factory.mktemp("replaydb") / "fixture.sqlite3"
    store = MemoryStore(MemoryConfig(db_path=db_path))

    node_a = store.create_node(
        level="trace",
        content="alembic migration checksum zr9042 failure fixed by pinning sqlalchemy-utils 0.41",
        context={"scope": SCOPE},
    )
    node_b = store.create_node(
        level="trace",
        content="frontend toolbar palette refactor moved swatches into ColorDock component",
        context={"scope": SCOPE},
    )
    node_c = store.create_node(
        level="trace",
        content="kubernetes ingress annotation cheat sheet for cert-manager wildcard certificates",
        context={"scope": SCOPE},
    )
    node_g = store.create_node(
        level="trace",
        content="rollback ordering bridge zr9042 requires draining feed workers before migration",
        context={"scope": SCOPE},
    )

    trace_1 = store.create_node(
        level="trace",
        content=(
            "Deployed the alembic migration checksum zr9042 failure fix by pinning "
            "sqlalchemy-utils 0.41 and verified rollback ordering bridge draining feed "
            "workers before migration."
        ),
        context={"scope": SCOPE},
    )
    trace_2 = store.create_node(
        level="trace",
        content=(
            "Second rollout: pinning sqlalchemy-utils 0.41 for the alembic migration "
            "checksum zr9042 failure again on staging."
        ),
        context={"scope": SCOPE},
    )
    trace_3 = store.create_node(
        level="trace",
        content=(
            "Holdout note: alembic migration checksum zr9042 failure resolved via "
            "sqlalchemy-utils 0.41 pin on the replica cluster."
        ),
        context={"scope": SCOPE},
    )

    event_1 = store.record_recall_event(
        query="alembic checksum failure",
        scope=SCOPE,
        resolved_scopes=[SCOPE, "global"],
        max_results=5,
        results=[
            result_dict(node_a.id, 1, bm25=0.8, vector=0.2),
            result_dict(node_b.id, 2, vector=0.55),
            result_dict(node_c.id, 3, bm25=0.3),
            result_dict(node_g.id, 4, graph=1.0),
        ],
    )
    store.mark_recall_event_feedback(event_1.id, trace_1.id)
    _set_times(store, event_1.id, "2026-01-01T10:00:00Z", "2026-01-01T10:05:00Z")

    event_2 = store.record_recall_event(
        query="staging rollout pin",
        scope=SCOPE,
        resolved_scopes=[SCOPE, "global"],
        max_results=5,
        results=[
            result_dict(node_b.id, 1, vector=0.6),
            result_dict(node_a.id, 2, bm25=0.7),
            result_dict(node_c.id, 3, bm25=0.2),
        ],
    )
    store.mark_recall_event_feedback(event_2.id, trace_2.id)
    _set_times(store, event_2.id, "2026-01-01T11:00:00Z", "2026-01-01T11:05:00Z")

    event_3 = store.record_recall_event(
        query="replica cluster checksum",
        scope=SCOPE,
        resolved_scopes=[SCOPE, "global"],
        max_results=5,
        results=[
            result_dict(node_c.id, 1, bm25=0.5),
            result_dict(node_a.id, 2, bm25=0.1, vector=0.3),
        ],
    )
    store.mark_recall_event_feedback(event_3.id, trace_3.id)
    _set_times(store, event_3.id, "2026-02-01T09:00:00Z", "2026-02-01T09:05:00Z")

    event_unlabeled = store.record_recall_event(
        query="unconsumed recall",
        scope=SCOPE,
        resolved_scopes=[SCOPE, "global"],
        max_results=5,
        results=[result_dict(node_b.id, 1, vector=0.4)],
    )
    _set_times(store, event_unlabeled.id, "2026-01-02T08:00:00Z", None)

    # Trajectory scope: one consumed event with a single vector-dominant result.
    node_r = store.create_node(
        level="trace",
        content="trajectory probe payload with entirely disjoint wording",
        context={"scope": TRAJ_SCOPE},
    )
    trace_7 = store.create_node(
        level="trace",
        content="Consuming note that shares no distinctive vocabulary whatsoever.",
        context={"scope": TRAJ_SCOPE},
    )
    event_7 = store.record_recall_event(
        query="trajectory probe",
        scope=TRAJ_SCOPE,
        resolved_scopes=[TRAJ_SCOPE, "global"],
        max_results=5,
        results=[result_dict(node_r.id, 1, bm25=0.2, vector=0.6, scope=TRAJ_SCOPE)],
    )
    store.mark_recall_event_feedback(event_7.id, trace_7.id)
    _set_times(store, event_7.id, "2026-01-02T12:00:00Z", "2026-01-02T12:05:00Z")

    # Unusable events: empty results (consumed) and a result missing `methods`.
    event_empty = store.record_recall_event(
        query="empty",
        scope=SCOPE,
        resolved_scopes=[SCOPE],
        max_results=5,
        results=[],
    )
    store.mark_recall_event_feedback(event_empty.id, trace_1.id)
    _set_times(store, event_empty.id, "2026-01-03T08:00:00Z", "2026-01-03T08:01:00Z")

    legacy = result_dict(node_a.id, 1, bm25=0.4)
    legacy.pop("methods")
    event_legacy = store.record_recall_event(
        query="legacy schema",
        scope=SCOPE,
        resolved_scopes=[SCOPE],
        max_results=5,
        results=[legacy],
    )
    _set_times(store, event_legacy.id, "2026-01-03T09:00:00Z", None)

    store.close()

    fixture_ids = {
        "a": node_a.id,
        "b": node_b.id,
        "c": node_c.id,
        "g": node_g.id,
        "r": node_r.id,
        "e1": event_1.id,
        "e2": event_2.id,
        "e3": event_3.id,
        "e7": event_7.id,
    }
    (db_path.parent / "ids.json").write_text(json.dumps(fixture_ids))
    return db_path


def _ids(db_path: Path) -> dict[str, str]:
    return json.loads((db_path.parent / "ids.json").read_text())


def _load_events(db_path: Path):
    connection = replay.open_readonly(db_path)
    try:
        return replay.load_replay_events(connection)
    finally:
        connection.close()


def test_loader_skips_unusable_events_gracefully(fixture_db: Path) -> None:
    events, stats = _load_events(fixture_db)
    assert stats["total_events"] == 7
    assert stats["empty_results"] == 1
    assert stats["incomplete_results"] == 1
    assert stats["usable_events"] == 5
    assert len(events) == 5
    # Chronological order and the labeled flag survive parsing.
    assert [event.created_at for event in events] == sorted(event.created_at for event in events)
    assert sum(1 for event in events if event.labeled) == 4


def test_labels_discriminate_within_events(fixture_db: Path) -> None:
    ids = _ids(fixture_db)
    events, _ = _load_events(fixture_db)
    connection = replay.open_readonly(fixture_db)
    try:
        data = replay.build_label_data(connection, events)
    finally:
        connection.close()
    summary = replay.apply_labels(events, data, replay.LabelConfig())

    by_id = {event.id: event for event in events}
    assert by_id[ids["e1"]].useful_ids == {ids["a"], ids["g"]}
    assert by_id[ids["e2"]].useful_ids == {ids["a"]}
    assert by_id[ids["e3"]].useful_ids == {ids["a"]}

    # HARD REQUIREMENT: labels must be a strict subset of results for a
    # meaningful fraction of multi-result events — here every one of them.
    discrimination = summary["discrimination"]
    assert discrimination["multi_result_events"] == 3
    assert discrimination["strict_subset_fraction"] == 1.0
    assert discrimination["all_useful_fraction"] == 0.0


def test_rerank_uses_real_ranking_path(fixture_db: Path) -> None:
    ids = _ids(fixture_db)
    events, _ = _load_events(fixture_db)
    event_2 = next(event for event in events if event.id == ids["e2"])
    event_1 = next(event for event in events if event.id == ids["e1"])

    bm25_only = replay.make_ranking_service(replay.explicit_weights(1.0, 0.0, 0.0))
    vector_only = replay.make_ranking_service(replay.explicit_weights(0.0, 1.0, 0.0))

    # bm25-only weights put A (bm25 0.7) first; vector-only drops it to a miss.
    assert replay.rerank_event(bm25_only, event_2)[0] == ids["a"]
    vector_order = replay.rerank_event(vector_only, event_2)
    assert vector_order[0] == ids["b"]
    assert ids["a"] not in vector_order  # zero score candidates leave the ranking

    # The in-rank graph rescue keeps the graph-only candidate retrievable
    # even under bm25-only weights (graph weight lifted to 0.25).
    order = replay.rerank_event(bm25_only, event_1)
    assert ids["g"] in order
    # Zeroing the graph channel removes the graph-only candidate entirely.
    zeroed = replay.rerank_event(bm25_only, event_1, zero_graph=True)
    assert ids["g"] not in zeroed
    assert ids["a"] == zeroed[0]


def test_recorded_scheme_metrics_hand_computed(fixture_db: Path) -> None:
    report = replay.run_replay(
        fixture_db,
        replay.HarnessConfig(
            cutoff=CUTOFF,
            schemes=("recorded",),
            evidence_mode="assume",
            label_sensitivity=False,
        ),
    )
    assert report["labeled_events"] == 4
    assert report["cutoff"] == CUTOFF

    overall = report["metrics"]["recorded"]["overall"]
    assert overall["events"] == 4
    assert overall["events_with_useful"] == 3
    assert overall["hit@1"] == pytest.approx(1 / 3, abs=1e-6)
    assert overall["hit@5"] == pytest.approx(1.0)
    assert overall["hit@10"] == pytest.approx(1.0)
    assert overall["mrr"] == pytest.approx(2 / 3, abs=1e-6)
    # Useful results: E1 {A, G}, E2 {A}, E3 {A}; only G is graph-only.
    assert overall["useful_results"] == 4
    assert overall["graph_unique_share"] == pytest.approx(0.25)

    zeroed = overall["graph_zeroed"]
    assert zeroed["dropped_entirely"] == pytest.approx(0.25)
    assert zeroed["dropped_top_max"] == pytest.approx(0.25)
    assert zeroed["dropped_top1"] == pytest.approx(0.0)

    train = report["metrics"]["recorded"]["train"]
    holdout = report["metrics"]["recorded"]["holdout"]
    assert train["events"] == 3 and holdout["events"] == 1
    assert train["hit@1"] == pytest.approx(0.5)
    assert holdout["mrr"] == pytest.approx(0.5)

    per_scope = overall["per_scope"]
    # Both fixture scopes fall under the min-events threshold -> "_other".
    assert set(per_scope) == {replay.OTHER_SCOPE_BUCKET}
    assert per_scope[replay.OTHER_SCOPE_BUCKET]["events"] == 4

    scoped = replay.run_replay(
        fixture_db,
        replay.HarnessConfig(
            cutoff=CUTOFF,
            schemes=("recorded",),
            evidence_mode="assume",
            label_sensitivity=False,
            min_scope_events=1,
        ),
    )
    scoped_overall = scoped["metrics"]["recorded"]["overall"]
    assert set(scoped_overall["per_scope"]) == {SCOPE, TRAJ_SCOPE}
    assert scoped_overall["per_scope"][SCOPE]["events_with_useful"] == 3


def test_weight_trajectory_mirrors_live_update_math(fixture_db: Path) -> None:
    events, _ = _load_events(fixture_db)

    trajectory = replay.WeightTrajectory(MemoryConfig(), evidence=replay.assumed_evidence)
    for _time, event in replay.reinforcement_order(events):
        trajectory.reinforce_event(event, replay.CREDIT_RULES["winner_take_all"])

    # One vector-dominant result (bm25 0.2, vector 0.6), rank 1 -> signal 1.0.
    # Winner-take-all: vector +1.0, bm25 -0.25, graph -0.25 at lr 0.05 from the
    # project family default (0.7, 0.3, 0.0):
    #   (0.6875, 0.35, 0) -> normalized (0.662651, 0.337349, 0)
    #   -> graph floor 0.05 taken from bm25 -> (0.612651, 0.337349, 0.05)
    weights = trajectory.weights[TRAJ_SCOPE]
    assert weights.bm25 == pytest.approx(0.6126506024, abs=1e-6)
    assert weights.vector == pytest.approx(0.3373493976, abs=1e-6)
    assert weights.graph == pytest.approx(0.05, abs=1e-9)
    assert trajectory.update_counts[TRAJ_SCOPE] == 1

    proportional = replay.WeightTrajectory(MemoryConfig(), evidence=replay.assumed_evidence)
    for _time, event in replay.reinforcement_order(events):
        proportional.reinforce_event(event, replay.CREDIT_RULES["proportional"])
    # Proportional credit: bm25 +0.25, vector +0.75 -> (0.7125, 0.3375, 0)
    #   -> normalized (0.678571, 0.321429, 0) -> graph floor from bm25.
    prop_weights = proportional.weights[TRAJ_SCOPE]
    assert prop_weights.bm25 == pytest.approx(0.6285714286, abs=1e-6)
    assert prop_weights.vector == pytest.approx(0.3214285714, abs=1e-6)
    assert prop_weights.graph == pytest.approx(0.05, abs=1e-9)
    # The pluggable rule genuinely changes the trajectory.
    assert prop_weights.bm25 != pytest.approx(weights.bm25, abs=1e-4)


def _credit_result(bm25: float, vector: float, graph: float) -> replay.ReplayResult:
    return replay.ReplayResult(
        node_id="credit-probe",
        rank=1,
        level="trace",
        scope="project:credit",
        score=max(bm25, vector, graph),
        bm25_score=bm25,
        vector_score=vector,
        graph_score=graph,
        trigger_score=0.0,
    )


def test_proportional_rule_is_feedback_method_signals() -> None:
    """Parity contract: CREDIT_RULES['proportional'] IS the live rule.

    feedback._method_signals is the single source of truth for proportional
    credit; the replay rule must agree with it on every (scores, signal)
    combination, including negative raw scores and zero evidence.
    """

    score_grid = (0.0, 0.05, 0.3, 0.7, 1.0, -0.2)
    signal_grid = (1.0, 0.5, 0.2, 0.0, -0.4, -1.0)
    for bm25, vector, graph in itertools.product(score_grid, repeat=3):
        for signal in signal_grid:
            expected = _method_signals(
                SimpleNamespace(bm25_score=bm25, vector_score=vector, graph_score=graph),
                signal,
            )
            actual = replay.CREDIT_RULES["proportional"](
                _credit_result(bm25, vector, graph), signal
            )
            assert actual == pytest.approx(expected), (bm25, vector, graph, signal)


def test_winner_take_all_rule_stays_frozen_for_ab_replays() -> None:
    """The pre-fix semantics live on in replay.py only; pin them exactly."""

    rule = replay.CREDIT_RULES["winner_take_all"]
    positive = rule(_credit_result(0.2, 0.6, 0.1), 1.0)
    assert positive == pytest.approx({"bm25": -0.25, "vector": 1.0, "graph": -0.25})
    negative = rule(_credit_result(0.2, 0.6, 0.1), -1.0)
    assert negative == pytest.approx({"bm25": 0.15, "vector": -1.0, "graph": 0.15})
    # Zero evidence keeps the historical bm25-dominant fallback here (the
    # live proportional rule instead yields all-zero signals).
    fallback = rule(_credit_result(0.0, 0.0, 0.0), 0.5)
    assert fallback == pytest.approx({"bm25": 0.5, "vector": -0.125, "graph": -0.125})


def test_replayed_scheme_freezes_weights_after_cutoff(fixture_db: Path) -> None:
    report = replay.run_replay(
        fixture_db,
        replay.HarnessConfig(
            cutoff="2026-01-01T00:00:00Z",  # before every event: nothing learns
            schemes=("replayed_winner_take_all",),
            evidence_mode="assume",
            label_sensitivity=False,
        ),
    )
    # All updates happen after the cutoff, so the trajectory never leaves the
    # configured family defaults.
    final = report["weights"]["replayed_final"]
    assert TRAJ_SCOPE not in final
    assert report["weights"]["replayed_update_counts"] == {}

    learned = replay.run_replay(
        fixture_db,
        replay.HarnessConfig(
            cutoff=CUTOFF,
            schemes=("replayed_winner_take_all",),
            evidence_mode="assume",
            label_sensitivity=False,
        ),
    )
    assert TRAJ_SCOPE in learned["weights"]["replayed_final"]
    assert learned["weights"]["replayed_update_counts"][SCOPE] > 0


def test_cli_end_to_end_produces_contract_report(fixture_db: Path, tmp_path: Path) -> None:
    report_path = tmp_path / "out" / "report.json"
    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "living_memory.replay",
            "--db",
            str(fixture_db),
            "--cutoff",
            CUTOFF,
            "--report",
            str(report_path),
            "--evidence",
            "assume",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    report = json.loads(report_path.read_text())
    for key in ("labeled_events", "cutoff", "metrics", "split", "labeling", "corpus"):
        assert key in report
    assert set(report["metrics"]) == {
        "recorded",
        "live_weights",
        "floor_defaults",
        "uniform",
        "replayed",
    }
    for blocks in report["metrics"].values():
        for split in ("overall", "train", "holdout"):
            block = blocks[split]
            for metric in ("hit@1", "hit@5", "hit@10", "mrr", "graph_unique_share"):
                assert metric in block
            for metric in (
                "dropped_top1",
                "dropped_top5",
                "dropped_top_max",
                "dropped_entirely",
            ):
                assert metric in block["graph_zeroed"]
            assert "per_scope" in block
    assert "label_sensitivity" in report

    markdown = report_path.with_suffix(".md").read_text()
    assert "## Metrics by scheme" in markdown
    assert "## Labeling protocol" in markdown
    assert "Candidate selection bias" in markdown


def test_replay_never_writes_to_the_database(fixture_db: Path) -> None:
    connection = replay.open_readonly(fixture_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("INSERT INTO metadata (key, value) VALUES ('x', 'y')")
    finally:
        connection.close()


@pytest.mark.skipif(not BASELINE_PATH.exists(), reason="baseline artifact not built yet")
def test_committed_baseline_satisfies_contract() -> None:
    baseline = json.loads(BASELINE_PATH.read_text())

    assert isinstance(baseline["labeled_events"], int)
    assert baseline["labeled_events"] >= 7000
    replay.normalize_cutoff(baseline["cutoff"])  # parses as ISO

    holdout = baseline["split"]["holdout"]
    assert holdout["labeled_events"] >= 1000
    assert holdout["event_share_of_usable"] >= 0.15

    schemes = baseline["metrics"]
    assert {"recorded", "live_weights", "floor_defaults", "uniform"} <= set(schemes)
    assert any(name.startswith("replayed") for name in schemes)
    for blocks in schemes.values():
        for split in ("overall", "train", "holdout"):
            block = blocks[split]
            for metric in ("hit@1", "hit@5", "hit@10", "mrr", "graph_unique_share"):
                assert metric in block
            assert block["per_scope"]
            for metric in ("dropped_top1", "dropped_top5", "dropped_top_max", "dropped_entirely"):
                assert metric in block["graph_zeroed"]

    discrimination = baseline["labeling"]["discrimination"]
    assert discrimination["strict_subset_fraction"] >= 0.2
    assert discrimination["all_useful_fraction"] <= 0.5

    markdown = BASELINE_PATH.with_suffix(".md").read_text()
    assert "## Labeling protocol" in markdown
    assert "Candidate selection bias" in markdown
