"""Tests for the end-to-end retrieval harness (fixture DBs only).

Runs under ``LIVING_MEMORY_EMBEDDING_BACKEND=hash`` (what ``scripts/test.sh``
exports) and must never load torch / sentence-transformers: the tail
classifier's real tokenizer is injectable and every test here passes a stub.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from living_memory import retrieval_harness as harness
from living_memory.config import MemoryConfig
from living_memory.storage import MemoryStore

SCOPE = "project:harness-fixture"
CUTOFF = "2026-01-01T00:00:00Z"

# A node whose distinctive vocabulary sits far past any plausible visible span.
TAIL_NODE_CONTENT = (
    "Deployment runbook preamble that repeats the usual boilerplate about "
    "environments, ownership, rollback windows, paging policy, and the change "
    "advisory checklist, none of which identifies anything in particular and "
    "all of which is duplicated across every runbook in the repository so that "
    "the first few hundred characters carry no discriminating signal at all. "
    "Only here, well past that preamble, does it say: the quorum wedge "
    "hypervolt latch must be disarmed before draining feed workers."
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class StubVisibleSpan:
    """Injected stand-in for the real tokenizer: a fixed visible prefix."""

    def __init__(self, visible_char_end: int) -> None:
        self._visible_char_end = visible_char_end
        self.calls: list[str] = []

    def visible_char_end(self, text: str) -> int:
        self.calls.append(text)
        return min(self._visible_char_end, len(text))


def _result_dict(
    node_id: str,
    rank: int,
    *,
    bm25: float = 0.0,
    vector: float = 0.0,
    graph: float = 0.0,
    trigger: float = 0.0,
    level: str = "trace",
    scope: str = SCOPE,
) -> dict[str, Any]:
    methods = [
        name
        for name, value in (
            ("bm25", bm25),
            ("vector", vector),
            ("graph", graph),
            ("trigger", trigger),
        )
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


def _set_created_at(store: MemoryStore, event_id: str, created_at: str) -> None:
    with store.connection as conn:
        conn.execute(
            "UPDATE recall_events SET created_at = ?, feedback_applied_at = ? WHERE id = ?",
            (created_at, created_at, event_id),
        )


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A small real ``MemoryStore`` corpus plus a goldset over it."""

    root = tmp_path_factory.mktemp("retrieval-harness")
    db_path = root / "source.sqlite3"
    store = MemoryStore(MemoryConfig(db_path=db_path))

    node_a = store.create_node(
        level="trace",
        content=(
            "alembic migration checksum zr9042 failure fixed by pinning "
            "sqlalchemy-utils 0.41 on the primary cluster"
        ),
        context={"scope": SCOPE},
    )
    node_b = store.create_node(
        level="trace",
        content="frontend toolbar palette refactor moved swatches into ColorDock component",
        context={"scope": SCOPE},
    )
    node_c = store.create_node(
        level="trace",
        content="kubernetes ingress annotation cheatsheet for certmanager wildcard certificates",
        context={"scope": SCOPE},
    )
    node_tail = store.create_node(
        level="trace",
        content=TAIL_NODE_CONTENT,
        context={"scope": SCOPE},
    )
    schema_node = store.create_node(
        level="schema",
        content=(
            "Procedure for reviewing a merge request end to end: read the diff, "
            "check the pipeline, leave blocking comments, then approve."
        ),
        context={"scope": SCOPE, "trigger": "review merge request approve"},
    )

    trace_one = store.create_node(
        level="trace",
        content=(
            "Rolled out the alembic migration checksum zr9042 failure fix by pinning "
            "sqlalchemy-utils 0.41 on the primary cluster."
        ),
        context={"scope": SCOPE},
    )
    trace_two = store.create_node(
        level="trace",
        content=(
            "Confirmed the quorum wedge hypervolt latch must be disarmed before "
            "draining feed workers during the rollout."
        ),
        context={"scope": SCOPE},
    )

    event_one = store.record_recall_event(
        query="alembic checksum failure",
        scope=SCOPE,
        requested_scope=SCOPE,
        resolved_scopes=[SCOPE, "global"],
        ambient_context={"agent": "fixture"},
        depth=1,
        max_results=5,
        results=[
            _result_dict(node_a.id, 1, bm25=0.8, vector=0.2),
            _result_dict(node_b.id, 2, vector=0.55),
            _result_dict(node_c.id, 3, bm25=0.3),
        ],
    )
    store.mark_recall_event_feedback(event_one.id, trace_one.id)
    _set_created_at(store, event_one.id, "2026-02-01T10:00:00Z")

    event_two = store.record_recall_event(
        query="quorum wedge latch",
        scope=SCOPE,
        requested_scope=SCOPE,
        resolved_scopes=[SCOPE, "global"],
        depth=1,
        max_results=5,
        results=[
            _result_dict(node_tail.id, 1, bm25=0.7, vector=0.4),
            _result_dict(node_c.id, 2, bm25=0.1),
        ],
    )
    store.mark_recall_event_feedback(event_two.id, trace_two.id)
    _set_created_at(store, event_two.id, "2026-02-02T10:00:00Z")

    # Consumed but before the cutoff: build-goldset must exclude it.
    event_old = store.record_recall_event(
        query="alembic checksum failure",
        scope=SCOPE,
        requested_scope=SCOPE,
        resolved_scopes=[SCOPE],
        depth=1,
        max_results=5,
        results=[_result_dict(node_a.id, 1, bm25=0.9)],
    )
    store.mark_recall_event_feedback(event_old.id, trace_one.id)
    _set_created_at(store, event_old.id, "2025-12-01T10:00:00Z")

    # Never consumed: build-goldset must exclude it too.
    event_unlabeled = store.record_recall_event(
        query="unconsumed recall",
        scope=SCOPE,
        requested_scope=SCOPE,
        resolved_scopes=[SCOPE],
        depth=1,
        max_results=5,
        results=[_result_dict(node_b.id, 1, vector=0.4)],
    )
    _set_created_at(store, event_unlabeled.id, "2026-02-03T10:00:00Z")

    store.close()

    ids = {
        "a": node_a.id,
        "b": node_b.id,
        "c": node_c.id,
        "tail": node_tail.id,
        "schema": schema_node.id,
        "trace_one": trace_one.id,
        "trace_two": trace_two.id,
        "event_one": event_one.id,
        "event_two": event_two.id,
        "event_old": event_old.id,
    }

    goldset_path = root / "goldset.jsonl"
    items = [
        _goldset_record(
            "content_grounded-0001",
            "alembic checksum failure",
            [ids["a"]],
            "content_grounded",
            tail=False,
            source_event_id=ids["event_one"],
        ),
        _goldset_record(
            "content_grounded-0002",
            "kubernetes ingress certmanager wildcard",
            [ids["c"]],
            "content_grounded",
            tail=False,
        ),
        _goldset_record(
            "content_grounded-0003",
            "quorum wedge hypervolt latch disarmed",
            [ids["tail"]],
            "content_grounded",
            tail=True,
            source_event_id=ids["event_two"],
        ),
        _goldset_record(
            "cross_lingual-0001",
            "ColorDock swatches toolbar palette",
            [ids["b"]],
            "cross_lingual",
            tail=False,
        ),
        _goldset_record(
            "role_query-0001",
            "review merge request approve",
            [ids["schema"]],
            "role_query",
            tail=True,
        ),
    ]
    goldset_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in items) + "\n",
        encoding="utf-8",
    )

    seed_queries_path = root / "seed-queries.json"
    seed_queries_path.write_text(
        json.dumps(
            {
                "cross_lingual": [
                    {
                        "jargon_query": "поревьювь мердж-реквест",
                        "paraphrase_query": "review merge request approve",
                        "scope": SCOPE,
                        "depth": 1,
                        "top_k": 2,
                    }
                ],
                "role_query": [
                    {
                        "query": "merge request review procedure",
                        "expected_node_ids": [schema_node.id],
                        "scope": SCOPE,
                        "depth": 1,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "root": root,
        "db_path": db_path,
        "ids": ids,
        "goldset": goldset_path,
        "seed_queries": seed_queries_path,
    }


def _goldset_record(
    query_id: str,
    query: str,
    relevant: list[str],
    stratum: str,
    *,
    tail: bool,
    source_event_id: str | None = None,
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "query": query,
        "scope": SCOPE,
        "ambient_context": None,
        "depth": 1,
        "max_results": 10,
        "relevant_node_ids": relevant,
        "stratum": stratum,
        "tail": tail,
        "source_event_id": source_event_id,
        "provenance": {"origin": "fixture"},
    }


@pytest.fixture(scope="module")
def snapshot(corpus: dict[str, Any]) -> Path:
    """Frozen snapshot of the fixture corpus, with its sidecar manifest."""

    snapshot_path = corpus["root"] / "frozen.sqlite3"
    harness.create_snapshot(corpus["db_path"], snapshot_path)
    return snapshot_path


def _file_state(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return harness.sha256_file(path), stat.st_mtime_ns, stat.st_size


def _row_counts(path: Path) -> dict[str, int]:
    connection = harness.open_readonly(path)
    try:
        return harness.snapshot_row_counts(connection)
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Snapshot discipline
# ---------------------------------------------------------------------------


def test_snapshot_manifest_records_source_identity(
    corpus: dict[str, Any], snapshot: Path
) -> None:
    manifest = json.loads(
        harness.manifest_path_for(snapshot).read_text(encoding="utf-8")
    )
    assert manifest["snapshot_sha256"] == harness.sha256_file(snapshot)
    assert manifest["source_path"] == str(corpus["db_path"])
    assert manifest["row_counts"] == _row_counts(corpus["db_path"])
    assert manifest["row_counts"]["recall_events"] == 4
    assert manifest["recall_events_created_at"]["min"] == "2025-12-01T10:00:00Z"
    assert manifest["recall_events_created_at"]["max"] == "2026-02-03T10:00:00Z"


def test_full_run_never_writes_the_source_or_the_frozen_snapshot(
    corpus: dict[str, Any], snapshot: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(corpus["db_path"])
    before_source = _file_state(source)
    before_snapshot = _file_state(snapshot)
    before_counts = _row_counts(source)

    opened: list[tuple[str, bool]] = []
    real_connect = sqlite3.connect

    def recording_connect(target, *args, **kwargs):  # type: ignore[no-untyped-def]
        opened.append((str(target), bool(kwargs.get("uri", False))))
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", recording_connect)

    exit_code = harness.main(
        [
            "run",
            "--source-db",
            str(source),
            "--goldset",
            str(corpus["goldset"]),
            "--report",
            str(tmp_path / "report.json"),
            "--markdown",
            str(tmp_path / "report.md"),
            "--agreement-sample",
            "20",
        ]
    )
    assert exit_code == 0

    # The source database is only ever reached through a read-only URI.
    source_opens = [entry for entry in opened if str(source) in entry[0]]
    assert source_opens, "the run never touched the source database"
    for target, uri in source_opens:
        assert target == f"file:{source}?mode=ro", target
        assert uri is True

    assert _file_state(source) == before_source
    assert _row_counts(source) == before_counts
    # The frozen snapshot is an input too: runs work off a temp copy of it.
    assert _file_state(snapshot) == before_snapshot


def test_working_copy_is_deleted_and_leaves_the_snapshot_untouched(snapshot: Path) -> None:
    before = _file_state(snapshot)
    with harness.working_copy(snapshot) as working_db:
        store = MemoryStore(MemoryConfig(db_path=working_db))
        store.create_node(level="trace", content="scribble", context={"scope": SCOPE})
        store.close()
        assert working_db.exists()
    assert not working_db.exists()
    assert not working_db.parent.exists()
    assert _file_state(snapshot) == before


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_runs_produce_byte_identical_metrics(
    corpus: dict[str, Any], snapshot: Path, tmp_path: Path
) -> None:
    reports = []
    for index in (1, 2):
        report_path = tmp_path / f"report-{index}.json"
        assert (
            harness.main(
                [
                    "--snapshot",  # bare form: no explicit `run` subcommand
                    str(snapshot),
                    "--goldset",
                    str(corpus["goldset"]),
                    "--report",
                    str(report_path),
                    "--seed",
                    "7",
                    "--cutoff",
                    CUTOFF,
                ]
            )
            == 0
        )
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))

    first, second = reports
    assert json.dumps(first["metrics"], sort_keys=True) == json.dumps(
        second["metrics"], sort_keys=True
    )
    assert json.dumps(first["runs"], sort_keys=True) == json.dumps(
        second["runs"], sort_keys=True
    )
    assert json.dumps(first["agreement"], sort_keys=True) == json.dumps(
        second["agreement"], sort_keys=True
    )
    # Provenance is where the wall clock lives, and it is the only difference.
    assert first["provenance"]["cutoff"] == "2026-01-01T00:00:00Z"
    assert first["provenance"]["seed"] == 7
    assert first["provenance"]["goldset_sha256"] == harness.sha256_file(corpus["goldset"])
    assert first["provenance"]["snapshot_sha256"] == harness.sha256_file(snapshot)
    assert first["provenance"]["embedding_backend"] == "hash"


def test_runs_are_ordered_by_query_id(corpus: dict[str, Any], snapshot: Path, tmp_path: Path) -> None:
    report_path = tmp_path / "ordered.json"
    assert harness.main(
        ["--snapshot", str(snapshot), "--goldset", str(corpus["goldset"]), "--report", str(report_path)]
    ) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    query_ids = [run["query_id"] for run in report["runs"]]
    assert query_ids == sorted(query_ids)


# ---------------------------------------------------------------------------
# Metrics, hand-computed
# ---------------------------------------------------------------------------


def _item(query_id: str, stratum: str, relevant: list[str], *, tail: bool) -> harness.GoldsetItem:
    return harness.GoldsetItem(
        query_id=query_id,
        query=query_id,
        scope=SCOPE,
        ambient_context=None,
        depth=1,
        max_results=10,
        relevant_node_ids=tuple(relevant),
        stratum=stratum,
        tail=tail,
        source_event_id=None,
        provenance={},
    )


def _res(
    node_id: str,
    rank: int,
    *,
    bm25: float = 0.0,
    vector: float = 0.0,
    graph: float = 0.0,
    trigger: float = 0.0,
) -> harness.HarnessResult:
    methods = tuple(
        name
        for name, value in (
            ("bm25", bm25),
            ("vector", vector),
            ("graph", graph),
            ("trigger", trigger),
        )
        if value > 0.0
    )
    return harness.HarnessResult(
        node_id=node_id,
        rank=rank,
        level="trace",
        scope=SCOPE,
        score=bm25 + vector + graph + trigger,
        bm25_score=bm25,
        vector_score=vector,
        graph_score=graph,
        trigger_score=trigger,
        methods=methods,
    )


@pytest.fixture()
def controlled_runs() -> list[harness.HarnessRun]:
    """Three runs with a ranking chosen so every metric is hand-computable."""

    return [
        harness.HarnessRun(
            item=_item("q1", "content_grounded", ["n2"], tail=False),
            results=(
                _res("n1", 1, bm25=0.9, vector=0.1),
                _res("n2", 2, bm25=0.2, vector=0.8),
                _res("n3", 3, graph=0.5),
            ),
        ),
        harness.HarnessRun(
            item=_item("q2", "cross_lingual", ["m1"], tail=True),
            results=(
                _res("m0", 1, bm25=0.1, vector=0.3),
                _res("m1", 2, trigger=0.7),
            ),
        ),
        # The relevant node is never returned: a miss, not an excluded item.
        harness.HarnessRun(
            item=_item("q3", "role_query", ["z9"], tail=True),
            results=(_res("k1", 1, bm25=0.5),),
        ),
    ]


def test_overall_and_stratum_metrics_match_hand_computation(
    controlled_runs: list[harness.HarnessRun],
) -> None:
    metrics = harness.compute_metrics(controlled_runs)

    assert metrics["items"] == 3
    assert metrics["tail_items"] == 2

    overall = metrics["overall"]
    # q1: first relevant at live rank 2 -> 1/2. q2: rank 2 -> 1/2. q3: miss -> 0.
    assert overall["events"] == 3
    assert overall["events_with_useful"] == 3
    assert overall["hit@1"] == 0.0
    assert overall["hit@5"] == pytest.approx(2 / 3, abs=5e-7)
    assert overall["hit@10"] == pytest.approx(2 / 3, abs=5e-7)
    assert overall["mrr"] == pytest.approx((0.5 + 0.5 + 0.0) / 3, abs=5e-7)
    assert overall["useful_results"] == 3

    strata = metrics["per_stratum"]
    assert set(strata) == {"content_grounded", "cross_lingual", "role_query"}
    assert strata["content_grounded"]["hit@1"] == 0.0
    assert strata["content_grounded"]["hit@5"] == 1.0
    assert strata["content_grounded"]["mrr"] == 0.5
    assert strata["cross_lingual"]["hit@5"] == 1.0
    assert strata["cross_lingual"]["mrr"] == 0.5
    # Total miss: counted, and counted as a zero.
    assert strata["role_query"]["events_with_useful"] == 1
    assert strata["role_query"]["hit@5"] == 0.0
    assert strata["role_query"]["mrr"] == 0.0

    # Tail subset is exactly q2 + q3.
    tail = metrics["tail"]
    assert tail["events"] == 2
    assert tail["events_with_useful"] == 2
    assert tail["hit@5"] == 0.5
    assert tail["mrr"] == 0.25


def test_per_channel_metrics_match_hand_computation(
    controlled_runs: list[harness.HarnessRun],
) -> None:
    metrics = harness.compute_metrics(controlled_runs)
    channels = metrics["per_channel"]

    # bm25: q1 -> n1(.9), n2(.2), n3(0) => rank 2. q2 -> m0(.1), m1(0) => rank 2.
    assert channels["bm25"]["mrr"] == pytest.approx((0.5 + 0.5) / 3, abs=5e-7)
    assert channels["bm25"]["hit@1"] == 0.0
    assert channels["bm25"]["hit@5"] == pytest.approx(2 / 3, abs=5e-7)

    # vector: q1 -> n2(.8) first => rank 1. q2 -> m0(.3), m1(0) => rank 2.
    assert channels["vector"]["mrr"] == pytest.approx((1.0 + 0.5) / 3, abs=5e-7)
    assert channels["vector"]["hit@1"] == pytest.approx(1 / 3, abs=5e-7)

    # graph: q1 -> n3(.5) first, then the zero-scored pair by live rank => n2 at 3.
    # q2 -> both zero, live-rank tie-break => m1 at 2.
    assert channels["graph"]["mrr"] == pytest.approx((1 / 3 + 0.5) / 3, abs=5e-7)
    assert channels["graph"]["hit@1"] == 0.0

    # trigger: q1 -> all zero, live order preserved => n2 at 2. q2 -> m1(.7) => rank 1.
    assert channels["trigger"]["mrr"] == pytest.approx((0.5 + 1.0) / 3, abs=5e-7)
    assert channels["trigger"]["hit@1"] == pytest.approx(1 / 3, abs=5e-7)

    attribution = metrics["channel_attribution"]
    assert attribution["items_with_relevant_hit"] == 2
    assert attribution["methods"] == {"bm25": 1, "vector": 1, "graph": 0, "trigger": 1}
    assert attribution["method_share"] == {
        "bm25": 0.5,
        "vector": 0.5,
        "graph": 0.0,
        "trigger": 0.5,
    }


def test_channel_order_breaks_ties_by_live_rank(
    controlled_runs: list[harness.HarnessRun],
) -> None:
    run = controlled_runs[0]
    assert harness.channel_order(run, "vector") == ["n2", "n1", "n3"]
    assert harness.channel_order(run, "trigger") == ["n1", "n2", "n3"]
    assert harness.live_order(run) == ["n1", "n2", "n3"]
    # n3 is graph-only and drops out of the graph-zeroed order.
    assert harness.zero_graph_order(run) == ["n1", "n2"]


def test_missing_relevant_node_becomes_a_scored_miss() -> None:
    run = harness.HarnessRun(
        item=_item("q", "content_grounded", ["present", "absent"], tail=False),
        results=(_res("present", 1, bm25=0.5), _res("other", 2, bm25=0.2)),
    )
    event = harness.to_replay_event(run)
    assert event.useful_ids == {"present", "absent"}
    # The placeholder never enters the ranking.
    assert "absent" not in harness.live_order(run)
    assert [result.node_id for result in event.results] == ["present", "other", "absent"]
    assert event.results[-1].score == 0.0


def test_end_to_end_metrics_match_an_independent_recomputation(
    corpus: dict[str, Any], snapshot: Path, tmp_path: Path
) -> None:
    """Recompute hit@k / MRR from the report's own run log, by hand."""

    report_path = tmp_path / "report.json"
    assert (
        harness.main(
            [
                "--snapshot",
                str(snapshot),
                "--goldset",
                str(corpus["goldset"]),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    hits1 = hits5 = 0
    reciprocal = 0.0
    with_relevant = 0
    for run in report["runs"]:
        relevant = set(run["relevant_node_ids"])
        assert relevant, "every goldset item has at least one relevant node"
        with_relevant += 1
        rank = next(
            (
                index + 1
                for index, node_id in enumerate(run["ranked_node_ids"])
                if node_id in relevant
            ),
            None,
        )
        if rank is None:
            continue
        reciprocal += 1.0 / rank
        hits1 += int(rank <= 1)
        hits5 += int(rank <= 5)

    overall = report["metrics"]["overall"]
    assert overall["events_with_useful"] == with_relevant
    assert overall["hit@1"] == pytest.approx(hits1 / with_relevant, abs=5e-7)
    assert overall["hit@5"] == pytest.approx(hits5 / with_relevant, abs=5e-7)
    assert overall["mrr"] == pytest.approx(reciprocal / with_relevant, abs=5e-7)


# ---------------------------------------------------------------------------
# Live-agreement sanity
# ---------------------------------------------------------------------------


def test_live_agreement_is_total_on_the_fixture(
    corpus: dict[str, Any], snapshot: Path, tmp_path: Path
) -> None:
    report_path = tmp_path / "report.json"
    assert (
        harness.main(
            [
                "--snapshot",
                str(snapshot),
                "--goldset",
                str(corpus["goldset"]),
                "--report",
                str(report_path),
                "--agreement-sample",
                "20",
            ]
        )
        == 0
    )
    agreement = json.loads(report_path.read_text(encoding="utf-8"))["agreement"]
    assert agreement["sampled_items"] == 5  # the whole fixture goldset
    assert agreement["agreement_rate"] == 1.0
    assert agreement["divergences"] == []


def test_live_agreement_reports_every_divergence(snapshot: Path) -> None:
    items = [_item("q1", "content_grounded", ["whatever"], tail=False)]
    doctored = [
        harness.HarnessRun(item=items[0], results=(_res("not-a-real-node", 1, bm25=1.0),))
    ]
    with harness.working_copy(snapshot) as working_db:
        agreement = harness.live_agreement(
            working_db, items, doctored, sample_size=5, seed=0
        )
    assert agreement["sampled_items"] == 1
    assert agreement["agreement_rate"] == 0.0
    assert agreement["divergences"][0]["query_id"] == "q1"
    assert agreement["divergences"][0]["harness_top_k"] == ["not-a-real-node"]


def test_divergence_below_one_fails_unless_a_note_is_supplied(
    corpus: dict[str, Any], snapshot: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def diverging(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "definition": harness.AGREEMENT_RULE,
            "seed": 0,
            "top_k": 5,
            "requested_sample": 20,
            "sampled_items": 2,
            "sampled_query_ids": ["a", "b"],
            "agreed_items": 1,
            "agreement_rate": 0.5,
            "divergences": [
                {"query_id": "b", "query": "q", "harness_top_k": [], "direct_top_k": ["x"]}
            ],
        }

    monkeypatch.setattr(harness, "live_agreement", diverging)
    argv = [
        "--snapshot",
        str(snapshot),
        "--goldset",
        str(corpus["goldset"]),
        "--report",
        str(tmp_path / "report.json"),
    ]
    assert harness.main(argv) == 1
    assert harness.main([*argv, "--divergence-note", "known FTS tokenizer drift"]) == 0

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["provenance"]["divergence_note"] == "known FTS tokenizer drift"
    markdown = harness.render_markdown(report)
    assert "known FTS tokenizer drift" in markdown
    assert "agreement_rate 0.500" in markdown


# ---------------------------------------------------------------------------
# Goldset schema validation
# ---------------------------------------------------------------------------


def _good_record() -> dict[str, Any]:
    return _goldset_record("content_grounded-0001", "some query", ["node-1"], "content_grounded", tail=False)


def _write_goldset(path: Path, records: list[Any]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_validator_accepts_a_good_file_and_sorts_by_query_id(tmp_path: Path) -> None:
    second = _goldset_record("role_query-0001", "another", ["node-2"], "role_query", tail=True)
    path = _write_goldset(tmp_path / "good.jsonl", [second, _good_record()])
    items = harness.load_goldset(path)
    assert [item.query_id for item in items] == [
        "content_grounded-0001",
        "role_query-0001",
    ]
    assert items[1].tail is True
    assert items[0].relevant_node_ids == ("node-1",)
    # Round-tripping through the writer preserves validity and order.
    round_trip = tmp_path / "round-trip.jsonl"
    harness.dump_goldset(items, round_trip)
    assert [item.query_id for item in harness.load_goldset(round_trip)] == [
        item.query_id for item in items
    ]


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda rec: rec.update(stratum="made_up"), "unknown stratum"),
        (lambda rec: rec.update(relevant_node_ids=[]), "relevant_node_ids"),
        (lambda rec: rec.update(tail="yes"), "tail must be a boolean"),
        (lambda rec: rec.update(tail=1), "tail must be a boolean"),
        (lambda rec: rec.pop("provenance"), "missing key(s): provenance"),
        (lambda rec: rec.pop("query_id"), "missing key(s): query_id"),
        (lambda rec: rec.update(extra_field=1), "unknown key(s): extra_field"),
        (lambda rec: rec.update(max_results=0), "max_results"),
        (lambda rec: rec.update(query=""), "query must be a non-empty string"),
        (lambda rec: rec.update(relevant_node_ids=["a", "a"]), "must not repeat"),
    ],
)
def test_validator_rejects_malformed_records_naming_the_line(
    tmp_path: Path, mutate: Any, needle: str
) -> None:
    good = _good_record()
    bad = _goldset_record("z-second", "other", ["node-9"], "cross_lingual", tail=False)
    mutate(bad)
    path = _write_goldset(tmp_path / "bad.jsonl", [good, bad])
    with pytest.raises(harness.GoldsetError) as excinfo:
        harness.load_goldset(path)
    message = str(excinfo.value)
    assert "goldset line 2:" in message
    assert needle in message


def test_validator_rejects_duplicate_query_ids(tmp_path: Path) -> None:
    path = _write_goldset(tmp_path / "dupe.jsonl", [_good_record(), _good_record()])
    with pytest.raises(harness.GoldsetError) as excinfo:
        harness.load_goldset(path)
    assert "goldset line 2: duplicate query_id" in str(excinfo.value)
    assert "first seen on line 1" in str(excinfo.value)


def test_validator_rejects_invalid_json_naming_the_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(_good_record()) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(harness.GoldsetError) as excinfo:
        harness.load_goldset(path)
    assert "goldset line 2: invalid JSON" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Tail classification
# ---------------------------------------------------------------------------


def test_token_spans_reproduce_tokenize_exactly() -> None:
    """The offset re-derivation must not drift from ``embeddings.tokenize``."""

    from living_memory.embeddings import tokenize

    samples = [
        "fooBar baz-qux alembic_migration ZR9042",
        "ColorDock/swatches.palette поревьювь МёрджReq",
        "plain english sentence with nothing special",
        TAIL_NODE_CONTENT,
    ]
    for sample in samples:
        assert [token for token, _start, _end in harness.token_spans(sample)] == tokenize(
            sample
        ), sample
        for token, start, end in harness.token_spans(sample):
            assert 0 <= start < end <= len(sample)
            assert tokenize(sample[start:end]) == [token]


def test_grounding_token_past_the_visible_span_is_tail() -> None:
    content = "alpha beta gamma delta zeta"
    tokenizer = StubVisibleSpan(10)  # visible prefix is "alpha beta"
    verdict = harness.classify_node_tail(content, "zeta", tokenizer=tokenizer)
    assert verdict.grounding_tokens == ("zeta",)
    assert verdict.visible_char_end == 10
    assert verdict.earliest_grounding_char == 23
    assert verdict.tail is True
    assert verdict.reason == harness.TAIL_REASON_BEYOND


def test_grounding_token_inside_the_visible_span_is_not_tail() -> None:
    content = "alpha beta gamma delta zeta"
    verdict = harness.classify_node_tail(content, "alpha", tokenizer=StubVisibleSpan(10))
    assert verdict.grounding_tokens == ("alpha",)
    assert verdict.earliest_grounding_char == 0
    assert verdict.tail is False
    assert verdict.reason == harness.TAIL_REASON_WITHIN


def test_every_occurrence_must_be_past_the_visible_span() -> None:
    """A repeated grounding token is judged by its *earliest* occurrence."""

    content = "zeta alpha beta gamma delta zeta"
    verdict = harness.classify_node_tail(content, "zeta", tokenizer=StubVisibleSpan(10))
    assert verdict.earliest_grounding_char == 0
    assert verdict.tail is False


def test_empty_grounding_tokens_are_not_tail() -> None:
    verdict = harness.classify_node_tail(
        "alpha beta gamma", "совершенно посторонний", tokenizer=StubVisibleSpan(4)
    )
    assert verdict.grounding_tokens == ()
    assert verdict.tail is False
    assert verdict.reason == harness.TAIL_REASON_NO_GROUNDING


def test_item_tail_requires_every_grounded_node_to_be_tail() -> None:
    contents = {
        "tail-node": "alpha beta gamma delta zeta",
        "head-node": "alpha beta gamma delta zeta",
        "unrelated": "совершенно посторонний текст",
    }
    tokenizer = StubVisibleSpan(10)

    tail, audit = harness.classify_item_tail(
        contents, ["tail-node"], "zeta", tokenizer=tokenizer
    )
    assert tail is True
    assert audit["reason"] == harness.ITEM_TAIL_REASON_TAIL
    assert audit["rule"] == harness.TAIL_RULE
    assert audit["max_seq_length"] == 128

    # One head-grounded relevant node is enough to disqualify the item.
    tail, audit = harness.classify_item_tail(
        contents, ["tail-node", "head-node"], "zeta alpha", tokenizer=tokenizer
    )
    assert tail is False
    assert audit["reason"] == harness.ITEM_TAIL_REASON_HEAD

    tail, audit = harness.classify_item_tail(
        contents, ["unrelated"], "zeta", tokenizer=tokenizer
    )
    assert tail is False
    assert audit["reason"] == harness.ITEM_TAIL_REASON_NO_GROUNDING


def test_real_tokenizer_is_never_loaded_under_the_hash_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default tail tokenizer refuses rather than silently mis-measuring.

    The env var is forced rather than inherited from ``scripts/test.sh`` so a
    bare ``pytest`` invocation cannot turn this into a torch import.
    """

    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    with pytest.raises(RuntimeError, match="tokenizer is unavailable"):
        harness.ModelVisibleSpan().visible_char_end("some content")
    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules


# ---------------------------------------------------------------------------
# build-goldset
# ---------------------------------------------------------------------------


def test_build_goldset_produces_a_valid_three_stratum_goldset(
    corpus: dict[str, Any], snapshot: Path, tmp_path: Path
) -> None:
    config = harness.GoldsetBuildConfig(
        cutoff=CUTOFF,
        seed=3,
        seed_queries=corpus["seed_queries"],
        cross_lingual_top_k=2,
    )
    out = tmp_path / "generated.jsonl"
    with harness.working_copy(snapshot) as working_db:
        items, stats = harness.build_goldset(
            working_db, config, tokenizer=StubVisibleSpan(10_000)
        )
    harness.dump_goldset(items, out)

    reloaded = harness.load_goldset(out)
    assert [item.query_id for item in reloaded] == [item.query_id for item in items]
    strata = {item.stratum for item in reloaded}
    assert strata == {"content_grounded", "cross_lingual", "role_query"}

    # Only the two consumed events after the cutoff are eligible.
    assert stats["content_grounded"]["candidate_events"] == 2
    assert stats["content_grounded"]["selected"] == len(
        [item for item in items if item.stratum == "content_grounded"]
    )

    grounded = [item for item in reloaded if item.stratum == "content_grounded"]
    assert grounded, "the fixture has grounded events after the cutoff"
    for item in grounded:
        # hit@10 is well defined even though the recorded event asked for 5.
        assert item.max_results >= harness.MIN_MAX_RESULTS
        assert item.provenance["source_max_results"] == 5
        assert item.provenance["requested_scope"] == SCOPE
        assert item.source_event_id in {corpus["ids"]["event_one"], corpus["ids"]["event_two"]}
        assert item.source_event_id != corpus["ids"]["event_old"]
        assert item.relevant_node_ids
        assert item.provenance["label"]["min_containment"] == config.min_containment

    cross = next(item for item in reloaded if item.stratum == "cross_lingual")
    assert cross.query == "поревьювь мердж-реквест"
    assert cross.provenance["paraphrase_query"] == "review merge request approve"
    assert cross.provenance["resolution"] == "paraphrase_top_k"
    assert len(cross.provenance["paraphrase_top_k"]) == len(cross.relevant_node_ids)
    assert set(cross.provenance["jargon_vector_scores"]) == set(cross.relevant_node_ids)
    # Russian jargon against English content shares no tokens by construction.
    assert cross.tail is False
    assert (
        cross.provenance["tail_classification"]["reason"]
        == harness.ITEM_TAIL_REASON_NO_GROUNDING
    )

    role = next(item for item in reloaded if item.stratum == "role_query")
    assert role.relevant_node_ids == (corpus["ids"]["schema"],)
    assert role.provenance["justification_triggers"] == {
        corpus["ids"]["schema"]: "review merge request approve"
    }


def test_build_goldset_is_reproducible_for_a_fixed_seed(
    corpus: dict[str, Any], snapshot: Path
) -> None:
    config = harness.GoldsetBuildConfig(
        cutoff=CUTOFF, seed=11, seed_queries=corpus["seed_queries"], cross_lingual_top_k=2
    )
    dumps = []
    for _ in range(2):
        with harness.working_copy(snapshot) as working_db:
            items, _stats = harness.build_goldset(
                working_db, config, tokenizer=StubVisibleSpan(10_000)
            )
        dumps.append(
            [json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=False) for item in items]
        )
    assert dumps[0] == dumps[1]


def test_build_goldset_rejects_a_role_spec_instead_of_dropping_it(
    corpus: dict[str, Any], snapshot: Path, tmp_path: Path
) -> None:
    bad_seed = tmp_path / "bad-seed.json"
    bad_seed.write_text(
        json.dumps(
            {
                "role_query": [
                    {
                        "query": "trace is not a schema",
                        # A trace node: not level='schema' and carries no trigger.
                        "expected_node_ids": [corpus["ids"]["a"]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = harness.GoldsetBuildConfig(cutoff=CUTOFF, seed=0, seed_queries=bad_seed)
    with harness.working_copy(snapshot) as working_db:
        with pytest.raises(harness.GoldsetError, match="expected 'schema'"):
            harness.build_goldset(working_db, config, tokenizer=StubVisibleSpan(10_000))


def test_build_goldset_caps_are_applied_deterministically(
    corpus: dict[str, Any], snapshot: Path
) -> None:
    config = harness.GoldsetBuildConfig(
        cutoff=CUTOFF,
        seed=5,
        seed_queries=corpus["seed_queries"],
        content_grounded_cap=1,
        cross_lingual_top_k=2,
    )
    with harness.working_copy(snapshot) as working_db:
        items, stats = harness.build_goldset(
            working_db, config, tokenizer=StubVisibleSpan(10_000)
        )
    grounded = [item for item in items if item.stratum == "content_grounded"]
    assert len(grounded) == 1
    assert stats["content_grounded"]["cap"] == 1


# ---------------------------------------------------------------------------
# Reporters
# ---------------------------------------------------------------------------


def test_markdown_spells_out_the_auditable_definitions(
    corpus: dict[str, Any], snapshot: Path, tmp_path: Path
) -> None:
    report_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    assert (
        harness.main(
            [
                "--snapshot",
                str(snapshot),
                "--goldset",
                str(corpus["goldset"]),
                "--report",
                str(report_path),
                "--markdown",
                str(markdown_path),
                "--cutoff",
                CUTOFF,
            ]
        )
        == 0
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    # The tail rule is reproduced verbatim so the report is self-auditing.
    assert harness.TAIL_RULE in markdown
    assert harness.PER_CHANNEL_RULE in markdown
    assert harness.AGREEMENT_RULE in markdown
    assert "| bucket | items | w/relevant | hit@1 | hit@5 | hit@10 | MRR |" in markdown
    for stratum in ("content_grounded", "cross_lingual", "role_query"):
        assert f"stratum:{stratum}" in markdown
    for channel in harness.CHANNELS:
        assert f"| {channel} |" in markdown
    assert harness.sha256_file(snapshot) in markdown
    assert harness.sha256_file(corpus["goldset"]) in markdown


def test_report_separates_metrics_from_provenance(
    corpus: dict[str, Any], snapshot: Path, tmp_path: Path
) -> None:
    report_path = tmp_path / "report.json"
    assert (
        harness.main(
            [
                "--snapshot",
                str(snapshot),
                "--goldset",
                str(corpus["goldset"]),
                "--report",
                str(report_path),
                "--cutoff",
                CUTOFF,
                "--seed",
                "2",
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) == {
        "metrics",
        "agreement",
        "runs",
        "anchor_annotations",
        "definitions",
        "provenance",
    }
    assert "generated_at" not in json.dumps(report["metrics"])
    provenance = report["provenance"]
    for key in (
        "source_db_path",
        "snapshot_sha256",
        "snapshot_row_counts",
        "cutoff",
        "seed",
        "git_commit",
        "goldset_sha256",
        "embedding_backend",
        "harness_version",
        "generated_at",
    ):
        assert key in provenance, key
    assert provenance["snapshot_row_counts"] == _row_counts(snapshot)
    assert provenance["harness_version"] == harness.HARNESS_VERSION
