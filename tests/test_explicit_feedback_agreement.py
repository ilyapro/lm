"""Tests for ``scripts/explicit_feedback_agreement.py`` on synthetic stores.

The fixture store is created with the real schema (``MemoryStore``) and then
filled with raw rows: every node shares one embedding direction, so any older
undelivered node is a valid placebo twin, and content is built from words
unique to each node, so grounding is decided by construction. Honest marks
point at the node the closing trace actually wrote about; ritual marks mark
everything, or only rank 1, or arrive with no work in between.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from living_memory.storage import MemoryStore, pack_chunk_embedding

REPO = Path(__file__).resolve().parents[1]


def _load_module() -> Any:
    name = "explicit_feedback_agreement"
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


agreement = _load_module()

BASE = datetime(2026, 9, 10, tzinfo=timezone.utc)


def _ts(minutes: float) -> str:
    return (BASE + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _word(seed: str) -> str:
    digest = hashlib.sha256(seed.encode()).digest()
    return "".join(chr(ord("a") + byte % 26) for byte in digest[:9])


def _words(seed: str, count: int = 8) -> list[str]:
    return [_word(f"{seed}:{index}") for index in range(count)]


class Fixture:
    def __init__(self, path: Path, *, with_marks: bool = True) -> None:
        with MemoryStore(path):
            pass
        self.path = path
        self.connection = sqlite3.connect(path)
        if not with_marks:
            # MemoryStore creates recall_feedback_marks itself; drop it to model
            # a store that predates the explicit-feedback migration.
            self.connection.execute("DROP TABLE recall_feedback_marks")
        self.vector = pack_chunk_embedding([1.0, 0.0, 0.0, 0.0])
        self.mark_count = 0

    def node(self, node_id: str, content: str, minutes: float, transport: str | None = None, agent: str | None = None) -> None:
        context: dict[str, Any] = {}
        if transport:
            context["transport_session_id"] = transport
        if agent:
            context["agent"] = agent
        stamp = _ts(minutes)
        self.connection.execute(
            "INSERT INTO nodes (id, level, content, scope, context, timestamp, created_at, updated_at) "
            "VALUES (?, 'trace', ?, 'project:lm', ?, ?, ?, ?)",
            (node_id, content, json.dumps(context), stamp, stamp, stamp),
        )
        self.connection.execute(
            "INSERT INTO node_chunk_embeddings (id, node_id, chunk_index, dimensions, embedding, "
            "token_start, token_end, content_fingerprint, created_at, updated_at) "
            "VALUES (?, ?, 0, 4, ?, 0, 1, 'fp', ?, ?)",
            (f"chunk-{node_id}", node_id, self.vector, stamp, stamp),
        )

    def event(
        self,
        event_id: str,
        minutes: float,
        results: list[str],
        transport: str,
        *,
        query: str = "how to proceed",
        trace_id: str | None = None,
        agent: str | None = "claude-opus",
    ) -> None:
        payload = [{"node_id": node_id, "rank": rank + 1} for rank, node_id in enumerate(results)]
        self.connection.execute(
            "INSERT INTO recall_events (id, query, scope, requested_scope, ambient_context, results, "
            "agent, transport_session_id, feedback_applied, feedback_trace_id, created_at) "
            "VALUES (?, ?, 'project:lm', 'project:lm', '{}', ?, ?, ?, ?, ?, ?)",
            (event_id, query, json.dumps(payload), agent, transport, 1 if trace_id else 0, trace_id, _ts(minutes)),
        )

    def mark(self, event_id: str, node_id: str, mark: str, rank: int, minutes: float, transport: str, *, via: str, source: str | None, accepted: int = 1, agent: str = "claude-opus") -> None:
        self.mark_count += 1
        self.connection.execute(
            "INSERT INTO recall_feedback_marks (recall_event_id, node_id, mark, accepted, reject_reason, "
            "via_tool, source_id, transport_session_id, agent, rank, marked_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id, node_id, mark, accepted,
                None if accepted else "not_delivered", via, source, transport, agent, rank, _ts(minutes),
            ),
        )

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def _background(fixture: Fixture, count: int = 30) -> None:
    for index in range(count):
        fixture.node(f"bg{index}", " ".join(_words(f"bg{index}")), -1000 + index)


def _delivered(fixture: Fixture, event: int) -> list[str]:
    ids = [f"e{event}n{rank}" for rank in range(3)]
    for rank, node_id in enumerate(ids):
        fixture.node(node_id, " ".join(_words(node_id)), event * 10 - 5 + rank * 0.1)
    return ids


def build_honest(path: Path, events: int = 40) -> None:
    fixture = Fixture(path)
    _background(fixture)
    for event in range(events):
        transport = f"s{event}"
        ids = _delivered(fixture, event)
        used_rank = event % 3
        trace_id = f"t{event}"
        fixture.event(f"E{event}", event * 10, ids, transport, trace_id=trace_id)
        trace_text = " ".join(_words(ids[used_rank]) + _words(f"extra{event}", 4))
        fixture.node(trace_id, trace_text, event * 10 + 2, transport=transport, agent="claude-opus")
        fixture.mark(f"E{event}", ids[used_rank], "used", used_rank, event * 10 + 2, transport, via="memory_remember", source=trace_id)
        other = (used_rank + 1) % 3
        fixture.mark(f"E{event}", ids[other], "irrelevant", other, event * 10 + 2, transport, via="memory_remember", source=trace_id)
    # A foreign id the server rejected: counted, never scored.
    fixture.mark("E0", "bg1", "used", 0, 3, "s0", via="memory_remember", source="t0", accepted=0)
    fixture.close()


def build_ritual_all_used(path: Path, events: int = 40) -> None:
    """Recall, then recall again marking every delivered node used: no work between."""

    fixture = Fixture(path)
    _background(fixture)
    for event in range(events):
        transport = f"s{event}"
        ids = _delivered(fixture, event)
        fixture.event(f"E{event}", event * 10, ids, transport, agent="codex")
        fixture.event(f"F{event}", event * 10 + 1, [f"bg{event % 30}"], transport, agent="codex")
        for rank, node_id in enumerate(ids):
            fixture.mark(f"E{event}", node_id, "used", rank, event * 10 + 1, transport, via="memory_recall", source=f"F{event}", agent="codex")
    fixture.close()


def build_ritual_rank1(path: Path, events: int = 45) -> None:
    """Closed events where the trace wrote about a varying rank, but only rank 1 is marked."""

    fixture = Fixture(path)
    _background(fixture)
    for event in range(events):
        transport = f"s{event}"
        ids = _delivered(fixture, event)
        used_rank = event % 3
        trace_id = f"t{event}"
        fixture.event(f"E{event}", event * 10, ids, transport, trace_id=trace_id)
        fixture.node(trace_id, " ".join(_words(ids[used_rank])), event * 10 + 2, transport=transport)
        fixture.mark(f"E{event}", ids[0], "used", 0, event * 10 + 2, transport, via="memory_remember", source=trace_id)
    fixture.close()


def _report(path: Path, since: str | None = None) -> dict[str, Any]:
    connection = agreement.open_readonly(path)
    try:
        return agreement.build_report(connection, host_label="test", since=since, db_label=str(path))
    finally:
        connection.close()


def test_honest_marks_score_above_random(tmp_path: Path) -> None:
    path = tmp_path / "honest.sqlite3"
    build_honest(path)
    report = _report(path)
    falsifier = report["falsifier"]
    assert falsifier["better_than_random"] == "pass"
    assert falsifier["permutation"]["p_value"] < 0.01
    assert falsifier["placebo_subtracted_evidence_excess_ci95"][0] > 0.5
    classes = report["classes"]
    assert classes["used"]["grounded"]["grounded_rate"] == pytest.approx(1.0)
    assert classes["used"]["grounded"]["twin_rate"] == pytest.approx(0.0)
    assert classes["irrelevant"]["grounded"]["grounded_rate"] == pytest.approx(0.0)
    agreement_block = report["agreement"]
    assert agreement_block["used_minus_irrelevant_excess"] == pytest.approx(1.0)
    assert agreement_block["used_minus_irrelevant_excess_ci95"][0] > 0.9
    assert agreement_block["auc_used_vs_irrelevant"] == pytest.approx(1.0)
    assert report["marks"]["rejected"] == 1
    assert report["marks"]["scored_in_window"] == 80
    ritual = report["ritual"]
    assert not any(detector["flagged"] for detector in ritual.values())
    assert ritual["no_closing_work"]["hits"] == 0
    assert set(report["marks_by_agent"]) == {"claude"}


def test_all_used_ritual_trips_detectors_and_falsifier(tmp_path: Path) -> None:
    path = tmp_path / "ritual.sqlite3"
    build_ritual_all_used(path)
    report = _report(path)
    ritual = report["ritual"]
    assert ritual["all_delivered_marked_used"]["share"] == pytest.approx(1.0)
    assert ritual["all_delivered_marked_used"]["flagged"]
    assert ritual["no_closing_work"]["share"] == pytest.approx(1.0)
    assert ritual["no_closing_work"]["flagged"]
    assert report["falsifier"]["better_than_random"] == "fail"
    assert set(report["marks_by_agent"]) == {"codex"}


def test_rank1_only_ritual_is_flagged_and_not_better_than_random(tmp_path: Path) -> None:
    path = tmp_path / "rank1.sqlite3"
    build_ritual_rank1(path)
    report = _report(path)
    assert report["ritual"]["rank1_only_marking"]["share"] == pytest.approx(1.0)
    assert report["ritual"]["rank1_only_marking"]["flagged"]
    # Only a third of rank-1 marks are backed by evidence: exactly chance.
    assert report["falsifier"]["better_than_random"] == "fail"
    assert report["falsifier"]["permutation"]["p_value"] > 0.05


def test_session_duplicate_marks_are_detected(tmp_path: Path) -> None:
    path = tmp_path / "dup.sqlite3"
    fixture = Fixture(path)
    _background(fixture)
    ids = _delivered(fixture, 1)
    fixture.event("E1", 10, ids, "s1")
    fixture.event("E2", 20, ids, "s1")
    fixture.node("t2", " ".join(_words(ids[0])), 21, transport="s1")
    fixture.mark("E2", ids[0], "used", 0, 21, "s1", via="memory_remember", source="t2")
    fixture.mark("E1", ids[1], "used", 1, 21, "s1", via="memory_remember", source="t2")
    fixture.close()
    report = _report(path)
    detector = report["ritual"]["session_duplicate_marked"]
    assert (detector["hits"], detector["marks"]) == (1, 2)
    assert report["falsifier"]["better_than_random"] == "insufficient"


def test_missing_marks_table_reports_no_marks_and_reference_rates(tmp_path: Path) -> None:
    path = tmp_path / "nomarks.sqlite3"
    fixture = Fixture(path, with_marks=False)
    _background(fixture)
    for event in range(12):
        ids = _delivered(fixture, event)
        fixture.event(f"E{event}", event * 10, ids, f"s{event}", trace_id=f"t{event}")
        fixture.node(f"t{event}", " ".join(_words(ids[0])), event * 10 + 2, transport=f"s{event}")
    fixture.close()
    report = _report(path)
    assert report["marks"]["status"] == "no marks"
    assert report["falsifier"]["better_than_random"] == "insufficient"
    r1 = report["reference"]["r1"]
    assert r1["n"] == 12
    assert r1["grounded_rate"] == pytest.approx(1.0)
    assert r1["twin_rate"] == pytest.approx(0.0)
    assert report["reference"]["top3"]["grounded_rate"] == pytest.approx(1 / 3)
    markdown = agreement.render_markdown(report)
    assert "No marks." in markdown
    assert agreement.FALSIFIER_TEXT in markdown


def test_ab_sessions_are_excluded_and_since_applies(tmp_path: Path) -> None:
    path = tmp_path / "ab.sqlite3"
    fixture = Fixture(path, with_marks=False)
    _background(fixture)
    for event in range(4):
        ids = _delivered(fixture, event)
        fixture.event(f"E{event}", event * 10, ids, f"s{event}")
    # Session s1 wrote an A/B fixture trace; its innocent-looking recall goes too.
    fixture.node("abnode", "closing", 12, transport="s1")
    fixture.connection.execute(
        "UPDATE nodes SET context = ? WHERE id = 'abnode'",
        (json.dumps({"transport_session_id": "s1", "run": "tree-context-ab/live-1"}),),
    )
    fixture.close()
    report = _report(path)
    assert report["counts"]["events"] == 3
    assert report["counts"]["excluded_ab_events"] == 1
    later = _report(path, since=_ts(15))
    assert later["counts"]["events"] == 2


def test_pick_twin_respects_band_and_eligibility() -> None:
    cos_all = np.array([0.50, 0.505, 0.52, 0.498, 0.70])
    eligible = np.array([False, True, True, True, True])
    ids = ["delivered", "near_hi", "far", "near_lo", "other"]
    picks = {
        agreement.pick_twin(0.50, cos_all, eligible, ids, f"seed{index}") for index in range(40)
    }
    assert picks == {"near_hi", "near_lo"}
    assert agreement.pick_twin(0.9, cos_all, eligible, ids, "x") is None
    assert agreement.pick_twin(0.50, cos_all, eligible, ids, "same") == agreement.pick_twin(
        0.50, cos_all, eligible, ids, "same"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("claude:ae-dashboard-chat", "claude"),
        ("codex-root", "codex"),
        ("opencode", "opencode"),
        ("gigacode-cli", "gigacode"),
        ("ae:tree_node", "other"),
        (None, "unknown"),
    ],
)
def test_agent_type(raw: str | None, expected: str) -> None:
    assert agreement.agent_type(raw) == expected


def test_auc_counts_ties_half() -> None:
    assert agreement.auc([1.0, 1.0], [0.0, 0.0]) == pytest.approx(1.0)
    assert agreement.auc([0.5], [0.5]) == pytest.approx(0.5)
    assert agreement.auc([], [1.0]) is None


def test_cli_is_read_only_and_writes_reports(tmp_path: Path) -> None:
    path = tmp_path / "cli.sqlite3"
    build_honest(path, events=12)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    md_out = tmp_path / "out" / "r.md"
    json_out = tmp_path / "out" / "r.json"
    assert agreement.main(["--db", str(path), "--host-label", "fx", "--md", str(md_out), "--json", str(json_out)]) == 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not Path(str(path) + "-wal").exists() or Path(str(path) + "-wal").stat().st_size == 0
    assert json.loads(json_out.read_text())["host"] == "fx"
    assert "Explicit-feedback agreement check — fx" in md_out.read_text()
    with pytest.raises(sqlite3.OperationalError):
        agreement.open_readonly(path).execute("CREATE TABLE t (x)")
