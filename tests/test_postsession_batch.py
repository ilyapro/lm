"""Synthetic batch extraction through the dry-run CLI and field op dump."""

import json
import weakref
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

from living_memory.postsession import runner, transcripts
from living_memory.postsession.corpus import SessionEntry, probe, write_index
from living_memory.postsession.session import WriteInteraction
from postsession_support import make_record, make_remember, make_teach
from scripts import post_session_extract as cli
from scripts import postsession_field_run as field


def test_batch_report_order_and_body_lifetime(tmp_path, monkeypatch, capsys):
    # One recall followed by a patch in each synthetic transcript.
    rows = [
        {"type": "event_msg", "payload": {"type": "mcp_tool_call_end",
         "invocation": {"server": "living-memory", "tool": "memory_recall"},
         "result": {"Ok": {"structuredContent": {"recall_event_id": "event"}}}}},
        {"type": "response_item", "payload": {"type": "custom_tool_call",
         "call_id": "edit", "name": "apply_patch"}},
        {"type": "event_msg", "payload": {"type": "patch_apply_end",
         "call_id": "edit", "success": True, "changes": {"file": {"type": "add", "content": "text"}}}},
    ]
    path = tmp_path / "sample.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    entries = [probe(transcripts.TranscriptRef("codex", "codex", path, f"codex:{i}"))
               for i in range(24)]
    for entry in entries:
        entry.split = "train"
    entries[-1].path = str(tmp_path / "missing")
    index = tmp_path / "index.jsonl"
    write_index(entries, index)
    order = sorted((e.session_key for e in entries[:-1]), key=cli._sample_rank)
    broken = order[-1]
    fixtures = Path(__file__).parent / "fixtures" / "postsession"
    out = tmp_path / "report.json"

    references = []
    counts = []

    class TrackedRecord(transcripts.SessionRecord):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            references.append(weakref.ref(self))
            counts.append(sum(ref() is not None for ref in references))

    monkeypatch.setattr(transcripts, "SessionRecord", TrackedRecord)
    attestations = runner.proposed_attestations

    def attest(record):
        if record.session_key == broken:
            raise ValueError("synthetic stage failure")
        if record.session_key in order[:workers]:
            barrier.wait(timeout=10)
        return attestations(record)

    monkeypatch.setattr(runner, "proposed_attestations", attest)
    decide = runner.ExtractionRunner.decide_and_execute

    def at_barrier(self, proposals):
        assert len(references) == len(proposals) == len(order)
        assert all(ref() is None for ref in references)
        return decide(self, proposals)

    monkeypatch.setattr(runner.ExtractionRunner, "decide_and_execute", at_barrier)
    reports = []
    for workers in (1, 2, 4):
        barrier = Barrier(workers)
        references.clear()
        counts.clear()
        assert cli.main([
            "--split", "train", "--index", str(index), "--max-sessions", "0",
            "--workers", str(workers), "--stages", "attest", "--judge", "none",
            "--baseline", str(fixtures / "gate-baseline.json"),
            "--gate-verdict", str(fixtures / "gate-control.json"), "--max-verdict-age-days", "100000",
            "--state-dir", str(tmp_path / "state"), "--out", str(out), "--max-total-attests", "2",
        ]) == 0
        assert max(counts) == workers
        assert f"load failed {entries[-1].session_key}" in capsys.readouterr().err
        report = json.loads(out.read_text())
        assert [s["session_key"] for s in report["sessions"]] == order
        assert report["sessions"][-1]["errors"] == ["attest: synthetic stage failure"]
        assert report["selection"]["sessions_loaded"] == 23
        assert report["selection"]["sessions_failed_to_load"] == 1
        assert [d["session_key"] for d in report["decisions"]] == order[:-1]
        assert [d["status"] for d in report["decisions"]] == ["would_write"] * 2 + ["dropped"] * 20
        del report["generated_at"], report["gate"]["details"]["checked_at"]
        del report["gate"]["details"]["verdict_age_days"]
        reports.append(report)
    assert reports[0] == reports[1] == reports[2]


def test_field_dump_preserves_attributed_payloads(tmp_path, monkeypatch):
    record = make_record(transport_session_ids=())
    record.writes = [WriteInteraction(0, "remember", record.span(1), transport_session_id="source")]
    ops = [make_teach(record, "original", "corrected"), make_remember(record, "observed")]
    config = runner.RunnerConfig(max_total_ops=1)
    expected = [runner.enforce_attribution(op, record, task=config.task) for op in ops]

    (tmp_path / "preregistration.json").write_text("{}")
    monkeypatch.setattr(field, "DEFAULT_STATE_DIR", tmp_path)
    monkeypatch.setattr(field, "evaluate_gate", lambda *a, **kw: runner.GateDecision("PASS"))
    entry = SessionEntry(record.session_key, record.source, record.cli, record.path)
    monkeypatch.setattr(field, "_select_sessions", lambda *a: ([entry], {"eligible": 1}))
    monkeypatch.setattr(field, "_runner_config", lambda: config)
    monkeypatch.setattr(field, "load_session", lambda entry: record)
    monkeypatch.setattr(runner.ExtractionRunner, "propose",
                        lambda self, record: runner.SessionProposals(record, ops=ops))

    assert field.cmd_extract(SimpleNamespace(split="eval", state_dir=tmp_path)) == 0
    dump = json.loads((tmp_path / "ops-eval.json").read_text())["ops"]
    assert [item["status"] for item in dump] == ["would_write", "dropped"]
    assert [item["payload"] for item in dump] == [op.payload for op in expected]
    assert [item["fingerprint"] for item in dump] == [runner.op_fingerprint(op) for op in expected]
    for item, op in zip(dump, expected, strict=True):
        assert item["session_key"] == record.session_key
        assert item["provenance"] == {
            "session_key": record.session_key, "source_span": op.provenance.source_span,
            "transcript_sha256": record.transcript_sha256,
        }
