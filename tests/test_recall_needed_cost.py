"""Synthetic contract tests for the frozen paired recall-cost reader.

No private packet, snapshot, query, or holdout answer is loaded here.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import recall_needed_cost as runner


FIXTURES = json.loads((runner.ROOT / "artifacts/recall-needed-cost/fixtures.json").read_text())
NODES = {node["key"]: node for node in FIXTURES["nodes"]}
CASES = {case["id"]: case for case in FIXTURES["cases"]}


def _node(key, content=None):
    source = NODES[key]
    return {"id": source["id"], "content": source["content"] if content is None else content}


def _replay(monkeypatch, tmp_path, case_ids, delivered, looked_up=None, recall_batches=None):
    """Run the real worker loop against deterministic synthetic tool responses."""
    import living_memory
    from living_memory import server as server_module

    looked_up = delivered if looked_up is None else looked_up
    lookup_requests = []
    recalls = iter(recall_batches) if recall_batches is not None else None

    def make_server(_path, mcp_factory):
        def recall(**_query):
            nodes = next(recalls) if recalls is not None else delivered
            return {"results": [{"node": node} for node in nodes]}

        def lookup(*, node_ids):
            lookup_requests.append(node_ids)
            return {"results": [node for node in looked_up if node["id"] in node_ids]}

        return SimpleNamespace(tools={"memory_recall": recall, "memory_lookup": lookup})

    monkeypatch.setattr(server_module, "create_mcp_server", make_server)
    # Pytest imports the repo-root namespace shim before the worker starts.
    # A standalone archive worker imports the package from src instead.
    monkeypatch.setattr(living_memory, "__file__", str(runner.ROOT / "src/living_memory/__init__.py"))
    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps({"schema_version": 1, "cases": [CASES[key] for key in case_ids]}))
    snapshot = tmp_path / "snapshot.sqlite3"
    snapshot.write_bytes(b"synthetic snapshot marker")
    out = tmp_path / "worker.json"
    runner._worker(argparse.Namespace(source_root=runner.ROOT, packet=packet, snapshot=snapshot,
                                      snapshot_sha256=runner.sha256(snapshot), out=out))
    return json.loads(out.read_text())["cases"], lookup_requests


def test_short_complete_response_needs_no_lookup_and_missing_control_stays_missing(monkeypatch, tmp_path):
    rows, lookups = _replay(monkeypatch, tmp_path, ["synthetic-short-a", "synthetic-missing"],
                            [_node("short_a")])
    short, absent = rows
    assert short["oracle_pass"] and short["sufficient"]
    assert short["responses"] == [{"tool": "memory_recall", "chars": runner.response_chars(
        {"results": [{"node": _node("short_a")}]})}]
    assert (short["recall_calls"], short["lookup_calls"], short["calls"]) == (1, 0, 1)
    assert short["ordered_retrieval_ids"] == [[NODES["short_a"]["id"]]]
    assert absent["expected"] == "missing" and not absent["sufficient"]
    assert (absent["recall_calls"], absent["lookup_calls"]) == (1, 0)
    assert lookups == []


def test_long_source_lookup_is_counted_through_sufficiency(monkeypatch, tmp_path):
    stub = _node("buried", "Evidence: archive rebuild cases. …")
    full = _node("buried")
    rows, requests = _replay(monkeypatch, tmp_path, ["synthetic-buried"], [stub], [full])
    row = rows[0]
    assert row["oracle_pass"] and row["sufficient"]
    assert requests == [[full["id"]]]
    assert [part["tool"] for part in row["responses"]] == ["memory_recall", "memory_lookup"]
    assert [part["chars"] for part in row["responses"]] == [
        runner.response_chars({"results": [{"node": stub}]}),
        runner.response_chars({"results": [full]}),
    ]
    assert row["response_chars"] == sum(part["chars"] for part in row["responses"])
    assert (row["recall_calls"], row["lookup_calls"], row["calls"]) == (1, 1, 2)
    assert row["latency_ms"] == pytest.approx(row["recall_ms"] + row["lookup_ms"], abs=0.001)


def test_every_recall_is_counted_before_later_source_lookup(monkeypatch, tmp_path):
    case = {**CASES["synthetic-buried"], "queries": [
        {"query": "archive index", "scope": "project:synthetic"},
        {"query": "sealed archive manifest", "scope": "project:synthetic"},
    ]}
    monkeypatch.setitem(CASES, "two-recalls", case)
    stub = _node("buried", "Evidence: archive rebuild cases. …")
    rows, requests = _replay(monkeypatch, tmp_path, ["two-recalls"], [stub], [_node("buried")],
                             recall_batches=[[_node("short_b")], [stub]])
    row = rows[0]
    assert row["oracle_pass"] and row["sufficient"]
    assert requests == [[NODES["buried"]["id"]]]
    assert [part["tool"] for part in row["responses"]] == [
        "memory_recall", "memory_recall", "memory_lookup"]
    assert (row["recall_calls"], row["lookup_calls"], row["calls"]) == (2, 1, 3)
    assert row["ordered_retrieval_ids"] == [[NODES["short_b"]["id"]], [NODES["buried"]["id"]]]


@pytest.mark.parametrize(
    ("case_id", "source_key", "damaged"),
    [
        ("synthetic-buried", "buried", "Evidence: archive rebuild cases. Old logs only."),
        ("synthetic-correction", "new", "Correction: amber exports now require a signed manifest."),
    ],
)
def test_oracle_detects_deliberate_instruction_and_correction_losses(
    monkeypatch, tmp_path, case_id, source_key, damaged
):
    rows, requests = _replay(monkeypatch, tmp_path, [case_id], [_node(source_key, damaged)])
    row = rows[0]
    assert requests == [[NODES[source_key]["id"]]]
    assert not row["sufficient"] and not row["oracle_pass"]
    assert row["remaining_requirements"] > 0
    assert (row["recall_calls"], row["lookup_calls"]) == (1, 1)


def test_correction_requires_explicit_negation_even_with_old_source(monkeypatch, tmp_path):
    old = _node("old")
    weakened = _node("new", "Amber exports now require a signed manifest.")
    rows, _ = _replay(monkeypatch, tmp_path, ["synthetic-correction"], [old, weakened])
    assert not rows[0]["oracle_pass"]
    rows, _ = _replay(monkeypatch, tmp_path, ["synthetic-correction"], [old, _node("new")])
    assert rows[0]["oracle_pass"] and rows[0]["sufficient"]


def test_paired_report_separates_ranking_from_presentation(monkeypatch, tmp_path):
    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps({"schema_version": 1, "snapshot_sha256": "unused",
                                  "cases": [CASES["synthetic-short-a"]]}))
    snapshot = tmp_path / "snapshot.sqlite3"
    snapshot.write_bytes(b"same bytes for both arms")
    fixtures = runner.ROOT / "artifacts/recall-needed-cost/fixtures.json"
    packet_data = json.loads(packet.read_text())
    packet_data["snapshot_sha256"] = runner.sha256(snapshot)
    packet.write_text(json.dumps(packet_data))
    revisions = iter(("baseline-commit", "candidate-commit"))
    monkeypatch.setattr(runner, "_revision", lambda _ref: next(revisions))
    monkeypatch.setattr(runner, "_archive", lambda _ref, destination: destination.mkdir())

    def fake_worker(command, **_kwargs):
        arm = Path(command[command.index("--source-root") + 1]).name
        source = NODES["short_a"]["id"]
        row = {"id": "synthetic-short-a", "split": "development", "response_chars": 100 if arm == "baseline" else 60,
               "calls": 1, "recall_calls": 1, "lookup_calls": 0, "latency_ms": 2.0, "oracle_pass": True,
               "ordered_retrieval_ids": [[source] if arm == "baseline" else [source, "other"]]}
        Path(command[command.index("--out") + 1]).write_text(json.dumps({"cases": [row]}))

    monkeypatch.setattr(runner.subprocess, "run", fake_worker)
    out, aggregate = tmp_path / "report.json", tmp_path / "aggregate.json"
    runner._run(argparse.Namespace(packet=packet, packet_sha256=runner.sha256(packet),
                                   fixtures=fixtures, fixtures_sha256=runner.sha256(fixtures),
                                   snapshot=snapshot, snapshot_sha256=runner.sha256(snapshot),
                                   baseline_ref="base", candidate_ref="candidate", out=out,
                                   aggregate_out=aggregate, timeout=3))
    report = json.loads(out.read_text())
    totals = json.loads(aggregate.read_text())
    assert report["ranking_changed_cases"] == 1
    assert report["cases"][0]["ranking_changed"]
    assert totals["by_split"]["development"]["baseline"]["response_chars"] == 100
    assert totals["by_split"]["development"]["candidate"]["response_chars"] == 60
    assert totals["by_split"]["development"]["candidate"]["calls"] == 1


def test_hash_mismatch_rejects_before_any_revision_or_worker(monkeypatch, tmp_path):
    packet = tmp_path / "packet.json"
    packet.write_text("{}")
    snapshot = tmp_path / "snapshot.sqlite3"
    snapshot.write_bytes(b"snapshot")
    fixtures = runner.ROOT / "artifacts/recall-needed-cost/fixtures.json"

    def unexpected(_ref):
        pytest.fail("hash failure must precede revision resolution")

    monkeypatch.setattr(runner, "_revision", unexpected)
    with pytest.raises(ValueError, match="frozen hash mismatch"):
        runner._run(argparse.Namespace(packet=packet, packet_sha256="0" * 64,
                                       fixtures=fixtures, fixtures_sha256=runner.sha256(fixtures),
                                       snapshot=snapshot, snapshot_sha256=runner.sha256(snapshot),
                                       baseline_ref="base", candidate_ref="candidate", out=tmp_path / "out.json",
                                       aggregate_out=None, timeout=3))
