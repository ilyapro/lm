"""Synthetic end-to-end checks for the private paired recall runner."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from living_memory.config import MemoryConfig
from living_memory.storage import MemoryStore
from scripts.recall_applicability_eval import ROOT, run_pair, score, validate_cases, wire


def test_oracle_scores_source_and_required_lookup() -> None:
    case = validate_cases({"cases": [{
        "case_id": "synthetic", "query": "opal valve pressure", "scope": "project:synthetic",
        "depth": 1, "max_results": 2, "category": "large-group",
        "required_facts": [{"text": "pressure is 31 psi", "acceptable_source_ids": ["SYNTHGOOD"]}],
    }]})[0]
    first = {"results": [
        {"node": {"id": "carrier", "content": "Evidence archive summary"}, "content_ref": {"node_id": "carrier"}},
        {"node": {"id": "noise", "content": "unrelated"}},
    ]}
    lookup = {"results": [{"id": "carrier", "content": "- SYNTHBAD: pressure is 31 psi\n- SYNTHGOOD: pressure is 31 psi"}]}
    calls = [{"tool": "memory_recall", "response": first, "response_bytes": len(wire(first)), "latency_ms": 2},
             {"tool": "memory_lookup", "response": lookup, "response_bytes": len(wire(lookup)), "latency_ms": 3}]
    result = score(case, {"calls": calls}, {
        "carrier": "- SYNTHBAD: pressure is 31 psi\n- SYNTHGOOD: pressure is 31 psi",
        "noise": "unrelated"})
    assert result["reached_fact_indexes"] == [0]
    assert result["necessary_sequence"] == ["memory_recall", "memory_lookup"]
    assert result["relevant_ranks"] == [1] and result["irrelevant_ranks"] == [2]
    assert result["response_bytes"] == len(wire(first)) + len(wire(lookup))
    lookup["results"][0]["content"] = "- SYNTHBAD: pressure is 31 psi"
    assert not score(case, {"calls": calls}, {"carrier": "- SYNTHGOOD: pressure is 31 psi"})["sufficient"]


def test_paired_processes_on_same_synthetic_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    snapshot = tmp_path / "synthetic.sqlite3"
    # Access in the creation second omits updated_at from sparse delivery;
    # access a second later includes it. Keep this snapshot older than both
    # workers, so their real response bytes cannot differ at that boundary.
    recorded_at = (datetime.now(UTC) - timedelta(minutes=1)).isoformat(timespec="seconds").replace("+00:00", "Z")
    with monkeypatch.context() as clock:
        clock.setattr("living_memory.storage._utc_now", lambda: recorded_at)
        with MemoryStore(MemoryConfig(db_path=snapshot)) as store:
            good = store.create_node(level="trace", content="Opal valve verified pressure is 31 psi.",
                                     context={"scope": "project:synthetic"})
            store.create_node(level="trace", content="Opal valve archive shipping note.",
                              context={"scope": "project:synthetic"})
            # Exclude an opportunistic sweep's measured duration_ms from this
            # equal-input check, without modifying the runner's byte accounting.
            store.set_last_decay_sweep_at(recorded_at)
    cases = validate_cases({"cases": [
        {"case_id": "found", "query": "opal valve verified pressure", "scope": "project:synthetic",
         "depth": 1, "max_results": 2, "category": "concrete",
         "required_facts": [{"text": "pressure is 31 psi", "acceptable_source_ids": [good.id]}]},
        {"case_id": "absent", "query": "zebra engine quotient", "scope": "project:synthetic",
         "depth": 1, "max_results": 2, "category": "absent", "required_facts": []},
    ]})
    report = run_pair(snapshot, cases, ROOT, ROOT)
    assert len(report["snapshot_sha256"]) == 64
    for arms in report["cases"].values():
        for key in ("reached_fact_indexes", "relevant_ranks", "irrelevant_ranks",
                    "response_bytes", "call_count", "recall_count", "lookup_count"):
            assert arms["baseline"][key] == arms["candidate"][key]
        assert arms["candidate"]["lost_fact_indexes"] == []
        assert arms["baseline"]["call_count"] >= 1
    assert report["cases"]["found"]["baseline"]["sufficient"]
    assert report["cases"]["found"]["baseline"]["relevant_ranks"]
    assert report["cases"]["absent"]["baseline"]["required_fact_count"] == 0


def test_reject_duplicate_cases_and_missing_oracle_sources() -> None:
    raw = {"case_id": "x", "query": "q", "scope": None, "depth": 1, "max_results": 1,
           "category": "concrete", "required_facts": [{"text": "a"}]}
    with pytest.raises(ValueError, match="source IDs"):
        validate_cases({"cases": [raw]})
    raw["required_facts"][0]["acceptable_source_ids"] = ["SOURCE"]
    with pytest.raises(ValueError, match="repeated"):
        validate_cases({"cases": [raw, dict(raw)]})
