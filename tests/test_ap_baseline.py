"""Current offline delivery and ranking metrics on a disposable corpus."""

import importlib.util
import json
from pathlib import Path

import pytest


def _load_baseline():
    script = Path(__file__).resolve().parents[1] / "scripts" / "ap_baseline.py"
    spec = importlib.util.spec_from_file_location("ap_baseline_test", script)
    ap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ap)
    return ap


def test_compare_keeps_ordinary_metrics_without_experimental_state(tmp_path, capsys):
    ap = _load_baseline()
    node = {
        "type": "node", "id": "survey", "level": "trace", "scope": "project:synthetic",
        "content_surrogate": "Harbour survey: the pilot countersigns on Tuesday.",
        "timestamp": "2099-01-01T00:00:00Z", "relations": [],
    }
    events = [{
        "type": "event", "id": f"event-{index}", "class": "automatic",
        "query_surrogate": "harbour survey", "scope": "project:synthetic",
        "requested_scope": "project:synthetic", "resolved_scopes": ["project:synthetic"],
        "created_at": f"2099-01-01T00:00:0{index}Z", "transport_session_id": "reader",
        "max_results": 1, "feedback_applied": False,
        "results": [{"node_id": "survey", "scope": "project:synthetic", "rank": 1,
                     "score": 1.0, "bm25_score": 1.0, "methods": ["bm25"]}],
    } for index in range(2)]
    (tmp_path / "dev.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [node, *events]))
    assert ap.main(["compare", "--corpus-root", str(tmp_path), "--split", "dev"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report["metrics"]) == set(ap.METRIC_FAMILIES)
    auto = report["metrics"]["auto_recall"]
    assert auto["after"]["delivery_classes"] == {"full": 1, "session_duplicate": 1}
    assert auto["after"]["content_bearing_duplicates"] == 0
    assert report["metrics"]["payload"]["population"]["all_split_events_replayed"] == 2
    assert "repeat_gating" not in auto


@pytest.mark.parametrize("direction", ["in", "out"])
@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"kind": "drain_near_duplicate"}, False),
        ({"kind": "duplicate_content"}, False),
        ({"by": "lm_collapse_near_dups.py", "reason": "near-duplicate collapse"}, False),
        ({"kind": "teach"}, True),
        ({"kind": "unknown"}, True),
        ({"by": "lm_collapse_near_dups.py"}, True),
        ({}, True),
        (None, True),
    ],
)
def test_snapshot_corrections_respect_available_metadata(direction, metadata, expected):
    ap = _load_baseline()
    relation = {"type": "supersedes", "direction": direction,
                "other_id": "other", "created_at": "2099-01-01T00:00:00Z"}
    if metadata is not None:
        relation["metadata"] = metadata
    nodes = {"node": {"id": "node", "relations": [relation, relation]}}

    edges = ap.supersedes_edges(nodes)

    pair = ("node", "other") if direction == "out" else ("other", "node")
    assert list(edges) == [(*pair, "2099-01-01T00:00:00Z")] * int(expected)
