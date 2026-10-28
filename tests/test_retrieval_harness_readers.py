"""Qualification survives evaluation caches and consumers of harness reports."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from living_memory import retrieval_harness as harness
from living_memory.postsession.counterfactual import _as_replay_results
from living_memory.postsession.usage_metric import _replay_results
from living_memory.replay import explicit_weights, run_scheme


def _script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runs() -> list[harness.HarnessRun]:
    item = harness.GoldsetItem(
        query_id="reference", query="checkpoint cursor", scope="project:fixture",
        ambient_context=None, depth=1, max_results=5, relevant_node_ids=("source",),
        stratum="content_grounded", tail=False, source_event_id=None,
        provenance={"recorded_created_at": "2026-01-01T00:00:00Z"},
    )
    result = harness.HarnessResult(
        node_id="source", rank=1, level="trace", scope="project:fixture", score=0.9,
        bm25_score=1.0, vector_score=0.9, graph_score=0.0, trigger_score=0.0,
        methods=("bm25", "vector"), withheld="below_threshold",
    )
    return [
        harness.HarnessRun(item=item, results=(result,)),
        harness.HarnessRun(
            item=replace(item, query_id="qualified"),
            results=(replace(result, withheld=None),),
        ),
    ]


def test_cached_runs_keep_qualification_and_raw_weight_replay(runs) -> None:
    script = _script("replay_post_chunking")
    payload = json.loads(json.dumps(script._runs_payload(runs)))
    restored = script._runs_from_payload(payload, [run.item for run in runs])
    assert harness.compute_metrics(restored) == harness.compute_metrics(runs)
    assert harness.compute_metrics(restored)["overall"]["mrr"] == 0.5
    events = script.dated_replay_events(restored)
    resolver = explicit_weights(1 / 3, 1 / 3, 1 / 3)
    scheme = script.Scheme(
        name="uniform", kind="static", resolver=resolver, description="uniform weights", weights=None
    )
    assert script.first_relevant_ranks(scheme, events) == {"qualified": 1, "reference": 1}
    raw = run_scheme("uniform", events, {}, "2026-02-01T00:00:00Z", resolver=resolver)
    assert raw.to_dict()["overall"]["mrr"] == 1.0
    assert raw.to_dict()["overall"]["result_basis"] == "raw_candidates"
    del payload["reference"][0]["withheld"]
    with pytest.raises(KeyError, match="withheld"):
        script._runs_from_payload(payload, [run.item for run in runs])


def test_anchor_headroom_and_subset_report_count_only_qualified_hits(runs, tmp_path) -> None:
    report = {"runs": [run.to_dict() for run in runs], "metrics": harness.compute_metrics(runs)}
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    anchors = _script("anchor_grounding_estimate")
    reach = {"per_item": [
        {"query_id": run.item.query_id, "oracle": True, "top10": True} for run in runs
    ]}
    headroom = anchors.headroom(path, reach)
    assert headroom["overall"]["depth1"] == {
        "baseline_hit": 0.5, "missed": 1, "misses_anchor_reachable_oracle": 1,
        "misses_anchor_reachable_top10": 1, "structural_ceiling": 1.0, "matched_ceiling": 1.0,
    }
    tokenizer = _script("cyrillic_tokenizer_report")
    metrics = tokenizer.harness_metrics(report, {"reference"})
    assert metrics["overall"]["mrr"] == 0.5
    assert metrics["russian_query"]["mrr"] == 0.0
    assert metrics["non_russian_query"]["mrr"] == 1.0

    # Reports which discarded the verdict cannot establish qualified hits.
    del report["metrics"]["overall"]["result_basis"]
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="qualification-aware"):
        anchors.headroom(path, reach)
    with pytest.raises(ValueError, match="qualification-aware"):
        tokenizer.harness_metrics(report, set())


def test_postsession_adapters_preserve_the_source_verdict(runs) -> None:
    for run in runs:
        expected = run.results[0].withheld
        assert _as_replay_results(run)[0].withheld == expected
        assert _replay_results([result.to_dict() for result in run.results])[0].withheld == expected
