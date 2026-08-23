from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_recall_map_relevance_report.py"
MANIFEST = ROOT / "artifacts" / "recall-map" / "relevance" / "dataset-manifest.json"
ANALYSIS = ROOT / "artifacts" / "recall-map" / "relevance" / "feature-analysis.json"
REPORT = ROOT / "artifacts" / "recall-map" / "relevance" / "feature-analysis.md"
EVALUATOR = ROOT / "scripts" / "recall_map_relevance_eval.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("recall_map_relevance_report_tested", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R = _load_module()


def _artifacts() -> tuple[dict[str, object], dict[str, object]]:
    return json.loads(MANIFEST.read_text()), json.loads(ANALYSIS.read_text())


def _rebind(manifest: dict[str, object], analysis: dict[str, object]) -> None:
    analysis["dataset_manifest_sha256"] = R._sha256_text(R.canonical_json(manifest))


def test_canonical_aggregate_inputs_reproduce_expanded_report_byte_for_byte(
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    outputs = []
    for hash_seed in ("1", "987654321"):
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--manifest",
                str(MANIFEST),
                "--analysis",
                str(ANALYSIS),
            ],
            cwd=tmp_path,
            env=environment,
            check=True,
            capture_output=True,
        )
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1] == REPORT.read_bytes()
    assert len(outputs[0]) == 25_252


def test_rendering_objects_never_reads_the_tracked_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, analysis = _artifacts()

    def reject_read(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("render_report must not read any file")

    monkeypatch.setattr(Path, "read_text", reject_read)
    rendered = R.render_report(manifest, analysis)
    assert rendered.startswith("# Historical recall-map relevance evidence\n")
    assert "feature-analysis.md" not in SCRIPT.read_bytes().decode()


def test_alternate_bound_corpus_receipt_renders_instead_of_keying_off_frozen_identity() -> None:
    manifest, analysis = _artifacts()
    manifest["as_of"] = "2031-02-03T04:05:06Z"
    analysis["as_of"] = manifest["as_of"]
    source = manifest["sources"][0]
    source["logical_name"] = "alternate"
    source["redacted_path"] = "$SNAPSHOT/alternate.sqlite3"
    source["parts"][0]["bytes"] = 12_345
    source["parts"][0]["sha256"] = "a" * 64
    source["snapshot_sha256"] = "b" * 64
    source["sqlite"]["schema_sha256"] = "c" * 64
    _rebind(manifest, analysis)

    rendered = R.render_report(manifest, analysis)

    assert "pinned at `2031-02-03T04:05:06Z`" in rendered
    assert "`$SNAPSHOT/alternate.sqlite3`" in rendered
    assert "12,345 bytes" in rendered
    assert f"SHA-256 `{'a' * 64}`" in rendered
    assert rendered.encode() != REPORT.read_bytes()


def test_representative_aggregate_mutations_change_their_rendered_sections() -> None:
    manifest, analysis = _artifacts()
    baseline = R.render_report(manifest, analysis)

    cohort_manifest = deepcopy(manifest)
    cohort_analysis = deepcopy(analysis)
    cohort_manifest["train"]["events"] = 1_041
    _rebind(cohort_manifest, cohort_analysis)
    cohort_report = R.render_report(cohort_manifest, cohort_analysis)
    assert (
        "| Organic train | sole fitting population | 369 | 1,041 | 3,102 |"
        in cohort_report
    )
    assert cohort_report != baseline

    association_analysis = deepcopy(analysis)
    association_analysis["associations"]["node_age_log_days"]["organic_train"].update(
        mean_consumed=9.87654321,
        difference=7.43185872,
    )
    association_report = R.render_report(manifest, association_analysis)
    assert "9.87654321 / 2.44468449; +7.43185872" in association_report
    assert association_report != baseline

    model_analysis = deepcopy(analysis)
    model_analysis["primary_model"]["coefficients"]["level_schema"] = 0.5
    model_report = R.render_report(manifest, model_analysis)
    assert "| `level_schema` | 0.2643455835 | 0.4409841221 | +0.5000000000 |" in model_report
    assert model_report != baseline

    ballast_analysis = deepcopy(analysis)
    ballast_analysis["content_forms"]["observed_map_eval"]["file_chunk_envelope"][
        "items"
    ] = 31
    ballast_report = R.render_report(manifest, ballast_analysis)
    assert "| Observed-map evaluation | 652 (57) | 31 (0) | 4 (0)" in ballast_report
    assert "contains 35 classified ballast items" in ballast_report
    assert ballast_report != baseline

    privacy_analysis = deepcopy(analysis)
    privacy_analysis["privacy_audit"]["raw_strings_compared"] = 110_314
    privacy_report = R.render_report(manifest, privacy_analysis)
    assert "privacy audit compared 110,314 raw strings" in privacy_report
    assert privacy_report != baseline


def test_manifest_mutation_requires_analysis_rebinding() -> None:
    manifest, analysis = _artifacts()
    manifest["train"]["events"] = 1_041
    with pytest.raises(R.ReportError, match="does not bind"):
        R.render_report(manifest, analysis)


def test_candidate_holdout_paths_and_access_claim_fail_closed(tmp_path: Path) -> None:
    prohibited = (
        tmp_path
        / "artifacts"
        / "recall-map"
        / "relevance"
        / "field"
        / "manifest.json"
    )
    with pytest.raises(R.ReportError, match="candidate-holdout access is prohibited"):
        R._assert_not_holdout(prohibited)

    manifest, analysis = _artifacts()
    analysis["leakage_audit"]["candidate_holdout_accessed"] = True
    with pytest.raises(R.ReportError, match="non-access assertion"):
        R.render_report(manifest, analysis)


def test_cli_writes_the_expanded_report_without_changing_historical_inputs(
    tmp_path: Path,
) -> None:
    before = {
        path: path.read_bytes()
        for path in (EVALUATOR, MANIFEST, ANALYSIS, REPORT)
    }
    output = tmp_path / "expanded.md"

    assert R.main(["--manifest", str(MANIFEST), "--analysis", str(ANALYSIS), "--out", str(output)]) == 0
    assert output.read_bytes() == REPORT.read_bytes()
    assert {path: path.read_bytes() for path in before} == before
