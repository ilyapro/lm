"""CLI checks for the aggregate-only paired report contract; no private data is read."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import recall_needed_cost as runner


FAMILIES = [
    ("development", "known_large_carrier"),
    ("development", "ordinary_short_fact"),
    ("development", "ordinary_short_fact"),
    ("development", "buried_applicable_instruction"),
    ("development", "obsolete_rule_with_correction"),
    ("development", "missing_knowledge"),
    ("holdout", "ordinary_short_fact"),
    ("holdout", "buried_applicable_instruction"),
    ("holdout", "missing_knowledge"),
]
FIELDS = ("response_chars", "calls", "recall_calls", "lookup_calls", "latency_ms", "oracle_pass")


def recalc(report):
    totals = {arm: {key: 0 for key in FIELDS} for arm in ("baseline", "candidate")}
    splits = {split: {arm: {key: 0 for key in FIELDS} for arm in totals}
              for split in ("development", "holdout")}
    for case in report["cases"]:
        for arm in totals:
            row = case[arm]
            row["response_chars"] = sum(item["chars"] for item in row["responses"])
            row["calls"] = len(row["responses"])
            row["recall_calls"] = sum(item["tool"] == "memory_recall" for item in row["responses"])
            row["lookup_calls"] = sum(item["tool"] == "memory_lookup" for item in row["responses"])
            for key in FIELDS:
                value = int(row[key]) if key == "oracle_pass" else row[key]
                totals[arm][key] += value
                splits[case["split"]][arm][key] += value
        case["delta_chars"] = case["candidate"]["response_chars"] - case["baseline"]["response_chars"]
        case["delta_calls"] = case["candidate"]["calls"] - case["baseline"]["calls"]
    report["totals"] = totals
    report["by_split"] = splits
    return report


def good_report():
    cases = []
    for index, (split, family) in enumerate(FAMILIES, 1):
        arms = {}
        for arm in ("baseline", "candidate"):
            chars = 500 if arm == "baseline" else (100 if index in (1, 8) else 500)
            arms[arm] = {"expected": "missing" if family == "missing_knowledge" else "sufficient",
                         "sufficient": family != "missing_knowledge", "oracle_pass": True,
                         "remaining_requirements": 0,
                         "responses": [{"tool": "memory_recall", "chars": chars}],
                         "recall_ms": 2.0, "lookup_ms": 0.0, "latency_ms": 2.0}
        cases.append({"number": index, "split": split, "family": family,
                      "ranking_identical": True, **arms})
    report = {"schema_version": 1, "method": "Frozen paired replay with independent oracle.",
        "limitations": ["Latency is descriptive."], "verdict": "PASS: measured acceptance met.", "inputs": {
        "private_packet_sha256": "a54df4594910edb62d421fe7eec1add260022a6779809cdf710eef5619fd5d64",
        "snapshot_sha256": "2891a098b23c7ad0ef2a1c64a8bdfdd9dadccdccf4e2693afd31b09e55f09e69",
        "synthetic_fixtures_sha256": "972bd2e0e33baba2d2a03d2c0238deb46eae92bf303582943bbaf042f285b293"},
        "revisions": {"baseline": "cd4ad6c3984c41badc55199299532199ed4579b1",
                      "candidate": "d3f4c26af2c6af5a72fbb177ecf53b42a469439a"},
        "population": {"cases": 9, "development": 6, "holdout": 3,
                       "positive": 7, "missing_knowledge": 2},
        "production_complexity": {"merge_base": "cd4ad6c3984c41badc55199299532199ed4579b1",
            "files": ["src/living_memory/delivery.py", "src/living_memory/server.py"],
            "baseline": {"lines": 200, "functions": 10, "ast_branch_nodes": 20},
            "candidate": {"lines": 230, "functions": 11, "ast_branch_nodes": 22},
            "delta": {"lines": 30, "functions": 1, "ast_branch_nodes": 2},
            "git_diff": {"insertions": 35, "deletions": 5}},
        "independent_oracle": {"all_cases_pass": True, "all_positive_sufficient": True,
                               "loss_probe": "Buried instruction and correction loss rejected.",
                               "targeted_tests_passed": 8},
        "acceptance": {key: True for key in (
            "aggregate_chars_reduced", "all_required_knowledge_preserved", "measured_cost_claim_passes",
            "no_forced_lookup_cost_increase", "no_mandatory_calls_added_on_sufficient_cases",
            "ranking_identical")},
        "ranking_changed_cases": 0, "cases": cases}
    return recalc(report)


def cli(tmp_path, report):
    path = tmp_path / "aggregate.json"
    content = json.dumps(report)
    path.write_text(content)
    result = subprocess.run([sys.executable, str(runner.ROOT / "scripts/recall_needed_cost.py"),
                             "verify", "--report", str(path)], cwd=runner.ROOT,
                            capture_output=True, text=True)
    assert path.read_text() == content  # verify is read-only
    return result


def test_valid_aggregate_passes_read_only(tmp_path):
    result = cli(tmp_path, good_report())
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("damage,expected", [
    (lambda d: d.update(schema_version=2), "schema_version"),
    (lambda d: d["inputs"].update(snapshot_sha256="0" * 64), "snapshot_sha256"),
    (lambda d: d["revisions"].update(candidate="0" * 40), "revisions"),
    (lambda d: d["production_complexity"].update(merge_base="0" * 40), "merge_base"),
    (lambda d: d["production_complexity"]["delta"].update(lines=0), "delta.lines"),
    (lambda d: d["production_complexity"].pop("git_diff"), "git_diff"),
    (lambda d: d["cases"].pop(), "nine frozen cases"),
    (lambda d: d["cases"][7].update(split="development"), "split"),
    (lambda d: d["cases"][3].update(family="ordinary_short_fact"), "family"),
    (lambda d: d["cases"][4]["candidate"].update(oracle_pass=False), "oracle_pass"),
    (lambda d: d["cases"][4]["candidate"].update(remaining_requirements=1), "remaining_requirements"),
    (lambda d: d["cases"][5]["candidate"].update(sufficient=True), "sufficient"),
    (lambda d: d["cases"][0].update(ranking_identical=False), "ranking identity"),
    (lambda d: d.update(ranking_changed_cases=1), "ranking_changed_cases"),
    (lambda d: d["cases"][0]["candidate"].update(response_chars=1), "response_chars"),
    (lambda d: d["cases"][0]["candidate"].update(calls=2), "calls"),
    (lambda d: d["cases"][0]["candidate"].update(latency_ms=4), "latency"),
    (lambda d: d["cases"][0]["candidate"].update(recall_ms=0, latency_ms=0), "latency evidence"),
    (lambda d: d["totals"]["candidate"].update(response_chars=1), "totals.candidate.response_chars"),
    (lambda d: d["by_split"]["holdout"]["baseline"].update(calls=2), "by_split.holdout.baseline.calls"),
    (lambda d: d["acceptance"].update(measured_cost_claim_passes=False), "acceptance"),
    (lambda d: d["independent_oracle"].update(loss_probe=""), "loss probe"),
    (lambda d: d.pop("method"), "method"),
    (lambda d: d.update(verdict="FAIL: stale claim"), "verdict"),
])
def test_rejects_incomplete_or_inconsistent_report(tmp_path, damage, expected):
    report = copy.deepcopy(good_report())
    damage(report)
    result = cli(tmp_path, report)
    assert result.returncode != 0
    assert expected in result.stderr


def test_aggregate_saving_cannot_hide_large_carrier_lookup(tmp_path):
    report = good_report()
    case = report["cases"][0]
    case["candidate"]["responses"] = [
        {"tool": "memory_recall", "chars": 100},
        {"tool": "memory_lookup", "chars": 700},
    ]
    case["candidate"]["lookup_ms"] = 1.0
    case["candidate"]["latency_ms"] = 3.0
    recalc(report)
    assert report["totals"]["candidate"]["response_chars"] < report["totals"]["baseline"]["response_chars"]
    assert cli(tmp_path, report).returncode != 0


def test_rejects_one_case_saving_even_with_lower_aggregate(tmp_path):
    report = good_report()
    report["cases"][7]["candidate"]["responses"][0]["chars"] = 500
    recalc(report)
    assert report["totals"]["candidate"]["response_chars"] < report["totals"]["baseline"]["response_chars"]
    assert "only one case" in cli(tmp_path, report).stderr


def test_rejects_absent_aggregate_saving(tmp_path):
    report = good_report()
    report["cases"][2]["candidate"]["responses"][0]["chars"] = 1300
    recalc(report)
    assert "aggregate full-chain" in cli(tmp_path, report).stderr
