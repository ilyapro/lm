"""Adversarial tests for the deterministic default-off release-v1 tool.

The only real semantic replay in this module is the public ``eval`` split run
by :mod:`scripts.ap_release`.  Holdout evidence is handled either as an
already-published aggregate or as deliberately invalid opaque bytes, so these
tests cannot become another semantic holdout reader.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ap_release.py"


@pytest.fixture(scope="module")
def ap() -> Any:
    name = "ap_release_test_subject"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _retention() -> dict[str, Any]:
    return {
        kind: {
            population: {"n": n, "before": 1.0, "after": 1.0, "ratio": 0.01}
            for population, n in (("matched_events", 10), ("all_events", 20))
        }
        for kind in ("top_result_content", "useful_feedback")
    }


def _payload(
    before_median: int | float,
    after_median: int | float,
    before_p90: int | float,
    after_p90: int | float,
) -> dict[str, Any]:
    return {
        "before": {"median": before_median, "p90": before_p90},
        "after": {"median": after_median, "p90": after_p90},
        "delta": {"median": 999999, "p90": 999999},
        "retention": _retention(),
    }


def _cross(
    before_results: int,
    before_cross: int,
    after_results: int,
    after_cross: int,
) -> dict[str, Any]:
    return {
        "before": {
            "results": before_results,
            "cross_scope_results": before_cross,
            "cross_scope_share": 0.999,
        },
        "after": {
            "results": after_results,
            "cross_scope_results": after_cross,
            "cross_scope_share": 0.999,
        },
        "delta": {"cross_scope_results": 999999},
    }


def _correction() -> dict[str, Any]:
    return {
        "population": {"pairs": 0, "events_with_pairs": 0},
        "after": {"violations": 0, "corrections_dropped": 0},
    }


def _auto_repeat(ap: Any) -> dict[str, Any]:
    return {
        "repeat_gating": {
            "policy": copy.deepcopy(ap.DEFAULT_REPEAT_POLICY),
            "population": {
                "repeated_fingerprints": 7,
                "repeated_fingerprint_events": 204,
                "gated_events": 0,
            },
            "chars": {
                "automatic_nonrepeated": {"delta_pct": 0.0},
                "organic": {
                    "ungated_total": 796731,
                    "gated_total": 796731,
                    "delta_pct": 0.0,
                },
                "repeated_automatic": {
                    "ungated_total": 1999022,
                    "gated_total": 1999022,
                    "ungated_median": 9360.0,
                    "gated_median": 9360.0,
                    "reduction_ratio": 0.0,
                },
            },
            "retention": {
                "organic_content_access": {
                    "ungated_share": 1.0,
                    "gated_share": 1.0,
                    "delta_pct": 0.0,
                }
            },
            "unseen_in_dev": {
                "reference_split": "dev",
                "population": {
                    "repeated_fingerprints": 1,
                    "repeated_fingerprint_events": 3,
                    "gated_events": 0,
                },
                "chars": {
                    "repeated_automatic": {
                        "ungated_total": 44835,
                        "gated_total": 44835,
                        "reduction_ratio": 0.0,
                    }
                },
                "retention": {
                    "repeated_automatic_content_access": {
                        "events": 3,
                        "ungated_retained": 3,
                        "gated_retained": 3,
                        "ungated_share": 1.0,
                        "gated_share": 1.0,
                        "delta_pct": 0.0,
                    }
                },
            },
        }
    }


def _valid_eval(ap: Any) -> dict[str, Any]:
    return {
        "corpus_root": ap.EVAL_CORPUS_ROOT,
        "delivery_defaults": copy.deepcopy(ap.DEFAULT_DELIVERY),
        "fidelity_limits": [],
        "split": "eval",
        "metrics": {
            "payload": _payload(25345, 14394, 30176, 17455),
            "cross_scope": _cross(3477, 788, 2980, 348),
            "correction_dominance": _correction(),
            "auto_recall": _auto_repeat(ap),
        },
    }


def _valid_holdout() -> dict[str, Any]:
    return {
        "metrics": {
            # The .0 suffixes reproduce the byte-pinned historical aggregate.
            "payload": _payload(23956.0, 13707.0, 29595, 17519),
            "cross_scope": _cross(32426, 5586, 27933, 2243),
            "correction_dominance": _correction(),
        }
    }


def _proof(command: tuple[str, ...], passed: int) -> dict[str, Any]:
    return {
        "command": list(command),
        "passed": passed,
        "failed": 0,
        "status": "pass",
    }


def _derive(ap: Any, eval_compare: dict[str, Any], holdout: dict[str, Any]) -> dict[str, Any]:
    return ap.derive_measurements(
        eval_compare,
        holdout,
        _proof(ap.CORRECTION_SUITE, 26),
        _proof(ap.ERA_SUITE, 45),
    )


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    current = value
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = replacement


def _set_json_path(
    value: Any,
    path: tuple[str | int, ...],
    replacement: Any,
) -> None:
    current = value
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = replacement


def _synthetic_bundle(
    ap: Any,
) -> tuple[dict[str, bytes], Any, dict[str, dict[str, Any]]]:
    eval_compare = _valid_eval(ap)
    holdout = _valid_holdout()
    effective = ap._validate_effective_configuration(eval_compare)
    measurements = _derive(ap, eval_compare, holdout)
    eval_stdout = ap._canonical_json(eval_compare)
    report = ap._build_report(eval_stdout, effective, measurements)
    contents = {
        "eval-compare.json": eval_stdout,
        "release-report.json": ap._canonical_json(report),
        "release-report.md": ap._render_markdown(report),
    }
    lineage = ap.LineageResult(
        evidence={"synthetic_mechanical_lineage": "pass"},
        historical_json={"holdout_compare": holdout},
    )
    implementation = {
        "synthetic.py": {"sha256": "a" * 64, "bytes": 1},
    }
    manifest = ap._build_manifest(
        contents,
        lineage.evidence,
        implementation,
        effective,
    )
    return (
        {**contents, "release-manifest.json": ap._canonical_json(manifest)},
        lineage,
        implementation,
    )


def _valid_control_watermark(ap: Any, manifest_raw: bytes) -> dict[str, Any]:
    implementation = {
        component: {
            "sha256": hashlib.sha256(component.encode()).hexdigest(),
            "bytes": index + 1,
        }
        for index, component in enumerate(ap.RUNTIME_IMPLEMENTATION_COMPONENTS, start=1)
    }
    services = [
        {
            "service_identity_sha256": "1" * 64,
            "boot_identity_sha256": "2" * 64,
            "boot_started_at": "2026-08-14T12:00:00Z",
            # Deliberately differs from the replay code control: deployment is
            # not a precondition for an eligible event-producing runtime.
            "serving_build": {
                "commit": "3" * 40,
                "tree": "4" * 40,
                "implementation": implementation,
            },
            "sanitized_configuration": {
                "schema": ap.RUNTIME_CONFIGURATION_SCHEMA,
                "encoding": ap.CANONICAL_JSON_ENCODING,
                "sha256": "a" * 64,
                "bytes": 37,
            },
            "effective_legacy_repeat_controls": {
                control: False for control in ap.LEGACY_REPEAT_CONTROLS
            },
        }
    ]
    state_hash = hashlib.sha256(ap._canonical_json(services)).hexdigest()
    return {
        "schema": ap.CONTROL_WATERMARK_SCHEMA,
        "schema_version": ap.CONTROL_WATERMARK_SCHEMA_VERSION,
        "release_id": "animal-planet-release-v1",
        "release_manifest": {
            "name": "release-manifest.json",
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "bytes": len(manifest_raw),
        },
        "replay_code_control": {
            "role": "replay_code_control_only",
            "commit": ap.REPLAY_CODE_CONTROL_COMMIT,
            "tree": ap.REPLAY_CODE_CONTROL_TREE,
        },
        "runtime_provenance": {
            "identity_derivation": {
                "schema": ap.RUNTIME_IDENTITY_DERIVATION_SCHEMA,
                "algorithm": "SHA-256",
                "encoding": ap.CANONICAL_JSON_ENCODING,
                "service_identity_input": [
                    "living-memory-service-identity-v1",
                    "raw_service_identity",
                    "raw_per_service_process_invocation_identity",
                ],
                "boot_identity_input": [
                    "living-memory-boot-identity-v1",
                    "raw_per_service_process_invocation_identity",
                ],
                "per_service_process_invocation_identity_high_entropy": True,
            },
            "all_event_producing_services_attested": True,
            "observation": {
                "method": "read_only_pre_post_canonical_services_sha256",
                "pre_observed_at": "2026-08-14T12:59:59Z",
                "pre_runtime_state_sha256": state_hash,
                "post_observed_at": "2026-08-14T13:00:01Z",
                "post_runtime_state_sha256": state_hash,
            },
            "services": services,
        },
        "selection": {
            "basis": "observed_runtime_provenance_instant",
            "lower_bound_exclusive_at": "2026-08-14T13:00:00Z",
        },
        "privacy": {
            "aggregates_and_hashes_only": True,
            "private_paths_included": False,
            "raw_service_identities_included": False,
            "queries_included": False,
            "rows_included": False,
            "raw_outcomes_included": False,
        },
    }


def _write_valid_control_watermark(ap: Any, root: Path) -> dict[str, Any]:
    value = _valid_control_watermark(ap, (root / "release-manifest.json").read_bytes())
    (root / ap.CONTROL_WATERMARK_NAME).write_bytes(ap._canonical_json(value))
    return value


def _publish_finalized_candidate(
    ap: Any, root: Path, bundle: dict[str, bytes]
) -> dict[str, Any]:
    ap._publish_bundle(root, bundle)
    return _write_valid_control_watermark(ap, root)


def _rebind_runtime_state(ap: Any, watermark: dict[str, Any]) -> None:
    state_hash = hashlib.sha256(
        ap._canonical_json(watermark["runtime_provenance"]["services"])
    ).hexdigest()
    observation = watermark["runtime_provenance"]["observation"]
    observation["pre_runtime_state_sha256"] = state_hash
    observation["post_runtime_state_sha256"] = state_hash


def _validate_watermark_bytes(
    ap: Any,
    raw: bytes,
    *,
    manifest_raw: bytes | None = None,
) -> dict[str, Any]:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    candidate = dict(bundle)
    if manifest_raw is not None:
        candidate["release-manifest.json"] = manifest_raw
    candidate[ap.CONTROL_WATERMARK_NAME] = raw
    manifest = ap._validate_manifest(candidate)
    return ap._validate_control_watermark(candidate, manifest)


def _configure_synthetic_validator(
    monkeypatch: pytest.MonkeyPatch,
    ap: Any,
    lineage: Any,
    implementation: dict[str, Any],
    regenerated: dict[str, bytes],
) -> None:
    monkeypatch.setattr(ap, "_verify_lineage", lambda _root: lineage)
    monkeypatch.setattr(ap, "_implementation_evidence", lambda _root: implementation)
    monkeypatch.setattr(ap, "_generate_bundle", lambda _root: regenerated)


def _rewrite_manifest_hash(ap: Any, root: Path, name: str) -> None:
    manifest_path = root / "release-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    raw = (root / name).read_bytes()
    manifest["bundle_files"][name] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    manifest_path.write_bytes(ap._canonical_json(manifest))


def test_sanitized_environment_removes_overrides_and_injection_channels(ap: Any) -> None:
    source = {
        "SAFE_MARKER": "preserved",
        "AWS_SECRET_ACCESS_KEY": "must-not-reach-child",
        "GITHUB_TOKEN": "must-not-reach-child",
        "HOME": "/tmp/synthetic-home",
        "PATH": "/tmp/attacker-bin",
        "LM_RECALL_REPEAT_GATING": "garbage",
        "LM_RECALL_REPEAT_DROP_TRAILING_STUBS": "1",
        "LM_RECALL_REPEAT_FUTURE": "yes",
        "LM_DELIVERY_SNIPPET_CHARS": "999999",
        "LM_DELIVERY_FUTURE": "on",
        "LM_RETRIEVAL_TUNING_POLICY": "/tmp/policy.json",
        "BASH_ENV": "/tmp/bashrc",
        "BASH_FUNC_x%%": "() { :; }",
        "DYLD_INSERT_LIBRARIES": "/tmp/dylib",
        "GIT_CONFIG_COUNT": "1",
        "LD_PRELOAD": "/tmp/preload",
        "PIP_INDEX_URL": "https://invalid.example",
        "PYTHONPATH": "/tmp/site",
        "PYTEST_ADDOPTS": "--capture=no",
        "PYTEST_PLUGINS": "hostile_plugin",
    }
    original = dict(source)

    clean = ap.sanitized_environment(source)

    assert source == original
    assert "SAFE_MARKER" not in clean
    assert "AWS_SECRET_ACCESS_KEY" not in clean
    assert "GITHUB_TOKEN" not in clean
    assert clean["HOME"] == "/tmp/synthetic-home"
    assert set(clean) == set(ap.SAFE_INHERITED_SUBPROCESS_ENV) & set(source) | {
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "LIVING_MEMORY_EMBEDDING_BACKEND",
        "PATH",
        "PIP_CONFIG_FILE",
        "PIP_DISABLE_PIP_VERSION_CHECK",
        "PIP_NO_INDEX",
        "PYTHON",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "PYTHONIOENCODING",
        "PYTHONNOUSERSITE",
        "PYTHONSAFEPATH",
        "PYTHONUTF8",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    }
    assert not any(key.startswith(ap.OVERRIDE_PREFIXES) for key in clean)
    for hostile in (
        "BASH_ENV",
        "BASH_FUNC_x%%",
        "DYLD_INSERT_LIBRARIES",
        "GIT_CONFIG_COUNT",
        "LD_PRELOAD",
        "LM_RETRIEVAL_TUNING_POLICY",
        "PYTHONPATH",
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
    ):
        assert hostile not in clean
    assert clean["PATH"] == "/usr/bin:/bin"
    assert clean["LIVING_MEMORY_EMBEDDING_BACKEND"] == "hash"
    assert clean["GIT_CONFIG_GLOBAL"] == os.devnull
    assert clean["GIT_CONFIG_NOSYSTEM"] == "1"
    assert clean["PIP_CONFIG_FILE"] == os.devnull
    assert clean["PIP_NO_INDEX"] == "1"
    assert clean["PYTHONDONTWRITEBYTECODE"] == "1"
    assert clean["PYTHONHASHSEED"] == "0"
    assert clean["PYTHONNOUSERSITE"] == "1"
    assert clean["PYTHONSAFEPATH"] == "1"
    assert clean["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"


@pytest.mark.parametrize("token", ["", "0", "false", "off", "garbage", "1", "true"])
def test_caller_repeat_tokens_are_always_absent_from_release_subprocesses(
    ap: Any, token: str
) -> None:
    clean = ap.sanitized_environment(
        {
            "LM_RECALL_REPEAT_GATING": token,
            "LM_RECALL_REPEAT_DROP_TRAILING_STUBS": token,
        }
    )
    assert "LM_RECALL_REPEAT_GATING" not in clean
    assert "LM_RECALL_REPEAT_DROP_TRAILING_STUBS" not in clean


def test_calculator_runner_uses_exact_relative_eval_command_and_clean_env(
    ap: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setenv("LM_RECALL_REPEAT_GATING", "1")
    monkeypatch.setenv("LM_DELIVERY_SNIPPET_CHARS", "999999")

    def fake_run(command: tuple[str, ...], **kwargs: Any) -> SimpleNamespace:
        captured.update(command=command, kwargs=kwargs)
        return SimpleNamespace(returncode=0, stdout=b"{}\n", stderr=b"")

    monkeypatch.setattr(ap.subprocess, "run", fake_run)
    assert ap._run_calculator(tmp_path) == b"{}\n"
    command = list(captured["command"])
    kwargs = captured["kwargs"]
    assert command == [sys.executable, *ap.CALCULATOR_ARGS]
    assert command[command.index("--corpus-root") + 1] == ap.EVAL_CORPUS_ROOT
    assert not Path(command[command.index("--corpus-root") + 1]).is_absolute()
    assert command[command.index("--split") + 1] == "eval"
    assert kwargs["cwd"] == tmp_path
    assert kwargs["timeout"] == 180
    assert "shell" not in kwargs
    assert not any(key.startswith(ap.OVERRIDE_PREFIXES) for key in kwargs["env"])


def test_only_allowlisted_nonsemantic_subprocess_sites_exist(ap: Any) -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))

    class RunVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []
            self.owners: list[str] = []
            self.keyword_sets: list[set[str | None]] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
                and node.func.attr == "run"
            ):
                self.owners.append(self.stack[-1])
                self.keyword_sets.append({keyword.arg for keyword in node.keywords})
            self.generic_visit(node)

    visitor = RunVisitor()
    visitor.visit(tree)
    assert visitor.owners == ["_run_calculator", "_run_focused_suite"]
    assert all("shell" not in keywords for keywords in visitor.keyword_sets)
    assert "holdout" not in ap.CALCULATOR_ARGS
    assert "eval" in ap.CALCULATOR_ARGS
    assert not Path(ap.EVAL_CORPUS_ROOT).is_absolute()


def test_semantic_holdout_and_alias_commands_are_rejected(ap: Any) -> None:
    exact = [sys.executable, *ap.CALCULATOR_ARGS]
    ap._assert_safe_calculator_command(exact)
    unsafe: list[list[str]] = []
    for flag, value in (
        ("--corpus-root", str(REPO_ROOT / ap.EVAL_CORPUS_ROOT)),
        ("--corpus-root", "./" + ap.EVAL_CORPUS_ROOT),
        ("--corpus-root", "artifacts/animal-planet/corpus/../corpus"),
        ("--split", "dev"),
        ("--split", "holdout"),
        ("--metrics", "payload"),
    ):
        command = list(exact)
        command[command.index(flag) + 1] = value
        unsafe.append(command)
    unsafe.extend(
        (
            [*exact, "--dev-fingerprint-index", "secret.json"],
            [sys.executable, "-B", ap.CALCULATOR_SCRIPT, "verify"],
            [sys.executable, "-B", "scripts/ap_confirmatory_readiness.py"],
            [
                sys.executable,
                "-B",
                "artifacts/animal-planet/evaluation/replacement-holdout/recipe/verify.py",
            ],
        )
    )
    for command in unsafe:
        with pytest.raises(ap.ReleaseError, match="unsafe|prohibited|relative"):
            ap._assert_safe_calculator_command(command)
    with pytest.raises(ap.ReleaseError, match="unsafe focused-suite command"):
        ap._run_focused_suite(
            REPO_ROOT,
            (sys.executable, "scripts/ap_baseline.py", "compare", "--split", "holdout"),
            1,
            "forbidden holdout reader",
        )


@pytest.mark.parametrize("mode", ["exit", "empty", "timeout"])
def test_calculator_reader_faults_fail_closed(
    ap: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    def fake_run(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        if mode == "timeout":
            raise subprocess.TimeoutExpired(cmd="calculator", timeout=180)
        if mode == "exit":
            return SimpleNamespace(returncode=7, stdout=b"", stderr=b"synthetic failure")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(ap.subprocess, "run", fake_run)
    with pytest.raises(ap.ReleaseError, match="could not complete|failed with exit|empty"):
        ap._run_calculator(tmp_path)


def test_strict_json_reader_rejects_duplicates_nonfinite_and_invalid_utf8(ap: Any) -> None:
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b"\xff"):
        with pytest.raises(ap.ReleaseError):
            ap._load_json_bytes(raw, "adversarial.json")


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("delivery_defaults", "session_dedup"), False),
        (("delivery_defaults", "snippet_max_chars"), 1201),
        (("delivery_defaults", "snippet_ladder"), [0, 1000]),
        (("metrics", "auto_recall", "repeat_gating", "policy", "enabled"), True),
        (
            (
                "metrics",
                "auto_recall",
                "repeat_gating",
                "policy",
                "drop_trailing_stubs",
            ),
            True,
        ),
        (("metrics", "auto_recall", "repeat_gating", "policy", "min_sessions"), 3),
        (("metrics", "auto_recall", "repeat_gating", "population", "gated_events"), 1),
        (
            (
                "metrics",
                "auto_recall",
                "repeat_gating",
                "unseen_in_dev",
                "population",
                "gated_events",
            ),
            1,
        ),
        (
            (
                "metrics",
                "auto_recall",
                "repeat_gating",
                "population",
                "repeated_fingerprints",
            ),
            0,
        ),
        (
            (
                "metrics",
                "auto_recall",
                "repeat_gating",
                "chars",
                "organic",
                "gated_total",
            ),
            796730,
        ),
        (
            (
                "metrics",
                "auto_recall",
                "repeat_gating",
                "retention",
                "organic_content_access",
                "gated_share",
            ),
            0.99,
        ),
    ],
)
def test_default_delivery_and_repeat_configuration_drift_is_rejected(
    ap: Any, path: tuple[str, ...], replacement: Any
) -> None:
    compare = _valid_eval(ap)
    _set_path(compare, path, replacement)
    with pytest.raises(ap.ReleaseError):
        ap._validate_effective_configuration(compare)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (
            (
                "metrics",
                "auto_recall",
                "repeat_gating",
                "unseen_in_dev",
                "population",
                "repeated_fingerprints",
            ),
            8,
        ),
        (
            (
                "metrics",
                "auto_recall",
                "repeat_gating",
                "unseen_in_dev",
                "population",
                "repeated_fingerprint_events",
            ),
            205,
        ),
        (
            (
                "metrics",
                "auto_recall",
                "repeat_gating",
                "unseen_in_dev",
                "retention",
                "repeated_automatic_content_access",
                "events",
            ),
            1,
        ),
        (
            (
                "metrics",
                "auto_recall",
                "repeat_gating",
                "chars",
                "automatic_nonrepeated",
                "delta_pct",
            ),
            99,
        ),
        (
            (
                "metrics",
                "auto_recall",
                "repeat_gating",
                "chars",
                "repeated_automatic",
                "reduction_ratio",
            ),
            0.5,
        ),
        (
            (
                "metrics",
                "auto_recall",
                "repeat_gating",
                "unseen_in_dev",
                "chars",
                "repeated_automatic",
                "reduction_ratio",
            ),
            0.5,
        ),
        (
            (
                "metrics",
                "auto_recall",
                "repeat_gating",
                "chars",
                "repeated_automatic",
                "gated_median",
            ),
            9359,
        ),
    ],
)
def test_contradictory_disabled_policy_evidence_is_rejected(
    ap: Any, path: tuple[str, ...], replacement: Any
) -> None:
    compare = _valid_eval(ap)
    _set_path(compare, path, replacement)
    with pytest.raises(ap.ReleaseError):
        ap._validate_effective_configuration(compare)


def test_valid_effective_configuration_discloses_strict_defaults_and_zero_delta(
    ap: Any,
) -> None:
    effective = ap._validate_effective_configuration(_valid_eval(ap))
    assert effective["delivery_defaults"] == ap.DEFAULT_DELIVERY
    assert effective["delivery_defaults"]["full_node_diet"] is True
    assert effective["delivery_defaults"]["session_dedup"] is True
    assert effective["repeat_gating_policy"] == ap.DEFAULT_REPEAT_POLICY
    assert effective["repeat_gating_policy"]["enabled"] is False
    assert effective["repeat_gating_policy"]["drop_trailing_stubs"] is False
    assert effective["zero_gated_events"] == {"all_eval": 0, "unseen_in_dev": 0}
    equalities = effective["disabled_policy_raw_equalities"]
    assert all(
        pair.get("ungated", pair.get("ungated_retained"))
        == pair.get("effective", pair.get("effective_retained"))
        for pair in equalities.values()
    )


@pytest.mark.parametrize(
    ("corpus_root", "split"),
    [
        (str(REPO_ROOT / "artifacts/animal-planet/corpus"), "eval"),
        ("./artifacts/animal-planet/corpus", "eval"),
        ("artifacts/animal-planet/corpus", "holdout"),
    ],
)
def test_calculator_identity_requires_exact_relative_eval_corpus(
    ap: Any, corpus_root: str, split: str
) -> None:
    compare = _valid_eval(ap)
    compare["corpus_root"] = corpus_root
    compare["split"] = split
    with pytest.raises(ap.ReleaseError):
        ap._validate_compare_identity(compare)


def test_ratios_are_derived_from_raw_counts_and_match_release_result(ap: Any) -> None:
    eval_compare = _valid_eval(ap)
    holdout = _valid_holdout()
    measurements = _derive(ap, eval_compare, holdout)

    p2 = measurements["P2_payload_and_retention"]
    assert p2["status"] == "pass"
    assert p2["eval"]["median"] == {
        "before_chars": 25345,
        "after_chars": 14394,
        "ratio": 0.5679226672,
        "reduction_ratio": 0.4320773328,
    }
    assert p2["eval"]["p90"]["ratio"] == 0.5784398197
    assert p2["preserved_holdout"]["median"]["before_chars"] == 23956
    assert p2["preserved_holdout"]["median"]["after_chars"] == 13707
    assert p2["preserved_holdout"]["median"]["ratio"] == 0.5721739856
    assert p2["preserved_holdout"]["p90"]["ratio"] == 0.591958101
    assert p2["eval"]["retention"]["minimum_ratio"] == 1.0
    assert p2["preserved_holdout"]["retention"]["minimum_ratio"] == 1.0

    p3 = measurements["P3_cross_scope"]
    assert p3["status"] == "pass"
    assert p3["eval"]["admission"]["reduction_ratio"] == 0.5583756345
    assert p3["eval"]["same_scope"]["retention_ratio"] == 0.9788025288
    assert p3["preserved_holdout"]["admission"]["reduction_ratio"] == 0.5984604368
    assert p3["preserved_holdout"]["same_scope"]["retention_ratio"] == 0.9571535022


def test_distribution_arithmetic_accepts_float_quantiles_and_rejects_bad_numbers(
    ap: Any,
) -> None:
    result, exact = ap._distribution_ratio(
        {"before": {"median": 10.5}, "after": {"median": 6.0}},
        "median",
        "synthetic",
    )
    assert result["before_chars"] == 10.5
    assert result["after_chars"] == 6
    assert result["ratio"] == 0.5714285714
    assert exact.numerator == 4 and exact.denominator == 7

    for bad in (True, -1, float("nan"), float("inf"), "10"):
        with pytest.raises(ap.ReleaseError):
            ap._distribution_ratio(
                {"before": {"median": bad}, "after": {"median": 1}},
                "median",
                "synthetic",
            )
    with pytest.raises(ap.ReleaseError, match="denominator must be positive"):
        ap._distribution_ratio(
            {"before": {"median": 0}, "after": {"median": 0}},
            "median",
            "synthetic",
        )


def test_exact_threshold_boundaries_pass(ap: Any) -> None:
    eval_compare = _valid_eval(ap)
    holdout = _valid_holdout()
    for source in (eval_compare, holdout):
        payload = source["metrics"]["payload"]
        payload["before"].update(median=100, p90=100)
        payload["after"].update(median=60, p90=60)
        for kind in ("top_result_content", "useful_feedback"):
            for population in ("matched_events", "all_events"):
                payload["retention"][kind][population].update(before=1.0, after=0.95)
        source["metrics"]["cross_scope"] = _cross(120, 20, 105, 10)
    measurements = _derive(ap, eval_compare, holdout)
    assert measurements["P2_payload_and_retention"]["status"] == "pass"
    assert measurements["P3_cross_scope"]["status"] == "pass"


def test_p2_payload_retention_and_generalization_failures_fail_closed(ap: Any) -> None:
    cases: list[tuple[str, Callable[[dict[str, Any], dict[str, Any]], None]]] = [
        (
            "eval median",
            lambda evaluation, _holdout: evaluation["metrics"]["payload"]["after"].__setitem__(
                "median", 16000
            ),
        ),
        (
            "eval p90",
            lambda evaluation, _holdout: evaluation["metrics"]["payload"]["after"].__setitem__(
                "p90", 19000
            ),
        ),
        (
            "holdout median",
            lambda _evaluation, holdout: holdout["metrics"]["payload"]["after"].__setitem__(
                "median", 15000
            ),
        ),
        (
            "holdout p90",
            lambda _evaluation, holdout: holdout["metrics"]["payload"]["after"].__setitem__(
                "p90", 18000
            ),
        ),
        (
            "eval retention",
            lambda evaluation, _holdout: evaluation["metrics"]["payload"]["retention"]
            ["top_result_content"]["all_events"].update(after=0.94),
        ),
        (
            "holdout retention",
            lambda _evaluation, holdout: holdout["metrics"]["payload"]["retention"]
            ["useful_feedback"]["matched_events"].update(after=0.94),
        ),
        (
            "gap",
            lambda evaluation, holdout: (
                evaluation["metrics"]["payload"]["before"].update(median=100),
                evaluation["metrics"]["payload"]["after"].update(median=50),
                holdout["metrics"]["payload"]["before"].update(median=100),
                holdout["metrics"]["payload"]["after"].update(median=56),
            ),
        ),
        (
            "precision epsilon",
            lambda evaluation, holdout: (
                evaluation["metrics"]["payload"]["before"].update(median=10**40),
                evaluation["metrics"]["payload"]["after"].update(
                    median=6 * 10**39 + 1
                ),
                holdout["metrics"]["payload"]["before"].update(median=10**40),
                holdout["metrics"]["payload"]["after"].update(median=6 * 10**39),
            ),
        ),
    ]
    for label, mutate in cases:
        eval_compare = _valid_eval(ap)
        holdout = _valid_holdout()
        mutate(eval_compare, holdout)
        measurements = _derive(ap, eval_compare, holdout)
        assert measurements["P2_payload_and_retention"]["status"] == "fail", label
        with pytest.raises(ap.ReleaseError, match="P2_payload_and_retention"):
            ap._require_p2_p5(measurements)


def test_p3_reduction_and_same_scope_failures_fail_closed(ap: Any) -> None:
    cases: list[tuple[str, Callable[[dict[str, Any], dict[str, Any]], None]]] = [
        (
            "eval reduction",
            lambda evaluation, _holdout: evaluation["metrics"]["cross_scope"]["after"].update(
                cross_scope_results=400
            ),
        ),
        (
            "holdout reduction",
            lambda _evaluation, holdout: holdout["metrics"]["cross_scope"]["after"].update(
                cross_scope_results=2800
            ),
        ),
        (
            "eval same scope",
            lambda evaluation, _holdout: evaluation["metrics"]["cross_scope"]["after"].update(
                results=2848
            ),
        ),
        (
            "holdout same scope",
            lambda _evaluation, holdout: holdout["metrics"]["cross_scope"]["after"].update(
                results=27000
            ),
        ),
    ]
    for label, mutate in cases:
        eval_compare = _valid_eval(ap)
        holdout = _valid_holdout()
        mutate(eval_compare, holdout)
        measurements = _derive(ap, eval_compare, holdout)
        assert measurements["P3_cross_scope"]["status"] == "fail", label
        with pytest.raises(ap.ReleaseError, match="P3_cross_scope"):
            ap._require_p2_p5(measurements)


@pytest.mark.parametrize("gate", ["P2_payload_and_retention", "P3_cross_scope", "P4_correction_dominance", "P5_era_safety"])
def test_every_p2_p5_gate_status_is_mandatory(ap: Any, gate: str) -> None:
    measurements = _derive(ap, _valid_eval(ap), _valid_holdout())
    measurements[gate]["status"] = "fail"
    with pytest.raises(ap.ReleaseError, match=gate):
        ap._require_p2_p5(measurements)


@pytest.mark.parametrize("suite", ["correction", "era"])
@pytest.mark.parametrize("fault", ["command", "count", "failed", "status"])
def test_p4_and_p5_focused_proof_faults_are_rejected(
    ap: Any, suite: str, fault: str
) -> None:
    correction = _proof(ap.CORRECTION_SUITE, 26)
    era = _proof(ap.ERA_SUITE, 45)
    target = correction if suite == "correction" else era
    if fault == "command":
        target["command"] = ["python3", "forbidden.py"]
    elif fault == "count":
        target["passed"] -= 1
    elif fault == "failed":
        target["failed"] = 1
    else:
        target["status"] = "fail"
    with pytest.raises(ap.ReleaseError, match="proof failed|bound focused suite"):
        ap.derive_measurements(
            _valid_eval(ap), _valid_holdout(), correction, era
        )


def test_correction_zero_pair_limitation_is_explicit_and_nonzero_drift_rejects(
    ap: Any,
) -> None:
    measurements = _derive(ap, _valid_eval(ap), _valid_holdout())
    p4 = measurements["P4_correction_dominance"]
    disclosure = p4["zero_pair_population_limitation"]
    assert p4["status"] == "pass"
    assert p4["focused_suite"]["passed"] == 26
    assert "zero correction/superseded pairs" in disclosure
    assert "eval" in disclosure and "preserved holdout" in disclosure
    assert "26-test live/property/SQLite focused suite" in disclosure

    evaluation = _valid_eval(ap)
    evaluation["metrics"]["correction_dominance"]["population"].update(
        pairs=1, events_with_pairs=1
    )
    with pytest.raises(ap.ReleaseError, match="zero-pair population"):
        _derive(ap, evaluation, _valid_holdout())


def test_p6_is_deferred_never_promoted_by_default_off_evidence(ap: Any) -> None:
    evaluation = _valid_eval(ap)
    effective = ap._validate_effective_configuration(evaluation)
    measurements = _derive(ap, evaluation, _valid_holdout())
    report = ap._build_report(ap._canonical_json(evaluation), effective, measurements)
    p6 = report["measurements"]["P6_automatic_only_policy"]
    gates = {gate["id"]: gate["status"] for gate in report["gates"]}
    assert report["overall"]["status"] == "pass_with_p6_deferred"
    assert p6["status"] == "deferred_pending_confirmatory_v3"
    assert p6["legacy_repeat_path"] == "class-blind_experimental_strict_opt_in"
    assert p6["automatic_only_compaction"] == "not_confirmed_by_this_release"
    assert p6["historical_negative_evidence"]["new_semantic_evaluation"] is False
    assert p6["effective_policy"]["enabled"] is False
    assert p6["effective_policy"]["drop_trailing_stubs"] is False
    assert p6["zero_gated_events"] == {"all_eval": 0, "unseen_in_dev": 0}
    assert gates["P6_automatic_only_policy"] == "deferred_pending_confirmatory_v3"
    assert "P6 deferred" in ap._render_markdown(report).decode("utf-8")


def test_every_historical_and_transition_pin_matches_and_attempt_note_is_bound(ap: Any) -> None:
    groups = (
        (ap.ORIGINAL_MANIFEST,),
        (ap.REPLACEMENT_MANIFEST,),
        ap.HISTORICAL_INPUTS,
        ap.RETIRED_V2_INPUTS,
        ap.PRESERVED_TRANSITIVE_INPUTS,
        ap.PRESERVED_RELEASE_TRANSITION_INPUTS,
    )
    pins = [pin for group in groups for pin in group]
    for pin in pins:
        raw = ap._read_and_verify_pinned(REPO_ROOT, pin)
        assert len(raw) == pin.bytes
        assert hashlib.sha256(raw).hexdigest() == pin.sha256
    retired_paths = {pin.path for pin in ap.RETIRED_V2_INPUTS}
    assert "artifacts/animal-planet/evaluation/confirmatory-holdout-v2/attempt-note.json" in retired_paths
    assert len(ap.HISTORICAL_INPUTS) == 7
    assert len(ap.RETIRED_V2_INPUTS) == 6


def test_same_size_historical_tamper_is_rejected(ap: Any, tmp_path: Path) -> None:
    pin = next(pin for pin in ap.RETIRED_V2_INPUTS if pin.path.endswith("attempt-note.json"))
    copied = tmp_path / pin.path
    copied.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / pin.path, copied)
    ap._read_and_verify_pinned(tmp_path, pin)
    raw = bytearray(copied.read_bytes())
    raw[len(raw) // 2] ^= 1
    copied.write_bytes(raw)
    with pytest.raises(ap.ReleaseError, match="pinned evidence changed"):
        ap._read_and_verify_pinned(tmp_path, pin)


def test_packet_members_are_opaque_hash_reads_not_semantic_json_reads(
    ap: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    packet = tmp_path / "packet"
    member = packet / "corpus" / "holdout.jsonl"
    member.parent.mkdir(parents=True)
    opaque = b"this is deliberately not JSON and must never be parsed\n"
    member.write_bytes(opaque)
    manifest = {
        "frozen": True,
        "files": {
            "corpus/holdout.jsonl": {
                "sha256": hashlib.sha256(opaque).hexdigest(),
                "bytes": len(opaque),
            }
        },
        "packet_files": {},
    }
    manifest_path = packet / "manifest.json"
    manifest_path.write_bytes(ap._canonical_json(manifest))
    pin = ap.PinnedFile(
        "packet/manifest.json",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        manifest_path.stat().st_size,
    )
    labels: list[str] = []
    original_load = ap._load_json_bytes

    def spy_load(raw: bytes, label: str) -> Any:
        labels.append(label)
        return original_load(raw, label)

    monkeypatch.setattr(ap, "_load_json_bytes", spy_load)
    evidence, _raw = ap._verify_packet_manifest(tmp_path, pin, "packet")
    assert evidence["manifest_members_hash_verified"] == 1
    assert evidence["semantic_case_reads"] == 0
    assert labels == ["packet/manifest.json"]
    member.write_bytes(opaque[:-1] + b"!")
    with pytest.raises(ap.ReleaseError, match="packet member changed"):
        ap._verify_packet_manifest(tmp_path, pin, "packet")


def test_repo_input_symlinks_are_rejected(ap: Any, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "evidence").symlink_to(outside)
    with pytest.raises(ap.ReleaseError, match="symlink"):
        ap._safe_repo_file(repo, "evidence", "test evidence")
    with pytest.raises(ap.ReleaseError, match="unsafe"):
        ap._safe_repo_file(repo, "../outside.txt", "test evidence")


def test_lineage_mechanically_verifies_packets_and_records_zero_readers(ap: Any) -> None:
    lineage = ap._verify_lineage(REPO_ROOT)
    evidence = lineage.evidence
    assert evidence["original_packet"]["manifest_members_hash_verified"] > 0
    assert evidence["replacement_packet"]["manifest_members_hash_verified"] > 0
    assert evidence["historical_aggregates"]["status"] == "byte_identical"
    assert evidence["retired_confirmatory_v2"]["status"] == "byte_identical_and_no_authority"
    assert len(evidence["retired_confirmatory_v2"]["files"]) == 6
    assert evidence["retired_confirmatory_v2"]["scanner_invocations"] == 0
    assert evidence["reader_boundary"] == {
        "calculator_split": "eval",
        "holdout_compare_invocations": 0,
        "replacement_reader_invocations": 0,
        "packet_verifier_invocations": 0,
        "readiness_scanner_invocations": 0,
        "semantic_case_reads": 0,
        "new_v3_authority_created": False,
    }
    compatibility = evidence["preserved_holdout_implementation_compatibility"]["files"]
    storage = compatibility["src/living_memory/storage.py"]
    assert storage["sha256"] == ap.PRESERVED_RELEASE_TRANSITION_INPUTS[0].sha256
    assert storage["pin_source"] == "reviewed strict-default release transition"


def test_implementation_binding_covers_all_runtime_modules_and_release_inputs(ap: Any) -> None:
    bound = set(ap.IMPLEMENTATION_PATHS)
    tracked_modules = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "src/living_memory").glob("*.py")
    }
    assert tracked_modules <= bound
    assert {
        "scripts/ap_release.py",
        "scripts/ap_baseline.py",
        "tests/test_ap_release.py",
        "tests/test_retrieval_correction_dominance.py",
        "tests/test_consolidation.py",
        "docs/mcp-interface.md",
        "pyproject.toml",
    } <= bound


def test_core_and_finalized_candidate_membership_contracts_are_distinct(ap: Any) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    assert ap.OUTPUT_ORDER == (
        "eval-compare.json",
        "release-report.json",
        "release-report.md",
        "release-manifest.json",
    )
    assert tuple(bundle) == ap.OUTPUT_ORDER
    assert ap.CANDIDATE_MEMBERS == (*ap.OUTPUT_ORDER, "control-watermark.json")
    assert ap.REPLAY_CODE_CONTROL_COMMIT == "46a9951842512333b0896370056d07a9e1c25bdf"
    assert ap.REPLAY_CODE_CONTROL_TREE == "c1606671b9d13bed78c21b7d2f9a4bb75a3d1c1c"
    assert ap.LEGACY_REPEAT_CONTROLS == (
        "LM_RECALL_REPEAT_GATING",
        "LM_RECALL_REPEAT_DROP_TRAILING_STUBS",
    )
    assert set(ap.RUNTIME_IMPLEMENTATION_COMPONENTS) == {
        path.stem for path in (REPO_ROOT / "src/living_memory").glob("*.py")
    }
    assert set(ap.RUNTIME_IMPLEMENTATION_COMPONENTS) == {
        "__init__",
        "config",
        "consolidation",
        "decay",
        "delivery",
        "edge_backfill",
        "edge_derivation",
        "embeddings",
        "feedback",
        "health_audit",
        "maintenance",
        "models",
        "phase",
        "prompts",
        "replay",
        "resources",
        "retrieval",
        "scope",
        "server",
        "storage",
        "temporal",
    }

    manifest = json.loads(bundle["release-manifest.json"])
    assert set(manifest["bundle_files"]) == set(ap.CONTENT_OUTPUTS)
    assert manifest["publication"]["write_order"] == list(ap.OUTPUT_ORDER)
    assert manifest["required_companion"] == {
        "name": "control-watermark.json",
        "schema": ap.CONTROL_WATERMARK_SCHEMA,
        "schema_version": ap.CONTROL_WATERMARK_SCHEMA_VERSION,
        "required_for_final_validation": True,
        "binding": "companion_reverse_pins_manifest_sha256_and_bytes",
    }
    assert "sha256" not in manifest["required_companion"]
    assert "bytes" not in manifest["required_companion"]


def test_valid_control_watermark_is_closed_canonical_and_allows_older_serving_build(
    ap: Any,
) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    value = _valid_control_watermark(ap, bundle["release-manifest.json"])

    validated = _validate_watermark_bytes(ap, ap._canonical_json(value))

    assert validated == value
    serving = validated["runtime_provenance"]["services"][0]["serving_build"]
    assert serving["commit"] != validated["replay_code_control"]["commit"]
    assert serving["tree"] != validated["replay_code_control"]["tree"]
    assert validated["selection"] == {
        "basis": "observed_runtime_provenance_instant",
        "lower_bound_exclusive_at": "2026-08-14T13:00:00Z",
    }


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("name", "renamed-watermark.json"),
        ("schema", "unknown-watermark-schema"),
        ("schema_version", True),
        ("required_for_final_validation", False),
        ("binding", "manifest_hashes_companion"),
        ("sha256", "f" * 64),
        ("bytes", 123),
    ],
)
def test_manifest_rejects_changed_or_circular_companion_declarations(
    ap: Any,
    field: str,
    replacement: Any,
) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    manifest = json.loads(bundle["release-manifest.json"])
    manifest["required_companion"][field] = replacement
    candidate = dict(bundle)
    candidate["release-manifest.json"] = ap._canonical_json(manifest)
    with pytest.raises(ap.ReleaseError, match="companion contract"):
        ap._validate_manifest(candidate)


def test_manifest_cannot_hash_watermark_or_replace_a_core_content_member(ap: Any) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    original = json.loads(bundle["release-manifest.json"])
    for replace_core in (False, True):
        manifest = copy.deepcopy(original)
        if replace_core:
            manifest["bundle_files"].pop("release-report.md")
        manifest["bundle_files"][ap.CONTROL_WATERMARK_NAME] = {
            "sha256": "f" * 64,
            "bytes": 1,
        }
        candidate = dict(bundle)
        candidate["release-manifest.json"] = ap._canonical_json(manifest)
        with pytest.raises(ap.ReleaseError, match="bundle file set"):
            ap._validate_manifest(candidate)


@pytest.mark.parametrize(
    "path,replacement",
    [
        (("schema",), "wrong-schema"),
        (("schema_version",), True),
        (("schema_version",), 1.0),
        (("release_id",), "wrong-release"),
        (("release_manifest", "name"), "renamed-manifest.json"),
        (("release_manifest", "sha256"), "b" * 64),
        (("release_manifest", "bytes"), True),
        (("release_manifest", "bytes"), 0),
        (("replay_code_control", "role"), "serving_runtime"),
        (("replay_code_control", "commit"), "f" * 40),
        (("replay_code_control", "tree"), "e" * 40),
        (("selection", "basis"), "git_commit_timestamp"),
        (("selection", "lower_bound_exclusive_at"), "2026-08-14T12:59:58Z"),
        (("selection", "lower_bound_exclusive_at"), "2026-08-14T13:00:02Z"),
        (("selection", "lower_bound_exclusive_at"), "2026-08-14T20:00:00+07:00"),
        (("selection", "lower_bound_exclusive_at"), "2026-08-14T13:00:00.0Z"),
        (("selection", "lower_bound_exclusive_at"), "2026-02-30T13:00:00Z"),
        (("runtime_provenance", "all_event_producing_services_attested"), False),
        (("runtime_provenance", "all_event_producing_services_attested"), 1),
        (("runtime_provenance", "identity_derivation", "algorithm"), "plain-SHA256"),
        (
            (
                "runtime_provenance",
                "identity_derivation",
                "per_service_process_invocation_identity_high_entropy",
            ),
            1,
        ),
        (
            ("runtime_provenance", "identity_derivation", "service_identity_input"),
            ["raw_service_identity"],
        ),
        (("runtime_provenance", "services"), []),
        (("runtime_provenance", "services", 0, "service_identity_sha256"), "raw-service"),
        (("runtime_provenance", "services", 0, "boot_identity_sha256"), "A" * 64),
        (
            ("runtime_provenance", "services", 0, "boot_started_at"),
            "2026-08-14T12:59:59.500000Z",
        ),
        (("runtime_provenance", "services", 0, "boot_started_at"), "2026-08-14T13:00:01Z"),
        (("runtime_provenance", "services", 0, "serving_build", "commit"), "3" * 39),
        (("runtime_provenance", "services", 0, "serving_build", "tree"), "z" * 40),
        (
            (
                "runtime_provenance",
                "services",
                0,
                "serving_build",
                "implementation",
                "retrieval",
                "bytes",
            ),
            True,
        ),
        (
            (
                "runtime_provenance",
                "services",
                0,
                "serving_build",
                "implementation",
                "server",
                "sha256",
            ),
            "F" * 64,
        ),
        (
            ("runtime_provenance", "services", 0, "sanitized_configuration", "sha256"),
            "not-a-hash",
        ),
        (
            ("runtime_provenance", "services", 0, "sanitized_configuration", "bytes"),
            0,
        ),
        (
            ("runtime_provenance", "services", 0, "sanitized_configuration", "bytes"),
            1.0,
        ),
        (
            ("runtime_provenance", "services", 0, "sanitized_configuration", "bytes"),
            "37",
        ),
        (
            ("runtime_provenance", "services", 0, "sanitized_configuration", "schema"),
            "arbitrary-config",
        ),
        (
            (
                "runtime_provenance",
                "services",
                0,
                "sanitized_configuration",
                "encoding",
            ),
            "repr",
        ),
        (
            ("runtime_provenance", "observation", "method"),
            "git_status_snapshot",
        ),
        (
            ("runtime_provenance", "observation", "pre_observed_at"),
            "2026-08-14T13:00:01Z",
        ),
        (
            ("runtime_provenance", "observation", "post_observed_at"),
            "2026-08-14T12:59:59Z",
        ),
        (
            ("runtime_provenance", "observation", "post_runtime_state_sha256"),
            "f" * 64,
        ),
    ],
)
def test_control_watermark_rejects_identity_binding_runtime_and_time_faults(
    ap: Any,
    path: tuple[str | int, ...],
    replacement: Any,
) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    value = _valid_control_watermark(ap, bundle["release-manifest.json"])
    _set_json_path(value, path, replacement)
    if path[:2] == ("runtime_provenance", "services"):
        _rebind_runtime_state(ap, value)

    with pytest.raises(ap.ReleaseError):
        _validate_watermark_bytes(ap, ap._canonical_json(value))


@pytest.mark.parametrize("control", [
    "LM_RECALL_REPEAT_GATING",
    "LM_RECALL_REPEAT_DROP_TRAILING_STUBS",
])
@pytest.mark.parametrize("enabled", [True, 0, None, "false"])
def test_control_watermark_requires_both_effective_controls_as_literal_false(
    ap: Any,
    control: str,
    enabled: Any,
) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    value = _valid_control_watermark(ap, bundle["release-manifest.json"])
    controls = value["runtime_provenance"]["services"][0][
        "effective_legacy_repeat_controls"
    ]
    controls[control] = enabled
    _rebind_runtime_state(ap, value)

    with pytest.raises(ap.ReleaseError, match="control"):
        _validate_watermark_bytes(ap, ap._canonical_json(value))


@pytest.mark.parametrize(
    "privacy_key,bad_value",
    [
        ("aggregates_and_hashes_only", False),
        ("private_paths_included", True),
        ("raw_service_identities_included", True),
        ("queries_included", True),
        ("rows_included", True),
        ("raw_outcomes_included", True),
        ("aggregates_and_hashes_only", 1),
        ("private_paths_included", 0),
    ],
)
def test_control_watermark_rejects_every_privacy_unsafe_claim(
    ap: Any,
    privacy_key: str,
    bad_value: Any,
) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    value = _valid_control_watermark(ap, bundle["release-manifest.json"])
    value["privacy"][privacy_key] = bad_value

    with pytest.raises(ap.ReleaseError, match="privacy"):
        _validate_watermark_bytes(ap, ap._canonical_json(value))


@pytest.mark.parametrize(
    "container_path,field,private_value",
    [
        ((), "query", "private query"),
        (("release_manifest",), "path", "/private/release-manifest.json"),
        (("replay_code_control",), "worktree", "/private/worktree"),
        (("runtime_provenance",), "rows", [{"private": "row"}]),
        (("runtime_provenance", "observation"), "notes", "private identity"),
        (("runtime_provenance", "services", 0), "service_identity", "raw-hostname"),
        (
            ("runtime_provenance", "services", 0, "serving_build"),
            "source_path",
            "/private/checkout",
        ),
        (
            (
                "runtime_provenance",
                "services",
                0,
                "serving_build",
                "implementation",
            ),
            "private-module",
            {"sha256": "f" * 64, "bytes": 1},
        ),
        (
            ("runtime_provenance", "services", 0, "sanitized_configuration"),
            "raw",
            {"db": "/private/database"},
        ),
        (
            (
                "runtime_provenance",
                "services",
                0,
                "effective_legacy_repeat_controls",
            ),
            "LM_RECALL_REPEAT_PRIVATE",
            False,
        ),
        (("selection",), "release_effective_at", "2026-08-14T13:00:00Z"),
        (("privacy",), "notes", "private row"),
    ],
)
def test_closed_watermark_schema_rejects_free_form_privacy_channels(
    ap: Any,
    container_path: tuple[str | int, ...],
    field: str,
    private_value: Any,
) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    value = _valid_control_watermark(ap, bundle["release-manifest.json"])
    container: Any = value
    for part in container_path:
        container = container[part]
    container[field] = private_value
    if container_path[:2] == ("runtime_provenance", "services"):
        _rebind_runtime_state(ap, value)

    with pytest.raises(ap.ReleaseError, match="field set"):
        _validate_watermark_bytes(ap, ap._canonical_json(value))


@pytest.mark.parametrize(
    "container_path,field",
    [
        ((), "privacy"),
        (("release_manifest",), "sha256"),
        (("replay_code_control",), "tree"),
        (("runtime_provenance",), "identity_derivation"),
        (
            ("runtime_provenance", "identity_derivation"),
            "per_service_process_invocation_identity_high_entropy",
        ),
        (("runtime_provenance",), "observation"),
        (("runtime_provenance", "observation"), "pre_runtime_state_sha256"),
        (("runtime_provenance", "services", 0), "boot_identity_sha256"),
        (("runtime_provenance", "services", 0, "serving_build"), "tree"),
        (
            (
                "runtime_provenance",
                "services",
                0,
                "serving_build",
                "implementation",
            ),
            "retrieval",
        ),
        (
            ("runtime_provenance", "services", 0, "sanitized_configuration"),
            "bytes",
        ),
        (
            ("runtime_provenance", "services", 0, "sanitized_configuration"),
            "schema",
        ),
        (("selection",), "lower_bound_exclusive_at"),
        (("privacy",), "rows_included"),
    ],
)
def test_control_watermark_rejects_missing_required_envelope_fields(
    ap: Any,
    container_path: tuple[str | int, ...],
    field: str,
) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    value = _valid_control_watermark(ap, bundle["release-manifest.json"])
    container: Any = value
    for part in container_path:
        container = container[part]
    container.pop(field)
    if container_path[:2] == ("runtime_provenance", "services"):
        _rebind_runtime_state(ap, value)

    with pytest.raises(ap.ReleaseError, match="field set"):
        _validate_watermark_bytes(ap, ap._canonical_json(value))


def test_control_watermark_rejects_missing_renamed_and_extra_control_fields(ap: Any) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    original = _valid_control_watermark(ap, bundle["release-manifest.json"])
    controls_path = ("runtime_provenance", "services", 0, "effective_legacy_repeat_controls")
    for control in ap.LEGACY_REPEAT_CONTROLS:
        for mode in ("missing", "renamed", "extra"):
            value = copy.deepcopy(original)
            controls: Any = value
            for part in controls_path:
                controls = controls[part]
            if mode == "missing":
                controls.pop(control)
            elif mode == "renamed":
                controls[f"renamed_{control}"] = controls.pop(control)
            else:
                controls[f"extra_{control}"] = False
            _rebind_runtime_state(ap, value)
            with pytest.raises(ap.ReleaseError, match="field set"):
                _validate_watermark_bytes(ap, ap._canonical_json(value))


def test_control_watermark_requires_unique_sorted_services_and_bound_state(ap: Any) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    original = _valid_control_watermark(ap, bundle["release-manifest.json"])
    second = copy.deepcopy(original["runtime_provenance"]["services"][0])
    second["service_identity_sha256"] = "5" * 64
    second["boot_identity_sha256"] = "6" * 64
    value = copy.deepcopy(original)
    value["runtime_provenance"]["services"].append(second)
    _rebind_runtime_state(ap, value)
    _validate_watermark_bytes(ap, ap._canonical_json(value))

    for fault in ("duplicate_service", "duplicate_boot", "unsorted", "stale_state"):
        broken = copy.deepcopy(value)
        services = broken["runtime_provenance"]["services"]
        if fault == "duplicate_service":
            services[1]["service_identity_sha256"] = services[0][
                "service_identity_sha256"
            ]
            _rebind_runtime_state(ap, broken)
        elif fault == "duplicate_boot":
            services[1]["boot_identity_sha256"] = services[0]["boot_identity_sha256"]
            _rebind_runtime_state(ap, broken)
        elif fault == "unsorted":
            services.reverse()
            _rebind_runtime_state(ap, broken)
        else:
            services[0]["sanitized_configuration"]["bytes"] += 1
        with pytest.raises(ap.ReleaseError):
            _validate_watermark_bytes(ap, ap._canonical_json(broken))


def test_runtime_observation_must_be_short_ordered_and_not_future_dated(ap: Any) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    original = _valid_control_watermark(ap, bundle["release-manifest.json"])
    for fault in ("equal", "wide", "future"):
        value = copy.deepcopy(original)
        observation = value["runtime_provenance"]["observation"]
        if fault == "equal":
            observation["pre_observed_at"] = "2026-08-14T13:00:00Z"
            observation["post_observed_at"] = "2026-08-14T13:00:00Z"
        elif fault == "wide":
            observation["pre_observed_at"] = "2026-08-14T12:00:00Z"
        else:
            value["runtime_provenance"]["services"][0][
                "boot_started_at"
            ] = "9999-01-01T00:00:00Z"
            observation["pre_observed_at"] = "9999-01-01T00:00:01Z"
            value["selection"]["lower_bound_exclusive_at"] = "9999-01-01T00:00:02Z"
            observation["post_observed_at"] = "9999-01-01T00:00:03Z"
            _rebind_runtime_state(ap, value)
        with pytest.raises(ap.ReleaseError, match="observation|future"):
            _validate_watermark_bytes(ap, ap._canonical_json(value))


def test_control_watermark_rejects_malformed_noncanonical_and_oversized_bytes(
    ap: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    value = _valid_control_watermark(ap, bundle["release-manifest.json"])
    canonical = ap._canonical_json(value)
    malformed = (
        b"",
        b"\xff",
        b"[]\n",
        b'{"schema":1,"schema":2}\n',
        b'{"schema":NaN}\n',
        b'{"schema":' + b"1" * 5000 + b"}\n",
        b"[" * 1500 + b"0" + b"]" * 1500,
        canonical + b" ",
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(),
    )
    for raw in malformed:
        with pytest.raises(ap.ReleaseError):
            _validate_watermark_bytes(ap, raw)

    monkeypatch.setattr(ap, "CONTROL_WATERMARK_MAX_BYTES", len(canonical) - 1)
    with pytest.raises(ap.ReleaseError, match="between"):
        _validate_watermark_bytes(ap, canonical)

    deeply_nested: Any = 0
    for _index in range(1500):
        deeply_nested = [deeply_nested]
    with pytest.raises(ap.ReleaseError, match="serialize deterministic JSON"):
        ap._canonical_json(deeply_nested)


def test_control_watermark_reverse_pin_uses_raw_manifest_hash_and_byte_count(ap: Any) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    manifest_raw = bundle["release-manifest.json"]
    original = _valid_control_watermark(ap, manifest_raw)
    for field, replacement in (
        ("sha256", "f" * 64),
        ("bytes", len(manifest_raw) + 1),
        ("bytes", True),
    ):
        value = copy.deepcopy(original)
        value["release_manifest"][field] = replacement
        with pytest.raises(ap.ReleaseError, match="manifest"):
            _validate_watermark_bytes(ap, ap._canonical_json(value))

    # JSON-equivalent trailing whitespace still changes the attested raw bytes.
    with pytest.raises(ap.ReleaseError, match="exact manifest bytes"):
        _validate_watermark_bytes(
            ap,
            ap._canonical_json(original),
            manifest_raw=manifest_raw + b" ",
        )


def test_manifest_is_written_last_and_content_is_durable_first(
    ap: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = {name: (name + "\n").encode() for name in ap.OUTPUT_ORDER}
    writes: list[str] = []
    original_write = ap._write_exclusive

    def observed_write(path: Path, raw: bytes) -> None:
        if path.name == "release-manifest.json":
            assert all((path.parent / name).is_file() for name in ap.CONTENT_OUTPUTS)
        writes.append(path.name)
        original_write(path, raw)

    monkeypatch.setattr(ap, "_write_exclusive", observed_write)
    root = tmp_path / "release"
    ap._publish_bundle(root, bundle)
    assert writes == list(ap.OUTPUT_ORDER)
    assert {path.name for path in root.iterdir()} == set(ap.OUTPUT_ORDER)


def test_pre_manifest_publication_fault_leaves_no_manifest(
    ap: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = {name: name.encode() for name in ap.OUTPUT_ORDER}
    original_write = ap._write_exclusive

    def failing_write(path: Path, raw: bytes) -> None:
        if path.name == "release-report.json":
            raise ap.ReleaseError("synthetic content write fault")
        original_write(path, raw)

    monkeypatch.setattr(ap, "_write_exclusive", failing_write)
    root = tmp_path / "release"
    with pytest.raises(ap.ReleaseError, match="synthetic content write fault"):
        ap._publish_bundle(root, bundle)
    assert not (root / "release-manifest.json").exists()


def test_manifest_write_and_final_fsync_faults_leave_no_publication_boundary(
    ap: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "partial-manifest"
    original_fsync = ap.os.fsync

    def fail_file_fsync(_descriptor: int) -> None:
        raise OSError("synthetic file fsync fault")

    monkeypatch.setattr(ap.os, "fsync", fail_file_fsync)
    with pytest.raises(ap.ReleaseError, match="cannot publish"):
        ap._write_exclusive(manifest, b"partial")
    assert not manifest.exists()
    monkeypatch.setattr(ap.os, "fsync", original_fsync)

    bundle = {name: name.encode() for name in ap.OUTPUT_ORDER}
    calls = 0
    original_directory_fsync = ap._fsync_directory

    def fail_final_once(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ap.ReleaseError("synthetic final directory fsync fault")
        original_directory_fsync(path)

    monkeypatch.setattr(ap, "_fsync_directory", fail_final_once)
    root = tmp_path / "release"
    with pytest.raises(ap.ReleaseError, match="manifest publication did not commit"):
        ap._publish_bundle(root, bundle)
    assert calls == 3
    assert not (root / "release-manifest.json").exists()


def test_existing_destination_never_runs_generator_or_overwrites(
    ap: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = 0

    def forbidden_generate(_repo: Path) -> dict[str, bytes]:
        nonlocal calls
        calls += 1
        raise AssertionError("generator must not run")

    monkeypatch.setattr(ap, "_generate_bundle", forbidden_generate)
    directory = tmp_path / "directory"
    directory.mkdir()
    marker = directory / "marker"
    marker.write_bytes(b"unchanged")
    regular = tmp_path / "regular"
    regular.write_bytes(b"unchanged")
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "missing")
    for destination in (directory, regular, broken):
        with pytest.raises(ap.ReleaseError, match="refusing to overwrite"):
            ap.build_release(destination, repo_root=REPO_ROOT)
    assert calls == 0
    assert marker.read_bytes() == b"unchanged"
    assert regular.read_bytes() == b"unchanged"
    assert broken.is_symlink()


@pytest.mark.parametrize("fault", ["lineage", "configuration", "reader", "focused proof"])
def test_prepublication_generation_faults_leave_destination_absent(
    ap: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault: str,
) -> None:
    def failing_generate(_repo: Path) -> dict[str, bytes]:
        raise ap.ReleaseError(f"synthetic {fault} fault")

    monkeypatch.setattr(ap, "_generate_bundle", failing_generate)
    root = tmp_path / "never-created"
    with pytest.raises(ap.ReleaseError, match=fault):
        ap.build_release(root, repo_root=REPO_ROOT)
    assert not root.exists()
    assert not root.is_symlink()


def test_synthetic_validator_accepts_exact_bundle_and_detects_regeneration_drift(
    ap: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle, lineage, implementation = _synthetic_bundle(ap)
    root = tmp_path / "candidate"
    _publish_finalized_candidate(ap, root, bundle)
    _configure_synthetic_validator(monkeypatch, ap, lineage, implementation, bundle)
    ap.validate_release(root, repo_root=tmp_path)

    changed = dict(bundle)
    changed["eval-compare.json"] += b" "
    monkeypatch.setattr(ap, "_generate_bundle", lambda _root: changed)
    with pytest.raises(ap.ReleaseError, match="fresh regeneration"):
        ap.validate_release(root, repo_root=tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "eval-compare.json",
        "release-report.json",
        "release-report.md",
        "release-manifest.json",
        "control-watermark.json",
    ],
)
def test_one_byte_tamper_of_every_output_is_rejected(
    ap: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
) -> None:
    bundle, lineage, implementation = _synthetic_bundle(ap)
    root = tmp_path / "candidate"
    _publish_finalized_candidate(ap, root, bundle)
    path = root / name
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 1
    path.write_bytes(raw)
    _configure_synthetic_validator(monkeypatch, ap, lineage, implementation, bundle)
    with pytest.raises(ap.ReleaseError):
        ap.validate_release(root, repo_root=tmp_path)


@pytest.mark.parametrize("name", ["eval-compare.json", "release-report.json", "release-report.md"])
def test_coherent_manifest_rehash_cannot_hide_tampered_content(
    ap: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
) -> None:
    bundle, lineage, implementation = _synthetic_bundle(ap)
    root = tmp_path / "candidate"
    _publish_finalized_candidate(ap, root, bundle)
    path = root / name
    if name == "eval-compare.json":
        value = json.loads(path.read_bytes())
        value["metrics"]["payload"]["after"]["median"] -= 1
        path.write_bytes(ap._canonical_json(value))
    elif name == "release-report.json":
        value = json.loads(path.read_bytes())
        value["overall"]["summary"] = "tampered but rehashed"
        path.write_bytes(ap._canonical_json(value))
    else:
        path.write_bytes(path.read_bytes() + b"tampered but rehashed\n")
    _rewrite_manifest_hash(ap, root, name)
    _write_valid_control_watermark(ap, root)
    _configure_synthetic_validator(monkeypatch, ap, lineage, implementation, bundle)
    expected_error = "Markdown" if name == "release-report.md" else "derive exactly"
    with pytest.raises(ap.ReleaseError, match=expected_error):
        ap.validate_release(root, repo_root=tmp_path)


def test_manifest_semantic_claim_tamper_is_caught_by_fresh_regeneration(
    ap: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle, lineage, implementation = _synthetic_bundle(ap)
    root = tmp_path / "candidate"
    _publish_finalized_candidate(ap, root, bundle)
    manifest_path = root / "release-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["privacy"]["semantic_holdout_reads"] = 1
    manifest_path.write_bytes(ap._canonical_json(manifest))
    _write_valid_control_watermark(ap, root)
    _configure_synthetic_validator(monkeypatch, ap, lineage, implementation, bundle)
    with pytest.raises(ap.ReleaseError, match="fresh regeneration"):
        ap.validate_release(root, repo_root=tmp_path)


@pytest.mark.parametrize("fault", ["missing", "renamed", "relocated", "extra", "directory"])
def test_candidate_requires_exact_five_root_members(
    ap: Any,
    tmp_path: Path,
    fault: str,
) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    root = tmp_path / "candidate"
    ap._publish_bundle(root, bundle)
    if fault == "missing":
        pass
    elif fault == "renamed":
        value = _valid_control_watermark(ap, bundle["release-manifest.json"])
        (root / "runtime-watermark.json").write_bytes(ap._canonical_json(value))
    elif fault == "relocated":
        nested = root / "runtime"
        nested.mkdir()
        value = _valid_control_watermark(ap, bundle["release-manifest.json"])
        (nested / ap.CONTROL_WATERMARK_NAME).write_bytes(ap._canonical_json(value))
    elif fault == "extra":
        _write_valid_control_watermark(ap, root)
        (root / "extra").write_bytes(b"extra")
    else:
        (root / ap.CONTROL_WATERMARK_NAME).mkdir()
    expected_error = "regular non-symlink" if fault == "directory" else "exactly"
    with pytest.raises(ap.ReleaseError, match=expected_error):
        ap._read_candidate_bundle(root)


@pytest.mark.parametrize("name", [
    "eval-compare.json",
    "release-report.json",
    "release-report.md",
    "release-manifest.json",
    "control-watermark.json",
])
def test_candidate_rejects_symlinked_members(
    ap: Any,
    tmp_path: Path,
    name: str,
) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    root = tmp_path / "candidate"
    _publish_finalized_candidate(ap, root, bundle)
    path = root / name
    outside = tmp_path / f"outside-{name}"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ap.ReleaseError, match="regular non-symlink"):
        ap._read_candidate_bundle(root)


def test_candidate_rejects_symlinked_release_root(ap: Any, tmp_path: Path) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    real_root = tmp_path / "real-candidate"
    _publish_finalized_candidate(ap, real_root, bundle)
    alias = tmp_path / "candidate-alias"
    alias.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ap.ReleaseError, match="release root"):
        ap._read_candidate_bundle(alias)


def test_candidate_member_swap_to_symlink_is_rejected_at_atomic_open(
    ap: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    root = tmp_path / "candidate"
    _publish_finalized_candidate(ap, root, bundle)
    watermark_path = root / ap.CONTROL_WATERMARK_NAME
    outside = tmp_path / "outside-watermark.json"
    outside.write_bytes(watermark_path.read_bytes())
    original_open = ap.os.open
    swapped = False

    def swapping_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal swapped
        if (
            not swapped
            and path == ap.CONTROL_WATERMARK_NAME
            and kwargs.get("dir_fd") is not None
        ):
            swapped = True
            watermark_path.unlink()
            watermark_path.symlink_to(outside)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(ap.os, "open", swapping_open)
    with pytest.raises(ap.ReleaseError, match="regular non-symlink"):
        ap._read_candidate_bundle(root)
    assert swapped is True


def test_candidate_reader_rejects_oversized_watermark_before_validation(
    ap: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    root = tmp_path / "candidate"
    _publish_finalized_candidate(ap, root, bundle)
    watermark_size = (root / ap.CONTROL_WATERMARK_NAME).stat().st_size
    monkeypatch.setattr(ap, "CONTROL_WATERMARK_MAX_BYTES", watermark_size - 1)
    with pytest.raises(ap.ReleaseError, match="exceeds"):
        ap._read_candidate_bundle(root)


def test_candidate_reader_rejects_fifo_without_blocking(
    ap: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    root = tmp_path / "candidate"
    ap._publish_bundle(root, bundle)
    os.mkfifo(root / ap.CONTROL_WATERMARK_NAME)
    real_open = os.open

    def assert_nonblocking_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if kwargs.get("dir_fd") is not None:
            assert flags & os.O_NONBLOCK
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(ap.os, "open", assert_nonblocking_open)
    with pytest.raises(ap.ReleaseError, match="regular non-symlink"):
        ap._read_candidate_bundle(root)


def test_candidate_reader_rechecks_exact_membership_after_atomic_reads(
    ap: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    root = tmp_path / "candidate"
    _publish_finalized_candidate(ap, root, bundle)
    original_open = ap.os.open
    inserted = False

    def inserting_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal inserted
        if (
            not inserted
            and path == ap.CONTROL_WATERMARK_NAME
            and kwargs.get("dir_fd") is not None
        ):
            inserted = True
            (root / "late-extra").write_bytes(b"extra")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(ap.os, "open", inserting_open)
    with pytest.raises(ap.ReleaseError, match="membership changed"):
        ap._read_candidate_bundle(root)
    assert inserted is True


def test_core_reader_accepts_only_four_and_candidate_reader_only_five(
    ap: Any,
    tmp_path: Path,
) -> None:
    bundle, _lineage, _implementation = _synthetic_bundle(ap)
    root = tmp_path / "candidate"
    ap._publish_bundle(root, bundle)
    assert tuple(ap._read_core_bundle(root)) == ap.OUTPUT_ORDER
    with pytest.raises(ap.ReleaseError, match="finalized release candidate"):
        ap._read_candidate_bundle(root)
    _write_valid_control_watermark(ap, root)
    assert tuple(ap._read_candidate_bundle(root)) == ap.CANDIDATE_MEMBERS
    with pytest.raises(ap.ReleaseError, match="release core"):
        ap._read_core_bundle(root)


def test_real_cli_build_validate_is_deterministic_and_never_launches_holdout_commands(
    ap: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    historical_pins = (
        ap.HISTORICAL_INPUTS
        + ap.RETIRED_V2_INPUTS
        + (ap.ORIGINAL_MANIFEST, ap.REPLACEMENT_MANIFEST)
    )
    before = {
        pin.path: hashlib.sha256((REPO_ROOT / pin.path).read_bytes()).hexdigest()
        for pin in historical_pins
    }
    for name, value in (
        ("LM_RECALL_REPEAT_GATING", "malformed"),
        ("LM_RECALL_REPEAT_DROP_TRAILING_STUBS", "1"),
        ("LM_DELIVERY_SNIPPET_CHARS", "999999"),
        ("PYTEST_ADDOPTS", "--capture=no"),
        ("PYTHONPATH", "/tmp/hostile-pythonpath"),
    ):
        monkeypatch.setenv(name, value)

    real_run = ap.subprocess.run
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def observed_run(command: Any, **kwargs: Any) -> Any:
        calls.append((list(command), kwargs))
        return real_run(command, **kwargs)

    monkeypatch.setattr(ap.subprocess, "run", observed_run)
    root = tmp_path / "real-release-v1"
    assert ap.main(["build", "--root", str(root)]) == 0
    assert {path.name for path in root.iterdir()} == set(ap.OUTPUT_ORDER)
    first = {name: (root / name).read_bytes() for name in ap.OUTPUT_ORDER}
    # The builder has no authority to fabricate runtime attestation bytes.
    assert ap.main(["validate", "--root", str(root)]) == 1
    assert len(calls) == 3
    _write_valid_control_watermark(ap, root)
    finalized = {
        name: (root / name).read_bytes() for name in ap.CANDIDATE_MEMBERS
    }
    assert ap.main(["validate", "--root", str(root)]) == 0
    assert {
        name: (root / name).read_bytes() for name in ap.CANDIDATE_MEMBERS
    } == finalized
    assert {name: (root / name).read_bytes() for name in ap.OUTPUT_ORDER} == first
    assert {path.name for path in root.iterdir()} == set(ap.CANDIDATE_MEMBERS)
    assert ap.main(["build", "--root", str(root)]) == 1
    output = capsys.readouterr()
    assert "built release-v1" in output.out
    assert "validated release-v1" in output.out
    assert "refusing to overwrite" in output.err

    allowed = {
        tuple([sys.executable, *ap.CALCULATOR_ARGS]),
        ap.CORRECTION_SUITE,
        ap.ERA_SUITE,
    }
    assert len(calls) == 6
    assert all(tuple(command) in allowed for command, _kwargs in calls)
    assert all("shell" not in kwargs for _command, kwargs in calls)
    assert all(
        not any("holdout" in part.lower() or "readiness" in part.lower() for part in command)
        for command, _kwargs in calls
    )
    assert all(
        not any(key.startswith(ap.OVERRIDE_PREFIXES) for key in kwargs["env"])
        for _command, kwargs in calls
    )

    report = json.loads((root / "release-report.json").read_bytes())
    manifest = json.loads((root / "release-manifest.json").read_bytes())
    eval_compare = json.loads((root / "eval-compare.json").read_bytes())
    p2 = report["measurements"]["P2_payload_and_retention"]
    p3 = report["measurements"]["P3_cross_scope"]
    p4 = report["measurements"]["P4_correction_dominance"]
    p5 = report["measurements"]["P5_era_safety"]
    p6 = report["measurements"]["P6_automatic_only_policy"]
    assert eval_compare["corpus_root"] == "artifacts/animal-planet/corpus"
    assert p2["eval"]["median"]["ratio"] == 0.5679226672
    assert p2["eval"]["p90"]["ratio"] == 0.5784398197
    assert p2["eval"]["retention"]["minimum_ratio"] == 1.0
    assert p3["eval"]["admission"]["reduction_ratio"] == 0.5583756345
    assert p3["eval"]["same_scope"]["retention_ratio"] == 0.9788025288
    assert p4["focused_suite"]["passed"] == 26
    assert "zero correction/superseded pairs" in p4["zero_pair_population_limitation"]
    assert p5["focused_suite"]["passed"] == 45
    assert p6["status"] == "deferred_pending_confirmatory_v3"
    assert p6["zero_gated_events"] == {"all_eval": 0, "unseen_in_dev": 0}
    assert report["overall"]["status"] == "pass_with_p6_deferred"
    assert manifest["publication"]["manifest_last"] is True
    assert manifest["publication"]["no_overwrite"] is True
    assert manifest["privacy"]["semantic_holdout_reads"] == 0
    assert manifest["measurement"]["corpus_root_is_relative"] is True
    assert manifest["measurement"]["sanitized_configuration"]["delivery_defaults"] == ap.DEFAULT_DELIVERY

    after = {
        pin.path: hashlib.sha256((REPO_ROOT / pin.path).read_bytes()).hexdigest()
        for pin in historical_pins
    }
    assert after == before
