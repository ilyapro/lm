from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import random
import sqlite3
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "recall_map_relevance_eval.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("recall_map_relevance_eval_tested", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R = _load_module()


def _result(
    node_id: str, *, score: float = 0.5, level: str = "trace"
) -> dict[str, object]:
    return {
        "node_id": node_id,
        "level": level,
        "score": score,
        "bm25_score": score,
        "vector_score": score / 2,
        "graph_score": 0.0,
        "trigger_score": 0.0,
    }


def _file_chunk_header(index: int = 1) -> str:
    return json.dumps(
        {
            "path": f"src/fixture_{index}.py",
            "kind": "source",
            "language": "python",
            "sha256": hashlib.sha256(f"fixture-{index}".encode()).hexdigest(),
            "chunk": f"{index}/{index + 1}",
            "lines": f"{index}-{index + 10}",
        },
        sort_keys=True,
    )


def _make_snapshot(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            level TEXT NOT NULL,
            content TEXT NOT NULL,
            context TEXT NOT NULL,
            provenance TEXT NOT NULL,
            source_traces TEXT NOT NULL,
            created_at TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0,
            usefulness_score REAL NOT NULL DEFAULT 0.0,
            last_accessed TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE recall_events (
            id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            scope TEXT,
            task TEXT,
            results TEXT NOT NULL,
            session_id TEXT,
            transport_session_id TEXT,
            ambient_context TEXT NOT NULL,
            created_at TEXT NOT NULL,
            recall_map TEXT
        );
        CREATE TABLE query_anchors (
            id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            decayed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE query_anchor_edges (
            anchor_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            weight REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )

    def add_node(node_id: str, content: str, *, level: str = "trace") -> None:
        connection.execute(
            "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                node_id,
                level,
                content,
                json.dumps({"shape": {"nested": [1, 2]}, "flag": True}),
                json.dumps({"kind": "fixture"}),
                "[]",
                "2026-07-01T00:00:00Z",
                "2026-07-01T00:00:00Z",
                0,
                0.0,
                None,
                "2026-07-01T00:00:00Z",
            ),
        )

    def add_event(
        event_id: str,
        at: str,
        task: str,
        transport: str,
        results: list[dict[str, object]],
        *,
        recall_map: dict[str, object] | None = None,
    ) -> None:
        connection.execute(
            "INSERT INTO recall_events VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                event_id,
                "fixture query",
                "project:fixture",
                task,
                json.dumps(results, sort_keys=True),
                f"logical-{transport}",
                transport,
                json.dumps({"task_pattern": task}),
                at,
                None if recall_map is None else json.dumps(recall_map, sort_keys=True),
            ),
        )

    # Many independent organic components make both deterministic sides of an
    # 80/20 component split overwhelmingly likely; the assertion below still
    # verifies the actual result rather than assuming a particular event id.
    for index in range(24):
        tail = f"organic-tail-{index:02d}"
        top = [f"organic-top-{index:02d}-{rank}" for rank in range(3)]
        for node_id in [*top, tail]:
            content = (
                f"[file-chunk] {_file_chunk_header(index + 1)}\nopaque body"
                if index == 0 and node_id == tail
                else f"durable fixture fact {index} {node_id}"
            )
            add_node(node_id, content, level=("trace", "concept", "schema")[index % 3])
        task = "map-linked" if index == 0 else f"organic-component-{index:02d}"
        transport = f"organic-transport-{index:02d}"
        day = 1 + index % 15
        add_event(
            f"organic-delivery-{index:02d}",
            f"2026-08-{day:02d}T00:00:00Z",
            task,
            transport,
            [
                _result(
                    node_id,
                    score=0.9 - rank * 0.1,
                    level=("trace", "concept", "schema")[index % 3],
                )
                for rank, node_id in enumerate([*top, tail])
            ],
        )
        later_results = [_result(tail)] if index % 2 == 0 else [_result(f"miss-{index:02d}")]
        add_event(
            f"organic-consumer-{index:02d}",
            f"2026-08-{day:02d}T01:00:00Z",
            task,
            transport,
            later_results,
        )

    map_node = "observed-map-node"
    add_node(map_node, "durable observed map candidate")
    payload = {
        "clusters": [
            {
                "label": "durable topic",
                "count": 2,
                "medoid": {"node_id": map_node, "score": 999.0, "residual_score": 777.0},
                "ask_hint": "durable topic",
                "plan_item": "recall durable topic",
            }
        ],
        "pool": 2,
        "covered": 2,
    }
    add_event(
        "map-delivery",
        "2026-08-20T00:00:00Z",
        "map-linked",
        "map-transport",
        [_result("map-top-0"), _result("map-top-1"), _result("map-top-2")],
        recall_map=payload,
    )
    add_event(
        "map-consumer",
        "2026-08-20T01:00:00Z",
        "map-linked",
        "map-transport",
        [_result(map_node)],
    )
    connection.commit()
    connection.close()


def test_sealed_hashes_are_checked_before_import(tmp_path: Path) -> None:
    bindings = R.load_frozen_bindings()
    assert bindings.prereg_sha256 == R.EXPECTED_PREREG_SHA256
    assert bindings.effect_sha256 == R.EXPECTED_EFFECT_SHA256
    assert bindings.protocol.horizon_hours == 24

    tampered = tmp_path / "recall_map_effect.py"
    marker = tmp_path / "executed"
    tampered.write_bytes(
        (ROOT / "scripts" / "recall_map_effect.py").read_bytes()
        + f"\nfrom pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n".encode()
    )
    with pytest.raises(R.EvaluationError, match="effect-tool hash mismatch"):
        R.load_frozen_bindings(effect_path=tampered)
    assert not marker.exists(), "tampered Python executed before its byte hash was rejected"


def test_form_classifiers_are_structural_and_keep_near_misses() -> None:
    assert (
        R.classify_content_form(f"[file-chunk] {_file_chunk_header(3)}\nbody")
        == "file_chunk_envelope"
    )
    assert R.classify_content_form('[file-chunk] {"chunk":"1/1"}\nbody') == "eligible"
    assert R.classify_content_form("[file-chunk] this is prose") == "eligible"
    assert R.classify_content_form("Strategy stagnation detected on retry-loop") == "strategy_stagnation"
    assert (
        R.classify_content_form("A user noted: Strategy stagnation detected on retry-loop")
        == "eligible"
    )
    assert (
        R.classify_content_form("structured payload", {"lesson_kind": "monitoring-journal"})
        == "supervision_journal"
    )


def test_independently_generated_fixed_seed_positive_variants() -> None:
    rng = random.Random(0x51A7C0DE)
    for index in range(48):
        part = rng.randint(1, 20)
        total = rng.randint(part, part + 20)
        start = rng.randint(1, 5_000)
        header = {
            "language": rng.choice(["python", "rust", "markdown", "text"]),
            "path": f"pkg/{rng.randrange(1_000_000):06d}/module.py",
            "lines": f"{start}-{start + rng.randint(0, 200)}",
            "chunk": f"{part}/{total}",
            "kind": rng.choice(["source", "test", "documentation"]),
            "sha256": f"{rng.getrandbits(256):064x}",
        }
        encoded = json.dumps(header, separators=(",", ":"), sort_keys=bool(index % 2))
        assert (
            R.classify_content_form(
                f"[file-chunk]{rng.choice([' ', '  ', chr(9)])}{encoded}\r\n```\nbody\n```"
            )
            == "file_chunk_envelope"
        )

        details = rng.sample(
            ["attempts", "strategy", "window", "reason"], rng.randrange(5)
        )
        stagnation = f"Strategy stagnation detected on branch-{rng.randrange(1_000_000)}"
        stagnation += "".join(f"\n{name}: value-{index}" for name in details)
        assert R.classify_content_form(stagnation) == "strategy_stagnation"

        kind = rng.choice(sorted(R.SUPERVISION_KINDS))
        key = rng.choice(["kind", "type", "lesson_kind", "record_kind"])
        mapping = {key: kind.replace("_", rng.choice(["_", "-"]))}
        if rng.randrange(2):
            assert R.classify_content_form("opaque structured row", mapping, {}) == "supervision_journal"
        else:
            assert R.classify_content_form("opaque structured row", {}, mapping) == "supervision_journal"


@pytest.mark.parametrize(
    ("content", "context", "provenance"),
    [
        (
            'I quoted [file-chunk] {"path":"mine.py","chunk":"1/1"} while explaining indexing.',
            {},
            {},
        ),
        ('[file-chunk] {"chunk":"1/1","note":"this is my outline"}', {}, {}),
        ("[file-chunk] ordinary prose after a marker", {}, {}),
        (
            '[file-chunk] {"path":"mine.py","kind":"source","language":"python",'
            '"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
            '"chunk":"9/1","lines":"20-10"}',
            {},
            {},
        ),
        ("A user noted: Strategy stagnation detected on retry-loop", {}, {}),
        ("Strategy stagnation was detected while I revised this paragraph.", {}, {}),
        (
            "Strategy stagnation detected on retry-loop\nThis is my diagnosis, not a watchdog row.",
            {},
            {},
        ),
        ("I maintain a supervision_journal by hand.", {}, {}),
        ("[supervision_journal] is a token discussed in this sentence.", {}, {}),
        ("My monitoring journal is prose.", {"topic": "monitoring_journal"}, {}),
    ],
)
def test_user_authored_near_misses_remain_eligible(
    content: str, context: dict[str, object], provenance: dict[str, object]
) -> None:
    assert R.classify_content_form(content, context, provenance) == "eligible"


def test_published_form_control_audit_is_aggregate_and_error_free() -> None:
    first = R.ballast_control_audit()
    second = R.ballast_control_audit()
    assert first == second
    assert first["seed"] == R.BALLAST_CONTROL_SEED
    assert first["cases"] == 3 * R.BALLAST_CONTROL_VARIANTS_PER_CLASS + len(
        R.USER_AUTHORED_NEAR_MISSES
    )
    assert first["false_positives"] == first["false_negatives"] == 0
    assert first["case_text_published"] is False
    serialized = R.canonical_json(first)
    assert "retry-loop" not in serialized
    assert "generated_" not in serialized


def test_source_qualified_connected_components_force_map_eval() -> None:
    records = [
        R.ItemRecord(source="one", arm="organic", event_id="organic", created_at="x"),
        R.ItemRecord(source="one", arm="observed_map", event_id="mapped", created_at="x"),
        R.ItemRecord(source="two", arm="organic", event_id="separate", created_at="x"),
    ]
    metas = {
        ("one", "organic"): SimpleNamespace(
            qualified_cache_key="one\0cache\0same", qualified_sessions=()
        ),
        ("one", "mapped"): SimpleNamespace(
            qualified_cache_key="one\0cache\0same", qualified_sessions=()
        ),
        # Same unqualified spelling is deliberately a different source token.
        ("two", "separate"): SimpleNamespace(
            qualified_cache_key="two\0cache\0same", qualified_sessions=()
        ),
    }
    components = R.assign_component_splits(records, metas)
    assert records[0].component_id == records[1].component_id
    assert records[0].split == records[1].split == "eval"
    assert records[2].component_id != records[0].component_id
    assert components[records[0].component_id]["reason"] == "touches_observed_map_delivery"


def test_cache_and_both_session_identities_bridge_even_without_item_rows() -> None:
    records = [
        R.ItemRecord(source="one", arm="organic", event_id="organic", created_at="x"),
        R.ItemRecord(source="one", arm="observed_map", event_id="mapped", created_at="x"),
        R.ItemRecord(source="two", arm="organic", event_id="other-source", created_at="x"),
    ]
    metas = {
        ("one", "organic"): SimpleNamespace(
            qualified_cache_key="one\0cache\0organic",
            qualified_sessions=("one\0transport\0bridge-a",),
        ),
        # This event deliberately has no ItemRecord. Its transport identity
        # joins the organic event while its logical session joins the map.
        ("one", "journal-only-bridge"): SimpleNamespace(
            qualified_cache_key="one\0cache\0bridge",
            qualified_sessions=(
                "one\0transport\0bridge-a",
                "one\0session\0bridge-b",
            ),
        ),
        ("one", "mapped"): SimpleNamespace(
            qualified_cache_key="one\0cache\0mapped",
            qualified_sessions=("one\0session\0bridge-b",),
        ),
        # Identical unqualified session spelling on another source must not join.
        ("two", "other-source"): SimpleNamespace(
            qualified_cache_key="two\0cache\0organic",
            qualified_sessions=("two\0transport\0bridge-a",),
        ),
    }
    components = R.assign_component_splits(
        records,
        metas,
        component_event_keys=metas,
        forced_eval_event_keys={("one", "mapped")},
    )
    assert records[0].component_id == records[1].component_id
    assert records[0].split == records[1].split == "eval"
    assert records[2].component_id != records[0].component_id
    assert components[records[0].component_id]["reason"] == "touches_observed_map_delivery"
    assert not ({record.component_id for record in records if record.split == "train"} & {
        record.component_id for record in records if record.split == "eval"
    })


def test_past_history_is_strictly_matured_before_delivery() -> None:
    history = {
        "n": [
            ("2026-08-01T23:00:00Z", "2026-07-31T23:00:00Z", False),
            ("2026-08-02T01:00:00Z", "2026-08-01T01:00:00Z", True),
            # A malformed/future delivery can never become prior history merely
            # because its claimed outcome timestamp sorts before the decision.
            ("2026-08-01T20:00:00Z", "2026-08-03T00:00:00Z", False),
        ]
    }
    before_second_matures = R.matured_past_features(history, "n", "2026-08-02T00:00:00Z")
    after_second_matures = R.matured_past_features(history, "n", "2026-08-02T01:00:00Z")
    assert before_second_matures["prior_consumption_rate"] == 0.0
    assert before_second_matures["prior_nonconsumption_streak_log"] > 0
    assert after_second_matures["prior_consumption_rate"] == 0.5
    assert after_second_matures["prior_nonconsumption_streak_log"] == 0.0


def test_model_feature_allowlist_rejects_identity_and_mutated_stats() -> None:
    for forbidden in (
        "node_id",
        "label_tokens",
        "task_name",
        "host",
        "source_identity",
        "cache_key",
        "transport_session",
        "current_access_count",
        "usefulness_score",
        "last_accessed",
        "updated_at_usage",
    ):
        names = list(R.PRIMARY_MODEL_FEATURES)
        names[-1] = forbidden
        with pytest.raises(R.EvaluationError):
            R.validate_model_feature_names(names)


def test_map_rows_never_enter_fitting_even_if_marked_train() -> None:
    organic: list[object] = []
    for index in range(12):
        features = {
            name: float((index + position) % 4)
            for position, name in enumerate(R.PRIMARY_MODEL_FEATURES)
        }
        organic.append(
            R.ItemRecord(
                source="fixture",
                arm="organic",
                event_id=f"organic-{index}",
                created_at="2026-08-01T00:00:00Z",
                split="train",
                consumed=bool(index % 3 == 0),
                features=features,
            )
        )
    baseline = R.fit_primary_model(organic)
    adversarial_map = R.ItemRecord(
        source="fixture",
        arm="observed_map",
        event_id="map",
        created_at="2026-08-20T00:00:00Z",
        # A corrupt split marker and extreme label/features still cannot pass
        # the arm allowlist in fit_primary_model.
        split="train",
        consumed=True,
        features={name: 1e12 for name in R.PRIMARY_MODEL_FEATURES},
    )
    assert R.fit_primary_model([*organic, adversarial_map]) == baseline


def test_end_to_end_evidence_is_deterministic_aggregate_and_verifiable(tmp_path: Path) -> None:
    snapshot = tmp_path / "frozen.sqlite3"
    _make_snapshot(snapshot)
    source = R.SourceSpec("fixture", snapshot)

    first_manifest, first_analysis = R.evaluate_sources([source])
    second_manifest, second_analysis = R.evaluate_sources([source])
    assert R.canonical_json(first_manifest) == R.canonical_json(second_manifest)
    assert R.canonical_json(first_analysis) == R.canonical_json(second_analysis)
    assert not R.verify_artifacts(first_manifest, first_analysis)

    assert first_manifest["train"]["items"] > 0
    assert first_manifest["eval"]["organic"]["items"] > 0
    assert first_manifest["eval"]["observed_map"]["items"] == 1
    assert first_manifest["eval"]["observed_map"]["consumed"] == 1
    assert not (
        set(first_manifest["train"]["components"])
        & set(first_manifest["eval"]["components"])
    )
    assert first_analysis["primary_model"]["fit_cohort"] == "organic_train_only"
    assert first_analysis["primary_model"]["fit_items"] == first_manifest["train"]["items"]
    assert set(first_analysis["feature_policy"]["primary_model_features"]) == set(
        R.PRIMARY_MODEL_FEATURES
    )
    assert not (
        set(first_analysis["feature_policy"]["primary_model_features"])
        & {
            "form_machine_ballast",
            "provenance_log_key_count",
            "context_log_key_count",
            "cascade_anchor",
            "score",
        }
    )
    assert (
        first_analysis["availability"]["observed_map_eval"]["recorded_delivery_scores"][
            "reasons"
        ]["not_recorded_in_recall_map_payload"]
        == 1
    )
    assert first_analysis["content_forms"]["organic_eval"]["file_chunk_envelope"]["items"] == 1
    assert first_analysis["leakage_audit"]["future_mutated_node_stats_read"] is False
    assert first_analysis["leakage_audit"]["map_rows_in_fit"] == 0
    assert first_analysis["privacy_audit"]["exact_raw_string_matches"] == 0
    assert first_analysis["form_classification_audit"]["synthetic"]["false_positives"] == 0
    assert first_analysis["form_classification_audit"]["synthetic"]["false_negatives"] == 0
    assert (
        first_analysis["form_classification_audit"]["real"]["observed_map_eval"]["items"]
        == first_manifest["eval"]["observed_map"]["items"]
    )
    for cohort, availability in first_analysis["availability"].items():
        if availability["cascade_stage"]["items"] == 0:
            assert availability["cascade_stage"]["reasons"] == {}
            continue
        assert availability["cascade_stage"]["reasons"] == {
            "not_recorded_at_delivery_not_reconstructed": availability["cascade_stage"]["items"]
        }, cohort
        assert availability["historical_context_provenance"]["reasons"] == {
            "not_versioned_snapshot_diagnostics_excluded_from_fitting": availability[
                "historical_context_provenance"
            ]["items"]
        }, cohort
    for feature in R.SCORE_FIELDS:
        assert first_analysis["associations"][feature]["observed_map_eval"]["available"] == 0
    assert all(
        R.OPAQUE_DIGEST_RE.fullmatch(component)
        for component in [
            *first_manifest["train"]["components"],
            *first_manifest["eval"]["components"],
        ]
    )

    serialized = R.canonical_json(first_manifest) + R.canonical_json(first_analysis)
    for secret in (
        "organic-tail-00",
        "map-delivery",
        "map-linked",
        "map-transport",
        "durable observed map candidate",
    ):
        assert secret not in serialized

    rendered = R.render_markdown(first_manifest, first_analysis)
    assert "organic train" in rendered
    assert "candidate holdout was not accessed" in rendered


def _semantic_outputs(
    manifest: dict[str, object], analysis: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    stable_manifest = deepcopy(manifest)
    stable_analysis = deepcopy(analysis)
    stable_manifest.pop("sources")
    stable_analysis.pop("dataset_manifest_sha256")
    stable_analysis.pop("privacy_audit")
    return stable_manifest, stable_analysis


def test_future_mutated_stats_and_post_horizon_rows_cannot_change_features_or_labels(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "frozen.sqlite3"
    _make_snapshot(snapshot)
    source = R.SourceSpec("fixture", snapshot)
    before_manifest, before_analysis = R.evaluate_sources([source])
    before_snapshot_hash = before_manifest["sources"][0]["snapshot_sha256"]

    connection = sqlite3.connect(snapshot)
    connection.execute(
        "UPDATE nodes SET access_count=999999, usefulness_score=-123.5, "
        "last_accessed='2026-08-22T17:59:59Z', updated_at='2026-08-22T17:59:59Z'"
    )
    connection.execute(
        "INSERT INTO recall_events VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            "post-horizon-row",
            "private post-horizon query",
            "project:fixture",
            "map-linked",
            json.dumps([_result("observed-map-node")]),
            "logical-map-transport",
            "map-transport",
            json.dumps({"task_pattern": "map-linked"}),
            "2026-08-22T17:00:00Z",
            None,
        ),
    )
    connection.commit()
    connection.close()

    after_manifest, after_analysis = R.evaluate_sources([source])
    assert after_manifest["sources"][0]["snapshot_sha256"] != before_snapshot_hash
    assert _semantic_outputs(before_manifest, before_analysis) == _semantic_outputs(
        after_manifest, after_analysis
    )


def test_mutable_historical_context_and_provenance_are_snapshot_only_not_fit_inputs(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "frozen.sqlite3"
    _make_snapshot(snapshot)
    source = R.SourceSpec("fixture", snapshot)
    before_manifest, before_analysis = R.evaluate_sources([source])

    connection = sqlite3.connect(snapshot)
    connection.execute(
        "UPDATE nodes SET context=?, provenance=?, updated_at=?",
        (
            json.dumps({"lesson_kind": "supervision_journal", "late": {"shape": [1, 2, 3]}}),
            json.dumps({"record_kind": "monitoring_journal", "late_source": "private"}),
            "2026-08-22T17:59:59Z",
        ),
    )
    connection.commit()
    connection.close()

    after_manifest, after_analysis = R.evaluate_sources([source])
    assert after_analysis["primary_model"] == before_analysis["primary_model"]
    for location in (
        ("train",),
        ("eval", "organic"),
        ("eval", "observed_map"),
    ):
        before: object = before_manifest
        after: object = after_manifest
        for key in location:
            before = before[key]
            after = after[key]
        assert before["items"] == after["items"]
        assert before["consumed"] == after["consumed"]
    assert before_analysis["content_forms"] != after_analysis["content_forms"]
    assert after_analysis["feature_policy"]["snapshot_only_role"].endswith(
        "excluded from fitting"
    )
    for availability in after_analysis["availability"].values():
        if availability["historical_context_provenance"]["items"]:
            assert availability["historical_context_provenance"]["reasons"] == {
                "not_versioned_snapshot_diagnostics_excluded_from_fitting": availability[
                    "historical_context_provenance"
                ]["items"]
            }


def test_artifact_verifier_rejects_privacy_split_fit_and_audit_claim_tampering(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "frozen.sqlite3"
    _make_snapshot(snapshot)
    manifest, analysis = R.evaluate_sources([R.SourceSpec("fixture", snapshot)])

    bad_manifest = deepcopy(manifest)
    bad_analysis = deepcopy(analysis)
    component = bad_manifest["train"]["components"][0]
    bad_manifest["eval"]["components"].append(component)
    bad_manifest["train"]["components"].append("raw-session-name")
    bad_analysis["primary_model"]["fit_items"] += 1
    bad_analysis["form_classification_audit"]["synthetic"]["false_positives"] = 1
    bad_analysis["privacy_audit"]["exact_raw_string_matches"] = 1
    bad_analysis["leakage_audit"]["map_rows_in_fit"] = 1
    bad_analysis["content"] = "organic-tail-00"
    bad_manifest["sources"][0]["redacted_path"] = "/private/source.sqlite3"
    problems = R.verify_artifacts(bad_manifest, bad_analysis, check_evaluator_hash=False)
    assert any("component overlap" in problem for problem in problems)
    assert any("non-opaque" in problem for problem in problems)
    assert any("fit denominator" in problem for problem in problems)
    assert any("classification errors" in problem for problem in problems)
    assert any("privacy audit" in problem for problem in problems)
    assert any("sensitive artifact key" in problem for problem in problems)
    assert any("canonically redacted" in problem for problem in problems)
    assert any("map fitting-row" in problem for problem in problems)


def test_candidate_holdout_path_is_fail_closed(tmp_path: Path) -> None:
    prohibited = tmp_path / "artifacts" / "recall-map" / "relevance" / "field" / "manifest.json"
    with pytest.raises(R.EvaluationError, match="candidate-holdout access is prohibited"):
        R._assert_not_holdout(prohibited)


def test_cli_runs_verifies_source_and_renders(tmp_path: Path) -> None:
    snapshot = tmp_path / "source.sqlite3"
    manifest = tmp_path / "manifest.json"
    analysis = tmp_path / "analysis.json"
    report = tmp_path / "report.md"
    _make_snapshot(snapshot)
    source_arg = f"fixture={snapshot}"
    assert (
        R.main(
            [
                "run",
                "--source",
                source_arg,
                "--manifest-out",
                str(manifest),
                "--analysis-out",
                str(analysis),
            ]
        )
        == 0
    )
    assert manifest.read_text().endswith("\n")
    assert analysis.read_text().endswith("\n")
    assert (
        R.main(
            [
                "verify",
                "--manifest",
                str(manifest),
                "--analysis",
                str(analysis),
                "--source",
                source_arg,
            ]
        )
        == 0
    )
    assert (
        R.main(
            [
                "render",
                "--manifest",
                str(manifest),
                "--analysis",
                str(analysis),
                "--out",
                str(report),
            ]
        )
        == 0
    )
    assert report.read_text().startswith("# Historical recall-map relevance evidence")


def test_cli_rerun_is_byte_identical_across_python_hash_seeds(tmp_path: Path) -> None:
    snapshot = tmp_path / "source.sqlite3"
    _make_snapshot(snapshot)
    outputs: list[tuple[bytes, bytes]] = []
    for hash_seed in ("1", "987654321"):
        manifest = tmp_path / f"manifest-{hash_seed}.json"
        analysis = tmp_path / f"analysis-{hash_seed}.json"
        environment = dict(os.environ)
        environment.update({"PYTHONPATH": str(ROOT / "src"), "PYTHONHASHSEED": hash_seed})
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "run",
                "--source",
                f"fixture={snapshot}",
                "--manifest-out",
                str(manifest),
                "--analysis-out",
                str(analysis),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append((manifest.read_bytes(), analysis.read_bytes()))
    assert outputs[0] == outputs[1]
