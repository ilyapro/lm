"""Privacy-safe unseen-in-dev measurement for the animal-planet calculator."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ap_baseline.py"
NODE_ID = "01TESTNODE00000000000000000"


@pytest.fixture(scope="module")
def ap() -> Any:
    spec = importlib.util.spec_from_file_location("ap_baseline_unseen_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _stable_replay_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_RECALL_REPEAT_GATING", "1")
    monkeypatch.setenv("LM_RECALL_REPEAT_DROP_TRAILING_STUBS", "1")
    for name in (
        "LM_RECALL_REPEAT_MIN_UNLINKED",
        "LM_RECALL_REPEAT_MAX_LINK_RATE",
        "LM_RECALL_REPEAT_MIN_SESSIONS",
        "LM_RECALL_REPEAT_PROBE_EVERY",
    ):
        monkeypatch.delenv(name, raising=False)


def _token(label: str) -> str:
    return hashlib.sha256(f"synthetic opaque family {label}".encode()).hexdigest()


def _node() -> dict[str, Any]:
    return {
        "type": "node",
        "id": NODE_ID,
        "level": "trace",
        "scope": "project:synthetic",
        "content_surrogate": "synthetic repeat-gating content " + "x" * 2400,
        "context_surrogate": {"scope": "project:synthetic"},
        "timestamp": "2026-08-13T00:00:00Z",
        "created_at": "2026-08-13T00:00:00Z",
        "updated_at": "2026-08-13T00:00:00Z",
        "stats": {},
        "source_traces": [],
        "corrections": [],
        "relations": [],
    }


def _event(
    number: int,
    *,
    event_class: str = "automatic",
    query: str | None = None,
    token: str | None = None,
    feedback_at: str | None = None,
) -> dict[str, Any]:
    created_at = f"2026-08-13T00:{number:02d}:00Z"
    event = {
        "type": "event",
        "id": f"01TESTEVENT{number:015d}",
        "class": event_class,
        "created_at": created_at,
        "query_surrogate": query if query is not None else f"surrogate query {number}",
        "scope": "project:synthetic",
        "requested_scope": "project:synthetic",
        "resolved_scopes": ["project:synthetic"],
        "transport_session_id": f"transport-{number}",
        "feedback_applied": feedback_at is not None,
        "feedback_applied_at": feedback_at,
        "max_results": 5,
        "results": [
            {
                "node_id": NODE_ID,
                "rank": 1,
                "score": 1.0,
                "bm25_score": 1.0,
                "vector_score": 0.0,
                "graph_score": 0.0,
                "trigger_score": 0.0,
                "methods": ["bm25"],
                "path": [],
            }
        ],
    }
    if token is not None:
        event["fingerprint_token"] = token
    return event


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _index_document(
    _ap: Any,
    *,
    seen_token: str,
) -> dict[str, Any]:
    tokens = {seen_token}
    counter = 0
    while len(tokens) < 353:
        tokens.add(_token(f"dev-{counter}"))
        counter += 1
    ordered = sorted(tokens)
    return {
        "schema_version": 1,
        "kind": "frozen-dev-automatic-fingerprint-token-index",
        "algorithm": "HMAC-SHA256",
        "normalization": '" ".join(query.split()) + "\\n" + requested_scope',
        "population_events": 732,
        "unique_tokens": len(ordered),
        "tokens": ordered,
    }


def _write_packet_manifest(
    corpus_root: Path,
    *,
    split: str,
    index_path: Path | None = None,
    automatic_dev_events: int | None = None,
) -> None:
    records = [
        json.loads(line)
        for line in (corpus_root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    events = [record for record in records if record.get("type") == "event"]
    files: dict[str, Any] = {
        f"corpus/{split}.jsonl": {
            "sha256": hashlib.sha256(
                (corpus_root / f"{split}.jsonl").read_bytes()
            ).hexdigest()
        }
    }
    if index_path is not None:
        files["corpus/dev-fingerprint-index.json"] = {
            "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest()
        }
        index = json.loads(index_path.read_text(encoding="utf-8"))
        dev_tokens = set(index["tokens"])
        by_token: dict[Any, list[dict[str, Any]]] = {}
        for event in events:
            if event.get("class") == "automatic":
                by_token.setdefault(event.get("fingerprint_token"), []).append(event)
        repeated = {
            token: rows
            for token, rows in by_token.items()
            if len(rows) >= 3
            and len({row.get("transport_session_id") for row in rows}) >= 2
        }
        unseen = {token: rows for token, rows in repeated.items() if token not in dev_tokens}
        repeated_events = sum(map(len, repeated.values()))
        unseen_events = sum(map(len, unseen.values()))
        automatic_events = sum(event.get("class") == "automatic" for event in events)
        organic_events = sum(event.get("class") == "organic" for event in events)
        project_ae_events = sum(
            event.get("requested_scope") == "project:ae" for event in events
        )
        project_online_events = sum(
            event.get("requested_scope") == "project:online" for event in events
        )
        nodes = [record for record in records if record.get("type") == "node"]
        verifier_path = corpus_root.parent / "recipe" / "verify.py"
        verifier_path.parent.mkdir(parents=True, exist_ok=True)
        verifier_path.write_text("# synthetic independently reviewed verifier\n", encoding="utf-8")
        receipt = {
            "schema_version": 1,
            "mode": "keyed-preseal",
            "status": "pass",
            "mismatches": 0,
            "token_collisions": 0,
            "semantic_reads": 0,
            "supplement_events": len(events),
            "automatic_events": automatic_events,
            "organic_events": organic_events,
            "project_ae_events": project_ae_events,
            "project_online_events": project_online_events,
            "direct_nodes": len(nodes),
            "nodes": len(nodes),
            "dev_automatic_events": 732,
            "dev_unique_tokens": 353,
            "repeated_automatic_families": len(repeated),
            "repeated_automatic_events": repeated_events,
            "unseen_in_dev_families": len(unseen),
            "unseen_in_dev_events": unseen_events,
        }
        original_manifest = REPO_ROOT / "artifacts" / "animal-planet" / "manifest.json"
        document = {
            "schema_version": 1,
            "packet": "animal-planet P6 replacement temporal supplement",
            "namespace": "replacement-holdout",
            "frozen": True,
            "semantic_reads": 0,
            "identity": {
                "algorithm": "HMAC-SHA256",
                "normalization": '" ".join(query.split()) + "\\n" + requested_scope',
                "one_key_for_dev_and_supplement": True,
                "key_persisted": False,
                "unkeyed_query_fingerprints_persisted": False,
            },
            "sources": {
                "original_packet_manifest": {
                    "sha256": hashlib.sha256(original_manifest.read_bytes()).hexdigest(),
                    "frozen": True,
                }
            },
            "implementation": {
                "verifier": {
                    "path": "recipe/verify.py",
                    "sha256": hashlib.sha256(verifier_path.read_bytes()).hexdigest(),
                }
            },
            "counts": {
                "supplement": {
                    "events": len(events),
                    "automatic": automatic_events,
                    "organic": organic_events,
                },
                "frozen_dev": {
                    "automatic": 732,
                    "unique_automatic_tokens": 353,
                },
                "repeated_automatic": {
                    "families": len(repeated),
                    "events": repeated_events,
                    "unseen_in_dev_families": len(unseen),
                    "unseen_in_dev_events": unseen_events,
                },
            },
            "files": files,
            "validation": {
                "keyed_preseal": receipt,
                "candidate_unchanged_after_validation": True,
            },
        }
    else:
        document = {"schema_version": 1, "files": files}
    if automatic_dev_events is not None:
        document["corpus"] = {
            "counts": {
                "dev": {"by_class": {"automatic": automatic_dev_events}}
            }
        }
    (corpus_root.parent / "manifest.json").write_text(
        json.dumps(document, sort_keys=True), encoding="utf-8"
    )


def _run_opaque(
    ap: Any,
    tmp_path: Path,
    events: list[dict[str, Any]],
    *,
    seen_token: str,
    name: str,
) -> dict[str, Any]:
    corpus_root = tmp_path / name / "corpus"
    _write_jsonl(corpus_root / "holdout.jsonl", [_node(), *events])
    index_path = corpus_root / "dev-fingerprint-index.json"
    index_path.write_text(
        json.dumps(
            _index_document(
                ap,
                seen_token=seen_token,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_packet_manifest(
        corpus_root, split="holdout", index_path=index_path
    )
    lm = ap._import_living_memory()
    corpus = ap.ReplayCorpus(corpus_root, "holdout", lm)
    delivery = ap.replay_delivery(corpus)
    return ap.family_repeat_gating(
        corpus, delivery, dev_fingerprint_index=index_path
    )


def test_real_eval_unseen_aggregate_matches_frozen_reference() -> None:
    env = dict(os.environ)
    env["LIVING_MEMORY_EMBEDDING_BACKEND"] = "hash"
    env["LM_RECALL_REPEAT_GATING"] = "1"
    env["LM_RECALL_REPEAT_DROP_TRAILING_STUBS"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "compare",
            "--split",
            "eval",
            "--metrics",
            "auto_recall",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    gate = result["metrics"]["auto_recall"]["repeat_gating"]
    unseen = gate["unseen_in_dev"]

    assert unseen["reference_split"] == "dev"
    assert unseen["population"]["automatic_events_in_dev"] == 732
    assert unseen["population"]["automatic_fingerprints_in_dev"] == 353
    assert unseen["population"]["repeated_fingerprints"] == 1
    assert unseen["population"]["repeated_fingerprint_events"] == 3
    assert unseen["chars"]["repeated_automatic"]["reduction_ratio"] == 0.0
    # Existing split-wide measurement remains unchanged by the additive slice.
    assert gate["population"]["repeated_fingerprints"] == 7
    assert gate["population"]["repeated_fingerprint_events"] == 204
    assert gate["population"]["gated_events"] == 169


def test_manifest_only_replacement_receipt_schema_matches_and_gates_synthetic_packet(
    ap: Any, tmp_path: Path
) -> None:
    seen = _token("receipt-schema-seen")
    unseen = _token("receipt-schema-unseen")
    events = [
        _event(number, event_class="organic", token=unseen)
        for number in range(1, 6)
    ]
    events += [_event(number, token=unseen) for number in range(6, 9)]
    corpus_root = tmp_path / "receipt-schema" / "corpus"
    _write_jsonl(corpus_root / "eval.jsonl", [_node(), *events])
    index_path = corpus_root / "dev-fingerprint-index.json"
    index_path.write_text(
        json.dumps(_index_document(ap, seen_token=seen), sort_keys=True),
        encoding="utf-8",
    )
    _write_packet_manifest(corpus_root, split="eval", index_path=index_path)

    synthetic_manifest = json.loads(
        (corpus_root.parent / "manifest.json").read_text(encoding="utf-8")
    )
    sealed_manifest = json.loads(
        (
            REPO_ROOT
            / "artifacts"
            / "animal-planet"
            / "evaluation"
            / "replacement-holdout"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    synthetic_receipt = synthetic_manifest["validation"]["keyed_preseal"]
    sealed_receipt = sealed_manifest["validation"]["keyed_preseal"]
    assert frozenset(synthetic_receipt) == frozenset(sealed_receipt)

    lm = ap._import_living_memory()
    corpus = ap.ReplayCorpus(corpus_root, "eval", lm)
    gate = ap.family_repeat_gating(
        corpus,
        ap.replay_delivery(corpus),
        dev_fingerprint_index=index_path,
    )
    assert gate["unseen_in_dev"]["population"] == {
        "automatic_events_in_dev": 732,
        "automatic_fingerprints_in_dev": 353,
        "repeated_fingerprints": 1,
        "repeated_fingerprint_events": 3,
        "gated_events": 2,
    }


def test_opaque_identity_replays_shared_organic_state_without_disclosure(
    ap: Any, tmp_path: Path
) -> None:
    seen = _token("seen")
    unseen = _token("unseen")
    events = [
        _event(number, event_class="organic", token=unseen)
        for number in range(1, 6)
    ]
    events += [_event(number, token=unseen) for number in range(6, 9)]
    events += [_event(number, token=seen) for number in range(9, 12)]

    gate = _run_opaque(ap, tmp_path, events, seen_token=seen, name="shared-state")
    aggregate = gate["unseen_in_dev"]
    assert aggregate["population"] == {
        "automatic_events_in_dev": 732,
        "automatic_fingerprints_in_dev": 353,
        "repeated_fingerprints": 1,
        "repeated_fingerprint_events": 3,
        "gated_events": 2,
    }
    assert aggregate["chars"]["repeated_automatic"]["reduction_ratio"] > 0.5
    assert aggregate["retention"]["repeated_automatic_content_access"]["events"] == 3

    rendered = json.dumps(gate, sort_keys=True)
    assert "fingerprint_token" not in rendered
    assert seen not in rendered
    assert unseen not in rendered
    assert all(event["id"] not in rendered for event in events)


def test_compare_cli_accepts_external_dev_index_without_exposing_it(
    ap: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _token("cli-seen")
    unseen = _token("cli-unseen")
    events = [
        _event(number, event_class="organic", token=unseen)
        for number in range(1, 6)
    ]
    events += [_event(number, token=unseen) for number in range(6, 9)]
    corpus_root = tmp_path / "cli" / "corpus"
    _write_jsonl(corpus_root / "holdout.jsonl", [_node(), *events])
    index_path = corpus_root / "dev-fingerprint-index.json"
    index_path.write_text(
        json.dumps(
            _index_document(
                ap,
                seen_token=seen,
            )
        ),
        encoding="utf-8",
    )
    _write_packet_manifest(corpus_root, split="holdout", index_path=index_path)

    exit_code = ap.main(
        [
            "compare",
            "--corpus-root",
            str(corpus_root),
            "--split",
            "holdout",
            "--metrics",
            "auto_recall",
            "--dev-fingerprint-index",
            str(index_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    result = json.loads(captured.out)
    unseen_result = result["metrics"]["auto_recall"]["repeat_gating"]["unseen_in_dev"]
    assert unseen_result["population"]["repeated_fingerprints"] == 1
    assert unseen_result["population"]["gated_events"] == 2
    rendered = captured.out + captured.err
    assert "fingerprint_token" not in rendered
    assert seen not in rendered
    assert unseen not in rendered


def test_opaque_token_values_do_not_change_payload_aggregate(
    ap: Any, tmp_path: Path
) -> None:
    first_seen, first_unseen = _token("seen-one"), _token("unseen-one")
    second_seen, second_unseen = _token("seen-two"), _token("unseen-two")

    def workload(seen: str, unseen: str) -> list[dict[str, Any]]:
        rows = [_event(number, event_class="organic", token=unseen) for number in range(1, 6)]
        rows += [_event(number, token=unseen) for number in range(6, 9)]
        rows += [_event(number, token=seen) for number in range(9, 12)]
        return rows

    first = _run_opaque(
        ap,
        tmp_path,
        workload(first_seen, first_unseen),
        seen_token=first_seen,
        name="token-values-one",
    )["unseen_in_dev"]
    second = _run_opaque(
        ap,
        tmp_path,
        workload(second_seen, second_unseen),
        seen_token=second_seen,
        name="token-values-two",
    )["unseen_in_dev"]
    assert first["population"]["repeated_fingerprint_events"] == 3
    assert first["population"]["gated_events"] == 2
    assert first["chars"]["repeated_automatic"]["ungated_total"] > 0
    assert first == second


def test_repeat_gating_aggregate_compares_explicit_on_with_explicit_off(
    ap: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _token("explicit-policy-seen")
    unseen = _token("explicit-policy-unseen")
    events = [
        _event(number, event_class="organic", token=unseen)
        for number in range(1, 6)
    ]
    events += [_event(number, token=unseen) for number in range(6, 9)]

    monkeypatch.setenv("LM_RECALL_REPEAT_GATING", "1")
    monkeypatch.setenv("LM_RECALL_REPEAT_DROP_TRAILING_STUBS", "1")
    enabled = _run_opaque(
        ap,
        tmp_path,
        events,
        seen_token=seen,
        name="explicit-policy-on",
    )

    monkeypatch.setenv("LM_RECALL_REPEAT_GATING", "0")
    monkeypatch.setenv("LM_RECALL_REPEAT_DROP_TRAILING_STUBS", "0")
    disabled = _run_opaque(
        ap,
        tmp_path,
        events,
        seen_token=seen,
        name="explicit-policy-off",
    )

    assert enabled["policy"]["enabled"] is True
    assert enabled["policy"]["drop_trailing_stubs"] is True
    assert disabled["policy"]["enabled"] is False
    assert disabled["policy"]["drop_trailing_stubs"] is False

    enabled_population = enabled["population"]
    disabled_population = disabled["population"]
    assert enabled_population["repeated_fingerprints"] == 1
    assert enabled_population["repeated_fingerprint_events"] == 3
    assert enabled_population["gated_events"] == 2
    assert disabled_population["repeated_fingerprints"] == 1
    assert disabled_population["repeated_fingerprint_events"] == 3
    assert disabled_population["gated_events"] == 0

    enabled_chars = enabled["chars"]["repeated_automatic"]
    disabled_chars = disabled["chars"]["repeated_automatic"]
    assert enabled_chars["ungated_total"] > 0
    assert enabled_chars["gated_total"] < enabled_chars["ungated_total"]
    assert disabled_chars["ungated_total"] == enabled_chars["ungated_total"]
    assert disabled_chars["gated_total"] == disabled_chars["ungated_total"]


def test_unseen_partition_depends_on_reference_membership_not_family_order(
    ap: Any, tmp_path: Path
) -> None:
    family_a = _token("membership-a")
    family_b = _token("membership-b")
    events = [_event(number, token=family_a) for number in range(1, 4)]
    events += [
        _event(number, event_class="organic", token=family_b)
        for number in range(4, 9)
    ]
    events += [_event(number, token=family_b) for number in range(9, 12)]

    unseen_b = _run_opaque(
        ap,
        tmp_path,
        events,
        seen_token=family_a,
        name="membership-a-seen",
    )["unseen_in_dev"]
    unseen_a = _run_opaque(
        ap,
        tmp_path,
        events,
        seen_token=family_b,
        name="membership-b-seen",
    )["unseen_in_dev"]

    assert unseen_b["population"]["gated_events"] == 2
    assert unseen_b["chars"]["repeated_automatic"]["reduction_ratio"] > 0.5
    assert unseen_a["population"]["gated_events"] == 0
    assert unseen_a["chars"]["repeated_automatic"]["reduction_ratio"] == 0.0


def test_delayed_feedback_is_applied_at_recorded_time(ap: Any, tmp_path: Path) -> None:
    seen = _token("different-seen")
    unseen = _token("delayed-feedback")
    events = [
        _event(
            number,
            token=unseen,
            feedback_at="2026-08-13T00:07:30Z" if number == 1 else None,
        )
        for number in range(1, 9)
    ]
    gate = _run_opaque(ap, tmp_path, events, seen_token=seen, name="delayed-feedback")

    # Delivery seven gates after the five-event warmup and deterministic probe;
    # the delayed link is applied before delivery eight and reopens the gate.
    assert gate["unseen_in_dev"]["population"]["gated_events"] == 1


def test_equal_timestamp_deliveries_precede_feedback(ap: Any, tmp_path: Path) -> None:
    seen = _token("tie-seen")
    unseen = _token("tie-feedback")
    events = [_event(number, token=unseen) for number in range(1, 9)]
    events[5]["feedback_applied"] = True
    events[5]["feedback_applied_at"] = events[5]["created_at"]
    events[6]["created_at"] = events[5]["created_at"]

    gate = _run_opaque(ap, tmp_path, events, seen_token=seen, name="tie-feedback")
    # Event seven still sees the pre-link state because all equal-timestamp
    # deliveries precede links; event eight sees the reset streak.
    assert gate["unseen_in_dev"]["population"]["gated_events"] == 1


def test_surviving_cross_session_stub_keeps_content_access(
    ap: Any, tmp_path: Path
) -> None:
    seen = _token("retention-seen")
    unseen = _token("retention-unseen")
    second_node = copy.deepcopy(_node())
    second_node["id"] = "01TESTNODE00000000000000001"
    second_node["content_surrogate"] = "new content bearer " + "y" * 2400
    events = [_event(number, token=unseen) for number in range(1, 8)]
    events[-1]["results"].append(
        {
            "node_id": second_node["id"],
            "rank": 2,
            "score": 0.9,
            "bm25_score": 0.9,
            "vector_score": 0.0,
            "graph_score": 0.0,
            "trigger_score": 0.0,
            "methods": ["bm25"],
            "path": [],
        }
    )

    corpus_root = tmp_path / "retention" / "corpus"
    _write_jsonl(corpus_root / "holdout.jsonl", [_node(), second_node, *events])
    index_path = corpus_root / "dev-fingerprint-index.json"
    index_path.write_text(
        json.dumps(
            _index_document(
                ap,
                seen_token=seen,
            )
        ),
        encoding="utf-8",
    )
    _write_packet_manifest(corpus_root, split="holdout", index_path=index_path)
    lm = ap._import_living_memory()
    corpus = ap.ReplayCorpus(corpus_root, "holdout", lm)
    aggregate = ap.family_repeat_gating(
        corpus,
        ap.replay_delivery(corpus),
        dev_fingerprint_index=index_path,
    )["unseen_in_dev"]

    retention = aggregate["retention"]["repeated_automatic_content_access"]
    assert aggregate["population"]["gated_events"] == 1
    assert retention["events"] == 7
    assert retention["gated_retained"] == 7
    assert retention["gated_share"] == 1.0


def test_dropped_trailing_twin_keeps_access_through_surviving_bearer(
    ap: Any, tmp_path: Path
) -> None:
    seen = _token("twin-seen")
    unseen = _token("twin-unseen")
    first_twin = copy.deepcopy(_node())
    first_twin["id"] = "01TESTNODE00000000000000001"
    first_twin["content_surrogate"] = "new identical content " + "z" * 2400
    second_twin = copy.deepcopy(first_twin)
    second_twin["id"] = "01TESTNODE00000000000000002"
    novel = copy.deepcopy(_node())
    novel["id"] = "01TESTNODE00000000000000003"
    novel["content_surrogate"] = "novel content between twins " + "q" * 2400
    events = [_event(number, token=unseen) for number in range(1, 8)]
    events[0]["results"][0]["node_id"] = first_twin["id"]
    events[-1]["results"] = [
        {
            "node_id": node["id"],
            "rank": rank,
            "score": 1.0 - rank / 10,
            "bm25_score": 1.0 - rank / 10,
            "vector_score": 0.0,
            "graph_score": 0.0,
            "trigger_score": 0.0,
            "methods": ["bm25"],
            "path": [],
        }
        for rank, node in enumerate((first_twin, novel, second_twin), start=1)
    ]

    corpus_root = tmp_path / "twin-retention" / "corpus"
    _write_jsonl(
        corpus_root / "holdout.jsonl",
        [_node(), first_twin, second_twin, novel, *events],
    )
    index_path = corpus_root / "dev-fingerprint-index.json"
    index_path.write_text(
        json.dumps(_index_document(ap, seen_token=seen)), encoding="utf-8"
    )
    _write_packet_manifest(corpus_root, split="holdout", index_path=index_path)
    lm = ap._import_living_memory()
    corpus = ap.ReplayCorpus(corpus_root, "holdout", lm)
    aggregate = ap.family_repeat_gating(
        corpus,
        ap.replay_delivery(corpus),
        dev_fingerprint_index=index_path,
    )["unseen_in_dev"]

    retention = aggregate["retention"]["repeated_automatic_content_access"]
    assert aggregate["population"]["gated_events"] == 1
    assert retention["events"] == 7
    assert retention["gated_retained"] == 7


def test_custom_native_replay_without_manifest_reports_unseen_gap(
    ap: Any, tmp_path: Path
) -> None:
    corpus_root = tmp_path / "legacy-native" / "corpus"
    events = [_event(number, query="legacy family") for number in range(1, 8)]
    _write_jsonl(corpus_root / "eval.jsonl", [_node(), *events])

    lm = ap._import_living_memory()
    corpus = ap.ReplayCorpus(corpus_root, "eval", lm)
    gate = ap.family_repeat_gating(corpus, ap.replay_delivery(corpus))

    assert gate["population"]["repeated_fingerprint_events"] == 7
    assert gate["unseen_in_dev"] == {
        "available": False,
        "reason": "complete_dev_reference_not_attested",
    }


def test_native_families_use_production_whitespace_normalization(
    ap: Any, tmp_path: Path
) -> None:
    corpus_root = tmp_path / "native" / "corpus"
    dev_reference = _event(1, query="different dev identity")
    dev_reference["results"] = []
    _write_jsonl(corpus_root / "dev.jsonl", [dev_reference])
    _write_packet_manifest(
        corpus_root, split="dev", automatic_dev_events=1
    )
    target = [
        _event(10, query="same normalized query"),
        _event(11, query="  same   normalized query "),
        _event(12, query="same\nnormalized\tquery"),
    ]
    _write_jsonl(corpus_root / "eval.jsonl", [_node(), *target])

    lm = ap._import_living_memory()
    corpus = ap.ReplayCorpus(corpus_root, "eval", lm)
    gate = ap.family_repeat_gating(corpus, ap.replay_delivery(corpus))
    unseen = gate["unseen_in_dev"]
    assert unseen["population"]["automatic_fingerprints_in_dev"] == 1
    assert unseen["population"]["repeated_fingerprints"] == 1
    assert unseen["population"]["repeated_fingerprint_events"] == 3


def test_native_dev_reference_requires_hash_and_population_attestation(
    ap: Any, tmp_path: Path
) -> None:
    corpus_root = tmp_path / "incomplete-native" / "corpus"
    dev_reference = _event(1, query="only surviving dev identity")
    dev_reference["results"] = []
    _write_jsonl(corpus_root / "dev.jsonl", [dev_reference])
    _write_packet_manifest(
        corpus_root, split="dev", automatic_dev_events=2
    )
    _write_jsonl(
        corpus_root / "eval.jsonl",
        [_node(), *[_event(number, query="target family") for number in range(3, 6)]],
    )
    lm = ap._import_living_memory()
    corpus = ap.ReplayCorpus(corpus_root, "eval", lm)
    with pytest.raises(ValueError, match="population is incomplete"):
        ap.family_repeat_gating(corpus, ap.replay_delivery(corpus))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document.__setitem__("population_events", 731),
        lambda document: document.__setitem__("unique_tokens", 352),
        lambda document: document.__setitem__("algorithm", "SHA256"),
        lambda document: document.__setitem__("normalization", "raw query"),
        lambda document: document.__setitem__("unexpected_private_field", "x"),
        lambda document: document["tokens"].__setitem__(0, "not-an-opaque-token"),
        lambda document: document["tokens"].append(document["tokens"][0]),
        lambda document: document["tokens"].reverse(),
    ],
)
def test_external_index_fails_closed_without_echoing_identities(
    ap: Any,
    tmp_path: Path,
    mutation: Any,
) -> None:
    secret = _token("must-never-echo")
    document = copy.deepcopy(_index_document(ap, seen_token=secret))
    mutation(document)
    path = tmp_path / f"bad-{hash(str(mutation))}.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        ap.load_dev_fingerprint_index(path)
    assert secret not in str(caught.value)


def test_external_packet_manifest_hash_binds_index(ap: Any, tmp_path: Path) -> None:
    seen = _token("manifest-seen")
    unseen = _token("manifest-unseen")
    events = [_event(number, token=unseen) for number in range(1, 4)]
    _run_opaque(
        ap,
        tmp_path,
        events,
        seen_token=seen,
        name="manifest-binding",
    )
    corpus_root = tmp_path / "manifest-binding" / "corpus"
    index_path = corpus_root / "dev-fingerprint-index.json"
    index_path.write_text(index_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    lm = ap._import_living_memory()
    corpus = ap.ReplayCorpus(corpus_root, "holdout", lm)
    with pytest.raises(ValueError, match="not hash-valid") as caught:
        ap.family_repeat_gating(
            corpus,
            ap.replay_delivery(corpus),
            dev_fingerprint_index=index_path,
        )
    assert seen not in str(caught.value)
    assert unseen not in str(caught.value)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest["sources"]["original_packet_manifest"].__setitem__(
            "sha256", "0" * 64
        ),
        lambda manifest: manifest["validation"]["keyed_preseal"].__setitem__(
            "semantic_reads", 1
        ),
        lambda manifest: manifest["validation"]["keyed_preseal"].__setitem__(
            "supplement_events",
            manifest["validation"]["keyed_preseal"]["supplement_events"] + 1,
        ),
    ],
)
def test_external_packet_requires_frozen_source_and_keyed_receipt(
    ap: Any, tmp_path: Path, mutation: Any
) -> None:
    seen = _token("receipt-seen")
    unseen = _token("receipt-unseen")
    events = [_event(number, token=unseen) for number in range(1, 4)]
    name = f"receipt-{abs(hash(str(mutation)))}"
    _run_opaque(ap, tmp_path, events, seen_token=seen, name=name)
    corpus_root = tmp_path / name / "corpus"
    manifest_path = corpus_root.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    lm = ap._import_living_memory()
    corpus = ap.ReplayCorpus(corpus_root, "holdout", lm)
    with pytest.raises(ValueError, match="valid sealed receipt") as caught:
        ap.family_repeat_gating(
            corpus,
            ap.replay_delivery(corpus),
            dev_fingerprint_index=corpus_root / "dev-fingerprint-index.json",
        )
    assert seen not in str(caught.value)
    assert unseen not in str(caught.value)


def test_external_packet_hash_binds_independent_verifier(ap: Any, tmp_path: Path) -> None:
    seen = _token("verifier-seen")
    unseen = _token("verifier-unseen")
    events = [_event(number, token=unseen) for number in range(1, 4)]
    _run_opaque(ap, tmp_path, events, seen_token=seen, name="verifier-binding")
    corpus_root = tmp_path / "verifier-binding" / "corpus"
    verifier_path = corpus_root.parent / "recipe" / "verify.py"
    verifier_path.write_text("# changed after sealing\n", encoding="utf-8")

    lm = ap._import_living_memory()
    corpus = ap.ReplayCorpus(corpus_root, "holdout", lm)
    with pytest.raises(ValueError, match="valid sealed receipt") as caught:
        ap.family_repeat_gating(
            corpus,
            ap.replay_delivery(corpus),
            dev_fingerprint_index=corpus_root / "dev-fingerprint-index.json",
        )
    assert seen not in str(caught.value)
    assert unseen not in str(caught.value)


def test_opaque_target_requires_all_event_identities(ap: Any, tmp_path: Path) -> None:
    seen = _token("coverage-seen")
    exposed = _token("coverage-present")
    events = [_event(1, token=exposed), _event(2, token=None), _event(3, token=exposed)]
    corpus_root = tmp_path / "coverage" / "corpus"
    _write_jsonl(corpus_root / "holdout.jsonl", [_node(), *events])
    index_path = corpus_root / "dev-fingerprint-index.json"
    index_path.write_text(
        json.dumps(
            _index_document(
                ap,
                seen_token=seen,
            )
        ),
        encoding="utf-8",
    )
    _write_packet_manifest(corpus_root, split="holdout", index_path=index_path)

    lm = ap._import_living_memory()
    corpus = ap.ReplayCorpus(corpus_root, "holdout", lm)
    delivery = ap.replay_delivery(corpus)
    with pytest.raises(ValueError) as no_index:
        ap.family_repeat_gating(corpus, delivery)
    assert exposed not in str(no_index.value)

    with pytest.raises(ValueError) as caught:
        ap.family_repeat_gating(
            corpus,
            delivery,
            dev_fingerprint_index=index_path,
        )
    message = str(caught.value)
    assert exposed not in message
    assert all(event["id"] not in message for event in events)
