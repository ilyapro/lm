"""Byte-lock and semantic checks for the frozen confirmatory-holdout-v4 protocol.

V4 is a self-contained successor to the retired v3 namespace, not an overlay.
These tests prove three things that a reviewer cannot get from the diff alone:

1. the v4 documents are byte-locked, canonically encoded, duplicate-key free,
   and source-blind, and the namespace is closed around them plus the accrual
   artifacts the frozen protocol itself declares -- nothing else, at no depth;
2. v4 differs from the frozen v3 plan in exactly one enumerated set of paths --
   fresh identities, the retired-v3 lineage pins, and the replacement bootstrap
   transition -- so the calendar, grace, horizon, floors, gates, runtime
   boundary, privacy rules, sealing sequence, and reader authorities are
   provably carried across unchanged;
3. the replacement `initial-runtime-change-closure` schema carries no
   prior-source-binding member at all, is mutually exclusive with the ordinary
   two-core closure, and cannot be replayed across protocol namespaces.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts" / "animal-planet" / "evaluation"
NAMESPACE = EVAL / "confirmatory-holdout-v4"
PLAN_PATH = NAMESPACE / "analysis-plan.json"
V3_NAMESPACE = EVAL / "confirmatory-holdout-v3"
V3_PLAN_PATH = V3_NAMESPACE / "analysis-plan.json"
V2_NAMESPACE = EVAL / "confirmatory-holdout-v2"
WATERMARK_PATH = EVAL / "release-v1" / "control-watermark.json"

DOCUMENT_HASHES = {
    "README.md": (
        "abaf8604d3d1b0d83b90ad2c1dbe62a54cec94765facde08814f75bf8e64c3f1",
        9545,
    ),
    "POLICY.md": (
        "8a229fa1edb71835ff3fb3d34ab38719ad5af68ef059f0717e7e164726f623c6",
        42277,
    ),
    "analysis-plan.json": (
        "bb049d1b5a237b345a5f94957e5cd82f37bd4940b14011e68e30dfb7aa7c8d70",
        177982,
    ),
}

# The namespace is closed, not frozen-empty. The three documents above are the
# only *protocol* files, but the protocol they carry declares artifacts of its
# own that come into existence during accrual and sealing. Each name below is
# admitted on the authority of the byte-locked bytes cited beside it; a name
# with no such authority is not admitted. Membership is mandatory, presence is
# not -- these appear progressively, so the guard checks the closure, never the
# count.
ALLOWED_ACCRUAL_DIRECTORIES = {
    # README.md, line 21 of the bytes locked above: "observed accrual belongs
    # only in append-only `segments/` and `probes/` artifacts defined by this
    # protocol".
    "segments",
    "probes",
    # analysis-plan.json `hash_binding/packet_manifest_must_bind` requires the
    # sealed packet to bind a "packet builder and independent verifier"; that
    # tooling lives beside the packet it builds, mirroring the original packet's
    # `artifacts/animal-planet/recipe/`.
    "recipe",
    # ... and to bind the "complete selected holdout partition" and "complete
    # selected shadow partition" plus the "partition/source seed states and
    # construction manifest" that this directory holds.
    "corpus",
}
ALLOWED_ACCRUAL_FILES = {
    # analysis-plan.json `publication_and_ordering` fixes the sealing order --
    # `content_before_manifest`, `manifest_last`, and a sealed state whose
    # `canonical_manifest_present` and `manifest_hash_valid` are both true.
    "manifest.json",
    # ... and grants exactly one `packet_publication_attempts` under
    # `no_overwrite`, which the seal launcher records here.
    "seal-receipt.json",
}

V3_FROZEN_FILES = {
    "README.md": (
        V3_NAMESPACE / "README.md",
        "51988680542c2a8361df3a126420abb4ba32d51b52ee588889bdfe2dbe3987c3",
        6371,
    ),
    "POLICY.md": (
        V3_NAMESPACE / "POLICY.md",
        "e5976cd3d6239dbf35cbd84495308c5a61b063e3d4c08fb0a6fc6fda2ff8a551",
        36475,
    ),
    "analysis-plan.json": (
        V3_PLAN_PATH,
        "7c8af7f53bfa6edbd8146e5593e585327aeedf2408e066d2e47c9856056570a9",
        158604,
    ),
    "protocol-test": (
        ROOT / "tests" / "test_confirmatory_evidence_protocol_v3.py",
        "805fb4863c4595e2508949c2638d363a1d9e81df3b7fe94365bc2047ba448c64",
        66316,
    ),
}

# The complete, enumerated v3 -> v4 identity freshening. Applying it to the
# frozen v3 plan must leave exactly EXPECTED_DELTA_PATHS differing.
IDENTITY_FRESHENING = (
    ("confirmatory-holdout-v3", "confirmatory-holdout-v4"),
    ("confirmatory-local-v3-ro", "confirmatory-local-v4-ro"),
    ("confirmatory-alt-v3-ro", "confirmatory-alt-v4-ro"),
    ("confirmatory-shadow-v3-eval", "confirmatory-shadow-v4-eval"),
    ("v3_canonical_json_utf8", "v4_canonical_json_utf8"),
    ("v3_semantic_reads", "v4_semantic_reads"),
    ("v3_lower_bound_exclusive", "v4_lower_bound_exclusive"),
    ("v3-protocol-frozen", "v4-protocol-frozen"),
    ("v3_evidence_protocol_supersessions", "v4_evidence_protocol_supersessions"),
    ("sealed v3 packet manifest", "sealed v4 packet manifest"),
    ("not a v3 source alias", "not a v4 source alias"),
    ("ephemeral v3 HMAC token", "ephemeral v4 HMAC token"),
    ("the v3 compact encoder", "the v4 compact encoder"),
    ("valid v3 ", "valid v4 "),
    ("exact v3 ", "exact v4 "),
    ("fresh v3 ", "fresh v4 "),
    ("this hash-bound v3 ", "this hash-bound v4 "),
)

EXPECTED_DELTA_PATHS = {
    # replacement bootstrap transition
    "accrual_and_stopping/bootstrap_resolution_rules [added]",
    "accrual_and_stopping/initial_state_implementation_alias [added]",
    "accrual_and_stopping/ledger/initial_runtime_change_closure_appends_counts_or_snapshots [added]",
    "accrual_and_stopping/state_machine/1/event [changed]",
    "operational_receipt_schemas/initial_runtime_change_closure_resolution [added]",
    "operational_receipt_schemas/ledger_entry/artifact_kind_rule/initial-runtime-change-closure [added]",
    "operational_receipt_schemas/ledger_entry/bootstrap_and_successor_order [changed]",
    "operational_receipt_schemas/ledger_entry/entry_kind_enum [len 14->15]",
    "operational_receipt_schemas/ledger_entry/slot_index_or_null [changed]",
    "operational_receipt_schemas/slot_segment_closure_resolution/pre_binding_use_action [added]",
    "operational_receipt_schemas/slot_segment_closure_resolution/prior_source_binding_core_required [added]",
    "operational_receipt_schemas/slot_segment_closure_resolution/valid_before_initial_source_binding_attestation [added]",
    "runtime_segments/initial_bootstrap_branch_selector [added]",
    "runtime_segments/initial_bootstrap_branches_exactly [added]",
    "runtime_segments/initial_bootstrap_branches_mutually_exclusive [added]",
    "runtime_segments/initial_runtime_change_closure_allowlist [added]",
    "runtime_segments/initial_runtime_change_closure_mismatch_kinds_exactly [added]",
    "runtime_segments/initial_runtime_change_closure_receipt_kind [added]",
    "runtime_segments/initial_runtime_change_closure_status [added]",
    "runtime_segments/prior_source_binding_core_for_unbound_initial_segment [added]",
    "runtime_segments/slot_change_action [changed]",
    "runtime_segments/closure_mismatch_kinds_rule [changed]",
    "sources/initial_source_binding_attempt [changed]",
    # fresh namespace header and namespace-derived values
    "schema_version [changed]",
    "operational_receipt_schemas/common_validation/constants/schema_version [changed]",
    "protocol_parent_commit [changed]",
    "protocol_parent_tree [changed]",
    "domains/initial_segment_id_value [changed]",
    "runtime_segments/initial_segment/segment_id [changed]",
    "partition/golden_vectors/0/bucket [changed]",
    "partition/golden_vectors/0/digest_sha256 [changed]",
    "partition/golden_vectors/0/partition [changed]",
    "partition/golden_vectors/1/bucket [changed]",
    "partition/golden_vectors/1/digest_sha256 [changed]",
    "partition/golden_vectors/2/bucket [changed]",
    "partition/golden_vectors/2/digest_sha256 [changed]",
    "uncertainty/seeds/automatic_component/value [changed]",
    "uncertainty/seeds/organic_component/value [changed]",
    "uncertainty/seeds/shadow_component/value [changed]",
    # retired-v3 lineage pins and retirement bookkeeping
    "authority/retired_v3_seal_id [added]",
    "authority/retired_v3_source_alias_ids [added]",
    "authority/retired_void_v3_readers [added]",
    "cross_protocol_replay_safety [added]",
    "domains/retired_domain_prefix [removed]",
    "domains/retired_domain_prefixes [added]",
    "hash_binding/packet_manifest_must_bind [len 25->26]",
    "lineage/retired_v3 [added]",
    "lineage/unchanged_repair_design/preserved_from_v3_without_change [added]",
    "lineage/unchanged_repair_design/v3_can_satisfy_precondition [added]",
    "lineage/unchanged_repair_design/v4_evidence_protocol_supersessions [len 8->11]",
    "source_blind_preregistration/allowed_inputs [len 12->16]",
    "source_blind_preregistration/reads_before_freeze/retired_v3_probe_or_seal_attempts [added]",
    "source_blind_preregistration/reads_before_freeze/source_availability_probe_attempts [added]",
    "uncertainty/retired_v3_seed_values_reusable [added]",
}

RETIRED_IDENTITIES = {
    "confirmatory-holdout-v2",
    "confirmatory-holdout-v3",
    "confirmatory-local-v2-ro",
    "confirmatory-local-v3-ro",
    "confirmatory-alt-v2-ro",
    "confirmatory-alt-v3-ro",
    "confirmatory-shadow-v2-eval",
    "confirmatory-shadow-v3-eval",
    "confirmatory-holdout-v2-eval",
    "confirmatory-holdout-v3-eval",
    "confirmatory-holdout-v3-seal",
    "shadow-eval",
    "replacement-holdout-eval",
}

PRIOR_BINDING_MEMBER = "prior_source_binding_core_sha256_and_bytes"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_namespace_is_closed(root: Path) -> None:
    """Every entry is a protocol document or a declared accrual artifact.

    Three separable claims, each falsifiable on its own:

    * closure -- all three documents are present, and no name outside them and
      the annotated allowlist exists;
    * no symlink at any depth. `Path.rglob` reports a symlink as itself and
      never descends into one, so a link can neither escape this sweep, hide an
      entry behind it, nor build a loop out of it;
    * kind -- an allowlisted directory name is a directory and an allowlisted
      file name is a regular file, so an admitted name cannot be smuggled in
      as something else.

    The symlink sweep runs before the kind checks so that a link is always
    diagnosed as a link: `is_dir`/`is_file` resolve through one, and a dangling
    link would otherwise be reported as the wrong kind.
    """
    names = {child.name for child in root.iterdir()}

    missing = sorted(set(DOCUMENT_HASHES) - names)
    assert not missing, f"missing protocol document: {missing}"
    undeclared = sorted(
        names - set(DOCUMENT_HASHES) - ALLOWED_ACCRUAL_DIRECTORIES - ALLOWED_ACCRUAL_FILES
    )
    assert not undeclared, f"undeclared namespace entry: {undeclared}"

    symlinks = sorted(
        str(child.relative_to(root)) for child in root.rglob("*") if child.is_symlink()
    )
    assert not symlinks, f"symlink in namespace: {symlinks}"

    for name in sorted(names & ALLOWED_ACCRUAL_DIRECTORIES):
        assert (root / name).is_dir(), f"declared directory is not a directory: {name}"
    for name in sorted(names & ALLOWED_ACCRUAL_FILES):
        assert (root / name).is_file(), f"declared file is not a regular file: {name}"


def _namespace_replica(root: Path) -> Path:
    """A minimal admissible namespace: the declared documents and nothing else.

    Only names and kinds reach the closure guard, so the replica carries no real
    bytes -- the byte locks are asserted against the real namespace, and running
    the closure cases off-tree keeps the frozen directory untouched.
    """
    root.mkdir(parents=True)
    for name in DOCUMENT_HASHES:
        (root / name).write_text("placeholder\n", encoding="utf-8")
    return root


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
    )


def _plan() -> dict[str, Any]:
    return _load_json(PLAN_PATH)


def _v3_plan() -> dict[str, Any]:
    return _load_json(V3_PLAN_PATH)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _utc(value: str) -> datetime:
    assert value.endswith("Z"), value
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")


def _freshen(value: Any) -> Any:
    if isinstance(value, str):
        for old, new in IDENTITY_FRESHENING:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_freshen(item) for item in value]
    if isinstance(value, dict):
        return {_freshen(key): _freshen(member) for key, member in value.items()}
    return value


def _diff(left: Any, right: Any, path: tuple[str, ...] = ()) -> Iterator[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left:
            if key not in right:
                yield "/".join(path + (key,)) + " [removed]"
            else:
                yield from _diff(left[key], right[key], path + (key,))
        for key in right:
            if key not in left:
                yield "/".join(path + (key,)) + " [added]"
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            yield "/".join(path) + f" [len {len(left)}->{len(right)}]"
            return
        for index, (a, b) in enumerate(zip(left, right)):
            yield from _diff(a, b, path + (str(index),))
        return
    if left != right:
        yield "/".join(path) + " [changed]"


def _strings(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, str]]:
    """Every object key and every string leaf, with its slash-joined path."""
    if isinstance(value, dict):
        for key, member in value.items():
            yield "/".join(path + (key,)), key
            yield from _strings(member, path + (key,))
    elif isinstance(value, list):
        for index, member in enumerate(value):
            yield from _strings(member, path + (str(index),))
    elif isinstance(value, str):
        yield "/".join(path), value


def _object_keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, member in value.items():
            yield key
            yield from _object_keys(member)
    elif isinstance(value, list):
        for member in value:
            yield from _object_keys(member)


def _string_leaves(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, member in value.items():
            yield from _string_leaves(member, path + (key,))
    elif isinstance(value, list):
        for index, member in enumerate(value):
            yield from _string_leaves(member, path + (str(index),))
    elif isinstance(value, str):
        yield "/".join(path), value


def test_v4_documents_are_nonempty_duplicate_free_and_byte_locked() -> None:
    assert set(DOCUMENT_HASHES) == {"README.md", "POLICY.md", "analysis-plan.json"}
    _assert_namespace_is_closed(NAMESPACE)
    for name, (expected_hash, expected_bytes) in DOCUMENT_HASHES.items():
        path = NAMESPACE / name
        assert path.is_file()
        assert not path.is_symlink()
        assert path.stat().st_size == expected_bytes, f"{name} size changed"
        assert _sha256(path) == expected_hash, f"{name} bytes changed"

    raw = PLAN_PATH.read_text(encoding="utf-8")
    plan = _plan()
    # Canonical, reproducible encoding: no hidden duplicate keys or stray bytes.
    assert json.dumps(plan, ensure_ascii=False, allow_nan=False, indent=2) + "\n" == raw

    assert plan["schema_version"] == 4
    assert plan["namespace"] == "confirmatory-holdout-v4"
    assert plan["protocol_id"] == "confirmatory-holdout-v4-protocol-v1"
    assert plan["status"] == "preregistered-source-unprobed-accruable-unsealed"
    assert plan["frozen"] is True
    assert plan["amendment_policy"].startswith("never in place;")
    assert re.fullmatch(r"[0-9a-f]{40}", plan["protocol_parent_commit"])
    assert re.fullmatch(r"[0-9a-f]{40}", plan["protocol_parent_tree"])


def test_namespace_closure_admits_exactly_the_declared_accrual_artifacts(
    tmp_path: Path,
) -> None:
    """The allowlist is a closed enumeration, and every member is optional."""
    assert ALLOWED_ACCRUAL_DIRECTORIES | ALLOWED_ACCRUAL_FILES == {
        "segments",
        "probes",
        "recipe",
        "corpus",
        "manifest.json",
        "seal-receipt.json",
    }
    # an accrual artifact never shadows a protocol document, so admitting one
    # can never be a way to replace locked bytes
    assert not (ALLOWED_ACCRUAL_DIRECTORIES | ALLOWED_ACCRUAL_FILES) & set(DOCUMENT_HASHES)
    assert not ALLOWED_ACCRUAL_DIRECTORIES & ALLOWED_ACCRUAL_FILES

    # presence is optional: the bare three documents are admissible ...
    _assert_namespace_is_closed(_namespace_replica(tmp_path / "bare"))

    # ... so is any single accrual artifact appearing on its own ...
    for name in sorted(ALLOWED_ACCRUAL_DIRECTORIES):
        case = _namespace_replica(tmp_path / f"only-{name}")
        (case / name).mkdir()
        _assert_namespace_is_closed(case)
    for name in sorted(ALLOWED_ACCRUAL_FILES):
        case = _namespace_replica(tmp_path / f"only-{name}")
        (case / name).write_text("{}\n", encoding="utf-8")
        _assert_namespace_is_closed(case)

    # ... and so is the fully accrued namespace, with real content at depth.
    # This is the positive control: the rejections below are not vacuous.
    full = _namespace_replica(tmp_path / "full")
    for name in sorted(ALLOWED_ACCRUAL_DIRECTORIES):
        (full / name).mkdir()
        (full / name / "nested").mkdir()
        (full / name / "nested" / "entry.json").write_text("{}\n", encoding="utf-8")
    for name in sorted(ALLOWED_ACCRUAL_FILES):
        (full / name).write_text("{}\n", encoding="utf-8")
    _assert_namespace_is_closed(full)


def test_namespace_closure_rejects_undeclared_entries(tmp_path: Path) -> None:
    """The closure still bites: an entry the protocol never declared fails."""
    for index, undeclared in enumerate(
        ("POLICY-v2.md", "analysis-plan.json.bak", "README.md.orig", "readiness.json")
    ):
        case = _namespace_replica(tmp_path / f"file-{index}")
        (case / undeclared).write_text("x\n", encoding="utf-8")
        with pytest.raises(AssertionError, match="undeclared namespace entry"):
            _assert_namespace_is_closed(case)

    # a stray directory, including one whose name merely resembles a declared
    # artifact, is rejected just as hard
    for index, undeclared in enumerate(("scratch", "segments-v2", "corpus.old", ".hidden")):
        case = _namespace_replica(tmp_path / f"dir-{index}")
        (case / undeclared).mkdir()
        with pytest.raises(AssertionError, match="undeclared namespace entry"):
            _assert_namespace_is_closed(case)

    # and the three documents are still mandatory, not merely allowlisted
    for name in sorted(DOCUMENT_HASHES):
        case = _namespace_replica(tmp_path / f"missing-{name}")
        (case / name).unlink()
        with pytest.raises(AssertionError, match="missing protocol document"):
            _assert_namespace_is_closed(case)


def test_namespace_closure_rejects_wrong_kinds_and_symlinks_at_any_depth(
    tmp_path: Path,
) -> None:
    """Beyond the original guard: an admitted name cannot change shape."""
    for name in sorted(ALLOWED_ACCRUAL_DIRECTORIES):
        case = _namespace_replica(tmp_path / f"asfile-{name}")
        (case / name).write_text("x\n", encoding="utf-8")
        with pytest.raises(AssertionError, match="declared directory is not a directory"):
            _assert_namespace_is_closed(case)
    for name in sorted(ALLOWED_ACCRUAL_FILES):
        case = _namespace_replica(tmp_path / f"asdir-{name}")
        (case / name).mkdir()
        with pytest.raises(AssertionError, match="declared file is not a regular file"):
            _assert_namespace_is_closed(case)

    # The hole the original guard left open: it checked `is_symlink` on the
    # three documents only, so a link anywhere else -- at any depth, pointing
    # anywhere, including at a private source outside the namespace -- passed.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.jsonl").write_text("x\n", encoding="utf-8")

    top = _namespace_replica(tmp_path / "symlink-top")
    (top / "manifest.json").symlink_to(outside / "private.jsonl")
    with pytest.raises(AssertionError, match="symlink in namespace"):
        _assert_namespace_is_closed(top)

    deep = _namespace_replica(tmp_path / "symlink-deep")
    (deep / "segments").mkdir()
    (deep / "segments" / "nested").mkdir()
    (deep / "segments" / "nested" / "entry.json").symlink_to(outside / "private.jsonl")
    with pytest.raises(AssertionError, match="symlink in namespace"):
        _assert_namespace_is_closed(deep)

    # a whole allowlisted directory swapped for a link to somewhere else ...
    swapped = _namespace_replica(tmp_path / "symlink-dir")
    (swapped / "probes").symlink_to(outside)
    with pytest.raises(AssertionError, match="symlink in namespace"):
        _assert_namespace_is_closed(swapped)

    # ... a dangling link that resolves to nothing ...
    dangling = _namespace_replica(tmp_path / "symlink-dangling")
    (dangling / "seal-receipt.json").symlink_to(tmp_path / "does-not-exist")
    with pytest.raises(AssertionError, match="symlink in namespace"):
        _assert_namespace_is_closed(dangling)

    # ... and a protocol document itself relinked. The per-document byte lock
    # already rejects this; the sweep is the same prohibition stated once for
    # the whole namespace, so no future entry can be added outside it.
    relinked = _namespace_replica(tmp_path / "symlink-document")
    (relinked / "README.md").unlink()
    (relinked / "README.md").symlink_to(outside / "private.jsonl")
    with pytest.raises(AssertionError, match="symlink in namespace"):
        _assert_namespace_is_closed(relinked)


def test_v4_is_source_blind_and_reads_only_hash_pinned_public_inputs() -> None:
    prereg = _plan()["source_blind_preregistration"]
    assert prereg["availability_values_used"] is False
    assert prereg["calendar_derived_from_availability"] is False
    assert prereg["power_floors_derived_from_current_availability"] is False
    assert prereg["repair_implementation_started"] is False
    assert prereg["semantic_evaluator_started"] is False
    assert prereg["reads_before_freeze"] == {
        "candidate_private_rows": 0,
        "candidate_source_aggregate_queries": 0,
        "candidate_source_snapshot_captures": 0,
        "original_sealed_corpus_rows": 0,
        "replacement_sealed_corpus_rows": 0,
        "consumed_packet_verifier_invocations": 0,
        "v4_semantic_reads": 0,
        "source_availability_probe_attempts": 0,
        "retired_v3_probe_or_seal_attempts": 0,
    }

    for entry in prereg["allowed_inputs"]:
        path = ROOT / entry["path"]
        assert path.is_file(), entry["path"]
        assert not path.is_symlink()
        assert path.stat().st_size == entry["bytes"], entry["path"]
        assert _sha256(path) == entry["sha256"], entry["path"]

    declared = {entry["path"] for entry in prereg["allowed_inputs"]}
    # The v3 predecessor is an input only as public frozen bytes.
    assert {
        "artifacts/animal-planet/evaluation/confirmatory-holdout-v3/README.md",
        "artifacts/animal-planet/evaluation/confirmatory-holdout-v3/POLICY.md",
        "artifacts/animal-planet/evaluation/confirmatory-holdout-v3/analysis-plan.json",
        "tests/test_confirmatory_evidence_protocol_v3.py",
    } <= declared
    # No private source, sealed corpus, or consumed verifier is an admitted input.
    for path in declared:
        assert "corpus/holdout" not in path
        assert "corpus/eval" not in path
        assert not path.endswith("recipe/verify.py")
        assert not path.endswith(".sqlite3")


def test_v3_is_pinned_immutable_unamended_and_non_authoritative() -> None:
    retired = _plan()["lineage"]["retired_v3"]
    assert retired["namespace"] == "confirmatory-holdout-v3"
    assert retired["state"] == "retired-unreachable-initial-bootstrap"
    assert retired["immutable"] is True
    assert retired["amended_in_place"] is False
    assert retired["authoritative"] is False
    assert retired["authority_grants"] == []
    assert retired["all_authorities_unusable"] is True
    for flag in (
        "source_binding_attestation",
        "readiness_receipt",
        "packet",
        "seal",
        "reader",
        "repair_implementation",
        "mentioning_a_retired_identifier_revives_it",
    ):
        assert retired[flag] is False, flag
    assert retired["probe_attempts"] == 0
    assert retired["source_reads"] == 0
    assert PRIOR_BINDING_MEMBER in retired["defect"]

    pinned = {entry["path"]: entry for entry in retired["documents"]}
    pinned[retired["protocol_test"]["path"]] = retired["protocol_test"]
    assert len(pinned) == 4
    for _, path_hash_size in V3_FROZEN_FILES.items():
        path, expected_hash, expected_bytes = path_hash_size
        relative = str(path.relative_to(ROOT))
        assert path.is_file()
        assert not path.is_symlink()
        # the v3 bytes on disk are unchanged ...
        assert _sha256(path) == expected_hash, relative
        assert path.stat().st_size == expected_bytes, relative
        # ... and v4 pins exactly those bytes.
        assert pinned[relative]["sha256"] == expected_hash
        assert pinned[relative]["bytes"] == expected_bytes

    assert set(retired["retired_identifiers"]) == {
        "confirmatory-local-v3-ro",
        "confirmatory-alt-v3-ro",
        "confirmatory-shadow-v3-eval",
        "confirmatory-holdout-v3-eval",
        "confirmatory-holdout-v3-seal",
    }
    assert retired["retired_domain_prefix"] == "confirmatory-holdout-v3/"

    design = _plan()["lineage"]["unchanged_repair_design"]
    assert design["sha256"] == (
        "76d34d33b7be57230ef0988662060fe70eea229f9b07f2cd5b8a2354600a79f5"
    )
    assert design["v2_can_satisfy_precondition"] is False
    assert design["v3_can_satisfy_precondition"] is False


def test_v4_differs_from_frozen_v3_in_exactly_the_enumerated_paths() -> None:
    """The strongest preservation proof: nothing drifted that we did not name."""
    expected_base = _freshen(_v3_plan())
    observed = sorted(set(_diff(expected_base, _plan())))
    unexpected = sorted(set(observed) - EXPECTED_DELTA_PATHS)
    missing = sorted(EXPECTED_DELTA_PATHS - set(observed))
    assert not unexpected, f"undeclared v3->v4 drift: {unexpected}"
    assert not missing, f"declared change absent: {missing}"


def test_calendar_grace_horizon_floors_and_gates_are_preserved_exactly() -> None:
    plan = _plan()
    v3 = _v3_plan()

    schedule = plan["schedule"]
    assert schedule == v3["schedule"]
    assert schedule["anchor_at"] == "2026-08-17T00:00:00Z"
    assert schedule["cadence_seconds"] == 86400
    assert schedule["first_slot_index"] == 0
    assert schedule["last_slot_index"] == 28
    assert schedule["slot_count"] == 29
    assert schedule["grace_seconds"] == 21600
    assert schedule["first_slot_at"] == "2026-08-17T00:00:00Z"
    assert schedule["final_selection_slot_at"] == "2026-09-14T00:00:00Z"
    assert schedule["absolute_horizon_expires_at"] == "2026-09-14T06:00:00Z"
    assert schedule["horizon_resets_on_segment_change"] is False
    anchor = _utc(schedule["anchor_at"])
    assert anchor + timedelta(seconds=28 * 86400) == _utc(
        schedule["final_selection_slot_at"]
    )
    assert _utc(schedule["final_selection_slot_at"]) + timedelta(seconds=21600) == _utc(
        schedule["absolute_horizon_expires_at"]
    )
    for flag in (
        "backfill_allowed",
        "catch_up_allowed",
        "slot_replacement_allowed",
        "slot_skipping_allowed",
        "orchestration_retry_creates_new_slot",
    ):
        assert schedule[flag] is False, flag

    # numeric floors, point gates, and uncertainty parameters are untouched;
    # sections carrying identity strings are compared through the freshening map.
    assert plan["a_priori_power_floor"] == v3["a_priori_power_floor"]
    assert plan["promotion"] == v3["promotion"]
    assert plan["privacy"] == v3["privacy"]
    assert plan["readiness"] == _freshen(v3["readiness"])
    assert plan["measurement"] == _freshen(v3["measurement"])
    assert plan["selection"] == _freshen(v3["selection"])
    assert plan["replayability"] == _freshen(v3["replayability"])
    assert plan["identity"] == _freshen(v3["identity"])

    uncertainty = plan["uncertainty"]
    assert uncertainty["resamples"] == 10000
    assert uncertainty["confidence_level"] == 0.95
    assert uncertainty["interval"] == v3["uncertainty"]["interval"]
    assert uncertainty["retired_v2_seed_values_reusable"] is False
    assert uncertainty["retired_v3_seed_values_reusable"] is False


def test_runtime_boundary_and_default_off_replay_control_are_preserved() -> None:
    plan = _plan()
    v3 = _v3_plan()
    watermark = _load_json(WATERMARK_PATH)

    control = plan["release_control_watermark"]
    assert control == _freshen(v3["release_control_watermark"])
    assert control["sha256"] == _sha256(WATERMARK_PATH)
    assert control["bytes"] == WATERMARK_PATH.stat().st_size
    assert control["release_effective_at"] == "2026-08-14T15:03:46.793603Z"
    assert control["release_effective_at"] == (
        watermark["selection"]["lower_bound_exclusive_at"]
    )
    assert control["release_effective_at_semantics"] == (
        "observed runtime-provenance boundary; not a deployment claim"
    )
    replay = control["replay_code_control"]
    assert replay["commit"] == "46a9951842512333b0896370056d07a9e1c25bdf"
    assert replay["tree"] == "c1606671b9d13bed78c21b7d2f9a4bb75a3d1c1c"
    assert replay["deployment_required"] is False
    assert replay["measurement_arm"] == "gating-off"

    initial = plan["runtime_segments"]["initial_segment"]
    assert initial["segment_index"] == 0
    assert initial["lower_bound_exclusive_at"] == control["release_effective_at"]
    assert initial["effective_legacy_repeat_controls"] == {
        "LM_RECALL_REPEAT_GATING": False,
        "LM_RECALL_REPEAT_DROP_TRAILING_STUBS": False,
    }
    assert initial["all_event_producing_services_attested"] is True

    provenance = watermark["runtime_provenance"]
    observation = provenance["observation"]
    stable = control["stable_observation"]
    assert stable["pre_watermark_services_state_sha256"] == (
        observation["pre_runtime_state_sha256"]
    )
    assert stable["post_watermark_services_state_sha256"] == (
        observation["post_runtime_state_sha256"]
    )
    # the pinned prior tuple digest is recomputable from the watermark itself
    recomputed = hashlib.sha256(
        (
            json.dumps(
                sorted(
                    provenance["services"],
                    key=lambda service: (
                        service["service_identity_sha256"],
                        service["boot_identity_sha256"],
                    ),
                ),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    assert initial["watermark_services_state_sha256"] == recomputed
    assert initial["active_services_state_sha256"] == recomputed
    assert observation["pre_runtime_state_sha256"] == recomputed


def test_every_v4_domain_alias_seal_and_reader_identity_is_fresh() -> None:
    plan = _plan()
    domains = plan["domains"]

    fresh_domains = [
        value
        for key, value in domains.items()
        if key.endswith("_utf8") and value.startswith("confirmatory-holdout-")
    ]
    assert len(fresh_domains) == 12
    for domain in fresh_domains:
        assert domain.startswith("confirmatory-holdout-v4/"), domain
    assert len(set(fresh_domains)) == len(fresh_domains)
    assert domains["all_active_domains_fresh"] is True
    assert domains["retired_domains_reusable"] is False
    assert domains["retired_domain_prefixes"] == [
        "confirmatory-holdout-v2/",
        "confirmatory-holdout-v3/",
    ]

    aliases = plan["sources"]["aliases"]
    assert aliases["local"]["alias_id"] == "confirmatory-local-v4-ro"
    assert aliases["alt"]["alias_id"] == "confirmatory-alt-v4-ro"
    assert plan["sources"]["source_binding_core"]["alias_ids_by_alias"] == {
        "local": "confirmatory-local-v4-ro",
        "alt": "confirmatory-alt-v4-ro",
    }

    authority = plan["authority"]
    assert authority["seal"]["id"] == "confirmatory-holdout-v4-seal"
    assert authority["seal"]["max_process_launches"] == 1
    assert authority["seal"]["max_publication_attempts"] == 1
    assert [reader["id"] for reader in authority["reserved_readers_exactly"]] == [
        "confirmatory-shadow-v4-eval",
        "confirmatory-holdout-v4-eval",
    ]
    for reader in authority["reserved_readers_exactly"]:
        assert reader["max_process_launches"] == 1
        assert reader["max_semantic_passes"] == 1
    for flag in (
        "reader_aliases_allowed",
        "reader_delegation_allowed",
        "reader_wildcards_allowed",
        "fallback_readers_allowed",
    ):
        assert authority[flag] is False, flag
    assert authority["retired_void_v3_readers"] == [
        "confirmatory-shadow-v3-eval",
        "confirmatory-holdout-v3-eval",
    ]
    assert authority["retired_v3_source_alias_ids"] == [
        "confirmatory-local-v3-ro",
        "confirmatory-alt-v3-ro",
    ]
    assert authority["retired_v3_seal_id"] == "confirmatory-holdout-v3-seal"


def test_no_retired_identity_is_reused_as_an_active_v4_authority() -> None:
    plan = _plan()
    retired_only_paths = (
        "lineage/retired_v2",
        "lineage/retired_v3",
        "lineage/unchanged_repair_design",
        "authority/retired_",
        "cross_protocol_replay_safety",
        "domains/retired_domain_prefixes",
        "source_blind_preregistration",
        "hash_binding/packet_manifest_must_bind",
        "operational_receipt_schemas/slot_segment_closure_resolution",
        "uncertainty/retired_",
    )
    for path, value in _strings(plan):
        if any(path.startswith(prefix) for prefix in retired_only_paths):
            continue
        for identity in RETIRED_IDENTITIES:
            assert identity not in value, f"retired identity {identity} at {path}"


def test_namespace_derived_values_recompute_from_the_v4_domains() -> None:
    plan = _plan()
    domains = plan["domains"]

    expected_segment_id = hashlib.sha256(
        domains["runtime_segment_utf8"].encode("utf-8")
        + b"\x00"
        + _canonical(
            {
                "segment_index": 0,
                "attestation_source_sha256": plan["release_control_watermark"]["sha256"],
                "lower_bound_exclusive_at": plan["release_control_watermark"][
                    "release_effective_at"
                ],
            }
        )
    ).hexdigest()
    assert domains["initial_segment_id_value"] == expected_segment_id
    assert plan["runtime_segments"]["initial_segment"]["segment_id"] == expected_segment_id
    # and it is genuinely distinct from the retired v3 identity
    assert expected_segment_id != _v3_plan()["domains"]["initial_segment_id_value"]

    partition = plan["partition"]
    assert partition["domain_separator_utf8"] == "confirmatory-holdout-v4/partition/v1"
    for vector in partition["golden_vectors"]:
        representative = vector["representative_display"].replace("\\0", "\x00")
        digest = hashlib.sha256(
            partition["domain_separator_utf8"].encode("utf-8")
            + b"\x00"
            + representative.encode("utf-8")
        ).digest()
        bucket = int.from_bytes(digest[0:4], "big") % 100
        assert vector["digest_sha256"] == digest.hex()
        assert vector["bucket"] == bucket
        assert vector["partition"] == ("holdout" if bucket < 50 else "shadow")

    v3_seeds = {seed["value"] for seed in _v3_plan()["uncertainty"]["seeds"].values()}
    for seed in plan["uncertainty"]["seeds"].values():
        expected = int.from_bytes(
            hashlib.sha256(
                plan["namespace"].encode("utf-8") + b"\x00" + seed["label"].encode("utf-8")
            ).digest()[0:4],
            "big",
        )
        assert seed["value"] == expected
        assert seed["value"] not in v3_seeds


def test_initial_runtime_change_closure_has_no_prior_source_binding_member() -> None:
    plan = _plan()
    schema = plan["operational_receipt_schemas"]["initial_runtime_change_closure_resolution"]
    fields = schema["fields_exactly"]

    # The defining property: the member is absent, not null and not a sentinel.
    assert PRIOR_BINDING_MEMBER not in fields
    assert schema["prior_source_binding_core_member_present"] is False
    assert schema["prior_source_binding_core_null_or_sentinel_allowed"] is False
    assert plan["runtime_segments"]["initial_runtime_change_closure_allowlist"] == fields

    # No nested object anywhere in the schema re-introduces the member as a key,
    # and no bound-evidence entry names it as carried data.
    assert PRIOR_BINDING_MEMBER not in set(_object_keys(schema))
    for entry in schema["bound_evidence_exactly"]:
        assert "prior source-binding" not in entry
        assert "prior source binding" not in entry
    # The only place the member name may appear is the rule that forbids it.
    mentions = {
        path for path, value in _string_leaves(schema) if PRIOR_BINDING_MEMBER in value
    }
    assert mentions == {"absent_member_rule"}, mentions
    for token in ("null", "sentinel", "placeholder", "invalid"):
        assert token in schema["absent_member_rule"], token

    # it binds exactly the evidence that does exist at slot 0
    assert fields == [
        "schema_version",
        "namespace",
        "receipt_kind",
        "status",
        "resolution_id",
        "slot_index",
        "scheduled_at",
        "grace_deadline_at",
        "launched_at",
        "validated_at",
        "segment_index",
        "segment_id",
        "mismatch_kinds",
        "prior_active_services_state_sha256_and_bytes",
        "observed_active_services_state_sha256_and_bytes",
        "observed_source_binding_core_sha256_and_bytes",
        "attestation_sha256_and_bytes",
        "attestation_attempt_marker_sha256_and_bytes",
        "previous_ledger_entry_sha256",
        "runtime_observer_sha256_and_bytes",
        "analysis_plan_sha256_and_bytes",
    ]
    assert schema["receipt_kind"] == "initial-runtime-change-closure"
    assert schema["status"] == "segment-closed"
    assert schema["slot_index"] == 0
    assert schema["segment_index"] == 0
    assert schema["mismatch_kinds"] == ["active-services-state-change"]
    assert plan["runtime_segments"][
        "initial_runtime_change_closure_mismatch_kinds_exactly"
    ] == ["active-services-state-change"]
    assert schema["count_free"] is True
    assert schema["carries_counts_snapshots_or_keys"] is False
    assert schema["reads_source_rows"] is False
    assert schema["consumes_slot_index"] == 0
    assert schema["next_probe_slot_index"] == 1
    assert schema["grants_authority"] is False
    assert schema["requires_active_segment_object"] is False
    assert schema["fabricated_predecessor_active_segment_allowed"] is False
    assert schema["reusable_in_any_other_slot"] is False

    # every field it does carry is a declared, typed field
    typed = plan["operational_receipt_schemas"]["common_validation"]["closed_field_type_rules"]
    known = {name for names in typed.values() for name in names}
    for field in fields:
        assert field in known or field in {"schema_version", "namespace"}, field


def test_both_bootstrap_branches_are_declared_and_mutually_exclusive() -> None:
    plan = _plan()
    segments = plan["runtime_segments"]

    branches = segments["initial_bootstrap_branches_exactly"]
    assert len(branches) == 2
    assert [branch["branch"] for branch in branches] == [
        "same-tuple-initial-binding",
        "changed-tuple-pre-binding-closure",
    ]
    assert [branch["receipt_kind"] for branch in branches] == [
        "source-binding-attestation",
        "initial-runtime-change-closure",
    ]
    for branch in branches:
        assert branch["prior_source_binding_core_exists"] is False
    assert segments["initial_bootstrap_branches_mutually_exclusive"] is True
    assert "equality" in segments["initial_bootstrap_branch_selector"]
    assert PRIOR_BINDING_MEMBER not in segments[
        "prior_source_binding_core_for_unbound_initial_segment"
    ]

    closure = plan["operational_receipt_schemas"]["initial_runtime_change_closure_resolution"]
    ordinary = plan["operational_receipt_schemas"]["slot_segment_closure_resolution"]
    # ordinary closure keeps the strict two-core schema
    assert PRIOR_BINDING_MEMBER in ordinary["fields_exactly"]
    assert "observed_source_binding_core_sha256_and_bytes" in ordinary["fields_exactly"]
    assert ordinary["prior_source_binding_core_required"] is True
    assert ordinary["receipt_kind"] == "slot-segment-closed"
    # the two schemas exclude each other
    assert ordinary["valid_before_initial_source_binding_attestation"] is False
    assert closure["valid_after_initial_source_binding_attestation"] is False
    assert closure["receipt_kind"] != ordinary["receipt_kind"]
    assert set(closure["fields_exactly"]) != set(ordinary["fields_exactly"])

    # the inherited closure prose is scoped, so no rule reads as authorizing the
    # two-core schema before a binding exists
    assert "before the initial source binding exists" in segments["slot_change_action"]
    assert "initial-runtime-change-closure" in segments["slot_change_action"]
    mismatch_rule = segments["closure_mismatch_kinds_rule"]
    assert "post-binding slot-segment-closed" in mismatch_rule
    assert "no prior source-binding core exists to differ from" in mismatch_rule
    attempt = plan["sources"]["initial_source_binding_attempt"]
    assert "equal to the watermark yields the source-binding-attestation" in attempt
    assert "different from the watermark yields only the count-free" in attempt

    # the state machine reaches the replacement transition from the initial state
    accrual = plan["accrual_and_stopping"]
    assert accrual["initial_state"] == "initial-segment-awaiting-source-binding"
    assert accrual["initial_state_implementation_alias"] == "initial-source-binding-required"
    from_initial = [
        transition
        for transition in accrual["state_machine"]
        if transition["from"] == "initial-segment-awaiting-source-binding"
    ]
    assert len(from_initial) == 4
    closure_transitions = [
        transition
        for transition in from_initial
        if "initial-runtime-change-closure" in transition["event"]
    ]
    assert len(closure_transitions) == 1
    transition = closure_transitions[0]
    assert transition["to"] == "segment-closed-awaiting-immediate-re-attestation"
    assert transition["terminal"] is False
    assert transition["authority_granted"] is False

    rules = accrual["bootstrap_resolution_rules"]
    assert rules["initial_runtime_change_closure_is_the_immutable_slot_0_resolution"] is True
    assert rules["next_probe_slot_index_after_initial_runtime_change_closure"] == 1
    for flag in (
        "initial_runtime_change_closure_appends_counts",
        "initial_runtime_change_closure_appends_snapshots",
        "initial_runtime_change_closure_constructs_active_segment_object",
        "ordinary_two_core_closure_before_initial_binding",
        "prior_source_binding_core_inference_or_backfill",
        "counts_snapshots_latches_or_evidence_crossing_the_break",
    ):
        assert rules[flag] is False, flag
    assert rules["initial_runtime_change_closure_clears_every_latch_and_source_authority"] is True
    assert rules["initial_runtime_change_closure_retains_typed_pre_binding_predecessor"] is True
    assert accrual["ledger"][
        "initial_runtime_change_closure_appends_counts_or_snapshots"
    ] is False


def test_ledger_accepts_the_new_closure_as_a_typed_slot_zero_entry() -> None:
    ledger = _plan()["operational_receipt_schemas"]["ledger_entry"]
    assert "initial-runtime-change-closure" in ledger["entry_kind_enum"]
    assert len(set(ledger["entry_kind_enum"])) == len(ledger["entry_kind_enum"])
    assert ledger["artifact_kind_rule"]["initial-runtime-change-closure"] == (
        "initial-runtime-change-closure with status segment-closed at slot 0"
    )
    assert "initial-runtime-change-closure is exactly 0" in ledger["slot_index_or_null"]
    order = ledger["bootstrap_and_successor_order"]
    assert "exactly one of" in order
    assert "initial source-binding" in order
    assert "initial-runtime-change-closure" in order
    assert ledger["genesis_rule"].startswith(
        "entry 0 is segment-attestation pointing to the exact release control-watermark"
    )


def test_cross_protocol_replay_is_structurally_impossible() -> None:
    plan = _plan()
    safety = plan["cross_protocol_replay_safety"]
    assert safety["retired_namespaces_exactly"] == [
        "confirmatory-holdout-v2",
        "confirmatory-holdout-v3",
    ]
    assert safety["retired_artifact_accepted_by_v4_validator"] is False
    assert safety["v4_artifact_accepted_by_retired_validator"] is False
    assert safety["retired_identity_reuse_allowed"] is False
    assert safety["retired_authority_transfer_allowed"] is False
    assert len(safety["separating_constants"]) >= 7

    constants = plan["operational_receipt_schemas"]["common_validation"]["constants"]
    assert constants == {"schema_version": 4, "namespace": "confirmatory-holdout-v4"}
    v3_constants = _v3_plan()["operational_receipt_schemas"]["common_validation"]["constants"]
    assert constants["schema_version"] != v3_constants["schema_version"]
    assert constants["namespace"] != v3_constants["namespace"]

    # every keyed preimage domain differs from its v3 counterpart
    v3_domains = _v3_plan()["domains"]
    for key, value in plan["domains"].items():
        if key.endswith("_utf8") and isinstance(value, str) and value.startswith(
            "confirmatory-holdout-"
        ):
            assert value != v3_domains[key], key

    # a source-binding core cannot collide across protocols: alias IDs differ
    assert plan["sources"]["source_binding_core"]["alias_ids_by_alias"] != (
        _v3_plan()["sources"]["source_binding_core"]["alias_ids_by_alias"]
    )
    assert plan["operational_receipt_schemas"]["common_validation"]["composite_types"][
        "alias_id_map"
    ] == "object exactly {local:confirmatory-local-v4-ro,alt:confirmatory-alt-v4-ro}"


def test_sealing_sequence_and_one_shot_reader_authorities_are_preserved() -> None:
    plan = _plan()
    v3 = _v3_plan()
    assert plan["publication_and_ordering"] == _freshen(v3["publication_and_ordering"])
    assert plan["publication_and_ordering"]["state_sequence"][0] == "v4-protocol-frozen"
    assert plan["publication_and_ordering"]["packet_publication_attempts"] == 1
    assert plan["publication_and_ordering"]["reseal_allowed"] is False
    assert plan["publication_and_ordering"][
        "packet_seal_commit_must_be_strict_ancestor_of_repair_and_evaluator_commits"
    ] is True
    assert plan["publication_and_ordering"]["sealed_state"]["semantic_reads"] == {
        "confirmatory-shadow-v4-eval": 0,
        "confirmatory-holdout-v4-eval": 0,
    }
    assert plan["authority"]["evaluation_state_machine"] == v3["authority"][
        "evaluation_state_machine"
    ]
    assert plan["authority"]["execution_order"] == [
        "confirmatory-shadow-v4-eval",
        "confirmatory-holdout-v4-eval",
    ]
    for key in (
        "evaluation_process_watchdog_seconds",
        "reader_watchdog_seconds",
        "forced_termination_grace_seconds",
        "controller_handoff_deadline_seconds",
        "joint_release_deadline_seconds",
        "development_and_public_start_deadline_seconds",
    ):
        assert plan["authority"][key] == v3["authority"][key], key


def test_protocol_text_states_the_bootstrap_rules_and_leaks_no_private_locator() -> None:
    readme = (NAMESPACE / "README.md").read_text(encoding="utf-8")
    policy = (NAMESPACE / "POLICY.md").read_text(encoding="utf-8")
    plan_text = PLAN_PATH.read_text(encoding="utf-8")

    for text in (readme, policy, plan_text):
        assert "/home/" not in text
        assert "/root/" not in text
        assert ".sqlite3" not in text
        assert "global.sqlite3" not in text
        assert "localhost:" not in text

    for text in (readme, policy):
        assert "2026-08-17T00:00:00Z" in text
        assert "2026-09-14T06:00:00Z" in text
        assert "initial-runtime-change-closure" in text
        assert "confirmatory-holdout-v4-eval" in text
        assert "confirmatory-shadow-v4-eval" in text

    # both documents state the defining absence rule in their own words
    for text in (readme, policy):
        assert "prior-source-binding member" in text
        assert "mutually exclusive" in text
        assert "source-binding-attestation" in text
    assert "not a null, not a sentinel" in readme
    assert "no source row" in readme and "no source row" in policy
    assert "immutable, non-authoritative" in policy
    assert "retired-unreachable-initial-bootstrap" in policy
    assert "46a9951842512333b0896370056d07a9e1c25bdf" in readme
    assert "76d34d33b7be57230ef0988662060fe70eea229f9b07f2cd5b8a2354600a79f5" in policy
