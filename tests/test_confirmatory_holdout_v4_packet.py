"""Adversarial falsification suite for the confirmatory-holdout-v4 tooling.

The packet builder, the independent verifier and the seal launcher are treated
as untrusted here.  Selection, component formation, partition bucketing and the
identity HMAC are reimplemented below as an independent oracle: no expected
value in this module is ever obtained from ``recipe/build.py``,
``recipe/verify.py``, ``scripts/ap_confirmatory_seal_v4.py`` or any of their
helpers.  The three are loaded solely so they can be *driven* -- as a
subprocess where a whole ceremony is under test, and by direct call where a
single guard is.  Every number, digest, bucket, encoding, row count and
timestamp this file asserts against is derived here, from the frozen protocol,
by code written a second time.

The fixture strategy mirrors ``tests/test_replacement_holdout_packet.py``.  One
real synthetic packet is built once per session with
``build.py self-check --work-dir``; each test copies it, makes the copy
writable (build outputs are 0o400/0o444), applies exactly one surgical
mutation, and then re-derives every affected ``sha256``/``bytes``/row count so
the mutation survives the hash binding and can only be caught by a semantic
check.  Those rehashed variants carry the real evidence; a mutation that dies
on a stale hash proves nothing about the invariant it was aimed at.

Two things this suite must never do, and does not.  It seals no real packet:
the builder and the seal launcher are driven only against temporary
directories, and both are shown *behaviourally* to refuse an output root that
resolves inside the frozen namespace.  And it asserts no blanket absence of
``corpus/``, ``manifest.json``, ``seal-receipt.json``, ``segments/`` or
``probes/`` inside that namespace: those five names are exactly what the
downstream sibling ``accrue-and-seal-confirmatory-v4`` legitimately creates, so
a directory snapshot would turn red the moment that sibling lands.  The
forward-compatible proof is ownership plus refusal, not a snapshot.
"""

from __future__ import annotations

import sys

# The builder and the seal launcher promise that a self-check touches nothing
# outside its work directory.  Both set this themselves at import time, but a
# stray ``scripts/__pycache__`` written before that assignment executes would
# already have broken the promise, so it is set here first.
sys.dont_write_bytecode = True

import ast
import base64
import hashlib
import hmac
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pytest

from living_memory.config import (
    DEFAULT_RETRIEVAL_WEIGHTS,
    MemoryConfig,
    RetrievalWeightConfig,
)
from living_memory.storage import MemoryStore


ROOT = Path(__file__).resolve().parents[1]
NAMESPACE_ROOT = ROOT / "artifacts/animal-planet/evaluation/confirmatory-holdout-v4"
RECIPE_ROOT = NAMESPACE_ROOT / "recipe"
BUILD = RECIPE_ROOT / "build.py"
VERIFY = RECIPE_ROOT / "verify.py"
SEAL = ROOT / "scripts/ap_confirmatory_seal_v4.py"
PLAN_PATH = NAMESPACE_ROOT / "analysis-plan.json"

# ``bindings.runtime_observer_aggregate_probe_and_accrual_ledger_with_tests``
# is the campaign tool group: everything that touches evidence, bound by its
# own repo-relative path.  The set is re-spelled here by hand, independently of
# the builder's enumeration and of the verifier's shape rule, so that a tool
# which lands unbound turns this file red rather than passing on the builder's
# own say-so.
CAMPAIGN_TOOL_BINDING_KEY = "runtime_observer_aggregate_probe_and_accrual_ledger_with_tests"
CAMPAIGN_TOOL_PATHS = (
    "scripts/ap_confirmatory_accrual_v4.py",
    "scripts/ap_confirmatory_probe_v4.py",
    "scripts/ap_confirmatory_runtime_v4.py",
    "scripts/ap_confirmatory_seal_v4.py",
    "scripts/ap_confirmatory_slot_v4.py",
    "scripts/ap_confirmatory_snapshot_v4.py",
    "scripts/v4_cadence.py",
    "tests/test_ap_confirmatory_accrual_v4.py",
    "tests/test_ap_confirmatory_probe_production_seed_v4.py",
    "tests/test_ap_confirmatory_probe_v4.py",
    "tests/test_ap_confirmatory_runtime_v4.py",
    "tests/test_ap_confirmatory_slot_v4.py",
    "tests/test_ap_confirmatory_snapshot_v4.py",
    "tests/test_v4_cadence.py",
)
# The snapshot launcher, the slot driver and the cadence renderer produced
# evidence while nothing bound them.  These are the six the group was extended
# to cover; the rest of ``CAMPAIGN_TOOL_PATHS`` was already bound.
CAMPAIGN_TOOLS_ADDED = (
    "scripts/ap_confirmatory_snapshot_v4.py",
    "scripts/ap_confirmatory_slot_v4.py",
    "scripts/v4_cadence.py",
    "tests/test_ap_confirmatory_snapshot_v4.py",
    "tests/test_ap_confirmatory_slot_v4.py",
    "tests/test_v4_cadence.py",
)


# --------------------------------------------------------------------------
# The frozen protocol, re-spelled independently
# --------------------------------------------------------------------------

NAMESPACE = "confirmatory-holdout-v4"
SCHEMA_VERSION = 4
SOURCE_ALIASES = ("local", "alt")

PARTITION_DOMAIN = b"confirmatory-holdout-v4/partition/v1\0"
IDENTITY_DOMAIN = b"confirmatory-holdout-v4/identity/v1\0"
SNAPSHOT_SET_DOMAIN = b"confirmatory-holdout-v4/snapshot-set/v1\0"
RETIRED_DOMAIN_PREFIXES = ("confirmatory-holdout-v2/", "confirmatory-holdout-v3/")

SPLIT_MODULUS = 100
HOLDOUT_UPPER_EXCLUSIVE = 50
HOLDOUT_PARTITION = "holdout"
SHADOW_PARTITION = "shadow"
PARTITIONS = (HOLDOUT_PARTITION, SHADOW_PARTITION)

MANIFEST_NAME = "manifest.json"
CORPUS_DIRECTORY = "corpus"
HOLDOUT_CORPUS = "holdout.jsonl"
SHADOW_CORPUS = "shadow.jsonl"
SEED_MANIFEST = "seed-state-manifest.json"
PRESEAL_RECEIPT = "preseal-receipt.json"
SEED_STATES = tuple(
    f"seed-state-{partition}-{alias}.sqlite3"
    for partition in PARTITIONS
    for alias in SOURCE_ALIASES
)
CORPUS_MEMBERS = frozenset(
    {HOLDOUT_CORPUS, SHADOW_CORPUS, *SEED_STATES, SEED_MANIFEST, PRESEAL_RECEIPT}
)

# ``recipe/`` is this subtree's own ownership boundary and is stable.
RECIPE_OWNED_ENTRIES = frozenset({"build.py", "verify.py"})

# The verifier's closed public failure vocabulary.  Nothing outside this may
# ever be printed, whatever the mutation.
PUBLIC_FAILURE_CHECKS = frozenset(
    {
        "overwrite",
        "schema",
        "production_shape",
        "privacy",
        "partition",
        "replayability",
        "runtime_segment",
        "reader_authority",
        "manifest_order",
        "pin",
        "boundary",
        "candidate_changed",
        "arguments",
        "internal",
    }
)

AGGREGATE_FIELDS = (
    "selected_event_count",
    "selected_replayable_event_count",
    "selected_nonreplayable_event_count",
    "holdout_unseen_automatic_family_count",
    "holdout_unseen_automatic_event_count",
    "holdout_unseen_automatic_component_count",
    "holdout_organic_event_count",
    "holdout_organic_session_count",
    "holdout_organic_component_count",
    "holdout_project_scope_count",
    "shadow_real_workflow_count",
    "shadow_replayable_logical_call_count",
    "shadow_real_workflow_component_count",
    "shadow_project_scope_count",
)
HOLDOUT_FLOORS = {
    "holdout_unseen_automatic_family_count": 30,
    "holdout_unseen_automatic_event_count": 150,
    "holdout_unseen_automatic_component_count": 30,
    "holdout_organic_event_count": 200,
    "holdout_organic_session_count": 30,
    "holdout_organic_component_count": 30,
    "holdout_project_scope_count": 2,
}
SHADOW_FLOORS = {
    "shadow_real_workflow_count": 30,
    "shadow_replayable_logical_call_count": 100,
    "shadow_real_workflow_component_count": 30,
    "shadow_project_scope_count": 2,
}

SEMANTIC_READER_IDS = ("confirmatory-shadow-v4-eval", "confirmatory-holdout-v4-eval")
RETIRED_READER_IDS = (
    "confirmatory-shadow-v3-eval",
    "confirmatory-holdout-v3-eval",
    "confirmatory-shadow-v2-eval",
    "confirmatory-holdout-v2-eval",
)

# ``measurement.state_isolation.construction`` -- packet column allowlist.
SEED_COLUMN_ALLOWLIST = {
    "nodes": (
        "packet_node_label",
        "level",
        "deidentified_content",
        "packet_scope_label",
        "created_at",
    ),
    "connections": (
        "source_packet_node_label",
        "target_packet_node_label",
        "relation_type",
        "weight",
    ),
    "retrieval_weights": (
        "packet_scope_label",
        "bm25_weight",
        "vector_weight",
        "graph_weight",
    ),
}
FORCED_EMPTY_TABLES = ("recall_events", "recall_fingerprints", "kv", "metadata")
REQUIRED_POLICY_KEYS = frozenset({"default", "project", "global", "session"})
NODE_LEVELS = frozenset({"trace", "concept", "schema"})
RELATION_TYPES = frozenset(
    {"related", "caused", "contradicts", "supersedes", "requires"}
)

# ``replayability.replay_input_schema.ambient_context``.
AMBIENT_KEYS = (
    "scope",
    "session_id",
    "session",
    "project",
    "project_name",
    "workspace",
    "workspace_path",
    "cwd",
    "project_scope",
    "agent",
    "task",
    "transport_session_id",
)

BUILD_EXIT_OK = 0
BUILD_EXIT_ERROR = 1
BUILD_EXIT_NAMESPACE_REFUSED = 3
NAMESPACE_REFUSAL_CODE = "namespace_output_root_refused"

SEAL_EXIT_SEALED = 0
SEAL_EXIT_TERMINAL = 1
SEAL_EXIT_NAMESPACE_REFUSED = 3
SEAL_STATUS_TERMINAL = "terminal-seal-failure"


# --------------------------------------------------------------------------
# The independent oracle
# --------------------------------------------------------------------------


def _canonical_bytes(value: Any) -> bytes:
    """``domains.v4_canonical_json_utf8``, spelled out here."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _identity_of(raw: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _representative(alias: str, event_id: str) -> bytes:
    """``partition.source_qualified_keys``: alias, NUL, event id."""

    return alias.encode("utf-8") + b"\0" + event_id.encode("utf-8")


def _partition_digest(representative: bytes) -> bytes:
    return hashlib.sha256(PARTITION_DOMAIN + representative).digest()


def _partition_bucket(representative: bytes) -> int:
    return int.from_bytes(_partition_digest(representative)[0:4], "big") % SPLIT_MODULUS


def _partition_for(representative: bytes) -> str:
    bucket = _partition_bucket(representative)
    return HOLDOUT_PARTITION if bucket < HOLDOUT_UPPER_EXCLUSIVE else SHADOW_PARTITION


def _identity_token(key: bytes, query: str, requested_scope: str) -> bytes:
    """``identity.normalization`` under the fresh v4 identity domain."""

    normalized = " ".join(query.split())
    message = IDENTITY_DOMAIN + (normalized + "\n" + requested_scope).encode("utf-8")
    return hmac.new(bytes(key), message, hashlib.sha256).digest()


def _snapshot_set_preimage(local: Mapping[str, Any], alt: Mapping[str, Any]) -> bytes:
    return SNAPSHOT_SET_DOMAIN + _canonical_bytes(
        {
            "local": {"sha256": local["sha256"], "bytes": local["bytes"]},
            "alt": {"sha256": alt["sha256"], "bytes": alt["bytes"]},
        }
    )


class _Components:
    """Union-find over exactly the two declared ``partition.component_edges``."""

    __slots__ = ("_parent",)

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self._parent.setdefault(value, value)

    def find(self, value: str) -> str:
        root = value
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[value] != root:
            self._parent[value], value = root, self._parent[value]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root

    def classes(self) -> dict[str, set[str]]:
        groups: dict[str, set[str]] = {}
        for member in self._parent:
            groups.setdefault(self.find(member), set()).add(member)
        return groups


def _oracle_components(records: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    components = _Components()
    for record in records:
        components.add(record["event_label"])
    groups: dict[tuple[str, str, str], list[str]] = {}
    for record in records:
        if record["record_kind"] != "case":
            continue
        for kind in ("family_label", "workflow_label"):
            value = record[kind]
            if value is not None:
                groups.setdefault((kind, record["source_label"], value), []).append(
                    record["event_label"]
                )
    for members in groups.values():
        for member in members[1:]:
            components.union(members[0], member)
    return components.classes()


_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _microseconds(value: str) -> int:
    """``selection.utc_timestamp_contract``: parse, never compare as text."""

    if len(value) != 27 or value[-1] != "Z" or value[10] != "T" or value[19] != ".":
        raise ValueError(value)
    year, month, day = int(value[0:4]), int(value[5:7]), int(value[8:10])
    hour, minute, second = int(value[11:13]), int(value[14:16]), int(value[17:19])
    fraction = int(value[20:26])
    days = sum(366 if _leap(step) else 365 for step in range(1970, year))
    for step in range(1, month):
        days += _DAYS_IN_MONTH[step - 1] + (1 if step == 2 and _leap(year) else 0)
    days += day - 1
    return (
        ((days * 24 + hour) * 60 + minute) * 60_000_000 + second * 1_000_000 + fraction
    )


def _oracle_aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """All fourteen counts, recomputed here from the sealed corpus alone."""

    cases = [record for record in records if record["record_kind"] == "case"]
    inventory = [record for record in records if record["record_kind"] == "inventory"]

    def automatic(record: Mapping[str, Any]) -> bool:
        return record["replay_input"]["agent"] is None

    holdout_cases = [
        record for record in cases if record["partition"] == HOLDOUT_PARTITION
    ]
    holdout_automatic = [
        record
        for record in holdout_cases
        if record["floor_counted"] and automatic(record)
    ]
    holdout_organic = [
        record
        for record in holdout_cases
        if record["floor_counted"] and not automatic(record)
    ]
    shadow_calls = [
        record
        for record in cases
        if record["partition"] == SHADOW_PARTITION and record["floor_counted"]
    ]

    def project_scopes(rows: Sequence[Mapping[str, Any]]) -> set[str]:
        return {
            row["replay_input"]["requested_scope"]
            for row in rows
            if row["replay_input"]["requested_scope"].startswith("project:")
        }

    return {
        "selected_event_count": len(records),
        "selected_replayable_event_count": len(cases),
        "selected_nonreplayable_event_count": len(inventory),
        "holdout_unseen_automatic_family_count": len(
            {(row["source_label"], row["family_label"]) for row in holdout_automatic}
        ),
        "holdout_unseen_automatic_event_count": len(holdout_automatic),
        "holdout_unseen_automatic_component_count": len(
            {row["component_label"] for row in holdout_automatic}
        ),
        "holdout_organic_event_count": len(holdout_organic),
        "holdout_organic_session_count": len(
            {
                (row["source_label"], row["workflow_label"])
                for row in holdout_organic
                if row["workflow_label"] is not None
            }
        ),
        "holdout_organic_component_count": len(
            {row["component_label"] for row in holdout_organic}
        ),
        "holdout_project_scope_count": len(project_scopes(holdout_cases)),
        "shadow_real_workflow_count": len(
            {(row["source_label"], row["workflow_label"]) for row in shadow_calls}
        ),
        "shadow_replayable_logical_call_count": len(shadow_calls),
        "shadow_real_workflow_component_count": len(
            {row["component_label"] for row in shadow_calls}
        ),
        "shadow_project_scope_count": len(project_scopes(shadow_calls)),
    }


def _oracle_floors_pass(counts: Mapping[str, int]) -> bool:
    return all(
        counts[field] >= minimum for field, minimum in HOLDOUT_FLOORS.items()
    ) and all(counts[field] >= minimum for field, minimum in SHADOW_FLOORS.items())


def _oracle_encodings(value: bytes) -> dict[str, bytes]:
    """The five encodings a leaked secret can wear."""

    return {
        "raw": value,
        "hex": value.hex().encode("ascii"),
        "upper_hex": value.hex().upper().encode("ascii"),
        "base64": base64.b64encode(value),
        "urlsafe_base64": base64.urlsafe_b64encode(value),
    }


PRIVACY_ENCODINGS = ("raw", "hex", "upper_hex", "base64", "urlsafe_base64")

_SCOPE_SAFE_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


def _scope_safe_secret(material: bytes) -> bytes:
    """Project secret material onto thirty-two printable ASCII bytes.

    The claim under test is that a leak is caught in *all five* encodings, and
    the raw encoding is only plantable inside a packet if its bytes are UTF-8
    without control characters.  Projecting the material through SHA-256 onto a
    printable alphabet keeps the raw form plantable while leaving every derived
    encoding exactly what the verifier's needle expansion would compute for the
    real thirty-two secret bytes.
    """

    digest = hashlib.sha256(material).digest()
    return "".join(
        _SCOPE_SAFE_ALPHABET[byte % len(_SCOPE_SAFE_ALPHABET)] for byte in digest
    ).encode("ascii")


# ``privacy``/``identity``: the four kinds of material that may never persist.
_ORACLE_KEY = hashlib.sha256(b"confirmatory-holdout-v4 oracle identity key").digest()
_ORACLE_SALT = hashlib.sha256(b"confirmatory-holdout-v4 oracle deidentification salt").digest()
_ORACLE_QUERY = "proj-alpha repeated automatic probe"
_ORACLE_SCOPE = "project:proj-alpha"
_ORACLE_NORMALIZED = (" ".join(_ORACLE_QUERY.split()) + "\n" + _ORACLE_SCOPE).encode("utf-8")

PRIVACY_SECRETS = {
    "raw_content": _scope_safe_secret(_ORACLE_QUERY.encode("utf-8")),
    "identity_token": _scope_safe_secret(
        _identity_token(_ORACLE_KEY, _ORACLE_QUERY, _ORACLE_SCOPE)
    ),
    "salt": _scope_safe_secret(_ORACLE_SALT),
    "unkeyed_digest": _scope_safe_secret(hashlib.sha256(_ORACLE_NORMALIZED).digest()),
}
PRIVACY_SECRET_KINDS = tuple(PRIVACY_SECRETS)


# ``partition.golden_vectors``, transcribed from the frozen plan and asserted
# against this module's own digest rather than against the plan's arithmetic.
GOLDEN_VECTORS = (
    ("local", "evt-a", "09981681b80beaee0cc70792c38ecf07d7ef08aa67f36f8fa604d4ac2d4c4c67", 77, SHADOW_PARTITION),
    ("alt", "evt-a", "9b2536ab81118aa26c47c0d8c291adda4656ab12a4e2531b79d93b3659e912ce", 7, HOLDOUT_PARTITION),
    ("local", "evt-c", "fc06bb285466722ddeb69fe6fb2ceadb2aa68a47a0cff7a7e4b52613e4b97173", 60, SHADOW_PARTITION),
)

# Witnesses either side of the single split boundary, found by search over the
# oracle digest.  49 is the last holdout bucket, 50 the first shadow bucket.
BOUNDARY_WITNESSES = (
    ("local", "boundary-70", 49, HOLDOUT_PARTITION),
    ("alt", "boundary-65", 49, HOLDOUT_PARTITION),
    ("local", "boundary-159", 50, SHADOW_PARTITION),
    ("alt", "boundary-89", 50, SHADOW_PARTITION),
)


# --------------------------------------------------------------------------
# Redacting containers
# --------------------------------------------------------------------------


class _Secret(bytes):
    """Secret-shaped test material that never renders into pytest output."""

    def __repr__(self) -> str:  # pragma: no cover - only reached on failure
        return "<redacted-secret>"


@dataclass(repr=False)
class _SyntheticPacket:
    """The session base packet plus the work tree that produced it."""

    packet: Path
    work: Path
    summary: dict

    def __repr__(self) -> str:  # pragma: no cover - only reached on failure
        return "<synthetic-packet>"


@dataclass(repr=False)
class _SealRun:
    """One completed synthetic seal ceremony."""

    work: Path
    summary: dict

    def __repr__(self) -> str:  # pragma: no cover - only reached on failure
        return "<seal-run>"


# --------------------------------------------------------------------------
# Process helpers
# --------------------------------------------------------------------------


def _run(command: Sequence[str], *, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _private_directory(path: Path) -> Path:
    """A work root the builder and sealer accept: owner-only, no group/other."""

    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def _run_builder(*arguments: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return _run(
        [sys.executable, "-I", os.fspath(BUILD), *arguments], timeout=timeout
    )


def _run_sealer(*arguments: str, timeout: int = 900) -> subprocess.CompletedProcess:
    return _run(
        [sys.executable, "-I", os.fspath(SEAL), *arguments], timeout=timeout
    )


def _run_verifier(packet: Path) -> subprocess.CompletedProcess:
    return _run(
        [sys.executable, "-I", os.fspath(VERIFY), "sealed", "--packet-dir", os.fspath(packet)]
    )


def _error_code(result: subprocess.CompletedProcess) -> str | None:
    try:
        payload = json.loads(result.stderr.decode("utf-8") or result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload.get("code") if type(payload) is dict else None


def _assert_no_leak(blob: bytes, private_values: Iterable[bytes]) -> None:
    for secret in private_values:
        for encoded in _oracle_encodings(bytes(secret)).values():
            if encoded and encoded in blob:
                pytest.fail("private material reached process output", pytrace=False)


def _assert_verifier_pass(result: subprocess.CompletedProcess) -> None:
    if result.stderr:
        pytest.fail("sealed verifier emitted stderr", pytrace=False)
    if result.returncode != 0:
        pytest.fail("sealed verifier rejected a conforming packet", pytrace=False)
    try:
        payload = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        pytest.fail("sealed verifier emitted invalid aggregate JSON", pytrace=False)
    if payload != {"mode": "v4-sealed", "status": "pass"}:
        pytest.fail("sealed verifier pass attestation is not exact", pytrace=False)


def _assert_verifier_rejects(
    result: subprocess.CompletedProcess,
    expected_check: str,
    *,
    private_values: Iterable[bytes] = (),
) -> None:
    """One aggregate-only failure line, carrying exactly the named class."""

    _assert_no_leak(result.stdout + result.stderr, private_values)
    if result.stderr:
        pytest.fail("sealed verifier emitted stderr", pytrace=False)
    if result.returncode == 0:
        pytest.fail("sealed verifier accepted an adversarial mutation", pytrace=False)
    if not result.stdout or len(result.stdout) >= 256:
        pytest.fail("sealed verifier emitted a non-aggregate attestation", pytrace=False)
    try:
        payload = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        pytest.fail("sealed verifier emitted invalid aggregate JSON", pytrace=False)
    if type(payload) is not dict or set(payload) != {"mode", "status", "check"}:
        pytest.fail("sealed verifier failure attestation is not aggregate-only", pytrace=False)
    if payload["mode"] != "v4-sealed" or payload["status"] != "fail":
        pytest.fail("sealed verifier failure attestation is malformed", pytrace=False)
    if payload["check"] not in PUBLIC_FAILURE_CHECKS:
        pytest.fail("sealed verifier printed a code outside the closed vocabulary", pytrace=False)
    if payload["check"] != expected_check:
        pytest.fail(
            f"expected the {expected_check} class, observed {payload['check']}",
            pytrace=False,
        )


# --------------------------------------------------------------------------
# Packet staging, mutation and resealing
# --------------------------------------------------------------------------


def _stage(base: Path, target: Path) -> Path:
    """Copy the session base packet and make the copy mutable and auditable.

    The published packet is 0o400/0o444 by construction, and the in-packet
    ``recipe/verify.py`` is the builder's self-check placeholder.  Installing
    the real verifier bytes is what makes the copy auditable at all: the
    verifier binds the packet's tracked copy to its own executing bytes, and
    the manifest binds that same identity.  Nothing else about the packet is
    touched, so the staged copy is a conforming packet and every rejection
    below is caused by exactly one deliberate mutation.
    """

    shutil.copytree(base, target)
    for item in [target, *target.rglob("*")]:
        item.chmod(0o755 if item.is_dir() else 0o644)
    (target / "recipe" / "verify.py").write_bytes(VERIFY.read_bytes())
    _seal(target, _manifest_of(target))
    return target


def _manifest_of(packet: Path) -> dict:
    return json.loads((packet / MANIFEST_NAME).read_text())


def _seal(packet: Path, manifest: dict, *, rebind_verifier: bool = True) -> None:
    """Re-establish content-before-manifest ordering after a mutation.

    Every corpus member is returned to read-only *before* the manifest is
    rewritten, so no installed byte carries a change stamp newer than the
    sealed-state marker, and the manifest is the last byte written.
    """

    if rebind_verifier:
        manifest["bindings"]["packet_builder_and_independent_verifier"][
            "independent_verifier"
        ] = _identity_of(VERIFY.read_bytes())
    for member in sorted((packet / CORPUS_DIRECTORY).iterdir()):
        member.chmod(0o444)
    path = packet / MANIFEST_NAME
    if path.exists():
        path.chmod(0o644)
        path.unlink()
    path.write_bytes(_canonical_bytes(manifest))
    path.chmod(0o444)


def _rebind(packet: Path, manifest: dict, name: str) -> None:
    """Re-derive the affected content binding so the mutation survives hashing."""

    identity = _identity_of((packet / CORPUS_DIRECTORY / name).read_bytes())
    bindings = manifest["bindings"]
    if name == HOLDOUT_CORPUS:
        bindings["complete_selected_holdout_partition"] = identity
    elif name == SHADOW_CORPUS:
        bindings["complete_selected_shadow_partition"] = identity
    elif name == PRESEAL_RECEIPT:
        bindings["keyed_aggregate_preseal_receipt"] = identity
    else:
        bindings["partition_source_seed_states_and_construction_manifest"][name] = identity


def _read_corpus(packet: Path, name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (packet / CORPUS_DIRECTORY / name).read_bytes().decode("utf-8").splitlines()
    ]


def _write_corpus(packet: Path, name: str, records: Sequence[Mapping[str, Any]]) -> None:
    path = packet / CORPUS_DIRECTORY / name
    path.chmod(0o644)
    path.write_bytes(b"".join(_canonical_bytes(record) + b"\n" for record in records))


def _read_member(packet: Path, name: str) -> dict:
    return json.loads((packet / CORPUS_DIRECTORY / name).read_bytes())


def _write_member(packet: Path, name: str, value: Mapping[str, Any]) -> None:
    path = packet / CORPUS_DIRECTORY / name
    path.chmod(0o644)
    path.write_bytes(_canonical_bytes(value))


def _open_seed(packet: Path, name: str) -> sqlite3.Connection:
    path = packet / CORPUS_DIRECTORY / name
    path.chmod(0o644)
    return sqlite3.connect(os.fspath(path))


def _first_case(
    records: Sequence[dict], predicate: Callable[[dict], bool] = lambda record: True
) -> dict:
    return next(
        record
        for record in records
        if record["record_kind"] == "case" and predicate(record)
    )


def _families(records: Sequence[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        if record["record_kind"] == "case" and record["family_label"] is not None:
            groups.setdefault(
                (record["source_label"], record["family_label"]), []
            ).append(record)
    return groups


# --------------------------------------------------------------------------
# Module fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def builder() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ap_confirmatory_v4_packet_builder_test", BUILD
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verifier() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ap_confirmatory_v4_packet_verifier_test", VERIFY
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sealer() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ap_confirmatory_seal_v4_packet_test", SEAL
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def plan() -> dict:
    return json.loads(PLAN_PATH.read_bytes())


@pytest.fixture(scope="session")
def synthetic_base(tmp_path_factory: pytest.TempPathFactory) -> _SyntheticPacket:
    """One real synthetic packet, built once, never mutated."""

    work = _private_directory(tmp_path_factory.mktemp("v4-self-check") / "work")
    result = _run_builder("self-check", "--work-dir", os.fspath(work))
    if result.returncode != BUILD_EXIT_OK:
        pytest.fail("builder self-check did not complete", pytrace=False)
    summary = json.loads(result.stdout)
    if summary.get("status") != "self-check-pass" or summary.get("real_packet_sealed") is not False:
        pytest.fail("builder self-check did not report a synthetic pass", pytrace=False)
    return _SyntheticPacket(packet=work / "packet", work=work, summary=summary)


@pytest.fixture(scope="session")
def seal_run(tmp_path_factory: pytest.TempPathFactory) -> _SealRun:
    """One complete synthetic seal ceremony, driven end to end."""

    work = _private_directory(tmp_path_factory.mktemp("v4-seal") / "work")
    result = _run_sealer("self-check", "--work-dir", os.fspath(work))
    if result.returncode != SEAL_EXIT_SEALED:
        pytest.fail("seal launcher self-check did not seal the fixture", pytrace=False)
    summary = json.loads(result.stdout)
    if summary.get("status") != "sealed" or summary.get("real_packet_sealed") is not False:
        pytest.fail("seal launcher did not report a synthetic seal", pytrace=False)
    return _SealRun(work=work, summary=summary)


@pytest.fixture
def packet(synthetic_base: _SyntheticPacket, tmp_path: Path) -> Path:
    return _stage(synthetic_base.packet, tmp_path / "packet")


# --------------------------------------------------------------------------
# The oracle proves itself before it is used to judge anything
# --------------------------------------------------------------------------


def test_synthetic_base_packet_is_accepted_and_seals_no_real_packet(
    synthetic_base: _SyntheticPacket, packet: Path
) -> None:
    """The unmutated fixture must pass, or every rejection below is vacuous."""

    _assert_verifier_pass(_run_verifier(packet))
    manifest = _manifest_of(packet)
    assert {item.name for item in (packet / CORPUS_DIRECTORY).iterdir()} == CORPUS_MEMBERS
    assert sorted(manifest["packet_layout"]["corpus"]) == sorted(CORPUS_MEMBERS)
    assert manifest["packet_layout"]["manifest"] == MANIFEST_NAME
    assert synthetic_base.summary["namespace"] == NAMESPACE
    assert synthetic_base.summary["schema_version"] == SCHEMA_VERSION
    assert synthetic_base.summary["synthetic_fixture"] is True
    assert synthetic_base.summary["real_packet_sealed"] is False
    assert synthetic_base.summary["semantic_reads"] == {
        reader: 0 for reader in SEMANTIC_READER_IDS
    }


def test_partition_golden_vectors_match_the_independent_oracle(plan: dict) -> None:
    """``partition.golden_vectors`` asserted directly, three of three."""

    declared = plan["partition"]["golden_vectors"]
    assert len(declared) == len(GOLDEN_VECTORS) == 3
    for vector, (alias, event_id, digest, bucket, partition) in zip(
        declared, GOLDEN_VECTORS, strict=True
    ):
        assert vector["representative_display"] == f"{alias}\\0{event_id}"
        assert vector["digest_sha256"] == digest
        assert vector["bucket"] == bucket
        assert vector["partition"] == partition
        representative = _representative(alias, event_id)
        assert _partition_digest(representative).hex() == digest
        assert _partition_bucket(representative) == bucket
        assert _partition_for(representative) == partition


@pytest.mark.parametrize(
    ("alias", "event_id", "bucket", "partition"), BOUNDARY_WITNESSES
)
def test_partition_bucket_boundary_is_exactly_forty_nine_over_fifty(
    plan: dict, alias: str, event_id: str, bucket: int, partition: str
) -> None:
    """The single split boundary, witnessed from both sides."""

    assert plan["partition"]["modulus"] == SPLIT_MODULUS
    assert plan["partition"]["holdout_buckets"] == {"gte": 0, "lt": HOLDOUT_UPPER_EXCLUSIVE}
    assert plan["partition"]["shadow_buckets"] == {
        "gte": HOLDOUT_UPPER_EXCLUSIVE,
        "lt": SPLIT_MODULUS,
    }
    representative = _representative(alias, event_id)
    assert _partition_bucket(representative) == bucket
    assert _partition_for(representative) == partition
    assert (bucket < HOLDOUT_UPPER_EXCLUSIVE) is (partition == HOLDOUT_PARTITION)


def test_identity_hmac_oracle_normalizes_and_separates_domains() -> None:
    """``identity.normalization`` plus a fresh, never-reused v4 domain."""

    key = _Secret(hashlib.sha256(b"identity-oracle").digest())
    spaced = _identity_token(key, "  alpha   beta \n", "project:p")
    tight = _identity_token(key, "alpha beta", "project:p")
    assert spaced == tight
    assert _identity_token(key, "alpha beta", "project:q") != tight
    assert _identity_token(_Secret(b"\0" * 32), "alpha beta", "project:p") != tight
    # The domain prefix is load-bearing: a bare HMAC over the same message is
    # a different token, and no retired prefix may appear in it.
    assert hmac.new(bytes(key), b"alpha beta\nproject:p", hashlib.sha256).digest() != tight
    for prefix in RETIRED_DOMAIN_PREFIXES:
        assert not IDENTITY_DOMAIN.decode("utf-8").startswith(prefix)
        assert not PARTITION_DOMAIN.decode("utf-8").startswith(prefix)


def test_oracle_reproduces_every_declared_aggregate_and_floor(packet: Path) -> None:
    """Fourteen counts and eleven floors, recomputed here from the corpus."""

    records = [
        record
        for name in (HOLDOUT_CORPUS, SHADOW_CORPUS)
        for record in _read_corpus(packet, name)
    ]
    manifest = _manifest_of(packet)
    counts = _oracle_aggregates(records)
    assert counts == manifest["aggregate_counts"]
    assert set(counts) == set(AGGREGATE_FIELDS)
    assert _oracle_floors_pass(counts) is manifest["floors_pass"] is True
    receipt = _read_member(packet, PRESEAL_RECEIPT)
    assert {field: receipt[field] for field in AGGREGATE_FIELDS} == counts
    # The components the oracle recomputes must sit inside the sealed ones.
    owner = {record["event_label"]: record["component_label"] for record in records}
    for members in _oracle_components(records).values():
        assert len({owner[label] for label in members}) == 1


def test_oracle_reproduces_the_canonical_two_alias_snapshot_set_identity(
    packet: Path,
) -> None:
    bindings = _manifest_of(packet)["bindings"]
    local = bindings["local_alias_immutable_snapshot"]
    alt = bindings["alt_alias_immutable_snapshot"]
    assert local["sha256"] != alt["sha256"]
    preimage = _snapshot_set_preimage(local, alt)
    assert bindings["canonical_two_alias_snapshot_set_identity"] == _identity_of(preimage)


# --------------------------------------------------------------------------
# overwrite
# --------------------------------------------------------------------------


def _fixture_publish_root(root: Path, *, sealed_marker: str | None = None) -> str:
    """A minimal namespace shape, built here rather than borrowed."""

    (root / "recipe").mkdir(parents=True)
    for name in ("README.md", "POLICY.md", "analysis-plan.json"):
        (root / name).write_bytes(b"synthetic self-check placeholder\n")
    (root / "recipe" / "build.py").write_bytes(BUILD.read_bytes())
    verifier = root / "recipe" / "verify.py"
    verifier.write_bytes(b"# synthetic self-check verifier placeholder\n")
    if sealed_marker == MANIFEST_NAME:
        (root / MANIFEST_NAME).write_bytes(b"{}\n")
    elif sealed_marker == CORPUS_DIRECTORY:
        (root / CORPUS_DIRECTORY).mkdir()
    return hashlib.sha256(verifier.read_bytes()).hexdigest()


@pytest.mark.parametrize("occupied", [MANIFEST_NAME, CORPUS_DIRECTORY])
def test_freeze_refuses_a_publish_root_that_already_holds_a_sealed_packet(
    synthetic_base: _SyntheticPacket, tmp_path: Path, occupied: str
) -> None:
    """One publication attempt, no reseal: an occupied root fails closed."""

    draft = tmp_path / "draft"
    shutil.copytree(synthetic_base.work / "draft", draft)
    publish = tmp_path / "publish"
    publish.mkdir(mode=0o755)
    digest = _fixture_publish_root(publish, sealed_marker=occupied)
    result = _run_builder(
        "freeze",
        "--draft-dir",
        os.fspath(draft),
        "--publish-dir",
        os.fspath(publish),
        "--verifier-sha256",
        digest,
    )
    assert result.returncode == BUILD_EXIT_ERROR
    assert _error_code(result) == "publish_target_exists"
    assert result.stdout == b""


def test_freeze_publishes_once_and_refuses_the_second_attempt(
    synthetic_base: _SyntheticPacket, tmp_path: Path
) -> None:
    """The no-overwrite guarantee, proved against a root that really sealed."""

    draft = tmp_path / "draft"
    shutil.copytree(synthetic_base.work / "draft", draft)
    publish = tmp_path / "publish"
    publish.mkdir(mode=0o755)
    digest = _fixture_publish_root(publish)
    arguments = (
        "freeze",
        "--draft-dir",
        os.fspath(draft),
        "--publish-dir",
        os.fspath(publish),
        "--verifier-sha256",
        digest,
    )
    first = _run_builder(*arguments)
    assert first.returncode == BUILD_EXIT_OK
    summary = json.loads(first.stdout)
    assert summary["status"] == "sealed" and summary["synthetic_fixture"] is True
    manifest_before = (publish / MANIFEST_NAME).read_bytes()

    second = _run_builder(*arguments)
    assert second.returncode == BUILD_EXIT_ERROR
    assert _error_code(second) == "publish_target_exists"
    # Not one sealed byte moved.
    assert (publish / MANIFEST_NAME).read_bytes() == manifest_before


def test_rename_noreplace_collision_refuses_to_replace_an_installed_corpus(
    builder: Any, tmp_path: Path
) -> None:
    """``renameat2(RENAME_NOREPLACE)`` is required, not preferred."""

    root = tmp_path / "rename"
    root.mkdir()
    (root / "staging").mkdir()
    (root / CORPUS_DIRECTORY).mkdir()
    (root / CORPUS_DIRECTORY / "installed").write_bytes(b"sealed\n")
    handle = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(builder.BuildError) as caught:
            builder.rename_directory_no_replace(handle, "staging", CORPUS_DIRECTORY)
    finally:
        os.close(handle)
    assert caught.value.code == "publish_target_exists"
    assert (root / CORPUS_DIRECTORY / "installed").read_bytes() == b"sealed\n"
    assert (root / "staging").is_dir()


def test_second_seal_attempt_against_a_consumed_marker_is_terminal(
    seal_run: _SealRun,
) -> None:
    """Authority is consumed at marker creation, never at manifest write."""

    markers = sorted((seal_run.work / "seal" / "markers").iterdir())
    assert len(markers) == 1
    marker_before = markers[0].read_bytes()
    receipt_before = (seal_run.work / "packet" / "seal-receipt.json").read_bytes()

    result = _run_sealer("self-check", "--work-dir", os.fspath(seal_run.work))
    assert result.returncode == SEAL_EXIT_TERMINAL
    summary = json.loads(result.stdout)
    assert summary["status"] == SEAL_STATUS_TERMINAL
    assert summary["failure_stage"] == "marker-create"
    assert summary["failure_reason"] == "marker-collision"
    assert summary["real_packet_sealed"] is False
    assert summary["seal_receipt_written"] is False
    assert summary["handoff_steps_completed"] == 0
    receipt = summary["seal_terminal_receipt"]
    assert receipt["receipt_kind"] == "seal-terminal"
    assert receipt["status"] == SEAL_STATUS_TERMINAL
    assert receipt["manifest_sha256_and_bytes_or_null"] is None
    assert receipt["seal_consumption_marker_sha256_and_bytes_or_null"] == _identity_of(
        marker_before
    )
    # A refused replacement child publishes nothing and rewrites nothing.
    assert sorted((seal_run.work / "seal" / "markers").iterdir()) == markers
    assert markers[0].read_bytes() == marker_before
    assert (seal_run.work / "packet" / "seal-receipt.json").read_bytes() == receipt_before


@pytest.mark.parametrize(
    "mutation",
    ["content_after_manifest", "corpus_member_writable", "manifest_writable", "two_attempts"],
)
def test_sealed_overwrite_violations_are_rejected(packet: Path, mutation: str) -> None:
    manifest = _manifest_of(packet)
    if mutation == "two_attempts":
        manifest["publication"]["packet_publication_attempts"] = 2
    _seal(packet, manifest)
    if mutation == "content_after_manifest":
        target = packet / CORPUS_DIRECTORY / HOLDOUT_CORPUS
        stamp = time.time() + 5
        os.utime(target, (stamp, stamp))
    elif mutation == "corpus_member_writable":
        (packet / CORPUS_DIRECTORY / SHADOW_CORPUS).chmod(0o644)
    elif mutation == "manifest_writable":
        (packet / MANIFEST_NAME).chmod(0o644)
    _assert_verifier_rejects(_run_verifier(packet), "overwrite")


def test_publication_flags_that_deny_no_overwrite_are_rejected(packet: Path) -> None:
    manifest = _manifest_of(packet)
    manifest["publication"]["no_overwrite"] = False
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "overwrite")


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

SCHEMA_MUTATIONS = (
    "unknown_member_nested",
    "unknown_member_root",
    "missing_member_nested",
    "duplicate_json_key",
    "free_text_value",
    "unknown_enum",
    "bool_where_int_required",
    "float_where_int_required",
    "sha256_not_lowercase_hex",
    "sha256_wrong_length",
    "timestamp_without_fraction",
    "timestamp_without_trailing_z",
    "non_canonical_encoding",
    "corpus_line_not_canonical",
    "stale_content_binding",
)


@pytest.mark.parametrize("mutation", SCHEMA_MUTATIONS)
def test_schema_violations_are_rejected(packet: Path, mutation: str) -> None:
    manifest = _manifest_of(packet)
    if mutation == "unknown_member_nested":
        manifest["publication"]["reseal_permitted"] = False
    elif mutation == "unknown_member_root":
        manifest["operator_note"] = "0" * 64
    elif mutation == "missing_member_nested":
        del manifest["bindings"]["retired_v3_documents"]["protocol-test"]
    elif mutation == "duplicate_json_key":
        raw = (packet / CORPUS_DIRECTORY / PRESEAL_RECEIPT).read_bytes()
        assert raw.startswith(b'{"active_segment')
        doctored = b'{"status":"pass",' + raw[1:]
        path = packet / CORPUS_DIRECTORY / PRESEAL_RECEIPT
        path.chmod(0o644)
        path.write_bytes(doctored)
        _rebind(packet, manifest, PRESEAL_RECEIPT)
    elif mutation == "free_text_value":
        manifest["bindings"]["unchanged_repair_design"]["sha256"] = "operator supplied note"
    elif mutation == "unknown_enum":
        manifest["receipt_kind"] = "packet-manifest-v2"
    elif mutation == "bool_where_int_required":
        manifest["aggregate_counts"]["selected_event_count"] = True
    elif mutation == "float_where_int_required":
        manifest["aggregate_counts"]["selected_event_count"] = float(
            manifest["aggregate_counts"]["selected_event_count"]
        )
    elif mutation == "sha256_not_lowercase_hex":
        binding = manifest["bindings"]["unchanged_repair_design"]
        binding["sha256"] = binding["sha256"].upper()
    elif mutation == "sha256_wrong_length":
        binding = manifest["bindings"]["unchanged_repair_design"]
        binding["sha256"] = binding["sha256"][:63]
    elif mutation == "timestamp_without_fraction":
        records = _read_corpus(packet, HOLDOUT_CORPUS)
        records[0]["created_at"] = records[0]["created_at"][:19] + "Z"
        _write_corpus(packet, HOLDOUT_CORPUS, records)
        _rebind(packet, manifest, HOLDOUT_CORPUS)
    elif mutation == "timestamp_without_trailing_z":
        records = _read_corpus(packet, HOLDOUT_CORPUS)
        records[0]["created_at"] = records[0]["created_at"][:-1]
        _write_corpus(packet, HOLDOUT_CORPUS, records)
        _rebind(packet, manifest, HOLDOUT_CORPUS)
    elif mutation == "non_canonical_encoding":
        receipt = _read_member(packet, PRESEAL_RECEIPT)
        path = packet / CORPUS_DIRECTORY / PRESEAL_RECEIPT
        path.chmod(0o644)
        path.write_bytes(
            json.dumps(receipt, sort_keys=True, separators=(", ", ": ")).encode("utf-8")
        )
        _rebind(packet, manifest, PRESEAL_RECEIPT)
    elif mutation == "corpus_line_not_canonical":
        records = _read_corpus(packet, SHADOW_CORPUS)
        lines = [_canonical_bytes(record) for record in records]
        lines[0] = json.dumps(records[0], sort_keys=True, separators=(", ", ":")).encode(
            "utf-8"
        )
        path = packet / CORPUS_DIRECTORY / SHADOW_CORPUS
        path.chmod(0o644)
        path.write_bytes(b"".join(line + b"\n" for line in lines))
        _rebind(packet, manifest, SHADOW_CORPUS)
    else:  # stale_content_binding
        records = _read_corpus(packet, HOLDOUT_CORPUS)
        _write_corpus(packet, HOLDOUT_CORPUS, records[:-1])
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "schema")


def test_receipt_timestamp_grammar_is_enforced_in_the_runtime_segment_class(
    packet: Path,
) -> None:
    """The same grammar, reached through the segment interval it bounds."""

    manifest = _manifest_of(packet)
    receipt = _read_member(packet, PRESEAL_RECEIPT)
    receipt["scheduled_at"] = receipt["scheduled_at"][:19] + "Z"
    _write_member(packet, PRESEAL_RECEIPT, receipt)
    _rebind(packet, manifest, PRESEAL_RECEIPT)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "runtime_segment")


# --------------------------------------------------------------------------
# production-shape
# --------------------------------------------------------------------------


def _seed_declared(packet: Path, name: str, table: str, value: int, manifest: dict) -> None:
    seed_manifest = _read_member(packet, SEED_MANIFEST)
    seed_manifest["states"][name]["table_row_counts"][table] = value
    _write_member(packet, SEED_MANIFEST, seed_manifest)
    _rebind(packet, manifest, SEED_MANIFEST)


@pytest.mark.parametrize("table", tuple(SEED_COLUMN_ALLOWLIST))
def test_each_required_seed_table_is_required(packet: Path, table: str) -> None:
    """Three copied tables; dropping any one is a production-shape failure."""

    manifest = _manifest_of(packet)
    name = SEED_STATES[0]
    connection = _open_seed(packet, name)
    try:
        connection.execute(f"DROP TABLE {table}")
        connection.commit()
    finally:
        connection.close()
    _rebind(packet, manifest, name)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "production_shape")


@pytest.mark.parametrize(
    ("table", "dropped"),
    [
        (table, column)
        for table, columns in SEED_COLUMN_ALLOWLIST.items()
        for column in columns
    ],
)
def test_each_allowlisted_seed_column_is_required(
    packet: Path, table: str, dropped: str
) -> None:
    """Every allowlisted column, one at a time, dropped from the projection."""

    kept = [column for column in SEED_COLUMN_ALLOWLIST[table] if column != dropped]
    manifest = _manifest_of(packet)
    name = SEED_STATES[1]
    connection = _open_seed(packet, name)
    try:
        connection.executescript(
            f"ALTER TABLE {table} RENAME TO packet_original;\n"
            f"CREATE TABLE {table} ({', '.join(kept)});\n"
            f"INSERT INTO {table} SELECT {', '.join(kept)} FROM packet_original;\n"
            "DROP TABLE packet_original;"
        )
        connection.commit()
    finally:
        connection.close()
    _rebind(packet, manifest, name)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "production_shape")


@pytest.mark.parametrize("table", tuple(SEED_COLUMN_ALLOWLIST))
def test_a_view_substituted_for_a_seed_table_is_rejected(
    packet: Path, table: str
) -> None:
    """A view answers the same SELECT and is still not the copied table."""

    manifest = _manifest_of(packet)
    name = SEED_STATES[2]
    connection = _open_seed(packet, name)
    try:
        connection.executescript(
            f"ALTER TABLE {table} RENAME TO packet_source;\n"
            f"CREATE VIEW {table} AS SELECT * FROM packet_source;"
        )
        connection.commit()
    finally:
        connection.close()
    _rebind(packet, manifest, name)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "production_shape")


@pytest.mark.parametrize("missing", sorted(REQUIRED_POLICY_KEYS))
def test_a_missing_required_policy_kind_is_rejected(packet: Path, missing: str) -> None:
    """The requirement is on policy *kinds*: ``project:p`` still is a project.

    Deleting only the bare key would leave the kind present through a concrete
    learned key, so every row of the kind has to go for the deletion to be a
    real violation.
    """

    manifest = _manifest_of(packet)
    name = SEED_STATES[3]
    connection = _open_seed(packet, name)
    try:
        before = connection.execute("SELECT COUNT(*) FROM retrieval_weights").fetchone()[0]
        connection.execute(
            "DELETE FROM retrieval_weights "
            "WHERE packet_scope_label = ? OR packet_scope_label LIKE ?",
            (missing, missing + ":%"),
        )
        removed = connection.execute("SELECT COUNT(*) FROM retrieval_weights").fetchone()[0]
        connection.commit()
    finally:
        connection.close()
    assert 0 < removed < before
    _seed_declared(packet, name, "retrieval_weights", removed, manifest)
    _rebind(packet, manifest, name)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "production_shape")


def test_a_duplicated_policy_key_is_rejected(packet: Path) -> None:
    """One policy key, one row: a duplicate is two policies for one scope."""

    manifest = _manifest_of(packet)
    name = SEED_STATES[0]
    connection = _open_seed(packet, name)
    try:
        connection.executescript(
            "ALTER TABLE retrieval_weights RENAME TO packet_original;\n"
            "CREATE TABLE retrieval_weights (packet_scope_label TEXT PRIMARY KEY, "
            "bm25_weight REAL NOT NULL, vector_weight REAL NOT NULL, "
            "graph_weight REAL NOT NULL);\n"
            "INSERT INTO retrieval_weights SELECT * FROM packet_original;\n"
            "DROP TABLE packet_original;\n"
            "INSERT INTO retrieval_weights VALUES ('default:duplicate', 1.0, 0.0, 0.0);"
        )
        total = connection.execute("SELECT COUNT(*) FROM retrieval_weights").fetchone()[0]
        connection.commit()
    finally:
        connection.close()
    _seed_declared(packet, name, "retrieval_weights", total, manifest)
    _rebind(packet, manifest, name)
    _seal(packet, manifest)
    # ``default`` is a policy key but never a prefix: ``default:x`` has no
    # mapping rule at all.
    _assert_verifier_rejects(_run_verifier(packet), "production_shape")


def test_an_empty_retrieval_weights_table_is_rejected(packet: Path) -> None:
    manifest = _manifest_of(packet)
    name = SEED_STATES[1]
    connection = _open_seed(packet, name)
    try:
        connection.execute("DELETE FROM retrieval_weights")
        connection.commit()
    finally:
        connection.close()
    _seed_declared(packet, name, "retrieval_weights", 0, manifest)
    _rebind(packet, manifest, name)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "production_shape")


@pytest.mark.parametrize("policy_key", ["project:", "session:", "scope:project:p", "unknown"])
def test_policy_keys_outside_the_exact_mapping_are_rejected(
    packet: Path, policy_key: str
) -> None:
    """``project:``/``session:`` with an empty suffix carry no mapping rule."""

    manifest = _manifest_of(packet)
    name = SEED_STATES[2]
    connection = _open_seed(packet, name)
    try:
        connection.execute(
            "UPDATE retrieval_weights SET packet_scope_label = ? "
            "WHERE packet_scope_label = 'default'",
            (policy_key,),
        )
        connection.commit()
    finally:
        connection.close()
    _rebind(packet, manifest, name)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "production_shape")


@pytest.mark.parametrize("mutation", ["unclassified_table", "declared_row_count", "forced_empty_not_empty"])
def test_seed_state_shape_mutations_are_rejected(packet: Path, mutation: str) -> None:
    manifest = _manifest_of(packet)
    name = SEED_STATES[3]
    connection = _open_seed(packet, name)
    try:
        if mutation == "unclassified_table":
            connection.execute("CREATE TABLE operator_scratch (value TEXT)")
        elif mutation == "declared_row_count":
            connection.execute(
                "DELETE FROM connections WHERE rowid = (SELECT MIN(rowid) FROM connections)"
            )
        else:
            connection.execute("INSERT INTO kv (packet_placeholder) VALUES ('x')")
        connection.commit()
    finally:
        connection.close()
    _rebind(packet, manifest, name)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "production_shape")


@pytest.mark.parametrize(
    "field",
    [
        "contract_id",
        "copied_table_column_allowlist",
        "forced_empty_state",
        "derived_index_prefixes",
        "policy_key_mapping_exact",
        "source_retrieval_policy_keys_required",
        "recall_events_at_seed",
        "unclassified_mutable_table_action",
    ],
)
def test_seed_manifest_contract_fields_are_rejected_when_altered(
    packet: Path, field: str
) -> None:
    manifest = _manifest_of(packet)
    seed_manifest = _read_member(packet, SEED_MANIFEST)
    value = seed_manifest[field]
    if type(value) is str:
        seed_manifest[field] = value + "-altered"
    elif type(value) is list:
        # A one-element list reads the same reversed, so extend it instead.
        seed_manifest[field] = (
            list(reversed(value)) if len(value) > 1 else [*value, "altered"]
        )
    else:
        altered = dict(value)
        altered[sorted(altered)[0]] = "altered"
        seed_manifest[field] = altered
    _write_member(packet, SEED_MANIFEST, seed_manifest)
    _rebind(packet, manifest, SEED_MANIFEST)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "production_shape")


def test_seed_state_names_are_bound_to_their_own_partition(packet: Path) -> None:
    """A holdout case may never point at a shadow seed state."""

    manifest = _manifest_of(packet)
    records = _read_corpus(packet, HOLDOUT_CORPUS)
    case = _first_case(records)
    alias_suffix = case["seed_state"].rsplit("-", 1)[1]
    case["seed_state"] = f"seed-state-{SHADOW_PARTITION}-{alias_suffix}"
    _write_corpus(packet, HOLDOUT_CORPUS, records)
    _rebind(packet, manifest, HOLDOUT_CORPUS)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "partition")


def test_an_undeclared_seed_state_name_is_rejected(packet: Path) -> None:
    manifest = _manifest_of(packet)
    records = _read_corpus(packet, HOLDOUT_CORPUS)
    _first_case(records)["seed_state"] = "seed-state-holdout-operator.sqlite3"
    _write_corpus(packet, HOLDOUT_CORPUS, records)
    _rebind(packet, manifest, HOLDOUT_CORPUS)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "production_shape")


# --------------------------------------------------------------------------
# privacy
# --------------------------------------------------------------------------


def _plant_needle(packet: Path, manifest: dict, planted: bytes) -> None:
    """Put one encoded needle where the packet still parses cleanly.

    A seed-state scope label is UTF-8 with a declared grammar and no digest
    constraint, so the planted bytes survive every structural check and can
    only be caught by the encoding-aware secret scan.
    """

    name = SEED_STATES[0]
    connection = _open_seed(packet, name)
    try:
        label = connection.execute(
            "SELECT packet_node_label FROM nodes ORDER BY packet_node_label LIMIT 1"
        ).fetchone()[0]
        connection.execute(
            "UPDATE nodes SET packet_scope_label = ? WHERE packet_node_label = ?",
            ("project:" + planted.decode("ascii"), label),
        )
        connection.commit()
    finally:
        connection.close()
    _rebind(packet, manifest, name)


@pytest.mark.parametrize("kind", PRIVACY_SECRET_KINDS)
@pytest.mark.parametrize("encoding", PRIVACY_ENCODINGS)
def test_planted_secret_material_is_caught_in_every_encoding(
    packet: Path, kind: str, encoding: str
) -> None:
    """Raw, hex, upper-hex, base64 and urlsafe-base64, for all four kinds."""

    secret = _Secret(PRIVACY_SECRETS[kind])
    planted = _oracle_encodings(bytes(secret))[encoding]
    manifest = _manifest_of(packet)
    # Bind the secret so the packet's own hash bindings make it a live needle,
    # exactly as a real leaked digest would be.
    manifest["bindings"]["unchanged_repair_design"]["sha256"] = bytes(secret).hex()
    _plant_needle(packet, manifest, planted)
    _seal(packet, manifest)
    _assert_verifier_rejects(
        _run_verifier(packet), "privacy", private_values=[secret]
    )


@pytest.mark.parametrize("kind", PRIVACY_SECRET_KINDS)
def test_the_privacy_scan_control_shows_the_needle_is_what_fails(
    packet: Path, kind: str
) -> None:
    """The same planted bytes, unbound: the packet must still verify.

    Without this the encoding matrix above could be passing for an unrelated
    reason, and would prove nothing about the needle expansion.
    """

    secret = _Secret(PRIVACY_SECRETS[kind])
    manifest = _manifest_of(packet)
    _plant_needle(packet, manifest, bytes(secret).hex().encode("ascii"))
    _seal(packet, manifest)
    _assert_verifier_pass(_run_verifier(packet))


@pytest.mark.parametrize(
    "plaintext",
    ["proj-alpha organic recall", "local-auto-0007", "/home/fixture/proj-alpha"],
)
def test_retained_raw_content_in_a_sealed_case_is_rejected(
    packet: Path, plaintext: str
) -> None:
    """A surrogate may keep length and whitespace, never a source character."""

    manifest = _manifest_of(packet)
    records = _read_corpus(packet, HOLDOUT_CORPUS)
    _first_case(records)["replay_input"]["query"] = plaintext
    _write_corpus(packet, HOLDOUT_CORPUS, records)
    _rebind(packet, manifest, HOLDOUT_CORPUS)
    _seal(packet, manifest)
    _assert_verifier_rejects(
        _run_verifier(packet), "privacy", private_values=[plaintext.encode("utf-8")]
    )


def test_a_digest_shaped_run_in_a_sealed_case_is_rejected(packet: Path) -> None:
    """Packet labels are thirty-two hex digits; sixty-four is secret-shaped."""

    manifest = _manifest_of(packet)
    records = _read_corpus(packet, HOLDOUT_CORPUS)
    token = _Secret(_identity_token(_ORACLE_KEY, _ORACLE_QUERY, _ORACLE_SCOPE))
    case = _first_case(records)
    case["replay_input"]["query"] = bytes(token).hex()
    _write_corpus(packet, HOLDOUT_CORPUS, records)
    _rebind(packet, manifest, HOLDOUT_CORPUS)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "privacy", private_values=[token])


def test_an_undeclared_corpus_member_is_rejected_as_a_persisted_map(
    packet: Path,
) -> None:
    """A tenth member is a persisted salt, map or fingerprint index."""

    manifest = _manifest_of(packet)
    (packet / CORPUS_DIRECTORY / "identity-map.json").write_bytes(b"{}\n")
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "privacy")


def test_deidentification_that_splits_a_repeated_family_is_rejected(
    packet: Path,
) -> None:
    """One family label, two surrogate identities: a false split."""

    manifest = _manifest_of(packet)
    records = _read_corpus(packet, HOLDOUT_CORPUS)
    family = next(members for members in _families(records).values() if len(members) > 1)
    member = family[0]
    original = member["replay_input"]["query"]
    assert family[1]["replay_input"]["query"] == original
    member["replay_input"]["query"] = "z" * len(original)
    _write_corpus(packet, HOLDOUT_CORPUS, records)
    _rebind(packet, manifest, HOLDOUT_CORPUS)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "privacy")


def test_deidentification_that_merges_two_families_is_rejected(packet: Path) -> None:
    """Two family labels, one surrogate identity: a false merge."""

    manifest = _manifest_of(packet)
    records = _read_corpus(packet, HOLDOUT_CORPUS)
    families = _families(records)
    source = next(iter(families))[0]
    keys = [key for key in families if key[0] == source][:2]
    assert len(keys) == 2
    left, right = families[keys[0]][0], families[keys[1]][0]
    assert left["replay_input"]["query"] != right["replay_input"]["query"]
    right["replay_input"]["query"] = left["replay_input"]["query"]
    right["replay_input"]["requested_scope"] = left["replay_input"]["requested_scope"]
    right["replay_input"]["resolved_scopes"] = list(left["replay_input"]["resolved_scopes"])
    _write_corpus(packet, HOLDOUT_CORPUS, records)
    _rebind(packet, manifest, HOLDOUT_CORPUS)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "privacy")


def test_deidentified_identity_equivalence_holds_on_the_conforming_packet(
    packet: Path,
) -> None:
    """The oracle's own bijection check, run over the unmutated corpus."""

    records = [
        record
        for name in (HOLDOUT_CORPUS, SHADOW_CORPUS)
        for record in _read_corpus(packet, name)
    ]
    forward: dict[tuple[str, str], tuple[str, str]] = {}
    reverse: dict[tuple[str, str, str], tuple[str, str]] = {}
    for record in records:
        if record["record_kind"] != "case" or record["family_label"] is None:
            continue
        source = record["source_label"]
        family = (source, record["family_label"])
        identity = (
            " ".join(record["replay_input"]["query"].split()),
            record["replay_input"]["requested_scope"],
        )
        assert forward.setdefault(family, identity) == identity
        assert reverse.setdefault((source, *identity), family) == family
    assert forward and reverse


# --------------------------------------------------------------------------
# partition
# --------------------------------------------------------------------------


def test_the_sealed_partition_covers_the_population_exactly_once(packet: Path) -> None:
    """``event_intersection_required: 0`` and a complete union."""

    holdout = _read_corpus(packet, HOLDOUT_CORPUS)
    shadow = _read_corpus(packet, SHADOW_CORPUS)
    holdout_labels = {record["event_label"] for record in holdout}
    shadow_labels = {record["event_label"] for record in shadow}
    assert not holdout_labels & shadow_labels
    assert len(holdout_labels) == len(holdout)
    assert len(shadow_labels) == len(shadow)
    assert all(record["partition"] == HOLDOUT_PARTITION for record in holdout)
    assert all(record["partition"] == SHADOW_PARTITION for record in shadow)
    counts = _manifest_of(packet)["aggregate_counts"]
    assert len(holdout_labels | shadow_labels) == counts["selected_event_count"]
    # One bucket per component, and zero family or workflow crossings.
    partition_of = {
        record["event_label"]: record["partition"] for record in (*holdout, *shadow)
    }
    for members in _oracle_components([*holdout, *shadow]).values():
        assert len({partition_of[label] for label in members}) == 1
    for kind in ("family_label", "workflow_label"):
        crossings: dict[tuple[str, str], set[str]] = {}
        for record in (*holdout, *shadow):
            if record["record_kind"] != "case" or record[kind] is None:
                continue
            crossings.setdefault((record["source_label"], record[kind]), set()).add(
                record["partition"]
            )
        assert crossings and all(len(value) == 1 for value in crossings.values())


@pytest.mark.parametrize("direction", ["holdout_to_shadow", "shadow_to_holdout"])
def test_a_reassigned_event_is_rejected(packet: Path, direction: str) -> None:
    """No reassignment, rebalancing, retry or move is admissible."""

    manifest = _manifest_of(packet)
    if direction == "holdout_to_shadow":
        source_name, target_name = HOLDOUT_CORPUS, SHADOW_CORPUS
        target_partition = SHADOW_PARTITION
    else:
        source_name, target_name = SHADOW_CORPUS, HOLDOUT_CORPUS
        target_partition = HOLDOUT_PARTITION
    source = _read_corpus(packet, source_name)
    target = _read_corpus(packet, target_name)
    moved = source.pop(0)
    moved["partition"] = target_partition
    if moved["record_kind"] == "case":
        moved["seed_state"] = (
            f"seed-state-{target_partition}-{moved['seed_state'].rsplit('-', 1)[1]}"
        )
    target.append(moved)
    target.sort(key=lambda record: _microseconds(record["created_at"]))
    _write_corpus(packet, source_name, source)
    _write_corpus(packet, target_name, target)
    _rebind(packet, manifest, source_name)
    _rebind(packet, manifest, target_name)
    manifest["aggregate_counts"] = _oracle_aggregates([*source, *target])
    receipt = _read_member(packet, PRESEAL_RECEIPT)
    receipt.update(manifest["aggregate_counts"])
    _write_member(packet, PRESEAL_RECEIPT, receipt)
    _rebind(packet, manifest, PRESEAL_RECEIPT)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "partition")


@pytest.mark.parametrize("edge", ["family_label", "workflow_label"])
def test_a_component_edge_crossing_the_split_is_rejected(
    packet: Path, edge: str
) -> None:
    """Zero automatic-family, organic-session and workflow crossings."""

    manifest = _manifest_of(packet)
    holdout = _read_corpus(packet, HOLDOUT_CORPUS)
    shadow = _read_corpus(packet, SHADOW_CORPUS)
    donor = _first_case(holdout, lambda record: record[edge] is not None)
    receiver = _first_case(shadow, lambda record: record[edge] is not None)
    receiver["source_label"] = donor["source_label"]
    receiver[edge] = donor[edge]
    _write_corpus(packet, SHADOW_CORPUS, shadow)
    _rebind(packet, manifest, SHADOW_CORPUS)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "partition")


def test_a_split_component_is_rejected(packet: Path) -> None:
    """A recomputed class must lie inside exactly one sealed component."""

    manifest = _manifest_of(packet)
    records = _read_corpus(packet, HOLDOUT_CORPUS)
    family = next(members for members in _families(records).values() if len(members) > 1)
    assert family[0]["component_label"] == family[1]["component_label"]
    family[0]["component_label"] = "0" * 32
    _write_corpus(packet, HOLDOUT_CORPUS, records)
    _rebind(packet, manifest, HOLDOUT_CORPUS)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "partition")


def test_records_out_of_created_at_order_are_rejected(packet: Path) -> None:
    """``selection.ordering`` is parsed microseconds, never source text."""

    manifest = _manifest_of(packet)
    records = _read_corpus(packet, HOLDOUT_CORPUS)
    records.reverse()
    assert _microseconds(records[0]["created_at"]) > _microseconds(
        records[-1]["created_at"]
    )
    _write_corpus(packet, HOLDOUT_CORPUS, records)
    _rebind(packet, manifest, HOLDOUT_CORPUS)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "partition")


@pytest.mark.parametrize(
    "field", ["feedback_applied", "case_outcome", "latency_ms", "index", "ordinal", "position"]
)
def test_an_outcome_or_ordinal_field_in_a_sealed_record_is_rejected(
    packet: Path, field: str
) -> None:
    """No sealed record may carry an outcome or an ordinal, under any name.

    The ambient envelope is a closed twelve-key set, so a planted outcome or
    ordinal is caught as a replayability violation before the partition audit
    is reached; the partition-side structural guard is falsified directly
    below, where the record schema cannot mask it.
    """

    manifest = _manifest_of(packet)
    records = _read_corpus(packet, SHADOW_CORPUS)
    _first_case(records)["replay_input"]["ambient_context"][field] = 1
    _write_corpus(packet, SHADOW_CORPUS, records)
    _rebind(packet, manifest, SHADOW_CORPUS)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "replayability")


@pytest.mark.parametrize(
    "field",
    [
        "index",
        "ordinal",
        "position",
        "feedback_applied",
        "case_outcome",
        "usefulness_score",
        "results",
        "latency_ms",
    ],
)
def test_the_partition_audit_refuses_an_outcome_or_ordinal_at_any_depth(
    verifier: Any, packet: Path, plan: dict, field: str
) -> None:
    """``outcome_field_invariant``/``input_order_invariant``, proved directly.

    Nothing the record schema admits can carry these names, so the partition
    class owns its own structural negative and is falsified here rather than
    through a packet whose schema check would fire first.
    """

    records = _read_corpus(packet, SHADOW_CORPUS)
    _first_case(records)["replay_input"]["ambient_context"] = {field: 1}
    with pytest.raises(verifier.VerificationError) as caught:
        verifier.audit_partition(records, plan["partition"]["golden_vectors"])
    assert caught.value.code == verifier.PARTITION


def _structural_profile(content: Path) -> list[tuple]:
    """Everything about a draft that outcome and input order may not move.

    Packet-local labels are minted fresh for every build, so they carry no
    information across two builds; what must be identical is the shape.
    """

    profile: list[tuple] = []
    for name in (HOLDOUT_CORPUS, SHADOW_CORPUS):
        for line in (content / name).read_bytes().decode("utf-8").splitlines():
            record = json.loads(line)
            replay = record.get("replay_input", {})
            profile.append(
                (
                    record["record_kind"],
                    record["created_at"],
                    record["partition"],
                    record.get("floor_counted"),
                    record.get("family_label") is not None,
                    record.get("workflow_label") is not None,
                    record.get("seed_state"),
                    replay.get("requested_scope", "").split(":", 1)[0],
                    tuple(sorted(replay.get("ambient_context", {}))),
                    replay.get("agent") is None,
                    replay.get("max_results"),
                )
            )
    return sorted(profile)


def test_outcome_fields_and_input_order_move_no_event(
    synthetic_base: _SyntheticPacket,
) -> None:
    """Two builds over perturbed outcomes and reversed rows agree exactly."""

    base = _structural_profile(synthetic_base.work / "draft" / "content")
    perturbed = _structural_profile(synthetic_base.work / "draft-invariance" / "content")
    assert base and base == perturbed
    assert len(base) == synthetic_base.summary["aggregate_counts"]["selected_event_count"]


# --------------------------------------------------------------------------
# replayability
# --------------------------------------------------------------------------

REPLAYABILITY_MUTATIONS = (
    "floor_counted_without_family_witness",
    "floor_counted_without_workflow_witness",
    "unknown_ambient_key",
    "malformed_requested_scope_project",
    "malformed_requested_scope_session",
    "malformed_requested_scope_unprefixed",
    "resolved_scope_plan_diverges",
    "agent_nullness_flip",
    "task_scalar_type_flip",
    "session_id_nullness_flip",
    "transport_workflow_nullness_disagrees",
    "forbidden_input_influences_the_predicate",
    "nonreplayable_event_carries_inputs",
    "max_results_not_positive_int",
)


@pytest.mark.parametrize("mutation", REPLAYABILITY_MUTATIONS)
def test_replayability_violations_are_rejected(packet: Path, mutation: str) -> None:
    manifest = _manifest_of(packet)
    name = SHADOW_CORPUS if "workflow_witness" in mutation else HOLDOUT_CORPUS
    records = _read_corpus(packet, name)
    if mutation == "floor_counted_without_family_witness":
        # A holdout event counted as an automatic floor witness while carrying
        # no family witness at all.  Nothing else moves: the case keeps its
        # component edges, so only the floor rule can object.
        case = _first_case(
            records,
            lambda record: record["floor_counted"]
            and record["family_label"] is None
            and record["replay_input"]["agent"] is not None,
        )
        case["replay_input"]["agent"] = None
        case["replay_input"]["ambient_context"]["agent"] = None
    elif mutation == "floor_counted_without_workflow_witness":
        case = _first_case(
            records,
            lambda record: not record["floor_counted"]
            and record["family_label"] is not None
            and record["workflow_label"] is not None,
        )
        case["floor_counted"] = True
        case["workflow_label"] = None
        case["replay_input"]["transport_session_id"] = None
        case["replay_input"]["ambient_context"].pop("transport_session_id", None)
    elif mutation == "unknown_ambient_key":
        _first_case(records)["replay_input"]["ambient_context"]["operator_note"] = "x"
    elif mutation == "malformed_requested_scope_project":
        _first_case(records)["replay_input"]["requested_scope"] = "project:"
    elif mutation == "malformed_requested_scope_session":
        _first_case(records)["replay_input"]["requested_scope"] = "session:"
    elif mutation == "malformed_requested_scope_unprefixed":
        _first_case(records)["replay_input"]["requested_scope"] = "operator"
    elif mutation == "resolved_scope_plan_diverges":
        case = _first_case(
            records,
            lambda record: record["replay_input"]["requested_scope"].startswith("project:"),
        )
        case["replay_input"]["resolved_scopes"] = ["global"]
    elif mutation == "agent_nullness_flip":
        case = _first_case(records, lambda record: record["replay_input"]["agent"] is not None)
        case["replay_input"]["agent"] = None
    elif mutation == "task_scalar_type_flip":
        case = _first_case(records, lambda record: record["replay_input"]["task"] is not None)
        case["replay_input"]["task"] = 7
    elif mutation == "session_id_nullness_flip":
        case = _first_case(
            records, lambda record: record["replay_input"]["session_id"] is not None
        )
        case["replay_input"]["session_id"] = None
    elif mutation == "transport_workflow_nullness_disagrees":
        case = _first_case(records, lambda record: record["workflow_label"] is not None)
        case["workflow_label"] = None
    elif mutation == "forbidden_input_influences_the_predicate":
        _first_case(records)["replay_input"]["ambient_context"]["feedback_applied"] = True
    elif mutation == "nonreplayable_event_carries_inputs":
        inventory = next(
            record for record in records if record["record_kind"] == "inventory"
        )
        inventory["replayable"] = True
    else:  # max_results_not_positive_int
        _first_case(records)["replay_input"]["max_results"] = 0
    _write_corpus(packet, name, records)
    _rebind(packet, manifest, name)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "replayability")


@pytest.mark.parametrize("field", ["floor_counted", "replay_input"])
def test_a_nonreplayable_event_may_never_become_a_floor_witness(
    verifier: Any, packet: Path, field: str
) -> None:
    """``selected_nonreplayable_floor_contribution: 0``, at both layers.

    The inventory schema has no such member, so a sealed packet is stopped by
    the schema class first; the replayability class keeps its own guard for
    exactly this, and it is falsified here where the schema cannot mask it.
    """

    inventory = next(
        record
        for record in _read_corpus(packet, SHADOW_CORPUS)
        if record["record_kind"] == "inventory"
    )
    inventory[field] = True if field == "floor_counted" else {}
    with pytest.raises(verifier.VerificationError) as caught:
        verifier.audit_inventory_record(inventory)
    assert caught.value.code == verifier.REPLAYABILITY


@pytest.mark.parametrize("field", ["floor_counted", "replay_input", "family_label"])
def test_an_inventory_record_carrying_case_members_is_rejected(
    packet: Path, field: str
) -> None:
    """The same claim, end to end: the inventory key set is exact."""

    manifest = _manifest_of(packet)
    records = _read_corpus(packet, SHADOW_CORPUS)
    inventory = next(
        record for record in records if record["record_kind"] == "inventory"
    )
    inventory[field] = True if field == "floor_counted" else None
    _write_corpus(packet, SHADOW_CORPUS, records)
    _rebind(packet, manifest, SHADOW_CORPUS)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "schema")


def test_every_floor_counted_case_carries_replayable_backing(packet: Path) -> None:
    """``selected_nonreplayable_floor_contribution: 0``, checked by the oracle."""

    records = [
        record
        for name in (HOLDOUT_CORPUS, SHADOW_CORPUS)
        for record in _read_corpus(packet, name)
    ]
    floor_counted = [
        record
        for record in records
        if record["record_kind"] == "case" and record["floor_counted"]
    ]
    assert floor_counted
    for record in floor_counted:
        assert set(record["replay_input"]) >= {"query", "requested_scope", "resolved_scopes"}
        if record["partition"] == HOLDOUT_PARTITION and record["replay_input"]["agent"] is None:
            assert record["family_label"] is not None
        if record["partition"] == SHADOW_PARTITION:
            assert record["workflow_label"] is not None
    for record in records:
        if record["record_kind"] == "inventory":
            assert record["replayable"] is False
            assert "floor_counted" not in record and "replay_input" not in record
        else:
            assert set(record["replay_input"]["ambient_context"]) <= set(AMBIENT_KEYS)


# --------------------------------------------------------------------------
# runtime-segment
# --------------------------------------------------------------------------


def test_one_snapshot_set_identity_holds_across_every_sealed_artifact(
    seal_run: _SealRun,
) -> None:
    """Marker, ready resolution, post-build observation, receipt, manifest.

    The five artifacts are produced at five different instants of one still
    open segment.  A packet built from a different retained snapshot set would
    disagree at exactly one of them, so the equality is what proves the whole
    ceremony read one segment.
    """

    work = seal_run.work
    markers = sorted((work / "seal" / "markers").iterdir())
    assert len(markers) == 1
    marker = json.loads(markers[0].read_bytes())
    observation_raw = (work / "seal" / "post-build-observation.json").read_bytes()
    observation = json.loads(observation_raw)
    resolution_raw = (work / "seal" / "ready" / "ready-resolution.json").read_bytes()
    receipt = json.loads((work / "packet" / "seal-receipt.json").read_bytes())
    manifest = json.loads((work / "packet" / MANIFEST_NAME).read_bytes())
    preseal = json.loads(
        (work / "packet" / CORPUS_DIRECTORY / PRESEAL_RECEIPT).read_bytes()
    )

    identity = marker["snapshot_set_sha256_and_bytes"]
    assert observation["snapshot_set_sha256_and_bytes"] == identity
    assert receipt["snapshot_set_sha256_and_bytes"] == identity
    assert preseal["snapshot_set_sha256_and_bytes"] == identity
    assert manifest["bindings"]["canonical_two_alias_snapshot_set_identity"] == identity

    # Each artifact is bound by its own bytes, so none can be substituted.
    assert receipt["seal_consumption_marker_sha256_and_bytes_or_null"] == _identity_of(
        markers[0].read_bytes()
    )
    assert receipt["slot_resolution_sha256_and_bytes_or_null"] == _identity_of(
        resolution_raw
    )
    assert receipt[
        "post_build_runtime_observation_sha256_and_bytes_or_null"
    ] == _identity_of(observation_raw)
    assert receipt["manifest_sha256_and_bytes_or_null"] == _identity_of(
        (work / "packet" / MANIFEST_NAME).read_bytes()
    )
    # The snapshot set is the digest of the two distinct alias snapshots.
    local = manifest["bindings"]["local_alias_immutable_snapshot"]
    alt = manifest["bindings"]["alt_alias_immutable_snapshot"]
    assert local["sha256"] != alt["sha256"]
    assert identity == _identity_of(_snapshot_set_preimage(local, alt))
    # One segment throughout, and it is the segment the observation attests.
    assert marker["segment_id"] == observation["segment_id"] == preseal["segment_id"]
    assert receipt["segment_attestation_sha256_and_bytes"] == observation[
        "segment_attestation_sha256_and_bytes"
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "equal_alias_snapshot_digests",
        "snapshot_set_not_the_pair_digest",
        "receipt_snapshot_set_disagrees",
        "receipt_alias_snapshot_disagrees",
        "slot_index_out_of_range",
        "segment_id_not_a_digest",
        "interval_not_monotone",
        "segment_binding_member_missing",
        "segment_binding_member_empty",
    ],
)
def test_runtime_segment_violations_are_rejected(packet: Path, mutation: str) -> None:
    manifest = _manifest_of(packet)
    bindings = manifest["bindings"]
    receipt = _read_member(packet, PRESEAL_RECEIPT)
    rewrite_receipt = False
    if mutation == "equal_alias_snapshot_digests":
        bindings["alt_alias_immutable_snapshot"] = dict(
            bindings["local_alias_immutable_snapshot"]
        )
    elif mutation == "snapshot_set_not_the_pair_digest":
        identity = bindings["canonical_two_alias_snapshot_set_identity"]
        identity["bytes"] = identity["bytes"] + 1
    elif mutation == "receipt_snapshot_set_disagrees":
        receipt["snapshot_set_sha256_and_bytes"] = {"sha256": "b" * 64, "bytes": 241}
        rewrite_receipt = True
    elif mutation == "receipt_alias_snapshot_disagrees":
        receipt["aliased_source_snapshot_sha256_and_bytes"]["alt"] = dict(
            receipt["aliased_source_snapshot_sha256_and_bytes"]["local"]
        )
        rewrite_receipt = True
    elif mutation == "slot_index_out_of_range":
        receipt["slot_index"] = 29
        rewrite_receipt = True
    elif mutation == "segment_id_not_a_digest":
        receipt["segment_id"] = receipt["segment_id"].upper()
        rewrite_receipt = True
    elif mutation == "interval_not_monotone":
        receipt["grace_deadline_at"] = receipt["active_segment_lower_bound_exclusive_at"]
        rewrite_receipt = True
    elif mutation == "segment_binding_member_missing":
        del bindings["active_runtime_segment_attestation_and_complete_tuple"][
            "complete_unaliased_service_tuple"
        ]
    else:  # segment_binding_member_empty
        bindings["active_runtime_segment_attestation_and_complete_tuple"][
            "segment_attestation"
        ]["bytes"] = 0
    if rewrite_receipt:
        _write_member(packet, PRESEAL_RECEIPT, receipt)
        _rebind(packet, manifest, PRESEAL_RECEIPT)
    _seal(packet, manifest)
    expected = "schema" if mutation == "segment_binding_member_missing" else "runtime_segment"
    _assert_verifier_rejects(_run_verifier(packet), expected)


def test_a_segment_changed_before_the_pre_manifest_observation_is_rejected(
    verifier: Any, packet: Path, plan: dict
) -> None:
    """The post-build, pre-manifest observation is a sealed-state precondition."""

    manifest = _manifest_of(packet)
    receipt = _read_member(packet, PRESEAL_RECEIPT)
    doctored = json.loads(json.dumps(plan))
    doctored["publication_and_ordering"]["sealed_state"][
        "active_segment_unchanged_at_post_build_pre_manifest_observation"
    ] = False
    with pytest.raises(verifier.VerificationError) as caught:
        verifier.audit_manifest_order(manifest, receipt, doctored)
    assert caught.value.code == verifier.MANIFEST_ORDER


def test_the_seal_launcher_declares_a_runtime_change_terminal(sealer: Any) -> None:
    """A segment that moves mid-ceremony has exactly one terminal outcome."""

    assert "runtime-change" in sealer.STAGE_REASON_MATRIX["post-build-observation"]
    assert "post-build-observation" in sealer.POST_GATE_STAGES
    assert sealer.STATUS_TERMINAL == SEAL_STATUS_TERMINAL
    for stage, reasons in sealer.STAGE_REASON_MATRIX.items():
        assert stage in sealer.FAILURE_STAGE_ENUM
        for reason in reasons:
            assert reason in sealer.FAILURE_REASON_ENUM


def _restarted_service_tuple(sealer: Any) -> Any:
    """One well-formed service tuple describing a *different* runtime.

    The frozen control watermark pins the segment, so bootstrapping a second
    synthetic ledger observes the very same tuple and cannot model a change at
    all.  A real one is modelled the way the protocol names it -- a service
    restarts, so it comes back under a new boot identity -- and the
    replacement is built through the runtime's own canonical factory rather
    than forged, so what the ceremony observes is a genuine ``ServiceTuple``
    that simply describes a runtime which moved.  ``sealer.runtime`` is the
    one shared runtime module object; a second importlib copy would break
    every ``type(x) is ...`` check downstream.
    """

    runtime = sealer.runtime
    observed = json.loads(
        runtime.load_frozen_contract().initial_services.raw.decode("utf-8")
    )
    digest = observed[0]["boot_identity_sha256"]
    observed[0]["boot_identity_sha256"] = (
        "0" if digest[0] != "0" else "1"
    ) + digest[1:]
    return runtime.canonical_service_tuple(observed)


def _run_with_failing_post_build_observation(
    sealer: Any,
    monkeypatch: pytest.MonkeyPatch,
    work: Path,
    second: Callable[[Any, Any], tuple[Any, Any]],
) -> dict:
    """Drive a whole synthetic ceremony whose post-build observation fails.

    ``observe_active_state`` is injected into the ceremony, and a run makes
    exactly two observations -- the marker-create check and the post-build
    one -- so perturbing every call after the first perturbs precisely the
    observation that happens *after* authority has been consumed.
    """

    original = sealer.SyntheticRuntimeObserver.observe
    calls = {"count": 0}

    def observe(self: Any) -> tuple[Any, Any]:
        calls["count"] += 1
        services, binding = original(self)
        if calls["count"] == 1:
            return services, binding
        return second(services, binding)

    monkeypatch.setattr(sealer.SyntheticRuntimeObserver, "observe", observe)
    status, summary = sealer.run_self_check(work)
    # Exactly the post-authority observation was perturbed: the ceremony got
    # past marker creation and really did observe a second time.
    assert calls["count"] == 2
    assert status == SEAL_EXIT_TERMINAL
    return summary


def test_the_post_build_observation_harness_seals_when_nothing_is_perturbed(
    sealer: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The control for the two rejections below.

    Same in-process driver, same patched ``observe``, same work directory
    layout -- only the second observation is left alone.  It seals.  So when
    the two tests below terminalize, the perturbation is what caused it and
    not the harness.
    """

    original = sealer.SyntheticRuntimeObserver.observe
    calls = {"count": 0}

    def observe(self: Any) -> tuple[Any, Any]:
        calls["count"] += 1
        return original(self)

    monkeypatch.setattr(sealer.SyntheticRuntimeObserver, "observe", observe)
    status, summary = sealer.run_self_check(_private_directory(tmp_path / "work"))
    assert calls["count"] == 2
    assert status == SEAL_EXIT_SEALED
    assert summary["status"] == "sealed"
    assert summary["failure_stage"] is None and summary["failure_reason"] is None
    assert summary["real_packet_sealed"] is False
    # ``sealed_conditional_fields``: a sealed status binds every artifact.
    receipt = summary["seal_terminal_receipt"]
    assert receipt["manifest_sha256_and_bytes_or_null"] is not None
    assert receipt["post_build_runtime_observation_sha256_and_bytes_or_null"] is not None


@pytest.mark.parametrize("reason", ["runtime-change", "observation-unavailable"])
def test_a_post_authority_observation_fault_is_terminal_not_a_return_to_accrual(
    sealer: Any,
    plan: dict,
    seal_run: _SealRun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reason: str,
) -> None:
    """The second observation is a real gate, not a declared constant.

    ``STAGE_REASON_MATRIX`` above only proves the launcher *says* a moved
    runtime is terminal.  These two drive the whole seven-step ceremony to the
    point where authority is already spent and the content is already built,
    then move the runtime under it -- once by restarting a service, once by
    making the runtime unobservable at all.  Both must terminalize with a
    receipt rather than a traceback, and neither may publish or fall back to
    accrual.
    """

    if reason == "runtime-change":
        changed = _restarted_service_tuple(sealer)
        second: Callable[[Any, Any], tuple[Any, Any]] = (
            lambda _services, binding: (changed, binding)
        )
    else:

        def second(_services: Any, _binding: Any) -> tuple[Any, Any]:
            raise OSError("the runtime is no longer observable")

    work = _private_directory(tmp_path / "work")
    summary = _run_with_failing_post_build_observation(
        sealer, monkeypatch, work, second
    )

    # The unperturbed ceremony seals this very fixture, so neither rejection
    # here is vacuous.
    assert seal_run.summary["status"] == "sealed"

    assert summary["status"] == SEAL_STATUS_TERMINAL
    assert summary["failure_stage"] == "post-build-observation"
    assert summary["failure_reason"] == reason
    assert summary["real_packet_sealed"] is False
    # All seven handoff steps ran: this is a fault after authority, not before.
    assert summary["handoff_steps_completed"] == len(sealer.ORDERING_EXACTLY)
    # Never a return to accrual, and never a partial publication.
    assert not (work / "packet" / MANIFEST_NAME).exists()

    receipt = summary["seal_terminal_receipt"]
    # The field set comes from the frozen plan, never from the launcher.
    assert set(receipt) == set(
        plan["operational_receipt_schemas"]["seal_terminal_receipt"]["fields_exactly"]
    )
    assert receipt["receipt_kind"] == "seal-terminal"
    assert receipt["status"] == SEAL_STATUS_TERMINAL
    assert receipt["failure_stage_or_null"] == "post-build-observation"
    assert receipt["failure_reason_or_null"] == reason
    # ``sealed_conditional_fields``: a terminal receipt claims no sealed
    # artifact, and the observation that failed is not recorded as having run.
    assert receipt["manifest_sha256_and_bytes_or_null"] is None
    assert receipt["post_build_runtime_observation_sha256_and_bytes_or_null"] is None
    # Authority was consumed at marker creation, so the marker stays bound,
    # and the ready bundle was already visible, so slot resolution is not null.
    assert receipt["seal_consumption_marker_sha256_and_bytes_or_null"] is not None
    assert receipt["slot_resolution_sha256_and_bytes_or_null"] is not None


# --------------------------------------------------------------------------
# reader-authority
# --------------------------------------------------------------------------


def test_exactly_two_v4_readers_stand_at_zero_semantic_reads(
    packet: Path, seal_run: _SealRun
) -> None:
    reads = _manifest_of(packet)["semantic_reads"]
    assert set(reads) == set(SEMANTIC_READER_IDS)
    assert all(value == 0 and type(value) is int for value in reads.values())
    assert _read_member(packet, PRESEAL_RECEIPT)["semantic_reads"] == reads
    assert seal_run.summary["semantic_reads"] == {
        reader: 0 for reader in SEMANTIC_READER_IDS
    }


@pytest.mark.parametrize("reader", SEMANTIC_READER_IDS)
def test_a_nonzero_semantic_read_entry_is_rejected(packet: Path, reader: str) -> None:
    manifest = _manifest_of(packet)
    manifest["semantic_reads"][reader] = 1
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "reader_authority")


@pytest.mark.parametrize(
    "reads",
    [
        {"confirmatory-shadow-v4-eval": 0, "confirmatory-holdout-v4-eval-alias": 0},
        {"confirmatory-holdout-v4-eval": 0, "confirmatory-shadow-v4": 0},
        {
            "confirmatory-shadow-v4-eval": 0,
            "confirmatory-holdout-v4-eval": 0,
            "confirmatory-extra-v4-eval": 0,
        },
        {"confirmatory-shadow-v4-eval": 0},
        {
            "confirmatory-shadow-v3-eval": 0,
            "confirmatory-holdout-v3-eval": 0,
        },
    ],
    ids=["aliased", "renamed", "third_reader", "missing_reader", "retired_v3_pair"],
)
def test_only_the_two_declared_reader_identities_validate(
    packet: Path, reads: dict
) -> None:
    manifest = _manifest_of(packet)
    manifest["semantic_reads"] = reads
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "reader_authority")


@pytest.mark.parametrize("retired", RETIRED_READER_IDS)
def test_a_retired_reader_identity_offered_as_v4_evidence_is_rejected(
    packet: Path, retired: str
) -> None:
    """A v2/v3 identity is not a v4 reader, wherever it is offered."""

    assert retired not in SEMANTIC_READER_IDS
    manifest = _manifest_of(packet)
    records = _read_corpus(packet, HOLDOUT_CORPUS)
    case = _first_case(records, lambda record: record["replay_input"]["agent"] is not None)
    case["replay_input"]["agent"] = retired
    case["replay_input"]["ambient_context"]["agent"] = retired
    _write_corpus(packet, HOLDOUT_CORPUS, records)
    _rebind(packet, manifest, HOLDOUT_CORPUS)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "reader_authority")


@pytest.mark.parametrize("prefix", RETIRED_DOMAIN_PREFIXES)
def test_a_retired_domain_prefix_in_the_sealed_corpus_is_rejected(
    packet: Path, prefix: str
) -> None:
    manifest = _manifest_of(packet)
    records = _read_corpus(packet, HOLDOUT_CORPUS)
    case = _first_case(records, lambda record: record["replay_input"]["task"] is not None)
    value = prefix + "partition/v1"
    case["replay_input"]["task"] = value
    case["replay_input"]["ambient_context"]["task"] = value
    _write_corpus(packet, HOLDOUT_CORPUS, records)
    _rebind(packet, manifest, HOLDOUT_CORPUS)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "reader_authority")


@pytest.mark.parametrize(
    "key", ["reader_alias_of", "reader_aliases", "reader_delegation", "wildcard_reader", "reader_fallback"]
)
def test_an_alias_key_planted_in_the_seed_manifest_is_rejected(
    packet: Path, key: str
) -> None:
    manifest = _manifest_of(packet)
    seed_manifest = _read_member(packet, SEED_MANIFEST)
    seed_manifest["states"][SEED_STATES[0]][key] = "confirmatory-holdout-v4-eval"
    _write_member(packet, SEED_MANIFEST, seed_manifest)
    _rebind(packet, manifest, SEED_MANIFEST)
    _seal(packet, manifest)
    # An unknown member inside the seed manifest is a production-shape failure
    # before the alias scan can reach it; the alias vocabulary itself is
    # falsified directly below.
    _assert_verifier_rejects(_run_verifier(packet), "production_shape")


@pytest.mark.parametrize(
    "key",
    [
        "reader_alias_of",
        "reader_aliases",
        "delegated_reader",
        "wildcard_reader",
        "reader_fallback",
        "inherit_reader",
        "any_reader",
        "additional_reader",
    ],
)
def test_the_reader_alias_vocabulary_is_closed(verifier: Any, key: str) -> None:
    """No alias, delegation, wildcard or fallback key may appear anywhere."""

    with pytest.raises(verifier.VerificationError) as caught:
        verifier.audit_reader_authority([{key: "confirmatory-holdout-v4-eval"}])
    assert caught.value.code == verifier.READER_AUTHORITY
    # And the two declared identities alone survive the same predicate.
    verifier.audit_reader_authority([{reader: 0 for reader in SEMANTIC_READER_IDS}])


# --------------------------------------------------------------------------
# manifest-order
# --------------------------------------------------------------------------


def test_the_manifest_is_the_last_byte_written(packet: Path) -> None:
    """Content before manifest, manifest last, on the conforming packet."""

    manifest_mark = (packet / MANIFEST_NAME).stat()
    assert manifest_mark.st_nlink == 1
    for member in (packet / CORPUS_DIRECTORY).iterdir():
        mark = member.stat()
        assert mark.st_nlink == 1
        assert mark.st_mtime_ns <= manifest_mark.st_mtime_ns
        assert mark.st_ctime_ns <= manifest_mark.st_ctime_ns
    corpus_mark = (packet / CORPUS_DIRECTORY).stat()
    assert corpus_mark.st_mtime_ns <= manifest_mark.st_mtime_ns
    assert corpus_mark.st_ctime_ns <= manifest_mark.st_ctime_ns
    publication = _manifest_of(packet)["publication"]
    assert publication == {
        "packet_publication_attempts": 1,
        "no_overwrite": True,
        "content_before_manifest": True,
        "manifest_last": True,
        "canonical_manifest_present": True,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "frozen_false",
        "content_before_manifest_false",
        "manifest_last_false",
        "canonical_manifest_absent",
        "keyed_preseal_mismatches_nonzero",
        "keyed_preseal_status_not_pass",
    ],
)
def test_manifest_order_violations_are_rejected(packet: Path, mutation: str) -> None:
    manifest = _manifest_of(packet)
    receipt = _read_member(packet, PRESEAL_RECEIPT)
    rewrite_receipt = False
    if mutation == "frozen_false":
        manifest["frozen"] = False
    elif mutation == "content_before_manifest_false":
        manifest["publication"]["content_before_manifest"] = False
    elif mutation == "manifest_last_false":
        manifest["publication"]["manifest_last"] = False
    elif mutation == "canonical_manifest_absent":
        manifest["publication"]["canonical_manifest_present"] = False
    elif mutation == "keyed_preseal_mismatches_nonzero":
        receipt["mismatch_count"] = 1
        rewrite_receipt = True
    else:  # keyed_preseal_status_not_pass
        receipt["status"] = "fail"
        rewrite_receipt = True
    if rewrite_receipt:
        _write_member(packet, PRESEAL_RECEIPT, receipt)
        _rebind(packet, manifest, PRESEAL_RECEIPT)
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "manifest_order")


def test_content_installed_after_the_manifest_is_rejected(packet: Path) -> None:
    """The sealed-state marker cannot predate a byte it is supposed to seal."""

    _seal(packet, _manifest_of(packet))
    _assert_verifier_pass(_run_verifier(packet))
    target = packet / CORPUS_DIRECTORY / SEED_MANIFEST
    target.chmod(0o644)
    target.write_bytes(target.read_bytes())
    target.chmod(0o444)
    _assert_verifier_rejects(_run_verifier(packet), "overwrite")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latched_slot_status", "pending"),
        ("latched_slot_status", "below-floor"),
        ("first_ready_predecessor_chain_valid", False),
        ("active_segment_open_at_atomic_handoff", False),
        ("active_segment_unchanged_at_post_build_pre_manifest_observation", False),
        ("keyed_preseal_status", "fail"),
        ("keyed_preseal_mismatches", 1),
        ("consumption_record_location", "the immutable packet manifest"),
        ("frozen", False),
    ],
)
def test_sealed_state_preconditions_are_rejected_when_denied(
    verifier: Any, packet: Path, plan: dict, field: str, value: Any
) -> None:
    """The sealed-state clauses the manifest cannot carry, falsified directly.

    ``latched_slot_status`` and its siblings live in the frozen plan, which is
    hash-pinned and unmutable from a packet, so the only honest way to falsify
    them is to hand the auditor a doctored plan.
    """

    manifest = _manifest_of(packet)
    receipt = _read_member(packet, PRESEAL_RECEIPT)
    doctored = json.loads(json.dumps(plan))
    doctored["publication_and_ordering"]["sealed_state"][field] = value
    with pytest.raises(verifier.VerificationError) as caught:
        verifier.audit_manifest_order(manifest, receipt, doctored)
    assert caught.value.code == verifier.MANIFEST_ORDER
    # The unaltered plan accepts the same manifest and receipt.
    verifier.audit_manifest_order(manifest, receipt, plan)


@pytest.mark.parametrize(
    "key",
    [
        "consumed_at",
        "consumption",
        "consumption_record",
        "consumption_records",
        "seal_consumption_marker_sha256_and_bytes",
        "launch_consumption_sha256_and_bytes_or_null",
        "latched_slot_status",
        "keyed_preseal_mismatches",
    ],
)
def test_a_consumption_record_written_into_the_manifest_is_rejected(
    verifier: Any, packet: Path, plan: dict, key: str
) -> None:
    """``consumption_record_location``: separate receipts only, never here."""

    manifest = _manifest_of(packet)
    receipt = _read_member(packet, PRESEAL_RECEIPT)
    manifest["publication"][key] = "2026-08-18T00:00:00.000000Z"
    with pytest.raises(verifier.VerificationError) as caught:
        verifier.audit_manifest_order(manifest, receipt, plan)
    assert caught.value.code == verifier.MANIFEST_ORDER


@pytest.mark.parametrize(
    "key",
    ["consumed_at", "consumption_record", "latched_slot_status", "keyed_preseal_mismatches"],
)
def test_a_consumption_key_in_a_sealed_manifest_is_rejected_end_to_end(
    packet: Path, key: str
) -> None:
    """The same key, through the sealed packet: the schema class stops it first."""

    manifest = _manifest_of(packet)
    manifest["publication"][key] = "2026-08-18T00:00:00.000000Z"
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "schema")


def test_the_sealed_seal_receipt_carries_the_consumption_record_instead(
    seal_run: _SealRun,
) -> None:
    """Where the consumption record does live: a separate hash-bound receipt."""

    manifest = json.loads((seal_run.work / "packet" / MANIFEST_NAME).read_bytes())
    receipt = json.loads((seal_run.work / "packet" / "seal-receipt.json").read_bytes())
    assert receipt["receipt_kind"] == "seal-terminal"
    assert receipt["status"] == "sealed"
    assert receipt["failure_stage_or_null"] is None
    assert receipt["failure_reason_or_null"] is None
    assert receipt["seal_consumption_marker_sha256_and_bytes_or_null"] is not None
    # And not one consumption key leaked into the immutable manifest.
    forbidden = {
        "consumed_at",
        "consumption",
        "consumption_record",
        "consumption_records",
        "seal_consumption_marker_sha256_and_bytes",
        "seal_consumption_marker_sha256_and_bytes_or_null",
        "launch_consumption_sha256_and_bytes_or_null",
        "latched_slot_status",
        "first_ready_predecessor_chain_valid",
        "active_segment_open_at_atomic_handoff",
        "active_segment_unchanged_at_post_build_pre_manifest_observation",
        "keyed_preseal_status",
        "keyed_preseal_mismatches",
    }
    stack: list[Any] = [manifest]
    while stack:
        node = stack.pop()
        if type(node) is dict:
            assert not set(node) & forbidden
            stack.extend(node.values())
        elif type(node) is list:
            stack.extend(node)


# --------------------------------------------------------------------------
# No real packet was sealed
# --------------------------------------------------------------------------


def test_this_subtree_owns_exactly_the_builder_and_the_verifier() -> None:
    """``recipe/`` is this subtree's ownership boundary, and nothing more.

    This is deliberately *not* a snapshot of the namespace root.  ``corpus/``,
    ``manifest.json``, ``seal-receipt.json``, ``segments/`` and ``probes/`` are
    exactly what the downstream accrue-and-seal sibling legitimately creates,
    so asserting their absence would turn red the moment that sibling lands.
    """

    assert set(os.listdir(RECIPE_ROOT)) == RECIPE_OWNED_ENTRIES
    for name in sorted(RECIPE_OWNED_ENTRIES):
        assert (RECIPE_ROOT / name).is_file()
        assert not (RECIPE_ROOT / name).is_symlink()


@pytest.mark.parametrize(
    "relative", ["", "scratch", "recipe/scratch", "corpus", "segments/scratch"]
)
def test_the_builder_refuses_a_namespace_resident_output_root(
    tmp_path: Path, relative: str
) -> None:
    """Only ``freeze --publish-dir`` may ever write inside the namespace."""

    target = NAMESPACE_ROOT / relative if relative else NAMESPACE_ROOT
    before = sorted(os.listdir(NAMESPACE_ROOT))
    check = _run_builder("self-check", "--work-dir", os.fspath(target))
    assert check.returncode == BUILD_EXIT_NAMESPACE_REFUSED
    assert _error_code(check) == NAMESPACE_REFUSAL_CODE
    draft = _run_builder(
        "draft",
        "--draft-dir",
        os.fspath(target),
        "--handoff",
        os.fspath(tmp_path / "absent-handoff.json"),
    )
    assert draft.returncode == BUILD_EXIT_NAMESPACE_REFUSED
    assert _error_code(draft) == NAMESPACE_REFUSAL_CODE
    # The refusal happens before a single byte is written.
    assert sorted(os.listdir(NAMESPACE_ROOT)) == before


@pytest.mark.parametrize("relative", ["", "scratch", "recipe/scratch", "probes/scratch"])
def test_the_seal_launcher_refuses_a_namespace_resident_work_root(
    relative: str,
) -> None:
    target = NAMESPACE_ROOT / relative if relative else NAMESPACE_ROOT
    before = sorted(os.listdir(NAMESPACE_ROOT))
    result = _run_sealer("self-check", "--work-dir", os.fspath(target))
    assert result.returncode == SEAL_EXIT_NAMESPACE_REFUSED
    payload = json.loads(result.stdout)
    assert payload["code"] == NAMESPACE_REFUSAL_CODE
    assert payload["real_packet_sealed"] is False
    assert sorted(os.listdir(NAMESPACE_ROOT)) == before


def test_a_symlinked_work_root_cannot_smuggle_a_namespace_write(
    tmp_path: Path,
) -> None:
    """Both the lexical and the link-resolved form are refused."""

    link = tmp_path / "work"
    link.symlink_to(NAMESPACE_ROOT / "scratch")
    before = sorted(os.listdir(NAMESPACE_ROOT))
    build_result = _run_builder("self-check", "--work-dir", os.fspath(link))
    assert build_result.returncode == BUILD_EXIT_NAMESPACE_REFUSED
    assert _error_code(build_result) == NAMESPACE_REFUSAL_CODE
    seal_result = _run_sealer("self-check", "--work-dir", os.fspath(link))
    assert seal_result.returncode == SEAL_EXIT_NAMESPACE_REFUSED
    assert json.loads(seal_result.stdout)["code"] == NAMESPACE_REFUSAL_CODE
    assert sorted(os.listdir(NAMESPACE_ROOT)) == before


def test_the_self_checks_leave_the_namespace_and_scripts_untouched(
    synthetic_base: _SyntheticPacket, seal_run: _SealRun
) -> None:
    """Both ceremonies stayed inside their own work directories.

    The comparison is a delta taken now against the directory listing, so it
    stays green whatever a sibling has legitimately published; what it proves
    is that re-running either self-check adds nothing here.
    """

    namespace_before = sorted(os.listdir(NAMESPACE_ROOT))
    recipe_before = sorted(os.listdir(RECIPE_ROOT))
    scripts_before = sorted(os.listdir(ROOT / "scripts"))
    work = _private_directory(seal_run.work.parent / "delta-work")
    result = _run_builder("self-check", "--work-dir", os.fspath(work))
    assert result.returncode == BUILD_EXIT_OK
    assert json.loads(result.stdout)["real_packet_sealed"] is False
    assert sorted(os.listdir(NAMESPACE_ROOT)) == namespace_before
    assert sorted(os.listdir(RECIPE_ROOT)) == recipe_before
    assert sorted(os.listdir(ROOT / "scripts")) == scripts_before
    # Every artefact of both ceremonies lives under its own work directory.
    assert (synthetic_base.packet / MANIFEST_NAME).is_file()
    assert synthetic_base.work in synthetic_base.packet.parents
    assert (seal_run.work / "packet" / "seal-receipt.json").is_file()


def test_neither_self_check_reports_a_real_packet(
    synthetic_base: _SyntheticPacket, seal_run: _SealRun
) -> None:
    for summary in (synthetic_base.summary, seal_run.summary):
        assert summary["real_packet_sealed"] is False
        assert summary["synthetic_fixture"] is True
        assert summary["namespace"] == NAMESPACE
        assert summary["schema_version"] == SCHEMA_VERSION
    assert _manifest_of(synthetic_base.packet)["synthetic_fixture"] is True
    assert seal_run.summary["seal_terminal_receipt"]["namespace"] == NAMESPACE


# --------------------------------------------------------------------------
# Production-seeded coverage
# --------------------------------------------------------------------------

# The projection contract, restated here against the real production column
# names so a silent rename in either schema is caught.
PRODUCTION_PROJECTION_COLUMNS = {
    "nodes": ("id", "level", "content", "scope", "created_at"),
    "connections": ("source_id", "target_id", "type", "weight"),
    "retrieval_weights": ("scope", "bm25", "vector", "graph"),
}


def _production_store(path: Path) -> None:
    """A real ``MemoryStore``, populated only through its public API."""

    with MemoryStore(path) as store:
        left = store.append_trace(
            "alpha production content", {"scope": "project:seed", "agent": "operator"}
        )
        right = store.append_trace("beta production content", {"scope": "project:seed"})
        third = store.append_trace("gamma production content", {"scope": "global"})
        store.create_connection(left.id, right.id, "related", weight=0.75)
        store.create_connection(right.id, third.id, "supersedes", weight=0.5)
        store.set_retrieval_weights("project:learned", bm25=0.6, vector=0.3, graph=0.1)


def _readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


@pytest.fixture
def production_snapshot(tmp_path: Path) -> Path:
    path = tmp_path / "production.sqlite3"
    _production_store(path)
    return path


def test_production_schema_carries_every_projected_column(
    production_snapshot: Path,
) -> None:
    """The allowlist is a projection of columns that really exist."""

    connection = _readonly(production_snapshot)
    try:
        for table, columns in PRODUCTION_PROJECTION_COLUMNS.items():
            observed = {
                row[1]
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            assert set(columns) <= observed
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type IN ('table','view')"
            ).fetchall()
            if not row[0].startswith("sqlite_")
        }
    finally:
        connection.close()
    assert set(PRODUCTION_PROJECTION_COLUMNS) <= names
    assert set(FORCED_EMPTY_TABLES) <= names
    assert any(name.startswith("nodes_fts") for name in names)


def test_the_builder_projects_a_real_production_snapshot_the_verifier_accepts(
    builder: Any, verifier: Any, production_snapshot: Path, tmp_path: Path
) -> None:
    """Genuine production schema in, conforming packet seed state out."""

    connection = _readonly(production_snapshot)
    secrets_ = builder.Secrets()
    try:
        classified = builder.classify_source_tables(connection)
        assert {classified[table] for table in PRODUCTION_PROJECTION_COLUMNS} == {"copied"}
        assert {classified[table] for table in FORCED_EMPTY_TABLES} == {"forced-empty"}
        assert all(
            classified[name] == "derived-index"
            for name in classified
            if name.startswith("nodes_fts")
        )
        target = tmp_path / "seed-state.sqlite3"
        counts = builder.project_seed_state(
            connection,
            target=target,
            deidentifier=builder.Deidentifier(secrets_),
            mint=builder.LabelMint(secrets_),
        )
    finally:
        secrets_.destroy()
        connection.close()

    assert counts["nodes"] == 3
    assert counts["connections"] == 2
    assert counts["retrieval_weights"] == len(DEFAULT_RETRIEVAL_WEIGHTS) + 1
    assert all(counts[table] == 0 for table in FORCED_EMPTY_TABLES)

    # The verifier's production-shape audit accepts the real projection ...
    verifier.audit_seed_state(target, counts)

    # ... and the oracle independently confirms the projection is allowlisted,
    # de-identified and self-contained.
    projected = _readonly(target)
    try:
        for table, columns in SEED_COLUMN_ALLOWLIST.items():
            observed = tuple(
                row[1] for row in projected.execute(f"PRAGMA table_info({table})").fetchall()
            )
            assert observed == columns
        labels = set()
        for label, level, content, scope, created_at in projected.execute(
            "SELECT packet_node_label, level, deidentified_content, packet_scope_label, "
            "created_at FROM nodes"
        ):
            assert len(label) == 32 and all(c in "0123456789abcdef" for c in label)
            assert level in NODE_LEVELS
            assert set(content) <= set("abcdefghijklmnopqrstuvwxyz \t\n\r")
            assert scope == "global" or scope.split(":", 1)[0] in ("project", "session")
            assert _microseconds(created_at) > 0
            labels.add(label)
        for source, target_label, relation, weight in projected.execute(
            "SELECT source_packet_node_label, target_packet_node_label, relation_type, "
            "weight FROM connections"
        ):
            assert source in labels and target_label in labels
            assert relation in RELATION_TYPES
            assert type(weight) is float
        kinds = set()
        for key, *weights in projected.execute(
            "SELECT packet_scope_label, bm25_weight, vector_weight, graph_weight "
            "FROM retrieval_weights"
        ):
            kinds.add(key if key in REQUIRED_POLICY_KEYS else key.split(":", 1)[0])
            assert all(type(value) is float for value in weights)
        assert REQUIRED_POLICY_KEYS <= kinds
        for table in FORCED_EMPTY_TABLES:
            assert projected.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    finally:
        projected.close()

    # No raw production content survived the projection.
    raw = target.read_bytes()
    for plaintext in (b"alpha production content", b"project:seed", b"operator"):
        assert plaintext not in raw


def test_the_projection_refuses_an_unclassified_mutable_production_table(
    builder: Any, production_snapshot: Path, tmp_path: Path
) -> None:
    """An unclassified mutable table is a fatal preseal integrity failure."""

    connection = sqlite3.connect(production_snapshot)
    try:
        connection.execute("CREATE TABLE operator_scratch (value TEXT)")
        connection.commit()
    finally:
        connection.close()
    connection = _readonly(production_snapshot)
    try:
        with pytest.raises(builder.BuildError) as caught:
            builder.classify_source_tables(connection)
    finally:
        connection.close()
    assert caught.value.code == "seed_unclassified_mutable_table"


@pytest.mark.parametrize("dropped", sorted(PRODUCTION_PROJECTION_COLUMNS) + ["recall_events"])
def test_the_projection_refuses_a_missing_required_production_table(
    builder: Any, production_snapshot: Path, dropped: str
) -> None:
    connection = sqlite3.connect(production_snapshot)
    try:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(f"DROP TABLE {dropped}")
        connection.commit()
    finally:
        connection.close()
    connection = _readonly(production_snapshot)
    try:
        with pytest.raises(builder.BuildError) as caught:
            builder.classify_source_tables(connection)
    finally:
        connection.close()
    assert caught.value.code == "seed_required_table_missing"


def _shared_contract_failure(builder: Any, path: Path) -> None:
    """The shared validator's failures carry no message at all."""

    connection = _readonly(path)
    try:
        with pytest.raises(builder.probe.IntegrityFailure) as caught:
            builder.probe.validate_production_seed_state(connection)
    finally:
        connection.close()
    assert caught.value.args == ()
    assert str(caught.value) == ""


def test_the_shared_seed_contract_rejects_a_duplicate_production_policy_key(
    builder: Any, production_snapshot: Path
) -> None:
    """One policy key, one row: the source side refuses the duplicate."""

    connection = sqlite3.connect(production_snapshot)
    try:
        connection.executescript(
            """
            ALTER TABLE retrieval_weights RENAME TO retrieval_weights_original;
            CREATE TABLE retrieval_weights (scope, bm25, vector, graph);
            INSERT INTO retrieval_weights
                SELECT scope, bm25, vector, graph FROM retrieval_weights_original;
            INSERT INTO retrieval_weights VALUES ('default', 1.0, 0.0, 0.0);
            DROP TABLE retrieval_weights_original;
            """
        )
        connection.commit()
    finally:
        connection.close()
    _shared_contract_failure(builder, production_snapshot)


@pytest.mark.parametrize("missing", sorted(REQUIRED_POLICY_KEYS))
def test_the_shared_seed_contract_rejects_a_missing_production_policy_key(
    builder: Any, production_snapshot: Path, missing: str
) -> None:
    connection = sqlite3.connect(production_snapshot)
    try:
        connection.execute("DELETE FROM retrieval_weights WHERE scope = ?", (missing,))
        connection.commit()
    finally:
        connection.close()
    _shared_contract_failure(builder, production_snapshot)


def test_the_shared_seed_contract_rejects_an_empty_production_weights_table(
    builder: Any, tmp_path: Path
) -> None:
    """``retrieval_weights_table_must_be_nonempty``, on a real store."""

    path = tmp_path / "empty-weights.sqlite3"
    config = MemoryConfig(db_path=path, retrieval_weights={})
    with MemoryStore(config):
        pass
    connection = sqlite3.connect(path)
    try:
        connection.execute("DELETE FROM retrieval_weights")
        connection.commit()
        assert connection.execute("SELECT COUNT(*) FROM retrieval_weights").fetchone()[0] == 0
    finally:
        connection.close()
    _shared_contract_failure(builder, path)


@pytest.mark.parametrize("policy_key", ["project:", "session:", "scope:project:p", "operator"])
def test_the_shared_seed_contract_rejects_ungrammatical_production_policy_keys(
    builder: Any, production_snapshot: Path, policy_key: str
) -> None:
    connection = sqlite3.connect(production_snapshot)
    try:
        connection.execute(
            "UPDATE retrieval_weights SET scope = ? WHERE scope = 'project:learned'",
            (policy_key,),
        )
        connection.commit()
    finally:
        connection.close()
    _shared_contract_failure(builder, production_snapshot)


def test_the_required_policy_keys_are_exactly_the_production_defaults() -> None:
    """The allowlist is the production default set, not a private table."""

    assert REQUIRED_POLICY_KEYS == set(DEFAULT_RETRIEVAL_WEIGHTS)
    assert all(
        type(value) is RetrievalWeightConfig for value in DEFAULT_RETRIEVAL_WEIGHTS.values()
    )


def test_a_view_substituted_for_a_production_table_is_not_a_copied_table(
    builder: Any, production_snapshot: Path
) -> None:
    """A view answers the same SELECT; the classifier must not copy it."""

    connection = sqlite3.connect(production_snapshot)
    try:
        connection.executescript(
            "ALTER TABLE connections RENAME TO connections_source;\n"
            "CREATE VIEW connections AS SELECT * FROM connections_source;"
        )
        connection.commit()
    finally:
        connection.close()
    connection = _readonly(production_snapshot)
    try:
        with pytest.raises(builder.BuildError) as caught:
            builder.classify_source_tables(connection)
    finally:
        connection.close()
    # ``connections_source`` is an unclassified mutable table, and the view
    # that shadows it is never a copied one.
    assert caught.value.code == "seed_unclassified_mutable_table"


# --------------------------------------------------------------------------
# Failure hygiene
# --------------------------------------------------------------------------


def test_the_module_identity_of_the_upstream_runtime_is_shared(
    builder: Any, sealer: Any
) -> None:
    """All three reach the runtime by attribute, never by a second import."""

    assert builder.runtime is builder.probe.runtime
    assert builder.probe is builder.accrual.probe
    assert sealer.runtime is sealer.accrual.runtime
    assert sealer.probe is sealer.accrual.probe
    assert builder.runtime is sealer.runtime
    assert builder.IntegrityFailure is sealer.IntegrityFailure


@pytest.mark.parametrize("script", [BUILD, SEAL], ids=["builder", "seal_launcher"])
def test_bytecode_writing_is_disabled_before_the_upstream_import(script: Path) -> None:
    """A self-check that wrote ``scripts/__pycache__`` already touched too much.

    Importing the accrual module by canonical name is what would create it, so
    the guard has to come first in source order, not merely be present.
    """

    source = script.read_text()
    guard = source.index("sys.dont_write_bytecode = True")
    assert guard < source.index("import ap_confirmatory_accrual_v4")
    assert sys.dont_write_bytecode is True


def test_the_verifier_imports_neither_the_builder_nor_the_seal_launcher() -> None:
    """Independence is structural: no import path back to the implementations."""

    source = VERIFY.read_text()
    for forbidden in (
        "ap_confirmatory_accrual_v4",
        "ap_confirmatory_probe_v4",
        "ap_confirmatory_runtime_v4",
        "ap_confirmatory_seal_v4",
        # The campaign tools the manifest now binds are named nowhere either.
        # The verifier recognises them by name shape and re-hashes their bytes,
        # so binding them cost it none of its independence.
        "ap_confirmatory_slot_v4",
        "ap_confirmatory_snapshot_v4",
        "v4_cadence",
        "spec_from_file_location",
        "importlib",
        "runpy",
        "living_memory",
    ):
        assert forbidden not in source
    assert "import build" not in source


def test_every_rejection_prints_one_aggregate_line_from_a_closed_vocabulary(
    packet: Path,
) -> None:
    """No path, label, digest, count or exception text may ever escape."""

    secret = _Secret(PRIVACY_SECRETS["identity_token"])
    manifest = _manifest_of(packet)
    manifest["bindings"]["unchanged_repair_design"]["sha256"] = bytes(secret).hex()
    _plant_needle(packet, manifest, bytes(secret))
    _seal(packet, manifest)
    result = _run_verifier(packet)
    _assert_verifier_rejects(result, "privacy", private_values=[secret])
    assert result.stderr == b""
    text = result.stdout.decode("utf-8")
    assert text.count("\n") == 1
    for leak in (os.fspath(packet), "Traceback", "sqlite", "File \"", "corpus/"):
        assert leak not in text


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["sealed"],
        ["--mechanical-only", "extra"],
        ["extra", "--mechanical-only"],
        ["unknown-mode", "--packet-dir", "."],
        ["sealed", "--packet-dir"],
        ["sealed", "--unknown-flag", "."],
        ["sealed", "--packet", "."],
        ["keyed", "--draft-dir", "."],
    ],
)
def test_the_verifier_refuses_arguments_it_does_not_recognise(
    arguments: list[str],
) -> None:
    """A usage error is silent: even argparse must not describe the command."""

    result = _run([sys.executable, "-I", os.fspath(VERIFY), *arguments])
    assert result.returncode != 0
    assert result.stdout == b""
    assert result.stderr == b""


def test_a_packet_directory_that_is_not_a_directory_is_refused(
    tmp_path: Path,
) -> None:
    """A missing, symlinked or non-directory candidate never opens."""

    absent = tmp_path / "absent"
    regular = tmp_path / "regular"
    regular.write_bytes(b"{}\n")
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "target")
    (tmp_path / "target").mkdir()
    for candidate in (absent, regular, link):
        result = _run(
            [
                sys.executable,
                "-I",
                os.fspath(VERIFY),
                "sealed",
                "--packet-dir",
                os.fspath(candidate),
            ]
        )
        _assert_verifier_rejects(result, "boundary")


def test_the_builder_never_lets_an_exception_representation_reach_stderr(
    tmp_path: Path,
) -> None:
    result = _run_builder(
        "draft",
        "--draft-dir",
        os.fspath(_private_directory(tmp_path / "draft")),
        "--handoff",
        os.fspath(tmp_path / "absent.json"),
    )
    assert result.returncode == BUILD_EXIT_ERROR
    payload = json.loads(result.stderr)
    assert set(payload) == {"status", "code"}
    assert payload["status"] == "error"
    assert "Traceback" not in result.stderr.decode("utf-8")
    assert os.fspath(tmp_path) not in result.stderr.decode("utf-8")


# --------------------------------------------------------------------------
# Campaign tool bindings
# --------------------------------------------------------------------------
#
# A tool that produces evidence while nothing binds it is provenance the packet
# claims but does not have.  The group is therefore closed against disk in both
# directions, and both directions are rejections rather than warnings.


def _campaign_group(packet: Path) -> dict[str, Any]:
    return _manifest_of(packet)["bindings"][CAMPAIGN_TOOL_BINDING_KEY]


def _synthetic_repository(root: Path, tools: Mapping[str, bytes]) -> Path:
    """A repository laid out like the real one, returning its namespace root.

    ``audit_campaign_tools`` derives the repository root from the namespace
    root by the same four steps the sealed entry point uses, so the whole
    closure can be exercised on files that are ours to move.
    """

    namespace = root / "artifacts/animal-planet/evaluation/confirmatory-holdout-v4"
    namespace.mkdir(parents=True)
    for directory in ("scripts", "tests"):
        (root / directory).mkdir(exist_ok=True)
    for relative, raw in tools.items():
        (root / relative).write_bytes(raw)
    return namespace


def _synthetic_tools() -> dict[str, bytes]:
    return {relative: f"# {relative}\n".encode("utf-8") for relative in CAMPAIGN_TOOL_PATHS}


def _synthetic_manifest(tools: Mapping[str, bytes]) -> dict[str, Any]:
    return {
        "bindings": {
            CAMPAIGN_TOOL_BINDING_KEY: {
                relative: _identity_of(raw) for relative, raw in tools.items()
            }
        }
    }


def test_the_manifest_binds_every_campaign_tool_by_path_digest_and_bytes(
    packet: Path,
) -> None:
    """Path, SHA-256 and byte size for each, matching the file on disk.

    ``hash_binding.mutable_head_or_path_only_binding_allowed`` is false, so the
    path is carried by the member name and never travels alone.
    """

    group = _campaign_group(packet)
    assert set(group) == set(CAMPAIGN_TOOL_PATHS)
    for relative in CAMPAIGN_TOOL_PATHS:
        raw = (ROOT / relative).read_bytes()
        assert group[relative] == _identity_of(raw)
        assert group[relative]["bytes"] == len(raw) > 0


def test_the_snapshot_launcher_slot_driver_and_cadence_renderer_are_bound(
    packet: Path,
) -> None:
    """The three tools that were producing evidence unbound, plus their tests."""

    group = _campaign_group(packet)
    for relative in CAMPAIGN_TOOLS_ADDED:
        assert (ROOT / relative).is_file()
        assert relative in group, f"campaign tool is not bound: {relative}"
        assert group[relative] == _identity_of((ROOT / relative).read_bytes())


@pytest.mark.parametrize("relative", CAMPAIGN_TOOLS_ADDED)
def test_a_drifted_campaign_tool_binding_is_rejected(packet: Path, relative: str) -> None:
    """A manifest digest that no longer matches the file is not a warning."""

    manifest = _manifest_of(packet)
    manifest["bindings"][CAMPAIGN_TOOL_BINDING_KEY][relative] = _identity_of(b"drifted")
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "pin")


@pytest.mark.parametrize("relative", CAMPAIGN_TOOLS_ADDED)
def test_a_campaign_tool_bound_at_the_wrong_byte_count_is_rejected(
    packet: Path, relative: str
) -> None:
    """The size is load-bearing on its own, not decoration beside the digest."""

    manifest = _manifest_of(packet)
    manifest["bindings"][CAMPAIGN_TOOL_BINDING_KEY][relative]["bytes"] += 1
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "pin")


@pytest.mark.parametrize("relative", CAMPAIGN_TOOLS_ADDED)
def test_a_campaign_tool_dropped_from_the_manifest_is_rejected(
    packet: Path, relative: str
) -> None:
    """Unbinding a tool that is on disk fails; it does not quietly shrink."""

    manifest = _manifest_of(packet)
    del manifest["bindings"][CAMPAIGN_TOOL_BINDING_KEY][relative]
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "pin")


def test_a_binding_for_a_campaign_tool_that_is_not_on_disk_is_rejected(
    packet: Path,
) -> None:
    """Closure runs the other way too: an invented tool has no bytes to match."""

    manifest = _manifest_of(packet)
    manifest["bindings"][CAMPAIGN_TOOL_BINDING_KEY]["scripts/v4_absent.py"] = _identity_of(
        b"invented"
    )
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "pin")


@pytest.mark.parametrize(
    "member",
    [
        "runtime_observer",
        "scripts/ap_confirmatory_readiness.py",
        "scripts/../scripts/v4_cadence.py",
        "docs/v4-campaign-runbook.md",
        "scripts/v4_cadence.py/",
        "v4_cadence.py",
    ],
    ids=[
        "logical_name",
        "not_generation_tagged",
        "traversal",
        "outside_tool_directories",
        "trailing_separator",
        "no_directory",
    ],
)
def test_a_campaign_tool_member_that_is_not_a_tool_path_is_rejected(
    packet: Path, member: str
) -> None:
    """The member name is the path, so a name that is not one never binds."""

    manifest = _manifest_of(packet)
    manifest["bindings"][CAMPAIGN_TOOL_BINDING_KEY][member] = _identity_of(b"whatever")
    _seal(packet, manifest)
    _assert_verifier_rejects(_run_verifier(packet), "schema")


def test_a_campaign_tool_present_on_disk_but_unbound_is_rejected(packet: Path) -> None:
    """The failure this binding exists to prevent, end to end on the real tree.

    A new tool lands in ``scripts/`` and nothing in the manifest names it.  The
    sealed packet was valid a moment ago and must stop being valid now.
    """

    _assert_verifier_pass(_run_verifier(packet))
    planted = ROOT / "scripts" / "v4_unbound_tool_probe.py"
    assert not planted.exists()
    planted.write_bytes(b"# a tool that touches evidence and that nothing binds\n")
    try:
        _assert_verifier_rejects(_run_verifier(packet), "pin")
    finally:
        planted.unlink()
    _assert_verifier_pass(_run_verifier(packet))


def test_the_tool_closure_holds_on_a_repository_the_test_controls(
    verifier: Any, tmp_path: Path
) -> None:
    """The same closure, on files that can be added and removed freely."""

    tools = _synthetic_tools()
    namespace = _synthetic_repository(tmp_path / "repo", tools)
    manifest = _synthetic_manifest(tools)
    verifier.audit_campaign_tools(namespace, manifest)


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/v4_unbound.py",
        "scripts/ap_confirmatory_unbound_v4.py",
        "tests/test_v4_unbound.py",
        "tests/test_ap_confirmatory_unbound_v4.py",
    ],
)
def test_an_unbound_tool_in_either_tool_directory_is_rejected(
    verifier: Any, tmp_path: Path, relative: str
) -> None:
    """Both directories are swept, and both naming shapes are recognised."""

    tools = _synthetic_tools()
    namespace = _synthetic_repository(tmp_path / "repo", tools)
    manifest = _synthetic_manifest(tools)

    (namespace.parents[3] / relative).write_bytes(b"# unbound\n")
    with pytest.raises(verifier.VerificationError) as caught:
        verifier.audit_campaign_tools(namespace, manifest)
    assert caught.value.code == verifier.PIN


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/ap_confirmatory_readiness.py",
        "scripts/ap_release.py",
        "scripts/README.md",
        "tests/test_ap_confirmatory_readiness.py",
        "tests/test_confirmatory_evidence_protocol_v4.py",
        "tests/test_confirmatory_holdout_v4_packet.py",
    ],
)
def test_a_file_outside_the_campaign_generation_does_not_have_to_be_bound(
    verifier: Any, tmp_path: Path, relative: str
) -> None:
    """The sweep is scoped to the v4 campaign, not to every file in the tree.

    A closure that demanded a binding for every neighbour would be unusable,
    and the retired v2 scanner and the v4 protocol suites are bound elsewhere
    or not evidence-producing at all.
    """

    tools = _synthetic_tools()
    namespace = _synthetic_repository(tmp_path / "repo", tools)
    manifest = _synthetic_manifest(tools)

    (namespace.parents[3] / relative).write_bytes(b"# not a campaign tool\n")
    verifier.audit_campaign_tools(namespace, manifest)


def test_a_bound_campaign_tool_removed_from_disk_is_rejected(
    verifier: Any, tmp_path: Path
) -> None:
    tools = _synthetic_tools()
    namespace = _synthetic_repository(tmp_path / "repo", tools)
    manifest = _synthetic_manifest(tools)

    (namespace.parents[3] / "scripts/v4_cadence.py").unlink()
    with pytest.raises(verifier.VerificationError) as caught:
        verifier.audit_campaign_tools(namespace, manifest)
    assert caught.value.code == verifier.PIN


@pytest.mark.parametrize(
    "relative", ["scripts/ap_confirmatory_slot_v4.py", "tests/test_v4_cadence.py"]
)
def test_a_campaign_tool_whose_bytes_drift_under_a_stale_binding_is_rejected(
    verifier: Any, tmp_path: Path, relative: str
) -> None:
    """The manifest is never the authority: the bytes on disk are re-hashed."""

    tools = _synthetic_tools()
    namespace = _synthetic_repository(tmp_path / "repo", tools)
    manifest = _synthetic_manifest(tools)

    (namespace.parents[3] / relative).write_bytes(b"# rewritten after the seal\n")
    with pytest.raises(verifier.VerificationError) as caught:
        verifier.audit_campaign_tools(namespace, manifest)
    assert caught.value.code == verifier.PIN


def test_a_campaign_tool_smuggled_in_as_a_symlink_is_rejected(
    verifier: Any, tmp_path: Path
) -> None:
    """A tool-shaped name that is not a regular file fails, never skips."""

    tools = _synthetic_tools()
    namespace = _synthetic_repository(tmp_path / "repo", tools)
    manifest = _synthetic_manifest(tools)

    target = tmp_path / "elsewhere.py"
    target.write_bytes(b"# outside the repository\n")
    (namespace.parents[3] / "scripts/v4_linked.py").symlink_to(target)
    with pytest.raises(verifier.VerificationError) as caught:
        verifier.audit_campaign_tools(namespace, manifest)
    assert caught.value.code == verifier.PIN


def test_the_builder_refuses_to_leave_a_campaign_tool_out_of_its_enumeration(
    builder: Any,
) -> None:
    """The builder settles its own enumeration against disk before binding.

    The verifier would reject the packet regardless; this makes the same
    omission fail at the build that would have shipped it.
    """

    assert builder.discover_campaign_tools() == set(builder.CAMPAIGN_TOOL_PATHS)
    assert set(builder.CAMPAIGN_TOOL_PATHS) == set(CAMPAIGN_TOOL_PATHS)
    assert set(builder.GROUP_FILE_BINDINGS[CAMPAIGN_TOOL_BINDING_KEY]) == set(
        CAMPAIGN_TOOL_PATHS
    )


def _module_literals(path: Path) -> dict[str, Any]:
    """Module-level constant assignments, read without importing the module."""

    assignments: dict[str, Any] = {}
    for node in ast.parse(path.read_bytes()).body:
        if type(node) is not ast.Assign or type(node.value) is not ast.Constant:
            continue
        for target in node.targets:
            if type(target) is ast.Name:
                assignments[target.id] = node.value.value
    return assignments


def test_the_verifier_pins_the_sibling_bytes_it_was_actually_written_against() -> None:
    """A stale literal would verify a builder and a protocol that moved on.

    The pins are the verifier's whole claim to independence, so they are worth
    nothing unless something fails when they drift out of date.
    """

    literals = _module_literals(VERIFY)
    expected = {
        ("BUILDER_SHA256", "BUILDER_BYTES"): BUILD,
        ("README_SHA256", "README_BYTES"): NAMESPACE_ROOT / "README.md",
        ("POLICY_SHA256", "POLICY_BYTES"): NAMESPACE_ROOT / "POLICY.md",
        ("PLAN_SHA256", "PLAN_BYTES"): PLAN_PATH,
    }
    for (digest_name, size_name), path in expected.items():
        raw = path.read_bytes()
        identity = _identity_of(raw)
        assert literals[digest_name] == identity["sha256"], f"stale pin: {digest_name}"
        assert literals[size_name] == identity["bytes"], f"stale pin: {size_name}"
