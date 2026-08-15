#!/usr/bin/env python3
"""Build and independently validate the animal-planet ``release-v1`` bundle.

The only semantic replay performed by this program is the public ``eval``
split through ``scripts/ap_baseline.py compare``.  Original and replacement
holdouts are consumed only as already-published aggregate JSON or as opaque
bytes for SHA-256/size checks.  In particular, this program never invokes a
holdout calculator, a packet verifier, a packet builder, or the retired v2
readiness scanner.

``build`` prepares the deterministic four-file core in memory, rejects a
pre-existing destination, and creates ``release-manifest.json`` last.  It does
not invent the separately observed runtime attestation.  ``validate`` requires
that attestation as the exact fifth candidate member, checks its one-way bind
to the manifest, regenerates only the core in a temporary directory, and
byte-compares all four deterministic outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RELEASE_ROOT = Path("artifacts/animal-planet/evaluation/release-v1")

EVAL_CORPUS_ROOT = "artifacts/animal-planet/corpus"
CALCULATOR_SCRIPT = "scripts/ap_baseline.py"
CALCULATOR_METRICS = "payload,cross_scope,correction_dominance,auto_recall"
CALCULATOR_ARGS = (
    "-B",
    CALCULATOR_SCRIPT,
    "compare",
    "--corpus-root",
    EVAL_CORPUS_ROOT,
    "--split",
    "eval",
    "--metrics",
    CALCULATOR_METRICS,
)
CALCULATOR_DISPLAY_COMMAND = ("python3", *CALCULATOR_ARGS)

OVERRIDE_PREFIXES = ("LM_RECALL_REPEAT_", "LM_DELIVERY_")
SAFE_INHERITED_SUBPROCESS_ENV = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TZ",
)
DEFAULT_DELIVERY = {
    "snippet_max_chars": 1200,
    "context_value_max_chars": 160,
    "session_dedup": True,
    "snippet_ladder": [0, 1000, 700, 500, 300, 200],
    "full_node_diet": True,
    "provenance_value_max_chars": 160,
    "stats_compaction": True,
    "sparse_entries": True,
}
DEFAULT_REPEAT_POLICY = {
    "enabled": False,
    "min_unlinked": 5,
    "max_link_rate": 0.2,
    "min_sessions": 2,
    "probe_every": 25,
    "drop_trailing_stubs": False,
}

PAYLOAD_RATIO_MAX = Decimal("0.60")
RETENTION_RATIO_MIN = Decimal("0.95")
PAYLOAD_GAP_MAX_PP = Decimal("5")
CROSS_SCOPE_REDUCTION_MIN = Decimal("0.50")
SAME_SCOPE_RETENTION_MIN = Decimal("0.95")

OUTPUT_ORDER = (
    "eval-compare.json",
    "release-report.json",
    "release-report.md",
    "release-manifest.json",
)
CONTENT_OUTPUTS = OUTPUT_ORDER[:-1]

# Runtime provenance cannot be generated deterministically from repository
# inputs.  The four-file core therefore remains the complete builder output,
# while a finalized candidate has a separate exact-membership contract.
CONTROL_WATERMARK_NAME = "control-watermark.json"
CONTROL_WATERMARK_SCHEMA = "animal-planet-runtime-provenance-control-watermark"
CONTROL_WATERMARK_SCHEMA_VERSION = 1
CANDIDATE_MEMBERS = (*OUTPUT_ORDER, CONTROL_WATERMARK_NAME)
CONTROL_WATERMARK_MAX_BYTES = 64 * 1024
RUNTIME_OBSERVATION_MAX_SECONDS = 300
RUNTIME_CLOCK_SKEW_SECONDS = 60

# This independently verified default-off release is the replay code control.
# A serving build named by the watermark supplies only the event population;
# it is never silently substituted for either replay arm.
REPLAY_CODE_CONTROL_COMMIT = "46a9951842512333b0896370056d07a9e1c25bdf"
REPLAY_CODE_CONTROL_TREE = "c1606671b9d13bed78c21b7d2f9a4bb75a3d1c1c"
# Each public component label below binds the raw bytes of
# ``src/living_memory/<component>.py`` in the actually serving checkout.
RUNTIME_IMPLEMENTATION_COMPONENTS = (
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
)
RUNTIME_IDENTITY_DERIVATION_SCHEMA = "living-memory-service-boot-identity-v1"
RUNTIME_CONFIGURATION_SCHEMA = "living-memory-effective-runtime-configuration-v1"
CANONICAL_JSON_ENCODING = "canonical-json-utf8"
# Identity v1 prevents a dictionary attack on a low-entropy service name:
# ``service_identity_sha256`` hashes the canonical JSON array
# ["living-memory-service-identity-v1", raw service identity,
#  raw per-service process-invocation identity]. ``boot_identity_sha256``
# hashes ["living-memory-boot-identity-v1", raw per-service process-invocation
# identity].  The invocation identity must be high entropy; it is not the
# host-wide OS boot ID.  Only the resulting digests enter the artifact.
# Effective-configuration v1 is the canonical JSON projection of runtime
# configuration after credentials, raw paths, and raw identities are removed;
# it includes every effective delivery/retrieval setting and both repeat
# controls.  The artifact carries only that projection's hash and byte count.
LEGACY_REPEAT_CONTROLS = (
    "LM_RECALL_REPEAT_GATING",
    "LM_RECALL_REPEAT_DROP_TRAILING_STUBS",
)


class ReleaseError(RuntimeError):
    """A fail-closed release construction or validation error."""


@dataclass(frozen=True)
class PinnedFile:
    path: str
    sha256: str
    bytes: int


ORIGINAL_MANIFEST = PinnedFile(
    "artifacts/animal-planet/manifest.json",
    "3f1a6a87a4d34e411f06e50161f9203afc3c4992dfc973d4f544453e8bbfde48",
    34934,
)
REPLACEMENT_MANIFEST = PinnedFile(
    "artifacts/animal-planet/evaluation/replacement-holdout/manifest.json",
    "fd36c972b049c296acbd2537f9af8f1db8d7db725f188c5dd90d27ab9aa3ab83",
    5537,
)

# These seven files are the consumed historical evidence.  Their contents are
# not rewritten, normalized, or copied into the release; fixed hashes make any
# change fatal before a report can be published.
HISTORICAL_INPUTS = (
    PinnedFile(
        "artifacts/animal-planet/evaluation/original-holdout-compare.json",
        "d031828d0f1f6c970c5030f869d07a03a8e83a99643d2899adaa9500ee5f0382",
        8493,
    ),
    PinnedFile(
        "artifacts/animal-planet/evaluation/original-holdout-report.json",
        "7d7fcbf44e81acb9a46e9e8137b165857b1627b9ac314ea5d9e3d8528ec2450b",
        13659,
    ),
    PinnedFile(
        "artifacts/animal-planet/evaluation/original-holdout-report.md",
        "5fabd400a19712c9efae5b21ba0627ea574f7de20caf2764dfcce6b4ef262e77",
        8654,
    ),
    PinnedFile(
        "artifacts/animal-planet/evaluation/shadow-latency.json",
        "40765127c0c5cbdcb2e3f6b365f6f2751a0ad659a0034e7c0180bd283c11c662",
        7389,
    ),
    PinnedFile(
        "artifacts/animal-planet/evaluation/final-report.json",
        "ec06f851ea4dab73609dfebbc54a6299bc48368bd05e11f78c164075fb9b2245",
        13607,
    ),
    PinnedFile(
        "artifacts/animal-planet/evaluation/final-report.md",
        "cb253fd36e1d90a68c8fd6fdd6d38573a79165797eb97cfaa03c212617d6786b",
        4252,
    ),
    PinnedFile(
        "artifacts/animal-planet/evaluation/replacement-auto-recall.json",
        "fc1739fac4843e989cf33cd7ca707f49cd9bcaa3d3f21bda08a63894f41781ba",
        4386,
    ),
)

RETIRED_V2_INPUTS = (
    PinnedFile(
        "artifacts/animal-planet/evaluation/confirmatory-holdout-v2/POLICY.md",
        "096263ddf554dc014c8cd971e6129d47300bd1c0b9710da90a5749ff186a1afb",
        19239,
    ),
    PinnedFile(
        "artifacts/animal-planet/evaluation/confirmatory-holdout-v2/README.md",
        "4455e267979284ffbe09c704de6f51069e2fd733e3092bddbbc5f2f482ac90e4",
        3581,
    ),
    PinnedFile(
        "artifacts/animal-planet/evaluation/confirmatory-holdout-v2/analysis-plan.json",
        "5f5f050330b97c8a35b0e76bd31cbf1b9dd73f4d1a2baa0175817c8201a0c653",
        35902,
    ),
    PinnedFile(
        "artifacts/animal-planet/evaluation/confirmatory-holdout-v2/repair-design.md",
        "76d34d33b7be57230ef0988662060fe70eea229f9b07f2cd5b8a2354600a79f5",
        68799,
    ),
    PinnedFile(
        "artifacts/animal-planet/evaluation/confirmatory-holdout-v2/attempt-note.json",
        "436b32d4cccdb293a6c06249d22f338328b19e49c59f45120d2da4c70f1a2087",
        1699,
    ),
    PinnedFile(
        "scripts/ap_confirmatory_readiness.py",
        "ce1875bf4c442bb481cc254e3b145fb45625b483d4e98644d9365d56cfa017ba",
        54274,
    ),
)

# The preserved holdout is applicable only while the calculator and the P2/P3
# production path used by that evaluation remain byte-identical.  Repeat
# policy files are deliberately absent: this release changes their unsafe
# defaults without changing delivery/reranking behavior.
PRESERVED_IMPLEMENTATION_PATHS = (
    "scripts/ap_baseline.py",
    "src/living_memory/config.py",
    "src/living_memory/delivery.py",
    "src/living_memory/models.py",
    "src/living_memory/replay.py",
    "src/living_memory/resources.py",
    "src/living_memory/retrieval.py",
    "src/living_memory/scope.py",
)
PRESERVED_EVALUATED_COMMIT = "ee042335dbcd763054819def24e7e487de5d8f11"
# The byte-pinned final report names the evaluated commit but recorded hashes
# for only the direct modules above.  These commit-derived pins close the
# package/import dependencies used by those modules.  storage.py is excluded
# intentionally because this release changes only its repeat-policy defaults.
PRESERVED_TRANSITIVE_INPUTS = (
    PinnedFile(
        "src/living_memory/__init__.py",
        "860b4bcc9e1ece2474178ab08200b05dc7c0de3ec0f9a634c7f7457a7fc0b88d",
        1576,
    ),
    PinnedFile(
        "src/living_memory/decay.py",
        "7fed798aa9d63ec44f8706abcd89b47fcb9fb364427094f3fd3ea827cd7dd5e5",
        3242,
    ),
    PinnedFile(
        "src/living_memory/edge_derivation.py",
        "e7e43f165e5172ccea34c70a377c62b44f3efdf2d71e2caf750ac8100ed211ee",
        22834,
    ),
    PinnedFile(
        "src/living_memory/embeddings.py",
        "d8b326f9594b6c781afa0a9fb3ed002ba7a18eef5118c200af5d3f16938c5c41",
        18320,
    ),
    PinnedFile(
        "src/living_memory/feedback.py",
        "f46983ba62e3f77c8c9993da86a160e99f2d1850a6bc41b76ce141d661db947b",
        13826,
    ),
    PinnedFile(
        "src/living_memory/health_audit.py",
        "f0826841d6553f8c23d8e4c493400c86ce34b1ea67ee6a429381de7d1bb2cd3c",
        20824,
    ),
    PinnedFile(
        "src/living_memory/phase.py",
        "0c6700587e5985ed6567b5fe67d7f83b64a8952a1eea58387fa4fdb67a92ec20",
        2121,
    ),
)

# ``storage.py`` intentionally differs from the evaluated holdout commit only
# by the reviewed strict-default repeat-policy transition.  Unlike the dynamic
# implementation manifest below, this fixed pin makes that exception exact:
# any later storage change invalidates preserved-holdout applicability until a
# new release review updates the sanctioned transition.
PRESERVED_RELEASE_TRANSITION_INPUTS = (
    PinnedFile(
        "src/living_memory/storage.py",
        "9c5a8dc275e40fe42673b23daa783bef96a73ad30cb8fe068505f8d3c6b065ac",
        87279,
    ),
)

IMPLEMENTATION_PATHS = (
    "conftest.py",
    "pyproject.toml",
    "scripts/ap_release.py",
    "scripts/ap_baseline.py",
    "scripts/setup-python.sh",
    "scripts/test.sh",
    "src/living_memory/__init__.py",
    "src/living_memory/config.py",
    "src/living_memory/consolidation.py",
    "src/living_memory/decay.py",
    "src/living_memory/delivery.py",
    "src/living_memory/edge_backfill.py",
    "src/living_memory/edge_derivation.py",
    "src/living_memory/embeddings.py",
    "src/living_memory/feedback.py",
    "src/living_memory/health_audit.py",
    "src/living_memory/maintenance.py",
    "src/living_memory/models.py",
    "src/living_memory/phase.py",
    "src/living_memory/prompts.py",
    "src/living_memory/replay.py",
    "src/living_memory/resources.py",
    "src/living_memory/retrieval.py",
    "src/living_memory/scope.py",
    "src/living_memory/server.py",
    "src/living_memory/storage.py",
    "src/living_memory/temporal.py",
    "docs/mcp-interface.md",
    "tests/test_storage.py",
    "tests/test_recall_gating.py",
    "tests/test_ap_baseline_unseen.py",
    "tests/test_ap_release.py",
    "tests/test_retrieval_correction_dominance.py",
    "tests/test_dedup_supersede.py",
    "tests/test_consolidation.py",
    "tests/test_schema_distillation.py",
    "tests/test_consensus_temporal_decay.py",
)

CORRECTION_SUITE = (
    "bash",
    "scripts/test.sh",
    "tests/test_retrieval_correction_dominance.py",
    "tests/test_dedup_supersede.py",
)
ERA_SUITE = (
    "bash",
    "scripts/test.sh",
    "tests/test_consolidation.py",
    "tests/test_schema_distillation.py",
    "tests/test_consensus_temporal_decay.py",
)


@dataclass(frozen=True)
class LineageResult:
    evidence: dict[str, Any]
    historical_json: dict[str, Any]


def sanitized_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a subprocess environment with every repeat/delivery override absent."""

    # Prevent caller-controlled secrets, Python/pytest hooks, or shell hooks
    # from reaching either subprocess.  Only deterministic locale/home/temp
    # inputs are inherited; every configuration channel is fixed below.
    inherited = os.environ if source is None else source
    clean = {
        key: inherited[key]
        for key in SAFE_INHERITED_SUBPROCESS_ENV
        if key in inherited
    }
    clean.update(
        {
            "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
            "PATH": "/usr/bin:/bin",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PYTHON": sys.executable,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONUTF8": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
    )
    return clean


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise ReleaseError(f"cannot hash required file {path}: {exc}") from exc
    return digest.hexdigest(), size


def _file_evidence(path: Path) -> dict[str, Any]:
    digest, size = _file_digest(path)
    return {"sha256": digest, "bytes": size}


def _safe_repo_file(repo_root: Path, relative: str, label: str) -> Path:
    pure = PurePosixPath(relative)
    if not relative or pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ReleaseError(f"unsafe {label} path: {relative!r}")
    if repo_root.is_symlink() or not repo_root.is_dir():
        raise ReleaseError(f"repository root is missing, unsafe, or a symlink: {repo_root}")
    current = repo_root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise ReleaseError(f"{label} is a symlink: {relative}")
    try:
        resolved_root = repo_root.resolve(strict=True)
        resolved = current.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ReleaseError(f"unsafe or missing {label} {relative}: {exc}") from exc
    if not resolved.is_file():
        raise ReleaseError(f"{label} is not a file: {relative}")
    return current


def _read_and_verify_pinned(repo_root: Path, pin: PinnedFile) -> bytes:
    path = _safe_repo_file(repo_root, pin.path, "pinned evidence")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReleaseError(f"cannot read pinned evidence {pin.path}: {exc}") from exc
    actual_hash = _sha256_bytes(raw)
    if len(raw) != pin.bytes or not hmac.compare_digest(actual_hash, pin.sha256):
        raise ReleaseError(
            f"pinned evidence changed: {pin.path} "
            f"(expected {pin.bytes} bytes/{pin.sha256}, "
            f"got {len(raw)} bytes/{actual_hash})"
        )
    return raw


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ReleaseError(f"non-finite JSON number: {value}")


def _load_json_bytes(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_json_object,
            parse_constant=_reject_json_constant,
        )
    except (ValueError, RecursionError) as exc:
        raise ReleaseError(f"invalid UTF-8 JSON in {label}: {exc}") from exc


def _canonical_json(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ReleaseError(f"cannot serialize deterministic JSON: {exc}") from exc
    return (rendered + "\n").encode("utf-8")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseError(f"{label} must be a JSON object")
    return value


def _exact_mapping(value: Any, keys: Sequence[str], label: str) -> dict[str, Any]:
    result = _mapping(value, label)
    expected = set(keys)
    if set(result) != expected:
        raise ReleaseError(
            f"{label} has an unexpected field set; "
            f"missing={sorted(expected - set(result))!r}, "
            f"extra={sorted(set(result) - expected)!r}"
        )
    return result


def _hex_digest(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        rf"[0-9a-f]{{{length}}}", value
    ):
        raise ReleaseError(f"{label} must be exactly {length} lowercase hex characters")
    return value


def _utc_instant(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:"
        r"[0-9]{2}(?:\.[0-9]{1,6})?Z",
        value,
    ):
        raise ReleaseError(f"{label} must be a canonical UTC instant ending in Z")
    try:
        result = datetime.fromisoformat(value[:-1]).replace(tzinfo=UTC)
    except ValueError as exc:
        raise ReleaseError(f"{label} is not a valid UTC instant: {exc}") from exc
    timespec = "microseconds" if result.microsecond else "seconds"
    canonical = result.isoformat(timespec=timespec).replace("+00:00", "Z")
    if value != canonical:
        raise ReleaseError(f"{label} is not canonically formatted")
    return result


def _json_exact(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int or int/float coercions."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _json_exact(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_exact(left, right) for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def _at(value: Any, *path: str) -> Any:
    current = value
    walked: list[str] = []
    for key in path:
        walked.append(key)
        if not isinstance(current, dict) or key not in current:
            raise ReleaseError("missing required field $." + ".".join(walked))
        current = current[key]
    return current


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReleaseError(f"{label} must be a finite JSON number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ReleaseError(f"{label} must be finite")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ReleaseError(f"{label} must be finite")
    return result


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReleaseError(f"{label} must be a non-negative integer")
    return value


def _nonnegative_decimal(value: Any, label: str) -> Decimal:
    result = _decimal(value, label)
    if result < 0:
        raise ReleaseError(f"{label} must be non-negative")
    return result


def _ratio(numerator: Decimal, denominator: Decimal, label: str) -> Fraction:
    if denominator <= 0:
        raise ReleaseError(f"{label} denominator must be positive")
    return Fraction(numerator) / Fraction(denominator)


def _rounded(value: Decimal | Fraction, places: int) -> float:
    if isinstance(value, Fraction):
        # Gate comparisons remain exact Fractions.  Decimal conversion happens
        # only here, with enough precision for deterministic report rounding.
        numerator_digits = len(str(abs(value.numerator)))
        denominator_digits = len(str(abs(value.denominator)))
        with localcontext() as context:
            context.prec = max(64, numerator_digits + denominator_digits + places + 8)
            value = Decimal(value.numerator) / Decimal(value.denominator)
    quantum = Decimal(1).scaleb(-places)
    return float(value.quantize(quantum, rounding=ROUND_HALF_UP))


def _plain_number(value: Decimal) -> int | float:
    integral = value.to_integral_value()
    if value == integral:
        return int(integral)
    return float(value)


def _safe_manifest_member(
    repo_root: Path, packet_root_relative: str, relative: str
) -> Path:
    pure = PurePosixPath(relative)
    if not relative or pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ReleaseError(f"unsafe packet manifest path: {relative!r}")
    combined = PurePosixPath(packet_root_relative).joinpath(pure).as_posix()
    return _safe_repo_file(repo_root, combined, "packet manifest member")


def _verify_packet_manifest(
    repo_root: Path,
    pin: PinnedFile,
    packet_root_relative: str,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_and_verify_pinned(repo_root, pin)
    manifest = _mapping(_load_json_bytes(raw, pin.path), pin.path)
    if manifest.get("frozen") is not True:
        raise ReleaseError(f"packet is not frozen: {pin.path}")
    files = _mapping(manifest.get("files"), f"{pin.path}.files")
    packet_files = _mapping(manifest.get("packet_files"), f"{pin.path}.packet_files")
    overlap = set(files) & set(packet_files)
    if overlap:
        raise ReleaseError(f"duplicate packet manifest members: {sorted(overlap)}")

    verified = 0
    for relative, entry_value in sorted({**files, **packet_files}.items()):
        entry = _mapping(entry_value, f"{pin.path}:{relative}")
        expected_hash = entry.get("sha256")
        expected_bytes = entry.get("bytes")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ReleaseError(f"invalid packet SHA-256 declaration for {relative}")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
            raise ReleaseError(f"invalid packet byte-size declaration for {relative}")
        member = _safe_manifest_member(repo_root, packet_root_relative, relative)
        actual_hash, actual_bytes = _file_digest(member)
        if actual_bytes != expected_bytes or not hmac.compare_digest(actual_hash, expected_hash):
            raise ReleaseError(
                f"packet member changed: {packet_root_relative}/{relative} "
                f"(expected {expected_bytes} bytes/{expected_hash}, "
                f"got {actual_bytes} bytes/{actual_hash})"
            )
        verified += 1

    return (
        {
            "path": pin.path,
            "sha256": pin.sha256,
            "bytes": pin.bytes,
            "frozen": True,
            "manifest_members_hash_verified": verified,
            "semantic_case_reads": 0,
        },
        raw,
    )


def _pinned_evidence(pin: PinnedFile) -> dict[str, Any]:
    return {"path": pin.path, "sha256": pin.sha256, "bytes": pin.bytes}


def _verify_lineage(repo_root: Path) -> LineageResult:
    original, _original_raw = _verify_packet_manifest(
        repo_root, ORIGINAL_MANIFEST, "artifacts/animal-planet"
    )
    replacement, _replacement_raw = _verify_packet_manifest(
        repo_root,
        REPLACEMENT_MANIFEST,
        "artifacts/animal-planet/evaluation/replacement-holdout",
    )

    historical_raw: dict[str, bytes] = {}
    for pin in HISTORICAL_INPUTS:
        historical_raw[pin.path] = _read_and_verify_pinned(repo_root, pin)
    for pin in RETIRED_V2_INPUTS:
        _read_and_verify_pinned(repo_root, pin)

    final_report_path = "artifacts/animal-planet/evaluation/final-report.json"
    final_report = _mapping(
        _load_json_bytes(historical_raw[final_report_path], final_report_path),
        final_report_path,
    )
    recorded_hashes = _mapping(
        _at(final_report, "evidence", "input_sha256"),
        f"{final_report_path}.evidence.input_sha256",
    )
    evaluated_commit = _at(final_report, "evidence", "evaluated_git_commit")
    if evaluated_commit != PRESERVED_EVALUATED_COMMIT:
        raise ReleaseError("historical report's evaluated implementation commit changed")
    compatible: dict[str, dict[str, Any]] = {}
    for relative in PRESERVED_IMPLEMENTATION_PATHS:
        expected = recorded_hashes.get(relative)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ReleaseError(f"historical report lacks implementation pin for {relative}")
        path = _safe_repo_file(repo_root, relative, "preserved implementation")
        actual, size = _file_digest(path)
        if not hmac.compare_digest(actual, expected):
            raise ReleaseError(
                f"preserved holdout is not applicable: {relative} changed "
                f"from {expected} to {actual}"
            )
        compatible[relative] = {"sha256": actual, "bytes": size}
    for pin in PRESERVED_TRANSITIVE_INPUTS:
        _read_and_verify_pinned(repo_root, pin)
        compatible[pin.path] = {
            "sha256": pin.sha256,
            "bytes": pin.bytes,
            "pin_source": f"evaluated commit {PRESERVED_EVALUATED_COMMIT}",
        }
    for pin in PRESERVED_RELEASE_TRANSITION_INPUTS:
        _read_and_verify_pinned(repo_root, pin)
        compatible[pin.path] = {
            "sha256": pin.sha256,
            "bytes": pin.bytes,
            "pin_source": "reviewed strict-default release transition",
        }

    holdout_path = "artifacts/animal-planet/evaluation/original-holdout-compare.json"
    holdout = _mapping(
        _load_json_bytes(historical_raw[holdout_path], holdout_path), holdout_path
    )

    evidence = {
        "original_packet": original,
        "replacement_packet": replacement,
        "historical_aggregates": {
            "status": "byte_identical",
            "files": [_pinned_evidence(pin) for pin in HISTORICAL_INPUTS],
        },
        "retired_confirmatory_v2": {
            "status": "byte_identical_and_no_authority",
            "files": [_pinned_evidence(pin) for pin in RETIRED_V2_INPUTS],
            "scanner_invocations": 0,
            "semantic_reads": 0,
        },
        "preserved_holdout_implementation_compatibility": {
            "status": "byte_identical",
            "files": compatible,
        },
        "reader_boundary": {
            "calculator_split": "eval",
            "holdout_compare_invocations": 0,
            "replacement_reader_invocations": 0,
            "packet_verifier_invocations": 0,
            "readiness_scanner_invocations": 0,
            "semantic_case_reads": 0,
            "new_v3_authority_created": False,
        },
    }
    return LineageResult(evidence=evidence, historical_json={"holdout_compare": holdout})


def _assert_safe_calculator_command(command: Sequence[str]) -> None:
    args = list(command)
    exact = [sys.executable, *CALCULATOR_ARGS]
    if args != exact:
        raise ReleaseError(f"unsafe calculator command: {args!r}")
    required = [
        CALCULATOR_SCRIPT,
        "compare",
        "--corpus-root",
        EVAL_CORPUS_ROOT,
        "--split",
        "eval",
        "--metrics",
        CALCULATOR_METRICS,
    ]
    try:
        start = args.index(CALCULATOR_SCRIPT)
    except ValueError as exc:
        raise ReleaseError("calculator command does not name scripts/ap_baseline.py") from exc
    if start != 2 or args[1] != "-B" or args[start:] != required:
        raise ReleaseError(f"unsafe calculator command: {args!r}")
    corpus_arg = args[start + 3]
    if Path(corpus_arg).is_absolute() or corpus_arg != EVAL_CORPUS_ROOT:
        raise ReleaseError("calculator corpus root must be the pinned relative eval corpus")
    if "holdout" in args[start:]:
        raise ReleaseError("holdout calculator invocation is prohibited")


def _run_calculator(repo_root: Path) -> bytes:
    command = (sys.executable, *CALCULATOR_ARGS)
    _assert_safe_calculator_command(command)
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=sanitized_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseError(f"eval calculator could not complete: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise ReleaseError(f"eval calculator failed with exit {completed.returncode}: {detail}")
    if not completed.stdout:
        raise ReleaseError("eval calculator emitted empty stdout")
    return completed.stdout


def _validate_compare_identity(compare: Any) -> dict[str, Any]:
    result = _mapping(compare, "eval calculator output")
    expected_top_level = {
        "corpus_root",
        "delivery_defaults",
        "fidelity_limits",
        "metrics",
        "split",
    }
    if set(result) != expected_top_level:
        raise ReleaseError(
            "calculator output has an unexpected top-level schema: "
            f"{sorted(result)!r}"
        )
    if result.get("split") != "eval":
        raise ReleaseError("calculator output is not the eval split")
    if result.get("corpus_root") != EVAL_CORPUS_ROOT:
        raise ReleaseError("calculator did not use the exact relative eval corpus root")
    metrics = _mapping(result.get("metrics"), "eval calculator output.metrics")
    required = {"payload", "cross_scope", "correction_dominance", "auto_recall"}
    if set(metrics) != required:
        raise ReleaseError(
            "calculator metric families are not the exact requested set: "
            f"{sorted(metrics)!r}"
        )
    return result


def _validate_effective_configuration(compare: dict[str, Any]) -> dict[str, Any]:
    delivery = _mapping(compare.get("delivery_defaults"), "delivery_defaults")
    if not _json_exact(delivery, DEFAULT_DELIVERY):
        raise ReleaseError(
            "override-free calculator did not expose the default-on delivery diet: "
            f"{delivery!r}"
        )

    repeat = _mapping(
        _at(compare, "metrics", "auto_recall", "repeat_gating"),
        "metrics.auto_recall.repeat_gating",
    )
    policy = _mapping(repeat.get("policy"), "repeat_gating.policy")
    if not _json_exact(policy, DEFAULT_REPEAT_POLICY):
        raise ReleaseError(
            "repeat controls are not strict override-free defaults: " f"{policy!r}"
        )

    population = _mapping(repeat.get("population"), "repeat_gating.population")
    repeated_fingerprints = _count(
        population.get("repeated_fingerprints"), "repeated_fingerprints"
    )
    repeated_events = _count(
        population.get("repeated_fingerprint_events"),
        "repeated_fingerprint_events",
    )
    gated = _count(population.get("gated_events"), "gated_events")
    if repeated_fingerprints == 0 or repeated_events == 0:
        raise ReleaseError("repeat-gating eval evidence has an empty repeated population")
    if repeated_events < 3 * repeated_fingerprints or gated > repeated_events:
        raise ReleaseError("repeat-gating population counts are inconsistent")
    if gated != 0:
        raise ReleaseError(f"override-free eval replay gated {gated} events")

    unseen = _mapping(repeat.get("unseen_in_dev"), "repeat_gating.unseen_in_dev")
    if unseen.get("reference_split") != "dev":
        raise ReleaseError("unseen-in-dev evidence is unavailable or not dev-referenced")
    unseen_population = _mapping(
        unseen.get("population"), "repeat_gating.unseen_in_dev.population"
    )
    unseen_fingerprints = _count(
        unseen_population.get("repeated_fingerprints"),
        "unseen_in_dev.repeated_fingerprints",
    )
    unseen_events = _count(
        unseen_population.get("repeated_fingerprint_events"),
        "unseen_in_dev.repeated_fingerprint_events",
    )
    unseen_gated = _count(
        unseen_population.get("gated_events"), "unseen_in_dev.gated_events"
    )
    if unseen_fingerprints == 0 or unseen_events == 0:
        raise ReleaseError("unseen-in-dev repeat evidence has an empty population")
    if (
        unseen_events < 3 * unseen_fingerprints
        or unseen_gated > unseen_events
        or unseen_fingerprints > repeated_fingerprints
        or unseen_events > repeated_events
        or unseen_gated > gated
    ):
        raise ReleaseError("unseen-in-dev population counts are inconsistent")
    if unseen_gated != 0:
        raise ReleaseError(f"override-free unseen-in-dev replay gated {unseen_gated} events")

    repeated_ungated = _count(
        _at(repeat, "chars", "repeated_automatic", "ungated_total"),
        "repeated automatic ungated chars",
    )
    repeated_effective = _count(
        _at(repeat, "chars", "repeated_automatic", "gated_total"),
        "repeated automatic effective chars",
    )
    organic_ungated = _count(
        _at(repeat, "chars", "organic", "ungated_total"), "organic ungated chars"
    )
    organic_effective = _count(
        _at(repeat, "chars", "organic", "gated_total"), "organic effective chars"
    )
    unseen_ungated = _count(
        _at(unseen, "chars", "repeated_automatic", "ungated_total"),
        "unseen repeated automatic ungated chars",
    )
    unseen_effective = _count(
        _at(unseen, "chars", "repeated_automatic", "gated_total"),
        "unseen repeated automatic effective chars",
    )
    if min(repeated_ungated, organic_ungated, unseen_ungated) <= 0:
        raise ReleaseError("disabled-policy character evidence must be nonempty")
    if repeated_ungated != repeated_effective:
        raise ReleaseError("disabled repeat policy changed repeated-automatic payload")
    if organic_ungated != organic_effective:
        raise ReleaseError("disabled repeat policy changed organic payload")
    if unseen_ungated != unseen_effective:
        raise ReleaseError("disabled repeat policy changed unseen repeated payload")

    zero_delta_fields = (
        (
            _at(repeat, "chars", "automatic_nonrepeated", "delta_pct"),
            "automatic nonrepeated delta_pct",
        ),
        (_at(repeat, "chars", "organic", "delta_pct"), "organic delta_pct"),
        (
            _at(repeat, "chars", "repeated_automatic", "reduction_ratio"),
            "repeated automatic reduction_ratio",
        ),
        (
            _at(unseen, "chars", "repeated_automatic", "reduction_ratio"),
            "unseen repeated automatic reduction_ratio",
        ),
        (
            _at(repeat, "retention", "organic_content_access", "delta_pct"),
            "organic content-access delta_pct",
        ),
        (
            _at(
                unseen,
                "retention",
                "repeated_automatic_content_access",
                "delta_pct",
            ),
            "unseen content-access delta_pct",
        ),
    )
    for value, label in zero_delta_fields:
        if _decimal(value, label) != 0:
            raise ReleaseError(f"disabled repeat policy reported nonzero {label}")

    repeated_ungated_median = _nonnegative_decimal(
        _at(repeat, "chars", "repeated_automatic", "ungated_median"),
        "repeated automatic ungated median",
    )
    repeated_effective_median = _nonnegative_decimal(
        _at(repeat, "chars", "repeated_automatic", "gated_median"),
        "repeated automatic effective median",
    )
    if repeated_ungated_median != repeated_effective_median:
        raise ReleaseError("disabled repeat policy changed repeated-automatic median")

    organic_before = _decimal(
        _at(repeat, "retention", "organic_content_access", "ungated_share"),
        "organic ungated access",
    )
    organic_after = _decimal(
        _at(repeat, "retention", "organic_content_access", "gated_share"),
        "organic effective access",
    )
    if not (Decimal(0) <= organic_before <= Decimal(1)) or not (
        Decimal(0) <= organic_after <= Decimal(1)
    ):
        raise ReleaseError("organic content-access shares must be between 0 and 1")
    unseen_access = _mapping(
        _at(unseen, "retention", "repeated_automatic_content_access"),
        "unseen repeated automatic content access",
    )
    unseen_access_events = _count(
        unseen_access.get("events"), "unseen content-access events"
    )
    unseen_before = _count(
        unseen_access.get("ungated_retained"),
        "unseen ungated retained",
    )
    unseen_after = _count(
        unseen_access.get("gated_retained"),
        "unseen effective retained",
    )
    if (
        unseen_access_events == 0
        or unseen_access_events != unseen_events
        or max(unseen_before, unseen_after) > unseen_access_events
    ):
        raise ReleaseError("unseen content-access counts are empty or inconsistent")
    if organic_before != organic_after or unseen_before != unseen_after:
        raise ReleaseError("disabled repeat policy changed content access")

    return {
        "environment": {
            "repeat_and_delivery_overrides": "absent",
            "retrieval_tuning_override": "absent_fixed_default",
            "sanitized_prefixes": list(OVERRIDE_PREFIXES),
            "inherited_environment_allowlist": list(SAFE_INHERITED_SUBPROCESS_ENV),
            "embedding_backend": "hash",
            "bytecode_writes_disabled": True,
            "python_hash_seed": "0",
            "python_io_encoding": "utf-8",
            "python_user_site_disabled": True,
            "python_safe_path": True,
            "pytest_plugin_autoload_disabled": True,
            "executable_lookup_path": "/usr/bin:/bin",
            "package_index_access_disabled": True,
            "caller_git_and_pip_configuration_removed": True,
        },
        "delivery_defaults": delivery,
        "repeat_gating_policy": policy,
        "zero_gated_events": {
            "all_eval": gated,
            "unseen_in_dev": unseen_gated,
        },
        "disabled_policy_raw_equalities": {
            "repeated_automatic_chars": {
                "ungated": repeated_ungated,
                "effective": repeated_effective,
            },
            "organic_chars": {
                "ungated": organic_ungated,
                "effective": organic_effective,
            },
            "unseen_repeated_automatic_chars": {
                "ungated": unseen_ungated,
                "effective": unseen_effective,
            },
            "organic_content_access": {
                "ungated": _plain_number(organic_before),
                "effective": _plain_number(organic_after),
            },
            "unseen_content_bearing_events": {
                "ungated_retained": unseen_before,
                "effective_retained": unseen_after,
            },
        },
    }


def _distribution_ratio(
    payload: dict[str, Any], name: str, label: str
) -> tuple[dict[str, Any], Fraction]:
    # Distribution statistics can be JSON floats even though their population
    # is character counts (the preserved holdout records integral medians with
    # a ``.0`` suffix, and an even population could legitimately yield ``.5``).
    before = _nonnegative_decimal(
        _at(payload, "before", name), f"{label}.before.{name}"
    )
    after = _nonnegative_decimal(
        _at(payload, "after", name), f"{label}.after.{name}"
    )
    ratio = _ratio(after, before, f"{label}.{name}")
    return (
        {
            "before_chars": _plain_number(before),
            "after_chars": _plain_number(after),
            "ratio": _rounded(ratio, 10),
            "reduction_ratio": _rounded(Fraction(1) - ratio, 10),
        },
        ratio,
    )


def _retention(payload: dict[str, Any], label: str) -> tuple[dict[str, Any], Fraction]:
    output: dict[str, Any] = {}
    exact_ratios: list[Fraction] = []
    for kind in ("top_result_content", "useful_feedback"):
        output[kind] = {}
        for population in ("matched_events", "all_events"):
            item = _mapping(
                _at(payload, "retention", kind, population),
                f"{label}.retention.{kind}.{population}",
            )
            n = _count(item.get("n"), f"{label}.{kind}.{population}.n")
            if n == 0:
                raise ReleaseError(f"{label}.{kind}.{population} has no observations")
            before = _decimal(item.get("before"), f"{label}.{kind}.{population}.before")
            after = _decimal(item.get("after"), f"{label}.{kind}.{population}.after")
            if not (Decimal(0) <= before <= Decimal(1)) or not (
                Decimal(0) <= after <= Decimal(1)
            ):
                raise ReleaseError(
                    f"{label}.{kind}.{population} shares must be between 0 and 1"
                )
            ratio = _ratio(after, before, f"{label}.{kind}.{population}")
            exact_ratios.append(ratio)
            output[kind][population] = {
                "events": n,
                "before_share": _plain_number(before),
                "after_share": _plain_number(after),
                "ratio": _rounded(ratio, 10),
            }
    minimum = min(exact_ratios)
    output["minimum_ratio"] = _rounded(minimum, 10)
    return output, minimum


def _cross_scope(
    cross: dict[str, Any], label: str
) -> tuple[dict[str, Any], Fraction, Fraction]:
    before_results = _count(_at(cross, "before", "results"), f"{label}.before.results")
    before_cross = _count(
        _at(cross, "before", "cross_scope_results"), f"{label}.before.cross_scope"
    )
    after_results = _count(_at(cross, "after", "results"), f"{label}.after.results")
    after_cross = _count(
        _at(cross, "after", "cross_scope_results"), f"{label}.after.cross_scope"
    )
    if before_cross > before_results or after_cross > after_results:
        raise ReleaseError(f"{label} cross-scope counts exceed total results")
    reduction = _ratio(
        Decimal(before_cross - after_cross), Decimal(before_cross), f"{label} cross reduction"
    )
    same_before = before_results - before_cross
    same_after = after_results - after_cross
    retention = _ratio(
        Decimal(same_after), Decimal(same_before), f"{label} same-scope retention"
    )
    return (
        {
            "admission": {
                "before_cross_scope_results": before_cross,
                "after_cross_scope_results": after_cross,
                "reduction_ratio": _rounded(reduction, 10),
            },
            "same_scope": {
                "before_results": same_before,
                "after_results": same_after,
                "retention_ratio": _rounded(retention, 10),
            },
            "total_results": {"before": before_results, "after": after_results},
        },
        reduction,
        retention,
    )


def _proof(
    proof: Mapping[str, Any],
    expected: int,
    label: str,
    expected_command: Sequence[str],
) -> dict[str, Any]:
    passed = _count(proof.get("passed"), f"{label}.passed")
    failed = _count(proof.get("failed"), f"{label}.failed")
    status = proof.get("status")
    command = proof.get("command")
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        raise ReleaseError(f"{label}.command must be a string list")
    if command != list(expected_command):
        raise ReleaseError(f"{label}.command does not name the bound focused suite")
    if status != "pass" or passed != expected or failed != 0:
        raise ReleaseError(
            f"{label} proof failed: expected exactly {expected} passed/0 failed, "
            f"got {passed} passed/{failed} failed ({status!r})"
        )
    return {"command": command, "passed": passed, "failed": failed, "status": "pass"}


def derive_measurements(
    eval_compare: Mapping[str, Any],
    holdout_compare: Mapping[str, Any],
    correction_proof: Mapping[str, Any],
    era_proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive P2--P5 exclusively from raw calculator/aggregate counts."""

    eval_metrics = _mapping(eval_compare.get("metrics"), "eval.metrics")
    holdout_metrics = _mapping(holdout_compare.get("metrics"), "holdout.metrics")
    eval_payload = _mapping(eval_metrics.get("payload"), "eval.metrics.payload")
    holdout_payload = _mapping(holdout_metrics.get("payload"), "holdout.metrics.payload")

    eval_median, eval_median_exact = _distribution_ratio(eval_payload, "median", "eval")
    eval_p90, eval_p90_exact = _distribution_ratio(eval_payload, "p90", "eval")
    holdout_median, holdout_median_exact = _distribution_ratio(
        holdout_payload, "median", "holdout"
    )
    holdout_p90, holdout_p90_exact = _distribution_ratio(holdout_payload, "p90", "holdout")
    eval_retention, eval_retention_exact = _retention(eval_payload, "eval")
    holdout_retention, holdout_retention_exact = _retention(holdout_payload, "holdout")
    median_gap = abs(eval_median_exact - holdout_median_exact) * 100
    p90_gap = abs(eval_p90_exact - holdout_p90_exact) * 100
    payload_ratio_max = Fraction(PAYLOAD_RATIO_MAX)
    retention_ratio_min = Fraction(RETENTION_RATIO_MIN)
    payload_gap_max_pp = Fraction(PAYLOAD_GAP_MAX_PP)
    p2_pass = all(
        (
            eval_median_exact <= payload_ratio_max,
            eval_p90_exact <= payload_ratio_max,
            holdout_median_exact <= payload_ratio_max,
            holdout_p90_exact <= payload_ratio_max,
            eval_retention_exact >= retention_ratio_min,
            holdout_retention_exact >= retention_ratio_min,
            median_gap <= payload_gap_max_pp,
            p90_gap <= payload_gap_max_pp,
        )
    )

    eval_cross, eval_reduction, eval_same = _cross_scope(
        _mapping(eval_metrics.get("cross_scope"), "eval.metrics.cross_scope"), "eval"
    )
    holdout_cross, holdout_reduction, holdout_same = _cross_scope(
        _mapping(holdout_metrics.get("cross_scope"), "holdout.metrics.cross_scope"),
        "holdout",
    )
    cross_scope_reduction_min = Fraction(CROSS_SCOPE_REDUCTION_MIN)
    same_scope_retention_min = Fraction(SAME_SCOPE_RETENTION_MIN)
    p3_pass = all(
        (
            eval_reduction >= cross_scope_reduction_min,
            holdout_reduction >= cross_scope_reduction_min,
            eval_same >= same_scope_retention_min,
            holdout_same >= same_scope_retention_min,
        )
    )

    correction = _proof(
        correction_proof, 26, "correction focused suite", CORRECTION_SUITE
    )
    era = _proof(era_proof, 45, "era focused suite", ERA_SUITE)
    eval_corr = _mapping(
        eval_metrics.get("correction_dominance"), "eval.metrics.correction_dominance"
    )
    holdout_corr = _mapping(
        holdout_metrics.get("correction_dominance"),
        "holdout.metrics.correction_dominance",
    )

    def correction_replay(value: dict[str, Any], label: str) -> tuple[dict[str, Any], bool]:
        pairs = _count(_at(value, "population", "pairs"), f"{label}.pairs")
        events = _count(
            _at(value, "population", "events_with_pairs"), f"{label}.events_with_pairs"
        )
        violations = _count(_at(value, "after", "violations"), f"{label}.violations")
        dropped = _count(
            _at(value, "after", "corrections_dropped"), f"{label}.corrections_dropped"
        )
        if events > pairs or violations > pairs or dropped > pairs:
            raise ReleaseError(f"{label} correction counts are inconsistent")
        if pairs != 0 or events != 0:
            raise ReleaseError(
                f"{label} no longer has the release-lineage zero-pair population"
            )
        return (
            {
                "candidate_pairs": pairs,
                "events_with_pairs": events,
                "after_violations": violations,
                "corrections_dropped": dropped,
            },
            violations == 0 and dropped == 0,
        )

    eval_replay, eval_corr_pass = correction_replay(eval_corr, "eval correction")
    holdout_replay, holdout_corr_pass = correction_replay(
        holdout_corr, "holdout correction"
    )
    p4_pass = eval_corr_pass and holdout_corr_pass
    zero_pair_splits = [
        label
        for label, replay in (
            ("eval", eval_replay),
            ("preserved holdout", holdout_replay),
        )
        if replay["candidate_pairs"] == 0
    ]
    if zero_pair_splits:
        pair_disclosure = (
            "Recorded-candidate replay exercises zero correction/superseded pairs in "
            + " and ".join(zero_pair_splits)
            + "; the 26-test live/property/SQLite focused suite is the substantive "
            "ordering proof."
        )
    else:
        pair_disclosure = (
            "Recorded-candidate replay exercises correction/superseded pairs in both "
            "eval and the preserved holdout; the focused suite independently covers "
            "live/property/SQLite ordering."
        )

    measurements = {
        "P2_payload_and_retention": {
            "status": "pass" if p2_pass else "fail",
            "thresholds": {
                "maximum_payload_ratio": float(PAYLOAD_RATIO_MAX),
                "minimum_retention_ratio": float(RETENTION_RATIO_MIN),
                "maximum_eval_to_holdout_reduction_gap_pp": float(PAYLOAD_GAP_MAX_PP),
            },
            "eval": {
                "median": eval_median,
                "p90": eval_p90,
                "retention": eval_retention,
            },
            "preserved_holdout": {
                "median": holdout_median,
                "p90": holdout_p90,
                "retention": holdout_retention,
                "semantic_rerun": False,
            },
            "eval_to_holdout_reduction_gap_pp": {
                "median": _rounded(median_gap, 6),
                "p90": _rounded(p90_gap, 6),
            },
        },
        "P3_cross_scope": {
            "status": "pass" if p3_pass else "fail",
            "thresholds": {
                "minimum_admission_reduction_ratio": float(CROSS_SCOPE_REDUCTION_MIN),
                "minimum_same_scope_retention_ratio": float(SAME_SCOPE_RETENTION_MIN),
            },
            "eval": eval_cross,
            "preserved_holdout": {**holdout_cross, "semantic_rerun": False},
        },
        "P4_correction_dominance": {
            "status": "pass" if p4_pass else "fail",
            "focused_suite": correction,
            "eval_replay": eval_replay,
            "preserved_holdout_replay": holdout_replay,
            "zero_pair_population_limitation": pair_disclosure,
        },
        "P5_era_safety": {
            "status": "pass",
            "focused_suite": era,
            "coverage": "mixed-era rejection/repair and ordinary single-era promotion",
        },
    }
    return measurements


def _require_p2_p5(measurements: Mapping[str, Any]) -> None:
    failures = [
        key
        for key in (
            "P2_payload_and_retention",
            "P3_cross_scope",
            "P4_correction_dominance",
            "P5_era_safety",
        )
        if _at(measurements, key, "status") != "pass"
    ]
    if failures:
        raise ReleaseError("release thresholds failed: " + ", ".join(failures))


def _parse_pytest_summary(stdout: bytes, expected: int, label: str) -> None:
    summaries = re.findall(rb"(?<![0-9])(\d+) passed(?:[, ]|$)", stdout)
    if not summaries:
        raise ReleaseError(f"{label} did not emit a pytest pass summary")
    passed = int(summaries[-1])
    negative = re.findall(
        rb"(?<![0-9])(\d+) (?:failed|errors?|skipped|xfailed|xpassed|deselected)",
        stdout,
    )
    if passed != expected or any(int(count) for count in negative):
        detail = stdout.decode("utf-8", errors="replace")[-2000:]
        raise ReleaseError(
            f"{label} expected exactly {expected} clean passes, got:\n{detail}"
        )


def _run_focused_suite(
    repo_root: Path, command: Sequence[str], expected: int, label: str
) -> dict[str, Any]:
    requested = (tuple(command), expected)
    allowed = {(CORRECTION_SUITE, 26), (ERA_SUITE, 45)}
    if requested not in allowed:
        raise ReleaseError(f"unsafe focused-suite command: {list(command)!r}")
    try:
        completed = subprocess.run(
            tuple(command),
            cwd=repo_root,
            env=sanitized_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseError(f"{label} could not complete: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stdout.decode("utf-8", errors="replace")[-2000:]
        raise ReleaseError(f"{label} failed with exit {completed.returncode}:\n{detail}")
    _parse_pytest_summary(completed.stdout, expected, label)
    return {"command": list(command), "passed": expected, "failed": 0, "status": "pass"}


def _implementation_evidence(repo_root: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = _safe_repo_file(repo_root, relative, "implementation binding")
        evidence[relative] = _file_evidence(path)
    return evidence


def _p6_measurement(effective: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "deferred_pending_confirmatory_v3",
        "legacy_repeat_path": "class-blind_experimental_strict_opt_in",
        "automatic_only_compaction": "not_confirmed_by_this_release",
        "effective_policy": effective["repeat_gating_policy"],
        "zero_gated_events": effective["zero_gated_events"],
        "disabled_policy_raw_equalities": effective["disabled_policy_raw_equalities"],
        "historical_negative_evidence": {
            "status": "preserved_byte_for_byte",
            "artifacts": [
                "artifacts/animal-planet/evaluation/final-report.json",
                "artifacts/animal-planet/evaluation/final-report.md",
                "artifacts/animal-planet/evaluation/replacement-auto-recall.json",
            ],
            "new_semantic_evaluation": False,
        },
    }


def _build_report(
    eval_stdout: bytes,
    effective: dict[str, Any],
    measurements: dict[str, Any],
) -> dict[str, Any]:
    _require_p2_p5(measurements)
    p6 = _p6_measurement(effective)
    return {
        "schema_version": 1,
        "release_id": "animal-planet-release-v1",
        "overall": {
            "status": "pass_with_p6_deferred",
            "summary": (
                "P2-P5 pass on override-free eval plus preserved aggregate evidence; "
                "P6 remains deferred pending an event-disjoint confirmatory v3 packet."
            ),
        },
        "configuration": effective,
        "measurements": {**measurements, "P6_automatic_only_policy": p6},
        "gates": [
            {"id": "P1_packet_lineage", "status": "pass"},
            {
                "id": "P2_payload_and_retention",
                "status": measurements["P2_payload_and_retention"]["status"],
            },
            {"id": "P3_cross_scope", "status": measurements["P3_cross_scope"]["status"]},
            {
                "id": "P4_correction_dominance",
                "status": measurements["P4_correction_dominance"]["status"],
            },
            {"id": "P5_era_safety", "status": measurements["P5_era_safety"]["status"]},
            {"id": "P6_automatic_only_policy", "status": p6["status"]},
            {"id": "P7_release_evidence", "status": "pass"},
        ],
        "evidence": {
            "calculator": {
                "command": list(CALCULATOR_DISPLAY_COMMAND),
                "stdout_artifact": "eval-compare.json",
                "stdout_sha256": _sha256_bytes(eval_stdout),
                "stdout_bytes": len(eval_stdout),
                "stdout_stored_byte_for_byte": True,
            },
            "focused_checks_executed_on_release_lineage": True,
            "lineage_and_implementation_hashes": "release-manifest.json",
            "private_raw_data_read_or_committed": False,
            "semantic_holdout_reads": 0,
        },
        "fidelity_limits": [
            "Recorded-candidate replay re-mixes historical candidate scores; "
            "it does not re-retrieve.",
            "The preserved original holdout is consumed only as a byte-pinned aggregate JSON file.",
            "Correction replay has zero candidate pairs; the focused live/property "
            "suite supplies the proof.",
            "feedback_applied is a lower-bound proxy for observed use, not ground-truth relevance.",
        ],
    }


def _render_markdown(report: Mapping[str, Any]) -> bytes:
    p2 = _at(report, "measurements", "P2_payload_and_retention")
    p3 = _at(report, "measurements", "P3_cross_scope")
    p4 = _at(report, "measurements", "P4_correction_dominance")
    p5 = _at(report, "measurements", "P5_era_safety")
    p6 = _at(report, "measurements", "P6_automatic_only_policy")

    def ten(value: Any) -> str:
        return f"{float(value):.10f}"

    lines = [
        "# Animal-planet default-off release v1",
        "",
        "Status: **PASS with P6 deferred**. P2-P5 pass; automatic-only compaction is "
        "not claimed by this release.",
        "",
        "## P2 — payload and retention",
        "",
        "| Evidence | Median after/before | Median ratio | P90 after/before | "
        "P90 ratio | Minimum retention |",
        "|---|---:|---:|---:|---:|---:|",
        (
            "| Override-free eval | "
            f"{p2['eval']['median']['after_chars']}/{p2['eval']['median']['before_chars']} | "
            f"{ten(p2['eval']['median']['ratio'])} | "
            f"{p2['eval']['p90']['after_chars']}/{p2['eval']['p90']['before_chars']} | "
            f"{ten(p2['eval']['p90']['ratio'])} | "
            f"{p2['eval']['retention']['minimum_ratio']} |"
        ),
        (
            "| Preserved holdout aggregate | "
            f"{p2['preserved_holdout']['median']['after_chars']}/"
            f"{p2['preserved_holdout']['median']['before_chars']} | "
            f"{ten(p2['preserved_holdout']['median']['ratio'])} | "
            f"{p2['preserved_holdout']['p90']['after_chars']}/"
            f"{p2['preserved_holdout']['p90']['before_chars']} | "
            f"{ten(p2['preserved_holdout']['p90']['ratio'])} | "
            f"{p2['preserved_holdout']['retention']['minimum_ratio']} |"
        ),
        "",
        (
            "Eval-to-holdout reduction gaps are "
            f"{p2['eval_to_holdout_reduction_gap_pp']['median']:.6f} pp (median) and "
            f"{p2['eval_to_holdout_reduction_gap_pp']['p90']:.6f} pp (p90)."
        ),
        "",
        "## P3 — scope precision",
        "",
        "| Evidence | Cross-scope admission reduction | Same-scope retention |",
        "|---|---:|---:|",
        (
            "| Override-free eval | "
            f"{ten(p3['eval']['admission']['reduction_ratio'])} | "
            f"{ten(p3['eval']['same_scope']['retention_ratio'])} |"
        ),
        (
            "| Preserved holdout aggregate | "
            f"{ten(p3['preserved_holdout']['admission']['reduction_ratio'])} | "
            f"{ten(p3['preserved_holdout']['same_scope']['retention_ratio'])} |"
        ),
        "",
        "## P4 and P5 — correction and era safety",
        "",
        (
            f"The correction suite passed {p4['focused_suite']['passed']} tests and the era-safety "
            f"suite passed {p5['focused_suite']['passed']} tests. Eval replay reports "
            f"{p4['eval_replay']['after_violations']} violations. "
            f"{p4['zero_pair_population_limitation']}"
        ),
        "",
        "## P6 — deferred",
        "",
        (
            f"Status: `{p6['status']}`. `LM_RECALL_REPEAT_GATING` and "
            "`LM_RECALL_REPEAT_DROP_TRAILING_STUBS` are both false with overrides absent; "
            "the override-free eval replay gated zero events. The legacy path is class-blind "
            "and experimental. Historical negative evidence remains byte-identical."
        ),
        "",
        "## Evidence boundary",
        "",
        "`eval-compare.json` is the calculator's stdout byte-for-byte. The release manifest "
        "hash-binds the implementation, effective configuration, original and replacement "
        "packet manifests, retired v2 design, latency/context-cost evidence, and all seven "
        "historical aggregates. No holdout calculator or semantic case reader was invoked.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _build_manifest(
    contents: Mapping[str, bytes],
    lineage: Mapping[str, Any],
    implementation: Mapping[str, Any],
    effective: Mapping[str, Any],
) -> dict[str, Any]:
    if set(contents) != set(CONTENT_OUTPUTS):
        raise ReleaseError("manifest construction received an unexpected output set")
    return {
        "schema_version": 1,
        "release_id": "animal-planet-release-v1",
        "required_companion": {
            "name": CONTROL_WATERMARK_NAME,
            "schema": CONTROL_WATERMARK_SCHEMA,
            "schema_version": CONTROL_WATERMARK_SCHEMA_VERSION,
            "required_for_final_validation": True,
            "binding": "companion_reverse_pins_manifest_sha256_and_bytes",
        },
        "bundle_files": {
            name: {"sha256": _sha256_bytes(contents[name]), "bytes": len(contents[name])}
            for name in CONTENT_OUTPUTS
        },
        "measurement": {
            "calculator_command": list(CALCULATOR_DISPLAY_COMMAND),
            "calculator_stdout": "eval-compare.json",
            "calculator_stdout_stored_byte_for_byte": True,
            "corpus_root_is_relative": True,
            "split": "eval",
            "sanitized_configuration": effective,
            "sanitized_configuration_sha256": _sha256_bytes(
                _canonical_json(effective)
            ),
        },
        "lineage": lineage,
        "implementation": implementation,
        "publication": {
            "no_overwrite": True,
            "content_before_manifest": True,
            "manifest_last": True,
            "write_order": list(OUTPUT_ORDER),
        },
        "privacy": {
            "private_raw_data_committed": False,
            "semantic_holdout_reads": 0,
            "mechanical_hash_reads_only_for_packet_case_files": True,
        },
    }


def _generate_bundle(repo_root: Path) -> dict[str, bytes]:
    lineage = _verify_lineage(repo_root)
    implementation = _implementation_evidence(repo_root)
    eval_stdout = _run_calculator(repo_root)
    eval_compare = _validate_compare_identity(
        _load_json_bytes(eval_stdout, "calculator stdout")
    )
    effective = _validate_effective_configuration(eval_compare)

    correction_proof = _run_focused_suite(
        repo_root, CORRECTION_SUITE, 26, "correction focused suite"
    )
    era_proof = _run_focused_suite(repo_root, ERA_SUITE, 45, "era focused suite")
    final_lineage = _verify_lineage(repo_root)
    final_implementation = _implementation_evidence(repo_root)
    if not _json_exact(final_lineage.evidence, lineage.evidence) or not _json_exact(
        final_lineage.historical_json, lineage.historical_json
    ):
        raise ReleaseError("lineage evidence changed while release checks were running")
    if not _json_exact(final_implementation, implementation):
        raise ReleaseError("implementation changed while release checks were running")
    holdout_compare = lineage.historical_json["holdout_compare"]
    measurements = derive_measurements(
        eval_compare, holdout_compare, correction_proof, era_proof
    )
    _require_p2_p5(measurements)

    report = _build_report(eval_stdout, effective, measurements)
    report_json = _canonical_json(report)
    report_markdown = _render_markdown(report)
    contents = {
        "eval-compare.json": eval_stdout,
        "release-report.json": report_json,
        "release-report.md": report_markdown,
    }
    manifest = _build_manifest(
        contents,
        lineage.evidence,
        implementation,
        effective,
    )
    return {**contents, "release-manifest.json": _canonical_json(manifest)}


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseError(f"cannot open output directory for fsync: {exc}") from exc
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            raise ReleaseError(f"cannot fsync output directory: {exc}") from exc
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, raw: bytes) -> None:
    created = False
    try:
        with path.open("xb") as handle:
            created = True
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        cleanup_error: OSError | None = None
        if created:
            try:
                path.unlink()
            except OSError as cleanup_exc:
                cleanup_error = cleanup_exc
        if cleanup_error is not None:
            raise ReleaseError(
                f"cannot publish {path.name} exclusively: {exc}; "
                f"cannot remove incomplete output: {cleanup_error}"
            ) from exc
        raise ReleaseError(f"cannot publish {path.name} exclusively: {exc}") from exc


def _publish_bundle(root: Path, bundle: Mapping[str, bytes]) -> None:
    if tuple(bundle) != OUTPUT_ORDER or set(bundle) != set(OUTPUT_ORDER):
        raise ReleaseError("generator produced an unexpected or misordered output set")
    if root.exists() or root.is_symlink():
        raise ReleaseError(f"release root already exists; refusing to overwrite: {root}")
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir()
    except OSError as exc:
        raise ReleaseError(f"cannot create release root {root}: {exc}") from exc
    for name in CONTENT_OUTPUTS:
        _write_exclusive(root / name, bundle[name])
    _fsync_directory(root)
    # The presence of this file is the deterministic core publication boundary.
    # Finalization remains impossible until an external observer supplies the
    # required companion attestation.
    manifest_path = root / "release-manifest.json"
    _write_exclusive(manifest_path, bundle["release-manifest.json"])
    try:
        _fsync_directory(root)
    except ReleaseError as exc:
        cleanup_detail = ""
        try:
            manifest_path.unlink()
        except OSError as cleanup_exc:
            cleanup_detail = f"; cannot remove uncommitted manifest: {cleanup_exc}"
        else:
            try:
                _fsync_directory(root)
            except ReleaseError as cleanup_exc:
                cleanup_detail = f"; cannot fsync manifest removal: {cleanup_exc}"
        raise ReleaseError(f"manifest publication did not commit: {exc}{cleanup_detail}") from exc


def _resolve_root(root: Path, repo_root: Path) -> Path:
    return root if root.is_absolute() else repo_root / root


def build_release(
    root: Path = DEFAULT_RELEASE_ROOT,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, bytes]:
    """Generate and publish one immutable deterministic release core."""

    destination = _resolve_root(Path(root), repo_root)
    if destination.exists() or destination.is_symlink():
        raise ReleaseError(
            f"release root already exists; refusing to overwrite: {destination}"
        )
    bundle = _generate_bundle(repo_root)
    _publish_bundle(destination, bundle)
    return bundle


def _read_exact_bundle_members(
    root: Path,
    members: Sequence[str],
    label: str,
) -> dict[str, bytes]:
    if not all(hasattr(os, flag) for flag in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")):
        raise ReleaseError("platform cannot enforce non-symlink release member reads")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
    )
    try:
        directory_fd = os.open(root, directory_flags)
    except OSError as exc:
        raise ReleaseError(
            f"release root is missing, not a directory, or a symlink: {root}: {exc}"
        ) from exc
    try:
        try:
            names = set(os.listdir(directory_fd))
        except OSError as exc:
            raise ReleaseError(f"cannot enumerate {label}: {exc}") from exc
        expected = set(members)
        if names != expected:
            raise ReleaseError(
                f"{label} must contain exactly {list(members)!r}; "
                f"missing={sorted(expected - names)!r}, "
                f"extra={sorted(names - expected)!r}"
            )

        bundle: dict[str, bytes] = {}
        member_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | os.O_NONBLOCK
            | os.O_NOFOLLOW
        )
        for name in members:
            try:
                member_fd = os.open(name, member_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise ReleaseError(
                    f"{label} member is not a readable regular non-symlink file: "
                    f"{name}: {exc}"
                ) from exc
            try:
                try:
                    member_stat = os.fstat(member_fd)
                except OSError as exc:
                    raise ReleaseError(f"cannot stat {label} member {name}: {exc}") from exc
                if not stat.S_ISREG(member_stat.st_mode):
                    raise ReleaseError(
                        f"{label} member is not a regular non-symlink file: {name}"
                    )
                read_limit = (
                    CONTROL_WATERMARK_MAX_BYTES + 1
                    if name == CONTROL_WATERMARK_NAME
                    else -1
                )
                if (
                    name == CONTROL_WATERMARK_NAME
                    and member_stat.st_size > CONTROL_WATERMARK_MAX_BYTES
                ):
                    raise ReleaseError(
                        f"{CONTROL_WATERMARK_NAME} exceeds "
                        f"{CONTROL_WATERMARK_MAX_BYTES} bytes"
                    )
                try:
                    with os.fdopen(member_fd, "rb", closefd=True) as handle:
                        member_fd = -1
                        raw = handle.read(read_limit)
                except OSError as exc:
                    raise ReleaseError(f"cannot read {label} member {name}: {exc}") from exc
                if (
                    name == CONTROL_WATERMARK_NAME
                    and len(raw) > CONTROL_WATERMARK_MAX_BYTES
                ):
                    raise ReleaseError(
                        f"{CONTROL_WATERMARK_NAME} exceeds "
                        f"{CONTROL_WATERMARK_MAX_BYTES} bytes"
                    )
                bundle[name] = raw
            finally:
                if member_fd >= 0:
                    os.close(member_fd)
        try:
            final_names = set(os.listdir(directory_fd))
        except OSError as exc:
            raise ReleaseError(f"cannot re-enumerate {label}: {exc}") from exc
        if final_names != expected:
            raise ReleaseError(
                f"{label} membership changed while it was being read; "
                f"missing={sorted(expected - final_names)!r}, "
                f"extra={sorted(final_names - expected)!r}"
            )
        return bundle
    finally:
        os.close(directory_fd)


def _read_core_bundle(root: Path) -> dict[str, bytes]:
    return _read_exact_bundle_members(root, OUTPUT_ORDER, "release core")


def _read_candidate_bundle(root: Path) -> dict[str, bytes]:
    return _read_exact_bundle_members(root, CANDIDATE_MEMBERS, "finalized release candidate")


def _positive_bytes(value: Any, label: str) -> int:
    result = _count(value, label)
    if result == 0:
        raise ReleaseError(f"{label} must be positive")
    return result


def _validate_file_identity(value: Any, label: str) -> dict[str, Any]:
    identity = _exact_mapping(value, ("sha256", "bytes"), label)
    _hex_digest(identity["sha256"], 64, f"{label}.sha256")
    _positive_bytes(identity["bytes"], f"{label}.bytes")
    return identity


def _validate_control_watermark(
    candidate: Mapping[str, bytes],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a supplied runtime attestation without ever constructing one."""

    discovered_runtime_components = {
        path.stem for path in (REPO_ROOT / "src/living_memory").glob("*.py")
    }
    if discovered_runtime_components != set(RUNTIME_IMPLEMENTATION_COMPONENTS):
        raise ReleaseError(
            "runtime implementation component contract is incomplete or stale"
        )
    raw = candidate[CONTROL_WATERMARK_NAME]
    if not raw or len(raw) > CONTROL_WATERMARK_MAX_BYTES:
        raise ReleaseError(
            f"{CONTROL_WATERMARK_NAME} must be between 1 and "
            f"{CONTROL_WATERMARK_MAX_BYTES} bytes"
        )
    watermark = _exact_mapping(
        _load_json_bytes(raw, CONTROL_WATERMARK_NAME),
        (
            "schema",
            "schema_version",
            "release_id",
            "release_manifest",
            "replay_code_control",
            "runtime_provenance",
            "selection",
            "privacy",
        ),
        CONTROL_WATERMARK_NAME,
    )
    if not hmac.compare_digest(raw, _canonical_json(watermark)):
        raise ReleaseError(f"{CONTROL_WATERMARK_NAME} must use canonical JSON bytes")
    if (
        watermark["schema"] != CONTROL_WATERMARK_SCHEMA
        or not _json_exact(
            watermark["schema_version"], CONTROL_WATERMARK_SCHEMA_VERSION
        )
        or watermark["release_id"] != "animal-planet-release-v1"
    ):
        raise ReleaseError("control watermark identity is invalid")

    declared_companion = _at(manifest, "required_companion")
    if not _json_exact(
        declared_companion,
        {
            "name": CONTROL_WATERMARK_NAME,
            "schema": CONTROL_WATERMARK_SCHEMA,
            "schema_version": CONTROL_WATERMARK_SCHEMA_VERSION,
            "required_for_final_validation": True,
            "binding": "companion_reverse_pins_manifest_sha256_and_bytes",
        },
    ):
        raise ReleaseError("manifest does not declare the required control watermark")

    manifest_identity = _exact_mapping(
        watermark["release_manifest"],
        ("name", "sha256", "bytes"),
        "control-watermark.release_manifest",
    )
    if manifest_identity["name"] != "release-manifest.json":
        raise ReleaseError("control watermark binds the wrong manifest member")
    manifest_hash = _hex_digest(
        manifest_identity["sha256"],
        64,
        "control-watermark.release_manifest.sha256",
    )
    manifest_bytes = _positive_bytes(
        manifest_identity["bytes"],
        "control-watermark.release_manifest.bytes",
    )
    manifest_raw = candidate["release-manifest.json"]
    if manifest_bytes != len(manifest_raw) or not hmac.compare_digest(
        manifest_hash, _sha256_bytes(manifest_raw)
    ):
        raise ReleaseError("control watermark does not bind the exact manifest bytes")

    replay_control = _exact_mapping(
        watermark["replay_code_control"],
        ("role", "commit", "tree"),
        "control-watermark.replay_code_control",
    )
    _hex_digest(replay_control["commit"], 40, "replay code-control commit")
    _hex_digest(replay_control["tree"], 40, "replay code-control tree")
    if not _json_exact(
        replay_control,
        {
            "role": "replay_code_control_only",
            "commit": REPLAY_CODE_CONTROL_COMMIT,
            "tree": REPLAY_CODE_CONTROL_TREE,
        },
    ):
        raise ReleaseError("control watermark replay code control is invalid")

    selection = _exact_mapping(
        watermark["selection"],
        ("basis", "lower_bound_exclusive_at"),
        "control-watermark.selection",
    )
    if selection["basis"] != "observed_runtime_provenance_instant":
        raise ReleaseError("control watermark selection basis is invalid")
    lower_bound = _utc_instant(
        selection["lower_bound_exclusive_at"],
        "control-watermark.selection.lower_bound_exclusive_at",
    )

    provenance = _exact_mapping(
        watermark["runtime_provenance"],
        (
            "identity_derivation",
            "all_event_producing_services_attested",
            "observation",
            "services",
        ),
        "control-watermark.runtime_provenance",
    )
    identity_derivation = _exact_mapping(
        provenance["identity_derivation"],
        (
            "schema",
            "algorithm",
            "encoding",
            "service_identity_input",
            "boot_identity_input",
            "per_service_process_invocation_identity_high_entropy",
        ),
        "control-watermark.runtime_provenance.identity_derivation",
    )
    if not _json_exact(
        identity_derivation,
        {
            "schema": RUNTIME_IDENTITY_DERIVATION_SCHEMA,
            "algorithm": "SHA-256",
            "encoding": CANONICAL_JSON_ENCODING,
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
    ):
        raise ReleaseError("control watermark identity derivation is invalid or unsafe")
    if provenance["all_event_producing_services_attested"] is not True:
        raise ReleaseError("not every event-producing service is attested")
    services = provenance["services"]
    if not isinstance(services, list) or not services:
        raise ReleaseError("control-watermark.runtime_provenance.services must be nonempty")

    service_keys: list[tuple[str, str]] = []
    boot_times: list[datetime] = []
    for index, service_value in enumerate(services):
        label = f"control-watermark.runtime_provenance.services[{index}]"
        service = _exact_mapping(
            service_value,
            (
                "service_identity_sha256",
                "boot_identity_sha256",
                "boot_started_at",
                "serving_build",
                "sanitized_configuration",
                "effective_legacy_repeat_controls",
            ),
            label,
        )
        service_identity = _hex_digest(
            service["service_identity_sha256"], 64, f"{label}.service_identity_sha256"
        )
        boot_identity = _hex_digest(
            service["boot_identity_sha256"], 64, f"{label}.boot_identity_sha256"
        )
        boot_started_at = _utc_instant(
            service["boot_started_at"], f"{label}.boot_started_at"
        )
        if boot_started_at > lower_bound:
            raise ReleaseError(f"{label} boot starts after the selection boundary")
        boot_times.append(boot_started_at)

        serving_build = _exact_mapping(
            service["serving_build"],
            ("commit", "tree", "implementation"),
            f"{label}.serving_build",
        )
        _hex_digest(serving_build["commit"], 40, f"{label}.serving_build.commit")
        _hex_digest(serving_build["tree"], 40, f"{label}.serving_build.tree")
        implementation = _exact_mapping(
            serving_build["implementation"],
            RUNTIME_IMPLEMENTATION_COMPONENTS,
            f"{label}.serving_build.implementation",
        )
        for component in RUNTIME_IMPLEMENTATION_COMPONENTS:
            _validate_file_identity(
                implementation[component],
                f"{label}.serving_build.implementation.{component}",
            )
        configuration = _exact_mapping(
            service["sanitized_configuration"],
            ("schema", "encoding", "sha256", "bytes"),
            f"{label}.sanitized_configuration",
        )
        if (
            configuration["schema"] != RUNTIME_CONFIGURATION_SCHEMA
            or configuration["encoding"] != CANONICAL_JSON_ENCODING
        ):
            raise ReleaseError(f"{label} sanitized configuration schema is invalid")
        _hex_digest(
            configuration["sha256"],
            64,
            f"{label}.sanitized_configuration.sha256",
        )
        _positive_bytes(
            configuration["bytes"],
            f"{label}.sanitized_configuration.bytes",
        )
        controls = _exact_mapping(
            service["effective_legacy_repeat_controls"],
            LEGACY_REPEAT_CONTROLS,
            f"{label}.effective_legacy_repeat_controls",
        )
        if not _json_exact(
            controls,
            {control: False for control in LEGACY_REPEAT_CONTROLS},
        ):
            raise ReleaseError(f"{label} has a legacy repeat control enabled or malformed")
        service_keys.append((service_identity, boot_identity))

    if len({service for service, _boot in service_keys}) != len(service_keys):
        raise ReleaseError("control watermark contains duplicate service identities")
    if len({boot for _service, boot in service_keys}) != len(service_keys):
        raise ReleaseError("control watermark contains duplicate boot identities")
    if service_keys != sorted(service_keys):
        raise ReleaseError("control watermark services must be sorted by hashed identity")

    observation = _exact_mapping(
        provenance["observation"],
        (
            "method",
            "pre_observed_at",
            "pre_runtime_state_sha256",
            "post_observed_at",
            "post_runtime_state_sha256",
        ),
        "control-watermark.runtime_provenance.observation",
    )
    if observation["method"] != "read_only_pre_post_canonical_services_sha256":
        raise ReleaseError("control watermark observation method is invalid")
    pre_at = _utc_instant(
        observation["pre_observed_at"],
        "control-watermark.runtime_provenance.observation.pre_observed_at",
    )
    post_at = _utc_instant(
        observation["post_observed_at"],
        "control-watermark.runtime_provenance.observation.post_observed_at",
    )
    if pre_at >= post_at:
        raise ReleaseError("runtime pre-observation must precede post-observation")
    if post_at - pre_at > timedelta(seconds=RUNTIME_OBSERVATION_MAX_SECONDS):
        raise ReleaseError("runtime observation envelope is wider than allowed")
    if post_at > datetime.now(UTC) + timedelta(seconds=RUNTIME_CLOCK_SKEW_SECONDS):
        raise ReleaseError("runtime post-observation is implausibly in the future")
    if not pre_at <= lower_bound <= post_at:
        raise ReleaseError("selection boundary is not bracketed by runtime observations")
    if any(boot_started_at > pre_at for boot_started_at in boot_times):
        raise ReleaseError("a service boot starts after the pre-boundary observation")
    pre_state = _hex_digest(
        observation["pre_runtime_state_sha256"],
        64,
        "control-watermark.runtime_provenance.observation.pre_runtime_state_sha256",
    )
    post_state = _hex_digest(
        observation["post_runtime_state_sha256"],
        64,
        "control-watermark.runtime_provenance.observation.post_runtime_state_sha256",
    )
    expected_state = _sha256_bytes(_canonical_json(services))
    if (
        not hmac.compare_digest(pre_state, post_state)
        or not hmac.compare_digest(pre_state, expected_state)
    ):
        raise ReleaseError("pre/post runtime state is unstable or does not bind services")

    privacy = _exact_mapping(
        watermark["privacy"],
        (
            "aggregates_and_hashes_only",
            "private_paths_included",
            "raw_service_identities_included",
            "queries_included",
            "rows_included",
            "raw_outcomes_included",
        ),
        "control-watermark.privacy",
    )
    if not _json_exact(
        privacy,
        {
            "aggregates_and_hashes_only": True,
            "private_paths_included": False,
            "raw_service_identities_included": False,
            "queries_included": False,
            "rows_included": False,
            "raw_outcomes_included": False,
        },
    ):
        raise ReleaseError("control watermark privacy contract is invalid")
    return watermark


def _validate_manifest(candidate: Mapping[str, bytes]) -> dict[str, Any]:
    manifest = _mapping(
        _load_json_bytes(candidate["release-manifest.json"], "release-manifest.json"),
        "release-manifest.json",
    )
    if not _json_exact(manifest.get("schema_version"), 1) or (
        manifest.get("release_id") != "animal-planet-release-v1"
    ):
        raise ReleaseError("release manifest identity is invalid")
    if not _json_exact(
        manifest.get("required_companion"),
        {
            "name": CONTROL_WATERMARK_NAME,
            "schema": CONTROL_WATERMARK_SCHEMA,
            "schema_version": CONTROL_WATERMARK_SCHEMA_VERSION,
            "required_for_final_validation": True,
            "binding": "companion_reverse_pins_manifest_sha256_and_bytes",
        },
    ):
        raise ReleaseError("release manifest companion contract is invalid")
    declared = _mapping(manifest.get("bundle_files"), "release-manifest.bundle_files")
    if set(declared) != set(CONTENT_OUTPUTS):
        raise ReleaseError("release manifest has an unexpected bundle file set")
    for name in CONTENT_OUTPUTS:
        evidence = _mapping(declared[name], f"release-manifest.bundle_files.{name}")
        expected_hash = evidence.get("sha256")
        expected_bytes = evidence.get("bytes")
        raw = candidate[name]
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or expected_bytes != len(raw)
            or not isinstance(expected_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
        ):
            raise ReleaseError(f"release manifest byte/hash declaration is invalid for {name}")
        if not hmac.compare_digest(expected_hash, _sha256_bytes(raw)):
            raise ReleaseError(f"release output hash mismatch: {name}")
    publication = _mapping(manifest.get("publication"), "release-manifest.publication")
    if not _json_exact(
        publication,
        {
            "no_overwrite": True,
            "content_before_manifest": True,
            "manifest_last": True,
            "write_order": list(OUTPUT_ORDER),
        },
    ):
        raise ReleaseError("release manifest publication contract is invalid")
    measurement = _mapping(manifest.get("measurement"), "release-manifest.measurement")
    if (
        measurement.get("calculator_command") != list(CALCULATOR_DISPLAY_COMMAND)
        or measurement.get("calculator_stdout") != "eval-compare.json"
        or measurement.get("calculator_stdout_stored_byte_for_byte") is not True
        or measurement.get("corpus_root_is_relative") is not True
        or measurement.get("split") != "eval"
    ):
        raise ReleaseError("release manifest measurement command/boundary is invalid")
    manifest_configuration = _mapping(
        measurement.get("sanitized_configuration"),
        "release-manifest.measurement.sanitized_configuration",
    )
    configuration_hash = measurement.get("sanitized_configuration_sha256")
    if (
        not isinstance(configuration_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", configuration_hash)
        or not hmac.compare_digest(
            configuration_hash, _sha256_bytes(_canonical_json(manifest_configuration))
        )
    ):
        raise ReleaseError("release manifest configuration hash is invalid")
    return manifest


def _validate_candidate_report(
    candidate: Mapping[str, bytes],
    lineage: LineageResult,
    manifest: Mapping[str, Any],
) -> None:
    eval_compare = _validate_compare_identity(
        _load_json_bytes(candidate["eval-compare.json"], "eval-compare.json")
    )
    effective = _validate_effective_configuration(eval_compare)
    if not _json_exact(
        _at(manifest, "measurement", "sanitized_configuration"), effective
    ):
        raise ReleaseError(
            "release manifest configuration is not the calculator's effective configuration"
        )
    report = _mapping(
        _load_json_bytes(candidate["release-report.json"], "release-report.json"),
        "release-report.json",
    )
    if not _json_exact(report.get("schema_version"), 1) or (
        report.get("release_id") != "animal-planet-release-v1"
    ):
        raise ReleaseError("release report identity is invalid")
    overall = _mapping(report.get("overall"), "release-report.overall")
    if overall.get("status") != "pass_with_p6_deferred":
        raise ReleaseError("release report overall status is invalid")
    stored_measurements = _mapping(report.get("measurements"), "release-report.measurements")
    correction_proof = _mapping(
        _at(stored_measurements, "P4_correction_dominance", "focused_suite"),
        "release-report correction proof",
    )
    era_proof = _mapping(
        _at(stored_measurements, "P5_era_safety", "focused_suite"),
        "release-report era proof",
    )
    rederived = derive_measurements(
        eval_compare,
        lineage.historical_json["holdout_compare"],
        correction_proof,
        era_proof,
    )
    _require_p2_p5(rederived)
    expected_report = _build_report(
        candidate["eval-compare.json"], effective, rederived
    )
    if not _json_exact(report, expected_report):
        raise ReleaseError("release report does not derive exactly from stored raw evidence")
    expected_markdown = _render_markdown(expected_report)
    if not hmac.compare_digest(expected_markdown, candidate["release-report.md"]):
        raise ReleaseError("release Markdown does not derive from release-report.json")


def validate_release(
    root: Path = DEFAULT_RELEASE_ROOT,
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Validate an exact-five candidate, then regenerate its four-file core."""

    destination = _resolve_root(Path(root), repo_root)
    candidate = _read_candidate_bundle(destination)
    manifest = _validate_manifest(candidate)
    _validate_control_watermark(candidate, manifest)
    lineage = _verify_lineage(repo_root)
    if not _json_exact(manifest.get("lineage"), lineage.evidence):
        raise ReleaseError("release manifest lineage does not match mechanical verification")
    if not _json_exact(
        manifest.get("implementation"), _implementation_evidence(repo_root)
    ):
        raise ReleaseError("release manifest implementation hashes are stale or invalid")
    _validate_candidate_report(candidate, lineage, manifest)

    with tempfile.TemporaryDirectory(prefix="ap-release-v1-validate-") as temporary:
        regenerated_root = Path(temporary) / "release-v1"
        build_release(regenerated_root, repo_root=repo_root)
        regenerated = _read_core_bundle(regenerated_root)
        for name in OUTPUT_ORDER:
            if not hmac.compare_digest(candidate[name], regenerated[name]):
                raise ReleaseError(f"release output differs from fresh regeneration: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -B scripts/ap_release.py",
        description=(
            "Build or independently validate the deterministic animal-planet "
            "default-off release-v1 evidence bundle."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("build", "generate the four-file core; never overwrite or invent attestation"),
        ("validate", "validate an exact-five candidate and regenerate its four-file core"),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument(
            "--root",
            type=Path,
            default=DEFAULT_RELEASE_ROOT,
            help=f"release directory (default: {DEFAULT_RELEASE_ROOT.as_posix()})",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            build_release(args.root)
            print(f"built release-v1 core at {_resolve_root(args.root, REPO_ROOT)}")
        else:
            validate_release(args.root)
            print(f"validated release-v1 at {_resolve_root(args.root, REPO_ROOT)}")
    except ReleaseError as exc:
        print(f"ap_release: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
