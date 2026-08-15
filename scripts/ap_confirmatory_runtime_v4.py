#!/usr/bin/env python3
"""Strict shared runtime primitives for ``confirmatory-holdout-v4``.

This module is deliberately observation-only.  It canonicalizes and validates
already captured synthetic or private launcher evidence; it has no code for
contacting a service, opening a database, or reading source rows.  Every
integrity failure is the same message-free exception so private evidence can
never escape through diagnostics.

The initial segment is attested by the release control watermark alone and has
no source binding, so it is modelled by :class:`UnboundWatermarkPredecessor`
rather than by :class:`ActiveSegment`.  Exactly two mutually exclusive bootstrap
branches exist, selected only by whether the complete stable observed service
tuple equals the pinned watermark tuple: an equal tuple yields a
source-binding attestation and the initial ``ActiveSegment``; a different tuple
yields the count-free ``InitialRuntimeChangeClosure``, which carries no prior
source-binding core in any spelling because none has ever existed.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import os
import re
import stat
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLAN = (
    REPO_ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "confirmatory-holdout-v4"
    / "analysis-plan.json"
)
DEFAULT_WATERMARK = (
    REPO_ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "release-v1"
    / "control-watermark.json"
)

NAMESPACE = "confirmatory-holdout-v4"
SCHEMA_VERSION = 4
PINNED_PLAN_SHA256 = "bb049d1b5a237b345a5f94957e5cd82f37bd4940b14011e68e30dfb7aa7c8d70"
PINNED_PLAN_BYTES = 177_982
PINNED_WATERMARK_SHA256 = (
    "a9e6e6a4dbd6178345e11e951e973a1af9c062fb50f8dd7bd51528d44217a86f"
)
PINNED_WATERMARK_BYTES = 10_553
PINNED_RELEASE_MANIFEST_SHA256 = (
    "dd1208d4f5196460428d7049fa9e33ca74ef71be77d074c2b479876653c61768"
)
PINNED_RELEASE_MANIFEST_BYTES = 16_970

RELEASE_EFFECTIVE_AT = "2026-08-14T15:03:46.793603Z"
REPLAY_CODE_CONTROL_COMMIT = "46a9951842512333b0896370056d07a9e1c25bdf"
REPLAY_CODE_CONTROL_TREE = "c1606671b9d13bed78c21b7d2f9a4bb75a3d1c1c"
INITIAL_SEGMENT_ID = "0e8c526343643b2c2019d05afa84012540d13ebe7e899ec5979fc8df90a21b0c"
INITIAL_SERVICES_SHA256 = (
    "84b53e60f899afa18c7ab401f8e992e754f80e2b3b6dbddac7b94e95fe070449"
)

SOURCE_ALIASES = ("local", "alt")
ALIAS_IDS = {
    "local": "confirmatory-local-v4-ro",
    "alt": "confirmatory-alt-v4-ro",
}
ANCHOR_AT = "2026-08-17T00:00:00Z"
CADENCE_SECONDS = 86_400
GRACE_SECONDS = 21_600
FIRST_SLOT_INDEX = 0
LAST_SLOT_INDEX = 28
SLOT_COUNT = 29
ABSOLUTE_HORIZON_AT = "2026-09-14T06:00:00Z"

RUNTIME_SEGMENT_DOMAIN = b"confirmatory-holdout-v4/segment/v1"
SOURCE_DATABASE_IDENTITY_DOMAIN = (
    b"confirmatory-holdout-v4/source-database-identity/v1"
)
ALIAS_SERVICE_DATABASE_BINDING_DOMAIN = (
    b"confirmatory-holdout-v4/alias-service-database-binding/v1"
)
SOURCE_BINDING_CORE_DOMAIN = b"confirmatory-holdout-v4/source-binding-core/v1"
SLOT_RECEIPT_DOMAIN = b"confirmatory-holdout-v4/slot-receipt/v1"

ACTIVE_DOMAIN_PREFIX = "confirmatory-holdout-v4/"
RETIRED_NAMESPACES = ("confirmatory-holdout-v2", "confirmatory-holdout-v3")
RETIRED_DOMAIN_PREFIXES = tuple(f"{name}/" for name in RETIRED_NAMESPACES)
NON_DOMAIN_ENCODING_KEYS = (
    "release_v1_canonical_json_utf8",
    "v4_canonical_json_utf8",
)

SAME_TUPLE_BRANCH = "same-tuple-initial-binding"
CHANGED_TUPLE_BRANCH = "changed-tuple-pre-binding-closure"
ACTIVE_SERVICES_STATE_CHANGE = "active-services-state-change"
SOURCE_BINDING_CHANGE = "source-binding-change"

CLOSURE_RECEIPT_KIND = "slot-segment-closed"
INITIAL_CLOSURE_RECEIPT_KIND = "initial-runtime-change-closure"
CLOSURE_STATUS = "segment-closed"
ATTEMPT_RECEIPT_KIND = "runtime-attestation-attempt"
INITIAL_SOURCE_BINDING_SCOPE = "initial-source-binding"
SUCCESSOR_SEGMENT_SCOPE = "successor-segment"

HEX_LOWER = frozenset("0123456789abcdef")
UTC_PATTERN = re.compile(
    r"(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"T(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})"
    r"(?:\.(?P<fraction>[0-9]{1,6}))?Z\Z"
)
UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)

IDENTITY_DERIVATION = {
    "schema": "living-memory-service-boot-identity-v1",
    "algorithm": "SHA-256",
    "encoding": "canonical-json-utf8",
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
}


class IntegrityFailure(Exception):
    """The sole, deliberately message-free integrity signal."""


def _fail() -> None:
    raise IntegrityFailure from None


def _strict_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail()
        result[key] = value
    return result


def _utf8(value: Any, *, nonempty: bool = False) -> bytes:
    if type(value) is not str or (nonempty and not value):
        _fail()
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError:
        _fail()


def _validate_json_value(
    value: Any,
    *,
    active: set[int] | None = None,
    depth: int = 0,
) -> None:
    if depth > 256:
        _fail()
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail()
        return
    if type(value) is str:
        _utf8(value)
        return
    if type(value) not in (list, dict):
        _fail()
    if active is None:
        active = set()
    marker = id(value)
    if marker in active:
        _fail()
    active.add(marker)
    try:
        if type(value) is list:
            for member in value:
                _validate_json_value(member, active=active, depth=depth + 1)
        else:
            for key, member in value.items():
                _utf8(key)
                _validate_json_value(member, active=active, depth=depth + 1)
    finally:
        active.remove(marker)


def load_json_bytes(raw: bytes, *, expected: type | None = None) -> Any:
    """Parse UTF-8 JSON while rejecting duplicate keys at every depth."""

    if type(raw) is not bytes:
        _fail()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda _value: _fail(),
        )
        _validate_json_value(value)
    except IntegrityFailure:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        _fail()
    if expected is not None and type(value) is not expected:
        _fail()
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the frozen compact v4 canonical JSON representation."""

    _validate_json_value(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
        _fail()


def release_v1_canonical_json_bytes(value: Any) -> bytes:
    """Return release-v1 pretty canonical JSON, including its final newline."""

    _validate_json_value(value)
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
        _fail()


def load_canonical_json_bytes(
    raw: bytes,
    *,
    expected: type | None = None,
    release_v1: bool = False,
) -> Any:
    value = load_json_bytes(raw, expected=expected)
    expected_raw = (
        release_v1_canonical_json_bytes(value)
        if release_v1
        else canonical_json_bytes(value)
    )
    if not hmac.compare_digest(raw, expected_raw):
        _fail()
    return value


def _is_int(value: Any) -> bool:
    return type(value) is int


def _nonnegative_int(value: Any) -> int:
    if not _is_int(value) or value < 0:
        _fail()
    return value


def _positive_int(value: Any) -> int:
    if not _is_int(value) or value <= 0:
        _fail()
    return value


def _bounded_int(value: Any, lower: int, upper: int) -> int:
    if not _is_int(value) or not lower <= value <= upper:
        _fail()
    return value


def _hex(value: Any, length: int) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in HEX_LOWER for character in value)
    ):
        _fail()
    return value


def _exact_mapping(value: Any, fields: Sequence[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        _fail()
    return value


def _json_exact(value: Any, expected: Any) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(
            _json_exact(value[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _json_exact(member, expected_member)
            for member, expected_member in zip(value, expected, strict=True)
        )
    return value == expected


def parse_utc(value: Any, *, receipt: bool = False) -> datetime:
    """Parse the v4 UTC grammar without accepting ISO aliases or offsets."""

    _utf8(value, nonempty=True)
    match = UTC_PATTERN.fullmatch(value)
    if match is None:
        _fail()
    fraction = match.group("fraction")
    if receipt and (fraction is None or len(fraction) != 6):
        _fail()
    microsecond = int((fraction or "").ljust(6, "0"))
    try:
        return datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second")),
            microsecond,
            tzinfo=UTC,
        )
    except ValueError:
        _fail()


def canonical_utc(value: datetime) -> str:
    if type(value) is not datetime or value.tzinfo is not UTC:
        _fail()
    return (
        f"{value.year:04d}-{value.month:02d}-{value.day:02d}"
        f"T{value.hour:02d}:{value.minute:02d}:{value.second:02d}"
        f".{value.microsecond:06d}Z"
    )


def utc_microseconds(value: Any, *, receipt: bool = False) -> int:
    parsed = parse_utc(value, receipt=receipt) if type(value) is str else value
    if type(parsed) is not datetime or parsed.tzinfo is not UTC:
        _fail()
    try:
        delta = parsed - datetime(1970, 1, 1, tzinfo=UTC)
    except (OverflowError, TypeError, ValueError):
        _fail()
    return (
        delta.days * 86_400_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )


def _add_seconds(value: datetime, seconds: int) -> datetime:
    if type(value) is not datetime or value.tzinfo is not UTC or not _is_int(seconds):
        _fail()
    try:
        return value + timedelta(seconds=seconds)
    except OverflowError:
        _fail()


def _add_microseconds(value: datetime, microseconds: int) -> datetime:
    if (
        type(value) is not datetime
        or value.tzinfo is not UTC
        or not _is_int(microseconds)
    ):
        _fail()
    try:
        return value + timedelta(microseconds=microseconds)
    except OverflowError:
        _fail()


@dataclass(frozen=True, slots=True)
class SlotTimes:
    slot_index: int
    scheduled_at: str
    grace_deadline_at: str


def slot_times(slot_index: Any) -> SlotTimes:
    index = _bounded_int(slot_index, FIRST_SLOT_INDEX, LAST_SLOT_INDEX)
    anchor = parse_utc(ANCHOR_AT)
    scheduled = _add_seconds(anchor, index * CADENCE_SECONDS)
    grace = _add_seconds(scheduled, GRACE_SECONDS)
    return SlotTimes(index, canonical_utc(scheduled), canonical_utc(grace))


def absolute_horizon_at() -> str:
    derived = slot_times(LAST_SLOT_INDEX).grace_deadline_at
    if derived != canonical_utc(parse_utc(ABSOLUTE_HORIZON_AT)):
        _fail()
    return derived


def validate_slot_times(
    *,
    slot_index: Any,
    scheduled_at: Any,
    grace_deadline_at: Any,
    launched_at: Any,
    validated_at: Any,
) -> SlotTimes:
    expected = slot_times(slot_index)
    if scheduled_at != expected.scheduled_at or grace_deadline_at != expected.grace_deadline_at:
        _fail()
    scheduled = parse_utc(scheduled_at, receipt=True)
    grace = parse_utc(grace_deadline_at, receipt=True)
    launched = parse_utc(launched_at, receipt=True)
    validated = parse_utc(validated_at, receipt=True)
    if not scheduled <= launched <= validated < grace:
        _fail()
    return expected


@dataclass(frozen=True, slots=True)
class HashAndBytes:
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        _hex(self.sha256, 64)
        _nonnegative_int(self.bytes)

    @classmethod
    def from_value(cls, value: Any) -> HashAndBytes:
        mapping = _exact_mapping(value, ("sha256", "bytes"))
        return cls(_hex(mapping["sha256"], 64), _nonnegative_int(mapping["bytes"]))

    def as_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "bytes": self.bytes}


def hash_and_bytes(raw: bytes) -> HashAndBytes:
    if type(raw) is not bytes:
        _fail()
    return HashAndBytes(hashlib.sha256(raw).hexdigest(), len(raw))


def validate_hash_and_bytes(raw: bytes, identity: HashAndBytes | Mapping[str, Any]) -> HashAndBytes:
    declared = identity if type(identity) is HashAndBytes else HashAndBytes.from_value(identity)
    actual = hash_and_bytes(raw)
    if actual.bytes != declared.bytes or not hmac.compare_digest(actual.sha256, declared.sha256):
        _fail()
    return declared


def _stat_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def read_regular_file(
    path: Path | str,
    *,
    expected: HashAndBytes | Mapping[str, Any] | None = None,
    maximum_bytes: int = 2_000_000,
) -> bytes:
    """Read one stable regular final path without following a symlink."""

    if (
        type(path) not in (str, type(Path()))
        or not _is_int(maximum_bytes)
        or maximum_bytes < 0
    ):
        _fail()
    candidate = Path(path)
    if ".." in candidate.parts:
        _fail()

    def reject_symlink_ancestors() -> None:
        absolute = Path(os.path.abspath(candidate))
        current = Path(absolute.anchor)
        for part in absolute.parts[1:-1]:
            current /= part
            try:
                info = os.lstat(current)
            except (OSError, ValueError):
                _fail()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                _fail()

    reject_symlink_ancestors()
    try:
        path_before = os.lstat(candidate)
    except (OSError, ValueError):
        _fail()
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        _fail()
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(candidate, flags)
    except (OSError, ValueError):
        _fail()
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino) != (path_before.st_dev, path_before.st_ino)
            or before.st_size > maximum_bytes
        ):
            _fail()
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            try:
                chunk = os.pread(fd, min(1 << 20, before.st_size - offset), offset)
            except (OSError, ValueError):
                _fail()
            if not chunk:
                _fail()
            chunks.append(chunk)
            offset += len(chunk)
        try:
            after = os.fstat(fd)
            path_after = os.lstat(candidate)
        except (OSError, ValueError):
            _fail()
        if (
            _stat_identity(before) != _stat_identity(after)
            or (after.st_dev, after.st_ino) != (path_after.st_dev, path_after.st_ino)
            or stat.S_ISLNK(path_after.st_mode)
        ):
            _fail()
        reject_symlink_ancestors()
        raw = b"".join(chunks)
        if len(raw) != after.st_size:
            _fail()
    finally:
        try:
            os.close(fd)
        except OSError:
            _fail()
    if expected is not None:
        validate_hash_and_bytes(raw, expected)
    return raw


def read_content_addressed_object(
    namespace_root: Path | str,
    identity: HashAndBytes | Mapping[str, Any],
) -> bytes:
    declared = identity if type(identity) is HashAndBytes else HashAndBytes.from_value(identity)
    if type(namespace_root) not in (str, type(Path())):
        _fail()
    root = Path(namespace_root)
    expected_path = root / "objects" / "sha256" / declared.sha256[:2] / declared.sha256[2:]
    return read_regular_file(expected_path, expected=declared)


def _domain_hash(domain: bytes, value: Any) -> HashAndBytes:
    if type(domain) is not bytes or not domain or b"\0" in domain:
        _fail()
    return hash_and_bytes(domain + b"\0" + canonical_json_bytes(value))


def _validate_file_identity_value(value: Any, *, positive: bool = False) -> None:
    identity = HashAndBytes.from_value(value)
    if positive and identity.bytes == 0:
        _fail()


def _validate_service(value: Any) -> tuple[str, str, datetime]:
    service = _exact_mapping(
        value,
        (
            "service_identity_sha256",
            "boot_identity_sha256",
            "boot_started_at",
            "serving_build",
            "sanitized_configuration",
            "effective_legacy_repeat_controls",
        ),
    )
    service_identity = _hex(service["service_identity_sha256"], 64)
    boot_identity = _hex(service["boot_identity_sha256"], 64)
    boot_started_at = parse_utc(service["boot_started_at"], receipt=True)

    build = _exact_mapping(
        service["serving_build"], ("commit", "tree", "implementation")
    )
    _hex(build["commit"], 40)
    _hex(build["tree"], 40)
    implementation = build["implementation"]
    if type(implementation) is not dict or not implementation:
        _fail()
    for name, identity in implementation.items():
        _utf8(name, nonempty=True)
        _validate_file_identity_value(identity)

    configuration = _exact_mapping(
        service["sanitized_configuration"],
        ("schema", "encoding", "sha256", "bytes"),
    )
    if (
        configuration["schema"] != "living-memory-effective-runtime-configuration-v1"
        or configuration["encoding"] != "canonical-json-utf8"
    ):
        _fail()
    _hex(configuration["sha256"], 64)
    _nonnegative_int(configuration["bytes"])

    controls = _exact_mapping(
        service["effective_legacy_repeat_controls"],
        ("LM_RECALL_REPEAT_GATING", "LM_RECALL_REPEAT_DROP_TRAILING_STUBS"),
    )
    if not _json_exact(
        controls,
        {
            "LM_RECALL_REPEAT_GATING": False,
            "LM_RECALL_REPEAT_DROP_TRAILING_STUBS": False,
        },
    ):
        _fail()
    return service_identity, boot_identity, boot_started_at


@dataclass(frozen=True, slots=True)
class ServiceTuple:
    raw: bytes
    identity: HashAndBytes
    service_pairs: tuple[tuple[str, str], ...]
    boot_started_at_by_pair: tuple[tuple[tuple[str, str], datetime], ...]

    def boot_started_at(self, pair: tuple[str, str]) -> datetime:
        for candidate, instant in self.boot_started_at_by_pair:
            if candidate == pair:
                return instant
        _fail()


def canonical_service_tuple(value: Any) -> ServiceTuple:
    """Validate an unordered service array and return its one canonical tuple."""

    if type(value) is not list or len(value) != len(SOURCE_ALIASES):
        _fail()
    validated: list[tuple[tuple[str, str], datetime, dict[str, Any]]] = []
    for member in value:
        pair_and_time = _validate_service(member)
        # Round-trip to detach caller-owned containers and prove pure JSON types.
        detached = load_json_bytes(canonical_json_bytes(member), expected=dict)
        validated.append((pair_and_time[:2], pair_and_time[2], detached))
    pairs = [pair for pair, _instant, _member in validated]
    if (
        len({service for service, _boot in pairs}) != len(pairs)
        or len({boot for _service, boot in pairs}) != len(pairs)
    ):
        _fail()
    validated.sort(key=lambda item: item[0])
    ordered = [member for _pair, _instant, member in validated]
    raw = release_v1_canonical_json_bytes(ordered)
    return ServiceTuple(
        raw=raw,
        identity=hash_and_bytes(raw),
        service_pairs=tuple(pair for pair, _instant, _member in validated),
        boot_started_at_by_pair=tuple(
            (pair, instant) for pair, instant, _member in validated
        ),
    )


def validate_service_tuple_bytes(raw: bytes) -> ServiceTuple:
    value = load_canonical_json_bytes(raw, expected=list, release_v1=True)
    result = canonical_service_tuple(value)
    if not hmac.compare_digest(raw, result.raw):
        _fail()
    return result


def _require_service_tuple_consistent(value: ServiceTuple) -> ServiceTuple:
    if (
        type(value) is not ServiceTuple
        or type(value.raw) is not bytes
        or type(value.identity) is not HashAndBytes
        or type(value.service_pairs) is not tuple
        or type(value.boot_started_at_by_pair) is not tuple
    ):
        _fail()
    for pair in value.service_pairs:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not str
            or _hex(pair[0], 64) != pair[0]
            or _hex(pair[1], 64) != pair[1]
        ):
            _fail()
    for member in value.boot_started_at_by_pair:
        if (
            type(member) is not tuple
            or len(member) != 2
            or type(member[0]) is not tuple
            or len(member[0]) != 2
            or type(member[0][0]) is not str
            or type(member[0][1]) is not str
            or _hex(member[0][0], 64) != member[0][0]
            or _hex(member[0][1], 64) != member[0][1]
            or member[0] not in value.service_pairs
            or type(member[1]) is not datetime
            or member[1].tzinfo is not UTC
        ):
            _fail()
    canonical = validate_service_tuple_bytes(value.raw)
    if canonical != value:
        _fail()
    return canonical


def _service_observation(value: Any) -> ServiceTuple:
    if type(value) is bytes:
        value = load_json_bytes(value, expected=list)
    return canonical_service_tuple(value)


def validate_stable_service_tuple_observations(
    pre_value: Any,
    post_value: Any,
) -> ServiceTuple:
    """Canonicalize unordered pre/post observations and require full stability."""

    pre = _service_observation(pre_value)
    post = _service_observation(post_value)
    if not hmac.compare_digest(pre.raw, post.raw):
        _fail()
    return pre


def derive_initial_segment_id(
    *,
    attestation_source_sha256: Any,
    lower_bound_exclusive_at: Any,
) -> str:
    _hex(attestation_source_sha256, 64)
    lower = canonical_utc(parse_utc(lower_bound_exclusive_at))
    payload = {
        "segment_index": 0,
        "attestation_source_sha256": attestation_source_sha256,
        "lower_bound_exclusive_at": lower,
    }
    return hashlib.sha256(
        RUNTIME_SEGMENT_DOMAIN + b"\0" + canonical_json_bytes(payload)
    ).hexdigest()


def _validate_control_watermark_document(value: Any) -> ServiceTuple:
    watermark = _exact_mapping(
        value,
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
    )
    if (
        watermark["schema"] != "animal-planet-runtime-provenance-control-watermark"
        or watermark["schema_version"] != 1
        or type(watermark["schema_version"]) is not int
        or watermark["release_id"] != "animal-planet-release-v1"
    ):
        _fail()
    manifest = _exact_mapping(
        watermark["release_manifest"], ("name", "sha256", "bytes")
    )
    if (
        manifest["name"] != "release-manifest.json"
        or manifest["sha256"] != PINNED_RELEASE_MANIFEST_SHA256
        or manifest["bytes"] != PINNED_RELEASE_MANIFEST_BYTES
        or type(manifest["bytes"]) is not int
    ):
        _fail()
    replay = _exact_mapping(
        watermark["replay_code_control"], ("role", "commit", "tree")
    )
    if replay != {
        "role": "replay_code_control_only",
        "commit": REPLAY_CODE_CONTROL_COMMIT,
        "tree": REPLAY_CODE_CONTROL_TREE,
    }:
        _fail()
    selection = _exact_mapping(
        watermark["selection"], ("basis", "lower_bound_exclusive_at")
    )
    if (
        selection["basis"] != "observed_runtime_provenance_instant"
        or selection["lower_bound_exclusive_at"] != RELEASE_EFFECTIVE_AT
    ):
        _fail()
    lower = parse_utc(selection["lower_bound_exclusive_at"], receipt=True)

    provenance = _exact_mapping(
        watermark["runtime_provenance"],
        (
            "identity_derivation",
            "all_event_producing_services_attested",
            "observation",
            "services",
        ),
    )
    if (
        not _json_exact(provenance["identity_derivation"], IDENTITY_DERIVATION)
        or provenance["all_event_producing_services_attested"] is not True
    ):
        _fail()
    services = canonical_service_tuple(provenance["services"])
    if services.identity.sha256 != INITIAL_SERVICES_SHA256:
        _fail()
    # The frozen watermark itself must already use the canonical unordered order.
    if provenance["services"] != load_json_bytes(services.raw, expected=list):
        _fail()
    observation = _exact_mapping(
        provenance["observation"],
        (
            "method",
            "pre_observed_at",
            "pre_runtime_state_sha256",
            "post_observed_at",
            "post_runtime_state_sha256",
        ),
    )
    pre = parse_utc(observation["pre_observed_at"], receipt=True)
    post = parse_utc(observation["post_observed_at"], receipt=True)
    if (
        observation["method"] != "read_only_pre_post_canonical_services_sha256"
        or not pre <= lower <= post
        or not pre < post
        or post - pre > timedelta(seconds=300)
        or observation["pre_runtime_state_sha256"] != services.identity.sha256
        or observation["post_runtime_state_sha256"] != services.identity.sha256
    ):
        _fail()
    if any(instant > pre for _pair, instant in services.boot_started_at_by_pair):
        _fail()
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
        _fail()
    return services


def validate_control_watermark_bytes(raw: bytes) -> ServiceTuple:
    validate_hash_and_bytes(
        raw, HashAndBytes(PINNED_WATERMARK_SHA256, PINNED_WATERMARK_BYTES)
    )
    value = load_canonical_json_bytes(raw, expected=dict, release_v1=True)
    return _validate_control_watermark_document(value)


@dataclass(frozen=True, slots=True)
class FrozenContract:
    plan_identity: HashAndBytes
    watermark_identity: HashAndBytes
    release_effective_at: str
    initial_segment_id: str
    initial_services: ServiceTuple


def _validate_frozen_bootstrap_branches(
    *,
    segments: Mapping[str, Any],
    initial_closure: Mapping[str, Any],
    slot_closure: Mapping[str, Any],
    attempt_marker: Mapping[str, Any],
) -> None:
    """Pin the two mutually exclusive bootstrap branches to the compiled shapes."""

    if (
        segments.get("segment_attestation_allowlist")
        != list(SUCCESSOR_ATTESTATION_FIELDS)
        or segments.get("slot_segment_closure_resolution_allowlist")
        != list(CLOSURE_FIELDS)
        or segments.get("initial_runtime_change_closure_allowlist")
        != list(INITIAL_CLOSURE_FIELDS)
        or segments.get("initial_runtime_change_closure_receipt_kind")
        != INITIAL_CLOSURE_RECEIPT_KIND
        or segments.get("initial_runtime_change_closure_status") != CLOSURE_STATUS
        or segments.get("initial_runtime_change_closure_mismatch_kinds_exactly")
        != [ACTIVE_SERVICES_STATE_CHANGE]
        or segments.get("segment_closure_mismatch_kind_enum")
        != [ACTIVE_SERVICES_STATE_CHANGE, SOURCE_BINDING_CHANGE]
        or segments.get("slot_closure_receipt_kind") != CLOSURE_RECEIPT_KIND
        or segments.get("slot_closure_status") != CLOSURE_STATUS
        or segments.get("initial_bootstrap_branches_mutually_exclusive") is not True
        or type(segments.get("prior_source_binding_core_for_unbound_initial_segment"))
        is not str
    ):
        _fail()
    branches = segments.get("initial_bootstrap_branches_exactly")
    if type(branches) is not list or len(branches) != 2:
        _fail()
    expected_branches = (
        (SAME_TUPLE_BRANCH, "source-binding-attestation"),
        (CHANGED_TUPLE_BRANCH, INITIAL_CLOSURE_RECEIPT_KIND),
    )
    for branch, (name, receipt_kind) in zip(branches, expected_branches, strict=True):
        if type(branch) is not dict:
            _fail()
        if (
            branch.get("branch") != name
            or branch.get("receipt_kind") != receipt_kind
            or branch.get("prior_source_binding_core_exists") is not False
        ):
            _fail()
    if (
        initial_closure.get("fields_exactly") != list(INITIAL_CLOSURE_FIELDS)
        or initial_closure.get("receipt_kind") != INITIAL_CLOSURE_RECEIPT_KIND
        or initial_closure.get("status") != CLOSURE_STATUS
        or initial_closure.get("mismatch_kinds") != [ACTIVE_SERVICES_STATE_CHANGE]
        or initial_closure.get("prior_source_binding_core_member_present") is not False
        or initial_closure.get("prior_source_binding_core_null_or_sentinel_allowed")
        is not False
        or initial_closure.get("requires_active_segment_object") is not False
        or initial_closure.get("fabricated_predecessor_active_segment_allowed")
        is not False
        or initial_closure.get("valid_after_initial_source_binding_attestation")
        is not False
        or initial_closure.get("count_free") is not True
        or initial_closure.get("carries_counts_snapshots_or_keys") is not False
        or initial_closure.get("reads_source_rows") is not False
        or initial_closure.get("grants_authority") is not False
        or initial_closure.get("reusable_in_any_other_slot") is not False
        or initial_closure.get("slot_index") != 0
        or type(initial_closure.get("slot_index")) is not int
        or initial_closure.get("segment_index") != 0
        or type(initial_closure.get("segment_index")) is not int
        or initial_closure.get("consumes_slot_index") != 0
        or initial_closure.get("next_probe_slot_index") != 1
    ):
        _fail()
    # The absent prior core is a schema fact, not a spelling the receipt may use.
    if "prior_source_binding_core_sha256_and_bytes" in list(INITIAL_CLOSURE_FIELDS):
        _fail()
    if (
        slot_closure.get("fields_exactly") != list(CLOSURE_FIELDS)
        or slot_closure.get("receipt_kind") != CLOSURE_RECEIPT_KIND
        or slot_closure.get("status") != CLOSURE_STATUS
        or slot_closure.get("prior_source_binding_core_required") is not True
        or slot_closure.get("valid_before_initial_source_binding_attestation")
        is not False
    ):
        _fail()
    if (
        attempt_marker.get("fields_exactly") != list(ATTEMPT_MARKER_FIELDS)
        or attempt_marker.get("receipt_kind") != "runtime-attestation-attempt"
        or attempt_marker.get("attempt_scope_enum")
        != [INITIAL_SOURCE_BINDING_SCOPE, SUCCESSOR_SEGMENT_SCOPE]
    ):
        _fail()


def _validate_frozen_plan(value: Any, watermark_identity: HashAndBytes) -> None:
    plan = value if type(value) is dict else _fail()
    try:
        domains = plan["domains"]
        release = plan["release_control_watermark"]
        schedule = plan["schedule"]
        segments = plan["runtime_segments"]
        sources = plan["sources"]
        schemas = plan["operational_receipt_schemas"]
        common = schemas["common_validation"]
        initial_closure = schemas["initial_runtime_change_closure_resolution"]
        slot_closure = schemas["slot_segment_closure_resolution"]
        attempt_marker = schemas["successor_attestation_attempt_marker"]
        replay_safety = plan["cross_protocol_replay_safety"]
    except (KeyError, TypeError):
        _fail()
    if not all(
        type(member) is dict
        for member in (
            domains,
            release,
            schedule,
            segments,
            sources,
            common,
            initial_closure,
            slot_closure,
            attempt_marker,
            replay_safety,
        )
    ):
        _fail()
    if (
        plan.get("schema_version") != SCHEMA_VERSION
        or type(plan.get("schema_version")) is not int
        or plan.get("namespace") != NAMESPACE
        or plan.get("frozen") is not True
        or plan.get("protocol_id") != "confirmatory-holdout-v4-protocol-v1"
    ):
        _fail()
    if (
        domains.get("runtime_segment_utf8") != RUNTIME_SEGMENT_DOMAIN.decode()
        or domains.get("source_database_identity_utf8")
        != SOURCE_DATABASE_IDENTITY_DOMAIN.decode()
        or domains.get("alias_service_database_binding_utf8")
        != ALIAS_SERVICE_DATABASE_BINDING_DOMAIN.decode()
        or domains.get("source_binding_core_utf8") != SOURCE_BINDING_CORE_DOMAIN.decode()
        or domains.get("slot_receipt_utf8") != SLOT_RECEIPT_DOMAIN.decode()
        or domains.get("initial_segment_id_value") != INITIAL_SEGMENT_ID
        or domains.get("retired_domain_prefixes") != list(RETIRED_DOMAIN_PREFIXES)
        or domains.get("retired_domains_reusable") is not False
    ):
        _fail()
    # Every active separator must carry the v4 prefix, so no retired preimage
    # can ever collide with a v4 one.
    for key, separator in domains.items():
        if type(key) is not str or not key.endswith("_utf8"):
            continue
        if key in NON_DOMAIN_ENCODING_KEYS:
            continue
        if type(separator) is not str or not separator.startswith(ACTIVE_DOMAIN_PREFIX):
            _fail()
        if any(separator.startswith(prefix) for prefix in RETIRED_DOMAIN_PREFIXES):
            _fail()
    if (
        replay_safety.get("retired_namespaces_exactly") != list(RETIRED_NAMESPACES)
        or replay_safety.get("retired_domain_prefixes_exactly")
        != list(RETIRED_DOMAIN_PREFIXES)
        or replay_safety.get("retired_artifact_accepted_by_v4_validator") is not False
        or replay_safety.get("v4_artifact_accepted_by_retired_validator") is not False
        or replay_safety.get("retired_identity_reuse_allowed") is not False
    ):
        _fail()
    if (
        release.get("sha256") != watermark_identity.sha256
        or release.get("bytes") != watermark_identity.bytes
        or release.get("release_effective_at") != RELEASE_EFFECTIVE_AT
        or release.get("schema")
        != "animal-planet-runtime-provenance-control-watermark"
        or release.get("schema_version") != 1
    ):
        _fail()
    if schedule != {
        **schedule,
        "anchor_at": ANCHOR_AT,
        "cadence_seconds": CADENCE_SECONDS,
        "first_slot_index": FIRST_SLOT_INDEX,
        "last_slot_index": LAST_SLOT_INDEX,
        "slot_count": SLOT_COUNT,
        "grace_seconds": GRACE_SECONDS,
        "absolute_horizon_expires_at": ABSOLUTE_HORIZON_AT,
    }:
        _fail()
    if not all(
        type(schedule.get(field)) is int
        for field in (
            "cadence_seconds",
            "first_slot_index",
            "last_slot_index",
            "slot_count",
            "grace_seconds",
        )
    ):
        _fail()
    if (
        schedule.get("first_slot_at") != ANCHOR_AT
        or schedule.get("final_selection_slot_at")
        != canonical_utc(parse_utc(slot_times(LAST_SLOT_INDEX).scheduled_at, receipt=True)).replace(
            ".000000Z", "Z"
        )
        or schedule.get("window_lower_inclusive") is not True
        or schedule.get("window_upper_exclusive") is not True
        or schedule.get("horizon_resets_on_segment_change") is not False
    ):
        _fail()
    initial = segments.get("initial_segment", {})
    if (
        initial.get("segment_index") != 0
        or type(initial.get("segment_index")) is not int
        or initial.get("segment_id") != INITIAL_SEGMENT_ID
        or initial.get("lower_bound_exclusive_at") != RELEASE_EFFECTIVE_AT
        or initial.get("active_services_state_sha256") != INITIAL_SERVICES_SHA256
        or initial.get("watermark_services_state_sha256") != INITIAL_SERVICES_SHA256
        or initial.get("service_count") != 2
        or initial.get("all_event_producing_services_attested") is not True
        or segments.get("tuple_order_cannot_hide_change") is not True
        or segments.get("cross_segment_pooling_allowed") is not False
    ):
        _fail()
    _validate_frozen_bootstrap_branches(
        segments=segments,
        initial_closure=initial_closure,
        slot_closure=slot_closure,
        attempt_marker=attempt_marker,
    )
    if (
        sources.get("source_aliases_exactly") != list(SOURCE_ALIASES)
        or sources.get("source_binding_core", {}).get("fields_exactly")
        != [
            "alias_ids_by_alias",
            "database_instance_identity_sha256_by_alias",
            "alias_service_database_binding_sha256_by_alias",
            "active_services_state_sha256",
        ]
        or sources.get("source_binding_attestation_allowlist")
        != list(SOURCE_BINDING_ATTESTATION_FIELDS)
        or sources.get("source_binding_core", {}).get("alias_ids_by_alias") != ALIAS_IDS
        or common.get("constants")
        != {"schema_version": SCHEMA_VERSION, "namespace": NAMESPACE}
        or common.get("duplicate_json_key_action") != "invalid"
        or common.get("recursive_unknown_members_action") != "invalid at every nested depth"
    ):
        _fail()
    derived = derive_initial_segment_id(
        attestation_source_sha256=watermark_identity.sha256,
        lower_bound_exclusive_at=RELEASE_EFFECTIVE_AT,
    )
    if derived != INITIAL_SEGMENT_ID:
        _fail()


def load_frozen_contract(
    *,
    plan_path: Path | str = DEFAULT_PLAN,
    watermark_path: Path | str = DEFAULT_WATERMARK,
) -> FrozenContract:
    plan_identity = HashAndBytes(PINNED_PLAN_SHA256, PINNED_PLAN_BYTES)
    watermark_identity = HashAndBytes(PINNED_WATERMARK_SHA256, PINNED_WATERMARK_BYTES)
    plan_raw = read_regular_file(plan_path, expected=plan_identity, maximum_bytes=PINNED_PLAN_BYTES)
    watermark_raw = read_regular_file(
        watermark_path,
        expected=watermark_identity,
        maximum_bytes=PINNED_WATERMARK_BYTES,
    )
    plan = load_json_bytes(plan_raw, expected=dict)
    services = validate_control_watermark_bytes(watermark_raw)
    _validate_frozen_plan(plan, watermark_identity)
    contract = FrozenContract(
        plan_identity=plan_identity,
        watermark_identity=watermark_identity,
        release_effective_at=RELEASE_EFFECTIVE_AT,
        initial_segment_id=INITIAL_SEGMENT_ID,
        initial_services=services,
    )
    _require_frozen_contract_consistent(contract)
    return contract


def _require_frozen_contract_consistent(contract: FrozenContract) -> None:
    """Rebind a caller-supplied frozen-contract wrapper to the compiled pins."""

    if (
        type(contract) is not FrozenContract
        or type(contract.plan_identity) is not HashAndBytes
        or type(contract.watermark_identity) is not HashAndBytes
        or type(contract.release_effective_at) is not str
        or type(contract.initial_segment_id) is not str
        or type(contract.initial_services) is not ServiceTuple
    ):
        _fail()
    services = _require_service_tuple_consistent(contract.initial_services)
    if (
        contract.plan_identity
        != HashAndBytes(PINNED_PLAN_SHA256, PINNED_PLAN_BYTES)
        or contract.watermark_identity
        != HashAndBytes(PINNED_WATERMARK_SHA256, PINNED_WATERMARK_BYTES)
        or contract.release_effective_at != RELEASE_EFFECTIVE_AT
        or contract.initial_segment_id != INITIAL_SEGMENT_ID
        or services.identity.sha256 != INITIAL_SERVICES_SHA256
        or derive_initial_segment_id(
            attestation_source_sha256=contract.watermark_identity.sha256,
            lower_bound_exclusive_at=contract.release_effective_at,
        )
        != contract.initial_segment_id
    ):
        _fail()


def derive_database_instance_identity(value: Any) -> str:
    private_input = _exact_mapping(
        value,
        ("filesystem_uuid", "statx_inode_uint64", "statx_birthtime_ns_int64"),
    )
    filesystem_uuid = private_input["filesystem_uuid"]
    _utf8(filesystem_uuid, nonempty=True)
    if UUID_PATTERN.fullmatch(filesystem_uuid) is None:
        _fail()
    try:
        parsed_uuid = uuid.UUID(filesystem_uuid)
    except (ValueError, AttributeError):
        _fail()
    if parsed_uuid.int == 0 or str(parsed_uuid) != filesystem_uuid:
        _fail()
    _bounded_int(private_input["statx_inode_uint64"], 1, (1 << 64) - 1)
    _bounded_int(private_input["statx_birthtime_ns_int64"], 1, (1 << 63) - 1)
    return hashlib.sha256(
        SOURCE_DATABASE_IDENTITY_DOMAIN
        + b"\0"
        + canonical_json_bytes(private_input)
    ).hexdigest()


def _validate_authority(alias: str, value: Any) -> dict[str, Any]:
    if alias == "local":
        authority = _exact_mapping(
            value,
            ("transport", "kernel_uid_uint32", "process_user_namespace_inode_uint64"),
        )
        if authority["transport"] != "local":
            _fail()
        _bounded_int(authority["kernel_uid_uint32"], 0, (1 << 32) - 1)
        _bounded_int(authority["process_user_namespace_inode_uint64"], 1, (1 << 64) - 1)
        return authority
    if alias != "alt":
        _fail()
    authority = _exact_mapping(
        value,
        (
            "transport",
            "ssh_host_key_algorithm",
            "ssh_host_key_sha256_base64",
            "remote_kernel_uid_uint32",
            "remote_process_user_namespace_inode_uint64",
        ),
    )
    if (
        authority["transport"] != "ssh"
        or authority["ssh_host_key_algorithm"]
        not in (
            "ssh-ed25519",
            "ecdsa-sha2-nistp256",
            "rsa-sha2-512",
            "rsa-sha2-256",
        )
    ):
        _fail()
    fingerprint = authority["ssh_host_key_sha256_base64"]
    raw_fingerprint = _utf8(fingerprint, nonempty=True)
    if b"=" in raw_fingerprint:
        _fail()
    try:
        decoded = base64.b64decode(raw_fingerprint + b"=", validate=True)
    except (binascii.Error, ValueError):
        _fail()
    if (
        len(decoded) != 32
        or base64.b64encode(decoded).rstrip(b"=") != raw_fingerprint
    ):
        _fail()
    _bounded_int(authority["remote_kernel_uid_uint32"], 0, (1 << 32) - 1)
    _bounded_int(
        authority["remote_process_user_namespace_inode_uint64"], 1, (1 << 64) - 1
    )
    return authority


def derive_alias_service_database_binding(
    alias: Any,
    *,
    alias_id: Any,
    authenticated_authority_identity: Any,
    service_identity_sha256: Any,
    boot_identity_sha256: Any,
    database_instance_identity_sha256: Any,
) -> str:
    if type(alias) is not str or alias not in SOURCE_ALIASES:
        _fail()
    if alias_id != ALIAS_IDS[alias]:
        _fail()
    authority = _validate_authority(alias, authenticated_authority_identity)
    payload = {
        "alias_id": alias_id,
        "authenticated_authority_identity": authority,
        "service_identity_sha256": _hex(service_identity_sha256, 64),
        "boot_identity_sha256": _hex(boot_identity_sha256, 64),
        "database_instance_identity_sha256": _hex(
            database_instance_identity_sha256, 64
        ),
    }
    return hashlib.sha256(
        ALIAS_SERVICE_DATABASE_BINDING_DOMAIN
        + b"\0"
        + canonical_json_bytes(payload)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class RetainedSourceBindingCore:
    raw: bytes
    identity: HashAndBytes
    active_services_state_sha256: str
    database_by_alias: tuple[tuple[str, str], ...]
    binding_by_alias: tuple[tuple[str, str], ...]

    def database_map(self) -> dict[str, str]:
        return dict(self.database_by_alias)

    def binding_map(self) -> dict[str, str]:
        return dict(self.binding_by_alias)


def _source_binding_core(
    *,
    active_services_state_sha256: Any,
    database_by_alias: Mapping[str, Any],
    binding_by_alias: Mapping[str, Any],
) -> RetainedSourceBindingCore:
    active_sha = _hex(active_services_state_sha256, 64)
    if type(database_by_alias) is not dict or set(database_by_alias) != set(SOURCE_ALIASES):
        _fail()
    if type(binding_by_alias) is not dict or set(binding_by_alias) != set(SOURCE_ALIASES):
        _fail()
    databases = {alias: _hex(database_by_alias[alias], 64) for alias in SOURCE_ALIASES}
    bindings = {alias: _hex(binding_by_alias[alias], 64) for alias in SOURCE_ALIASES}
    if len(set(databases.values())) != 2 or len(set(bindings.values())) != 2:
        _fail()
    value = {
        "alias_ids_by_alias": dict(ALIAS_IDS),
        "database_instance_identity_sha256_by_alias": databases,
        "alias_service_database_binding_sha256_by_alias": bindings,
        "active_services_state_sha256": active_sha,
    }
    raw = SOURCE_BINDING_CORE_DOMAIN + b"\0" + canonical_json_bytes(value)
    return RetainedSourceBindingCore(
        raw=raw,
        identity=hash_and_bytes(raw),
        active_services_state_sha256=active_sha,
        database_by_alias=tuple((alias, databases[alias]) for alias in SOURCE_ALIASES),
        binding_by_alias=tuple((alias, bindings[alias]) for alias in SOURCE_ALIASES),
    )


def validate_source_binding_core_bytes(raw: bytes) -> RetainedSourceBindingCore:
    if type(raw) is not bytes:
        _fail()
    prefix = SOURCE_BINDING_CORE_DOMAIN + b"\0"
    if not raw.startswith(prefix):
        _fail()
    value = load_canonical_json_bytes(raw[len(prefix) :], expected=dict)
    core = _exact_mapping(
        value,
        (
            "alias_ids_by_alias",
            "database_instance_identity_sha256_by_alias",
            "alias_service_database_binding_sha256_by_alias",
            "active_services_state_sha256",
        ),
    )
    if core["alias_ids_by_alias"] != ALIAS_IDS:
        _fail()
    result = _source_binding_core(
        active_services_state_sha256=core["active_services_state_sha256"],
        database_by_alias=core["database_instance_identity_sha256_by_alias"],
        binding_by_alias=core["alias_service_database_binding_sha256_by_alias"],
    )
    if not hmac.compare_digest(raw, result.raw):
        _fail()
    return result


@dataclass(frozen=True, slots=True)
class SourceBinding:
    core: RetainedSourceBindingCore
    service_pair_by_alias: tuple[tuple[str, tuple[str, str]], ...]
    authority_bytes_by_alias: tuple[tuple[str, bytes], ...]

    def service_map(self) -> dict[str, tuple[str, str]]:
        return dict(self.service_pair_by_alias)


def validate_source_binding_integrity(
    binding: SourceBinding,
    services: ServiceTuple | None = None,
) -> None:
    """Recompute every in-memory private binding member from its retained proof."""

    if (
        type(binding) is not SourceBinding
        or type(binding.core) is not RetainedSourceBindingCore
        or type(binding.core.raw) is not bytes
        or type(binding.core.identity) is not HashAndBytes
        or type(binding.core.active_services_state_sha256) is not str
        or _hex(binding.core.active_services_state_sha256, 64)
        != binding.core.active_services_state_sha256
        or type(binding.core.database_by_alias) is not tuple
        or type(binding.core.binding_by_alias) is not tuple
        or type(binding.service_pair_by_alias) is not tuple
        or type(binding.authority_bytes_by_alias) is not tuple
        or (services is not None and type(services) is not ServiceTuple)
    ):
        _fail()
    for members in (binding.core.database_by_alias, binding.core.binding_by_alias):
        if len(members) != len(SOURCE_ALIASES):
            _fail()
        for member in members:
            if (
                type(member) is not tuple
                or len(member) != 2
                or type(member[0]) is not str
                or member[0] not in SOURCE_ALIASES
                or type(member[1]) is not str
                or _hex(member[1], 64) != member[1]
            ):
                _fail()
    for member in binding.service_pair_by_alias:
        if (
            type(member) is not tuple
            or len(member) != 2
            or type(member[0]) is not str
            or member[0] not in SOURCE_ALIASES
            or type(member[1]) is not tuple
            or len(member[1]) != 2
            or type(member[1][0]) is not str
            or type(member[1][1]) is not str
            or _hex(member[1][0], 64) != member[1][0]
            or _hex(member[1][1], 64) != member[1][1]
        ):
            _fail()
    for member in binding.authority_bytes_by_alias:
        if (
            type(member) is not tuple
            or len(member) != 2
            or type(member[0]) is not str
            or member[0] not in SOURCE_ALIASES
            or type(member[1]) is not bytes
        ):
            _fail()
    if services is not None:
        _require_service_tuple_consistent(services)
    core = validate_source_binding_core_bytes(binding.core.raw)
    if core != binding.core:
        _fail()
    if (
        len(binding.service_pair_by_alias) != len(SOURCE_ALIASES)
        or {alias for alias, _pair in binding.service_pair_by_alias}
        != set(SOURCE_ALIASES)
        or len(binding.authority_bytes_by_alias) != len(SOURCE_ALIASES)
        or {alias for alias, _raw in binding.authority_bytes_by_alias}
        != set(SOURCE_ALIASES)
    ):
        _fail()
    service_map = binding.service_map()
    authority_map = dict(binding.authority_bytes_by_alias)
    database_map = binding.core.database_map()
    declared_bindings = binding.core.binding_map()
    if len(set(service_map.values())) != len(SOURCE_ALIASES):
        _fail()
    for alias in SOURCE_ALIASES:
        pair = service_map[alias]
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or _hex(pair[0], 64) != pair[0]
            or _hex(pair[1], 64) != pair[1]
        ):
            _fail()
        authority_raw = authority_map[alias]
        authority = load_canonical_json_bytes(authority_raw, expected=dict)
        _validate_authority(alias, authority)
        expected_binding = derive_alias_service_database_binding(
            alias,
            alias_id=ALIAS_IDS[alias],
            authenticated_authority_identity=authority,
            service_identity_sha256=pair[0],
            boot_identity_sha256=pair[1],
            database_instance_identity_sha256=database_map[alias],
        )
        if expected_binding != declared_bindings[alias]:
            _fail()
    if services is not None and (
        sorted(service_map.values()) != list(services.service_pairs)
        or binding.core.active_services_state_sha256 != services.identity.sha256
    ):
        _fail()


def validate_source_binding_observation(
    services: ServiceTuple,
    value: Any,
) -> SourceBinding:
    """Validate one private, row-blind alias/authority/database observation."""

    if type(services) is not ServiceTuple:
        _fail()
    _require_service_tuple_consistent(services)
    observations = _exact_mapping(value, SOURCE_ALIASES)
    databases: dict[str, str] = {}
    bindings: dict[str, str] = {}
    service_map: dict[str, tuple[str, str]] = {}
    authorities: dict[str, bytes] = {}
    for alias in SOURCE_ALIASES:
        member = _exact_mapping(
            observations[alias],
            (
                "alias_id",
                "authenticated_authority_identity",
                "service_identity_sha256",
                "boot_identity_sha256",
                "database_instance_identity",
            ),
        )
        if member["alias_id"] != ALIAS_IDS[alias]:
            _fail()
        pair = (
            _hex(member["service_identity_sha256"], 64),
            _hex(member["boot_identity_sha256"], 64),
        )
        database_sha = derive_database_instance_identity(
            member["database_instance_identity"]
        )
        binding_sha = derive_alias_service_database_binding(
            alias,
            alias_id=member["alias_id"],
            authenticated_authority_identity=member["authenticated_authority_identity"],
            service_identity_sha256=pair[0],
            boot_identity_sha256=pair[1],
            database_instance_identity_sha256=database_sha,
        )
        service_map[alias] = pair
        databases[alias] = database_sha
        bindings[alias] = binding_sha
        authorities[alias] = canonical_json_bytes(
            _validate_authority(alias, member["authenticated_authority_identity"])
        )
    if sorted(service_map.values()) != list(services.service_pairs):
        _fail()
    core = _source_binding_core(
        active_services_state_sha256=services.identity.sha256,
        database_by_alias=databases,
        binding_by_alias=bindings,
    )
    return SourceBinding(
        core=core,
        service_pair_by_alias=tuple((alias, service_map[alias]) for alias in SOURCE_ALIASES),
        authority_bytes_by_alias=tuple((alias, authorities[alias]) for alias in SOURCE_ALIASES),
    )


def validate_stable_source_binding_observations(
    services: ServiceTuple,
    pre_value: Any,
    post_value: Any,
) -> SourceBinding:
    pre = validate_source_binding_observation(services, pre_value)
    post = validate_source_binding_observation(services, post_value)
    if (
        pre.service_pair_by_alias != post.service_pair_by_alias
        or pre.authority_bytes_by_alias != post.authority_bytes_by_alias
        or not hmac.compare_digest(pre.core.raw, post.core.raw)
    ):
        _fail()
    validate_source_binding_integrity(pre, services)
    return pre


def validate_binding_transition(
    prior_services: ServiceTuple,
    prior_binding: SourceBinding,
    observed_services: ServiceTuple,
    observed_binding: SourceBinding,
) -> tuple[str, ...]:
    """Classify a complete change while rejecting same-tuple alias swaps."""

    if not all(
        type(item) is expected
        for item, expected in (
            (prior_services, ServiceTuple),
            (prior_binding, SourceBinding),
            (observed_services, ServiceTuple),
            (observed_binding, SourceBinding),
        )
    ):
        _fail()
    validate_source_binding_integrity(prior_binding, prior_services)
    validate_source_binding_integrity(observed_binding, observed_services)
    if prior_binding.core.active_services_state_sha256 != prior_services.identity.sha256:
        _fail()
    if observed_binding.core.active_services_state_sha256 != observed_services.identity.sha256:
        _fail()
    if prior_binding.authority_bytes_by_alias != observed_binding.authority_bytes_by_alias:
        _fail()
    services_changed = not hmac.compare_digest(prior_services.raw, observed_services.raw)
    binding_changed = not hmac.compare_digest(prior_binding.core.raw, observed_binding.core.raw)
    prior_alias_by_pair = {
        pair: alias for alias, pair in prior_binding.service_pair_by_alias
    }
    observed_alias_by_pair = {
        pair: alias for alias, pair in observed_binding.service_pair_by_alias
    }
    for pair in set(prior_alias_by_pair).intersection(observed_alias_by_pair):
        if prior_alias_by_pair[pair] != observed_alias_by_pair[pair]:
            # A surviving service may not move between the fixed authorities.
            _fail()
    prior_alias_by_database = {
        database_sha: alias
        for alias, database_sha in prior_binding.core.database_by_alias
    }
    observed_alias_by_database = {
        database_sha: alias
        for alias, database_sha in observed_binding.core.database_by_alias
    }
    for database_sha in set(prior_alias_by_database).intersection(
        observed_alias_by_database
    ):
        if (
            prior_alias_by_database[database_sha]
            != observed_alias_by_database[database_sha]
        ):
            # A retained database identity may not move between fixed aliases.
            _fail()
    kinds: list[str] = []
    if services_changed:
        kinds.append("active-services-state-change")
    if binding_changed:
        kinds.append("source-binding-change")
    if not kinds:
        _fail()
    return tuple(sorted(kinds, key=lambda item: item.encode("utf-8")))


def validate_pre_binding_transition(
    watermark_services: ServiceTuple,
    observed_services: ServiceTuple,
) -> tuple[str, ...]:
    """Classify the sole change provable before any source binding exists.

    This deliberately takes no binding argument at all.  Before the initial
    source-binding attestation there is no prior source-binding core anywhere in
    the protocol, so ``source-binding-change`` is unprovable and the only honest
    classification is the single-element active-services change.
    """

    if (
        type(watermark_services) is not ServiceTuple
        or type(observed_services) is not ServiceTuple
    ):
        _fail()
    _require_service_tuple_consistent(watermark_services)
    _require_service_tuple_consistent(observed_services)
    if hmac.compare_digest(watermark_services.raw, observed_services.raw):
        # An equal tuple is the same-tuple initial-binding branch, never a closure.
        _fail()
    if watermark_services.identity == observed_services.identity:
        _fail()
    return (ACTIVE_SERVICES_STATE_CHANGE,)


def _identity_value(value: HashAndBytes | Mapping[str, Any]) -> HashAndBytes:
    return value if type(value) is HashAndBytes else HashAndBytes.from_value(value)


def _validate_alias_sha_map(value: Any) -> dict[str, str]:
    mapping = _exact_mapping(value, SOURCE_ALIASES)
    return {alias: _hex(mapping[alias], 64) for alias in SOURCE_ALIASES}


def _validate_alias_timestamp_map(value: Any) -> dict[str, datetime]:
    mapping = _exact_mapping(value, SOURCE_ALIASES)
    return {
        alias: parse_utc(mapping[alias], receipt=True) for alias in SOURCE_ALIASES
    }


def _load_receipt(raw: bytes, fields: Sequence[str]) -> dict[str, Any]:
    receipt = load_canonical_json_bytes(raw, expected=dict)
    _exact_mapping(receipt, fields)
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or type(receipt["schema_version"]) is not int
        or receipt["namespace"] != NAMESPACE
    ):
        _fail()
    return receipt


def _same_identity(
    value: Any,
    expected: HashAndBytes | Mapping[str, Any],
) -> HashAndBytes:
    declared = HashAndBytes.from_value(value)
    target = _identity_value(expected)
    if declared != target:
        _fail()
    return declared


@dataclass(frozen=True, slots=True)
class AttestationAttempt:
    raw: bytes
    identity: HashAndBytes
    attempt_scope: str
    segment_index: int
    slot_index: int | None
    predecessor_closure_sha256: str | None
    authorized_at: datetime
    written_at: datetime
    start_deadline_at: datetime
    previous_ledger_entry_sha256: str
    runtime_observer_identity: HashAndBytes
    analysis_plan_identity: HashAndBytes


@dataclass(frozen=True, slots=True)
class SourceBindingAttestation:
    raw: bytes
    identity: HashAndBytes
    segment_index: int
    attempt: AttestationAttempt
    binding: SourceBinding
    pre_observed_at: datetime
    post_observed_at: datetime
    previous_ledger_entry_sha256: str
    runtime_observer_identity: HashAndBytes
    analysis_plan_identity: HashAndBytes


@dataclass(frozen=True, slots=True)
class UnboundWatermarkPredecessor:
    """The watermark-attested initial segment before any source binding exists.

    It is intentionally *not* an :class:`ActiveSegment`: it has no source
    binding, no source-binding attestation, and no attestation bytes of its own,
    because none of those exist before the initial ceremony.  Nothing in this
    module can turn it into an ``ActiveSegment`` or read a prior core from it.
    """

    segment_index: int
    segment_id: str
    lower_bound_exclusive_at: str
    services: ServiceTuple
    attestation_identity: HashAndBytes


@dataclass(frozen=True, slots=True)
class ActiveSegment:
    segment_index: int
    segment_id: str
    lower_bound_exclusive_at: str
    services: ServiceTuple
    source_binding: SourceBinding
    attestation_identity: HashAndBytes
    source_binding_attestation: SourceBindingAttestation
    attestation_raw: bytes | None
    attestation_previous_ledger_sha256: str | None
    predecessor_closure: ValidatedClosure | InitialRuntimeChangeClosure | None
    predecessor_segment: ActiveSegment | UnboundWatermarkPredecessor | None


@dataclass(frozen=True, slots=True)
class ValidatedClosure:
    raw: bytes
    identity: HashAndBytes
    slot_index: int
    validated_at: datetime
    segment_index: int
    segment_id: str
    resolution_id: str


@dataclass(frozen=True, slots=True)
class InitialRuntimeChangeClosure:
    """The sole pre-binding closure, consuming slot 0 of an unbound segment.

    It deliberately retains no observed service tuple and no observed
    source-binding core.  The receipt bytes record both as evidence of what was
    observed, but the wrapper exposes nothing a successor could reuse in place
    of its own fresh ceremony, and it never holds a prior source-binding core
    because none has ever existed.
    """

    raw: bytes
    identity: HashAndBytes
    slot_index: int
    validated_at: datetime
    segment_index: int
    segment_id: str
    resolution_id: str
    predecessor: UnboundWatermarkPredecessor
    attempt_identity: HashAndBytes
    observer_identity: HashAndBytes


ATTEMPT_MARKER_FIELDS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "attempt_scope",
    "segment_index",
    "slot_index_or_null",
    "predecessor_closure_sha256_or_null",
    "authorized_at",
    "written_at",
    "start_deadline_at",
    "previous_ledger_entry_sha256",
    "runtime_observer_sha256_and_bytes",
    "analysis_plan_sha256_and_bytes",
)

SOURCE_BINDING_ATTESTATION_FIELDS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "segment_index",
    "attestation_attempt_marker_sha256_and_bytes",
    "alias_ids_by_alias",
    "pre_database_instance_identity_sha256_by_alias",
    "post_database_instance_identity_sha256_by_alias",
    "pre_alias_service_database_binding_sha256_by_alias",
    "post_alias_service_database_binding_sha256_by_alias",
    "active_services_state_sha256",
    "source_binding_core_sha256_and_bytes",
    "status",
    "pre_observed_at",
    "post_observed_at",
    "runtime_observer_sha256_and_bytes",
    "analysis_plan_sha256_and_bytes",
    "previous_ledger_entry_sha256",
)

CLOSURE_FIELDS = (
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
    "prior_source_binding_core_sha256_and_bytes",
    "observed_source_binding_core_sha256_and_bytes",
    "attestation_sha256_and_bytes",
    "previous_ledger_entry_sha256",
    "runtime_observer_sha256_and_bytes",
    "analysis_plan_sha256_and_bytes",
)

# The pre-binding closure carries the slot-0 attempt marker instead of a prior
# source-binding core.  The prior member is absent, not nullable: the exact
# field set below is what makes every spelling of a fabricated prior core fail.
INITIAL_CLOSURE_FIELDS = (
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
)

SUCCESSOR_ATTESTATION_FIELDS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "status",
    "segment_index",
    "segment_id",
    "attestation_attempt_marker_sha256_and_bytes",
    "predecessor_closure_sha256",
    "lower_bound_exclusive_at",
    "pre_launcher_at",
    "pre_source_clock_observed_at_by_alias",
    "boundary_launcher_at",
    "boundary_at",
    "post_phase_started_at",
    "post_source_clock_observed_at_by_alias",
    "post_launcher_at",
    "complete_unaliased_service_tuple_sha256_and_bytes",
    "source_binding_attestation_sha256_and_bytes",
    "active_services_state_sha256",
    "replay_code_control_commit_and_tree",
    "analysis_plan_sha256_and_bytes",
    "previous_ledger_entry_sha256",
)

CLOSURE_TYPES = (ValidatedClosure, InitialRuntimeChangeClosure)
PREDECESSOR_TYPES = (ActiveSegment, UnboundWatermarkPredecessor)


def _require_attempt_wrapper_consistent(attempt: AttestationAttempt) -> None:
    if (
        type(attempt) is not AttestationAttempt
        or type(attempt.raw) is not bytes
        or type(attempt.identity) is not HashAndBytes
        or type(attempt.attempt_scope) is not str
        or type(attempt.segment_index) is not int
        or not (attempt.slot_index is None or type(attempt.slot_index) is int)
        or not (
            attempt.predecessor_closure_sha256 is None
            or type(attempt.predecessor_closure_sha256) is str
        )
        or type(attempt.previous_ledger_entry_sha256) is not str
        or type(attempt.runtime_observer_identity) is not HashAndBytes
        or type(attempt.analysis_plan_identity) is not HashAndBytes
    ):
        _fail()
    if any(
        type(value) is not datetime or value.tzinfo is not UTC
        for value in (
            attempt.authorized_at,
            attempt.written_at,
            attempt.start_deadline_at,
        )
    ):
        _fail()
    receipt = _load_receipt(attempt.raw, ATTEMPT_MARKER_FIELDS)
    predecessor = receipt["predecessor_closure_sha256_or_null"]
    if predecessor is not None:
        predecessor = _hex(predecessor, 64)
    if (
        hash_and_bytes(attempt.raw) != attempt.identity
        or receipt["receipt_kind"] != ATTEMPT_RECEIPT_KIND
        or receipt["attempt_scope"] != attempt.attempt_scope
        or _bounded_int(receipt["segment_index"], 0, LAST_SLOT_INDEX)
        != attempt.segment_index
        or receipt["slot_index_or_null"] != attempt.slot_index
        or predecessor != attempt.predecessor_closure_sha256
        or parse_utc(receipt["authorized_at"], receipt=True) != attempt.authorized_at
        or parse_utc(receipt["written_at"], receipt=True) != attempt.written_at
        or parse_utc(receipt["start_deadline_at"], receipt=True)
        != attempt.start_deadline_at
        or _hex(receipt["previous_ledger_entry_sha256"], 64)
        != _hex(attempt.previous_ledger_entry_sha256, 64)
        or HashAndBytes.from_value(receipt["runtime_observer_sha256_and_bytes"])
        != attempt.runtime_observer_identity
        or HashAndBytes.from_value(receipt["analysis_plan_sha256_and_bytes"])
        != attempt.analysis_plan_identity
    ):
        _fail()
    if attempt.attempt_scope == INITIAL_SOURCE_BINDING_SCOPE:
        if (
            type(attempt.slot_index) is not int
            or attempt.slot_index != 0
            or attempt.segment_index != 0
            or attempt.predecessor_closure_sha256 is not None
            or canonical_utc(attempt.authorized_at) != slot_times(0).scheduled_at
            or canonical_utc(attempt.start_deadline_at)
            != slot_times(0).grace_deadline_at
            or not attempt.authorized_at
            <= attempt.written_at
            < attempt.start_deadline_at
        ):
            _fail()
    elif attempt.attempt_scope == SUCCESSOR_SEGMENT_SCOPE:
        if (
            attempt.slot_index is not None
            or attempt.predecessor_closure_sha256 is None
            or attempt.start_deadline_at != _add_seconds(attempt.authorized_at, 30)
            or not attempt.authorized_at
            <= attempt.written_at
            < attempt.start_deadline_at
        ):
            _fail()
    else:
        _fail()


def _require_source_attestation_wrapper_consistent(
    attestation: SourceBindingAttestation,
) -> None:
    if (
        type(attestation) is not SourceBindingAttestation
        or type(attestation.raw) is not bytes
        or type(attestation.identity) is not HashAndBytes
        or type(attestation.segment_index) is not int
        or type(attestation.attempt) is not AttestationAttempt
        or type(attestation.binding) is not SourceBinding
        or type(attestation.previous_ledger_entry_sha256) is not str
        or type(attestation.runtime_observer_identity) is not HashAndBytes
        or type(attestation.analysis_plan_identity) is not HashAndBytes
    ):
        _fail()
    _require_attempt_wrapper_consistent(attestation.attempt)
    if any(
        type(value) is not datetime or value.tzinfo is not UTC
        for value in (attestation.pre_observed_at, attestation.post_observed_at)
    ):
        _fail()
    validate_source_binding_integrity(attestation.binding)
    if not (
        attestation.attempt.written_at
        <= attestation.pre_observed_at
        < attestation.post_observed_at
    ):
        _fail()
    if attestation.attempt.attempt_scope == INITIAL_SOURCE_BINDING_SCOPE:
        if not attestation.post_observed_at < attestation.attempt.start_deadline_at:
            _fail()
    elif not attestation.pre_observed_at < attestation.attempt.start_deadline_at:
        _fail()
    receipt = _load_receipt(attestation.raw, SOURCE_BINDING_ATTESTATION_FIELDS)
    databases = attestation.binding.core.database_map()
    bindings = attestation.binding.core.binding_map()
    if (
        hash_and_bytes(attestation.raw) != attestation.identity
        or receipt["receipt_kind"] != "source-binding-attestation"
        or receipt["status"] != "pass"
        or _bounded_int(receipt["segment_index"], 0, LAST_SLOT_INDEX)
        != attestation.segment_index
        or attestation.segment_index != attestation.attempt.segment_index
        or HashAndBytes.from_value(
            receipt["attestation_attempt_marker_sha256_and_bytes"]
        )
        != attestation.attempt.identity
        or receipt["alias_ids_by_alias"] != ALIAS_IDS
        or _validate_alias_sha_map(
            receipt["pre_database_instance_identity_sha256_by_alias"]
        )
        != databases
        or _validate_alias_sha_map(
            receipt["post_database_instance_identity_sha256_by_alias"]
        )
        != databases
        or _validate_alias_sha_map(
            receipt["pre_alias_service_database_binding_sha256_by_alias"]
        )
        != bindings
        or _validate_alias_sha_map(
            receipt["post_alias_service_database_binding_sha256_by_alias"]
        )
        != bindings
        or receipt["active_services_state_sha256"]
        != attestation.binding.core.active_services_state_sha256
        or HashAndBytes.from_value(receipt["source_binding_core_sha256_and_bytes"])
        != attestation.binding.core.identity
        or parse_utc(receipt["pre_observed_at"], receipt=True)
        != attestation.pre_observed_at
        or parse_utc(receipt["post_observed_at"], receipt=True)
        != attestation.post_observed_at
        or _hex(receipt["previous_ledger_entry_sha256"], 64)
        != _hex(attestation.previous_ledger_entry_sha256, 64)
        or HashAndBytes.from_value(receipt["runtime_observer_sha256_and_bytes"])
        != attestation.runtime_observer_identity
        or HashAndBytes.from_value(receipt["analysis_plan_sha256_and_bytes"])
        != attestation.analysis_plan_identity
        or attestation.runtime_observer_identity
        != attestation.attempt.runtime_observer_identity
        or attestation.analysis_plan_identity
        != attestation.attempt.analysis_plan_identity
    ):
        _fail()


def validate_attestation_attempt_marker(
    raw: bytes,
    *,
    contract: FrozenContract,
    runtime_observer_identity: HashAndBytes | Mapping[str, Any],
    expected_previous_ledger_sha256: Any,
    initial_probe_launched_at: Any | None = None,
    predecessor_closure: ValidatedClosure | InitialRuntimeChangeClosure | None = None,
) -> AttestationAttempt:
    if type(contract) is not FrozenContract:
        _fail()
    _require_frozen_contract_consistent(contract)
    marker = _load_receipt(raw, ATTEMPT_MARKER_FIELDS)
    if marker["receipt_kind"] != ATTEMPT_RECEIPT_KIND:
        _fail()
    scope = marker["attempt_scope"]
    if scope not in (INITIAL_SOURCE_BINDING_SCOPE, SUCCESSOR_SEGMENT_SCOPE):
        _fail()
    segment_index = _bounded_int(marker["segment_index"], 0, LAST_SLOT_INDEX)
    authorized_at = parse_utc(marker["authorized_at"], receipt=True)
    written = parse_utc(marker["written_at"], receipt=True)
    deadline = parse_utc(marker["start_deadline_at"], receipt=True)
    previous = _hex(marker["previous_ledger_entry_sha256"], 64)
    if previous != _hex(expected_previous_ledger_sha256, 64):
        _fail()
    _same_identity(
        marker["runtime_observer_sha256_and_bytes"], runtime_observer_identity
    )
    _same_identity(marker["analysis_plan_sha256_and_bytes"], contract.plan_identity)

    predecessor_sha: str | None
    if scope == INITIAL_SOURCE_BINDING_SCOPE:
        if predecessor_closure is not None or initial_probe_launched_at is None:
            _fail()
        launched = parse_utc(initial_probe_launched_at, receipt=True)
        slot = slot_times(0)
        scheduled = parse_utc(slot.scheduled_at, receipt=True)
        if (
            segment_index != 0
            or marker["slot_index_or_null"] != 0
            or type(marker["slot_index_or_null"]) is not int
            or marker["predecessor_closure_sha256_or_null"] is not None
            or marker["authorized_at"] != slot.scheduled_at
            or marker["start_deadline_at"] != slot.grace_deadline_at
            or not scheduled <= launched <= written < deadline
            or authorized_at != scheduled
        ):
            _fail()
        predecessor_sha = None
    else:
        if (
            type(predecessor_closure) not in CLOSURE_TYPES
            or initial_probe_launched_at is not None
        ):
            _fail()
        closure = predecessor_closure
        _require_closure_wrapper_consistent(contract, closure)
        if closure.slot_index >= LAST_SLOT_INDEX:
            _fail()
        expected_deadline = _add_seconds(closure.validated_at, 30)
        predecessor_sha = _hex(marker["predecessor_closure_sha256_or_null"], 64)
        if (
            segment_index != closure.segment_index + 1
            or marker["slot_index_or_null"] is not None
            or predecessor_sha != closure.identity.sha256
            or marker["authorized_at"] != canonical_utc(closure.validated_at)
            or marker["start_deadline_at"] != canonical_utc(expected_deadline)
            or not closure.validated_at <= written < deadline
            or deadline != expected_deadline
        ):
            _fail()
        if type(closure) is InitialRuntimeChangeClosure:
            # A successor ceremony is a fresh attempt; it may never re-present
            # the slot-0 initial-source-binding marker the closure consumed.
            if hash_and_bytes(raw) == closure.attempt_identity:
                _fail()
    return AttestationAttempt(
        raw=raw,
        identity=hash_and_bytes(raw),
        attempt_scope=scope,
        segment_index=segment_index,
        slot_index=marker["slot_index_or_null"],
        predecessor_closure_sha256=predecessor_sha,
        authorized_at=authorized_at,
        written_at=written,
        start_deadline_at=deadline,
        previous_ledger_entry_sha256=previous,
        runtime_observer_identity=_identity_value(runtime_observer_identity),
        analysis_plan_identity=contract.plan_identity,
    )


def validate_source_binding_attestation(
    raw: bytes,
    *,
    contract: FrozenContract,
    attempt: AttestationAttempt,
    binding: SourceBinding,
    runtime_observer_identity: HashAndBytes | Mapping[str, Any],
    expected_previous_ledger_sha256: Any,
) -> SourceBindingAttestation:
    if (
        type(contract) is not FrozenContract
        or type(attempt) is not AttestationAttempt
        or type(binding) is not SourceBinding
    ):
        _fail()
    _require_frozen_contract_consistent(contract)
    _require_attempt_wrapper_consistent(attempt)
    validate_source_binding_integrity(binding)
    if (
        attempt.runtime_observer_identity != _identity_value(runtime_observer_identity)
        or attempt.analysis_plan_identity != contract.plan_identity
    ):
        _fail()
    receipt = _load_receipt(raw, SOURCE_BINDING_ATTESTATION_FIELDS)
    if (
        receipt["receipt_kind"] != "source-binding-attestation"
        or receipt["status"] != "pass"
    ):
        _fail()
    segment_index = _bounded_int(receipt["segment_index"], 0, LAST_SLOT_INDEX)
    if segment_index != attempt.segment_index:
        _fail()
    _same_identity(
        receipt["attestation_attempt_marker_sha256_and_bytes"], attempt.identity
    )
    if receipt["alias_ids_by_alias"] != ALIAS_IDS:
        _fail()
    pre_databases = _validate_alias_sha_map(
        receipt["pre_database_instance_identity_sha256_by_alias"]
    )
    post_databases = _validate_alias_sha_map(
        receipt["post_database_instance_identity_sha256_by_alias"]
    )
    pre_bindings = _validate_alias_sha_map(
        receipt["pre_alias_service_database_binding_sha256_by_alias"]
    )
    post_bindings = _validate_alias_sha_map(
        receipt["post_alias_service_database_binding_sha256_by_alias"]
    )
    if (
        pre_databases != post_databases
        or pre_bindings != post_bindings
        or pre_databases != binding.core.database_map()
        or pre_bindings != binding.core.binding_map()
        or len(set(pre_databases.values())) != 2
        or len(set(pre_bindings.values())) != 2
        or receipt["active_services_state_sha256"]
        != binding.core.active_services_state_sha256
    ):
        _fail()
    _same_identity(
        receipt["source_binding_core_sha256_and_bytes"], binding.core.identity
    )
    pre = parse_utc(receipt["pre_observed_at"], receipt=True)
    post = parse_utc(receipt["post_observed_at"], receipt=True)
    if not attempt.written_at <= pre < post:
        _fail()
    if attempt.attempt_scope == INITIAL_SOURCE_BINDING_SCOPE:
        if not post < attempt.start_deadline_at:
            _fail()
    elif not pre < attempt.start_deadline_at:
        _fail()
    _same_identity(
        receipt["runtime_observer_sha256_and_bytes"], runtime_observer_identity
    )
    _same_identity(receipt["analysis_plan_sha256_and_bytes"], contract.plan_identity)
    previous = _hex(receipt["previous_ledger_entry_sha256"], 64)
    if previous != _hex(expected_previous_ledger_sha256, 64):
        _fail()
    return SourceBindingAttestation(
        raw=raw,
        identity=hash_and_bytes(raw),
        segment_index=segment_index,
        attempt=attempt,
        binding=binding,
        pre_observed_at=pre,
        post_observed_at=post,
        previous_ledger_entry_sha256=previous,
        runtime_observer_identity=_identity_value(runtime_observer_identity),
        analysis_plan_identity=contract.plan_identity,
    )


def initial_bootstrap_branch(
    contract: FrozenContract,
    observed_services: ServiceTuple,
) -> str:
    """Select the bootstrap branch from tuple equality and nothing else.

    No operator, outcome, availability, or count may influence this: the branch
    is exactly whether the complete stable observed active-services state equals
    the pinned watermark active-services state.
    """

    if type(contract) is not FrozenContract or type(observed_services) is not ServiceTuple:
        _fail()
    _require_frozen_contract_consistent(contract)
    _require_service_tuple_consistent(observed_services)
    same_bytes = hmac.compare_digest(
        contract.initial_services.raw, observed_services.raw
    )
    same_identity = contract.initial_services.identity == observed_services.identity
    if same_bytes != same_identity:
        # A digest that disagrees with the bytes is forged, not a branch.
        _fail()
    return SAME_TUPLE_BRANCH if same_bytes else CHANGED_TUPLE_BRANCH


def make_unbound_watermark_predecessor(
    contract: FrozenContract,
) -> UnboundWatermarkPredecessor:
    """Build the typed pre-binding predecessor straight from the frozen pins."""

    if type(contract) is not FrozenContract:
        _fail()
    _require_frozen_contract_consistent(contract)
    return UnboundWatermarkPredecessor(
        segment_index=0,
        segment_id=contract.initial_segment_id,
        lower_bound_exclusive_at=contract.release_effective_at,
        services=contract.initial_services,
        attestation_identity=contract.watermark_identity,
    )


def _require_unbound_predecessor_consistent(
    contract: FrozenContract,
    predecessor: UnboundWatermarkPredecessor,
) -> None:
    if (
        type(contract) is not FrozenContract
        or type(predecessor) is not UnboundWatermarkPredecessor
        or type(predecessor.segment_index) is not int
        or type(predecessor.segment_id) is not str
        or type(predecessor.lower_bound_exclusive_at) is not str
        or type(predecessor.services) is not ServiceTuple
        or type(predecessor.attestation_identity) is not HashAndBytes
    ):
        _fail()
    _require_frozen_contract_consistent(contract)
    services = _require_service_tuple_consistent(predecessor.services)
    if (
        predecessor.segment_index != 0
        or predecessor.segment_id != contract.initial_segment_id
        or predecessor.lower_bound_exclusive_at != contract.release_effective_at
        or predecessor.attestation_identity != contract.watermark_identity
        or services.identity != contract.initial_services.identity
        or not hmac.compare_digest(services.raw, contract.initial_services.raw)
    ):
        _fail()


def make_initial_segment(
    contract: FrozenContract,
    source_binding_attestation: SourceBindingAttestation,
) -> ActiveSegment:
    """Open the initial segment from same-tuple evidence alone.

    This is the whole of the ``same-tuple-initial-binding`` branch: no closure,
    no predecessor object, and no prior source-binding core participate.
    """

    if (
        type(contract) is not FrozenContract
        or type(source_binding_attestation) is not SourceBindingAttestation
    ):
        _fail()
    _require_frozen_contract_consistent(contract)
    _require_source_attestation_wrapper_consistent(source_binding_attestation)
    if (
        source_binding_attestation.segment_index != 0
        or source_binding_attestation.attempt.attempt_scope
        != INITIAL_SOURCE_BINDING_SCOPE
        or source_binding_attestation.binding.core.active_services_state_sha256
        != contract.initial_services.identity.sha256
        or source_binding_attestation.analysis_plan_identity != contract.plan_identity
    ):
        _fail()
    validate_source_binding_integrity(
        source_binding_attestation.binding, contract.initial_services
    )
    return ActiveSegment(
        segment_index=0,
        segment_id=contract.initial_segment_id,
        lower_bound_exclusive_at=contract.release_effective_at,
        services=contract.initial_services,
        source_binding=source_binding_attestation.binding,
        attestation_identity=contract.watermark_identity,
        source_binding_attestation=source_binding_attestation,
        attestation_raw=None,
        attestation_previous_ledger_sha256=None,
        predecessor_closure=None,
        predecessor_segment=None,
    )


def derive_slot_resolution_id(value: Any) -> str:
    if type(value) is not dict or "resolution_id" not in value:
        _fail()
    core = dict(value)
    del core["resolution_id"]
    core.pop("artifact_sha256", None)
    return hashlib.sha256(
        SLOT_RECEIPT_DOMAIN + b"\0" + canonical_json_bytes(core)
    ).hexdigest()


def _require_initial_closure_wrapper_consistent(
    contract: FrozenContract,
    closure: InitialRuntimeChangeClosure,
) -> None:
    """Rebind the pre-binding closure wrapper to its own receipt bytes."""

    if (
        type(closure) is not InitialRuntimeChangeClosure
        or type(closure.raw) is not bytes
        or type(closure.identity) is not HashAndBytes
        or type(closure.slot_index) is not int
        or type(closure.segment_index) is not int
        or type(closure.segment_id) is not str
        or type(closure.resolution_id) is not str
        or type(closure.predecessor) is not UnboundWatermarkPredecessor
        or type(closure.attempt_identity) is not HashAndBytes
        or type(closure.observer_identity) is not HashAndBytes
        or type(closure.validated_at) is not datetime
        or closure.validated_at.tzinfo is not UTC
    ):
        _fail()
    _require_unbound_predecessor_consistent(contract, closure.predecessor)
    receipt = _load_receipt(closure.raw, INITIAL_CLOSURE_FIELDS)
    prior = HashAndBytes.from_value(
        receipt["prior_active_services_state_sha256_and_bytes"]
    )
    observed = HashAndBytes.from_value(
        receipt["observed_active_services_state_sha256_and_bytes"]
    )
    if (
        hash_and_bytes(closure.raw) != closure.identity
        or receipt["receipt_kind"] != INITIAL_CLOSURE_RECEIPT_KIND
        or receipt["status"] != CLOSURE_STATUS
        or receipt["slot_index"] != 0
        or type(receipt["slot_index"]) is not int
        or closure.slot_index != 0
        or receipt["segment_index"] != 0
        or type(receipt["segment_index"]) is not int
        or closure.segment_index != 0
        or _hex(receipt["segment_id"], 64) != closure.segment_id
        or closure.segment_id != contract.initial_segment_id
        or parse_utc(receipt["validated_at"], receipt=True) != closure.validated_at
        or _hex(receipt["resolution_id"], 64) != closure.resolution_id
        or closure.resolution_id != derive_slot_resolution_id(receipt)
        or not _json_exact(
            receipt["mismatch_kinds"], [ACTIVE_SERVICES_STATE_CHANGE]
        )
        or prior != contract.initial_services.identity
        or prior == observed
        or prior.sha256 == observed.sha256
        or HashAndBytes.from_value(receipt["attestation_sha256_and_bytes"])
        != contract.watermark_identity
        or HashAndBytes.from_value(
            receipt["attestation_attempt_marker_sha256_and_bytes"]
        )
        != closure.attempt_identity
        or HashAndBytes.from_value(receipt["runtime_observer_sha256_and_bytes"])
        != closure.observer_identity
        or HashAndBytes.from_value(receipt["analysis_plan_sha256_and_bytes"])
        != contract.plan_identity
    ):
        _fail()
    HashAndBytes.from_value(receipt["observed_source_binding_core_sha256_and_bytes"])
    _hex(receipt["previous_ledger_entry_sha256"], 64)
    validate_slot_times(
        slot_index=receipt["slot_index"],
        scheduled_at=receipt["scheduled_at"],
        grace_deadline_at=receipt["grace_deadline_at"],
        launched_at=receipt["launched_at"],
        validated_at=receipt["validated_at"],
    )
    if not parse_utc(
        closure.predecessor.lower_bound_exclusive_at, receipt=True
    ) < parse_utc(receipt["scheduled_at"], receipt=True):
        _fail()


def _require_closure_wrapper_consistent(
    contract: FrozenContract,
    closure: ValidatedClosure | InitialRuntimeChangeClosure,
) -> None:
    """Rebind every successor-relevant closure projection to its receipt bytes."""

    _require_frozen_contract_consistent(contract)
    if type(closure) is InitialRuntimeChangeClosure:
        _require_initial_closure_wrapper_consistent(contract, closure)
        return
    if (
        type(closure) is not ValidatedClosure
        or type(closure.raw) is not bytes
        or type(closure.identity) is not HashAndBytes
        or type(closure.slot_index) is not int
        or type(closure.segment_index) is not int
        or type(closure.segment_id) is not str
        or type(closure.resolution_id) is not str
    ):
        _fail()
    if type(closure.validated_at) is not datetime or closure.validated_at.tzinfo is not UTC:
        _fail()
    receipt = _load_receipt(closure.raw, CLOSURE_FIELDS)
    allowed_kinds = {ACTIVE_SERVICES_STATE_CHANGE, SOURCE_BINDING_CHANGE}
    kinds = receipt["mismatch_kinds"]
    if (
        hash_and_bytes(closure.raw) != closure.identity
        or receipt["receipt_kind"] != CLOSURE_RECEIPT_KIND
        or receipt["status"] != CLOSURE_STATUS
        or _bounded_int(receipt["slot_index"], 0, LAST_SLOT_INDEX)
        != closure.slot_index
        or parse_utc(receipt["validated_at"], receipt=True) != closure.validated_at
        or _bounded_int(receipt["segment_index"], 0, LAST_SLOT_INDEX)
        != closure.segment_index
        or _hex(receipt["segment_id"], 64) != closure.segment_id
        or _hex(receipt["resolution_id"], 64) != closure.resolution_id
        or closure.resolution_id != derive_slot_resolution_id(receipt)
        or type(kinds) is not list
        or not kinds
        or any(type(kind) is not str or kind not in allowed_kinds for kind in kinds)
        or kinds != sorted(set(kinds), key=lambda item: item.encode("utf-8"))
        or HashAndBytes.from_value(receipt["analysis_plan_sha256_and_bytes"])
        != contract.plan_identity
    ):
        _fail()
    for field in (
        "prior_active_services_state_sha256_and_bytes",
        "observed_active_services_state_sha256_and_bytes",
        "prior_source_binding_core_sha256_and_bytes",
        "observed_source_binding_core_sha256_and_bytes",
        "attestation_sha256_and_bytes",
        "runtime_observer_sha256_and_bytes",
    ):
        HashAndBytes.from_value(receipt[field])
    _hex(receipt["previous_ledger_entry_sha256"], 64)
    validate_slot_times(
        slot_index=receipt["slot_index"],
        scheduled_at=receipt["scheduled_at"],
        grace_deadline_at=receipt["grace_deadline_at"],
        launched_at=receipt["launched_at"],
        validated_at=receipt["validated_at"],
    )
    if closure.segment_index > closure.slot_index:
        _fail()


def validate_initial_runtime_change_closure(
    raw: bytes,
    *,
    contract: FrozenContract,
    attempt: AttestationAttempt,
    observed_services: ServiceTuple,
    observed_services_raw: bytes,
    observed_binding: SourceBinding,
    observed_source_binding_core_raw: bytes,
    runtime_observer_identity: HashAndBytes | Mapping[str, Any],
    expected_previous_ledger_sha256: Any,
) -> InitialRuntimeChangeClosure:
    """Validate the sole pre-binding closure of the unbound initial segment.

    Everything it needs is the frozen contract, the one slot-0 attestation
    attempt, and the complete stable observed tuple and binding.  No
    ``ActiveSegment`` is required or constructed, and no prior source-binding
    core is read, inferred, or invented, because none has ever existed.
    """

    if (
        type(contract) is not FrozenContract
        or type(attempt) is not AttestationAttempt
        or type(observed_services) is not ServiceTuple
        or type(observed_binding) is not SourceBinding
    ):
        _fail()
    _require_frozen_contract_consistent(contract)
    _require_attempt_wrapper_consistent(attempt)
    observer_identity = _identity_value(runtime_observer_identity)
    if (
        attempt.attempt_scope != INITIAL_SOURCE_BINDING_SCOPE
        or attempt.segment_index != 0
        or attempt.slot_index != 0
        or attempt.predecessor_closure_sha256 is not None
        or attempt.runtime_observer_identity != observer_identity
        or attempt.analysis_plan_identity != contract.plan_identity
    ):
        _fail()
    predecessor = make_unbound_watermark_predecessor(contract)

    # The retained objects must reproduce the supplied stable observations.
    retained_services = validate_service_tuple_bytes(observed_services_raw)
    retained_core = validate_source_binding_core_bytes(
        observed_source_binding_core_raw
    )
    validate_source_binding_integrity(observed_binding, observed_services)
    if (
        retained_services.identity != observed_services.identity
        or not hmac.compare_digest(retained_services.raw, observed_services.raw)
        or retained_core.identity != observed_binding.core.identity
        or not hmac.compare_digest(retained_core.raw, observed_binding.core.raw)
        or retained_core.active_services_state_sha256
        != retained_services.identity.sha256
    ):
        _fail()
    # The only classification provable without a prior core.
    mismatch_kinds = validate_pre_binding_transition(
        predecessor.services, retained_services
    )

    receipt = _load_receipt(raw, INITIAL_CLOSURE_FIELDS)
    if (
        receipt["receipt_kind"] != INITIAL_CLOSURE_RECEIPT_KIND
        or receipt["status"] != CLOSURE_STATUS
    ):
        _fail()
    supplied_kinds = receipt["mismatch_kinds"]
    if type(supplied_kinds) is not list or supplied_kinds != list(mismatch_kinds):
        _fail()
    _same_identity(
        receipt["prior_active_services_state_sha256_and_bytes"],
        predecessor.services.identity,
    )
    _same_identity(
        receipt["observed_active_services_state_sha256_and_bytes"],
        retained_services.identity,
    )
    _same_identity(
        receipt["observed_source_binding_core_sha256_and_bytes"],
        retained_core.identity,
    )
    _same_identity(receipt["attestation_sha256_and_bytes"], contract.watermark_identity)
    _same_identity(
        receipt["attestation_attempt_marker_sha256_and_bytes"], attempt.identity
    )
    _same_identity(receipt["runtime_observer_sha256_and_bytes"], observer_identity)
    _same_identity(receipt["analysis_plan_sha256_and_bytes"], contract.plan_identity)

    if (
        receipt["slot_index"] != 0
        or type(receipt["slot_index"]) is not int
        or receipt["segment_index"] != 0
        or type(receipt["segment_index"]) is not int
        or receipt["segment_id"] != predecessor.segment_id
    ):
        _fail()
    _hex(receipt["segment_id"], 64)
    validate_slot_times(
        slot_index=receipt["slot_index"],
        scheduled_at=receipt["scheduled_at"],
        grace_deadline_at=receipt["grace_deadline_at"],
        launched_at=receipt["launched_at"],
        validated_at=receipt["validated_at"],
    )
    launched_at = parse_utc(receipt["launched_at"], receipt=True)
    validated_at = parse_utc(receipt["validated_at"], receipt=True)
    if not launched_at <= attempt.written_at <= validated_at:
        _fail()
    if any(
        retained_services.boot_started_at(pair) > validated_at
        for pair in retained_services.service_pairs
    ):
        _fail()
    if not parse_utc(
        predecessor.lower_bound_exclusive_at, receipt=True
    ) < parse_utc(receipt["scheduled_at"], receipt=True):
        _fail()
    previous = _hex(receipt["previous_ledger_entry_sha256"], 64)
    if previous != _hex(expected_previous_ledger_sha256, 64):
        _fail()
    resolution_id = _hex(receipt["resolution_id"], 64)
    if resolution_id != derive_slot_resolution_id(receipt):
        _fail()
    return InitialRuntimeChangeClosure(
        raw=raw,
        identity=hash_and_bytes(raw),
        slot_index=0,
        validated_at=validated_at,
        segment_index=0,
        segment_id=receipt["segment_id"],
        resolution_id=resolution_id,
        predecessor=predecessor,
        attempt_identity=attempt.identity,
        observer_identity=observer_identity,
    )


def validate_segment_closure(
    raw: bytes,
    *,
    contract: FrozenContract,
    active_segment: ActiveSegment,
    prior_services_raw: bytes,
    observed_services_raw: bytes,
    prior_source_binding_core_raw: bytes,
    observed_source_binding_core_raw: bytes,
    observed_binding: SourceBinding,
    runtime_observer_identity: HashAndBytes | Mapping[str, Any],
    expected_previous_ledger_sha256: Any,
) -> ValidatedClosure:
    """Validate one complete, count-free closure against the open segment.

    This is the strict two-core post-binding schema.  It is only reachable once a
    validator-valid source-binding attestation has opened an ``ActiveSegment``;
    before that the sole authorized closure is
    :func:`validate_initial_runtime_change_closure`, and this schema may never be
    satisfied by inventing a prior core.
    """

    if (
        type(contract) is not FrozenContract
        or type(active_segment) is not ActiveSegment
        or type(observed_binding) is not SourceBinding
    ):
        _fail()
    _require_active_segment_consistent(contract, active_segment)
    _require_source_attestation_wrapper_consistent(
        active_segment.source_binding_attestation
    )
    validate_source_binding_integrity(
        active_segment.source_binding, active_segment.services
    )
    observer_identity = _identity_value(runtime_observer_identity)
    if (
        active_segment.source_binding
        != active_segment.source_binding_attestation.binding
        or observer_identity
        != active_segment.source_binding_attestation.runtime_observer_identity
    ):
        _fail()
    receipt = _load_receipt(raw, CLOSURE_FIELDS)
    if (
        receipt["receipt_kind"] != "slot-segment-closed"
        or receipt["status"] != "segment-closed"
    ):
        _fail()

    prior_services = validate_service_tuple_bytes(prior_services_raw)
    observed_services = validate_service_tuple_bytes(observed_services_raw)
    prior_core = validate_source_binding_core_bytes(prior_source_binding_core_raw)
    observed_core = validate_source_binding_core_bytes(observed_source_binding_core_raw)
    if (
        not hmac.compare_digest(prior_services.raw, active_segment.services.raw)
        or prior_services.identity != active_segment.services.identity
        or not hmac.compare_digest(prior_core.raw, active_segment.source_binding.core.raw)
        or prior_core.identity != active_segment.source_binding.core.identity
        or prior_core.active_services_state_sha256 != prior_services.identity.sha256
        or observed_core.active_services_state_sha256 != observed_services.identity.sha256
        or not hmac.compare_digest(observed_core.raw, observed_binding.core.raw)
        or observed_core.identity != observed_binding.core.identity
    ):
        _fail()
    mismatch_kinds = validate_binding_transition(
        prior_services,
        active_segment.source_binding,
        observed_services,
        observed_binding,
    )
    supplied_kinds = receipt["mismatch_kinds"]
    if type(supplied_kinds) is not list or supplied_kinds != list(mismatch_kinds):
        _fail()

    _same_identity(
        receipt["prior_active_services_state_sha256_and_bytes"], prior_services.identity
    )
    _same_identity(
        receipt["observed_active_services_state_sha256_and_bytes"],
        observed_services.identity,
    )
    _same_identity(
        receipt["prior_source_binding_core_sha256_and_bytes"], prior_core.identity
    )
    _same_identity(
        receipt["observed_source_binding_core_sha256_and_bytes"], observed_core.identity
    )
    _same_identity(receipt["attestation_sha256_and_bytes"], active_segment.attestation_identity)
    _same_identity(receipt["runtime_observer_sha256_and_bytes"], observer_identity)
    _same_identity(receipt["analysis_plan_sha256_and_bytes"], contract.plan_identity)

    segment_index = _bounded_int(receipt["segment_index"], 0, LAST_SLOT_INDEX)
    if (
        segment_index != active_segment.segment_index
        or receipt["segment_id"] != active_segment.segment_id
    ):
        _fail()
    _hex(receipt["segment_id"], 64)
    validate_slot_times(
        slot_index=receipt["slot_index"],
        scheduled_at=receipt["scheduled_at"],
        grace_deadline_at=receipt["grace_deadline_at"],
        launched_at=receipt["launched_at"],
        validated_at=receipt["validated_at"],
    )
    if segment_index > _bounded_int(receipt["slot_index"], 0, LAST_SLOT_INDEX):
        _fail()
    validated_at = parse_utc(receipt["validated_at"], receipt=True)
    if any(
        observed_services.boot_started_at(pair) > validated_at
        for pair in observed_services.service_pairs
    ):
        _fail()
    if not parse_utc(active_segment.lower_bound_exclusive_at, receipt=True) < parse_utc(
        receipt["scheduled_at"], receipt=True
    ):
        _fail()
    previous = _hex(receipt["previous_ledger_entry_sha256"], 64)
    if previous != _hex(expected_previous_ledger_sha256, 64):
        _fail()
    resolution_id = _hex(receipt["resolution_id"], 64)
    if resolution_id != derive_slot_resolution_id(receipt):
        _fail()
    return ValidatedClosure(
        raw=raw,
        identity=hash_and_bytes(raw),
        slot_index=_bounded_int(receipt["slot_index"], 0, LAST_SLOT_INDEX),
        validated_at=validated_at,
        segment_index=segment_index,
        segment_id=receipt["segment_id"],
        resolution_id=resolution_id,
    )


def validate_active_segment_observations(
    active_segment: ActiveSegment,
    pre_services: ServiceTuple,
    pre_binding: SourceBinding,
    post_services: ServiceTuple,
    post_binding: SourceBinding,
    *,
    contract: FrozenContract,
) -> None:
    """Prove both sides of a count-bearing capture equal the open segment."""

    if not all(
        type(item) is expected
        for item, expected in (
            (contract, FrozenContract),
            (active_segment, ActiveSegment),
            (pre_services, ServiceTuple),
            (pre_binding, SourceBinding),
            (post_services, ServiceTuple),
            (post_binding, SourceBinding),
        )
    ):
        _fail()
    _require_active_segment_consistent(contract, active_segment)
    _require_source_attestation_wrapper_consistent(
        active_segment.source_binding_attestation
    )
    validate_source_binding_integrity(active_segment.source_binding, active_segment.services)
    validate_source_binding_integrity(pre_binding, pre_services)
    validate_source_binding_integrity(post_binding, post_services)
    if active_segment.source_binding != active_segment.source_binding_attestation.binding:
        _fail()
    if (
        not hmac.compare_digest(pre_services.raw, active_segment.services.raw)
        or not hmac.compare_digest(post_services.raw, active_segment.services.raw)
        or not hmac.compare_digest(pre_binding.core.raw, active_segment.source_binding.core.raw)
        or not hmac.compare_digest(post_binding.core.raw, active_segment.source_binding.core.raw)
        or pre_binding.service_pair_by_alias
        != active_segment.source_binding.service_pair_by_alias
        or post_binding.service_pair_by_alias
        != active_segment.source_binding.service_pair_by_alias
        or pre_binding.authority_bytes_by_alias
        != active_segment.source_binding.authority_bytes_by_alias
        or post_binding.authority_bytes_by_alias
        != active_segment.source_binding.authority_bytes_by_alias
    ):
        _fail()


def _closure_receipt_fields(
    closure: ValidatedClosure | InitialRuntimeChangeClosure,
) -> Sequence[str]:
    if type(closure) is InitialRuntimeChangeClosure:
        return INITIAL_CLOSURE_FIELDS
    if type(closure) is ValidatedClosure:
        return CLOSURE_FIELDS
    _fail()


def _require_predecessor_pair_consistent(
    predecessor: ActiveSegment | UnboundWatermarkPredecessor,
    closure: ValidatedClosure | InitialRuntimeChangeClosure,
) -> None:
    """A pre-binding predecessor pairs only with the pre-binding closure."""

    unbound = type(predecessor) is UnboundWatermarkPredecessor
    pre_binding = type(closure) is InitialRuntimeChangeClosure
    if unbound != pre_binding:
        _fail()
    if not unbound and type(predecessor) is not ActiveSegment:
        _fail()
    if not pre_binding and type(closure) is not ValidatedClosure:
        _fail()


def _require_active_segment_consistent(
    contract: FrozenContract,
    active: ActiveSegment,
) -> None:
    if type(contract) is not FrozenContract or type(active) is not ActiveSegment:
        _fail()
    if (
        type(active.segment_index) is not int
        or type(active.segment_id) is not str
        or type(active.lower_bound_exclusive_at) is not str
        or type(active.services) is not ServiceTuple
        or type(active.source_binding) is not SourceBinding
        or type(active.attestation_identity) is not HashAndBytes
        or type(active.source_binding_attestation) is not SourceBindingAttestation
    ):
        _fail()
    _bounded_int(active.segment_index, 0, LAST_SLOT_INDEX)
    _require_frozen_contract_consistent(contract)
    _require_source_attestation_wrapper_consistent(active.source_binding_attestation)
    validate_source_binding_integrity(active.source_binding, active.services)
    if (
        active.source_binding != active.source_binding_attestation.binding
        or active.source_binding_attestation.segment_index != active.segment_index
        or active.source_binding_attestation.analysis_plan_identity
        != contract.plan_identity
    ):
        _fail()
    if active.segment_index == 0:
        if (
            active.segment_id != contract.initial_segment_id
            or active.lower_bound_exclusive_at != contract.release_effective_at
            or active.services.identity != contract.initial_services.identity
            or not hmac.compare_digest(
                active.services.raw, contract.initial_services.raw
            )
            or active.attestation_identity != contract.watermark_identity
            or active.attestation_raw is not None
            or active.attestation_previous_ledger_sha256 is not None
            or active.predecessor_closure is not None
            or active.predecessor_segment is not None
        ):
            _fail()
        return
    if type(active.attestation_raw) is not bytes:
        _fail()
    if type(active.predecessor_closure) not in CLOSURE_TYPES:
        _fail()
    if type(active.predecessor_segment) not in PREDECESSOR_TYPES:
        _fail()
    predecessor_closure = active.predecessor_closure
    predecessor_segment = active.predecessor_segment
    _require_predecessor_pair_consistent(predecessor_segment, predecessor_closure)
    unbound = type(predecessor_segment) is UnboundWatermarkPredecessor
    if predecessor_segment.segment_index != active.segment_index - 1:
        _fail()
    if unbound:
        _require_unbound_predecessor_consistent(contract, predecessor_segment)
    else:
        _require_active_segment_consistent(contract, predecessor_segment)
    _require_closure_wrapper_consistent(contract, predecessor_closure)
    closure_receipt = _load_receipt(
        predecessor_closure.raw, _closure_receipt_fields(predecessor_closure)
    )
    receipt = _load_receipt(active.attestation_raw, SUCCESSOR_ATTESTATION_FIELDS)
    attempt = active.source_binding_attestation.attempt
    pre_launcher = parse_utc(receipt["pre_launcher_at"], receipt=True)
    pre_clocks = _validate_alias_timestamp_map(
        receipt["pre_source_clock_observed_at_by_alias"]
    )
    boundary_launcher = parse_utc(receipt["boundary_launcher_at"], receipt=True)
    boundary = parse_utc(receipt["boundary_at"], receipt=True)
    lower = parse_utc(receipt["lower_bound_exclusive_at"], receipt=True)
    post_phase = parse_utc(receipt["post_phase_started_at"], receipt=True)
    post_clocks = _validate_alias_timestamp_map(
        receipt["post_source_clock_observed_at_by_alias"]
    )
    post_launcher = parse_utc(receipt["post_launcher_at"], receipt=True)
    predecessor_sha = _hex(receipt["predecessor_closure_sha256"], 64)
    closure_observer = HashAndBytes.from_value(
        closure_receipt["runtime_observer_sha256_and_bytes"]
    )
    if (
        hash_and_bytes(active.attestation_raw) != active.attestation_identity
        or receipt["receipt_kind"] != "runtime-segment-attestation"
        or receipt["status"] != "pass"
        or _bounded_int(receipt["segment_index"], 0, LAST_SLOT_INDEX)
        != active.segment_index
        or active.segment_index != attempt.segment_index
        or active.segment_index != active.source_binding_attestation.segment_index
        or active.segment_index != predecessor_closure.segment_index + 1
        or predecessor_closure.segment_index != predecessor_segment.segment_index
        or predecessor_closure.segment_id != predecessor_segment.segment_id
        or active.source_binding_attestation.runtime_observer_identity
        != closure_observer
        or receipt["segment_id"] != active.segment_id
        or receipt["lower_bound_exclusive_at"] != active.lower_bound_exclusive_at
        or lower != boundary
        or receipt["lower_bound_exclusive_at"] != receipt["boundary_at"]
        or attempt.attempt_scope != SUCCESSOR_SEGMENT_SCOPE
        or predecessor_sha != predecessor_closure.identity.sha256
        or predecessor_sha != attempt.predecessor_closure_sha256
        or predecessor_closure.slot_index >= LAST_SLOT_INDEX
        or attempt.authorized_at != predecessor_closure.validated_at
        or HashAndBytes.from_value(
            closure_receipt["prior_active_services_state_sha256_and_bytes"]
        )
        != predecessor_segment.services.identity
        or HashAndBytes.from_value(closure_receipt["attestation_sha256_and_bytes"])
        != predecessor_segment.attestation_identity
        or not parse_utc(
            predecessor_segment.lower_bound_exclusive_at,
            receipt=True,
        )
        < parse_utc(closure_receipt["scheduled_at"], receipt=True)
        or HashAndBytes.from_value(
            receipt["attestation_attempt_marker_sha256_and_bytes"]
        )
        != attempt.identity
        or HashAndBytes.from_value(
            receipt["complete_unaliased_service_tuple_sha256_and_bytes"]
        )
        != active.services.identity
        or HashAndBytes.from_value(
            receipt["source_binding_attestation_sha256_and_bytes"]
        )
        != active.source_binding_attestation.identity
        or receipt["active_services_state_sha256"] != active.services.identity.sha256
        or not _json_exact(
            receipt["replay_code_control_commit_and_tree"],
            {
                "commit": REPLAY_CODE_CONTROL_COMMIT,
                "tree": REPLAY_CODE_CONTROL_TREE,
            },
        )
        or HashAndBytes.from_value(receipt["analysis_plan_sha256_and_bytes"])
        != contract.plan_identity
        or type(active.attestation_previous_ledger_sha256) is not str
        or _hex(receipt["previous_ledger_entry_sha256"], 64)
        != _hex(active.attestation_previous_ledger_sha256, 64)
        or pre_launcher != active.source_binding_attestation.pre_observed_at
        or post_launcher != active.source_binding_attestation.post_observed_at
        or attempt.start_deadline_at != _add_seconds(attempt.authorized_at, 30)
        or not attempt.authorized_at <= attempt.written_at < attempt.start_deadline_at
        or not attempt.written_at <= pre_launcher < attempt.start_deadline_at
        or boundary != max(boundary_launcher, *pre_clocks.values())
        or not pre_launcher <= boundary_launcher <= boundary
        or post_phase < _add_microseconds(boundary, 250_000)
        or post_phase > post_launcher
        or not all(instant > boundary for instant in post_clocks.values())
        or post_launcher <= boundary
        or not post_launcher - pre_launcher < timedelta(seconds=300)
        or not post_launcher
        < parse_utc(
            slot_times(predecessor_closure.slot_index + 1).scheduled_at,
            receipt=True,
        )
        or receipt["segment_id"] != derive_successor_segment_id(receipt)
    ):
        _fail()
    if unbound:
        # No prior source-binding core exists, so only the tuple change is
        # provable, and the closure's observed core is never compared against.
        transition_kinds = validate_pre_binding_transition(
            predecessor_segment.services, active.services
        )
        if (
            HashAndBytes.from_value(
                closure_receipt["attestation_attempt_marker_sha256_and_bytes"]
            )
            != predecessor_closure.attempt_identity
            or attempt.identity == predecessor_closure.attempt_identity
            or not predecessor_closure.validated_at < pre_launcher
        ):
            _fail()
    else:
        transition_kinds = validate_binding_transition(
            predecessor_segment.services,
            predecessor_segment.source_binding,
            active.services,
            active.source_binding,
        )
        if (
            HashAndBytes.from_value(
                closure_receipt["prior_source_binding_core_sha256_and_bytes"]
            )
            != predecessor_segment.source_binding.core.identity
            or HashAndBytes.from_value(
                closure_receipt["observed_active_services_state_sha256_and_bytes"]
            )
            != active.services.identity
            or HashAndBytes.from_value(
                closure_receipt["observed_source_binding_core_sha256_and_bytes"]
            )
            != active.source_binding.core.identity
        ):
            _fail()
    if closure_receipt["mismatch_kinds"] != list(transition_kinds):
        _fail()
    if any(
        active.services.boot_started_at(pair) > predecessor_closure.validated_at
        for pair in active.services.service_pairs
    ):
        _fail()
    if not lower > parse_utc(
        predecessor_segment.lower_bound_exclusive_at,
        receipt=True,
    ):
        _fail()
    for alias in SOURCE_ALIASES:
        pair = active.source_binding.service_map()[alias]
        if active.services.boot_started_at(pair) > pre_clocks[alias]:
            _fail()


def derive_successor_segment_id(value: Any) -> str:
    if type(value) is not dict or "segment_id" not in value:
        _fail()
    core = dict(value)
    del core["segment_id"]
    core.pop("attestation_sha256", None)
    return hashlib.sha256(
        RUNTIME_SEGMENT_DOMAIN + b"\0" + canonical_json_bytes(core)
    ).hexdigest()


def validate_successor_segment(
    raw: bytes,
    *,
    contract: FrozenContract,
    predecessor: ActiveSegment | UnboundWatermarkPredecessor,
    predecessor_closure: ValidatedClosure | InitialRuntimeChangeClosure,
    attempt: AttestationAttempt,
    source_binding_attestation: SourceBindingAttestation,
    pre_services_observation: Any,
    post_services_observation: Any,
    complete_services_raw: bytes,
    expected_previous_ledger_sha256: Any,
) -> ActiveSegment:
    """Validate a successor attestation and return the next active segment.

    The predecessor may be an ordinary closed :class:`ActiveSegment` or the typed
    :class:`UnboundWatermarkPredecessor`.  Either way the successor is proven by
    its own fresh boundary ceremony: on the pre-binding path the closure's
    recorded observation is evidence only and is never accepted in place of that
    ceremony.
    """

    if not all(
        type(item) is expected
        for item, expected in (
            (contract, FrozenContract),
            (attempt, AttestationAttempt),
            (source_binding_attestation, SourceBindingAttestation),
        )
    ):
        _fail()
    if (
        type(predecessor) not in PREDECESSOR_TYPES
        or type(predecessor_closure) not in CLOSURE_TYPES
    ):
        _fail()
    _require_predecessor_pair_consistent(predecessor, predecessor_closure)
    unbound = type(predecessor) is UnboundWatermarkPredecessor
    _require_closure_wrapper_consistent(contract, predecessor_closure)
    if unbound:
        _require_unbound_predecessor_consistent(contract, predecessor)
        if predecessor_closure.predecessor != predecessor:
            _fail()
        observer_identity = predecessor_closure.observer_identity
    else:
        _require_active_segment_consistent(contract, predecessor)
        _require_source_attestation_wrapper_consistent(
            predecessor.source_binding_attestation
        )
        validate_source_binding_integrity(
            predecessor.source_binding, predecessor.services
        )
        if predecessor.source_binding != predecessor.source_binding_attestation.binding:
            _fail()
        observer_identity = (
            predecessor.source_binding_attestation.runtime_observer_identity
        )
    _require_attempt_wrapper_consistent(attempt)
    _require_source_attestation_wrapper_consistent(source_binding_attestation)
    if (
        attempt.analysis_plan_identity != contract.plan_identity
        or source_binding_attestation.analysis_plan_identity != contract.plan_identity
        or attempt.runtime_observer_identity != observer_identity
        or source_binding_attestation.runtime_observer_identity != observer_identity
    ):
        _fail()
    if (
        predecessor_closure.slot_index >= LAST_SLOT_INDEX
        or predecessor_closure.segment_index != predecessor.segment_index
        or predecessor_closure.segment_id != predecessor.segment_id
        or attempt.attempt_scope != SUCCESSOR_SEGMENT_SCOPE
        or attempt.predecessor_closure_sha256 != predecessor_closure.identity.sha256
        or source_binding_attestation.attempt.identity != attempt.identity
    ):
        _fail()
    if unbound and attempt.identity == predecessor_closure.attempt_identity:
        # The slot-0 marker the closure consumed can never stand in for the
        # successor ceremony's own marker.
        _fail()
    expected_start_deadline = _add_seconds(predecessor_closure.validated_at, 30)
    if (
        attempt.segment_index != predecessor.segment_index + 1
        or attempt.authorized_at != predecessor_closure.validated_at
        or attempt.start_deadline_at != expected_start_deadline
        or not predecessor_closure.validated_at
        <= attempt.written_at
        < attempt.start_deadline_at
    ):
        _fail()
    # Rebind the validated wrapper to its exact bytes before trusting its hash.
    if (
        hash_and_bytes(predecessor_closure.raw) != predecessor_closure.identity
        or hash_and_bytes(attempt.raw) != attempt.identity
        or hash_and_bytes(source_binding_attestation.raw)
        != source_binding_attestation.identity
    ):
        _fail()
    receipt = _load_receipt(raw, SUCCESSOR_ATTESTATION_FIELDS)
    if (
        receipt["receipt_kind"] != "runtime-segment-attestation"
        or receipt["status"] != "pass"
    ):
        _fail()
    segment_index = _bounded_int(receipt["segment_index"], 0, LAST_SLOT_INDEX)
    if (
        segment_index != predecessor.segment_index + 1
        or segment_index != attempt.segment_index
        or segment_index != source_binding_attestation.segment_index
    ):
        _fail()
    services = validate_stable_service_tuple_observations(
        pre_services_observation,
        post_services_observation,
    )
    retained_services = validate_service_tuple_bytes(complete_services_raw)
    if services.identity != retained_services.identity or not hmac.compare_digest(
        services.raw, retained_services.raw
    ):
        _fail()
    binding = source_binding_attestation.binding
    validate_source_binding_integrity(binding, services)
    if binding.core.active_services_state_sha256 != services.identity.sha256:
        _fail()
    closure_receipt = _load_receipt(
        predecessor_closure.raw, _closure_receipt_fields(predecessor_closure)
    )
    if unbound:
        transition_kinds = validate_pre_binding_transition(
            predecessor.services, services
        )
        _same_identity(
            closure_receipt["attestation_attempt_marker_sha256_and_bytes"],
            predecessor_closure.attempt_identity,
        )
    else:
        transition_kinds = validate_binding_transition(
            predecessor.services,
            predecessor.source_binding,
            services,
            binding,
        )
        _same_identity(
            closure_receipt["observed_active_services_state_sha256_and_bytes"],
            services.identity,
        )
        _same_identity(
            closure_receipt["prior_source_binding_core_sha256_and_bytes"],
            predecessor.source_binding.core.identity,
        )
        _same_identity(
            closure_receipt["observed_source_binding_core_sha256_and_bytes"],
            binding.core.identity,
        )
    if closure_receipt["mismatch_kinds"] != list(transition_kinds):
        _fail()
    _same_identity(
        closure_receipt["prior_active_services_state_sha256_and_bytes"],
        predecessor.services.identity,
    )
    _same_identity(
        closure_receipt["attestation_sha256_and_bytes"],
        predecessor.attestation_identity,
    )
    _same_identity(
        receipt["attestation_attempt_marker_sha256_and_bytes"], attempt.identity
    )
    predecessor_sha = _hex(receipt["predecessor_closure_sha256"], 64)
    if predecessor_sha != predecessor_closure.identity.sha256:
        _fail()
    _same_identity(
        receipt["complete_unaliased_service_tuple_sha256_and_bytes"], services.identity
    )
    _same_identity(
        receipt["source_binding_attestation_sha256_and_bytes"],
        source_binding_attestation.identity,
    )
    if receipt["active_services_state_sha256"] != services.identity.sha256:
        _fail()
    if receipt["replay_code_control_commit_and_tree"] != {
        "commit": REPLAY_CODE_CONTROL_COMMIT,
        "tree": REPLAY_CODE_CONTROL_TREE,
    }:
        _fail()
    _same_identity(receipt["analysis_plan_sha256_and_bytes"], contract.plan_identity)
    previous = _hex(receipt["previous_ledger_entry_sha256"], 64)
    if previous != _hex(expected_previous_ledger_sha256, 64):
        _fail()

    pre_launcher = parse_utc(receipt["pre_launcher_at"], receipt=True)
    pre_clocks = _validate_alias_timestamp_map(
        receipt["pre_source_clock_observed_at_by_alias"]
    )
    boundary_launcher = parse_utc(receipt["boundary_launcher_at"], receipt=True)
    boundary = parse_utc(receipt["boundary_at"], receipt=True)
    lower = parse_utc(receipt["lower_bound_exclusive_at"], receipt=True)
    post_phase = parse_utc(receipt["post_phase_started_at"], receipt=True)
    post_clocks = _validate_alias_timestamp_map(
        receipt["post_source_clock_observed_at_by_alias"]
    )
    post_launcher = parse_utc(receipt["post_launcher_at"], receipt=True)
    expected_boundary = max(boundary_launcher, *pre_clocks.values())
    next_slot = parse_utc(
        slot_times(predecessor_closure.slot_index + 1).scheduled_at,
        receipt=True,
    )
    if (
        receipt["lower_bound_exclusive_at"] != receipt["boundary_at"]
        or lower != boundary
        or boundary != expected_boundary
        or not pre_launcher <= boundary_launcher <= boundary
        or pre_launcher != source_binding_attestation.pre_observed_at
        or post_launcher != source_binding_attestation.post_observed_at
        or not attempt.written_at <= pre_launcher < attempt.start_deadline_at
        or not post_phase >= _add_microseconds(boundary, 250_000)
        or not post_phase <= post_launcher
        or not all(instant > boundary for instant in post_clocks.values())
        or not post_launcher > boundary
        or not post_launcher - pre_launcher < timedelta(seconds=300)
        or not post_launcher < next_slot
        or not lower > parse_utc(predecessor.lower_bound_exclusive_at, receipt=True)
    ):
        _fail()
    if unbound and not predecessor_closure.validated_at < pre_launcher:
        # The ceremony must genuinely follow the closure it succeeds.
        _fail()
    for alias in SOURCE_ALIASES:
        pair = binding.service_map()[alias]
        if services.boot_started_at(pair) > pre_clocks[alias]:
            _fail()

    segment_id = _hex(receipt["segment_id"], 64)
    if segment_id != derive_successor_segment_id(receipt):
        _fail()
    return ActiveSegment(
        segment_index=segment_index,
        segment_id=segment_id,
        lower_bound_exclusive_at=receipt["lower_bound_exclusive_at"],
        services=services,
        source_binding=binding,
        attestation_identity=hash_and_bytes(raw),
        source_binding_attestation=source_binding_attestation,
        attestation_raw=raw,
        attestation_previous_ledger_sha256=previous,
        predecessor_closure=predecessor_closure,
        predecessor_segment=predecessor,
    )
