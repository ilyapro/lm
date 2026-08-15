#!/usr/bin/env python3
"""Independent confirmatory-holdout-v4 packet verifier.

Three entry points:

* ``verify.py sealed --packet-dir DIR`` audits a frozen packet and prints
  exactly one JSON line, ``{"mode":"v4-sealed","status":"pass"}`` or
  ``{"mode":"v4-sealed","status":"fail","check":<code>}``.
* ``verify.py keyed --draft-dir DIR --identity-key-fd N --receipt-fd N`` is the
  in-build keyed phase.  It prints nothing at all -- stdout is a leak channel
  during a keyed build -- and writes one canonical aggregate receipt to an
  inherited anonymous pipe.
* ``verify.py --mechanical-only`` is a strict expansion of ``sealed`` against
  this file's own parent packet directory, accepted only as an exact-match sole
  argument.

Independence is the whole point.  ``BUILDER_SHA256``/``BUILDER_BYTES``,
``README_SHA256``, ``POLICY_SHA256`` and ``PLAN_SHA256`` below are this
module's own source literals, computed once from the final sibling files and
checked directly against bytes read from disk.  The manifest's self-report is
never the authority: rewriting a packet file and rehashing the manifest still
fails, because both sides are compared against these literals rather than
against each other.  This file pins no hash of itself; instead the packet's
tracked ``recipe/verify.py`` is bound to the bytes of the executing verifier,
so a swapped in-packet verifier is caught.

The campaign tools that produce the evidence are bound the same way, by path,
SHA-256 and byte size, but their set is settled against disk rather than
spelled: ``audit_campaign_tools`` sweeps the tool directories, hashes what it
finds, and requires the manifest to name exactly that.  A tool nothing binds
and a bound tool that has drifted both fail closed.

Nothing here imports the builder, the seal launcher, or any v4 runtime module,
and nothing here opens a consumed corpus, invokes a consumed packet verifier,
inspects a private source, or runs a v4 semantic reader.  Every schema, domain,
grammar and column allowlist is re-spelled from the frozen protocol.

Failures are a small closed vocabulary of category codes.  No path, label,
digest, count or exception text ever escapes: ``except BaseException``
collapses to ``internal``.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import hmac
import json
import os
import re
import resource
import sqlite3
import stat
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# --------------------------------------------------------------------------
# Independent source pins
# --------------------------------------------------------------------------
#
# Computed from the now-final sibling files and never re-derived at runtime
# from anything the packet supplies.

BUILDER_SHA256 = "e2932b3ad77a5540e2009d7370964527dd67bfbfd217b6313ff9ade23ed0773b"
BUILDER_BYTES = 157_965
README_SHA256 = "abaf8604d3d1b0d83b90ad2c1dbe62a54cec94765facde08814f75bf8e64c3f1"
README_BYTES = 9_545
POLICY_SHA256 = "8a229fa1edb71835ff3fb3d34ab38719ad5af68ef059f0717e7e164726f623c6"
POLICY_BYTES = 42_277
PLAN_SHA256 = "bb049d1b5a237b345a5f94957e5cd82f37bd4940b14011e68e30dfb7aa7c8d70"
PLAN_BYTES = 177_982

# ``measurement.reference_behavior``: the gating-off control implementation.
# The manifest binds a digest over the pair, so the pair is pinned here and the
# digest is recomputed rather than trusted.
REPLAY_CODE_CONTROL_COMMIT = "46a9951842512333b0896370056d07a9e1c25bdf"
REPLAY_CODE_CONTROL_TREE = "c1606671b9d13bed78c21b7d2f9a4bb75a3d1c1c"

# ``identity.complete_frozen_dev_reference.raw_identity_reference``.
RAW_IDENTITY_REFERENCE_SHA256 = (
    "45dbfdb389324a8702ba1a84b28be028991f23dcfe1d7ab9d771928c37c2cc7a"
)
RAW_IDENTITY_REFERENCE_BYTES = 4_162_914

# --------------------------------------------------------------------------
# Frozen protocol constants, re-spelled
# --------------------------------------------------------------------------

NAMESPACE = "confirmatory-holdout-v4"
SCHEMA_VERSION = 4
SOURCE_ALIASES = ("local", "alt")
RELEASE_EFFECTIVE_AT = "2026-08-14T15:03:46.793603Z"
FIRST_SLOT_INDEX = 0
LAST_SLOT_INDEX = 28

# ``domains``.  Only the two this verifier actually recomputes are spelled as
# preimage bytes; the retired prefixes are spelled to prove they never appear.
PARTITION_DOMAIN = b"confirmatory-holdout-v4/partition/v1\0"
SNAPSHOT_SET_DOMAIN = b"confirmatory-holdout-v4/snapshot-set/v1\0"
RETIRED_DOMAIN_PREFIXES = ("confirmatory-holdout-v2/", "confirmatory-holdout-v3/")

SPLIT_MODULUS = 100
HOLDOUT_UPPER_EXCLUSIVE = 50
HOLDOUT_PARTITION = "holdout"
SHADOW_PARTITION = "shadow"
PARTITIONS = (HOLDOUT_PARTITION, SHADOW_PARTITION)

IDENTITY_KEY_BYTES = 32

# ``publication_and_ordering`` packet layout.
PACKET_MANIFEST_NAME = "manifest.json"
PACKET_CORPUS_DIRECTORY = "corpus"
HOLDOUT_CORPUS_NAME = "holdout.jsonl"
SHADOW_CORPUS_NAME = "shadow.jsonl"
SEED_MANIFEST_NAME = "seed-state-manifest.json"
PRESEAL_RECEIPT_NAME = "preseal-receipt.json"
SEED_STATE_NAMES = tuple(
    f"seed-state-{partition}-{alias}.sqlite3"
    for partition in PARTITIONS
    for alias in SOURCE_ALIASES
)
CORPUS_MEMBER_NAMES = frozenset(
    {
        HOLDOUT_CORPUS_NAME,
        SHADOW_CORPUS_NAME,
        *SEED_STATE_NAMES,
        SEED_MANIFEST_NAME,
        PRESEAL_RECEIPT_NAME,
    }
)
CORPUS_NAME_BY_PARTITION = {
    HOLDOUT_PARTITION: HOLDOUT_CORPUS_NAME,
    SHADOW_PARTITION: SHADOW_CORPUS_NAME,
}

# The namespace freeze allowlist.  A sealed packet root may carry only these,
# and must carry the six that make it a published packet.
NAMESPACE_ALLOWED_ENTRIES = frozenset(
    {
        "segments",
        "probes",
        "recipe",
        "corpus",
        "manifest.json",
        "seal-receipt.json",
        "README.md",
        "POLICY.md",
        "analysis-plan.json",
    }
)
PACKET_REQUIRED_ENTRIES = frozenset(
    {"README.md", "POLICY.md", "analysis-plan.json", "recipe", "corpus", "manifest.json"}
)
RECIPE_REQUIRED_ENTRIES = frozenset({"build.py", "verify.py"})

# --------------------------------------------------------------------------
# Closed failure vocabulary
# --------------------------------------------------------------------------
#
# One code per named violation class, plus the four structural categories.
# Nothing outside this frozenset can ever be printed.

OVERWRITE = "overwrite"
SCHEMA = "schema"
PRODUCTION_SHAPE = "production_shape"
PRIVACY = "privacy"
PARTITION = "partition"
REPLAYABILITY = "replayability"
RUNTIME_SEGMENT = "runtime_segment"
READER_AUTHORITY = "reader_authority"
MANIFEST_ORDER = "manifest_order"
PIN = "pin"
BOUNDARY = "boundary"
CANDIDATE_CHANGED = "candidate_changed"
ARGUMENTS = "arguments"
INTERNAL = "internal"

CHECK_CODES = frozenset(
    {
        OVERWRITE,
        SCHEMA,
        PRODUCTION_SHAPE,
        PRIVACY,
        PARTITION,
        REPLAYABILITY,
        RUNTIME_SEGMENT,
        READER_AUTHORITY,
        MANIFEST_ORDER,
        PIN,
        BOUNDARY,
        CANDIDATE_CHANGED,
        ARGUMENTS,
        INTERNAL,
    }
)


class VerificationError(RuntimeError):
    """Fail-closed error carrying only a public aggregate category."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code if code in CHECK_CODES else INTERNAL


def fail(code: str) -> Any:
    raise VerificationError(code)


# --------------------------------------------------------------------------
# ``operational_receipt_schemas.common_validation``
# --------------------------------------------------------------------------


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """``duplicate_json_key_action: invalid`` at every nesting depth."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            fail(SCHEMA)
        result[key] = value
    return result


def _reject_constant(_literal: str) -> Any:
    """NaN and +-Infinity are not JSON and must never enter the audit."""

    return fail(SCHEMA)


def load_json(raw: bytes, *, expected: type | None = None, code: str = SCHEMA) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return fail(code)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except VerificationError:
        raise
    except (ValueError, RecursionError):
        return fail(code)
    if expected is not None and type(value) is not expected:
        return fail(code)
    return value


def canonical_bytes(value: Any) -> bytes:
    """``domains.v4_canonical_json_utf8``."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        return fail(SCHEMA)


def load_canonical(raw: bytes, *, expected: type | None = None, code: str = SCHEMA) -> Any:
    """Parse, then require byte-exact canonical round-trip."""

    value = load_json(raw, expected=expected, code=code)
    if canonical_bytes(value) != raw:
        fail(code)
    return value


def is_int(value: Any) -> bool:
    """``bytes_type``/``slot_index_type``: booleans are not integers."""

    return type(value) is int


def is_bool(value: Any) -> bool:
    return type(value) is bool


HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
TIMESTAMP_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d{6})Z$"
)
DEIDENTIFIED_RE = re.compile(r"^[a-z \t\n\r]*$")
# A sixty-four character hexadecimal run is digest-shaped.  De-identified
# surrogates are drawn from ``a-z`` plus four whitespace characters and can
# never contain a decimal digit, so requiring one discriminates exactly.
DIGEST_SHAPED_RE = re.compile(r"(?<![0-9A-Fa-f])(?=[0-9a-f]{64}(?![0-9A-Fa-f]))(?=[a-f]*[0-9])[0-9a-f]{64}")


def hex64(value: Any, code: str = SCHEMA) -> str:
    """``sha256_type``: exactly 64 lowercase hexadecimal characters."""

    if type(value) is not str or HEX64_RE.match(value) is None:
        return fail(code)
    return value


def nonnegative_int(value: Any, code: str = SCHEMA) -> int:
    if not is_int(value) or value < 0:
        return fail(code)
    return value


def positive_int(value: Any, code: str = SCHEMA) -> int:
    if not is_int(value) or value <= 0:
        return fail(code)
    return value


_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def timestamp(value: Any, code: str = SCHEMA) -> str:
    """``canonical_receipt_encoding``: exactly six fractional digits, UTC."""

    if type(value) is not str:
        return fail(code)
    match = TIMESTAMP_RE.match(value)
    if match is None:
        return fail(code)
    year, month, day, hour, minute, second = (int(part) for part in match.groups()[:6])
    if not 1 <= month <= 12 or not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return fail(code)
    if not 0 <= second <= 59:
        return fail(code)
    limit = _DAYS_IN_MONTH[month - 1] + (1 if month == 2 and _leap(year) else 0)
    if not 1 <= day <= limit:
        return fail(code)
    return value


def timestamp_microseconds(value: Any, code: str = SCHEMA) -> int:
    """``comparison``: parse exactly to integer microseconds, never text."""

    text = timestamp(value, code)
    year, month, day, hour, minute, second, fraction = (
        int(part) for part in TIMESTAMP_RE.match(text).groups()
    )
    days = 0
    for step in range(1970, year):
        days += 366 if _leap(step) else 365
    for step in range(1, month):
        days += _DAYS_IN_MONTH[step - 1] + (1 if step == 2 and _leap(year) else 0)
    days += day - 1
    return ((days * 24 + hour) * 60 + minute) * 60_000_000 + second * 1_000_000 + fraction


def exact_keys(value: Any, keys: Iterable[str], code: str = SCHEMA) -> dict[str, Any]:
    """Exact key SETS, recursively.

    The canonical encoder sorts keys, so tuple order carries no information;
    an unknown or missing member at any depth is what must be rejected.
    """

    if type(value) is not dict:
        return fail(code)
    if set(value) != set(keys):
        return fail(code)
    return value


def identity_shape(value: Any, code: str = SCHEMA) -> dict[str, Any]:
    """``composite_types.sha256_and_bytes``."""

    identity = exact_keys(value, ("sha256", "bytes"), code)
    hex64(identity["sha256"], code)
    nonnegative_int(identity["bytes"], code)
    return identity


def same_identity(observed: Mapping[str, Any], sha256: str, size: int) -> bool:
    return (
        hmac.compare_digest(str(observed.get("sha256")), sha256)
        and is_int(observed.get("bytes"))
        and observed["bytes"] == size
    )


def digest_of(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def same_bytes(left: bytes, right: bytes) -> bool:
    return hmac.compare_digest(hashlib.sha256(left).digest(), hashlib.sha256(right).digest())


def walk(value: Any) -> Iterable[tuple[tuple[str, ...], Any]]:
    """Every (key path, node) pair, for recursive structural scans."""

    stack: list[tuple[tuple[str, ...], Any]] = [((), value)]
    while stack:
        path, node = stack.pop()
        yield path, node
        if type(node) is dict:
            for key, child in node.items():
                stack.append((path + (key,), child))
        elif type(node) is list:
            for child in node:
                stack.append((path + ("[]",), child))


def strings_in(value: Any) -> Iterable[str]:
    for _path, node in walk(value):
        if type(node) is str:
            yield node


# --------------------------------------------------------------------------
# Filesystem access
# --------------------------------------------------------------------------

MAX_MEMBER_BYTES = 4_000_000_000


def _disable_core_dumps() -> None:
    """A core file would be an unaudited copy of everything read."""

    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError):
        pass


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _file_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def open_directory(path: Path, code: str = BOUNDARY) -> int:
    try:
        return os.open(os.fspath(path), _directory_flags())
    except OSError:
        return fail(code)


def open_child_directory(parent_fd: int, name: str, code: str = BOUNDARY) -> int:
    try:
        return os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError:
        return fail(code)


def close_quietly(fd: int) -> None:
    if fd >= 0:
        try:
            os.close(fd)
        except OSError:
            pass


def stat_at(parent_fd: int, name: str, code: str = BOUNDARY) -> os.stat_result:
    try:
        return os.lstat(name, dir_fd=parent_fd)
    except OSError:
        return fail(code)


def read_at(parent_fd: int, name: str, code: str = BOUNDARY) -> bytes:
    """Read one regular, non-symlink, single-link member through its parent."""

    try:
        fd = os.open(name, _file_flags(), dir_fd=parent_fd)
    except OSError:
        return fail(code)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            fail(code)
        if info.st_size > MAX_MEMBER_BYTES:
            fail(code)
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) != info.st_size or os.read(fd, 1):
            fail(code)
        return raw
    except OSError:
        return fail(code)
    finally:
        close_quietly(fd)


def listdir_at(fd: int, code: str = BOUNDARY) -> set[str]:
    try:
        return set(os.listdir(fd))
    except OSError:
        return fail(code)


def directory_identity(fd: int, code: str = BOUNDARY) -> tuple[int, int]:
    try:
        info = os.fstat(fd)
    except OSError:
        return fail(code)
    if not stat.S_ISDIR(info.st_mode):
        fail(code)
    return info.st_dev, info.st_ino


def sweep_for_symlinks(root_fd: int, code: str = BOUNDARY) -> None:
    """No entry anywhere under the packet may be a symlink or a device."""

    pending: list[tuple[int, bool]] = [(root_fd, False)]
    try:
        while pending:
            fd, owned = pending.pop()
            try:
                for name in listdir_at(fd, code):
                    info = stat_at(fd, name, code)
                    if stat.S_ISDIR(info.st_mode):
                        pending.append((open_child_directory(fd, name, code), True))
                    elif not stat.S_ISREG(info.st_mode):
                        fail(code)
            finally:
                if owned:
                    close_quietly(fd)
    finally:
        for fd, owned in pending:
            if owned:
                close_quietly(fd)


def read_tree(root_fd: int, code: str = BOUNDARY) -> tuple[dict[str, bytes], dict[str, tuple]]:
    """Every regular file under the packet, by slash-joined relative name.

    The second return value is the identity map used to detect mutation during
    verification: device, inode, size, link count and both change stamps.
    """

    blobs: dict[str, bytes] = {}
    marks: dict[str, tuple] = {}
    pending: list[tuple[int, str, bool]] = [(root_fd, "", False)]
    try:
        while pending:
            fd, prefix, owned = pending.pop()
            try:
                for name in sorted(listdir_at(fd, code)):
                    relative = f"{prefix}{name}"
                    info = stat_at(fd, name, code)
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(
                            (open_child_directory(fd, name, code), relative + "/", True)
                        )
                        marks[relative + "/"] = (
                            info.st_dev,
                            info.st_ino,
                            info.st_mode,
                            info.st_mtime_ns,
                            info.st_ctime_ns,
                        )
                        continue
                    if not stat.S_ISREG(info.st_mode):
                        fail(code)
                    blobs[relative] = read_at(fd, name, code)
                    marks[relative] = (
                        info.st_dev,
                        info.st_ino,
                        info.st_mode,
                        info.st_nlink,
                        info.st_size,
                        info.st_mtime_ns,
                        info.st_ctime_ns,
                    )
            finally:
                if owned:
                    close_quietly(fd)
    finally:
        for fd, _prefix, owned in pending:
            if owned:
                close_quietly(fd)
    return blobs, marks


# --------------------------------------------------------------------------
# The frozen input-only resolver, re-spelled
# --------------------------------------------------------------------------

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
AMBIENT_KEY_SET = frozenset(AMBIENT_KEYS)

# ``replayability.persisted_source_fields_required``.
PERSISTED_SOURCE_FIELDS_REQUIRED = (
    "id",
    "created_at",
    "query",
    "scope",
    "requested_scope",
    "resolved_scopes",
    "ambient_context",
    "depth",
    "max_results",
    "agent",
    "task",
    "session_id",
)

# ``replayability.forbidden_inputs``.
FORBIDDEN_REPLAY_INPUTS = (
    "feedback_applied",
    "success",
    "error",
    "result content or count",
    "payload size",
    "latency",
    "access result",
    "candidate behavior",
    "baseline behavior",
)
# The concrete field spellings those nine forbid, as they would appear in a
# record.  A sealed case that carries any of them read an outcome.
FORBIDDEN_REPLAY_INPUT_FIELDS = frozenset(
    {
        "feedback_applied",
        "feedback_applied_at",
        "feedback_trace_id",
        "success",
        "error",
        "results",
        "result",
        "result_count",
        "content",
        "payload",
        "payload_size",
        "latency",
        "latency_ms",
        "access",
        "access_result",
        "candidate",
        "candidate_behavior",
        "candidate_metric",
        "baseline",
        "baseline_behavior",
        "baseline_metric",
        "case_outcome",
        "outcome",
        "usefulness_score",
        "feedback",
    }
)


def node_scope_valid(value: Any) -> bool:
    """``replayability.requested_scope_grammar``: exactly three shapes."""

    if type(value) is not str:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    if value == "global":
        return True
    for prefix in ("project:", "session:"):
        if value.startswith(prefix) and len(value) > len(prefix):
            return not any(ord(character) < 32 or ord(character) == 127 for character in value)
    return False


def normalize_scope(value: Any) -> str:
    cleaned = str(value).strip()
    if not cleaned or cleaned == "global":
        return "global"
    if cleaned.startswith("project:") or cleaned.startswith("session:"):
        prefix, suffix = cleaned.split(":", 1)
        if not suffix:
            raise ValueError
        return f"{prefix}:{suffix}"
    if ":" in cleaned:
        prefix, suffix = cleaned.split(":", 1)
        if prefix in {"workspace", "repo"} and suffix:
            return f"project:{suffix}"
        raise ValueError
    return f"project:{cleaned}"


def ambient_scope(ambient: Mapping[str, Any]) -> str | None:
    raw_scope = ambient.get("scope")
    if raw_scope:
        return normalize_scope(str(raw_scope))
    raw_session = ambient.get("session_id") or ambient.get("session")
    if raw_session:
        return normalize_scope(f"session:{raw_session}")
    raw_project = ambient.get("project") or ambient.get("project_name") or ambient.get("workspace")
    if raw_project:
        return normalize_scope(str(raw_project))
    raw_path = ambient.get("workspace_path") or ambient.get("cwd")
    if raw_path:
        name = Path(str(raw_path)).name
        if name:
            return normalize_scope(name)
    return None


def ambient_project(ambient: Mapping[str, Any], scope: str | None) -> str | None:
    if scope is not None and scope.startswith("project:"):
        return scope
    project_scope = ambient.get("project_scope")
    if project_scope:
        normalized = normalize_scope(str(project_scope))
        if normalized.startswith("project:"):
            return normalized
    raw_project = ambient.get("project") or ambient.get("project_name") or ambient.get("workspace")
    if raw_project:
        normalized = normalize_scope(str(raw_project))
        if normalized.startswith("project:"):
            return normalized
    raw_path = ambient.get("workspace_path") or ambient.get("cwd")
    if raw_path:
        name = Path(str(raw_path)).name
        if name:
            normalized = normalize_scope(name)
            if normalized.startswith("project:"):
                return normalized
    return None


def expected_scope_plan(requested_scope: Any, ambient: Mapping[str, Any]) -> tuple[str, ...] | None:
    """``canonical_reconstruction_equivalence``, from inputs alone."""

    if not node_scope_valid(requested_scope):
        return None
    try:
        scope = ambient_scope(ambient)
        project = ambient_project(ambient, scope)
        normalized = normalize_scope(requested_scope)
    except (TypeError, ValueError, UnicodeError, OSError):
        return None
    if normalized != requested_scope:
        return None
    if normalized == "global":
        return ("global",)
    if normalized.startswith("project:"):
        return (normalized, "global")
    values = [normalized]
    if project is not None and project != normalized:
        values.append(project)
    values.append("global")
    return tuple(dict.fromkeys(values))


def depth_accepted(value: Any) -> bool:
    if value is None:
        return True
    if type(value) is not str:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def ambient_scalars_valid(ambient: Any) -> bool:
    """The twelve allowlisted keys; any other key is unconditionally invalid."""

    if type(ambient) is not dict:
        return False
    for key, value in ambient.items():
        if type(key) is not str or key not in AMBIENT_KEY_SET:
            return False
        if value is None or type(value) in (str, bool, int):
            if type(value) is str:
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError:
                    return False
            continue
        if type(value) is float and value == value and value not in (
            float("inf"),
            float("-inf"),
        ):
            continue
        return False
    return True


def same_json_scalar(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


# ``replayability.shared_seed_state_validation.policy_key_mapping_exact``.  A
# retrieval-policy key has its own grammar; the node-scope grammar is never
# applied to it.
POLICY_KEY_MAPPING_EXACT = {
    "default": "default",
    "project": "project",
    "global": "global",
    "session": "session",
    "project:<nonempty UTF-8 suffix>": "project",
    "session:<nonempty UTF-8 suffix>": "session",
}
REQUIRED_PRODUCTION_POLICY_KEYS = frozenset({"default", "project", "global", "session"})
PRODUCTION_SEED_CONTRACT_ID = "confirmatory-holdout-v4-production-seed-state-v1"


def policy_key_kind(value: Any) -> str:
    """Exactly the six mapping rules, and nothing else."""

    if type(value) is not str:
        return fail(PRODUCTION_SHAPE)
    if value in ("default", "project", "global", "session"):
        return value
    for prefix in ("project:", "session:"):
        if value.startswith(prefix):
            # ``project:`` and ``session:`` with an empty suffix reject.
            if len(value) > len(prefix):
                return prefix[:-1]
            return fail(PRODUCTION_SHAPE)
    # ``scope:project:p`` is not a policy key.
    return fail(PRODUCTION_SHAPE)


def policy_key_accepted(value: Any) -> str | None:
    try:
        return policy_key_kind(value)
    except VerificationError:
        return None


# --------------------------------------------------------------------------
# ``partition``
# --------------------------------------------------------------------------


def partition_digest(representative: bytes) -> bytes:
    return hashlib.sha256(PARTITION_DOMAIN + representative).digest()


def partition_bucket(representative: bytes) -> int:
    return int.from_bytes(partition_digest(representative)[0:4], "big") % SPLIT_MODULUS


def partition_for_representative(representative: bytes) -> str:
    bucket = partition_bucket(representative)
    return HOLDOUT_PARTITION if bucket < HOLDOUT_UPPER_EXCLUSIVE else SHADOW_PARTITION


class DisjointSet:
    """Independent component formation over the two ``component_edges``."""

    __slots__ = ("parent",)

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != root:
            self.parent[value], value = root, self.parent[value]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def classes(self) -> dict[str, set[str]]:
        groups: dict[str, set[str]] = {}
        for member in self.parent:
            groups.setdefault(self.find(member), set()).add(member)
        return groups


# --------------------------------------------------------------------------
# Document schemas, re-spelled
# --------------------------------------------------------------------------

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
POPULATION_FIELDS = (
    "selected_event_count",
    "selected_replayable_event_count",
    "selected_nonreplayable_event_count",
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

MANIFEST_KEYS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "frozen",
    "synthetic_fixture",
    "packet_layout",
    "aggregate_counts",
    "floors_pass",
    "publication",
    "semantic_reads",
    "bindings",
)
PACKET_LAYOUT_KEYS = ("corpus", "manifest")
PUBLICATION_KEYS = (
    "packet_publication_attempts",
    "no_overwrite",
    "content_before_manifest",
    "manifest_last",
    "canonical_manifest_present",
)

# ``hash_binding.packet_manifest_must_bind`` -- twenty-six items, every one a
# SHA-256 plus byte size, nothing path-only.
SINGLE_BINDING_KEYS = (
    "release_control_watermark",
    "release_manifest_reverse_pin",
    "replay_code_control_commit_and_tree",
    "unchanged_repair_design",
    "local_alias_immutable_snapshot",
    "alt_alias_immutable_snapshot",
    "canonical_two_alias_snapshot_set_identity",
    "policy_document",
    "readme_document",
    "analysis_plan_document",
    "retired_replacement_manifest",
    "complete_selected_holdout_partition",
    "complete_selected_shadow_partition",
    "keyed_aggregate_preseal_receipt",
)
GROUP_BINDING_MEMBERS = {
    "active_runtime_segment_attestation_and_complete_tuple": (
        "segment_attestation",
        "complete_unaliased_service_tuple",
    ),
    "active_source_binding_attestation_and_core": (
        "source_binding_attestation",
        "source_binding_core",
    ),
    "first_ready_scheduled_slot_and_ledger_predecessor_chain": (
        "ready_resolution",
        "ledger_predecessor_chain",
    ),
    "distinct_source_authority_and_alias_binding_attestations": (
        "local_authority",
        "alt_authority",
        "alias_binding",
    ),
    "original_frozen_manifest_and_split_declaration": (
        "original_manifest",
        "split_declaration",
    ),
    "complete_frozen_dev_reference_and_raw_identity_reference": (
        "frozen_dev_reference",
        "raw_identity_reference",
    ),
    "retired_v2_documents": (
        "README.md",
        "POLICY.md",
        "analysis-plan.json",
        "attempt-note.json",
        "scanner",
    ),
    "retired_v3_documents": (
        "README.md",
        "POLICY.md",
        "analysis-plan.json",
        "protocol-test",
    ),
    "packet_builder_and_independent_verifier": (
        "packet_builder",
        "independent_verifier",
    ),
    "seal_launcher_and_deidentification_implementation": (
        "seal_launcher",
        "deidentification_implementation",
    ),
    "partition_source_seed_states_and_construction_manifest": (
        *SEED_STATE_NAMES,
        SEED_MANIFEST_NAME,
    ),
}
# The one binding group whose membership is not spelled here but settled
# against disk.  Every campaign tool that touches evidence is bound by its own
# repo-relative path, and ``audit_campaign_tools`` requires the bound set and
# the set on disk to be equal: a tool no manifest entry names is rejected
# exactly as hard as a bound entry whose file has drifted.
#
# The tools are recognised by name shape rather than by module name on purpose.
# This verifier keeps no import path back to any implementation, so it must not
# spell one -- and a rule about shape also catches the tool that does not exist
# yet, which an enumeration of names could not.
CAMPAIGN_TOOL_BINDING_KEY = "runtime_observer_aggregate_probe_and_accrual_ledger_with_tests"
CAMPAIGN_TOOL_DIRECTORIES = ("scripts", "tests")
CAMPAIGN_TOOL_PATTERNS = {
    "scripts": (("ap_confirmatory_", "_v4.py"), ("v4_", ".py")),
    "tests": (("test_ap_confirmatory_", "_v4.py"), ("test_v4_", ".py")),
}

MANIFEST_BINDING_KEYS = (
    *SINGLE_BINDING_KEYS,
    *GROUP_BINDING_MEMBERS,
    CAMPAIGN_TOOL_BINDING_KEY,
)


def is_campaign_tool(directory: str, name: str) -> bool:
    """``<prefix>...<suffix>`` in a tool directory, with a nonempty middle."""

    for prefix, suffix in CAMPAIGN_TOOL_PATTERNS[directory]:
        if name.startswith(prefix) and name.endswith(suffix):
            if len(name) > len(prefix) + len(suffix):
                return True
    return False


def campaign_tool_path(value: Any) -> bool:
    """A binding member name that is a repo-relative campaign tool path."""

    if type(value) is not str:
        return False
    directory, separator, name = value.partition("/")
    if not separator or directory not in CAMPAIGN_TOOL_DIRECTORIES:
        return False
    return is_campaign_tool(directory, name)


PRESEAL_RECEIPT_KEYS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "status",
    "mismatch_count",
    "slot_index",
    "scheduled_at",
    "grace_deadline_at",
    "active_segment_lower_bound_exclusive_at",
    "release_effective_at",
    "segment_id",
    "snapshot_set_sha256_and_bytes",
    "aliased_source_snapshot_sha256_and_bytes",
    *AGGREGATE_FIELDS,
    "floors_pass",
    "semantic_reads",
    "synthetic_fixture",
)
SEED_MANIFEST_KEYS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "contract_id",
    "copied_table_column_allowlist",
    "forced_empty_state",
    "derived_index_prefixes",
    "policy_key_mapping_exact",
    "source_retrieval_policy_keys_required",
    "recall_events_at_seed",
    "unclassified_mutable_table_action",
    "states",
)

CASE_RECORD_KEYS = (
    "record_kind",
    "event_label",
    "created_at",
    "partition",
    "source_label",
    "component_label",
    "family_label",
    "workflow_label",
    "floor_counted",
    "seed_state",
    "replay_input",
)
INVENTORY_RECORD_KEYS = (
    "record_kind",
    "event_label",
    "created_at",
    "partition",
    "source_label",
    "component_label",
    "replayable",
)
REPLAY_INPUT_KEYS = (
    "query",
    "requested_scope",
    "resolved_scopes",
    "max_results",
    "depth",
    "ambient_context",
    "agent",
    "task",
    "session_id",
    "transport_session_id",
)

# ``reader-authority``: the only two v4 readers, both at zero semantic reads.
SEMANTIC_READER_IDS = ("confirmatory-shadow-v4-eval", "confirmatory-holdout-v4-eval")
# No reader aliases, delegation, wildcards or fallbacks anywhere.
READER_AUTHORITY_FORBIDDEN_KEY_PARTS = (
    "alias_of",
    "aliases",
    "delegat",
    "wildcard",
    "fallback",
    "inherit",
    "any_reader",
    "additional_reader",
    "reader_alias",
)

# ``measurement.state_isolation.construction`` -- three copied tables, exact
# columns, and the forced-empty state.
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
DERIVED_INDEX_PREFIXES = ("nodes_fts",)
SEED_TABLES = frozenset({*SEED_COLUMN_ALLOWLIST, *FORCED_EMPTY_TABLES})
NODE_LEVELS = frozenset({"trace", "concept", "schema"})
RELATION_TYPES = frozenset({"related", "caused", "contradicts", "supersedes", "requires"})
SOURCE_EVENT_COLUMNS = (
    "id",
    "created_at",
    "query",
    "scope",
    "requested_scope",
    "resolved_scopes",
    "ambient_context",
    "depth",
    "max_results",
    "agent",
    "task",
    "session_id",
    "transport_session_id",
)


# --------------------------------------------------------------------------
# Sealed corpus records
# --------------------------------------------------------------------------


def opaque_label(value: Any, code: str) -> str:
    """A fresh packet-local label: exactly thirty-two lowercase hex digits."""

    if type(value) is not str or HEX32_RE.match(value) is None:
        return fail(code)
    return value


def deidentified_text(value: Any, code: str = PRIVACY) -> str:
    """``privacy.deidentification``: only ``a-z`` plus the four whitespace.

    Length and the space/tab/newline/CR positions are all the surrogate may
    preserve, so any other character is retained plaintext by definition.
    """

    if type(value) is not str or DEIDENTIFIED_RE.match(value) is None:
        return fail(code)
    return value


def parse_corpus(raw: bytes, partition: str) -> list[dict[str, Any]]:
    """One canonical JSON object per line, newline-terminated, in order."""

    if raw and not raw.endswith(b"\n"):
        fail(SCHEMA)
    records: list[dict[str, Any]] = []
    previous_microseconds = -1
    for line in raw.split(b"\n")[:-1]:
        record = load_canonical(line, expected=dict)
        kind = record.get("record_kind")
        if kind == "case":
            exact_keys(record, CASE_RECORD_KEYS)
        elif kind == "inventory":
            exact_keys(record, INVENTORY_RECORD_KEYS)
        else:
            fail(SCHEMA)
        if record["partition"] != partition:
            fail(PARTITION)
        opaque_label(record["event_label"], SCHEMA)
        opaque_label(record["source_label"], SCHEMA)
        opaque_label(record["component_label"], SCHEMA)
        microseconds = timestamp_microseconds(record["created_at"])
        # ``selection.ordering``: parsed microseconds, never source text.
        if microseconds < previous_microseconds:
            fail(PARTITION)
        previous_microseconds = microseconds
        records.append(record)
    return records


def audit_case_record(record: Mapping[str, Any]) -> None:
    """One outcome-free, input-only replayable case."""

    if not is_bool(record["floor_counted"]):
        fail(SCHEMA)
    if record["seed_state"] not in SEED_STATE_NAMES:
        fail(PRODUCTION_SHAPE)
    if not record["seed_state"].startswith(f"seed-state-{record['partition']}-"):
        fail(PARTITION)
    for key in ("family_label", "workflow_label"):
        if record[key] is not None:
            opaque_label(record[key], SCHEMA)

    replay_input = exact_keys(record["replay_input"], REPLAY_INPUT_KEYS)
    # Not one of the nine forbidden inputs may appear at any depth, under any
    # spelling, anywhere in the record.
    for path, _node in walk(record):
        for key in path:
            if key in FORBIDDEN_REPLAY_INPUT_FIELDS:
                fail(REPLAYABILITY)

    query = replay_input["query"]
    if type(query) is not str or not query:
        fail(REPLAYABILITY)
    deidentified_text(query)
    if not node_scope_valid(replay_input["requested_scope"]):
        fail(REPLAYABILITY)
    resolved = replay_input["resolved_scopes"]
    if type(resolved) is not list or not all(
        type(value) is str and node_scope_valid(value) for value in resolved
    ):
        fail(REPLAYABILITY)
    if not is_int(replay_input["max_results"]) or replay_input["max_results"] <= 0:
        fail(REPLAYABILITY)
    if not depth_accepted(replay_input["depth"]):
        fail(REPLAYABILITY)
    ambient = replay_input["ambient_context"]
    if not ambient_scalars_valid(ambient):
        fail(REPLAYABILITY)
    plan = expected_scope_plan(replay_input["requested_scope"], ambient)
    if plan is None or tuple(resolved) != plan:
        fail(REPLAYABILITY)
    if not same_json_scalar(replay_input["agent"], ambient.get("agent")):
        fail(REPLAYABILITY)
    if not same_json_scalar(replay_input["task"], ambient.get("task")):
        fail(REPLAYABILITY)
    caller_session = ambient.get("session_id") or ambient.get("session")
    if not same_json_scalar(replay_input["session_id"], caller_session):
        fail(REPLAYABILITY)
    if not same_json_scalar(
        replay_input["transport_session_id"], ambient.get("transport_session_id")
    ):
        fail(REPLAYABILITY)
    # ``transport_session_id`` is the workflow key: nullness and equality must
    # agree with the sealed workflow label.
    if (replay_input["transport_session_id"] is None) != (record["workflow_label"] is None):
        fail(REPLAYABILITY)


def audit_inventory_record(record: Mapping[str, Any]) -> None:
    """A selected nonreplayable event: accounted for, carrying no inputs."""

    if record["replayable"] is not False:
        fail(REPLAYABILITY)
    # ``selected_nonreplayable_floor_contribution: 0``.  The field is absent by
    # schema, so an inventory record can never be a floor witness.
    if "floor_counted" in record or "replay_input" in record:
        fail(REPLAYABILITY)


def audit_records(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        if record["record_kind"] == "case":
            audit_case_record(record)
        else:
            audit_inventory_record(record)


# --------------------------------------------------------------------------
# ``partition``: independent recomputation
# --------------------------------------------------------------------------


def audit_partition(
    records: Sequence[Mapping[str, Any]], vectors: Sequence[Mapping[str, Any]]
) -> None:
    """Recompute components, representatives and buckets, then the invariants."""

    if not vectors:
        fail(PARTITION)
    for vector in vectors:
        exact_keys(
            vector,
            ("representative_display", "digest_sha256", "bucket", "partition"),
            PARTITION,
        )
        display = vector["representative_display"]
        if type(display) is not str or display.count("\\0") != 1:
            fail(PARTITION)
        alias, _, event_id = display.partition("\\0")
        representative = alias.encode("utf-8") + b"\0" + event_id.encode("utf-8")
        if not hmac.compare_digest(
            partition_digest(representative).hex(), hex64(vector["digest_sha256"], PARTITION)
        ):
            fail(PARTITION)
        if not is_int(vector["bucket"]) or partition_bucket(representative) != vector["bucket"]:
            fail(PARTITION)
        if partition_for_representative(representative) != vector["partition"]:
            fail(PARTITION)

    labels = [record["event_label"] for record in records]
    if len(labels) != len(set(labels)):
        fail(PARTITION)
    by_label = {record["event_label"]: record for record in records}

    # ``event_intersection_required: 0`` and complete union: every record sits
    # in exactly one partition file, and the two files cover the population.
    holdout = {label for label, record in by_label.items() if record["partition"] == HOLDOUT_PARTITION}
    shadow = {label for label, record in by_label.items() if record["partition"] == SHADOW_PARTITION}
    if holdout & shadow or holdout | shadow != set(by_label):
        fail(PARTITION)

    # Component formation, from the two declared edges only.
    disjoint = DisjointSet()
    for label in by_label:
        disjoint.add(label)
    groups: dict[tuple[str, str, str], list[str]] = {}
    for record in records:
        if record["record_kind"] != "case":
            continue
        for kind, value in (("family", record["family_label"]), ("workflow", record["workflow_label"])):
            if value is not None:
                groups.setdefault((kind, record["source_label"], value), []).append(
                    record["event_label"]
                )
    for members in groups.values():
        for member in members[1:]:
            disjoint.union(members[0], member)

    packet_components: dict[str, set[str]] = {}
    for record in records:
        packet_components.setdefault(record["component_label"], set()).add(record["event_label"])
    owner = {
        label: component
        for component, members in packet_components.items()
        for label in members
    }

    for members in disjoint.classes().values():
        # Every recomputed class must lie inside exactly one packet component.
        if len({owner[label] for label in members}) != 1:
            fail(PARTITION)
    for component, members in packet_components.items():
        recomputed = {disjoint.find(label) for label in members}
        if len(recomputed) == 1:
            continue
        # A nonreplayable event keeps its component edges but exposes no
        # family or workflow label, so extra merging is admissible only where
        # an inventory record can explain it.
        if not any(by_label[label]["record_kind"] == "inventory" for label in members):
            fail(PARTITION)

    # One bucket per component; nothing searched, retried, rebalanced or moved.
    for members in packet_components.values():
        if len({by_label[label]["partition"] for label in members}) != 1:
            fail(PARTITION)

    # Zero family, session and workflow crossings.
    for kind in ("family_label", "workflow_label"):
        crossings: dict[tuple[str, str], set[str]] = {}
        for record in records:
            if record["record_kind"] != "case" or record[kind] is None:
                continue
            crossings.setdefault((record["source_label"], record[kind]), set()).add(
                record["partition"]
            )
        if any(len(value) > 1 for value in crossings.values()):
            fail(PARTITION)

    # ``outcome_field_invariant``/``input_order_invariant``: a record carries
    # no outcome and no ordinal, so no field the packet exposes could have
    # steered its bucket.  Prove the negative structurally.
    for record in records:
        for path, _node in walk(record):
            for key in path:
                if key in FORBIDDEN_REPLAY_INPUT_FIELDS or key in ("index", "ordinal", "position"):
                    fail(PARTITION)


# --------------------------------------------------------------------------
# ``replayability.deidentified_identity_equivalence``
# --------------------------------------------------------------------------


def normalized_surrogate_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    """``identity.normalization`` applied to the sealed surrogate."""

    replay_input = record["replay_input"]
    return " ".join(replay_input["query"].split()), replay_input["requested_scope"]


def audit_identity_equivalence(records: Sequence[Mapping[str, Any]]) -> None:
    """Within each source alias, family equality iff surrogate equality.

    ``family_label`` is the raw-side equality class: it is minted from the
    ephemeral HMAC token over the raw normalized identity.  The surrogate side
    is the de-identified normalized identity.  If de-identification split a
    repeated family, two cases would share a family label and differ in
    surrogate identity; if it merged two families, two cases would share a
    surrogate identity and differ in family label.  Both are fatal.
    """

    forward: dict[tuple[str, str], tuple[str, str]] = {}
    reverse: dict[tuple[str, str, str], tuple[str, str]] = {}
    for record in records:
        if record["record_kind"] != "case" or record["family_label"] is None:
            continue
        source = record["source_label"]
        family = (source, record["family_label"])
        identity = normalized_surrogate_identity(record)
        if forward.setdefault(family, identity) != identity:
            fail(PRIVACY)
        if reverse.setdefault((source, *identity), family) != family:
            fail(PRIVACY)


# --------------------------------------------------------------------------
# Aggregate reconstruction
# --------------------------------------------------------------------------


def recompute_aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """All fourteen counts, recomputed from the sealed corpus alone."""

    cases = [record for record in records if record["record_kind"] == "case"]
    inventory = [record for record in records if record["record_kind"] == "inventory"]

    def automatic(record: Mapping[str, Any]) -> bool:
        # ``agent IS NULL`` is the automatic classification; organic is its
        # complement, exactly as the frozen selection defines them.
        return record["replay_input"]["agent"] is None

    holdout_cases = [record for record in cases if record["partition"] == HOLDOUT_PARTITION]
    holdout_automatic = [
        record for record in holdout_cases if record["floor_counted"] and automatic(record)
    ]
    holdout_organic = [
        record for record in holdout_cases if record["floor_counted"] and not automatic(record)
    ]
    shadow_calls = [
        record
        for record in cases
        if record["partition"] == SHADOW_PARTITION and record["floor_counted"]
    ]

    # ``floor_counted_rule``: every floor witness carries a sealed replayable
    # case, and every counted automatic event carries its family witness.
    if any(record["family_label"] is None for record in holdout_automatic):
        fail(REPLAYABILITY)
    if any(record["workflow_label"] is None for record in shadow_calls):
        fail(REPLAYABILITY)

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
        "holdout_organic_component_count": len({row["component_label"] for row in holdout_organic}),
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


def floors_pass(counts: Mapping[str, int]) -> bool:
    return all(counts[field] >= minimum for field, minimum in HOLDOUT_FLOORS.items()) and all(
        counts[field] >= minimum for field, minimum in SHADOW_FLOORS.items()
    )


# --------------------------------------------------------------------------
# ``privacy``
# --------------------------------------------------------------------------


def encoded_needles(values: Iterable[bytes]) -> set[bytes]:
    """Raw, hex, HEX, base64 and urlsafe-base64, padded and unpadded."""

    needles: set[bytes] = set()
    for value in values:
        if not value:
            continue
        needles.update(
            (
                value,
                value.hex().encode("ascii"),
                value.hex().upper().encode("ascii"),
                base64.b64encode(value),
                base64.b64encode(value).rstrip(b"="),
                base64.urlsafe_b64encode(value),
                base64.urlsafe_b64encode(value).rstrip(b"="),
            )
        )
    return {needle for needle in needles if needle}


def privacy_scan(blobs: Iterable[bytes], needles: set[bytes]) -> None:
    for blob in blobs:
        for needle in needles:
            if needle in blob:
                fail(PRIVACY)


def audit_no_digest_material(blobs: Iterable[bytes]) -> None:
    """No secret-shaped material may live in the sealed case records.

    Packet-local labels are thirty-two hex digits by construction; a sixty-four
    digit run is an HMAC token, a commitment or an unkeyed fingerprint, none of
    which may be persisted.

    Only the two textual corpora are scanned this way.  A seed state is a
    packed SQLite file in which two adjacent thirty-two digit labels are
    physically contiguous, so a byte-run scan cannot discriminate there; its
    values are instead constrained exactly, column by column, in
    ``audit_seed_state``.
    """

    for raw in blobs:
        if DIGEST_SHAPED_RE.search(raw.decode("utf-8", errors="ignore")) is not None:
            fail(PRIVACY)


# --------------------------------------------------------------------------
# ``production-shape``: seed state
# --------------------------------------------------------------------------


def audit_seed_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    exact_keys(manifest, SEED_MANIFEST_KEYS)
    if manifest["schema_version"] != SCHEMA_VERSION or not is_int(manifest["schema_version"]):
        fail(SCHEMA)
    if manifest["namespace"] != NAMESPACE:
        fail(SCHEMA)
    if manifest["receipt_kind"] != "packet-seed-state-manifest":
        fail(SCHEMA)
    if manifest["contract_id"] != PRODUCTION_SEED_CONTRACT_ID:
        fail(PRODUCTION_SHAPE)

    allowlist = exact_keys(
        manifest["copied_table_column_allowlist"], SEED_COLUMN_ALLOWLIST, PRODUCTION_SHAPE
    )
    for table, columns in SEED_COLUMN_ALLOWLIST.items():
        if allowlist[table] != list(columns):
            fail(PRODUCTION_SHAPE)
    if manifest["forced_empty_state"] != list(FORCED_EMPTY_TABLES):
        fail(PRODUCTION_SHAPE)
    if manifest["derived_index_prefixes"] != list(DERIVED_INDEX_PREFIXES):
        fail(PRODUCTION_SHAPE)
    if manifest["policy_key_mapping_exact"] != POLICY_KEY_MAPPING_EXACT:
        fail(PRODUCTION_SHAPE)
    if manifest["source_retrieval_policy_keys_required"] != sorted(
        REQUIRED_PRODUCTION_POLICY_KEYS
    ):
        fail(PRODUCTION_SHAPE)
    if manifest["recall_events_at_seed"] != "empty":
        fail(PRODUCTION_SHAPE)
    if manifest["unclassified_mutable_table_action"] != "fatal-preseal-integrity-failure":
        fail(PRODUCTION_SHAPE)

    states = exact_keys(manifest["states"], SEED_STATE_NAMES, PRODUCTION_SHAPE)
    for name in SEED_STATE_NAMES:
        entry = exact_keys(states[name], ("table_row_counts",), PRODUCTION_SHAPE)
        counts = exact_keys(entry["table_row_counts"], SEED_TABLES, PRODUCTION_SHAPE)
        for table in FORCED_EMPTY_TABLES:
            if counts[table] != 0 or not is_int(counts[table]):
                fail(PRODUCTION_SHAPE)
        for table in SEED_COLUMN_ALLOWLIST:
            nonnegative_int(counts[table], PRODUCTION_SHAPE)
        if counts["retrieval_weights"] <= 0:
            fail(PRODUCTION_SHAPE)
    return states


def expected_seed_schema() -> dict[str, str]:
    """The exact CREATE statements a conforming seed state must carry."""

    statements = {
        "nodes": (
            "CREATE TABLE nodes ("
            "packet_node_label TEXT PRIMARY KEY, level TEXT NOT NULL, "
            "deidentified_content TEXT NOT NULL, packet_scope_label TEXT NOT NULL, "
            "created_at TEXT NOT NULL)"
        ),
        "connections": (
            "CREATE TABLE connections ("
            "source_packet_node_label TEXT NOT NULL, target_packet_node_label TEXT NOT NULL, "
            "relation_type TEXT NOT NULL, weight REAL NOT NULL, "
            "PRIMARY KEY (source_packet_node_label, target_packet_node_label, relation_type))"
        ),
        "retrieval_weights": (
            "CREATE TABLE retrieval_weights ("
            "packet_scope_label TEXT PRIMARY KEY, bm25_weight REAL NOT NULL, "
            "vector_weight REAL NOT NULL, graph_weight REAL NOT NULL)"
        ),
    }
    for table in FORCED_EMPTY_TABLES:
        if table == "recall_events":
            columns = ", ".join(f"{column} TEXT" for column in SOURCE_EVENT_COLUMNS)
            statements[table] = f"CREATE TABLE recall_events ({columns})"
        else:
            statements[table] = f"CREATE TABLE {table} (packet_placeholder TEXT)"
    return statements


def audit_seed_state(path: Path, declared: Mapping[str, int]) -> None:
    """Three copied tables, an allowlisted column set, and nothing else."""

    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    except sqlite3.Error:
        return fail(PRODUCTION_SHAPE)
    try:
        rows = connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY name"
        ).fetchall()
        expected = expected_seed_schema()
        seen: set[str] = set()
        for kind, name, sql in rows:
            if kind == "index" and name.startswith("sqlite_autoindex_"):
                continue
            if kind != "table" or name not in SEED_TABLES:
                # An unclassified mutable table is a fatal integrity failure.
                fail(PRODUCTION_SHAPE)
            if str(sql) != expected[name]:
                fail(PRODUCTION_SHAPE)
            seen.add(name)
        if seen != SEED_TABLES:
            fail(PRODUCTION_SHAPE)

        for table in FORCED_EMPTY_TABLES:
            if connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] != 0:
                fail(PRODUCTION_SHAPE)
        for table in SEED_COLUMN_ALLOWLIST:
            count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if count != declared[table]:
                fail(PRODUCTION_SHAPE)
        # ``retrieval_weights_table_must_be_nonempty``.
        weights = connection.execute(
            "SELECT packet_scope_label, bm25_weight, vector_weight, graph_weight "
            "FROM retrieval_weights ORDER BY packet_scope_label"
        ).fetchall()
        if not weights:
            fail(PRODUCTION_SHAPE)
        kinds: set[str] = set()
        for label, bm25, vector, graph in weights:
            kinds.add(policy_key_kind(label))
            for value in (bm25, vector, graph):
                if type(value) is not float or value != value:
                    fail(PRODUCTION_SHAPE)
        if not REQUIRED_PRODUCTION_POLICY_KEYS.issubset(kinds):
            fail(PRODUCTION_SHAPE)

        node_labels: set[str] = set()
        for label, level, content, scope_label, created_at in connection.execute(
            "SELECT packet_node_label, level, deidentified_content, packet_scope_label, "
            "created_at FROM nodes ORDER BY packet_node_label"
        ):
            opaque_label(label, PRODUCTION_SHAPE)
            if level not in NODE_LEVELS:
                fail(PRODUCTION_SHAPE)
            deidentified_text(content)
            if not node_scope_valid(scope_label):
                fail(PRODUCTION_SHAPE)
            timestamp(created_at, PRODUCTION_SHAPE)
            node_labels.add(label)
        for source, target, relation, weight in connection.execute(
            "SELECT source_packet_node_label, target_packet_node_label, relation_type, "
            "weight FROM connections"
        ):
            if source not in node_labels or target not in node_labels:
                fail(PRODUCTION_SHAPE)
            if relation not in RELATION_TYPES:
                fail(PRODUCTION_SHAPE)
            if type(weight) is not float or weight != weight:
                fail(PRODUCTION_SHAPE)
    except sqlite3.Error:
        return fail(PRODUCTION_SHAPE)
    finally:
        try:
            connection.close()
        except sqlite3.Error:
            pass


def audit_policy_key_grammar(plan: Mapping[str, Any]) -> None:
    """Replay the plan's own policy-key vectors through this module."""

    validation = plan["replayability"]["shared_seed_state_validation"]
    if validation["contract_id"] != PRODUCTION_SEED_CONTRACT_ID:
        fail(PRODUCTION_SHAPE)
    if validation["retrieval_weights_table_must_be_nonempty"] is not True:
        fail(PRODUCTION_SHAPE)
    if sorted(validation["production_policy_keys_required"]) != sorted(
        REQUIRED_PRODUCTION_POLICY_KEYS
    ):
        fail(PRODUCTION_SHAPE)
    if validation["policy_key_mapping_exact"] != POLICY_KEY_MAPPING_EXACT:
        fail(PRODUCTION_SHAPE)
    # The node-scope grammar is never applied to a retrieval-policy key.
    if validation["node_scope_grammar_applied_to_retrieval_weight_policy_key"] is not False:
        fail(PRODUCTION_SHAPE)
    for vector in validation["policy_key_vectors"]:
        kind = policy_key_accepted(vector["input"])
        if bool(vector["accepted"]) != (kind is not None) or kind != vector["kind"]:
            fail(PRODUCTION_SHAPE)
    # ``project:`` and ``session:`` with an empty suffix, and ``scope:project:p``,
    # all reject; ``default`` is a policy key but not a node scope.
    for rejected in ("project:", "session:", "scope:project:p", "unknown", ""):
        if policy_key_accepted(rejected) is not None:
            fail(PRODUCTION_SHAPE)
    if node_scope_valid("default") or policy_key_accepted("default") != "default":
        fail(PRODUCTION_SHAPE)
    for value, valid in (
        ("global", True),
        ("project:p", True),
        ("session:s", True),
        ("project:", False),
        ("session:", False),
        ("scope:project:p", False),
        ("unknown", False),
        ("project:\n", False),
    ):
        if node_scope_valid(value) is not valid:
            fail(REPLAYABILITY)


# --------------------------------------------------------------------------
# ``reader-authority``
# --------------------------------------------------------------------------


def audit_semantic_reads(value: Any) -> None:
    """Exactly the two-key map, both zero, with no alias or fallback."""

    reads = exact_keys(value, SEMANTIC_READER_IDS, READER_AUTHORITY)
    for reader in SEMANTIC_READER_IDS:
        if not is_int(reads[reader]) or reads[reader] != 0:
            fail(READER_AUTHORITY)


def audit_reader_authority(documents: Iterable[Any]) -> None:
    for document in documents:
        for path, node in walk(document):
            for key in path:
                lowered = key.lower()
                for part in READER_AUTHORITY_FORBIDDEN_KEY_PARTS:
                    if part in lowered:
                        fail(READER_AUTHORITY)
                # A retired v2/v3 reader identity must never validate as v4.
                if lowered.endswith("-eval") and key not in SEMANTIC_READER_IDS:
                    fail(READER_AUTHORITY)
            if type(node) is str:
                if node.endswith("-eval") and node not in SEMANTIC_READER_IDS:
                    fail(READER_AUTHORITY)
                for prefix in RETIRED_DOMAIN_PREFIXES:
                    if node.startswith(prefix):
                        fail(READER_AUTHORITY)
    # Self-vectors: the retired identities must not pass this predicate.
    for retired in (
        "confirmatory-shadow-v3-eval",
        "confirmatory-holdout-v3-eval",
        "confirmatory-shadow-v2-eval",
        "confirmatory-holdout-v2-eval",
        "confirmatory-holdout-v4-eval-shadow",
        "*",
    ):
        if retired in SEMANTIC_READER_IDS:
            fail(READER_AUTHORITY)


# --------------------------------------------------------------------------
# ``runtime-segment``
# --------------------------------------------------------------------------


def snapshot_set_preimage(local: Mapping[str, Any], alt: Mapping[str, Any]) -> bytes:
    """``domains.snapshot_set_preimage``, spelled out exactly."""

    return SNAPSHOT_SET_DOMAIN + canonical_bytes(
        {
            "local": {"sha256": local["sha256"], "bytes": local["bytes"]},
            "alt": {"sha256": alt["sha256"], "bytes": alt["bytes"]},
        }
    )


def audit_runtime_segment(manifest: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    """The manifest must bind the whole still-open segment story."""

    bindings = manifest["bindings"]
    local = identity_shape(bindings["local_alias_immutable_snapshot"], RUNTIME_SEGMENT)
    alt = identity_shape(bindings["alt_alias_immutable_snapshot"], RUNTIME_SEGMENT)
    # ``the local and alt snapshot SHA-256 values must differ``.
    if hmac.compare_digest(local["sha256"], alt["sha256"]):
        fail(RUNTIME_SEGMENT)

    preimage = snapshot_set_preimage(local, alt)
    declared = identity_shape(
        bindings["canonical_two_alias_snapshot_set_identity"], RUNTIME_SEGMENT
    )
    if not same_identity(declared, digest_of(preimage), len(preimage)):
        fail(RUNTIME_SEGMENT)

    # The keyed preseal receipt must agree, alias by alias, on the same bytes.
    aliased = exact_keys(
        receipt["aliased_source_snapshot_sha256_and_bytes"], SOURCE_ALIASES, RUNTIME_SEGMENT
    )
    for alias, identity in (("local", local), ("alt", alt)):
        observed = identity_shape(aliased[alias], RUNTIME_SEGMENT)
        if not same_identity(observed, identity["sha256"], identity["bytes"]):
            fail(RUNTIME_SEGMENT)
    observed_set = identity_shape(receipt["snapshot_set_sha256_and_bytes"], RUNTIME_SEGMENT)
    if not same_identity(observed_set, declared["sha256"], declared["bytes"]):
        fail(RUNTIME_SEGMENT)

    # Active segment, source binding, first-ready slot and ledger chain: each
    # group must be present with its exact member set, every member an
    # independently hash-bound artifact.
    for key in (
        "active_runtime_segment_attestation_and_complete_tuple",
        "active_source_binding_attestation_and_core",
        "first_ready_scheduled_slot_and_ledger_predecessor_chain",
        "distinct_source_authority_and_alias_binding_attestations",
    ):
        group = exact_keys(bindings[key], GROUP_BINDING_MEMBERS[key], RUNTIME_SEGMENT)
        for member in GROUP_BINDING_MEMBERS[key]:
            identity_shape(group[member], RUNTIME_SEGMENT)
            if group[member]["bytes"] <= 0:
                fail(RUNTIME_SEGMENT)

    # The slot the packet consumed, and the interval it was allowed to select.
    slot_index = receipt["slot_index"]
    if not is_int(slot_index) or not FIRST_SLOT_INDEX <= slot_index <= LAST_SLOT_INDEX:
        fail(RUNTIME_SEGMENT)
    hex64(receipt["segment_id"], RUNTIME_SEGMENT)
    scheduled = timestamp_microseconds(receipt["scheduled_at"], RUNTIME_SEGMENT)
    grace = timestamp_microseconds(receipt["grace_deadline_at"], RUNTIME_SEGMENT)
    lower = timestamp_microseconds(
        receipt["active_segment_lower_bound_exclusive_at"], RUNTIME_SEGMENT
    )
    release = timestamp_microseconds(receipt["release_effective_at"], RUNTIME_SEGMENT)
    if not release <= lower < scheduled <= grace:
        fail(RUNTIME_SEGMENT)
    if not bool(manifest["synthetic_fixture"]) and receipt["release_effective_at"] != (
        RELEASE_EFFECTIVE_AT
    ):
        fail(RUNTIME_SEGMENT)


# --------------------------------------------------------------------------
# ``manifest-order``
# --------------------------------------------------------------------------

# ``publication_and_ordering.sealed_state``, re-spelled.  ``latched_slot_status``
# and the two segment observations are proved through the bindings the sealer
# had to install, and the keyed preseal fields through the receipt those
# bindings hash-bind, because the immutable manifest may not carry consumption
# records at all.
SEALED_STATE_CONSUMPTION_KEYS = frozenset(
    {
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
)


def audit_manifest_order(
    manifest: Mapping[str, Any], receipt: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    ordering = plan["publication_and_ordering"]
    sealed_state = ordering["sealed_state"]

    if manifest["frozen"] is not True or sealed_state["frozen"] is not True:
        fail(MANIFEST_ORDER)
    publication = exact_keys(manifest["publication"], PUBLICATION_KEYS, MANIFEST_ORDER)
    if publication["content_before_manifest"] is not True or ordering[
        "content_before_manifest"
    ] is not True:
        fail(MANIFEST_ORDER)
    if publication["manifest_last"] is not True or ordering["manifest_last"] is not True:
        fail(MANIFEST_ORDER)
    if publication["canonical_manifest_present"] is not True:
        fail(MANIFEST_ORDER)
    if publication["no_overwrite"] is not True or ordering["no_overwrite"] is not True:
        fail(OVERWRITE)
    if not is_int(publication["packet_publication_attempts"]) or publication[
        "packet_publication_attempts"
    ] != 1:
        fail(OVERWRITE)
    if ordering["packet_publication_attempts"] != 1 or ordering["reseal_allowed"] is not False:
        fail(OVERWRITE)

    # ``latched_slot_status: ready`` and the complete predecessor chain: the
    # sealer could only bind these after the first-ready resolution latched.
    if sealed_state["latched_slot_status"] != "ready":
        fail(MANIFEST_ORDER)
    for key in ("ready_resolution", "ledger_predecessor_chain"):
        if key not in manifest["bindings"]["first_ready_scheduled_slot_and_ledger_predecessor_chain"]:
            fail(MANIFEST_ORDER)
    if sealed_state["first_ready_predecessor_chain_valid"] is not True:
        fail(MANIFEST_ORDER)
    # ``active_segment_open_at_atomic_handoff`` and the post-build,
    # pre-manifest observation: both are attested by the segment binding group.
    if sealed_state["active_segment_open_at_atomic_handoff"] is not True:
        fail(MANIFEST_ORDER)
    if sealed_state["active_segment_unchanged_at_post_build_pre_manifest_observation"] is not True:
        fail(MANIFEST_ORDER)
    for key in ("segment_attestation", "complete_unaliased_service_tuple"):
        if key not in manifest["bindings"]["active_runtime_segment_attestation_and_complete_tuple"]:
            fail(MANIFEST_ORDER)

    # ``keyed_preseal_status: pass`` with zero mismatches.
    if sealed_state["keyed_preseal_status"] != "pass" or receipt["status"] != "pass":
        fail(MANIFEST_ORDER)
    if sealed_state["keyed_preseal_mismatches"] != 0:
        fail(MANIFEST_ORDER)
    if not is_int(receipt["mismatch_count"]) or receipt["mismatch_count"] != 0:
        fail(MANIFEST_ORDER)

    # ``consumption_record_location``: separate hash-bound aggregate receipts
    # only, so no consumption record may appear in the immutable manifest.
    if sealed_state["consumption_record_location"] != (
        "separate hash-bound aggregate receipts only"
    ):
        fail(MANIFEST_ORDER)
    for path, _node in walk(manifest):
        for key in path:
            if key in SEALED_STATE_CONSUMPTION_KEYS:
                fail(MANIFEST_ORDER)


# --------------------------------------------------------------------------
# Manifest and receipt schemas
# --------------------------------------------------------------------------


def audit_manifest_schema(manifest: Mapping[str, Any]) -> None:
    exact_keys(manifest, MANIFEST_KEYS)
    if not is_int(manifest["schema_version"]) or manifest["schema_version"] != SCHEMA_VERSION:
        fail(SCHEMA)
    if manifest["namespace"] != NAMESPACE:
        fail(SCHEMA)
    if manifest["receipt_kind"] != "packet-manifest":
        fail(SCHEMA)
    for key in ("frozen", "synthetic_fixture", "floors_pass"):
        if not is_bool(manifest[key]):
            fail(SCHEMA)

    layout = exact_keys(manifest["packet_layout"], PACKET_LAYOUT_KEYS)
    if layout["manifest"] != PACKET_MANIFEST_NAME:
        fail(SCHEMA)
    if type(layout[PACKET_CORPUS_DIRECTORY]) is not list:
        fail(SCHEMA)
    if sorted(layout[PACKET_CORPUS_DIRECTORY]) != sorted(CORPUS_MEMBER_NAMES):
        fail(SCHEMA)

    counts = exact_keys(manifest["aggregate_counts"], AGGREGATE_FIELDS)
    for field in AGGREGATE_FIELDS:
        nonnegative_int(counts[field], SCHEMA)
    # Key sets are this schema's business at every depth; the values inside
    # ``publication`` belong to the manifest-order class.
    exact_keys(manifest["publication"], PUBLICATION_KEYS)
    audit_semantic_reads(manifest["semantic_reads"])

    bindings = exact_keys(manifest["bindings"], MANIFEST_BINDING_KEYS)
    for key in SINGLE_BINDING_KEYS:
        identity_shape(bindings[key])
    for key, members in GROUP_BINDING_MEMBERS.items():
        group = exact_keys(bindings[key], members)
        for member in members:
            identity_shape(group[member])

    # The campaign tool group names each member by its repo-relative path, so
    # only the shape of those names is this class's business.  Which paths must
    # be there is settled against disk in ``audit_campaign_tools``.
    tools = bindings[CAMPAIGN_TOOL_BINDING_KEY]
    if type(tools) is not dict or not tools:
        fail(SCHEMA)
    for member, identity in tools.items():
        if not campaign_tool_path(member):
            fail(SCHEMA)
        identity_shape(identity)

    # ``free_text_fields_allowed: false``: every string the manifest carries
    # must be a declared constant, a digest, or a declared member name.
    permitted = {
        NAMESPACE,
        "packet-manifest",
        PACKET_MANIFEST_NAME,
        *CORPUS_MEMBER_NAMES,
        *SEMANTIC_READER_IDS,
    }
    for path, node in walk(manifest):
        if type(node) is not str:
            continue
        if node in permitted or HEX64_RE.match(node) is not None:
            continue
        fail(SCHEMA)


def audit_preseal_receipt(receipt: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    exact_keys(receipt, PRESEAL_RECEIPT_KEYS)
    if not is_int(receipt["schema_version"]) or receipt["schema_version"] != SCHEMA_VERSION:
        fail(SCHEMA)
    if receipt["namespace"] != NAMESPACE:
        fail(SCHEMA)
    if receipt["receipt_kind"] != "packet-keyed-preseal":
        fail(SCHEMA)
    # ``status`` and ``mismatch_count`` are sealed-state values, so the
    # manifest-order class owns them.
    if type(receipt["status"]) is not str:
        fail(SCHEMA)
    if not is_bool(receipt["floors_pass"]) or not is_bool(receipt["synthetic_fixture"]):
        fail(SCHEMA)
    for field in AGGREGATE_FIELDS:
        nonnegative_int(receipt[field], SCHEMA)
    audit_semantic_reads(receipt["semantic_reads"])
    identity_shape(receipt["snapshot_set_sha256_and_bytes"])

    # The keyed receipt is aggregate-only and must agree with the manifest.
    counts = manifest["aggregate_counts"]
    if any(receipt[field] != counts[field] for field in AGGREGATE_FIELDS):
        fail(SCHEMA)
    if receipt["floors_pass"] is not manifest["floors_pass"]:
        fail(SCHEMA)
    if receipt["synthetic_fixture"] is not manifest["synthetic_fixture"]:
        fail(SCHEMA)


# --------------------------------------------------------------------------
# Independent source pins
# --------------------------------------------------------------------------

PINNED_DOCUMENTS = (
    ("README.md", README_SHA256, README_BYTES, "readme_document"),
    ("POLICY.md", POLICY_SHA256, POLICY_BYTES, "policy_document"),
    ("analysis-plan.json", PLAN_SHA256, PLAN_BYTES, "analysis_plan_document"),
)


def audit_source_pins(namespace_root: Path, manifest: Mapping[str, Any]) -> dict[str, bytes]:
    """Check the literals against disk bytes, then against the manifest.

    Both sides are compared to this module's constants, never to each other, so
    rewriting a packet file and rehashing the manifest still fails.
    """

    root_fd = open_directory(namespace_root, PIN)
    recipe_fd = -1
    documents: dict[str, bytes] = {}
    try:
        for name, sha256, size, binding in PINNED_DOCUMENTS:
            raw = read_at(root_fd, name, PIN)
            if len(raw) != size or not hmac.compare_digest(digest_of(raw), sha256):
                fail(PIN)
            if not same_identity(manifest["bindings"][binding], sha256, size):
                fail(PIN)
            documents[name] = raw
        recipe_fd = open_child_directory(root_fd, "recipe", PIN)
        builder = read_at(recipe_fd, "build.py", PIN)
        if len(builder) != BUILDER_BYTES or not hmac.compare_digest(
            digest_of(builder), BUILDER_SHA256
        ):
            fail(PIN)
        documents["recipe/build.py"] = builder
    finally:
        close_quietly(recipe_fd)
        close_quietly(root_fd)

    # The builder binds itself twice: as the packet builder and as the
    # de-identification implementation.
    group = manifest["bindings"]["packet_builder_and_independent_verifier"]
    if not same_identity(group["packet_builder"], BUILDER_SHA256, BUILDER_BYTES):
        fail(PIN)
    deidentification = manifest["bindings"]["seal_launcher_and_deidentification_implementation"]
    if not same_identity(
        deidentification["deidentification_implementation"], BUILDER_SHA256, BUILDER_BYTES
    ):
        fail(PIN)

    # ``replay_code_control_commit_and_tree`` binds content, not a mutable head.
    for value in (REPLAY_CODE_CONTROL_COMMIT, REPLAY_CODE_CONTROL_TREE):
        if HEX40_RE.match(value) is None:
            fail(PIN)
    preimage = canonical_bytes(
        {"commit": REPLAY_CODE_CONTROL_COMMIT, "tree": REPLAY_CODE_CONTROL_TREE}
    )
    if not same_identity(
        manifest["bindings"]["replay_code_control_commit_and_tree"],
        digest_of(preimage),
        len(preimage),
    ):
        fail(PIN)

    # The operator-private raw identity reference is a declaration only, and
    # is pinned here so a substituted reference cannot pass.
    reference = manifest["bindings"]["complete_frozen_dev_reference_and_raw_identity_reference"]
    if not same_identity(
        reference["raw_identity_reference"],
        RAW_IDENTITY_REFERENCE_SHA256,
        RAW_IDENTITY_REFERENCE_BYTES,
    ):
        fail(PIN)
    return documents


def audit_campaign_tools(namespace_root: Path, manifest: Mapping[str, Any]) -> None:
    """Bind every evidence-touching campaign tool by path, digest and size.

    Closed in both directions, and fail-closed in both.  A tool that exists on
    disk but that no manifest entry names is rejected exactly as hard as a
    bound entry whose file has since drifted: unbound code that produces
    evidence is the hole this exists to shut, not something to warn about.

    The bytes are read and hashed here.  Nothing is imported and no path the
    manifest supplies is ever opened -- the set of files to read comes from
    ``listdir`` on the two tool directories, and the manifest only ever gets
    compared against what that sweep already found.
    """

    repository_root = namespace_root.parents[3]
    bound = manifest["bindings"][CAMPAIGN_TOOL_BINDING_KEY]

    observed: dict[str, tuple[str, int]] = {}
    root_fd = open_directory(repository_root, PIN)
    try:
        for directory in CAMPAIGN_TOOL_DIRECTORIES:
            directory_fd = open_child_directory(root_fd, directory, PIN)
            try:
                for name in sorted(listdir_at(directory_fd, PIN)):
                    if not is_campaign_tool(directory, name):
                        continue
                    # A tool-shaped name that is not a plain regular file --
                    # a symlink, a directory, a device -- fails here rather
                    # than being skipped past the closure check below.
                    raw = read_at(directory_fd, name, PIN)
                    observed[f"{directory}/{name}"] = (digest_of(raw), len(raw))
            finally:
                close_quietly(directory_fd)
    finally:
        close_quietly(root_fd)

    if set(observed) != set(bound):
        fail(PIN)
    for relative, (sha256, size) in observed.items():
        if not same_identity(bound[relative], sha256, size):
            fail(PIN)


def read_pinned_plan(namespace_root: Path) -> bytes:
    """The frozen plan, checked against ``PLAN_SHA256`` before it is parsed."""

    root_fd = open_directory(namespace_root, PIN)
    try:
        raw = read_at(root_fd, "analysis-plan.json", PIN)
    finally:
        close_quietly(root_fd)
    if len(raw) != PLAN_BYTES or not hmac.compare_digest(digest_of(raw), PLAN_SHA256):
        fail(PIN)
    return raw


def audit_plan_pins(plan: Mapping[str, Any]) -> None:
    """The plan this verifier was written against is the plan on disk."""

    if plan["namespace"] != NAMESPACE or plan["schema_version"] != SCHEMA_VERSION:
        fail(PIN)
    domains = plan["domains"]
    if domains["partition_utf8"] != PARTITION_DOMAIN[:-1].decode("ascii"):
        fail(PIN)
    if domains["snapshot_set_utf8"] != SNAPSHOT_SET_DOMAIN[:-1].decode("ascii"):
        fail(PIN)
    if tuple(domains["retired_domain_prefixes"]) != RETIRED_DOMAIN_PREFIXES:
        fail(PIN)
    if domains["all_active_domains_fresh"] is not True:
        fail(PIN)
    if domains["retired_domains_reusable"] is not False:
        fail(PIN)

    partition = plan["partition"]
    if partition["modulus"] != SPLIT_MODULUS:
        fail(PIN)
    if partition["holdout_buckets"] != {"gte": 0, "lt": HOLDOUT_UPPER_EXCLUSIVE}:
        fail(PIN)
    if partition["shadow_buckets"] != {"gte": HOLDOUT_UPPER_EXCLUSIVE, "lt": SPLIT_MODULUS}:
        fail(PIN)
    for key in (
        "event_intersection_required",
        "automatic_family_cross_partition_count_required",
        "organic_session_cross_partition_count_required",
        "workflow_cross_partition_count_required",
    ):
        if partition[key] != 0:
            fail(PIN)
    if partition["event_union_equals_selected_population"] is not True:
        fail(PIN)
    for key in ("input_order_invariant", "outcome_field_invariant", "source_qualified_keys"):
        if partition[key] is not True:
            fail(PIN)
    for key in (
        "key_search_allowed",
        "rebalancing_allowed",
        "truncation_allowed",
        "stratification_allowed",
        "reassignment_allowed",
    ):
        if partition[key] is not False:
            fail(PIN)

    replay = plan["replayability"]
    if tuple(replay["persisted_source_fields_required"]) != PERSISTED_SOURCE_FIELDS_REQUIRED:
        fail(PIN)
    if len(PERSISTED_SOURCE_FIELDS_REQUIRED) != 12:
        fail(PIN)
    ambient = replay["replay_input_schema"]["ambient_context"]
    if tuple(ambient["behaviorally_relevant_keys_exactly"]) != AMBIENT_KEYS:
        fail(PIN)
    if len(AMBIENT_KEYS) != 12:
        fail(PIN)
    if not str(ambient["unknown_key_action"]).startswith("unconditionally nonreplayable"):
        fail(PIN)
    if tuple(replay["forbidden_inputs"]) != FORBIDDEN_REPLAY_INPUTS:
        fail(PIN)
    if len(FORBIDDEN_REPLAY_INPUTS) != 9:
        fail(PIN)
    if replay["selected_nonreplayable_floor_contribution"] != 0:
        fail(PIN)
    if replay["role"] != "input-only floor and measurement eligibility; never selection":
        fail(PIN)

    selection = plan["selection"]["predicate"]
    if selection["scope_filter"] is not None or selection["replayability_filter"] is not None:
        fail(PIN)
    if selection["outcome_filtering"] is not False:
        fail(PIN)
    if selection["created_at"]["gt_release_effective_at"] != RELEASE_EFFECTIVE_AT:
        fail(PIN)

    privacy = plan["privacy"]["deidentification"]
    if privacy["persist_salt_or_map"] is not False:
        fail(PIN)
    if tuple(privacy["preserve"]) != (
        "exact Python character length",
        "space, tab, newline, and carriage-return positions",
        "bijective equality classes within one packet build",
    ):
        fail(PIN)
    if privacy["replace_other_characters_with"] != "lowercase a-z":
        fail(PIN)
    identity = plan["identity"]
    for key in (
        "persist_key",
        "persist_key_commitment",
        "persist_hmac_tokens_or_samples",
        "persist_plaintext_or_normalized_identity",
        "persist_unkeyed_fingerprint",
    ):
        if identity[key] is not False:
            fail(PIN)
    if identity["normalization"] != '" ".join(query.split()) + "\\n" + requested_scope':
        fail(PIN)
    if plan["privacy"]["consumed_corpus_opened"] is not False:
        fail(PIN)
    if len(plan["hash_binding"]["packet_manifest_must_bind"]) != len(MANIFEST_BINDING_KEYS):
        fail(PIN)

    validation = plan["operational_receipt_schemas"]["common_validation"]
    if validation["encoding"] != "domains.v4_canonical_json_utf8":
        fail(PIN)
    for key in ("unknown_or_missing_field_action", "duplicate_json_key_action"):
        if validation[key] != "invalid":
            fail(PIN)
    if validation["free_text_fields_allowed"] is not False:
        fail(PIN)
    if validation["sha256_type"] != "exactly 64 lowercase hexadecimal characters":
        fail(PIN)
    if validation["constants"] != {"schema_version": SCHEMA_VERSION, "namespace": NAMESPACE}:
        fail(PIN)
    contract = plan["selection"]["utc_timestamp_contract"]
    if contract["canonical_receipt_encoding"] != "YYYY-MM-DDTHH:MM:SS.ffffffZ":
        fail(PIN)


# --------------------------------------------------------------------------
# Corpus-level audit, shared by both phases
# --------------------------------------------------------------------------


def audit_corpus(
    members: Mapping[str, bytes], seed_paths: Mapping[str, Path], plan: Mapping[str, Any]
) -> tuple[dict[str, int], dict[str, Any]]:
    """Everything provable from the nine corpus members alone."""

    if set(members) != set(CORPUS_MEMBER_NAMES):
        # A tenth member would be a persisted salt, map or fingerprint index.
        fail(PRIVACY)

    records: list[dict[str, Any]] = []
    for partition, name in CORPUS_NAME_BY_PARTITION.items():
        records.extend(parse_corpus(members[name], partition))
    audit_records(records)

    audit_partition(records, plan["partition"]["golden_vectors"])
    audit_identity_equivalence(records)
    counts = recompute_aggregates(records)

    seed_manifest = load_canonical(members[SEED_MANIFEST_NAME], expected=dict)
    states = audit_seed_manifest(seed_manifest)
    audit_policy_key_grammar(plan)
    for name in SEED_STATE_NAMES:
        audit_seed_state(seed_paths[name], states[name]["table_row_counts"])

    # No secret-shaped material anywhere in the sealed case records.
    audit_no_digest_material(members[name] for name in CORPUS_NAME_BY_PARTITION.values())
    audit_reader_authority([seed_manifest, *records])
    return counts, seed_manifest


def audit_packet_documents(
    blobs: Mapping[str, bytes],
    marks: Mapping[str, tuple],
    seed_paths: Mapping[str, Path],
    namespace_root: Path,
    running_verifier: bytes,
) -> None:
    """The complete sealed audit, from the packet bytes and this file's pins."""

    # Tree shape first: exactly one manifest, exactly the nine corpus members,
    # and nothing outside the frozen namespace allowlist.
    top = {name.split("/", 1)[0] for name in blobs}
    top |= {name.rstrip("/").split("/", 1)[0] for name in marks if name.endswith("/")}
    if not PACKET_REQUIRED_ENTRIES.issubset(top) or top - NAMESPACE_ALLOWED_ENTRIES:
        fail(BOUNDARY)
    corpus_members = {
        name.split("/", 1)[1]: raw
        for name, raw in blobs.items()
        if name.startswith(PACKET_CORPUS_DIRECTORY + "/")
    }
    if set(corpus_members) - set(CORPUS_MEMBER_NAMES):
        # An undeclared corpus member is a persisted salt, map or fingerprint
        # index, which ``privacy.deidentification`` forbids outright.
        fail(PRIVACY)
    if set(corpus_members) != set(CORPUS_MEMBER_NAMES):
        fail(BOUNDARY)
    recipe_members = {
        name.split("/", 1)[1] for name in blobs if name.startswith("recipe/")
    }
    if not RECIPE_REQUIRED_ENTRIES.issubset(recipe_members):
        fail(BOUNDARY)

    # The executing verifier is the external trust anchor.  Bind the packet's
    # tracked copy byte for byte, so a swapped in-packet verifier is caught.
    if not same_bytes(blobs["recipe/verify.py"], running_verifier):
        fail(PIN)

    manifest_raw = blobs[PACKET_MANIFEST_NAME]
    manifest = load_canonical(manifest_raw, expected=dict)
    audit_manifest_schema(manifest)

    # Pins first: the plan is only parsed from bytes this module has already
    # matched against ``PLAN_SHA256``.
    documents = audit_source_pins(namespace_root, manifest)
    plan = load_json(documents["analysis-plan.json"], expected=dict, code=PIN)
    audit_plan_pins(plan)
    audit_campaign_tools(namespace_root, manifest)
    if not same_identity(
        manifest["bindings"]["packet_builder_and_independent_verifier"]["independent_verifier"],
        digest_of(running_verifier),
        len(running_verifier),
    ):
        fail(PIN)

    # Every content binding must match the installed bytes.  A rewritten file
    # fails here; a rewritten file with a rehashed manifest fails the pins.
    bound = {
        HOLDOUT_CORPUS_NAME: manifest["bindings"]["complete_selected_holdout_partition"],
        SHADOW_CORPUS_NAME: manifest["bindings"]["complete_selected_shadow_partition"],
        PRESEAL_RECEIPT_NAME: manifest["bindings"]["keyed_aggregate_preseal_receipt"],
    }
    seed_group = manifest["bindings"]["partition_source_seed_states_and_construction_manifest"]
    for name in (*SEED_STATE_NAMES, SEED_MANIFEST_NAME):
        bound[name] = seed_group[name]
    for name, identity in bound.items():
        raw = corpus_members[name]
        if not same_identity(identity, digest_of(raw), len(raw)):
            fail(SCHEMA)

    receipt = load_canonical(corpus_members[PRESEAL_RECEIPT_NAME], expected=dict)
    audit_preseal_receipt(receipt, manifest)
    audit_runtime_segment(manifest, receipt)
    audit_manifest_order(manifest, receipt, plan)
    audit_reader_authority([manifest, receipt])

    counts, _seed_manifest = audit_corpus(corpus_members, seed_paths, plan)
    declared = dict(manifest["aggregate_counts"])
    # ``event_union_equals_selected_population``: the three population counts
    # are a partition invariant, the eleven floor counts a replayability one.
    for field in POPULATION_FIELDS:
        if counts[field] != declared[field]:
            fail(PARTITION)
    if counts != declared:
        fail(REPLAYABILITY)
    if floors_pass(counts) is not bool(manifest["floors_pass"]):
        fail(SCHEMA)

    # No manifest digest may be echoed into the sealed corpus or seed state.
    digests: set[bytes] = set()
    for value in strings_in(manifest["bindings"]):
        if HEX64_RE.match(value) is not None:
            digests.add(bytes.fromhex(value))
    privacy_scan(
        [
            raw
            for name, raw in corpus_members.items()
            if name not in (SEED_MANIFEST_NAME, PRESEAL_RECEIPT_NAME)
        ],
        encoded_needles(digests),
    )

    audit_publication_order(marks)


def audit_publication_order(marks: Mapping[str, tuple]) -> None:
    """``content_before_manifest`` and ``manifest_last``, from the inodes.

    The manifest is the sealed-state marker and the last byte written, so no
    installed content may be newer than it.  A member rewritten after the seal
    bumps its change stamp above the manifest's even if its modification stamp
    is forged backwards.
    """

    manifest_mark = marks.get(PACKET_MANIFEST_NAME)
    if manifest_mark is None:
        fail(BOUNDARY)
    _dev, _ino, mode, nlink, _size, manifest_mtime, manifest_ctime = manifest_mark
    if nlink != 1 or stat.S_IMODE(mode) & 0o222:
        fail(OVERWRITE)

    for name, mark in marks.items():
        if name.endswith("/"):
            if not name.startswith(PACKET_CORPUS_DIRECTORY + "/"):
                continue
            if mark[3] > manifest_mtime or mark[4] > manifest_ctime:
                fail(OVERWRITE)
            continue
        if not name.startswith(PACKET_CORPUS_DIRECTORY + "/"):
            continue
        _cdev, _cino, cmode, cnlink, _csize, cmtime, cctime = mark
        if cnlink != 1 or stat.S_IMODE(cmode) & 0o222:
            fail(OVERWRITE)
        if cmtime > manifest_mtime or cctime > manifest_ctime:
            fail(OVERWRITE)
    corpus_mark = marks.get(PACKET_CORPUS_DIRECTORY + "/")
    if corpus_mark is None or corpus_mark[3] > manifest_mtime or corpus_mark[4] > manifest_ctime:
        fail(OVERWRITE)


# --------------------------------------------------------------------------
# ``sealed``
# --------------------------------------------------------------------------


def verify_sealed(packet_dir: Path) -> None:
    packet = Path(os.path.normpath(os.fspath(packet_dir)))
    try:
        before = os.lstat(packet)
    except OSError:
        return fail(BOUNDARY)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        fail(BOUNDARY)

    running_verifier = Path(__file__).resolve().read_bytes()
    namespace_root = Path(__file__).resolve().parents[1]

    packet_fd = open_directory(packet)
    try:
        if directory_identity(packet_fd) != (before.st_dev, before.st_ino):
            fail(CANDIDATE_CHANGED)
        sweep_for_symlinks(packet_fd)
        blobs, marks = read_tree(packet_fd)
        seed_paths = {
            name: packet / PACKET_CORPUS_DIRECTORY / name for name in SEED_STATE_NAMES
        }
        audit_packet_documents(blobs, marks, seed_paths, namespace_root, running_verifier)

        # Re-read the whole packet after auditing: a member mutated while the
        # audit ran must not be able to pass through it.
        final_blobs, final_marks = read_tree(packet_fd, CANDIDATE_CHANGED)
        if set(blobs) != set(final_blobs) or set(marks) != set(final_marks):
            fail(CANDIDATE_CHANGED)
        for name, raw in blobs.items():
            if not same_bytes(raw, final_blobs[name]):
                fail(CANDIDATE_CHANGED)
        for name, mark in marks.items():
            if mark != final_marks[name]:
                fail(CANDIDATE_CHANGED)
        if directory_identity(packet_fd, CANDIDATE_CHANGED) != (before.st_dev, before.st_ino):
            fail(CANDIDATE_CHANGED)
    finally:
        close_quietly(packet_fd)

    # Re-``lstat`` the packet directory itself, to catch a swap of the whole
    # tree behind the descriptor we held.
    try:
        after = os.lstat(packet)
    except OSError:
        return fail(CANDIDATE_CHANGED)
    if stat.S_ISLNK(after.st_mode) or (after.st_dev, after.st_ino) != (
        before.st_dev,
        before.st_ino,
    ):
        fail(CANDIDATE_CHANGED)


# --------------------------------------------------------------------------
# ``keyed``
# --------------------------------------------------------------------------

DRAFT_CONTENT_DIRECTORY = "content"
KEYED_RECEIPT_KIND = "packet-independent-keyed-preseal"


STANDARD_STREAM_FDS = frozenset({0, 1, 2})


def assert_private_channel(fd: int) -> int:
    """A standard stream is never a private inherited channel.

    Without this, ``--receipt-fd 1`` would turn the keyed phase's own receipt
    into console output whenever stdout happened to be a pipe.
    """

    if not is_int(fd) or fd in STANDARD_STREAM_FDS or fd < 0:
        return fail(ARGUMENTS)
    return fd


def assert_anonymous_pipe(fd: int, *, readable: bool) -> None:
    """``privacy.key_transport``: inherited anonymous descriptors only."""

    assert_private_channel(fd)
    try:
        info = os.fstat(fd)
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        target = os.readlink(f"/proc/self/fd/{fd}")
    except (OSError, ValueError):
        return fail(PRIVACY)
    mode = flags & os.O_ACCMODE
    if (
        not stat.S_ISFIFO(info.st_mode)
        or not target.startswith("pipe:[")
        or (readable and mode not in (os.O_RDONLY, os.O_RDWR))
        or (not readable and mode not in (os.O_WRONLY, os.O_RDWR))
    ):
        fail(PRIVACY)


def read_identity_key(fd: int) -> bytearray:
    assert_anonymous_pipe(fd, readable=True)
    chunks: list[bytes] = []
    total = 0
    while total <= IDENTITY_KEY_BYTES:
        try:
            chunk = os.read(fd, IDENTITY_KEY_BYTES + 1 - total)
        except OSError:
            return fail(PRIVACY)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    close_quietly(fd)
    raw = b"".join(chunks)
    if len(raw) != IDENTITY_KEY_BYTES:
        fail(PRIVACY)
    return bytearray(raw)


def write_receipt(fd: int, raw: bytes) -> None:
    assert_anonymous_pipe(fd, readable=False)
    offset = 0
    while offset < len(raw):
        try:
            written = os.write(fd, raw[offset:])
        except OSError:
            return fail(PRIVACY)
        if written <= 0:
            fail(PRIVACY)
        offset += written


def verify_keyed(draft_dir: Path, identity_key_fd: int, receipt_fd: int) -> dict[str, Any]:
    """Audit the draft content while the build's identity key is still live.

    The key is drained from its inherited pipe, used to prove that no encoding
    of it reached a single content byte, and zeroed.  Nothing here opens a raw
    source, a consumed corpus, or a private reference.
    """

    key = read_identity_key(identity_key_fd)
    try:
        draft = Path(os.path.normpath(os.fspath(draft_dir)))
        draft_fd = open_directory(draft)
        content_fd = -1
        try:
            content_fd = open_child_directory(draft_fd, DRAFT_CONTENT_DIRECTORY)
            sweep_for_symlinks(content_fd)
            members: dict[str, bytes] = {}
            for name in sorted(listdir_at(content_fd)):
                members[name] = read_at(content_fd, name)
        finally:
            close_quietly(content_fd)
            close_quietly(draft_fd)

        namespace_root = Path(__file__).resolve().parents[1]
        plan = load_json(read_pinned_plan(namespace_root), expected=dict, code=PIN)
        audit_plan_pins(plan)

        seed_paths = {
            name: draft / DRAFT_CONTENT_DIRECTORY / name for name in SEED_STATE_NAMES
        }
        counts, _seed_manifest = audit_corpus(members, seed_paths, plan)

        receipt = load_canonical(members[PRESEAL_RECEIPT_NAME], expected=dict)
        exact_keys(receipt, PRESEAL_RECEIPT_KEYS)
        if any(receipt[field] != counts[field] for field in AGGREGATE_FIELDS):
            fail(SCHEMA)
        audit_semantic_reads(receipt["semantic_reads"])

        # The keyed phase exists to prove exactly this: no encoding of the live
        # identity key reached any sealed byte.
        privacy_scan(members.values(), encoded_needles([bytes(key)]))
        return {
            "mode": "v4-keyed",
            "schema_version": SCHEMA_VERSION,
            "namespace": NAMESPACE,
            "receipt_kind": KEYED_RECEIPT_KIND,
            "status": "pass",
            "mismatch_count": 0,
            "floors_pass": floors_pass(counts),
            **{field: counts[field] for field in AGGREGATE_FIELDS},
        }
    finally:
        for index in range(len(key)):
            key[index] = 0
        del key


def keyed_failure_receipt(code: str) -> dict[str, Any]:
    return {
        "mode": "v4-keyed",
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "receipt_kind": KEYED_RECEIPT_KIND,
        "status": "fail",
        "check": code,
        "mismatch_count": 1,
    }


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


class _QuietParser(argparse.ArgumentParser):
    """argparse usage text is a leak channel; raise the closed error instead."""

    def error(self, _message: str) -> None:
        raise VerificationError(ARGUMENTS)

    def exit(self, status: int = 0, _message: str | None = None) -> None:
        raise VerificationError(ARGUMENTS)


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "--mechanical-only":
        # Accepted only as an exact-match sole argument.
        if arguments != ["--mechanical-only"]:
            fail(ARGUMENTS)
        arguments = ["sealed", "--packet-dir", os.fspath(Path(__file__).resolve().parents[1])]
    parser = _QuietParser(add_help=False, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="mode", required=True, parser_class=_QuietParser)
    sealed = subparsers.add_parser("sealed", add_help=False, allow_abbrev=False)
    sealed.add_argument("--packet-dir", type=Path, required=True)
    keyed = subparsers.add_parser("keyed", add_help=False, allow_abbrev=False)
    keyed.add_argument("--draft-dir", type=Path, required=True)
    keyed.add_argument("--identity-key-fd", type=int, required=True)
    keyed.add_argument("--receipt-fd", type=int, required=True)
    return parser.parse_args(arguments)


def _print_sealed(status: str, code: str | None = None) -> None:
    line = {"mode": "v4-sealed", "status": status}
    if code is not None:
        line["check"] = code if code in CHECK_CODES else INTERNAL
    sys.stdout.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    _disable_core_dumps()
    mode = "unknown"
    receipt_fd = -1
    try:
        arguments = parse_arguments(argv)
        mode = arguments.mode
        if mode == "sealed":
            verify_sealed(arguments.packet_dir)
            _print_sealed("pass")
            return 0
        if mode == "keyed":
            # Both channels must be private and distinct before anything is
            # read or written; a bad channel yields no output at all.
            assert_private_channel(arguments.identity_key_fd)
            receipt_fd = assert_private_channel(arguments.receipt_fd)
            if receipt_fd == arguments.identity_key_fd:
                return fail(ARGUMENTS)
            receipt = verify_keyed(
                arguments.draft_dir, arguments.identity_key_fd, receipt_fd
            )
            write_receipt(receipt_fd, canonical_bytes(receipt) + b"\n")
            return 0
        return fail(ARGUMENTS)
    except VerificationError as error:
        if mode == "sealed":
            _print_sealed("fail", error.code)
        elif mode == "keyed" and receipt_fd >= 0:
            try:
                write_receipt(receipt_fd, canonical_bytes(keyed_failure_receipt(error.code)) + b"\n")
            except BaseException:
                pass
        return 1
    except BaseException:
        # No exception text ever escapes; every unexpected failure collapses.
        if mode == "sealed":
            _print_sealed("fail", INTERNAL)
        elif mode == "keyed" and receipt_fd >= 0:
            try:
                write_receipt(receipt_fd, canonical_bytes(keyed_failure_receipt(INTERNAL)) + b"\n")
            except BaseException:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
