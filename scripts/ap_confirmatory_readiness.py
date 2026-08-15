#!/usr/bin/env python3
"""Aggregate-only readiness scanner for ``confirmatory-holdout-v2``.

The public ``scan`` command accepts private inputs only as already-open,
read-only file descriptors plus the trusted launcher's UTC watermark.  The
source; the scanner rejects future watermarks or capture starts outside the
fixed 60-second launch window before touching the descriptors.  It hashes
inputs before and after use and delegates all
raw-row and HMAC work to a fresh ``python -I`` child.  The child returns only
aggregate counts and a SHA-256 of the ephemeral candidate plan; it never writes
a corpus or imports an evaluator.

The public ``validate`` command mechanically validates a tracked readiness
receipt against the frozen analysis plan.  A valid ``insufficient`` receipt is
a successful, terminal one-shot decision.

Attempt consumption belongs to the trusted readiness launcher: this scanner's
pure core is intentionally repeatable for synthetic tests.  A fatal integrity
failure cannot truthfully populate the frozen 20-field receipt (for example,
there may be no stable source hash), so ``scan`` emits only the allowlisted
terminal fragment ``{"status":"insufficient"}``, returns nonzero, and does not
write ``--receipt``.  That fragment is a fatal signal, not a validator-valid
receipt; the launcher must record the consumed attempt and must not retry.

Candidate-plan canonicalization
-------------------------------
The private plan is an RFC-8785-compatible JSON array (all keys are ASCII and
all values are strings, booleans, or the integers 0/1) in canonical event
order.  Each record has exactly ``CANDIDATE_PLAN_FIELDS``.  Only the SHA-256 of
those bytes crosses the keyed worker boundary.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import math
import os
import secrets
import signal
import sqlite3
import stat
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLAN = (
    REPO_ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "confirmatory-holdout-v2"
    / "analysis-plan.json"
)
DEFAULT_POLICY = DEFAULT_PLAN.with_name("POLICY.md")

PINNED_ANALYSIS_PLAN_SHA256 = (
    "5f5f050330b97c8a35b0e76bd31cbf1b9dd73f4d1a2baa0175817c8201a0c653"
)
PINNED_POLICY_SHA256 = (
    "096263ddf554dc014c8cd971e6129d47300bd1c0b9710da90a5749ff186a1afb"
)
NAMESPACE = "confirmatory-holdout-v2"
SOURCE_ALIASES = ("local", "alt")
FIXED_LOWER_BOUND = "2026-08-13T20:16:51Z"
PARTITION_DOMAIN = b"confirmatory-holdout-v2/partition/v1\0"
ARM_ORDER_DOMAIN = b"confirmatory-holdout-v2/arm-order/v1\0"
DEV_SPLIT_LOWER = 15
DEV_SPLIT_UPPER = 66
SPLIT_MODULUS = 100
KEY_BYTES = 32
WORKER_TIMEOUT_SECONDS = 600
MAX_PLAN_BYTES = 256_000
MAX_RECEIPT_BYTES = 32_768
MAX_WORKER_RESULT_BYTES = 16_384
READ_CHUNK_BYTES = 1 << 20

RECEIPT_FIELDS = (
    "schema_version",
    "namespace",
    "status",
    "fixed_lower_bound",
    "capture_watermark",
    "aliased_source_snapshot_sha256_and_bytes",
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
    "selected_event_count",
    "candidate_plan_sha256",
    "analysis_plan_sha256",
)

COUNT_FIELDS = RECEIPT_FIELDS[6:18]
WORKER_FIELDS = (*COUNT_FIELDS, "candidate_plan_sha256")
SOURCE_META_FIELDS = ("sha256", "bytes")
HEX64 = frozenset("0123456789abcdef")

EVENT_COLUMNS = (
    "id",
    "query",
    "requested_scope",
    "agent",
    "task",
    "session_id",
    "transport_session_id",
    "created_at",
)
SEED_STATE_COLUMNS = {
    "nodes": frozenset({"id", "level", "content", "scope", "created_at"}),
    "connections": frozenset({"source_id", "target_id", "type", "weight"}),
    "retrieval_weights": frozenset({"scope", "bm25", "vector", "graph"}),
}

CANDIDATE_PLAN_FIELDS = (
    "source_qualified_event_key",
    "created_at",
    "connected_component_representative",
    "partition",
    "structural_validity",
    "family_floor_eligibility",
    "workflow_floor_eligibility",
    "replayability",
    "paired_arm_dispatch_order",
)


class IntegrityFailure(Exception):
    """A deliberately message-free, non-public integrity failure."""


def _fail() -> None:
    raise IntegrityFailure


def _strict_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail()
        result[key] = value
    return result


def _load_json(raw: bytes, *, expected: type | None = None) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda _value: _fail(),
        )
    except IntegrityFailure:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        _fail()
    if expected is not None and type(value) is not expected:
        _fail()
    return value


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_nonnegative_int(value: Any) -> bool:
    return _is_int(value) and value >= 0


def _is_hex64(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in HEX64 for character in value)
    )


def _utf8(value: Any, *, nonempty: bool = False) -> bytes:
    if type(value) is not str or (nonempty and not value):
        _fail()
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError:
        _fail()


def _parse_utc(value: Any) -> datetime:
    raw = _utf8(value, nonempty=True)
    del raw
    text_value = value
    try:
        parsed = datetime.fromisoformat(
            text_value[:-1] + "+00:00" if text_value.endswith("Z") else text_value
        )
    except ValueError:
        _fail()
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        _fail()
    return parsed.astimezone(UTC)


def _utc_now() -> tuple[str, datetime]:
    value = datetime.now(UTC)
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z"), value


def _canonical_json_bytes(value: Any) -> bytes:
    """Return RFC-8785-compatible bytes for this script's restricted values."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError):
        _fail()


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            receipt,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_fd(fd: int, *, maximum: int | None = None) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    total = 0
    while True:
        try:
            chunk = os.pread(fd, READ_CHUNK_BYTES, offset)
        except OSError:
            _fail()
        if not chunk:
            break
        total += len(chunk)
        if maximum is not None and total > maximum:
            _fail()
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _hash_fd(fd: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    offset = 0
    size = 0
    while True:
        try:
            chunk = os.pread(fd, READ_CHUNK_BYTES, offset)
        except OSError:
            _fail()
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@dataclass(frozen=True, slots=True)
class FileProof:
    identity: tuple[int, int, int, int, int]
    sha256: str
    bytes: int


def _proof_fd(fd: int) -> FileProof:
    try:
        before = os.fstat(fd)
    except OSError:
        _fail()
    if not stat.S_ISREG(before.st_mode):
        _fail()
    digest, size = _hash_fd(fd)
    try:
        after = os.fstat(fd)
    except OSError:
        _fail()
    if _stat_identity(before) != _stat_identity(after) or size != after.st_size:
        _fail()
    return FileProof(_stat_identity(after), digest, size)


def _require_unchanged(fd: int, proof: FileProof) -> None:
    if _proof_fd(fd) != proof:
        _fail()


def _duplicate_readonly_regular_fd(fd: int) -> int:
    if not _is_int(fd) or fd < 0:
        _fail()
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        duplicate = os.dup(fd)
    except OSError:
        _fail()
    try:
        if flags & os.O_ACCMODE != os.O_RDONLY:
            _fail()
        info = os.fstat(duplicate)
        if not stat.S_ISREG(info.st_mode):
            _fail()
        os.set_inheritable(duplicate, False)
        return duplicate
    except BaseException:
        os.close(duplicate)
        raise


def _read_regular_path(path: Path, *, maximum: int) -> bytes:
    try:
        path_info = os.lstat(path)
    except OSError:
        _fail()
    if stat.S_ISLNK(path_info.st_mode) or not stat.S_ISREG(path_info.st_mode):
        _fail()
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError:
        _fail()
    try:
        before = os.fstat(fd)
        raw = _read_fd(fd, maximum=maximum)
        after = os.fstat(fd)
        if _stat_identity(before) != _stat_identity(after) or len(raw) != after.st_size:
            _fail()
        return raw
    finally:
        os.close(fd)


def _validate_plan(
    raw: bytes,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    if not _is_hex64(expected_sha256):
        _fail()
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256):
        _fail()
    plan = _load_json(raw, expected=dict)
    try:
        readiness = plan["readiness"]
        identity = plan["identity"]
        partition = plan["partition"]
        sources = plan["sources"]
        selection = plan["selection"]
        privacy = plan["privacy"]
    except (KeyError, TypeError):
        _fail()

    conditions = (
        plan.get("schema_version") == 2,
        plan.get("namespace") == NAMESPACE,
        plan.get("frozen") is True,
        tuple(selection.get("source_aliases_exactly", ())) == SOURCE_ALIASES,
        selection.get("predicate", {}).get("created_at", {}).get("gt")
        == FIXED_LOWER_BOUND,
        selection.get("all_eligible_events_included") is True,
        selection.get("floors_are_not_inclusion_filters") is True,
        selection.get("predicate", {}).get("outcome_filtering") is False,
        identity.get("algorithm") == "HMAC-SHA256",
        identity.get("normalization")
        == '" ".join(query.split()) + "\\n" + requested_scope',
        identity.get("persist_key") is False,
        identity.get("persist_hmac_tokens_or_samples") is False,
        partition.get("domain_separator_utf8")
        == "confirmatory-holdout-v2/partition/v1",
        partition.get("modulus") == SPLIT_MODULUS,
        partition.get("holdout_buckets") == {"gte": 0, "lt": 50},
        partition.get("shadow_buckets") == {"gte": 50, "lt": 100},
        partition.get("event_intersection_required") == 0,
        readiness.get("receipt_status_allowed") == ["ready", "insufficient"],
        readiness.get("decision_operator") == "all",
        readiness.get("individual_or_per_source_counts_allowed") is False,
        tuple(readiness.get("readiness_receipt_allowlist", ())) == RECEIPT_FIELDS,
        tuple(readiness.get("candidate_plan", {}).get("record_fields", ()))
        == (
            "source-qualified event key",
            "created_at",
            "connected-component representative",
            "partition",
            "structural validity",
            "family-floor eligibility",
            "workflow-floor eligibility",
            "replayability",
            "paired-arm dispatch order",
        ),
        readiness.get("candidate_plan", {}).get("outcome_fields_present") is False,
        privacy.get("readiness_output_is_allowlist_only") is True,
        privacy.get("final_reports_aggregate_only") is True,
        set(sources.get("aliases", {})) == set(SOURCE_ALIASES),
        sources.get("distinct_authorities_and_database_instances_required") is True,
        sources.get("maximum_snapshot_capture_start_delay_seconds") == 60,
    )
    if not all(conditions):
        _fail()

    try:
        family = identity["repeated_automatic_family"]
        dev = identity["complete_frozen_dev_reference"]["raw_identity_reference"]
        holdout_floor = readiness["holdout_floor"]
        shadow_floor = readiness["shadow_floor"]
    except (KeyError, TypeError):
        _fail()
    if (
        family.get("minimum_events") != 3
        or family.get("minimum_distinct_nonempty_source_qualified_transport_sessions")
        != 2
        or not _is_hex64(dev.get("sha256"))
        or not _is_nonnegative_int(dev.get("bytes"))
        or dev.get("split_membership")
        != "source alt and 15 <= int.from_bytes(SHA256(event_id UTF-8)[0:8], 'big') % 100 < 66"
    ):
        _fail()
    if set(holdout_floor) != {
        "unseen_in_dev_repeated_automatic_families_min",
        "unseen_in_dev_repeated_automatic_events_min",
        "distinct_components_containing_qualifying_unseen_automatic_family_min",
        "organic_events_min",
        "distinct_nonempty_source_qualified_organic_sessions_min",
        "distinct_components_containing_organic_event_min",
        "distinct_requested_project_scopes_min",
    }:
        _fail()
    if set(shadow_floor) != {
        "genuine_nonsynthetic_real_workflows_min",
        "unique_replayable_logical_calls_min",
        "distinct_components_containing_counted_workflow_min",
        "distinct_requested_project_scopes_min",
    }:
        _fail()
    if not all(_is_int(value) and value > 0 for value in holdout_floor.values()):
        _fail()
    if not all(_is_int(value) and value > 0 for value in shadow_floor.values()):
        _fail()
    return plan


def _load_plan_path(path: Path, *, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_path(path, maximum=MAX_PLAN_BYTES)
    return _validate_plan(raw, expected_sha256=expected_sha256), raw


def _validate_frozen_policy() -> None:
    raw = _read_regular_path(DEFAULT_POLICY, maximum=256_000)
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), PINNED_POLICY_SHA256):
        _fail()


def _split_bucket(event_id: str) -> int:
    digest = hashlib.sha256(_utf8(event_id, nonempty=True)).digest()
    return int.from_bytes(digest[:8], "big") % SPLIT_MODULUS


def _identity_message(query: str, requested_scope: str) -> bytes:
    normalized = " ".join(query.split())
    return _utf8(normalized + "\n" + requested_scope)


def _identity_token(
    key: bytes | bytearray,
    query: str,
    requested_scope: str,
    collision_map: dict[bytes, bytes],
) -> bytes:
    message = _identity_message(query, requested_scope)
    token = hmac.new(key, message, hashlib.sha256).digest()
    previous = collision_map.get(token)
    if previous is not None and not hmac.compare_digest(previous, message):
        _fail()
    collision_map[token] = message
    return token


def _dev_tokens(
    fd: int,
    *,
    plan: Mapping[str, Any],
    key: bytes | bytearray,
    collision_map: dict[bytes, bytes],
) -> set[bytes]:
    dev_contract = plan["identity"]["complete_frozen_dev_reference"][
        "raw_identity_reference"
    ]
    raw = _read_fd(fd, maximum=dev_contract["bytes"])
    if (
        len(raw) != dev_contract["bytes"]
        or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), dev_contract["sha256"])
    ):
        _fail()
    if raw and not raw.endswith(b"\n"):
        _fail()
    tokens: set[bytes] = set()
    identifiers: set[str] = set()
    for line in raw.splitlines():
        if not line:
            _fail()
        row = _load_json(line, expected=dict)
        if not {"event_id", "query", "requested_scope", "agent"}.issubset(row):
            # The pinned export historically names the identifier ``id``.
            if not {"id", "query", "requested_scope", "agent"}.issubset(row):
                _fail()
        event_id = row.get("event_id", row.get("id"))
        _utf8(event_id, nonempty=True)
        if event_id in identifiers:
            _fail()
        identifiers.add(event_id)
        query = row["query"]
        requested_scope = row["requested_scope"]
        if type(query) is not str or type(requested_scope) is not str:
            _fail()
        _utf8(query)
        _utf8(requested_scope)
        if (
            row["agent"] is None
            and bool(query)
            and _scope_kind_valid(requested_scope)
            and DEV_SPLIT_LOWER <= _split_bucket(event_id) < DEV_SPLIT_UPPER
        ):
            tokens.add(
                _identity_token(
                    key,
                    query,
                    requested_scope,
                    collision_map,
                )
            )
    return tokens


@dataclass(slots=True)
class Event:
    alias: str
    event_id: str
    created_at: str
    created_instant: datetime
    query: Any
    requested_scope: Any
    agent: Any
    task: Any
    session_id: Any
    transport_session_id: Any
    token: bytes | None
    replayable: bool

    @property
    def key_bytes(self) -> bytes:
        return _utf8(self.alias) + b"\0" + _utf8(self.event_id, nonempty=True)

    @property
    def key_text(self) -> str:
        return self.alias + "\0" + self.event_id

    @property
    def session_key(self) -> tuple[str, str] | None:
        if type(self.transport_session_id) is str and self.transport_session_id:
            return self.alias, self.transport_session_id
        return None


def _table_columns(connection: sqlite3.Connection, table: str) -> frozenset[str]:
    try:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        _fail()
    return frozenset(row[1] for row in rows if type(row[1]) is str)


def _validate_snapshot_schema(connection: sqlite3.Connection) -> None:
    if not set(EVENT_COLUMNS).issubset(_table_columns(connection, "recall_events")):
        _fail()
    for table, required in SEED_STATE_COLUMNS.items():
        if not required.issubset(_table_columns(connection, table)):
            _fail()


def _nullable_string(value: Any) -> bool:
    return value is None or type(value) is str


def _scope_kind_valid(value: Any) -> bool:
    return type(value) is str and (
        value == "global"
        or (value.startswith("project:") and len(value) > len("project:"))
        or (value.startswith("session:") and len(value) > len("session:"))
    )


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _validate_seed_state(connection: sqlite3.Connection) -> None:
    """Prove the frozen column allowlist can form a self-contained seed state."""

    try:
        node_ids: set[str] = set()
        for row in connection.execute(
            "SELECT id, level, content, scope, created_at FROM nodes"
        ):
            node_id = row["id"]
            _utf8(node_id, nonempty=True)
            if (
                node_id in node_ids
                or row["level"] not in ("trace", "concept", "schema")
                or type(row["content"]) is not str
                or not _scope_kind_valid(row["scope"])
            ):
                _fail()
            _utf8(row["content"])
            _parse_utc(row["created_at"])
            node_ids.add(node_id)

        for row in connection.execute(
            "SELECT source_id, target_id, type, weight FROM connections"
        ):
            if (
                row["source_id"] not in node_ids
                or row["target_id"] not in node_ids
                or row["type"]
                not in ("related", "caused", "contradicts", "supersedes", "requires")
                or not _finite_number(row["weight"])
            ):
                _fail()

        weight_scopes: set[str] = set()
        for row in connection.execute(
            "SELECT scope, bm25, vector, graph FROM retrieval_weights"
        ):
            if (
                not _scope_kind_valid(row["scope"])
                or row["scope"] in weight_scopes
                or not all(
                    _finite_number(row[name]) for name in ("bm25", "vector", "graph")
                )
            ):
                _fail()
            weight_scopes.add(row["scope"])
    except sqlite3.Error:
        _fail()


def _open_snapshot(fd: int) -> sqlite3.Connection:
    uri = f"file:/proc/self/fd/{fd}?mode=ro&immutable=1&cache=private"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
            _fail()
        if connection.execute("PRAGMA temp_store").fetchone()[0] != 2:
            _fail()
        check = connection.execute("PRAGMA quick_check").fetchall()
        if len(check) != 1 or check[0][0] != "ok":
            _fail()
        _validate_snapshot_schema(connection)
        _validate_seed_state(connection)
        return connection
    except IntegrityFailure:
        try:
            connection.close()
        except (UnboundLocalError, sqlite3.Error):
            pass
        raise
    except sqlite3.Error:
        try:
            connection.close()
        except (UnboundLocalError, sqlite3.Error):
            pass
        _fail()


def _snapshot_events(
    fd: int,
    *,
    alias: str,
    lower: datetime,
    watermark: datetime,
    key: bytes | bytearray,
    collision_map: dict[bytes, bytes],
) -> list[Event]:
    if alias not in SOURCE_ALIASES:
        _fail()
    connection = _open_snapshot(fd)
    events: list[Event] = []
    identifiers: set[str] = set()
    try:
        try:
            rows = connection.execute(
                "SELECT id, query, requested_scope, agent, task, session_id, "
                "transport_session_id, created_at FROM recall_events"
            )
            for row in rows:
                event_id = row["id"]
                _utf8(event_id, nonempty=True)
                if event_id in identifiers:
                    _fail()
                identifiers.add(event_id)
                instant = _parse_utc(row["created_at"])
                if not (lower < instant <= watermark):
                    continue
                query = row["query"]
                requested_scope = row["requested_scope"]
                agent = row["agent"]
                task = row["task"]
                session_id = row["session_id"]
                transport_session_id = row["transport_session_id"]
                token: bytes | None = None
                if (
                    agent is None
                    and type(query) is str
                    and bool(query)
                    and _scope_kind_valid(requested_scope)
                ):
                    try:
                        token = _identity_token(
                            key,
                            query,
                            requested_scope,
                            collision_map,
                        )
                    except UnicodeEncodeError:
                        token = None
                replayable = (
                    type(query) is str
                    and bool(query)
                    and _scope_kind_valid(requested_scope)
                    and _nullable_string(agent)
                    and _nullable_string(task)
                    and _nullable_string(session_id)
                    and _nullable_string(transport_session_id)
                )
                events.append(
                    Event(
                        alias=alias,
                        event_id=event_id,
                        created_at=row["created_at"],
                        created_instant=instant,
                        query=query,
                        requested_scope=requested_scope,
                        agent=agent,
                        task=task,
                        session_id=session_id,
                        transport_session_id=transport_session_id,
                        token=token,
                        replayable=replayable,
                    )
                )
        except sqlite3.Error:
            _fail()
    finally:
        connection.close()
    return events


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _join_group(disjoint: _DisjointSet, indices: Sequence[int]) -> None:
    if not indices:
        return
    first = indices[0]
    for other in indices[1:]:
        disjoint.union(first, other)


def _partition_for_representative(representative: bytes) -> str:
    digest = hashlib.sha256(PARTITION_DOMAIN + representative).digest()
    bucket = int.from_bytes(digest[:4], "big") % SPLIT_MODULUS
    return "holdout" if bucket < 50 else "shadow"


def _assert_one_partition(indices: Iterable[int], partitions: Sequence[str]) -> None:
    if len({partitions[index] for index in indices}) > 1:
        _fail()


def _candidate_plan_digest(
    events: Sequence[Event],
    *,
    representatives: Sequence[bytes],
    partitions: Sequence[str],
    qualifying_event_indices: set[int],
    counted_workflows: set[tuple[str, str]],
) -> str:
    order = sorted(
        range(len(events)),
        key=lambda index: (
            events[index].created_instant,
            _utf8(events[index].alias),
            _utf8(events[index].event_id),
        ),
    )
    records: list[dict[str, Any]] = []
    try:
        for index in order:
            event = events[index]
            arm_bit = hashlib.sha256(ARM_ORDER_DOMAIN + event.key_bytes).digest()[0] & 1
            record = {
                "source_qualified_event_key": event.key_text,
                "created_at": event.created_at,
                "connected_component_representative": representatives[index].decode(
                    "utf-8"
                ),
                "partition": partitions[index],
                "structural_validity": True,
                "family_floor_eligibility": index in qualifying_event_indices,
                "workflow_floor_eligibility": event.session_key in counted_workflows,
                "replayability": event.replayable,
                "paired_arm_dispatch_order": arm_bit,
            }
            if tuple(record) != CANDIDATE_PLAN_FIELDS:
                _fail()
            records.append(record)
        return hashlib.sha256(_canonical_json_bytes(records)).hexdigest()
    finally:
        records.clear()


def _floor_status(counts: Mapping[str, int], plan: Mapping[str, Any]) -> str:
    holdout = plan["readiness"]["holdout_floor"]
    shadow = plan["readiness"]["shadow_floor"]
    checks = (
        counts["holdout_unseen_automatic_family_count"]
        >= holdout["unseen_in_dev_repeated_automatic_families_min"],
        counts["holdout_unseen_automatic_event_count"]
        >= holdout["unseen_in_dev_repeated_automatic_events_min"],
        counts["holdout_unseen_automatic_component_count"]
        >= holdout[
            "distinct_components_containing_qualifying_unseen_automatic_family_min"
        ],
        counts["holdout_organic_event_count"] >= holdout["organic_events_min"],
        counts["holdout_organic_session_count"]
        >= holdout["distinct_nonempty_source_qualified_organic_sessions_min"],
        counts["holdout_organic_component_count"]
        >= holdout["distinct_components_containing_organic_event_min"],
        counts["holdout_project_scope_count"]
        >= holdout["distinct_requested_project_scopes_min"],
        counts["shadow_real_workflow_count"]
        >= shadow["genuine_nonsynthetic_real_workflows_min"],
        counts["shadow_replayable_logical_call_count"]
        >= shadow["unique_replayable_logical_calls_min"],
        counts["shadow_real_workflow_component_count"]
        >= shadow["distinct_components_containing_counted_workflow_min"],
        counts["shadow_project_scope_count"]
        >= shadow["distinct_requested_project_scopes_min"],
    )
    return "ready" if all(checks) else "insufficient"


def _analyze(
    *,
    plan: Mapping[str, Any],
    local_fd: int,
    alt_fd: int,
    dev_fd: int,
    watermark_text: str,
) -> dict[str, Any]:
    watermark = _parse_utc(watermark_text)
    lower = _parse_utc(FIXED_LOWER_BOUND)
    if watermark <= lower:
        _fail()

    key = bytearray(secrets.token_bytes(KEY_BYTES))
    collision_map: dict[bytes, bytes] = {}
    dev_tokens: set[bytes] = set()
    events: list[Event] = []
    try:
        dev_tokens = _dev_tokens(
            dev_fd,
            plan=plan,
            key=key,
            collision_map=collision_map,
        )
        for alias, fd in (("local", local_fd), ("alt", alt_fd)):
            events.extend(
                _snapshot_events(
                    fd,
                    alias=alias,
                    lower=lower,
                    watermark=watermark,
                    key=key,
                    collision_map=collision_map,
                )
            )

        event_keys = [event.key_bytes for event in events]
        if len(event_keys) != len(set(event_keys)):
            _fail()

        family_groups: dict[tuple[str, bytes], list[int]] = defaultdict(list)
        session_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, event in enumerate(events):
            if event.token is not None:
                family_groups[(event.alias, event.token)].append(index)
            if event.session_key is not None:
                session_groups[event.session_key].append(index)

        disjoint = _DisjointSet(len(events))
        for indices in family_groups.values():
            _join_group(disjoint, indices)
        for indices in session_groups.values():
            _join_group(disjoint, indices)

        components: dict[int, list[int]] = defaultdict(list)
        for index in range(len(events)):
            components[disjoint.find(index)].append(index)

        component_rep: dict[int, bytes] = {}
        component_partition: dict[int, str] = {}
        partition_digests: dict[bytes, bytes] = {}
        for root, indices in components.items():
            representative = min(events[index].key_bytes for index in indices)
            digest = hashlib.sha256(PARTITION_DOMAIN + representative).digest()
            previous = partition_digests.get(digest)
            if previous is not None and previous != representative:
                _fail()
            partition_digests[digest] = representative
            component_rep[root] = representative
            component_partition[root] = _partition_for_representative(representative)

        representatives = [component_rep[disjoint.find(index)] for index in range(len(events))]
        partitions = [component_partition[disjoint.find(index)] for index in range(len(events))]
        holdout_indices = {index for index, value in enumerate(partitions) if value == "holdout"}
        shadow_indices = {index for index, value in enumerate(partitions) if value == "shadow"}
        if holdout_indices & shadow_indices or holdout_indices | shadow_indices != set(
            range(len(events))
        ):
            _fail()
        for indices in family_groups.values():
            _assert_one_partition(indices, partitions)
        for indices in session_groups.values():
            _assert_one_partition(indices, partitions)

        qualifying_families: dict[tuple[str, bytes], list[int]] = {}
        qualifying_event_indices: set[int] = set()
        for family_key, indices in family_groups.items():
            nonempty_sessions = {
                events[index].session_key
                for index in indices
                if events[index].session_key is not None
            }
            if (
                len(indices) >= 3
                and len(nonempty_sessions) >= 2
                and family_key[1] not in dev_tokens
            ):
                qualifying_families[family_key] = indices
                qualifying_event_indices.update(indices)

        counted_workflows: set[tuple[str, str]] = set()
        for workflow, indices in session_groups.items():
            if any(events[index].agent is not None for index in indices) and any(
                events[index].replayable for index in indices
            ):
                counted_workflows.add(workflow)
                _assert_one_partition(indices, partitions)

        holdout_families = {
            family
            for family, indices in qualifying_families.items()
            if partitions[indices[0]] == "holdout"
        }
        holdout_automatic_indices = {
            index
            for family in holdout_families
            for index in qualifying_families[family]
        }
        holdout_organic_indices = {
            index
            for index in holdout_indices
            if events[index].agent is not None
        }
        holdout_organic_sessions = {
            events[index].session_key
            for index in holdout_organic_indices
            if events[index].session_key is not None
        }
        shadow_workflows = {
            workflow
            for workflow in counted_workflows
            if partitions[session_groups[workflow][0]] == "shadow"
        }
        shadow_replayable_indices = {
            index
            for workflow in shadow_workflows
            for index in session_groups[workflow]
            if events[index].replayable
        }

        counts = {
            "holdout_unseen_automatic_family_count": len(holdout_families),
            "holdout_unseen_automatic_event_count": len(holdout_automatic_indices),
            "holdout_unseen_automatic_component_count": len(
                {disjoint.find(index) for index in holdout_automatic_indices}
            ),
            "holdout_organic_event_count": len(holdout_organic_indices),
            "holdout_organic_session_count": len(holdout_organic_sessions),
            "holdout_organic_component_count": len(
                {disjoint.find(index) for index in holdout_organic_indices}
            ),
            "holdout_project_scope_count": len(
                {
                    events[index].requested_scope
                    for index in holdout_indices
                    if type(events[index].requested_scope) is str
                    and events[index].requested_scope.startswith("project:")
                }
            ),
            "shadow_real_workflow_count": len(shadow_workflows),
            "shadow_replayable_logical_call_count": len(shadow_replayable_indices),
            "shadow_real_workflow_component_count": len(
                {
                    disjoint.find(index)
                    for workflow in shadow_workflows
                    for index in session_groups[workflow]
                }
            ),
            "shadow_project_scope_count": len(
                {
                    events[index].requested_scope
                    for index in shadow_indices
                    if type(events[index].requested_scope) is str
                    and events[index].requested_scope.startswith("project:")
                }
            ),
            "selected_event_count": len(events),
        }
        if tuple(counts) != COUNT_FIELDS:
            _fail()
        result: dict[str, Any] = dict(counts)
        result["candidate_plan_sha256"] = _candidate_plan_digest(
            events,
            representatives=representatives,
            partitions=partitions,
            qualifying_event_indices=qualifying_event_indices,
            counted_workflows=counted_workflows,
        )
        if tuple(result) != WORKER_FIELDS:
            _fail()
        return result
    finally:
        for index in range(len(key)):
            key[index] = 0
        collision_map.clear()
        dev_tokens.clear()
        events.clear()


def _write_all(fd: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        try:
            written = os.write(fd, raw[offset:])
        except OSError:
            _fail()
        if written <= 0:
            _fail()
        offset += written


def _worker_entry(arguments: Sequence[str]) -> int:
    # Fixed-position parsing avoids argparse diagnostics in the keyed process.
    if len(arguments) != 12:
        return 73
    expected_names = (
        "--plan-fd",
        "--local-fd",
        "--alt-fd",
        "--dev-fd",
        "--result-fd",
        "--watermark",
    )
    if tuple(arguments[0::2]) != expected_names:
        return 73
    try:
        plan_fd, local_fd, alt_fd, dev_fd, result_fd = map(int, arguments[1:10:2])
        watermark = arguments[11]
        plan_raw = _read_fd(plan_fd, maximum=MAX_PLAN_BYTES)
        expected_plan_hash = hashlib.sha256(plan_raw).hexdigest()
        plan = _validate_plan(plan_raw, expected_sha256=expected_plan_hash)
        result = _analyze(
            plan=plan,
            local_fd=local_fd,
            alt_fd=alt_fd,
            dev_fd=dev_fd,
            watermark_text=watermark,
        )
        raw = _canonical_json_bytes(result)
        if len(raw) > MAX_WORKER_RESULT_BYTES:
            _fail()
        _write_all(result_fd, raw)
        return 0
    except BaseException:
        return 73
    finally:
        for offset in (1, 3, 5, 7, 9):
            try:
                os.close(int(arguments[offset]))
            except (OSError, ValueError, IndexError):
                pass


def _read_worker_result(fd: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            _fail()
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_WORKER_RESULT_BYTES:
            _fail()
        chunks.append(chunk)
    return b"".join(chunks)


def _terminate_worker_group(process: subprocess.Popen[bytes]) -> None:
    """Destroy the keyed process group and prove no descriptor holder remains."""

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        _fail()
    if process.poll() is None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _fail()
    deadline = time.monotonic() + 10
    while True:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        except OSError:
            _fail()
        if time.monotonic() >= deadline:
            _fail()
        time.sleep(0.01)


def _validate_worker_result(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(WORKER_FIELDS):
        _fail()
    for field in COUNT_FIELDS:
        if not _is_nonnegative_int(value[field]):
            _fail()
    if not _is_hex64(value["candidate_plan_sha256"]):
        _fail()
    return value


def _run_keyed_worker(
    *,
    plan_fd: int,
    local_fd: int,
    alt_fd: int,
    dev_fd: int,
    watermark: str,
) -> dict[str, Any]:
    result_read, result_write = os.pipe()
    command = [
        sys.executable,
        "-B",
        "-I",
        os.fspath(Path(__file__).resolve()),
        "__worker",
        "--plan-fd",
        str(plan_fd),
        "--local-fd",
        str(local_fd),
        "--alt-fd",
        str(alt_fd),
        "--dev-fd",
        str(dev_fd),
        "--result-fd",
        str(result_write),
        "--watermark",
        watermark,
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(plan_fd, local_fd, alt_fd, dev_fd, result_write),
            start_new_session=True,
        )
    except (OSError, ValueError):
        os.close(result_read)
        os.close(result_write)
        _fail()
    os.close(result_write)
    try:
        try:
            return_code = process.wait(timeout=WORKER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _terminate_worker_group(process)
            _fail()
        _terminate_worker_group(process)
        raw = _read_worker_result(result_read)
        if return_code != 0:
            _fail()
        return _validate_worker_result(_load_json(raw, expected=dict))
    finally:
        try:
            _terminate_worker_group(process)
            if process.poll() is None:
                process.wait(timeout=10)
        finally:
            os.close(result_read)


def _source_metadata(proofs: Mapping[str, FileProof]) -> dict[str, dict[str, Any]]:
    if set(proofs) != set(SOURCE_ALIASES):
        _fail()
    return {
        alias: {"sha256": proofs[alias].sha256, "bytes": proofs[alias].bytes}
        for alias in SOURCE_ALIASES
    }


def _validate_capture_envelope(
    *,
    watermark: datetime,
    capture_starts: Mapping[str, str] | None,
    observed_launch: datetime,
    maximum_delay_seconds: int,
) -> None:
    if watermark > observed_launch:
        _fail()
    if capture_starts is None or set(capture_starts) != set(SOURCE_ALIASES):
        _fail()
    latest_start = watermark + timedelta(seconds=maximum_delay_seconds)
    for alias in SOURCE_ALIASES:
        capture_start = _parse_utc(capture_starts[alias])
        if not (watermark <= capture_start <= latest_start):
            _fail()
        if capture_start > observed_launch:
            _fail()


def _assemble_receipt(
    *,
    plan: Mapping[str, Any],
    analysis_plan_sha256: str,
    watermark: str,
    source_proofs: Mapping[str, FileProof],
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    counts = {field: aggregate[field] for field in COUNT_FIELDS}
    receipt = {
        "schema_version": plan["schema_version"],
        "namespace": NAMESPACE,
        "status": _floor_status(counts, plan),
        "fixed_lower_bound": FIXED_LOWER_BOUND,
        "capture_watermark": watermark,
        "aliased_source_snapshot_sha256_and_bytes": _source_metadata(source_proofs),
        **counts,
        "candidate_plan_sha256": aggregate["candidate_plan_sha256"],
        "analysis_plan_sha256": analysis_plan_sha256,
    }
    _validate_receipt(receipt, plan=plan, plan_sha256=analysis_plan_sha256)
    return receipt


def _scan_fds(
    *,
    plan_path: Path,
    local_input_fd: int,
    alt_input_fd: int,
    dev_input_fd: int,
    watermark: str,
    capture_starts: Mapping[str, str] | None = None,
    launch_instant: datetime | None = None,
    expected_plan_sha256: str = PINNED_ANALYSIS_PLAN_SHA256,
    validate_policy: bool = True,
    enforce_capture_age: bool = True,
) -> dict[str, Any]:
    if validate_policy:
        _validate_frozen_policy()
    plan, plan_raw = _load_plan_path(plan_path, expected_sha256=expected_plan_sha256)
    watermark_instant = _parse_utc(watermark)
    observed_launch = datetime.now(UTC) if launch_instant is None else launch_instant
    maximum_delay = plan["sources"]["maximum_snapshot_capture_start_delay_seconds"]
    if enforce_capture_age:
        _validate_capture_envelope(
            watermark=watermark_instant,
            capture_starts=capture_starts,
            observed_launch=observed_launch,
            maximum_delay_seconds=maximum_delay,
        )
    elif watermark_instant > observed_launch:
        _fail()

    owned: list[int] = []
    try:
        local_fd = _duplicate_readonly_regular_fd(local_input_fd)
        owned.append(local_fd)
        alt_fd = _duplicate_readonly_regular_fd(alt_input_fd)
        owned.append(alt_fd)
        dev_fd = _duplicate_readonly_regular_fd(dev_input_fd)
        owned.append(dev_fd)
        for input_fd in {local_input_fd, alt_input_fd, dev_input_fd}:
            try:
                os.close(input_fd)
            except OSError:
                pass

        try:
            plan_read = os.memfd_create("confirmatory-analysis-plan", os.MFD_CLOEXEC)
        except (AttributeError, OSError):
            _fail()
        owned.append(plan_read)
        _write_all(plan_read, plan_raw)

        proofs = {
            "local": _proof_fd(local_fd),
            "alt": _proof_fd(alt_fd),
        }
        if proofs["local"].identity[:2] == proofs["alt"].identity[:2]:
            _fail()
        if hmac.compare_digest(proofs["local"].sha256, proofs["alt"].sha256):
            _fail()
        dev_proof = _proof_fd(dev_fd)
        dev_contract = plan["identity"]["complete_frozen_dev_reference"][
            "raw_identity_reference"
        ]
        if (
            dev_proof.bytes != dev_contract["bytes"]
            or not hmac.compare_digest(dev_proof.sha256, dev_contract["sha256"])
        ):
            _fail()

        aggregate = _run_keyed_worker(
            plan_fd=plan_read,
            local_fd=local_fd,
            alt_fd=alt_fd,
            dev_fd=dev_fd,
            watermark=watermark,
        )
        _require_unchanged(local_fd, proofs["local"])
        _require_unchanged(alt_fd, proofs["alt"])
        _require_unchanged(dev_fd, dev_proof)
        return _assemble_receipt(
            plan=plan,
            analysis_plan_sha256=hashlib.sha256(plan_raw).hexdigest(),
            watermark=watermark,
            source_proofs=proofs,
            aggregate=aggregate,
        )
    finally:
        for fd in reversed(owned):
            try:
                os.close(fd)
            except OSError:
                pass


def _validate_receipt(
    receipt: Any,
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
) -> dict[str, Any]:
    if type(receipt) is not dict or set(receipt) != set(RECEIPT_FIELDS):
        _fail()
    if (
        receipt["schema_version"] != plan["schema_version"]
        or receipt["namespace"] != NAMESPACE
        or receipt["status"] not in ("ready", "insufficient")
        or receipt["fixed_lower_bound"] != FIXED_LOWER_BOUND
        or receipt["analysis_plan_sha256"] != plan_sha256
        or not _is_hex64(receipt["analysis_plan_sha256"])
        or not _is_hex64(receipt["candidate_plan_sha256"])
    ):
        _fail()
    watermark = _parse_utc(receipt["capture_watermark"])
    if watermark <= _parse_utc(FIXED_LOWER_BOUND) or watermark > datetime.now(UTC):
        _fail()

    sources = receipt["aliased_source_snapshot_sha256_and_bytes"]
    if type(sources) is not dict or set(sources) != set(SOURCE_ALIASES):
        _fail()
    source_hashes: list[str] = []
    for alias in SOURCE_ALIASES:
        value = sources[alias]
        if type(value) is not dict or set(value) != set(SOURCE_META_FIELDS):
            _fail()
        if not _is_hex64(value["sha256"]):
            _fail()
        if not _is_int(value["bytes"]) or value["bytes"] <= 0:
            _fail()
        source_hashes.append(value["sha256"])
    if len(set(source_hashes)) != len(source_hashes):
        _fail()

    counts: dict[str, int] = {}
    for field in COUNT_FIELDS:
        value = receipt[field]
        if not _is_nonnegative_int(value):
            _fail()
        counts[field] = value
    if receipt["status"] != _floor_status(counts, plan):
        _fail()

    selected = counts["selected_event_count"]
    families = counts["holdout_unseen_automatic_family_count"]
    automatic_events = counts["holdout_unseen_automatic_event_count"]
    automatic_components = counts["holdout_unseen_automatic_component_count"]
    organic_events = counts["holdout_organic_event_count"]
    organic_sessions = counts["holdout_organic_session_count"]
    organic_components = counts["holdout_organic_component_count"]
    workflows = counts["shadow_real_workflow_count"]
    calls = counts["shadow_replayable_logical_call_count"]
    workflow_components = counts["shadow_real_workflow_component_count"]
    if (
        automatic_events < 3 * families
        or automatic_components > families
        or (families == 0) != (automatic_events == 0)
        or (families == 0) != (automatic_components == 0)
        or organic_sessions > organic_events
        or organic_components > organic_events
        or (organic_events == 0) != (organic_components == 0)
        or workflows > calls
        or workflow_components > workflows
        or (workflows == 0) != (calls == 0)
        or (workflows == 0) != (workflow_components == 0)
        or automatic_events + organic_events + calls > selected
        or counts["holdout_project_scope_count"] > selected
        or counts["shadow_project_scope_count"] > selected
    ):
        _fail()
    return dict(receipt)


def _validate_receipt_paths(plan_path: Path, receipt_path: Path) -> None:
    plan, plan_raw = _load_plan_path(
        plan_path,
        expected_sha256=PINNED_ANALYSIS_PLAN_SHA256,
    )
    _validate_frozen_policy()
    raw = _read_regular_path(receipt_path, maximum=MAX_RECEIPT_BYTES)
    receipt = _load_json(raw, expected=dict)
    _validate_receipt(
        receipt,
        plan=plan,
        plan_sha256=hashlib.sha256(plan_raw).hexdigest(),
    )


def _write_receipt(path: str, receipt: Mapping[str, Any]) -> None:
    raw = _receipt_bytes(receipt)
    if len(raw) > MAX_RECEIPT_BYTES:
        _fail()
    if path == "-":
        try:
            sys.stdout.buffer.write(raw)
            sys.stdout.buffer.flush()
        except OSError:
            _fail()
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError:
        _fail()
    try:
        _write_all(fd, raw)
        os.fsync(fd)
    except BaseException:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    finally:
        os.close(fd)


class _QuietParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        _fail()

    def exit(self, status: int = 0, message: str | None = None) -> None:
        del status, message
        _fail()


def _parser() -> argparse.ArgumentParser:
    parser = _QuietParser(add_help=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", add_help=False)
    scan.add_argument("--plan", required=True)
    scan.add_argument("--local-snapshot-fd", required=True, type=int)
    scan.add_argument("--alt-snapshot-fd", required=True, type=int)
    scan.add_argument("--dev-reference-fd", required=True, type=int)
    scan.add_argument("--capture-watermark", required=True)
    scan.add_argument("--local-capture-start", required=True)
    scan.add_argument("--alt-capture-start", required=True)
    scan.add_argument("--receipt", required=True)

    validate = subparsers.add_parser("validate", add_help=False)
    validate.add_argument("--plan", required=True)
    validate.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # This happens before argument parsing or any source descriptor access.
    _, launch_instant = _utc_now()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "__worker":
        return _worker_entry(arguments[1:])
    command = arguments[0] if arguments else None
    try:
        parsed = _parser().parse_args(arguments)
        if parsed.command == "validate":
            _validate_receipt_paths(Path(parsed.plan), Path(parsed.receipt))
            return 0
        receipt = _scan_fds(
            plan_path=Path(parsed.plan),
            local_input_fd=parsed.local_snapshot_fd,
            alt_input_fd=parsed.alt_snapshot_fd,
            dev_input_fd=parsed.dev_reference_fd,
            watermark=parsed.capture_watermark,
            capture_starts={
                "local": parsed.local_capture_start,
                "alt": parsed.alt_capture_start,
            },
            launch_instant=launch_instant,
        )
        _write_receipt(parsed.receipt, receipt)
        return 0
    except BaseException:
        # No exception text, path, private value, or diagnostic crosses the boundary.
        if command == "scan":
            try:
                sys.stdout.buffer.write(b'{"status":"insufficient"}\n')
                sys.stdout.buffer.flush()
            except BaseException:
                pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
