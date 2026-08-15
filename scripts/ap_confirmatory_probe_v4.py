#!/usr/bin/env python3
"""Aggregate-only mechanics for ``confirmatory-holdout-v4``.

The module deliberately has no dependency on :mod:`living_memory`.  A trusted
launcher supplies already-open, read-only snapshots and runtime objects that
were produced by :mod:`ap_confirmatory_runtime_v4`.  Raw rows, the fresh HMAC
key, equality classes, and component membership exist only in an isolated
``python -I`` worker.  The worker can return only the frozen aggregate counts.

An all-floor result is *provisional*: this leaf never serializes it as
``ready``.  ``ready`` becomes validator-valid only after the downstream atomic
seal-consumption/blocked-spawn handoff.  A below-floor result can be converted
to the exact public slot-resolution allowlist with
``build_below_floor_resolution``.

Every integrity error is the shared, message-free ``IntegrityFailure``.  The
internal worker writes nothing on failure, and direct command-line misuse is
silent.  No corpus, candidate plan, semantic read, equality map, or case
artifact is created here.

Two boundaries are specific to v4.  The runtime models the unbound watermark
predecessor as its own type, so ordinary probe work must refuse anything that
is not a bound :class:`~ap_confirmatory_runtime_v4.ActiveSegment`: before the
initial source binding exists the only authorized slot-0 resolution is the
runtime's count-free ``initial-runtime-change-closure``, never a probe
resolution.  And every keyed, partition, and receipt preimage carries the
``confirmatory-holdout-v4/`` prefix, so a retired v2/v3 artifact can never
replay into this namespace even when its structure is otherwise identical.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import math
import os
import secrets
import signal
import sqlite3
import socket
import stat
import subprocess
import sys
import time
from array import array
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


# Loading by the canonical module name matters: the shared runtime intentionally
# checks exact dataclass type identity.  importlib-based tests should construct
# runtime wrappers through ``probe.runtime`` rather than loading a second copy.
try:
    import ap_confirmatory_runtime_v4 as runtime
except ModuleNotFoundError:  # pragma: no cover - exercised by importlib callers
    import importlib.util

    _RUNTIME_PATH = Path(__file__).resolve().with_name(
        "ap_confirmatory_runtime_v4.py"
    )
    _RUNTIME_SPEC = importlib.util.spec_from_file_location(
        "ap_confirmatory_runtime_v4", _RUNTIME_PATH
    )
    if _RUNTIME_SPEC is None or _RUNTIME_SPEC.loader is None:
        raise
    runtime = importlib.util.module_from_spec(_RUNTIME_SPEC)
    sys.modules[_RUNTIME_SPEC.name] = runtime
    _RUNTIME_SPEC.loader.exec_module(runtime)


IntegrityFailure = runtime.IntegrityFailure

NAMESPACE = runtime.NAMESPACE
SCHEMA_VERSION = runtime.SCHEMA_VERSION
SOURCE_ALIASES = runtime.SOURCE_ALIASES
RELEASE_EFFECTIVE_AT = runtime.RELEASE_EFFECTIVE_AT

PRODUCTION_SEED_CONTRACT_ID = (
    "confirmatory-holdout-v4-production-seed-state-v1"
)
REQUIRED_PRODUCTION_POLICY_KEYS = frozenset(
    {"default", "project", "global", "session"}
)

# The frozen plan spells the retrieval-policy-key grammar out as data so the
# implementation and the protocol cannot drift apart silently.  The bare keys
# are exactly the four production seeds shipped by the serving configuration;
# they are *not* memory scopes, and the node-scope predicate rejects three of
# the four.
POLICY_KEY_MAPPING_EXACT = {
    "default": "default",
    "project": "project",
    "global": "global",
    "session": "session",
    "project:<nonempty UTF-8 suffix>": "project",
    "session:<nonempty UTF-8 suffix>": "session",
}

IDENTITY_DOMAIN = b"confirmatory-holdout-v4/identity/v1\0"
PARTITION_DOMAIN = b"confirmatory-holdout-v4/partition/v1\0"
DEV_SPLIT_LOWER = 15
DEV_SPLIT_UPPER = 66
SPLIT_MODULUS = 100
KEY_BYTES = 32

WORKER_TIMEOUT_SECONDS = 600
WORKER_HANDSHAKE_TIMEOUT_SECONDS = 30
MAX_PLAN_BYTES = 256_000
MAX_WORKER_RESULT_BYTES = 16_384
READ_CHUNK_BYTES = 1 << 20

WORKER_BOOTSTRAP = (
    "import importlib.util,importlib.machinery,sys;"
    "r=sys.argv.pop(1);p=sys.argv.pop(1);"
    "l=importlib.machinery.SourceFileLoader('ap_confirmatory_runtime_v4',r);"
    "s=importlib.util.spec_from_loader(l.name,l);"
    "m=importlib.util.module_from_spec(s);"
    "sys.modules[s.name]=m;s.loader.exec_module(m);"
    "l=importlib.machinery.SourceFileLoader('__main__',p);"
    "s=importlib.util.spec_from_loader(l.name,l);"
    "m=importlib.util.module_from_spec(s);"
    "sys.modules['__main__']=m;s.loader.exec_module(m)"
)

EVENT_COLUMNS = (
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

SEED_STATE_COLUMNS = {
    "nodes": frozenset({"id", "level", "content", "scope", "created_at"}),
    "connections": frozenset({"source_id", "target_id", "type", "weight"}),
    "retrieval_weights": frozenset({"scope", "bm25", "vector", "graph"}),
}

AMBIENT_KEYS = frozenset(
    {
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

PROBE_RESOLUTION_FIELDS = (
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
    "release_effective_at",
    "active_segment_lower_bound_exclusive_at",
    "segment_id",
    "segment_attestation_sha256_and_bytes",
    "source_binding_attestation_sha256_and_bytes",
    "previous_ledger_entry_sha256",
    "pre_active_services_state_sha256",
    "post_active_services_state_sha256",
    "pre_database_instance_identity_sha256_by_alias",
    "post_database_instance_identity_sha256_by_alias",
    "pre_alias_service_database_binding_sha256_by_alias",
    "post_alias_service_database_binding_sha256_by_alias",
    "aliased_source_snapshot_sha256_and_bytes",
    *AGGREGATE_FIELDS,
    "runtime_observer_sha256_and_bytes",
    "probe_sha256_and_bytes",
    "analysis_plan_sha256_and_bytes",
    "seal_consumption_marker_sha256_and_bytes_or_null",
    "sealer_process_launched_at_or_null",
)

PROBE_ATTEMPT_FIELDS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "slot_index",
    "scheduled_at",
    "grace_deadline_at",
    "attempt_ordinal",
    "launched_at",
    "watchdog_deadline_at",
    "segment_id",
    "previous_ledger_entry_sha256",
    "runtime_observer_sha256_and_bytes",
    "probe_sha256_and_bytes",
    "analysis_plan_sha256_and_bytes",
)

PROBE_FAILURE_FIELDS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "slot_index",
    "attempt_ordinal",
    "probe_attempt_marker_sha256_and_bytes",
    "failed_at",
    "phase_at_failure",
    "failure_class",
    "source_open_count_or_null",
    "key_created_or_null",
    "retry_authorized",
    "controller_synthesized",
    "previous_ledger_entry_sha256",
    "analysis_plan_sha256_and_bytes",
)

PROBE_TERMINAL_FIELDS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "status",
    "slot_index",
    "attempt_ordinal",
    "probe_failure_marker_sha256_and_bytes",
    "recorded_at",
    "previous_ledger_entry_sha256",
    "analysis_plan_sha256_and_bytes",
)

PHASE_PROGRESS = {
    "pre-key-pre-source": (0, False),
    "key-created-pre-source": (0, True),
    "first-source-opened": (1, True),
    "snapshots-latched": (2, True),
    "keyed-count-computed": (2, True),
    "atomic-seal-handoff": (2, True),
}

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


def _fail() -> None:
    raise IntegrityFailure from None


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


def _hex64(value: Any) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail()
    return value


def _utf8(value: Any, *, nonempty: bool = False) -> bytes:
    if type(value) is not str or (nonempty and not value):
        _fail()
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError:
        _fail()


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _same_json_scalar(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _exact_mapping(value: Any, fields: Sequence[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        _fail()
    return value


def _load_metadata_json(value: Any, expected: type) -> Any | None:
    if type(value) is not str:
        return None
    try:
        raw = value.encode("utf-8")
        return runtime.load_json_bytes(raw, expected=expected)
    except (IntegrityFailure, UnicodeEncodeError):
        return None


def production_policy_key_kind(value: Any) -> str:
    """Map exactly one supported retrieval-policy key to its policy family.

    This is deliberately *not* :func:`_node_scope_valid`, and the two must
    never be merged.  ``retrieval_weights.scope`` stores a retrieval-policy
    key, not a memory scope: the serving defaults are the bare strings
    ``default``, ``project``, ``global``, and ``session``, of which the
    node-scope grammar accepts only ``global``.  Validating policy keys with
    the node predicate therefore rejects every real database and passes only
    an empty table — the exact defect this port repairs.  Normalizing them
    instead is worse, because the serving normalizer silently rewrites
    ``default`` to ``project:default``.

    Bare production keys map to themselves; only concrete nonempty
    ``project:*`` and ``session:*`` learned keys are admitted beyond them.
    """

    _utf8(value, nonempty=True)
    if value in REQUIRED_PRODUCTION_POLICY_KEYS:
        return value
    if value.startswith("project:") and value != "project:":
        return "project"
    if value.startswith("session:") and value != "session:":
        return "session"
    _fail()


def _require_policy_key_vectors(value: Any) -> None:
    """Drive the policy-key grammar over every vector the plan declares.

    The plan pins accepted and rejected spellings as data.  Replaying them
    here means a future edit that widened or narrowed the grammar — including
    one that reintroduced the node-scope predicate — fails plan validation
    before it can reach a source row.
    """

    if type(value) is not list or not value:
        _fail()
    for vector in value:
        if type(vector) is not dict or set(vector) != {"input", "accepted", "kind"}:
            _fail()
        accepted = vector["accepted"]
        if type(accepted) is not bool:
            _fail()
        if not accepted:
            if vector["kind"] is not None:
                _fail()
            try:
                production_policy_key_kind(vector["input"])
            except IntegrityFailure:
                continue
            _fail()
        if production_policy_key_kind(vector["input"]) != vector["kind"]:
            _fail()


def _require_partition_golden_vectors(value: Any) -> None:
    """Replay the plan's partition vectors through this module's own digest.

    Every vector carries the v4 domain separator, so a retired v2/v3
    partition implementation cannot satisfy them: all three digests change,
    and the published ``local\\0evt-a`` representative even crosses into the
    other partition.
    """

    if type(value) is not list or not value:
        _fail()
    for vector in value:
        if type(vector) is not dict or set(vector) != {
            "representative_display",
            "digest_sha256",
            "bucket",
            "partition",
        }:
            _fail()
        display = vector["representative_display"]
        bucket = vector["bucket"]
        if (
            type(display) is not str
            or type(bucket) is not int
            or type(bucket) is bool
            or not 0 <= bucket < SPLIT_MODULUS
            or display.count("\\0") != 1
        ):
            _fail()
        alias, _, event_id = display.partition("\\0")
        representative = _utf8(alias, nonempty=True) + b"\0" + _utf8(
            event_id, nonempty=True
        )
        digest = hashlib.sha256(PARTITION_DOMAIN + representative).digest()
        if (
            not hmac.compare_digest(digest.hex(), _hex64(vector["digest_sha256"]))
            or int.from_bytes(digest[:4], "big") % SPLIT_MODULUS != bucket
            or _partition_for_representative(representative) != vector["partition"]
        ):
            _fail()


def _node_scope_valid(value: Any) -> bool:
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


def _table_columns(connection: sqlite3.Connection, table: str) -> frozenset[str]:
    if table not in SEED_STATE_COLUMNS and table != "recall_events":
        _fail()
    try:
        objects = connection.execute(
            "SELECT type, sql FROM main.sqlite_schema WHERE name = ?", (table,)
        ).fetchall()
        if (
            len(objects) != 1
            or objects[0][0] != "table"
            or type(objects[0][1]) is not str
            or objects[0][1].lstrip().upper().startswith("CREATE VIRTUAL TABLE")
        ):
            _fail()
        rows = connection.execute(f"PRAGMA main.table_xinfo({table})").fetchall()
    except sqlite3.Error:
        _fail()
    columns: set[str] = set()
    for row in rows:
        if len(row) < 7 or type(row[1]) is not str:
            _fail()
        if row[1] in SEED_STATE_COLUMNS.get(table, frozenset(EVENT_COLUMNS)):
            if row[6] != 0 or type(row[6]) is not int:
                _fail()
        columns.add(row[1])
    return frozenset(columns)


def validate_production_seed_state(connection: sqlite3.Connection) -> None:
    """Validate the shared v3 production-shaped seed projection.

    Callers must pass the already-open source snapshot connection.  The
    function reads only the frozen ``nodes``, ``connections``, and
    ``retrieval_weights`` projections; result/history columns are irrelevant.
    """

    if type(connection) is not sqlite3.Connection:
        _fail()
    for table, required in SEED_STATE_COLUMNS.items():
        if not required.issubset(_table_columns(connection, table)):
            _fail()
    try:
        node_ids: set[str] = set()
        for row in connection.execute(
            "SELECT id, level, content, scope, created_at FROM main.nodes"
        ):
            node_id, level, content, scope, created_at = tuple(row)
            _utf8(node_id, nonempty=True)
            if (
                node_id in node_ids
                or level not in ("trace", "concept", "schema")
                or type(content) is not str
                or not _node_scope_valid(scope)
            ):
                _fail()
            _utf8(content)
            runtime.parse_utc(created_at)
            node_ids.add(node_id)

        connection_keys: set[tuple[str, str, str]] = set()
        for row in connection.execute(
            "SELECT source_id, target_id, type, weight FROM main.connections"
        ):
            source_id, target_id, relation_type, weight = tuple(row)
            key = (source_id, target_id, relation_type)
            if (
                source_id not in node_ids
                or target_id not in node_ids
                or relation_type
                not in ("related", "caused", "contradicts", "supersedes", "requires")
                or not _finite_number(weight)
                or key in connection_keys
            ):
                _fail()
            connection_keys.add(key)

        policy_keys: set[str] = set()
        row_count = 0
        for row in connection.execute(
            "SELECT scope, bm25, vector, graph FROM main.retrieval_weights"
        ):
            scope, bm25, vector, graph = tuple(row)
            production_policy_key_kind(scope)
            if (
                scope in policy_keys
                or not all(_finite_number(value) for value in (bm25, vector, graph))
            ):
                _fail()
            policy_keys.add(scope)
            row_count += 1
        # The emptiness clause is subsumed by the subset clause -- zero rows
        # can never contain four required keys -- and is kept only because the
        # plan states `retrieval_weights_table_must_be_nonempty` as its own
        # requirement.  An empty table is what let the retired scanner pass
        # while rejecting every real database, so the intent is spelled out
        # here even though it is not independently reachable.
        if row_count == 0 or not REQUIRED_PRODUCTION_POLICY_KEYS.issubset(policy_keys):
            _fail()
    except IntegrityFailure:
        raise
    except (sqlite3.Error, TypeError, ValueError, UnicodeError, OverflowError):
        _fail()


def _validate_snapshot_schema(connection: sqlite3.Connection) -> None:
    if not set(EVENT_COLUMNS).issubset(_table_columns(connection, "recall_events")):
        _fail()
    validate_production_seed_state(connection)


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

    def hash_and_bytes(self) -> runtime.HashAndBytes:
        return runtime.HashAndBytes(self.sha256, self.bytes)


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


def _stat_regular_fd(fd: int) -> tuple[int, int, int, int, int]:
    try:
        info = os.fstat(fd)
    except OSError:
        _fail()
    if not stat.S_ISREG(info.st_mode):
        _fail()
    return _stat_identity(info)


def _require_stat_unchanged(
    fd: int, identity: tuple[int, int, int, int, int]
) -> None:
    if _stat_regular_fd(fd) != identity:
        _fail()


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
        if not stat.S_ISREG(os.fstat(duplicate).st_mode):
            _fail()
        os.set_inheritable(duplicate, False)
        return duplicate
    except BaseException:
        os.close(duplicate)
        raise


def _read_regular_path(path: Path, *, maximum: int) -> bytes:
    try:
        info = os.lstat(path)
    except OSError:
        _fail()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
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


def _open_readonly_regular_path(path: Path) -> int:
    try:
        info = os.lstat(path)
    except OSError:
        _fail()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        _fail()
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(path, flags)
    except OSError:
        _fail()


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


def _validate_probe_plan(plan: Any) -> dict[str, Any]:
    """Validate every plan member consumed by the aggregate worker.

    Production additionally pins the complete file through
    ``runtime.load_frozen_contract``.  Keeping this projection validator
    separate permits outcome-free synthetic population tests with only the
    private dev-reference hash/size substituted.
    """

    if type(plan) is not dict:
        _fail()
    try:
        selection = plan["selection"]
        identity = plan["identity"]
        replay = plan["replayability"]
        partition = plan["partition"]
        readiness = plan["readiness"]
        schemas = plan["operational_receipt_schemas"]
        dev = identity["complete_frozen_dev_reference"]["raw_identity_reference"]
        family = identity["repeated_automatic_family"]
        seed = replay["shared_seed_state_validation"]
    except (KeyError, TypeError):
        _fail()
    predicate = selection.get("predicate") if type(selection) is dict else None
    replay_schema = (
        replay.get("replay_input_schema") if type(replay) is dict else None
    )
    ambient_schema = (
        replay_schema.get("ambient_context")
        if type(replay_schema) is dict
        else None
    )
    schema_members = (
        tuple(schemas.get(name) for name in (
            "slot_probe_resolution",
            "probe_attempt_marker",
            "probe_failure_marker",
            "probe_terminal_marker",
        ))
        if type(schemas) is dict
        else (None,) * 4
    )
    if not all(
        type(value) is dict
        for value in (
            selection,
            identity,
            replay,
            partition,
            readiness,
            schemas,
            dev,
            family,
            seed,
            predicate,
            replay_schema,
            ambient_schema,
            *schema_members,
        )
    ):
        _fail()
    if (
        plan.get("schema_version") != SCHEMA_VERSION
        or type(plan.get("schema_version")) is not int
        or plan.get("namespace") != NAMESPACE
        or plan.get("frozen") is not True
        or selection.get("source_aliases_exactly") != list(SOURCE_ALIASES)
        or type(predicate.get("created_at")) is not dict
        or predicate.get("created_at", {}).get(
            "gt_release_effective_at"
        )
        != RELEASE_EFFECTIVE_AT
        or predicate.get("created_at", {}).get(
            "gt_active_segment_lower_bound_exclusive_at"
        )
        is not True
        or predicate.get("created_at", {}).get(
            "lte_scheduled_slot_at"
        )
        is not True
        or predicate.get("outcome_filtering") is not False
        or selection.get("replayability_is_not_inclusion_filter") is not True
        or identity.get("algorithm") != "HMAC-SHA256"
        or identity.get("domain_separator_utf8") != IDENTITY_DOMAIN[:-1].decode()
        or identity.get("normalization")
        != '" ".join(query.split()) + "\\n" + requested_scope'
        or family.get("minimum_events") != 3
        or type(family.get("minimum_events")) is not int
        or family.get("minimum_distinct_nonempty_source_qualified_transport_sessions")
        != 2
        or type(
            family.get("minimum_distinct_nonempty_source_qualified_transport_sessions")
        )
        is not int
        or partition.get("domain_separator_utf8") != PARTITION_DOMAIN[:-1].decode()
        or partition.get("modulus") != SPLIT_MODULUS
        or type(partition.get("modulus")) is not int
        or partition.get("holdout_buckets") != {"gte": 0, "lt": 50}
        or partition.get("shadow_buckets") != {"gte": 50, "lt": 100}
        or readiness.get("probe_receipt_allowlist") != list(PROBE_RESOLUTION_FIELDS)
        or schemas.get("slot_probe_resolution", {}).get("fields_exactly")
        != list(PROBE_RESOLUTION_FIELDS)
        or schemas.get("probe_attempt_marker", {}).get("fields_exactly")
        != list(PROBE_ATTEMPT_FIELDS)
        or schemas.get("probe_failure_marker", {}).get("fields_exactly")
        != list(PROBE_FAILURE_FIELDS)
        or schemas.get("probe_terminal_marker", {}).get("fields_exactly")
        != list(PROBE_TERMINAL_FIELDS)
        or seed.get("contract_id") != PRODUCTION_SEED_CONTRACT_ID
        or seed.get("production_policy_keys_required")
        != ["default", "project", "global", "session"]
        or seed.get("retrieval_weights_table_must_be_nonempty") is not True
        # The frozen plan states outright that the node-scope grammar is not
        # the retrieval-policy-key grammar.  A plan that dropped or relaxed
        # that statement would re-authorize the defect this port repairs.
        or seed.get("node_scope_grammar_applied_to_retrieval_weight_policy_key")
        is not False
        or seed.get("policy_key_mapping_exact") != POLICY_KEY_MAPPING_EXACT
        or replay.get("probe_builds_corpus") is not False
        or replay.get("probe_builds_candidate_plan") is not False
        or readiness.get("probe_launches_consume_authority") is not False
        or ambient_schema.get("behaviorally_relevant_keys_exactly")
        != [
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
        ]
    ):
        _fail()
    expected_holdout = readiness.get("holdout_floor", {})
    expected_shadow = readiness.get("shadow_floor", {})
    if expected_holdout != {
        "unseen_in_dev_repeated_automatic_families_min": 30,
        "unseen_in_dev_repeated_automatic_events_min": 150,
        "distinct_components_containing_qualifying_unseen_automatic_family_min": 30,
        "organic_events_min": 200,
        "distinct_nonempty_source_qualified_organic_sessions_min": 30,
        "distinct_components_containing_organic_event_min": 30,
        "distinct_requested_project_scopes_min": 2,
    } or expected_shadow != {
        "genuine_nonsynthetic_real_workflows_min": 30,
        "unique_replayable_logical_calls_min": 100,
        "distinct_components_containing_counted_workflow_min": 30,
        "distinct_requested_project_scopes_min": 2,
    }:
        _fail()
    _require_policy_key_vectors(seed.get("policy_key_vectors"))
    _require_partition_golden_vectors(partition.get("golden_vectors"))
    if (
        type(dev) is not dict
        or type(dev.get("bytes")) is not int
        or dev.get("bytes", -1) < 0
        or type(dev.get("sha256")) is not str
    ):
        _fail()
    _hex64(dev["sha256"])
    return plan


def _load_probe_plan_bytes(raw: bytes, *, expected_sha256: str) -> dict[str, Any]:
    _hex64(expected_sha256)
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256):
        _fail()
    return _validate_probe_plan(runtime.load_json_bytes(raw, expected=dict))


def _open_snapshot(fd: int) -> sqlite3.Connection:
    try:
        header = os.pread(fd, 100, 0)
    except OSError:
        _fail()
    # Immutable SQLite must be a self-contained rollback-journal snapshot.
    # A WAL-mode main file can otherwise hide uncheckpointed selected rows.
    if (
        len(header) != 100
        or header[:16] != b"SQLite format 3\0"
        or header[18] != 1
        or header[19] != 1
    ):
        _fail()
    uri = f"file:/proc/self/fd/{fd}?mode=ro&immutable=1&cache=private"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
            _fail()
        if connection.execute("PRAGMA temp_store").fetchone()[0] != 2:
            _fail()
        check = connection.execute("PRAGMA main.quick_check").fetchall()
        if len(check) != 1 or check[0][0] != "ok":
            _fail()
        _validate_snapshot_schema(connection)
        return connection
    except IntegrityFailure:
        try:
            connection.close()
        except (UnboundLocalError, sqlite3.Error):
            pass
        raise
    except (sqlite3.Error, TypeError, ValueError):
        try:
            connection.close()
        except (UnboundLocalError, sqlite3.Error):
            pass
        _fail()


def _split_bucket(event_id: str) -> int:
    digest = hashlib.sha256(_utf8(event_id, nonempty=True)).digest()
    return int.from_bytes(digest[:8], "big") % SPLIT_MODULUS


def _identity_message(query: str, requested_scope: str) -> bytes:
    normalized = " ".join(query.split())
    return IDENTITY_DOMAIN + _utf8(normalized + "\n" + requested_scope)


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
) -> tuple[set[bytes], FileProof]:
    contract = plan["identity"]["complete_frozen_dev_reference"][
        "raw_identity_reference"
    ]
    before = _stat_regular_fd(fd)
    raw = _read_fd(fd, maximum=contract["bytes"])
    after = _stat_regular_fd(fd)
    if (
        len(raw) != contract["bytes"]
        or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), contract["sha256"])
        or before != after
        or (raw and not raw.endswith(b"\n"))
    ):
        _fail()
    identifiers: set[str] = set()
    tokens: set[bytes] = set()
    for line in raw.splitlines():
        if not line:
            _fail()
        row = runtime.load_json_bytes(line, expected=dict)
        if not {"event_id", "query", "requested_scope", "agent"}.issubset(row):
            _fail()
        event_id = row["event_id"]
        if "id" in row and row["id"] != event_id:
            _fail()
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
            and DEV_SPLIT_LOWER <= _split_bucket(event_id) < DEV_SPLIT_UPPER
        ):
            tokens.add(
                _identity_token(key, query, requested_scope, collision_map)
            )
    return tokens, FileProof(
        identity=after,
        sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw),
    )


def _normalize_scope(value: Any) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        return "global"
    if cleaned == "global":
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


def _ambient_scope(ambient: Mapping[str, Any]) -> str | None:
    raw_scope = ambient.get("scope")
    if raw_scope:
        return _normalize_scope(str(raw_scope))
    raw_session = ambient.get("session_id") or ambient.get("session")
    if raw_session:
        return _normalize_scope(f"session:{raw_session}")
    raw_project = (
        ambient.get("project")
        or ambient.get("project_name")
        or ambient.get("workspace")
    )
    if raw_project:
        return _normalize_scope(str(raw_project))
    raw_path = ambient.get("workspace_path") or ambient.get("cwd")
    if raw_path:
        name = Path(str(raw_path)).name
        if name:
            return _normalize_scope(name)
    return None


def _ambient_project(
    ambient: Mapping[str, Any], ambient_scope: str | None
) -> str | None:
    if ambient_scope is not None and ambient_scope.startswith("project:"):
        return ambient_scope

    project_scope = ambient.get("project_scope")
    if project_scope:
        normalized = _normalize_scope(str(project_scope))
        if normalized.startswith("project:"):
            return normalized
    raw_project = (
        ambient.get("project")
        or ambient.get("project_name")
        or ambient.get("workspace")
    )
    if raw_project:
        normalized = _normalize_scope(str(raw_project))
        if normalized.startswith("project:"):
            return normalized
    raw_path = ambient.get("workspace_path") or ambient.get("cwd")
    if raw_path:
        name = Path(str(raw_path)).name
        if name:
            normalized = _normalize_scope(name)
            if normalized.startswith("project:"):
                return normalized
    return None


def _expected_scope_plan(
    requested_scope: Any, ambient: Mapping[str, Any]
) -> tuple[str, ...] | None:
    if not _node_scope_valid(requested_scope):
        return None
    try:
        ambient_scope = _ambient_scope(ambient)
        project = _ambient_project(ambient, ambient_scope)
        normalized_requested = _normalize_scope(requested_scope)
    except (TypeError, ValueError, UnicodeError, OSError):
        return None
    if normalized_requested != requested_scope:
        return None
    if normalized_requested == "global":
        return ("global",)
    if normalized_requested.startswith("project:"):
        return (normalized_requested, "global")
    values = [normalized_requested]
    if project is not None and project != normalized_requested:
        values.append(project)
    values.append("global")
    return tuple(dict.fromkeys(values))


def _depth_accepted(value: Any) -> bool:
    if value is None:
        return True
    if type(value) is not str:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    # Serving _parse_depth accepts every string: named/numeric spellings are
    # normalized and unknown strings deterministically fall back to depth 1.
    return True


def _ambient_scalars_valid(ambient: Mapping[str, Any]) -> bool:
    for key, value in ambient.items():
        if type(key) is not str or key not in AMBIENT_KEYS:
            return False
        if value is None or type(value) in (str, bool, int):
            if type(value) is str:
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError:
                    return False
            continue
        if type(value) is float and math.isfinite(value):
            continue
        return False
    return True


def _event_replayable(
    *,
    query: Any,
    scope: Any,
    requested_scope: Any,
    resolved_scopes_raw: Any,
    ambient_context_raw: Any,
    depth: Any,
    max_results: Any,
    agent: Any,
    task: Any,
    session_id: Any,
    transport_session_id: Any,
) -> bool:
    if (
        type(query) is not str
        or not query
        or not _node_scope_valid(requested_scope)
        or type(scope) is not str
        or scope != requested_scope
        or not _is_int(max_results)
        or max_results <= 0
        or not _depth_accepted(depth)
    ):
        return False
    try:
        query.encode("utf-8")
    except UnicodeEncodeError:
        return False
    resolved = _load_metadata_json(resolved_scopes_raw, list)
    ambient = _load_metadata_json(ambient_context_raw, dict)
    if resolved is None or ambient is None or not _ambient_scalars_valid(ambient):
        return False
    if not all(type(value) is str and _node_scope_valid(value) for value in resolved):
        return False
    expected_plan = _expected_scope_plan(requested_scope, ambient)
    if expected_plan is None or tuple(resolved) != expected_plan:
        return False

    if not _same_json_scalar(agent, ambient.get("agent")):
        return False
    if not _same_json_scalar(task, ambient.get("task")):
        return False
    caller_session = ambient.get("session_id") or ambient.get("session")
    if not _same_json_scalar(session_id, caller_session):
        return False
    if not _same_json_scalar(
        transport_session_id, ambient.get("transport_session_id")
    ):
        return False
    return True


@dataclass(slots=True)
class _Event:
    alias: str
    event_id: str
    created_at_us: int
    query: Any
    requested_scope: Any
    agent: Any
    transport_session_id: Any
    token: bytes | None
    replayable: bool

    @property
    def key_bytes(self) -> bytes:
        return _utf8(self.alias, nonempty=True) + b"\0" + _utf8(
            self.event_id, nonempty=True
        )

    @property
    def session_key(self) -> tuple[str, str] | None:
        if type(self.transport_session_id) is str and self.transport_session_id:
            return self.alias, self.transport_session_id
        return None


def _snapshot_events(
    fd: int,
    *,
    alias: str,
    lower_us: int,
    release_us: int,
    upper_us: int,
    key: bytes | bytearray,
    collision_map: dict[bytes, bytes],
) -> list[_Event]:
    if alias not in SOURCE_ALIASES:
        _fail()
    connection = _open_snapshot(fd)
    events: list[_Event] = []
    identifiers: set[str] = set()
    try:
        try:
            rows = connection.execute(
                "SELECT id, created_at, query, scope, requested_scope, "
                "resolved_scopes, ambient_context, depth, max_results, agent, "
                "task, session_id, transport_session_id FROM main.recall_events"
            )
            for row in rows:
                (
                    event_id,
                    created_at,
                    query,
                    scope,
                    requested_scope,
                    resolved_scopes,
                    ambient_context,
                    depth,
                    max_results,
                    agent,
                    task,
                    session_id,
                    transport_session_id,
                ) = tuple(row)
                _utf8(event_id, nonempty=True)
                if event_id in identifiers:
                    _fail()
                identifiers.add(event_id)
                created_us = runtime.utc_microseconds(created_at)
                if not (
                    created_us > release_us
                    and created_us > lower_us
                    and created_us <= upper_us
                ):
                    continue
                token: bytes | None = None
                if agent is None and type(query) is str and type(requested_scope) is str:
                    try:
                        query.encode("utf-8")
                        requested_scope.encode("utf-8")
                    except UnicodeEncodeError:
                        # Invalid UTF-8 metadata removes only the identity edge.
                        token = None
                    else:
                        # Any failure after encoding, notably an HMAC collision,
                        # is a slot-fatal integrity error rather than missing
                        # family identity.
                        token = _identity_token(
                            key, query, requested_scope, collision_map
                        )
                replayable = _event_replayable(
                    query=query,
                    scope=scope,
                    requested_scope=requested_scope,
                    resolved_scopes_raw=resolved_scopes,
                    ambient_context_raw=ambient_context,
                    depth=depth,
                    max_results=max_results,
                    agent=agent,
                    task=task,
                    session_id=session_id,
                    transport_session_id=transport_session_id,
                )
                events.append(
                    _Event(
                        alias=alias,
                        event_id=event_id,
                        created_at_us=created_us,
                        query=query,
                        requested_scope=requested_scope,
                        agent=agent,
                        transport_session_id=transport_session_id,
                        token=token,
                        replayable=replayable,
                    )
                )
        except IntegrityFailure:
            raise
        except (sqlite3.Error, TypeError, ValueError, UnicodeError, OverflowError):
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
    for index in indices[1:]:
        disjoint.union(indices[0], index)


def _partition_for_representative(representative: bytes) -> str:
    digest = hashlib.sha256(PARTITION_DOMAIN + representative).digest()
    bucket = int.from_bytes(digest[:4], "big") % SPLIT_MODULUS
    return "holdout" if bucket < 50 else "shadow"


def _aggregate_invariants(counts: Mapping[str, Any]) -> None:
    _exact_mapping(counts, AGGREGATE_FIELDS)
    for field in AGGREGATE_FIELDS:
        _nonnegative_int(counts[field])
    selected = counts["selected_event_count"]
    replayable = counts["selected_replayable_event_count"]
    nonreplayable = counts["selected_nonreplayable_event_count"]
    families = counts["holdout_unseen_automatic_family_count"]
    automatic = counts["holdout_unseen_automatic_event_count"]
    automatic_components = counts["holdout_unseen_automatic_component_count"]
    organic = counts["holdout_organic_event_count"]
    organic_sessions = counts["holdout_organic_session_count"]
    organic_components = counts["holdout_organic_component_count"]
    workflows = counts["shadow_real_workflow_count"]
    calls = counts["shadow_replayable_logical_call_count"]
    workflow_components = counts["shadow_real_workflow_component_count"]
    if (
        selected != replayable + nonreplayable
        or automatic < 3 * families
        or automatic_components > families
        or (families == 0) != (automatic == 0)
        or (families == 0) != (automatic_components == 0)
        or organic_sessions > organic
        or organic_components > organic
        or (organic == 0) != (organic_components == 0)
        or workflows > calls
        or workflow_components > workflows
        or (workflows == 0) != (calls == 0)
        or (workflows == 0) != (workflow_components == 0)
        or automatic + organic + calls > replayable
        or counts["holdout_project_scope_count"] > replayable
        or counts["shadow_project_scope_count"] > calls
    ):
        _fail()


def _all_floors_pass(counts: Mapping[str, int]) -> bool:
    return all(counts[field] >= minimum for field, minimum in HOLDOUT_FLOORS.items()) and all(
        counts[field] >= minimum for field, minimum in SHADOW_FLOORS.items()
    )


def _analyze(
    *,
    plan: Mapping[str, Any],
    local_fd: int,
    alt_fd: int,
    dev_fd: int,
    slot_index: int,
    active_lower_bound_exclusive_at: str,
    key: bytearray,
) -> tuple[dict[str, int], dict[str, runtime.HashAndBytes]]:
    slot = runtime.slot_times(slot_index)
    release_us = runtime.utc_microseconds(RELEASE_EFFECTIVE_AT, receipt=True)
    lower_us = runtime.utc_microseconds(
        active_lower_bound_exclusive_at, receipt=True
    )
    upper_us = runtime.utc_microseconds(slot.scheduled_at, receipt=True)
    if lower_us < release_us or upper_us <= lower_us:
        _fail()

    if type(key) is not bytearray or len(key) != KEY_BYTES:
        _fail()
    collision_map: dict[bytes, bytes] = {}
    dev_tokens: set[bytes] = set()
    events: list[_Event] = []
    try:
        local_proof = _proof_fd(local_fd)
        alt_proof = _proof_fd(alt_fd)
        if (
            local_proof.identity[:2] == alt_proof.identity[:2]
            or hmac.compare_digest(local_proof.sha256, alt_proof.sha256)
        ):
            _fail()
        dev_tokens, dev_proof = _dev_tokens(
            dev_fd, plan=plan, key=key, collision_map=collision_map
        )
        events.extend(
            _snapshot_events(
                local_fd,
                alias="local",
                lower_us=lower_us,
                release_us=release_us,
                upper_us=upper_us,
                key=key,
                collision_map=collision_map,
            )
        )
        events.extend(
            _snapshot_events(
                alt_fd,
                alias="alt",
                lower_us=lower_us,
                release_us=release_us,
                upper_us=upper_us,
                key=key,
                collision_map=collision_map,
            )
        )
        events.sort(
            key=lambda event: (
                event.created_at_us,
                _utf8(event.alias),
                _utf8(event.event_id),
            )
        )
        event_keys = [event.key_bytes for event in events]
        if len(event_keys) != len(set(event_keys)):
            _fail()

        family_groups: dict[tuple[str, bytes], list[int]] = defaultdict(list)
        session_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, event in enumerate(events):
            if event.agent is None and event.token is not None:
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
        component_partition: dict[int, str] = {}
        digest_representatives: dict[bytes, bytes] = {}
        for root, indices in components.items():
            representative = min(events[index].key_bytes for index in indices)
            digest = hashlib.sha256(PARTITION_DOMAIN + representative).digest()
            previous = digest_representatives.get(digest)
            if previous is not None and previous != representative:
                _fail()
            digest_representatives[digest] = representative
            component_partition[root] = _partition_for_representative(representative)
        partitions = [
            component_partition[disjoint.find(index)] for index in range(len(events))
        ]
        holdout_indices = {
            index for index, partition in enumerate(partitions) if partition == "holdout"
        }
        shadow_indices = {
            index for index, partition in enumerate(partitions) if partition == "shadow"
        }
        if holdout_indices & shadow_indices or holdout_indices | shadow_indices != set(
            range(len(events))
        ):
            _fail()

        qualifying_families: dict[tuple[str, bytes], list[int]] = {}
        for family, all_indices in family_groups.items():
            eligible = [
                index
                for index in all_indices
                if events[index].replayable and events[index].agent is None
            ]
            sessions = {
                events[index].session_key
                for index in eligible
                if events[index].session_key is not None
            }
            if (
                len(eligible) >= 3
                and len(sessions) >= 2
                and family[1] not in dev_tokens
            ):
                qualifying_families[family] = eligible

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
            if events[index].replayable and events[index].agent is not None
        }
        holdout_organic_sessions = {
            events[index].session_key
            for index in holdout_organic_indices
            if events[index].session_key is not None
        }

        counted_workflows: set[tuple[str, str]] = set()
        for workflow, indices in session_groups.items():
            if partitions[indices[0]] != "shadow":
                continue
            replayable_indices = [
                index for index in indices if events[index].replayable
            ]
            if replayable_indices and any(
                events[index].replayable and events[index].agent is not None
                for index in indices
            ):
                counted_workflows.add(workflow)
        shadow_replayable_indices = {
            index
            for workflow in counted_workflows
            for index in session_groups[workflow]
            if events[index].replayable
        }

        replayable_count = sum(event.replayable for event in events)
        counts: dict[str, int] = {
            "selected_event_count": len(events),
            "selected_replayable_event_count": replayable_count,
            "selected_nonreplayable_event_count": len(events) - replayable_count,
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
                    if events[index].replayable
                    and type(events[index].requested_scope) is str
                    and events[index].requested_scope.startswith("project:")
                }
            ),
            "shadow_real_workflow_count": len(counted_workflows),
            "shadow_replayable_logical_call_count": len(
                shadow_replayable_indices
            ),
            "shadow_real_workflow_component_count": len(
                {
                    disjoint.find(session_groups[workflow][0])
                    for workflow in counted_workflows
                }
            ),
            "shadow_project_scope_count": len(
                {
                    events[index].requested_scope
                    for index in shadow_replayable_indices
                    if type(events[index].requested_scope) is str
                    and events[index].requested_scope.startswith("project:")
                }
            ),
        }
        if tuple(counts) != AGGREGATE_FIELDS:
            _fail()
        _aggregate_invariants(counts)
        _require_unchanged(local_fd, local_proof)
        _require_unchanged(alt_fd, alt_proof)
        _require_unchanged(dev_fd, dev_proof)
        return counts, {
            "local": local_proof.hash_and_bytes(),
            "alt": alt_proof.hash_and_bytes(),
        }
    finally:
        for index in range(len(key)):
            key[index] = 0
        collision_map.clear()
        dev_tokens.clear()
        events.clear()


def _validate_worker_result(value: Any) -> dict[str, int]:
    if type(value) is not dict:
        _fail()
    _aggregate_invariants(value)
    return {field: value[field] for field in AGGREGATE_FIELDS}


def _validate_worker_envelope(
    value: Any,
    *,
    expected_plan_identity: runtime.HashAndBytes,
    expected_slot_index: int,
    expected_active_lower_bound_exclusive_at: str,
) -> tuple[dict[str, int], dict[str, runtime.HashAndBytes]]:
    envelope = _exact_mapping(
        value,
        (
            "slot_index",
            "scheduled_at",
            "grace_deadline_at",
            "active_segment_lower_bound_exclusive_at",
            "aggregate",
            "aliased_source_snapshot_sha256_and_bytes",
            "analysis_plan_sha256_and_bytes",
        ),
    )
    slot = runtime.slot_times(envelope["slot_index"])
    if (
        slot.slot_index != expected_slot_index
        or envelope["scheduled_at"] != slot.scheduled_at
        or envelope["grace_deadline_at"] != slot.grace_deadline_at
        or envelope["active_segment_lower_bound_exclusive_at"]
        != expected_active_lower_bound_exclusive_at
        or runtime.HashAndBytes.from_value(
            envelope["analysis_plan_sha256_and_bytes"]
        )
        != expected_plan_identity
    ):
        _fail()
    return (
        _validate_worker_result(envelope["aggregate"]),
        _validate_snapshot_map(
            envelope["aliased_source_snapshot_sha256_and_bytes"]
        ),
    )


def _receive_source_fds(
    control_fd: int, key: bytearray
) -> tuple[int, int]:
    if type(key) is not bytearray or len(key) != KEY_BYTES:
        _fail()
    try:
        control = socket.socket(fileno=control_fd)
        control.settimeout(WORKER_HANDSHAKE_TIMEOUT_SECONDS)
        control.sendall(b"K")
        payload, ancillary, flags, _ = control.recvmsg(
            1, socket.CMSG_SPACE(2 * array("i").itemsize)
        )
    except (OSError, TimeoutError, ValueError):
        _fail()
    finally:
        try:
            control.close()
        except (UnboundLocalError, OSError):
            pass
    if payload != b"S" or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
        _fail()
    descriptor_arrays: list[array[int]] = []
    for level, kind, raw in ancillary:
        if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
            _fail()
        descriptors = array("i")
        usable = len(raw) - (len(raw) % descriptors.itemsize)
        descriptors.frombytes(raw[:usable])
        descriptor_arrays.append(descriptors)
    if len(descriptor_arrays) != 1 or len(descriptor_arrays[0]) != 2:
        _fail()
    local_fd, alt_fd = tuple(descriptor_arrays[0])
    if local_fd == alt_fd:
        _fail()
    try:
        for descriptor in (local_fd, alt_fd):
            flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
            if flags & os.O_ACCMODE != os.O_RDONLY:
                _fail()
            os.set_inheritable(descriptor, False)
    except OSError:
        for descriptor in (local_fd, alt_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass
        _fail()
    return local_fd, alt_fd


def _worker_entry(arguments: Sequence[str]) -> int:
    expected_names = (
        "--plan-fd",
        "--dev-fd",
        "--result-fd",
        "--control-fd",
        "--slot-index",
        "--active-lower-bound-exclusive-at",
        "--expected-plan-sha256",
    )
    if len(arguments) != len(expected_names) * 2 or tuple(arguments[0::2]) != expected_names:
        return 73
    key = bytearray()
    source_fds: tuple[int, int] = ()
    try:
        plan_fd, dev_fd, result_fd, control_fd = map(
            int, arguments[1:8:2]
        )
        slot_index = int(arguments[9])
        active_lower = arguments[11]
        expected_plan_sha256 = arguments[13]
        plan_raw = _read_fd(plan_fd, maximum=MAX_PLAN_BYTES)
        plan = _load_probe_plan_bytes(
            plan_raw, expected_sha256=expected_plan_sha256
        )
        key = bytearray(secrets.token_bytes(KEY_BYTES))
        if len(key) != KEY_BYTES:
            _fail()
        source_fds = _receive_source_fds(control_fd, key)
        local_fd, alt_fd = source_fds
        aggregate, snapshots = _analyze(
            plan=plan,
            local_fd=local_fd,
            alt_fd=alt_fd,
            dev_fd=dev_fd,
            slot_index=slot_index,
            active_lower_bound_exclusive_at=active_lower,
            key=key,
        )
        slot = runtime.slot_times(slot_index)
        raw = runtime.canonical_json_bytes(
            {
                "slot_index": slot.slot_index,
                "scheduled_at": slot.scheduled_at,
                "grace_deadline_at": slot.grace_deadline_at,
                "active_segment_lower_bound_exclusive_at": active_lower,
                "aggregate": aggregate,
                "aliased_source_snapshot_sha256_and_bytes": {
                    alias: snapshots[alias].as_dict()
                    for alias in SOURCE_ALIASES
                },
                "analysis_plan_sha256_and_bytes": {
                    "sha256": expected_plan_sha256,
                    "bytes": len(plan_raw),
                },
            }
        )
        if len(raw) > MAX_WORKER_RESULT_BYTES:
            _fail()
        _write_all(result_fd, raw)
        return 0
    except BaseException:
        return 73
    finally:
        for index in range(len(key)):
            key[index] = 0
        for descriptor in source_fds:
            try:
                os.close(descriptor)
            except OSError:
                pass
        for offset in (1, 3, 5, 7):
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


def _run_keyed_worker(
    *,
    probe_fd: int,
    runtime_fd: int,
    plan_fd: int,
    dev_fd: int,
    source_fd_supplier: Callable[[], dict[str, int]],
    slot_index: int,
    active_lower_bound_exclusive_at: str,
    expected_plan_sha256: str,
) -> tuple[dict[str, int], dict[str, runtime.HashAndBytes], bytes]:
    try:
        result_read, result_write = os.pipe()
    except OSError:
        _fail()
    try:
        control_parent, control_child = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_STREAM
        )
    except OSError:
        os.close(result_read)
        os.close(result_write)
        _fail()
    command = [
        sys.executable,
        "-B",
        "-I",
        "-S",
        "-c",
        WORKER_BOOTSTRAP,
        f"/proc/self/fd/{runtime_fd}",
        f"/proc/self/fd/{probe_fd}",
        "__worker",
        "--plan-fd",
        str(plan_fd),
        "--dev-fd",
        str(dev_fd),
        "--result-fd",
        str(result_write),
        "--control-fd",
        str(control_child.fileno()),
        "--slot-index",
        str(slot_index),
        "--active-lower-bound-exclusive-at",
        active_lower_bound_exclusive_at,
        "--expected-plan-sha256",
        expected_plan_sha256,
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(
                probe_fd,
                runtime_fd,
                plan_fd,
                dev_fd,
                result_write,
                control_child.fileno(),
            ),
            start_new_session=True,
        )
    except (OSError, ValueError):
        os.close(result_read)
        os.close(result_write)
        control_parent.close()
        control_child.close()
        _fail()
    os.close(result_write)
    control_child.close()
    try:
        try:
            control_parent.settimeout(WORKER_HANDSHAKE_TIMEOUT_SECONDS)
            if control_parent.recv(1) != b"K":
                _fail()
            supplied = _exact_mapping(source_fd_supplier(), SOURCE_ALIASES)
            source_fds: list[int] = []
            for alias in SOURCE_ALIASES:
                descriptor = supplied[alias]
                if (
                    not _is_int(descriptor)
                    or descriptor < 0
                    or descriptor in source_fds
                    or fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
                    != os.O_RDONLY
                    or not stat.S_ISREG(os.fstat(descriptor).st_mode)
                ):
                    _fail()
                source_fds.append(descriptor)
            rights = array("i", source_fds)
            if control_parent.sendmsg(
                [b"S"],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)],
            ) != 1:
                _fail()
        except IntegrityFailure:
            raise
        except BaseException:
            _fail()
        try:
            return_code = process.wait(timeout=WORKER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _terminate_worker_group(process)
            _fail()
        _terminate_worker_group(process)
        raw = _read_worker_result(result_read)
        if return_code != 0:
            _fail()
        value = runtime.load_canonical_json_bytes(raw, expected=dict)
        aggregate, snapshots = _validate_worker_envelope(
            value,
            expected_plan_identity=runtime.HashAndBytes(
                expected_plan_sha256, os.fstat(plan_fd).st_size
            ),
            expected_slot_index=slot_index,
            expected_active_lower_bound_exclusive_at=(
                active_lower_bound_exclusive_at
            ),
        )
        return aggregate, snapshots, raw
    finally:
        try:
            _terminate_worker_group(process)
            if process.poll() is None:
                process.wait(timeout=10)
        finally:
            control_parent.close()
            os.close(result_read)


def _capture_snapshot_fds(
    active_segment: runtime.ActiveSegment,
    capture_snapshot: Callable[[str, str, str], int],
) -> dict[str, int]:
    """Invoke the authenticated capture boundary once for each exact alias."""

    if not callable(capture_snapshot):
        _fail()
    databases = active_segment.source_binding.core.database_map()
    result: dict[str, int] = {}
    descriptors: set[int] = set()
    for alias in SOURCE_ALIASES:
        try:
            descriptor = capture_snapshot(
                alias, runtime.ALIAS_IDS[alias], databases[alias]
            )
        except IntegrityFailure:
            raise
        except BaseException:
            _fail()
        if (
            not _is_int(descriptor)
            or descriptor < 0
            or descriptor in descriptors
        ):
            _fail()
        descriptors.add(descriptor)
        result[alias] = descriptor
    return result


_COMPUTATION_ORIGIN = object()


@dataclass(frozen=True, slots=True, init=False)
class ProbeComputation:
    slot_index: int
    scheduled_at: str
    grace_deadline_at: str
    active_segment_lower_bound_exclusive_at: str
    aggregate_items: tuple[tuple[str, int], ...]
    snapshot_items: tuple[tuple[str, runtime.HashAndBytes], ...]
    analysis_plan_identity: runtime.HashAndBytes
    segment_id: str | None
    segment_attestation_identity: runtime.HashAndBytes | None
    source_binding_attestation_identity: runtime.HashAndBytes | None
    source_database_identity_items: tuple[tuple[str, str], ...] | None
    worker_envelope_raw: bytes
    probe_identity: runtime.HashAndBytes
    _origin: object

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _fail()

    def aggregate(self) -> dict[str, int]:
        value = dict(self.aggregate_items)
        _aggregate_invariants(value)
        return value

    def snapshots(self) -> dict[str, runtime.HashAndBytes]:
        value = dict(self.snapshot_items)
        if set(value) != set(SOURCE_ALIASES):
            _fail()
        return value

    @property
    def provisional_ready(self) -> bool:
        return _all_floors_pass(self.aggregate())


def _make_probe_computation(**values: Any) -> ProbeComputation:
    expected = {
        field
        for field in ProbeComputation.__dataclass_fields__
        if field != "_origin"
    }
    if set(values) != expected:
        _fail()
    result = object.__new__(ProbeComputation)
    for field in expected:
        object.__setattr__(result, field, values[field])
    object.__setattr__(result, "_origin", _COMPUTATION_ORIGIN)
    return result


def _compute_aggregate_supplier(
    *,
    plan_raw: bytes,
    expected_plan_sha256: str,
    source_input_supplier: Callable[[], dict[str, int]],
    dev_input_fd: int,
    slot_index: int,
    active_lower_bound_exclusive_at: str,
) -> ProbeComputation:
    """Testable FD core; production callers use ``compute_probe_aggregate``."""

    plan = _load_probe_plan_bytes(
        plan_raw, expected_sha256=expected_plan_sha256
    )
    slot = runtime.slot_times(slot_index)
    lower = runtime.parse_utc(active_lower_bound_exclusive_at, receipt=True)
    if (
        lower < runtime.parse_utc(RELEASE_EFFECTIVE_AT, receipt=True)
        or lower >= runtime.parse_utc(slot.scheduled_at, receipt=True)
    ):
        _fail()
    owned: list[int] = []
    try:
        dev_fd = _duplicate_readonly_regular_fd(dev_input_fd)
        owned.append(dev_fd)
        probe_fd = _open_readonly_regular_path(Path(__file__).resolve())
        owned.append(probe_fd)
        runtime_fd = _open_readonly_regular_path(
            Path(runtime.__file__).resolve()
        )
        owned.append(runtime_fd)
        probe_proof = _proof_fd(probe_fd)
        runtime_proof = _proof_fd(runtime_fd)
        try:
            plan_write_fd = os.memfd_create(
                "confirmatory-v4-analysis-plan",
                os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
            )
        except (AttributeError, OSError):
            _fail()
        owned.append(plan_write_fd)
        _write_all(plan_write_fd, plan_raw)
        try:
            fcntl.fcntl(
                plan_write_fd,
                fcntl.F_ADD_SEALS,
                fcntl.F_SEAL_WRITE
                | fcntl.F_SEAL_GROW
                | fcntl.F_SEAL_SHRINK
                | fcntl.F_SEAL_SEAL,
            )
            plan_fd = os.open(
                f"/proc/self/fd/{plan_write_fd}",
                os.O_RDONLY | os.O_CLOEXEC,
            )
        except (AttributeError, OSError):
            _fail()
        owned.append(plan_fd)

        states = {"dev": _stat_regular_fd(dev_fd)}
        source_descriptors: dict[str, int] = {}

        def supply_sources() -> dict[str, int]:
            supplied = _exact_mapping(
                source_input_supplier(), SOURCE_ALIASES
            )
            for alias in SOURCE_ALIASES:
                descriptor = _duplicate_readonly_regular_fd(supplied[alias])
                owned.append(descriptor)
                source_descriptors[alias] = descriptor
                states[alias] = _stat_regular_fd(descriptor)
            if states["local"][:2] == states["alt"][:2]:
                _fail()
            return dict(source_descriptors)

        dev_contract = plan["identity"]["complete_frozen_dev_reference"][
            "raw_identity_reference"
        ]
        if states["dev"][2] != dev_contract["bytes"]:
            _fail()
        aggregate, snapshots, worker_raw = _run_keyed_worker(
            probe_fd=probe_fd,
            runtime_fd=runtime_fd,
            plan_fd=plan_fd,
            dev_fd=dev_fd,
            source_fd_supplier=supply_sources,
            slot_index=slot_index,
            active_lower_bound_exclusive_at=active_lower_bound_exclusive_at,
            expected_plan_sha256=expected_plan_sha256,
        )
        for alias in SOURCE_ALIASES:
            _require_stat_unchanged(
                source_descriptors[alias], states[alias]
            )
        _require_stat_unchanged(dev_fd, states["dev"])
        _require_unchanged(probe_fd, probe_proof)
        _require_unchanged(runtime_fd, runtime_proof)
        return _make_probe_computation(
            slot_index=slot.slot_index,
            scheduled_at=slot.scheduled_at,
            grace_deadline_at=slot.grace_deadline_at,
            active_segment_lower_bound_exclusive_at=active_lower_bound_exclusive_at,
            aggregate_items=tuple((field, aggregate[field]) for field in AGGREGATE_FIELDS),
            snapshot_items=tuple(
                (alias, snapshots[alias]) for alias in SOURCE_ALIASES
            ),
            analysis_plan_identity=runtime.HashAndBytes(
                expected_plan_sha256, len(plan_raw)
            ),
            segment_id=None,
            segment_attestation_identity=None,
            source_binding_attestation_identity=None,
            source_database_identity_items=None,
            worker_envelope_raw=worker_raw,
            probe_identity=probe_proof.hash_and_bytes(),
        )
    finally:
        for fd in reversed(owned):
            try:
                os.close(fd)
            except OSError:
                pass


def _compute_aggregate_fds(
    *,
    plan_raw: bytes,
    expected_plan_sha256: str,
    local_input_fd: int,
    alt_input_fd: int,
    dev_input_fd: int,
    slot_index: int,
    active_lower_bound_exclusive_at: str,
) -> ProbeComputation:
    """Synthetic FD adapter; production captures only after the key handshake."""

    return _compute_aggregate_supplier(
        plan_raw=plan_raw,
        expected_plan_sha256=expected_plan_sha256,
        source_input_supplier=lambda: {
            "local": local_input_fd,
            "alt": alt_input_fd,
        },
        dev_input_fd=dev_input_fd,
        slot_index=slot_index,
        active_lower_bound_exclusive_at=active_lower_bound_exclusive_at,
    )


def compute_probe_aggregate(
    *,
    active_segment: runtime.ActiveSegment,
    observe_active_state: Callable[
        [], tuple[runtime.ServiceTuple, runtime.SourceBinding]
    ],
    capture_snapshot: Callable[[str, str, str], int],
    dev_reference_fd: int,
    slot_index: int,
    contract: runtime.FrozenContract | None = None,
    plan_path: Path | str = runtime.DEFAULT_PLAN,
    watermark_path: Path | str = runtime.DEFAULT_WATERMARK,
) -> ProbeComputation:
    """Run one ordered production probe through authenticated callbacks.

    ``capture_snapshot`` is the trusted launcher boundary: each invocation
    receives the exact alias, alias authority id, and attested database
    identity, performs its raw copy in its own isolation boundary, and returns
    only a retained read-only FD.  The keyed worker handshake occurs before
    either invocation; the two ``observe_active_state`` calls are owned here
    and strictly bracket both captures.
    """

    frozen = (
        runtime.load_frozen_contract(
            plan_path=plan_path, watermark_path=watermark_path
        )
        if contract is None
        else contract
    )
    _active_segment_recheck(frozen, active_segment)
    _require_segment_slot_alignment(active_segment, slot_index)
    if not callable(observe_active_state):
        _fail()

    def supply_sources_after_key() -> dict[str, int]:
        try:
            pre_services, pre_binding = observe_active_state()
        except IntegrityFailure:
            raise
        except BaseException:
            _fail()
        runtime.validate_active_segment_observations(
            active_segment,
            pre_services,
            pre_binding,
            pre_services,
            pre_binding,
            contract=frozen,
        )
        captures = _capture_snapshot_fds(
            active_segment, capture_snapshot
        )
        try:
            post_services, post_binding = observe_active_state()
        except IntegrityFailure:
            raise
        except BaseException:
            _fail()
        runtime.validate_active_segment_observations(
            active_segment,
            pre_services,
            pre_binding,
            post_services,
            post_binding,
            contract=frozen,
        )
        return captures

    plan_raw = _read_regular_path(Path(plan_path), maximum=MAX_PLAN_BYTES)
    runtime.validate_hash_and_bytes(plan_raw, frozen.plan_identity)
    computation = _compute_aggregate_supplier(
        plan_raw=plan_raw,
        expected_plan_sha256=frozen.plan_identity.sha256,
        source_input_supplier=supply_sources_after_key,
        dev_input_fd=dev_reference_fd,
        slot_index=slot_index,
        active_lower_bound_exclusive_at=active_segment.lower_bound_exclusive_at,
    )
    if computation.analysis_plan_identity != frozen.plan_identity:
        _fail()
    return _make_probe_computation(
        slot_index=computation.slot_index,
        scheduled_at=computation.scheduled_at,
        grace_deadline_at=computation.grace_deadline_at,
        active_segment_lower_bound_exclusive_at=(
            computation.active_segment_lower_bound_exclusive_at
        ),
        aggregate_items=computation.aggregate_items,
        snapshot_items=computation.snapshot_items,
        analysis_plan_identity=computation.analysis_plan_identity,
        segment_id=active_segment.segment_id,
        segment_attestation_identity=active_segment.attestation_identity,
        source_binding_attestation_identity=(
            active_segment.source_binding_attestation.identity
        ),
        source_database_identity_items=tuple(
            (alias, active_segment.source_binding.core.database_map()[alias])
            for alias in SOURCE_ALIASES
        ),
        worker_envelope_raw=computation.worker_envelope_raw,
        probe_identity=computation.probe_identity,
    )


def _self_identity() -> runtime.HashAndBytes:
    raw = _read_regular_path(Path(__file__).resolve(), maximum=2_000_000)
    return runtime.hash_and_bytes(raw)


def _hash_value(value: Any) -> runtime.HashAndBytes:
    return (
        value
        if type(value) is runtime.HashAndBytes
        else runtime.HashAndBytes.from_value(value)
    )


def _require_bound_active_segment(active_segment: Any) -> None:
    """Refuse every predecessor that is not a bound active segment.

    V4 models the watermark predecessor as its own
    :class:`~ap_confirmatory_runtime_v4.UnboundWatermarkPredecessor` type
    precisely because no source binding exists yet on that branch.  It carries
    a segment index, a segment id, a lower bound, and a service tuple, so it is
    duck-type compatible with almost everything below; only exact type identity
    keeps it out.  Ordinary probe work — key creation, source open, counting —
    is unauthorized until a validator-valid source-binding attestation has
    produced a real :class:`ActiveSegment`.
    """

    if type(active_segment) is not runtime.ActiveSegment:
        _fail()


def _active_segment_recheck(
    contract: runtime.FrozenContract, active_segment: runtime.ActiveSegment
) -> None:
    if type(contract) is not runtime.FrozenContract:
        _fail()
    _require_bound_active_segment(active_segment)
    runtime.validate_active_segment_observations(
        active_segment,
        active_segment.services,
        active_segment.source_binding,
        active_segment.services,
        active_segment.source_binding,
        contract=contract,
    )


def _require_segment_slot_alignment(
    active_segment: runtime.ActiveSegment, slot_index: int
) -> None:
    _require_bound_active_segment(active_segment)
    slot = runtime.slot_times(slot_index)
    lower = runtime.parse_utc(
        active_segment.lower_bound_exclusive_at, receipt=True
    )
    if (
        lower < runtime.parse_utc(RELEASE_EFFECTIVE_AT, receipt=True)
        or lower >= runtime.parse_utc(slot.scheduled_at, receipt=True)
        or active_segment.segment_index > slot.slot_index
    ):
        _fail()


def _require_production_computation_binding(
    computation: ProbeComputation,
    contract: runtime.FrozenContract,
    active_segment: runtime.ActiveSegment,
) -> None:
    if (
        type(computation) is not ProbeComputation
        or getattr(computation, "_origin", None) is not _COMPUTATION_ORIGIN
        or type(computation.analysis_plan_identity) is not runtime.HashAndBytes
        or computation.analysis_plan_identity != contract.plan_identity
        or computation.segment_id != active_segment.segment_id
        or computation.segment_attestation_identity
        != active_segment.attestation_identity
        or computation.source_binding_attestation_identity
        != active_segment.source_binding_attestation.identity
        or computation.active_segment_lower_bound_exclusive_at
        != active_segment.lower_bound_exclusive_at
        or computation.source_database_identity_items
        != tuple(
            (
                alias,
                active_segment.source_binding.core.database_map()[alias],
            )
            for alias in SOURCE_ALIASES
        )
        or type(computation.worker_envelope_raw) is not bytes
        or type(computation.probe_identity) is not runtime.HashAndBytes
        or computation.probe_identity != _self_identity()
    ):
        _fail()
    worker_value = runtime.load_canonical_json_bytes(
        computation.worker_envelope_raw, expected=dict
    )
    worker_aggregate, worker_snapshots = _validate_worker_envelope(
        worker_value,
        expected_plan_identity=contract.plan_identity,
        expected_slot_index=computation.slot_index,
        expected_active_lower_bound_exclusive_at=(
            active_segment.lower_bound_exclusive_at
        ),
    )
    if (
        computation.aggregate() != worker_aggregate
        or computation.snapshots() != worker_snapshots
    ):
        _fail()


def build_below_floor_resolution(
    computation: ProbeComputation,
    *,
    active_segment: runtime.ActiveSegment,
    launched_at: str,
    validated_at: str,
    previous_ledger_entry_sha256: str,
    contract: runtime.FrozenContract | None = None,
    probe_identity: runtime.HashAndBytes | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the sole public resolution this leaf can independently publish."""

    frozen = runtime.load_frozen_contract() if contract is None else contract
    _active_segment_recheck(frozen, active_segment)
    _require_production_computation_binding(
        computation, frozen, active_segment
    )
    if computation.provisional_ready:
        _fail()
    slot = runtime.validate_slot_times(
        slot_index=computation.slot_index,
        scheduled_at=computation.scheduled_at,
        grace_deadline_at=computation.grace_deadline_at,
        launched_at=launched_at,
        validated_at=validated_at,
    )
    _require_segment_slot_alignment(active_segment, slot.slot_index)
    previous = _hex64(previous_ledger_entry_sha256)
    probe = (
        computation.probe_identity
        if probe_identity is None
        else _hash_value(probe_identity)
    )
    if probe != computation.probe_identity:
        _fail()
    aggregate = computation.aggregate()
    snapshots = computation.snapshots()
    source_core = active_segment.source_binding.core
    databases = source_core.database_map()
    bindings = source_core.binding_map()
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "receipt_kind": "slot-probe-resolution",
        "status": "below-floor",
        "resolution_id": "",
        "slot_index": slot.slot_index,
        "scheduled_at": slot.scheduled_at,
        "grace_deadline_at": slot.grace_deadline_at,
        "launched_at": launched_at,
        "validated_at": validated_at,
        "release_effective_at": frozen.release_effective_at,
        "active_segment_lower_bound_exclusive_at": active_segment.lower_bound_exclusive_at,
        "segment_id": active_segment.segment_id,
        "segment_attestation_sha256_and_bytes": active_segment.attestation_identity.as_dict(),
        "source_binding_attestation_sha256_and_bytes": (
            active_segment.source_binding_attestation.identity.as_dict()
        ),
        "previous_ledger_entry_sha256": previous,
        "pre_active_services_state_sha256": active_segment.services.identity.sha256,
        "post_active_services_state_sha256": active_segment.services.identity.sha256,
        "pre_database_instance_identity_sha256_by_alias": databases,
        "post_database_instance_identity_sha256_by_alias": dict(databases),
        "pre_alias_service_database_binding_sha256_by_alias": bindings,
        "post_alias_service_database_binding_sha256_by_alias": dict(bindings),
        "aliased_source_snapshot_sha256_and_bytes": {
            alias: snapshots[alias].as_dict() for alias in SOURCE_ALIASES
        },
        **aggregate,
        "runtime_observer_sha256_and_bytes": (
            active_segment.source_binding_attestation.runtime_observer_identity.as_dict()
        ),
        "probe_sha256_and_bytes": probe.as_dict(),
        "analysis_plan_sha256_and_bytes": frozen.plan_identity.as_dict(),
        "seal_consumption_marker_sha256_and_bytes_or_null": None,
        "sealer_process_launched_at_or_null": None,
    }
    if tuple(receipt) != PROBE_RESOLUTION_FIELDS:
        _fail()
    receipt["resolution_id"] = runtime.derive_slot_resolution_id(receipt)
    raw = runtime.canonical_json_bytes(receipt)
    validate_probe_resolution(
        raw,
        contract=frozen,
        active_segment=active_segment,
        expected_previous_ledger_sha256=previous,
        expected_probe_identity=probe,
        expected_snapshot_identities=snapshots,
    )
    return receipt


@dataclass(frozen=True, slots=True)
class ValidatedProbeResolution:
    raw: bytes
    identity: runtime.HashAndBytes
    receipt: dict[str, Any]
    slot_index: int
    status: str
    validated_at: datetime


def _validate_alias_hash_map(value: Any) -> dict[str, str]:
    mapping = _exact_mapping(value, SOURCE_ALIASES)
    return {alias: _hex64(mapping[alias]) for alias in SOURCE_ALIASES}


def _validate_snapshot_map(
    value: Any,
) -> dict[str, runtime.HashAndBytes]:
    mapping = _exact_mapping(value, SOURCE_ALIASES)
    result = {
        alias: runtime.HashAndBytes.from_value(mapping[alias])
        for alias in SOURCE_ALIASES
    }
    if (
        any(identity.bytes <= 0 for identity in result.values())
        or len({identity.sha256 for identity in result.values()}) != len(SOURCE_ALIASES)
    ):
        _fail()
    return result


def validate_probe_resolution(
    raw: bytes,
    *,
    contract: runtime.FrozenContract,
    active_segment: runtime.ActiveSegment,
    expected_previous_ledger_sha256: str,
    expected_probe_identity: runtime.HashAndBytes | Mapping[str, Any] | None = None,
    expected_snapshot_identities: Mapping[
        str, runtime.HashAndBytes | Mapping[str, Any]
    ]
    | None = None,
    expected_seal_consumption_identity: runtime.HashAndBytes
    | Mapping[str, Any]
    | None = None,
    expected_sealer_process_launched_at: str | None = None,
) -> ValidatedProbeResolution:
    """Strictly validate one canonical below-floor or downstream-ready receipt."""

    _active_segment_recheck(contract, active_segment)
    receipt = runtime.load_canonical_json_bytes(raw, expected=dict)
    _exact_mapping(receipt, PROBE_RESOLUTION_FIELDS)
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or type(receipt["schema_version"]) is not int
        or receipt["namespace"] != NAMESPACE
        or receipt["receipt_kind"] != "slot-probe-resolution"
        or receipt["status"] not in ("below-floor", "ready")
    ):
        _fail()
    slot = runtime.validate_slot_times(
        slot_index=receipt["slot_index"],
        scheduled_at=receipt["scheduled_at"],
        grace_deadline_at=receipt["grace_deadline_at"],
        launched_at=receipt["launched_at"],
        validated_at=receipt["validated_at"],
    )
    _require_segment_slot_alignment(active_segment, slot.slot_index)
    if (
        receipt["release_effective_at"] != contract.release_effective_at
        or receipt["active_segment_lower_bound_exclusive_at"]
        != active_segment.lower_bound_exclusive_at
        or receipt["segment_id"] != active_segment.segment_id
        or runtime.HashAndBytes.from_value(
            receipt["segment_attestation_sha256_and_bytes"]
        )
        != active_segment.attestation_identity
        or runtime.HashAndBytes.from_value(
            receipt["source_binding_attestation_sha256_and_bytes"]
        )
        != active_segment.source_binding_attestation.identity
        or _hex64(receipt["previous_ledger_entry_sha256"])
        != _hex64(expected_previous_ledger_sha256)
        or receipt["pre_active_services_state_sha256"]
        != active_segment.services.identity.sha256
        or receipt["post_active_services_state_sha256"]
        != active_segment.services.identity.sha256
    ):
        _fail()
    databases = active_segment.source_binding.core.database_map()
    bindings = active_segment.source_binding.core.binding_map()
    if (
        _validate_alias_hash_map(
            receipt["pre_database_instance_identity_sha256_by_alias"]
        )
        != databases
        or _validate_alias_hash_map(
            receipt["post_database_instance_identity_sha256_by_alias"]
        )
        != databases
        or _validate_alias_hash_map(
            receipt["pre_alias_service_database_binding_sha256_by_alias"]
        )
        != bindings
        or _validate_alias_hash_map(
            receipt["post_alias_service_database_binding_sha256_by_alias"]
        )
        != bindings
    ):
        _fail()
    snapshots = _validate_snapshot_map(
        receipt["aliased_source_snapshot_sha256_and_bytes"]
    )
    if expected_snapshot_identities is not None:
        if set(expected_snapshot_identities) != set(SOURCE_ALIASES):
            _fail()
        expected_snapshots = {
            alias: _hash_value(expected_snapshot_identities[alias])
            for alias in SOURCE_ALIASES
        }
        if snapshots != expected_snapshots:
            _fail()

    counts = {field: receipt[field] for field in AGGREGATE_FIELDS}
    _aggregate_invariants(counts)
    expected_status = "ready" if _all_floors_pass(counts) else "below-floor"
    if receipt["status"] != expected_status:
        _fail()
    marker_value = receipt[
        "seal_consumption_marker_sha256_and_bytes_or_null"
    ]
    sealer_value = receipt["sealer_process_launched_at_or_null"]
    if expected_status == "below-floor":
        if marker_value is not None or sealer_value is not None:
            _fail()
        if (
            expected_seal_consumption_identity is not None
            or expected_sealer_process_launched_at is not None
        ):
            _fail()
    else:
        marker = runtime.HashAndBytes.from_value(marker_value)
        sealer_at = runtime.parse_utc(sealer_value, receipt=True)
        if not (
            runtime.parse_utc(receipt["launched_at"], receipt=True)
            <= sealer_at
            <= runtime.parse_utc(receipt["validated_at"], receipt=True)
        ):
            _fail()
        if (
            expected_seal_consumption_identity is None
            or marker != _hash_value(expected_seal_consumption_identity)
            or expected_sealer_process_launched_at is None
            or sealer_value != expected_sealer_process_launched_at
            or runtime.parse_utc(
                expected_sealer_process_launched_at, receipt=True
            )
            != sealer_at
        ):
            # Structural non-nullness and self-asserted time are not an
            # authenticated atomic-handoff proof.
            _fail()

    observer = runtime.HashAndBytes.from_value(
        receipt["runtime_observer_sha256_and_bytes"]
    )
    if observer != active_segment.source_binding_attestation.runtime_observer_identity:
        _fail()
    probe = runtime.HashAndBytes.from_value(receipt["probe_sha256_and_bytes"])
    expected_probe = (
        _self_identity()
        if expected_probe_identity is None
        else _hash_value(expected_probe_identity)
    )
    if probe != expected_probe:
        _fail()
    if (
        runtime.HashAndBytes.from_value(receipt["analysis_plan_sha256_and_bytes"])
        != contract.plan_identity
        or _hex64(receipt["resolution_id"])
        != runtime.derive_slot_resolution_id(receipt)
    ):
        _fail()
    return ValidatedProbeResolution(
        raw=raw,
        identity=runtime.hash_and_bytes(raw),
        receipt=dict(receipt),
        slot_index=slot.slot_index,
        status=receipt["status"],
        validated_at=runtime.parse_utc(receipt["validated_at"], receipt=True),
    )


@dataclass(frozen=True, slots=True)
class ValidatedProbeAttempt:
    raw: bytes
    identity: runtime.HashAndBytes
    slot_index: int
    attempt_ordinal: int
    launched_at: datetime
    watchdog_deadline_at: datetime
    previous_ledger_entry_sha256: str
    segment_id: str
    runtime_observer_identity: runtime.HashAndBytes
    probe_identity: runtime.HashAndBytes
    analysis_plan_identity: runtime.HashAndBytes


def _require_probe_attempt_wrapper_consistent(
    attempt: ValidatedProbeAttempt,
    contract: runtime.FrozenContract,
    active_segment: runtime.ActiveSegment | None = None,
) -> None:
    if (
        type(attempt) is not ValidatedProbeAttempt
        or type(attempt.raw) is not bytes
        or type(attempt.identity) is not runtime.HashAndBytes
        or type(attempt.slot_index) is not int
        or type(attempt.attempt_ordinal) is not int
        or type(attempt.launched_at) is not datetime
        or attempt.launched_at.tzinfo is not UTC
        or type(attempt.watchdog_deadline_at) is not datetime
        or attempt.watchdog_deadline_at.tzinfo is not UTC
        or type(attempt.previous_ledger_entry_sha256) is not str
        or type(attempt.segment_id) is not str
        or type(attempt.runtime_observer_identity) is not runtime.HashAndBytes
        or type(attempt.probe_identity) is not runtime.HashAndBytes
        or type(attempt.analysis_plan_identity) is not runtime.HashAndBytes
    ):
        _fail()
    if active_segment is not None:
        _active_segment_recheck(contract, active_segment)
        if (
            attempt.segment_id != active_segment.segment_id
            or attempt.runtime_observer_identity
            != active_segment.source_binding_attestation.runtime_observer_identity
        ):
            _fail()
    receipt = runtime.load_canonical_json_bytes(attempt.raw, expected=dict)
    _exact_mapping(receipt, PROBE_ATTEMPT_FIELDS)
    slot = runtime.slot_times(receipt["slot_index"])
    if active_segment is not None:
        _require_segment_slot_alignment(active_segment, slot.slot_index)
    launched = runtime.parse_utc(receipt["launched_at"], receipt=True)
    watchdog = runtime.parse_utc(receipt["watchdog_deadline_at"], receipt=True)
    grace = runtime.parse_utc(slot.grace_deadline_at, receipt=True)
    scheduled = runtime.parse_utc(slot.scheduled_at, receipt=True)
    latest_launch = grace - timedelta(seconds=60)
    if not scheduled <= launched < latest_launch:
        _fail()
    expected_watchdog = min(
        launched + timedelta(seconds=3600), latest_launch
    )
    if (
        runtime.hash_and_bytes(attempt.raw) != attempt.identity
        or receipt["schema_version"] != SCHEMA_VERSION
        or type(receipt["schema_version"]) is not int
        or receipt["namespace"] != NAMESPACE
        or receipt["receipt_kind"] != "probe-attempt"
        or receipt["scheduled_at"] != slot.scheduled_at
        or receipt["grace_deadline_at"] != slot.grace_deadline_at
        or _nonnegative_int(receipt["attempt_ordinal"])
        != attempt.attempt_ordinal
        or slot.slot_index != attempt.slot_index
        or launched != attempt.launched_at
        or watchdog != attempt.watchdog_deadline_at
        or watchdog != expected_watchdog
        or not launched < watchdog < grace
        or _hex64(receipt["previous_ledger_entry_sha256"])
        != _hex64(attempt.previous_ledger_entry_sha256)
        or _hex64(receipt["segment_id"]) != attempt.segment_id
        or runtime.HashAndBytes.from_value(
            receipt["runtime_observer_sha256_and_bytes"]
        )
        != attempt.runtime_observer_identity
        or runtime.HashAndBytes.from_value(receipt["probe_sha256_and_bytes"])
        != attempt.probe_identity
        or runtime.HashAndBytes.from_value(
            receipt["analysis_plan_sha256_and_bytes"]
        )
        != attempt.analysis_plan_identity
        or attempt.analysis_plan_identity != contract.plan_identity
    ):
        _fail()


def validate_probe_attempt_marker(
    raw: bytes,
    *,
    contract: runtime.FrozenContract,
    active_segment: runtime.ActiveSegment | None = None,
    expected_segment_id: str | None = None,
    expected_runtime_observer_identity: runtime.HashAndBytes
    | Mapping[str, Any]
    | None = None,
    expected_previous_ledger_sha256: str,
    expected_slot_index: int,
    expected_attempt_ordinal: int,
    expected_probe_identity: runtime.HashAndBytes | Mapping[str, Any] | None = None,
) -> ValidatedProbeAttempt:
    if type(contract) is not runtime.FrozenContract:
        _fail()
    # The initial slot-0 attempt precedes the source-binding ceremony that can
    # construct an ActiveSegment.  Every later attempt must bind an existing
    # active segment.  Do not silently mix these two authority paths.
    if active_segment is None:
        runtime._require_frozen_contract_consistent(contract)
        if (
            expected_segment_id != contract.initial_segment_id
            or expected_runtime_observer_identity is None
            or _nonnegative_int(expected_slot_index) != 0
            or _nonnegative_int(expected_attempt_ordinal) != 0
        ):
            _fail()
        segment_id = _hex64(expected_segment_id)
        observer_identity = _hash_value(expected_runtime_observer_identity)
    else:
        _active_segment_recheck(contract, active_segment)
        segment_id = active_segment.segment_id
        observer_identity = (
            active_segment.source_binding_attestation.runtime_observer_identity
        )
        if (
            expected_segment_id is not None
            and _hex64(expected_segment_id) != segment_id
        ) or (
            expected_runtime_observer_identity is not None
            and _hash_value(expected_runtime_observer_identity)
            != observer_identity
        ):
            _fail()
    receipt = runtime.load_canonical_json_bytes(raw, expected=dict)
    _exact_mapping(receipt, PROBE_ATTEMPT_FIELDS)
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or type(receipt["schema_version"]) is not int
        or receipt["namespace"] != NAMESPACE
        or receipt["receipt_kind"] != "probe-attempt"
        or receipt["segment_id"] != segment_id
    ):
        _fail()
    slot = runtime.slot_times(receipt["slot_index"])
    if (
        receipt["scheduled_at"] != slot.scheduled_at
        or receipt["grace_deadline_at"] != slot.grace_deadline_at
    ):
        _fail()
    launched = runtime.parse_utc(receipt["launched_at"], receipt=True)
    grace = runtime.parse_utc(slot.grace_deadline_at, receipt=True)
    watchdog = runtime.parse_utc(receipt["watchdog_deadline_at"], receipt=True)
    scheduled = runtime.parse_utc(slot.scheduled_at, receipt=True)
    latest_launch = grace - timedelta(seconds=60)
    if not scheduled <= launched < latest_launch:
        _fail()
    expected_watchdog = min(
        launched + timedelta(seconds=3600), latest_launch
    )
    if (
        watchdog != expected_watchdog
        or not launched < watchdog < grace
    ):
        _fail()
    ordinal = _nonnegative_int(receipt["attempt_ordinal"])
    if (
        slot.slot_index != _nonnegative_int(expected_slot_index)
        or ordinal != _nonnegative_int(expected_attempt_ordinal)
    ):
        _fail()
    expected_probe = (
        _self_identity()
        if expected_probe_identity is None
        else _hash_value(expected_probe_identity)
    )
    if (
        _hex64(receipt["previous_ledger_entry_sha256"])
        != _hex64(expected_previous_ledger_sha256)
        or runtime.HashAndBytes.from_value(
            receipt["runtime_observer_sha256_and_bytes"]
        )
        != observer_identity
        or runtime.HashAndBytes.from_value(receipt["probe_sha256_and_bytes"])
        != expected_probe
        or runtime.HashAndBytes.from_value(
            receipt["analysis_plan_sha256_and_bytes"]
        )
        != contract.plan_identity
    ):
        _fail()
    result = ValidatedProbeAttempt(
        raw=raw,
        identity=runtime.hash_and_bytes(raw),
        slot_index=slot.slot_index,
        attempt_ordinal=ordinal,
        launched_at=launched,
        watchdog_deadline_at=watchdog,
        previous_ledger_entry_sha256=receipt["previous_ledger_entry_sha256"],
        segment_id=segment_id,
        runtime_observer_identity=observer_identity,
        probe_identity=expected_probe,
        analysis_plan_identity=contract.plan_identity,
    )
    _require_probe_attempt_wrapper_consistent(
        result, contract, active_segment
    )
    return result


@dataclass(frozen=True, slots=True)
class ValidatedProbeFailure:
    raw: bytes
    identity: runtime.HashAndBytes
    attempt: ValidatedProbeAttempt
    failed_at: datetime
    phase_at_failure: str
    failure_class: str
    retry_authorized: bool
    previous_ledger_entry_sha256: str
    analysis_plan_identity: runtime.HashAndBytes


def _require_probe_failure_wrapper_consistent(
    failure: ValidatedProbeFailure,
    contract: runtime.FrozenContract,
    active_segment: runtime.ActiveSegment,
) -> None:
    if (
        type(failure) is not ValidatedProbeFailure
        or type(failure.raw) is not bytes
        or type(failure.identity) is not runtime.HashAndBytes
        or type(failure.attempt) is not ValidatedProbeAttempt
        or type(failure.failed_at) is not datetime
        or failure.failed_at.tzinfo is not UTC
        or type(failure.phase_at_failure) is not str
        or type(failure.failure_class) is not str
        or type(failure.retry_authorized) is not bool
        or type(failure.previous_ledger_entry_sha256) is not str
        or type(failure.analysis_plan_identity) is not runtime.HashAndBytes
    ):
        _fail()
    _require_probe_attempt_wrapper_consistent(
        failure.attempt, contract, active_segment
    )
    rebound = validate_probe_failure_marker(
        failure.raw,
        contract=contract,
        active_segment=active_segment,
        attempt=failure.attempt,
        expected_previous_ledger_sha256=failure.previous_ledger_entry_sha256,
    )
    if rebound != failure:
        _fail()


def validate_probe_failure_marker(
    raw: bytes,
    *,
    contract: runtime.FrozenContract,
    active_segment: runtime.ActiveSegment,
    attempt: ValidatedProbeAttempt,
    expected_previous_ledger_sha256: str,
) -> ValidatedProbeFailure:
    _require_probe_attempt_wrapper_consistent(attempt, contract, active_segment)
    receipt = runtime.load_canonical_json_bytes(raw, expected=dict)
    _exact_mapping(receipt, PROBE_FAILURE_FIELDS)
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or type(receipt["schema_version"]) is not int
        or receipt["namespace"] != NAMESPACE
        or receipt["receipt_kind"] != "probe-failure"
        or receipt["slot_index"] != attempt.slot_index
        or receipt["attempt_ordinal"] != attempt.attempt_ordinal
        or runtime.HashAndBytes.from_value(
            receipt["probe_attempt_marker_sha256_and_bytes"]
        )
        != attempt.identity
        or _hex64(receipt["previous_ledger_entry_sha256"])
        != _hex64(expected_previous_ledger_sha256)
        or runtime.HashAndBytes.from_value(
            receipt["analysis_plan_sha256_and_bytes"]
        )
        != contract.plan_identity
        or type(receipt["retry_authorized"]) is not bool
        or type(receipt["controller_synthesized"]) is not bool
    ):
        _fail()
    failed_at = runtime.parse_utc(receipt["failed_at"], receipt=True)
    if not attempt.launched_at <= failed_at <= attempt.watchdog_deadline_at:
        _fail()
    phase = receipt["phase_at_failure"]
    failure_class = receipt["failure_class"]
    if type(phase) is not str or type(failure_class) is not str:
        _fail()
    if phase == "unknown-after-attempt":
        if (
            receipt["source_open_count_or_null"] is not None
            or receipt["key_created_or_null"] is not None
            or receipt["controller_synthesized"] is not True
            or failure_class != "post-source-terminal"
            or receipt["retry_authorized"] is not False
            or failed_at != attempt.watchdog_deadline_at
        ):
            _fail()
    elif phase in PHASE_PROGRESS:
        source_count, key_created = PHASE_PROGRESS[phase]
        if (
            receipt["source_open_count_or_null"] != source_count
            or type(receipt["source_open_count_or_null"]) is not int
            or receipt["key_created_or_null"] is not key_created
        ):
            _fail()
        retryable = phase == "pre-key-pre-source"
        if (
            failure_class
            != ("pre-source-retryable" if retryable else "post-source-terminal")
            or receipt["retry_authorized"] is not retryable
            or (retryable and receipt["controller_synthesized"] is not False)
        ):
            _fail()
    else:
        _fail()
    return ValidatedProbeFailure(
        raw=raw,
        identity=runtime.hash_and_bytes(raw),
        attempt=attempt,
        failed_at=failed_at,
        phase_at_failure=phase,
        failure_class=failure_class,
        retry_authorized=receipt["retry_authorized"],
        previous_ledger_entry_sha256=receipt[
            "previous_ledger_entry_sha256"
        ],
        analysis_plan_identity=contract.plan_identity,
    )


def validate_probe_terminal_marker(
    raw: bytes,
    *,
    contract: runtime.FrozenContract,
    active_segment: runtime.ActiveSegment,
    failure: ValidatedProbeFailure,
    expected_previous_ledger_sha256: str,
) -> runtime.HashAndBytes:
    _require_probe_failure_wrapper_consistent(
        failure, contract, active_segment
    )
    if failure.failure_class != "post-source-terminal" or failure.retry_authorized:
        _fail()
    receipt = runtime.load_canonical_json_bytes(raw, expected=dict)
    _exact_mapping(receipt, PROBE_TERMINAL_FIELDS)
    recorded_at = runtime.parse_utc(receipt["recorded_at"], receipt=True)
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or type(receipt["schema_version"]) is not int
        or receipt["namespace"] != NAMESPACE
        or receipt["receipt_kind"] != "probe-terminal"
        or receipt["status"] != "terminal-probe-integrity-failure"
        or receipt["slot_index"] != failure.attempt.slot_index
        or receipt["attempt_ordinal"] != failure.attempt.attempt_ordinal
        or runtime.HashAndBytes.from_value(
            receipt["probe_failure_marker_sha256_and_bytes"]
        )
        != failure.identity
        or recorded_at < failure.failed_at
        or _hex64(receipt["previous_ledger_entry_sha256"])
        != _hex64(expected_previous_ledger_sha256)
        or runtime.HashAndBytes.from_value(
            receipt["analysis_plan_sha256_and_bytes"]
        )
        != contract.plan_identity
    ):
        _fail()
    return runtime.hash_and_bytes(raw)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "__worker":
        return _worker_entry(arguments[1:])
    # The frozen protocol defines no standalone serialized ActiveSegment
    # request envelope.  Integration therefore uses the typed API above rather
    # than an unsafe path-only CLI.  Misuse is deliberately silent.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
