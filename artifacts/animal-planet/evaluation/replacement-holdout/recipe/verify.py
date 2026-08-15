#!/usr/bin/env python3
"""Independent two-phase verifier for the replacement P6 supplement.

The keyed phase is invoked by ``build.py freeze`` while its single ephemeral
identity key is still live.  It independently reads the pinned raw sources,
recomputes every dev and supplement identity, and writes only an aggregate
receipt to an inherited descriptor.  The sealed phase needs no raw sources or
key; it validates the hash-bound, frozen packet mechanically.

This file intentionally imports neither the builder nor its de-identification
helpers.  Failures never include paths, ids, queries, tokens, keys, or private
diagnostics.
"""

from __future__ import annotations

import argparse
import base64
import collections
import fcntl
import hashlib
import hmac
import json
import math
import os
import re
import resource
import sqlite3
import stat
import sys
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


START_EXCLUSIVE = "2026-08-12T23:13:24Z"
END_EXCLUSIVE = "2026-08-13T20:16:51Z"
REQUESTED_SCOPES = ("project:ae", "project:online")
PACKET_SCOPES = frozenset({"global", "project:ae", "project:online", "project:x"})

METADATA_SHA256 = "025e734f84f452711e86a31d1639f2ee9833213e7ec94ced69b95d315ebac855"
SNAPSHOT_SHA256 = "4d6648f9e3c33e8ffaa7bf15620bd26a18bdf9f7ddf5b97641aa8c082e9aaa67"
STAGING_EVENTS_SHA256 = "45dbfdb389324a8702ba1a84b28be028991f23dcfe1d7ab9d771928c37c2cc7a"
ORIGINAL_MANIFEST_SHA256 = "3f1a6a87a4d34e411f06e50161f9203afc3c4992dfc973d4f544453e8bbfde48"
SPLITS_SHA256 = "66f04caa41e0302fd7b8582f54197a396a2bb8483ddd543e564740a8e4c3a7b7"
DEID_SHA256 = "503de08cea164660c2f1c3b3675ad082f70109d8f4360194bb1e02c7423bd6e3"
README_SHA256 = "c0483824717fb3323600b94db945c4ed8718e23749863f493f3013ecd2fae409"
POLICY_SHA256 = "05d4f3d148b86a933c107341ac1157487509efe011428dc4858c18b9b1c28a90"
BUILDER_SHA256 = "a2f420bec70e0ec7a87e0a99ddd9d030b9131e72c27b3b7facc622f6bcf4cd65"
README_BYTES = 11_153
POLICY_BYTES = 6_958
BUILDER_BYTES = 98_378

STAGING_EVENTS_BYTES = 4_162_914
STAGING_EVENTS_ROWS = 1_576
DEV_HOLDOUT_BOUND = 15
DEV_BOUND = 66
SPLIT_MOD = 100

EXPECTED_SUPPLEMENT_EVENTS = 769
EXPECTED_AUTOMATIC_EVENTS = 332
EXPECTED_ORGANIC_EVENTS = 437
EXPECTED_SCOPE_COUNTS = {"project:ae": 502, "project:online": 267}
EXPECTED_DIRECT_NODES = 619
EXPECTED_NODES = 804
EXPECTED_DEV_EVENTS = 826
EXPECTED_DEV_AUTOMATIC = 732
EXPECTED_DEV_ORGANIC = 94
EXPECTED_DEV_UNIQUE_TOKENS = 353
EXPECTED_REPEATED_FAMILIES = 18
EXPECTED_REPEATED_EVENTS = 238
EXPECTED_UNSEEN_FAMILIES = 16
EXPECTED_UNSEEN_EVENTS = 113

IDENTITY_KEY_BYTES = 32
TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")
PURITY_RE = re.compile(r"^[a-z \t\n\r]*$")
KEEP_WS = frozenset(" \t\n\r")
ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
HEXID_RE = re.compile(r"^(?=.*[0-9])[0-9a-f]{7,64}$")
ISO_TS_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$"
)
UTC_INSTANT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
SCOPE_RE = re.compile(r"^(global|project:[a-z0-9_-]+)$")
STRUCTURAL_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,63}$")
NUMERIC_RE = re.compile(r"^[0-9]+([.][0-9]+)?$")
TEMPORAL_HINT_RE = re.compile(r"^[a-z]+(:[a-z0-9_-]+)?$")
FLAG_LITERALS = frozenset({"true", "false", "yes", "no"})
LEVELS = frozenset({"trace", "concept", "schema"})
RESULT_METHODS = frozenset({"bm25", "vector", "graph", "trigger"})
RESULT_METHOD_ORDER = {name: position for position, name in enumerate(("bm25", "vector", "graph", "trigger"))}
DEPTH_VALUES = frozenset({"1", "2", "3", "4", "shallow", "causal", "full"})
TEMPORAL_HINT_VALUES = frozenset({f"weekly:{day}" for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")})
SUPPLEMENT_STRUCTURAL_KEYS = frozenset({
    "api", "chat_analyzed", "cli", "commit_removed", "commits_absorbed",
    "confluence_version", "correct_commit", "delivered", "diff_version_id",
    "discussion_id", "evidence_discussions", "evidence_file", "finding_kind",
    "fingerprint_token", "hosts", "identifiers", "incorrect_commit",
    "jira_comment", "jira_issue", "lines", "mattermost_post",
    "misleading_ticket", "notes_total", "open_threads", "overall_mean_sm_pct",
    "pipeline_status", "purpose", "pushed", "related_node", "related_posts",
    "resolvable_threads", "revision", "saved_patch", "session_outcome",
    "source_files", "stages_strict_fail", "step_description", "step_order",
    "summary_note", "target_sha", "unresolved_threads", "user_intent",
    "verification_kind", "wall_time_seconds",
})

RECALL_EVENT_COLUMNS = (
    "id", "query", "scope", "requested_scope", "resolved_scopes",
    "ambient_context", "depth", "max_results", "results", "agent", "task",
    "session_id", "feedback_applied", "feedback_trace_id",
    "feedback_applied_at", "created_at", "transport_session_id",
)
NODE_COLUMNS = (
    "id", "level", "content", "scope", "agent", "task", "context",
    "timestamp", "decayed", "decay_reason", "access_count", "last_accessed",
    "usefulness_score", "confidence", "unique_agents", "temporal_hint",
    "source_traces", "corrections", "provenance", "created_at", "updated_at",
)
RESULT_FIELDS = (
    "node_id", "level", "scope", "rank", "score", "bm25_score",
    "vector_score", "graph_score", "trigger_score", "methods", "path",
)
EVENT_FIELDS = (
    "type", "id", "source", "created_at", "scope", "requested_scope",
    "resolved_scopes", "depth", "max_results", "class", "template_id",
    "fingerprint_token", "query_surrogate", "query_chars", "agent_surrogate",
    "agent_chars", "task_surrogate", "task_chars", "session_id_surrogate",
    "session_id_chars", "transport_session_id", "ambient_context_surrogate",
    "ambient_context_chars", "results", "feedback_applied",
    "feedback_trace_id", "feedback_applied_at", "supersedes_bearing",
    "transcript_matched", "transcript_serialized_chars",
)
NODE_FIELDS = (
    "type", "id", "level", "scope", "included_via", "agent_surrogate",
    "agent_chars", "task_surrogate", "task_chars", "timestamp", "created_at",
    "updated_at", "decayed", "decay_reason_surrogate", "decay_reason_chars",
    "stats", "content_surrogate", "content_chars", "context_surrogate",
    "context_chars", "source_traces", "corrections", "corrections_chars",
    "provenance_shape", "relations",
)
STATS_FIELDS = (
    "access_count", "usefulness_score", "confidence", "unique_agents",
    "last_accessed", "temporal_hint",
)
RELATION_FIELDS = ("type", "direction", "other_id", "weight", "created_at")
INDEX_FIELDS = (
    "schema_version", "kind", "algorithm", "normalization",
    "population_events", "unique_tokens", "tokens",
)
MANIFEST_FIELDS = (
    "schema_version", "packet", "namespace", "frozen", "generated_at",
    "semantic_reads", "selection", "classification", "identity",
    "deidentification", "repeated_family", "sources", "implementation",
    "counts", "files", "packet_files", "validation", "publication",
)
RECEIPT = {
    "schema_version": 1,
    "mode": "keyed-preseal",
    "status": "pass",
    "mismatches": 0,
    "token_collisions": 0,
    "semantic_reads": 0,
    "supplement_events": EXPECTED_SUPPLEMENT_EVENTS,
    "automatic_events": EXPECTED_AUTOMATIC_EVENTS,
    "organic_events": EXPECTED_ORGANIC_EVENTS,
    "project_ae_events": EXPECTED_SCOPE_COUNTS["project:ae"],
    "project_online_events": EXPECTED_SCOPE_COUNTS["project:online"],
    "direct_nodes": EXPECTED_DIRECT_NODES,
    "nodes": EXPECTED_NODES,
    "dev_automatic_events": EXPECTED_DEV_AUTOMATIC,
    "dev_unique_tokens": EXPECTED_DEV_UNIQUE_TOKENS,
    "repeated_automatic_families": EXPECTED_REPEATED_FAMILIES,
    "repeated_automatic_events": EXPECTED_REPEATED_EVENTS,
    "unseen_in_dev_families": EXPECTED_UNSEEN_FAMILIES,
    "unseen_in_dev_events": EXPECTED_UNSEEN_EVENTS,
}


class VerificationError(RuntimeError):
    """Fail-closed error carrying only a public aggregate category."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def fail(code: str) -> None:
    raise VerificationError(code)


def _disable_core_dumps() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError):
        pass


def _safe_int(value: Any) -> bool:
    return type(value) is int


def _safe_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _safe_float(value: Any) -> bool:
    return type(value) is float and math.isfinite(value)


def _is_ulid(value: Any) -> bool:
    return isinstance(value, str) and ULID_RE.fullmatch(value) is not None


def _is_scope(value: Any) -> bool:
    return isinstance(value, str) and SCOPE_RE.fullmatch(value) is not None


def _is_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or ISO_TS_RE.fullmatch(value) is None:
        return False
    try:
        if "T" in value or " " in value:
            datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
        else:
            date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _is_utc_instant(value: Any) -> bool:
    return (
        isinstance(value, str)
        and UTC_INSTANT_RE.fullmatch(value) is not None
        and _is_timestamp(value)
    )


def _instant(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        fail("schema")
    return parsed


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return tuple(actual) == tuple(expected) and all(
            _strict_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _normalized_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _path_inside(child: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath((os.fspath(child), os.fspath(parent))) == os.fspath(parent)
    except ValueError:
        return False


def _assert_regular(path: Path) -> os.stat_result:
    absolute = _normalized_path(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except OSError:
            fail("source")
        if stat.S_ISLNK(info.st_mode):
            fail("boundary")
    info = os.lstat(absolute)
    if not stat.S_ISREG(info.st_mode):
        fail("source")
    return info


def _assert_directory(path: Path) -> os.stat_result:
    absolute = _normalized_path(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except OSError:
            fail("boundary")
        if stat.S_ISLNK(info.st_mode):
            fail("boundary")
    info = os.lstat(absolute)
    if not stat.S_ISDIR(info.st_mode):
        fail("boundary")
    return info


def _read_regular(path: Path, *, maximum: int | None = None) -> bytes:
    before = _assert_regular(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        fail("source")
    try:
        bound = os.fstat(fd)
        if (bound.st_dev, bound.st_ino) != (before.st_dev, before.st_ino):
            fail("source_changed")
        chunks: list[bytes] = []
        total = 0
        while True:
            try:
                chunk = os.read(fd, 1024 * 1024)
            except OSError:
                fail("source")
            if not chunk:
                break
            total += len(chunk)
            if maximum is not None and total > maximum:
                fail("source_size")
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    final = os.lstat(path)
    signatures = (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        (bound.st_dev, bound.st_ino, bound.st_size, bound.st_mtime_ns),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns),
    )
    if len(set(signatures)) != 1:
        fail("source_changed")
    return b"".join(chunks)


def _read_at(parent_fd: int, name: str, *, maximum: int | None = None) -> bytes:
    if not name or "/" in name or name in (".", ".."):
        fail("boundary")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError:
        fail("boundary")
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            fail("boundary")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if maximum is not None and total > maximum:
                fail("source_size")
            chunks.append(chunk)
        after = os.fstat(fd)
        linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            stat.S_ISLNK(linked.st_mode)
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (linked.st_dev, linked.st_ino, linked.st_size, linked.st_mtime_ns)
        ):
            fail("candidate_changed")
        return b"".join(chunks)
    except OSError:
        fail("boundary")
    finally:
        os.close(fd)


def _read_keyed_candidate(draft_fd: int) -> tuple[bytes, bytes]:
    try:
        if sorted(os.listdir(draft_fd)) != ["corpus"]:
            fail("extra_file")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        corpus_fd = os.open("corpus", flags, dir_fd=draft_fd)
    except VerificationError:
        raise
    except OSError:
        fail("boundary")
    try:
        corpus_info = os.fstat(corpus_fd)
        linked = os.stat("corpus", dir_fd=draft_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(corpus_info.st_mode)
            or stat.S_ISLNK(linked.st_mode)
            or (corpus_info.st_dev, corpus_info.st_ino) != (linked.st_dev, linked.st_ino)
            or sorted(os.listdir(corpus_fd))
            != ["dev-fingerprint-index.json", "holdout.jsonl"]
        ):
            fail("boundary")
        corpus = _read_at(corpus_fd, "holdout.jsonl")
        index = _read_at(corpus_fd, "dev-fingerprint-index.json")
        if sorted(os.listdir(corpus_fd)) != ["dev-fingerprint-index.json", "holdout.jsonl"]:
            fail("candidate_changed")
        rebound = os.stat("corpus", dir_fd=draft_fd, follow_symlinks=False)
        if (rebound.st_dev, rebound.st_ino) != (corpus_info.st_dev, corpus_info.st_ino):
            fail("candidate_changed")
        return corpus, index
    finally:
        os.close(corpus_fd)


def _open_dir_at(parent_fd: int, name: str) -> int:
    if not name or "/" in name or name in (".", ".."):
        fail("boundary")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
        linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        bound = os.fstat(fd)
    except OSError:
        fail("boundary")
    if (
        stat.S_ISLNK(linked.st_mode)
        or not stat.S_ISDIR(bound.st_mode)
        or (linked.st_dev, linked.st_ino) != (bound.st_dev, bound.st_ino)
    ):
        os.close(fd)
        fail("boundary")
    return fd


def _read_packet_files(packet_fd: int) -> tuple[dict[str, bytes], tuple[tuple[str, tuple[str, ...]], ...]]:
    expected_root = ["POLICY.md", "README.md", "corpus", "manifest.json", "recipe"]
    try:
        if sorted(os.listdir(packet_fd)) != expected_root:
            fail("extra_file")
        corpus_fd = _open_dir_at(packet_fd, "corpus")
        recipe_fd = _open_dir_at(packet_fd, "recipe")
    except VerificationError:
        raise
    except OSError:
        fail("boundary")
    try:
        if sorted(os.listdir(corpus_fd)) != ["dev-fingerprint-index.json", "holdout.jsonl"]:
            fail("extra_file")
        if sorted(os.listdir(recipe_fd)) != ["build.py", "verify.py"]:
            fail("extra_file")
        blobs = {
            "README.md": _read_at(packet_fd, "README.md"),
            "POLICY.md": _read_at(packet_fd, "POLICY.md"),
            "manifest.json": _read_at(packet_fd, "manifest.json"),
            "corpus/holdout.jsonl": _read_at(corpus_fd, "holdout.jsonl"),
            "corpus/dev-fingerprint-index.json": _read_at(corpus_fd, "dev-fingerprint-index.json"),
            "recipe/build.py": _read_at(recipe_fd, "build.py"),
            "recipe/verify.py": _read_at(recipe_fd, "verify.py"),
        }
        tree = (
            ("root", tuple(sorted(os.listdir(packet_fd)))),
            ("corpus", tuple(sorted(os.listdir(corpus_fd)))),
            ("recipe", tuple(sorted(os.listdir(recipe_fd)))),
        )
        return blobs, tree
    finally:
        os.close(corpus_fd)
        os.close(recipe_fd)


def _sha(path: Path) -> tuple[str, int]:
    raw = _read_regular(path)
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _require_sha(path: Path, expected: str) -> tuple[str, int]:
    digest, size = _sha(path)
    if not hmac.compare_digest(digest, expected):
        fail("source_pin")
    return digest, size


def _pairs_no_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("schema")
        result[key] = value
    return result


def _json(raw: bytes, *, expected: type | None = None) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda _value: fail("schema"),
        )
    except VerificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError):
        fail("schema")
    if expected is not None and type(value) is not expected:
        fail("schema")
    return value


def _json_column(raw: Any, expected: type) -> Any:
    if not isinstance(raw, str):
        fail("source_schema")
    value = _json(raw.encode("utf-8"))
    if type(value) is not expected:
        fail("source_schema")
    return value


def _canonical_line(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _canonical_document(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False) + "\n").encode("utf-8")


def _classification(agent: Any) -> str:
    return "automatic" if agent is None else "organic"


def _template(query: str) -> str | None:
    for value in ("reopen_lesson", "architectural_decision"):
        if value in query:
            return value
    return None


def _bucket(event_id: str) -> int:
    digest = hashlib.sha256(event_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % SPLIT_MOD


def _identity(query: str, requested_scope: str) -> tuple[str, str, bytes]:
    normalized = " ".join(query.split())
    return normalized, requested_scope, (normalized + "\n" + requested_scope).encode("utf-8")


def _token(key: bytes | bytearray, query: str, requested_scope: str) -> tuple[str, tuple[str, str], bytes]:
    normalized, scope, message = _identity(query, requested_scope)
    return hmac.new(key, message, hashlib.sha256).hexdigest(), (normalized, scope), message


def _validate_result(raw: Any) -> dict[str, Any]:
    if type(raw) is not dict or tuple(raw) != RESULT_FIELDS:
        fail("schema")
    if (
        not _is_ulid(raw["node_id"])
        or raw["level"] not in LEVELS
        or not _is_scope(raw["scope"])
    ):
        fail("schema")
    if not _safe_int(raw["rank"]) or raw["rank"] < 0:
        fail("schema")
    for name in ("score", "bm25_score", "vector_score", "graph_score", "trigger_score"):
        if not _safe_float(raw[name]):
            fail("schema")
    if (
        type(raw["methods"]) is not list
        or not all(type(x) is str and x in RESULT_METHODS for x in raw["methods"])
        or len(raw["methods"]) != len(set(raw["methods"]))
    ):
        fail("schema")
    if type(raw["path"]) is not list or not all(_is_ulid(x) for x in raw["path"]):
        fail("schema")
    return raw


def _load_staging(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = _read_regular(path, maximum=STAGING_EVENTS_BYTES)
    if len(raw) != STAGING_EVENTS_BYTES or hashlib.sha256(raw).hexdigest() != STAGING_EVENTS_SHA256:
        fail("source_pin")
    if not raw.endswith(b"\n"):
        fail("source_schema")
    lines = raw.splitlines()
    if len(lines) != STAGING_EVENTS_ROWS:
        fail("source_schema")
    rows: list[dict[str, Any]] = []
    for line in lines:
        row = _json(line, expected=dict)
        if set(row) != set(RECALL_EVENT_COLUMNS):
            fail("source_schema")
        rows.append(row)
    if len({row.get("id") for row in rows}) != len(rows):
        fail("source_schema")
    return rows, raw


def _collect_structural_keys(value: Any, result: set[str]) -> None:
    if type(value) is dict:
        result.update(key for key in value if isinstance(key, str))
        for item in value.values():
            _collect_structural_keys(item, result)
    elif type(value) is list:
        for item in value:
            _collect_structural_keys(item, result)


def _load_original_packet(original_manifest_path: Path) -> tuple[dict[str, Any], set[str], frozenset[str]]:
    raw = _read_regular(original_manifest_path)
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_MANIFEST_SHA256:
        fail("original_packet")
    manifest = _json(raw, expected=dict)
    if manifest.get("frozen") is not True:
        fail("original_packet")
    root = _normalized_path(original_manifest_path).parent
    files = manifest.get("files")
    packet_files = manifest.get("packet_files")
    if type(files) is not dict or type(packet_files) is not dict:
        fail("original_packet")
    verified: dict[str, bytes] = {}
    for relative, metadata in {**files, **packet_files}.items():
        if type(relative) is not str or type(metadata) is not dict:
            fail("original_packet")
        target = _normalized_path(root / relative)
        if not _path_inside(target, root):
            fail("original_packet")
        data = _read_regular(target)
        if hashlib.sha256(data).hexdigest() != metadata.get("sha256") or len(data) != metadata.get("bytes"):
            fail("original_packet")
        verified[relative] = data
    split_meta = files.get("corpus/splits.json")
    deid_meta = packet_files.get("recipe/transform/deid.py")
    if (
        type(split_meta) is not dict
        or split_meta.get("sha256") != SPLITS_SHA256
        or type(deid_meta) is not dict
        or deid_meta.get("sha256") != DEID_SHA256
    ):
        fail("original_packet")
    event_ids: set[str] = set()
    structural_keys: set[str] = set()
    for relative in ("corpus/dev.jsonl", "corpus/eval.jsonl", "corpus/holdout.jsonl"):
        if relative not in files:
            fail("original_packet")
        data = verified[relative]
        if not data.endswith(b"\n"):
            fail("original_packet")
        for line in data.splitlines():
            record = _json(line, expected=dict)
            _collect_structural_keys(record, structural_keys)
            if record.get("type") == "event":
                event_id = record.get("id")
                if not isinstance(event_id, str) or event_id in event_ids:
                    fail("original_packet")
                event_ids.add(event_id)
    if len(event_ids) != 6_076:
        fail("original_packet")
    if not structural_keys:
        fail("original_packet")
    return manifest, event_ids, frozenset(structural_keys)


def _source_pins(staging: Path, snapshot: Path, original_manifest: Path, splits: Path) -> dict[str, Any]:
    staging = _normalized_path(staging)
    metadata_path = staging / "METADATA.json"
    events_path = staging / "alt-db" / "recall_events.jsonl"
    original_root = _normalized_path(original_manifest).parent
    if any(
        _path_inside(snapshot, candidate) or _path_inside(candidate, snapshot)
        for candidate in (staging, original_root, splits)
    ):
        fail("boundary")
    metadata_raw = _read_regular(metadata_path)
    events_raw = _read_regular(events_path, maximum=STAGING_EVENTS_BYTES)
    snapshot_digest, snapshot_bytes = _require_sha(snapshot, SNAPSHOT_SHA256)
    manifest_digest, manifest_bytes = _require_sha(original_manifest, ORIGINAL_MANIFEST_SHA256)
    splits_raw = _read_regular(splits)
    splits_digest = hashlib.sha256(splits_raw).hexdigest()
    splits_bytes = len(splits_raw)
    if not hmac.compare_digest(splits_digest, SPLITS_SHA256):
        fail("source_pin")
    if hashlib.sha256(metadata_raw).hexdigest() != METADATA_SHA256:
        fail("source_pin")
    if hashlib.sha256(events_raw).hexdigest() != STAGING_EVENTS_SHA256 or len(events_raw) != STAGING_EVENTS_BYTES:
        fail("source_pin")
    metadata = _json(metadata_raw, expected=dict)
    try:
        export = metadata["exports"]["alt-db/recall_events.jsonl"]
    except (KeyError, TypeError):
        fail("source_pin")
    if export != {"sha256": STAGING_EVENTS_SHA256, "bytes": STAGING_EVENTS_BYTES, "rows": STAGING_EVENTS_ROWS}:
        fail("source_pin")
    splits_doc = _json(splits_raw, expected=dict)
    try:
        rule = splits_doc["rule"]
        dev = splits_doc["counts"]["dev"]
    except (KeyError, TypeError):
        fail("source_pin")
    if (
        splits_doc.get("staging_metadata_sha256") != METADATA_SHA256
        or rule.get("mod") != SPLIT_MOD
        or rule.get("holdout_bound") != DEV_HOLDOUT_BOUND
        or rule.get("dev_bound") != DEV_BOUND
        or dev.get("events") != EXPECTED_DEV_EVENTS
        or dev.get("by_class") != {"automatic": EXPECTED_DEV_AUTOMATIC, "organic": EXPECTED_DEV_ORGANIC}
    ):
        fail("source_pin")
    wal = Path(os.fspath(snapshot) + "-wal")
    if wal.exists():
        info = os.lstat(wal)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size != 0:
            fail("source_pin")
    return {
        "metadata": (METADATA_SHA256, len(metadata_raw)),
        "events": (STAGING_EVENTS_SHA256, len(events_raw)),
        "snapshot": (snapshot_digest, snapshot_bytes),
        "manifest": (manifest_digest, manifest_bytes),
        "splits": (splits_digest, splits_bytes),
    }


def _select_snapshot(snapshot: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    before = _assert_regular(snapshot)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(snapshot, flags)
    except OSError:
        fail("snapshot")
    connection: sqlite3.Connection | None = None
    try:
        bound = os.fstat(fd)
        if (bound.st_dev, bound.st_ino) != (before.st_dev, before.st_ino):
            fail("source_changed")
        snapshot_hash = hashlib.sha256()
        snapshot_size = 0
        offset = 0
        while True:
            chunk = os.pread(fd, 1024 * 1024, offset)
            if not chunk:
                break
            snapshot_hash.update(chunk)
            snapshot_size += len(chunk)
            offset += len(chunk)
        if (
            snapshot_size != bound.st_size
            or not hmac.compare_digest(snapshot_hash.hexdigest(), SNAPSHOT_SHA256)
        ):
            fail("source_pin")
        uri = f"file:/proc/self/fd/{fd}?mode=ro&immutable=1&cache=private"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        quick = connection.execute("PRAGMA quick_check").fetchone()
        if quick is None or quick[0] != "ok":
            fail("snapshot")
        event_columns = {row[1] for row in connection.execute("PRAGMA table_info(recall_events)")}
        node_columns = {row[1] for row in connection.execute("PRAGMA table_info(nodes)")}
        if not set(RECALL_EVENT_COLUMNS).issubset(event_columns) or not set(NODE_COLUMNS).issubset(node_columns):
            fail("source_schema")
        sql_rows = connection.execute(
            f"SELECT {', '.join(RECALL_EVENT_COLUMNS)} FROM recall_events "
            "WHERE created_at > ? AND created_at < ? AND requested_scope IN (?, ?) "
            "ORDER BY created_at ASC, id ASC",
            (START_EXCLUSIVE, END_EXCLUSIVE, *REQUESTED_SCOPES),
        ).fetchall()
        events: list[dict[str, Any]] = []
        direct: set[str] = set()
        for sql_row in sql_rows:
            row = {name: sql_row[name] for name in RECALL_EVENT_COLUMNS}
            if not isinstance(row["id"], str) or not isinstance(row["query"], str):
                fail("source_schema")
            if row["requested_scope"] not in REQUESTED_SCOPES or not (START_EXCLUSIVE < row["created_at"] < END_EXCLUSIVE):
                fail("selection")
            row["resolved_scopes"] = _json_column(row["resolved_scopes"], list)
            row["ambient_context"] = _json_column(row["ambient_context"], dict)
            results = _json_column(row["results"], list)
            if not all(type(item) is dict for item in results):
                fail("source_schema")
            # Raw key order is irrelevant; output order is independently fixed.
            row["results"] = [{name: item[name] for name in RESULT_FIELDS} for item in results]
            for result in row["results"]:
                _validate_result(result)
                direct.add(result["node_id"])
            if row["feedback_trace_id"] is not None:
                if not isinstance(row["feedback_trace_id"], str):
                    fail("source_schema")
                direct.add(row["feedback_trace_id"])
            events.append(row)
        if len(direct) != EXPECTED_DIRECT_NODES:
            fail("containment")
        direct_list = sorted(direct)
        placeholders = ",".join("?" for _ in direct_list)
        partner_rows = connection.execute(
            f"SELECT source_id,target_id FROM connections WHERE type IN ('supersedes','contradicts') "
            f"AND (source_id IN ({placeholders}) OR target_id IN ({placeholders}))",
            [*direct_list, *direct_list],
        ).fetchall()
        node_ids = set(direct)
        for row in partner_rows:
            node_ids.update((row["source_id"], row["target_id"]))
        ordered_ids = sorted(node_ids)
        placeholders = ",".join("?" for _ in ordered_ids)
        node_rows = connection.execute(
            f"SELECT {', '.join(NODE_COLUMNS)} FROM nodes WHERE id IN ({placeholders}) ORDER BY id ASC",
            ordered_ids,
        ).fetchall()
        nodes = {row["id"]: {name: row[name] for name in NODE_COLUMNS} for row in node_rows}
        if set(nodes) != node_ids:
            fail("containment")
        edge_rows = connection.execute(
            "SELECT source_id,target_id,type,weight,created_at FROM connections "
            f"WHERE type IN ('supersedes','contradicts') AND source_id IN ({placeholders}) "
            f"AND target_id IN ({placeholders}) ORDER BY type,source_id,target_id,created_at",
            [*ordered_ids, *ordered_ids],
        ).fetchall()
        edges = [dict(row) for row in edge_rows]
        after = os.fstat(fd)
        final_hash = hashlib.sha256()
        final_size = 0
        offset = 0
        while True:
            chunk = os.pread(fd, 1024 * 1024, offset)
            if not chunk:
                break
            final_hash.update(chunk)
            final_size += len(chunk)
            offset += len(chunk)
        if (
            final_size != snapshot_size
            or not hmac.compare_digest(final_hash.digest(), snapshot_hash.digest())
        ):
            fail("source_changed")
    except VerificationError:
        raise
    except (sqlite3.Error, KeyError, TypeError):
        fail("snapshot")
    finally:
        if connection is not None:
            connection.close()
        os.close(fd)
    final = os.lstat(snapshot)
    signatures = {
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        (bound.st_dev, bound.st_ino, bound.st_size, bound.st_mtime_ns),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns),
    }
    if len(signatures) != 1:
        fail("source_changed")
    return events, nodes, edges


def _relations(node_ids: set[str], edges: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    result: dict[str, list[dict[str, Any]]] = {node_id: [] for node_id in node_ids}
    supersedes: set[str] = set()
    for edge in edges:
        source = edge.get("source_id")
        target = edge.get("target_id")
        edge_type = edge.get("type")
        if source not in node_ids or target not in node_ids or edge_type not in ("supersedes", "contradicts"):
            fail("containment")
        if not isinstance(edge.get("created_at"), str) or not _safe_number(edge.get("weight")):
            fail("source_schema")
        result[source].append({
            "type": edge_type, "direction": "out", "other_id": target,
            "weight": edge["weight"], "created_at": edge["created_at"],
        })
        if target != source:
            result[target].append({
                "type": edge_type, "direction": "in", "other_id": source,
                "weight": edge["weight"], "created_at": edge["created_at"],
            })
        if edge_type == "supersedes":
            supersedes.update((source, target))
    for values in result.values():
        values.sort(key=lambda item: (item["type"], item["direction"], item["other_id"], item["created_at"]))
    return result, supersedes


def _known_scopes(events: Sequence[Mapping[str, Any]], nodes: Mapping[str, Mapping[str, Any]]) -> frozenset[str]:
    result = {"global"}
    for row in events:
        for value in (row.get("scope"), row.get("requested_scope")):
            if isinstance(value, str) and value:
                result.add(value)
        result.update(value for value in row["resolved_scopes"] if isinstance(value, str) and value)
        result.update(item["scope"] for item in row["results"] if isinstance(item["scope"], str) and item["scope"])
    for row in nodes.values():
        if isinstance(row.get("scope"), str) and row["scope"]:
            result.add(row["scope"])
    return frozenset(result)


def _keeps_real(value: str, scopes: frozenset[str]) -> bool:
    return bool(
        value in scopes or ULID_RE.fullmatch(value) or HEX32_RE.fullmatch(value)
        or UUID_RE.fullmatch(value) or HEXID_RE.fullmatch(value)
        or ISO_TS_RE.fullmatch(value) or NUMERIC_RE.fullmatch(value)
        or value in FLAG_LITERALS
    )


class PrivacyAudit:
    def __init__(self, scopes: frozenset[str], allowed_keys: frozenset[str]) -> None:
        self.scopes = scopes
        self.allowed_keys = allowed_keys
        self.original_to_surrogate: dict[str, str] = {}
        self.surrogate_to_original: dict[str, str] = {}
        self.private_originals: set[str] = set()

    def surrogate(self, original: Any, surrogate: Any, count: Any) -> None:
        if original is None:
            if surrogate is not None or count is not None:
                fail("privacy")
            return
        if not isinstance(original, str) or not isinstance(surrogate, str) or not _safe_int(count):
            fail("privacy")
        if count != len(original) or len(surrogate) != len(original) or not PURITY_RE.fullmatch(surrogate):
            fail("privacy")
        for left, right in zip(original, surrogate):
            if left in KEEP_WS:
                if left != right:
                    fail("privacy")
            elif right in KEEP_WS:
                fail("privacy")
        prior = self.original_to_surrogate.setdefault(original, surrogate)
        holder = self.surrogate_to_original.setdefault(surrogate, original)
        if prior != surrogate or holder != original:
            fail("privacy")
        self.private_originals.add(original)

    def tree(self, original: Any, surrogate: Any) -> None:
        if type(original) is dict:
            if type(surrogate) is not dict or tuple(surrogate) != tuple(original):
                fail("privacy")
            for key in original:
                if not isinstance(key, str) or key not in self.allowed_keys:
                    fail("privacy")
                self.tree(original[key], surrogate[key])
            return
        if type(original) is list:
            if type(surrogate) is not list or len(surrogate) != len(original):
                fail("privacy")
            for left, right in zip(original, surrogate):
                self.tree(left, right)
            return
        if isinstance(original, str):
            if _keeps_real(original, self.scopes):
                if surrogate != original:
                    fail("row_mismatch")
            else:
                self.surrogate(original, surrogate, len(original))
            return
        if type(surrogate) is not type(original) or surrogate != original:
            fail("row_mismatch")


def _delivery_chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _context_chars(value: Mapping[str, Any]) -> dict[str, int]:
    return {key: _delivery_chars(item) for key, item in value.items()}


def _provenance_shape(provenance: Mapping[str, Any], source_traces: list[Any], corrections: list[Any]) -> dict[str, Any]:
    full = {**provenance, "source_traces": source_traces, "corrections": corrections}
    return {
        "keys": {
            key: {
                "kind": type(value).__name__,
                **({"count": len(value)} if isinstance(value, (list, dict)) else {}),
                "chars": _delivery_chars(value),
            }
            for key, value in provenance.items()
        },
        "serialized_chars": _delivery_chars(full),
    }


def _parse_corpus(raw: bytes) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not raw or not raw.endswith(b"\n") or b"\r\n" in raw:
        fail("canonical")
    nodes: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    seen_event = False
    for line in raw.splitlines(keepends=True):
        record = _json(line[:-1], expected=dict)
        if _canonical_line(record) != line:
            fail("canonical")
        if record.get("type") == "node":
            if seen_event or tuple(record) != NODE_FIELDS:
                fail("schema")
            nodes.append(record)
        elif record.get("type") == "event":
            seen_event = True
            if tuple(record) != EVENT_FIELDS:
                fail("schema")
            events.append(record)
        else:
            fail("schema")
    if [row.get("id") for row in nodes] != sorted(row.get("id") for row in nodes):
        fail("ordering")
    if [(row.get("created_at"), row.get("id")) for row in events] != sorted((row.get("created_at"), row.get("id")) for row in events):
        fail("ordering")
    if len({row.get("id") for row in nodes}) != len(nodes) or len({row.get("id") for row in events}) != len(events):
        fail("membership")
    return nodes, events


def _parse_index(raw: bytes) -> dict[str, Any]:
    index = _json(raw, expected=dict)
    if tuple(index) != INDEX_FIELDS or _canonical_document(index) != raw:
        fail("canonical")
    if not _safe_int(index.get("schema_version")) or index.get("schema_version") != 1 or index.get("kind") != "frozen-dev-automatic-fingerprint-token-index":
        fail("schema")
    if index.get("algorithm") != "HMAC-SHA256" or index.get("normalization") != '" ".join(query.split()) + "\\n" + requested_scope':
        fail("schema")
    tokens = index.get("tokens")
    if type(tokens) is not list or not all(isinstance(value, str) and TOKEN_RE.fullmatch(value) for value in tokens):
        fail("token")
    if tokens != sorted(set(tokens)):
        fail("token")
    if (
        not _safe_int(index.get("population_events"))
        or not _safe_int(index.get("unique_tokens"))
        or index.get("population_events") != EXPECTED_DEV_AUTOMATIC
        or index.get("unique_tokens") != EXPECTED_DEV_UNIQUE_TOKENS
        or index.get("unique_tokens") != len(tokens)
    ):
        fail("aggregate")
    return index


def _audit_records(
    raw_nodes: Mapping[str, Mapping[str, Any]],
    raw_events: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    corpus_nodes: Sequence[Mapping[str, Any]],
    corpus_events: Sequence[Mapping[str, Any]],
    index: Mapping[str, Any],
    *,
    key: bytes | bytearray | None,
    original_event_ids: set[str] | None,
    allowed_keys: frozenset[str],
) -> tuple[dict[str, int], set[bytes], PrivacyAudit]:
    if len(raw_events) != EXPECTED_SUPPLEMENT_EVENTS or len(corpus_events) != EXPECTED_SUPPLEMENT_EVENTS:
        fail("aggregate")
    if len(raw_nodes) != EXPECTED_NODES or len(corpus_nodes) != EXPECTED_NODES:
        fail("aggregate")
    raw_event_by_id = {row["id"]: row for row in raw_events}
    corpus_event_by_id = {row.get("id"): row for row in corpus_events}
    if len(raw_event_by_id) != len(raw_events) or set(corpus_event_by_id) != set(raw_event_by_id):
        fail("membership")
    if original_event_ids is not None and set(corpus_event_by_id) & original_event_ids:
        fail("disjointness")
    if set(raw_nodes) != {row.get("id") for row in corpus_nodes}:
        fail("membership")
    relations, supersedes = _relations(set(raw_nodes), edges)
    scopes = _known_scopes(raw_events, raw_nodes)
    privacy = PrivacyAudit(scopes, allowed_keys)
    node_reasons: dict[str, set[str]] = collections.defaultdict(set)
    for row in raw_events:
        for item in row["results"]:
            node_reasons[item["node_id"]].add("results")
        if row["feedback_trace_id"]:
            node_reasons[row["feedback_trace_id"]].add("feedback")
    if not set(node_reasons).issubset(raw_nodes) or len(node_reasons) != EXPECTED_DIRECT_NODES:
        fail("containment")
    token_identities: dict[str, tuple[str, str]] = {}
    forbidden_digests: set[bytes] = set()

    for record in corpus_nodes:
        node_id = record["id"]
        raw = raw_nodes[node_id]
        expected_reasons = sorted(node_reasons.get(node_id, {"typed_edge_partner"}))
        if record["included_via"] != expected_reasons:
            fail("containment")
        if tuple(record["stats"]) != STATS_FIELDS or type(record["relations"]) is not list:
            fail("schema")
        if not _strict_equal(record["relations"], relations[node_id]) or any(tuple(item) != RELATION_FIELDS for item in record["relations"]):
            fail("containment")
        for name in ("level", "scope", "timestamp", "created_at", "updated_at", "decayed", "source_traces"):
            expected = raw[name]
            if name == "source_traces":
                expected = _json_column(expected, list)
            if not _strict_equal(record[name], expected):
                fail("row_mismatch")
        expected_stats = {
            "access_count": raw["access_count"],
            "usefulness_score": raw["usefulness_score"],
            "confidence": raw["confidence"],
            "unique_agents": raw["unique_agents"],
            "last_accessed": raw["last_accessed"],
        }
        got_stats = dict(record["stats"])
        hint = got_stats.pop("temporal_hint")
        if not _strict_equal(got_stats, expected_stats):
            fail("row_mismatch")
        raw_hint = raw["temporal_hint"]
        if raw_hint is None or (isinstance(raw_hint, str) and TEMPORAL_HINT_RE.fullmatch(raw_hint)):
            if hint != raw_hint:
                fail("row_mismatch")
        else:
            privacy.surrogate(raw_hint, hint, len(raw_hint))
        privacy.surrogate(raw["agent"], record["agent_surrogate"], record["agent_chars"])
        privacy.surrogate(raw["task"], record["task_surrogate"], record["task_chars"])
        privacy.surrogate(raw["decay_reason"], record["decay_reason_surrogate"], record["decay_reason_chars"])
        privacy.surrogate(raw["content"], record["content_surrogate"], record["content_chars"])
        context = _json_column(raw["context"], dict)
        privacy.tree(context, record["context_surrogate"])
        if not _strict_equal(record["context_chars"], _context_chars(context)):
            fail("privacy")
        source_traces = _json_column(raw["source_traces"], list)
        corrections = _json_column(raw["corrections"], list)
        provenance = _json_column(raw["provenance"], dict)
        if type(record["corrections"]) is not list or type(record["corrections_chars"]) is not list:
            fail("schema")
        if len(record["corrections"]) != len(corrections) or len(record["corrections_chars"]) != len(corrections):
            fail("schema")
        for original, surrogate, chars in zip(corrections, record["corrections"], record["corrections_chars"]):
            if type(original) is not dict or type(surrogate) is not dict or type(chars) is not dict or tuple(surrogate) != tuple(original):
                fail("schema")
            if any(not isinstance(name, str) or name not in privacy.allowed_keys for name in original):
                fail("privacy")
            if set(chars) != {name for name in ("old", "new") if isinstance(original.get(name), str)}:
                fail("privacy")
            for name, value in original.items():
                if name in ("old", "new") and isinstance(value, str):
                    privacy.surrogate(value, surrogate[name], chars[name])
                else:
                    privacy.tree(value, surrogate[name])
        if not _strict_equal(record["provenance_shape"], _provenance_shape(provenance, source_traces, corrections)):
            fail("privacy")
        if any(not isinstance(name, str) or name not in privacy.allowed_keys for name in provenance):
            fail("privacy")

    class_counts: collections.Counter[str] = collections.Counter()
    scope_counts: collections.Counter[str] = collections.Counter()
    automatic_groups: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for raw in raw_events:
        record = corpus_event_by_id[raw["id"]]
        for name in (
            "created_at", "scope", "requested_scope", "resolved_scopes", "depth",
            "max_results", "results", "feedback_applied", "feedback_trace_id",
            "feedback_applied_at",
        ):
            if not _strict_equal(record[name], raw[name]):
                fail("row_mismatch")
        if record["source"] != "local" or record["class"] != _classification(raw["agent"]):
            fail("classification")
        if record["template_id"] != _template(raw["query"]):
            fail("classification")
        if record["requested_scope"] not in REQUESTED_SCOPES or not (START_EXCLUSIVE < record["created_at"] < END_EXCLUSIVE):
            fail("selection")
        token_value = record["fingerprint_token"]
        if not isinstance(token_value, str) or not TOKEN_RE.fullmatch(token_value):
            fail("token")
        _normalized, scope, message = _identity(raw["query"], raw["requested_scope"])
        forbidden_digests.add(hashlib.sha256(message).digest())
        if key is not None:
            expected_token, identity_value, _message = _token(key, raw["query"], raw["requested_scope"])
            if not hmac.compare_digest(token_value, expected_token):
                fail("token_mismatch")
            holder = token_identities.setdefault(token_value, identity_value)
            if holder != identity_value:
                fail("token_collision")
        raw_results = raw["results"]
        if any(tuple(item) != RESULT_FIELDS for item in record["results"]):
            fail("schema")
        for item in record["results"]:
            _validate_result(item)
        references = {item["node_id"] for item in raw_results}
        if raw["feedback_trace_id"]:
            references.add(raw["feedback_trace_id"])
        if not references.issubset(raw_nodes):
            fail("containment")
        if record["supersedes_bearing"] is not bool({item["node_id"] for item in raw_results} & supersedes):
            fail("containment")
        if record["transcript_matched"] is not False or record["transcript_serialized_chars"] is not None:
            fail("schema")
        privacy.surrogate(raw["query"], record["query_surrogate"], record["query_chars"])
        privacy.surrogate(raw["agent"], record["agent_surrogate"], record["agent_chars"])
        privacy.surrogate(raw["task"], record["task_surrogate"], record["task_chars"])
        privacy.surrogate(raw["session_id"], record["session_id_surrogate"], record["session_id_chars"])
        transport = raw["transport_session_id"]
        if transport is None or (isinstance(transport, str) and HEX32_RE.fullmatch(transport)):
            if record["transport_session_id"] != transport:
                fail("row_mismatch")
        else:
            privacy.surrogate(transport, record["transport_session_id"], len(transport))
        privacy.tree(raw["ambient_context"], record["ambient_context_surrogate"])
        if not _strict_equal(record["ambient_context_chars"], _context_chars(raw["ambient_context"])):
            fail("privacy")
        class_counts[record["class"]] += 1
        scope_counts[record["requested_scope"]] += 1
        if record["class"] == "automatic":
            automatic_groups[token_value].append(record)

    if class_counts != {"automatic": EXPECTED_AUTOMATIC_EVENTS, "organic": EXPECTED_ORGANIC_EVENTS}:
        fail("aggregate")
    if scope_counts != EXPECTED_SCOPE_COUNTS:
        fail("aggregate")
    repeated = {
        token: rows for token, rows in automatic_groups.items()
        if len(rows) >= 3 and len({row["transport_session_id"] for row in rows}) >= 2
    }
    dev_tokens = set(index["tokens"])
    unseen = {token: rows for token, rows in repeated.items() if token not in dev_tokens}
    counts = {
        "repeated_families": len(repeated),
        "repeated_events": sum(len(rows) for rows in repeated.values()),
        "unseen_families": len(unseen),
        "unseen_events": sum(len(rows) for rows in unseen.values()),
    }
    if counts != {
        "repeated_families": EXPECTED_REPEATED_FAMILIES,
        "repeated_events": EXPECTED_REPEATED_EVENTS,
        "unseen_families": EXPECTED_UNSEEN_FAMILIES,
        "unseen_events": EXPECTED_UNSEEN_EVENTS,
    }:
        fail("aggregate")
    return counts, forbidden_digests, privacy


def _encoded_needles(values: Sequence[bytes]) -> set[bytes]:
    needles: set[bytes] = set()
    for value in values:
        needles.update((
            value,
            value.hex().encode("ascii"),
            value.hex().upper().encode("ascii"),
            base64.b64encode(value),
            base64.b64encode(value).rstrip(b"="),
            base64.urlsafe_b64encode(value),
            base64.urlsafe_b64encode(value).rstrip(b"="),
        ))
    return {needle for needle in needles if needle}


def _privacy_scan(
    blobs: Sequence[bytes],
    *,
    key: bytes | bytearray | None,
    unkeyed_digests: set[bytes],
    raw_private: set[str],
    raw_queries: set[str],
    surrogates: set[str],
) -> None:
    values = list(unkeyed_digests)
    if key is not None:
        values.append(bytes(key))
    needles = _encoded_needles(values)
    for blob in blobs:
        if any(needle in blob for needle in needles):
            fail("secret_material")
    for query in raw_queries:
        if query in ("reopen_lesson", "architectural_decision"):
            # These two declared template fingerprints are intentionally kept
            # real in ``template_id`` and therefore are not private plaintext.
            continue
        encoded = query.encode("utf-8")
        if encoded and any(encoded in blob for blob in blobs):
            fail("plaintext")
    # Exhaustive private strings of twelve or more characters are checked
    # against surrogate leaves.  Restricting the haystack to audited private
    # leaves avoids false positives in JSON field names and kept-real values;
    # exact schemas prevent an unaudited text channel.
    gram_hashes: set[int] = set()
    encoded_private: list[bytes] = []
    for value in raw_private:
        encoded = value.encode("utf-8")
        encoded_private.append(encoded)
        for offset in range(max(0, len(encoded) - 11)):
            gram = encoded[offset : offset + 12]
            if len(gram) == 12 and sum(byte not in b" \t\n\r" for byte in gram) >= 7:
                gram_hashes.add(hash(gram))
    for surrogate in surrogates:
        encoded = surrogate.encode("utf-8")
        for offset in range(max(0, len(encoded) - 11)):
            gram = encoded[offset : offset + 12]
            if len(gram) != 12 or hash(gram) not in gram_hashes:
                continue
            # Hashes are only an acceleration index.  Confirm the exact
            # n-gram before failing, so randomized hash collisions are benign.
            if any(gram in original for original in encoded_private):
                fail("plaintext")


def _assert_anonymous_pipe(fd: int, *, readable: bool) -> None:
    try:
        info = os.fstat(fd)
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        target = os.readlink(f"/proc/self/fd/{fd}")
    except (OSError, ValueError):
        fail("key_channel" if readable else "receipt_channel")
    mode = flags & os.O_ACCMODE
    if (
        not stat.S_ISFIFO(info.st_mode)
        or not target.startswith("pipe:[")
        or (readable and mode not in (os.O_RDONLY, os.O_RDWR))
        or (not readable and mode not in (os.O_WRONLY, os.O_RDWR))
    ):
        fail("key_channel" if readable else "receipt_channel")


def _read_exact_key(fd: int) -> bytearray:
    _assert_anonymous_pipe(fd, readable=True)
    chunks: list[bytes] = []
    total = 0
    while total <= IDENTITY_KEY_BYTES:
        try:
            chunk = os.read(fd, IDENTITY_KEY_BYTES + 1 - total)
        except OSError:
            fail("key_channel")
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    try:
        os.close(fd)
    except OSError:
        pass
    raw = b"".join(chunks)
    if len(raw) != IDENTITY_KEY_BYTES:
        fail("key_channel")
    return bytearray(raw)


def _write_all(fd: int, raw: bytes) -> None:
    _assert_anonymous_pipe(fd, readable=False)
    offset = 0
    while offset < len(raw):
        try:
            written = os.write(fd, raw[offset:])
        except OSError:
            fail("receipt_channel")
        if written <= 0:
            fail("receipt_channel")
        offset += written


def _verify_dev(staging_rows: Sequence[Mapping[str, Any]], key: bytes | bytearray, index: Mapping[str, Any]) -> tuple[set[bytes], dict[str, tuple[str, str]]]:
    total = automatic = organic = 0
    tokens: list[str] = []
    identities: dict[str, tuple[str, str]] = {}
    unkeyed: set[bytes] = set()
    for row in staging_rows:
        event_id = row.get("id")
        if not isinstance(event_id, str):
            fail("source_schema")
        if not (DEV_HOLDOUT_BOUND <= _bucket(event_id) < DEV_BOUND):
            continue
        total += 1
        if row.get("agent") is None:
            automatic += 1
            query = row.get("query")
            scope = row.get("requested_scope")
            if not isinstance(query, str) or not isinstance(scope, str):
                fail("source_schema")
            token_value, identity_value, message = _token(key, query, scope)
            holder = identities.setdefault(token_value, identity_value)
            if holder != identity_value:
                fail("token_collision")
            tokens.append(token_value)
            unkeyed.add(hashlib.sha256(message).digest())
        else:
            organic += 1
    if (total, automatic, organic, len(set(tokens))) != (
        EXPECTED_DEV_EVENTS, EXPECTED_DEV_AUTOMATIC, EXPECTED_DEV_ORGANIC,
        EXPECTED_DEV_UNIQUE_TOKENS,
    ):
        fail("aggregate")
    if index["tokens"] != sorted(set(tokens)):
        fail("token_mismatch")
    return unkeyed, identities


def _verify_keyed(args: argparse.Namespace) -> None:
    key = _read_exact_key(args.identity_key_fd)
    try:
        draft = _normalized_path(args.draft_dir)
        draft_info = _assert_directory(draft)
        try:
            fd_info = os.fstat(args.draft_dir_fd)
        except OSError:
            fail("boundary")
        if not stat.S_ISDIR(fd_info.st_mode) or (fd_info.st_dev, fd_info.st_ino) != (draft_info.st_dev, draft_info.st_ino):
            fail("boundary")
        roots = tuple(map(_normalized_path, (
            args.staging,
            args.snapshot,
            _normalized_path(args.original_manifest).parent,
        )))
        if any(_path_inside(draft, root) or _path_inside(root, draft) for root in roots):
            fail("boundary")
        corpus_raw, index_raw = _read_keyed_candidate(args.draft_dir_fd)
        before_pins = _source_pins(args.staging, args.snapshot, args.original_manifest, args.splits)
        _original_manifest, original_ids, original_keys = _load_original_packet(args.original_manifest)
        staging_rows, _staging_raw = _load_staging(args.staging / "alt-db" / "recall_events.jsonl")
        raw_events, raw_nodes, edges = _select_snapshot(args.snapshot)
        if set(original_ids) & {row["id"] for row in raw_events}:
            fail("disjointness")
        corpus_nodes, corpus_events = _parse_corpus(corpus_raw)
        index = _parse_index(index_raw)
        dev_unkeyed, dev_identities = _verify_dev(staging_rows, key, index)
        _counts, supplement_unkeyed, privacy = _audit_records(
            raw_nodes, raw_events, edges, corpus_nodes, corpus_events, index,
            key=key,
            original_event_ids=original_ids,
            allowed_keys=original_keys | SUPPLEMENT_STRUCTURAL_KEYS,
        )
        # Joint collision proof: recompute the mapping over all 732+769 raw
        # events, not merely over unique persisted values.
        joint = dict(dev_identities)
        for row in raw_events:
            token_value, identity_value, _message = _token(key, row["query"], row["requested_scope"])
            holder = joint.setdefault(token_value, identity_value)
            if holder != identity_value:
                fail("token_collision")
        private_values = set(privacy.private_originals)
        private_values.update(row["query"] for row in staging_rows if isinstance(row.get("query"), str))
        _privacy_scan(
            (corpus_raw, index_raw), key=key,
            unkeyed_digests=dev_unkeyed | supplement_unkeyed,
            raw_private=private_values,
            raw_queries={row["query"] for row in raw_events},
            surrogates=set(privacy.surrogate_to_original),
        )
        after_pins = _source_pins(args.staging, args.snapshot, args.original_manifest, args.splits)
        if before_pins != after_pins:
            fail("source_changed")
        final_corpus, final_index = _read_keyed_candidate(args.draft_dir_fd)
        if (
            not hmac.compare_digest(hashlib.sha256(final_corpus).digest(), hashlib.sha256(corpus_raw).digest())
            or not hmac.compare_digest(hashlib.sha256(final_index).digest(), hashlib.sha256(index_raw).digest())
        ):
            fail("candidate_changed")
        final_draft = os.lstat(draft)
        if stat.S_ISLNK(final_draft.st_mode) or (final_draft.st_dev, final_draft.st_ino) != (fd_info.st_dev, fd_info.st_ino):
            fail("candidate_changed")
        _write_all(args.receipt_fd, json.dumps(RECEIPT, separators=(",", ":"), ensure_ascii=True).encode("ascii"))
    finally:
        for index_position in range(len(key)):
            key[index_position] = 0
        try:
            os.close(args.receipt_fd)
        except OSError:
            pass


def _expected_manifest_constants() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "packet": "animal-planet P6 replacement temporal supplement",
        "namespace": "replacement-holdout",
        "frozen": True,
        "semantic_reads": 0,
        "selection": {
            "source": "pinned immutable SQLite main snapshot",
            "predicate": {
                "created_at": {"gt": START_EXCLUSIVE, "lt": END_EXCLUSIVE},
                "requested_scope_in": list(REQUESTED_SCOPES),
            },
            "outcome_filtering": False,
            "ordering": ["created_at ASC", "id ASC"],
        },
        "classification": {"automatic": "agent IS NULL", "organic": "agent IS NOT NULL"},
        "identity": {
            "algorithm": "HMAC-SHA256",
            "normalization": '" ".join(query.split()) + "\\n" + requested_scope',
            "one_key_for_dev_and_supplement": True,
            "key_transport": "inherited anonymous descriptor to keyed verifier only",
            "key_persisted": False,
            "unkeyed_query_fingerprints_persisted": False,
            "token_encoding": "lowercase hexadecimal full digest",
        },
        "deidentification": {
            "salt": "fresh independent ephemeral 256-bit value; not persisted",
            "private_string_invariants": [
                "exact Python character length",
                "space/tab/newline/carriage-return positions preserved",
                "all other characters replaced by lowercase a-z",
                "global original-to-surrogate equality classes are bijective per build",
            ],
            "private_mapping_persisted": False,
        },
        "repeated_family": {
            "population": "automatic events only",
            "identity": "fingerprint_token",
            "minimum_events": 3,
            "minimum_distinct_transport_session_values": 2,
            "unseen_in_dev": "fingerprint_token absent from frozen-dev automatic index",
        },
        "publication": {
            "no_overwrite": True,
            "content_and_index_before_manifest": True,
            "manifest_last": True,
            "canonical_namespace_sealed_iff_manifest_present": True,
        },
    }


def _validate_sealed_manifest(manifest: Mapping[str, Any], packet_blobs: Mapping[str, bytes]) -> None:
    if tuple(manifest) != MANIFEST_FIELDS:
        fail("schema")
    constants = _expected_manifest_constants()
    for name, expected in constants.items():
        if not _strict_equal(manifest.get(name), expected):
            fail("manifest")
    if not _is_utc_instant(manifest.get("generated_at")):
        fail("manifest")
    expected_counts = {
        "supplement": {
            "events": EXPECTED_SUPPLEMENT_EVENTS,
            "direct_nodes": EXPECTED_DIRECT_NODES,
            "nodes": EXPECTED_NODES,
            "automatic": EXPECTED_AUTOMATIC_EVENTS,
            "organic": EXPECTED_ORGANIC_EVENTS,
            "by_requested_scope": EXPECTED_SCOPE_COUNTS,
        },
        "frozen_dev": {
            "events": EXPECTED_DEV_EVENTS,
            "automatic": EXPECTED_DEV_AUTOMATIC,
            "organic": EXPECTED_DEV_ORGANIC,
            "unique_automatic_tokens": EXPECTED_DEV_UNIQUE_TOKENS,
        },
        "repeated_automatic": {
            "families": EXPECTED_REPEATED_FAMILIES,
            "events": EXPECTED_REPEATED_EVENTS,
            "unseen_in_dev_families": EXPECTED_UNSEEN_FAMILIES,
            "unseen_in_dev_events": EXPECTED_UNSEEN_EVENTS,
        },
    }
    if not _strict_equal(manifest.get("counts"), expected_counts):
        fail("aggregate")
    files = manifest.get("files")
    packet_files = manifest.get("packet_files")
    if type(files) is not dict or tuple(files) != ("corpus/holdout.jsonl", "corpus/dev-fingerprint-index.json"):
        fail("manifest")
    if type(packet_files) is not dict or tuple(packet_files) != ("README.md", "POLICY.md", "recipe/build.py", "recipe/verify.py"):
        fail("manifest")
    for relative, metadata in {**files, **packet_files}.items():
        expected_metadata_fields = (
            ("sha256", "bytes", "records", "nodes", "events")
            if relative == "corpus/holdout.jsonl"
            else ("sha256", "bytes", "population_events", "unique_tokens")
            if relative == "corpus/dev-fingerprint-index.json"
            else ("sha256", "bytes")
        )
        if type(metadata) is not dict or tuple(metadata) != expected_metadata_fields:
            fail("manifest")
        if relative not in packet_blobs:
            fail("manifest")
        data = packet_blobs[relative]
        digest, size = hashlib.sha256(data).hexdigest(), len(data)
        if digest != metadata.get("sha256") or size != metadata.get("bytes"):
            fail("hash")
    fixed_static = {
        "README.md": (README_SHA256, README_BYTES),
        "POLICY.md": (POLICY_SHA256, POLICY_BYTES),
        "recipe/build.py": (BUILDER_SHA256, BUILDER_BYTES),
    }
    for relative, expected in fixed_static.items():
        metadata = packet_files[relative]
        if (metadata.get("sha256"), metadata.get("bytes")) != expected:
            fail("hash")
    implementation = manifest.get("implementation")
    if type(implementation) is not dict or tuple(implementation) != ("builder", "verifier"):
        fail("manifest")
    for name in ("builder", "verifier"):
        value = implementation.get(name)
        if type(value) is not dict or tuple(value) != ("path", "sha256", "bytes") or value.get("path") != f"recipe/{'build.py' if name == 'builder' else 'verify.py'}":
            fail("manifest")
        pinned = packet_files[value["path"]]
        if value.get("sha256") != pinned["sha256"] or value.get("bytes") != pinned["bytes"]:
            fail("manifest")
    validation = manifest.get("validation")
    if type(validation) is not dict or tuple(validation) != ("keyed_preseal", "candidate_unchanged_after_validation"):
        fail("manifest")
    receipt = validation.get("keyed_preseal")
    if type(receipt) is not dict or not _strict_equal(receipt, RECEIPT):
        fail("manifest")
    if validation.get("candidate_unchanged_after_validation") is not True:
        fail("manifest")
    sources = manifest.get("sources")
    if type(sources) is not dict or tuple(sources) != (
        "staging_metadata", "staging_events", "sqlite_snapshot",
        "original_packet_manifest", "original_splits", "deid_implementation",
    ):
        fail("manifest")
    try:
        staging_metadata = sources["staging_metadata"]
        staging_events = sources["staging_events"]
        snapshot = sources["sqlite_snapshot"]
        original = sources["original_packet_manifest"]
        splits = sources["original_splits"]
        deid = sources["deid_implementation"]
    except (KeyError, TypeError):
        fail("manifest")
    expected_sources = {
        "staging_metadata": {"sha256": METADATA_SHA256, "bytes": 25_382},
        "staging_events": {"sha256": STAGING_EVENTS_SHA256, "bytes": STAGING_EVENTS_BYTES, "rows": STAGING_EVENTS_ROWS},
        "sqlite_snapshot": {
            "sha256": SNAPSHOT_SHA256, "bytes": 492_367_872,
            "open_mode": "mode=ro&immutable=1&cache=private",
            "query_only": True, "temp_store": "MEMORY",
        },
        "original_packet_manifest": {"sha256": ORIGINAL_MANIFEST_SHA256, "bytes": 34_934, "frozen": True},
        "original_splits": {"sha256": SPLITS_SHA256, "bytes": 4_040},
        "deid_implementation": {"sha256": DEID_SHA256, "bytes": 8_216},
    }
    if not _strict_equal(sources, expected_sources):
        fail("manifest")


def _audit_keyless_tree(value: Any, scopes: frozenset[str], allowed_keys: frozenset[str]) -> None:
    if type(value) is dict:
        if not all(
            isinstance(key, str)
            and STRUCTURAL_KEY_RE.fullmatch(key)
            and key in allowed_keys
            for key in value
        ):
            fail("schema")
        for item in value.values():
            _audit_keyless_tree(item, scopes, allowed_keys)
    elif type(value) is list:
        for item in value:
            _audit_keyless_tree(item, scopes, allowed_keys)
    elif isinstance(value, str):
        if not _keeps_real(value, scopes) and not PURITY_RE.fullmatch(value):
            fail("privacy")
    elif value is not None and type(value) not in (bool, int, float):
        fail("schema")
    elif type(value) is float and not math.isfinite(value):
        fail("schema")


def _audit_keyless_structure(
    nodes: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    index: Mapping[str, Any],
    *,
    allowed_keys: frozenset[str],
) -> None:
    if len(nodes) != EXPECTED_NODES or len(events) != EXPECTED_SUPPLEMENT_EVENTS:
        fail("aggregate")
    node_ids = {row["id"] for row in nodes}
    node_by_id = {row["id"]: row for row in nodes}
    # Generic de-identification may preserve syntactically valid scope names,
    # but the allowlist must never be derived from unvalidated packet text.
    scope_values = (
        [row.get("scope") for row in nodes]
        + [row.get("scope") for row in events]
        + [row.get("requested_scope") for row in events]
        + [value for row in events for value in row.get("resolved_scopes", [])]
    )
    if not all(_is_scope(value) and value in PACKET_SCOPES for value in scope_values):
        fail("schema")
    scopes = frozenset(scope_values) | {"global"}
    declared_event_scopes = frozenset({"global", *REQUESTED_SCOPES})
    direct: set[str] = set()
    for row in nodes:
        if (
            not _is_ulid(row["id"])
            or row["level"] not in LEVELS
            or not _is_scope(row["scope"])
            or not all(_is_timestamp(row[name]) for name in ("timestamp", "created_at", "updated_at"))
            or not _safe_int(row["decayed"])
            or row["decayed"] not in (0, 1)
        ):
            fail("schema")
        if type(row["stats"]) is not dict or tuple(row["stats"]) != STATS_FIELDS or type(row["relations"]) is not list:
            fail("schema")
        stats = row["stats"]
        if (
            not _safe_int(stats["access_count"])
            or stats["access_count"] < 0
            or not _safe_float(stats["usefulness_score"])
            or not _safe_float(stats["confidence"])
            or not _safe_int(stats["unique_agents"])
            or stats["unique_agents"] < 0
            or stats["last_accessed"] is not None
            and not _is_timestamp(stats["last_accessed"])
            or stats["temporal_hint"] is not None
            and not isinstance(stats["temporal_hint"], str)
        ):
            fail("schema")
        hint = stats["temporal_hint"]
        if isinstance(hint, str) and hint not in TEMPORAL_HINT_VALUES:
            fail("privacy")
        if type(row["included_via"]) is not list or not row["included_via"] or row["included_via"] != sorted(set(row["included_via"])):
            fail("containment")
        if not set(row["included_via"]).issubset({"results", "feedback", "typed_edge_partner"}):
            fail("containment")
        if "typed_edge_partner" in row["included_via"] and len(row["included_via"]) != 1:
            fail("containment")
        for relation in row["relations"]:
            if tuple(relation) != RELATION_FIELDS or relation["other_id"] not in node_ids or not _is_ulid(relation["other_id"]):
                fail("containment")
            if relation["type"] not in ("supersedes", "contradicts") or relation["direction"] not in ("in", "out"):
                fail("schema")
            if not _safe_float(relation["weight"]) or not _is_timestamp(relation["created_at"]):
                fail("schema")
        if len({tuple(item.values()) for item in row["relations"]}) != len(row["relations"]):
            fail("containment")
        if row["relations"] != sorted(row["relations"], key=lambda item: (item["type"], item["direction"], item["other_id"], item["created_at"])):
            fail("ordering")
        for field in ("agent", "task", "decay_reason"):
            surrogate = row[f"{field}_surrogate"]
            count = row[f"{field}_chars"]
            if surrogate is None:
                if count is not None:
                    fail("privacy")
            elif not isinstance(surrogate, str) or not _safe_int(count) or len(surrogate) != count or not PURITY_RE.fullmatch(surrogate):
                fail("privacy")
        if (row["decayed"] == 1) is not (row["decay_reason_surrogate"] is not None):
            fail("schema")
        if _is_utc_instant(row["updated_at"]) and _is_utc_instant(row["created_at"]) and _instant(row["updated_at"]) < _instant(row["created_at"]):
            fail("ordering")
        if stats["last_accessed"] is not None and _is_utc_instant(stats["last_accessed"]) and _is_utc_instant(row["created_at"]) and _instant(stats["last_accessed"]) < _instant(row["created_at"]):
            fail("ordering")
        if not isinstance(row["content_surrogate"], str) or not _safe_int(row["content_chars"]) or len(row["content_surrogate"]) != row["content_chars"] or not PURITY_RE.fullmatch(row["content_surrogate"]):
            fail("privacy")
        if type(row["context_surrogate"]) is not dict or type(row["context_chars"]) is not dict or not all(isinstance(key, str) and _safe_int(value) for key, value in row["context_chars"].items()):
            fail("schema")
        if tuple(row["context_surrogate"]) != tuple(row["context_chars"]):
            fail("schema")
        for key, value in row["context_surrogate"].items():
            visible = _delivery_chars(value)
            declared = row["context_chars"][key]
            if declared < visible or (not isinstance(value, (dict, list)) and declared != visible):
                fail("privacy")
        _audit_keyless_tree(row["context_surrogate"], scopes, allowed_keys)
        if type(row["source_traces"]) is not list or not all(_is_ulid(value) for value in row["source_traces"]):
            fail("schema")
        if type(row["corrections"]) is not list or type(row["corrections_chars"]) is not list or len(row["corrections"]) != len(row["corrections_chars"]):
            fail("schema")
        for correction, chars in zip(row["corrections"], row["corrections_chars"]):
            if type(correction) is not dict or type(chars) is not dict:
                fail("schema")
            expected_char_fields = tuple(
                name for name in ("old", "new") if isinstance(correction.get(name), str)
            )
            if tuple(chars) != expected_char_fields:
                fail("privacy")
            for name, value in correction.items():
                if (
                    not isinstance(name, str)
                    or not STRUCTURAL_KEY_RE.fullmatch(name)
                    or name not in allowed_keys
                ):
                    fail("schema")
                if name in ("old", "new") and isinstance(value, str):
                    if not _safe_int(chars[name]) or len(value) != chars[name] or not PURITY_RE.fullmatch(value):
                        fail("privacy")
                else:
                    _audit_keyless_tree(value, scopes, allowed_keys)
        if type(row["provenance_shape"]) is not dict or tuple(row["provenance_shape"]) != ("keys", "serialized_chars") or not _safe_int(row["provenance_shape"]["serialized_chars"]):
            fail("schema")
        shape_keys = row["provenance_shape"]["keys"]
        if type(shape_keys) is not dict:
            fail("schema")
        for name, shape in shape_keys.items():
            if (
                not isinstance(name, str)
                or not STRUCTURAL_KEY_RE.fullmatch(name)
                or name not in allowed_keys
                or type(shape) is not dict
            ):
                fail("schema")
            if tuple(shape) not in (("kind", "chars"), ("kind", "count", "chars")):
                fail("schema")
            if shape["kind"] not in (
                "NoneType", "bool", "dict", "float", "int", "list", "str",
            ) or not _safe_int(shape["chars"]) or shape["chars"] < 0:
                fail("schema")
            minimum_chars = {
                "NoneType": 4,
                "bool": 4,
                "dict": 2,
                "list": 2,
                "float": 1,
                "int": 1,
                "str": 0,
            }[shape["kind"]]
            if shape["chars"] < minimum_chars:
                fail("schema")
            if (
                shape["kind"] == "NoneType" and shape["chars"] != 4
                or shape["kind"] == "bool" and shape["chars"] not in (4, 5)
                or shape["kind"] in ("list", "dict")
                and shape.get("count") == 0
                and shape["chars"] != 2
            ):
                fail("schema")
            if "count" in shape and (not _safe_int(shape["count"]) or shape["count"] < 0):
                fail("schema")
            if (shape["kind"] in ("list", "dict")) is not ("count" in shape):
                fail("schema")
        provenance_floor = _delivery_chars({
            "source_traces": row["source_traces"],
            "corrections": row["corrections"],
        })
        if row["provenance_shape"]["serialized_chars"] < provenance_floor:
            fail("schema")
    class_counts: collections.Counter[str] = collections.Counter()
    scope_counts: collections.Counter[str] = collections.Counter()
    groups: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    expected_reasons: dict[str, set[str]] = collections.defaultdict(set)
    supersedes_endpoints: set[str] = set()
    for row in nodes:
        for relation in row["relations"]:
            if relation["type"] == "supersedes":
                supersedes_endpoints.update((row["id"], relation["other_id"]))
    for row in events:
        if (
            not _is_ulid(row["id"])
            or row["source"] != "local"
            or not _is_utc_instant(row["created_at"])
            or not _is_scope(row["scope"])
            or row["requested_scope"] not in REQUESTED_SCOPES
            or row["depth"] not in DEPTH_VALUES
        ):
            fail("schema")
        if len(row["results"]) > row["max_results"]:
            fail("schema")
        if row["feedback_applied_at"] is not None and _instant(row["feedback_applied_at"]) < _instant(row["created_at"]):
            fail("ordering")
        if type(row["resolved_scopes"]) is not list or not all(
            _is_scope(value) for value in row["resolved_scopes"]
        ):
            fail("schema")
        if (
            row["scope"] != row["requested_scope"]
            or row["resolved_scopes"] != [row["requested_scope"], "global"]
        ):
            fail("selection")
        if (
            not _safe_int(row["max_results"])
            or row["max_results"] < 0
            or not _safe_int(row["feedback_applied"])
            or row["feedback_applied"] not in (0, 1)
            or row["feedback_trace_id"] is not None
            and not _is_ulid(row["feedback_trace_id"])
            or row["feedback_applied_at"] is not None
            and not _is_utc_instant(row["feedback_applied_at"])
            or (row["feedback_applied"] == 1)
            is not (row["feedback_trace_id"] is not None and row["feedback_applied_at"] is not None)
        ):
            fail("schema")
        if row["scope"] not in declared_event_scopes or any(
            scope not in declared_event_scopes for scope in row["resolved_scopes"]
        ):
            fail("selection")
        if row["source"] != "local" or row["requested_scope"] not in REQUESTED_SCOPES or not (START_EXCLUSIVE < row["created_at"] < END_EXCLUSIVE):
            fail("selection")
        if row["class"] not in ("automatic", "organic") or row["template_id"] not in (None, "reopen_lesson", "architectural_decision"):
            fail("classification")
        if (row["class"] == "automatic") is not (row["agent_surrogate"] is None):
            fail("classification")
        if not isinstance(row["fingerprint_token"], str) or not TOKEN_RE.fullmatch(row["fingerprint_token"]):
            fail("token")
        if type(row["results"]) is not list or any(tuple(item) != RESULT_FIELDS for item in row["results"]):
            fail("schema")
        if [item["rank"] for item in row["results"]] != list(range(1, len(row["results"]) + 1)):
            fail("ordering")
        if len({item["node_id"] for item in row["results"]}) != len(row["results"]):
            fail("membership")
        for item in row["results"]:
            _validate_result(item)
            if item["methods"] != sorted(item["methods"], key=RESULT_METHOD_ORDER.__getitem__):
                fail("ordering")
            if item["node_id"] not in node_ids:
                fail("containment")
            referenced = node_by_id[item["node_id"]]
            if item["level"] != referenced["level"] or item["scope"] != referenced["scope"]:
                fail("containment")
            expected_reasons[item["node_id"]].add("results")
        if row["feedback_trace_id"] is not None:
            if row["feedback_trace_id"] not in node_ids:
                fail("containment")
            expected_reasons[row["feedback_trace_id"]].add("feedback")
        bearing = bool({item["node_id"] for item in row["results"]} & supersedes_endpoints)
        if row["supersedes_bearing"] is not bearing:
            fail("containment")
        if row["transcript_matched"] is not False or row["transcript_serialized_chars"] is not None:
            fail("schema")
        for field in ("query", "agent", "task", "session_id"):
            surrogate = row[f"{field}_surrogate"]
            count = row[f"{field}_chars"]
            if surrogate is None:
                if count is not None:
                    fail("privacy")
            elif not isinstance(surrogate, str) or not _safe_int(count) or len(surrogate) != count or not PURITY_RE.fullmatch(surrogate):
                fail("privacy")
        if row["query_surrogate"] is None:
            fail("privacy")
        transport = row["transport_session_id"]
        if not isinstance(transport, str) or not HEX32_RE.fullmatch(transport):
            fail("privacy")
        if type(row["ambient_context_surrogate"]) is not dict or type(row["ambient_context_chars"]) is not dict or not all(isinstance(key, str) and _safe_int(value) for key, value in row["ambient_context_chars"].items()):
            fail("schema")
        if tuple(row["ambient_context_surrogate"]) != tuple(row["ambient_context_chars"]):
            fail("schema")
        for key, value in row["ambient_context_surrogate"].items():
            visible = _delivery_chars(value)
            declared = row["ambient_context_chars"][key]
            if declared < visible or (not isinstance(value, (dict, list)) and declared != visible):
                fail("privacy")
        _audit_keyless_tree(row["ambient_context_surrogate"], scopes, allowed_keys)
        class_counts[row["class"]] += 1
        scope_counts[row["requested_scope"]] += 1
        if row["class"] == "automatic":
            groups[row["fingerprint_token"]].append(row)
    if class_counts != {"automatic": EXPECTED_AUTOMATIC_EVENTS, "organic": EXPECTED_ORGANIC_EVENTS} or scope_counts != EXPECTED_SCOPE_COUNTS:
        fail("aggregate")
    index_tokens = set(index["tokens"])
    if len(expected_reasons) != EXPECTED_DIRECT_NODES:
        fail("containment")
    for row in nodes:
        expected = sorted(expected_reasons.get(row["id"], {"typed_edge_partner"}))
        if row["included_via"] != expected:
            fail("containment")
        if expected == ["typed_edge_partner"] and not any(
            relation["other_id"] in expected_reasons for relation in row["relations"]
        ):
            fail("containment")
        for relation in row["relations"]:
            if relation["other_id"] == row["id"]:
                if relation["direction"] != "out":
                    fail("containment")
                continue
            reciprocal = {
                "type": relation["type"],
                "direction": "in" if relation["direction"] == "out" else "out",
                "other_id": row["id"],
                "weight": relation["weight"],
                "created_at": relation["created_at"],
            }
            partner = node_by_id[relation["other_id"]]
            if reciprocal not in partner["relations"]:
                fail("containment")
    repeated = {token: rows for token, rows in groups.items() if len(rows) >= 3 and len({row["transport_session_id"] for row in rows}) >= 2}
    unseen = {token: rows for token, rows in repeated.items() if token not in index_tokens}
    if (len(repeated), sum(map(len, repeated.values())), len(unseen), sum(map(len, unseen.values()))) != (
        EXPECTED_REPEATED_FAMILIES, EXPECTED_REPEATED_EVENTS, EXPECTED_UNSEEN_FAMILIES, EXPECTED_UNSEEN_EVENTS,
    ):
        fail("aggregate")


def _verify_sealed(args: argparse.Namespace) -> None:
    packet = _normalized_path(args.packet_dir)
    packet_info = _assert_directory(packet)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        packet_fd = os.open(packet, flags)
    except OSError:
        fail("boundary")
    bound_packet = os.fstat(packet_fd)
    if (bound_packet.st_dev, bound_packet.st_ino) != (packet_info.st_dev, packet_info.st_ino):
        os.close(packet_fd)
        fail("candidate_changed")
    # The executing verifier is the external trust anchor.  Bind the packet's
    # tracked copy byte-for-byte so it cannot become a post-seal text channel
    # while merely updating its self-reported manifest hash.
    running_verifier = _read_regular(Path(__file__))
    blobs, tree = _read_packet_files(packet_fd)
    manifest_raw = blobs["manifest.json"]
    manifest = _json(manifest_raw, expected=dict)
    if _canonical_document(manifest) != manifest_raw:
        fail("canonical")
    _validate_sealed_manifest(manifest, blobs)
    _original_manifest, original_ids, original_keys = _load_original_packet(args.original_manifest)
    corpus_raw = blobs["corpus/holdout.jsonl"]
    index_raw = blobs["corpus/dev-fingerprint-index.json"]
    packet_verifier = blobs["recipe/verify.py"]
    if not hmac.compare_digest(
        hashlib.sha256(packet_verifier).digest(),
        hashlib.sha256(running_verifier).digest(),
    ):
        fail("hash")
    if (
        hashlib.sha256(corpus_raw).hexdigest()
        != manifest["files"]["corpus/holdout.jsonl"]["sha256"]
        or len(corpus_raw) != manifest["files"]["corpus/holdout.jsonl"]["bytes"]
        or hashlib.sha256(index_raw).hexdigest()
        != manifest["files"]["corpus/dev-fingerprint-index.json"]["sha256"]
        or len(index_raw)
        != manifest["files"]["corpus/dev-fingerprint-index.json"]["bytes"]
    ):
        fail("hash")
    nodes, events = _parse_corpus(corpus_raw)
    index = _parse_index(index_raw)
    if {row["id"] for row in events} & original_ids:
        fail("disjointness")
    _audit_keyless_structure(
        nodes,
        events,
        index,
        allowed_keys=original_keys | SUPPLEMENT_STRUCTURAL_KEYS,
    )
    files = manifest["files"]
    if files["corpus/holdout.jsonl"].get("records") != len(nodes) + len(events) or files["corpus/holdout.jsonl"].get("nodes") != len(nodes) or files["corpus/holdout.jsonl"].get("events") != len(events):
        fail("aggregate")
    if files["corpus/dev-fingerprint-index.json"].get("population_events") != EXPECTED_DEV_AUTOMATIC or files["corpus/dev-fingerprint-index.json"].get("unique_tokens") != len(index["tokens"]):
        fail("aggregate")
    final_blobs, final_tree = _read_packet_files(packet_fd)
    if tree != final_tree or set(blobs) != set(final_blobs) or any(
        not hmac.compare_digest(hashlib.sha256(blobs[name]).digest(), hashlib.sha256(final_blobs[name]).digest())
        for name in blobs
    ):
        fail("candidate_changed")
    final_packet = os.lstat(packet)
    if (
        stat.S_ISLNK(final_packet.st_mode)
        or (final_packet.st_dev, final_packet.st_ino)
        != (packet_info.st_dev, packet_info.st_ino)
    ):
        fail("candidate_changed")
    for relative, metadata in {**manifest["files"], **manifest["packet_files"]}.items():
        data = final_blobs[relative]
        if (
            len(data) != metadata["bytes"]
            or not hmac.compare_digest(hashlib.sha256(data).hexdigest(), metadata["sha256"])
        ):
            fail("candidate_changed")
    os.close(packet_fd)


class _QuietParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise VerificationError("arguments")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--mechanical-only"]:
        packet_dir = Path(__file__).resolve().parents[1]
        arguments = [
            "sealed",
            "--packet-dir",
            os.fspath(packet_dir),
            "--original-manifest",
            os.fspath(packet_dir.parents[1] / "manifest.json"),
        ]
    parser = _QuietParser(add_help=True, description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True, parser_class=_QuietParser)
    keyed = subparsers.add_parser("keyed", add_help=True)
    keyed.add_argument("--staging", type=Path, required=True)
    keyed.add_argument("--snapshot", type=Path, required=True)
    keyed.add_argument("--original-manifest", type=Path, required=True)
    keyed.add_argument("--splits", type=Path, required=True)
    keyed.add_argument("--draft-dir", type=Path, required=True)
    keyed.add_argument("--draft-dir-fd", type=int, required=True)
    keyed.add_argument("--identity-key-fd", type=int, required=True)
    keyed.add_argument("--receipt-fd", type=int, required=True)
    sealed = subparsers.add_parser("sealed", add_help=True)
    sealed.add_argument("--packet-dir", type=Path, required=True)
    sealed.add_argument("--original-manifest", type=Path, required=True)
    return parser.parse_args(arguments)


def main(argv: Sequence[str] | None = None) -> int:
    _disable_core_dumps()
    mode = "unknown"
    try:
        args = _parse_args(argv)
        mode = args.mode
        if mode == "keyed":
            _verify_keyed(args)
            return 0
        if mode == "sealed":
            _verify_sealed(args)
            print(json.dumps({"mode": "keyless-sealed", "semantic_reads": 0, "status": "pass"}, sort_keys=True, separators=(",", ":")))
            return 0
        fail("arguments")
    except VerificationError as error:
        if mode == "sealed":
            print(json.dumps({"mode": "keyless-sealed", "status": "fail", "check": error.code}, sort_keys=True, separators=(",", ":")))
        return 1
    except BaseException:
        if mode == "sealed":
            print(json.dumps({"mode": "keyless-sealed", "status": "fail", "check": "internal"}, sort_keys=True, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
