#!/usr/bin/env python3
"""Build and, only after independent keyed validation, freeze the P6 supplement.

This runner has two deliberately separate lifecycles::

    build.py draft --draft-dir DIR [source options]
    build.py freeze --draft-dir DIR --publish-dir DIR \
      --verifier-sha256 HEX [source options]

``draft`` writes a non-sealed candidate.  ``freeze`` regenerates the candidate
in an empty draft directory, invokes the fixed sibling ``verify.py`` in keyed
pre-seal mode, lets every key-bearing process exit, and only then publishes.
There is no command that can publish an old or unvalidated draft.

The verifier protocol is intentionally small and contains no secret-bearing
channel other than an inherited anonymous descriptor.  The runner invokes::

    python -I verify.py keyed \
      --staging STAGING --snapshot SNAPSHOT \
      --original-manifest MANIFEST --splits SPLITS \
      --draft-dir DRAFT --draft-dir-fd FD \
      --identity-key-fd FD --receipt-fd FD

The identity descriptor contains exactly 32 bytes and reaches the verifier
through ``subprocess.Popen(pass_fds=...)``; it is never placed in argv, the
environment, a regular file, or output.  The receipt descriptor must contain
one JSON object with the aggregate fields declared in ``RECEIPT_REQUIRED``.
Verifier stdout and stderr are discarded so a faulty verifier cannot disclose
private diagnostics through this runner.

Only aggregate, privacy-safe status is printed.  Raw queries, surrogate text,
fingerprint tokens, salts, and keys are never printed or included in errors.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import errno
import fcntl
import hashlib
import hmac
import json
import os
import resource
import secrets
import select
import signal
import sqlite3
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence


HERE = Path(__file__).resolve().parent
CANONICAL_ROOT = HERE.parent
ORIGINAL_PACKET_ROOT = HERE.parents[2]

DEFAULT_STAGING = Path("/home/sfx/.cache/ap-audit/staging")
DEFAULT_SNAPSHOT = Path("/tmp/lm-shadow-eval.O3vX3k/global.sqlite3")
DEFAULT_ORIGINAL_MANIFEST = ORIGINAL_PACKET_ROOT / "manifest.json"
DEFAULT_SPLITS = ORIGINAL_PACKET_ROOT / "corpus" / "splits.json"
DEID_PATH = ORIGINAL_PACKET_ROOT / "recipe" / "transform" / "deid.py"
VERIFIER_PATH = HERE / "verify.py"

START_EXCLUSIVE = "2026-08-12T23:13:24Z"
END_EXCLUSIVE = "2026-08-13T20:16:51Z"
REQUESTED_SCOPES = ("project:ae", "project:online")

METADATA_SHA256 = "025e734f84f452711e86a31d1639f2ee9833213e7ec94ced69b95d315ebac855"
SNAPSHOT_SHA256 = "4d6648f9e3c33e8ffaa7bf15620bd26a18bdf9f7ddf5b97641aa8c082e9aaa67"
STAGING_EVENTS_SHA256 = "45dbfdb389324a8702ba1a84b28be028991f23dcfe1d7ab9d771928c37c2cc7a"
ORIGINAL_MANIFEST_SHA256 = "3f1a6a87a4d34e411f06e50161f9203afc3c4992dfc973d4f544453e8bbfde48"
SPLITS_SHA256 = "66f04caa41e0302fd7b8582f54197a396a2bb8483ddd543e564740a8e4c3a7b7"
DEID_SHA256 = "503de08cea164660c2f1c3b3675ad082f70109d8f4360194bb1e02c7423bd6e3"

STAGING_EVENTS_BYTES = 4_162_914
STAGING_EVENTS_ROWS = 1_576
DEV_HOLDOUT_BOUND = 15
DEV_BOUND = 66
SPLIT_MOD = 100

EXPECTED_SUPPLEMENT_EVENTS = 769
EXPECTED_AUTOMATIC_EVENTS = 332
EXPECTED_ORGANIC_EVENTS = 437
EXPECTED_SCOPE_COUNTS = {"project:ae": 502, "project:online": 267}
EXPECTED_NODE_COUNT = 619
EXPECTED_TOTAL_NODE_COUNT = 804
EXPECTED_DEV_EVENTS = 826
EXPECTED_DEV_AUTOMATIC = 732
EXPECTED_DEV_ORGANIC = 94
EXPECTED_DEV_UNIQUE_TOKENS = 353
EXPECTED_REPEATED_FAMILIES = 18
EXPECTED_REPEATED_EVENTS = 238
EXPECTED_UNSEEN_FAMILIES = 16
EXPECTED_UNSEEN_EVENTS = 113

TOKEN_HEX_LENGTH = 64
IDENTITY_KEY_BYTES = 32
SURROGATE_SALT_BYTES = 32
MAX_RECEIPT_BYTES = 16_384
VERIFIER_TIMEOUT_SECONDS = 600

RECALL_EVENT_COLUMNS = (
    "id",
    "query",
    "scope",
    "requested_scope",
    "resolved_scopes",
    "ambient_context",
    "depth",
    "max_results",
    "results",
    "agent",
    "task",
    "session_id",
    "feedback_applied",
    "feedback_trace_id",
    "feedback_applied_at",
    "created_at",
    "transport_session_id",
)
NODE_COLUMNS = (
    "id",
    "level",
    "content",
    "scope",
    "agent",
    "task",
    "context",
    "timestamp",
    "decayed",
    "decay_reason",
    "access_count",
    "last_accessed",
    "usefulness_score",
    "confidence",
    "unique_agents",
    "temporal_hint",
    "source_traces",
    "corrections",
    "provenance",
    "created_at",
    "updated_at",
)
RESULT_FIELDS = (
    "node_id",
    "level",
    "scope",
    "rank",
    "score",
    "bm25_score",
    "vector_score",
    "graph_score",
    "trigger_score",
    "methods",
    "path",
)

RECEIPT_REQUIRED: dict[str, Any] = {
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
    "direct_nodes": EXPECTED_NODE_COUNT,
    "nodes": EXPECTED_TOTAL_NODE_COUNT,
    "dev_automatic_events": EXPECTED_DEV_AUTOMATIC,
    "dev_unique_tokens": EXPECTED_DEV_UNIQUE_TOKENS,
    "repeated_automatic_families": EXPECTED_REPEATED_FAMILIES,
    "repeated_automatic_events": EXPECTED_REPEATED_EVENTS,
    "unseen_in_dev_families": EXPECTED_UNSEEN_FAMILIES,
    "unseen_in_dev_events": EXPECTED_UNSEEN_EVENTS,
}

DRAFT_CORPUS_REL = Path("corpus/holdout.jsonl")
DRAFT_INDEX_REL = Path("corpus/dev-fingerprint-index.json")
DRAFT_MANIFEST_REL = Path("manifest.json")
DATA_FILE_RELS = (DRAFT_CORPUS_REL, DRAFT_INDEX_REL)
ALL_FILE_RELS = (*DATA_FILE_RELS, DRAFT_MANIFEST_REL)
PACKET_FILE_RELS = (
    Path("README.md"),
    Path("POLICY.md"),
    Path("recipe/build.py"),
    Path("recipe/verify.py"),
)


class BuildError(RuntimeError):
    """A fail-closed error whose code is safe to expose."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _raise(code: str) -> None:
    raise BuildError(code)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalized_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(path: Path, *, allow_missing_leaf: bool) -> None:
    candidate = _normalized_path(path)
    current = Path(candidate.anchor)
    parts = candidate.parts[1:] if candidate.is_absolute() else candidate.parts
    for index, part in enumerate(parts):
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf and index == len(parts) - 1:
                return
            _raise("path_missing")
        if stat.S_ISLNK(info.st_mode):
            _raise("symlink_path_rejected")


def _paths_overlap(left: Path, right: Path) -> bool:
    left_s = os.fspath(_normalized_path(left))
    right_s = os.fspath(_normalized_path(right))
    try:
        common = os.path.commonpath((left_s, right_s))
    except ValueError:
        return False
    return common in (left_s, right_s)


def _open_flags_readonly() -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_directory_chain(
    path: Path, *, create_leaf: bool = False, leaf_mode: int = 0o700
) -> int:
    """Open an absolute directory without following any path-component link."""

    candidate = _normalized_path(path)
    parts = candidate.parts
    if not candidate.is_absolute() or not parts:
        _raise("directory_path_invalid")
    try:
        current = os.open(candidate.anchor, _directory_open_flags())
    except OSError:
        _raise("directory_open_failed")
    try:
        for index, part in enumerate(parts[1:]):
            last = index == len(parts[1:]) - 1
            try:
                child = os.open(part, _directory_open_flags(), dir_fd=current)
            except FileNotFoundError:
                if not (create_leaf and last):
                    _raise("path_missing")
                try:
                    os.mkdir(part, leaf_mode, dir_fd=current)
                    child = os.open(part, _directory_open_flags(), dir_fd=current)
                except OSError:
                    _raise("directory_create_failed")
            except OSError:
                _raise("symlink_path_rejected")
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _open_child_directory(
    parent_fd: int, name: str, *, create: bool = False, mode: int = 0o700
) -> int:
    if not name or "/" in name or name in (".", ".."):
        _raise("directory_name_invalid")
    try:
        return os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            _raise("path_missing")
        try:
            os.mkdir(name, mode, dir_fd=parent_fd)
            return os.open(name, _directory_open_flags(), dir_fd=parent_fd)
        except OSError:
            _raise("directory_create_failed")
    except OSError:
        _raise("symlink_path_rejected")


def _directory_identity(fd: int) -> tuple[int, int, int, int]:
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode):
        _raise("directory_fd_invalid")
    return (info.st_dev, info.st_ino, info.st_uid, stat.S_IMODE(info.st_mode))


def _require_directory_identity(
    fd: int, expected: tuple[int, int, int, int], label: str
) -> None:
    if _directory_identity(fd) != expected:
        _raise(f"{label}_directory_changed")


def _require_path_directory_identity(
    path: Path, expected: tuple[int, int, int, int], label: str
) -> None:
    reopened = _open_directory_chain(path)
    try:
        _require_directory_identity(reopened, expected, label)
    finally:
        os.close(reopened)


def _require_child_directory_identity(
    parent_fd: int,
    name: str,
    expected: tuple[int, int, int, int],
    label: str,
) -> None:
    reopened = _open_child_directory(parent_fd, name)
    try:
        _require_directory_identity(reopened, expected, label)
    finally:
        os.close(reopened)


@contextmanager
def _relative_parent_fd(root_fd: int, relative: Path) -> Iterator[tuple[int, str]]:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        _raise("relative_path_invalid")
    current = os.dup(root_fd)
    try:
        for part in relative.parts[:-1]:
            child = _open_child_directory(current, part)
            os.close(current)
            current = child
        yield current, relative.parts[-1]
    finally:
        os.close(current)


def _hash_at(parent_fd: int, name: str) -> tuple[str, int]:
    try:
        fd = os.open(name, _open_flags_readonly(), dir_fd=parent_fd)
    except OSError:
        _raise("source_open_failed")
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            _raise("source_not_regular")
        digest, size = _hash_fd(fd)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            _raise("source_changed_during_hash")
        return digest, size
    finally:
        os.close(fd)


def _hash_relative(root_fd: int, relative: Path) -> tuple[str, int]:
    with _relative_parent_fd(root_fd, relative) as (parent_fd, name):
        return _hash_at(parent_fd, name)


def _read_at(parent_fd: int, name: str, *, max_bytes: int | None = None) -> bytes:
    try:
        fd = os.open(name, _open_flags_readonly(), dir_fd=parent_fd)
    except OSError:
        _raise("source_open_failed")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            _raise("source_not_regular")
        if max_bytes is not None and info.st_size > max_bytes:
            _raise("source_size_limit")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                _raise("source_short_read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            _raise("source_changed_during_read")
        after = os.fstat(fd)
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            _raise("source_changed_during_read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _read_relative(
    root_fd: int, relative: Path, *, max_bytes: int | None = None
) -> bytes:
    with _relative_parent_fd(root_fd, relative) as (parent_fd, name):
        return _read_at(parent_fd, name, max_bytes=max_bytes)


def _file_metadata_relative(root_fd: int, relative: Path) -> dict[str, Any]:
    digest, size = _hash_relative(root_fd, relative)
    return {"sha256": digest, "bytes": size}


@contextmanager
def _regular_fd(path: Path) -> Iterator[int]:
    _assert_no_symlink_components(path, allow_missing_leaf=False)
    try:
        fd = os.open(path, _open_flags_readonly())
    except OSError:
        _raise("source_open_failed")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            _raise("source_not_regular")
        yield fd
    finally:
        os.close(fd)


def _hash_fd(fd: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    offset = 0
    while True:
        chunk = os.pread(fd, 1024 * 1024, offset)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
        offset += len(chunk)
    return digest.hexdigest(), total


def _hash_regular(path: Path) -> tuple[str, int]:
    with _regular_fd(path) as fd:
        before = os.fstat(fd)
        digest, size = _hash_fd(fd)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            _raise("source_changed_during_hash")
        return digest, size


def _read_regular(path: Path, *, max_bytes: int | None = None) -> bytes:
    with _regular_fd(path) as fd:
        info = os.fstat(fd)
        if max_bytes is not None and info.st_size > max_bytes:
            _raise("source_size_limit")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                _raise("source_short_read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            _raise("source_changed_during_read")
        after = os.fstat(fd)
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            _raise("source_changed_during_read")
        return b"".join(chunks)


def _require_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    digest, size = _hash_regular(path)
    if not hmac.compare_digest(digest, expected):
        _raise(f"{label}_hash_mismatch")
    return {"sha256": expected, "bytes": size}


def _load_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        _raise(f"{label}_json_invalid")


def _load_json_field(raw: Any, expected_type: type, label: str) -> Any:
    if not isinstance(raw, str):
        _raise(f"{label}_storage_type")
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        _raise(f"{label}_json_invalid")
    if not isinstance(value, expected_type):
        _raise(f"{label}_json_shape")
    return value


def _verify_public_source_pins(args: argparse.Namespace) -> dict[str, Any]:
    staging = _normalized_path(args.staging)
    metadata_path = staging / "METADATA.json"
    events_path = staging / "alt-db" / "recall_events.jsonl"

    builder_hash, builder_bytes = _hash_regular(Path(__file__).resolve())
    pins = {
        "staging_metadata": _require_hash(metadata_path, METADATA_SHA256, "metadata"),
        "staging_events": _require_hash(
            events_path, STAGING_EVENTS_SHA256, "staging_events"
        ),
        "snapshot": _require_hash(args.snapshot, SNAPSHOT_SHA256, "snapshot"),
        "original_manifest": _require_hash(
            args.original_manifest,
            ORIGINAL_MANIFEST_SHA256,
            "original_manifest",
        ),
        "original_splits": _require_hash(args.splits, SPLITS_SHA256, "splits"),
        "deid_implementation": _require_hash(DEID_PATH, DEID_SHA256, "deid"),
        "builder": {
            "sha256": builder_hash,
            "bytes": builder_bytes,
        },
    }
    if pins["staging_events"]["bytes"] != STAGING_EVENTS_BYTES:
        _raise("staging_events_size_mismatch")

    metadata = _load_json(_read_regular(metadata_path), "metadata")
    try:
        export_pin = metadata["exports"]["alt-db/recall_events.jsonl"]
    except (KeyError, TypeError):
        _raise("metadata_export_pin_missing")
    if not isinstance(export_pin, dict):
        _raise("metadata_export_pin_shape")
    if (
        export_pin.get("sha256") != STAGING_EVENTS_SHA256
        or export_pin.get("bytes") != STAGING_EVENTS_BYTES
        or export_pin.get("rows") != STAGING_EVENTS_ROWS
    ):
        _raise("metadata_export_pin_mismatch")

    original_manifest = _load_json(
        _read_regular(args.original_manifest), "original_manifest"
    )
    try:
        manifest_splits = original_manifest["files"]["corpus/splits.json"]
        manifest_deid = original_manifest["packet_files"][
            "recipe/transform/deid.py"
        ]
    except (KeyError, TypeError):
        _raise("original_manifest_pin_missing")
    if original_manifest.get("frozen") is not True:
        _raise("original_packet_not_frozen")
    if (
        manifest_splits.get("sha256") != SPLITS_SHA256
        or manifest_deid.get("sha256") != DEID_SHA256
    ):
        _raise("original_manifest_nested_pin_mismatch")

    splits = _load_json(_read_regular(args.splits), "splits")
    try:
        rule = splits["rule"]
        dev_counts = splits["counts"]["dev"]
    except (KeyError, TypeError):
        _raise("splits_contract_missing")
    if (
        splits.get("staging_metadata_sha256") != METADATA_SHA256
        or rule.get("mod") != SPLIT_MOD
        or rule.get("holdout_bound") != DEV_HOLDOUT_BOUND
        or rule.get("dev_bound") != DEV_BOUND
        or dev_counts.get("events") != EXPECTED_DEV_EVENTS
        or dev_counts.get("by_class", {}).get("automatic")
        != EXPECTED_DEV_AUTOMATIC
        or dev_counts.get("by_class", {}).get("organic") != EXPECTED_DEV_ORGANIC
    ):
        _raise("splits_contract_mismatch")

    wal_path = Path(os.fspath(args.snapshot) + "-wal")
    try:
        wal_info = os.lstat(wal_path)
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISLNK(wal_info.st_mode) or not stat.S_ISREG(wal_info.st_mode):
            _raise("snapshot_wal_invalid")
        if wal_info.st_size != 0:
            _raise("snapshot_wal_nonempty")
    return pins


def _import_deid() -> ModuleType:
    source = _read_regular(DEID_PATH, max_bytes=1024 * 1024)
    digest = hashlib.sha256(source).hexdigest()
    if not hmac.compare_digest(digest, DEID_SHA256):
        _raise("deid_hash_mismatch")
    module = ModuleType("replacement_packet_deid")
    module.__file__ = "pinned-deid.py"
    try:
        code = compile(source, "pinned-deid.py", "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception:
        _raise("deid_import_failed")
    return module


def _bucket_of(event_id: str) -> int:
    digest = hashlib.sha256(event_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % SPLIT_MOD


def _fingerprint_token(
    identity_key: bytearray, query: str, requested_scope: str
) -> tuple[str, tuple[str, str]]:
    normalized = " ".join(query.split())
    identity = (normalized, requested_scope)
    message = (normalized + "\n" + requested_scope).encode("utf-8")
    token = hmac.new(identity_key, message, hashlib.sha256).hexdigest()
    if len(token) != TOKEN_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in token
    ):
        _raise("fingerprint_token_encoding_invalid")
    return token, identity


def _load_staging_events(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = _read_regular(path, max_bytes=STAGING_EVENTS_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(digest, STAGING_EVENTS_SHA256):
        _raise("staging_events_hash_mismatch")
    if len(raw) != STAGING_EVENTS_BYTES:
        _raise("staging_events_size_mismatch")
    lines = raw.splitlines()
    if len(lines) != STAGING_EVENTS_ROWS or not raw.endswith(b"\n"):
        _raise("staging_events_row_count_mismatch")
    rows: list[dict[str, Any]] = []
    for line in lines:
        value = _load_json(line, "staging_event")
        if not isinstance(value, dict) or set(value) != set(RECALL_EVENT_COLUMNS):
            _raise("staging_event_schema_mismatch")
        rows.append(value)
    return rows, {
        "sha256": STAGING_EVENTS_SHA256,
        "bytes": len(raw),
        "rows": len(rows),
    }


def _validate_result(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != set(RESULT_FIELDS):
        _raise("result_schema_mismatch")
    if not all(isinstance(raw.get(name), str) for name in ("node_id", "level", "scope")):
        _raise("result_string_field_invalid")
    if isinstance(raw.get("rank"), bool) or not isinstance(raw.get("rank"), int):
        _raise("result_rank_invalid")
    for field in (
        "score",
        "bm25_score",
        "vector_score",
        "graph_score",
        "trigger_score",
    ):
        if isinstance(raw.get(field), bool) or not isinstance(raw.get(field), (int, float)):
            _raise("result_score_invalid")
    if not isinstance(raw.get("methods"), list) or not all(
        isinstance(item, str) for item in raw["methods"]
    ):
        _raise("result_methods_invalid")
    if not isinstance(raw.get("path"), list) or not all(
        isinstance(item, str) for item in raw["path"]
    ):
        _raise("result_path_invalid")
    return {name: raw[name] for name in RESULT_FIELDS}


def _select_snapshot_population(
    snapshot: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    _assert_no_symlink_components(snapshot, allow_missing_leaf=False)
    wal_path = Path(os.fspath(snapshot) + "-wal")
    try:
        wal_info = os.lstat(wal_path)
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISLNK(wal_info.st_mode) or not stat.S_ISREG(wal_info.st_mode):
            _raise("snapshot_wal_invalid")
        if wal_info.st_size:
            _raise("snapshot_wal_nonempty")

    with _regular_fd(snapshot) as snapshot_fd:
        before_info = os.fstat(snapshot_fd)
        before_hash, before_size = _hash_fd(snapshot_fd)
        if not hmac.compare_digest(before_hash, SNAPSHOT_SHA256):
            _raise("snapshot_hash_mismatch")

        uri = (
            f"file:/proc/self/fd/{snapshot_fd}"
            "?mode=ro&immutable=1&cache=private"
        )
        try:
            connection = sqlite3.connect(uri, uri=True)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA temp_store=MEMORY")
        except sqlite3.Error:
            _raise("snapshot_open_failed")

        try:
            quick_check = connection.execute("PRAGMA quick_check").fetchone()
            if quick_check is None or quick_check[0] != "ok":
                _raise("snapshot_integrity_failed")

            event_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(recall_events)")
            }
            node_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(nodes)")
            }
            if not set(RECALL_EVENT_COLUMNS).issubset(event_columns):
                _raise("snapshot_event_schema_mismatch")
            if not set(NODE_COLUMNS).issubset(node_columns):
                _raise("snapshot_node_schema_mismatch")

            selected_rows = connection.execute(
                f"""
                SELECT {', '.join(RECALL_EVENT_COLUMNS)}
                FROM recall_events
                WHERE created_at > ?
                  AND created_at < ?
                  AND requested_scope IN (?, ?)
                ORDER BY created_at ASC, id ASC
                """,
                (
                    START_EXCLUSIVE,
                    END_EXCLUSIVE,
                    REQUESTED_SCOPES[0],
                    REQUESTED_SCOPES[1],
                ),
            ).fetchall()

            events: list[dict[str, Any]] = []
            referenced: dict[str, set[str]] = {}
            for sql_row in selected_rows:
                row = {name: sql_row[name] for name in RECALL_EVENT_COLUMNS}
                if not isinstance(row["id"], str) or not isinstance(row["query"], str):
                    _raise("snapshot_event_required_string_invalid")
                if row["requested_scope"] not in REQUESTED_SCOPES:
                    _raise("snapshot_selection_scope_mismatch")
                if not (
                    isinstance(row["created_at"], str)
                    and START_EXCLUSIVE < row["created_at"] < END_EXCLUSIVE
                ):
                    _raise("snapshot_selection_time_mismatch")
                resolved = _load_json_field(
                    row["resolved_scopes"], list, "resolved_scopes"
                )
                ambient = _load_json_field(
                    row["ambient_context"], dict, "ambient_context"
                )
                raw_results = _load_json_field(row["results"], list, "results")
                if not all(isinstance(item, str) for item in resolved):
                    _raise("resolved_scopes_item_invalid")
                results = [_validate_result(item) for item in raw_results]
                row["resolved_scopes"] = resolved
                row["ambient_context"] = ambient
                row["results"] = results
                for result in results:
                    referenced.setdefault(result["node_id"], set()).add("results")
                if row["feedback_trace_id"] is not None:
                    if not isinstance(row["feedback_trace_id"], str):
                        _raise("feedback_trace_id_invalid")
                    referenced.setdefault(row["feedback_trace_id"], set()).add(
                        "feedback"
                    )
                events.append(row)

            direct_node_ids = sorted(referenced)
            if len(direct_node_ids) != EXPECTED_NODE_COUNT:
                _raise("snapshot_node_count_mismatch")

            # Reproduce the original packet's one-hop typed-edge closure:
            # supersedes/contradicts partners are replay evidence, while the
            # generic related graph is deliberately excluded.
            direct_placeholders = ",".join("?" for _ in direct_node_ids)
            partner_rows = connection.execute(
                f"""
                SELECT source_id, target_id
                FROM connections
                WHERE type IN ('supersedes', 'contradicts')
                  AND (
                    source_id IN ({direct_placeholders})
                    OR target_id IN ({direct_placeholders})
                  )
                """,
                [*direct_node_ids, *direct_node_ids],
            ).fetchall()
            node_id_set = set(direct_node_ids)
            for partner_row in partner_rows:
                node_id_set.update(
                    (partner_row["source_id"], partner_row["target_id"])
                )
            node_ids = sorted(node_id_set)
            placeholders = ",".join("?" for _ in node_ids)
            node_rows = connection.execute(
                f"""
                SELECT {', '.join(NODE_COLUMNS)}
                FROM nodes
                WHERE id IN ({placeholders})
                ORDER BY id ASC
                """,
                node_ids,
            ).fetchall()
            nodes = {
                row["id"]: {name: row[name] for name in NODE_COLUMNS}
                for row in node_rows
            }
            if set(nodes) != set(node_ids):
                _raise("snapshot_node_containment_failed")

            connection_rows = connection.execute(
                f"""
                SELECT source_id, target_id, type, weight, created_at
                FROM connections
                WHERE type IN ('supersedes', 'contradicts')
                  AND source_id IN ({placeholders})
                  AND target_id IN ({placeholders})
                ORDER BY type ASC, source_id ASC, target_id ASC, created_at ASC
                """,
                [*node_ids, *node_ids],
            ).fetchall()
            connections = [dict(row) for row in connection_rows]
        except sqlite3.Error:
            _raise("snapshot_query_failed")
        finally:
            connection.close()

        after_hash, after_size = _hash_fd(snapshot_fd)
        after_info = os.fstat(snapshot_fd)
        if (
            not hmac.compare_digest(after_hash, SNAPSHOT_SHA256)
            or before_hash != after_hash
            or before_size != after_size
            or (
                before_info.st_dev,
                before_info.st_ino,
                before_info.st_size,
                before_info.st_mtime_ns,
            )
            != (
                after_info.st_dev,
                after_info.st_ino,
                after_info.st_size,
                after_info.st_mtime_ns,
            )
        ):
            _raise("snapshot_changed_during_read")

    final_hash, final_size = _hash_regular(snapshot)
    if not hmac.compare_digest(final_hash, SNAPSHOT_SHA256) or final_size != before_size:
        _raise("snapshot_path_changed_during_read")
    return events, nodes, connections


def _nullable_surrogate(
    value: Any, pool: Any
) -> tuple[str | None, int | None]:
    if value is None:
        return None, None
    if not isinstance(value, str):
        _raise("private_string_field_invalid")
    surrogate = pool.surrogate(value)
    return surrogate, len(value)


def _build_node_record(
    row: Mapping[str, Any],
    pool: Any,
    deid: ModuleType,
    known_scopes: frozenset[str],
    included_via: Sequence[str],
    relations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    context = _load_json_field(row["context"], dict, "node_context")
    source_traces = _load_json_field(
        row["source_traces"], list, "node_source_traces"
    )
    corrections = _load_json_field(row["corrections"], list, "node_corrections")
    provenance = _load_json_field(row["provenance"], dict, "node_provenance")

    agent_surrogate, agent_chars = _nullable_surrogate(row["agent"], pool)
    task_surrogate, task_chars = _nullable_surrogate(row["task"], pool)
    decay_surrogate, decay_chars = _nullable_surrogate(row["decay_reason"], pool)

    temporal_hint = row["temporal_hint"]
    if temporal_hint is not None:
        if not isinstance(temporal_hint, str):
            _raise("temporal_hint_invalid")
        if not deid.TEMPORAL_HINT_RE.match(temporal_hint):
            temporal_hint = pool.surrogate(temporal_hint)

    correction_records: list[dict[str, Any]] = []
    correction_chars: list[dict[str, int]] = []
    for correction in corrections:
        if not isinstance(correction, dict):
            _raise("node_correction_shape")
        entry = dict(correction)
        chars: dict[str, int] = {}
        for key in ("old", "new"):
            if isinstance(entry.get(key), str):
                chars[key] = len(entry[key])
                entry[key] = pool.surrogate(entry[key])
        for key, value in list(entry.items()):
            if key not in ("old", "new"):
                entry[key] = deid.transform_value(value, pool, known_scopes)
        correction_records.append(entry)
        correction_chars.append(chars)

    content = row["content"]
    if not isinstance(content, str):
        _raise("node_content_invalid")
    return {
        "type": "node",
        "id": row["id"],
        "level": row["level"],
        "scope": row["scope"],
        "included_via": list(included_via),
        "agent_surrogate": agent_surrogate,
        "agent_chars": agent_chars,
        "task_surrogate": task_surrogate,
        "task_chars": task_chars,
        "timestamp": row["timestamp"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "decayed": row["decayed"],
        "decay_reason_surrogate": decay_surrogate,
        "decay_reason_chars": decay_chars,
        "stats": {
            "access_count": row["access_count"],
            "usefulness_score": row["usefulness_score"],
            "confidence": row["confidence"],
            "unique_agents": row["unique_agents"],
            "last_accessed": row["last_accessed"],
            "temporal_hint": temporal_hint,
        },
        "content_surrogate": pool.surrogate(content),
        "content_chars": len(content),
        "context_surrogate": deid.transform_value(context, pool, known_scopes),
        "context_chars": deid.context_chars_map(context),
        "source_traces": source_traces,
        "corrections": correction_records,
        "corrections_chars": correction_chars,
        "provenance_shape": deid.provenance_shape_of(
            provenance, source_traces, corrections
        ),
        "relations": list(relations),
    }


def _build_event_record(
    row: Mapping[str, Any],
    token: str,
    pool: Any,
    deid: ModuleType,
    known_scopes: frozenset[str],
    supersedes_endpoints: set[str],
) -> dict[str, Any]:
    query_surrogate, query_chars = _nullable_surrogate(row["query"], pool)
    agent_surrogate, agent_chars = _nullable_surrogate(row["agent"], pool)
    task_surrogate, task_chars = _nullable_surrogate(row["task"], pool)
    session_surrogate, session_chars = _nullable_surrogate(row["session_id"], pool)

    transport = row["transport_session_id"]
    if transport is not None:
        if not isinstance(transport, str):
            _raise("transport_session_id_invalid")
        if not deid.HEX32_RE.match(transport):
            transport = pool.surrogate(transport)

    result_node_ids = {item["node_id"] for item in row["results"]}
    return {
        "type": "event",
        "id": row["id"],
        "source": "local",
        "created_at": row["created_at"],
        "scope": row["scope"],
        "requested_scope": row["requested_scope"],
        "resolved_scopes": row["resolved_scopes"],
        "depth": row["depth"],
        "max_results": row["max_results"],
        "class": deid.classify_class(row["agent"]),
        "template_id": deid.classify_template(row["query"]),
        "fingerprint_token": token,
        "query_surrogate": query_surrogate,
        "query_chars": query_chars,
        "agent_surrogate": agent_surrogate,
        "agent_chars": agent_chars,
        "task_surrogate": task_surrogate,
        "task_chars": task_chars,
        "session_id_surrogate": session_surrogate,
        "session_id_chars": session_chars,
        "transport_session_id": transport,
        "ambient_context_surrogate": deid.transform_value(
            row["ambient_context"], pool, known_scopes
        ),
        "ambient_context_chars": deid.context_chars_map(row["ambient_context"]),
        "results": row["results"],
        "feedback_applied": row["feedback_applied"],
        "feedback_trace_id": row["feedback_trace_id"],
        "feedback_applied_at": row["feedback_applied_at"],
        "supersedes_bearing": bool(result_node_ids & supersedes_endpoints),
        # This replacement source has no transcript capture.  Preserve the
        # original replay schema while stating the absence rather than
        # fabricating a measurement.
        "transcript_matched": False,
        "transcript_serialized_chars": None,
    }


def _relations_by_node(
    node_ids: set[str], connections: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    relations: dict[str, list[dict[str, Any]]] = {node_id: [] for node_id in node_ids}
    supersedes_endpoints: set[str] = set()
    for edge in connections:
        source_id = edge.get("source_id")
        target_id = edge.get("target_id")
        edge_type = edge.get("type")
        if source_id not in node_ids or target_id not in node_ids:
            _raise("connection_containment_failed")
        if not all(
            isinstance(value, str)
            for value in (source_id, target_id, edge_type, edge.get("created_at"))
        ):
            _raise("connection_schema_invalid")
        if isinstance(edge.get("weight"), bool) or not isinstance(
            edge.get("weight"), (int, float)
        ):
            _raise("connection_weight_invalid")
        relations[source_id].append(
            {
                "type": edge_type,
                "direction": "out",
                "other_id": target_id,
                "weight": edge["weight"],
                "created_at": edge["created_at"],
            }
        )
        if target_id != source_id:
            relations[target_id].append(
                {
                    "type": edge_type,
                    "direction": "in",
                    "other_id": source_id,
                    "weight": edge["weight"],
                    "created_at": edge["created_at"],
                }
            )
        if edge_type == "supersedes":
            supersedes_endpoints.update((source_id, target_id))
    for values in relations.values():
        values.sort(
            key=lambda item: (
                item["type"],
                item["direction"],
                item["other_id"],
                item["created_at"],
            )
        )
    return relations, supersedes_endpoints


def _json_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _json_document(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")


def _fsync_directory_fd(fd: int) -> None:
    try:
        os.fsync(fd)
    except OSError:
        _raise("directory_sync_failed")


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        try:
            written = os.write(fd, view[offset:])
        except OSError:
            _raise("file_write_failed")
        if written <= 0:
            _raise("file_write_failed")
        offset += written


def _atomic_write_new_at(
    parent_fd: int, name: str, data: bytes, *, mode: int
) -> dict[str, Any]:
    if not name or "/" in name or name in (".", ".."):
        _raise("output_name_invalid")
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        _raise("output_exists")

    temp = "." + name + ".tmp-" + secrets.token_hex(12)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = -1
    try:
        fd = os.open(temp, flags, 0o600, dir_fd=parent_fd)
        _write_all(fd, data)
        os.fchmod(fd, mode)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.link(
            temp,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.unlink(temp, dir_fd=parent_fd)
        _fsync_directory_fd(parent_fd)
    except BuildError:
        raise
    except OSError:
        _raise("atomic_output_install_failed")
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temp, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        except OSError:
            pass
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _atomic_write_relative(
    root_fd: int, relative: Path, data: bytes, *, mode: int
) -> dict[str, Any]:
    with _relative_parent_fd(root_fd, relative) as (parent_fd, name):
        return _atomic_write_new_at(parent_fd, name, data, mode=mode)


def _prepare_draft(path: Path) -> tuple[Path, int, tuple[int, int, int, int]]:
    draft = _normalized_path(path)
    if draft == Path(draft.anchor):
        _raise("draft_path_invalid")
    draft_fd = _open_directory_chain(draft, create_leaf=True, leaf_mode=0o700)
    identity = _directory_identity(draft_fd)
    if identity[2] != os.geteuid():
        os.close(draft_fd)
        _raise("draft_owner_mismatch")
    if identity[3] != 0o700:
        os.close(draft_fd)
        _raise("draft_permissions_not_private")
    try:
        if os.listdir(draft_fd):
            os.close(draft_fd)
            _raise("draft_not_empty")
    except OSError:
        os.close(draft_fd)
        _raise("draft_scan_failed")
    return draft, draft_fd, identity


def _validate_path_boundaries(
    args: argparse.Namespace,
    draft: Path,
    publish: Path | None,
    *,
    canonical_root: Path,
) -> None:
    staging = _normalized_path(args.staging)
    original_root = _normalized_path(ORIGINAL_PACKET_ROOT)
    for source in (
        staging,
        _normalized_path(args.snapshot),
        original_root,
    ):
        if _paths_overlap(draft, source):
            _raise("draft_source_overlap")
    if publish is None:
        return
    if _paths_overlap(draft, publish):
        _raise("draft_publish_overlap")
    for source in (
        staging,
        _normalized_path(args.snapshot),
        _normalized_path(args.original_manifest),
        _normalized_path(args.splits),
        _normalized_path(DEID_PATH),
    ):
        if _paths_overlap(publish, source) and publish != canonical_root:
            _raise("publish_source_overlap")
    if _paths_overlap(publish, original_root) and publish != canonical_root:
        _raise("publish_original_packet_boundary")


def _preflight_output_boundaries(
    args: argparse.Namespace,
    draft: Path,
    publish: Path | None,
    *,
    canonical_root: Path,
) -> None:
    """Reject output/source overlap before creating either output root."""

    draft_candidate = _normalized_path(draft)
    publish_candidate = _normalized_path(publish) if publish is not None else None
    staging = _normalized_path(args.staging)
    original_root = _normalized_path(ORIGINAL_PACKET_ROOT)
    for source in (staging, _normalized_path(args.snapshot), original_root):
        if _paths_overlap(draft_candidate, source):
            _raise("draft_source_overlap")
    if publish_candidate is None:
        return
    if _paths_overlap(draft_candidate, publish_candidate):
        _raise("draft_publish_overlap")
    for source in (
        staging,
        _normalized_path(args.snapshot),
        _normalized_path(args.original_manifest),
        _normalized_path(args.splits),
        _normalized_path(DEID_PATH),
    ):
        if _paths_overlap(
            publish_candidate, source
        ) and publish_candidate != canonical_root:
            _raise("publish_source_overlap")
    if _paths_overlap(publish_candidate, original_root) and publish_candidate != (
        canonical_root
    ):
        _raise("publish_original_packet_boundary")


def _cleanup_fixed_draft_outputs(draft_fd: int) -> None:
    for relative in (DRAFT_MANIFEST_REL,):
        try:
            os.unlink(relative.name, dir_fd=draft_fd)
        except FileNotFoundError:
            pass
        except OSError:
            pass
    try:
        corpus_fd = _open_child_directory(draft_fd, "corpus")
    except BuildError:
        corpus_fd = -1
    if corpus_fd >= 0:
        try:
            for relative in DATA_FILE_RELS:
                try:
                    os.unlink(relative.name, dir_fd=corpus_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
        finally:
            os.close(corpus_fd)
    try:
        os.rmdir("corpus", dir_fd=draft_fd)
    except OSError:
        pass


def _build_candidate(
    args: argparse.Namespace,
    draft: Path,
    draft_fd: int,
    *,
    run_verifier: bool,
) -> dict[str, Any]:
    pins = _verify_public_source_pins(args)
    deid = _import_deid()
    identity_key = bytearray(os.urandom(IDENTITY_KEY_BYTES))
    surrogate_salt = bytearray(os.urandom(SURROGATE_SALT_BYTES))
    token_identities: dict[str, tuple[str, str]] = {}
    receipt: dict[str, Any] | None = None
    verifier_sha256: str | None = None

    try:
        staging_events, staging_info = _load_staging_events(
            _normalized_path(args.staging) / "alt-db" / "recall_events.jsonl"
        )
        dev_total = 0
        dev_automatic = 0
        dev_organic = 0
        dev_tokens: list[str] = []
        for row in staging_events:
            event_id = row.get("id")
            if not isinstance(event_id, str):
                _raise("staging_event_id_invalid")
            bucket = _bucket_of(event_id)
            if not (DEV_HOLDOUT_BOUND <= bucket < DEV_BOUND):
                continue
            dev_total += 1
            if row.get("agent") is None:
                dev_automatic += 1
                query = row.get("query")
                requested_scope = row.get("requested_scope")
                if not isinstance(query, str) or not isinstance(requested_scope, str):
                    _raise("staging_identity_field_invalid")
                token, identity = _fingerprint_token(
                    identity_key, query, requested_scope
                )
                prior = token_identities.setdefault(token, identity)
                if prior != identity:
                    _raise("fingerprint_token_collision")
                dev_tokens.append(token)
            else:
                dev_organic += 1

        unique_dev_tokens = sorted(set(dev_tokens))
        if (
            dev_total != EXPECTED_DEV_EVENTS
            or dev_automatic != EXPECTED_DEV_AUTOMATIC
            or dev_organic != EXPECTED_DEV_ORGANIC
            or len(unique_dev_tokens) != EXPECTED_DEV_UNIQUE_TOKENS
        ):
            _raise("dev_population_mismatch")

        events, nodes, connections = _select_snapshot_population(args.snapshot)
        if len(events) != EXPECTED_SUPPLEMENT_EVENTS:
            _raise("supplement_population_mismatch")
        event_ids = [row["id"] for row in events]
        if len(event_ids) != len(set(event_ids)):
            _raise("supplement_duplicate_event_id")
        if set(event_ids) & {row["id"] for row in staging_events}:
            _raise("supplement_original_alt_overlap")

        class_counts = collections.Counter(
            "automatic" if row["agent"] is None else "organic" for row in events
        )
        scope_counts = collections.Counter(row["requested_scope"] for row in events)
        if (
            class_counts
            != collections.Counter(
                {
                    "automatic": EXPECTED_AUTOMATIC_EVENTS,
                    "organic": EXPECTED_ORGANIC_EVENTS,
                }
            )
            or dict(scope_counts) != EXPECTED_SCOPE_COUNTS
        ):
            _raise("supplement_aggregate_mismatch")

        event_tokens: dict[str, str] = {}
        automatic_groups: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(
            list
        )
        for row in events:
            token, identity = _fingerprint_token(
                identity_key, row["query"], row["requested_scope"]
            )
            prior = token_identities.setdefault(token, identity)
            if prior != identity:
                _raise("fingerprint_token_collision")
            event_tokens[row["id"]] = token
            if row["agent"] is None:
                automatic_groups[token].append(row)

        repeated = {
            token: family
            for token, family in automatic_groups.items()
            if len(family) >= 3
            and len({row["transport_session_id"] for row in family}) >= 2
        }
        repeated_events = sum(len(family) for family in repeated.values())
        dev_token_set = set(unique_dev_tokens)
        unseen = {
            token: family
            for token, family in repeated.items()
            if token not in dev_token_set
        }
        unseen_events = sum(len(family) for family in unseen.values())
        if (
            len(repeated) != EXPECTED_REPEATED_FAMILIES
            or repeated_events != EXPECTED_REPEATED_EVENTS
            or len(unseen) != EXPECTED_UNSEEN_FAMILIES
            or unseen_events != EXPECTED_UNSEEN_EVENTS
        ):
            _raise("repeated_family_aggregate_mismatch")

        known_scopes: set[str] = {"global"}
        for row in events:
            for value in (row["scope"], row["requested_scope"]):
                if isinstance(value, str) and value:
                    known_scopes.add(value)
            known_scopes.update(
                value for value in row["resolved_scopes"] if isinstance(value, str)
            )
            known_scopes.update(result["scope"] for result in row["results"])
        for row in nodes.values():
            if isinstance(row["scope"], str) and row["scope"]:
                known_scopes.add(row["scope"])
        frozen_scopes = frozenset(known_scopes)

        node_reasons: dict[str, set[str]] = collections.defaultdict(set)
        for row in events:
            for result in row["results"]:
                node_reasons[result["node_id"]].add("results")
            if row["feedback_trace_id"]:
                node_reasons[row["feedback_trace_id"]].add("feedback")
        if not set(node_reasons).issubset(nodes):
            _raise("node_reason_containment_failed")
        for node_id in set(nodes) - set(node_reasons):
            node_reasons[node_id].add("typed_edge_partner")

        relations, supersedes_endpoints = _relations_by_node(
            set(nodes), connections
        )
        pool = deid.SurrogatePool(bytes(surrogate_salt))
        node_records = [
            _build_node_record(
                nodes[node_id],
                pool,
                deid,
                frozen_scopes,
                sorted(node_reasons[node_id]),
                relations[node_id],
            )
            for node_id in sorted(nodes)
        ]
        if len(node_records) != EXPECTED_TOTAL_NODE_COUNT:
            _raise("supplement_total_node_count_mismatch")
        event_records = [
            _build_event_record(
                row,
                event_tokens[row["id"]],
                pool,
                deid,
                frozen_scopes,
                supersedes_endpoints,
            )
            for row in events
        ]
        if [(row["created_at"], row["id"]) for row in events] != sorted(
            (row["created_at"], row["id"]) for row in events
        ):
            # The SQL ORDER BY is the authoritative order.  This assertion is
            # intentionally value-free on failure.
            _raise("event_order_mismatch")
        if any(
            record["class"]
            != ("automatic" if row["agent"] is None else "organic")
            for record, row in zip(event_records, events)
        ):
            _raise("event_classification_mismatch")

        corpus_bytes = b"".join(
            _json_line(record) for record in [*node_records, *event_records]
        )
        index_document = {
            "schema_version": 1,
            "kind": "frozen-dev-automatic-fingerprint-token-index",
            "algorithm": "HMAC-SHA256",
            "normalization": '" ".join(query.split()) + "\\n" + requested_scope',
            "population_events": EXPECTED_DEV_AUTOMATIC,
            "unique_tokens": EXPECTED_DEV_UNIQUE_TOKENS,
            "tokens": unique_dev_tokens,
        }
        index_bytes = _json_document(index_document)

        corpus_fd = _open_child_directory(
            draft_fd, "corpus", create=True, mode=0o700
        )
        try:
            corpus_info = _atomic_write_new_at(
                corpus_fd, DRAFT_CORPUS_REL.name, corpus_bytes, mode=0o400
            )
            corpus_info.update(
                {
                    "records": len(node_records) + len(event_records),
                    "nodes": len(node_records),
                    "events": len(event_records),
                }
            )
            index_info = _atomic_write_new_at(
                corpus_fd, DRAFT_INDEX_REL.name, index_bytes, mode=0o400
            )
            index_info.update(
                {
                    "population_events": EXPECTED_DEV_AUTOMATIC,
                    "unique_tokens": len(unique_dev_tokens),
                }
            )
        finally:
            os.close(corpus_fd)

        before_verify_hashes = {
            DRAFT_CORPUS_REL.as_posix(): corpus_info["sha256"],
            DRAFT_INDEX_REL.as_posix(): index_info["sha256"],
        }
        if run_verifier:
            receipt, verifier_sha256 = _run_external_verifier(
                args, draft, draft_fd, identity_key
            )
            for relative, expected_hash in before_verify_hashes.items():
                actual_hash, _ = _hash_relative(draft_fd, Path(relative))
                if not hmac.compare_digest(actual_hash, expected_hash):
                    _raise("verifier_mutated_candidate")

        # Pin every private source again after the verifier has closed its own
        # read-only handles.  No private value is returned in this metadata.
        final_pins = _verify_public_source_pins(args)
        if final_pins != pins:
            _raise("source_pin_changed")
        if staging_info != pins["staging_events"] | {"rows": STAGING_EVENTS_ROWS}:
            _raise("staging_source_info_mismatch")

        validation = (
            dict(receipt)
            if receipt is not None
            else {
                "status": "not-run",
                "mode": "draft-only",
                "semantic_reads": 0,
            }
        )
        return {
            "source_pins": pins,
            "files": {
                DRAFT_CORPUS_REL.as_posix(): corpus_info,
                DRAFT_INDEX_REL.as_posix(): index_info,
            },
            "counts": {
                "supplement": {
                    "events": EXPECTED_SUPPLEMENT_EVENTS,
                    "direct_nodes": EXPECTED_NODE_COUNT,
                    "nodes": len(node_records),
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
            },
            "validation": validation,
            "verifier_sha256": verifier_sha256,
        }
    finally:
        for index in range(len(identity_key)):
            identity_key[index] = 0
        for index in range(len(surrogate_salt)):
            surrogate_salt[index] = 0


def _write_pipe_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        try:
            written = os.write(fd, data[offset:])
        except OSError:
            _raise("anonymous_descriptor_write_failed")
        if written <= 0:
            _raise("anonymous_descriptor_write_failed")
        offset += written


def _read_pipe_limited(
    fd: int, limit: int, *, timeout_seconds: float | None = None
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    deadline = (
        time.monotonic() + timeout_seconds
        if timeout_seconds is not None
        else None
    )
    while True:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _raise("anonymous_descriptor_timeout")
            try:
                readable, _, _ = select.select([fd], [], [], remaining)
            except OSError:
                _raise("anonymous_descriptor_read_failed")
            if not readable:
                _raise("anonymous_descriptor_timeout")
        try:
            chunk = os.read(fd, min(4096, limit + 1 - total))
        except OSError:
            _raise("anonymous_descriptor_read_failed")
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            _raise("verifier_receipt_too_large")
    return b"".join(chunks)


def _validate_receipt(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > MAX_RECEIPT_BYTES:
        _raise("verifier_receipt_invalid")
    receipt = _load_json(raw, "verifier_receipt")
    if not isinstance(receipt, dict):
        _raise("verifier_receipt_shape")
    if set(receipt) != set(RECEIPT_REQUIRED):
        _raise("verifier_receipt_shape")
    for field, expected in RECEIPT_REQUIRED.items():
        actual = receipt.get(field)
        if type(actual) is not type(expected) or actual != expected:
            _raise("verifier_receipt_failed")
    # Return a freshly constructed allowlist only.  Even if a future verifier
    # adds diagnostics, no unreviewed value can cross into the manifest.
    return {field: expected for field, expected in RECEIPT_REQUIRED.items()}


def _disable_core_dumps() -> None:
    """Best-effort inherited hardening for processes that handle raw data."""

    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError):
        pass


@contextmanager
def _termination_guard() -> Iterator[None]:
    """Turn catchable launcher termination into ordinary worker cleanup."""

    previous: dict[signal.Signals, Any] = {}

    def interrupted(_signum: int, _frame: Any) -> None:
        _raise("launcher_interrupted")

    for signum in (signal.SIGTERM, signal.SIGHUP):
        try:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        except (OSError, ValueError):
            previous.pop(signum, None)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):
                pass


def _sealed_verifier_memfd(expected_sha256: str) -> tuple[int, str]:
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        _raise("verifier_pin_invalid")
    _assert_no_symlink_components(VERIFIER_PATH, allow_missing_leaf=False)
    source = _read_regular(VERIFIER_PATH, max_bytes=8 * 1024 * 1024)
    digest = hashlib.sha256(source).hexdigest()
    if not hmac.compare_digest(digest, expected_sha256):
        _raise("verifier_hash_mismatch")
    if not hasattr(os, "memfd_create"):
        _raise("sealed_verifier_unsupported")
    flags = getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0)
    fd = -1
    try:
        fd = os.memfd_create("replacement-packet-verifier", flags)
        _write_all(fd, source)
        os.lseek(fd, 0, os.SEEK_SET)
        seals = (
            getattr(fcntl, "F_SEAL_SEAL", 0)
            | getattr(fcntl, "F_SEAL_SHRINK", 0)
            | getattr(fcntl, "F_SEAL_GROW", 0)
            | getattr(fcntl, "F_SEAL_WRITE", 0)
        )
        if not seals or not hasattr(fcntl, "F_ADD_SEALS"):
            _raise("sealed_verifier_unsupported")
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals)
    except BuildError:
        if fd >= 0:
            os.close(fd)
        raise
    except OSError:
        if fd >= 0:
            os.close(fd)
        _raise("sealed_verifier_create_failed")
    return fd, digest


def _run_external_verifier(
    args: argparse.Namespace,
    draft: Path,
    draft_fd: int,
    identity_key: bytearray,
) -> tuple[dict[str, Any], str]:
    verifier_fd, verifier_hash = _sealed_verifier_memfd(args.verifier_sha256)
    key_read, key_write = os.pipe2(os.O_CLOEXEC)
    receipt_read, receipt_write = os.pipe2(os.O_CLOEXEC)
    command = [
        sys.executable,
        "-I",
        f"/proc/self/fd/{verifier_fd}",
        "keyed",
        "--staging",
        os.fspath(_normalized_path(args.staging)),
        "--snapshot",
        os.fspath(_normalized_path(args.snapshot)),
        "--original-manifest",
        os.fspath(_normalized_path(args.original_manifest)),
        "--splits",
        os.fspath(_normalized_path(args.splits)),
        "--draft-dir",
        os.fspath(draft),
        "--draft-dir-fd",
        str(draft_fd),
        "--identity-key-fd",
        str(key_read),
        "--receipt-fd",
        str(receipt_write),
    ]
    environment = {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    process: subprocess.Popen[bytes] | None = None
    try:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                pass_fds=(verifier_fd, draft_fd, key_read, receipt_write),
                env=environment,
                preexec_fn=_disable_core_dumps,
            )
        except OSError:
            _raise("verifier_start_failed")
        os.close(key_read)
        key_read = -1
        os.close(receipt_write)
        receipt_write = -1
        _write_pipe_all(key_write, bytes(identity_key))
        os.close(key_write)
        key_write = -1
        try:
            return_code = process.wait(timeout=VERIFIER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            _raise("verifier_timeout")
        receipt_raw = _read_pipe_limited(
            receipt_read, MAX_RECEIPT_BYTES, timeout_seconds=5.0
        )
        os.close(receipt_read)
        receipt_read = -1
        if return_code != 0:
            _raise("verifier_failed")
        receipt = _validate_receipt(receipt_raw)
        return receipt, verifier_hash
    finally:
        for fd in (
            verifier_fd,
            key_read,
            key_write,
            receipt_read,
            receipt_write,
        ):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def _send_worker_status(fd: int, value: Mapping[str, Any]) -> None:
    data = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    try:
        _write_pipe_all(fd, data)
    except BuildError:
        pass


def _run_keyed_worker(
    args: argparse.Namespace,
    draft: Path,
    draft_fd: int,
    *,
    run_verifier: bool,
) -> dict[str, Any]:
    status_read, status_write = os.pipe2(os.O_CLOEXEC)
    try:
        pid = os.fork()
    except OSError:
        os.close(status_read)
        os.close(status_write)
        _raise("worker_fork_failed")
    if pid == 0:
        os.close(status_read)
        try:
            os.setsid()
            _disable_core_dumps()
            result = _build_candidate(
                args, draft, draft_fd, run_verifier=run_verifier
            )
            _send_worker_status(status_write, {"status": "ok", "result": result})
            exit_code = 0
        except BuildError as error:
            _cleanup_fixed_draft_outputs(draft_fd)
            _send_worker_status(
                status_write, {"status": "error", "code": error.code}
            )
            exit_code = 1
        except BaseException:
            _cleanup_fixed_draft_outputs(draft_fd)
            _send_worker_status(
                status_write, {"status": "error", "code": "internal_failure"}
            )
            exit_code = 1
        finally:
            # Deliberately leave the status descriptor open until os._exit.
            # EOF then proves that the process-group leader has exited (and
            # is still a reserved zombie PID), so the parent can safely kill
            # any verifier descendants before reaping that leader.
            pass
        os._exit(exit_code)

    os.close(status_write)
    status_write = -1
    waited = False
    try:
        with _termination_guard():
            raw = _read_pipe_limited(
                status_read,
                64 * 1024,
                timeout_seconds=(
                    VERIFIER_TIMEOUT_SECONDS + 120 if run_verifier else 120
                ),
            )
            os.close(status_read)
            status_read = -1
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            _, wait_status = os.waitpid(pid, 0)
            waited = True
    finally:
        if status_read >= 0:
            try:
                os.close(status_read)
            except OSError:
                pass
        if not waited:
            try:
                os.killpg(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    waited_pid, wait_status = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    waited = True
                    break
                if waited_pid == pid:
                    waited = True
                    break
                time.sleep(0.02)
            if not waited:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                try:
                    _, wait_status = os.waitpid(pid, 0)
                except ChildProcessError:
                    pass
                waited = True
    if not os.WIFEXITED(wait_status) or os.WEXITSTATUS(wait_status) != 0:
        if raw:
            parsed = _load_json(raw, "worker_status")
            if isinstance(parsed, dict) and isinstance(parsed.get("code"), str):
                _raise(parsed["code"])
        _raise("worker_failed")
    parsed = _load_json(raw, "worker_status")
    if (
        not isinstance(parsed, dict)
        or parsed.get("status") != "ok"
        or not isinstance(parsed.get("result"), dict)
    ):
        _raise("worker_status_invalid")
    return parsed["result"]


def _assert_exact_draft_tree(draft_fd: int, *, include_manifest: bool) -> None:
    expected_root = {"corpus"}
    if include_manifest:
        expected_root.add("manifest.json")
    try:
        root_entries = set(os.listdir(draft_fd))
        corpus_fd = _open_child_directory(draft_fd, "corpus")
        try:
            corpus_entries = set(os.listdir(corpus_fd))
        finally:
            os.close(corpus_fd)
    except OSError:
        _raise("draft_tree_scan_failed")
    if root_entries != expected_root or corpus_entries != {
        "holdout.jsonl",
        "dev-fingerprint-index.json",
    }:
        _raise("draft_tree_not_allowlisted")


def _rehash_candidate(
    draft_fd: int, expected: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    _assert_exact_draft_tree(draft_fd, include_manifest=False)
    result: dict[str, dict[str, Any]] = {}
    for relative in DATA_FILE_RELS:
        key = relative.as_posix()
        if key not in expected:
            _raise("worker_file_metadata_missing")
        digest, size = _hash_relative(draft_fd, relative)
        if (
            digest != expected[key].get("sha256")
            or size != expected[key].get("bytes")
        ):
            _raise("candidate_changed_after_worker_exit")
        result[key] = dict(expected[key])
    return result


def _manifest_document(
    result: Mapping[str, Any],
    *,
    frozen: bool,
    packet_files: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    pins = result["source_pins"]
    validation = result["validation"]
    if frozen and validation.get("status") != "pass":
        _raise("cannot_freeze_without_validation")
    return {
        "schema_version": 1,
        "packet": "animal-planet P6 replacement temporal supplement",
        "namespace": "replacement-holdout",
        "frozen": frozen,
        "generated_at": _utc_now(),
        "semantic_reads": 0,
        "selection": {
            "source": "pinned immutable SQLite main snapshot",
            "predicate": {
                "created_at": {
                    "gt": START_EXCLUSIVE,
                    "lt": END_EXCLUSIVE,
                },
                "requested_scope_in": list(REQUESTED_SCOPES),
            },
            "outcome_filtering": False,
            "ordering": ["created_at ASC", "id ASC"],
        },
        "classification": {
            "automatic": "agent IS NULL",
            "organic": "agent IS NOT NULL",
        },
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
        "sources": {
            "staging_metadata": pins["staging_metadata"],
            "staging_events": {
                **pins["staging_events"],
                "rows": STAGING_EVENTS_ROWS,
            },
            "sqlite_snapshot": {
                **pins["snapshot"],
                "open_mode": "mode=ro&immutable=1&cache=private",
                "query_only": True,
                "temp_store": "MEMORY",
            },
            "original_packet_manifest": {
                **pins["original_manifest"],
                "frozen": True,
            },
            "original_splits": pins["original_splits"],
            "deid_implementation": pins["deid_implementation"],
        },
        "implementation": {
            "builder": {
                "path": "recipe/build.py",
                **pins["builder"],
            },
            "verifier": (
                {
                    "path": "recipe/verify.py",
                    "sha256": result["verifier_sha256"],
                    **(
                        {"bytes": packet_files["recipe/verify.py"]["bytes"]}
                        if packet_files
                        else {}
                    ),
                }
                if result.get("verifier_sha256") is not None
                else None
            ),
        },
        "counts": result["counts"],
        "files": result["files"],
        "packet_files": dict(packet_files or {}),
        "validation": {
            "keyed_preseal": validation,
            "candidate_unchanged_after_validation": validation.get("status")
            == "pass",
        },
        "publication": {
            "no_overwrite": True,
            "content_and_index_before_manifest": True,
            "manifest_last": True,
            "canonical_namespace_sealed_iff_manifest_present": frozen,
        },
    }


def _copy_install_no_replace(
    source_root_fd: int,
    source_relative: Path,
    destination_fd: int,
    destination_name: str,
    expected_sha256: str,
) -> None:
    if not destination_name or "/" in destination_name:
        _raise("publish_name_invalid")
    try:
        os.stat(destination_name, dir_fd=destination_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        _raise("publish_target_exists")

    temp = "." + destination_name + ".publish-" + secrets.token_hex(12)
    source_file_fd = -1
    destination_file_fd = -1
    digest = hashlib.sha256()
    try:
        with _relative_parent_fd(source_root_fd, source_relative) as (
            source_parent_fd,
            source_name,
        ):
            source_file_fd = os.open(
                source_name, _open_flags_readonly(), dir_fd=source_parent_fd
            )
        source_before = os.fstat(source_file_fd)
        if not stat.S_ISREG(source_before.st_mode):
            _raise("publish_source_not_regular")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        destination_file_fd = os.open(temp, flags, 0o600, dir_fd=destination_fd)
        while True:
            chunk = os.read(source_file_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            _write_all(destination_file_fd, chunk)
        source_after = os.fstat(source_file_fd)
        if (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
            source_before.st_mtime_ns,
        ) != (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_size,
            source_after.st_mtime_ns,
        ):
            _raise("publish_source_changed")
        if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
            _raise("publish_source_hash_mismatch")
        os.fchmod(destination_file_fd, 0o444)
        os.fsync(destination_file_fd)
        os.close(destination_file_fd)
        destination_file_fd = -1
        os.close(source_file_fd)
        source_file_fd = -1
        os.link(
            temp,
            destination_name,
            src_dir_fd=destination_fd,
            dst_dir_fd=destination_fd,
            follow_symlinks=False,
        )
        os.unlink(temp, dir_fd=destination_fd)
        _fsync_directory_fd(destination_fd)
    except BuildError:
        raise
    except OSError:
        _raise("publish_install_failed")
    finally:
        for fd in (source_file_fd, destination_file_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        try:
            os.unlink(temp, dir_fd=destination_fd)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _rename_directory_no_replace(
    parent_fd: int, source_name: str, destination_name: str
) -> None:
    """Atomically install a directory without replacing a raced target."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError):
        _raise("publish_noreplace_unsupported")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        os.fsencode(source_name),
        parent_fd,
        os.fsencode(destination_name),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            _raise("publish_target_exists")
        _raise("publish_corpus_install_failed")


def _prepare_publish_root(
    path: Path,
    *,
    expected_builder: Mapping[str, Any],
    expected_verifier_sha256: str,
) -> tuple[
    Path,
    int,
    tuple[int, int, int, int],
    dict[str, dict[str, Any]],
]:
    publish = _normalized_path(path)
    publish_fd = _open_directory_chain(publish)
    publish_identity = _directory_identity(publish_fd)
    if publish_identity[2] != os.geteuid() or publish_identity[3] & 0o022:
        os.close(publish_fd)
        _raise("publish_root_permissions_unsafe")
    try:
        if set(os.listdir(publish_fd)) != {"README.md", "POLICY.md", "recipe"}:
            _raise("publish_root_tree_not_allowlisted")
        recipe_fd = _open_child_directory(publish_fd, "recipe")
        try:
            recipe_identity = _directory_identity(recipe_fd)
            if recipe_identity[2] != os.geteuid() or recipe_identity[3] & 0o022:
                _raise("publish_recipe_permissions_unsafe")
            if set(os.listdir(recipe_fd)) != {"build.py", "verify.py"}:
                _raise("publish_recipe_tree_not_allowlisted")
        finally:
            os.close(recipe_fd)
        packet_files = {
            relative.as_posix(): _file_metadata_relative(publish_fd, relative)
            for relative in PACKET_FILE_RELS
        }
        current_builder = _file_metadata_relative(
            publish_fd, Path("recipe/build.py")
        )
        if current_builder != dict(expected_builder):
            _raise("publish_builder_mismatch")
        if packet_files["recipe/verify.py"]["sha256"] != expected_verifier_sha256:
            _raise("publish_verifier_mismatch")
        return publish, publish_fd, publish_identity, packet_files
    except BaseException:
        os.close(publish_fd)
        raise


def _packet_files_for_freeze(
    publish_dir: Path,
    *,
    expected_builder: Mapping[str, Any],
    expected_verifier_sha256: str,
) -> dict[str, dict[str, Any]]:
    _publish, publish_fd, _identity, packet_files = _prepare_publish_root(
        publish_dir,
        expected_builder=expected_builder,
        expected_verifier_sha256=expected_verifier_sha256,
    )
    try:
        return packet_files
    finally:
        os.close(publish_fd)


def _remove_installed_corpus(
    publish_fd: int,
    corpus_fd: int,
    corpus_identity: tuple[int, int, int, int],
) -> None:
    """Rollback only the exact runner-installed unsealed corpus directory."""

    try:
        os.stat(
            DRAFT_MANIFEST_REL.name,
            dir_fd=publish_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    except OSError:
        _raise("publish_rollback_manifest_check_failed")
    else:
        # Manifest presence is the commit marker.  Never remove content once
        # it may have become part of a sealed packet.
        return
    _require_directory_identity(corpus_fd, corpus_identity, "rollback_corpus")
    _require_child_directory_identity(
        publish_fd, "corpus", corpus_identity, "rollback_corpus_binding"
    )
    if set(os.listdir(corpus_fd)) != {
        DRAFT_CORPUS_REL.name,
        DRAFT_INDEX_REL.name,
    }:
        _raise("publish_rollback_tree_changed")
    try:
        os.unlink(DRAFT_CORPUS_REL.name, dir_fd=corpus_fd)
        os.unlink(DRAFT_INDEX_REL.name, dir_fd=corpus_fd)
        _fsync_directory_fd(corpus_fd)
        os.rmdir("corpus", dir_fd=publish_fd)
        _fsync_directory_fd(publish_fd)
    except OSError:
        _raise("publish_rollback_failed")


def _remove_staging_corpus(
    publish_fd: int,
    staging_name: str,
    staging_fd: int,
    staging_identity: tuple[int, int, int, int],
) -> None:
    """Remove only this invocation's bound, never-published staging tree."""

    _require_directory_identity(staging_fd, staging_identity, "rollback_staging")
    _require_child_directory_identity(
        publish_fd, staging_name, staging_identity, "rollback_staging_binding"
    )
    entries = set(os.listdir(staging_fd))
    allowed = {DRAFT_CORPUS_REL.name, DRAFT_INDEX_REL.name}
    if not entries.issubset(allowed):
        _raise("publish_staging_rollback_tree_changed")
    try:
        for name in sorted(entries):
            os.unlink(name, dir_fd=staging_fd)
        _fsync_directory_fd(staging_fd)
        os.rmdir(staging_name, dir_fd=publish_fd)
        _fsync_directory_fd(publish_fd)
    except OSError:
        _raise("publish_staging_rollback_failed")


def _child_directory_matches(
    parent_fd: int, name: str, expected: tuple[int, int, int, int]
) -> bool:
    try:
        child_fd = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    except FileNotFoundError:
        return False
    except OSError:
        _raise("publish_child_binding_probe_failed")
    try:
        return _directory_identity(child_fd) == expected
    finally:
        os.close(child_fd)


def _publish_validated(
    draft_fd: int,
    publish_dir: Path,
    *,
    expected_manifest_sha256: str,
) -> None:
    _assert_exact_draft_tree(draft_fd, include_manifest=True)
    manifest_raw = _read_relative(
        draft_fd, DRAFT_MANIFEST_REL, max_bytes=1024 * 1024
    )
    if not hmac.compare_digest(
        hashlib.sha256(manifest_raw).hexdigest(), expected_manifest_sha256
    ):
        _raise("draft_manifest_changed_before_publish")
    manifest = _load_json(manifest_raw, "draft_manifest")
    if (
        not isinstance(manifest, dict)
        or manifest.get("frozen") is not True
        or manifest.get("semantic_reads") != 0
        or manifest.get("validation", {})
        .get("keyed_preseal", {})
        .get("status")
        != "pass"
    ):
        _raise("draft_manifest_not_validated")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {
        relative.as_posix() for relative in DATA_FILE_RELS
    }:
        _raise("draft_manifest_file_set_invalid")
    for relative in DATA_FILE_RELS:
        expected = files[relative.as_posix()]
        if not isinstance(expected, dict) or not isinstance(
            expected.get("sha256"), str
        ):
            _raise("draft_manifest_file_pin_invalid")
        digest, size = _hash_relative(draft_fd, relative)
        if digest != expected["sha256"] or size != expected.get("bytes"):
            _raise("draft_file_pin_mismatch")

    packet_files = manifest.get("packet_files")
    if (
        not isinstance(packet_files, dict)
        or set(packet_files) != {relative.as_posix() for relative in PACKET_FILE_RELS}
        or any(
            not isinstance(value, dict)
            or set(value) != {"sha256", "bytes"}
            or not isinstance(value.get("sha256"), str)
            or type(value.get("bytes")) is not int
            for value in packet_files.values()
        )
    ):
        _raise("draft_manifest_packet_file_set_invalid")

    _publish, publish_fd, publish_identity, current_packet_files = (
        _prepare_publish_root(
            publish_dir,
            expected_builder={
                "sha256": manifest["implementation"]["builder"]["sha256"],
                "bytes": manifest["implementation"]["builder"]["bytes"],
            },
            expected_verifier_sha256=manifest["implementation"]["verifier"][
                "sha256"
            ],
        )
    )
    if current_packet_files != packet_files:
        os.close(publish_fd)
        _raise("publish_packet_file_pin_mismatch")

    staging_name = ".corpus.publish-" + secrets.token_hex(12)
    staging_corpus_fd = -1
    installed_corpus_fd = -1
    corpus_installed = False
    manifest_installed = False
    staging_created = False
    staging_identity: tuple[int, int, int, int] | None = None
    try:
        with _termination_guard():
            try:
                os.mkdir(staging_name, 0o700, dir_fd=publish_fd)
                staging_created = True
                staging_corpus_fd = _open_child_directory(
                    publish_fd, staging_name
                )
                staging_identity = _directory_identity(staging_corpus_fd)
            except OSError:
                _raise("publish_staging_create_failed")
            for relative in DATA_FILE_RELS:
                _copy_install_no_replace(
                    draft_fd,
                    relative,
                    staging_corpus_fd,
                    relative.name,
                    files[relative.as_posix()]["sha256"],
                )
            if set(os.listdir(staging_corpus_fd)) != {
                DRAFT_CORPUS_REL.name,
                DRAFT_INDEX_REL.name,
            }:
                _raise("publish_staging_tree_invalid")
            os.fchmod(staging_corpus_fd, 0o755)
            _fsync_directory_fd(staging_corpus_fd)
            staging_identity = _directory_identity(staging_corpus_fd)
            _require_directory_identity(
                staging_corpus_fd, staging_identity, "publish_staging_corpus"
            )
            _require_path_directory_identity(
                publish_dir, publish_identity, "publish_root"
            )
            if set(os.listdir(publish_fd)) != {
                "README.md",
                "POLICY.md",
                "recipe",
                staging_name,
            }:
                _raise("publish_root_tree_changed")
            # Keep this FD open across the rename.  The handler guard turns
            # SIGTERM/SIGHUP into a catchable failure, while this assignment
            # before rename ensures rollback has an anchored FD even if a
            # signal lands immediately after the atomic install.
            installed_corpus_fd = staging_corpus_fd
            staging_corpus_fd = -1
            _rename_directory_no_replace(publish_fd, staging_name, "corpus")
            corpus_installed = True
            _require_directory_identity(
                installed_corpus_fd, staging_identity, "publish_corpus"
            )
            _require_child_directory_identity(
                publish_fd, "corpus", staging_identity, "publish_corpus_binding"
            )
            if set(os.listdir(installed_corpus_fd)) != {
                DRAFT_CORPUS_REL.name,
                DRAFT_INDEX_REL.name,
            }:
                _raise("publish_corpus_tree_invalid")
            for relative in DATA_FILE_RELS:
                actual = _file_metadata_relative(
                    publish_fd, Path("corpus") / relative.name
                )
                expected = files[relative.as_posix()]
                if actual != {
                    "sha256": expected["sha256"],
                    "bytes": expected["bytes"],
                }:
                    _raise("publish_data_rehash_mismatch")
            if {
                relative.as_posix(): _file_metadata_relative(publish_fd, relative)
                for relative in PACKET_FILE_RELS
            } != packet_files:
                _raise("publish_packet_file_changed")
            if set(os.listdir(publish_fd)) != {
                "README.md",
                "POLICY.md",
                "recipe",
                "corpus",
            }:
                _raise("publish_root_tree_changed")
            _require_path_directory_identity(
                publish_dir, publish_identity, "publish_root"
            )
            _require_child_directory_identity(
                publish_fd, "corpus", staging_identity, "publish_corpus_binding"
            )
            manifest_hash = hashlib.sha256(manifest_raw).hexdigest()
            _copy_install_no_replace(
                draft_fd,
                DRAFT_MANIFEST_REL,
                publish_fd,
                DRAFT_MANIFEST_REL.name,
                manifest_hash,
            )
            manifest_installed = True
            _require_directory_identity(
                publish_fd, publish_identity, "publish_root"
            )
            _require_path_directory_identity(
                publish_dir, publish_identity, "publish_root"
            )
            _require_child_directory_identity(
                publish_fd, "corpus", staging_identity, "publish_corpus_binding"
            )
            if set(os.listdir(publish_fd)) != {
                "README.md",
                "POLICY.md",
                "recipe",
                "corpus",
                "manifest.json",
            }:
                _raise("published_tree_not_allowlisted")
            _fsync_directory_fd(publish_fd)
    except BaseException:
        previous_handlers: dict[signal.Signals, Any] = {}
        for signum in (signal.SIGTERM, signal.SIGHUP):
            try:
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, signal.SIG_IGN)
            except (OSError, ValueError):
                previous_handlers.pop(signum, None)
        try:
            if not manifest_installed and installed_corpus_fd >= 0:
                try:
                    corpus_is_bound = _child_directory_matches(
                        publish_fd, "corpus", staging_identity
                    )
                    if corpus_is_bound:
                        _remove_installed_corpus(
                            publish_fd,
                            installed_corpus_fd,
                            staging_identity,
                        )
                        corpus_installed = False
                except BuildError:
                    _raise("publish_failed_and_rollback_failed")
            if (
                not corpus_installed
                and staging_corpus_fd >= 0
                and staging_identity is not None
            ):
                try:
                    _remove_staging_corpus(
                        publish_fd,
                        staging_name,
                        staging_corpus_fd,
                        staging_identity,
                    )
                    staging_created = False
                except BuildError:
                    _raise("publish_failed_and_rollback_failed")
            elif not corpus_installed and staging_created:
                try:
                    os.rmdir(staging_name, dir_fd=publish_fd)
                    _fsync_directory_fd(publish_fd)
                    staging_created = False
                except OSError:
                    _raise("publish_failed_and_rollback_failed")
        finally:
            for signum, handler in previous_handlers.items():
                try:
                    signal.signal(signum, handler)
                except (OSError, ValueError):
                    pass
        raise
    finally:
        if installed_corpus_fd >= 0:
            os.close(installed_corpus_fd)
        if staging_corpus_fd >= 0:
            os.close(staging_corpus_fd)
        os.close(publish_fd)


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--original-manifest", type=Path, default=DEFAULT_ORIGINAL_MANIFEST
    )
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--draft-dir", type=Path, required=True)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    draft = subparsers.add_parser(
        "draft", help="build a non-sealed candidate in a new empty private directory"
    )
    _add_source_arguments(draft)
    freeze = subparsers.add_parser(
        "freeze",
        help="regenerate, externally validate, and publish with manifest last",
    )
    _add_source_arguments(freeze)
    freeze.add_argument("--publish-dir", type=Path, required=True)
    freeze.add_argument(
        "--verifier-sha256",
        required=True,
        help="pre-reviewed SHA-256 of the fixed sibling recipe/verify.py",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    draft_fd = -1
    try:
        _disable_core_dumps()
        args = _parse_args(argv)
        args.staging = _normalized_path(args.staging)
        args.snapshot = _normalized_path(args.snapshot)
        args.original_manifest = _normalized_path(args.original_manifest)
        args.splits = _normalized_path(args.splits)
        publish = (
            _normalized_path(args.publish_dir) if args.command == "freeze" else None
        )
        canonical_root = _normalized_path(Path(__file__).resolve().parent.parent)
        _preflight_output_boundaries(
            args,
            args.draft_dir,
            publish,
            canonical_root=canonical_root,
        )
        draft, draft_fd, draft_identity = _prepare_draft(args.draft_dir)
        _validate_path_boundaries(
            args,
            draft,
            publish,
            canonical_root=canonical_root,
        )
        _verify_public_source_pins(args)
        result = _run_keyed_worker(
            args,
            draft,
            draft_fd,
            run_verifier=args.command == "freeze",
        )
        _require_directory_identity(draft_fd, draft_identity, "draft")
        _require_path_directory_identity(draft, draft_identity, "draft")
        result["files"] = _rehash_candidate(draft_fd, result["files"])
        packet_files = (
            _packet_files_for_freeze(
                publish,
                expected_builder=result["source_pins"]["builder"],
                expected_verifier_sha256=result["verifier_sha256"],
            )
            if publish is not None
            else {}
        )
        manifest = _manifest_document(
            result,
            frozen=args.command == "freeze",
            packet_files=packet_files,
        )
        manifest_bytes = _json_document(manifest)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        _atomic_write_new_at(
            draft_fd, DRAFT_MANIFEST_REL.name, manifest_bytes, mode=0o400
        )
        _assert_exact_draft_tree(draft_fd, include_manifest=True)
        _require_path_directory_identity(draft, draft_identity, "draft")
        if publish is not None:
            _publish_validated(
                draft_fd,
                publish,
                expected_manifest_sha256=manifest_sha256,
            )
        summary = {
            "status": "published" if publish is not None else "draft",
            "events": EXPECTED_SUPPLEMENT_EVENTS,
            "automatic": EXPECTED_AUTOMATIC_EVENTS,
            "organic": EXPECTED_ORGANIC_EVENTS,
            "direct_nodes": EXPECTED_NODE_COUNT,
            "nodes": result["counts"]["supplement"]["nodes"],
            "dev_automatic_events": EXPECTED_DEV_AUTOMATIC,
            "dev_unique_tokens": EXPECTED_DEV_UNIQUE_TOKENS,
            "repeated_automatic_families": EXPECTED_REPEATED_FAMILIES,
            "repeated_automatic_events": EXPECTED_REPEATED_EVENTS,
            "unseen_in_dev_families": EXPECTED_UNSEEN_FAMILIES,
            "unseen_in_dev_events": EXPECTED_UNSEEN_EVENTS,
            "semantic_reads": 0,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0
    except BuildError as error:
        print(
            json.dumps(
                {"status": "error", "code": error.code},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    except Exception:
        # The unkeyed launcher still fails closed without allowing an
        # unexpected exception or object representation onto stderr.
        print(
            json.dumps(
                {"status": "error", "code": "internal_failure"},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    finally:
        if draft_fd >= 0:
            try:
                os.close(draft_fd)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
