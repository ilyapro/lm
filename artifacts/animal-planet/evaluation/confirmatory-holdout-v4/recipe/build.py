#!/usr/bin/env python3
"""Confirmatory-holdout-v4 packet builder.

Three commands, one pipeline:

* ``self-check --work-dir DIR`` generates a synthetic, production-shaped
  fixture, runs the complete draft and freeze pipeline over it into
  ``DIR/packet``, replays the frozen protocol's own golden vectors as
  assertions, prints one aggregate JSON line, and touches nothing outside
  ``DIR``.  It seals no real packet.
* ``draft --draft-dir DIR --handoff PATH`` is the real content build.  Every
  keyed step runs in a forked worker whose process group is destroyed before
  control returns; the launcher never holds the HMAC key or the salt.
* ``freeze --draft-dir DIR --publish-dir DIR --verifier-sha256 HEX`` installs
  that content with ``renameat2(RENAME_NOREPLACE)`` and writes ``manifest.json``
  last, through ``O_EXCL`` + ``os.link``.  There is no ``--force`` and no
  ``--overwrite``: a second attempt fails closed.

The manifest is deliberately the last byte written, because
``publication_and_ordering`` makes its presence the sealed-state marker.  The
launcher constructs it only after the keyed worker has exited and been reaped,
which is the plan's ``manifest_construction_after_hard_boundary``.

Authority is ``POLICY.md`` sections 4, 5, 7 and 10 plus ``analysis-plan.json``.
Nothing here may be relaxed by a caller: floors, domains, column allowlists and
the forbidden-filter set are module constants, not options.
"""

from __future__ import annotations

import sys

# Importing the shared v4 modules must not write ``scripts/__pycache__``.
# ``self-check`` promises to touch nothing outside its work directory, and a
# stray bytecode file would break that before the first fixture is generated.
sys.dont_write_bytecode = True

import argparse
import ctypes
import errno
import hashlib
import hmac
import json
import os
import secrets
import select
import signal
import sqlite3
import stat
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

HERE = Path(__file__).resolve().parent
NAMESPACE_ROOT = HERE.parent
REPO_ROOT = NAMESPACE_ROOT.parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# The accrual module imports the probe module, which imports the runtime
# module, all by canonical name.  Reaching the runtime through
# ``accrual.runtime`` is mandatory: all three use ``type(x) is SomeDataclass``
# identity checks, so a second importlib copy would silently invalidate every
# wrapper we hand back to them.
import ap_confirmatory_accrual_v4 as accrual  # noqa: E402

probe = accrual.probe
runtime = accrual.runtime

if runtime is not probe.runtime or probe is not accrual.probe:
    raise SystemExit("v4 runtime module identity is not shared")

IntegrityFailure = runtime.IntegrityFailure

NAMESPACE = runtime.NAMESPACE
SCHEMA_VERSION = runtime.SCHEMA_VERSION
SOURCE_ALIASES = runtime.SOURCE_ALIASES
RELEASE_EFFECTIVE_AT = runtime.RELEASE_EFFECTIVE_AT

PARTITION_DOMAIN = b"confirmatory-holdout-v4/partition/v1\0"
IDENTITY_DOMAIN = b"confirmatory-holdout-v4/identity/v1\0"
SPLIT_MODULUS = 100
HOLDOUT_UPPER_EXCLUSIVE = 50
KEY_BYTES = 32
SALT_BYTES = 32
LABEL_BYTES = 16

HOLDOUT_PARTITION = "holdout"
SHADOW_PARTITION = "shadow"
PARTITIONS = (HOLDOUT_PARTITION, SHADOW_PARTITION)

# ``selection.forbidden_filter_fields`` -- none of these may influence
# membership, ordering, component formation or partitioning.  The builder never
# reads them; ``self-check`` proves the claim by perturbing them.
FORBIDDEN_FILTER_FIELDS = (
    "feedback_applied",
    "feedback_applied_at",
    "success",
    "error",
    "payload",
    "latency",
    "access",
    "results",
    "content",
    "case_outcome",
    "candidate_metric",
    "baseline_metric",
    "family_size",
    "session_presence",
    "workflow_presence",
    "replayability",
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

# ``privacy.forbidden_in_prompts_logs_diagnostics_or_tracked_nonsealed_artifacts``.
FORBIDDEN_OUTSIDE_SEALED_CORPUS = (
    "raw SQLite rows",
    "raw transcripts",
    "raw paths or source locators",
    "raw queries or raw content",
    "query or content surrogates outside sealed corpus",
    "HMAC keys or commitments",
    "identity tokens or samples",
    "unkeyed query fingerprints",
    "raw event identifiers or packet event labels outside sealed corpus",
    "family membership or per-family sizes",
    "session or workflow identifiers",
    "raw or de-identified individual case records outside sealed corpus",
    "case-level outcomes",
    "scope spelling lists",
    "per-source counts or outcomes",
)

# ``replayability.ambient_context.behaviorally_relevant_keys_exactly``.  Any
# other key is unconditionally nonreplayable; there is no later exception.
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

SOURCE_EVENT_COLUMNS = probe.EVENT_COLUMNS
AGGREGATE_FIELDS = probe.AGGREGATE_FIELDS
HOLDOUT_FLOORS = probe.HOLDOUT_FLOORS
SHADOW_FLOORS = probe.SHADOW_FLOORS
PRODUCTION_SEED_CONTRACT_ID = probe.PRODUCTION_SEED_CONTRACT_ID
REQUIRED_PRODUCTION_POLICY_KEYS = probe.REQUIRED_PRODUCTION_POLICY_KEYS

MINIMUM_FAMILY_EVENTS = 3
MINIMUM_FAMILY_SESSIONS = 2

NODE_LEVELS = ("trace", "concept", "schema")
RELATION_TYPES = ("related", "caused", "contradicts", "supersedes", "requires")

# ``measurement.state_isolation.construction.copied_table_column_allowlist``.
# Left side is the packet column, right side the source column it projects.
SEED_COLUMN_ALLOWLIST = {
    "nodes": (
        ("packet_node_label", "id"),
        ("level", "level"),
        ("deidentified_content", "content"),
        ("packet_scope_label", "scope"),
        ("created_at", "created_at"),
    ),
    "connections": (
        ("source_packet_node_label", "source_id"),
        ("target_packet_node_label", "target_id"),
        ("relation_type", "type"),
        ("weight", "weight"),
    ),
    "retrieval_weights": (
        ("packet_scope_label", "scope"),
        ("bm25_weight", "bm25"),
        ("vector_weight", "vector"),
        ("graph_weight", "graph"),
    ),
}
COPIED_TABLES = tuple(SEED_COLUMN_ALLOWLIST)

# ``measurement.state_isolation.construction.forced_empty_state`` mapped onto
# the concrete production schema.  ``recall_events`` is empty at seed by name;
# the rest are the automatic-signal, delivery, session, cache and pending
# feedback state the plan forces empty.  A source table outside these two
# classifications and DERIVED_INDEX_TABLES is a fatal preseal failure.
FORCED_EMPTY_TABLES = (
    "recall_events",
    "recall_fingerprints",
    "kv",
    "metadata",
)
# FTS5 index and its shadow tables are derived from ``nodes``; the evaluator's
# own hash-bound migration rebuilds them, so they are never copied and are not
# unclassified mutable state.
DERIVED_INDEX_PREFIXES = ("nodes_fts",)

SEED_TABLE_ORDER = (*COPIED_TABLES, *FORCED_EMPTY_TABLES)

# Whitespace whose position de-identification preserves exactly.  Everything
# else becomes a lowercase letter.
KEEP_WHITESPACE = frozenset(" \t\n\r")
LOWERCASE = "abcdefghijklmnopqrstuvwxyz"

MAX_HANDOFF_BYTES = 1_000_000
MAX_DEV_REFERENCE_BYTES = 64_000_000
MAX_SNAPSHOT_BYTES = 4_000_000_000
MAX_WORKER_STATUS_BYTES = 1_000_000
WORKER_TIMEOUT_SECONDS = 1_800

PACKET_CORPUS_DIRECTORY = "corpus"
PACKET_MANIFEST_NAME = "manifest.json"
HOLDOUT_CORPUS_NAME = "holdout.jsonl"
SHADOW_CORPUS_NAME = "shadow.jsonl"
SEED_MANIFEST_NAME = "seed-state-manifest.json"
PRESEAL_RECEIPT_NAME = "preseal-receipt.json"


def seed_state_name(partition: str, alias: str) -> str:
    return f"seed-state-{partition}-{alias}.sqlite3"


CORPUS_MEMBER_NAMES = (
    HOLDOUT_CORPUS_NAME,
    SHADOW_CORPUS_NAME,
    *(
        seed_state_name(partition, alias)
        for partition in PARTITIONS
        for alias in SOURCE_ALIASES
    ),
    SEED_MANIFEST_NAME,
    PRESEAL_RECEIPT_NAME,
)

# The namespace freeze guard admits exactly these entries.  ``corpus`` and
# ``manifest.json`` are what this builder produces; the rest belong to the
# accrual ledger, this recipe directory and the seal launcher.
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
PUBLISH_REQUIRED_ENTRIES = frozenset({"README.md", "POLICY.md", "analysis-plan.json", "recipe"})
PUBLISH_REFUSED_ENTRIES = frozenset({PACKET_CORPUS_DIRECTORY, PACKET_MANIFEST_NAME})

RENAME_NOREPLACE = 1 << 0
_RENAMEAT2_SYSCALL_BY_MACHINE = {
    "x86_64": 316,
    "aarch64": 276,
    "armv7l": 382,
    "armv8l": 382,
    "ppc64le": 357,
    "s390x": 347,
    "riscv64": 276,
    "loongarch64": 276,
}


class BuildError(RuntimeError):
    """A fail-closed error whose code carries no private detail."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _raise(code: str) -> Any:
    raise BuildError(code)


# A refused output root is its own exit status so a caller cannot confuse it
# with an ordinary build failure.
EXIT_OK = 0
EXIT_BUILD_ERROR = 1
EXIT_NAMESPACE_REFUSED = 3

NAMESPACE_REFUSAL_CODE = "namespace_output_root_refused"


def _exit_code_for(code: str) -> int:
    return EXIT_NAMESPACE_REFUSED if code == NAMESPACE_REFUSAL_CODE else EXIT_BUILD_ERROR


# --------------------------------------------------------------------------
# Path safety
# --------------------------------------------------------------------------


def _normalized_path(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _resolved_path(path: Path | str) -> Path:
    """Follow every link once, for namespace-containment decisions only."""

    try:
        return Path(os.path.realpath(os.fspath(path)))
    except OSError:
        return _raise("path_resolve_failed")


def _paths_overlap(left: Path, right: Path) -> bool:
    left_text = os.fspath(left)
    right_text = os.fspath(right)
    try:
        common = os.path.commonpath((left_text, right_text))
    except ValueError:
        return False
    return common in (left_text, right_text)


def _assert_no_symlink_components(path: Path, *, allow_missing_leaf: bool) -> None:
    candidate = _normalized_path(path)
    current = Path(candidate.anchor)
    parts = candidate.parts[1:] if candidate.is_absolute() else candidate.parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf and index == len(parts) - 1:
                return
            _raise("path_missing")
        except OSError:
            _raise("path_stat_failed")
        if stat.S_ISLNK(info.st_mode):
            _raise("symlink_path_rejected")


def require_output_root_outside_namespace(path: Path | str, *, sealing: bool) -> Path:
    """Refuse a namespace-resident output root outside the real seal path.

    ``self-check`` and ``draft`` may never write inside
    ``artifacts/animal-planet/evaluation/confirmatory-holdout-v4/``; only
    ``freeze --publish-dir`` may, and only because that *is* the publication
    step the protocol authorizes.  Both the lexical and the link-resolved form
    are checked so a symlinked work directory cannot smuggle the write in.
    """

    candidate = _normalized_path(path)
    namespace = _normalized_path(NAMESPACE_ROOT)
    for probe_path, root in (
        (candidate, namespace),
        (_resolved_path(candidate), _resolved_path(namespace)),
    ):
        if _paths_overlap(probe_path, root) and not sealing:
            _raise(NAMESPACE_REFUSAL_CODE)
    return candidate


def _open_flags_readonly() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


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
    """Open an absolute directory without following any path component."""

    candidate = _normalized_path(path)
    if not candidate.is_absolute() or not candidate.parts:
        _raise("directory_path_invalid")
    try:
        current = os.open(candidate.anchor, _directory_open_flags())
    except OSError:
        return _raise("directory_open_failed")
    parts = candidate.parts[1:]
    try:
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            try:
                child = os.open(part, _directory_open_flags(), dir_fd=current)
            except FileNotFoundError:
                if not (create_leaf and last):
                    _raise("path_missing")
                try:
                    os.mkdir(part, leaf_mode, dir_fd=current)
                    child = os.open(part, _directory_open_flags(), dir_fd=current)
                except OSError:
                    return _raise("directory_create_failed")
            except OSError:
                return _raise("symlink_path_rejected")
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
            return _raise("path_missing")
        try:
            os.mkdir(name, mode, dir_fd=parent_fd)
            return os.open(name, _directory_open_flags(), dir_fd=parent_fd)
        except OSError:
            return _raise("directory_create_failed")
    except OSError:
        return _raise("symlink_path_rejected")


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


def _sweep_for_symlinks(root_fd: int, *, label: str) -> None:
    """Refuse a link at any depth beneath an output root."""

    stack: list[int] = [os.dup(root_fd)]
    try:
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        if entry.is_symlink():
                            _raise(f"{label}_symlink_present")
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(
                                _open_child_directory(current, entry.name)
                            )
            except OSError:
                _raise(f"{label}_scan_failed")
            finally:
                os.close(current)
    finally:
        for leftover in stack:
            try:
                os.close(leftover)
            except OSError:
                pass


# --------------------------------------------------------------------------
# Reading, hashing, atomic writing
# --------------------------------------------------------------------------


def _hash_fd(fd: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    offset = 0
    while True:
        try:
            chunk = os.pread(fd, 1 << 20, offset)
        except OSError:
            return _raise("source_read_failed")
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
        offset += len(chunk)
    return digest.hexdigest(), size


def _hash_regular(path: Path) -> tuple[str, int]:
    _assert_no_symlink_components(path, allow_missing_leaf=False)
    try:
        fd = os.open(path, _open_flags_readonly())
    except OSError:
        return _raise("source_open_failed")
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


def identity_of_path(path: Path | str) -> dict[str, Any]:
    """The only binding shape this builder emits: SHA-256 plus byte size.

    ``hash_binding.mutable_head_or_path_only_binding_allowed`` is false, so a
    path never travels alone.  Callers store the returned mapping, not the
    path.
    """

    digest, size = _hash_regular(_normalized_path(path))
    return {"sha256": digest, "bytes": size}


def identity_of_bytes(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes:
        _raise("binding_input_not_bytes")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _require_identity(raw: bytes, expected: Mapping[str, Any], label: str) -> None:
    actual = identity_of_bytes(raw)
    if actual["bytes"] != expected.get("bytes") or not hmac.compare_digest(
        actual["sha256"], str(expected.get("sha256"))
    ):
        _raise(f"{label}_identity_mismatch")


def _read_regular(path: Path, *, max_bytes: int) -> bytes:
    try:
        return runtime.read_regular_file(_normalized_path(path), maximum_bytes=max_bytes)
    except IntegrityFailure:
        return _raise("source_read_rejected")


def _read_at(parent_fd: int, name: str, *, max_bytes: int) -> bytes:
    try:
        fd = os.open(name, _open_flags_readonly(), dir_fd=parent_fd)
    except OSError:
        return _raise("source_open_failed")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            _raise("source_not_regular")
        if info.st_size > max_bytes:
            _raise("source_size_limit")
        chunks: list[bytes] = []
        offset = 0
        while offset < info.st_size:
            chunk = os.pread(fd, min(1 << 20, info.st_size - offset), offset)
            if not chunk:
                _raise("source_short_read")
            chunks.append(chunk)
            offset += len(chunk)
        return b"".join(chunks)
    except OSError:
        return _raise("source_read_failed")
    finally:
        os.close(fd)


def _identity_relative(root_fd: int, relative: Path) -> dict[str, Any]:
    with _relative_parent_fd(root_fd, relative) as (parent_fd, name):
        try:
            fd = os.open(name, _open_flags_readonly(), dir_fd=parent_fd)
        except OSError:
            return _raise("source_open_failed")
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                _raise("source_not_regular")
            digest, size = _hash_fd(fd)
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_size) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
            ):
                _raise("source_changed_during_hash")
            return {"sha256": digest, "bytes": size}
        finally:
            os.close(fd)


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


def write_new_at(parent_fd: int, name: str, data: bytes, *, mode: int = 0o400) -> dict[str, Any]:
    """Create one file that cannot replace an existing name.

    ``O_EXCL`` on a private temporary, then ``os.link`` for the final name.
    ``link`` fails with ``EEXIST`` instead of clobbering, so no publication
    step can overwrite a sealed byte even under a race.
    """

    if not name or "/" in name or name in (".", ".."):
        _raise("output_name_invalid")
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError:
        return _raise("output_probe_failed")
    else:
        _raise("output_exists")

    temporary = "." + name + ".tmp-" + secrets.token_hex(12)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = -1
    try:
        fd = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
        _write_all(fd, data)
        os.fchmod(fd, mode)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.link(
            temporary,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary, dir_fd=parent_fd)
        _fsync_directory_fd(parent_fd)
    except BuildError:
        raise
    except FileExistsError:
        _raise("output_exists")
    except OSError:
        _raise("atomic_output_install_failed")
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except OSError:
            pass
    return identity_of_bytes(data)


def _renameat2_number() -> int:
    machine = os.uname().machine
    number = _RENAMEAT2_SYSCALL_BY_MACHINE.get(machine)
    if number is None:
        _raise("noreplace_rename_unsupported")
    return int(number)


def rename_directory_no_replace(parent_fd: int, source_name: str, target_name: str) -> None:
    """Install a fully-populated directory atomically, never over a target.

    ``renameat2(RENAME_NOREPLACE)`` is required rather than preferred: plain
    ``rename`` would silently replace an existing ``corpus/``, which
    ``publication_and_ordering.no_overwrite`` forbids.  A kernel or libc
    without it fails closed.
    """

    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError:
        _raise("noreplace_rename_unsupported")
    call: Any
    if hasattr(libc, "renameat2"):
        call = libc.renameat2
        call.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        call.restype = ctypes.c_int
        arguments = (
            parent_fd,
            os.fsencode(source_name),
            parent_fd,
            os.fsencode(target_name),
            RENAME_NOREPLACE,
        )
    elif hasattr(libc, "syscall"):
        number = _renameat2_number()
        call = libc.syscall
        call.argtypes = (
            ctypes.c_long,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        call.restype = ctypes.c_long
        arguments = (
            number,
            parent_fd,
            os.fsencode(source_name),
            parent_fd,
            os.fsencode(target_name),
            RENAME_NOREPLACE,
        )
    else:
        return _raise("noreplace_rename_unsupported")

    ctypes.set_errno(0)
    result = call(*arguments)
    if result != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            _raise("publish_target_exists")
        if number in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
            _raise("noreplace_rename_unsupported")
        _raise("directory_install_failed")


# --------------------------------------------------------------------------
# Canonical JSON and exact key sets
# --------------------------------------------------------------------------


def canonical_bytes(value: Any) -> bytes:
    """``domains.v4_canonical_json_utf8`` -- sorted keys, no trailing newline."""

    try:
        return runtime.canonical_json_bytes(value)
    except IntegrityFailure:
        return _raise("canonical_encoding_failed")


def canonical_line(value: Any) -> bytes:
    return canonical_bytes(value) + b"\n"


def load_canonical(raw: bytes, *, expected: type | None = None) -> Any:
    """Parse canonical JSON, rejecting duplicate keys and non-canonical bytes."""

    try:
        return runtime.load_canonical_json_bytes(raw, expected=expected)
    except IntegrityFailure:
        return _raise("canonical_decoding_failed")


def load_json(raw: bytes, *, expected: type | None = None) -> Any:
    try:
        return runtime.load_json_bytes(raw, expected=expected)
    except IntegrityFailure:
        return _raise("json_decoding_failed")


def require_exact_keys(value: Any, keys: Sequence[str], label: str) -> dict[str, Any]:
    """Enforce the schema by key SET, because canonical JSON sorts keys.

    Key order carries no information here, so ``common_validation``'s
    ``recursive_unknown_members_action`` is implemented as exact set equality
    applied again at every nested depth by the caller.
    """

    if type(value) is not dict or set(value) != set(keys):
        _raise(f"{label}_key_set_invalid")
    return value


def _is_int(value: Any) -> bool:
    return type(value) is int


def _nonnegative_int(value: Any, label: str) -> int:
    if not _is_int(value) or value < 0:
        _raise(f"{label}_not_nonnegative_int")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not _is_int(value) or value <= 0:
        _raise(f"{label}_not_positive_int")
    return value


def _hex64(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _raise(f"{label}_not_sha256_hex")
    return value


def require_identity_shape(value: Any, label: str) -> dict[str, Any]:
    mapping = require_exact_keys(value, ("sha256", "bytes"), label)
    _hex64(mapping["sha256"], label)
    _nonnegative_int(mapping["bytes"], label)
    return mapping


def canonical_timestamp(value: Any, label: str) -> str:
    """Parse under ``selection.utc_timestamp_contract`` and re-emit canonically."""

    try:
        return runtime.canonical_utc(runtime.parse_utc(value))
    except IntegrityFailure:
        return _raise(f"{label}_timestamp_invalid")


def timestamp_microseconds(value: Any, label: str) -> int:
    try:
        return runtime.utc_microseconds(value)
    except IntegrityFailure:
        return _raise(f"{label}_timestamp_invalid")


# --------------------------------------------------------------------------
# Ephemeral secrets: one salt, one HMAC key, one label mint
# --------------------------------------------------------------------------


class Secrets:
    """The three key-bearing objects, all destroyed in one ``finally``.

    ``salt`` drives de-identification, ``key`` drives the automatic-identity
    HMAC, and the label mint is neither: ``identity.sealed_packet_equality_labels``
    requires packet-local labels to be *unrelated to HMAC output*, so they come
    straight from :mod:`secrets`.  ``privacy`` forbids persisting any of them,
    or any commitment to them, so nothing here is ever serialized.
    """

    __slots__ = ("salt", "key", "_labels", "_taken", "destroyed")

    def __init__(self) -> None:
        self.salt = bytearray(os.urandom(SALT_BYTES))
        self.key = bytearray(os.urandom(KEY_BYTES))
        if bytes(self.salt) == bytes(self.key):
            _raise("secret_material_not_independent")
        self._labels: dict[tuple[str, str], str] = {}
        self._taken: set[str] = set()
        self.destroyed = False

    def label(self, space: str, value: str) -> str:
        """One fresh opaque surrogate per distinct value, per label space."""

        if self.destroyed:
            _raise("secrets_already_destroyed")
        cache_key = (space, value)
        existing = self._labels.get(cache_key)
        if existing is not None:
            return existing
        while True:
            candidate = secrets.token_hex(LABEL_BYTES)
            if candidate not in self._taken:
                break
        self._labels[cache_key] = candidate
        self._taken.add(candidate)
        return candidate

    def identity_token(self, query: str, requested_scope: str) -> bytes:
        if self.destroyed:
            _raise("secrets_already_destroyed")
        normalized = " ".join(query.split())
        message = IDENTITY_DOMAIN + (normalized + "\n" + requested_scope).encode("utf-8")
        return hmac.new(bytes(self.key), message, hashlib.sha256).digest()

    def destroy(self) -> None:
        for buffer in (self.salt, self.key):
            for index in range(len(buffer)):
                buffer[index] = 0
        self._labels.clear()
        self._taken.clear()
        self.destroyed = True


class Deidentifier:
    """Length-, whitespace- and equality-preserving surrogates.

    ``privacy.deidentification`` permits exactly three preserved properties:
    Python character length, the positions of space/tab/newline/carriage
    return, and bijective equality within one build.  Every other character
    becomes a lowercase letter.

    Surrogates are rendered from the *normalized* identity
    (``" ".join(value.split())``) and re-injected into the original whitespace
    layout.  Rendering the raw string instead would give ``"a b"`` and
    ``"a  b"`` -- one identity family under ``identity.normalization`` --
    different normalized surrogates, splitting the family.
    """

    __slots__ = ("_secrets", "_by_normalized", "_taken", "_by_raw")

    def __init__(self, secret: Secrets) -> None:
        self._secrets = secret
        self._by_normalized: dict[str, str] = {}
        self._taken: dict[str, str] = {}
        self._by_raw: dict[str, str] = {}

    def _letters(self, normalized: str, attempt: int, count: int) -> list[str]:
        letters: list[str] = []
        seed = attempt.to_bytes(4, "big") + normalized.encode("utf-8")
        block = 0
        while len(letters) < count:
            digest = hmac.new(
                bytes(self._secrets.salt),
                seed + block.to_bytes(4, "big"),
                hashlib.sha256,
            ).digest()
            letters.extend(LOWERCASE[byte % 26] for byte in digest)
            block += 1
        return letters[:count]

    def _render(self, normalized: str, attempt: int) -> str:
        needed = sum(1 for character in normalized if character not in KEEP_WHITESPACE)
        letters = iter(self._letters(normalized, attempt, needed))
        return "".join(
            character if character in KEEP_WHITESPACE else next(letters)
            for character in normalized
        )

    def _normalized_surrogate(self, normalized: str) -> str:
        existing = self._by_normalized.get(normalized)
        if existing is not None:
            return existing
        attempt = 0
        candidate = self._render(normalized, attempt)
        while candidate in self._taken and self._taken[candidate] != normalized:
            attempt += 1
            if attempt > 1024:
                _raise("deidentification_collision_unresolved")
            candidate = self._render(normalized, attempt)
        self._by_normalized[normalized] = candidate
        self._taken[candidate] = normalized
        return candidate

    def surrogate(self, value: str) -> str:
        if type(value) is not str:
            _raise("deidentification_input_not_string")
        cached = self._by_raw.get(value)
        if cached is not None:
            return cached
        if any(
            character.isspace() and character not in KEEP_WHITESPACE
            for character in value
        ):
            # A whitespace character outside the preserved set would be
            # replaced by a letter, merging two tokens that ``str.split``
            # separates -- a false identity merge.  Fail closed.
            _raise("deidentification_unsupported_whitespace")
        tokens = value.split()
        normalized = " ".join(tokens)
        surrogate_tokens = self._normalized_surrogate(normalized).split(" ") if tokens else []
        if len(surrogate_tokens) != len(tokens) or any(
            len(left) != len(right)
            for left, right in zip(surrogate_tokens, tokens, strict=True)
        ):
            _raise("deidentification_token_shape_invalid")

        pieces: list[str] = []
        index = 0
        position = 0
        length = len(value)
        while position < length:
            character = value[position]
            if character in KEEP_WHITESPACE:
                pieces.append(character)
                position += 1
                continue
            start = position
            while position < length and value[position] not in KEEP_WHITESPACE:
                position += 1
            token = value[start:position]
            if index >= len(surrogate_tokens) or len(surrogate_tokens[index]) != len(token):
                _raise("deidentification_token_shape_invalid")
            pieces.append(surrogate_tokens[index])
            index += 1
        if index != len(surrogate_tokens):
            _raise("deidentification_token_shape_invalid")
        result = "".join(pieces)
        if len(result) != len(value):
            _raise("deidentification_length_not_preserved")
        for left, right in zip(result, value, strict=True):
            if (right in KEEP_WHITESPACE) != (left in KEEP_WHITESPACE) or (
                right in KEEP_WHITESPACE and left != right
            ):
                _raise("deidentification_whitespace_not_preserved")
        self._by_raw[value] = result
        return result

    def normalized_surrogate_of(self, value: str) -> str:
        return " ".join(self.surrogate(value).split())

    def clear(self) -> None:
        self._by_normalized.clear()
        self._taken.clear()
        self._by_raw.clear()


# --------------------------------------------------------------------------
# The frozen input-only resolver
# --------------------------------------------------------------------------


def node_scope_valid(value: Any) -> bool:
    """``replayability.requested_scope_grammar`` -- exactly three shapes."""

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
            return not any(
                ord(character) < 32 or ord(character) == 127 for character in value
            )
    return False


def normalize_scope(value: Any) -> str:
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


def ambient_scope(ambient: Mapping[str, Any]) -> str | None:
    raw_scope = ambient.get("scope")
    if raw_scope:
        return normalize_scope(str(raw_scope))
    raw_session = ambient.get("session_id") or ambient.get("session")
    if raw_session:
        return normalize_scope(f"session:{raw_session}")
    raw_project = (
        ambient.get("project") or ambient.get("project_name") or ambient.get("workspace")
    )
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
    raw_project = (
        ambient.get("project") or ambient.get("project_name") or ambient.get("workspace")
    )
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


def expected_scope_plan(
    requested_scope: Any, ambient: Mapping[str, Any]
) -> tuple[str, ...] | None:
    """Recompute the scope plan from inputs alone, before either replay arm."""

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


def ambient_scalars_valid(ambient: Mapping[str, Any]) -> bool:
    for key, value in ambient.items():
        if type(key) is not str or key not in AMBIENT_KEY_SET:
            # An unknown key is unconditionally nonreplayable; no later
            # implementation proof or case-specific exception is allowed.
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


def load_metadata(value: Any, expected: type) -> Any | None:
    if type(value) is expected:
        return value
    if type(value) is not str:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, RecursionError):
        return None
    return parsed if type(parsed) is expected else None


def event_replayable(fields: Mapping[str, Any]) -> bool:
    """The frozen input-only predicate. Never an inclusion filter.

    Not one of ``replayability.forbidden_inputs`` is read here, and the
    function has no access to either replay arm's behaviour.
    """

    query = fields["query"]
    requested_scope = fields["requested_scope"]
    if (
        type(query) is not str
        or not query
        or not node_scope_valid(requested_scope)
        or type(fields["scope"]) is not str
        or fields["scope"] != requested_scope
        or not _is_int(fields["max_results"])
        or fields["max_results"] <= 0
        or not depth_accepted(fields["depth"])
    ):
        return False
    try:
        query.encode("utf-8")
    except UnicodeEncodeError:
        return False
    resolved = load_metadata(fields["resolved_scopes"], list)
    ambient = load_metadata(fields["ambient_context"], dict)
    if resolved is None or ambient is None or not ambient_scalars_valid(ambient):
        return False
    if not all(type(value) is str and node_scope_valid(value) for value in resolved):
        return False
    plan = expected_scope_plan(requested_scope, ambient)
    if plan is None or tuple(resolved) != plan:
        return False
    if not same_json_scalar(fields["agent"], ambient.get("agent")):
        return False
    if not same_json_scalar(fields["task"], ambient.get("task")):
        return False
    caller_session = ambient.get("session_id") or ambient.get("session")
    if not same_json_scalar(fields["session_id"], caller_session):
        return False
    if not same_json_scalar(
        fields["transport_session_id"], ambient.get("transport_session_id")
    ):
        return False
    return True


# --------------------------------------------------------------------------
# Packet-local labels that commute with the resolver
# --------------------------------------------------------------------------


class LabelMint:
    """Structural surrogates chosen so the resolver cannot tell the difference.

    A scope label keeps its grammar class, an ambient project label is the
    *bare* token so ``normalize_scope(token) == "project:" + token``, and a
    path label is ``"/" + project-like label of its basename`` so
    ``Path(label).name`` still derives the same scope.  The empty string maps
    to itself because :func:`ambient_scope` selects keys by truthiness: a
    nonempty token for ``""`` would change which key wins.
    """

    __slots__ = ("_secrets",)

    def __init__(self, secret: Secrets) -> None:
        self._secrets = secret

    def opaque(self, space: str, value: Any) -> Any:
        if value is None:
            return None
        if type(value) is not str:
            # Nullness, scalar type and equality are preserved; a non-string
            # scalar carries no identifying text and stays exact.
            if type(value) in (bool, int, float):
                return value
            return _raise("label_input_type_invalid")
        if value == "":
            return ""
        return self._secrets.label(space, value)

    def scope(self, value: Any) -> str:
        """Label a node scope, preserving ``global``/``project:``/``session:``."""

        try:
            normalized = normalize_scope(value)
        except (TypeError, ValueError, UnicodeError):
            return _raise("scope_label_input_invalid")
        if normalized == "global":
            return "global"
        prefix, _, _ = normalized.partition(":")
        if prefix not in ("project", "session"):
            return _raise("scope_label_input_invalid")
        return f"{prefix}:{self._secrets.label('scope', normalized)}"

    def project_like(self, value: Any) -> str:
        """Label an ambient value whose behaviour is ``normalize_scope`` of it."""

        labelled = self.scope(value)
        if labelled.startswith("project:"):
            # Bare token: ``normalize_scope(token)`` re-derives ``project:token``.
            return labelled.split(":", 1)[1]
        return labelled

    def path_like(self, value: Any) -> str:
        """Label a workspace path so only its derived scope survives."""

        if type(value) is not str:
            return _raise("path_label_input_invalid")
        if value == "":
            return ""
        name = Path(value).name
        if not name:
            return "/"
        return "/" + self.project_like(name)

    def policy_key(self, value: Any) -> str:
        """Label a retrieval-policy key, never with the node-scope grammar.

        ``retrieval_weights.scope`` stores a policy key, not a memory scope.
        The four production keys map to themselves so the required set stays
        present; only concrete learned keys are tokenized.
        """

        kind = production_policy_key_kind(value)
        if value in REQUIRED_PRODUCTION_POLICY_KEYS:
            return str(value)
        return f"{kind}:{self._secrets.label('policy', str(value))}"


def production_policy_key_kind(value: Any) -> str:
    try:
        return probe.production_policy_key_kind(value)
    except IntegrityFailure:
        return _raise("retrieval_policy_key_invalid")


def deidentify_ambient(
    ambient: Mapping[str, Any], mint: LabelMint
) -> dict[str, Any]:
    """Rebuild the caller envelope from labels, preserving every scalar rule."""

    result: dict[str, Any] = {}
    for key, value in ambient.items():
        if key not in AMBIENT_KEY_SET:
            _raise("ambient_unknown_key")
        if value is None or type(value) in (bool, int, float) or value == "":
            result[key] = value
            continue
        if key == "scope":
            result[key] = mint.scope(value)
        elif key in ("session_id", "session"):
            result[key] = session_label(value, mint)
        elif key in ("project", "project_name", "workspace", "project_scope"):
            result[key] = mint.project_like(value)
        elif key in ("workspace_path", "cwd"):
            result[key] = mint.path_like(value)
        elif key == "agent":
            result[key] = mint.opaque("agent", value)
        elif key == "task":
            result[key] = mint.opaque("task", value)
        elif key == "transport_session_id":
            result[key] = mint.opaque("transport", value)
        else:  # pragma: no cover - AMBIENT_KEYS is closed
            _raise("ambient_unknown_key")
    return result


def session_label(value: Any, mint: LabelMint) -> Any:
    """The one label function shared by persisted and ambient session values."""

    if value is None or type(value) in (bool, int, float) or value == "":
        return value
    return mint.project_like(f"session:{value}").split(":", 1)[-1]


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


class SelectedEvent:
    """One selected source row, plus everything derived from inputs only."""

    __slots__ = (
        "alias",
        "event_id",
        "created_at",
        "created_us",
        "fields",
        "token",
        "replayable",
        "component_root",
        "partition",
    )

    def __init__(
        self,
        *,
        alias: str,
        event_id: str,
        created_at: str,
        created_us: int,
        fields: dict[str, Any],
    ) -> None:
        self.alias = alias
        self.event_id = event_id
        self.created_at = created_at
        self.created_us = created_us
        self.fields = fields
        self.token: bytes | None = None
        self.replayable = False
        self.component_root = -1
        self.partition = ""

    @property
    def key_bytes(self) -> bytes:
        return self.alias.encode("utf-8") + b"\0" + self.event_id.encode("utf-8")

    @property
    def automatic(self) -> bool:
        return self.fields["agent"] is None

    @property
    def session_key(self) -> tuple[str, str] | None:
        value = self.fields["transport_session_id"]
        if type(value) is str and value:
            return self.alias, value
        return None


def open_snapshot_readonly(path: Path) -> sqlite3.Connection:
    """Open one immutable snapshot with no write path and no shared cache."""

    _assert_no_symlink_components(path, allow_missing_leaf=False)
    try:
        info = os.lstat(path)
    except OSError:
        return _raise("snapshot_stat_failed")
    if not stat.S_ISREG(info.st_mode):
        _raise("snapshot_not_regular")
    uri = (
        "file:"
        + os.fspath(path).replace("?", "%3f").replace("#", "%23")
        + "?mode=ro&immutable=1&cache=private"
    )
    try:
        connection = sqlite3.connect(uri, uri=True, isolation_level=None)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA temp_store=MEMORY")
    except sqlite3.Error:
        return _raise("snapshot_open_failed")
    return connection


def select_population(
    connection: sqlite3.Connection,
    *,
    alias: str,
    release_us: int,
    lower_us: int,
    upper_us: int,
) -> list[SelectedEvent]:
    """Membership from exactly the three ``created_at`` clauses.

    ``scope_filter`` and ``replayability_filter`` are null and
    ``outcome_filtering`` is false, so this reads no scope, no outcome and no
    floor.  The ``SELECT`` list is exactly the structural and replay-input
    columns; not one ``forbidden_filter_fields`` name appears in it.
    """

    if alias not in SOURCE_ALIASES:
        _raise("source_alias_unknown")
    events: list[SelectedEvent] = []
    seen: set[str] = set()
    columns = ", ".join(SOURCE_EVENT_COLUMNS)
    try:
        rows = connection.execute(f"SELECT {columns} FROM main.recall_events")
        for row in rows:
            record = dict(zip(SOURCE_EVENT_COLUMNS, tuple(row), strict=True))
            event_id = record["id"]
            if type(event_id) is not str or not event_id:
                _raise("selection_event_id_invalid")
            try:
                event_id.encode("utf-8")
            except UnicodeEncodeError:
                return _raise("selection_event_id_invalid")
            if event_id in seen:
                _raise("selection_duplicate_source_qualified_key")
            seen.add(event_id)
            created_us = timestamp_microseconds(record["created_at"], "selection_created_at")
            if not (created_us > release_us and created_us > lower_us and created_us <= upper_us):
                continue
            events.append(
                SelectedEvent(
                    alias=alias,
                    event_id=event_id,
                    created_at=canonical_timestamp(record["created_at"], "selection_created_at"),
                    created_us=created_us,
                    fields=record,
                )
            )
    except BuildError:
        raise
    except IntegrityFailure:
        return _raise("selection_structural_field_invalid")
    except (sqlite3.Error, TypeError, ValueError, UnicodeError, OverflowError):
        return _raise("selection_query_failed")
    return events


def order_population(events: list[SelectedEvent]) -> list[SelectedEvent]:
    """``selection.ordering`` exactly: parsed microseconds, alias, event id."""

    ordered = sorted(
        events,
        key=lambda event: (
            event.created_us,
            event.alias.encode("utf-8"),
            event.event_id.encode("utf-8"),
        ),
    )
    keys = [event.key_bytes for event in ordered]
    if len(keys) != len(set(keys)):
        _raise("selection_duplicate_source_qualified_key")
    return ordered


# --------------------------------------------------------------------------
# Components and partitions
# --------------------------------------------------------------------------


class DisjointSet:
    __slots__ = ("parent", "rank")

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != root:
            self.parent[value], value = root, self.parent[value]
        return root

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


def partition_digest(representative: bytes) -> bytes:
    return hashlib.sha256(PARTITION_DOMAIN + representative).digest()


def partition_bucket(representative: bytes) -> int:
    return int.from_bytes(partition_digest(representative)[0:4], "big") % SPLIT_MODULUS


def partition_for_representative(representative: bytes) -> str:
    bucket = partition_bucket(representative)
    return HOLDOUT_PARTITION if bucket < HOLDOUT_UPPER_EXCLUSIVE else SHADOW_PARTITION


def build_components(
    events: Sequence[SelectedEvent],
) -> tuple[DisjointSet, dict[tuple[str, bytes], list[int]], dict[tuple[str, str], list[int]]]:
    """Exactly the two ``partition.component_edges``; missing both is a singleton."""

    families: dict[tuple[str, bytes], list[int]] = {}
    sessions: dict[tuple[str, str], list[int]] = {}
    for index, event in enumerate(events):
        if event.automatic and event.token is not None:
            families.setdefault((event.alias, event.token), []).append(index)
        session = event.session_key
        if session is not None:
            sessions.setdefault(session, []).append(index)

    disjoint = DisjointSet(len(events))
    for group in (*families.values(), *sessions.values()):
        for member in group[1:]:
            disjoint.union(group[0], member)
    return disjoint, families, sessions


def assign_partitions(events: Sequence[SelectedEvent], disjoint: DisjointSet) -> None:
    """Every component gets one bucket; nothing is searched, retried or moved."""

    members: dict[int, list[int]] = {}
    for index in range(len(events)):
        members.setdefault(disjoint.find(index), []).append(index)
    representatives: dict[bytes, bytes] = {}
    for root, indices in members.items():
        representative = min(events[index].key_bytes for index in indices)
        digest = partition_digest(representative)
        previous = representatives.get(digest)
        if previous is not None and previous != representative:
            _raise("partition_digest_collision")
        representatives[digest] = representative
        chosen = partition_for_representative(representative)
        for index in indices:
            events[index].component_root = root
            events[index].partition = chosen

    holdout = {index for index, event in enumerate(events) if event.partition == HOLDOUT_PARTITION}
    shadow = {index for index, event in enumerate(events) if event.partition == SHADOW_PARTITION}
    if holdout & shadow or holdout | shadow != set(range(len(events))):
        _raise("partition_cover_invalid")


def verify_partition_golden_vectors(vectors: Sequence[Mapping[str, Any]]) -> int:
    """Replay the plan's own vectors through this module's digest."""

    if not vectors:
        _raise("partition_golden_vectors_missing")
    for vector in vectors:
        require_exact_keys(
            vector,
            ("representative_display", "digest_sha256", "bucket", "partition"),
            "partition_golden_vector",
        )
        display = vector["representative_display"]
        if type(display) is not str or display.count("\\0") != 1:
            _raise("partition_golden_vector_display_invalid")
        alias, _, event_id = display.partition("\\0")
        representative = alias.encode("utf-8") + b"\0" + event_id.encode("utf-8")
        digest = partition_digest(representative)
        bucket = partition_bucket(representative)
        if (
            not hmac.compare_digest(digest.hex(), _hex64(vector["digest_sha256"], "vector"))
            or bucket != vector["bucket"]
            or partition_for_representative(representative) != vector["partition"]
        ):
            _raise("partition_golden_vector_mismatch")
    return len(vectors)


# --------------------------------------------------------------------------
# Frozen-dev reference and family qualification
# --------------------------------------------------------------------------


def dev_reference_tokens(
    raw: bytes, *, secret: Secrets, split_lower: int, split_upper: int
) -> set[bytes]:
    """Token set of the complete frozen-dev automatic reference.

    The reference is opened only inside the keyed worker, under the same fresh
    key as the candidate identities, and nothing derived from it is persisted.
    """

    if raw and not raw.endswith(b"\n"):
        _raise("dev_reference_not_newline_terminated")
    tokens: set[bytes] = set()
    identifiers: set[str] = set()
    for line in raw.splitlines():
        if not line:
            _raise("dev_reference_blank_line")
        row = load_json(line, expected=dict)
        if not {"event_id", "query", "requested_scope", "agent"}.issubset(row):
            _raise("dev_reference_row_schema_invalid")
        event_id = row["event_id"]
        if "id" in row and row["id"] != event_id:
            _raise("dev_reference_row_schema_invalid")
        if type(event_id) is not str or not event_id or event_id in identifiers:
            _raise("dev_reference_event_id_invalid")
        identifiers.add(event_id)
        query = row["query"]
        requested_scope = row["requested_scope"]
        if type(query) is not str or type(requested_scope) is not str:
            _raise("dev_reference_row_schema_invalid")
        bucket = (
            int.from_bytes(hashlib.sha256(event_id.encode("utf-8")).digest()[0:8], "big")
            % SPLIT_MODULUS
        )
        if row["agent"] is None and split_lower <= bucket < split_upper:
            tokens.add(secret.identity_token(query, requested_scope))
    return tokens


class Analysis:
    """Everything the packet needs, derived from inputs only."""

    __slots__ = (
        "events",
        "disjoint",
        "families",
        "sessions",
        "qualifying_families",
        "family_of_index",
        "counted_workflows",
        "floor_counted",
        "counts",
    )

    def __init__(self, events: list[SelectedEvent]) -> None:
        self.events = events
        self.disjoint: DisjointSet | None = None
        self.families: dict[tuple[str, bytes], list[int]] = {}
        self.sessions: dict[tuple[str, str], list[int]] = {}
        self.qualifying_families: dict[tuple[str, bytes], list[int]] = {}
        self.family_of_index: dict[int, tuple[str, bytes]] = {}
        self.counted_workflows: set[tuple[str, str]] = set()
        self.floor_counted: set[int] = set()
        self.counts: dict[str, int] = {}


def analyze(events: list[SelectedEvent], dev_tokens: set[bytes]) -> Analysis:
    """Components, qualification and the fourteen aggregate counts.

    A selected nonreplayable event keeps its component edges but contributes to
    no floor: ``replayability`` is eligibility, never membership.
    """

    analysis = Analysis(events)
    disjoint, families, sessions = build_components(events)
    assign_partitions(events, disjoint)
    analysis.disjoint = disjoint
    analysis.families = families
    analysis.sessions = sessions

    for family, indices in families.items():
        eligible = [
            index
            for index in indices
            if events[index].replayable and events[index].automatic
        ]
        distinct_sessions = {
            events[index].session_key
            for index in eligible
            if events[index].session_key is not None
        }
        if (
            len(eligible) >= MINIMUM_FAMILY_EVENTS
            and len(distinct_sessions) >= MINIMUM_FAMILY_SESSIONS
            and family[1] not in dev_tokens
        ):
            analysis.qualifying_families[family] = eligible
        for index in indices:
            analysis.family_of_index[index] = family

    holdout_families = {
        family
        for family, indices in analysis.qualifying_families.items()
        if events[indices[0]].partition == HOLDOUT_PARTITION
    }
    holdout_automatic = {
        index
        for family in holdout_families
        for index in analysis.qualifying_families[family]
    }
    holdout_organic = {
        index
        for index, event in enumerate(events)
        if event.partition == HOLDOUT_PARTITION and event.replayable and not event.automatic
    }
    holdout_sessions = {
        events[index].session_key
        for index in holdout_organic
        if events[index].session_key is not None
    }

    for workflow, indices in sessions.items():
        if events[indices[0]].partition != SHADOW_PARTITION:
            continue
        replayable_members = [index for index in indices if events[index].replayable]
        organic_anchor = any(
            events[index].replayable and not events[index].automatic for index in indices
        )
        if replayable_members and organic_anchor:
            analysis.counted_workflows.add(workflow)
    shadow_calls = {
        index
        for workflow in analysis.counted_workflows
        for index in sessions[workflow]
        if events[index].replayable
    }

    def project_scopes(indices: Sequence[int] | set[int]) -> set[str]:
        return {
            events[index].fields["requested_scope"]
            for index in indices
            if events[index].replayable
            and type(events[index].fields["requested_scope"]) is str
            and events[index].fields["requested_scope"].startswith("project:")
        }

    holdout_indices = {
        index for index, event in enumerate(events) if event.partition == HOLDOUT_PARTITION
    }
    replayable_count = sum(1 for event in events if event.replayable)
    counts = {
        "selected_event_count": len(events),
        "selected_replayable_event_count": replayable_count,
        "selected_nonreplayable_event_count": len(events) - replayable_count,
        "holdout_unseen_automatic_family_count": len(holdout_families),
        "holdout_unseen_automatic_event_count": len(holdout_automatic),
        "holdout_unseen_automatic_component_count": len(
            {disjoint.find(index) for index in holdout_automatic}
        ),
        "holdout_organic_event_count": len(holdout_organic),
        "holdout_organic_session_count": len(holdout_sessions),
        "holdout_organic_component_count": len(
            {disjoint.find(index) for index in holdout_organic}
        ),
        "holdout_project_scope_count": len(project_scopes(holdout_indices)),
        "shadow_real_workflow_count": len(analysis.counted_workflows),
        "shadow_replayable_logical_call_count": len(shadow_calls),
        "shadow_real_workflow_component_count": len(
            {disjoint.find(sessions[workflow][0]) for workflow in analysis.counted_workflows}
        ),
        "shadow_project_scope_count": len(project_scopes(shadow_calls)),
    }
    if tuple(counts) != AGGREGATE_FIELDS:
        _raise("aggregate_field_set_invalid")
    try:
        probe._aggregate_invariants(counts)
    except IntegrityFailure:
        _raise("aggregate_invariants_violated")
    analysis.counts = counts

    # Every event, call, family witness, session witness, component witness and
    # scope witness that reaches a floor must carry a sealed replayable case.
    analysis.floor_counted = set(holdout_automatic) | holdout_organic | shadow_calls
    if any(not events[index].replayable for index in analysis.floor_counted):
        _raise("floor_counted_case_not_replayable")
    return analysis


def floors_pass(counts: Mapping[str, int]) -> bool:
    return all(counts[field] >= minimum for field, minimum in HOLDOUT_FLOORS.items()) and all(
        counts[field] >= minimum for field, minimum in SHADOW_FLOORS.items()
    )


# --------------------------------------------------------------------------
# Sealed corpus records
# --------------------------------------------------------------------------

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


def build_case_record(
    event: SelectedEvent,
    *,
    analysis: Analysis,
    deidentifier: Deidentifier,
    mint: LabelMint,
) -> dict[str, Any]:
    """One outcome-free replayable case. No result, feedback or latency field."""

    fields = event.fields
    ambient = load_metadata(fields["ambient_context"], dict)
    resolved = load_metadata(fields["resolved_scopes"], list)
    if ambient is None or resolved is None:
        _raise("case_metadata_invalid")

    deidentified_ambient = deidentify_ambient(ambient, mint)
    requested_scope_label = mint.scope(fields["requested_scope"])
    replay_input = {
        "query": deidentifier.surrogate(fields["query"]),
        "requested_scope": requested_scope_label,
        "resolved_scopes": [mint.scope(value) for value in resolved],
        "max_results": _positive_int(fields["max_results"], "case_max_results"),
        "depth": fields["depth"],
        "ambient_context": deidentified_ambient,
        "agent": mint.opaque("agent", fields["agent"]),
        "task": mint.opaque("task", fields["task"]),
        "session_id": session_label(fields["session_id"], mint),
        "transport_session_id": mint.opaque("transport", fields["transport_session_id"]),
    }
    require_exact_keys(replay_input, REPLAY_INPUT_KEYS, "replay_input")

    # The de-identified case must resolve to exactly the de-identified plan the
    # source produced.  A mismatch would mean de-identification changed
    # behaviour, so it is fatal rather than silently nonreplayable here: the
    # source-side predicate already accepted this event.
    reconstructed = {
        "query": replay_input["query"],
        "scope": replay_input["requested_scope"],
        "requested_scope": replay_input["requested_scope"],
        "resolved_scopes": replay_input["resolved_scopes"],
        "ambient_context": replay_input["ambient_context"],
        "depth": replay_input["depth"],
        "max_results": replay_input["max_results"],
        "agent": replay_input["agent"],
        "task": replay_input["task"],
        "session_id": replay_input["session_id"],
        "transport_session_id": replay_input["transport_session_id"],
    }
    if not event_replayable(reconstructed):
        _raise("deidentified_reconstruction_not_replayable")
    if tuple(replay_input["resolved_scopes"]) != expected_scope_plan(
        replay_input["requested_scope"], deidentified_ambient
    ):
        _raise("deidentified_scope_plan_mismatch")

    record = {
        "record_kind": "case",
        "event_label": mint.opaque("event", event.key_bytes.hex()),
        "created_at": event.created_at,
        "partition": event.partition,
        "source_label": mint.opaque("source", event.alias),
        "component_label": mint.opaque("component", str(event.component_root)),
        "family_label": None,
        "workflow_label": (
            mint.opaque("workflow", fields["transport_session_id"])
            if event.session_key is not None
            else None
        ),
        "floor_counted": False,
        "seed_state": seed_state_name(event.partition, event.alias),
        "replay_input": replay_input,
    }
    return require_exact_keys(record, CASE_RECORD_KEYS, "case_record")


def build_inventory_record(
    event: SelectedEvent, *, mint: LabelMint
) -> dict[str, Any]:
    """A selected nonreplayable event: accounted for, but carrying no inputs."""

    record = {
        "record_kind": "inventory",
        "event_label": mint.opaque("event", event.key_bytes.hex()),
        "created_at": event.created_at,
        "partition": event.partition,
        "source_label": mint.opaque("source", event.alias),
        "component_label": mint.opaque("component", str(event.component_root)),
        "replayable": False,
    }
    return require_exact_keys(record, INVENTORY_RECORD_KEYS, "inventory_record")


def build_corpus(
    analysis: Analysis, *, deidentifier: Deidentifier, mint: LabelMint
) -> dict[str, bytes]:
    """Both partition files, in canonical order, covering every selected event."""

    events = analysis.events
    family_labels: dict[tuple[str, bytes], str] = {}
    for family in analysis.families:
        family_labels[family] = mint.opaque("family", family[0] + ":" + family[1].hex())

    lines: dict[str, list[bytes]] = {partition: [] for partition in PARTITIONS}
    for index, event in enumerate(events):
        if event.replayable:
            record = build_case_record(
                event, analysis=analysis, deidentifier=deidentifier, mint=mint
            )
            family = analysis.family_of_index.get(index)
            if family is not None and event.automatic:
                record["family_label"] = family_labels[family]
            record["floor_counted"] = index in analysis.floor_counted
        else:
            record = build_inventory_record(event, mint=mint)
        lines[event.partition].append(canonical_line(record))

    total = sum(len(value) for value in lines.values())
    if total != len(events):
        _raise("corpus_population_incomplete")
    return {
        HOLDOUT_CORPUS_NAME: b"".join(lines[HOLDOUT_PARTITION]),
        SHADOW_CORPUS_NAME: b"".join(lines[SHADOW_PARTITION]),
    }


def require_identity_equivalence(
    analysis: Analysis, *, deidentifier: Deidentifier, mint: LabelMint
) -> None:
    """Raw normalized identity equality iff surrogate equality, per alias.

    A false merge or split would silently change repeated-family membership,
    so it fails closed instead of degrading a case to nonreplayable.
    """

    for alias in SOURCE_ALIASES:
        raw_to_surrogate: dict[tuple[str, str], tuple[str, str]] = {}
        surrogate_to_raw: dict[tuple[str, str], tuple[str, str]] = {}
        for event in analysis.events:
            if event.alias != alias or not event.replayable:
                continue
            fields = event.fields
            raw_identity = (
                " ".join(str(fields["query"]).split()),
                str(fields["requested_scope"]),
            )
            surrogate_identity = (
                deidentifier.normalized_surrogate_of(fields["query"]),
                mint.scope(fields["requested_scope"]),
            )
            previous = raw_to_surrogate.setdefault(raw_identity, surrogate_identity)
            if previous != surrogate_identity:
                _raise("deidentified_identity_split")
            mirrored = surrogate_to_raw.setdefault(surrogate_identity, raw_identity)
            if mirrored != raw_identity:
                _raise("deidentified_identity_merge")


# --------------------------------------------------------------------------
# Partition/source seed states
# --------------------------------------------------------------------------


def classify_source_tables(connection: sqlite3.Connection) -> dict[str, str]:
    """Every mutable source table must land in a declared class.

    ``measurement.state_isolation.construction.unclassified_mutable_table`` is
    a fatal preseal integrity failure, so an unrecognised name stops the build
    instead of being quietly dropped from the projection.
    """

    try:
        rows = connection.execute(
            "SELECT name, type FROM main.sqlite_schema WHERE type IN ('table','view')"
        ).fetchall()
    except sqlite3.Error:
        return _raise("seed_schema_scan_failed")
    classified: dict[str, str] = {}
    for name, kind in rows:
        if type(name) is not str:
            _raise("seed_schema_scan_failed")
        if name.startswith("sqlite_"):
            continue
        if name in COPIED_TABLES:
            classified[name] = "copied"
        elif name in FORCED_EMPTY_TABLES:
            classified[name] = "forced-empty"
        elif any(name.startswith(prefix) for prefix in DERIVED_INDEX_PREFIXES):
            classified[name] = "derived-index"
        elif kind == "view":
            classified[name] = "derived-index"
        else:
            _raise("seed_unclassified_mutable_table")
    for required in (*COPIED_TABLES, "recall_events"):
        if classified.get(required) is None:
            _raise("seed_required_table_missing")
    return classified


def validate_shared_seed_contract(connection: sqlite3.Connection) -> None:
    """The byte-identical validator the probe, sealer and evaluator all share."""

    try:
        probe.validate_production_seed_state(connection)
    except IntegrityFailure:
        _raise("production_seed_state_invalid")


def _seed_schema_statements() -> tuple[str, ...]:
    statements: list[str] = [
        "CREATE TABLE nodes ("
        "packet_node_label TEXT PRIMARY KEY, level TEXT NOT NULL, "
        "deidentified_content TEXT NOT NULL, packet_scope_label TEXT NOT NULL, "
        "created_at TEXT NOT NULL)",
        "CREATE TABLE connections ("
        "source_packet_node_label TEXT NOT NULL, target_packet_node_label TEXT NOT NULL, "
        "relation_type TEXT NOT NULL, weight REAL NOT NULL, "
        "PRIMARY KEY (source_packet_node_label, target_packet_node_label, relation_type))",
        "CREATE TABLE retrieval_weights ("
        "packet_scope_label TEXT PRIMARY KEY, bm25_weight REAL NOT NULL, "
        "vector_weight REAL NOT NULL, graph_weight REAL NOT NULL)",
    ]
    for table in FORCED_EMPTY_TABLES:
        if table == "recall_events":
            columns = ", ".join(f"{column} TEXT" for column in SOURCE_EVENT_COLUMNS)
            statements.append(f"CREATE TABLE recall_events ({columns})")
        else:
            statements.append(f"CREATE TABLE {table} (packet_placeholder TEXT)")
    return tuple(statements)


def project_seed_state(
    connection: sqlite3.Connection,
    *,
    target: Path,
    deidentifier: Deidentifier,
    mint: LabelMint,
) -> dict[str, int]:
    """Build one fresh seed database from the allowlisted columns only.

    The projection is outcome-independent and complete: every allowed row in
    the alias snapshot is represented, so it can serve any selected call.
    """

    validate_shared_seed_contract(connection)
    counts: dict[str, int] = {}
    try:
        seed = sqlite3.connect(os.fspath(target), isolation_level=None)
    except sqlite3.Error:
        return _raise("seed_state_create_failed")
    try:
        seed.execute("PRAGMA journal_mode=DELETE")
        seed.execute("PRAGMA temp_store=MEMORY")
        seed.execute("BEGIN")
        for statement in _seed_schema_statements():
            seed.execute(statement)

        node_labels: dict[str, str] = {}
        rows = connection.execute(
            "SELECT id, level, content, scope, created_at FROM main.nodes ORDER BY id"
        ).fetchall()
        for node_id, level, content, scope, created_at in rows:
            if level not in NODE_LEVELS:
                _raise("seed_node_level_invalid")
            label = mint.opaque("node", str(node_id))
            node_labels[str(node_id)] = str(label)
            seed.execute(
                "INSERT INTO nodes (packet_node_label, level, deidentified_content, "
                "packet_scope_label, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    label,
                    level,
                    deidentifier.surrogate(str(content)),
                    mint.scope(scope),
                    canonical_timestamp(created_at, "seed_node_created_at"),
                ),
            )
        counts["nodes"] = len(rows)

        rows = connection.execute(
            "SELECT source_id, target_id, type, weight FROM main.connections "
            "ORDER BY source_id, target_id, type"
        ).fetchall()
        for source_id, target_id, relation_type, weight in rows:
            if relation_type not in RELATION_TYPES:
                _raise("seed_relation_type_invalid")
            source_label = node_labels.get(str(source_id))
            target_label = node_labels.get(str(target_id))
            if source_label is None or target_label is None:
                _raise("seed_connection_containment_failed")
            seed.execute(
                "INSERT INTO connections (source_packet_node_label, "
                "target_packet_node_label, relation_type, weight) VALUES (?, ?, ?, ?)",
                (source_label, target_label, relation_type, float(weight)),
            )
        counts["connections"] = len(rows)

        rows = connection.execute(
            "SELECT scope, bm25, vector, graph FROM main.retrieval_weights ORDER BY scope"
        ).fetchall()
        policy_kinds: set[str] = set()
        for scope, bm25, vector, graph in rows:
            policy_kinds.add(production_policy_key_kind(scope))
            seed.execute(
                "INSERT INTO retrieval_weights (packet_scope_label, bm25_weight, "
                "vector_weight, graph_weight) VALUES (?, ?, ?, ?)",
                (mint.policy_key(scope), float(bm25), float(vector), float(graph)),
            )
        counts["retrieval_weights"] = len(rows)
        if not rows or not REQUIRED_PRODUCTION_POLICY_KEYS.issubset(policy_kinds):
            _raise("seed_retrieval_policy_keys_incomplete")

        seed.execute("COMMIT")
        for table in FORCED_EMPTY_TABLES:
            remaining = seed.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if remaining != 0:
                _raise("seed_forced_empty_table_not_empty")
            counts[table] = 0
    except BuildError:
        raise
    except sqlite3.Error:
        return _raise("seed_state_write_failed")
    finally:
        try:
            seed.close()
        except sqlite3.Error:
            pass
    return counts


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


def build_seed_manifest(states: Mapping[str, Any]) -> dict[str, Any]:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "receipt_kind": "packet-seed-state-manifest",
        "contract_id": PRODUCTION_SEED_CONTRACT_ID,
        "copied_table_column_allowlist": {
            table: [packet for packet, _ in columns]
            for table, columns in SEED_COLUMN_ALLOWLIST.items()
        },
        "forced_empty_state": list(FORCED_EMPTY_TABLES),
        "derived_index_prefixes": list(DERIVED_INDEX_PREFIXES),
        "policy_key_mapping_exact": dict(probe.POLICY_KEY_MAPPING_EXACT),
        "source_retrieval_policy_keys_required": sorted(REQUIRED_PRODUCTION_POLICY_KEYS),
        "recall_events_at_seed": "empty",
        "unclassified_mutable_table_action": "fatal-preseal-integrity-failure",
        "states": dict(states),
    }
    return require_exact_keys(manifest, SEED_MANIFEST_KEYS, "seed_manifest")


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


def build_preseal_receipt(
    *,
    handoff: Mapping[str, Any],
    counts: Mapping[str, int],
    snapshot_set: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate-only keyed preseal evidence: no case, family or scope detail."""

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "receipt_kind": "packet-keyed-preseal",
        "status": "pass",
        "mismatch_count": 0,
        "slot_index": handoff["slot_index"],
        "scheduled_at": handoff["scheduled_at"],
        "grace_deadline_at": handoff["grace_deadline_at"],
        "active_segment_lower_bound_exclusive_at": handoff[
            "active_segment_lower_bound_exclusive_at"
        ],
        "release_effective_at": handoff["release_effective_at"],
        "segment_id": handoff["segment_id"],
        "snapshot_set_sha256_and_bytes": dict(snapshot_set),
        "aliased_source_snapshot_sha256_and_bytes": {
            alias: dict(handoff["aliased_source_snapshot_sha256_and_bytes"][alias])
            for alias in SOURCE_ALIASES
        },
        **{field: counts[field] for field in AGGREGATE_FIELDS},
        "floors_pass": floors_pass(counts),
        "semantic_reads": {
            "confirmatory-shadow-v4-eval": 0,
            "confirmatory-holdout-v4-eval": 0,
        },
        "synthetic_fixture": bool(handoff["synthetic_fixture"]),
    }
    return require_exact_keys(receipt, PRESEAL_RECEIPT_KEYS, "preseal_receipt")


# --------------------------------------------------------------------------
# Handoff document
# --------------------------------------------------------------------------

HANDOFF_KEYS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "slot_index",
    "scheduled_at",
    "grace_deadline_at",
    "active_segment_lower_bound_exclusive_at",
    "release_effective_at",
    "segment_id",
    "snapshot_paths",
    "aliased_source_snapshot_sha256_and_bytes",
    "dev_reference_path",
    "dev_reference_sha256_and_bytes",
    "dev_split_lower",
    "dev_split_upper",
    "raw_identity_reference_sha256_and_bytes",
    "seal_launcher_path",
    "expected_aggregate_counts",
    "declared_bindings",
    "synthetic_fixture",
)
DECLARED_BINDING_KEYS = (
    "active_runtime_segment_attestation_and_complete_tuple",
    "active_source_binding_attestation_and_core",
    "first_ready_scheduled_slot_and_ledger_predecessor_chain",
    "distinct_source_authority_and_alias_binding_attestations",
)
HANDOFF_RECEIPT_KIND = "packet-build-handoff"


def _require_nested_identities(value: Any, label: str) -> dict[str, Any]:
    """A binding group: a nonempty object whose every member is sha256+bytes."""

    if type(value) is not dict or not value:
        _raise(f"{label}_binding_group_invalid")
    for member, identity in value.items():
        if type(member) is not str or not member:
            _raise(f"{label}_binding_member_invalid")
        require_identity_shape(identity, f"{label}_{member}")
    return value


def validate_handoff(value: Any) -> dict[str, Any]:
    """Exact key sets at every depth, then the plan pins for a real build."""

    handoff = require_exact_keys(value, HANDOFF_KEYS, "handoff")
    if handoff["schema_version"] != SCHEMA_VERSION or handoff["namespace"] != NAMESPACE:
        _raise("handoff_namespace_invalid")
    if handoff["receipt_kind"] != HANDOFF_RECEIPT_KIND:
        _raise("handoff_receipt_kind_invalid")
    if type(handoff["synthetic_fixture"]) is not bool:
        _raise("handoff_synthetic_flag_invalid")

    slot_index = handoff["slot_index"]
    if not _is_int(slot_index) or not runtime.FIRST_SLOT_INDEX <= slot_index <= runtime.LAST_SLOT_INDEX:
        _raise("handoff_slot_index_invalid")
    slot = runtime.slot_times(slot_index)
    if (
        handoff["scheduled_at"] != slot.scheduled_at
        or handoff["grace_deadline_at"] != slot.grace_deadline_at
    ):
        _raise("handoff_schedule_mismatch")

    canonical_timestamp(handoff["active_segment_lower_bound_exclusive_at"], "handoff_lower_bound")
    canonical_timestamp(handoff["release_effective_at"], "handoff_release")
    _hex64(handoff["segment_id"], "handoff_segment_id")

    require_exact_keys(handoff["snapshot_paths"], SOURCE_ALIASES, "handoff_snapshot_paths")
    for alias in SOURCE_ALIASES:
        if type(handoff["snapshot_paths"][alias]) is not str or not handoff["snapshot_paths"][alias]:
            _raise("handoff_snapshot_path_invalid")
    require_exact_keys(
        handoff["aliased_source_snapshot_sha256_and_bytes"],
        SOURCE_ALIASES,
        "handoff_snapshot_identity",
    )
    for alias in SOURCE_ALIASES:
        require_identity_shape(
            handoff["aliased_source_snapshot_sha256_and_bytes"][alias],
            f"handoff_snapshot_{alias}",
        )

    if type(handoff["dev_reference_path"]) is not str or not handoff["dev_reference_path"]:
        _raise("handoff_dev_reference_path_invalid")
    require_identity_shape(handoff["dev_reference_sha256_and_bytes"], "handoff_dev_reference")
    require_identity_shape(
        handoff["raw_identity_reference_sha256_and_bytes"], "handoff_raw_identity_reference"
    )
    lower = _nonnegative_int(handoff["dev_split_lower"], "handoff_dev_split_lower")
    upper = _nonnegative_int(handoff["dev_split_upper"], "handoff_dev_split_upper")
    if not 0 <= lower < upper <= SPLIT_MODULUS:
        _raise("handoff_dev_split_invalid")

    if type(handoff["seal_launcher_path"]) is not str or not handoff["seal_launcher_path"]:
        _raise("handoff_seal_launcher_path_invalid")

    expected = handoff["expected_aggregate_counts"]
    if expected is not None:
        require_exact_keys(expected, AGGREGATE_FIELDS, "handoff_expected_counts")
        for field in AGGREGATE_FIELDS:
            _nonnegative_int(expected[field], f"handoff_expected_{field}")

    declared = require_exact_keys(
        handoff["declared_bindings"], DECLARED_BINDING_KEYS, "handoff_declared_bindings"
    )
    for key, group in declared.items():
        _require_nested_identities(group, f"handoff_{key}")

    if not handoff["synthetic_fixture"]:
        # A real build cannot choose its own boundary, dev split or reference.
        if handoff["release_effective_at"] != canonical_timestamp(
            RELEASE_EFFECTIVE_AT, "release"
        ):
            _raise("handoff_release_not_plan_pinned")
        if (lower, upper) != (probe.DEV_SPLIT_LOWER, probe.DEV_SPLIT_UPPER):
            _raise("handoff_dev_split_not_plan_pinned")
    return handoff


# --------------------------------------------------------------------------
# Manifest bindings
# --------------------------------------------------------------------------

# Every campaign tool that touches evidence, bound by its own repo-relative
# path.  The member name *is* the path, so each entry carries path, SHA-256 and
# byte size, and the independent verifier can rediscover the same set from disk
# by name shape alone -- it may not spell a tool's module name, because it must
# keep no import path back to one.  Adding a tool without adding it here is the
# failure this enumeration exists to make impossible, and
# ``require_campaign_tools_are_enumerated`` refuses to build past it.
CAMPAIGN_TOOL_BINDING_KEY = "runtime_observer_aggregate_probe_and_accrual_ledger_with_tests"
CAMPAIGN_TOOL_DIRECTORIES = ("scripts", "tests")
CAMPAIGN_TOOL_PATTERNS = {
    "scripts": (("ap_confirmatory_", "_v4.py"), ("v4_", ".py")),
    "tests": (("test_ap_confirmatory_", "_v4.py"), ("test_v4_", ".py")),
}
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

MANIFEST_BINDING_KEYS = (
    "release_control_watermark",
    "release_manifest_reverse_pin",
    "replay_code_control_commit_and_tree",
    "unchanged_repair_design",
    "active_runtime_segment_attestation_and_complete_tuple",
    "active_source_binding_attestation_and_core",
    "first_ready_scheduled_slot_and_ledger_predecessor_chain",
    "local_alias_immutable_snapshot",
    "alt_alias_immutable_snapshot",
    "canonical_two_alias_snapshot_set_identity",
    "distinct_source_authority_and_alias_binding_attestations",
    "policy_document",
    "readme_document",
    "analysis_plan_document",
    "original_frozen_manifest_and_split_declaration",
    "complete_frozen_dev_reference_and_raw_identity_reference",
    "retired_v2_documents",
    "retired_v3_documents",
    "retired_replacement_manifest",
    CAMPAIGN_TOOL_BINDING_KEY,
    "packet_builder_and_independent_verifier",
    "seal_launcher_and_deidentification_implementation",
    "complete_selected_holdout_partition",
    "complete_selected_shadow_partition",
    "partition_source_seed_states_and_construction_manifest",
    "keyed_aggregate_preseal_receipt",
)

EVALUATION_ROOT = REPO_ROOT / "artifacts" / "animal-planet" / "evaluation"
SINGLE_FILE_BINDINGS = {
    "release_control_watermark": EVALUATION_ROOT / "release-v1" / "control-watermark.json",
    "release_manifest_reverse_pin": EVALUATION_ROOT / "release-v1" / "release-manifest.json",
    "unchanged_repair_design": EVALUATION_ROOT
    / "confirmatory-holdout-v2"
    / "repair-design.md",
    "policy_document": NAMESPACE_ROOT / "POLICY.md",
    "readme_document": NAMESPACE_ROOT / "README.md",
    "analysis_plan_document": NAMESPACE_ROOT / "analysis-plan.json",
    "retired_replacement_manifest": EVALUATION_ROOT / "replacement-holdout" / "manifest.json",
}
GROUP_FILE_BINDINGS = {
    "original_frozen_manifest_and_split_declaration": {
        "original_manifest": REPO_ROOT / "artifacts" / "animal-planet" / "manifest.json",
        "split_declaration": REPO_ROOT
        / "artifacts"
        / "animal-planet"
        / "corpus"
        / "splits.json",
    },
    "retired_v2_documents": {
        "README.md": EVALUATION_ROOT / "confirmatory-holdout-v2" / "README.md",
        "POLICY.md": EVALUATION_ROOT / "confirmatory-holdout-v2" / "POLICY.md",
        "analysis-plan.json": EVALUATION_ROOT
        / "confirmatory-holdout-v2"
        / "analysis-plan.json",
        "attempt-note.json": EVALUATION_ROOT
        / "confirmatory-holdout-v2"
        / "attempt-note.json",
        "scanner": SCRIPTS_DIR / "ap_confirmatory_readiness.py",
    },
    "retired_v3_documents": {
        "README.md": EVALUATION_ROOT / "confirmatory-holdout-v3" / "README.md",
        "POLICY.md": EVALUATION_ROOT / "confirmatory-holdout-v3" / "POLICY.md",
        "analysis-plan.json": EVALUATION_ROOT
        / "confirmatory-holdout-v3"
        / "analysis-plan.json",
        "protocol-test": REPO_ROOT / "tests" / "test_confirmatory_evidence_protocol_v3.py",
    },
    CAMPAIGN_TOOL_BINDING_KEY: {
        relative: REPO_ROOT / relative for relative in CAMPAIGN_TOOL_PATHS
    },
}


def is_campaign_tool(directory: str, name: str) -> bool:
    """Name shape only, matching the verifier's independent discovery rule."""

    for prefix, suffix in CAMPAIGN_TOOL_PATTERNS[directory]:
        if name.startswith(prefix) and name.endswith(suffix):
            if len(name) > len(prefix) + len(suffix):
                return True
    return False


def discover_campaign_tools() -> set[str]:
    """Every campaign tool on disk, by repo-relative path."""

    discovered: set[str] = set()
    for directory in CAMPAIGN_TOOL_DIRECTORIES:
        for entry in sorted((REPO_ROOT / directory).iterdir()):
            if entry.is_file() and is_campaign_tool(directory, entry.name):
                discovered.add(f"{directory}/{entry.name}")
    return discovered


def require_campaign_tools_are_enumerated() -> None:
    """Refuse to build a manifest that leaves an evidence-touching tool loose.

    The verifier settles this against disk on its own and would reject the
    packet anyway; failing here means a new tool is caught at the build that
    would have shipped it, not after sealing.
    """

    if discover_campaign_tools() != set(CAMPAIGN_TOOL_PATHS):
        _raise("campaign_tool_enumeration_not_closed")


def replay_code_control_identity() -> dict[str, Any]:
    """Bind the commit *and* tree as content, never as a mutable head name."""

    return identity_of_bytes(
        canonical_bytes(
            {
                "commit": runtime.REPLAY_CODE_CONTROL_COMMIT,
                "tree": runtime.REPLAY_CODE_CONTROL_TREE,
            }
        )
    )


def snapshot_set_identity(handoff: Mapping[str, Any]) -> dict[str, Any]:
    identity = accrual.derive_snapshot_set_identity(
        {
            alias: runtime.HashAndBytes.from_value(
                handoff["aliased_source_snapshot_sha256_and_bytes"][alias]
            )
            for alias in SOURCE_ALIASES
        }
    )
    return identity.as_dict()


def build_manifest_bindings(
    *,
    handoff: Mapping[str, Any],
    content_identities: Mapping[str, dict[str, Any]],
    verifier_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """All twenty-six items, each SHA-256 plus byte size, nothing path-only."""

    require_campaign_tools_are_enumerated()
    bindings: dict[str, Any] = {}
    for key, path in SINGLE_FILE_BINDINGS.items():
        bindings[key] = identity_of_path(path)
    for key, members in GROUP_FILE_BINDINGS.items():
        bindings[key] = {name: identity_of_path(path) for name, path in members.items()}

    bindings["replay_code_control_commit_and_tree"] = replay_code_control_identity()
    for key in DECLARED_BINDING_KEYS:
        bindings[key] = {
            member: dict(identity)
            for member, identity in handoff["declared_bindings"][key].items()
        }
    for alias in SOURCE_ALIASES:
        bindings[f"{alias}_alias_immutable_snapshot"] = dict(
            handoff["aliased_source_snapshot_sha256_and_bytes"][alias]
        )
    bindings["canonical_two_alias_snapshot_set_identity"] = snapshot_set_identity(handoff)
    bindings["complete_frozen_dev_reference_and_raw_identity_reference"] = {
        "frozen_dev_reference": dict(handoff["dev_reference_sha256_and_bytes"]),
        # Declaration only: the raw identity reference is operator-private and
        # is never opened outside the isolated keyed process.
        "raw_identity_reference": dict(handoff["raw_identity_reference_sha256_and_bytes"]),
    }
    bindings["packet_builder_and_independent_verifier"] = {
        "packet_builder": identity_of_path(Path(__file__)),
        "independent_verifier": dict(verifier_identity),
    }
    bindings["seal_launcher_and_deidentification_implementation"] = {
        "seal_launcher": identity_of_path(handoff["seal_launcher_path"]),
        # De-identification is implemented in this module, so it binds itself.
        "deidentification_implementation": identity_of_path(Path(__file__)),
    }
    bindings["complete_selected_holdout_partition"] = dict(
        content_identities[HOLDOUT_CORPUS_NAME]
    )
    bindings["complete_selected_shadow_partition"] = dict(
        content_identities[SHADOW_CORPUS_NAME]
    )
    bindings["partition_source_seed_states_and_construction_manifest"] = {
        name: dict(content_identities[name])
        for name in CORPUS_MEMBER_NAMES
        if name.startswith("seed-state-")
    }
    bindings["keyed_aggregate_preseal_receipt"] = dict(
        content_identities[PRESEAL_RECEIPT_NAME]
    )

    require_exact_keys(bindings, MANIFEST_BINDING_KEYS, "manifest_bindings")
    for key, value in bindings.items():
        if "sha256" in value:
            require_identity_shape(value, f"manifest_binding_{key}")
        else:
            _require_nested_identities(value, f"manifest_binding_{key}")
    return bindings


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
PUBLICATION_KEYS = (
    "packet_publication_attempts",
    "no_overwrite",
    "content_before_manifest",
    "manifest_last",
    "canonical_manifest_present",
)


def build_manifest(
    *,
    handoff: Mapping[str, Any],
    counts: Mapping[str, int],
    content_identities: Mapping[str, dict[str, Any]],
    verifier_identity: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "receipt_kind": "packet-manifest",
        "frozen": True,
        "synthetic_fixture": bool(handoff["synthetic_fixture"]),
        "packet_layout": {
            PACKET_CORPUS_DIRECTORY: sorted(CORPUS_MEMBER_NAMES),
            "manifest": PACKET_MANIFEST_NAME,
        },
        "aggregate_counts": {field: counts[field] for field in AGGREGATE_FIELDS},
        "floors_pass": floors_pass(counts),
        "publication": {
            "packet_publication_attempts": 1,
            "no_overwrite": True,
            "content_before_manifest": True,
            "manifest_last": True,
            "canonical_manifest_present": True,
        },
        "semantic_reads": {
            "confirmatory-shadow-v4-eval": 0,
            "confirmatory-holdout-v4-eval": 0,
        },
        "bindings": build_manifest_bindings(
            handoff=handoff,
            content_identities=content_identities,
            verifier_identity=verifier_identity,
        ),
    }
    require_exact_keys(manifest, MANIFEST_KEYS, "manifest")
    require_exact_keys(manifest["publication"], PUBLICATION_KEYS, "manifest_publication")
    require_exact_keys(
        manifest["aggregate_counts"], AGGREGATE_FIELDS, "manifest_aggregate_counts"
    )
    return manifest


# --------------------------------------------------------------------------
# The keyed content build and its hard destruction boundary
# --------------------------------------------------------------------------

DRAFT_CONTENT_DIRECTORY = "content"
DRAFT_STATE_NAME = "draft-state.json"
DRAFT_STATE_KEYS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "handoff",
    "aggregate_counts",
    "floors_pass",
    "content_identities",
    "seed_state_counts",
)


def build_content(
    handoff: Mapping[str, Any], content_fd: int, content_dir: Path
) -> dict[str, Any]:
    """Everything that touches raw rows, keys or salt happens only here."""

    secret = Secrets()
    deidentifier = Deidentifier(secret)
    mint = LabelMint(secret)
    connections: list[sqlite3.Connection] = []
    try:
        release_us = timestamp_microseconds(handoff["release_effective_at"], "release")
        lower_us = timestamp_microseconds(
            handoff["active_segment_lower_bound_exclusive_at"], "lower_bound"
        )
        upper_us = timestamp_microseconds(handoff["scheduled_at"], "scheduled")
        if lower_us < release_us or upper_us <= lower_us:
            _raise("selection_interval_invalid")

        dev_raw = _read_regular(
            Path(handoff["dev_reference_path"]), max_bytes=MAX_DEV_REFERENCE_BYTES
        )
        _require_identity(dev_raw, handoff["dev_reference_sha256_and_bytes"], "dev_reference")
        dev_tokens = dev_reference_tokens(
            dev_raw,
            secret=secret,
            split_lower=handoff["dev_split_lower"],
            split_upper=handoff["dev_split_upper"],
        )
        del dev_raw

        events: list[SelectedEvent] = []
        for alias in SOURCE_ALIASES:
            path = _normalized_path(handoff["snapshot_paths"][alias])
            _require_identity(
                _read_regular(path, max_bytes=MAX_SNAPSHOT_BYTES),
                handoff["aliased_source_snapshot_sha256_and_bytes"][alias],
                f"snapshot_{alias}",
            )
            connection = open_snapshot_readonly(path)
            connections.append(connection)
            classify_source_tables(connection)
            events.extend(
                select_population(
                    connection,
                    alias=alias,
                    release_us=release_us,
                    lower_us=lower_us,
                    upper_us=upper_us,
                )
            )
        events = order_population(events)

        # Identity tokens and the input-only replayability label.  Neither
        # reads an outcome, and neither can remove an event from the
        # population.
        for event in events:
            fields = event.fields
            if (
                event.automatic
                and type(fields["query"]) is str
                and type(fields["requested_scope"]) is str
            ):
                try:
                    fields["query"].encode("utf-8")
                    fields["requested_scope"].encode("utf-8")
                except UnicodeEncodeError:
                    event.token = None
                else:
                    event.token = secret.identity_token(
                        fields["query"], fields["requested_scope"]
                    )
            event.replayable = event_replayable(fields)

        tokens = {
            (event.alias, event.token) for event in events if event.token is not None
        }
        messages = {
            (
                event.alias,
                " ".join(str(event.fields["query"]).split()),
                str(event.fields["requested_scope"]),
            )
            for event in events
            if event.token is not None
        }
        if len(tokens) != len(messages):
            _raise("identity_token_collision")

        analysis = analyze(events, dev_tokens)
        require_identity_equivalence(analysis, deidentifier=deidentifier, mint=mint)

        expected = handoff["expected_aggregate_counts"]
        if expected is not None and dict(expected) != analysis.counts:
            _raise("preseal_aggregate_mismatch")
        if not floors_pass(analysis.counts):
            _raise("preseal_floor_not_met")

        content: dict[str, bytes] = dict(
            build_corpus(analysis, deidentifier=deidentifier, mint=mint)
        )

        seed_counts: dict[str, dict[str, int]] = {}
        for partition in PARTITIONS:
            for alias, connection in zip(SOURCE_ALIASES, connections, strict=True):
                name = seed_state_name(partition, alias)
                seed_counts[name] = project_seed_state(
                    connection,
                    target=content_dir / name,
                    deidentifier=deidentifier,
                    mint=mint,
                )
        seed_manifest = build_seed_manifest(
            {
                name: {"table_row_counts": dict(sorted(counts.items()))}
                for name, counts in sorted(seed_counts.items())
            }
        )
        content[SEED_MANIFEST_NAME] = canonical_bytes(seed_manifest)
        content[PRESEAL_RECEIPT_NAME] = canonical_bytes(
            build_preseal_receipt(
                handoff=handoff,
                counts=analysis.counts,
                snapshot_set=snapshot_set_identity(handoff),
            )
        )

        identities: dict[str, dict[str, Any]] = {}
        for name in CORPUS_MEMBER_NAMES:
            if name in content:
                identities[name] = write_new_at(content_fd, name, content[name])
            else:
                identities[name] = _identity_relative(content_fd, Path(name))
        return {
            "aggregate_counts": dict(analysis.counts),
            "floors_pass": floors_pass(analysis.counts),
            "content_identities": identities,
            "seed_state_counts": {
                name: dict(sorted(counts.items())) for name, counts in sorted(seed_counts.items())
            },
        }
    finally:
        for connection in connections:
            try:
                connection.close()
            except sqlite3.Error:
                pass
        deidentifier.clear()
        secret.destroy()


def _write_pipe_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        try:
            written = os.write(fd, data[offset:])
        except OSError:
            return
        if written <= 0:
            return
        offset += written


def _read_pipe_limited(fd: int, limit: int, *, timeout_seconds: float) -> bytes:
    chunks: list[bytes] = []
    total = 0
    deadline = time.monotonic() + timeout_seconds
    os.set_blocking(fd, False)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _raise("worker_timeout")
        try:
            readable, _, _ = select.select([fd], [], [], min(remaining, 1.0))
        except OSError:
            return _raise("worker_pipe_read_failed")
        if not readable:
            continue
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            continue
        except OSError:
            return _raise("worker_pipe_read_failed")
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            _raise("worker_status_too_large")
    return b"".join(chunks)


def run_keyed_worker(
    handoff: Mapping[str, Any], content_fd: int, content_dir: Path
) -> dict[str, Any]:
    """Fork, build, then destroy the whole key-bearing process group.

    ``privacy.hard_destruction_boundary`` requires descriptor closure,
    descendant process-group termination and an observed exit before manifest
    construction.  The launcher returns only aggregate counts and file
    identities -- never a key, a token, a surrogate or a case.
    """

    status_read, status_write = os.pipe2(os.O_CLOEXEC)
    try:
        pid = os.fork()
    except OSError:
        os.close(status_read)
        os.close(status_write)
        return _raise("worker_fork_failed")

    if pid == 0:  # pragma: no cover - exercised in the forked child
        os.close(status_read)
        code = 0
        try:
            os.setsid()
            result = build_content(handoff, content_fd, content_dir)
            payload = {"status": "ok", "result": result}
        except BuildError as error:
            payload = {"status": "error", "code": error.code}
            code = 1
        except BaseException:
            payload = {"status": "error", "code": "worker_internal_failure"}
            code = 1
        try:
            _write_pipe_all(status_write, json.dumps(payload, sort_keys=True).encode("utf-8"))
        finally:
            os._exit(code)

    os.close(status_write)
    raw = b""
    wait_status = 0
    try:
        raw = _read_pipe_limited(
            status_read, MAX_WORKER_STATUS_BYTES, timeout_seconds=WORKER_TIMEOUT_SECONDS
        )
    finally:
        try:
            os.close(status_read)
        except OSError:
            pass
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            _, wait_status = os.waitpid(pid, 0)
        except ChildProcessError:
            wait_status = 0

    if raw:
        parsed = load_json(raw, expected=dict)
        if parsed.get("status") == "error" and type(parsed.get("code")) is str:
            _raise(parsed["code"])
        if (
            os.WIFEXITED(wait_status)
            and os.WEXITSTATUS(wait_status) == 0
            and parsed.get("status") == "ok"
            and type(parsed.get("result")) is dict
        ):
            return parsed["result"]
    return _raise("worker_failed")


# --------------------------------------------------------------------------
# draft
# --------------------------------------------------------------------------


def _prepare_private_directory(path: Path, *, label: str) -> tuple[Path, int, tuple[int, int, int, int]]:
    candidate = _normalized_path(path)
    if candidate == Path(candidate.anchor):
        _raise(f"{label}_path_invalid")
    _assert_no_symlink_components(candidate, allow_missing_leaf=True)
    fd = _open_directory_chain(candidate, create_leaf=True, leaf_mode=0o700)
    identity = _directory_identity(fd)
    try:
        if identity[2] != os.geteuid():
            _raise(f"{label}_owner_mismatch")
        if identity[3] & 0o077:
            _raise(f"{label}_permissions_not_private")
        if os.listdir(fd):
            _raise(f"{label}_not_empty")
    except BaseException:
        os.close(fd)
        raise
    return candidate, fd, identity


def run_draft(*, draft_dir: Path, handoff_path: Path, sealing: bool = False) -> dict[str, Any]:
    require_output_root_outside_namespace(draft_dir, sealing=sealing)
    handoff_raw = _read_regular(_normalized_path(handoff_path), max_bytes=MAX_HANDOFF_BYTES)
    handoff = validate_handoff(load_canonical(handoff_raw, expected=dict))

    draft, draft_fd, draft_identity = _prepare_private_directory(draft_dir, label="draft")
    content_fd = -1
    try:
        content_fd = _open_child_directory(
            draft_fd, DRAFT_CONTENT_DIRECTORY, create=True, mode=0o700
        )
        result = run_keyed_worker(
            handoff, content_fd, draft / DRAFT_CONTENT_DIRECTORY
        )
        # The keyed process group is gone; re-hash from disk before trusting a
        # single byte the worker reported.
        _require_directory_identity(draft_fd, draft_identity, "draft")
        _sweep_for_symlinks(draft_fd, label="draft")
        identities = {}
        for name in CORPUS_MEMBER_NAMES:
            observed = _identity_relative(content_fd, Path(name))
            declared = result["content_identities"].get(name)
            if declared != observed:
                _raise("content_changed_after_worker_exit")
            identities[name] = observed
        if set(os.listdir(content_fd)) != set(CORPUS_MEMBER_NAMES):
            _raise("draft_content_tree_not_allowlisted")

        state = {
            "schema_version": SCHEMA_VERSION,
            "namespace": NAMESPACE,
            "receipt_kind": "packet-draft-state",
            "handoff": dict(handoff),
            "aggregate_counts": dict(result["aggregate_counts"]),
            "floors_pass": bool(result["floors_pass"]),
            "content_identities": identities,
            "seed_state_counts": dict(result["seed_state_counts"]),
        }
        require_exact_keys(state, DRAFT_STATE_KEYS, "draft_state")
        write_new_at(draft_fd, DRAFT_STATE_NAME, canonical_bytes(state))
        if set(os.listdir(draft_fd)) != {DRAFT_CONTENT_DIRECTORY, DRAFT_STATE_NAME}:
            _raise("draft_tree_not_allowlisted")
        return state
    finally:
        for fd in (content_fd, draft_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass


# --------------------------------------------------------------------------
# freeze
# --------------------------------------------------------------------------


def _prepare_publish_root(path: Path) -> tuple[Path, int, tuple[int, int, int, int]]:
    """The publish root must already be a protocol namespace, and unsealed."""

    publish = _normalized_path(path)
    _assert_no_symlink_components(publish, allow_missing_leaf=False)
    fd = _open_directory_chain(publish)
    identity = _directory_identity(fd)
    try:
        if identity[2] != os.geteuid() or identity[3] & 0o022:
            _raise("publish_root_permissions_unsafe")
        entries = set(os.listdir(fd))
        if not PUBLISH_REQUIRED_ENTRIES.issubset(entries):
            _raise("publish_root_not_a_namespace")
        if entries & PUBLISH_REFUSED_ENTRIES:
            # A present corpus or manifest is a sealed packet.  There is one
            # publication attempt and no reseal.
            _raise("publish_target_exists")
        undeclared = entries - NAMESPACE_ALLOWED_ENTRIES
        if undeclared:
            _raise("publish_root_tree_not_allowlisted")
        _sweep_for_symlinks(fd, label="publish")
        return publish, fd, identity
    except BaseException:
        os.close(fd)
        raise


def _remove_staging_directory(parent_fd: int, name: str) -> None:
    try:
        staging_fd = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    except OSError:
        return
    try:
        for entry in os.listdir(staging_fd):
            try:
                os.unlink(entry, dir_fd=staging_fd)
            except OSError:
                return
    finally:
        try:
            os.close(staging_fd)
        except OSError:
            pass
    try:
        os.rmdir(name, dir_fd=parent_fd)
    except OSError:
        pass


def _install_file_no_replace(
    source_fd: int, source_name: str, target_fd: int, target_name: str, expected: Mapping[str, Any]
) -> None:
    raw = _read_at(source_fd, source_name, max_bytes=MAX_SNAPSHOT_BYTES)
    _require_identity(raw, expected, "publish_content")
    write_new_at(target_fd, target_name, raw, mode=0o444)


def run_freeze(
    *, draft_dir: Path, publish_dir: Path, verifier_sha256: str
) -> dict[str, Any]:
    require_output_root_outside_namespace(draft_dir, sealing=False)
    # The publish root is the one place the namespace may legitimately be
    # written, and only through this seal path.
    require_output_root_outside_namespace(publish_dir, sealing=True)
    _hex64(verifier_sha256, "verifier_sha256")

    draft = _normalized_path(draft_dir)
    _assert_no_symlink_components(draft, allow_missing_leaf=False)
    draft_fd = _open_directory_chain(draft)
    publish_fd = -1
    content_fd = -1
    staging_fd = -1
    staging_name = ".corpus.publish-" + secrets.token_hex(12)
    try:
        _sweep_for_symlinks(draft_fd, label="draft")
        state = require_exact_keys(
            load_canonical(
                _read_at(draft_fd, DRAFT_STATE_NAME, max_bytes=MAX_HANDOFF_BYTES),
                expected=dict,
            ),
            DRAFT_STATE_KEYS,
            "draft_state",
        )
        handoff = validate_handoff(state["handoff"])
        counts = require_exact_keys(
            state["aggregate_counts"], AGGREGATE_FIELDS, "draft_aggregate_counts"
        )
        content_fd = _open_child_directory(draft_fd, DRAFT_CONTENT_DIRECTORY)
        if set(os.listdir(content_fd)) != set(CORPUS_MEMBER_NAMES):
            _raise("draft_content_tree_not_allowlisted")
        identities: dict[str, dict[str, Any]] = {}
        for name in CORPUS_MEMBER_NAMES:
            observed = _identity_relative(content_fd, Path(name))
            if state["content_identities"].get(name) != observed:
                _raise("draft_content_changed_before_publish")
            identities[name] = observed

        publish, publish_fd, publish_identity = _prepare_publish_root(publish_dir)
        verifier_identity = _identity_relative(publish_fd, Path("recipe/verify.py"))
        if not hmac.compare_digest(verifier_identity["sha256"], verifier_sha256):
            _raise("publish_verifier_mismatch")
        builder_identity = identity_of_path(Path(__file__))
        if _identity_relative(publish_fd, Path("recipe/build.py")) != builder_identity:
            _raise("publish_builder_mismatch")

        # Content first, in a private staging directory that is installed
        # atomically and can never replace an existing corpus.
        try:
            os.mkdir(staging_name, 0o700, dir_fd=publish_fd)
        except OSError:
            _raise("publish_staging_create_failed")
        staging_fd = _open_child_directory(publish_fd, staging_name)
        for name in CORPUS_MEMBER_NAMES:
            _install_file_no_replace(content_fd, name, staging_fd, name, identities[name])
        if set(os.listdir(staging_fd)) != set(CORPUS_MEMBER_NAMES):
            _raise("publish_staging_tree_invalid")
        os.fchmod(staging_fd, 0o755)
        _fsync_directory_fd(staging_fd)
        os.close(staging_fd)
        staging_fd = -1

        _require_directory_identity(publish_fd, publish_identity, "publish_root")
        rename_directory_no_replace(publish_fd, staging_name, PACKET_CORPUS_DIRECTORY)
        staging_name = ""
        _fsync_directory_fd(publish_fd)

        installed_fd = _open_child_directory(publish_fd, PACKET_CORPUS_DIRECTORY)
        try:
            for name in CORPUS_MEMBER_NAMES:
                if _identity_relative(installed_fd, Path(name)) != identities[name]:
                    _raise("publish_content_rehash_mismatch")
        finally:
            os.close(installed_fd)

        # Manifest last, after every content byte is installed and verified.
        manifest = build_manifest(
            handoff=handoff,
            counts=counts,
            content_identities=identities,
            verifier_identity=verifier_identity,
        )
        manifest_raw = canonical_bytes(manifest)
        manifest_identity = write_new_at(
            publish_fd, PACKET_MANIFEST_NAME, manifest_raw, mode=0o444
        )
        _fsync_directory_fd(publish_fd)

        entries = set(os.listdir(publish_fd))
        if not {PACKET_CORPUS_DIRECTORY, PACKET_MANIFEST_NAME}.issubset(entries):
            _raise("published_tree_incomplete")
        if entries - NAMESPACE_ALLOWED_ENTRIES:
            _raise("published_tree_not_allowlisted")
        return {
            "publish_dir": os.fspath(publish),
            "manifest": manifest_identity,
            "aggregate_counts": dict(counts),
            "floors_pass": bool(state["floors_pass"]),
            "synthetic_fixture": bool(handoff["synthetic_fixture"]),
        }
    finally:
        if staging_fd >= 0:
            try:
                os.close(staging_fd)
            except OSError:
                pass
        if publish_fd >= 0 and staging_name:
            # A staging directory that never became ``corpus`` is private
            # scratch, so removing it cannot touch a published byte.
            _remove_staging_directory(publish_fd, staging_name)
        for fd in (content_fd, publish_fd, draft_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass


# --------------------------------------------------------------------------
# Synthetic, production-shaped fixtures (self-check only)
# --------------------------------------------------------------------------

# Outcome columns the source really carries.  They exist in the fixture
# precisely so that reading one would be visible: the builder's SELECT list
# never names them, and ``self-check`` perturbs them and re-derives identical
# aggregate counts.
FIXTURE_OUTCOME_COLUMNS = FORBIDDEN_FILTER_FIELDS

FIXTURE_FAMILIES_PER_ALIAS = 40
FIXTURE_FAMILY_EVENTS = 6
FIXTURE_SESSIONS_PER_ALIAS = 45
FIXTURE_SESSION_EVENTS = 7
FIXTURE_PROJECTS = ("proj-alpha", "proj-beta", "proj-gamma", "proj-delta")
FIXTURE_BASE_MICROSECONDS = 1_000_000


# SQLite column affinity is load-bearing: ``max_results`` must come back as an
# integer or the resolver rejects every row, so the fixture mirrors the real
# production declaration rather than typing everything as TEXT.
FIXTURE_EVENT_COLUMN_TYPES = {
    "id": "TEXT PRIMARY KEY",
    "created_at": "TEXT NOT NULL",
    "query": "TEXT NOT NULL",
    "scope": "TEXT NOT NULL DEFAULT 'global'",
    "requested_scope": "TEXT NOT NULL DEFAULT 'global'",
    "resolved_scopes": "TEXT NOT NULL DEFAULT '[]'",
    "ambient_context": "TEXT NOT NULL DEFAULT '{}'",
    "depth": "TEXT",
    "max_results": "INTEGER NOT NULL DEFAULT 10",
    "agent": "TEXT",
    "task": "TEXT",
    "session_id": "TEXT",
    "transport_session_id": "TEXT",
}


def _fixture_schema(connection: sqlite3.Connection) -> None:
    event_columns = ", ".join(
        f"{name} {FIXTURE_EVENT_COLUMN_TYPES[name]}" for name in SOURCE_EVENT_COLUMNS
    )
    outcome_columns = ", ".join(f"{name} TEXT" for name in FIXTURE_OUTCOME_COLUMNS)
    connection.executescript(
        "CREATE TABLE nodes (id TEXT PRIMARY KEY, level TEXT, content TEXT, "
        "scope TEXT, created_at TEXT, decayed INTEGER DEFAULT 0);"
        "CREATE TABLE connections (source_id TEXT, target_id TEXT, type TEXT, weight REAL);"
        "CREATE TABLE retrieval_weights (scope TEXT PRIMARY KEY, bm25 REAL, "
        "vector REAL, graph REAL);"
        f"CREATE TABLE recall_events ({event_columns}, {outcome_columns});"
        "CREATE TABLE recall_fingerprints (fingerprint TEXT, seen_at TEXT);"
        "CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT);"
        "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);"
    )


def _fixture_timestamp(offset: int) -> str:
    # Strictly after the release and segment boundaries and strictly before
    # slot 1, so every generated row is inside the selected interval.
    base = runtime.parse_utc("2026-08-17T00:00:00Z")
    return runtime.canonical_utc(
        base.replace(microsecond=0) + timedelta(microseconds=offset)
    )


def _fixture_event_row(
    *,
    event_id: str,
    created_at: str,
    query: str,
    requested_scope: str,
    resolved: Sequence[str],
    ambient: Mapping[str, Any],
    agent: Any,
    task: Any,
    session_id: Any,
    transport_session_id: Any,
    depth: str,
    max_results: int,
    outcome_marker: str,
) -> tuple[Any, ...]:
    values = {
        "id": event_id,
        "created_at": created_at,
        "query": query,
        "scope": requested_scope,
        "requested_scope": requested_scope,
        "resolved_scopes": json.dumps(list(resolved)),
        "ambient_context": json.dumps(dict(ambient)),
        "depth": depth,
        "max_results": max_results,
        "agent": agent,
        "task": task,
        "session_id": session_id,
        "transport_session_id": transport_session_id,
    }
    row = [values[name] for name in SOURCE_EVENT_COLUMNS]
    row.extend(f"{name}:{outcome_marker}" for name in FIXTURE_OUTCOME_COLUMNS)
    return tuple(row)


def _fixture_rows(alias: str, *, outcome_marker: str) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    offset = FIXTURE_BASE_MICROSECONDS

    for family in range(FIXTURE_FAMILIES_PER_ALIAS):
        project = FIXTURE_PROJECTS[family % len(FIXTURE_PROJECTS)]
        scope = f"project:{project}"
        query = f"repeated automatic probe {alias} family {family:03d}"
        for member in range(FIXTURE_FAMILY_EVENTS):
            transport = f"{alias}-auto-session-{family:03d}-{member % 2}"
            offset += 1
            rows.append(
                _fixture_event_row(
                    event_id=f"{alias}-auto-{family:03d}-{member}",
                    created_at=_fixture_timestamp(offset),
                    query=query,
                    requested_scope=scope,
                    resolved=[scope, "global"],
                    ambient={"transport_session_id": transport},
                    agent=None,
                    task=None,
                    session_id=None,
                    transport_session_id=transport,
                    depth="1",
                    max_results=5,
                    outcome_marker=outcome_marker,
                )
            )

    for session in range(FIXTURE_SESSIONS_PER_ALIAS):
        project = FIXTURE_PROJECTS[session % len(FIXTURE_PROJECTS)]
        transport = f"{alias}-organic-transport-{session:03d}"
        session_name = f"{alias}-organic-session-{session:03d}"
        agent = f"{alias}-agent-{session % 5}"
        for member in range(FIXTURE_SESSION_EVENTS):
            offset += 1
            # Every third session asks for a session scope, which is the only
            # shape whose resolved plan depends on ambient provenance.
            if session % 3 == 0:
                scope = f"session:{session_name}"
                resolved = [scope, f"project:{project}", "global"]
                ambient = {
                    "session_id": session_name,
                    "workspace_path": f"/home/fixture/{project}",
                    "agent": agent,
                    "task": f"task-{session % 4}",
                    "transport_session_id": transport,
                }
            else:
                scope = f"project:{project}"
                resolved = [scope, "global"]
                ambient = {
                    "session_id": session_name,
                    "agent": agent,
                    "task": f"task-{session % 4}",
                    "transport_session_id": transport,
                }
            rows.append(
                _fixture_event_row(
                    event_id=f"{alias}-org-{session:03d}-{member}",
                    created_at=_fixture_timestamp(offset),
                    query=f"organic recall {alias} {session:03d} {member}",
                    requested_scope=scope,
                    resolved=resolved,
                    ambient=ambient,
                    agent=agent,
                    task=f"task-{session % 4}",
                    session_id=session_name,
                    transport_session_id=transport,
                    depth="causal" if member % 2 else "1",
                    max_results=10,
                    outcome_marker=outcome_marker,
                )
            )

    # Selected-but-nonreplayable events: an unknown ambient key, a resolved
    # plan that disagrees with the resolver, and an ungrammatical scope.  Each
    # stays selected and partitioned and reaches no floor.
    for index, (scope, resolved, ambient) in enumerate(
        (
            ("project:proj-alpha", ["project:proj-alpha", "global"], {"unknown_key": "x"}),
            ("project:proj-beta", ["global"], {}),
            ("not-a-scope", ["global"], {}),
        )
    ):
        offset += 1
        rows.append(
            _fixture_event_row(
                event_id=f"{alias}-nonrep-{index}",
                created_at=_fixture_timestamp(offset),
                query=f"nonreplayable {alias} {index}",
                requested_scope=scope,
                resolved=resolved,
                ambient=ambient,
                agent=f"{alias}-agent-nonrep",
                task=None,
                session_id=None,
                transport_session_id=None,
                depth="1",
                max_results=10,
                outcome_marker=outcome_marker,
            )
        )
    return rows


def write_fixture_snapshot(
    path: Path, *, alias: str, outcome_marker: str, reverse_rows: bool
) -> None:
    connection = sqlite3.connect(os.fspath(path), isolation_level=None)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        _fixture_schema(connection)
        connection.execute("BEGIN")
        for index, project in enumerate(FIXTURE_PROJECTS):
            connection.execute(
                "INSERT INTO nodes (id, level, content, scope, created_at) VALUES (?,?,?,?,?)",
                (
                    f"{alias}-node-{index}",
                    NODE_LEVELS[index % len(NODE_LEVELS)],
                    f"seed content for {project} in {alias}",
                    f"project:{project}",
                    _fixture_timestamp(index),
                ),
            )
        connection.execute(
            "INSERT INTO nodes (id, level, content, scope, created_at) VALUES (?,?,?,?,?)",
            (f"{alias}-node-global", "trace", "global seed content", "global", _fixture_timestamp(9)),
        )
        for index in range(len(FIXTURE_PROJECTS)):
            connection.execute(
                "INSERT INTO connections (source_id, target_id, type, weight) VALUES (?,?,?,?)",
                (
                    f"{alias}-node-{index}",
                    f"{alias}-node-global",
                    RELATION_TYPES[index % len(RELATION_TYPES)],
                    1.0 + index,
                ),
            )
        for index, key in enumerate(("default", "project", "global", "session", "project:proj-alpha")):
            connection.execute(
                "INSERT INTO retrieval_weights (scope, bm25, vector, graph) VALUES (?,?,?,?)",
                (key, 1.0 + index, 0.5 + index, 0.25 + index),
            )

        rows = _fixture_rows(alias, outcome_marker=outcome_marker)
        if reverse_rows:
            rows = list(reversed(rows))
        placeholders = ", ".join("?" for _ in range(len(SOURCE_EVENT_COLUMNS) + len(FIXTURE_OUTCOME_COLUMNS)))
        columns = ", ".join((*SOURCE_EVENT_COLUMNS, *FIXTURE_OUTCOME_COLUMNS))
        connection.executemany(
            f"INSERT INTO recall_events ({columns}) VALUES ({placeholders})", rows
        )
        connection.execute("COMMIT")
    finally:
        connection.close()


def write_fixture_dev_reference(path: Path) -> None:
    lines = []
    for index in range(64):
        lines.append(
            json.dumps(
                {
                    "event_id": f"dev-{index:04d}",
                    "query": f"frozen dev automatic query {index}",
                    "requested_scope": "project:dev",
                    "agent": None,
                },
                sort_keys=True,
            ).encode("utf-8")
        )
    path.write_bytes(b"\n".join(lines) + b"\n")


def build_fixture_handoff(
    *,
    fixture_dir: Path,
    seal_launcher_path: Path,
    slot_index: int = 1,
) -> dict[str, Any]:
    slot = runtime.slot_times(slot_index)
    snapshots = {
        alias: identity_of_path(fixture_dir / f"{alias}.sqlite3") for alias in SOURCE_ALIASES
    }
    declared = {
        key: {member: identity_of_bytes(f"{key}/{member}".encode("utf-8")) for member in members}
        for key, members in (
            (
                "active_runtime_segment_attestation_and_complete_tuple",
                ("segment_attestation", "complete_unaliased_service_tuple"),
            ),
            (
                "active_source_binding_attestation_and_core",
                ("source_binding_attestation", "source_binding_core"),
            ),
            (
                "first_ready_scheduled_slot_and_ledger_predecessor_chain",
                ("ready_resolution", "ledger_predecessor_chain"),
            ),
            (
                "distinct_source_authority_and_alias_binding_attestations",
                ("local_authority", "alt_authority", "alias_binding"),
            ),
        )
    }
    handoff = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "receipt_kind": HANDOFF_RECEIPT_KIND,
        "slot_index": slot_index,
        "scheduled_at": slot.scheduled_at,
        "grace_deadline_at": slot.grace_deadline_at,
        "active_segment_lower_bound_exclusive_at": canonical_timestamp(
            RELEASE_EFFECTIVE_AT, "fixture_lower_bound"
        ),
        "release_effective_at": canonical_timestamp(RELEASE_EFFECTIVE_AT, "fixture_release"),
        "segment_id": runtime.INITIAL_SEGMENT_ID,
        "snapshot_paths": {
            alias: os.fspath(fixture_dir / f"{alias}.sqlite3") for alias in SOURCE_ALIASES
        },
        "aliased_source_snapshot_sha256_and_bytes": snapshots,
        "dev_reference_path": os.fspath(fixture_dir / "dev-reference.jsonl"),
        "dev_reference_sha256_and_bytes": identity_of_path(fixture_dir / "dev-reference.jsonl"),
        "dev_split_lower": probe.DEV_SPLIT_LOWER,
        "dev_split_upper": probe.DEV_SPLIT_UPPER,
        "raw_identity_reference_sha256_and_bytes": {
            "sha256": "45dbfdb389324a8702ba1a84b28be028991f23dcfe1d7ab9d771928c37c2cc7a",
            "bytes": 4_162_914,
        },
        "seal_launcher_path": os.fspath(seal_launcher_path),
        "expected_aggregate_counts": None,
        "declared_bindings": declared,
        "synthetic_fixture": True,
    }
    return validate_handoff(handoff)


def write_fixture_publish_root(root: Path) -> str:
    """A minimal namespace shape so the real publish path can be exercised."""

    (root / "recipe").mkdir(parents=True)
    for name in ("README.md", "POLICY.md", "analysis-plan.json"):
        (root / name).write_bytes(b"synthetic self-check placeholder\n")
    # The publish check pins the executing builder byte for byte.
    (root / "recipe" / "build.py").write_bytes(Path(__file__).read_bytes())
    verifier = root / "recipe" / "verify.py"
    verifier.write_bytes(b"# synthetic self-check verifier placeholder\n")
    return identity_of_path(verifier)["sha256"]


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


def _expect(condition: bool, code: str) -> None:
    if not condition:
        _raise(code)


def _expect_build_error(code: str, action: Callable[[], Any], check_code: str) -> None:
    try:
        action()
    except BuildError as error:
        if error.code != code:
            _raise(check_code)
        return
    _raise(check_code)


def _read_corpus(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        _raise("selfcheck_corpus_not_newline_terminated")
    records = []
    for line in raw.split(b"\n"):
        if not line:
            continue
        records.append(load_canonical(line, expected=dict))
    return records


def _check_golden_vectors() -> int:
    plan = load_json(
        _read_regular(NAMESPACE_ROOT / "analysis-plan.json", max_bytes=4_000_000),
        expected=dict,
    )
    vectors = plan["partition"]["golden_vectors"]
    count = verify_partition_golden_vectors(vectors)
    _expect(count == 3, "selfcheck_golden_vector_count")
    # Spelled out as unit assertions so a silent plan edit cannot weaken them.
    for display, bucket, partition in (
        ("local\0evt-a", 77, SHADOW_PARTITION),
        ("alt\0evt-a", 7, HOLDOUT_PARTITION),
        ("local\0evt-c", 60, SHADOW_PARTITION),
    ):
        representative = display.encode("utf-8")
        _expect(partition_bucket(representative) == bucket, "selfcheck_golden_bucket")
        _expect(
            partition_for_representative(representative) == partition,
            "selfcheck_golden_partition",
        )
    return count


def _check_frozen_contracts() -> None:
    """Replay the plan's own vectors and closed lists against this module.

    Every list below is frozen at a stated size, so a future edit that widened
    or narrowed one is caught here rather than in a sealed packet.
    """

    plan = load_json(
        _read_regular(NAMESPACE_ROOT / "analysis-plan.json", max_bytes=4_000_000),
        expected=dict,
    )
    selection = plan["selection"]
    replay = plan["replayability"]

    _expect(len(FORBIDDEN_FILTER_FIELDS) == 16, "selfcheck_forbidden_filter_count")
    _expect(
        tuple(selection["forbidden_filter_fields"]) == FORBIDDEN_FILTER_FIELDS,
        "selfcheck_forbidden_filter_drift",
    )
    _expect(len(FORBIDDEN_REPLAY_INPUTS) == 9, "selfcheck_forbidden_replay_input_count")
    _expect(
        tuple(replay["forbidden_inputs"]) == FORBIDDEN_REPLAY_INPUTS,
        "selfcheck_forbidden_replay_input_drift",
    )
    _expect(len(AMBIENT_KEYS) == 12, "selfcheck_ambient_key_count")
    _expect(
        tuple(replay["replay_input_schema"]["ambient_context"]["behaviorally_relevant_keys_exactly"])
        == AMBIENT_KEYS,
        "selfcheck_ambient_key_drift",
    )
    _expect(
        len(PERSISTED_SOURCE_FIELDS_REQUIRED) == 12,
        "selfcheck_persisted_field_count",
    )
    _expect(
        tuple(replay["persisted_source_fields_required"]) == PERSISTED_SOURCE_FIELDS_REQUIRED,
        "selfcheck_persisted_field_drift",
    )
    _expect(
        set(PERSISTED_SOURCE_FIELDS_REQUIRED).issubset(SOURCE_EVENT_COLUMNS),
        "selfcheck_persisted_field_not_read",
    )
    _expect(
        len(FORBIDDEN_OUTSIDE_SEALED_CORPUS) == 15,
        "selfcheck_privacy_list_count",
    )
    _expect(
        tuple(
            plan["privacy"][
                "forbidden_in_prompts_logs_diagnostics_or_tracked_nonsealed_artifacts"
            ]
        )
        == FORBIDDEN_OUTSIDE_SEALED_CORPUS,
        "selfcheck_privacy_list_drift",
    )
    _expect(
        len(plan["hash_binding"]["packet_manifest_must_bind"]) == len(MANIFEST_BINDING_KEYS),
        "selfcheck_manifest_binding_arity",
    )
    _expect(
        replay["shared_seed_state_validation"]["contract_id"] == PRODUCTION_SEED_CONTRACT_ID,
        "selfcheck_seed_contract_id_drift",
    )
    _expect(
        tuple(sorted(SEED_COLUMN_ALLOWLIST))
        == tuple(
            sorted(
                plan["measurement"]["state_isolation"]["construction"][
                    "copied_table_column_allowlist"
                ]
            )
        ),
        "selfcheck_seed_table_drift",
    )
    for table, columns in plan["measurement"]["state_isolation"]["construction"][
        "copied_table_column_allowlist"
    ].items():
        _expect(
            tuple(columns) == tuple(packet for packet, _ in SEED_COLUMN_ALLOWLIST[table]),
            "selfcheck_seed_column_drift",
        )

    # The three ``created_at`` clauses, replayed from the plan's own vectors.
    release_us = timestamp_microseconds(
        selection["predicate"]["created_at"]["gt_release_effective_at"], "vector_release"
    )
    for vector in selection["initial_boundary_vectors"]:
        included = timestamp_microseconds(vector["created_at"], "vector") > release_us
        _expect(included == vector["included"], "selfcheck_lower_boundary_vector")
    for vector in selection["upper_boundary_vectors"]:
        included = timestamp_microseconds(vector["created_at"], "vector") <= (
            timestamp_microseconds(vector["scheduled_at"], "vector")
        )
        _expect(included == vector["included"], "selfcheck_upper_boundary_vector")
    _expect(
        selection["predicate"]["scope_filter"] is None
        and selection["predicate"]["replayability_filter"] is None
        and selection["predicate"]["outcome_filtering"] is False,
        "selfcheck_selection_predicate_drift",
    )

    # Retrieval-policy keys use their own grammar, never the node-scope one.
    for vector in replay["shared_seed_state_validation"]["policy_key_vectors"]:
        if vector["accepted"]:
            _expect(
                production_policy_key_kind(vector["input"]) == vector["kind"],
                "selfcheck_policy_key_vector",
            )
        else:
            _expect_build_error(
                "retrieval_policy_key_invalid",
                lambda value=vector["input"]: production_policy_key_kind(value),
                "selfcheck_policy_key_vector",
            )
    _expect(
        not node_scope_valid("default") and production_policy_key_kind("default") == "default",
        "selfcheck_policy_key_grammar_conflated",
    )

    # The requested-scope grammar admits exactly three shapes.
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
        _expect(node_scope_valid(value) is valid, "selfcheck_requested_scope_grammar")


def _check_deidentification() -> None:
    secret = Secrets()
    try:
        deidentifier = Deidentifier(secret)
        for value in ("alpha beta", "alpha  beta", "  padded\tvalue\n", "", "one"):
            surrogate = deidentifier.surrogate(value)
            _expect(len(surrogate) == len(value), "selfcheck_deid_length")
            for left, right in zip(surrogate, value, strict=True):
                _expect(
                    (right in KEEP_WHITESPACE) == (left in KEEP_WHITESPACE)
                    and (right not in KEEP_WHITESPACE or left == right),
                    "selfcheck_deid_whitespace",
                )
            _expect(
                all(
                    character in KEEP_WHITESPACE or character in LOWERCASE
                    for character in surrogate
                ),
                "selfcheck_deid_alphabet",
            )
        _expect(
            deidentifier.surrogate("alpha beta") != deidentifier.surrogate("alpha gamma"),
            "selfcheck_deid_injective",
        )
        _expect(
            deidentifier.surrogate("alpha beta") == deidentifier.surrogate("alpha beta"),
            "selfcheck_deid_stable",
        )
        # One identity family, two whitespace layouts: the normalized
        # surrogates must agree or the family would split.
        _expect(
            deidentifier.normalized_surrogate_of("alpha  beta")
            == deidentifier.normalized_surrogate_of("alpha beta"),
            "selfcheck_deid_identity_equivalence",
        )
    finally:
        secret.destroy()


def _check_seed_states(corpus_dir: Path) -> None:
    expected_columns = {
        table: {packet for packet, _ in columns}
        for table, columns in SEED_COLUMN_ALLOWLIST.items()
    }
    for partition in PARTITIONS:
        for alias in SOURCE_ALIASES:
            path = corpus_dir / seed_state_name(partition, alias)
            connection = open_snapshot_readonly(path)
            try:
                names = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM main.sqlite_schema WHERE type='table'"
                    )
                    if not str(row[0]).startswith("sqlite_")
                }
                _expect(names == set(SEED_TABLE_ORDER), "selfcheck_seed_table_set")
                for table, columns in expected_columns.items():
                    observed = {
                        row[1]
                        for row in connection.execute(f"PRAGMA main.table_info({table})")
                    }
                    _expect(observed == columns, "selfcheck_seed_column_allowlist")
                for table in FORCED_EMPTY_TABLES:
                    count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    _expect(count == 0, "selfcheck_seed_forced_empty")
                kinds = {
                    production_policy_key_kind(row[0])
                    for row in connection.execute(
                        "SELECT packet_scope_label FROM retrieval_weights"
                    )
                }
                _expect(
                    REQUIRED_PRODUCTION_POLICY_KEYS.issubset(kinds),
                    "selfcheck_seed_policy_keys",
                )
                nodes = connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
                _expect(nodes > 0, "selfcheck_seed_nodes_present")
            finally:
                connection.close()


def _check_privacy(packet_dir: Path, markers: Sequence[bytes]) -> None:
    """No raw value may reach anything this builder wrote.

    The scan covers every produced byte, not just the non-sealed manifest: the
    sealed corpus is allowed to carry *de-identified* cases, so a raw query,
    scope, agent, path or event identifier appearing there would mean
    de-identification silently failed.  Files that were already in the publish
    root -- the protocol documents and this recipe directory -- are inputs, not
    outputs, and are deliberately out of scope.
    """

    produced = [packet_dir / PACKET_MANIFEST_NAME]
    produced.extend(packet_dir / PACKET_CORPUS_DIRECTORY / name for name in CORPUS_MEMBER_NAMES)
    for path in produced:
        _expect(path.is_file(), "selfcheck_produced_artifact_missing")
        raw = path.read_bytes()
        for marker in markers:
            _expect(marker not in raw, "selfcheck_privacy_leak")


def _check_corpus_invariants(packet_dir: Path, counts: Mapping[str, int]) -> None:
    corpus = packet_dir / PACKET_CORPUS_DIRECTORY
    holdout = _read_corpus(corpus / HOLDOUT_CORPUS_NAME)
    shadow = _read_corpus(corpus / SHADOW_CORPUS_NAME)

    _expect(
        len(holdout) + len(shadow) == counts["selected_event_count"],
        "selfcheck_union_incomplete",
    )
    labels_holdout = {record["event_label"] for record in holdout}
    labels_shadow = {record["event_label"] for record in shadow}
    _expect(not labels_holdout & labels_shadow, "selfcheck_intersection_nonempty")
    _expect(
        len(labels_holdout) + len(labels_shadow) == counts["selected_event_count"],
        "selfcheck_duplicate_event_label",
    )

    cases = [r for r in (*holdout, *shadow) if r["record_kind"] == "case"]
    inventory = [r for r in (*holdout, *shadow) if r["record_kind"] == "inventory"]
    _expect(
        len(cases) == counts["selected_replayable_event_count"],
        "selfcheck_replayable_count",
    )
    _expect(
        len(inventory) == counts["selected_nonreplayable_event_count"],
        "selfcheck_nonreplayable_count",
    )
    _expect(len(inventory) > 0, "selfcheck_nonreplayable_absent")

    for record in inventory:
        require_exact_keys(record, INVENTORY_RECORD_KEYS, "selfcheck_inventory")
        _expect(record["replayable"] is False, "selfcheck_inventory_replayable_flag")
    for record in cases:
        require_exact_keys(record, CASE_RECORD_KEYS, "selfcheck_case")
        require_exact_keys(record["replay_input"], REPLAY_INPUT_KEYS, "selfcheck_replay_input")
        _expect(
            record["seed_state"] in CORPUS_MEMBER_NAMES
            and record["seed_state"].startswith(f"seed-state-{record['partition']}-"),
            "selfcheck_case_seed_state",
        )
        for forbidden in (*FORBIDDEN_FILTER_FIELDS, *FORBIDDEN_REPLAY_INPUTS):
            _expect(forbidden not in record["replay_input"], "selfcheck_case_forbidden_input")

    # No component, family or workflow may straddle the split.
    for field in ("component_label", "family_label", "workflow_label"):
        by_value: dict[Any, str] = {}
        for record in (*holdout, *shadow):
            value = record.get(field)
            if value is None:
                continue
            previous = by_value.setdefault(value, record["partition"])
            _expect(previous == record["partition"], f"selfcheck_{field}_crosses_split")

    floor_counted = [record for record in cases if record["floor_counted"]]
    _expect(len(floor_counted) > 0, "selfcheck_no_floor_counted_case")
    for record in floor_counted:
        _expect(
            bool(record["replay_input"]["query"]),
            "selfcheck_floor_counted_case_incomplete",
        )


def _check_manifest(packet_dir: Path, counts: Mapping[str, int]) -> dict[str, Any]:
    manifest_path = packet_dir / PACKET_MANIFEST_NAME
    manifest = require_exact_keys(
        load_canonical(manifest_path.read_bytes(), expected=dict), MANIFEST_KEYS, "selfcheck_manifest"
    )
    bindings = require_exact_keys(
        manifest["bindings"], MANIFEST_BINDING_KEYS, "selfcheck_manifest_bindings"
    )
    _expect(len(bindings) == 26, "selfcheck_manifest_binding_count")
    _expect(manifest["frozen"] is True, "selfcheck_manifest_not_frozen")
    _expect(manifest["synthetic_fixture"] is True, "selfcheck_manifest_not_synthetic")
    _expect(
        manifest["aggregate_counts"] == dict(counts), "selfcheck_manifest_counts_mismatch"
    )

    corpus = packet_dir / PACKET_CORPUS_DIRECTORY
    for name, key in (
        (HOLDOUT_CORPUS_NAME, "complete_selected_holdout_partition"),
        (SHADOW_CORPUS_NAME, "complete_selected_shadow_partition"),
        (PRESEAL_RECEIPT_NAME, "keyed_aggregate_preseal_receipt"),
    ):
        _expect(bindings[key] == identity_of_path(corpus / name), "selfcheck_binding_mismatch")
    seeds = bindings["partition_source_seed_states_and_construction_manifest"]
    for name in CORPUS_MEMBER_NAMES:
        if name.startswith("seed-state-"):
            _expect(
                seeds.get(name) == identity_of_path(corpus / name),
                "selfcheck_seed_binding_mismatch",
            )

    # Manifest last: every content byte predates it.
    manifest_mtime = manifest_path.stat().st_mtime_ns
    for name in CORPUS_MEMBER_NAMES:
        _expect(
            (corpus / name).stat().st_mtime_ns <= manifest_mtime,
            "selfcheck_manifest_not_last",
        )
    return manifest


def run_self_check(work_dir: Path) -> dict[str, Any]:
    """Build one complete synthetic packet and falsify the claims about it."""

    work = require_output_root_outside_namespace(work_dir, sealing=False)
    _, work_fd, _ = _prepare_private_directory(work, label="work")
    os.close(work_fd)
    checks = 0

    _expect(_check_golden_vectors() == 3, "selfcheck_golden_vector_count")
    checks += 1
    _check_frozen_contracts()
    checks += 1
    _check_deidentification()
    checks += 1

    # A namespace-resident output root is refused outside the seal path.
    _expect_build_error(
        NAMESPACE_REFUSAL_CODE,
        lambda: require_output_root_outside_namespace(
            NAMESPACE_ROOT / "recipe" / "scratch", sealing=False
        ),
        "selfcheck_namespace_refusal_missing",
    )
    _expect_build_error(
        NAMESPACE_REFUSAL_CODE,
        lambda: run_self_check(NAMESPACE_ROOT / "scratch"),
        "selfcheck_namespace_refusal_missing",
    )
    checks += 1

    fixtures = work / "fixtures"
    fixtures.mkdir(mode=0o700)
    for alias in SOURCE_ALIASES:
        write_fixture_snapshot(
            fixtures / f"{alias}.sqlite3",
            alias=alias,
            outcome_marker="base",
            reverse_rows=False,
        )
    write_fixture_dev_reference(fixtures / "dev-reference.jsonl")
    seal_launcher = fixtures / "seal-launcher.py"
    seal_launcher.write_bytes(b"# synthetic self-check seal launcher placeholder\n")

    handoff = build_fixture_handoff(fixture_dir=fixtures, seal_launcher_path=seal_launcher)
    handoff_path = fixtures / "handoff.json"
    handoff_path.write_bytes(canonical_bytes(handoff))

    state = run_draft(draft_dir=work / "draft", handoff_path=handoff_path)
    counts = state["aggregate_counts"]
    _expect(floors_pass(counts), "selfcheck_floors_not_met")
    checks += 1

    packet = work / "packet"
    packet.mkdir(mode=0o755)
    verifier_sha256 = write_fixture_publish_root(packet)
    result = run_freeze(
        draft_dir=work / "draft", publish_dir=packet, verifier_sha256=verifier_sha256
    )
    checks += 1

    manifest = _check_manifest(packet, counts)
    checks += 1
    _check_corpus_invariants(packet, counts)
    checks += 1
    _check_seed_states(packet / PACKET_CORPUS_DIRECTORY)
    checks += 1

    markers = [
        b"proj-alpha",
        b"organic recall",
        b"repeated automatic probe",
        b"-organic-session-",
        b"-agent-",
        b"/home/fixture/",
        b"local-auto-",
        b"alt-org-",
    ]
    _check_privacy(packet, markers)
    checks += 1

    # A second publication attempt cannot replace a sealed byte.
    _expect_build_error(
        "publish_target_exists",
        lambda: run_freeze(
            draft_dir=work / "draft", publish_dir=packet, verifier_sha256=verifier_sha256
        ),
        "selfcheck_overwrite_not_refused",
    )
    checks += 1

    # Outcome fields and input order cannot move a single event.
    invariance = work / "invariance"
    invariance.mkdir(mode=0o700)
    for alias in SOURCE_ALIASES:
        write_fixture_snapshot(
            invariance / f"{alias}.sqlite3",
            alias=alias,
            outcome_marker="perturbed",
            reverse_rows=True,
        )
    write_fixture_dev_reference(invariance / "dev-reference.jsonl")
    invariance_launcher = invariance / "seal-launcher.py"
    invariance_launcher.write_bytes(b"# synthetic self-check seal launcher placeholder\n")
    invariance_handoff = build_fixture_handoff(
        fixture_dir=invariance, seal_launcher_path=invariance_launcher
    )
    invariance_handoff_path = invariance / "handoff.json"
    invariance_handoff_path.write_bytes(canonical_bytes(invariance_handoff))
    invariance_state = run_draft(
        draft_dir=work / "draft-invariance", handoff_path=invariance_handoff_path
    )
    _expect(
        invariance_state["aggregate_counts"] == counts,
        "selfcheck_outcome_or_order_influenced_membership",
    )
    checks += 1

    return {
        "namespace": NAMESPACE,
        "schema_version": SCHEMA_VERSION,
        "status": "self-check-pass",
        "checks": checks,
        "aggregate_counts": dict(counts),
        "floors_pass": True,
        "packet_manifest_sha256_and_bytes": dict(result["manifest"]),
        "manifest_binding_count": len(manifest["bindings"]),
        "semantic_reads": {
            "confirmatory-shadow-v4-eval": 0,
            "confirmatory-holdout-v4-eval": 0,
        },
        "synthetic_fixture": True,
        "real_packet_sealed": False,
    }


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="confirmatory-holdout-v4 packet builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "self-check", help="build a complete synthetic packet under a temporary directory"
    )
    check.add_argument("--work-dir", type=Path, required=True)

    draft = subparsers.add_parser("draft", help="build packet content from a real handoff")
    draft.add_argument("--draft-dir", type=Path, required=True)
    draft.add_argument("--handoff", type=Path, required=True)

    freeze = subparsers.add_parser(
        "freeze", help="install content, then write the manifest last"
    )
    freeze.add_argument("--draft-dir", type=Path, required=True)
    freeze.add_argument("--publish-dir", type=Path, required=True)
    freeze.add_argument("--verifier-sha256", required=True)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = parse_arguments(argv)
        if arguments.command == "self-check":
            summary = run_self_check(arguments.work_dir)
        elif arguments.command == "draft":
            state = run_draft(
                draft_dir=arguments.draft_dir, handoff_path=arguments.handoff
            )
            summary = {
                "namespace": NAMESPACE,
                "schema_version": SCHEMA_VERSION,
                "status": "draft",
                "aggregate_counts": dict(state["aggregate_counts"]),
                "floors_pass": bool(state["floors_pass"]),
                "semantic_reads": {
                    "confirmatory-shadow-v4-eval": 0,
                    "confirmatory-holdout-v4-eval": 0,
                },
                "synthetic_fixture": bool(state["handoff"]["synthetic_fixture"]),
            }
        else:
            result = run_freeze(
                draft_dir=arguments.draft_dir,
                publish_dir=arguments.publish_dir,
                verifier_sha256=arguments.verifier_sha256,
            )
            summary = {
                "namespace": NAMESPACE,
                "schema_version": SCHEMA_VERSION,
                "status": "sealed",
                "aggregate_counts": dict(result["aggregate_counts"]),
                "floors_pass": bool(result["floors_pass"]),
                "packet_manifest_sha256_and_bytes": dict(result["manifest"]),
                "semantic_reads": {
                    "confirmatory-shadow-v4-eval": 0,
                    "confirmatory-holdout-v4-eval": 0,
                },
                "synthetic_fixture": bool(result["synthetic_fixture"]),
            }
        sys.stdout.write(
            json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n"
        )
        return EXIT_OK
    except BuildError as error:
        sys.stderr.write(
            json.dumps({"status": "error", "code": error.code}, sort_keys=True)
            + "\n"
        )
        return _exit_code_for(error.code)
    except Exception:
        # Fail closed without letting an object representation reach stderr.
        sys.stderr.write(
            json.dumps({"status": "error", "code": "internal_failure"}, sort_keys=True)
            + "\n"
        )
        return EXIT_BUILD_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
