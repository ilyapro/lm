#!/usr/bin/env python3
"""The trusted read-only source snapshot launcher for ``confirmatory-holdout-v4``.

:mod:`ap_confirmatory_probe_v4` states that "a trusted launcher supplies
already-open, read-only snapshots and runtime objects".  This module is that
launcher.  It owns exactly one boundary: turning the two frozen source aliases
into one immutable SQLite snapshot each and handing the probe already-open
read-only descriptors through the exact ``capture_snapshot(alias, alias_id,
database_instance_identity_sha256) -> int`` callback that
``compute_probe_aggregate`` invokes.

Three frozen rules shape every decision here.

*Nothing opens before the window is proven.*  A :class:`SourceSnapshotLauncher`
cannot be constructed without a :class:`SlotWindow`, and a ``SlotWindow`` cannot
exist unless the observed instant lies in the half-open ``[slot_at(i),
slot_at(i) + grace)`` interval of the frozen calendar.  The window is re-derived
from the frozen schedule again on every capture, so a forged wrapper is refused
before a source is touched.

*A pre-source failure must be provable.*  ``POLICY.md`` authorizes an in-grace
retry only when the launcher proves it failed before key creation and before any
source open.  The launcher therefore counts live-source opens itself and records
one explicit ``mark_keyed_worker_handoff`` transition, and
:meth:`SourceSnapshotLauncher.failure_facts` reports the frozen phase implied by
those two proven facts.  Everything the launcher can validate on its own --
locator shape, alias distinctness, staging root, window -- is validated in the
constructor, which runs before the handoff and therefore stays retryable.

*The launcher never names a source.*  Alias locators are supplied by the caller
and are operator-private, matching the plan's ``external-untracked-operator-
mapping``; the module contains no source path, transport, or host of any kind
and deliberately exposes no command line that could open one.  Its only public
entry points are typed calls made by a trusted in-process driver.

Both aliases are addressed identically, as one resolved local filesystem path
each.  Resolving an alias whose authority is not local into such a path is the
operator's untracked responsibility, performed before the slot opens; the
launcher deliberately owns no transport, so no tracked file here can ever carry
an operator mapping.  The alias/authority/database binding those locators must
satisfy is proven separately by the runtime observer, and the identity the
launcher checks arrives through ``observe_database_identity`` -- which observes
the live configured database, not whatever local file the operator resolved.

The snapshot itself is produced with ``VACUUM INTO`` over a ``mode=ro``
connection: it never writes to the source, it is consistent even against a
live writer, and its output is always a self-contained rollback-journal file --
exactly what the probe's ``immutable=1`` open requires and what a raw byte copy
of a WAL database could not give.

Every integrity error is the shared, message-free
:class:`~ap_confirmatory_runtime_v4.IntegrityFailure`, so nothing about a
private source can escape through a diagnostic.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import sys
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping


# Import through the probe's canonical runtime module.  Both upstream modules
# use exact dataclass type identity as a wrapper-integrity check, so a second
# copy of the runtime would be rejected by everything downstream.
try:
    import ap_confirmatory_probe_v4 as probe
except ModuleNotFoundError:  # pragma: no cover - exercised by importlib callers
    import importlib.util

    _PROBE_PATH = Path(__file__).resolve().with_name(
        "ap_confirmatory_probe_v4.py"
    )
    _PROBE_SPEC = importlib.util.spec_from_file_location(
        "ap_confirmatory_probe_v4", _PROBE_PATH
    )
    if _PROBE_SPEC is None or _PROBE_SPEC.loader is None:
        raise
    probe = importlib.util.module_from_spec(_PROBE_SPEC)
    sys.modules[_PROBE_SPEC.name] = probe
    _PROBE_SPEC.loader.exec_module(probe)


runtime = probe.runtime
IntegrityFailure = runtime.IntegrityFailure

NAMESPACE = runtime.NAMESPACE
SCHEMA_VERSION = runtime.SCHEMA_VERSION
SOURCE_ALIASES = runtime.SOURCE_ALIASES
ALIAS_IDS = runtime.ALIAS_IDS

REPO_ROOT = Path(__file__).resolve().parent.parent
NAMESPACE_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "confirmatory-holdout-v4"
)

SNAPSHOT_SET_DOMAIN = b"confirmatory-holdout-v4/snapshot-set/v1"

# ``sources.aliases.*.snapshot_open`` verbatim.  The probe opens the retained
# descriptor with the identical query string; the launcher proves the snapshot
# survives that exact open before it hands the descriptor over.
SNAPSHOT_OPEN_QUERY = "mode=ro&immutable=1&cache=private"

SQLITE_HEADER_BYTES = 100
SQLITE_MAGIC = b"SQLite format 3\0"
# Header bytes 18/19 are the file-format write/read versions.  ``1`` is the
# rollback journal; ``2`` is WAL, whose uncheckpointed pages an ``immutable=1``
# open would silently ignore.
ROLLBACK_JOURNAL_FORMAT = 1

SNAPSHOT_FILE_MODE = 0o400
STAGING_DIRECTORY_MODE = 0o700

# The subset of ``probe.PHASE_PROGRESS`` the launcher can prove on its own,
# with the exact ``(source_open_count, key_created)`` progress the frozen probe
# assigns to each name.
LAUNCHER_PHASE_PROGRESS = (
    ("pre-key-pre-source", 0, False),
    ("key-created-pre-source", 0, True),
    ("first-source-opened", 1, True),
    ("snapshots-latched", 2, True),
)
RETRYABLE_PHASE = "pre-key-pre-source"
RETRYABLE_FAILURE_CLASS = "pre-source-retryable"
TERMINAL_FAILURE_CLASS = "post-source-terminal"

# The exact ``probe.PROBE_FAILURE_FIELDS`` members the launcher is the witness
# for.  The driver owns everything else in that receipt.
FAILURE_MARKER_FIELDS = (
    "phase_at_failure",
    "failure_class",
    "source_open_count_or_null",
    "key_created_or_null",
    "retry_authorized",
)


def _fail() -> None:
    raise IntegrityFailure from None


def _is_int(value: Any) -> bool:
    return type(value) is int


def _hex64(value: Any) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail()
    return value


def _exact_mapping(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        _fail()
    return value


def _require_frozen_phase_alignment() -> None:
    """Refuse to load if the frozen probe's phase ledger ever drifts.

    The retry authorization in ``POLICY.md`` is expressed entirely through
    these ``(source_open_count, key_created)`` pairs.  If the probe renamed a
    phase or moved a counter, a launcher that kept its own copy would start
    minting markers the probe's validator rejects -- or worse, markers it
    accepts with the wrong retry verdict.
    """

    progress = probe.PHASE_PROGRESS
    if type(progress) is not dict:
        _fail()
    for phase, count, key_created in LAUNCHER_PHASE_PROGRESS:
        if progress.get(phase) != (count, key_created):
            _fail()
    if len({phase for phase, _count, _key in LAUNCHER_PHASE_PROGRESS}) != len(
        LAUNCHER_PHASE_PROGRESS
    ):
        _fail()
    if not set(FAILURE_MARKER_FIELDS).issubset(probe.PROBE_FAILURE_FIELDS):
        _fail()
    if set(SOURCE_ALIASES) != set(ALIAS_IDS):
        _fail()


_require_frozen_phase_alignment()


def _identity(value: Any) -> runtime.HashAndBytes:
    if type(value) is runtime.HashAndBytes:
        return value
    return runtime.HashAndBytes.from_value(value)


def derive_snapshot_set_identity(value: Any) -> runtime.HashAndBytes:
    """``domains.snapshot_set_preimage`` and ``snapshot_set_value``, spelled out.

    The preimage is ``snapshot_set_utf8 + NUL +
    v4_canonical_json_utf8({local:{sha256,bytes},alt:{sha256,bytes}})``; the
    published ``sha256`` is the digest of that preimage and ``bytes`` is the
    preimage's own length, not any snapshot's length.  ``sources`` declares
    ``equal_snapshot_sha256`` a fatal integrity error, so two aliases that
    hashed alike can never produce a set identity.
    """

    mapping = _exact_mapping(value, SOURCE_ALIASES)
    identities = {alias: _identity(mapping[alias]) for alias in SOURCE_ALIASES}
    if any(identity.bytes <= 0 for identity in identities.values()):
        _fail()
    if len({identity.sha256 for identity in identities.values()}) != len(
        SOURCE_ALIASES
    ):
        _fail()
    normalized = {
        alias: identities[alias].as_dict() for alias in SOURCE_ALIASES
    }
    preimage = (
        SNAPSHOT_SET_DOMAIN + b"\0" + runtime.canonical_json_bytes(normalized)
    )
    return runtime.HashAndBytes(
        hashlib.sha256(preimage).hexdigest(), len(preimage)
    )


@dataclass(frozen=True, slots=True)
class SlotWindow:
    """Proof that one observed instant fell inside one frozen slot window."""

    slot_index: int
    scheduled_at: str
    grace_deadline_at: str
    observed_at: str


def open_slot_window(slot_index: Any, observed_at: Any) -> SlotWindow:
    """Prove ``slot_at(i) <= observed_at < slot_at(i) + grace_seconds``.

    The bounds come from :func:`ap_confirmatory_runtime_v4.slot_times`, so the
    calendar cannot be re-anchored here.  The upper bound is exclusive because
    ``POLICY.md`` defines the execution window as half-open.
    """

    slot = runtime.slot_times(slot_index)
    observed = runtime.parse_utc(observed_at, receipt=True)
    scheduled = runtime.parse_utc(slot.scheduled_at, receipt=True)
    grace = runtime.parse_utc(slot.grace_deadline_at, receipt=True)
    if not scheduled <= observed < grace:
        _fail()
    return SlotWindow(
        slot_index=slot.slot_index,
        scheduled_at=slot.scheduled_at,
        grace_deadline_at=slot.grace_deadline_at,
        observed_at=runtime.canonical_utc(observed),
    )


def open_slot_window_now(slot_index: Any) -> SlotWindow:
    """Prove the window against the real clock.

    There is deliberately no override parameter and no environment lookup: a
    launcher that could be told what time it is could open a source outside its
    authorized window, which the frozen schedule treats as unrecoverable.
    """

    return open_slot_window(
        slot_index, runtime.canonical_utc(datetime.now(UTC))
    )


@dataclass(frozen=True, slots=True)
class SourceLocator:
    """One operator-private alias mapping.  The launcher stores no default."""

    alias: str
    path: Path


@dataclass(frozen=True, slots=True)
class FailureFacts:
    """The part of a ``probe-failure`` receipt the launcher can prove."""

    phase_at_failure: str
    failure_class: str
    source_open_count: int
    key_created: bool
    retry_authorized: bool

    def marker_fields(self) -> dict[str, Any]:
        """Return the exact frozen receipt members, ready for the driver."""

        return {
            "phase_at_failure": self.phase_at_failure,
            "failure_class": self.failure_class,
            "source_open_count_or_null": self.source_open_count,
            "key_created_or_null": self.key_created,
            "retry_authorized": self.retry_authorized,
        }


def failure_facts_for(source_open_count: Any, key_created: Any) -> FailureFacts:
    """Map the two proven counters onto their one frozen phase.

    ``probe.PHASE_PROGRESS`` maps three distinct late phases onto ``(2, True)``;
    only ``snapshots-latched`` is one the launcher itself reaches, so the
    inverse is single-valued over :data:`LAUNCHER_PHASE_PROGRESS`.
    """

    if not _is_int(source_open_count) or type(key_created) is not bool:
        _fail()
    for phase, count, key in LAUNCHER_PHASE_PROGRESS:
        if (count, key) != (source_open_count, key_created):
            continue
        retryable = phase == RETRYABLE_PHASE
        return FailureFacts(
            phase_at_failure=phase,
            failure_class=(
                RETRYABLE_FAILURE_CLASS if retryable else TERMINAL_FAILURE_CLASS
            ),
            source_open_count=source_open_count,
            key_created=key_created,
            retry_authorized=retryable,
        )
    _fail()


@dataclass(frozen=True, slots=True)
class SnapshotHandle:
    """One latched immutable snapshot and its retained read-only descriptor."""

    alias: str
    alias_id: str
    path: Path
    fd: int
    identity: runtime.HashAndBytes
    database_instance_identity_sha256: str


def _normalized(path: Any) -> Path:
    if type(path) not in (str, type(Path())):
        _fail()
    candidate = Path(path)
    if ".." in candidate.parts or not candidate.is_absolute():
        _fail()
    return Path(os.path.realpath(os.fspath(candidate)))


def require_root_outside_namespace(path: Any) -> Path:
    """Refuse any staging root resolving inside the byte-locked namespace."""

    candidate = _normalized(path)
    namespace = Path(os.path.realpath(os.fspath(NAMESPACE_ROOT)))
    if candidate == namespace or namespace in candidate.parents:
        _fail()
    return candidate


def _lstat_regular(path: Path) -> os.stat_result:
    try:
        info = os.lstat(path)
    except (OSError, ValueError):
        _fail()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        _fail()
    return info


def _open_readonly_regular(path: Path) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except (OSError, ValueError):
        _fail()
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            _fail()
        os.set_inheritable(fd, False)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _hash_fd(fd: int) -> runtime.HashAndBytes:
    """Hash one stable regular descriptor without disturbing its offset."""

    try:
        before = os.fstat(fd)
    except OSError:
        _fail()
    if not stat.S_ISREG(before.st_mode):
        _fail()
    digest = hashlib.sha256()
    offset = 0
    while True:
        try:
            chunk = os.pread(fd, probe.READ_CHUNK_BYTES, offset)
        except OSError:
            _fail()
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    try:
        after = os.fstat(fd)
    except OSError:
        _fail()
    if _stat_identity(before) != _stat_identity(after) or offset != after.st_size:
        _fail()
    return runtime.HashAndBytes(digest.hexdigest(), offset)


def _source_uri(path: Path) -> str:
    """Build the read-only source URI without letting a path become options."""

    quoted = urllib.parse.quote(os.fspath(path), safe="/")
    return f"file:{quoted}?mode=ro"


def _close_quietly(connection: sqlite3.Connection | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except sqlite3.Error:
        pass


class SourceSnapshotLauncher:
    """Capture one immutable read-only snapshot per frozen source alias.

    The launcher is single-use and single-slot.  Construction validates every
    precondition it can reach without opening a source, so a bad locator, a
    forged window, an aliased pair of sources, or an unusable staging root all
    fail while :meth:`failure_facts` still proves ``pre-key-pre-source`` and the
    in-grace retry remains authorized.
    """

    def __init__(
        self,
        *,
        window: SlotWindow,
        locators: Mapping[str, SourceLocator],
        observe_database_identity: Callable[[str], Mapping[str, Any]],
        staging_root: Path | str,
    ) -> None:
        self._source_open_count = 0
        self._key_created = False
        self._destroyed = False
        self._handles: dict[str, SnapshotHandle] = {}
        self._pre_identities: dict[str, str] | None = None
        self._latched = False
        self._staging_root: Path | None = None

        self._window = _require_window(window)
        self._locators = _require_locators(locators)
        if not callable(observe_database_identity):
            _fail()
        self._observe_database_identity = observe_database_identity

        root = require_root_outside_namespace(staging_root)
        try:
            root.mkdir(mode=STAGING_DIRECTORY_MODE)
            info = root.lstat()
        except (OSError, ValueError):
            _fail()
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != STAGING_DIRECTORY_MODE
        ):
            _fail()
        self._staging_root = root

    # -- proven state -------------------------------------------------------

    @property
    def window(self) -> SlotWindow:
        return self._window

    @property
    def slot_index(self) -> int:
        return self._window.slot_index

    @property
    def source_open_count(self) -> int:
        return self._source_open_count

    @property
    def key_created(self) -> bool:
        return self._key_created

    @property
    def latched(self) -> bool:
        return self._latched

    def failure_facts(self) -> FailureFacts:
        """Report the frozen phase implied by the two counters proven so far.

        Called at any point, including after :meth:`destroy`, because the
        driver needs it precisely when the slot has already gone wrong.
        """

        return failure_facts_for(self._source_open_count, self._key_created)

    def mark_keyed_worker_handoff(self) -> None:
        """Record that the keyed worker is about to exist.

        The frozen probe creates its HMAC key before it calls
        ``capture_snapshot``, so from this call onward the launcher can no
        longer prove ``pre-key-pre-source`` and must not claim an authorized
        retry.  The launcher deliberately does not infer the transition from
        the first capture: a probe that died inside the keyed handshake would
        otherwise leave the counters claiming a retry the policy forbids.
        """

        self._require_live()
        if self._key_created or self._handles:
            _fail()
        self._key_created = True

    # -- the probe's capture boundary ---------------------------------------

    def capture_snapshot(
        self,
        alias: str,
        alias_id: str,
        database_instance_identity_sha256: str,
    ) -> int:
        """The exact callback ``compute_probe_aggregate`` invokes per alias.

        Returns an already-open read-only descriptor for a validated immutable
        snapshot.  The descriptor stays owned by the launcher; the probe
        duplicates it and never closes this one.
        """

        self._require_live()
        _require_window(self._window)
        if not self._key_created:
            _fail()
        if type(alias) is not str or alias not in SOURCE_ALIASES:
            _fail()
        # Exactly one snapshot per alias, in the frozen local-then-alt order
        # the probe and the binding ceremony both use.
        if len(self._handles) >= len(SOURCE_ALIASES):
            _fail()
        if alias != SOURCE_ALIASES[len(self._handles)]:
            _fail()
        if alias_id != ALIAS_IDS[alias]:
            _fail()
        attested = _hex64(database_instance_identity_sha256)

        if self._pre_identities is None:
            # ``probe_binding_capture``: one metadata observation immediately
            # before the first snapshot open.
            self._pre_identities = self._observe_all()
        if self._pre_identities[alias] != attested:
            _fail()

        handle = self._materialize(alias, attested)
        self._handles[alias] = handle
        if len(self._handles) == len(SOURCE_ALIASES):
            # ...and one more immediately after both captures.
            if self._observe_all() != self._pre_identities:
                _fail()
            self._latched = True
        return handle.fd

    # -- latched results ----------------------------------------------------

    def snapshot_identities(self) -> dict[str, runtime.HashAndBytes]:
        self._require_latched()
        return {
            alias: self._handles[alias].identity for alias in SOURCE_ALIASES
        }

    def snapshot_set_identity(self) -> runtime.HashAndBytes:
        return derive_snapshot_set_identity(self.snapshot_identities())

    def snapshot_paths(self) -> dict[str, Path]:
        self._require_latched()
        return {alias: self._handles[alias].path for alias in SOURCE_ALIASES}

    def snapshot_fds(self) -> dict[str, int]:
        self._require_latched()
        return {alias: self._handles[alias].fd for alias in SOURCE_ALIASES}

    def database_instance_identities(self) -> dict[str, str]:
        self._require_latched()
        return {
            alias: self._handles[alias].database_instance_identity_sha256
            for alias in SOURCE_ALIASES
        }

    def destroy(self) -> None:
        """Close every descriptor and remove every snapshot byte.

        ``sources.below_floor_snapshot_action`` requires destruction once the
        aggregate receipt is validator-valid, and a closed segment destroys
        captured snapshots outright.  Destruction is idempotent so a driver can
        call it from a failure path without knowing how far it got.
        """

        self._destroyed = True
        self._latched = False
        for handle in self._handles.values():
            try:
                os.close(handle.fd)
            except OSError:
                pass
            _unlink_quietly(handle.path)
        self._handles = {}
        if self._staging_root is not None:
            for alias in SOURCE_ALIASES:
                _unlink_quietly(self._staging_root / _snapshot_name(alias))
            try:
                self._staging_root.rmdir()
            except OSError:
                pass
            self._staging_root = None

    # -- internals ----------------------------------------------------------

    def _require_live(self) -> None:
        if self._destroyed or self._staging_root is None:
            _fail()

    def _require_latched(self) -> None:
        self._require_live()
        if not self._latched or set(self._handles) != set(SOURCE_ALIASES):
            _fail()

    def _observe_all(self) -> dict[str, str]:
        """Derive both alias database identities, local then alt.

        Only the derived digest is retained: ``database_instance_identity_
        derivation.raw_inputs_persisted`` is false, so the private statx input
        must not outlive this frame.
        """

        observed: dict[str, str] = {}
        for alias in SOURCE_ALIASES:
            try:
                private_input = self._observe_database_identity(alias)
            except IntegrityFailure:
                raise
            except BaseException:
                _fail()
            observed[alias] = runtime.derive_database_instance_identity(
                private_input
            )
        if len(set(observed.values())) != len(SOURCE_ALIASES):
            _fail()
        return observed

    def _materialize(self, alias: str, attested: str) -> SnapshotHandle:
        """Snapshot one live source and validate the result before returning it."""

        assert self._staging_root is not None  # guarded by _require_live
        destination = self._staging_root / _snapshot_name(alias)
        if destination.exists():
            _fail()
        source = self._locators[alias].path

        # From here the live source is opened, so the failure is terminal.
        self._source_open_count += 1
        fd: int | None = None
        try:
            connection: sqlite3.Connection | None = None
            try:
                connection = sqlite3.connect(
                    _source_uri(source), uri=True, isolation_level=None
                )
                # ``VACUUM INTO`` writes only the destination.  ``mode=ro``
                # already forbids every write to the source, and it is the
                # strongest read guarantee available that still yields a
                # consistent copy of a database with a live writer.
                connection.execute("VACUUM INTO ?", (os.fspath(destination),))
            except (sqlite3.Error, OSError, ValueError, TypeError):
                _fail()
            finally:
                _close_quietly(connection)

            info = _lstat_regular(destination)
            if info.st_size <= 0:
                _fail()
            try:
                os.chmod(destination, SNAPSHOT_FILE_MODE)
            except (OSError, ValueError):
                _fail()
            fd = _open_readonly_regular(destination)
            self._require_snapshot_header(fd)
            identity = _hash_fd(fd)
            self._require_distinct(fd, identity)
            self._require_probe_readable(fd)
        except BaseException:
            # A partial or rejected snapshot never survives the attempt: the
            # policy forbids a recapture, so a half-written file could only
            # ever be mistaken for evidence.
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            _unlink_quietly(destination)
            raise
        return SnapshotHandle(
            alias=alias,
            alias_id=ALIAS_IDS[alias],
            path=destination,
            fd=fd,
            identity=identity,
            database_instance_identity_sha256=attested,
        )

    @staticmethod
    def _require_snapshot_header(fd: int) -> None:
        try:
            header = os.pread(fd, SQLITE_HEADER_BYTES, 0)
        except OSError:
            _fail()
        if (
            len(header) != SQLITE_HEADER_BYTES
            or header[: len(SQLITE_MAGIC)] != SQLITE_MAGIC
            or header[18] != ROLLBACK_JOURNAL_FORMAT
            or header[19] != ROLLBACK_JOURNAL_FORMAT
        ):
            _fail()

    def _require_distinct(
        self, fd: int, identity: runtime.HashAndBytes
    ) -> None:
        """``equal_snapshot_sha256`` and a reused source are both fatal."""

        try:
            info = os.fstat(fd)
        except OSError:
            _fail()
        for handle in self._handles.values():
            if handle.identity.sha256 == identity.sha256:
                _fail()
            try:
                existing = os.fstat(handle.fd)
            except OSError:
                _fail()
            if (existing.st_dev, existing.st_ino) == (info.st_dev, info.st_ino):
                _fail()

    @staticmethod
    def _require_probe_readable(fd: int) -> None:
        """Open the snapshot exactly as the probe will, and validate its shape.

        Doing this before the descriptor is handed over turns an unusable
        snapshot into a launcher failure instead of a worker failure deep
        inside the keyed subprocess, where no diagnosis is possible at all.
        """

        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                f"file:/proc/self/fd/{fd}?{SNAPSHOT_OPEN_QUERY}", uri=True
            )
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
            columns = ", ".join(probe.EVENT_COLUMNS)
            connection.execute(
                f"SELECT {columns} FROM main.recall_events LIMIT 0"
            ).fetchall()
            probe.validate_production_seed_state(connection)
        except IntegrityFailure:
            raise
        except (sqlite3.Error, TypeError, ValueError, IndexError):
            _fail()
        finally:
            _close_quietly(connection)


def _snapshot_name(alias: str) -> str:
    if type(alias) is not str or alias not in SOURCE_ALIASES:
        _fail()
    return f"{alias}.sqlite3"


def _unlink_quietly(path: Path) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _require_window(window: Any) -> SlotWindow:
    """Re-derive a caller-supplied window from the frozen schedule.

    ``SlotWindow`` is an ordinary frozen dataclass, so a caller can build one
    with any fields it likes.  Rebinding it through :func:`open_slot_window`
    means only an instant that really falls inside a real frozen slot window
    survives, however the wrapper was produced.
    """

    if type(window) is not SlotWindow:
        _fail()
    if open_slot_window(window.slot_index, window.observed_at) != window:
        _fail()
    return window


def _require_locators(value: Any) -> dict[str, SourceLocator]:
    """Validate both alias mappings without opening either source."""

    mapping = _exact_mapping(value, SOURCE_ALIASES)
    locators: dict[str, SourceLocator] = {}
    identities: set[tuple[int, int]] = set()
    for alias in SOURCE_ALIASES:
        locator = mapping[alias]
        if type(locator) is not SourceLocator or locator.alias != alias:
            _fail()
        path = locator.path
        if type(path) is not type(Path()) or ".." in path.parts:
            _fail()
        if not path.is_absolute():
            _fail()
        info = _lstat_regular(path)
        identity = (info.st_dev, info.st_ino)
        # ``alias swaps, additions, removals, mirrors, duplicate database
        # instances`` are fatal: two aliases may never name one file.
        if identity in identities:
            _fail()
        identities.add(identity)
        locators[alias] = locator
    return locators


def main(argv: Any = None) -> int:
    """No command line exists.

    Every entry point that can open a source takes typed, already-validated
    objects from a trusted in-process driver.  A path-only command line would
    be a standing invitation to point this module at a real database outside a
    slot window, so misuse is deliberately silent, like the probe's.
    """

    del argv
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
