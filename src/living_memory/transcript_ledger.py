"""Offline import of transcript-grounding verdicts (phase 3 of transcript-grounding).

The phase-2 grader emits one JSON object per graded (recall event, node)
delivery: the IDF-containment of the delivered content against the session
transcript remainder *after* the delivery instant, plus the thresholded
``grounded`` verdict and the ``method_version`` that names the calibration
those numbers were produced under. This module is the only writer of
``transcript_grounding_verdicts``; the live recall/remember path neither
reads nor writes the table, and the phase-5 valve is its only intended
consumer.

Import semantics, in decreasing order of importance:

* **Replay, never duplicate.** ``UNIQUE(recall_event_id, node_id,
  method_version)`` is the idempotency key. Re-importing a file whose rows
  are already present counts them as ``replayed`` and writes nothing.
* **Never update.** A same-key row whose numbers differ from what the ledger
  already holds is a method_version discipline violation — the same method
  on the same delivery must grade the same. It is counted as ``mismatched``,
  reported, and NOT applied; the first-written row stands. A legitimate
  re-grade ships under a new ``method_version`` and lands beside the old
  rows.
* **All-or-nothing.** The whole import runs in one transaction, so a crash
  mid-import leaves no partial state and the re-run replays cleanly.
* **Fail fast on malformed input.** A line that does not parse, or a verdict
  outside the documented shape, aborts before anything is written. Because
  import is idempotent, fix-and-re-run is always safe.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import json
import math
import sqlite3
from typing import Any

from living_memory.storage import (
    TRANSCRIPT_GROUNDING_SCHEMA_SQL,
    TRANSCRIPT_GROUNDING_TABLE,
    new_ulid,
)
from living_memory.temporal import parse_timestamp

#: Which store's ``recall_events`` the row's ids belong to. Phases 1-2 run on
#: each host separately (the one-host blind-spot lesson), so a ledger can
#: carry both fields' verdicts without their event ids colliding in meaning.
TRANSCRIPT_GROUNDING_MEASUREMENT_FIELDS = ("local", "alt")

#: How many offending idempotency keys an import report carries verbatim.
#: Enough to locate the discipline violation, bounded so a wholly re-graded
#: file cannot balloon the report.
_MAX_REPORTED_MISMATCHES = 20


class TranscriptLedgerError(ValueError):
    """A verdict payload outside the documented phase-2 shape."""


@dataclass(frozen=True)
class TranscriptGroundingVerdict:
    """One graded delivery, exactly the phase-2 per-event JSONL shape."""

    recall_event_id: str
    node_id: str
    containment: float
    grounded: int
    method_version: str
    field: str
    delivered_at: str
    graded_at: str
    transcript_session_key: str

    @property
    def key(self) -> tuple[str, str, str]:
        """The ledger's idempotency key."""

        return (self.recall_event_id, self.node_id, self.method_version)

    def payload(self) -> tuple[float, int, str, str, str, str]:
        """Everything the replay comparison covers, in one comparable value."""

        return (
            self.containment,
            self.grounded,
            self.field,
            self.delivered_at,
            self.graded_at,
            self.transcript_session_key,
        )


@dataclass(frozen=True)
class TranscriptLedgerImportStats:
    """What one import (or dry run) did, in replayable terms."""

    total: int
    inserted: int
    replayed: int
    mismatched: int
    mismatches: tuple[tuple[str, str, str], ...]
    imported_at: str
    applied: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "inserted": self.inserted,
            "replayed": self.replayed,
            "mismatched": self.mismatched,
            "mismatches": [list(key) for key in self.mismatches],
            "imported_at": self.imported_at,
            "applied": self.applied,
        }


def _required_str(payload: Mapping[str, Any], name: str, source: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise TranscriptLedgerError(
            f"{source}: '{name}' must be a non-empty string, got {value!r}"
        )
    return value


def parse_verdict(
    payload: Mapping[str, Any], *, source: str = "verdict"
) -> TranscriptGroundingVerdict:
    """Validate one phase-2 verdict object into the ledger shape.

    Unknown keys are ignored — the phase-2 report may carry extra diagnostic
    fields — but every ledger column must be present and well-formed.
    """

    if not isinstance(payload, Mapping):
        raise TranscriptLedgerError(f"{source}: expected a JSON object, got {payload!r}")

    recall_event_id = _required_str(payload, "recall_event_id", source)
    node_id = _required_str(payload, "node_id", source)
    method_version = _required_str(payload, "method_version", source)
    transcript_session_key = _required_str(payload, "transcript_session_key", source)

    containment = payload.get("containment")
    if isinstance(containment, bool) or not isinstance(containment, (int, float)):
        raise TranscriptLedgerError(
            f"{source}: 'containment' must be a number, got {containment!r}"
        )
    containment = float(containment)
    if not math.isfinite(containment) or not 0.0 <= containment <= 1.0:
        raise TranscriptLedgerError(
            f"{source}: 'containment' must be within [0.0, 1.0], got {containment!r}"
        )

    grounded = payload.get("grounded")
    if isinstance(grounded, bool):
        grounded = int(grounded)
    if grounded not in (0, 1):
        raise TranscriptLedgerError(
            f"{source}: 'grounded' must be 0, 1 or a boolean, got {payload.get('grounded')!r}"
        )

    field = _required_str(payload, "field", source)
    if field not in TRANSCRIPT_GROUNDING_MEASUREMENT_FIELDS:
        raise TranscriptLedgerError(
            f"{source}: 'field' must be one of"
            f" {TRANSCRIPT_GROUNDING_MEASUREMENT_FIELDS}, got {field!r}"
        )

    timestamps: dict[str, str] = {}
    for name in ("delivered_at", "graded_at"):
        raw = _required_str(payload, name, source)
        if parse_timestamp(raw) is None:
            raise TranscriptLedgerError(
                f"{source}: '{name}' must be an ISO-8601 timestamp, got {raw!r}"
            )
        timestamps[name] = raw

    return TranscriptGroundingVerdict(
        recall_event_id=recall_event_id,
        node_id=node_id,
        containment=containment,
        grounded=int(grounded),
        method_version=method_version,
        field=field,
        delivered_at=timestamps["delivered_at"],
        graded_at=timestamps["graded_at"],
        transcript_session_key=transcript_session_key,
    )


def read_verdicts(path: Path | str) -> list[TranscriptGroundingVerdict]:
    """Parse a phase-2 per-event verdict JSONL file, failing on the first bad line."""

    verdicts: list[TranscriptGroundingVerdict] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            source = f"{path}:{line_number}"
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as error:
                raise TranscriptLedgerError(f"{source}: not valid JSON: {error}") from error
            verdicts.append(parse_verdict(payload, source=source))
    return verdicts


def ensure_ledger_table(conn: sqlite3.Connection) -> None:
    """Create the ledger objects if absent — the importer's half of the shared DDL.

    The same script ``MemoryStore._initialize_schema`` runs on every open, so
    a database the redeployed server has already opened is a no-op here, and
    a file it has not opened yet gains exactly the objects it will later
    carry anyway. Purely additive; touches nothing that exists.
    """

    conn.executescript(TRANSCRIPT_GROUNDING_SCHEMA_SQL)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def import_verdicts(
    conn: sqlite3.Connection,
    verdicts: Iterable[TranscriptGroundingVerdict],
    *,
    imported_at: str | None = None,
    apply: bool = True,
) -> TranscriptLedgerImportStats:
    """Insert absent keys, replay present ones, never touch an existing row.

    With ``apply=False`` the same accounting runs read-only (the dry run):
    in-file duplicates are tracked in memory so the plan matches what a real
    run would do. The caller is responsible for :func:`ensure_ledger_table`.
    """

    stamp = imported_at or _utc_now()
    table_present = (
        conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?",
            (TRANSCRIPT_GROUNDING_TABLE,),
        ).fetchone()
        is not None
    )
    if not table_present and apply:
        raise TranscriptLedgerError(
            f"{TRANSCRIPT_GROUNDING_TABLE} is absent — run ensure_ledger_table first"
        )
    total = inserted = replayed = mismatched = 0
    mismatches: list[tuple[str, str, str]] = []
    # The canonical verdict per key seen this run: the stored row when one
    # exists, else the first occurrence in the file (which is what a real run
    # inserts). Later duplicates compare against it, so a dry run and an
    # applied run report identical numbers.
    seen: dict[tuple[str, str, str], tuple[float, int, str, str, str, str]] = {}

    with conn:
        for verdict in verdicts:
            total += 1
            canonical = seen.get(verdict.key)
            if canonical is None and table_present:
                row = conn.execute(
                    f"""
                    SELECT containment, grounded, field, delivered_at,
                           graded_at, transcript_session_key
                    FROM {TRANSCRIPT_GROUNDING_TABLE}
                    WHERE recall_event_id = ? AND node_id = ? AND method_version = ?
                    """,
                    verdict.key,
                ).fetchone()
                if row is not None:
                    canonical = (
                        float(row[0]),
                        int(row[1]),
                        str(row[2]),
                        str(row[3]),
                        str(row[4]),
                        str(row[5]),
                    )
                    seen[verdict.key] = canonical
            if canonical is not None:
                if verdict.payload() == canonical:
                    replayed += 1
                else:
                    mismatched += 1
                    if len(mismatches) < _MAX_REPORTED_MISMATCHES:
                        mismatches.append(verdict.key)
                continue
            seen[verdict.key] = verdict.payload()
            inserted += 1
            if apply:
                conn.execute(
                    f"""
                    INSERT INTO {TRANSCRIPT_GROUNDING_TABLE} (
                        id, recall_event_id, node_id, containment, grounded,
                        method_version, field, delivered_at, graded_at,
                        transcript_session_key, imported_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_ulid(),
                        verdict.recall_event_id,
                        verdict.node_id,
                        verdict.containment,
                        verdict.grounded,
                        verdict.method_version,
                        verdict.field,
                        verdict.delivered_at,
                        verdict.graded_at,
                        verdict.transcript_session_key,
                        stamp,
                    ),
                )

    return TranscriptLedgerImportStats(
        total=total,
        inserted=inserted,
        replayed=replayed,
        mismatched=mismatched,
        mismatches=tuple(mismatches),
        imported_at=stamp,
        applied=apply,
    )
