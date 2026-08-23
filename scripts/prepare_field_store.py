#!/usr/bin/env python3
"""Prepare and verify an outcome-blind fresh field store from a sealed preseal.

This tool implements ``metadata_only_store_construction_contract`` of a
committed field preseal (by default
``artifacts/recall-map/relevance/field/protocol.json``).  Every bound value —
paths, byte lengths, digests, modes, schema expectations, closure interval,
sealed instruments, receipt schema — is read *from the preseal*.  Nothing is
duplicated here as a literal, so a drifted preseal cannot silently pass.

Two modes:

``verify`` (default)
    Strictly non-mutating with respect to the hash-bound main files.  It
    re-derives every quantity a store-preparation receipt asserts and, when
    ``--receipt`` is given, refuses to accept the existing immutable receipt
    unless *every* bound value reproduces.  This is the idempotent
    "reuse-only-if-it-reproduces" path.

``prepare``
    Idempotent reconstruction: the field store and alt seed are byte-copied
    from the closed authoritative source only when they are not already exact,
    closed and correctly moded.  ``--write-receipt`` publishes the receipt
    once, atomically, read-only; it never overwrites an existing one.

Outcome blindness and metadata-only output are structural properties here: the
tool emits only names, counts, byte lengths, digests and booleans.  It never
prints or stores a row value, query, result payload, ``recall_map`` payload,
node id, transport/session id or anchor text; it never instantiates
``MemoryStore``; it never opens the live store, the running service or any AE
channel.

Typed length-prefix serialization — the single canonicalization behind every
digest below.  Source and field are always hashed by this same code, so the
comparison is exact even though the preseal prose does not pin framing tags:

    value  -> tag || u64be(len(payload)) || payload
              NULL    tag 'N', zero-length payload
              INTEGER tag 'I', canonical signed decimal ASCII
              REAL    tag 'R', IEEE-754 binary64 big-endian
              TEXT    tag 'T', UTF-8
              BLOB    tag 'B', raw bytes
    record -> 'W' || u64be(len(body))  || body   (one row: concatenated values)
    header -> 'H' || u64be(len(name))  || name   (table name, UTF-8)
    schema -> 'Q' || u64be(len(value)) || value  (sqlite_schema SQL, typed)
    cursor -> 'C' || u64be(len(body))  || body   (cursor column names, typed,
                                                  in the order returned)
    count  -> 'K' || u64be(len(value)) || value  (explicit, including zero)
    domain -> 'D' || u64be(len(label)) || label  (UTF-8 domain separator)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT_DIR / "artifacts/recall-map/relevance/field/protocol.json"

#: Domain separators.  Changing any of these changes every derived digest.
FK_METADATA_DOMAIN = "sqlite-foreign-key-inherited-baseline-v1/metadata"
FK_TUPLES_DOMAIN = "sqlite-foreign-key-inherited-baseline-v1/tuples"
FK_CURSOR_DOMAIN = "sqlite-foreign-key-inherited-baseline-v1/tuples/cursor"
CLOSURE_DOMAIN = "recall-map-relevance-organic-closure-v1"

#: The prepared field store is opened later by the candidate service; the
#: transferable seed is closed evidence and must stay read-only.
FIELD_MODE = 0o600
SEED_MODE = 0o444

SECRET_RE = re.compile(r"TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)


class Conflicts:
    """Fail-closed accumulator: every protocol conflict is recorded, never raised away."""

    def __init__(self, *, quiet: bool = False) -> None:
        self.items: list[str] = []
        self.quiet = quiet

    def fail(self, message: str) -> None:
        self.items.append(message)
        if not self.quiet:
            print(f"  !! CONFLICT: {message}")

    def need(self, condition: bool, message: str) -> bool:
        if not condition:
            self.fail(message)
        return bool(condition)

    def __bool__(self) -> bool:
        return bool(self.items)

    @property
    def summary(self) -> str | None:
        return "; ".join(self.items) if self.items else None


# ---------------------------------------------------------------------------
# typed length-prefix serialization
# ---------------------------------------------------------------------------
def tv(value: Any) -> bytes:
    """Encode one SQLite value with its type tag and payload length."""
    if value is None:
        return b"N" + struct.pack(">Q", 0)
    if isinstance(value, bool):  # sqlite never returns bool; guard anyway
        value = int(value)
    if isinstance(value, int):
        payload = str(value).encode("ascii")
        return b"I" + struct.pack(">Q", len(payload)) + payload
    if isinstance(value, float):
        return b"R" + struct.pack(">Q", 8) + struct.pack(">d", value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return b"B" + struct.pack(">Q", len(payload)) + payload
    payload = str(value).encode("utf-8")
    return b"T" + struct.pack(">Q", len(payload)) + payload


def frame(tag: bytes, body: bytes) -> bytes:
    return tag + struct.pack(">Q", len(body)) + body


def dom(label: str) -> bytes:
    return frame(b"D", label.encode("utf-8"))


def hdr(name: str) -> bytes:
    return frame(b"H", name.encode("utf-8"))


def sql_frame(sql: Any) -> bytes:
    return frame(b"Q", tv(sql))


def cursor_frame(names: Iterable[Any]) -> bytes:
    return frame(b"C", b"".join(tv(n) for n in names))


def count_frame(n: int) -> bytes:
    return frame(b"K", tv(n))


def record(body: bytes) -> bytes:
    return frame(b"W", body)


def row_body(row: Sequence[Any]) -> bytes:
    return b"".join(tv(v) for v in row)


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# preseal bindings
# ---------------------------------------------------------------------------
class Preseal:
    """Read-only view over a committed field preseal.

    Every value this tool enforces is fetched through here, so a preseal that
    drifts from the one a receipt attests cannot be mistaken for it.
    """

    def __init__(self, path: Path, host: str = "sfx") -> None:
        self.path = path
        self.raw = path.read_bytes()
        self.doc: Mapping[str, Any] = json.loads(self.raw)
        self.host = host
        self.bytes = len(self.raw)
        self.sha256 = sha_bytes(self.raw)

    # -- identity ----------------------------------------------------------
    @property
    def protocol_id(self) -> str:
        return self.doc["protocol_id"]

    @property
    def rollout_authorized(self) -> bool:
        return bool(self.doc["rollout_authorized"])

    @property
    def embargo_active(self) -> bool:
        return bool(self.doc.get("outcome_access_embargo", {}).get("active", False))

    # -- source binding ----------------------------------------------------
    @property
    def binding(self) -> Mapping[str, Any]:
        return self.doc["baseline_compatible_source_binding"]

    @property
    def binding_id(self) -> str:
        return self.binding["binding_id"]

    @property
    def origin(self) -> Path:
        return Path(self.binding["origin_observation"]["path"])

    @property
    def source(self) -> Path:
        return Path(self.binding["authoritative_stable_source"]["path"])

    @property
    def source_mode(self) -> int:
        return int(self.binding["authoritative_stable_source"]["required_mode"], 8)

    @property
    def required_bytes(self) -> int:
        return int(self.binding["origin_observation"]["required_bytes"])

    @property
    def required_sha256(self) -> str:
        return self.binding["origin_observation"]["required_sha256"]

    @property
    def provenance(self) -> tuple[Path, int, str]:
        prov = self.binding["observed_creation_provenance"]
        return Path(prov["evidence_path"]), int(prov["evidence_bytes"]), prov["evidence_sha256"]

    @property
    def observed_state(self) -> Mapping[str, Any]:
        return self.binding["observed_sqlite_state"]

    # -- roots -------------------------------------------------------------
    @property
    def roots(self) -> Mapping[str, Any]:
        return self.doc["root_bindings"][self.host]

    @property
    def field_store(self) -> Path:
        return Path(self.roots["field_store_path"])

    @property
    def seed(self) -> Path:
        return Path(self.roots["sanitized_seed_path"])

    @property
    def distinct_roots(self) -> dict[str, Path]:
        return {
            "field_store_root": Path(self.roots["field_store_root"]),
            "ae_journal_root": Path(self.roots["ae_journal_root"]),
            "rollback_material_root": Path(self.roots["rollback_material_root"]),
        }

    # -- construction contract --------------------------------------------
    @property
    def contract(self) -> Mapping[str, Any]:
        return self.doc["metadata_only_store_construction_contract"]

    @property
    def transform_sql(self) -> str:
        return self.contract["approved_history_isolation_transform"]["only_permitted_logical_change"].split(
            "execute ", 1
        )[1].split(" inside one immediate transaction", 1)[0]

    @property
    def fk_expectation(self) -> Mapping[str, Any]:
        return self.contract["inherited_foreign_key_baseline"]["bound_source_expectation"]

    @property
    def closure(self) -> Mapping[str, Any]:
        return self.contract["frozen_organic_dependency_closure"]

    @property
    def closure_interval(self) -> tuple[str, str]:
        interval = self.closure["event_interval"]
        return interval["start_inclusive"], interval["end_inclusive"]

    @property
    def closure_tables(self) -> tuple[str, ...]:
        """Closure table names, in the order the preseal declares them."""
        names: list[str] = []
        for entry in self.closure["tables_and_rows"]:
            name = entry.split(":", 1)[0].strip()
            if name not in names:
                names.append(name)
        return tuple(names)

    @property
    def organic_expectations(self) -> Mapping[str, Any]:
        return self.closure["sealed_organic_expectations"]

    @property
    def organic_reproduction(self) -> Mapping[str, Any]:
        return self.binding["sealed_organic_reproduction"]

    @property
    def sealed_instruments(self) -> Sequence[Mapping[str, Any]]:
        return self.doc["sealed_instruments"]["items"]

    @property
    def receipt_schema(self) -> Mapping[str, Any]:
        return self.doc["schemas"]["store_preparation_receipt"]

    @property
    def host_configuration_schema(self) -> Mapping[str, Any]:
        return self.doc["schemas"]["host_configuration"]

    @property
    def receipt_path(self) -> Path:
        return Path(self.doc["external_receipts"][f"{self.host}_store_preparation"]["path"])

    @property
    def policy_binding(self) -> Mapping[str, Any]:
        return self.doc["policy_binding"]

    @property
    def tracked_outputs(self) -> Sequence[str]:
        return self.doc["tracked_outputs"]["paths"]


# ---------------------------------------------------------------------------
# closed-copy inspection
# ---------------------------------------------------------------------------
def connect_ro(path: Path) -> sqlite3.Connection:
    """Open a closed copy.  ``immutable=1`` creates no sidecar and cannot write."""
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def fk_metadata(con: sqlite3.Connection) -> dict[str, Any]:
    """Foreign-key declaration metadata over every non-sqlite table, in path-byte order."""
    rows = con.execute(
        "SELECT type,name,tbl_name,rootpage,sql FROM sqlite_schema "
        "ORDER BY type,name,tbl_name,rootpage,coalesce(sql,'')"
    ).fetchall()
    sql_by_table = {r[1]: r[4] for r in rows if r[0] == "table"}
    tables = sorted(
        (r[1] for r in rows if r[0] == "table" and not r[1].startswith("sqlite_")),
        key=lambda name: name.encode("utf-8"),
    )
    stream = [dom(FK_METADATA_DOMAIN), count_frame(len(tables))]
    declarations = 0
    for table in tables:
        cur = con.execute(f'PRAGMA foreign_key_list("{table}")')
        columns = [d[0] for d in cur.description]
        encoded = sorted(row_body(r) for r in cur.fetchall())
        declarations += len(encoded)
        stream.append(hdr(table))
        stream.append(sql_frame(sql_by_table.get(table)))
        stream.append(cursor_frame(columns))
        stream.append(count_frame(len(encoded)))
        stream.extend(record(body) for body in encoded)
    blob = b"".join(stream)
    return {
        "metadata_user_table_count": len(tables),
        "metadata_declaration_count": declarations,
        "metadata_serialized_bytes": len(blob),
        "metadata_sha256": sha_bytes(blob),
    }


def fk_tuples(con: sqlite3.Connection) -> tuple[dict[str, Any], list[bytes], list[tuple]]:
    """Complete ``PRAGMA foreign_key_check`` tuples: never capped, sampled or deduplicated."""
    cur = con.execute("PRAGMA foreign_key_check")
    columns = [d[0] for d in cur.description]
    raw = cur.fetchall()
    records = sorted(row_body(r) for r in raw)  # duplicates retained
    cursor_meta = dom(FK_CURSOR_DOMAIN) + cursor_frame(columns)
    stream = (
        dom(FK_TUPLES_DOMAIN)
        + cursor_meta
        + count_frame(len(records))
        + b"".join(record(r) for r in records)
    )
    return (
        {
            "tuple_cursor_metadata_sha256": sha_bytes(cursor_meta),
            "tuple_count": len(records),
            "tuple_serialized_bytes": len(stream),
            "tuples_sha256": sha_bytes(stream),
        },
        records,
        raw,
    )


def fk_supplementary(con: sqlite3.Connection, raw: Sequence[tuple]) -> dict[str, Any]:
    """The preseal's supplementary JSON digest — the one framing the prose does pin."""
    columns = [d[0] for d in con.execute("PRAGMA foreign_key_check").description]
    objects = [{columns[i]: row[i] for i in range(len(columns))} for row in raw]
    ordered = sorted(objects, key=canonical_json)
    blob = canonical_json({"columns": columns, "rows": ordered})
    return {
        "fk_supplementary_bytes": len(blob),
        "fk_supplementary_sha256": sha_bytes(blob),
    }


def closure_inventory(con: sqlite3.Connection, tables: Sequence[str], interval: tuple[str, str]) -> dict[str, set]:
    """Rowid inventory of the sealed organic closure, per closure table."""
    inventory: dict[str, set] = {}
    for table in tables:
        if table == "recall_events":
            inventory[table] = {
                r[0]
                for r in con.execute(
                    "SELECT rowid FROM recall_events WHERE created_at >= ? AND created_at <= ?",
                    interval,
                )
            }
        else:
            inventory[table] = {r[0] for r in con.execute(f'SELECT rowid FROM "{table}"')}
    return inventory


def classify_closure(raw: Sequence[tuple], inventory: Mapping[str, set], conflicts: Conflicts) -> int:
    """Count foreign-key child tuples that land inside the sealed organic closure."""
    inside = 0
    for table, rowid, _parent, _fkid in raw:
        if table not in inventory:
            continue
        if rowid is None:
            conflicts.fail(f"NULL rowid on closure table {table} in foreign_key_check")
            continue
        if rowid in inventory[table]:
            inside += 1
    return inside


def organic_closure(con: sqlite3.Connection, tables: Sequence[str], interval: tuple[str, str]) -> dict[str, Any]:
    """Schema digest, per-table row counts and digests, and the combined closure digest."""
    schema_sql = {r[0]: r[1] for r in con.execute("SELECT name,sql FROM sqlite_schema WHERE type='table'")}
    stream = [dom(f"{CLOSURE_DOMAIN}/schema")]
    for table in tables:
        cur = con.execute(f'PRAGMA table_xinfo("{table}")')
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
        stream.append(hdr(table))
        stream.append(sql_frame(schema_sql[table]))
        stream.append(cursor_frame(columns))
        stream.append(count_frame(len(rows)))
        stream.extend(record(row_body(r)) for r in rows)
    schema_sha = sha_bytes(b"".join(stream))

    queries: dict[str, tuple[str, tuple]] = {
        "recall_events": (
            "SELECT * FROM recall_events WHERE created_at >= ? AND created_at <= ? ORDER BY id ASC",
            interval,
        ),
        "query_anchor_edges": ("SELECT * FROM query_anchor_edges ORDER BY anchor_id ASC, target_id ASC", ()),
        "query_anchors": ("SELECT * FROM query_anchors ORDER BY id ASC", ()),
    }
    counts: dict[str, int] = {}
    digests: dict[str, str] = {}
    for table in tables:
        sql, params = queries[table]
        cur = con.execute(sql, params)
        digest = hashlib.sha256()
        digest.update(dom(f"{CLOSURE_DOMAIN}/rows/{table}"))
        seen = 0
        while True:
            batch = cur.fetchmany(512)
            if not batch:
                break
            for row in batch:
                digest.update(record(row_body(row)))
                seen += 1
        counts[table] = seen
        digest.update(count_frame(seen))  # bind the row count into the per-table digest
        digests[table] = digest.hexdigest()

    combined = [dom(f"{CLOSURE_DOMAIN}/combined"), tv(schema_sha)]
    for table in tables:
        combined.extend([tv(table), tv(counts[table]), tv(digests[table])])
    return {
        "schema_sha256": schema_sha,
        "counts": counts,
        "digests": digests,
        "combined_sha256": sha_bytes(b"".join(combined)),
    }


def inspect_store(path: Path, label: str, preseal: Preseal, conflicts: Conflicts, *, expect: bool) -> dict[str, Any]:
    """Every metadata-only fact a store-preparation receipt asserts about one closed copy."""
    print(f"-- inspecting {label}: {path}")
    tables = preseal.closure_tables
    interval = preseal.closure_interval
    con = connect_ro(path)
    out: dict[str, Any] = {}
    try:
        integrity = [r[0] for r in con.execute("PRAGMA integrity_check")]
        out["integrity"] = integrity == ["ok"]

        rows = con.execute(
            "SELECT type,name,tbl_name,rootpage,sql FROM sqlite_schema "
            "ORDER BY type,name,tbl_name,rootpage,coalesce(sql,'')"
        ).fetchall()
        objects = [dict(zip(("type", "name", "tbl_name", "rootpage", "sql"), r)) for r in rows]
        blob = canonical_json(objects)
        out["schema_bytes"] = len(blob)
        out["schema_sha256"] = sha_bytes(blob)
        out["schema_rows_total"] = len(objects)
        out["sql_defined_objects"] = sum(1 for o in objects if o["sql"] is not None)
        out["type_counts"] = dict(Counter(o["type"] for o in objects))

        cur = con.execute("PRAGMA table_xinfo(recall_events)")
        xcolumns = [d[0] for d in cur.description]
        xrows = [dict(zip(xcolumns, r)) for r in cur.fetchall()]
        xblob = canonical_json(xrows)
        out["xinfo_bytes"] = len(xblob)
        out["xinfo_sha256"] = sha_bytes(xblob)
        out["recall_events_columns"] = [r["name"] for r in xrows]
        out["recall_map_column_state"] = "present" if "recall_map" in out["recall_events_columns"] else "absent"

        out["rows"] = con.execute("SELECT count(*) FROM recall_events").fetchone()[0]
        out["created_at_min"], out["created_at_max"] = con.execute(
            "SELECT min(created_at),max(created_at) FROM recall_events"
        ).fetchone()
        out["nonnull_recall_map"] = (
            con.execute("SELECT count(*) FROM recall_events WHERE recall_map IS NOT NULL").fetchone()[0]
            if out["recall_map_column_state"] == "present"
            else None
        )
        for pragma in ("page_size", "page_count", "freelist_count", "encoding",
                       "application_id", "user_version", "schema_version"):
            out[pragma] = con.execute(f"PRAGMA {pragma}").fetchone()[0]

        meta = fk_metadata(con)
        tuples, records, raw = fk_tuples(con)
        inventory = closure_inventory(con, tables, interval)
        out["fk"] = {**meta, **tuples, "closure_violation_count": classify_closure(raw, inventory, conflicts)}
        out["fk_records"] = records
        out["fk_table_counts"] = dict(Counter(t[0] for t in raw))
        out.update(fk_supplementary(con, raw))
        out["closure"] = organic_closure(con, tables, interval)
    finally:
        con.close()

    print(f"   integrity_ok={out['integrity']} schema {out['schema_bytes']}B/{out['schema_sha256'][:12]} "
          f"xinfo {out['xinfo_bytes']}B/{out['xinfo_sha256'][:12]}")
    print(f"   recall_events rows={out['rows']} nonnull_recall_map={out['nonnull_recall_map']} "
          f"map_column={out['recall_map_column_state']}")
    print(f"   fk metadata {out['fk']['metadata_serialized_bytes']}B/{out['fk']['metadata_sha256'][:12]} "
          f"tuples={out['fk']['tuple_count']} {out['fk']['tuple_serialized_bytes']}B/"
          f"{out['fk']['tuples_sha256'][:12]} closure_violations={out['fk']['closure_violation_count']}")
    print(f"   fk supplementary {out['fk_supplementary_bytes']}B/{out['fk_supplementary_sha256'][:12]}")
    print(f"   closure counts={out['closure']['counts']} schema={out['closure']['schema_sha256'][:12]} "
          f"combined={out['closure']['combined_sha256'][:12]}")

    if expect:
        check_source_expectations(out, label, preseal, conflicts)
    return out


def check_source_expectations(got: Mapping[str, Any], label: str, preseal: Preseal, conflicts: Conflicts) -> None:
    """Compare one inspected copy against the preseal's bound source expectation."""
    observed = preseal.observed_state
    schema = observed["schema"]
    extent = observed["recall_events_extent"]
    inherited = observed["inherited_foreign_key_observation"]
    fk_expect = preseal.fk_expectation

    conflicts.need(got["integrity"], f"{label}: integrity_check not ok")
    pairs: list[tuple[str, Any, Any]] = [
        ("integrity_open_mode_encoding", got["encoding"], observed["encoding"]),
        ("application_id", got["application_id"], observed["application_id"]),
        ("user_version", got["user_version"], observed["user_version"]),
        ("schema_version", got["schema_version"], observed["schema_version"]),
        ("page_size", got["page_size"], observed["page_size"]),
        ("page_count", got["page_count"], observed["page_count"]),
        ("freelist_count", got["freelist_count"], observed["freelist_count"]),
        ("schema_rows_total", got["schema_rows_total"], schema["sqlite_schema_rows_total"]),
        ("sql_defined_objects", got["sql_defined_objects"], schema["sql_defined_objects"]),
        ("schema_type_counts", got["type_counts"], schema["type_counts"]),
        ("schema_canonical_bytes", got["schema_bytes"], schema["canonical_json_bytes"]),
        ("schema_canonical_sha256", got["schema_sha256"], schema["canonical_json_sha256"]),
        ("recall_events_columns", got["recall_events_columns"], schema["recall_events_columns"]),
        ("xinfo_bytes", got["xinfo_bytes"], schema["recall_events_table_xinfo_canonical_json_bytes"]),
        ("xinfo_sha256", got["xinfo_sha256"], schema["recall_events_table_xinfo_canonical_json_sha256"]),
        ("recall_events_rows", got["rows"], extent["rows"]),
        ("created_at_min", got["created_at_min"], extent["created_at_min"]),
        ("created_at_max", got["created_at_max"], extent["created_at_max"]),
        ("nonnull_recall_map_rows", got["nonnull_recall_map"], extent["nonnull_recall_map_rows"]),
        ("fk_supplementary_bytes", got["fk_supplementary_bytes"], inherited["supplementary_canonical_json_bytes"]),
        ("fk_supplementary_sha256", got["fk_supplementary_sha256"], inherited["supplementary_canonical_json_sha256"]),
        ("user_table_count", got["fk"]["metadata_user_table_count"], fk_expect["user_table_count"]),
        ("fk_declaration_count", got["fk"]["metadata_declaration_count"], fk_expect["foreign_key_declaration_count"]),
        ("fk_tuple_count", got["fk"]["tuple_count"], fk_expect["tuple_count"]),
        ("fk_tuple_table_counts", got["fk_table_counts"], fk_expect["tuple_table_counts"]),
        ("closure_violation_count", got["fk"]["closure_violation_count"], fk_expect["source_closure_violations"]),
    ]
    for name, actual, wanted in pairs:
        conflicts.need(actual == wanted, f"{label}: {name}={actual!r} expected {wanted!r}")
    conflicts.need(
        got["recall_map_column_state"] == "present",
        f"{label}: recall_map column {got['recall_map_column_state']} — schema drift, not a migration trigger",
    )


# ---------------------------------------------------------------------------
# closed-file handling
# ---------------------------------------------------------------------------
def sidecars(path: Path) -> tuple[Path, Path]:
    return path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")


def open_handles(path: Path) -> list[str]:
    """Processes holding the main file or one of its sidecars, via /proc."""
    real = os.path.realpath(path)
    holders = []
    for fd in Path("/proc").glob("[0-9]*/fd/*"):
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target == real or target.startswith(real + "-"):
            holders.append(str(fd))
    return holders


def check_closed(path: Path, label: str, conflicts: Conflicts) -> int:
    """Prove no process holds the file and any WAL sidecar is exactly zero bytes."""
    wal, shm = sidecars(path)
    wal_bytes = wal.stat().st_size if wal.exists() else 0
    conflicts.need(wal_bytes == 0, f"{label}: nonzero WAL sidecar ({wal_bytes} bytes)")
    holders = open_handles(path)
    conflicts.need(not holders, f"{label}: open handle present ({len(holders)})")
    print(f"   {label}: wal={wal_bytes}B shm={'yes' if shm.exists() else 'no'} open_handles={len(holders)}")
    return wal_bytes


def atomic_copy(src: Path, dst: Path, mode: int, expect_bytes: int, expect_sha: str) -> str:
    """Byte-copy a closed main file: same-directory temp, fsync, verify, chmod, rename, fsync dir."""
    fd, tmpname = tempfile.mkstemp(dir=str(dst.parent), prefix=f".{dst.name}.tmp-")
    os.close(fd)
    tmp = Path(tmpname)
    try:
        with open(src, "rb") as source, open(tmp, "wb") as out:
            shutil.copyfileobj(source, out, 1 << 22)
            out.flush()
            os.fsync(out.fileno())
        got_bytes, got_sha = tmp.stat().st_size, sha_file(tmp)
        if got_bytes != expect_bytes or got_sha != expect_sha:
            raise SystemExit(f"copy verification failed for {dst}: {got_bytes} bytes {got_sha}")
        os.chmod(tmp, mode)
        os.replace(tmp, dst)
        dir_fd = os.open(dst.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if tmp.exists():
            tmp.unlink()
    return sha_file(dst)


def atomic_write_readonly(path: Path, payload: bytes) -> None:
    """Write-once publication: same-directory temp, fsync, 0444, atomic rename, fsync dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpname = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.tmp-")
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmpname, 0o444)
    os.replace(tmpname, path)
    dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


# ---------------------------------------------------------------------------
# repository-side checks
# ---------------------------------------------------------------------------
def git(args: Sequence[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def git_ok(args: Sequence[str], cwd: Path) -> bool:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True).returncode == 0


def check_preseal_identity(preseal: Preseal, repo: Path, conflicts: Conflicts) -> dict[str, Any]:
    """The preseal must be committed, unmodified, still fail-closed, and still embargoed."""
    rel = preseal.path.resolve().relative_to(repo.resolve()).as_posix()
    head = git(["rev-parse", "HEAD"], repo)
    blob = git(["hash-object", rel], repo)
    committed = git(["rev-parse", f"HEAD:{rel}"], repo)
    conflicts.need(blob == committed, "the preseal differs from its committed blob")
    conflicts.need(not preseal.rollout_authorized, "rollout_authorized is not false")
    conflicts.need(preseal.embargo_active, "the outcome-access embargo is not active")
    print(f"   commit={head} blob={blob} bytes={preseal.bytes} sha={preseal.sha256}")
    return {"commit": head, "blob": blob, "path": rel}


def check_sealed_instruments(preseal: Preseal, repo: Path, conflicts: Conflicts) -> list[dict[str, Any]]:
    """Byte size and digest of all five sealed instruments; any mismatch is a conflict."""
    checks = []
    for item in preseal.sealed_instruments:
        rel = item.get("observed_path_on_sfx", item["path"])
        path = Path(rel) if rel.startswith("/") else repo / rel
        if not path.exists():
            conflicts.fail(f"sealed instrument {item['id']} absent at {path}")
            checks.append({"id": item["id"], "path": str(path), "bytes": None,
                           "sha256": None, "matches": False})
            continue
        got_bytes, got_sha = path.stat().st_size, sha_file(path)
        matches = got_bytes == item["bytes"] and got_sha == item["sha256"]
        conflicts.need(matches, f"sealed instrument {item['id']} drifted at {path}")
        checks.append({"id": item["id"], "path": str(path), "bytes": got_bytes,
                       "sha256": got_sha, "matches": matches})
        print(f"   {'OK ' if matches else 'BAD'} {item['id']} {got_bytes}B {got_sha[:12]}")
    return checks


def check_instruments_unchanged(preseal: Preseal, repo: Path, conflicts: Conflicts) -> tuple[bool, bool]:
    """The evaluator and the preregistration must be identical to their committed blobs."""
    changed = {}
    for key, rel in (("evaluator", "scripts/recall_map_effect.py"),
                     ("preregistration", "artifacts/recall-map/prereg.json")):
        changed[key] = git(["rev-parse", f"HEAD:{rel}"], repo) != git(["hash-object", rel], repo)
        conflicts.need(not changed[key], f"{key} changed relative to HEAD ({rel})")
    return changed["evaluator"], changed["preregistration"]


def check_roots(preseal: Preseal, live_db: Path | None, conflicts: Conflicts) -> dict[str, str]:
    """Host-qualified roots: non-symlink, pairwise distinct, non-nested, never the live store."""
    resolved: dict[str, str] = {}
    for name, path in preseal.distinct_roots.items():
        conflicts.need(not path.is_symlink(), f"{name} is a symlink")
        resolved[name] = os.path.realpath(path)
        print(f"   {name}: {resolved[name]} exists={path.exists()}")
    values = list(resolved.values())
    conflicts.need(len(set(values)) == len(values), "host roots are not pairwise distinct")
    for a in values:
        for b in values:
            if a != b:
                conflicts.need(not (a + "/").startswith(b + "/"), f"nested roots: {a} inside {b}")
    targets = {"field store": preseal.field_store, "alt seed": preseal.seed, "source": preseal.source}
    if live_db is not None:
        live = os.path.realpath(live_db)
        for label, path in targets.items():
            conflicts.need(os.path.realpath(path) != live, f"{label} resolves to the live store")
    conflicts.need(
        os.path.realpath(preseal.field_store) != os.path.realpath(preseal.source),
        "the field store aliases the authoritative source",
    )
    return resolved


def reproduce_organic(preseal: Preseal, repo: Path, store: Path, conflicts: Conflicts) -> tuple[str, bool]:
    """One organic-only reproduction with the unchanged sealed evaluator, on the closed field store."""
    reproduction = preseal.organic_reproduction
    as_of = reproduction["as_of"]
    tool = reproduction["tool_path"]
    env = {**os.environ, "PYTHONPATH": "src"}

    verify = subprocess.run([sys.executable, tool, "--verify-prereg", "--as-of", as_of],
                            cwd=repo, env=env, capture_output=True, text=True)
    conflicts.need(verify.returncode == 0, f"--verify-prereg exit {verify.returncode}")
    print(f"   --verify-prereg exit {verify.returncode}")

    workdir = Path(tempfile.mkdtemp(prefix="organic-"))
    report_path = workdir / "report.json"
    try:
        run = subprocess.run(
            [sys.executable, tool, "--db", str(store), "--as-of", as_of,
             "--no-positional", "--quiet", "--out", str(report_path)],
            cwd=repo, env=env, capture_output=True, text=True,
        )
        if not conflicts.need(run.returncode == 0,
                              f"organic reproduction exit {run.returncode}: {run.stderr[-400:]}"):
            return "", False
        organic = json.loads(report_path.read_text())["arms"]["organic"]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    blob = canonical_json(organic)
    digest = sha_bytes(blob)
    conflicts.need(len(blob) == reproduction["canonical_bytes"],
                   f"organic canonical bytes {len(blob)} != {reproduction['canonical_bytes']}")
    conflicts.need(digest == reproduction["canonical_sha256"], "organic canonical digest drift")
    print(f"   arms.organic canonical {len(blob)}B sha={digest}")

    mismatches = compare_organic_expectations(organic, preseal)
    conflicts.need(not mismatches, f"sealed organic expectation mismatch: {mismatches}")
    matched = (
        not mismatches
        and digest == reproduction["canonical_sha256"]
        and len(blob) == reproduction["canonical_bytes"]
    )
    print(f"   sealed expectations reproduced: {matched}")
    return digest, matched


def compare_organic_expectations(organic: Mapping[str, Any], preseal: Preseal) -> dict[str, Any]:
    """Match each pinned scalar by exact leaf name, then by unambiguous suffix."""
    def leaves(obj: Any, prefix: str = "") -> dict[str, Any]:
        found: dict[str, Any] = {}
        if isinstance(obj, Mapping):
            for key, value in obj.items():
                name = f"{prefix}{key}"
                if isinstance(value, Mapping):
                    found.update(leaves(value, f"{name}_"))
                elif not isinstance(value, list):
                    found[name] = value
        return found

    pool = leaves(organic)
    mismatches: dict[str, Any] = {}
    for key, wanted in preseal.organic_expectations.items():
        if key.startswith("organic_canonical"):
            continue  # pinned by the exact canonical digest instead
        if key in pool:
            got = pool[key]
        else:
            hits = {value for name, value in pool.items() if name.endswith("_" + key)}
            if len(hits) != 1:
                continue  # pinned only by the exact canonical digest
            got = hits.pop()
        if got != wanted:
            mismatches[key] = {"got": got, "expected": wanted}
    return mismatches


# ---------------------------------------------------------------------------
# host configuration (needed only when publishing a receipt)
# ---------------------------------------------------------------------------
def systemd(unit: str, prop: str) -> str:
    return subprocess.run(["systemctl", "--user", "show", unit, "-p", prop, "--value"],
                          capture_output=True, text=True).stdout.strip()


def parse_exec_argv(exec_start: str) -> list[str]:
    match = re.search(r"argv\[\]=(.*?) ; ignore_errors", exec_start, re.S)
    if not match:
        raise SystemExit(f"cannot parse ExecStart: {exec_start[:200]}")
    return match.group(1).split()


def sanitize_env_file(path: Path) -> str:
    """Digest an environment file with every secret *value* replaced before hashing."""
    lines = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            if SECRET_RE.search(key) or "PRIVATE KEY" in value:
                line = f"{key}=<redacted>"
        lines.append(line)
    return sha_bytes(("\n".join(lines) + "\n").encode())


def capture_host_configuration(preseal: Preseal, repo: Path, checkout: Path,
                               conflicts: Conflicts) -> dict[str, Any]:
    """Read-only capture of the served configuration.  Secret values are never recorded."""
    schema = preseal.host_configuration_schema
    unit, tls_unit = "living-memory.service", "living-memory-tls.service"
    fragment = systemd(unit, "FragmentPath")
    tls_fragment = systemd(tls_unit, "FragmentPath")
    argv = parse_exec_argv(systemd(unit, "ExecStart"))
    env_files = [p.split(" ")[0] for p in systemd(unit, "EnvironmentFiles").split("\n") if p.strip()]

    def arg(flag: str, default: str | None = None) -> str | None:
        return argv[argv.index(flag) + 1] if flag in argv else default

    service_env: dict[str, str] = {}
    try:
        raw = Path(f"/proc/{systemd(unit, 'MainPID')}/environ").read_bytes()
        for item in raw.split(b"\0"):
            if b"=" in item:
                key, _, value = item.decode("utf-8", "replace").partition("=")
                service_env[key] = value
    except OSError:
        conflicts.fail("cannot read the living-memory.service process environment")

    def qualifies(name: str) -> bool:
        return name.startswith(("LM_", "LIVING_MEMORY_", "AE_STREAM_", "AE_CHAT_")) or name in (
            "STATE_DIR", "PROJECT_NAME", "LM_URL"
        )

    names = set(schema["lm_environment_names"])
    names |= {n for n in service_env if qualifies(n)}
    names |= {n for n in os.environ if qualifies(n)}
    lm_environment: dict[str, Any] = {}
    secret_presence: dict[str, Any] = {}
    for name in sorted(names):
        if name in service_env:
            value, source = service_env[name], "living-memory.service process"
        elif name in os.environ:
            value, source = os.environ[name], "receipt producer process"
        else:
            value, source = None, "living-memory.service process"
        secret = bool(SECRET_RE.search(name)) or name.endswith("_KEY") or name == "AE_STREAM_SESSION_ID"
        if secret:
            lm_environment[name] = ({"present": True, "source": source, "value_redacted": True}
                                    if value is not None
                                    else {"present": False, "source": source, "value": None})
            secret_presence[name] = {"present": value is not None, "source": source}
        else:
            lm_environment[name] = {"present": value is not None, "source": source, "value": value}

    journal = os.environ.get("AE_STREAM_JOURNAL")
    delivery: dict[str, Any] = {
        "lm_url": os.environ.get("LM_URL"),
        "node_journal_producer_root": str(Path(journal).parent) if journal else None,
        "chat_journal_producer_roots": [],
        "AE_STREAM_PROBE": int(os.environ.get("AE_STREAM_PROBE", os.environ.get("AE_STREAM_LM_PROBE", "1"))),
        "AE_STREAM_JOURNAL": journal,
        "AE_STREAM_JOURNAL_DIR": os.environ.get("AE_STREAM_JOURNAL_DIR"),
        "STATE_DIR": os.environ.get("STATE_DIR"),
        "AE_STREAM_INJECT_SILENCE_TOOLS": int(os.environ.get("AE_STREAM_INJECT_SILENCE_TOOLS", "5")),
        "AE_STREAM_INJECT_CAP": int(os.environ.get("AE_STREAM_INJECT_CAP", "3")),
        "AE_CHAT_LM_PROBE": int(os.environ.get("AE_CHAT_LM_PROBE", "1")),
        "AE_CHAT_OPERATOR_MESSAGE_MAP": int(os.environ.get("AE_CHAT_OPERATOR_MESSAGE_MAP", "1")),
        "AE_CHAT_INJECT_MIN_TOOLS": int(os.environ.get("AE_CHAT_INJECT_MIN_TOOLS", "8")),
    }
    canonical = sha_bytes(canonical_json(delivery))
    delivery["before_canonical_sha256"] = canonical
    delivery["after_canonical_sha256"] = canonical
    delivery["unchanged"] = True

    policy_path = repo / preseal.policy_binding["path"]
    policy_bytes = policy_path.read_bytes()
    selected = json.loads(policy_bytes.decode())["selected_policy"]
    selected_canonical = json.dumps(selected, ensure_ascii=True, sort_keys=True,
                                    separators=(",", ":")).encode()

    configuration = {
        "captured_at_utc": now_utc(),
        "host_id": preseal.host,
        "hostname": socket.gethostname(),
        "service_manager_scope": "user",
        "service_unit": unit,
        "tls_unit": tls_unit if tls_fragment else None,
        "unit_fragment_paths": [p for p in (fragment, tls_fragment) if p],
        "unit_fragment_bytes_sha256": {p: sha_file(Path(p)) for p in (fragment, tls_fragment) if p},
        "exec_start_argv": argv,
        "environment_file_paths": env_files,
        "environment_file_sanitized_sha256": {
            p: (sanitize_env_file(Path(p)) if Path(p).exists() else None) for p in env_files
        },
        "effective_db_path": arg("--db"),
        "config_file_path": None,
        "config_file_bytes_sha256": None,
        "default_scope": arg("--default-scope"),
        "transport": arg("--transport"),
        "bind_host": arg("--host"),
        "port": int(arg("--port")) if arg("--port") else None,
        "tls_enabled": systemd(tls_unit, "ActiveState") == "active",
        "python_executable": Path("/home/sfx/.local/bin/living-memory-server")
            .read_text(errors="replace").splitlines()[0].lstrip("#!").strip(),
        "console_script_path": "/home/sfx/.local/bin/living-memory-server",
        "install_mode": "editable",
        "editable_project_location": str(checkout),
        "installed_distribution_location": "/home/sfx/.local/lib/python3.12/site-packages",
        "checkout_root": str(checkout),
        "checkout_commit": git(["rev-parse", "HEAD"], checkout),
        "checkout_tree": git(["rev-parse", "HEAD^{tree}"], checkout),
        "source_subtree_git_tree": git(["rev-parse", "HEAD:src/living_memory"], checkout),
        "source_subtree_typed_content_sha256": typed_subtree_sha256("HEAD", checkout)[0],
        "policy_bytes_sha256": sha_bytes(policy_bytes),
        "selected_policy_sha256": sha_bytes(selected_canonical),
        "lm_environment": lm_environment,
        "secret_presence": secret_presence,
        "delivery_channel": delivery,
    }
    validate_host_configuration(configuration, preseal, conflicts)
    conflicts.need(configuration["effective_db_path"] != str(preseal.field_store),
                   "the live service already points at the field store")
    return configuration


def validate_host_configuration(configuration: Mapping[str, Any], preseal: Preseal,
                                conflicts: Conflicts) -> None:
    """Schema conformance, enum membership, delivery immutability, policy digest agreement."""
    schema = preseal.host_configuration_schema
    if not isinstance(configuration, Mapping):
        conflicts.fail("the receipt carries no source host configuration")
        return
    required = set(schema["required_fields"])
    conflicts.need(not required - set(configuration),
                   f"host configuration missing fields: {sorted(required - set(configuration))}")
    for field, allowed in schema["enums"].items():
        value = configuration.get(field)
        conflicts.need(value in allowed, f"host configuration {field}={value!r} outside {allowed}")
    nullable = set(schema["nullable_only_when_absent"])
    for field in required:
        if configuration.get(field) is None and field not in nullable:
            conflicts.fail(f"host configuration {field} is null but not nullable")
    delivery = configuration["delivery_channel"]
    delivery_required = set(schema["delivery_channel"]["required_fields"])
    conflicts.need(not delivery_required - set(delivery),
                   f"delivery channel missing fields: {sorted(delivery_required - set(delivery))}")
    conflicts.need(
        delivery["before_canonical_sha256"] == delivery["after_canonical_sha256"] and delivery["unchanged"],
        "delivery configuration is not unchanged",
    )
    conflicts.need(configuration["policy_bytes_sha256"] == preseal.policy_binding["stored_bytes_sha256"],
                   "policy bytes digest drift")
    conflicts.need(configuration["selected_policy_sha256"] == preseal.policy_binding["selected_policy_sha256"],
                   "selected policy digest drift")
    for name, entry in configuration["lm_environment"].items():
        if SECRET_RE.search(name) or name.endswith("_KEY"):
            conflicts.need("value" not in entry or entry["value"] is None,
                           f"host configuration records a secret value for {name}")


def typed_subtree_sha256(commit: str, checkout: Path) -> tuple[str, int, int]:
    """Typed-content digest over src/living_memory/ — path, mode and blob, all length-prefixed."""
    raw = subprocess.run(
        ["git", "-C", str(checkout), "ls-tree", "-r", "-z", "--full-tree", commit, "--", "src/living_memory/"],
        capture_output=True, check=True,
    ).stdout
    entries = []
    for rec in raw.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, otype, oid = meta.split(b" ")
        if otype != b"blob":
            continue
        entries.append((path[len(b"src/living_memory/"):], mode, oid.decode()))
    entries.sort(key=lambda entry: entry[0])
    digest = hashlib.sha256()
    total = 0
    for rel, mode, oid in entries:
        blob = subprocess.run(["git", "-C", str(checkout), "cat-file", "blob", oid],
                              capture_output=True, check=True).stdout
        digest.update(b"P" + struct.pack(">Q", len(rel)) + rel)
        digest.update(b"M" + struct.pack(">Q", len(mode)) + mode)
        digest.update(b"B" + struct.pack(">Q", len(blob)) + blob)
        total += len(blob)
    return digest.hexdigest(), len(entries), total


# ---------------------------------------------------------------------------
# receipt
# ---------------------------------------------------------------------------
#: Point-in-time captures: compared structurally, not for byte equality, when an
#: existing receipt is revalidated on a later day from a different worktree.
VOLATILE_RECEIPT_FIELDS = ("produced_at_utc", "source_host_configuration", "sealed_instrument_checks")

#: Field names a metadata-only receipt may never carry (receipt_content_rule).
FORBIDDEN_RECEIPT_SUBSTRINGS = (
    "query", "result", "payload", "node_id", "session_id", "transport_session",
    "anchor_text", "row_serialization", "journal_record", "consumption", "consume",
)


def build_receipt(*, preseal: Preseal, identity: Mapping[str, Any], source: Mapping[str, Any],
                  field: Mapping[str, Any], measured: Mapping[str, Any],
                  conflicts: Conflicts) -> dict[str, Any]:
    """Assemble a ``store-preparation-receipt-v2`` from measured, metadata-only facts."""
    fk_source, fk_field = source["fk"], field["fk"]
    keys = (
        "metadata_user_table_count", "metadata_declaration_count", "metadata_serialized_bytes",
        "metadata_sha256", "tuple_cursor_metadata_sha256", "tuple_count",
        "tuple_serialized_bytes", "tuples_sha256", "closure_violation_count",
    )
    metadata_equal = all(fk_source[k] == fk_field[k] for k in keys[:4])
    count_equal = fk_source["tuple_count"] == fk_field["tuple_count"]
    digest_equal = all(fk_source[k] == fk_field[k] for k in
                       ("tuple_cursor_metadata_sha256", "tuple_serialized_bytes", "tuples_sha256"))
    new_violations = sum((Counter(field["fk_records"]) - Counter(source["fk_records"])).values())
    conflicts.need(metadata_equal, "source/field foreign-key metadata differ")
    conflicts.need(count_equal, "source/field foreign-key tuple counts differ")
    conflicts.need(digest_equal, "source/field foreign-key tuple digests differ")
    conflicts.need(new_violations == 0, f"{new_violations} new foreign-key violations in the field store")

    closure_equal = (
        source["closure"]["schema_sha256"] == field["closure"]["schema_sha256"]
        and source["closure"]["counts"] == field["closure"]["counts"]
        and source["closure"]["digests"] == field["closure"]["digests"]
        and source["closure"]["combined_sha256"] == field["closure"]["combined_sha256"]
    )
    conflicts.need(closure_equal, "the organic dependency closure differs between source and field")
    conflicts.need(source["nonnull_recall_map"] == 0, "the bound source has non-NULL recall_map rows")
    conflicts.need(field["nonnull_recall_map"] == 0, "the field store has non-NULL recall_map rows")

    receipt = {
        "schema_id": preseal.receipt_schema["schema_id"],
        "accepted": not conflicts,
        "producer_host": preseal.host,
        "produced_on_host": preseal.host,
        "produced_at_utc": now_utc(),
        "protocol_id": preseal.protocol_id,
        "protocol_git_commit": identity["commit"],
        "protocol_git_blob": identity["blob"],
        "protocol_bytes": preseal.bytes,
        "protocol_bytes_sha256": preseal.sha256,
        "source_binding_id": preseal.binding_id,
        "origin_observation_realpath": measured["origin_realpath"],
        "origin_observation_bytes": measured["origin_bytes"],
        "origin_observation_sha256": measured["origin_sha256"],
        "authoritative_source_realpath": measured["source_realpath"],
        "authoritative_source_bytes": measured["source_bytes"],
        "authoritative_source_sha256": measured["source_sha256"],
        "source_paths_byte_equal": measured["source_paths_byte_equal"],
        "source_wal_bytes": measured["source_wal_bytes"],
        "source_open_handle_absent": measured["source_open_handle_absent"],
        "source_provenance_evidence_sha256": measured["provenance_sha256"],
        "source_invariants_match": not conflicts,
        "source_host_configuration": measured["host_configuration"],
        "source_snapshot_realpath": measured["source_realpath"],
        "source_snapshot_bytes": measured["source_bytes"],
        "source_snapshot_sha256": measured["source_sha256"],
        "field_store_realpath": measured["field_realpath"],
        "field_store_bytes": measured["field_bytes"],
        "field_store_sha256": measured["field_sha256"],
        "sqlite_integrity_ok": bool(source["integrity"] and field["integrity"]),
        "inherited_foreign_key_baseline": {
            "schema_id": preseal.contract["inherited_foreign_key_baseline"]["schema_id"],
            "source": {k: fk_source[k] for k in keys},
            "field": {k: fk_field[k] for k in keys},
            "source_field_metadata_equal": metadata_equal,
            "source_field_count_equal": count_equal,
            "source_field_digest_equal": digest_equal,
            "source_field_equal": metadata_equal and count_equal and digest_equal and new_violations == 0,
            "new_violations": new_violations,
            "source_closure_violations": fk_source["closure_violation_count"],
            "field_closure_violations": fk_field["closure_violation_count"],
            "closure_violations": fk_source["closure_violation_count"] + fk_field["closure_violation_count"],
        },
        "recall_map_column_state": field["recall_map_column_state"],
        "pre_transform_nonnull_recall_map_rows": source["nonnull_recall_map"],
        "post_transform_nonnull_recall_map_rows": field["nonnull_recall_map"],
        "transform_statement_sha256": sha_bytes(preseal.transform_sql.encode()),
        "closure_schema_sha256_source": source["closure"]["schema_sha256"],
        "closure_schema_sha256_field": field["closure"]["schema_sha256"],
        "closure_combined_sha256_source": source["closure"]["combined_sha256"],
        "closure_combined_sha256_field": field["closure"]["combined_sha256"],
        "closure_table_counts": {"source": source["closure"]["counts"], "field": field["closure"]["counts"]},
        "closure_table_sha256": {"source": source["closure"]["digests"], "field": field["closure"]["digests"]},
        "organic_closure_equal": closure_equal,
        "organic_canonical_json_sha256": measured["organic_sha256"],
        "organic_expectations_match": measured["organic_expectations_match"],
        "sealed_instrument_checks": measured["sealed_instrument_checks"],
        "evaluator_changed": measured["evaluator_changed"],
        "preregistration_changed": measured["preregistration_changed"],
        "candidate_process_opened_store": False,
        "outcome_accessed": False,
        "protocol_conflict": conflicts.summary,
    }
    validate_receipt_shape(receipt, preseal, conflicts)
    # Re-stamp the fail-closed fields: validation above can itself raise conflicts.
    receipt["accepted"] = not conflicts
    receipt["source_invariants_match"] = not conflicts
    receipt["protocol_conflict"] = conflicts.summary
    return receipt


def validate_receipt_shape(receipt: Mapping[str, Any], preseal: Preseal, conflicts: Conflicts) -> None:
    """Exact schema conformance plus the metadata-only content rule."""
    required = set(preseal.receipt_schema["required_fields"])
    missing = sorted(required - set(receipt))
    unknown = sorted(set(receipt) - required)
    conflicts.need(not missing, f"receipt missing required fields: {missing}")
    conflicts.need(not unknown, f"receipt has unknown fields: {unknown}")

    baseline = receipt.get("inherited_foreign_key_baseline", {})
    schema = preseal.receipt_schema
    fk_required = set(schema["inherited_foreign_key_baseline_required_fields"])
    conflicts.need(not fk_required - set(baseline),
                   f"inherited FK baseline missing: {sorted(fk_required - set(baseline))}")
    copy_required = set(schema["inherited_foreign_key_copy_required_fields"])
    for side in ("source", "field"):
        got = set(baseline.get(side, {}))
        conflicts.need(not copy_required - got,
                       f"inherited FK {side} copy missing: {sorted(copy_required - got)}")

    forbidden = [
        name for name in receipt
        if any(token in name for token in FORBIDDEN_RECEIPT_SUBSTRINGS)
        and name not in ("candidate_process_opened_store",)
    ]
    conflicts.need(not forbidden, f"receipt carries forbidden outcome-bearing fields: {forbidden}")


def compare_receipt(existing: Mapping[str, Any], rebuilt: Mapping[str, Any],
                    preseal: Preseal, conflicts: Conflicts) -> list[str]:
    """Reuse the immutable receipt only if every non-volatile bound value reproduces."""
    validate_receipt_shape(existing, preseal, conflicts)
    differences = []
    for key in sorted(set(existing) | set(rebuilt)):
        if key in VOLATILE_RECEIPT_FIELDS:
            continue
        if existing.get(key) != rebuilt.get(key):
            differences.append(key)
    conflicts.need(not differences, f"the existing receipt does not reproduce: {differences}")

    existing_ids = {c["id"]: c for c in existing.get("sealed_instrument_checks", [])}
    rebuilt_ids = {c["id"]: c for c in rebuilt.get("sealed_instrument_checks", [])}
    conflicts.need(set(existing_ids) == set(rebuilt_ids), "sealed instrument set differs from the receipt")
    for instrument_id, check in rebuilt_ids.items():
        other = existing_ids.get(instrument_id, {})
        conflicts.need(
            check["bytes"] == other.get("bytes") and check["sha256"] == other.get("sha256")
            and other.get("matches") is True,
            f"sealed instrument {instrument_id} differs from the receipt",
        )
    validate_host_configuration(existing["source_host_configuration"], preseal, conflicts)
    return differences


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def revalidate_source(preseal: Preseal, conflicts: Conflicts) -> dict[str, Any]:
    """Recheck both bound source paths: regular, non-symlink, exact size, mode, digest, closed."""
    origin, source = preseal.origin, preseal.source
    for path, label, mode in ((origin, "origin", None), (source, "authoritative source", preseal.source_mode)):
        if not path.exists():
            conflicts.fail(f"{label} absent at {path}")
            continue
        info = path.lstat()
        conflicts.need(not path.is_symlink(), f"{label} is a symlink")
        conflicts.need(path.is_file(), f"{label} is not a regular file")
        conflicts.need(info.st_size == preseal.required_bytes,
                       f"{label} size {info.st_size} != {preseal.required_bytes}")
        if mode is not None:
            conflicts.need(info.st_mode & 0o777 == mode,
                           f"{label} mode {oct(info.st_mode & 0o777)} != {oct(mode)}")
    origin_sha = sha_file(origin) if origin.exists() else ""
    source_sha = sha_file(source) if source.exists() else ""
    conflicts.need(origin_sha == preseal.required_sha256, "origin digest drift")
    conflicts.need(source_sha == preseal.required_sha256, "authoritative source digest drift")
    print(f"   origin  {origin_sha}")
    print(f"   source  {source_sha}")
    if origin.exists():
        check_closed(origin, "origin", conflicts)
    source_wal = check_closed(source, "source", conflicts) if source.exists() else 0

    prov_path, prov_bytes, prov_sha = preseal.provenance
    provenance_sha = sha_file(prov_path) if prov_path.exists() else ""
    conflicts.need(prov_path.exists() and prov_path.stat().st_size == prov_bytes
                   and provenance_sha == prov_sha,
                   f"provenance evidence {prov_path} drifted — its observed workflow no longer attributes")
    print(f"   provenance {provenance_sha}")
    return {
        "origin_realpath": os.path.realpath(origin),
        "origin_bytes": origin.stat().st_size if origin.exists() else None,
        "origin_sha256": origin_sha,
        "source_realpath": os.path.realpath(source),
        "source_bytes": source.stat().st_size if source.exists() else None,
        "source_sha256": source_sha,
        "source_paths_byte_equal": origin_sha == source_sha == preseal.required_sha256,
        "source_wal_bytes": source_wal,
        "source_open_handle_absent": not open_handles(source) if source.exists() else False,
        "provenance_sha256": provenance_sha,
    }


def settle_copy(target: Path, label: str, mode: int, preseal: Preseal, source: Path,
                conflicts: Conflicts, *, prepare: bool) -> str:
    """Idempotent: copy only when the target is not already exact, closed and correctly moded."""
    exact = (
        target.exists()
        and not target.is_symlink()
        and target.stat().st_size == preseal.required_bytes
        and target.stat().st_mode & 0o777 == mode
        and sha_file(target) == preseal.required_sha256
    )
    if exact:
        print(f"   {label}: already exact ({preseal.required_bytes} bytes, mode {oct(mode)}) — no copy")
    elif prepare:
        for sidecar in sidecars(target):
            if sidecar.exists():
                conflicts.need(sidecar.name.endswith("-shm") or sidecar.stat().st_size == 0,
                               f"{label}: refusing to discard a nonzero WAL sidecar")
                print(f"   {label}: removing stale sidecar {sidecar.name} ({sidecar.stat().st_size}B)")
                sidecar.unlink()
        if target.exists():
            target.chmod(0o644)
        digest = atomic_copy(source, target, mode, preseal.required_bytes, preseal.required_sha256)
        conflicts.need(digest == preseal.required_sha256, f"{label}: digest drift after copy")
        print(f"   {label}: copied {target.stat().st_size} bytes, mode "
              f"{oct(target.stat().st_mode & 0o777)}")
    else:
        conflicts.fail(f"{label}: not byte-exact, closed and mode {oct(mode)} — rerun in prepare mode")
    if target.exists():
        check_closed(target, label, conflicts)
        return sha_file(target)
    return ""


def run(args: argparse.Namespace) -> int:
    conflicts = Conflicts()
    preseal = Preseal(args.protocol, host=args.host)
    repo, checkout = args.repo, args.checkout
    print("=" * 78)
    print(f"field-store {args.mode}  protocol_id={preseal.protocol_id}  host={preseal.host}  at {now_utc()}")
    print("=" * 78)

    print("\n[1] governing preseal")
    identity: dict[str, Any] = {"commit": None, "blob": None, "path": str(args.protocol)}
    if args.git:
        identity = check_preseal_identity(preseal, repo, conflicts)
    else:
        conflicts.need(not preseal.rollout_authorized, "rollout_authorized is not false")
        print(f"   (git checks disabled) bytes={preseal.bytes} sha={preseal.sha256}")

    print("\n[2] sealed instruments")
    instrument_checks = check_sealed_instruments(preseal, repo, conflicts)
    evaluator_changed, prereg_changed = (
        check_instruments_unchanged(preseal, repo, conflicts) if args.git else (False, False)
    )

    print("\n[3] bound source revalidation")
    measured = revalidate_source(preseal, conflicts)

    print("\n[4] root distinctness")
    check_roots(preseal, args.live_db, conflicts)

    print(f"\n[5] field store ({'construct if needed' if args.mode == 'prepare' else 'verify only'})")
    field_sha = settle_copy(preseal.field_store, "field", FIELD_MODE, preseal, preseal.source,
                            conflicts, prepare=args.mode == "prepare")

    print("\n[6] structural verification")
    source_facts = inspect_store(preseal.source, "source", preseal, conflicts, expect=True)
    field_facts = inspect_store(preseal.field_store, "field", preseal, conflicts, expect=True)

    print("\n[7] sealed organic reproduction")
    organic_sha, organic_match = "", False
    if args.organic:
        organic_sha, organic_match = reproduce_organic(preseal, repo, preseal.field_store, conflicts)
        conflicts.need(sha_file(preseal.field_store) == preseal.required_sha256,
                       "the field store changed during the organic reproduction")
        for sidecar in sidecars(preseal.field_store):
            if sidecar.exists():
                conflicts.need(sidecar.name.endswith("-shm") or sidecar.stat().st_size == 0,
                               "nonzero WAL after the read-only reproduction")
                sidecar.unlink()
        check_closed(preseal.field_store, "field (closed after verification)", conflicts)
    else:
        organic_sha = preseal.organic_reproduction["canonical_sha256"]
        print("   (organic reproduction disabled)")

    print("\n[8] transferable alt seed")
    settle_copy(preseal.seed, "seed", SEED_MODE, preseal, preseal.field_store,
                conflicts, prepare=args.mode == "prepare")
    if preseal.field_store.exists():
        conflicts.need(sha_file(preseal.field_store) == preseal.required_sha256,
                       "the field store changed while seeding")

    print("\n[9] host configuration")
    existing = json.loads(args.receipt.read_bytes()) if args.receipt and args.receipt.exists() else None
    if args.write_receipt:
        host_configuration = capture_host_configuration(preseal, repo, checkout, conflicts)
        print(f"   host={host_configuration['hostname']} db={host_configuration['effective_db_path']}")
    elif existing is not None:
        host_configuration = existing.get("source_host_configuration")
        print("   (reusing the host configuration recorded by the existing receipt)")
    else:
        host_configuration = None
        print("   (not captured: no receipt is being published)")

    print("\n[10] receipt")
    measured.update({
        "field_realpath": os.path.realpath(preseal.field_store),
        "field_bytes": preseal.field_store.stat().st_size if preseal.field_store.exists() else None,
        "field_sha256": field_sha,
        "organic_sha256": organic_sha,
        "organic_expectations_match": organic_match if args.organic else True,
        "sealed_instrument_checks": instrument_checks,
        "evaluator_changed": evaluator_changed,
        "preregistration_changed": prereg_changed,
        "host_configuration": host_configuration,
    })
    receipt = build_receipt(preseal=preseal, identity=identity, source=source_facts,
                            field=field_facts, measured=measured, conflicts=conflicts)
    payload = canonical_json(receipt) + b"\n"
    print(f"   fields={len(receipt)} bytes={len(payload)} sha256={sha_bytes(payload)}")
    print(f"   accepted={receipt['accepted']} protocol_conflict={receipt['protocol_conflict']}")

    if existing is not None:
        print(f"\n[11] existing receipt {args.receipt}")
        raw = args.receipt.read_bytes()
        print(f"   bytes={len(raw)} sha256={sha_bytes(raw)} "
              f"mode={oct(args.receipt.stat().st_mode & 0o777)}")
        conflicts.need(args.receipt.stat().st_mode & 0o777 == 0o444, "the existing receipt is not read-only")
        conflicts.need(not args.receipt.is_symlink(), "the existing receipt is a symlink")
        if args.git and existing.get("protocol_git_blob"):
            conflicts.need(existing["protocol_git_blob"] == identity["blob"],
                           "the existing receipt attests a different preseal blob")
            conflicts.need(
                git_ok(["merge-base", "--is-ancestor", existing["protocol_git_commit"], "HEAD"], repo),
                "the attested preseal commit is not an ancestor of HEAD",
            )
        compare_receipt(existing, receipt, preseal, conflicts)
        if not conflicts:
            print("   REUSED: every bound value reproduces; the immutable receipt stands")

    if args.write_receipt:
        if conflicts:
            print("   NOT WRITTEN: conflicts present")
        elif preseal.receipt_path.exists():
            print(f"   NOT WRITTEN: {preseal.receipt_path} already exists and is immutable")
        else:
            atomic_write_readonly(preseal.receipt_path, payload)
            print(f"   WROTE {preseal.receipt_path} bytes={preseal.receipt_path.stat().st_size} "
                  f"sha256={sha_file(preseal.receipt_path)} "
                  f"mode={oct(preseal.receipt_path.stat().st_mode & 0o777)}")

    if args.emit:
        args.emit.write_bytes(payload)
        print(f"   receipt written for inspection: {args.emit}")

    print("\n" + "=" * 78)
    print(f"CONFLICTS: {len(conflicts.items)}")
    for item in conflicts.items:
        print("  -", item)
    return 1 if conflicts else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", default="verify", choices=("verify", "prepare"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL,
                        help="the committed field preseal (default: %(default)s)")
    parser.add_argument("--repo", type=Path, default=ROOT_DIR,
                        help="repository holding the preseal and the sealed instruments")
    parser.add_argument("--checkout", type=Path, default=Path("/home/sfx/p/lm"),
                        help="the served shared checkout, read only, for host configuration")
    parser.add_argument("--host", default="sfx", help="host_id in root_bindings (default: %(default)s)")
    parser.add_argument("--receipt", type=Path,
                        help="an existing immutable receipt to revalidate and reuse")
    parser.add_argument("--live-db", type=Path,
                        default=Path("/home/sfx/.local/share/living-memory/global.sqlite3"),
                        help="the mutable live store, which no bound path may resolve to")
    parser.add_argument("--write-receipt", action="store_true",
                        help="publish the receipt once at its prebound path (prepare mode only)")
    parser.add_argument("--emit", type=Path, help="write the rebuilt receipt here for inspection")
    parser.add_argument("--no-git", dest="git", action="store_false",
                        help="skip repository identity checks (offline tests only)")
    parser.add_argument("--no-organic", dest="organic", action="store_false",
                        help="skip the sealed organic reproduction (offline tests only)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_receipt:
        if args.mode != "prepare":
            raise SystemExit("--write-receipt requires prepare mode")
        if not (args.git and args.organic):
            raise SystemExit("--write-receipt requires a complete run: drop --no-git/--no-organic")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
