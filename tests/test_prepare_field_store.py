"""Tests for scripts/prepare_field_store.py — outcome-blind field-store preparation.

The real store is a half-gigabyte host artifact, so these tests build a tiny
synthetic namespace instead: a miniature SQLite store carrying the structures
the contract reasons about (a ``recall_map`` column, an inherited foreign-key
violation, a dated closure window) plus a preseal derived from the *shipped*
protocol, so receipt conformance is checked against the real schema rather than
a convenient copy.

The preseal expectations the tests pin — canonical schema JSON, ``table_xinfo``
JSON and the supplementary foreign-key JSON — are recomputed here from the
framings the protocol prose fixes, independently of the module under test.  The
implementation-scoped digests (typed foreign-key and closure streams) are never
pinned to a literal; what is asserted about them is the property the contract
actually needs — that source and field agree, and that any drift breaks them.

The sealed instrument set is synthesized alongside the store, for the reason
given on ``Namespace.write_instruments``: two of the shipped instruments live in
another repository on the host, and a unit run cannot be bound to a working tree
it does not own.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPT = ROOT_DIR / "scripts" / "prepare_field_store.py"
REAL_PROTOCOL = ROOT_DIR / "artifacts/recall-map/relevance/field/protocol.json"

CLOSURE_START = "2026-08-01T00:00:00Z"
CLOSURE_END = "2026-08-19T11:00:00Z"


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_field_store", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pfs = _load_module()


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def build_store(path: Path, *, nonnull_map: int = 0, extra_orphan: bool = False) -> None:
    """A miniature store: dated recall events, anchors, and one inherited FK violation."""
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE query_anchors (id INTEGER PRIMARY KEY, label TEXT NOT NULL);
        CREATE TABLE query_anchor_edges (
            anchor_id INTEGER NOT NULL REFERENCES query_anchors(id),
            target_id INTEGER NOT NULL,
            weight REAL NOT NULL
        );
        CREATE TABLE recall_events (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            anchor_id INTEGER REFERENCES query_anchors(id),
            gated INTEGER NOT NULL DEFAULT 0,
            recall_map TEXT
        );
        CREATE INDEX recall_events_created_at ON recall_events(created_at);
        """
    )
    con.executemany("INSERT INTO query_anchors VALUES (?,?)", [(i, f"anchor-{i}") for i in range(1, 5)])
    con.executemany(
        "INSERT INTO query_anchor_edges VALUES (?,?,?)",
        [(a, a * 10, a / 4) for a in range(1, 5)],
    )
    rows = [
        # inside the closure window
        (1, "2026-08-02T00:00:00Z", 1, 0, None),
        (2, "2026-08-09T12:00:00Z", 2, 1, None),
        (3, "2026-08-19T10:59:59Z", 3, 0, None),
        # outside the closure window
        (4, "2026-07-30T00:00:00Z", 4, 0, None),
        (5, "2026-08-20T00:00:00Z", 1, 0, None),
        # inherited violation: anchor 99 does not exist.  Deliberately outside the
        # closure window, matching the bound source, where closure-local violations
        # must be zero while inherited ones are preserved untouched.
        (6, "2026-07-25T00:00:00Z", 99, 0, None),
    ]
    con.executemany("INSERT INTO recall_events VALUES (?,?,?,?,?)", rows)
    if extra_orphan:
        con.execute(
            "INSERT INTO recall_events VALUES (?,?,?,?,?)",
            (7, "2026-07-26T00:00:00Z", 98, 0, None),
        )
    for event_id in range(1, nonnull_map + 1):
        con.execute("UPDATE recall_events SET recall_map = ? WHERE id = ?", ('{"clusters":[]}', event_id))
    con.commit()
    con.execute("PRAGMA journal_mode=DELETE")  # keep the fixture a single closed main file
    con.commit()
    con.close()


def measure_expectations(path: Path) -> dict[str, Any]:
    """Recompute the preseal-pinned framings independently of the module under test."""
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        schema_rows = con.execute(
            "SELECT type,name,tbl_name,rootpage,sql FROM sqlite_schema "
            "ORDER BY type,name,tbl_name,rootpage,coalesce(sql,'')"
        ).fetchall()
        objects = [dict(zip(("type", "name", "tbl_name", "rootpage", "sql"), r)) for r in schema_rows]
        schema_blob = canonical(objects)

        cur = con.execute("PRAGMA table_xinfo(recall_events)")
        xcolumns = [d[0] for d in cur.description]
        xrows = [dict(zip(xcolumns, r)) for r in cur.fetchall()]
        xinfo_blob = canonical(xrows)

        cur = con.execute("PRAGMA foreign_key_check")
        fk_columns = [d[0] for d in cur.description]
        raw = cur.fetchall()
        fk_objects = sorted(
            ({fk_columns[i]: row[i] for i in range(len(fk_columns))} for row in raw), key=canonical
        )
        fk_blob = canonical({"columns": fk_columns, "rows": fk_objects})

        counts: dict[str, int] = {}
        for table, *_ in raw:
            counts[table] = counts.get(table, 0) + 1
        declarations = sum(
            len(con.execute(f'PRAGMA foreign_key_list("{name}")').fetchall())
            for name in (r[1] for r in schema_rows if r[0] == "table" and not r[1].startswith("sqlite_"))
        )
        user_tables = [r[1] for r in schema_rows if r[0] == "table" and not r[1].startswith("sqlite_")]
        closure_rowids = {
            r[0]
            for r in con.execute(
                "SELECT rowid FROM recall_events WHERE created_at >= ? AND created_at <= ?",
                (CLOSURE_START, CLOSURE_END),
            )
        }
        closure_violations = sum(
            1 for table, rowid, *_ in raw if table == "recall_events" and rowid in closure_rowids
        )
        rows_total = con.execute("SELECT count(*) FROM recall_events").fetchone()[0]
        created_min, created_max = con.execute(
            "SELECT min(created_at),max(created_at) FROM recall_events"
        ).fetchone()
        nonnull = con.execute(
            "SELECT count(*) FROM recall_events WHERE recall_map IS NOT NULL"
        ).fetchone()[0]
        pragmas = {
            name: con.execute(f"PRAGMA {name}").fetchone()[0]
            for name in ("page_size", "page_count", "freelist_count", "encoding",
                         "application_id", "user_version", "schema_version")
        }
    finally:
        con.close()

    type_counts: dict[str, int] = {}
    for obj in objects:
        type_counts[obj["type"]] = type_counts.get(obj["type"], 0) + 1
    return {
        **pragmas,
        "schema": {
            "sqlite_schema_rows_total": len(objects),
            "sql_defined_objects": sum(1 for o in objects if o["sql"] is not None),
            "type_counts": type_counts,
            "canonical_json_bytes": len(schema_blob),
            "canonical_json_sha256": sha(schema_blob),
            "recall_events_columns": [r["name"] for r in xrows],
            "recall_events_table_xinfo_canonical_json_bytes": len(xinfo_blob),
            "recall_events_table_xinfo_canonical_json_sha256": sha(xinfo_blob),
        },
        "recall_events_extent": {
            "rows": rows_total,
            "created_at_min": created_min,
            "created_at_max": created_max,
            "recall_map_column_present": True,
            "nonnull_recall_map_rows": nonnull,
        },
        "inherited_foreign_key_observation": {
            "supplementary_canonical_json_bytes": len(fk_blob),
            "supplementary_canonical_json_sha256": sha(fk_blob),
        },
        "_fk": {
            "user_table_count": len(user_tables),
            "foreign_key_declaration_count": declarations,
            "tuple_count": len(raw),
            "tuple_table_counts": counts,
            "source_closure_violations": closure_violations,
        },
    }


class Namespace:
    """A synthetic field namespace plus the preseal that binds it."""

    def __init__(self, base: Path, **store_kwargs: Any) -> None:
        self.base = base
        self.sfx = base / "sfx"
        self.sfx.mkdir(parents=True, exist_ok=True)
        self.origin = base / "origin.sqlite3"
        self.provenance = base / "provenance.py"
        self.source = self.sfx / "source.sqlite3"
        self.field = self.sfx / "field.sqlite3"
        self.seed = self.sfx / "alt-seed.sqlite3"
        self.receipt = base / "receipts" / "sfx-store-preparation.json"
        self.protocol = base / "protocol.json"
        self.instruments = base / "instruments"

        build_store(self.origin, **store_kwargs)
        self.provenance.write_text("# synthetic provenance evidence\n")
        shutil.copyfile(self.origin, self.source)
        os.chmod(self.source, 0o444)
        self.expectations = measure_expectations(self.source)
        self.write_protocol()

    def write_instruments(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Seal a synthetic instrument set — same ids and count as the shipped one.

        Two of the five shipped instruments are sealed by absolute path inside a
        *different* repository on this host.  A unit test owns neither that
        working tree nor its release cadence, so binding the synthetic namespace
        to it makes every run of this file hostage to unrelated work over there.
        The instruments are therefore synthesized here like every other binding;
        the seals this repository is actually answerable for are asserted by
        ``test_the_repo_holds_the_sealed_instruments_it_owns``.
        """
        self.instruments.mkdir(parents=True, exist_ok=True)
        sealed = []
        for item in items:
            path = self.instruments / f"{item['id']}.sealed"
            path.write_text(f"synthetic sealed instrument {item['id']}\n")
            sealed.append({**item, "observed_path_on_sfx": str(path),
                           "bytes": path.stat().st_size, "sha256": pfs.sha_file(path)})
        return sealed

    @property
    def required_bytes(self) -> int:
        return self.source.stat().st_size

    @property
    def required_sha256(self) -> str:
        return pfs.sha_file(self.source)

    def write_protocol(self) -> None:
        """Derive a preseal from the shipped one: real schemas, synthetic bindings."""
        doc = json.loads(REAL_PROTOCOL.read_bytes())
        expectations = copy.deepcopy(self.expectations)
        fk_expect = expectations.pop("_fk")

        binding = doc["baseline_compatible_source_binding"]
        binding["origin_observation"].update(
            path=str(self.origin),
            required_bytes=self.required_bytes,
            required_sha256=self.required_sha256,
        )
        binding["authoritative_stable_source"].update(
            path=str(self.source), required_mode="0444",
            bytes=self.required_bytes, sha256=self.required_sha256,
        )
        binding["observed_creation_provenance"].update(
            evidence_path=str(self.provenance),
            evidence_bytes=self.provenance.stat().st_size,
            evidence_sha256=pfs.sha_file(self.provenance),
        )
        binding["observed_sqlite_state"].update(expectations)

        contract = doc["metadata_only_store_construction_contract"]
        contract["inherited_foreign_key_baseline"]["bound_source_expectation"].update(fk_expect)

        doc["root_bindings"]["sfx"].update(
            field_store_root=str(self.sfx),
            source_snapshot_path=str(self.source),
            field_store_path=str(self.field),
            sanitized_seed_path=str(self.seed),
            ae_journal_root=str(self.base / "ae"),
            rollback_material_root=str(self.base / "rollback"),
        )
        doc["external_receipts"]["sfx_store_preparation"]["path"] = str(self.receipt)
        doc["sealed_instruments"]["items"] = self.write_instruments(
            doc["sealed_instruments"]["items"])
        self.protocol.write_bytes(canonical(doc) + b"\n")

    def invoke(self, *args: str, expect: int | None = None) -> subprocess.CompletedProcess:
        command = [sys.executable, str(SCRIPT), *args,
                   "--protocol", str(self.protocol), "--no-git", "--no-organic",
                   "--live-db", str(self.base / "nonexistent-live.sqlite3")]
        result = subprocess.run(command, capture_output=True, text=True)
        if expect is not None:
            assert result.returncode == expect, result.stdout + result.stderr
        return result

    def digest_tree(self) -> dict[str, str]:
        return {
            str(path.relative_to(self.base)): pfs.sha_file(path)
            for path in sorted(self.base.rglob("*")) if path.is_file()
        }

    def host_configuration(self) -> dict[str, Any]:
        """A schema-valid host configuration, since offline runs capture none."""
        doc = json.loads(self.protocol.read_bytes())
        schema = doc["schemas"]["host_configuration"]
        delivery = {name: None for name in schema["delivery_channel"]["required_fields"]}
        delivery.update(chat_journal_producer_roots=[], AE_STREAM_PROBE=1,
                        AE_STREAM_INJECT_SILENCE_TOOLS=5, AE_STREAM_INJECT_CAP=3,
                        AE_CHAT_LM_PROBE=1, AE_CHAT_OPERATOR_MESSAGE_MAP=1,
                        AE_CHAT_INJECT_MIN_TOOLS=8, unchanged=True)
        digest = sha(canonical({k: v for k, v in delivery.items()
                                if not k.endswith("_canonical_sha256") and k != "unchanged"}))
        delivery["before_canonical_sha256"] = delivery["after_canonical_sha256"] = digest
        configuration = {name: f"synthetic-{name}" for name in schema["required_fields"]}
        configuration.update(
            host_id="sfx", service_manager_scope="user", install_mode="editable", transport="http",
            port=8765, tls_enabled=False, tls_unit=None, config_file_path=None,
            config_file_bytes_sha256=None, unit_fragment_paths=[],
            unit_fragment_bytes_sha256={}, exec_start_argv=[], environment_file_paths=[],
            environment_file_sanitized_sha256={}, lm_environment={}, secret_presence={},
            policy_bytes_sha256=doc["policy_binding"]["stored_bytes_sha256"],
            selected_policy_sha256=doc["policy_binding"]["selected_policy_sha256"],
            delivery_channel=delivery,
        )
        return configuration

    def publish(self, path: Path, receipt: dict[str, Any]) -> Path:
        pfs.atomic_write_readonly(path, canonical(receipt) + b"\n")
        return path

    def rebuilt_receipt(self) -> dict[str, Any]:
        emitted = self.base / "emitted.json"
        self.invoke("verify", "--emit", str(emitted), expect=0)
        receipt = json.loads(emitted.read_bytes())
        receipt["source_host_configuration"] = self.host_configuration()
        return receipt


@pytest.fixture
def namespace(tmp_path: Path) -> Namespace:
    return Namespace(tmp_path / "field")


# ---------------------------------------------------------------------------
# typed serialization
# ---------------------------------------------------------------------------
def test_typed_encoding_separates_types_and_lengths() -> None:
    """Distinct SQLite values must never collide after encoding."""
    encodings = [
        pfs.tv(None), pfs.tv(""), pfs.tv(0), pfs.tv("0"), pfs.tv(b"0"),
        pfs.tv(1), pfs.tv("1"), pfs.tv(1.0), pfs.tv(-1), pfs.tv("-1"),
        pfs.tv("ab"), pfs.tv("a") + pfs.tv("b"),
    ]
    assert len(set(encodings)) == len(encodings)
    assert pfs.tv(None) != pfs.tv("")
    assert pfs.tv(1) != pfs.tv(1.0)
    assert pfs.tv("é").startswith(b"T")


def test_framing_is_length_prefixed_and_deterministic() -> None:
    assert pfs.row_body((1, "a")) == pfs.tv(1) + pfs.tv("a")
    assert pfs.record(b"xy") == b"W" + (2).to_bytes(8, "big") + b"xy"
    assert pfs.dom("x") != pfs.hdr("x")
    assert pfs.count_frame(0) != pfs.count_frame(1)
    assert pfs.canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


# ---------------------------------------------------------------------------
# preparation, idempotence, verification
# ---------------------------------------------------------------------------
def test_prepare_then_verify_accepts_a_clean_namespace(namespace: Namespace) -> None:
    namespace.invoke("prepare", expect=0)
    assert pfs.sha_file(namespace.field) == namespace.required_sha256
    assert pfs.sha_file(namespace.seed) == namespace.required_sha256
    assert namespace.field.stat().st_mode & 0o777 == 0o600
    assert namespace.seed.stat().st_mode & 0o777 == 0o444, "the transferable seed must stay read-only"
    namespace.invoke("verify", expect=0)


def test_prepare_is_idempotent_and_reuses_exact_copies(namespace: Namespace) -> None:
    namespace.invoke("prepare", expect=0)
    before = namespace.digest_tree()
    inode = namespace.field.stat().st_ino
    result = namespace.invoke("prepare", expect=0)
    assert "already exact" in result.stdout
    assert namespace.digest_tree() == before
    assert namespace.field.stat().st_ino == inode, "an exact field store must not be recopied"


def test_verify_refuses_a_namespace_that_was_never_prepared(namespace: Namespace) -> None:
    result = namespace.invoke("verify", expect=1)
    assert "rerun in prepare mode" in result.stdout


def test_verify_does_not_mutate_the_namespace(namespace: Namespace) -> None:
    namespace.invoke("prepare", expect=0)
    before = namespace.digest_tree()
    stats = {name: namespace.base.joinpath(name).stat() for name in before}
    namespace.invoke("verify", expect=0)
    assert namespace.digest_tree() == before
    for name, previous in stats.items():
        current = namespace.base.joinpath(name).stat()
        assert current.st_mtime == previous.st_mtime, f"verify rewrote {name}"
        assert current.st_mode == previous.st_mode, f"verify remoded {name}"


# ---------------------------------------------------------------------------
# fail-closed behaviour
# ---------------------------------------------------------------------------
def test_source_digest_drift_is_a_conflict(namespace: Namespace) -> None:
    namespace.invoke("prepare", expect=0)
    os.chmod(namespace.source, 0o644)
    with open(namespace.source, "r+b") as handle:
        handle.seek(namespace.required_bytes - 1)
        last = handle.read(1)
        handle.seek(namespace.required_bytes - 1)
        handle.write(b"\x01" if last == b"\x00" else b"\x00")
    os.chmod(namespace.source, 0o444)
    result = namespace.invoke("verify", expect=1)
    assert "authoritative source digest drift" in result.stdout


def test_source_mode_drift_is_a_conflict(namespace: Namespace) -> None:
    namespace.invoke("prepare", expect=0)
    os.chmod(namespace.source, 0o644)
    result = namespace.invoke("verify", expect=1)
    assert "authoritative source mode 0o644 != 0o444" in result.stdout


def test_nonzero_wal_sidecar_is_a_conflict(namespace: Namespace) -> None:
    namespace.invoke("prepare", expect=0)
    namespace.source.with_name(namespace.source.name + "-wal").write_bytes(b"not-empty")
    result = namespace.invoke("verify", expect=1)
    assert "nonzero WAL sidecar" in result.stdout


def test_writable_seed_is_a_conflict(namespace: Namespace) -> None:
    namespace.invoke("prepare", expect=0)
    os.chmod(namespace.seed, 0o644)
    result = namespace.invoke("verify", expect=1)
    assert "seed: not byte-exact, closed and mode 0o444" in result.stdout


def test_missing_provenance_evidence_is_a_conflict(namespace: Namespace) -> None:
    namespace.invoke("prepare", expect=0)
    namespace.provenance.unlink()
    result = namespace.invoke("verify", expect=1)
    assert "provenance evidence" in result.stdout and "no longer attributes" in result.stdout


def test_a_drifted_sealed_instrument_is_a_conflict(namespace: Namespace) -> None:
    doc = json.loads(namespace.protocol.read_bytes())
    doc["sealed_instruments"]["items"][0]["sha256"] = "0" * 64
    namespace.protocol.write_bytes(canonical(doc) + b"\n")
    result = namespace.invoke("prepare", expect=1)
    assert "sealed instrument recall-map-prereg drifted" in result.stdout


def test_the_repo_holds_the_sealed_instruments_it_owns() -> None:
    """The shipped preseal still describes this checkout's own sealed files.

    Only the instruments sealed in *this* repository are asserted.  The ones
    sealed in another repository are host state — their bytes drift with work
    that has nothing to do with this suite, and the operator verifies them at
    field-verification time, per the runbook in docs/.  Sealing them into a unit
    test turns a foreign commit into a red suite here.
    """
    items = json.loads(REAL_PROTOCOL.read_bytes())["sealed_instruments"]["items"]
    owned = [item for item in items if item["repository"] == "lm"]
    assert owned, "the preseal seals nothing in this repository — check the fixture"
    drifted = []
    for item in owned:
        path = ROOT_DIR / item["path"]
        assert path.exists(), f"sealed instrument {item['id']} absent at {path}"
        if (path.stat().st_size, pfs.sha_file(path)) != (item["bytes"], item["sha256"]):
            drifted.append(item["id"])
    assert not drifted, f"sealed instruments drifted: {drifted}"


def test_a_non_null_recall_map_in_the_source_is_a_conflict(tmp_path: Path) -> None:
    """A nonzero pre-transform count is source drift, never permission to sanitize."""
    namespace = Namespace(tmp_path / "field", nonnull_map=2)
    doc = json.loads(namespace.protocol.read_bytes())
    extent = doc["baseline_compatible_source_binding"]["observed_sqlite_state"]["recall_events_extent"]
    extent["nonnull_recall_map_rows"] = 0  # what the contract requires of a bound source
    namespace.protocol.write_bytes(canonical(doc) + b"\n")
    result = namespace.invoke("prepare", expect=1)
    assert "nonnull_recall_map_rows=2 expected 0" in result.stdout
    assert "has non-NULL recall_map rows" in result.stdout


def test_a_new_foreign_key_violation_in_the_field_store_is_a_conflict(
    namespace: Namespace, tmp_path: Path
) -> None:
    """Inherited violations are preserved; a violation the source lacks must be rejected."""
    namespace.invoke("prepare", expect=0)
    drifted = Namespace(tmp_path / "drifted", extra_orphan=True)
    os.chmod(namespace.field, 0o600)
    shutil.copyfile(drifted.source, namespace.field)
    result = namespace.invoke("verify", expect=1)
    assert "new foreign-key violations in the field store" in result.stdout


def test_inherited_violations_are_preserved_not_repaired(namespace: Namespace) -> None:
    namespace.invoke("prepare", expect=0)
    receipt_path = namespace.base / "rebuilt.json"
    namespace.invoke("verify", "--emit", str(receipt_path), expect=0)
    baseline = json.loads(receipt_path.read_bytes())["inherited_foreign_key_baseline"]
    assert baseline["source"]["tuple_count"] == 1, "the fixture carries one inherited violation"
    assert baseline["field"]["tuple_count"] == baseline["source"]["tuple_count"]
    assert baseline["new_violations"] == 0
    assert baseline["closure_violations"] == 0
    assert baseline["source_field_equal"] is True


# ---------------------------------------------------------------------------
# receipt contract
# ---------------------------------------------------------------------------
def test_receipt_matches_the_shipped_schema_exactly(namespace: Namespace) -> None:
    """Field set is fixed by the preseal: unknown fields are rejected, none may be missing."""
    namespace.invoke("prepare", expect=0)
    emitted = namespace.base / "rebuilt.json"
    namespace.invoke("verify", "--emit", str(emitted), expect=0)
    receipt = json.loads(emitted.read_bytes())
    schema = json.loads(REAL_PROTOCOL.read_bytes())["schemas"]["store_preparation_receipt"]
    assert set(receipt) == set(schema["required_fields"])
    assert receipt["schema_id"] == schema["schema_id"]
    assert receipt["accepted"] is True
    assert receipt["protocol_conflict"] is None
    assert receipt["candidate_process_opened_store"] is False
    assert receipt["outcome_accessed"] is False
    assert receipt["organic_closure_equal"] is True
    assert receipt["pre_transform_nonnull_recall_map_rows"] == 0
    assert receipt["post_transform_nonnull_recall_map_rows"] == 0
    for side in ("source", "field"):
        assert set(receipt["inherited_foreign_key_baseline"][side]) >= set(
            schema["inherited_foreign_key_copy_required_fields"]
        )


def test_receipt_carries_no_outcome_bearing_content(namespace: Namespace) -> None:
    """receipt_content_rule: names, counts, lengths and digests only."""
    namespace.invoke("prepare", expect=0)
    emitted = namespace.base / "rebuilt.json"
    namespace.invoke("verify", "--emit", str(emitted), expect=0)
    receipt = json.loads(emitted.read_bytes())
    for forbidden in ("results", "query", "recall_map", "node_id", "transport_session_id", "session_id"):
        assert forbidden not in receipt
    text = emitted.read_text()
    assert "anchor-1" not in text, "an anchor label leaked into the receipt"
    assert '{\\"clusters\\":[]}' not in text, "a recall_map payload leaked into the receipt"


def test_transform_digest_is_derived_from_the_preseal_not_hardcoded() -> None:
    """The bound source needs no transform, but its statement digest must still be exact."""
    preseal = pfs.Preseal(REAL_PROTOCOL)
    assert preseal.transform_sql == (
        "UPDATE recall_events SET recall_map = NULL WHERE recall_map IS NOT NULL"
    )
    assert pfs.sha_bytes(preseal.transform_sql.encode()) == (
        "5c35b056df344fd94fcf2f660ef90ac5474bd6d28818759363ca0c66e646e4ef"
    )
    assert preseal.transform_sql not in SCRIPT.read_text(), "the statement must come from the preseal"


def test_write_receipt_requires_a_complete_run(namespace: Namespace) -> None:
    """A receipt may never be published from a run with checks disabled."""
    for args in (["prepare", "--write-receipt"], ["verify", "--write-receipt"]):
        result = namespace.invoke(*args)
        assert result.returncode != 0
        assert "requires" in result.stderr
    assert not namespace.receipt.exists()


def test_published_receipt_is_never_overwritten(namespace: Namespace, tmp_path: Path) -> None:
    """Immutability: the publish path refuses an existing receipt instead of replacing it."""
    namespace.invoke("prepare", expect=0)
    namespace.receipt.parent.mkdir(parents=True, exist_ok=True)
    pfs.atomic_write_readonly(namespace.receipt, b'{"schema_id":"already-here"}\n')
    before = namespace.receipt.read_bytes()
    assert namespace.receipt.stat().st_mode & 0o777 == 0o444

    preseal = pfs.Preseal(namespace.protocol)
    assert preseal.receipt_path == namespace.receipt
    with pytest.raises(SystemExit):
        pfs.main(["prepare", "--protocol", str(namespace.protocol), "--write-receipt", "--no-organic"])
    assert namespace.receipt.read_bytes() == before


def test_existing_receipt_must_reproduce_to_be_reused(namespace: Namespace) -> None:
    namespace.invoke("prepare", expect=0)
    receipt = namespace.rebuilt_receipt()
    published = namespace.publish(namespace.base / "published.json", receipt)

    accepted = namespace.invoke("verify", "--receipt", str(published), expect=0)
    assert "REUSED" in accepted.stdout

    tampered = {**receipt, "field_store_sha256": "0" * 64}
    rejected = namespace.invoke(
        "verify", "--receipt", str(namespace.publish(namespace.base / "tampered.json", tampered)),
        expect=1,
    )
    assert "the existing receipt does not reproduce" in rejected.stdout
    assert "field_store_sha256" in rejected.stdout


def test_a_receipt_without_a_host_configuration_is_rejected(namespace: Namespace) -> None:
    namespace.invoke("prepare", expect=0)
    receipt = {**namespace.rebuilt_receipt(), "source_host_configuration": None}
    result = namespace.invoke(
        "verify", "--receipt", str(namespace.publish(namespace.base / "no-host.json", receipt)),
        expect=1,
    )
    assert "carries no source host configuration" in result.stdout


def test_a_receipt_with_a_drifted_policy_digest_is_rejected(namespace: Namespace) -> None:
    """The served policy must be the one the preseal binds, not merely some policy."""
    namespace.invoke("prepare", expect=0)
    receipt = namespace.rebuilt_receipt()
    receipt["source_host_configuration"]["selected_policy_sha256"] = "0" * 64
    result = namespace.invoke(
        "verify", "--receipt", str(namespace.publish(namespace.base / "drifted.json", receipt)),
        expect=1,
    )
    assert "selected policy digest drift" in result.stdout


def test_a_writable_receipt_is_rejected(namespace: Namespace) -> None:
    namespace.invoke("prepare", expect=0)
    emitted = namespace.base / "receipt.json"
    namespace.invoke("verify", "--emit", str(emitted), expect=0)
    result = namespace.invoke("verify", "--receipt", str(emitted), expect=1)
    assert "the existing receipt is not read-only" in result.stdout
