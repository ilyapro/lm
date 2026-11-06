#!/usr/bin/env python3
"""Back up the live Living Memory SQLite database to one private S3 object."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import uuid
import zlib
from datetime import datetime, timezone
from urllib.parse import quote


DATABASE = Path.home() / ".local/share/living-memory/global.sqlite3"
STATE_DIR = Path.home() / ".local/state/living-memory-backup"
CONFIG = Path.home() / ".config/living-memory/s3cmd-backup.conf"
BUCKET = "lm-private-backup-sfx"
FINAL = f"s3://{BUCKET}/global.sqlite3.gz"
STAGING = f"s3://{BUCKET}/.incoming.sqlite3.gz"
PRIVATE_ACL = ["sfx: FULL_CONTROL"]
READ_TIMEOUT = 120
TRANSFER_TIMEOUT = 3600


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_status() -> dict:
    path = STATE_DIR / "status.json"
    try:
        status = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    if not isinstance(status, dict):
        raise RuntimeError("invalid backup status; preserve it for reconciliation")
    transaction = status.get("transaction")
    if transaction is not None and (not isinstance(transaction, dict) or
                                   not {"phase", "pending", "safety", "previous", "expected"} <= transaction.keys()):
        raise RuntimeError("invalid backup transaction; preserve it for reconciliation")
    return status


def save_status(**updates: object) -> None:
    status = read_status()
    status.update(updates)
    temporary = STATE_DIR / "status.json.tmp"
    with temporary.open("w") as stream:
        os.chmod(temporary, 0o600)
        stream.write(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(STATE_DIR / "status.json")
    descriptor = os.open(STATE_DIR, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def s3(*args: str) -> str:
    try:
        result = subprocess.run(
            ["/usr/bin/s3cmd", "-c", str(CONFIG), "--no-progress", "--max-retries=0", *args],
            capture_output=True, text=True,
            timeout=TRANSFER_TIMEOUT if args[0] in {"get", "put", "cp"} else READ_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        # Never include commands, config, request details or credentials in status.
        raise RuntimeError(f"s3cmd {args[0]} transport failed ({type(error).__name__})") from None
    if result.returncode:
        raise RuntimeError(f"s3cmd {args[0]} failed (exit {result.returncode})")
    return result.stdout


def mutate(transaction: dict, phase: str, *args: str, completed: str | None = None) -> None:
    # Write intent BEFORE sending. A crash even after a successful response is
    # conservatively UNKNOWN until the existing backup owner reconciles it.
    transaction["pending"] = {"operation": args[0], "target": args[1] if args[0] == "abortmp" else args[-1]}
    if args[0] == "abortmp":
        transaction["pending"]["upload_id"] = args[2]
    save_status(transaction=transaction)
    s3(*args)
    transaction["pending"] = None
    transaction["phase"] = phase
    if completed:
        transaction[completed] = True
    save_status(transaction=transaction)


def info(target: str) -> dict[str, str]:
    output = s3("info", target)
    fields = {}
    acl = []
    for line in output.splitlines():
        key, separator, value = line.strip().partition(":")
        if separator:
            fields[key] = value.strip()
            if key == "ACL":
                acl.append(value.strip())
    if acl != PRIVATE_ACL or fields.get("Policy") != "none":
        raise RuntimeError("S3 privacy verification failed")
    return fields


def objects() -> set[str]:
    keys = set()
    for line in s3("ls", f"s3://{BUCKET}/").splitlines():
        columns = line.split()
        if not columns or not columns[-1].startswith(f"s3://{BUCKET}/"):
            raise RuntimeError("unrecognized S3 object listing")
        keys.add(columns[-1])
    return keys


def multipart() -> list[tuple[str, str]]:
    uploads = []
    lines = s3("multipart", f"s3://{BUCKET}").splitlines()
    if lines[:2] != [f"s3://{BUCKET}/", "Initiated\tPath\tId"]:
        raise RuntimeError("unrecognized S3 multipart listing")
    for line in lines[2:]:
        columns = line.split("\t")
        if len(columns) != 3 or not columns[1].startswith(f"s3://{BUCKET}/"):
            raise RuntimeError("unrecognized S3 multipart listing")
        uploads.append((columns[1], columns[2]))
    return uploads


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def check_sqlite(path: Path) -> None:
    connection = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True)
    try:
        result = connection.execute("PRAGMA integrity_check").fetchone()
        if result != ("ok",):
            raise RuntimeError("SQLite integrity_check failed")
    finally:
        connection.close()


def compress(source: Path, destination: Path) -> None:
    with source.open("rb") as plain, gzip.open(destination, "wb", compresslevel=1) as packed:
        shutil.copyfileobj(plain, packed, length=8 * 1024 * 1024)


def decompress(source: Path, destination: Path) -> None:
    try:
        with gzip.open(source, "rb") as packed, destination.open("wb") as plain:
            shutil.copyfileobj(packed, plain, length=8 * 1024 * 1024)
    except (EOFError, zlib.error, gzip.BadGzipFile):
        raise RuntimeError("invalid backup gzip archive") from None


def clean_staging(transaction: dict) -> None:
    phase = transaction["phase"]
    if not transaction.get("staging_deleted"):
        mutate(transaction, phase, "del", STAGING, completed="staging_deleted")
    if not transaction.get("multipart_clean"):
        uploads = multipart()
        if any(key != STAGING for key, _ in uploads):
            raise RuntimeError("unowned incomplete S3 uploads; preserving them")
        for key, identifier in uploads:
            mutate(transaction, phase, "abortmp", key, identifier)
        transaction["multipart_clean"] = True
        save_status(transaction=transaction)
    safety = transaction["safety"]
    if safety and not transaction.get("safety_deleted"):
        mutate(transaction, phase, "del", safety, completed="safety_deleted")
    expected = {FINAL} if transaction["phase"] == "verified" or transaction["previous"] else set()
    if objects() != expected:
        raise RuntimeError("unexpected S3 objects after cleanup; preserving them")


def finish(transaction: dict) -> int:
    """Cleanup is outside the replacement/rollback boundary, including on resume."""
    try:
        clean_staging(transaction)
    except (OSError, RuntimeError) as error:
        unknown = bool(transaction.get("pending"))
        save_status(cleanup="unknown" if unknown else "unverified",
                    cleanup_error={"at": now(), "message": str(error)})
        print(f"Living Memory backup {transaction['phase']}; cleanup unverified: {error}", file=sys.stderr)
        return 3 if unknown else (2 if transaction["phase"] == "verified" else 1)
    save_status(transaction=None, cleanup="complete", cleanup_error=None)
    if transaction["phase"] == "verified":
        print("Living Memory backup verified; cleanup complete")
        return 0
    return 1


def snapshot(path: Path) -> None:
    if not DATABASE.is_file():
        raise FileNotFoundError("Living Memory live database is missing")
    source = sqlite3.connect(f"file:{quote(str(DATABASE))}?mode=ro", uri=True, timeout=60)
    try:
        destination = sqlite3.connect(path)
        try:
            source.backup(destination, pages=4096, sleep=0.1)
        finally:
            destination.close()
    finally:
        source.close()


def verify_download(target: str, work: Path, name: str, expected: dict | None = None) -> dict:
    info(target)  # ACL and bucket policy are part of verification, before success.
    archive = work / f"{name}.sqlite3.gz"
    restored = work / f"{name}.sqlite3"
    s3("get", target, str(archive))
    archive_hash = digest(archive)
    if expected and archive_hash != expected["archive_sha256"]:
        raise RuntimeError(f"{name} archive checksum mismatch")
    decompress(archive, restored)
    checksum = digest(restored)
    if expected and checksum != expected["sha256"]:
        raise RuntimeError(f"{name} download checksum mismatch")
    check_sqlite(restored)
    return {"archive_sha256": archive_hash, "sha256": checksum, "bytes": restored.stat().st_size}


def preserve_previous(transaction: dict, work: Path, before: dict) -> None:
    safety = transaction["safety"]
    mutate(transaction, "safety_copied", "cp", "--acl-private", FINAL, safety)
    copied = info(safety)
    for key in ("File size", "MD5 sum"):
        if not before.get(key) or copied.get(key) != before[key]:
            raise RuntimeError("previous safety metadata mismatch")
    previous = verify_download(safety, work, "previous")
    archive = work / "previous.sqlite3.gz"
    # s3cmd's info uses the original content MD5 from s3cmd-attrs, including
    # multipart objects. Missing/opaque multipart ETags fail closed here.
    md5 = hashlib.md5(usedforsecurity=False)
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(block)
    if str(archive.stat().st_size) != before["File size"] or md5.hexdigest() != before["MD5 sum"]:
        raise RuntimeError("previous safety content mismatch")
    transaction.update(phase="safety_verified", previous=previous)
    save_status(transaction=transaction)


def rollback(transaction: dict, work: Path) -> None:
    # Verify the retained source again before using it, never delete it here.
    verify_download(transaction["safety"], work, "rollback-source", transaction["previous"])
    mutate(transaction, "restored", "cp", "--acl-private", transaction["safety"], FINAL)
    verify_download(FINAL, work, "rollback", transaction["previous"])
    transaction["phase"] = "rolled_back"
    save_status(transaction=transaction, rollback="verified")


def check_config() -> None:
    if not CONFIG.is_file() or CONFIG.stat().st_mode & 0o077:
        raise RuntimeError("private S3 configuration is missing or accessible to others")


def preflight() -> dict | None:
    check_config()
    # Positive inventory establishes absence. A timeout/403 is never "no FINAL".
    if info(f"s3://{BUCKET}").get("Versioning") != "none":
        raise RuntimeError("S3 bucket versioning must be disabled")
    keys = objects()
    if keys - {FINAL}:
        raise RuntimeError("unowned S3 objects; reconcile retained copies before backup")
    if multipart():
        raise RuntimeError("unowned incomplete S3 uploads; reconcile before backup")
    if FINAL not in keys:
        return None
    previous = info(FINAL)
    return {key: previous.get(key, "") for key in ("File size", "MD5 sum")}


def run() -> int:
    """Called only by main(), while holding backup.lock for the whole operation."""
    status = read_status()
    transaction = status.get("transaction")
    if transaction:
        if transaction.get("pending"):
            completed = transaction["phase"] in {"verified", "rolled_back", "unchanged"}
            save_status(result=status.get("result") if completed else "unknown",
                        cleanup="unknown" if completed else "unverified")
            print("Living Memory backup UNKNOWN mutation; owner reconciliation required; copies retained", file=sys.stderr)
            return 3
        if transaction.get("phase") in {"verified", "rolled_back", "unchanged"}:
            check_config()
            return finish(transaction)  # No snapshot, upload or duplicate FINAL verification.
        save_status(result="unknown", cleanup="unverified")
        print("Living Memory backup interrupted/unresolved; owner reconciliation required; copies retained", file=sys.stderr)
        return 3

    try:
        before = preflight()
    except (OSError, RuntimeError) as error:
        save_status(result="failed", last_attempt=now(), cleanup="unverified",
                    last_error={"at": now(), "message": str(error)})
        raise
    transaction = {
        "phase": "preparing", "pending": None,
        "safety": f"s3://{BUCKET}/.previous-{uuid.uuid4().hex}.sqlite3.gz" if before else None,
        "previous": before, "expected": None,
    }
    save_status(transaction=transaction, result="in_progress", cleanup="pending",
                last_attempt=now(), last_error=None, cleanup_error=None, rollback=None, rollback_error=None)
    with tempfile.TemporaryDirectory(prefix="work-", dir=STATE_DIR) as directory:
        work = Path(directory)
        local = work / "snapshot.sqlite3"
        archive = work / "snapshot.sqlite3.gz"
        try:
            snapshot(local)
            check_sqlite(local)
            compress(local, archive)
            expected = {"archive_sha256": digest(archive), "sha256": digest(local), "bytes": local.stat().st_size}
            transaction["expected"] = expected
            if before:
                preserve_previous(transaction, work, before)
            mutate(transaction, "uploaded", "put", "--acl-private", str(archive), STAGING)
            verify_download(STAGING, work, "staging", expected)
            mutate(transaction, "replaced", "cp", "--acl-private", STAGING, FINAL)
            verify_download(FINAL, work, "final", expected)
        except (OSError, sqlite3.Error, RuntimeError, EOFError) as error:
            save_status(result="unknown" if transaction["pending"] else "failed",
                        last_error={"at": now(), "message": str(error)}, cleanup="unverified")
            print(f"Living Memory backup not verified: {error}", file=sys.stderr)
            if transaction["pending"]:
                return 3  # Never rollback or clean up an unconfirmed mutation.
            if transaction["phase"] == "replaced":
                if not transaction["previous"]:
                    save_status(cleanup="unverified", rollback="unavailable")
                    return 1  # Preserve the checked staging copy, no prior FINAL exists.
                try:
                    rollback(transaction, work)
                except (OSError, sqlite3.Error, RuntimeError, EOFError) as restore_error:
                    save_status(result="unknown" if transaction["pending"] else "failed",
                                rollback="unknown" if transaction["pending"] else "failed",
                                rollback_error={"at": now(), "message": str(restore_error)}, cleanup="unverified")
                    print("Living Memory rollback unverified; safety retained", file=sys.stderr)
                    return 3 if transaction["pending"] else 1
            else:
                # FINAL was never replaced, so removing our temporary copies is safe.
                transaction["phase"] = "unchanged"
                save_status(transaction=transaction)
            return finish(transaction)

        # Commit the verified result BEFORE cleanup. No later exception enters
        # rollback, erases success or authorizes a consumer to repeat this work.
        transaction["phase"] = "verified"
        save_status(transaction=transaction, result="verified", last_success=now(),
                    last_error=None, sha256=expected["sha256"], bytes=expected["bytes"],
                    archive_sha256=expected["archive_sha256"])
    return finish(transaction)


def main() -> int:
    if len(sys.argv) != 1:
        print("Usage: backup_living_memory_s3.py (no arguments)", file=sys.stderr)
        return 1
    os.umask(0o077)
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(STATE_DIR, 0o700)
    with (STATE_DIR / "backup.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Living Memory backup already running", file=sys.stderr)
            return 1
        try:
            return run()
        except (OSError, sqlite3.Error, RuntimeError, ValueError) as error:
            # In particular, do not reset unreadable status or overwrite a
            # durable pending mutation after a local status-write failure.
            print(f"Living Memory backup stopped: {type(error).__name__}: {error}", file=sys.stderr)
            try:
                status = read_status()
                pending = bool((status.get("transaction") or {}).get("pending"))
                if status.get("result") == "verified":
                    save_status(cleanup="unknown" if pending else "unverified",
                                cleanup_error={"at": now(), "message": str(error)})
                    return 3 if pending else 2
                if pending:
                    return 3
            except (OSError, RuntimeError, ValueError):
                pass  # Unreadable status is retained, never reset to an empty run.
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
