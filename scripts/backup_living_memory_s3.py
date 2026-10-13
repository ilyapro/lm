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
from datetime import datetime, timezone
from urllib.parse import quote


DATABASE = Path.home() / ".local/share/living-memory/global.sqlite3"
STATE_DIR = Path.home() / ".local/state/living-memory-backup"
CONFIG = Path.home() / ".config/living-memory/s3cmd-backup.conf"
BUCKET = "lm-private-backup-sfx"
FINAL = f"s3://{BUCKET}/global.sqlite3.gz"
STAGING = f"s3://{BUCKET}/.incoming.sqlite3.gz"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_status(**updates: object) -> None:
    path = STATE_DIR / "status.json"
    try:
        status = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        status = {}
    status.update(updates)
    temporary = STATE_DIR / "status.json.tmp"
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def s3(*args: str) -> None:
    result = subprocess.run(
        ["/usr/bin/s3cmd", "-c", str(CONFIG), "--no-progress", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        # s3cmd output may contain request details; keep secrets out of status/logs.
        raise RuntimeError(f"s3cmd {args[0]} failed (exit {result.returncode})")


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
    with gzip.open(source, "rb") as packed, destination.open("wb") as plain:
        shutil.copyfileobj(packed, plain, length=8 * 1024 * 1024)


def clean_staging() -> None:
    s3("del", STAGING)
    result = subprocess.run(
        ["/usr/bin/s3cmd", "-c", str(CONFIG), "multipart", f"s3://{BUCKET}"],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError("cannot list incomplete S3 uploads")
    for line in result.stdout.splitlines():
        columns = line.split("\t")
        if len(columns) >= 3 and columns[1] == STAGING:
            s3("abortmp", STAGING, columns[2])


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


def run() -> None:
    if not CONFIG.is_file() or CONFIG.stat().st_mode & 0o077:
        raise RuntimeError("private S3 configuration is missing or accessible to others")
    with tempfile.TemporaryDirectory(prefix="work-", dir=STATE_DIR) as directory:
        work = Path(directory)
        local = work / "snapshot.sqlite3"
        archive = work / "snapshot.sqlite3.gz"
        restored_archive = work / "restored.sqlite3.gz"
        restored = work / "restored.sqlite3"
        final_archive = work / "final-restored.sqlite3.gz"
        final_restored = work / "final-restored.sqlite3"
        snapshot(local)
        check_sqlite(local)
        expected = digest(local)
        size = local.stat().st_size
        compress(local, archive)
        archive_hash = digest(archive)

        try:
            s3("put", "--acl-private", str(archive), STAGING)
            s3("get", STAGING, str(restored_archive))
            if digest(restored_archive) != archive_hash:
                raise RuntimeError("staging archive checksum mismatch")
            decompress(restored_archive, restored)
            if digest(restored) != expected:
                raise RuntimeError("staging download checksum mismatch")
            check_sqlite(restored)

            # S3 copy replaces this one key only after the new object has been
            # uploaded and round-trip checked. A failed preparation leaves FINAL.
            s3("cp", "--acl-private", STAGING, FINAL)
            s3("get", FINAL, str(final_archive))
            if digest(final_archive) != archive_hash:
                raise RuntimeError("final archive checksum mismatch")
            decompress(final_archive, final_restored)
            if digest(final_restored) != expected:
                raise RuntimeError("final download checksum mismatch")
            check_sqlite(final_restored)
        finally:
            # Also abort incomplete multipart uploads after a failed transfer.
            clean_staging()

        save_status(last_success=now(), last_error=None, sha256=expected, bytes=size)
        print(f"Living Memory backup verified: {size} bytes, sha256={expected}")


def main() -> int:
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(STATE_DIR, 0o700)
    with (STATE_DIR / "backup.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Living Memory backup already running", file=sys.stderr)
            return 1
        try:
            run()
        except (OSError, sqlite3.Error, RuntimeError) as error:
            message = str(error)
            save_status(last_error={"at": now(), "message": message})
            print(f"Living Memory backup failed: {message}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
